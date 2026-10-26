"""
Page-level text extraction from validated PDF documents.

1. Purpose: Extracts plain text and page metadata from PDF files using PyMuPDF (pymupdf),
   with font-aware character remapping for symbol fonts and smart punctuation.
2. Pipeline Role: Bridges document validation (loaders.py) and downstream semantic chunking.
3. Expected Inputs: ValidatedDocument instance pointing to an existing, validated PDF on disk.
4. Outputs: ExtractedDocument containing an ordered sequence of 1-indexed ExtractedPage records.
5. Dependencies: PyMuPDF (`pymupdf`), src.due_diligence.ingestion.document,
   src.due_diligence.ingestion.font_mappings.
6. Outside Scope: Optical Character Recognition (OCR), document chunking, table extraction, and embeddings.

NOTE ON MAPPING PROVENANCE & LIVING SPECIFICATION:
The symbol font translations applied during extraction are externalized in
`config/symbol_font_mappings.yaml`. They were initially derived from a single corporate
filing (HDFC Bank FY25-26 Annual Report, confirmed 2026-09-25) and represent a living,
extensible mapping. Unmapped symbol-font codepoints are replaced with structured
`[UNMAPPED:Font:0xHEX]` placeholders and logged with document and page provenance so they
can be audited and added without silent corruption.
"""

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

from due_diligence.ingestion.document import ExtractedDocument, ExtractedPage, ValidatedDocument
from due_diligence.ingestion.font_mappings import (
    format_unmapped_placeholder,
    get_symbol_font_category,
    is_symbol_font,
    load_symbol_mappings,
    strip_subset_prefix,
)
from due_diligence.ingestion.preflight import partition_preflight_batch, preflight_scan

logger = logging.getLogger(__name__)


class DocumentParseError(Exception):
    """Base exception for all document parsing failures."""


class EncryptedDocumentError(DocumentParseError, ValueError):
    """Raised when attempting to parse an encrypted or password-protected PDF."""


class CorruptedPDFParseError(DocumentParseError, ValueError):
    """Raised when PyMuPDF cannot read or process the PDF file stream."""


# Marker for standalone en-dash/em-dash representing nil or omitted metrics
NIL_NOT_DISCLOSED_MARKER = "[NIL/NOT DISCLOSED]"

# Matches standalone en-dash or em-dash bounded by whitespace or line boundaries
_STANDALONE_DASH_PATTERN = re.compile(r"(?<!\S)[–—](?!\S)")


def normalize_standalone_nil_dashes(text: str) -> str:
    """
    Replace standalone en-dashes and em-dashes with an explicit [NIL/NOT DISCLOSED] marker.

    HEURISTIC STOPGAP LIMITATION:
    This is a lightweight regex heuristic, not a structural table parsing fix.
    It distinguishes standalone dashes (e.g. 'Non-performing Assets\\t–\\t1,420') from
    mid-word dashes ('facility—maturing') or negative number prefixes ('-12%', '–1,420').
    However, it cannot replace true table-cell boundary recognition and may occasionally
    substitute a standalone em-dash used as a sentence-level parenthetical aside with spaces.
    Phase 1 Task 3's structured table extraction is the real architectural solution;
    this heuristic serves as an interim safeguard for documents ingested prior to that milestone.
    """
    return _STANDALONE_DASH_PATTERN.sub(NIL_NOT_DISCLOSED_MARKER, text)


def extract_page_text_with_font_remapping(
    page: pymupdf.Page,
    document_id: str = "unknown",
    page_number: int | None = None,
    config_path: Path | str | None = None,
) -> str:
    """
    Extract page text while remapping symbol fonts and normalizing smart punctuation.

    Traverses the page dictionary (blocks -> lines -> spans) to ensure symbol-font
    replacements are strictly font-scoped. Unmapped symbol codepoints are replaced with
    traceable placeholders and logged with document and page context.

    Args:
        page: PyMuPDF Page instance to extract.
        document_id: Optional document identifier or filename for structured log tracing.
        page_number: Optional 1-indexed page number for structured log tracing.
        config_path: Optional custom path to symbol font mappings YAML.

    Returns:
        Reassembled text string preserving line and block structure.
    """
    symbol_map, text_punct_map = load_symbol_mappings(config_path)
    page_num_str = str(page_number) if page_number is not None else str(page.number + 1)

    page_dict = page.get_text("dict")  # type: ignore[no-untyped-call]
    blocks_text: list[str] = []

    for block in page_dict.get("blocks", []):
        if "lines" not in block:
            continue
        lines_text: list[str] = []
        for line in block.get("lines", []):
            spans_text: list[str] = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                font = span.get("font", "")
                if not text:
                    continue

                clean_font = strip_subset_prefix(font)

                if is_symbol_font(font):
                    category = get_symbol_font_category(font)
                    remapped_chars: list[str] = []
                    for char in text:
                        cp = ord(char)
                        key = (category, cp)
                        if key in symbol_map:
                            remapped_chars.append(symbol_map[key])
                        else:
                            logger.warning(
                                "Unmapped codepoint in symbol font: doc='%s', page=%s, font='%s', codepoint=0x%04X (%d)",
                                document_id,
                                page_num_str,
                                clean_font,
                                cp,
                                cp,
                            )
                            # Never pass unmapped symbol glyphs raw: emit space-padded placeholder
                            remapped_chars.append(format_unmapped_placeholder(clean_font, cp))
                    spans_text.append("".join(remapped_chars))
                else:
                    remapped_chars = [text_punct_map.get(ord(char), char) for char in text]
                    spans_text.append("".join(remapped_chars))

            lines_text.append("".join(spans_text))
        blocks_text.append("\n".join(lines_text))

    if not blocks_text:
        return ""
    full_text = "\n".join(blocks_text) + "\n"
    return normalize_standalone_nil_dashes(full_text)


class PDFParser:
    """
    Extracts text page-by-page from validated PDF documents using PyMuPDF.

    Preserves 1-indexed physical page numbering for downstream citation attribution.
    """

    def __init__(self, config_path: Path | str | None = None) -> None:
        """Initialize parser with optional custom symbol mappings config path."""
        self.config_path = config_path

    def parse(
        self,
        validated_doc: ValidatedDocument,
        symbol_font_status: str | None = None,
    ) -> ExtractedDocument:
        """
        Extract page text and metadata from a validated PDF.

        Args:
            validated_doc: Verified document record from DocumentLoader.
            symbol_font_status: Optional explicit symbol font status ('clean', 'reviewed_<date>',
                or 'unmapped_allowed_<date>'). If None, automatically determined based on
                whether unmapped placeholders are detected during extraction.

        Returns:
            ExtractedDocument containing ordered, 1-indexed ExtractedPage objects.

        Raises:
            EncryptedDocumentError: If the PDF is encrypted or requires a password.
            CorruptedPDFParseError: If the PDF cannot be opened or parsed.
        """
        path: Path = validated_doc.file_path

        try:
            doc = pymupdf.open(path)  # type: ignore[no-untyped-call]
        except Exception as exc:
            raise CorruptedPDFParseError(
                f"Failed to open PDF at '{path}' for parsing: {exc}"
            ) from exc

        with doc:
            if doc.is_encrypted:
                raise EncryptedDocumentError(f"PDF at '{path}' is encrypted or password-protected.")

            pages: list[ExtractedPage] = []
            total_characters = 0
            total_unmapped_symbols = 0

            for page_idx in range(len(doc)):
                # 1-indexed physical page numbers ensure direct parity with user-visible citations
                page_num = page_idx + 1
                page = doc[page_idx]

                raw_text = extract_page_text_with_font_remapping(
                    page,
                    document_id=validated_doc.document_name,
                    page_number=page_num,
                    config_path=self.config_path,
                )
                page_unmapped_count = raw_text.count("[UNMAPPED:")
                total_unmapped_symbols += page_unmapped_count

                cleaned_text = self._clean_page_text(raw_text)
                char_count = len(cleaned_text)

                total_characters += char_count
                pages.append(
                    ExtractedPage(
                        page_number=page_num,
                        text=cleaned_text,
                        character_count=char_count,
                        is_blank=(char_count == 0),
                        has_unmapped_symbols=(page_unmapped_count > 0),
                    )
                )

            # Resolve symbol_font_status
            if symbol_font_status is not None:
                resolved_status = symbol_font_status
            elif total_unmapped_symbols > 0:
                today_str = datetime.now(UTC).strftime("%Y-%m-%d")
                resolved_status = f"unmapped_allowed_{today_str}"
            else:
                resolved_status = "clean"

            return ExtractedDocument(
                document=validated_doc,
                pages=pages,
                total_pages=len(doc),
                total_characters=total_characters,
                symbol_font_status=resolved_status,
                unmapped_symbol_count=total_unmapped_symbols,
            )

    def parse_batch(
        self,
        validated_docs: list[ValidatedDocument],
        allow_unmapped: bool = False,
        report_path: Path | str | None = None,
    ) -> tuple[list[ExtractedDocument], list[ValidatedDocument]]:
        """
        Parse a batch of validated documents with preflight quality gatekeeping.

        Clean documents ingest normally. Flagged (needs_review) documents are individually
        skipped and recorded in a review report unless allow_unmapped=True is specified.

        Args:
            validated_docs: Sequence of ValidatedDocument instances to parse.
            allow_unmapped: If True, force-ingests flagged documents with placeholders.
            report_path: Optional custom path for the review report JSON.

        Returns:
            tuple of (extracted_documents, skipped_documents).
        """
        if not validated_docs:
            return [], []

        doc_paths = [doc.file_path for doc in validated_docs]
        scan_results = preflight_scan(doc_paths, config_path=self.config_path)

        approved_paths, _ = partition_preflight_batch(
            scan_results,
            allow_unmapped=allow_unmapped,
            report_path=report_path,
        )
        approved_set = {p.resolve() for p in approved_paths}

        extracted_docs: list[ExtractedDocument] = []
        skipped_docs: list[ValidatedDocument] = []

        for doc in validated_docs:
            doc_key = str(doc.file_path.resolve())
            if doc.file_path.resolve() in approved_set:
                doc_info = scan_results.get("documents", {}).get(doc_key, {})
                if doc_info.get("status") == "needs_review":
                    today_str = datetime.now(UTC).strftime("%Y-%m-%d")
                    status_flag = f"unmapped_allowed_{today_str}"
                else:
                    status_flag = "clean"

                extracted = self.parse(doc, symbol_font_status=status_flag)
                extracted_docs.append(extracted)
            else:
                logger.warning(
                    "Skipping document '%s' from ingestion batch pending preflight review.",
                    doc.document_name,
                )
                skipped_docs.append(doc)

        return extracted_docs, skipped_docs

    @staticmethod
    def _clean_page_text(text: str) -> str:
        """Normalize line breaks, trim trailing whitespace, and strip null bytes."""
        if not text:
            return ""
        # Null characters can cause issues in downstream JSON/database serializers
        cleaned = text.replace("\x00", "")
        # Trim trailing whitespace per line to avoid trailing artifacts
        cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
        # Collapse excessive blank lines (3 or more) to standard double newlines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
