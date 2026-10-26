"""
Batch preflight scanner for symbol font audit across multiple PDF documents.

1. Purpose: Audits batches of PDFs for unmapped symbol font codepoints prior to ingestion.
2. Pipeline Role: Quality gatekeeper ensuring glyph integrity before semantic chunking and embedding.
3. Expected Inputs: Sequence of PDF file paths and optional symbol mappings config path.
4. Outputs: Structured dictionary summarizing document status, unmapped glyphs, and batch totals;
   generates review JSON reports when unmapped codepoints are flagged.
5. Dependencies: PyMuPDF (`pymupdf`), standard library (`json`, `datetime`, `pathlib`, `logging`),
   `due_diligence.ingestion.font_mappings`.
6. Outside Scope: Chunking, embedding generation, text indexing, vector database writes.
"""

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf

from due_diligence.ingestion.font_mappings import (
    get_symbol_font_category,
    is_symbol_font,
    load_symbol_mappings,
    strip_subset_prefix,
)

logger = logging.getLogger(__name__)


class PreflightReviewRequiredError(Exception):
    """Raised when batch ingestion contains documents with unmapped symbol codepoints."""

    def __init__(
        self,
        message: str,
        review_documents: int,
        report_path: Path | str | None = None,
        scan_results: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.review_documents = review_documents
        self.report_path = report_path
        self.scan_results = scan_results or {}


def preflight_scan(
    pdf_paths: Sequence[str | Path],
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Scan a batch of PDFs for unmapped symbol font codepoints.

    Args:
        pdf_paths: List of filesystem paths to PDF files.
        config_path: Optional path to symbol font mappings YAML.

    Returns:
        Structured audit dictionary with per-document and batch summary metrics.
    """
    symbol_map, _ = load_symbol_mappings(config_path)

    documents_result: dict[str, Any] = {}
    clean_count = 0
    review_count = 0
    total_unmapped_in_batch = 0

    for path_item in pdf_paths:
        pdf_path = Path(path_item)
        doc_key = str(pdf_path)

        if not pdf_path.is_file():
            logger.warning("PDF file not found during preflight scan: %s", pdf_path)
            documents_result[doc_key] = {
                "status": "error",
                "error": f"File not found: {pdf_path}",
                "total_pages": 0,
                "unmapped_codepoints": [],
            }
            continue

        try:
            doc = pymupdf.open(pdf_path)  # type: ignore[no-untyped-call]
        except (pymupdf.FileDataError, pymupdf.EmptyFileError, OSError, ValueError) as exc:
            logger.error("Could not open PDF %s for preflight scan: %s", pdf_path, exc)
            documents_result[doc_key] = {
                "status": "error",
                "error": str(exc),
                "total_pages": 0,
                "unmapped_codepoints": [],
            }
            continue

        total_pages = len(doc)
        unmapped_records: dict[tuple[str, int], dict[str, Any]] = {}

        for page_idx in range(total_pages):
            page_num = page_idx + 1  # 1-indexed page
            page = doc[page_idx]
            page_dict = page.get_text("dict")  # type: ignore[no-untyped-call]

            for block in page_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font = span.get("font", "")
                        text = span.get("text", "")
                        if not text or not is_symbol_font(font):
                            continue

                        clean_font = strip_subset_prefix(font)
                        category = get_symbol_font_category(font)

                        for idx, char in enumerate(text):
                            cp = ord(char)
                            if (category, cp) in symbol_map:
                                continue

                            key = (clean_font, cp)
                            if key not in unmapped_records:
                                start = max(0, idx - 20)
                                end = min(len(text), idx + 21)
                                context = text[start:end]

                                unmapped_records[key] = {
                                    "font_name": clean_font,
                                    "codepoint_hex": f"0x{cp:04X}",
                                    "codepoint_dec": cp,
                                    "count": 0,
                                    "page_numbers": [],
                                    "total_pages": 0,
                                    "example_context": context,
                                }

                            rec = unmapped_records[key]
                            rec["count"] += 1
                            if page_num not in rec["page_numbers"]:
                                rec["page_numbers"].append(page_num)

        doc.close()  # type: ignore[no-untyped-call]

        unmapped_list: list[dict[str, Any]] = []
        doc_unmapped_total = 0
        for rec in unmapped_records.values():
            all_pages = rec["page_numbers"]
            rec["total_pages"] = len(all_pages)
            rec["page_numbers"] = all_pages[:5]
            doc_unmapped_total += rec["count"]
            unmapped_list.append(rec)

        if unmapped_list:
            doc_status = "needs_review"
            review_count += 1
            total_unmapped_in_batch += doc_unmapped_total
        else:
            doc_status = "clean"
            clean_count += 1

        documents_result[doc_key] = {
            "status": doc_status,
            "total_pages": total_pages,
            "unmapped_codepoints": unmapped_list,
        }

    return {
        "documents": documents_result,
        "batch_summary": {
            "total_documents": len(pdf_paths),
            "clean_documents": clean_count,
            "review_documents": review_count,
            "total_unmapped_codepoints": total_unmapped_in_batch,
        },
    }


def save_preflight_report(
    scan_results: dict[str, Any],
    output_path: Path | str | None = None,
) -> Path:
    """
    Save structured review report JSON for unmapped codepoint occurrences.

    Args:
        scan_results: Output dictionary from preflight_scan.
        output_path: Optional custom destination path. Defaults to needs_review_<YYYY-MM-DD>.json.

    Returns:
        Path to the written JSON file.
    """
    if output_path is not None:
        target = Path(output_path)
    else:
        today_str = datetime.now(UTC).strftime("%Y-%m-%d")
        target = Path(f"needs_review_{today_str}.json")

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(scan_results, f, indent=2, ensure_ascii=False)

    logger.info("Saved preflight review report to '%s'", target)
    return target


def partition_preflight_batch(
    scan_results: dict[str, Any],
    allow_unmapped: bool = False,
    report_path: Path | str | None = None,
) -> tuple[list[Path], list[Path]]:
    """
    Partition batch documents into approved-for-ingestion vs excluded-for-review.

    Clean documents always proceed to ingestion without requiring allow_unmapped.
    Flagged (needs_review) documents are individually excluded by default and saved to a review report.
    If allow_unmapped is True, flagged documents are also included in approved list.

    Args:
        scan_results: Output dictionary from preflight_scan.
        allow_unmapped: If True, flagged documents are included in approved list.
        report_path: Optional custom path for the review report JSON.

    Returns:
        tuple of (approved_paths, skipped_paths).
    """
    approved: list[Path] = []
    skipped: list[Path] = []

    documents = scan_results.get("documents", {})
    review_docs_exist = False

    for doc_path_str, info in documents.items():
        doc_path = Path(doc_path_str)
        status = info.get("status")

        if status == "clean":
            approved.append(doc_path)
        elif status == "needs_review":
            review_docs_exist = True
            if allow_unmapped:
                approved.append(doc_path)
            else:
                skipped.append(doc_path)
        else:
            # File missing or corrupted
            skipped.append(doc_path)

    if review_docs_exist:
        save_preflight_report(scan_results, output_path=report_path)

    return approved, skipped


def assert_preflight_clean(
    scan_results: dict[str, Any],
    document_path: str | Path | None = None,
    allow_unmapped: bool = False,
    report_path: Path | str | None = None,
) -> None:
    """
    Assert that a document or batch meets preflight quality requirements.

    Args:
        scan_results: Output dictionary from preflight_scan.
        document_path: Optional specific document path to evaluate. If specified,
            only that document's status is checked. If None, checks the batch as a whole.
        allow_unmapped: If True, permits flagged documents with unmapped glyphs.
        report_path: Optional destination for the review report JSON.

    Raises:
        PreflightReviewRequiredError:
            - When document_path is specified and that document requires review (and allow_unmapped=False).
            - When document_path is None and ALL documents in the batch require review (and allow_unmapped=False).
    """
    documents = scan_results.get("documents", {})
    summary = scan_results.get("batch_summary", {})
    review_docs = summary.get("review_documents", 0)
    clean_docs = summary.get("clean_documents", 0)

    # 1. Single-document check
    if document_path is not None:
        doc_key = str(Path(document_path))
        doc_info = documents.get(doc_key, {})
        if doc_info.get("status") == "needs_review" and not allow_unmapped:
            saved_report = save_preflight_report(scan_results, report_path)
            raise PreflightReviewRequiredError(
                f"Document '{doc_key}' requires review before ingestion. "
                f"See review report at '{saved_report}'. Ingestion skipped. "
                "Use allow_unmapped=True to override.",
                review_documents=1,
                report_path=saved_report,
                scan_results=scan_results,
            )
        return

    # 2. Whole-batch check: only raises if NO documents can proceed
    if review_docs > 0:
        saved_report = save_preflight_report(scan_results, report_path)
        if clean_docs == 0 and not allow_unmapped:
            raise PreflightReviewRequiredError(
                f"Batch preflight check failed: all {review_docs} document(s) require review. "
                f"Zero clean documents available to ingest. See report at '{saved_report}'.",
                review_documents=review_docs,
                report_path=saved_report,
                scan_results=scan_results,
            )
        elif not allow_unmapped:
            logger.warning(
                "Batch preflight: %d clean document(s) will proceed to ingestion; "
                "%d document(s) require review and will be skipped. Report saved to '%s'.",
                clean_docs,
                review_docs,
                saved_report,
            )
        else:
            logger.warning(
                "Batch preflight: allow_unmapped=True. All %d document(s) will be ingested.",
                summary.get("total_documents", 0),
            )
