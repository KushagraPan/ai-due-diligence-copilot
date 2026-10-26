"""
Document schema definitions for ingestion, validation, and page extraction.

1. Purpose: Defines domain models and data contracts for validated documents and extracted page text.
2. Pipeline Role: Acts as the validated data boundary between raw files, page extraction, and downstream chunking.
3. Expected Inputs: Validated file attributes and extracted page representations.
4. Outputs: Immutable ValidatedDocument, ExtractedPage, and ExtractedDocument instances.
5. Dependencies: External `pydantic` only.
6. Outside Scope: File I/O, PDF parsing execution, chunking, and embedding generation.
"""

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DocumentType(str, Enum):
    """Supported document formats for due diligence ingestion."""

    PDF = "pdf"


class ValidatedDocument(BaseModel):
    """
    Validated document metadata and provenance record.

    Ensures file integrity and records initial provenance prior to parsing.
    """

    model_config = ConfigDict(frozen=True)

    document_name: str = Field(..., description="Original filename with extension")
    document_type: DocumentType = Field(DocumentType.PDF, description="Validated document format")
    file_path: Path = Field(..., description="Absolute path to the validated source document")
    file_size_bytes: int = Field(..., gt=0, description="File size in bytes (must be non-empty)")
    content_hash: str = Field(
        ..., min_length=64, max_length=64, description="Deterministic SHA-256 checksum"
    )
    mime_type: str = Field(default="application/pdf", description="Detected or verified MIME type")
    company: str | None = Field(None, description="Target company or entity name if specified")
    financial_period: str | None = Field(
        None, description="Fiscal year or reporting period if specified"
    )
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of successful validation",
    )


class ExtractedPage(BaseModel):
    """
    Extracted plain text and metadata for a single document page.

    Preserves 1-indexed page numbering for exact downstream citations.
    """

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(..., ge=1, description="1-indexed page number in the source file")
    text: str = Field(..., description="Normalized text content extracted from the page")
    character_count: int = Field(..., ge=0, description="Total characters in the extracted text")
    is_blank: bool = Field(..., description="True if the page contains no extractable text")
    has_unmapped_symbols: bool = Field(
        default=False,
        description="True if unmapped symbol font placeholders are present on this page",
    )


class ExtractedDocument(BaseModel):
    """
    Complete page-level extraction record for a validated document.

    Maintains source provenance, ordered sequence of extracted pages, and symbol font audit status.
    """

    model_config = ConfigDict(frozen=True)

    document: ValidatedDocument = Field(..., description="Source document validation record")
    pages: list[ExtractedPage] = Field(..., description="Ordered list of extracted pages")
    total_pages: int = Field(..., ge=0, description="Total physical pages in the document")
    total_characters: int = Field(..., ge=0, description="Sum of characters across all pages")
    symbol_font_status: str = Field(
        default="clean",
        description="Audit status of symbol fonts ('clean', 'reviewed_<date>', or 'unmapped_allowed_<date>')",
    )
    unmapped_symbol_count: int = Field(
        default=0,
        ge=0,
        description="Total occurrences of unmapped symbol placeholders across all pages",
    )
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when extraction completed",
    )

    def get_page(self, page_number: int) -> ExtractedPage | None:
        """Retrieve an extracted page by its 1-indexed page number."""
        if 1 <= page_number <= len(self.pages):
            return self.pages[page_number - 1]
        return None
