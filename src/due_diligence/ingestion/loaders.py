"""
Document validation and ingestion loading.

1. Purpose: Validates file existence, non-emptiness, format integrity, and computes deterministic checksums.
2. Pipeline Role: Serves as the first gate of the RAG ingestion pipeline before document parsing.
3. Expected Inputs: Path-like file reference, optional company name, and optional financial period.
4. Outputs: ValidatedDocument instance with verified metadata and SHA-256 checksum.
5. Dependencies: src.due_diligence.ingestion.document (ValidatedDocument, DocumentType).
6. Outside Scope: Text extraction, layout parsing, table extraction, chunking, and embedding generation.
"""

import hashlib
import mimetypes
from pathlib import Path

from due_diligence.ingestion.document import DocumentType, ValidatedDocument

# Minimum bytes needed to inspect the PDF file signature (%PDF-)
_PDF_MAGIC_BYTES = b"%PDF-"
_CHUNK_SIZE_BYTES = 65536


class DocumentLoadError(Exception):
    """Base exception for all document loading and validation failures."""


class DocumentNotFoundError(DocumentLoadError, FileNotFoundError):
    """Raised when the specified document file does not exist on disk."""


class EmptyDocumentError(DocumentLoadError, ValueError):
    """Raised when the document file has zero bytes."""


class UnsupportedDocumentTypeError(DocumentLoadError, ValueError):
    """Raised when the document format or extension is not supported in the MVP."""


class CorruptedDocumentError(DocumentLoadError, ValueError):
    """Raised when file content signature does not match expected format magic bytes."""


class DocumentLoader:
    """
    Validates and loads corporate diligence documents from disk.

    Enforces format constraints, non-emptiness, and generates deterministic content hashes.
    """

    def __init__(self, supported_types: set[str] | None = None) -> None:
        """Initialize loader with allowed extensions. Defaults to PDF only for MVP."""
        self.supported_extensions = supported_types or {".pdf"}

    def validate_and_load(
        self,
        file_path: str | Path,
        company: str | None = None,
        financial_period: str | None = None,
    ) -> ValidatedDocument:
        """
        Validate document presence, format, and content integrity.

        Args:
            file_path: Absolute or relative filesystem path to the target document.
            company: Optional corporate entity name for provenance tagging.
            financial_period: Optional reporting period (e.g., 'FY2023', 'Q3 2024').

        Returns:
            ValidatedDocument with verified path, size, MIME type, and SHA-256 checksum.

        Raises:
            DocumentNotFoundError: If the file path does not point to an existing file.
            UnsupportedDocumentTypeError: If the file extension is not in supported extensions.
            EmptyDocumentError: If the file is 0 bytes.
            CorruptedDocumentError: If the file header does not contain valid PDF magic bytes.
        """
        path = Path(file_path).resolve()

        if not path.is_file():
            raise DocumentNotFoundError(f"Document not found at path: {path}")

        # Extension check catches misrouted files early before reading disk bytes
        if path.suffix.lower() not in self.supported_extensions:
            raise UnsupportedDocumentTypeError(
                f"Unsupported file format '{path.suffix}'. Supported formats: {sorted(self.supported_extensions)}"
            )

        file_size = path.stat().st_size
        if file_size == 0:
            raise EmptyDocumentError(f"Document at '{path}' is empty (0 bytes).")

        self._verify_pdf_signature(path)
        content_hash = self.compute_sha256(path)

        mime_type, _ = mimetypes.guess_type(path)
        resolved_mime = mime_type or "application/pdf"

        return ValidatedDocument(
            document_name=path.name,
            document_type=DocumentType.PDF,
            file_path=path,
            file_size_bytes=file_size,
            content_hash=content_hash,
            mime_type=resolved_mime,
            company=company,
            financial_period=financial_period,
        )

    @staticmethod
    def _verify_pdf_signature(path: Path) -> None:
        """Inspect leading header bytes to guard against renamed non-PDF files."""
        with open(path, "rb") as f:
            header = f.read(len(_PDF_MAGIC_BYTES))
            if header != _PDF_MAGIC_BYTES:
                raise CorruptedDocumentError(
                    f"File '{path.name}' lacks valid PDF magic bytes header (%PDF-)."
                )

    @staticmethod
    def compute_sha256(path: Path) -> str:
        """Compute SHA-256 in fixed-size chunks to avoid high memory spikes on large filings."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(_CHUNK_SIZE_BYTES):
                sha256.update(chunk)
        return sha256.hexdigest()
