"""
Unit tests for document validation and loading.

Tests the DocumentLoader and ValidatedDocument schema across valid inputs and realistic edge cases.
"""

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from due_diligence.ingestion import (
    CorruptedDocumentError,
    DocumentLoader,
    DocumentLoadError,
    DocumentNotFoundError,
    DocumentType,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
    ValidatedDocument,
)


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Minimal valid PDF byte sequence including standard header."""
    return b"%PDF-1.4\n%mock sample binary stream for test\n%%EOF"


@pytest.fixture
def valid_pdf_file(tmp_path: Path, sample_pdf_content: bytes) -> Path:
    """Creates a temporary valid PDF file on disk."""
    pdf_path = tmp_path / "sample_filing.pdf"
    pdf_path.write_bytes(sample_pdf_content)
    return pdf_path


def test_validate_and_load_valid_pdf(
    valid_pdf_file: Path, sample_pdf_content: bytes
) -> None:
    """Verifies that a valid PDF produces a complete ValidatedDocument record."""
    loader = DocumentLoader()
    doc = loader.validate_and_load(
        valid_pdf_file,
        company="Acme Corp",
        financial_period="FY2023",
    )

    assert isinstance(doc, ValidatedDocument)
    assert doc.document_name == "sample_filing.pdf"
    assert doc.document_type == DocumentType.PDF
    assert doc.file_path == valid_pdf_file.resolve()
    assert doc.file_size_bytes == len(sample_pdf_content)
    assert doc.company == "Acme Corp"
    assert doc.financial_period == "FY2023"
    assert doc.mime_type == "application/pdf"

    expected_hash = hashlib.sha256(sample_pdf_content).hexdigest()
    assert doc.content_hash == expected_hash
    assert len(doc.content_hash) == 64


def test_validate_nonexistent_file_raises_error(tmp_path: Path) -> None:
    """Verifies that attempting to load a non-existent path raises DocumentNotFoundError."""
    loader = DocumentLoader()
    missing_path = tmp_path / "nonexistent.pdf"

    with pytest.raises(DocumentNotFoundError) as exc_info:
        loader.validate_and_load(missing_path)

    assert "Document not found" in str(exc_info.value)
    assert issubclass(DocumentNotFoundError, FileNotFoundError)
    assert issubclass(DocumentNotFoundError, DocumentLoadError)


def test_validate_empty_file_raises_error(tmp_path: Path) -> None:
    """Verifies that 0-byte files are rejected with EmptyDocumentError."""
    loader = DocumentLoader()
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    with pytest.raises(EmptyDocumentError) as exc_info:
        loader.validate_and_load(empty_file)

    assert "is empty (0 bytes)" in str(exc_info.value)
    assert issubclass(EmptyDocumentError, ValueError)
    assert issubclass(EmptyDocumentError, DocumentLoadError)


def test_validate_unsupported_format_raises_error(tmp_path: Path) -> None:
    """Verifies that unsupported file formats are rejected before reading."""
    loader = DocumentLoader()
    docx_file = tmp_path / "filing.docx"
    docx_file.write_bytes(b"PK\x03\x04mock docx content")

    with pytest.raises(UnsupportedDocumentTypeError) as exc_info:
        loader.validate_and_load(docx_file)

    assert "Unsupported file format '.docx'" in str(exc_info.value)
    assert issubclass(UnsupportedDocumentTypeError, ValueError)
    assert issubclass(UnsupportedDocumentTypeError, DocumentLoadError)


def test_validate_renamed_non_pdf_file_raises_corrupted_error(tmp_path: Path) -> None:
    """Verifies that a non-PDF file renamed with a .pdf extension is flagged via magic bytes."""
    loader = DocumentLoader()
    fake_pdf = tmp_path / "fake_report.pdf"
    fake_pdf.write_bytes(b"NOT A REAL PDF CONTENT STREAM")

    with pytest.raises(CorruptedDocumentError) as exc_info:
        loader.validate_and_load(fake_pdf)

    assert "lacks valid PDF magic bytes header" in str(exc_info.value)
    assert issubclass(CorruptedDocumentError, ValueError)
    assert issubclass(CorruptedDocumentError, DocumentLoadError)


def test_deterministic_sha256_checksum(tmp_path: Path, sample_pdf_content: bytes) -> None:
    """Verifies that hash generation is deterministic and unique across distinct content."""
    pdf1 = tmp_path / "report1.pdf"
    pdf1.write_bytes(sample_pdf_content)

    pdf2 = tmp_path / "report2.pdf"
    pdf2.write_bytes(sample_pdf_content + b"\n%extra byte")

    loader = DocumentLoader()
    hash1_a = loader.compute_sha256(pdf1)
    hash1_b = loader.compute_sha256(pdf1)
    hash2 = loader.compute_sha256(pdf2)

    assert hash1_a == hash1_b
    assert hash1_a != hash2


def test_validated_document_immutability(valid_pdf_file: Path) -> None:
    """Verifies that ValidatedDocument instances are frozen and cannot be mutated."""
    loader = DocumentLoader()
    doc = loader.validate_and_load(valid_pdf_file)

    with pytest.raises(ValidationError):
        # Direct mutation should be prevented by ConfigDict(frozen=True)
        doc.company = "Different Entity"
