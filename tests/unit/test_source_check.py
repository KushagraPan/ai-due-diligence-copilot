"""Tests for the shared source-document guard."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from due_diligence.core.source_check import (
    SourceDocumentError,
    ensure_source_document,
    exit_with_source_error,
)
from due_diligence.ingestion.parser import PDFParser


def test_ensure_source_document_returns_existing_file(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4\n")

    assert ensure_source_document(source) == source


def test_missing_source_error_points_to_readme(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"

    with pytest.raises(SourceDocumentError, match="README > Source document") as exc_info:
        ensure_source_document(missing)

    assert str(missing) in str(exc_info.value)


def test_parser_uses_shared_guard(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"
    fake_doc = SimpleNamespace(file_path=missing)

    with pytest.raises(SourceDocumentError):
        PDFParser().parse(fake_doc)  # type: ignore[arg-type]


def test_exit_with_source_error_prints_and_exits(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        exit_with_source_error(SourceDocumentError("missing source"))

    assert exc_info.value.code == 1
    assert "Error: missing source" in capsys.readouterr().err
