"""Tests for the source-document verifier."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf
import pytest

from scripts import verify_source


def _write_source(path: Path) -> str:
    path.write_bytes(b"%PDF-1.4\n% synthetic source\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    path = tmp_path / "source.pdf"
    digest = _write_source(path)
    monkeypatch.setattr(verify_source, "EXPECTED_PAGE_COUNT", 2)
    monkeypatch.setattr(verify_source, "_read_page_count", lambda _: 2)
    return path, digest


def test_verify_source_success(
    source_pdf: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, digest = source_pdf
    monkeypatch.setattr(verify_source, "EXPECTED_SHA256", digest)

    result = verify_source.verify_source(path)

    assert result.ok
    assert result.page_count == 2
    assert result.sha256 == digest
    assert result.errors == ()
    assert result.warnings == ()


def test_missing_source_is_fatal(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"

    result = verify_source.verify_source(missing)

    assert not result.ok
    assert result.page_count is None
    assert result.sha256 is None
    assert any("README > Source document" in error for error in result.errors)


def test_unreadable_source_is_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "unreadable.pdf"
    path.write_bytes(b"not readable")

    def raise_os_error(_: Path) -> str:
        raise OSError("permission denied")

    monkeypatch.setattr(verify_source, "_sha256_file", raise_os_error)

    result = verify_source.verify_source(path)

    assert not result.ok
    assert any("Could not read source PDF" in error for error in result.errors)


def test_wrong_page_count_is_fatal(
    source_pdf: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, digest = source_pdf
    monkeypatch.setattr(verify_source, "_read_page_count", lambda _: 3)
    monkeypatch.setattr(verify_source, "EXPECTED_SHA256", digest)

    result = verify_source.verify_source(path)

    assert not result.ok
    assert any("expected 2, found 3" in error for error in result.errors)


def test_629_page_report_gets_specific_hint(
    source_pdf: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, digest = source_pdf
    monkeypatch.setattr(verify_source, "EXPECTED_PAGE_COUNT", 678)
    monkeypatch.setattr(verify_source, "_read_page_count", lambda _: 629)
    monkeypatch.setattr(verify_source, "EXPECTED_SHA256", digest)

    result = verify_source.verify_source(path)

    assert not result.ok
    assert any("629-page standalone" in error for error in result.errors)


def test_placeholder_checksum_warns(
    source_pdf: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = source_pdf
    monkeypatch.setattr(verify_source, "EXPECTED_SHA256", verify_source.PLACEHOLDER_SHA256)

    result = verify_source.verify_source(path)

    assert result.ok
    assert any("not pinned" in warning for warning in result.warnings)


def test_checksum_mismatch_warns_by_default(
    source_pdf: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = source_pdf
    monkeypatch.setattr(verify_source, "EXPECTED_SHA256", "0" * 64)

    result = verify_source.verify_source(path)

    assert result.ok
    assert any("Checksum mismatch" in warning for warning in result.warnings)


def test_checksum_mismatch_strict_is_fatal(
    source_pdf: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = source_pdf
    monkeypatch.setattr(verify_source, "EXPECTED_SHA256", "0" * 64)

    result = verify_source.verify_source(path, strict_checksum=True)

    assert not result.ok
    assert any("Checksum mismatch" in error for error in result.errors)


def test_read_page_count_uses_pymupdf(tmp_path: Path) -> None:
    path = tmp_path / "two-pages.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    doc.save(path)
    doc.close()

    assert verify_source._read_page_count(path) == 2


def test_cli_returns_zero_for_valid_source(
    source_pdf: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, digest = source_pdf
    monkeypatch.setattr(verify_source, "EXPECTED_SHA256", digest)

    assert verify_source.main(["--path", str(path)]) == 0
    assert "OK: source document verified." in capsys.readouterr().out


def test_cli_returns_one_for_missing_source(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.pdf"

    assert verify_source.main(["--path", str(missing)]) == 1
    assert "not found" in capsys.readouterr().err
