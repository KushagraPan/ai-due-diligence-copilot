"""
Verify the local HDFC Bank source PDF before ingestion.

The report is not committed because of size and copyright. This script makes the
expected edition explicit: 678 pages and a pinned SHA-256. It is standalone
(PyMuPDF + hashlib) and does not contact Weaviate or any LLM.

Inputs: --path, defaulting to data/raw/hdfc-bank__annual-report__fy2025-26.pdf.
Outputs: a human-readable verification report and a process exit code.

A missing file or wrong page count is fatal. A checksum mismatch is a warning by
default because HDFC may re-upload the same content; pass --strict-checksum to
make it fatal. A placeholder checksum also warns instead of failing.

Dependencies: due_diligence.core.source_check, pymupdf, hashlib.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from due_diligence.core.source_check import DEFAULT_SOURCE_PATH

EXPECTED_PAGE_COUNT = 678
EXPECTED_SHA256 = "92de3c75b6fc05afe8731e97705c9d3534cc192e677af02f4b325e3d82eafca6"
PLACEHOLDER_SHA256 = "REPLACE_WITH_REAL_SHA256"
STANDALONE_REPORT_PAGE_COUNT = 629
_CHUNK_SIZE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class SourceVerificationResult:
    """Outcome of a source-document verification run."""

    path: Path
    page_count: int | None
    sha256: str | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when no fatal verification error occurred."""
        return not self.errors


def _sha256_file(path: Path) -> str:
    """Compute a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_page_count(path: Path) -> int:
    """Read the PDF page count with PyMuPDF."""
    with pymupdf.open(str(path)) as doc:  # type: ignore[no-untyped-call]
        return int(doc.page_count)


def verify_source(
    path: Path | str | None = None,
    *,
    strict_checksum: bool = False,
) -> SourceVerificationResult:
    """
    Verify existence, page count, and checksum of the source PDF.

    Args:
        path: Optional override. Defaults to DEFAULT_SOURCE_PATH.
        strict_checksum: Treat a checksum mismatch as fatal. Defaults to False
            because a re-uploaded PDF can have different bytes while preserving
            the same page count and content.

    Returns:
        SourceVerificationResult containing fatal errors and non-fatal warnings.
    """
    source_path = Path(path) if path is not None else DEFAULT_SOURCE_PATH
    errors: list[str] = []
    warnings: list[str] = []

    if not source_path.is_file():
        errors.append(
            f"Source PDF not found at '{source_path}'. "
            "Place the HDFC Bank FY2025-26 annual report there. "
            "See README > Source document for the exact edition, page count, and checksum."
        )
        return SourceVerificationResult(
            path=source_path,
            page_count=None,
            sha256=None,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    try:
        sha256 = _sha256_file(source_path)
    except OSError as exc:
        errors.append(f"Could not read source PDF '{source_path}': {exc}")
        return SourceVerificationResult(
            path=source_path,
            page_count=None,
            sha256=None,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    try:
        page_count = _read_page_count(source_path)
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, OSError, ValueError) as exc:
        errors.append(f"Could not read PDF '{source_path}': {exc}")
        return SourceVerificationResult(
            path=source_path,
            page_count=None,
            sha256=sha256,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    if page_count != EXPECTED_PAGE_COUNT:
        message = (
            f"Page count mismatch: expected {EXPECTED_PAGE_COUNT}, found {page_count}."
        )
        if page_count == STANDALONE_REPORT_PAGE_COUNT:
            message += (
                " This is the 629-page standalone Integrated Annual Report published on "
                "HDFC Bank's website. This repository uses the 678-page exchange filing; "
                "its PDF page numbers are 25 higher. See README > Source document."
            )
        errors.append(message)

    if EXPECTED_SHA256 == PLACEHOLDER_SHA256:
        warnings.append(
            "Checksum is not pinned: EXPECTED_SHA256 is still the placeholder. "
            "Page count verification is still enforced."
        )
    elif sha256 != EXPECTED_SHA256.lower():
        message = (
            f"Checksum mismatch: expected {EXPECTED_SHA256.lower()}, found {sha256}. "
            "A re-uploaded PDF can have different bytes. "
            "Pass --strict-checksum to make this fatal."
        )
        if strict_checksum:
            errors.append(message)
        else:
            warnings.append(message)

    return SourceVerificationResult(
        path=source_path,
        page_count=page_count,
        sha256=sha256,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Verify the local HDFC Bank FY2025-26 source PDF."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help=f"Source PDF path (default: {DEFAULT_SOURCE_PATH})",
    )
    parser.add_argument(
        "--strict-checksum",
        action="store_true",
        help="Fail on a SHA-256 mismatch instead of warning.",
    )
    args = parser.parse_args(argv)

    result = verify_source(args.path, strict_checksum=args.strict_checksum)
    print(f"Source PDF: {result.path}")
    if result.page_count is not None:
        print(f"Pages:      {result.page_count} (expected {EXPECTED_PAGE_COUNT})")
    if result.sha256 is not None:
        print(f"SHA-256:    {result.sha256}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    for error in result.errors:
        print(f"Error: {error}", file=sys.stderr)

    if result.ok:
        print("OK: source document verified.")
        return 0

    print("FAILED: source document verification failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
