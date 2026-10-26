"""
Shared guard for commands that require the source PDF.

The annual report is intentionally not committed. This helper turns a missing
source file into one clear, actionable error instead of a raw traceback.

Inputs: an optional path to the source PDF (defaults to the canonical HDFC file).
Outputs: the path when it exists; raises SourceDocumentError otherwise.
Dependencies: pathlib only.

Page count and checksum verification live in scripts/verify_source.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn

DEFAULT_SOURCE_PATH = Path("data/raw/hdfc-bank__annual-report__fy2025-26.pdf")
README_SOURCE_ANCHOR = "README > Source document"


class SourceDocumentError(RuntimeError):
    """Raised when the canonical source PDF is missing."""


def ensure_source_document(path: Path | str | None = None) -> Path:
    """
    Return the source PDF path if it exists.

    Args:
        path: Optional override. Defaults to DEFAULT_SOURCE_PATH.

    Raises:
        SourceDocumentError: If the path does not point to an existing file.
    """
    source_path = Path(path) if path is not None else DEFAULT_SOURCE_PATH
    if not source_path.is_file():
        raise SourceDocumentError(
            f"Source PDF not found at '{source_path}'. "
            "Place the HDFC Bank FY2025-26 annual report there. "
            f"See {README_SOURCE_ANCHOR} for the exact edition, page count, and checksum."
        )
    return source_path


def exit_with_source_error(exc: SourceDocumentError) -> NoReturn:
    """Print a source-document error to stderr and exit with status 1."""
    print(f"Error: {exc}", file=sys.stderr)
    raise SystemExit(1)
