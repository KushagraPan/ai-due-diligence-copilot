"""
CLI command for batch preflight scanning of diligence PDF documents.

Audits PDF files for unmapped symbol font codepoints prior to ingestion.
Exits with non-zero status when unmapped codepoints are detected unless --allow-unmapped is set.
"""

import argparse
import sys
from pathlib import Path

from due_diligence.ingestion import (
    preflight_scan,
    save_preflight_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch preflight scanner for symbol font codepoints in diligence PDFs."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["data/raw"],
        help="One or more PDF file paths or directories to scan (default: data/raw).",
    )
    parser.add_argument(
        "--allow-unmapped",
        action="store_true",
        help="Exit with status 0 even if unmapped codepoints are detected.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Custom path to symbol font mappings YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Custom output path for review report JSON.",
    )
    return parser.parse_args()


def collect_pdf_paths(input_paths: list[str]) -> list[Path]:
    """Collect all PDF files from given file paths, directory paths, or glob patterns."""
    pdf_paths: list[Path] = []
    for item in input_paths:
        path = Path(item)
        if path.is_dir():
            pdf_paths.extend(sorted(path.glob("**/*.pdf")))
        elif path.is_file() and path.suffix.lower() == ".pdf":
            pdf_paths.append(path)
        else:
            # Try globbing if path string contains wildcards
            parent = path.parent if path.parent.is_dir() else Path(".")
            pattern = path.name
            matches = list(parent.glob(pattern))
            pdf_paths.extend([m for m in matches if m.is_file() and m.suffix.lower() == ".pdf"])
    return sorted(set(pdf_paths))


def main() -> None:
    args = parse_args()

    # Ensure UTF-8 output on Windows terminal
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        reconfigure_fn = getattr(sys.stdout, "reconfigure", None)
        if callable(reconfigure_fn):
            try:
                reconfigure_fn(encoding="utf-8")
            except (AttributeError, ValueError):
                pass

    pdf_paths = collect_pdf_paths(args.paths)
    if not pdf_paths:
        print(f"No PDF documents found matching: {args.paths}")
        sys.exit(0)

    print(f"Scanning {len(pdf_paths)} document(s) for symbol font mappings...")
    scan_results = preflight_scan(pdf_paths, config_path=args.config)
    summary = scan_results.get("batch_summary", {})

    total_docs = summary.get("total_documents", 0)
    clean_docs = summary.get("clean_documents", 0)
    review_docs = summary.get("review_documents", 0)
    total_unmapped = summary.get("total_unmapped_codepoints", 0)

    print("\n" + "=" * 80)
    print("PREFLIGHT SCAN SUMMARY")
    print("=" * 80)
    print(f"Total Documents:           {total_docs}")
    print(f"Clean Documents:           {clean_docs}")
    print(f"Documents Needing Review:  {review_docs}")
    print(f"Total Unmapped Codepoints: {total_unmapped}")
    print("=" * 80)

    if review_docs > 0:
        print("\nUNMAPPED CODEPOINTS DETECTED:")
        print("-" * 80)
        docs = scan_results.get("documents", {})
        for doc_path, doc_info in docs.items():
            if doc_info.get("status") == "needs_review":
                print(f"\nDocument: {doc_path} ({doc_info.get('total_pages', 0)} pages)")
                for unmapped in doc_info.get("unmapped_codepoints", []):
                    font = unmapped.get("font_name")
                    hex_cp = unmapped.get("codepoint_hex")
                    count = unmapped.get("count")
                    pages = unmapped.get("page_numbers", [])
                    total_p = unmapped.get("total_pages", len(pages))
                    context = unmapped.get("example_context", "").strip()
                    pages_str = ", ".join(map(str, pages))
                    if total_p > len(pages):
                        pages_str += f" (+{total_p - len(pages)} more)"
                    print(
                        f"  - Font: {font:<20} CP: {hex_cp:<8} Count: {count:<4} Pages: [{pages_str}]"
                    )
                    if context:
                        print(f"    Sample: {context!r}")

        saved_report = save_preflight_report(scan_results, output_path=args.output)
        print("\n" + "-" * 80)
        print(f"Review report saved to: {saved_report.resolve()}")
        print("-" * 80)

        print(f"\n{clean_docs} document(s) clean and ready for ingestion.")
        print(f"{review_docs} document(s) require review and will be skipped during ingestion.")

        if not args.allow_unmapped:
            print("\nPreflight check: 1 or more documents require review.")
            print("To force-ingest flagged documents with placeholders anyway, pass --allow-unmapped.")
            sys.exit(1)
        else:
            print("\nWARNING: Proceeding with force-ingest because --allow-unmapped was specified.")
            sys.exit(0)
    else:
        print("\nAll documents clean. Zero unmapped symbol-font codepoints detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
