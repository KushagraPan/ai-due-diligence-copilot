"""
Full-document layout preflight scanner and reconciliation audit.

1. Purpose:
   Executes layout-aware text extraction, spatial partitioning, multi-column detection,
   heading classification, list item detection, and character reconciliation across
   all 678 pages of the HDFC Bank FY2025-26 annual report.

2. Role in Overall RAG Pipeline:
   Acts as the layout reconciliation audit before semantic chunking. Verifies that
   every source text block across the entire corpus is owned by exactly one category,
   diagnoses character reconciliation deltas, detects multi-column reading orders,
   and outputs a machine-readable preflight artifact.

3. Inputs:
   - Path to PDF document (default: data/raw/hdfc-bank__annual-report__fy2025-26.pdf)
   - Optional output JSON path (default: data/processed/layout_preflight_2026-09-25.json)

4. Outputs:
   - Persistent JSON artifact containing corpus summary and 678 per-page records:
     data/processed/layout_preflight_2026-09-25.json

5. Dependencies:
   - pymupdf (fitz)
   - due_diligence.ingestion.layout (reconcile_page_elements, LayoutConfig, PageLayoutReport)
   - due_diligence.ingestion.table_extractor (extract_tables_from_page)
   - due_diligence.ingestion.elements (Table)

6. Non-Responsibilities:
   - Does not perform Tier 2 Gemini vision LLM calls (deterministic local audit).
   - Does not generate vector embeddings or write to Weaviate.
   - Does not split elements into chunk tokens.
"""

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf

from due_diligence.core.source_check import (
    SourceDocumentError,
    ensure_source_document,
    exit_with_source_error,
)
from due_diligence.ingestion.elements import Table
from due_diligence.ingestion.layout import LayoutConfig, reconcile_page_elements
from due_diligence.ingestion.table_extractor import extract_tables_from_page

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_layout_preflight(
    pdf_path: Path | str,
    output_path: Path | str,
    tolerance_pct: float = 5.0,
) -> dict[str, Any]:
    """
    Runs full-document layout reconciliation audit across all pages.
    Generates per-page records and corpus-level summary metrics.
    """
    path = ensure_source_document(pdf_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(path))  # type: ignore[no-untyped-call]
    total_pages = len(doc)
    doc_name = path.name

    logger.info("Starting layout preflight scan across %d pages of '%s'...", total_pages, doc_name)
    start_time = time.time()

    config = LayoutConfig(reconciliation_tolerance_pct=tolerance_pct)
    page_reports: list[dict[str, Any]] = []

    heading_stack: list[tuple[int, str]] = []
    prev_tables: list[Table] | None = None

    # Corpus accumulators
    pages_exceeding_tolerance = 0
    total_tables = 0
    total_headings = 0
    total_list_items = 0
    total_narrative_blocks = 0
    candidate_multi_column_pages = 0
    pages_with_ownership_conflicts = 0
    pages_with_unowned_content = 0
    pages_needing_review = 0

    mc_page_examples: list[dict[str, Any]] = []

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]

        # Extract local Tier 1 tables with continuation context
        tables = extract_tables_from_page(
            page,
            page_number=page_num,
            document_name=doc_name,
            prev_page_tables=prev_tables,
            enable_tier2=False,
        )

        _, report, heading_stack = reconcile_page_elements(
            page=page,
            page_number=page_num,
            document_name=doc_name,
            tables=tables,
            prev_page_tables=prev_tables,
            initial_heading_stack=heading_stack,
            config=config,
        )

        prev_tables = tables if tables else None

        rep_dict = report.model_dump()
        page_reports.append(rep_dict)

        # Accumulate metrics
        if report.reconciliation_delta_pct > tolerance_pct:
            pages_exceeding_tolerance += 1
        total_tables += report.table_count
        total_headings += report.heading_count
        total_list_items += report.list_item_count
        total_narrative_blocks += report.narrative_count

        if report.multi_column:
            candidate_multi_column_pages += 1
            if len(mc_page_examples) < 10:
                mc_page_examples.append({
                    "page_number": page_num,
                    "columns": report.column_count,
                })

        if report.multiply_owned_block_count > 0:
            pages_with_ownership_conflicts += 1
        if report.unowned_block_count > 0:
            pages_with_unowned_content += 1
        if report.needs_review:
            pages_needing_review += 1

        if page_num % 100 == 0 or page_num == total_pages:
            elapsed = time.time() - start_time
            logger.info("Processed %d/%d pages (%.1fs elapsed)...", page_num, total_pages, elapsed)

    elapsed_total = time.time() - start_time

    summary = {
        "document_name": doc_name,
        "scanned_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(elapsed_total, 2),
        "total_pages": total_pages,
        "pages_exceeding_tolerance": pages_exceeding_tolerance,
        "reconciliation_tolerance_pct": tolerance_pct,
        "total_tables": total_tables,
        "total_headings": total_headings,
        "total_list_items": total_list_items,
        "total_narrative_blocks": total_narrative_blocks,
        "candidate_multi_column_pages": candidate_multi_column_pages,
        "candidate_multi_column_pct": round((candidate_multi_column_pages / total_pages) * 100, 2),
        "pages_with_ownership_conflicts": pages_with_ownership_conflicts,
        "pages_with_unowned_content": pages_with_unowned_content,
        "pages_needing_review": pages_needing_review,
        "multi_column_sample": mc_page_examples,
    }

    output_data = {
        "summary": summary,
        "pages": page_reports,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Layout preflight scan complete. Saved report to '%s'.", out_file)
    return output_data


def main() -> None:
    """CLI entrypoint for layout preflight scan."""
    parser = argparse.ArgumentParser(description="Full-document layout preflight scanner and reconciliation audit.")
    parser.add_argument(
        "--pdf-path",
        type=str,
        default="data/raw/hdfc-bank__annual-report__fy2025-26.pdf",
        help="Path to raw PDF document",
    )
    today_str = datetime.now(UTC).strftime("%Y-%m-%d")
    parser.add_argument(
        "--output-path",
        type=str,
        default=f"data/processed/layout_preflight_{today_str}.json",
        help="Path to output JSON artifact",
    )
    parser.add_argument(
        "--tolerance-pct",
        type=float,
        default=5.0,
        help="Reconciliation character delta tolerance percentage (default: 5.0)",
    )

    args = parser.parse_args()
    res = run_layout_preflight(args.pdf_path, args.output_path, args.tolerance_pct)
    summary = res["summary"]

    print("\n" + "=" * 60)
    print("LAYOUT PREFLIGHT RECONCILIATION SUMMARY")
    print("=" * 60)
    print(f"Total Pages:                     {summary['total_pages']}")
    print(f"Elapsed Time:                    {summary['elapsed_seconds']}s")
    print(f"Total Tables:                    {summary['total_tables']}")
    print(f"Total Headings:                  {summary['total_headings']}")
    print(f"Total List Items:                {summary['total_list_items']}")
    print(f"Total Narrative Blocks:          {summary['total_narrative_blocks']}")
    print(f"Multi-Column Pages:              {summary['candidate_multi_column_pages']} ({summary['candidate_multi_column_pct']}%)")
    print(f"Pages with Ownership Conflicts:  {summary['pages_with_ownership_conflicts']}")
    print(f"Pages with Unowned Content:      {summary['pages_with_unowned_content']}")
    print(f"Pages Exceeding Delta Tolerance: {summary['pages_exceeding_tolerance']}")
    print(f"Pages Needing Review:            {summary['pages_needing_review']}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except SourceDocumentError as exc:
        exit_with_source_error(exc)
