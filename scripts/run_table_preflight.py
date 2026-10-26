"""
Full-document table preflight scanner and quality gate audit.

1. Purpose:
   Executes deterministic table detection, Tier 1 column-projection extraction,
   and quality gate audit across all 678 pages of the HDFC Bank FY2025-26
   annual report. Produces exact machine-readable per-page classification records
   with complete arithmetic reconciliation between Tier 1 passes and Tier 2 routes.

2. Role in Overall RAG Pipeline:
   Acts as the preflight triage stage before ingestion. Separates pure narrative
   pages from table pages, accepts clean tabular structures directly via Tier 1
   local extraction (saving LLM cost and latency), routes degraded or complex
   tables to Tier 2 multimodal vision, and flags multi-page continuation tables
   for cross-page stitching.

3. Inputs:
   - Path to PDF document (default: data/raw/hdfc-bank__annual-report__fy2025-26.pdf)

4. Outputs:
   - JSON artifact containing summary reconciliation and 678 per-page records:
     data/processed/table_preflight_2026-09-25.json

5. Dependencies:
   - pymupdf (fitz)
   - due_diligence.ingestion.table_extractor

6. Non-Responsibilities:
   - Does not perform Tier 2 Gemini Flash API calls (audit & triage only).
   - Does not generate vector embeddings or write to vector stores.
   - Does not construct final RAG chunk elements (handled in elements.py).
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pymupdf

from due_diligence.ingestion.table_extractor import (
    detect_all_table_regions,
    evaluate_quality_gate,
    extract_table_tier1,
)


def evaluate_page(
    page_num: int,
    page: pymupdf.Page,
    prev_page_tables: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Evaluates a single page for tabular structures.
    Rejects 1-column narrative callouts so they are parsed cleanly as prose by layout.py.
    Applies Tier 1 column-projection and comprehensive quality gates.
    """
    raw_boxes = detect_all_table_regions(page)
    boxes: list[tuple[float, float, float, float]] = []
    box_t1_map: dict[tuple[float, float, float, float], dict[str, Any]] = {}

    for b in raw_boxes:
        t_res = extract_table_tier1(page, b)
        if t_res.get("col_count", 0) >= 2:
            boxes.append(b)
            box_t1_map[b] = t_res

    table_detected = len(boxes) > 0
    if not table_detected:
        rec = {
            "page_number": page_num,
            "table_detected": False,
            "table_region_count": 0,
            "tier1_extracted": False,
            "extracted_row_count": 0,
            "extracted_column_count": 0,
            "row_consistency": 0.0,
            "fragmented_word_count": 0,
            "continuation_detected": False,
            "gate_passed": False,
            "tier2_routed": False,
            "failure_reasons": [],
            "primary_failure_reason": None,
            "regions": [],
        }
        return rec, []

    region_results = []
    page_failures = []
    page_is_continuation = False
    page_tier1_extracted = True

    for reg_idx, bbox in enumerate(boxes):
        t_res = box_t1_map[bbox]
        row_count = t_res.get("row_count", 0)
        col_count = t_res.get("col_count", 0)
        if row_count == 0 or col_count == 0:
            page_tier1_extracted = False

        prev_t = prev_page_tables[-1] if (reg_idx == 0 and prev_page_tables) else None
        gate_res = evaluate_quality_gate(t_res, page, prev_table=prev_t)

        is_cont = gate_res["is_continuation"]
        if is_cont:
            page_is_continuation = True

        reg_failures = [f"region_{reg_idx}_{f}" for f in gate_res["failures"]]
        page_failures.extend(reg_failures)

        region_results.append({
            "region_index": reg_idx,
            "bbox": [round(x, 1) for x in bbox],
            "row_count": row_count,
            "col_count": col_count,
            "row_consistency": gate_res["consistency"],
            "fragmented_word_count": gate_res["fragmented_word_count"],
            "merged_cell_count": gate_res["merged_cell_count"],
            "is_continuation": is_cont,
            "gate_passed": gate_res["gate_passed"],
            "failures": reg_failures,
            "t_res": t_res,
        })

    page_gate_passed = (len(page_failures) == 0)
    page_tier2_routed = not page_gate_passed

    # Primary failure category assignment
    # Priority: continuation > col_count > row_count > row_consistency > merged_cells > fragmented_words > divider_cuts_words > header_drift > other
    primary_failure = None
    if page_tier2_routed:
        if page_is_continuation:
            primary_failure = "continuation"
        elif any("col_count" in f for f in page_failures):
            primary_failure = "col_count"
        elif any("row_count" in f for f in page_failures):
            primary_failure = "row_count"
        elif any("row_consistency" in f for f in page_failures):
            primary_failure = "row_consistency"
        elif any("merged_cell" in f for f in page_failures):
            primary_failure = "merged_cells"
        elif any("fragmented_words" in f for f in page_failures):
            primary_failure = "fragmented_words"
        elif any("divider_cuts" in f for f in page_failures):
            primary_failure = "divider_cuts_words"
        elif any("header_drift" in f for f in page_failures):
            primary_failure = "header_drift"
        else:
            primary_failure = "other"

    total_rows = sum(r["row_count"] for r in region_results)
    max_cols = max(r["col_count"] for r in region_results) if region_results else 0
    avg_cons = sum(r["row_consistency"] for r in region_results) / len(region_results) if region_results else 0.0
    total_frags = sum(r["fragmented_word_count"] for r in region_results)

    clean_regions = []
    extracted_tables = []
    for r in region_results:
        extracted_tables.append(r["t_res"])
        cr = dict(r)
        cr.pop("t_res", None)
        clean_regions.append(cr)

    rec = {
        "page_number": page_num,
        "table_detected": True,
        "table_region_count": len(boxes),
        "tier1_extracted": page_tier1_extracted,
        "extracted_row_count": total_rows,
        "extracted_column_count": max_cols,
        "row_consistency": round(avg_cons, 1),
        "fragmented_word_count": total_frags,
        "continuation_detected": page_is_continuation,
        "gate_passed": page_gate_passed,
        "tier2_routed": page_tier2_routed,
        "failure_reasons": page_failures,
        "primary_failure_reason": primary_failure,
        "regions": clean_regions,
    }
    return rec, extracted_tables


def run_full_preflight_audit(pdf_path: Path, output_path: Path) -> dict[str, Any]:
    """
    Executes the 678-page scan, validates all arithmetic identities,
    and writes the machine-readable artifact.
    """
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    start_t = time.perf_counter()

    records = []
    prev_page_tables = None

    for idx in range(total_pages):
        page_num = idx + 1
        page = doc[idx]
        rec, tables = evaluate_page(page_num, page, prev_page_tables)
        prev_page_tables = tables if rec["table_detected"] else None
        records.append(rec)

    duration = time.perf_counter() - start_t
    doc.close()

    pure_narrative_pages = sum(1 for r in records if not r["table_detected"])
    true_table_pages = sum(1 for r in records if r["table_detected"])
    tier1_clean_pass = sum(1 for r in records if r["table_detected"] and r["gate_passed"])
    tier2_routed = sum(1 for r in records if r["table_detected"] and r["tier2_routed"])
    other_unclassified = sum(1 for r in records if r["table_detected"] and not r["gate_passed"] and not r["tier2_routed"])

    primary_counts: dict[str, int] = {}
    for r in records:
        if r["tier2_routed"]:
            p = r["primary_failure_reason"] or "unspecified"
            primary_counts[p] = primary_counts.get(p, 0) + 1

    occurrence_counts: dict[str, int] = {}
    for r in records:
        if r["tier2_routed"]:
            for f in r["failure_reasons"]:
                cat = f.split(" ")[0]
                occurrence_counts[cat] = occurrence_counts.get(cat, 0) + 1

    assert len(records) == total_pages, f"Record count {len(records)} != total pages {total_pages}"
    assert total_pages == true_table_pages + pure_narrative_pages, "Total pages identity violated"
    assert true_table_pages == tier1_clean_pass + tier2_routed + other_unclassified, "Reconciliation identity violated"
    assert sum(primary_counts.values()) == tier2_routed, "Primary failure reasons sum violated"

    summary = {
        "metadata": {
            "document_path": str(pdf_path),
            "total_pages": total_pages,
            "scan_duration_seconds": round(duration, 2),
            "pages_per_second": round(total_pages / duration, 1),
            "scan_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "reconciliation": {
            "formula": "total_pages = true_table_pages + pure_narrative_pages; true_table_pages = tier1_clean_pass + tier2_routed + other_unclassified",
            "total_pages": total_pages,
            "pure_narrative_pages": pure_narrative_pages,
            "true_table_pages": true_table_pages,
            "tier1_clean_pass": tier1_clean_pass,
            "tier2_routed": tier2_routed,
            "other_unclassified": other_unclassified,
            "tier1_clean_pass_pct": round((tier1_clean_pass / true_table_pages) * 100, 2) if true_table_pages else 0.0,
            "tier2_routed_pct": round((tier2_routed / true_table_pages) * 100, 2) if true_table_pages else 0.0,
        },
        "primary_failure_categories": primary_counts,
        "occurrence_failure_counts": occurrence_counts,
        "pages": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Full 678-page table preflight scanner and gate audit.")
    parser.add_argument("--pdf", type=Path, default=Path("data/raw/hdfc-bank__annual-report__fy2025-26.pdf"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/table_preflight_2026-09-25.json"))
    args = parser.parse_args()

    print(f"Starting fresh full-document preflight scan on {args.pdf}...")
    summary = run_full_preflight_audit(args.pdf, args.output)

    recon = summary["reconciliation"]
    print("\n" + "=" * 80)
    print("VERIFICATION A: EXACT TABLE-PAGE ARITHMETIC RECONCILIATION")
    print("=" * 80)
    print(f"Total Pages:                     {recon['total_pages']}")
    print(f"Pure Narrative Pages:            {recon['pure_narrative_pages']} ({(recon['pure_narrative_pages']/recon['total_pages']):.1%})")
    print(f"True Table Pages:                {recon['true_table_pages']} ({(recon['true_table_pages']/recon['total_pages']):.1%})")
    print(f"Tier 1 Clean Pass:               {recon['tier1_clean_pass']} ({recon['tier1_clean_pass_pct']}%)")
    print(f"Tier 2 Routed:                   {recon['tier2_routed']} ({recon['tier2_routed_pct']}%)")
    print(f"Other Unclassified:              {recon['other_unclassified']}")
    print(f"Exact Identity Verified:         {recon['true_table_pages']} == {recon['tier1_clean_pass']} + {recon['tier2_routed']} + {recon['other_unclassified']}")

    print("\n" + "=" * 80)
    print("VERIFICATION B: MUTUALLY EXCLUSIVE PRIMARY FAILURE REASONS")
    print("=" * 80)
    for cat, count in sorted(summary["primary_failure_categories"].items(), key=lambda x: -x[1]):
        print(f"  - {cat:<25}: {count:>4} pages ({count / recon['tier2_routed']:>5.1%})")
    print(f"  Sum of Primary Categories:     {sum(summary['primary_failure_categories'].values())} == {recon['tier2_routed']}")

    print("\nTotal Non-Exclusive Failure Occurrences:")
    for cat, count in sorted(summary["occurrence_failure_counts"].items(), key=lambda x: -x[1]):
        print(f"  - {cat:<25}: {count:>4} occurrences")
    print(f"  Total Occurrences Sum:         {sum(summary['occurrence_failure_counts'].values())}")

    print("\n" + "=" * 80)
    print(f"Artifact saved successfully to: {args.output.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
