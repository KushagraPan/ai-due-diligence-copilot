"""
Validation script running DocumentChunker on the real HDFC Bank Annual Report.
Computes corpus chunking statistics, validates the provenance invariant,
and extracts representative chunk examples.
"""

import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

import pymupdf

from due_diligence.core.source_check import (
    SourceDocumentError,
    ensure_source_document,
    exit_with_source_error,
)
from due_diligence.ingestion.chunker import (
    ChunkingConfig,
    DocumentChunker,
    count_tokens,
    verify_chunking_provenance,
)
from due_diligence.ingestion.elements import DocumentElement, Table
from due_diligence.ingestion.layout import LayoutConfig, reconcile_page_elements
from due_diligence.ingestion.table_extractor import extract_tables_from_page

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CORPUS_PDF = Path("data/raw/hdfc-bank__annual-report__fy2025-26.pdf")


def main() -> None:
    ensure_source_document(CORPUS_PDF)

    logger.info("Opening PDF: %s", CORPUS_PDF)
    doc = pymupdf.open(str(CORPUS_PDF))
    total_pages = len(doc)
    doc_name = CORPUS_PDF.name

    logger.info("Extracting DocumentElement stream across %d pages...", total_pages)
    t0 = time.time()

    all_elements: list[DocumentElement] = []
    heading_stack: list[tuple[int, str]] = []
    prev_tables: list[Table] | None = None
    layout_cfg = LayoutConfig()

    for p_idx in range(total_pages):
        page_num = p_idx + 1
        page = doc[p_idx]

        tables = extract_tables_from_page(
            page,
            page_number=page_num,
            document_name=doc_name,
            prev_page_tables=prev_tables,
            enable_tier2=False,
        )

        elements, _, heading_stack = reconcile_page_elements(
            page=page,
            page_number=page_num,
            document_name=doc_name,
            tables=tables,
            prev_page_tables=prev_tables,
            initial_heading_stack=heading_stack,
            config=layout_cfg,
        )

        all_elements.extend(elements)
        prev_tables = tables if tables else None

        if page_num % 100 == 0 or page_num == total_pages:
            logger.info("Processed %d / %d pages (%d elements accumulated)...", page_num, total_pages, len(all_elements))

    t_extract = time.time() - t0
    logger.info("Extracted %d DocumentElements in %.2f seconds.", len(all_elements), t_extract)

    # Run Chunking
    logger.info("Running DocumentChunker...")
    t1 = time.time()
    config = ChunkingConfig(
        target_max_tokens=600,
        hard_max_tokens=750,
        narrative_overlap_tokens=50,
        table_max_tokens=1000,
    )
    chunker = DocumentChunker(config)
    chunks = chunker.chunk_document_elements(all_elements)
    t_chunk = time.time() - t1
    logger.info("Generated %d Chunks in %.2f seconds.", len(chunks), t_chunk)

    # Compute Statistics
    narrative_chunks = [c for c in chunks if c.chunk_type == "narrative"]
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    list_chunks = [c for c in chunks if c.chunk_type == "list"]
    mixed_chunks = [c for c in chunks if c.chunk_type == "mixed"]

    token_counts = [count_tokens(c.text) for c in chunks]
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    med_tokens = round(statistics.median(token_counts), 2) if token_counts else 0
    mean_tokens = round(statistics.mean(token_counts), 2) if token_counts else 0

    overlap_chunks = [c for c in chunks if c.overlap_tokens > 0]
    num_overlap_chunks = len(overlap_chunks)
    avg_overlap_tokens = (
        round(statistics.mean(c.overlap_tokens for c in overlap_chunks), 2)
        if overlap_chunks
        else 0.0
    )

    # Table slices & oversized tables
    table_group_ids = {c.table_group_id for c in table_chunks if c.table_group_id}
    oversized_table_ids = set()
    table_slices_count = 0
    table_slice_tokens: list[int] = []

    for t_id in table_group_ids:
        slices = [c for c in table_chunks if c.table_group_id == t_id]
        if len(slices) > 1:
            oversized_table_ids.add(t_id)
            table_slices_count += len(slices)
            for s in slices:
                table_slice_tokens.append(count_tokens(s.text))

    max_table_slice_tokens = max(table_slice_tokens) if table_slice_tokens else 0

    # Chunks exceeding normal limits
    # Normal limits: 750 for narrative/mixed/list, 1000 for table
    exceeding_chunks = []
    for c in chunks:
        toks = count_tokens(c.text)
        if (c.chunk_type in ("narrative", "mixed", "list") and toks > config.hard_max_tokens) or (
            c.chunk_type == "table" and toks > config.table_max_tokens
        ):
            exceeding_chunks.append((c.chunk_id, c.chunk_type, toks))

    # Provenance Validation
    logger.info("Validating provenance invariant...")
    is_valid, errors = verify_chunking_provenance(all_elements, chunks)

    # Find representative examples
    # 1. Normal narrative
    normal_narrative = next((c for c in narrative_chunks if 100 <= count_tokens(c.text) <= 400 and c.overlap_tokens == 0), narrative_chunks[0] if narrative_chunks else None)
    # 2. Heading + narrative
    heading_narrative = next((c for c in narrative_chunks if c.section_heading and len(c.heading_path) > 1 and 100 <= count_tokens(c.text) <= 400), narrative_chunks[0] if narrative_chunks else None)
    # 3. List
    sample_list = next((c for c in list_chunks if len(c.source_element_ids) >= 3), list_chunks[0] if list_chunks else None)
    # 4. Normal financial table
    normal_table = next((c for c in table_chunks if c.table_slice_count == 1 and 100 <= count_tokens(c.text) <= 600), table_chunks[0] if table_chunks else None)
    # 5. Oversized financial table
    oversized_example_slices = []
    if oversized_table_ids:
        sample_ot_id = next(iter(oversized_table_ids))
        oversized_example_slices = [c for c in table_chunks if c.table_group_id == sample_ot_id]
    # 6. Narrative chunk split with overlap (Chunk A and Chunk B)
    overlap_pair = None
    for i in range(len(chunks) - 1):
        cA = chunks[i]
        cB = chunks[i + 1]
        if cA.chunk_type == "narrative" and cB.chunk_type == "narrative" and cB.overlap_tokens > 0:
            overlap_pair = (cA, cB)
            break
    # 7. Content continuing across a page boundary
    cross_page_chunk = next((c for c in chunks if len(c.page_numbers) > 1), None)

    summary = {
        "total_input_elements": len(all_elements),
        "total_output_chunks": len(chunks),
        "narrative_chunks": len(narrative_chunks),
        "table_chunks": len(table_chunks),
        "list_chunks": len(list_chunks),
        "mixed_chunks": len(mixed_chunks),
        "min_chunk_tokens": min_tokens,
        "median_chunk_tokens": med_tokens,
        "mean_chunk_tokens": mean_tokens,
        "max_chunk_tokens": max_tokens,
        "num_narrative_overlap_chunks": num_overlap_chunks,
        "avg_narrative_overlap_tokens": avg_overlap_tokens,
        "num_oversized_tables": len(oversized_table_ids),
        "num_oversized_table_slices": table_slices_count,
        "max_table_slice_tokens": max_table_slice_tokens,
        "chunks_exceeding_limits_count": len(exceeding_chunks),
        "provenance_valid": is_valid,
        "provenance_errors_count": len(errors),
    }

    out_path = Path("data/processed/hdfc_chunking_validation_2026-09-25.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def chunk_to_dict(c: DocumentChunker | Any) -> dict[str, Any]:
        if not c:
            return {}
        return {
            "chunk_id": c.chunk_id,
            "chunk_type": c.chunk_type,
            "page_numbers": c.page_numbers,
            "section_heading": c.section_heading,
            "heading_path": c.heading_path,
            "source_element_ids": c.source_element_ids,
            "overlap_tokens": c.overlap_tokens,
            "table_metadata": c.table_metadata,
            "token_count": count_tokens(c.text),
            "text": c.text,
        }

    report_payload = {
        "summary": summary,
        "examples": {
            "normal_narrative": chunk_to_dict(normal_narrative),
            "heading_narrative": chunk_to_dict(heading_narrative),
            "list": chunk_to_dict(sample_list),
            "normal_table": chunk_to_dict(normal_table),
            "oversized_table_slices": [chunk_to_dict(s) for s in oversized_example_slices],
            "overlap_pair": {
                "chunk_a": chunk_to_dict(overlap_pair[0]) if overlap_pair else None,
                "chunk_b": chunk_to_dict(overlap_pair[1]) if overlap_pair else None,
            },
            "cross_page_chunk": chunk_to_dict(cross_page_chunk),
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    logger.info("Validation report written to %s", out_path)
    print("=== SUMMARY METRICS ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except SourceDocumentError as exc:
        exit_with_source_error(exc)
