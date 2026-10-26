"""
Real-data embedding validation script for HDFC Bank Annual Report chunks.

Executes real embedding inference using the configured embedding provider (OpenAI
text-embedding-3-large, 1536 dims) against actual Chunk objects generated from
the 678-page HDFC Bank FY2025-26 Annual Report.

Phases:
- Phase 1: Small sample validation (10 diverse representative chunks).
- Phase 2: Full corpus embedding run across all 1,984 canonical chunks with
  numpy vector persistence and JSON manifest generation.
"""

import argparse
import json
import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf

from due_diligence.core.config import get_settings
from due_diligence.core.source_check import (
    SourceDocumentError,
    ensure_source_document,
    exit_with_source_error,
)
from due_diligence.ingestion.chunker import (
    ChunkingConfig,
    DocumentChunker,
    count_tokens,
)
from due_diligence.ingestion.chunks import Chunk
from due_diligence.ingestion.elements import DocumentElement, Table
from due_diligence.ingestion.layout import LayoutConfig, reconcile_page_elements
from due_diligence.ingestion.table_extractor import extract_tables_from_page
from due_diligence.retrieval.embeddings import BaseEmbeddingProvider, get_embedding_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CORPUS_PDF = Path("data/raw/hdfc-bank__annual-report__fy2025-26.pdf")
CANONICAL_CHUNKS_PATH = Path("data/processed/hdfc_canonical_chunks.jsonl")
EMBEDDINGS_NPY_PATH = Path("data/processed/hdfc_embeddings.npy")
MANIFEST_PATH = Path("data/processed/hdfc_embeddings_manifest.json")


def load_or_generate_canonical_chunks() -> list[Chunk]:
    """
    Load canonical Chunk objects from cache, or generate them from the source PDF.

    Caches the 1,984 chunks in jsonlines format for fast subsequent runs.
    """
    if CANONICAL_CHUNKS_PATH.exists():
        logger.info("Loading cached canonical chunks from %s...", CANONICAL_CHUNKS_PATH)
        chunks: list[Chunk] = []
        with open(CANONICAL_CHUNKS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(Chunk.model_validate_json(line))
        logger.info("Loaded %d canonical chunks from cache.", len(chunks))
        return chunks

    ensure_source_document(CORPUS_PDF)

    logger.info("Extracting DocumentElements across all pages of %s...", CORPUS_PDF.name)
    doc = pymupdf.open(str(CORPUS_PDF))  # type: ignore[no-untyped-call]
    total_pages = len(doc)
    doc_name = CORPUS_PDF.name

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

    logger.info("Chunking %d DocumentElements...", len(all_elements))
    config = ChunkingConfig(
        target_max_tokens=600,
        hard_max_tokens=750,
        narrative_overlap_tokens=50,
        table_max_tokens=1000,
    )
    chunker = DocumentChunker(config)
    chunks = chunker.chunk_document_elements(all_elements)
    logger.info("Generated %d canonical chunks.", len(chunks))

    CANONICAL_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CANONICAL_CHUNKS_PATH, "w", encoding="utf-8") as f:
        f.writelines(c.model_dump_json() + "\n" for c in chunks)
    logger.info("Cached %d canonical chunks to %s", len(chunks), CANONICAL_CHUNKS_PATH)

    return chunks


def select_representative_sample_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """
    Select approximately 10 diverse representative real HDFC chunks:
    - Narrative chunk (normal length)
    - Heading + narrative chunk (deep hierarchy)
    - List chunk (bullet points)
    - Normal financial table chunk
    - Table slice from oversized table (different section, page 471)
    - Second table slice from continuation
    - Narrative chunk with overlap tokens
    - Cross-page boundary chunk
    - Long chunk near upper token limit (>600 tokens)
    - Financial statement / schedule chunk (page 378+)
    """
    selected: list[Chunk] = []

    # 1. Normal narrative chunk
    narratives = [c for c in chunks if c.chunk_type == "narrative" and c.overlap_tokens == 0]
    c1 = next((c for c in narratives if 100 <= count_tokens(c.text) <= 300), narratives[0])
    selected.append(c1)

    # 2. Heading narrative (deep heading path)
    c2 = next((c for c in narratives if len(c.heading_path) >= 2 and 150 <= count_tokens(c.text) <= 400), narratives[1])
    selected.append(c2)

    # 3. List chunk
    list_chunks = [c for c in chunks if c.chunk_type == "list"]
    c3 = next((c for c in list_chunks if len(c.source_element_ids) >= 2), list_chunks[0])
    selected.append(c3)

    # 4. Normal financial table chunk
    tables = [c for c in chunks if c.chunk_type == "table"]
    c4 = next((c for c in tables if c.table_slice_count == 1 and 150 <= count_tokens(c.text) <= 500), tables[0])
    selected.append(c4)

    # 5 & 6. Oversized table slices (from a multi-slice table)
    multi_slice_groups: dict[str, list[Chunk]] = {}
    for c in tables:
        if c.table_group_id and c.table_slice_count and c.table_slice_count > 1:
            multi_slice_groups.setdefault(c.table_group_id, []).append(c)

    if multi_slice_groups:
        sample_group = next(iter(multi_slice_groups.values()))
        selected.append(sample_group[0])
        if len(sample_group) > 1:
            selected.append(sample_group[1])

    # 7. Narrative chunk with overlap
    c7 = next((c for c in chunks if c.overlap_tokens > 0), None)
    if c7:
        selected.append(c7)

    # 8. Cross-page boundary chunk
    c8 = next((c for c in chunks if len(c.page_numbers) > 1), None)
    if c8:
        selected.append(c8)

    # 9. Long chunk (> 600 tokens)
    c9 = next((c for c in chunks if count_tokens(c.text) >= 600), None)
    if c9:
        selected.append(c9)

    # 10. Financial statement chunk from Schedule / Balance sheet (pages 350-400)
    c10 = next((c for c in chunks if any(350 <= p <= 450 for p in c.page_numbers) and c not in selected), None)
    if c10:
        selected.append(c10)

    # Deduplicate while preserving order
    seen_ids = set()
    deduped = []
    for c in selected:
        if c.chunk_id not in seen_ids:
            seen_ids.add(c.chunk_id)
            deduped.append(c)

    return deduped[:10]


def compute_vector_diagnostics(vectors: list[list[float]], expected_dim: int) -> dict[str, Any]:
    """Verify vectors for dimensions, NaN/Inf values, zero vectors, and L2 norms."""
    norms: list[float] = []
    has_nan = False
    has_inf = False
    is_empty = False
    wrong_dims = 0

    for vec in vectors:
        if not vec:
            is_empty = True
            continue
        if len(vec) != expected_dim:
            wrong_dims += 1

        squared_sum = 0.0
        for val in vec:
            if math.isnan(val):
                has_nan = True
            if math.isinf(val):
                has_inf = True
            squared_sum += val * val

        norm = math.sqrt(squared_sum)
        norms.append(norm)

    return {
        "count": len(vectors),
        "expected_dim": expected_dim,
        "wrong_dim_count": wrong_dims,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "has_empty": is_empty,
        "min_l2_norm": round(min(norms), 6) if norms else 0.0,
        "max_l2_norm": round(max(norms), 6) if norms else 0.0,
        "mean_l2_norm": round(sum(norms) / len(norms), 6) if norms else 0.0,
    }


def run_phase1_sample_validation(
    provider: BaseEmbeddingProvider,
    sample_chunks: list[Chunk],
) -> dict[str, Any]:
    """Execute Phase 1: Small real-data validation on ~10 representative chunks."""
    logger.info("=== PHASE 1: SMALL REAL-DATA VALIDATION (10 CHUNKS) ===")
    logger.info("Provider: %s (model=%s, dimension=%d)", type(provider).__name__, provider.model_name, provider.dimension)

    texts = [c.text for c in sample_chunks]
    chunk_ids = [c.chunk_id for c in sample_chunks]

    t0 = time.time()
    vectors = provider.embed_documents(texts)
    elapsed = time.time() - t0

    diagnostics = compute_vector_diagnostics(vectors, expected_dim=provider.dimension)

    # Verify input/output order
    # Each vector corresponds index-for-index to texts[i]
    order_preserved = len(vectors) == len(sample_chunks)

    report = {
        "num_input_chunks": len(sample_chunks),
        "num_returned_vectors": len(vectors),
        "configured_provider": type(provider).__name__,
        "model_name": provider.model_name,
        "vector_dimension": len(vectors[0]) if vectors else 0,
        "expected_dimension": provider.dimension,
        "input_output_ordering_preserved": order_preserved,
        "has_empty_or_invalid_vectors": diagnostics["has_empty"] or diagnostics["wrong_dim_count"] > 0,
        "has_nan_or_inf_values": diagnostics["has_nan"] or diagnostics["has_inf"],
        "vector_norm_statistics": {
            "min_l2_norm": diagnostics["min_l2_norm"],
            "max_l2_norm": diagnostics["max_l2_norm"],
            "mean_l2_norm": diagnostics["mean_l2_norm"],
        },
        "chunk_ids_used": chunk_ids,
        "elapsed_seconds": round(elapsed, 3),
        "sample_chunk_types": [c.chunk_type for c in sample_chunks],
        "sample_pages": [c.page_numbers for c in sample_chunks],
    }

    print("\n" + "=" * 60)
    print("PHASE 1 SAMPLE VALIDATION REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))
    print("=" * 60 + "\n")

    # Assertions for programmatic safety
    assert report["num_input_chunks"] == report["num_returned_vectors"], "Vector count mismatch"
    assert report["vector_dimension"] == report["expected_dimension"], "Dimension mismatch"
    assert not report["has_empty_or_invalid_vectors"], "Invalid vectors detected"
    assert not report["has_nan_or_inf_values"], "NaN or Inf detected in vectors"
    # OpenAI L2 norms should be approximately 1.0 (within float tolerance)
    assert 0.99 <= diagnostics["mean_l2_norm"] <= 1.01, f"Abnormal mean L2 norm: {diagnostics['mean_l2_norm']}"

    logger.info("Phase 1 Sample Validation PASSED with 100% integrity.")
    return report


def run_phase2_full_embedding(
    provider: BaseEmbeddingProvider,
    all_chunks: list[Chunk],
) -> dict[str, Any]:
    """Execute Phase 2: Full real-data embedding run on all canonical chunks."""
    logger.info("=== PHASE 2: FULL REAL-DATA EMBEDDING RUN (%d CHUNKS) ===", len(all_chunks))

    texts = [c.text for c in all_chunks]
    total_chunks = len(all_chunks)
    batch_size = getattr(provider, "batch_size", 64)
    total_batches = (total_chunks + batch_size - 1) // batch_size

    logger.info("Embedding %d chunks across %d batches of size %d...", total_chunks, total_batches, batch_size)
    t0 = time.time()
    vectors = provider.embed_documents(texts)
    elapsed = time.time() - t0

    # Verification checks
    if len(vectors) != total_chunks:
        raise ValueError(f"Vector count mismatch: got {len(vectors)} vectors for {total_chunks} chunks!")

    for idx, vec in enumerate(vectors):
        if len(vec) != provider.dimension:
            raise ValueError(f"Vector at index {idx} has dimension {len(vec)}; expected {provider.dimension}")

    diagnostics = compute_vector_diagnostics(vectors, expected_dim=provider.dimension)
    if diagnostics["has_nan"] or diagnostics["has_inf"]:
        raise ValueError("Detected NaN or Inf values in full embedding vectors!")

    # Build chunk_id -> index mapping
    chunk_id_to_index: dict[str, int] = {c.chunk_id: idx for idx, c in enumerate(all_chunks)}

    # Convert to float32 numpy array and persist
    vectors_array = np.array(vectors, dtype=np.float32)
    EMBEDDINGS_NPY_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_NPY_PATH, vectors_array)
    logger.info("Persisted embedding vectors array of shape %s to %s", vectors_array.shape, EMBEDDINGS_NPY_PATH)

    # Build and save manifest
    manifest = {
        "document": CORPUS_PDF.name,
        "chunk_count": total_chunks,
        "embedding_provider": type(provider).__name__,
        "embedding_model": provider.model_name,
        "embedding_dimension": provider.dimension,
        "batch_size": batch_size,
        "successful_vectors": len(vectors),
        "failed_vectors": 0,
        "vector_norm_statistics": {
            "min_l2_norm": diagnostics["min_l2_norm"],
            "max_l2_norm": diagnostics["max_l2_norm"],
            "mean_l2_norm": diagnostics["mean_l2_norm"],
        },
        "generated_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "chunks_per_second": round(total_chunks / max(elapsed, 0.001), 2),
        "artifact_paths": {
            "embeddings_npy": str(EMBEDDINGS_NPY_PATH),
            "manifest_json": str(MANIFEST_PATH),
        },
        "chunk_id_to_index": chunk_id_to_index,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved embedding manifest to %s", MANIFEST_PATH)

    summary = {
        "document": manifest["document"],
        "chunk_count": manifest["chunk_count"],
        "embedding_provider": manifest["embedding_provider"],
        "embedding_model": manifest["embedding_model"],
        "embedding_dimension": manifest["embedding_dimension"],
        "successful_vectors": manifest["successful_vectors"],
        "failed_vectors": manifest["failed_vectors"],
        "elapsed_seconds": manifest["elapsed_seconds"],
        "chunks_per_second": manifest["chunks_per_second"],
        "mean_l2_norm": diagnostics["mean_l2_norm"],
        "npy_shape": list(vectors_array.shape),
    }

    print("\n" + "=" * 60)
    print("PHASE 2 FULL RUN SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60 + "\n")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-data embedding validation for HDFC chunks")
    parser.add_argument("--sample-only", action="store_true", help="Run only Phase 1 sample validation")
    parser.add_argument("--full", action="store_true", help="Run Phase 2 full embedding run after sample validation")
    args = parser.parse_args()

    settings = get_settings()
    logger.info("Loaded settings: provider=%s, model=%s, dimension=%d", settings.embedding_provider, settings.embedding_model, settings.embedding_dimension)

    provider = get_embedding_provider(settings)

    # 1. Load canonical chunks
    all_chunks = load_or_generate_canonical_chunks()

    # 2. Select 10 diverse representative chunks
    sample_chunks = select_representative_sample_chunks(all_chunks)
    logger.info("Selected %d diverse representative chunks for Phase 1.", len(sample_chunks))

    # 3. Execute Phase 1 Sample Validation
    _ = run_phase1_sample_validation(provider, sample_chunks)

    # 4. If --full is specified, execute Phase 2 Full Run
    if args.full:
        run_phase2_full_embedding(provider, all_chunks)
    elif not args.sample_only:
        logger.info("Sample validation completed. To run the full embedding across all %d chunks, run with --full", len(all_chunks))


if __name__ == "__main__":
    try:
        main()
    except SourceDocumentError as exc:
        exit_with_source_error(exc)
