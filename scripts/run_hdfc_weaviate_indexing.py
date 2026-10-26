"""
Real-data Weaviate Cloud Indexing Script for HDFC Bank Annual Report.

1. Purpose:
   Indexes the pre-computed canonical HDFC Bank chunks and OpenAI embeddings
   into Weaviate Cloud under the DiligenceChunk collection.

2. Role in Overall RAG Pipeline:
   Acts as the operational bridge between offline embedding persistence (Phase B2)
   and online hybrid retrieval (Phase B4).

3. Expected Inputs:
   - Canonical chunks: data/processed/hdfc_canonical_chunks.jsonl (1,984 chunks)
   - Dense embeddings: data/processed/hdfc_embeddings.npy (1984 x 1536 float32)
   - Embeddings manifest: data/processed/hdfc_embeddings_manifest.json
   - Source document: data/raw/hdfc-bank__annual-report__fy2025-26.pdf

4. Outputs:
   - Indexed data objects in Weaviate Cloud collection 'DiligenceChunk'.
   - Detailed execution report: data/processed/hdfc_weaviate_indexing_report.json.

5. Dependencies:
   - due_diligence.retrieval.weaviate_store (WeaviateStore, validation functions)
   - due_diligence.ingestion.loader (DocumentLoader)
   - due_diligence.core.config (get_settings)

6. Non-Responsibilities:
   - Does NOT call the OpenAI API or regenerate vectors.
   - Does NOT perform search or reranking queries.
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from due_diligence.core.config import get_settings
from due_diligence.ingestion import DocumentLoader
from due_diligence.retrieval.weaviate_store import (
    DEFAULT_COLLECTION_NAME,
    WeaviateAlignmentError,
    WeaviateBatchIndexError,
    WeaviateConnectionError,
    WeaviateStore,
    WeaviateVerificationError,
    validate_pre_indexing_artifacts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

CORPUS_PDF = Path("data/raw/hdfc-bank__annual-report__fy2025-26.pdf")
CANONICAL_CHUNKS_PATH = Path("data/processed/hdfc_canonical_chunks.jsonl")
EMBEDDINGS_NPY_PATH = Path("data/processed/hdfc_embeddings.npy")
MANIFEST_PATH = Path("data/processed/hdfc_embeddings_manifest.json")
REPORT_PATH = Path("data/processed/hdfc_weaviate_indexing_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index pre-computed HDFC Bank chunks and embeddings into Weaviate Cloud."
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Target Weaviate collection name (default: {DEFAULT_COLLECTION_NAME}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of objects per batch request (default: 100).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        default=False,
        help="Explicitly delete and recreate collection before indexing (DANGEROUS).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        default=False,
        help="Skip indexing and run only post-indexing health verification.",
    )
    return parser.parse_args()


def run_indexing(args: argparse.Namespace) -> None:
    settings = get_settings()
    logger.info("Initializing Weaviate Cloud Indexer for HDFC Bank Annual Report...")
    logger.info("Target collection: %s", args.collection_name)
    logger.info("Configured embedding dimension: %d", settings.embedding_dimension)
    logger.info("Bring Your Own Vectors (BYOV): Enabled (OpenAI API will NOT be contacted).")

    # Step 1: Validate pre-indexing artifacts locally
    logger.info("Validating local artifacts and 10 pre-indexing invariants...")
    try:
        chunks, embeddings, manifest = validate_pre_indexing_artifacts(
            chunks_path=CANONICAL_CHUNKS_PATH,
            embeddings_path=EMBEDDINGS_NPY_PATH,
            manifest_path=MANIFEST_PATH,
            expected_dimension=settings.embedding_dimension,
        )
    except WeaviateAlignmentError as err:
        logger.error("Pre-indexing artifact validation failed: %s", err)
        sys.exit(1)

    # Step 2: Validate and load source document provenance
    if not CORPUS_PDF.exists():
        logger.error("Source document not found at %s", CORPUS_PDF)
        sys.exit(1)

    logger.info("Loading source document provenance from %s...", CORPUS_PDF)
    loader = DocumentLoader()
    doc = loader.validate_and_load(
        CORPUS_PDF,
        company="HDFC Bank",
        financial_period="FY2025-26",
    )
    logger.info("Validated document: %s (hash: %s...)", doc.document_name, doc.content_hash[:16])

    # Step 3: Select 5 representative probe chunk IDs for verification
    probes: list[str] = [
        chunks[0].chunk_id,  # First chunk
        chunks[-1].chunk_id,  # Last chunk
        chunks[len(chunks) // 2].chunk_id,  # Middle chunk
    ]
    # First table chunk
    for c in chunks:
        if c.chunk_type == "table":
            probes.append(c.chunk_id)
            break
    # Long narrative chunk
    narrative_chunks = [c for c in chunks if c.chunk_type == "narrative"]
    if narrative_chunks:
        longest_narrative = max(narrative_chunks, key=lambda c: len(c.text))
        probes.append(longest_narrative.chunk_id)

    probes = list(dict.fromkeys(probes))  # preserve order, unique
    logger.info("Selected %d representative probe chunks for verification.", len(probes))

    # Step 4: Execute indexing in WeaviateStore
    indexing_summary: dict[str, Any] = {}
    health_report: dict[str, Any] = {}

    try:
        with WeaviateStore(settings=settings) as store:
            if not args.verify_only:
                logger.info(
                    "Executing batch indexing into Weaviate Cloud (recreate=%s, batch_size=%d)...",
                    args.recreate,
                    args.batch_size,
                )
                indexing_summary = store.index_canonical_corpus(
                    chunks_path=CANONICAL_CHUNKS_PATH,
                    embeddings_path=EMBEDDINGS_NPY_PATH,
                    manifest_path=MANIFEST_PATH,
                    document=doc,
                    collection_name=args.collection_name,
                    batch_size=args.batch_size,
                    recreate_collection=args.recreate,
                )
            else:
                logger.info("Skipping indexing (--verify-only flag specified).")

            # Step 5: Verify collection health and probe chunk integrity
            logger.info("Running post-indexing health verification on '%s'...", args.collection_name)
            health_report = store.verify_collection_health(
                collection_name=args.collection_name,
                expected_count=len(chunks),
                probe_chunk_ids=probes,
            )
            logger.info("Health verification passed: Total objects = %d", health_report["total_objects"])

    except WeaviateConnectionError as err:
        logger.error("Weaviate Cloud connection error: %s", err)
        sys.exit(2)
    except WeaviateBatchIndexError as err:
        logger.error("Weaviate batch indexing error: %s", err)
        sys.exit(3)
    except WeaviateVerificationError as err:
        logger.error("Weaviate post-indexing verification failed: %s", err)
        sys.exit(4)
    except Exception:
        logger.exception("Unexpected error during Weaviate indexing.")
        sys.exit(5)

    # Step 6: Generate structured output report
    final_report = {
        "status": "success",
        "timestamp": datetime.now(UTC).isoformat(),
        "collection_name": args.collection_name,
        "document_provenance": {
            "document_name": doc.document_name,
            "company": doc.company,
            "financial_period": doc.financial_period,
            "content_hash": doc.content_hash,
            "file_size_bytes": doc.file_size_bytes,
        },
        "pre_indexing_validation": {
            "chunks_count": len(chunks),
            "embeddings_shape": list(embeddings.shape),
            "manifest_model": manifest.get("embedding_model"),
            "manifest_dimension": manifest.get("embedding_dimension"),
            "invariants_checked": 10,
            "all_invariants_passed": True,
        },
        "indexing_execution": indexing_summary if not args.verify_only else {"skipped": True},
        "health_verification": health_report,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    logger.info("Saved indexing report to %s", REPORT_PATH)

    # Step 7: Print console report
    print("\n" + "=" * 70)
    print("  WEAVIATE CLOUD INDEXING & VERIFICATION COMPLETE (PHASE B3)")
    print("=" * 70)
    print(f"  Collection Name     : {args.collection_name}")
    print(f"  Document            : {doc.document_name}")
    print(f"  Company             : {doc.company} ({doc.financial_period})")
    print(f"  Total Chunks        : {len(chunks):,}")
    print(f"  Total Vectors       : {embeddings.shape[0]:,} (dim: {embeddings.shape[1]})")
    if not args.verify_only:
        print(f"  Indexed Objects     : {indexing_summary.get('total_objects_indexed', 0):,}")
        print(f"  Batch Size          : {indexing_summary.get('batch_size', 0)}")
        print(f"  Batches Executed    : {indexing_summary.get('total_batches', 0)}")
        print(f"  Elapsed Time        : {indexing_summary.get('elapsed_seconds', 0):.2f}s")
        print(f"  Throughput          : {indexing_summary.get('throughput_obj_per_sec', 0):.1f} obj/sec")
    print(f"  Cluster Object Count: {health_report.get('total_objects', 0):,}")
    print(f"  Probes Verified     : {health_report.get('probes_verified_count', 0)}")
    print(f"  Cluster Health      : {health_report.get('status', 'unknown').upper()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_indexing(parse_args())
