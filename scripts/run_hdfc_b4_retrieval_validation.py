"""
Live Read-Only B4 Hybrid Retrieval Validation against Weaviate Cloud.

1. Purpose:
   Executes real-time read-only hybrid retrieval queries against the
   canonical DiligenceChunk collection in Weaviate Cloud to validate
   Phase B4 retrieval behavior on the HDFC Bank FY2025-26 Annual Report.

2. Role in Overall RAG Pipeline:
   Validates Phase B4 (WeaviateHybridRetriever) on live data prior to
   LLM reranking (Phase B5).

3. Queries Tested:
   1. "What are the major risks facing HDFC Bank?"
   2. "What was the net interest income?"
   3. "What is the CET1 ratio?"
   4. "What are HDFC Bank's business segments?"
   5. "What does the annual report say about cybersecurity?"
   6. "What was the net interest income?" with company="HDFC Bank"
   7. "What is the CET1 ratio?" with company="A COMPANY THAT DOES NOT EXIST"

4. Outputs:
   Saves full validation report to data/processed/hdfc_b4_retrieval_validation.json.
"""

import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from due_diligence.core.config import get_settings
from due_diligence.retrieval.retriever import WeaviateHybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/processed/hdfc_b4_retrieval_validation.json")

TEST_QUERIES: list[dict[str, Any]] = [
    {
        "id": 1,
        "query": "What are the major risks facing HDFC Bank?",
        "company": None,
        "description": "Standard broad risk query (unfiltered)",
    },
    {
        "id": 2,
        "query": "What was the net interest income?",
        "company": None,
        "description": "Financial metric query (unfiltered)",
    },
    {
        "id": 3,
        "query": "What is the CET1 ratio?",
        "company": None,
        "description": "Regulatory capital metric query (unfiltered)",
    },
    {
        "id": 4,
        "query": "What are HDFC Bank's business segments?",
        "company": None,
        "description": "Segment reporting query (unfiltered)",
    },
    {
        "id": 5,
        "query": "What does the annual report say about cybersecurity?",
        "company": None,
        "description": "Technology and operational risk query (unfiltered)",
    },
    {
        "id": 6,
        "query": "What was the net interest income?",
        "company": "HDFC Bank",
        "description": "Financial metric query with hard company filter",
    },
    {
        "id": 7,
        "query": "What is the CET1 ratio?",
        "company": "A COMPANY THAT DOES NOT EXIST",
        "description": "Regulatory capital query with non-existent company filter (verify zero results)",
    },
]


def run_validation() -> dict[str, Any]:
    """Execute live read-only validation suite using WeaviateHybridRetriever."""
    settings = get_settings()
    logger.info("Initializing WeaviateHybridRetriever (READ-ONLY mode)...")

    retriever = WeaviateHybridRetriever(settings=settings)
    embedding_dimension = retriever.embedding_provider.dimension

    logger.info(
        "Retriever initialized: collection='%s', alpha=%.2f, top_n=%d, embedding_dim=%d",
        retriever.collection_name,
        retriever.alpha,
        retriever.top_n,
        embedding_dimension,
    )

    query_results: list[dict[str, Any]] = []

    try:
        for tq in TEST_QUERIES:
            qid = tq["id"]
            query_str = tq["query"]
            company_filter = tq["company"]
            desc = tq["description"]

            logger.info(
                "Executing Query #%d: '%s' (company=%s)...",
                qid,
                query_str,
                company_filter,
            )

            t0 = time.perf_counter()
            exception_occurred: str | None = None
            raw_results: list[Any] = []

            try:
                raw_results = retriever.retrieve(
                    query=query_str,
                    company=company_filter,
                    top_n=10,
                )
            except Exception as err:  # noqa: BLE001 - one failed query must not abort the audit
                logger.error("Exception during Query #%d: %s", qid, err)
                exception_occurred = f"{type(err).__name__}: {err}"

            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            num_returned = len(raw_results)

            logger.info(
                "Query #%d completed: %d results in %.2f ms (exception=%s)",
                qid,
                num_returned,
                latency_ms,
                exception_occurred,
            )

            # Format top 10 items for inspection
            formatted_items: list[dict[str, Any]] = []
            for rank, r in enumerate(raw_results, start=1):
                is_table = (r.chunk_type == "table") or (r.table_metadata is not None)
                formatted_items.append({
                    "rank": rank,
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "page_numbers": r.page_numbers,
                    "section_heading": r.section_heading,
                    "chunk_type": r.chunk_type,
                    "is_table": is_table,
                    "company": r.company,
                    "text_preview": r.text[:200].replace("\n", " ").strip(),
                })

            query_results.append({
                "query_id": qid,
                "description": desc,
                "query": query_str,
                "company_filter": company_filter,
                "latency_ms": latency_ms,
                "num_results_returned": num_returned,
                "exception_occurred": exception_occurred,
                "results": formatted_items,
            })

    finally:
        retriever.close()

    # Compile validation summary
    q6 = next(q for q in query_results if q["query_id"] == 6)
    q7 = next(q for q in query_results if q["query_id"] == 7)

    q6_verified = (
        q6["num_results_returned"] > 0
        and all(item["company"] == "HDFC Bank" for item in q6["results"])
    )
    q7_verified = q7["num_results_returned"] == 0

    validation_summary: dict[str, Any] = {
        "status": "success" if (q6_verified and q7_verified) else "failed",
        "timestamp": datetime.now(UTC).isoformat(),
        "collection_name": settings.weaviate_url,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": embedding_dimension,
        "hybrid_alpha": settings.hybrid_alpha,
        "retrieval_top_n_requested": 10,
        "total_queries_executed": len(TEST_QUERIES),
        "company_filter_q6_verified": q6_verified,
        "non_existent_company_q7_verified_zero_results": q7_verified,
        "queries": query_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(validation_summary, f, indent=2)

    logger.info("Validation report written to: %s", OUTPUT_PATH)
    return validation_summary


if __name__ == "__main__":
    summary = run_validation()
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Status: {summary['status']}")
    print(f"Embedding Dimension: {summary['embedding_dimension']}")
    print(f"Query 6 (company='HDFC Bank') verified: {summary['company_filter_q6_verified']}")
    print(f"Query 7 (company='A COMPANY THAT DOES NOT EXIST') zero results verified: {summary['non_existent_company_q7_verified_zero_results']}")
    for q in summary["queries"]:
        print(f"\nQuery #{q['query_id']}: '{q['query']}' (company={q['company_filter']})")
        print(f"  Latency: {q['latency_ms']} ms | Results: {q['num_results_returned']} | Exception: {q['exception_occurred']}")
        for r in q["results"][:3]:  # Print top 3 to console
            print(f"    Rank {r['rank']} [{r['chunk_id']}] (score: {r['score']}, pages: {r['page_numbers']}, heading: {r['section_heading']})")
            print(f"      Preview: {r['text_preview'][:120]}...")
