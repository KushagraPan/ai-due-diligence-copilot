"""
Operational CLI runner for the Phase B9 Golden Benchmark Evaluation suite.

Executes the complete online diligence pipeline against the live Weaviate Cloud
corpus (1,984 chunks) and evaluates questions against the authoritative golden
benchmark dataset (tests/eval/data/golden_qa_25.json).

Pipeline under test (no stages bypassed):
User Query -> Query Rewriter -> B4 Weaviate Hybrid Retrieval (Top 40)
  -> B5 LLM Reranker (Top 7) -> B6 Context Construction
  -> B7 Grounded Generation (GPT-4o) -> B8 Citation Verification -> DiligenceResponse

Outputs:
- Machine-readable JSON report: data/processed/b9_golden_evaluation_report.json
- Human-readable Markdown report: data/processed/b9_golden_evaluation_report.md
"""

import argparse
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from due_diligence.core.config import get_settings
from due_diligence.evaluation.runner import BenchmarkRunner
from due_diligence.evaluation.schemas import GoldenDataset
from due_diligence.pipeline import DiligencePipeline
from due_diligence.retrieval.weaviate_store import WeaviateStore

logger = logging.getLogger("due_diligence.evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase B9 Golden Benchmark Evaluation against live DiligencePipeline.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/eval/data/golden_qa_25.json"),
        help="Path to curated golden QA JSON dataset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of questions to evaluate (for dry runs or smoke tests)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/processed/b9_golden_evaluation_report.json"),
        help="Path to persist machine-readable JSON evaluation results",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("data/processed/b9_golden_evaluation_report.md"),
        help="Path to persist human-readable Markdown evaluation report",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging level",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    settings = get_settings()

    print("\n" + "=" * 80)
    print("AI DUE DILIGENCE COPILOT — PHASE B9 GOLDEN BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"Weaviate Endpoint: {settings.weaviate_url}")
    print(f"Embedding Model:   {settings.embedding_model}")
    print(
        f"Reranker:          {settings.llm_reranker_model} [single-stage LLM] "
        f"(Top {settings.retrieval_top_n} -> Top {settings.rerank_top_k})"
    )
    print(f"Generation Model:  {settings.generation_model}")
    print(f"Golden Dataset:    {args.dataset}")
    print(f"Limit:             {args.limit or 'All questions'}")
    print(f"Output JSON:       {args.output_json}")
    print(f"Output Markdown:   {args.output_md}")
    print("=" * 80 + "\n")

    # 1. Verify Weaviate Collection Integrity
    store = WeaviateStore(settings=settings)
    collection = store.ensure_collection("DiligenceChunk")
    agg = collection.aggregate.over_all(total_count=True)
    cluster_count = agg.total_count
    store.close()
    logger.info("Verified active Weaviate cluster count: %d objects", cluster_count)
    if cluster_count != 1984:
        logger.warning(
            "Expected 1,984 indexed objects for HDFC Bank report, found %d",
            cluster_count,
        )

    # 2. Load Curated Golden Dataset
    dataset = GoldenDataset.from_file(args.dataset)
    logger.info("Loaded golden dataset with %d questions", len(dataset.questions))

    # 3. Instantiate DiligencePipeline and BenchmarkRunner
    pipeline = DiligencePipeline(settings=settings)
    runner = BenchmarkRunner(pipeline=pipeline, settings=settings)

    # 4. Execute Benchmark Run
    report = runner.run_benchmark(
        dataset=dataset,
        limit=args.limit,
    )

    # 5. Persist Results
    report.to_json_file(args.output_json)
    report.to_markdown_file(args.output_md)
    logger.info("Persisted JSON report to %s", args.output_json)
    logger.info("Persisted Markdown report to %s", args.output_md)

    # Clean up connections
    retriever = pipeline.retrieval_pipeline.retriever
    weaviate_store = getattr(retriever, "_weaviate_store", None)
    if weaviate_store is not None and hasattr(weaviate_store, "close"):
        weaviate_store.close()

    # 6. Display Executive Console Summary
    print("\n" + report.generate_markdown_report())
    print("\n" + "=" * 80)
    print(f"Benchmark evaluation finished successfully in {report.duration_seconds:.2f}s.")
    print(f"JSON Results:     {args.output_json.resolve()}")
    print(f"Markdown Report:  {args.output_md.resolve()}")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
