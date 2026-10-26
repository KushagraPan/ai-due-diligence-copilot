"""
Execution runner for the full 25-question Golden Benchmark against live DiligencePipeline (B4-B8).

Executes the complete online diligence pipeline against the live Weaviate Cloud
HDFC Bank corpus (1,984 chunks) and evaluates every question against the authoritative
ground-truth expectations in tests/eval/data/golden_qa_25.json.

Pipeline under test:
User Query -> Query Rewriter -> B4 Weaviate Hybrid Retrieval (Top 40)
  -> B5 LLM Reranker (Top 7) -> B6 EvidenceContextBuilder
  -> B7 GroundedGenerator (GPT-4o) -> B8 CitationVerifier -> DiligenceResponse

Evaluator:
DiligenceEvaluator scoring Retrieval Recall@40, Rerank Recall@7, Fact Coverage,
Citation Precision, and Uncertainty Accuracy.

Persists full machine-readable results to:
data/processed/golden_benchmark_results_2026-09-26.json
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from due_diligence.core.config import get_settings
from due_diligence.evaluation.evaluator import DiligenceEvaluator
from due_diligence.evaluation.schemas import GoldenDataset, PipelineQueryEval
from due_diligence.pipeline import DiligencePipeline
from due_diligence.retrieval.query_rewriter import ConversationTurn
from due_diligence.retrieval.weaviate_store import WeaviateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path("tests/eval/data/golden_qa_25.json")
OUTPUT_REPORT_PATH = Path("data/processed/golden_benchmark_results_2026-09-26.json")


def run_benchmark() -> None:
    """Execute the full 25-question golden evaluation suite."""
    settings = get_settings()

    print("\n" + "=" * 80)
    print("AI DUE DILIGENCE CO-PILOT — 25-QUESTION GOLDEN BENCHMARK RUNNER")
    print("=" * 80)
    print(f"Weaviate Endpoint: {settings.weaviate_url}")
    print(f"Embedding Model:   {settings.embedding_model}")
    print(f"Reranker Model:    {settings.llm_reranker_model} [single-stage LLM] (Top {settings.retrieval_top_n} -> Top {settings.rerank_top_k})")
    print(f"Generation Model:  {settings.generation_model}")
    print(f"Golden Dataset:    {GOLDEN_DATASET_PATH}")
    print("=" * 80 + "\n")

    # 1. Verify Weaviate Collection Integrity (Read-Only)
    store = WeaviateStore(settings=settings)
    collection = store.ensure_collection("DiligenceChunk")
    agg = collection.aggregate.over_all(total_count=True)
    cluster_count = agg.total_count
    logger.info("Verified active Weaviate cluster count: %d objects", cluster_count)
    assert cluster_count == 1984, f"Expected 1,984 objects, found {cluster_count}"

    # 2. Load Golden Dataset
    dataset = GoldenDataset.from_file(GOLDEN_DATASET_PATH)
    logger.info("Loaded golden dataset with %d questions", len(dataset.questions))

    # 3. Instantiate DiligencePipeline once (reused across all 25 queries)
    pipeline = DiligencePipeline(settings=settings)
    evaluator = DiligenceEvaluator()

    query_evals: list[PipelineQueryEval] = []
    per_question_records: list[dict[str, Any]] = []

    total_start_time = time.perf_counter()

    for idx, question in enumerate(dataset.questions, start=1):
        print("\n" + "-" * 80)
        print(f"[{idx}/25] {question.id}: {question.question}")
        print(f"Category: {question.category} | Expected Pages: {question.expected_pages}")
        if question.is_uncertain:
            print("Target Behavior: NEGATIVE CONTROL / EXPLICIT UNCERTAINTY EXPECTED")
        print("-" * 80)

        # Prepare conversation history for conversational queries (e.g. Q23)
        history_turns: list[ConversationTurn] | None = None
        if question.conversation_history:
            history_turns = [
                ConversationTurn(role=item["role"], content=item["content"])
                for item in question.conversation_history
            ]

        q_start = time.perf_counter()
        try:
            response = pipeline.run(
                user_query=question.question,
                conversation_history=history_turns,
                company="HDFC Bank",
                top_n=40,
                top_k=7,
            )
        except Exception:
            logger.exception("Pipeline execution failed for question %s", question.id)
            raise

        q_latency = time.perf_counter() - q_start

        # Evaluate response
        b4_raw = response.pipeline_result.raw_candidates
        eval_result = evaluator.evaluate_pipeline_response(
            question=question,
            response=response,
            retrieved_candidates=b4_raw,
        )
        query_evals.append(eval_result)

        # Extract diagnostic fields
        b4_pages = eval_result.retrieval.retrieved_pages if eval_result.retrieval else []
        b5_pages = eval_result.rerank.reranked_pages if eval_result.rerank else []
        cited_pages = eval_result.generation.cited_pages if eval_result.generation else []
        cov_ratio = eval_result.generation.coverage_ratio if eval_result.generation else 0.0
        cit_prec = eval_result.generation.citation_precision if eval_result.generation else 0.0
        unc_matched = eval_result.generation.uncertainty_matched if eval_result.generation else False

        ret_hit = eval_result.retrieval.hit_at_n if eval_result.retrieval else False
        rer_hit = eval_result.rerank.hit_at_k if eval_result.rerank else False

        print(f"  Retrieval (B4 Top 40): {'HIT' if ret_hit else 'MISS'} | Pages: {b4_pages[:8]}...")
        print(f"  Rerank    (B5 Top 7):  {'HIT' if rer_hit else 'MISS'} | Pages: {b5_pages}")
        print(f"  Answer:   {response.answer_text[:220]}...")
        print(f"  Facts:    Coverage {cov_ratio:.1%} | Covered: {eval_result.generation.facts_covered if eval_result.generation else []}")
        if eval_result.generation and eval_result.generation.facts_missing:
            print(f"            Missing: {eval_result.generation.facts_missing}")
        print(f"  Citations: Precision {cit_prec:.1%} | Cited Pages: {cited_pages}")
        print(f"  Uncertainty Matched: {unc_matched} (is_uncertain={response.is_uncertain})")
        print(f"  Latency:  {q_latency:.2f}s | Result: {'PASSED' if eval_result.passed else 'FAILED'}")
        if not eval_result.passed:
            print(f"  Failure Reasons: {eval_result.failure_reasons}")

        per_question_records.append({
            "id": question.id,
            "question": question.question,
            "category": question.category,
            "failure_mode": question.failure_mode,
            "expected_pages": question.expected_pages,
            "expected_facts": question.expected_facts,
            "is_uncertain_expected": question.is_uncertain,
            "retrieval_query": response.retrieval_query,
            "latency_seconds": round(q_latency, 3),
            "b4_candidates_count": len(b4_raw),
            "b4_pages_retrieved": b4_pages,
            "retrieval_hit": ret_hit,
            "b5_candidates_count": len(response.pipeline_result.candidates),
            "b5_pages_reranked": b5_pages,
            "rerank_hit": rer_hit,
            "answer_text": response.answer_text,
            "raw_answer_text": response.verified_answer.raw_answer_text,
            "is_uncertain": response.is_uncertain,
            "uncertainty_matched": unc_matched,
            "cited_pages": cited_pages,
            "citations_formatted": [c.format_citation() for c in response.verified_answer.citations],
            "unverified_citations": response.verified_answer.unverified_citations,
            "facts_covered": eval_result.generation.facts_covered if eval_result.generation else [],
            "facts_missing": eval_result.generation.facts_missing if eval_result.generation else [],
            "fact_coverage_ratio": round(cov_ratio, 4),
            "citation_precision": round(cit_prec, 4),
            "passed": eval_result.passed,
            "failure_reasons": eval_result.failure_reasons,
        })

    total_latency = time.perf_counter() - total_start_time

    # Compute aggregate summary
    summary = evaluator.evaluate_benchmark(dataset, query_evals)

    # Persist JSON report
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "benchmark_summary": {
            "timestamp": summary.timestamp,
            "dataset_version": summary.dataset_version,
            "total_questions": summary.total_questions,
            "total_latency_seconds": round(total_latency, 2),
            "average_latency_per_query": round(total_latency / len(dataset.questions), 2),
            "retrieval_recall_at_n": round(summary.retrieval_recall_at_n, 4),
            "rerank_recall_at_k": round(summary.rerank_recall_at_k, 4),
            "fact_coverage_rate": round(summary.fact_coverage_rate, 4),
            "citation_precision": round(summary.citation_accuracy, 4),
            "uncertainty_accuracy": round(summary.uncertainty_accuracy, 4),
            "overall_pass_rate": round(summary.overall_pass_rate, 4),
        },
        "per_question_results": per_question_records,
    }

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    logger.info("Persisted benchmark results to %s", OUTPUT_REPORT_PATH)

    # Print Aggregate Summary Table
    print("\n" + "=" * 80)
    print("25-QUESTION GOLDEN BENCHMARK — AGGREGATE SUMMARY")
    print("=" * 80)
    print(f"Total Questions Evaluated:  {summary.total_questions}")
    print(f"Total Benchmark Runtime:    {total_latency:.2f}s ({total_latency/60:.1f} min)")
    print(f"Average Latency / Question: {total_latency / len(dataset.questions):.2f}s")
    print("-" * 80)
    print(f"Retrieval Recall@25:        {summary.retrieval_recall_at_n * 100:.1f}%")
    print(f"Rerank Recall@5:            {summary.rerank_recall_at_k * 100:.1f}%")
    print(f"Fact Coverage Rate:         {summary.fact_coverage_rate * 100:.1f}%")
    print(f"Citation Precision:         {summary.citation_accuracy * 100:.1f}%")
    print(f"Uncertainty Accuracy:       {summary.uncertainty_accuracy * 100:.1f}%")
    print(f"Overall Benchmark Pass:     {summary.overall_pass_rate * 100:.1f}% ({sum(1 for q in query_evals if q.passed)}/{len(query_evals)} passed)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_benchmark()
