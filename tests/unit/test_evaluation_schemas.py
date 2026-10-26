"""
Unit tests for evaluation schemas and data contracts.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from due_diligence.evaluation.schemas import (
    BenchmarkSummary,
    GenerationStageEval,
    GoldenDataset,
    GoldenQuestion,
    PipelineQueryEval,
    RerankStageEval,
    RetrievalStageEval,
)


def test_golden_question_creation() -> None:
    q = GoldenQuestion(
        id="Q01",
        question="What were the total deposits?",
        category="financial_metric",
        failure_mode="direct_factual_lookup",
        expected_facts=["3,105,250 crore"],
        expected_pages=[378],
        expected_section="Balance Sheet",
        calculation_required=False,
        evidence_should_exist=True,
        is_uncertain=False,
    )
    assert q.id == "Q01"
    assert q.expected_pages == [378]
    assert q.calculation_required is False
    assert q.target_chunk_ids == []


def test_golden_question_frozen() -> None:
    q = GoldenQuestion(
        id="Q01",
        question="What were the total deposits?",
        category="financial_metric",
        failure_mode="direct_factual_lookup",
        expected_facts=["3,105,250 crore"],
        expected_pages=[378],
    )
    with pytest.raises(ValidationError):
        # Model is frozen
        q.question = "Modified"


def test_golden_dataset_operations(tmp_path: Path) -> None:
    q1 = GoldenQuestion(
        id="Q01",
        question="What were the total deposits?",
        category="financial_metric",
        failure_mode="direct_factual_lookup",
        expected_facts=["3,105,250 crore"],
        expected_pages=[378],
    )
    q2 = GoldenQuestion(
        id="Q02",
        question="What was the CET1 ratio?",
        category="ratio_capital",
        failure_mode="ambiguous_terminology",
        expected_facts=["17.28%"],
        expected_pages=[403],
    )

    dataset = GoldenDataset(
        version="1.0.0",
        company="HDFC Bank",
        document_name="doc.pdf",
        description="Test dataset",
        questions=[q1, q2],
    )

    assert dataset.question_ids == ["Q01", "Q02"]
    assert dataset.get_question("Q01").id == "Q01"
    assert dataset.get_question("Q02").id == "Q02"

    with pytest.raises(KeyError):
        dataset.get_question("Q99")

    # Filter operations
    fm_filtered = dataset.filter_by_failure_mode("ambiguous_terminology")
    assert len(fm_filtered) == 1
    assert fm_filtered[0].id == "Q02"

    cat_filtered = dataset.filter_by_category("financial_metric")
    assert len(cat_filtered) == 1
    assert cat_filtered[0].id == "Q01"

    # Persistence round-trip
    file_path = tmp_path / "dataset.json"
    dataset.to_file(file_path)
    loaded = GoldenDataset.from_file(file_path)
    assert loaded.version == "1.0.0"
    assert len(loaded.questions) == 2
    assert loaded.questions[0].id == "Q01"


def test_stage_eval_models() -> None:
    retrieval_eval = RetrievalStageEval(
        query_id="Q01",
        retrieved_chunk_ids=["chunk_1", "chunk_2"],
        retrieved_pages=[378, 379],
        hit_at_n=True,
        target_chunks_hit=["chunk_1"],
        reciprocal_rank=1.0,
    )
    assert retrieval_eval.hit_at_n is True
    assert retrieval_eval.reciprocal_rank == 1.0

    rerank_eval = RerankStageEval(
        query_id="Q01",
        reranked_chunk_ids=["chunk_1"],
        reranked_pages=[378],
        hit_at_k=True,
        target_chunks_hit=["chunk_1"],
        top1_hit=True,
    )
    assert rerank_eval.top1_hit is True

    gen_eval = GenerationStageEval(
        query_id="Q01",
        facts_covered=["3,105,250 crore"],
        facts_missing=[],
        coverage_ratio=1.0,
        uncertainty_matched=True,
        cited_pages=[378],
        citation_precision=1.0,
        unverified_citations_count=0,
    )
    assert gen_eval.coverage_ratio == 1.0

    query_eval = PipelineQueryEval(
        question_id="Q01",
        category="financial_metric",
        failure_mode="direct_factual_lookup",
        retrieval=retrieval_eval,
        rerank=rerank_eval,
        generation=gen_eval,
        passed=True,
    )
    assert query_eval.passed is True
    assert len(query_eval.failure_reasons) == 0

    summary = BenchmarkSummary(
        timestamp="2026-09-26T12:00:00Z",
        dataset_version="1.0.0",
        total_questions=1,
        retrieval_recall_at_n=1.0,
        rerank_recall_at_k=1.0,
        fact_coverage_rate=1.0,
        uncertainty_accuracy=1.0,
        citation_accuracy=1.0,
        overall_pass_rate=1.0,
        query_results=[query_eval],
    )
    assert summary.overall_pass_rate == 1.0
