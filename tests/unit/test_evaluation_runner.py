"""
Unit tests for the Phase B9 BenchmarkRunner and reporting engine.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from due_diligence.agents.citation_verifier import Citation, VerifiedAnswer
from due_diligence.agents.generator import GroundedAnswer
from due_diligence.evaluation.evaluator import DiligenceEvaluator
from due_diligence.evaluation.runner import BenchmarkRunner
from due_diligence.evaluation.schemas import (
    BenchmarkRunReport,
    GoldenDataset,
    GoldenQuestion,
    QuestionEvaluationRecord,
)
from due_diligence.retrieval.schemas import (
    EvidenceContext,
    EvidenceItem,
    RerankedResult,
    RetrievalResult,
)


@pytest.fixture
def sample_golden_question() -> GoldenQuestion:
    return GoldenQuestion(
        id="Q01",
        question="What was the bank's total deposit base as of March 31, 2026?",
        category="financial_metric",
        failure_mode="direct_factual_lookup",
        expected_facts=["3105250 crore", "31,052,504,740 thousand"],
        expected_pages=[378],
        expected_section="Standalone Balance Sheet (Schedule 3)",
        calculation_required=False,
        evidence_should_exist=True,
        is_uncertain=False,
        target_chunk_ids=["chunk_378_1"],
    )


@pytest.fixture
def negative_golden_question() -> GoldenQuestion:
    return GoldenQuestion(
        id="Q24",
        question="What was the bank's exact market share percentage in private commercial credit cards?",
        category="unanswerable_absent",
        failure_mode="not_disclosed_question",
        expected_facts=["Not disclosed", "not disclosed"],
        expected_pages=[],
        expected_section=None,
        calculation_required=False,
        evidence_should_exist=False,
        is_uncertain=True,
        target_chunk_ids=[],
    )


@pytest.fixture
def mock_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.retrieval_pipeline = MagicMock()
    pipeline.retrieval_pipeline.query_rewriter = MagicMock()
    pipeline.retrieval_pipeline.retriever = MagicMock()
    pipeline.retrieval_pipeline.reranker = MagicMock()
    pipeline.context_builder = MagicMock()
    pipeline.generator = MagicMock()
    pipeline.citation_verifier = MagicMock()
    return pipeline


def test_evaluate_question_answerable_success(
    mock_pipeline: MagicMock,
    sample_golden_question: GoldenQuestion,
) -> None:
    # 1. Setup mock returns
    mock_pipeline.retrieval_pipeline.query_rewriter.rewrite.return_value = (
        "What was HDFC Bank total deposit base as of March 31, 2026?"
    )

    retrieval_res = [
        RetrievalResult(
            chunk_id="chunk_378_1",
            text="Deposits: 31,052,504,740 thousand INR (₹31,05,250 crore)",
            chunk_type="table",
            source_element_ids=["elem_1"],
            company="HDFC Bank",
            document_name="annual_report.pdf",
            document_type="pdf",
            page_numbers=[378],
            section_heading="Balance Sheet",
            score=0.92,
        ),
        RetrievalResult(
            chunk_id="chunk_200_1",
            text="Unrelated text",
            chunk_type="narrative",
            source_element_ids=["elem_2"],
            company="HDFC Bank",
            document_name="annual_report.pdf",
            document_type="pdf",
            page_numbers=[200],
            section_heading="Unrelated",
            score=0.45,
        ),
    ]
    mock_pipeline.retrieval_pipeline.retriever.retrieve.return_value = retrieval_res

    reranked_res = [
        RerankedResult.from_retrieval_result(retrieval_res[0], rerank_score=0.98)
    ]
    mock_pipeline.retrieval_pipeline.reranker.rerank.return_value = reranked_res

    item_e1 = EvidenceItem.from_reranked_result("E1", reranked_res[0])
    ctx = EvidenceContext(
        original_query=sample_golden_question.question,
        retrieval_query="What was HDFC Bank total deposit base as of March 31, 2026?",
        formatted_context="<diligence_context><evidence id='E1'>31,05,250 crore</evidence></diligence_context>",
        evidence_items=[item_e1],
        evidence_map={"E1": item_e1},
        omitted_items=[],
    )
    mock_pipeline.context_builder.build_context.return_value = ctx

    mock_pipeline.generator.generate.return_value = GroundedAnswer(
        answer_text="Total deposits stood at ₹31,05,250 crore [E1].",
        cited_evidence_ids=["E1"],
        is_uncertain=False,
    )

    mock_pipeline.citation_verifier.verify.return_value = VerifiedAnswer(
        answer_text="Total deposits stood at ₹31,05,250 crore [1].",
        raw_answer_text="Total deposits stood at ₹31,05,250 crore [E1].",
        citations=[
            Citation(
                citation_number=1,
                evidence_id="E1",
                chunk_id="chunk_378_1",
                document_name="annual_report.pdf",
                page_numbers=[378],
                section_heading="Balance Sheet",
            )
        ],
        unverified_citations=[],
        is_uncertain=False,
    )

    runner = BenchmarkRunner(pipeline=mock_pipeline, evaluator=DiligenceEvaluator())
    record = runner.evaluate_question(sample_golden_question)

    assert isinstance(record, QuestionEvaluationRecord)
    assert record.question_id == "Q01"
    assert record.original_query == sample_golden_question.question
    assert record.rewritten_query == "What was HDFC Bank total deposit base as of March 31, 2026?"
    assert record.retrieved_candidates == ["chunk_378_1", "chunk_200_1"]
    assert record.retrieved_pages == [200, 378]
    assert record.reranked_candidates == ["chunk_378_1"]
    assert record.reranked_pages == [378]
    assert record.b6_evidence_ids == ["E1"]
    assert record.b7_citation_ids == ["E1"]
    assert len(record.b8_verified_citations) == 1
    assert record.b8_verified_citations[0].page_numbers == [378]
    assert record.b8_verified_citations[0].is_expected_page is True
    assert record.retrieval_recall_at_n == 1.0
    assert record.reranker_recall_at_k == 1.0
    assert record.evidence_recall == 1.0
    assert record.citation_validity == 1.0
    assert record.no_evidence_behavior == "correct_grounded_answer"
    assert record.stage_latencies.retrieval_ms >= 0.0
    assert record.stage_latencies.rerank_ms >= 0.0
    assert record.total_latency_ms >= 0.0
    assert record.passed is True
    assert record.error is None


def test_evaluate_question_negative_control_refusal(
    mock_pipeline: MagicMock,
    negative_golden_question: GoldenQuestion,
) -> None:
    mock_pipeline.retrieval_pipeline.query_rewriter.rewrite.return_value = (
        negative_golden_question.question
    )
    mock_pipeline.retrieval_pipeline.retriever.retrieve.return_value = []
    mock_pipeline.retrieval_pipeline.reranker.rerank.return_value = []

    ctx = EvidenceContext(
        original_query=negative_golden_question.question,
        retrieval_query=negative_golden_question.question,
        formatted_context="<diligence_context></diligence_context>",
        evidence_items=[],
        evidence_map={},
        omitted_items=[],
    )
    mock_pipeline.context_builder.build_context.return_value = ctx

    mock_pipeline.generator.generate.return_value = GroundedAnswer(
        answer_text="Not disclosed in the ingested document.",
        cited_evidence_ids=[],
        is_uncertain=True,
    )

    mock_pipeline.citation_verifier.verify.return_value = VerifiedAnswer(
        answer_text="Not disclosed in the ingested document.",
        raw_answer_text="Not disclosed in the ingested document.",
        citations=[],
        unverified_citations=[],
        is_uncertain=True,
    )

    runner = BenchmarkRunner(pipeline=mock_pipeline, evaluator=DiligenceEvaluator())
    record = runner.evaluate_question(negative_golden_question)

    assert record.question_id == "Q24"
    assert record.no_evidence_behavior == "correct_refusal"
    assert record.retrieval_recall_at_n == 1.0
    assert record.reranker_recall_at_k == 1.0
    assert record.citation_validity == 1.0
    assert len(record.b8_verified_citations) == 0
    assert record.passed is True


def test_evaluate_question_handles_exception_gracefully(
    mock_pipeline: MagicMock,
    sample_golden_question: GoldenQuestion,
) -> None:
    mock_pipeline.retrieval_pipeline.query_rewriter.rewrite.side_effect = RuntimeError(
        "Remote connection timed out"
    )

    runner = BenchmarkRunner(pipeline=mock_pipeline, evaluator=DiligenceEvaluator())
    record = runner.evaluate_question(sample_golden_question)

    assert record.question_id == "Q01"
    assert record.passed is False
    assert "Remote connection timed out" in str(record.error)
    assert record.no_evidence_behavior == "pipeline_execution_error"
    assert len(record.failure_reasons) > 0


def test_run_benchmark_aggregates_four_pillars(
    mock_pipeline: MagicMock,
    sample_golden_question: GoldenQuestion,
    negative_golden_question: GoldenQuestion,
    tmp_path: Path,
) -> None:
    dataset = GoldenDataset(
        version="1.0.0",
        company="HDFC Bank Limited",
        document_name="annual_report.pdf",
        description="Test dataset",
        questions=[sample_golden_question, negative_golden_question],
    )

    # Mock behavior for Q01
    retrieval_res = [
        RetrievalResult(
            chunk_id="chunk_378_1",
            text="31,05,250 crore",
            chunk_type="table",
            source_element_ids=["elem_1"],
            company="HDFC Bank",
            document_name="annual_report.pdf",
            document_type="pdf",
            page_numbers=[378],
            section_heading="Balance Sheet",
            score=0.9,
        )
    ]
    reranked_res = [
        RerankedResult.from_retrieval_result(retrieval_res[0], rerank_score=0.95)
    ]
    item_e1 = EvidenceItem.from_reranked_result("E1", reranked_res[0])
    ctx = EvidenceContext(
        original_query=sample_golden_question.question,
        retrieval_query=sample_golden_question.question,
        formatted_context="<evidence>31,05,250 crore</evidence>",
        evidence_items=[item_e1],
        evidence_map={"E1": item_e1},
        omitted_items=[],
    )

    mock_pipeline.retrieval_pipeline.query_rewriter.rewrite.return_value = (
        sample_golden_question.question
    )
    mock_pipeline.retrieval_pipeline.retriever.retrieve.return_value = retrieval_res
    mock_pipeline.retrieval_pipeline.reranker.rerank.return_value = reranked_res
    mock_pipeline.context_builder.build_context.return_value = ctx

    mock_pipeline.generator.generate.return_value = GroundedAnswer(
        answer_text="Total deposits stood at ₹31,05,250 crore [E1]. Not disclosed for unanswerables.",
        cited_evidence_ids=["E1"],
        is_uncertain=False,
    )

    mock_pipeline.citation_verifier.verify.return_value = VerifiedAnswer(
        answer_text="Total deposits stood at ₹31,05,250 crore [1].",
        raw_answer_text="Total deposits stood at ₹31,05,250 crore [E1].",
        citations=[
            Citation(
                citation_number=1,
                evidence_id="E1",
                chunk_id="chunk_378_1",
                document_name="annual_report.pdf",
                page_numbers=[378],
                section_heading="Balance Sheet",
            )
        ],
        unverified_citations=[],
        is_uncertain=False,
    )

    runner = BenchmarkRunner(pipeline=mock_pipeline, evaluator=DiligenceEvaluator())
    report = runner.run_benchmark(dataset)

    assert isinstance(report, BenchmarkRunReport)
    assert report.total_questions == 2
    assert report.retrieval_metrics.recall_at_n == 1.0
    assert report.reranker_metrics.recall_at_k == 1.0
    assert report.reranker_metrics.top1_accuracy == 1.0
    assert report.citation_metrics.citation_provenance_integrity_rate == 1.0
    assert report.latency_metrics.avg_total_ms >= 0.0

    # Test report persistence
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    report.to_json_file(json_path)
    report.to_markdown_file(md_path)

    assert json_path.exists()
    assert md_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    assert "# AI Due Diligence Copilot - Golden Benchmark Evaluation Report" in md_content
    assert "Four Metric Pillars" in md_content
    assert "Latency Breakdown" in md_content
