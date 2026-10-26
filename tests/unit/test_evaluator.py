"""
Unit tests for DiligenceEvaluator logic across retrieval, reranking, and generation stages.
"""

from due_diligence.agents.citation_verifier import Citation, VerifiedAnswer
from due_diligence.evaluation.evaluator import DiligenceEvaluator, _contains_fact, _normalize_text
from due_diligence.evaluation.schemas import GoldenDataset, GoldenQuestion
from due_diligence.pipeline import DiligenceResponse
from due_diligence.retrieval.schemas import (
    EvidenceContext,
    EvidenceItem,
    PipelineQueryResult,
    RerankedResult,
    RetrievalResult,
)


def _make_retrieval_result(
    chunk_id: str,
    text: str,
    page_numbers: list[int],
    score: float = 1.0,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        company="HDFC Bank",
        document_name="doc.pdf",
        document_type="annual_report",
        chunk_type="narrative",
        page_numbers=page_numbers,
        source_element_ids=["elem_1"],
        score=score,
    )


def _make_reranked_result(
    chunk_id: str,
    text: str,
    page_numbers: list[int],
    score: float = 1.0,
    rerank_score: float = 1.0,
) -> RerankedResult:
    return RerankedResult(
        chunk_id=chunk_id,
        text=text,
        company="HDFC Bank",
        document_name="doc.pdf",
        document_type="annual_report",
        chunk_type="narrative",
        page_numbers=page_numbers,
        source_element_ids=["elem_1"],
        score=score,
        rerank_score=rerank_score,
    )


def test_text_normalization() -> None:
    assert _normalize_text("₹31,05,250 crore") == "3105250 crore"
    assert _normalize_text("H 74,671.3 crore") == "74671.3 crore"
    assert _normalize_text("  260,501,479   thousand  ") == "260501479 thousand"
    assert _normalize_text("` 15.50 per share") == "15.50 per share"


def test_contains_fact() -> None:
    raw_ans = "As of March 31, 2026, total deposits stood at ₹31,05,250 crore (31,052,504,740 thousand INR)."
    ans = _normalize_text(raw_ans)
    assert _contains_fact(ans, "3,105,250 crore") is True
    assert _contains_fact(ans, "31,052,504,740 thousand") is True
    assert _contains_fact(ans, "99,999 crore") is False
    assert _contains_fact(ans, "99,999 crore | 3,105,250 crore") is True
    assert _contains_fact(ans, "11,111 crore | 22,222 crore") is False


def test_evaluate_retrieval_hit() -> None:
    evaluator = DiligenceEvaluator()
    question = GoldenQuestion(
        id="Q01",
        question="Total deposits?",
        category="financial_metric",
        failure_mode="direct_factual_lookup",
        expected_facts=["3105250 crore"],
        expected_pages=[378],
        target_chunk_ids=["chunk_target"],
    )

    candidates = [
        _make_retrieval_result("chunk_other", "Some text", [100], score=0.9),
        _make_retrieval_result("chunk_target", "Deposit text", [378], score=0.85),
    ]

    res = evaluator.evaluate_retrieval(question, candidates)
    assert res.hit_at_n is True
    assert res.target_chunks_hit == ["chunk_target"]
    assert res.reciprocal_rank == 0.5  # Rank 2


def test_evaluate_retrieval_miss() -> None:
    evaluator = DiligenceEvaluator()
    question = GoldenQuestion(
        id="Q01",
        question="Total deposits?",
        category="financial_metric",
        failure_mode="direct_factual_lookup",
        expected_facts=["3105250 crore"],
        expected_pages=[378],
    )

    candidates = [
        _make_retrieval_result("chunk_1", "Some text", [10, 11], score=0.9)
    ]

    res = evaluator.evaluate_retrieval(question, candidates)
    assert res.hit_at_n is False
    assert res.reciprocal_rank == 0.0


def test_evaluate_retrieval_negative_control() -> None:
    evaluator = DiligenceEvaluator()
    question = GoldenQuestion(
        id="Q25",
        question="FinTech Corp valuation?",
        category="negative_control",
        failure_mode="negative_no_evidence_question",
        expected_facts=["Not disclosed"],
        expected_pages=[],
        evidence_should_exist=False,
        is_uncertain=True,
    )

    res = evaluator.evaluate_retrieval(question, [])
    assert res.hit_at_n is True
    assert res.reciprocal_rank == 1.0


def test_evaluate_rerank_hit_and_top1() -> None:
    evaluator = DiligenceEvaluator()
    question = GoldenQuestion(
        id="Q01",
        question="Total deposits?",
        category="financial_metric",
        failure_mode="direct_factual_lookup",
        expected_facts=["3105250 crore"],
        expected_pages=[378],
        target_chunk_ids=["chunk_target"],
    )

    reranked = [
        _make_reranked_result("chunk_target", "Deposit text", [378], score=0.85, rerank_score=4.5),
        _make_reranked_result("chunk_other", "Other text", [200], score=0.9, rerank_score=2.1),
    ]

    res = evaluator.evaluate_rerank(question, reranked)
    assert res.hit_at_k is True
    assert res.top1_hit is True
    assert res.target_chunks_hit == ["chunk_target"]


def test_evaluate_generation_all_covered() -> None:
    evaluator = DiligenceEvaluator()
    question = GoldenQuestion(
        id="Q05",
        question="EPS for FY26?",
        category="financial_metric",
        failure_mode="financial_metric_lookup",
        expected_facts=["48.62", "48.40"],
        expected_pages=[379],
    )

    verified_answer = VerifiedAnswer(
        answer_text="Basic EPS was 48.62 and Diluted EPS was 48.40 [1].",
        raw_answer_text="Basic EPS was 48.62 and Diluted EPS was 48.40 [E1].",
        citations=[
            Citation(
                citation_number=1,
                evidence_id="E1",
                chunk_id="c1",
                document_name="doc.pdf",
                page_numbers=[379],
                section_heading="Income",
            )
        ],
        unverified_citations=[],
        is_uncertain=False,
    )

    res = evaluator.evaluate_generation(question, verified_answer)
    assert res.coverage_ratio == 1.0
    assert res.facts_missing == []
    assert res.uncertainty_matched is True
    assert res.citation_precision == 1.0
    assert res.unverified_citations_count == 0


def test_evaluate_generation_missing_fact_and_wrong_citation() -> None:
    evaluator = DiligenceEvaluator()
    question = GoldenQuestion(
        id="Q05",
        question="EPS for FY26?",
        category="financial_metric",
        failure_mode="financial_metric_lookup",
        expected_facts=["48.62", "48.40"],
        expected_pages=[379],
    )

    verified_answer = VerifiedAnswer(
        answer_text="Basic EPS was 48.62 [1].",
        raw_answer_text="Basic EPS was 48.62 [E1].",
        citations=[
            Citation(
                citation_number=1,
                evidence_id="E1",
                chunk_id="c1",
                document_name="doc.pdf",
                page_numbers=[999],  # Wrong page
                section_heading=None,
            )
        ],
        unverified_citations=["E99"],
        is_uncertain=False,
    )

    res = evaluator.evaluate_generation(question, verified_answer)
    assert res.coverage_ratio == 0.5
    assert res.facts_missing == ["48.40"]
    assert res.citation_precision == 0.0  # 999 not in [379]
    assert res.unverified_citations_count == 1


def test_evaluate_generation_negative_control_refusal() -> None:
    evaluator = DiligenceEvaluator()
    question = GoldenQuestion(
        id="Q25",
        question="FinTech Corp valuation?",
        category="negative_control",
        failure_mode="negative_no_evidence_question",
        expected_facts=["Not disclosed"],
        expected_pages=[],
        evidence_should_exist=False,
        is_uncertain=True,
    )

    verified_answer = VerifiedAnswer(
        answer_text="This information is not disclosed in the ingested document.",
        raw_answer_text="This information is not disclosed in the ingested document.",
        citations=[],
        unverified_citations=[],
        is_uncertain=True,
    )

    res = evaluator.evaluate_generation(question, verified_answer)
    assert res.coverage_ratio == 1.0
    assert res.uncertainty_matched is True
    assert res.citation_precision == 1.0
    assert res.unverified_citations_count == 0


def test_evaluate_pipeline_response_pass() -> None:
    evaluator = DiligenceEvaluator()
    question = GoldenQuestion(
        id="Q01",
        question="Total deposits?",
        category="financial_metric",
        failure_mode="direct_factual_lookup",
        expected_facts=["3105250 crore"],
        expected_pages=[378],
        target_chunk_ids=["chunk_1"],
    )

    retrieval_res = [
        _make_retrieval_result("chunk_1", "Total deposits: ₹31,05,250 crore", [378], score=0.9)
    ]
    rerank_res = [
        _make_reranked_result("chunk_1", "Total deposits: ₹31,05,250 crore", [378], score=0.9, rerank_score=5.0)
    ]
    item = EvidenceItem.from_reranked_result("E1", rerank_res[0])
    evidence_ctx = EvidenceContext(
        original_query="Total deposits?",
        retrieval_query="Total deposits?",
        formatted_context="<diligence_context></diligence_context>",
        evidence_items=[item],
        evidence_map={"E1": item},
        omitted_items=[],
    )
    verified_answer = VerifiedAnswer(
        answer_text="Total deposits were ₹31,05,250 crore [1].",
        raw_answer_text="Total deposits were ₹31,05,250 crore [E1].",
        citations=[
            Citation(
                citation_number=1,
                evidence_id="E1",
                chunk_id="chunk_1",
                document_name="doc.pdf",
                page_numbers=[378],
                section_heading="Balance Sheet",
            )
        ],
        unverified_citations=[],
        is_uncertain=False,
    )
    pipeline_res = PipelineQueryResult(
        original_query="Total deposits?",
        retrieval_query="Total deposits?",
        candidates=rerank_res,
        raw_retrieval_count=len(retrieval_res),
    )

    response = DiligenceResponse(
        verified_answer=verified_answer,
        evidence_context=evidence_ctx,
        pipeline_result=pipeline_res,
    )

    eval_res = evaluator.evaluate_pipeline_response(
        question, response, retrieved_candidates=retrieval_res
    )
    assert eval_res.passed is True
    assert eval_res.failure_reasons == []


def test_evaluate_benchmark_summary() -> None:
    evaluator = DiligenceEvaluator()
    q1 = GoldenQuestion(
        id="Q01",
        question="Total deposits?",
        category="financial_metric",
        failure_mode="direct_factual_lookup",
        expected_facts=["3105250 crore"],
        expected_pages=[378],
    )
    q2 = GoldenQuestion(
        id="Q25",
        question="FinTech Corp?",
        category="negative_control",
        failure_mode="negative_no_evidence_question",
        expected_facts=["Not disclosed"],
        expected_pages=[],
        evidence_should_exist=False,
        is_uncertain=True,
    )
    dataset = GoldenDataset(
        version="1.0.0",
        company="HDFC Bank",
        document_name="doc.pdf",
        description="Test",
        questions=[q1, q2],
    )

    # Mock query evaluations
    eval1 = evaluator.evaluate_pipeline_response(
        q1,
        DiligenceResponse(
            verified_answer=VerifiedAnswer(
                answer_text="Deposits were 3105250 crore [1].",
                raw_answer_text="Deposits were 3105250 crore [E1].",
                citations=[
                    Citation(
                        citation_number=1,
                        evidence_id="E1",
                        chunk_id="c1",
                        document_name="doc.pdf",
                        page_numbers=[378],
                        section_heading=None,
                    )
                ],
                unverified_citations=[],
                is_uncertain=False,
            ),
            evidence_context=EvidenceContext(
                original_query="q",
                retrieval_query="q",
                formatted_context="",
                evidence_items=[],
                omitted_items=[],
                evidence_map={},
            ),
            pipeline_result=PipelineQueryResult(
                original_query="q",
                retrieval_query="q",
                candidates=[
                    _make_reranked_result("c1", "t", [378])
                ],
                raw_retrieval_count=1,
            ),
        ),
        retrieved_candidates=[_make_retrieval_result("c1", "t", [378])],
    )

    eval2 = evaluator.evaluate_pipeline_response(
        q2,
        DiligenceResponse(
            verified_answer=VerifiedAnswer(
                answer_text="Not disclosed in the document.",
                raw_answer_text="Not disclosed in the document.",
                citations=[],
                unverified_citations=[],
                is_uncertain=True,
            ),
            evidence_context=EvidenceContext(
                original_query="q",
                retrieval_query="q",
                formatted_context="",
                evidence_items=[],
                omitted_items=[],
                evidence_map={},
            ),
            pipeline_result=PipelineQueryResult(
                original_query="q",
                retrieval_query="q",
                candidates=[],
                raw_retrieval_count=0,
            ),
        ),
        retrieved_candidates=[],
    )

    summary = evaluator.evaluate_benchmark(dataset, [eval1, eval2])
    assert summary.total_questions == 2
    assert summary.retrieval_recall_at_n == 1.0
    assert summary.rerank_recall_at_k == 1.0
    assert summary.fact_coverage_rate == 1.0
    assert summary.uncertainty_accuracy == 1.0
    assert summary.citation_accuracy == 1.0
    assert summary.overall_pass_rate == 1.0
