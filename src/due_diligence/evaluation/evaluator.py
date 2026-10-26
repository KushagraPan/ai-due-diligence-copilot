"""
Automated benchmark evaluator for the AI Due Diligence RAG pipeline.

1. Purpose:
   Evaluates pipeline responses against curated golden benchmark questions across
   all stages: retrieval recall (Phase B4), reranker precision (Phase B5),
   fact coverage (Phase B7), uncertainty compliance, and citation accuracy (Phase B8).

2. Role in Overall RAG Pipeline:
   Operates in Phase B9 (Evaluation Subsystem). Consumes DiligenceResponse objects
   emitted by DiligencePipeline and scores them against GoldenQuestion ground truth
   without modifying pipeline execution, models, or data stores.

3. Expected Inputs:
   - GoldenQuestion, GoldenDataset domain models.
   - DiligenceResponse instances (or stage-level candidate lists).

4. Outputs:
   - RetrievalStageEval, RerankStageEval, GenerationStageEval, PipelineQueryEval,
     and BenchmarkSummary metrics structures.

5. Dependencies:
   - re, datetime
   - due_diligence.evaluation.schemas
   - due_diligence.retrieval.schemas (RetrievalResult, RerankedResult, EvidenceContext)
   - due_diligence.agents.citation_verifier (VerifiedAnswer)
   - due_diligence.pipeline (DiligenceResponse)

6. Non-Responsibilities:
   - Does NOT perform LLM generation or database queries.
   - Does NOT modify candidate rankings or rewrite answers.
   - Does NOT alter Weaviate collections or configuration parameters.
"""

import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime

from due_diligence.agents.citation_verifier import VerifiedAnswer
from due_diligence.evaluation.schemas import (
    BenchmarkSummary,
    GenerationStageEval,
    GoldenDataset,
    GoldenQuestion,
    PipelineQueryEval,
    RerankStageEval,
    RetrievalStageEval,
)
from due_diligence.pipeline import DiligenceResponse
from due_diligence.retrieval.schemas import (
    EvidenceContext,
    RerankedResult,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


# ── Text & Number Normalization Helpers ─────────────────────────────────────


def _normalize_text(text: str) -> str:
    """Normalize text by collapsing whitespace, removing commas, colons, and standardizing scale units."""
    # Strip standalone currency symbols
    text = re.sub(r"[₹`]", "", text)
    # Strip single-letter or short currency prefixes preceding numbers (e.g. H 74,671, C 15.50, J 29,37,166)
    text = re.sub(r"\b[HJCj]\s*(?=\d)", "", text)
    text = re.sub(r"\b(?:rs\.?|inr)\s*", "", text, flags=re.IGNORECASE)
    # Normalize colons to whitespace to prevent token glueing in facts like 'Gross NPA: 1.15%'
    text = text.replace(":", " ")
    # Normalize thousand scale headers e.g. ('000), (in '000), '000 to 'thousand'
    text = re.sub(r"\(\s*['’]000\s*\)", " thousand", text)
    text = re.sub(r"\b(?:in\s+)?['’]000\b", " thousand", text)
    # Strip brackets, parentheses, and quote characters
    text = re.sub(r"[()\[\]'\"`]", " ", text)
    # Remove commas in numbers or clauses
    text = text.replace(",", "")
    # Lowercase
    text = text.lower()
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text



def _contains_fact(normalized_answer: str, fact: str) -> bool:
    """Check whether a single ground-truth fact is reflected in the normalized answer."""
    if "|" in fact:
        return any(_contains_fact(normalized_answer, alt.strip()) for alt in fact.split("|"))

    normalized_fact = _normalize_text(fact)
    if not normalized_fact:
        return True

    # Exact normalized substring match
    if normalized_fact in normalized_answer:
        return True

    # Token-set match for multi-word phrases (allows slight reordering)
    fact_tokens = [t for t in normalized_fact.split() if len(t) > 1]
    return bool(fact_tokens and all(token in normalized_answer for token in fact_tokens))


# ── Evaluator Engine ─────────────────────────────────────────────────────────


class DiligenceEvaluator:
    """
    Evaluation engine measuring retrieval recall, reranking precision, fact coverage,
    uncertainty compliance, and citation accuracy against golden ground truth.
    """

    def evaluate_retrieval(
        self,
        question: GoldenQuestion,
        retrieved_candidates: Sequence[RetrievalResult],
    ) -> RetrievalStageEval:
        """
        Evaluate Phase B4 Hybrid Retrieval candidates against expected pages and chunks.
        """
        retrieved_ids = [c.chunk_id for c in retrieved_candidates]
        all_pages: list[int] = []
        for c in retrieved_candidates:
            all_pages.extend(c.page_numbers)
        distinct_pages = sorted(set(all_pages))

        target_chunks_hit = [
            cid for cid in question.target_chunk_ids if cid in retrieved_ids
        ]

        if not question.evidence_should_exist:
            # Negative controls or absent info: retrieval is vacuously satisfied
            return RetrievalStageEval(
                query_id=question.id,
                retrieved_chunk_ids=retrieved_ids,
                retrieved_pages=distinct_pages,
                hit_at_n=True,
                target_chunks_hit=[],
                reciprocal_rank=1.0,
            )

        expected_page_set = set(question.expected_pages)
        reciprocal_rank = 0.0

        for rank_idx, candidate in enumerate(retrieved_candidates, start=1):
            if any(p in expected_page_set for p in candidate.page_numbers) or (
                candidate.chunk_id in question.target_chunk_ids
            ):
                reciprocal_rank = 1.0 / rank_idx
                break

        hit_at_n = reciprocal_rank > 0.0

        return RetrievalStageEval(
            query_id=question.id,
            retrieved_chunk_ids=retrieved_ids,
            retrieved_pages=distinct_pages,
            hit_at_n=hit_at_n,
            target_chunks_hit=target_chunks_hit,
            reciprocal_rank=reciprocal_rank,
        )

    def evaluate_rerank(
        self,
        question: GoldenQuestion,
        reranked_candidates: Sequence[RerankedResult],
    ) -> RerankStageEval:
        """
        Evaluate Phase B5 LLM Reranking candidates against expected pages and chunks.
        """
        reranked_ids = [c.chunk_id for c in reranked_candidates]
        all_pages: list[int] = []
        for c in reranked_candidates:
            all_pages.extend(c.page_numbers)
        distinct_pages = sorted(set(all_pages))

        target_chunks_hit = [
            cid for cid in question.target_chunk_ids if cid in reranked_ids
        ]

        if not question.evidence_should_exist:
            return RerankStageEval(
                query_id=question.id,
                reranked_chunk_ids=reranked_ids,
                reranked_pages=distinct_pages,
                hit_at_k=True,
                target_chunks_hit=[],
                top1_hit=True,
            )

        expected_page_set = set(question.expected_pages)
        hit_at_k = any(
            any(p in expected_page_set for p in c.page_numbers)
            or (c.chunk_id in question.target_chunk_ids)
            for c in reranked_candidates
        )

        top1_hit = False
        if reranked_candidates:
            top1 = reranked_candidates[0]
            top1_hit = any(p in expected_page_set for p in top1.page_numbers) or (
                top1.chunk_id in question.target_chunk_ids
            )

        return RerankStageEval(
            query_id=question.id,
            reranked_chunk_ids=reranked_ids,
            reranked_pages=distinct_pages,
            hit_at_k=hit_at_k,
            target_chunks_hit=target_chunks_hit,
            top1_hit=top1_hit,
        )

    def evaluate_generation(
        self,
        question: GoldenQuestion,
        verified_answer: VerifiedAnswer,
        evidence_context: EvidenceContext | None = None,
    ) -> GenerationStageEval:
        """
        Evaluate Phase B7 Grounded Generation and Phase B8 Citation Verification.
        """
        answer_text = verified_answer.answer_text
        normalized_answer = _normalize_text(answer_text)

        # 1. Fact Coverage
        facts_covered: list[str] = []
        facts_missing: list[str] = []

        if not question.evidence_should_exist or question.is_uncertain:
            # For unanswerable questions, check for explicit refusal phrases
            refusal_terms = [
                "not disclosed",
                "not mentioned",
                "not provided",
                "no evidence",
                "not available",
                "uncertain",
                "cannot be determined",
            ]
            has_refusal_phrase = any(term in normalized_answer for term in refusal_terms)
            if verified_answer.is_uncertain or has_refusal_phrase:
                facts_covered = ["uncertainty_or_refusal_acknowledged"]
                coverage_ratio = 1.0
            else:
                facts_missing = ["expected_uncertainty_or_refusal"]
                coverage_ratio = 0.0
        else:
            for fact in question.expected_facts:
                if _contains_fact(normalized_answer, fact):
                    facts_covered.append(fact)
                else:
                    facts_missing.append(fact)

            coverage_ratio = (
                len(facts_covered) / len(question.expected_facts)
                if question.expected_facts
                else 1.0
            )

        # 2. Uncertainty Alignment
        if question.is_uncertain:
            uncertainty_matched = verified_answer.is_uncertain or any(
                term in normalized_answer
                for term in ["not disclosed", "not mentioned", "not available", "uncertain"]
            )
        else:
            uncertainty_matched = not verified_answer.is_uncertain

        # 3. Citation Evaluation
        cited_pages: list[int] = []
        for cit in verified_answer.citations:
            cited_pages.extend(cit.page_numbers)
        distinct_cited_pages = sorted(set(cited_pages))

        if not question.evidence_should_exist:
            citation_precision = 1.0 if len(distinct_cited_pages) == 0 else 0.0
        elif not distinct_cited_pages:
            citation_precision = 0.0
        else:
            expected_set = set(question.expected_pages)
            matching_pages = [p for p in distinct_cited_pages if p in expected_set]
            citation_precision = len(matching_pages) / len(distinct_cited_pages)

        unverified_count = len(verified_answer.unverified_citations)

        return GenerationStageEval(
            query_id=question.id,
            facts_covered=facts_covered,
            facts_missing=facts_missing,
            coverage_ratio=coverage_ratio,
            uncertainty_matched=uncertainty_matched,
            cited_pages=distinct_cited_pages,
            citation_precision=citation_precision,
            unverified_citations_count=unverified_count,
        )

    def evaluate_pipeline_response(
        self,
        question: GoldenQuestion,
        response: DiligenceResponse,
        retrieved_candidates: Sequence[RetrievalResult] | None = None,
    ) -> PipelineQueryEval:
        """
        Evaluate complete end-to-end DiligenceResponse for a golden question.
        """
        retrieval_eval = (
            self.evaluate_retrieval(question, retrieved_candidates)
            if retrieved_candidates is not None
            else None
        )
        rerank_eval = self.evaluate_rerank(
            question, response.pipeline_result.candidates
        )
        generation_eval = self.evaluate_generation(
            question, response.verified_answer, response.evidence_context
        )

        failure_reasons: list[str] = []

        if question.evidence_should_exist:
            if retrieval_eval is not None and not retrieval_eval.hit_at_n:
                failure_reasons.append(
                    f"Retrieval Recall Failed: Expected pages {question.expected_pages} not in Top-N"
                )
            if not rerank_eval.hit_at_k:
                failure_reasons.append(
                    f"Reranking Recall Failed: Expected pages {question.expected_pages} not in Top-K"
                )
            if generation_eval.coverage_ratio < 0.5:
                failure_reasons.append(
                    f"Fact Coverage Low ({generation_eval.coverage_ratio:.1%}): Missing {generation_eval.facts_missing}"
                )
            if not generation_eval.uncertainty_matched:
                failure_reasons.append("Unexpected uncertainty/refusal flag on answerable question")
            if generation_eval.unverified_citations_count > 0:
                failure_reasons.append(
                    f"Unverified Citations Detected ({generation_eval.unverified_citations_count})"
                )
        else:
            if not generation_eval.uncertainty_matched:
                failure_reasons.append("Failed to declare uncertainty or refusal on unanswerable question")
            if generation_eval.unverified_citations_count > 0:
                failure_reasons.append("Unverified citations detected on unanswerable question")

        passed = len(failure_reasons) == 0

        return PipelineQueryEval(
            question_id=question.id,
            category=question.category,
            failure_mode=question.failure_mode,
            retrieval=retrieval_eval,
            rerank=rerank_eval,
            generation=generation_eval,
            passed=passed,
            failure_reasons=failure_reasons,
        )

    def evaluate_benchmark(
        self,
        dataset: GoldenDataset,
        query_evals: Sequence[PipelineQueryEval],
    ) -> BenchmarkSummary:
        """
        Aggregate individual query evaluations into an authoritative benchmark summary.
        """
        total = len(query_evals)
        if total == 0:
            return BenchmarkSummary(
                timestamp=datetime.now(UTC).isoformat(),
                dataset_version=dataset.version,
                total_questions=0,
                retrieval_recall_at_n=0.0,
                rerank_recall_at_k=0.0,
                fact_coverage_rate=0.0,
                uncertainty_accuracy=0.0,
                citation_accuracy=0.0,
                overall_pass_rate=0.0,
                query_results=[],
            )

        answerable_evals = [
            q for q in query_evals
            if dataset.get_question(q.question_id).evidence_should_exist
        ]
        num_answerable = len(answerable_evals)

        retrieval_hits = sum(
            1 for q in answerable_evals if q.retrieval and q.retrieval.hit_at_n
        )
        retrieval_recall = retrieval_hits / num_answerable if num_answerable > 0 else 1.0

        rerank_hits = sum(
            1 for q in answerable_evals if q.rerank and q.rerank.hit_at_k
        )
        rerank_recall = rerank_hits / num_answerable if num_answerable > 0 else 1.0

        fact_coverage_sum = sum(
            q.generation.coverage_ratio for q in answerable_evals if q.generation
        )
        fact_coverage_rate = (
            fact_coverage_sum / num_answerable if num_answerable > 0 else 1.0
        )

        uncertainty_matches = sum(
            1 for q in query_evals if q.generation and q.generation.uncertainty_matched
        )
        uncertainty_accuracy = uncertainty_matches / total

        citation_precisions = [
            q.generation.citation_precision for q in query_evals if q.generation
        ]
        citation_accuracy = (
            sum(citation_precisions) / len(citation_precisions)
            if citation_precisions
            else 1.0
        )

        passed_count = sum(1 for q in query_evals if q.passed)
        overall_pass_rate = passed_count / total

        return BenchmarkSummary(
            timestamp=datetime.now(UTC).isoformat(),
            dataset_version=dataset.version,
            total_questions=total,
            retrieval_recall_at_n=retrieval_recall,
            rerank_recall_at_k=rerank_recall,
            fact_coverage_rate=fact_coverage_rate,
            uncertainty_accuracy=uncertainty_accuracy,
            citation_accuracy=citation_accuracy,
            overall_pass_rate=overall_pass_rate,
            query_results=list(query_evals),
        )
