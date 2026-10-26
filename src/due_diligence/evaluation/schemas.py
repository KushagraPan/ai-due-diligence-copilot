"""
Domain schemas and data contracts for Golden Benchmark Evaluation.

1. Purpose:
   Defines typed data models for curated evaluation questions, ground-truth
   expectations, stage-wise retrieval/rerank/generation metrics, and aggregate
   benchmark reporting across the AI Due Diligence pipeline.

2. Role in Overall RAG Pipeline:
   Operates in Phase B9 (Evaluation Subsystem). Provides immutable contracts
   consumed by benchmark runners and evaluation suites to measure pipeline
   faithfulness, retrieval recall, reranker precision, and citation integrity.

3. Expected Inputs:
   - JSON benchmark files or programmatic question definitions.
   - Outputs from B4 Hybrid Retrieval, B5 LLM Reranker, B6 Context Builder,
     B7 Grounded Generator, and B8 Citation Verifier.

4. Outputs:
   - GoldenQuestion, GoldenDataset domain models.
   - RetrievalStageEval, RerankStageEval, GenerationStageEval, PipelineQueryEval,
     and BenchmarkSummary metrics structures.

5. Dependencies:
   - pydantic (BaseModel, ConfigDict, Field)
   - pathlib, json

6. Non-Responsibilities:
   - Does NOT execute retrieval queries or LLM calls (handled by DiligencePipeline).
   - Does NOT modify Weaviate collections or retriever settings.
   - Does NOT mutate source chunks or provenance metadata.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ── Golden Dataset Models ───────────────────────────────────────────────────


class GoldenQuestion(BaseModel):
    """
    Immutable specification of a curated due diligence evaluation question.

    Contains verified ground-truth facts, expected source pages, section breadcrumbs,
    and failure-mode categorizations derived directly from corporate source filings.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        ...,
        description="Unique identifier for the benchmark question (e.g., 'Q01')",
    )
    question: str = Field(
        ...,
        description="The natural-language question posed to the diligence copilot",
    )
    category: str = Field(
        ...,
        description="Broad diligence category (e.g., 'financial_metric', 'risk_management')",
    )
    failure_mode: str = Field(
        ...,
        description="Target evaluation failure mode (e.g., 'direct_factual_lookup', 'ratio_capital')",
    )
    expected_facts: list[str] = Field(
        ...,
        description="Key factual phrases, numbers, or terms that must appear in the answer",
    )
    expected_pages: list[int] = Field(
        ...,
        description="1-indexed source document pages where ground-truth evidence resides",
    )
    expected_section: str | None = Field(
        default=None,
        description="Expected document section or note title where evidence is found",
    )
    calculation_required: bool = Field(
        default=False,
        description="Whether synthesizing the answer requires explicit numerical calculation",
    )
    evidence_should_exist: bool = Field(
        default=True,
        description="Whether supporting evidence exists in the document (False for unanswerable/negative)",
    )
    is_uncertain: bool = Field(
        default=False,
        description="Whether the expected response must declare uncertainty or refusal",
    )
    conversation_history: list[dict[str, str]] | None = Field(
        default=None,
        description="Prior conversational turns for testing follow-up queries and query rewriting",
    )
    ambiguity_notes: str | None = Field(
        default=None,
        description="Explanatory notes regarding definitions, precision nuances, or dual sources",
    )
    target_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Known canonical chunk IDs containing the primary evidence",
    )


class GoldenDataset(BaseModel):
    """
    Collection of curated golden questions representing an authoritative benchmark suite.
    """

    model_config = ConfigDict(frozen=True)

    version: str = Field(
        ...,
        description="Benchmark dataset version identifier (e.g., '1.0.0')",
    )
    company: str = Field(
        ...,
        description="Target entity name (e.g., 'HDFC Bank Limited')",
    )
    document_name: str = Field(
        ...,
        description="Source document filename against which the benchmark is defined",
    )
    description: str = Field(
        ...,
        description="Human-readable description of benchmark scope and coverage",
    )
    questions: list[GoldenQuestion] = Field(
        ...,
        description="Ordered list of curated benchmark questions",
    )

    def get_question(self, question_id: str) -> GoldenQuestion:
        """Retrieve a question by its unique ID, raising KeyError if not found."""
        for q in self.questions:
            if q.id == question_id:
                return q
        raise KeyError(f"Question ID '{question_id}' not found in golden dataset.")

    def filter_by_failure_mode(self, failure_mode: str) -> list[GoldenQuestion]:
        """Return all questions belonging to a specific failure mode."""
        return [q for q in self.questions if q.failure_mode == failure_mode]

    def filter_by_category(self, category: str) -> list[GoldenQuestion]:
        """Return all questions belonging to a specific category."""
        return [q for q in self.questions if q.category == category]

    @property
    def question_ids(self) -> list[str]:
        """List of all question IDs in dataset order."""
        return [q.id for q in self.questions]

    @classmethod
    def from_file(cls, path: str | Path) -> "GoldenDataset":
        """Load and validate a golden dataset from a JSON file."""
        file_path = Path(path)
        with open(file_path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        return cls.model_validate(data)

    def to_file(self, path: str | Path) -> None:
        """Persist the dataset to a JSON file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))


# ── Stage-Wise Evaluation Result Models ──────────────────────────────────────


class RetrievalStageEval(BaseModel):
    """Evaluation metrics for Phase B4 Hybrid Retrieval."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    retrieved_chunk_ids: list[str]
    retrieved_pages: list[int]
    hit_at_n: bool = Field(
        ...,
        description="True if retrieved candidates include at least one expected page or target chunk",
    )
    target_chunks_hit: list[str] = Field(
        default_factory=list,
        description="Target chunk IDs successfully retrieved in top-N candidates",
    )
    reciprocal_rank: float = Field(
        default=0.0,
        description="Reciprocal rank (1/rank) of the highest-ranking expected page",
    )


class RerankStageEval(BaseModel):
    """Evaluation metrics for Phase B5 LLM Reranking."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    reranked_chunk_ids: list[str]
    reranked_pages: list[int]
    hit_at_k: bool = Field(
        ...,
        description="True if top-K reranked evidence includes at least one expected page or target chunk",
    )
    target_chunks_hit: list[str] = Field(
        default_factory=list,
        description="Target chunk IDs present in top-K reranked evidence",
    )
    top1_hit: bool = Field(
        default=False,
        description="True if the rank-1 evidence item is from an expected page",
    )


class GenerationStageEval(BaseModel):
    """Evaluation metrics for Phase B7 Grounded Generation and Phase B8 Verification."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    facts_covered: list[str] = Field(
        default_factory=list,
        description="Expected facts successfully identified in the generated answer",
    )
    facts_missing: list[str] = Field(
        default_factory=list,
        description="Expected facts that were missing from the generated answer",
    )
    coverage_ratio: float = Field(
        ...,
        description="Ratio of covered facts to total expected facts (0.0 to 1.0)",
    )
    uncertainty_matched: bool = Field(
        ...,
        description="True if generator's uncertainty declaration matched expected is_uncertain",
    )
    cited_pages: list[int] = Field(
        default_factory=list,
        description="Distinct 1-indexed pages cited in the verified answer",
    )
    citation_precision: float = Field(
        ...,
        description="Proportion of cited pages that match expected source pages",
    )
    unverified_citations_count: int = Field(
        default=0,
        description="Count of unverified or hallucinated citation tokens",
    )


class PipelineQueryEval(BaseModel):
    """Combined evaluation result for a single benchmark query across all stages."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    category: str
    failure_mode: str
    retrieval: RetrievalStageEval | None = None
    rerank: RerankStageEval | None = None
    generation: GenerationStageEval | None = None
    passed: bool = Field(
        default=True,
        description="Overall pass/fail judgment for this benchmark query",
    )
    failure_reasons: list[str] = Field(
        default_factory=list,
        description="Explanations for any stage-level failures",
    )


class BenchmarkSummary(BaseModel):
    """
    Comprehensive aggregate benchmark report across the entire evaluation suite.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: str
    dataset_version: str
    total_questions: int
    retrieval_recall_at_n: float = Field(
        ...,
        description="Proportion of answerable queries where retrieval candidate pool contained expected evidence",
    )
    rerank_recall_at_k: float = Field(
        ...,
        description="Proportion of answerable queries where reranker Top-K contained expected evidence",
    )
    fact_coverage_rate: float = Field(
        ...,
        description="Average fact coverage ratio across answerable queries",
    )
    uncertainty_accuracy: float = Field(
        ...,
        description="Accuracy of uncertainty/refusal handling across all queries",
    )
    citation_accuracy: float = Field(
        ...,
        description="Average citation precision across all generated answers",
    )
    overall_pass_rate: float = Field(
        ...,
        description="Proportion of benchmark queries passing all applicable quality gates",
    )
    query_results: list[PipelineQueryEval] = Field(
        default_factory=list,
        description="Detailed evaluation results for each benchmark query",
    )


# ── Granular Runner & Metric Pillar Models ───────────────────────────────────


class StageLatencies(BaseModel):
    """
    Execution latency measurements across all six pipeline stages in milliseconds.
    """

    model_config = ConfigDict(frozen=True)

    rewriter_ms: float = Field(default=0.0, description="Query rewriting latency in ms")
    retrieval_ms: float = Field(default=0.0, description="Phase B4 Weaviate hybrid retrieval latency in ms")
    rerank_ms: float = Field(default=0.0, description="Phase B5 LLM reranking latency in ms")
    context_ms: float = Field(default=0.0, description="Phase B6 XML context construction latency in ms")
    generation_ms: float = Field(default=0.0, description="Phase B7 Grounded LLM generation latency in ms")
    verification_ms: float = Field(default=0.0, description="Phase B8 Citation verification latency in ms")
    total_ms: float = Field(default=0.0, description="Total end-to-end pipeline execution latency in ms")


class CitationSummary(BaseModel):
    """
    Authoritative citation provenance record for evaluation reporting.
    """

    model_config = ConfigDict(frozen=True)

    citation_number: int = Field(..., description="1-indexed footnote citation number")
    evidence_id: str = Field(..., description="Context evidence tag (e.g. 'E1')")
    chunk_id: str = Field(..., description="Canonical chunk identifier")
    document_name: str = Field(..., description="Source document filename")
    page_numbers: list[int] = Field(..., description="Exact 1-indexed document page numbers")
    section_heading: str | None = Field(default=None, description="Section heading breadcrumb")
    is_verified: bool = Field(default=True, description="Whether citation was verified against context")
    is_expected_page: bool = Field(default=False, description="Whether cited page matches golden expected pages")


class QuestionEvaluationRecord(BaseModel):
    """
    Detailed end-to-end evaluation record for a single benchmark query.

    Captures intermediate artifacts, stage-wise metrics, latencies, and validation
    judgments across all six production pipeline stages.
    """

    model_config = ConfigDict(frozen=True)

    question_id: str = Field(..., description="Unique question identifier (e.g. 'Q01')")
    category: str = Field(..., description="Diligence category")
    failure_mode: str = Field(..., description="Target failure mode")
    original_query: str = Field(..., description="User's original query wording")
    rewritten_query: str = Field(..., description="Disambiguated retrieval query from QueryRewriter")
    retrieved_candidates: list[str] = Field(
        ...,
        description="Chunk IDs of the Top-N candidates retrieved in Phase B4",
    )
    retrieved_pages: list[int] = Field(
        default_factory=list,
        description="1-indexed document pages present in Phase B4 retrieved candidates",
    )
    reranked_candidates: list[str] = Field(
        ...,
        description="Chunk IDs of the Top-K candidates selected by Phase B5 LLM reranker",
    )
    reranked_pages: list[int] = Field(
        default_factory=list,
        description="1-indexed document pages present in Phase B5 reranked candidates",
    )
    b6_evidence_ids: list[str] = Field(
        ...,
        description="Active evidence identifiers in Phase B6 context (e.g. ['E1', 'E2'])",
    )
    b7_answer: str = Field(
        ...,
        description="Raw diligence answer generated in Phase B7 before verification",
    )
    b7_citation_ids: list[str] = Field(
        ...,
        description="Raw citation tags emitted by Phase B7 generator (e.g. ['[E1]', '[E2]'])",
    )
    b8_verified_citations: list[CitationSummary] = Field(
        default_factory=list,
        description="Structured, authoritative citations verified in Phase B8",
    )
    b8_unverified_citations: list[str] = Field(
        default_factory=list,
        description="Unverified or hallucinated citation tokens stripped in Phase B8",
    )
    expected_evidence: dict[str, Any] = Field(
        ...,
        description="Ground-truth expected evidence annotations from GoldenQuestion",
    )
    retrieval_recall_at_n: float = Field(
        ...,
        description="Phase B4 recall hit (1.0 if expected page/chunk retrieved, else 0.0)",
    )
    reranker_recall_at_k: float = Field(
        ...,
        description="Phase B5 recall hit (1.0 if expected page/chunk reranked into Top-K, else 0.0)",
    )
    evidence_recall: float = Field(
        ...,
        description="Proportion of expected pages present in Phase B5 reranked evidence",
    )
    fact_coverage_ratio: float = Field(
        default=1.0,
        description="Proportion of expected facts identified in the generated answer",
    )
    citation_validity: float = Field(
        ...,
        description="Precision of Phase B8 citations against expected source pages",
    )
    no_evidence_behavior: str = Field(
        ...,
        description="Behavioral assessment for negative/uncertain handling ('correct_refusal', 'correct_grounded_answer', etc.)",
    )
    stage_latencies: StageLatencies = Field(
        ...,
        description="Execution latencies broken down across all six pipeline stages in milliseconds",
    )
    total_latency_ms: float = Field(
        ...,
        description="Total end-to-end pipeline execution latency in milliseconds",
    )
    error: str | None = Field(
        default=None,
        description="Error message if pipeline execution failed on this question",
    )
    passed: bool = Field(
        default=True,
        description="Whether this query passed all applicable quality gates",
    )
    failure_reasons: list[str] = Field(
        default_factory=list,
        description="Itemized explanations for any quality gate failures",
    )


class RetrievalMetricsSummary(BaseModel):
    """Aggregate metrics for Phase B4 Hybrid Retrieval."""

    model_config = ConfigDict(frozen=True)

    recall_at_n: float = Field(..., description="Proportion of answerable queries with evidence in Top-N")
    mrr_at_n: float = Field(..., description="Mean Reciprocal Rank of first expected evidence item in Top-N")
    total_answerable: int = Field(..., description="Number of evaluated queries with evidence_should_exist=True")
    hits_at_n: int = Field(..., description="Count of answerable queries hitting expected evidence in Top-N")


class RerankerMetricsSummary(BaseModel):
    """Aggregate metrics for Phase B5 LLM Reranking."""

    model_config = ConfigDict(frozen=True)

    recall_at_k: float = Field(..., description="Proportion of answerable queries with evidence in Top-K")
    top1_accuracy: float = Field(..., description="Proportion of answerable queries where rank-1 matches expected page")
    hits_at_k: int = Field(..., description="Count of answerable queries hitting expected evidence in Top-K")


class CitationMetricsSummary(BaseModel):
    """Aggregate provenance and verification metrics for Phase B8 Citation Verification."""

    model_config = ConfigDict(frozen=True)

    citation_precision: float = Field(..., description="Average precision of cited pages against golden expected pages")
    unverified_citations_total: int = Field(..., description="Total unverified or hallucinated citation tokens detected")
    citation_provenance_integrity_rate: float = Field(
        ...,
        description="Percentage of citations verified exclusively via B6 evidence map",
    )


class AnswerQualitySummary(BaseModel):
    """Aggregate answer correctness, fact coverage, and uncertainty compliance metrics."""

    model_config = ConfigDict(frozen=True)

    fact_coverage_rate: float = Field(..., description="Average proportion of expected facts present in answers")
    uncertainty_accuracy: float = Field(..., description="Accuracy of uncertainty and refusal compliance")
    refusal_precision: float = Field(..., description="Proportion of negative/unanswerable queries correctly refused")
    overall_pass_rate: float = Field(..., description="Proportion of queries passing all stage quality gates")
    passed_count: int = Field(..., description="Number of passed benchmark queries")
    total_evaluated: int = Field(..., description="Total number of evaluated queries")


class LatencyMetricsSummary(BaseModel):
    """Aggregate latency metrics across all evaluated queries in milliseconds."""

    model_config = ConfigDict(frozen=True)

    avg_rewriter_ms: float = Field(..., description="Average query rewriting latency in ms")
    avg_retrieval_ms: float = Field(..., description="Average Phase B4 retrieval latency in ms")
    avg_rerank_ms: float = Field(..., description="Average Phase B5 reranking latency in ms")
    avg_context_ms: float = Field(..., description="Average Phase B6 context construction latency in ms")
    avg_generation_ms: float = Field(..., description="Average Phase B7 generation latency in ms")
    avg_verification_ms: float = Field(..., description="Average Phase B8 verification latency in ms")
    avg_total_ms: float = Field(..., description="Average end-to-end latency in ms")
    p50_total_ms: float = Field(..., description="Median end-to-end latency in ms")
    p95_total_ms: float = Field(..., description="95th percentile end-to-end latency in ms")
    min_total_ms: float = Field(..., description="Minimum end-to-end latency in ms")
    max_total_ms: float = Field(..., description="Maximum end-to-end latency in ms")


class BenchmarkRunReport(BaseModel):
    """
    Authoritative benchmark execution report encapsulating all 4 metric pillars,
    stage latencies, and granular question records.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Unique benchmark execution identifier")
    timestamp: str = Field(..., description="ISO 8601 execution timestamp")
    duration_seconds: float = Field(..., description="Total benchmark evaluation duration in seconds")
    dataset_version: str = Field(..., description="Golden dataset version identifier")
    dataset_company: str = Field(..., description="Target corporate entity name")
    dataset_document: str = Field(..., description="Source document filename")
    total_questions: int = Field(..., description="Total number of questions evaluated")
    retrieval_metrics: RetrievalMetricsSummary = Field(..., description="Pillar 1: Retrieval metrics")
    reranker_metrics: RerankerMetricsSummary = Field(..., description="Pillar 2: Reranker metrics")
    citation_metrics: CitationMetricsSummary = Field(..., description="Pillar 3: Citation provenance metrics")
    answer_quality: AnswerQualitySummary = Field(..., description="Pillar 4: Answer quality metrics")
    latency_metrics: LatencyMetricsSummary = Field(..., description="Stage and pipeline latency metrics")
    question_records: list[QuestionEvaluationRecord] = Field(
        default_factory=list,
        description="Granular question evaluation records",
    )

    def to_json_file(self, path: str | Path) -> None:
        """Persist the machine-readable evaluation report to a JSON file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    def generate_markdown_report(self) -> str:
        """Generate a concise, human-readable Markdown evaluation report."""
        lines: list[str] = [
            "# AI Due Diligence Copilot - Golden Benchmark Evaluation Report (Phase B9)",
            "",
            f"- **Run ID:** `{self.run_id}`",
            f"- **Timestamp:** {self.timestamp}",
            f"- **Target Entity:** {self.dataset_company}",
            f"- **Source Document:** `{self.dataset_document}`",
            f"- **Dataset Version:** {self.dataset_version}",
            f"- **Questions Evaluated:** {self.total_questions}",
            f"- **Total Duration:** {self.duration_seconds:.2f}s",
            "",
            "## 1. Executive Summary: Four Metric Pillars",
            "",
            "| Metric Pillar | Evaluation Dimension | Target | Actual | Assessment |",
            "| :--- | :--- | :--- | :--- | :--- |",
            f"| **1. Retrieval (Phase B4)** | Recall@N (Evidence in Top-N) | >= 95.0% | {self.retrieval_metrics.recall_at_n:.1%} | {'PASSED' if self.retrieval_metrics.recall_at_n >= 0.95 else 'FAILED'} |",
            f"| **1. Retrieval (Phase B4)** | Mean Reciprocal Rank (MRR@N) | >= 0.70 | {self.retrieval_metrics.mrr_at_n:.3f} | {'PASSED' if self.retrieval_metrics.mrr_at_n >= 0.70 else 'REVIEW'} |",
            f"| **2. Reranking (Phase B5)** | Recall@K (Evidence in Top-K) | >= 85.0% | {self.reranker_metrics.recall_at_k:.1%} | {'PASSED' if self.reranker_metrics.recall_at_k >= 0.85 else 'FAILED'} |",
            f"| **2. Reranking (Phase B5)** | Top-1 Accuracy (Rank 1 Hit) | >= 70.0% | {self.reranker_metrics.top1_accuracy:.1%} | {'PASSED' if self.reranker_metrics.top1_accuracy >= 0.70 else 'REVIEW'} |",
            f"| **3. Citation & Provenance (Phase B8)** | Citation Precision (Page match) | >= 90.0% | {self.citation_metrics.citation_precision:.1%} | {'PASSED' if self.citation_metrics.citation_precision >= 0.90 else 'FAILED'} |",
            f"| **3. Citation & Provenance (Phase B8)** | Unverified / Hallucinated Citations | 0 | {self.citation_metrics.unverified_citations_total} | {'PASSED' if self.citation_metrics.unverified_citations_total == 0 else 'FAILED'} |",
            f"| **3. Citation & Provenance (Phase B8)** | Provenance Integrity Rate | 100.0% | {self.citation_metrics.citation_provenance_integrity_rate:.1%} | {'PASSED' if self.citation_metrics.citation_provenance_integrity_rate == 1.0 else 'FAILED'} |",
            f"| **4. Answer Quality (Phase B7)** | Fact Coverage Rate | >= 80.0% | {self.answer_quality.fact_coverage_rate:.1%} | {'PASSED' if self.answer_quality.fact_coverage_rate >= 0.80 else 'REVIEW'} |",
            f"| **4. Answer Quality (Phase B7)** | Uncertainty / Refusal Accuracy | 100.0% | {self.answer_quality.uncertainty_accuracy:.1%} | {'PASSED' if self.answer_quality.uncertainty_accuracy >= 0.95 else 'FAILED'} |",
            f"| **Overall Benchmark** | Overall Pipeline Pass Rate | >= 80.0% | {self.answer_quality.overall_pass_rate:.1%} ({self.answer_quality.passed_count}/{self.answer_quality.total_evaluated}) | {'PASSED' if self.answer_quality.overall_pass_rate >= 0.80 else 'FAILED'} |",
            "",
            "## 2. Latency Breakdown Across Pipeline Stages",
            "",
            "| Pipeline Stage | Component | Avg Latency (ms) | % of Total Latency |",
            "| :--- | :--- | :--- | :--- |",
            f"| Query Rewriter | GPT-4o-mini / Heuristic Disambiguation | {self.latency_metrics.avg_rewriter_ms:.1f} ms | {self.latency_metrics.avg_rewriter_ms / max(self.latency_metrics.avg_total_ms, 0.001) * 100:.1f}% |",
            f"| Phase B4: Hybrid Retrieval | Weaviate Cloud (Vector + BM25) | {self.latency_metrics.avg_retrieval_ms:.1f} ms | {self.latency_metrics.avg_retrieval_ms / max(self.latency_metrics.avg_total_ms, 0.001) * 100:.1f}% |",
            f"| Phase B5: Reranking | LLM Reranker (gpt-4o-mini) | {self.latency_metrics.avg_rerank_ms:.1f} ms | {self.latency_metrics.avg_rerank_ms / max(self.latency_metrics.avg_total_ms, 0.001) * 100:.1f}% |",
            f"| Phase B6: Context Builder | Structured XML Assembly & Provenance Map | {self.latency_metrics.avg_context_ms:.1f} ms | {self.latency_metrics.avg_context_ms / max(self.latency_metrics.avg_total_ms, 0.001) * 100:.1f}% |",
            f"| Phase B7: Grounded LLM | GPT-4o Source-Grounded Generation | {self.latency_metrics.avg_generation_ms:.1f} ms | {self.latency_metrics.avg_generation_ms / max(self.latency_metrics.avg_total_ms, 0.001) * 100:.1f}% |",
            f"| Phase B8: Citation Verifier | Strict Provenance Footnote Formatting | {self.latency_metrics.avg_verification_ms:.1f} ms | {self.latency_metrics.avg_verification_ms / max(self.latency_metrics.avg_total_ms, 0.001) * 100:.1f}% |",
            f"| **Total Pipeline** | **End-to-End Execution** | **{self.latency_metrics.avg_total_ms:.1f} ms** (P50: {self.latency_metrics.p50_total_ms:.1f} ms, P95: {self.latency_metrics.p95_total_ms:.1f} ms) | **100.0%** |",
            "",
            "## 3. Question-by-Question Evaluation Results",
            "",
            "| ID | Failure Mode | B4 Hit | B5 Hit | Fact Cov | Uncertainty | Citations (Valid/Total) | Latency | Status |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for q in self.question_records:
            b4_icon = "PASS" if q.retrieval_recall_at_n > 0.0 else "FAIL"
            b5_icon = "PASS" if q.reranker_recall_at_k > 0.0 else "FAIL"
            status_badge = "**PASS**" if q.passed else "**FAIL**"
            valid_cits = sum(1 for c in q.b8_verified_citations if c.is_expected_page)
            total_cits = len(q.b8_verified_citations)
            cit_str = f"{valid_cits}/{total_cits}" if total_cits > 0 else "0/0"
            if q.no_evidence_behavior in ("correct_refusal", "n/a") and total_cits == 0:
                cit_str = "Refusal (0)"

            lines.append(
                f"| `{q.question_id}` | {q.failure_mode} | {b4_icon} | {b5_icon} | "
                f"{q.fact_coverage_ratio:.0%} | {q.no_evidence_behavior} | {cit_str} | "
                f"{q.total_latency_ms:.0f}ms | {status_badge} |"
            )

        lines.extend([
            "",
            "## 4. Failure Mode & Edge Case Analysis",
            "",
        ])

        failed_records = [q for q in self.question_records if not q.passed]
        if not failed_records:
            lines.append("No failures detected. All evaluated questions satisfied their applicable quality gates.")
        else:
            lines.append(f"Detected {len(failed_records)} questions with quality gate deviations:")
            lines.append("")
            for q in failed_records:
                lines.append(f"### Question `{q.question_id}`: {q.original_query}")
                lines.append(f"- **Category / Mode:** {q.category} / `{q.failure_mode}`")
                lines.append(f"- **Rewritten Query:** {q.rewritten_query}")
                lines.append(f"- **Failure Reasons:** {'; '.join(q.failure_reasons)}")
                lines.append(f"- **Expected Evidence:** Pages {q.expected_evidence.get('expected_pages', [])}, Target Chunks: {q.expected_evidence.get('target_chunk_ids', [])}")
                lines.append(f"- **Retrieved Pages (B4):** {q.retrieved_pages[:10]}")
                lines.append(f"- **Reranked Pages (B5):** {q.reranked_pages}")
                lines.append(f"- **B7 Answer Snippet:** `{q.b7_answer[:200]}...`")
                lines.append(f"- **Verified Citations (B8):** {[c.page_numbers for c in q.b8_verified_citations]}")
                lines.append("")

        return "\n".join(lines)

    def to_markdown_file(self, path: str | Path) -> None:
        """Persist the human-readable Markdown evaluation report to disk."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown_report())

