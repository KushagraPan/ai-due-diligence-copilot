"""
Evaluation subsystem for the AI Due Diligence Copilot.

1. Purpose:
   Provides schemas, benchmark datasets, and evaluation harnesses to validate
   the complete RAG pipeline against curated corporate due diligence ground truth.

2. Role in Overall RAG Pipeline:
   Operates in Phase B9 (Evaluation Subsystem). Consumes pipeline execution traces
   and generated answers to assess retrieval recall, reranking precision, fact coverage,
   uncertainty compliance, and citation accuracy.

3. Expected Inputs:
   - Golden benchmark datasets (`GoldenDataset`, `GoldenQuestion`).
   - Pipeline artifacts (`DiligenceResponse`, `RetrievalResult`, `RerankedResult`, `VerifiedAnswer`).

4. Outputs:
   - Stage-level evaluation records (`RetrievalStageEval`, `RerankStageEval`, `GenerationStageEval`).
   - Query and benchmark summary reports (`PipelineQueryEval`, `BenchmarkSummary`).

5. Dependencies:
   - due_diligence.evaluation.schemas
   - due_diligence.evaluation.evaluator

6. Non-Responsibilities:
   - Does NOT perform runtime document retrieval or LLM inference.
   - Does NOT modify vector store indices or retriever configurations.
"""

from due_diligence.evaluation.evaluator import DiligenceEvaluator
from due_diligence.evaluation.runner import BenchmarkRunner
from due_diligence.evaluation.schemas import (
    AnswerQualitySummary,
    BenchmarkRunReport,
    BenchmarkSummary,
    CitationMetricsSummary,
    CitationSummary,
    GenerationStageEval,
    GoldenDataset,
    GoldenQuestion,
    LatencyMetricsSummary,
    PipelineQueryEval,
    QuestionEvaluationRecord,
    RerankerMetricsSummary,
    RerankStageEval,
    RetrievalMetricsSummary,
    RetrievalStageEval,
    StageLatencies,
)

__all__ = [
    "AnswerQualitySummary",
    "BenchmarkRunReport",
    "BenchmarkRunner",
    "BenchmarkSummary",
    "CitationMetricsSummary",
    "CitationSummary",
    "DiligenceEvaluator",
    "GenerationStageEval",
    "GoldenDataset",
    "GoldenQuestion",
    "LatencyMetricsSummary",
    "PipelineQueryEval",
    "QuestionEvaluationRecord",
    "RerankStageEval",
    "RerankerMetricsSummary",
    "RetrievalMetricsSummary",
    "RetrievalStageEval",
    "StageLatencies",
]

