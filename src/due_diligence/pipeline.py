"""
End-to-end online diligence pipeline coordinating Phases B4 through B8.

1. Purpose:
   Coordinates the complete online diligence pipeline from user query to
   verified, citation-grounded answer:
   Query Rewriting -> B4 Hybrid Retrieval -> B5 LLM Reranking ->
   B6 Context Construction -> B7 Grounded Generation -> B8 Citation Verification.

2. Role in Overall RAG Pipeline:
   Acts as the central end-to-end execution orchestrator connecting the retrieval
   subsystem (Phases B4, B5), context assembly (Phase B6), LLM generation (Phase B7),
   and citation verification (Phase B8) into a cohesive, deterministic pipeline.

3. Expected Inputs:
   - user_query: str (Current user query)
   - conversation_history: Sequence[ConversationTurn] | None (Prior conversation context)
   - company: str | None (Optional target entity filter)
   - financial_period: str | None (Optional reporting period filter)
   - top_n: int (B4 hybrid candidate budget, default: 40)
   - top_k: int (B5 reranked evidence budget, default: 7)
   - rerank_candidate_count: int | None (Number of B4 candidates scored by B5)

4. Outputs:
   - DiligenceResponse: Immutable container with verified_answer (Phase B8),
     evidence_context (Phase B6), and pipeline_result (Phases B4-B5).

5. Dependencies:
   - due_diligence.core.config (Settings, get_settings)
   - due_diligence.retrieval.pipeline (RetrievalPipeline)
   - due_diligence.retrieval.context_builder (EvidenceContextBuilder)
   - due_diligence.retrieval.query_rewriter (ConversationTurn)
   - due_diligence.retrieval.schemas (PipelineQueryResult, EvidenceContext)
   - due_diligence.agents.generator (GroundedGenerator, GroundedAnswer)
   - due_diligence.agents.citation_verifier (CitationVerifier, VerifiedAnswer)

6. Non-Responsibilities:
   - Does NOT manage database clustering or vector schemas (Phase B3).
   - Does NOT manage terminal I/O or CLI prompt loops (cli.py).
   - Does NOT implement multi-agent task routing or autonomous agent loops.
"""

import logging
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from due_diligence.agents.citation_verifier import CitationVerifier, VerifiedAnswer
from due_diligence.agents.generator import GroundedGenerator
from due_diligence.core.config import Settings, get_settings
from due_diligence.retrieval.context_builder import EvidenceContextBuilder
from due_diligence.retrieval.pipeline import RetrievalPipeline
from due_diligence.retrieval.query_rewriter import ConversationTurn
from due_diligence.retrieval.schemas import EvidenceContext, PipelineQueryResult
from due_diligence.trace import TraceRecorder

logger = logging.getLogger(__name__)


# ── Response Model ──────────────────────────────────────────────────────────


class DiligenceResponse(BaseModel):
    """
    Immutable container representing the complete end-to-end diligence pipeline output.
    """

    model_config = ConfigDict(frozen=True)

    verified_answer: VerifiedAnswer = Field(
        ...,
        description="Grounded diligence answer with verified footnote citations from Phase B8",
    )
    evidence_context: EvidenceContext = Field(
        ...,
        description="Structured XML context, active evidence items, and lookup map from Phase B6",
    )
    pipeline_result: PipelineQueryResult = Field(
        ...,
        description="Retrieval trace preserving original query, retrieval query, and candidates",
    )

    @property
    def answer_text(self) -> str:
        """Resolved answer text with footnote citations."""
        return self.verified_answer.answer_text

    @property
    def original_query(self) -> str:
        """The user's original query wording."""
        return self.pipeline_result.original_query

    @property
    def retrieval_query(self) -> str:
        """Disambiguated query used for retrieval and reranking."""
        return self.pipeline_result.retrieval_query

    @property
    def is_uncertain(self) -> bool:
        """Whether the answer represents an explicit uncertainty or refusal."""
        return self.verified_answer.is_uncertain


# ── End-to-End Orchestrator Component ───────────────────────────────────────


class DiligencePipeline:
    """
    End-to-end AI Due Diligence Copilot pipeline.

    Orchestrates the complete 6-stage flow:
    1. Query Rewriter: resolves conversational pronouns into a standalone retrieval_query,
       strictly preserving the user's original_query.
    2. B4 Hybrid Retrieval: searches Weaviate Cloud (vector + BM25) using retrieval_query.
    3. B5 LLM Reranker: ranks candidate chunk IDs by direct evidence value using
       the fast LLM with both retrieval_query and original_query, selecting Top 7.
    4. B6 Evidence Context Builder: accepts both original_query and retrieval_query along
       with top candidates, formats structured XML context, and prepares the evidence map.
    5. B7 Grounded Generator: synthesizes a factual diligence answer strictly from
       original_query and the B6 EvidenceContext.
    6. B8 Citation Verifier: validates [EX] tokens against the B6 evidence map and formats
       clean numbered footnotes [1], [2].
    """

    def __init__(
        self,
        settings: Settings | None = None,
        retrieval_pipeline: RetrievalPipeline | None = None,
        context_builder: EvidenceContextBuilder | None = None,
        generator: GroundedGenerator | None = None,
        citation_verifier: CitationVerifier | None = None,
    ) -> None:
        """
        Initialize the end-to-end DiligencePipeline with injectable components.

        Reuses component and model instances across execution calls.
        """
        self._settings = settings or get_settings()
        self._retrieval_pipeline = retrieval_pipeline or RetrievalPipeline(settings=self._settings)
        self._context_builder = context_builder or EvidenceContextBuilder(settings=self._settings)

        # Share the OpenAI client between query_rewriter and generator if available
        shared_client: Any = getattr(self._retrieval_pipeline.query_rewriter, "_client", None)
        self._generator = generator or GroundedGenerator(
            settings=self._settings,
            client=shared_client,
        )
        self._citation_verifier = citation_verifier or CitationVerifier()

        logger.info(
            "Initialized DiligencePipeline (generator_model='%s', reranker='%s')",
            self._generator.model_name,
            self._retrieval_pipeline.reranker.model_name,
        )

    @property
    def settings(self) -> Settings:
        """Active application settings."""
        return self._settings

    @property
    def retrieval_pipeline(self) -> RetrievalPipeline:
        """Underlying retrieval pipeline (Phases B4, B5)."""
        return self._retrieval_pipeline

    @property
    def context_builder(self) -> EvidenceContextBuilder:
        """Active evidence context builder (Phase B6)."""
        return self._context_builder

    @property
    def generator(self) -> GroundedGenerator:
        """Active grounded LLM generator (Phase B7)."""
        return self._generator

    @property
    def citation_verifier(self) -> CitationVerifier:
        """Active citation verifier (Phase B8)."""
        return self._citation_verifier

    def run(
        self,
        user_query: str,
        conversation_history: Sequence[ConversationTurn] | None = None,
        company: str | None = None,
        financial_period: str | None = None,
        top_n: int | None = None,
        top_k: int | None = None,
        rerank_candidate_count: int | None = None,
        trace: TraceRecorder | None = None,
    ) -> DiligenceResponse:
        """
        Execute the end-to-end diligence pipeline for a user question.

        Args:
            user_query: The user's input question.
            conversation_history: Optional sequence of prior dialogue turns.
            company: Optional company filter for deal room isolation.
            financial_period: Optional financial period filter.
            top_n: Number of candidates to retrieve in B4 hybrid search (defaults to settings.retrieval_top_n).
            top_k: Number of candidates to retain after B5 reranking (defaults to settings.rerank_top_k).
            rerank_candidate_count: Number of candidates to score in B5 reranker (defaults to settings.rerank_candidate_count).
            trace: Optional observability recorder (no behavior change when None).

        Returns:
            DiligenceResponse containing the verified answer, context, and retrieval trace.
        """
        effective_top_n = top_n if top_n is not None else self._settings.retrieval_top_n
        effective_top_k = top_k if top_k is not None else self._settings.rerank_top_k
        effective_cand_count = (
            rerank_candidate_count
            if rerank_candidate_count is not None
            else self._settings.rerank_candidate_count
        )

        # Step 1-3: Retrieval pipeline (QueryRewriter -> B4 Weaviate Hybrid -> B5 LLM Reranker)
        # Note: B4 and B5 strictly use retrieval_query while original_query is preserved.
        pipeline_result = self._retrieval_pipeline.run(
            user_query=user_query,
            conversation_history=conversation_history,
            company=company,
            financial_period=financial_period,
            top_n=effective_top_n,
            top_k=effective_top_k,
            rerank_candidate_count=effective_cand_count,
            trace=trace,
        )

        # Step 4: B6 Context Assembly receives both original_query and retrieval_query
        if trace is not None:
            with trace.time("b6"):
                evidence_context = self._context_builder.build_context(
                    candidates=pipeline_result.candidates,
                    original_query=pipeline_result.original_query,
                    retrieval_query=pipeline_result.retrieval_query,
                )
        else:
            evidence_context = self._context_builder.build_context(
                candidates=pipeline_result.candidates,
                original_query=pipeline_result.original_query,
                retrieval_query=pipeline_result.retrieval_query,
            )

        # Step 5: B7 Grounded Generation from user's original query + B6 formatted context
        grounded_answer = self._generator.generate(
            query=pipeline_result.original_query,
            context=evidence_context,
            fallback_on_error=True,
            trace=trace,
        )

        # Step 6: B8 Citation Verification against B6 evidence map
        if trace is not None:
            with trace.time("b8"):
                verified_answer = self._citation_verifier.verify(
                    grounded_answer=grounded_answer,
                    context=evidence_context,
                )
        else:
            verified_answer = self._citation_verifier.verify(
                grounded_answer=grounded_answer,
                context=evidence_context,
            )

        return DiligenceResponse(
            verified_answer=verified_answer,
            evidence_context=evidence_context,
            pipeline_result=pipeline_result,
        )
