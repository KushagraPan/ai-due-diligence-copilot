"""
Retrieval pipeline orchestrating Query Rewriter, B4 Hybrid Retrieval, and B5 LLM Reranker.

1. Purpose:
   Coordinates user query processing, conversational reference resolution,
   hybrid candidate search, and LLM evidence reranking into a
   unified retrieval pipeline.

2. Role in Overall RAG Pipeline:
   Sits between the user interaction layer and Phase B6 Context Builder.
   Transforms inbound conversational queries into standalone retrieval queries,
   executes Weaviate hybrid search (B4), reranks candidates by direct evidence
   value with the LLM reranker (B5), and emits a ``PipelineQueryResult``
   retaining both original and retrieval queries.

3. Expected Inputs:
   - ``user_query: str``: Current user question.
   - ``conversation_history: Sequence[Any] | None``: Prior conversational turns.
   - Optional metadata filters: ``company``, ``document_name``, ``document_type``,
     ``financial_period``, ``is_table``.
   - Optional budget overrides: ``top_n`` (B4 candidates), ``top_k`` (B5 evidence).

4. Outputs:
   - ``PipelineQueryResult``: Frozen container with ``original_query``,
     ``retrieval_query``, ``candidates`` (Top-K ``RerankedResult``), and
     ``raw_retrieval_count``.

5. Dependencies:
   - ``due_diligence.core.config.Settings``, ``get_settings``
   - ``due_diligence.retrieval.query_rewriter.QueryRewriter``
   - ``due_diligence.retrieval.retriever.WeaviateHybridRetriever``
   - ``due_diligence.retrieval.reranker.BaseReranker``, ``LLMReranker``
   - ``due_diligence.retrieval.schemas.PipelineQueryResult``

6. Non-Responsibilities:
   - Does NOT format XML evidence contexts (Phase B6).
   - Does NOT call LLMs for final diligence answer synthesis (Phase B7).
   - Does NOT verify claim-level citations (Phase B8).
   - Does NOT modify B4 retrieval algorithm or Weaviate index structures.
   - Does NOT rewrite queries separately before B5.
"""

import logging
from collections.abc import Sequence
from typing import Any

from due_diligence.core.config import Settings, get_settings
from due_diligence.core.entity_scope import filter_subsidiary_candidates
from due_diligence.retrieval.query_rewriter import (
    QueryRewriter,
    QueryRewriterInputError,
)
from due_diligence.retrieval.reranker import (
    BaseReranker,
    LLMReranker,
)
from due_diligence.retrieval.retriever import WeaviateHybridRetriever
from due_diligence.retrieval.schemas import PipelineQueryResult
from due_diligence.trace import TraceRecorder

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """
    Online retrieval pipeline coordinating Query Rewriting -> B4 Retrieval -> B5 Reranking.

    Invariants:
    - user_query is preserved verbatim in original_query.
    - retrieval_query is produced by QueryRewriter and fed into B4 hybrid search.
    - B5 Reranker scores using the EXACT same retrieval_query.
    - Context Builder and Generation handoff contract is satisfied via PipelineQueryResult.
    """

    def __init__(
        self,
        retriever: WeaviateHybridRetriever | None = None,
        reranker: BaseReranker | None = None,
        query_rewriter: QueryRewriter | None = None,
        settings: Settings | None = None,
        fallback_on_rewrite_error: bool = True,
        rerank_candidate_count: int | None = None,
    ) -> None:
        """
        Initialize the retrieval pipeline with injectable components.

        Args:
            retriever: Optional injected WeaviateHybridRetriever instance.
            reranker: Optional injected BaseReranker instance.
            query_rewriter: Optional injected QueryRewriter instance.
            settings: Optional Settings instance. Defaults to get_settings().
            fallback_on_rewrite_error: Whether rewriting failures safely fall back
                                       to the original query. Defaults to True.
            rerank_candidate_count: Optional override for the number of B4 candidates
                                   forwarded to B5 LLM reranking.
        """
        self._settings = settings or get_settings()
        self._retriever = retriever or WeaviateHybridRetriever(settings=self._settings)
        self._reranker = reranker or LLMReranker(settings=self._settings)
        self._query_rewriter = query_rewriter or QueryRewriter(settings=self._settings)
        self._fallback_on_rewrite_error = fallback_on_rewrite_error
        self._rerank_candidate_count = (
            rerank_candidate_count
            if rerank_candidate_count is not None
            else self._settings.rerank_candidate_count
        )

        logger.info(
            "Initialized RetrievalPipeline (rewriter_model='%s', b4_alpha=%.2f, b4_top_n=%d, "
            "b5_candidates=%d, b5_top_k=%d)",
            self._query_rewriter.model_name,
            self._retriever.alpha,
            self._retriever.top_n,
            self._rerank_candidate_count,
            self._reranker.top_k,
        )

    @property
    def retriever(self) -> WeaviateHybridRetriever:
        """Active Weaviate hybrid retriever."""
        return self._retriever

    @property
    def reranker(self) -> BaseReranker:
        """Active LLM passage reranker."""
        return self._reranker

    @property
    def query_rewriter(self) -> QueryRewriter:
        """Active query rewriter."""
        return self._query_rewriter

    @property
    def rerank_candidate_count(self) -> int:
        """Default number of B4 candidates forwarded to B5 reranking."""
        return self._rerank_candidate_count

    def close(self) -> None:
        """Close owned retrieval resources, including the Weaviate connection."""
        self._retriever.close()

    def run(
        self,
        user_query: str,
        conversation_history: Sequence[Any] | None = None,
        *,
        company: str | None = None,
        document_name: str | None = None,
        document_type: str | None = None,
        financial_period: str | None = None,
        is_table: bool | None = None,
        top_n: int | None = None,
        top_k: int | None = None,
        rerank_candidate_count: int | None = None,
        trace: TraceRecorder | None = None,
    ) -> PipelineQueryResult:
        """
        Execute the retrieval pipeline: Rewriter -> B4 Hybrid Retrieval -> B5 Reranker.

        Args:
            user_query: Inbound user question string.
            conversation_history: Prior conversation turns (bounded by rewriter).
            company: Optional metadata filter for company entity isolation.
            document_name: Optional metadata filter for source filename.
            document_type: Optional metadata filter for document classification.
            financial_period: Optional metadata filter for fiscal period.
            is_table: Optional boolean flag to filter tabular or narrative chunks.
            top_n: Optional candidate pool size override for B4 hybrid retrieval.
            top_k: Optional evidence context size override for B5 reranking.
            rerank_candidate_count: Optional override for candidates forwarded to B5.
            trace: Optional observability recorder (no behavior change when None).

        Returns:
            PipelineQueryResult containing original_query, retrieval_query,
            and top-K reranked evidence passages.

        Raises:
            QueryRewriterInputError: If user_query is empty or whitespace-only.
        """
        if not isinstance(user_query, str):
            raise QueryRewriterInputError(
                f"Expected user_query to be str, got {type(user_query).__name__}."
            )
        cleaned_query = user_query.strip()
        if not cleaned_query:
            raise QueryRewriterInputError("User query cannot be empty or whitespace-only.")

        # Step 1: Query Rewriter produces standalone retrieval_query
        retrieval_query = self._query_rewriter.rewrite(
            user_query=cleaned_query,
            conversation_history=conversation_history,
            fallback_on_error=self._fallback_on_rewrite_error,
            trace=trace,
        )

        # Step 2: Phase B4 Hybrid Retrieval using retrieval_query (Top-N, default: 40)
        b4_candidates = self._retriever.retrieve(
            query=retrieval_query,
            company=company,
            document_name=document_name,
            document_type=document_type,
            financial_period=financial_period,
            is_table=is_table,
            top_n=top_n,
            trace=trace,
        )

        # Step 2b: Entity Scope Filtering — prevent cross-entity subsidiary conflation
        # Uses both the user's original query and rewritten retrieval query
        query_for_filter = cleaned_query if "hdfc bank" in cleaned_query.lower() or "the bank" in cleaned_query.lower() else retrieval_query
        filtered_candidates = filter_subsidiary_candidates(
            query=query_for_filter,
            candidates=b4_candidates,
        )
        if trace is not None:
            removed = [c.chunk_id for c in b4_candidates if c.chunk_id not in {f.chunk_id for f in filtered_candidates}]
            trace.note("entity_filter", "query_used", query_for_filter)
            trace.note("entity_filter", "input_count", len(b4_candidates))
            trace.note("entity_filter", "output_count", len(filtered_candidates))
            trace.note("entity_filter", "removed_chunk_ids", removed)
            trace.note(
                "entity_filter",
                "summary",
                f"{len(removed)} of {len(b4_candidates)} B4 candidates removed as subsidiary-scoped",
            )

        # Step 3: Phase B5 LLM Reranking using the SAME retrieval_query, with the
        # original user query attached for conversational intent/reference context
        effective_candidate_count = (
            rerank_candidate_count
            if rerank_candidate_count is not None
            else self._rerank_candidate_count
        )
        reranked_candidates = self._reranker.rerank(
            query=retrieval_query,
            candidates=filtered_candidates,
            top_k=top_k,
            candidate_count=effective_candidate_count,
            original_query=cleaned_query,
            trace=trace,
        )

        # Step 4: Package handoff container preserving both queries and raw B4 count
        return PipelineQueryResult(
            original_query=cleaned_query,
            retrieval_query=retrieval_query,
            candidates=reranked_candidates,
            raw_retrieval_count=len(b4_candidates),
            raw_candidates=b4_candidates,
        )
