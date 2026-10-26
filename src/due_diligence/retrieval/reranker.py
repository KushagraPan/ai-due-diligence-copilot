"""
LLM-based passage reranking layer for Phase B5.

1. Purpose:
   Reranks B4 hybrid retrieval candidates (Top 40) into Top-K evidence passages
   (default: 7) using a fast LLM that judges direct evidence value. The model
   ranks chunk IDs — it never answers the diligence question.

2. Role in Overall RAG Pipeline:
   Acts as Phase B5 between hybrid retrieval (retriever.py) and context
   construction (context_builder.py). Consumes (retrieval_query,
   original_query, RetrievalResult[]) triples, requests an ordered chunk-ID
   ranking, validates every ID against the supplied candidate set, and outputs
   deterministically ordered RerankedResult[] instances preserving B4 scores
   and provenance.

3. Expected Inputs:
   - ``query: str``: Disambiguated retrieval query used for B4 search.
   - ``original_query: str | None``: Original user wording (intent/reference context).
   - ``candidates: list[RetrievalResult]``: Top-N candidates from Phase B4.
   - Optional overrides: ``top_k``, ``candidate_count``.

4. Outputs:
   - ``list[RerankedResult]``: Top-K passages ordered by LLM rank with
     descending pseudo-scores, preserving the original Weaviate hybrid score
     in ``score``.

5. Dependencies:
   - ``openai`` (chat completions with JSON object response format)
   - ``due_diligence.core.config`` (Settings, get_settings)
   - ``due_diligence.prompts.reranker`` (system prompt, candidate formatting)
   - ``due_diligence.retrieval.schemas`` (RetrievalResult, RerankedResult)

6. Non-Responsibilities:
   - Does NOT perform vector or lexical retrieval (Phase B4).
   - Does NOT modify or connect to Weaviate.
   - Does NOT rewrite user queries.
   - Does NOT assemble prompt contexts or XML wire formats (Phase B6).
   - Does NOT generate answers (Phase B7) — the ranking model must not answer.
   - Does NOT use cross-encoders, local scoring models, or two-stage hybrids.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence

import openai

from due_diligence.core.config import Settings, get_settings
from due_diligence.prompts.reranker import (
    LLM_RERANKER_SYSTEM_PROMPT,
    format_llm_reranker_prompt,
)
from due_diligence.retrieval.schemas import RerankedResult, RetrievalResult
from due_diligence.trace import TokenUsage, TraceRecorder

logger = logging.getLogger(__name__)


# ── Typed Exceptions ────────────────────────────────────────────────────────


class RerankerError(Exception):
    """Base exception for all reranker operations."""


class RerankerConfigurationError(RerankerError):
    """Raised when reranker configuration or initialization parameters are invalid."""


class RerankerInputError(RerankerError):
    """Raised when input query or candidate list is invalid."""


class RerankerModelError(RerankerError):
    """Raised when the ranking model call fails irrecoverably."""


# ── Abstract Base Interface ─────────────────────────────────────────────────


class BaseReranker(ABC):
    """
    Abstract interface for passage reranking.

    Consumes a retrieval query (plus optional original user wording) and a
    candidate RetrievalResult list, and returns the top_k candidates as
    RerankedResult.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier."""

    @property
    @abstractmethod
    def top_k(self) -> int:
        """Default number of evidence passages to return."""

    @property
    @abstractmethod
    def candidate_count(self) -> int:
        """Default number of B4 candidates forwarded to reranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
        candidate_count: int | None = None,
        original_query: str | None = None,
        trace: TraceRecorder | None = None,
    ) -> list[RerankedResult]:
        """
        Rerank candidate passages and return top_k sorted by relevance.

        Args:
            query: Disambiguated retrieval query used for B4 search.
            candidates: Candidate passages from hybrid retrieval.
            top_k: Optional evidence limit override.
            candidate_count: Optional candidate count override for reranking.
            original_query: Optional original user wording for intent/reference
                context on conversational queries.
            trace: Optional observability recorder (no behavior change when None).

        Returns:
            List of RerankedResult instances ordered most to least relevant.
        """


def _raw_id_list(content: str) -> list[str]:
    """Best-effort extraction of raw string IDs from a ranking response."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, dict):
        return []
    raw = parsed.get("ranked_chunk_ids")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str)]


def _count_duplicates(ids: list[str]) -> int:
    """Count duplicate entries beyond first occurrence, preserving order-insensitivity."""
    return len(ids) - len(set(ids))


def parse_ranked_chunk_ids(
    content: str,
    valid_chunk_ids: Sequence[str],
) -> list[str]:
    """
    Parse and strictly validate an LLM ranking response.

    Accepts only IDs present in ``valid_chunk_ids``, drops duplicates and
    non-string entries, and preserves the model's ordering. Raises
    ``RerankerModelError`` when the response is not valid JSON or lacks a
    usable ``ranked_chunk_ids`` list.
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RerankerModelError(f"LLM reranker returned invalid JSON: {exc}") from exc

    if not isinstance(parsed, dict) or not isinstance(parsed.get("ranked_chunk_ids"), list):
        raise RerankerModelError(
            "LLM reranker response must be a JSON object with a 'ranked_chunk_ids' list."
        )

    valid_set = set(valid_chunk_ids)
    ranked: list[str] = []
    seen: set[str] = set()
    for item in parsed["ranked_chunk_ids"]:
        if isinstance(item, str) and item in valid_set and item not in seen:
            ranked.append(item)
            seen.add(item)
    return ranked


# ── Production LLM Reranker ─────────────────────────────────────────────────


class LLMReranker(BaseReranker):
    """
    Passage reranker powered by a fast LLM (default: gpt-4o-mini).

    Sends the retrieval query, original question, and B4 candidate chunks to
    the model and converts its ordered chunk-ID ranking into RerankedResult
    instances. Unknown, duplicate, or malformed IDs are rejected; unranked
    candidates are appended in input order for deterministic coverage; fewer
    than top_k valid IDs yields fewer results. API/timeout/parse failures fall
    back to B4 input order without raising.
    """

    def __init__(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
        candidate_count: int | None = None,
        client: openai.OpenAI | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        settings: Settings | None = None,
    ) -> None:
        """
        Initialize the LLM reranker.

        Args:
            model_name: Model identifier (defaults to settings.llm_reranker_model).
            top_k: Number of evidence passages to return (defaults to settings.rerank_top_k).
            candidate_count: Number of candidates to score (defaults to settings.rerank_candidate_count).
            client: Optional injected OpenAI client (useful for unit testing).
            api_key: Optional explicit API key.
            timeout: Per-request timeout in seconds for the ranking call.
            settings: Optional Settings instance. Defaults to get_settings().
        """
        self._settings = settings or get_settings()
        self._model_name = model_name or self._settings.llm_reranker_model

        resolved_top_k = top_k if top_k is not None else self._settings.rerank_top_k
        if resolved_top_k < 1:
            raise RerankerConfigurationError(
                f"top_k must be a positive integer, got {resolved_top_k}."
            )
        self._top_k = resolved_top_k

        resolved_candidate_count = (
            candidate_count if candidate_count is not None else self._settings.rerank_candidate_count
        )
        if resolved_candidate_count < 1:
            raise RerankerConfigurationError(
                f"candidate_count must be a positive integer, got {resolved_candidate_count}."
            )
        self._candidate_count = resolved_candidate_count

        if timeout <= 0:
            raise RerankerConfigurationError(
                f"timeout must be positive, got {timeout}."
            )
        self._timeout = timeout

        if client is not None:
            self._client = client
        else:
            resolved_key = (
                api_key or self._settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
            )
            if not resolved_key or not resolved_key.strip():
                raise RerankerConfigurationError(
                    "OpenAI API key must be provided via settings.openai_api_key or OPENAI_API_KEY env var for LLMReranker."
                )
            self._client = openai.OpenAI(api_key=resolved_key.strip(), timeout=self._timeout)

        logger.info(
            "Initialized LLMReranker (model='%s', top_k=%d, candidate_count=%d)",
            self._model_name,
            self._top_k,
            self._candidate_count,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def candidate_count(self) -> int:
        return self._candidate_count

    def _request_ranking(
        self,
        retrieval_query: str,
        candidates_to_score: list[RetrievalResult],
        original_query: str | None,
        trace: TraceRecorder | None = None,
    ) -> list[str]:
        """Call the ranking model and return validated chunk IDs (may be empty)."""
        prompt = format_llm_reranker_prompt(
            retrieval_query=retrieval_query,
            candidates=candidates_to_score,
            original_query=original_query,
        )
        if trace is not None:
            trace.note("b5", "model", self._model_name)
            trace.note("b5", "original_query", original_query)
            trace.note("b5", "prompt", prompt)
        resp = self._client.chat.completions.create(
            model=self._model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": LLM_RERANKER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content or "{}"
        valid_ids = [c.chunk_id for c in candidates_to_score]
        ranked = parse_ranked_chunk_ids(content, valid_ids)
        if trace is not None:
            raw_ids = _raw_id_list(content)
            valid_set = set(valid_ids)
            trace.note("b5", "raw_response", content)
            trace.note("b5", "parsed_ids", ranked)
            trace.note("b5", "invalid_ids", [i for i in raw_ids if i not in valid_set])
            trace.note("b5", "duplicate_count", _count_duplicates(raw_ids))
            trace.note("b5", "usage", TokenUsage.from_openai_response(resp))
        return ranked

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
        candidate_count: int | None = None,
        original_query: str | None = None,
        trace: TraceRecorder | None = None,
    ) -> list[RerankedResult]:
        """
        Rerank B4 candidates with the LLM and return the top_k chunk IDs as evidence.

        Args:
            query: Disambiguated retrieval query used for B4 search.
            candidates: Candidate passages from Phase B4 hybrid retrieval.
            top_k: Optional evidence limit override.
            candidate_count: Optional candidate count override.
            original_query: Optional original user wording for conversational intent.
            trace: Optional observability recorder (no behavior change when None).

        Returns:
            List of RerankedResult instances ordered most to least relevant,
            with descending pseudo-scores preserving B4 scores in ``score``.
        """
        # 1. Validate query
        if not isinstance(query, str) or not query.strip():
            raise RerankerInputError("Query cannot be empty or whitespace-only for reranking.")

        # 2. Validate candidates
        if not isinstance(candidates, list):
            raise RerankerInputError(
                f"Expected list of RetrievalResult, got {type(candidates).__name__}."
            )
        if not candidates:
            logger.debug("Empty candidate list provided to LLM reranker; returning [].")
            return []

        # 3. Validate top_k
        effective_top_k = top_k if top_k is not None else self._top_k
        if effective_top_k < 1:
            raise RerankerConfigurationError(
                f"top_k override must be a positive integer, got {effective_top_k}."
            )

        # 4. Validate candidate_count and slice candidates
        effective_candidate_count = (
            candidate_count if candidate_count is not None else self._candidate_count
        )
        if effective_candidate_count < 1:
            raise RerankerConfigurationError(
                f"candidate_count override must be a positive integer, got {effective_candidate_count}."
            )

        candidates_to_score = candidates[:effective_candidate_count]
        if not candidates_to_score:
            return []

        # 5. Request LLM ranking; fall back to B4 input order on any failure
        ranked_ids: list[str] = []
        try:
            if trace is not None:
                with trace.time("b5"):
                    ranked_ids = self._request_ranking(
                        retrieval_query=query,
                        candidates_to_score=candidates_to_score,
                        original_query=original_query,
                        trace=trace,
                    )
                trace.note("b5", "fallback", False)
            else:
                ranked_ids = self._request_ranking(
                    retrieval_query=query,
                    candidates_to_score=candidates_to_score,
                    original_query=original_query,
                )
        except (RerankerModelError, openai.OpenAIError, ValueError, TypeError,
                KeyError, IndexError, AttributeError, RuntimeError) as err:
            logger.warning(
                "LLMReranker ranking call failed (%s). Falling back to input candidate order.",
                err,
            )
            if trace is not None:
                trace.note("b5", "fallback", True)
                trace.note("b5", "error", f"{type(err).__name__}: {err}")

        # 6. Append unranked candidates in input order for deterministic coverage
        seen = set(ranked_ids)
        for cand in candidates_to_score:
            if cand.chunk_id not in seen:
                ranked_ids.append(cand.chunk_id)
                seen.add(cand.chunk_id)

        # 7. Convert to RerankedResult with descending pseudo-scores
        by_id = {c.chunk_id: c for c in candidates_to_score}
        reranked_results: list[RerankedResult] = []
        for rank_pos, chunk_id in enumerate(ranked_ids):
            pseudo_score = max(0.01, 1.0 - (rank_pos * 0.02))
            reranked_results.append(
                RerankedResult.from_retrieval_result(
                    candidate=by_id[chunk_id], rerank_score=pseudo_score
                )
            )

        limited_results = reranked_results[:effective_top_k]
        if trace is not None:
            trace.note("b5", "input_count", len(candidates_to_score))
            trace.note("b5", "output_count", len(limited_results))
        return limited_results
