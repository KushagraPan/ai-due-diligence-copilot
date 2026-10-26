"""
Query rewriting and reference disambiguation component.

1. Purpose:
   Transforms conversational, elliptical, or pronoun-heavy user queries into
   standalone, disambiguated retrieval queries optimized for downstream hybrid search
   against corporate diligence documents.

2. Role in Overall RAG Pipeline:
   Operates in the Online Query Pipeline prior to Phase B4 Hybrid Retrieval.
   Disambiguates follow-up questions while leaving standalone queries intact,
   without modifying retriever behavior, Weaviate collections, or answer generation.

3. Expected Inputs:
   - ``user_query: str``: Current user query.
   - ``conversation_history: Sequence[ConversationTurn | dict[str, str] | tuple[str, str]] | None``:
     Prior conversational turns (bounded by ``max_history_turns``).
   - ``fallback_on_error: bool``: Whether to fall back to the original query if LLM calls fail.

4. Outputs:
   - ``str``: Disambiguated standalone retrieval query.

5. Dependencies:
   - ``openai`` (OpenAI client for chat completions)
   - ``pydantic`` (BaseModel, ConfigDict, Field)
   - ``due_diligence.core.config`` (Settings, get_settings)
   - ``due_diligence.prompts.query_rewriter`` (System prompt and message formatting)

6. Non-Responsibilities:
   - Does NOT perform document retrieval or search Weaviate (Phase B4).
   - Does NOT rerank candidate passages (Phase B5).
   - Does NOT synthesize diligence answers or generate reports (Phase B7).
   - Does NOT invent facts, tickers, or numbers absent from conversation history.
   - Does NOT modify B4 retriever behavior or Weaviate schemas.
"""

import logging
import os
import re
from collections.abc import Sequence
from typing import Any

import openai
from pydantic import BaseModel, ConfigDict, Field

from due_diligence.core.config import Settings, get_settings
from due_diligence.prompts.query_rewriter import (
    QUERY_REWRITER_SYSTEM_PROMPT,
    QUERY_REWRITER_TAG,
    format_query_rewriter_user_prompt,
)
from due_diligence.trace import TokenUsage, TraceRecorder

logger = logging.getLogger(__name__)


# ── Typed Exceptions ────────────────────────────────────────────────────────


class QueryRewriterError(Exception):
    """Base exception for all query rewriting operations."""


class QueryRewriterConfigurationError(QueryRewriterError):
    """Raised when configuration or initialization parameters are invalid."""


class QueryRewriterInputError(QueryRewriterError):
    """Raised when input query or history format is invalid."""


class QueryRewriterModelError(QueryRewriterError):
    """Raised when the LLM API call fails or encounters an unrecoverable error."""


# ── Typed Schemas ───────────────────────────────────────────────────────────


class ConversationTurn(BaseModel):
    """
    Typed representation of a single conversation turn.
    """

    model_config = ConfigDict(frozen=True)

    role: str = Field(
        ...,
        description="Role of the speaker (e.g. 'user', 'assistant', 'system')",
    )
    content: str = Field(
        ...,
        description="Message content text",
    )


def normalize_history(
    history: Sequence[Any] | None,
) -> list[ConversationTurn]:
    """
    Normalize diverse conversation history representations into typed ConversationTurn instances.

    Supports:
    - ``ConversationTurn`` instances
    - Dictionaries with ``role`` and ``content`` keys, or single key-value mappings
    - Two-element tuples: ``(role, content)``

    Raises:
        QueryRewriterInputError: If any history item cannot be safely converted.
    """
    if history is None:
        return []

    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise QueryRewriterInputError(
            f"Expected conversation history to be a sequence, got {type(history).__name__}."
        )

    normalized: list[ConversationTurn] = []
    for idx, item in enumerate(history):
        if isinstance(item, ConversationTurn):
            normalized.append(item)
        elif isinstance(item, dict):
            if "role" in item or "content" in item:
                if "role" not in item or "content" not in item:
                    raise QueryRewriterInputError(
                        f"Invalid history dictionary at index {idx}: missing 'role' and 'content' keys."
                    )
                role = str(item["role"])
                content = str(item["content"])
            elif len(item) == 1:
                role, content = next(iter(item.items()))
                role, content = str(role), str(content)
            else:
                raise QueryRewriterInputError(
                    f"Invalid history dictionary at index {idx}: expected 'role' and 'content' keys or single mapping."
                )
            if not content.strip():
                continue
            normalized.append(ConversationTurn(role=role, content=content))
        elif isinstance(item, (tuple, list)):
            if len(item) != 2:
                raise QueryRewriterInputError(
                    f"Invalid history tuple at index {idx}: expected (role, content), got length {len(item)}."
                )
            role, content = str(item[0]), str(item[1])
            if not content.strip():
                continue
            normalized.append(ConversationTurn(role=role, content=content))
        else:
            raise QueryRewriterInputError(
                f"Unsupported history item type at index {idx}: {type(item).__name__}."
            )

    return normalized


def parse_rewritten_query(raw_response: str, fallback_query: str) -> str:
    """
    Deterministically parse the model output to extract the clean rewritten query.

    Checks:
    1. <rewritten_query>...</rewritten_query> XML tag enclosure.
    2. Markdown code blocks (```).
    3. Prefixes like 'Rewritten query:', 'Standalone query:', 'Query:'.
    4. Enclosing quotation marks.
    5. Falls back to original query if extracted string is empty.

    Args:
        raw_response: Raw text returned by the model completion.
        fallback_query: Original user query used as safe fallback.

    Returns:
        Clean, disambiguated query string.
    """
    if not raw_response or not raw_response.strip():
        return fallback_query.strip()

    # 1. Search for XML tag
    tag_pattern = rf"<{QUERY_REWRITER_TAG}>(.*?)</{QUERY_REWRITER_TAG}>"
    match = re.search(tag_pattern, raw_response, re.DOTALL | re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
    else:
        candidate = raw_response.strip()

    # 2. Strip code fences if present
    fence_match = re.search(r"```(?:xml|text|json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()

    # 3. Strip common query prefixes
    candidate = re.sub(
        r"^(?:rewritten\s+query|standalone\s+query|reformulated\s+query|query)\s*:\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()

    # 4. Strip surrounding single or double quotes
    if len(candidate) >= 2 and (
        (candidate.startswith('"') and candidate.endswith('"'))
        or (candidate.startswith("'") and candidate.endswith("'"))
    ):
        candidate = candidate[1:-1].strip()

    # 5. Fallback if empty or whitespace-only
    if not candidate:
        return fallback_query.strip()

    return candidate


# ── Query Rewriter Component ────────────────────────────────────────────────


class QueryRewriter:
    """
    Reformulates conversational queries into standalone retrieval queries.

    Design Principles:
    - Uses the configured query-rewriter LLM (default: gpt-4o-mini).
    - Zero fact injection: strictly preserves explicit entities, metrics, and periods.
    - Standalone queries: avoids unnecessary rewriting when history is empty or query is self-contained.
    - Untrusted boundary: treats conversation history strictly as data to prevent prompt injection.
    - Bounded window: limits history to recent turns to avoid topic confusion.
    - Resilience: supports deterministic parsing and configurable failure fallback.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: openai.OpenAI | None = None,
        model_name: str | None = None,
        max_history_turns: int = 4,
        api_key: str | None = None,
    ) -> None:
        """
        Initialize the Query Rewriter.

        Args:
            settings: Optional application settings. Defaults to get_settings().
            client: Optional injected OpenAI client (useful for unit testing without API keys).
            model_name: Optional LLM model identifier. Defaults to settings.query_rewriter_model.
            max_history_turns: Maximum number of recent conversation turns to consider. Defaults to 4.
            api_key: Optional explicit API key. Defaults to settings.openai_api_key or OPENAI_API_KEY env var.
        """
        self._settings = settings or get_settings()
        self._model_name = model_name or self._settings.query_rewriter_model
        self._max_history_turns = max_history_turns

        if self._max_history_turns < 1:
            raise QueryRewriterConfigurationError(
                f"max_history_turns must be a positive integer, got {self._max_history_turns}."
            )

        self._client = client
        if self._client is None:
            resolved_key = (
                api_key or self._settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
            )
            if not resolved_key or not resolved_key.strip():
                raise QueryRewriterConfigurationError(
                    "OpenAI API key must be provided via settings, api_key argument, or OPENAI_API_KEY environment variable."
                )
            self._client = openai.OpenAI(
                api_key=resolved_key,
                timeout=30.0,
                max_retries=2,
            )

        logger.info(
            "Initialized QueryRewriter (model='%s', max_history_turns=%d)",
            self._model_name,
            self._max_history_turns,
        )

    @property
    def model_name(self) -> str:
        """LLM model identifier used for rewriting."""
        return self._model_name

    @property
    def max_history_turns(self) -> int:
        """Maximum bounded number of conversation turns inspected."""
        return self._max_history_turns

    def rewrite(
        self,
        user_query: str,
        conversation_history: Sequence[Any] | None = None,
        *,
        fallback_on_error: bool = False,
        trace: TraceRecorder | None = None,
    ) -> str:
        """
        Rewrite a user query into a standalone retrieval query using conversation context.

        Args:
            user_query: Inbound user question string.
            conversation_history: Optional sequence of prior conversation turns.
            fallback_on_error: If True, returns original user query on LLM failures instead of raising.
            trace: Optional observability recorder (no behavior change when None).

        Returns:
            Disambiguated, standalone search query string.

        Raises:
            QueryRewriterInputError: If user_query is empty, whitespace-only, or not a string.
            QueryRewriterModelError: If the LLM invocation fails and fallback_on_error is False.
        """
        # Validate query
        if not isinstance(user_query, str):
            raise QueryRewriterInputError(
                f"Expected user_query to be str, got {type(user_query).__name__}."
            )
        cleaned_query = user_query.strip()
        if not cleaned_query:
            raise QueryRewriterInputError("User query cannot be empty or whitespace-only.")

        # Normalize history
        normalized_history = normalize_history(conversation_history)

        if trace is not None:
            trace.note("rewriter", "model", self._model_name)
            trace.note(
                "rewriter",
                "history",
                [{"role": t.role, "content": t.content} for t in normalized_history],
            )
            trace.note("rewriter", "history_turns", len(normalized_history))

        # Fast path: Empty history means query has no previous context to resolve
        if not normalized_history:
            if trace is not None:
                trace.note("rewriter", "fast_path", True)
                trace.note("rewriter", "llm_called", False)
            return cleaned_query

        # Bound history window to most recent turns
        if len(normalized_history) > self._max_history_turns:
            bounded_history = normalized_history[-self._max_history_turns:]
        else:
            bounded_history = normalized_history

        # Format prompt
        user_prompt = format_query_rewriter_user_prompt(cleaned_query, bounded_history)
        if trace is not None:
            trace.note("rewriter", "fast_path", False)
            trace.note("rewriter", "llm_called", True)
            trace.note("rewriter", "system_prompt", QUERY_REWRITER_SYSTEM_PROMPT)
            trace.note("rewriter", "user_prompt", user_prompt)

        # Execute LLM call
        try:
            assert self._client is not None
            if trace is not None:
                with trace.time("rewriter"):
                    response = self._client.chat.completions.create(
                        model=self._model_name,
                        messages=[
                            {"role": "system", "content": QUERY_REWRITER_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                        max_tokens=256,
                    )
            else:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": QUERY_REWRITER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
            raw_response = response.choices[0].message.content or ""
            parsed = parse_rewritten_query(raw_response, fallback_query=cleaned_query)
            if trace is not None:
                trace.note("rewriter", "raw_response", raw_response)
                trace.note("rewriter", "usage", TokenUsage.from_openai_response(response))
            return parsed

        except Exception as exc:
            if fallback_on_error:
                logger.warning(
                    "Query rewriting LLM invocation failed (%s). Falling back to original query.",
                    exc,
                )
                if trace is not None:
                    trace.note("rewriter", "error", f"{type(exc).__name__}: {exc}")
                return cleaned_query
            raise QueryRewriterModelError(
                f"Query rewriting LLM invocation failed: {exc}"
            ) from exc


def rewrite_query(
    user_query: str,
    conversation_history: Sequence[Any] | None = None,
    settings: Settings | None = None,
    client: openai.OpenAI | None = None,
    model_name: str | None = None,
    max_history_turns: int = 4,
    *,
    fallback_on_error: bool = False,
) -> str:
    """
    Functional convenience wrapper for query rewriting.

    Instantiates a temporary ``QueryRewriter`` or uses injected dependencies.
    """
    rewriter = QueryRewriter(
        settings=settings,
        client=client,
        model_name=model_name,
        max_history_turns=max_history_turns,
    )
    return rewriter.rewrite(
        user_query=user_query,
        conversation_history=conversation_history,
        fallback_on_error=fallback_on_error,
    )
