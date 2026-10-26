"""
Grounded generation subsystem for Phase B7 of the RAG pipeline.

1. Purpose:
   Synthesizes source-grounded, audit-ready diligence answers strictly from B6
   EvidenceContext and the user's original query, enforcing verifiable [EX] citation
   tokens and explicit uncertainty declarations.

2. Role in Overall RAG Pipeline:
   Acts as Phase B7 Generation between Phase B6 (Evidence Context Builder) and Phase B8
   (Citation Engine). Consumes B6 EvidenceContext, constructs a strict grounding prompt,
   invokes the generation LLM with JSON structured output, and returns a GroundedAnswer.

3. Expected Inputs:
   - query: str | None (Original user question. Falls back to context.original_query)
   - context: EvidenceContext (Structured XML evidence context from Phase B6)
   - fallback_on_error: bool (Whether to return a safe uncertain answer on failure)

4. Outputs:
   - GroundedAnswer: Pydantic model with structured claims (text + evidence_ids),
     assembled answer_text, cited_evidence_ids, and is_uncertain.

5. Dependencies:
   - openai (OpenAI client for chat completions)
   - pydantic (BaseModel, ConfigDict, Field)
   - due_diligence.core.config (Settings, get_settings)
   - due_diligence.retrieval.schemas (EvidenceContext)
   - due_diligence.prompts.generation (GENERATION_SYSTEM_PROMPT, format_generation_user_prompt)

6. Non-Responsibilities:
   - Does NOT perform document retrieval or query Weaviate (Phase B4).
   - Does NOT rerank candidate passages (Phase B5).
   - Does NOT rewrite user queries (QueryRewriter).
   - Does NOT assemble or format evidence context (Phase B6).
   - Does NOT resolve [EX] tokens to physical page numbers or filenames (Phase B8).
   - Does NOT invent financial figures or facts absent from context.
"""

import json
import logging
import os
import re
from typing import Any

import openai
from pydantic import BaseModel, ConfigDict, Field

from due_diligence.core.config import Settings, get_settings
from due_diligence.prompts.generation import (
    GENERATION_SYSTEM_PROMPT,
    format_generation_user_prompt,
)
from due_diligence.retrieval.schemas import EvidenceContext
from due_diligence.trace import TokenUsage, TraceRecorder

logger = logging.getLogger(__name__)


# ── Typed Exceptions ────────────────────────────────────────────────────────


class GeneratorError(Exception):
    """Base exception for all generation operations."""


class GeneratorConfigurationError(GeneratorError):
    """Raised when generator initialization or configuration is invalid."""


class GeneratorInputError(GeneratorError):
    """Raised when input query or context is invalid."""


class GeneratorModelError(GeneratorError):
    """Raised when the LLM API call fails or produces unparseable output."""


# ── Structured Domain Models ────────────────────────────────────────────────


class Claim(BaseModel):
    """
    One atomic factual claim with the evidence IDs that support it.

    A claim may cite several evidence IDs, and the same evidence ID may appear in
    several claims. B7 assembles the user-facing text and [EX] tokens from this
    structure; B8 verifies every claim against every evidence ID it cites.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(
        ...,
        description="Atomic factual claim without [EX] tokens",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Context-local evidence IDs supporting this claim (e.g. ['E1', 'E2'])",
    )


class GroundedAnswer(BaseModel):
    """
    Synthesized diligence answer strictly grounded in retrieved source evidence.

    Preserves verifiable evidence citations ([E1], [E2]) and explicit uncertainty flags.
    """

    model_config = ConfigDict(frozen=True)

    answer_text: str = Field(
        ...,
        description="Synthesized text with [EX] citation tokens",
    )
    cited_evidence_ids: list[str] = Field(
        default_factory=list,
        description="List of context-local evidence IDs cited (e.g. ['E1', 'E2'])",
    )
    claims: list[Claim] = Field(
        default_factory=list,
        description="Structured atomic claims with per-claim evidence IDs",
    )
    is_uncertain: bool = Field(
        default=False,
        description="True if evidence was insufficient or information was not disclosed",
    )


# ── Response Parser ─────────────────────────────────────────────────────────

_EVIDENCE_TOKEN_PATTERN = re.compile(r"\[(E\d+)\]", re.IGNORECASE)


def _normalize_evidence_id(value: Any) -> str:
    """Normalize an evidence identifier to the canonical ``E<n>`` form."""
    cleaned = str(value).strip().strip("[]")
    if re.fullmatch(r"e\d+", cleaned, re.IGNORECASE):
        return f"E{cleaned[1:]}"
    return cleaned


def _coerce_evidence_ids(raw: Any) -> list[str]:
    """Flatten and de-duplicate evidence IDs from model output."""
    if raw is None:
        return []

    values: list[Any]
    if isinstance(raw, list):
        values = raw
    else:
        values = [raw]

    evidence_ids: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, str):
            extracted = re.findall(r"E\d+", value, re.IGNORECASE)
            candidates: list[Any] = extracted or [value]
        else:
            candidates = [value]
        for candidate in candidates:
            normalized = _normalize_evidence_id(candidate)
            if normalized and normalized not in seen:
                seen.add(normalized)
                evidence_ids.append(normalized)
    return evidence_ids


def _raise_for_invalid_evidence_ids(
    cited: set[str],
    valid_evidence_ids: set[str] | None,
) -> None:
    """Reject evidence tokens that are not present in the active context."""
    if valid_evidence_ids is None:
        return
    invalid_ids = sorted(c for c in cited if c not in valid_evidence_ids)
    if invalid_ids:
        raise GeneratorModelError(
            f"Model response cited invalid evidence token(s) {invalid_ids} not present in active "
            f"context evidence IDs: {sorted(valid_evidence_ids)}. Hallucinated citations are strictly prohibited."
        )


def _parse_claims(
    raw_claims: Any,
    valid_evidence_ids: set[str] | None,
) -> list[Claim]:
    """Parse the model's structured ``claims`` array into atomic claims."""
    if not isinstance(raw_claims, list):
        raise GeneratorModelError("Model response field 'claims' must be a JSON array.")

    claims: list[Claim] = []
    all_cited: set[str] = set()

    for raw_claim in raw_claims:
        if isinstance(raw_claim, str):
            text = raw_claim.strip()
            raw_evidence: Any = []
        elif isinstance(raw_claim, dict):
            text = str(
                raw_claim.get("text")
                or raw_claim.get("claim")
                or raw_claim.get("statement")
                or ""
            ).strip()
            raw_evidence = (
                raw_claim.get("evidence_ids")
                or raw_claim.get("citations")
                or raw_claim.get("evidence")
                or []
            )
        else:
            raise GeneratorModelError(
                f"Each claim must be an object or string, got {type(raw_claim).__name__}."
            )

        if not text:
            continue

        # Reconcile tokens accidentally embedded in the claim text, then strip
        # them so B7 can place exactly one citation group after the claim.
        text_tokens = [match.group(1) for match in _EVIDENCE_TOKEN_PATTERN.finditer(text)]
        text = _EVIDENCE_TOKEN_PATTERN.sub("", text)
        text = re.sub(r"\s+([.,;:!?])", r"\1", text).strip()
        if not text:
            continue

        evidence_ids = _coerce_evidence_ids(raw_evidence)
        for token in text_tokens:
            normalized = _normalize_evidence_id(token)
            if normalized not in evidence_ids:
                evidence_ids.append(normalized)

        all_cited.update(evidence_ids)
        claims.append(Claim(text=text, evidence_ids=evidence_ids))

    if not claims:
        raise GeneratorModelError("Parsed response contains no non-empty 'claims'.")

    _raise_for_invalid_evidence_ids(all_cited, valid_evidence_ids)
    return claims


def _assemble_answer_text(claims: list[Claim]) -> str:
    """Build the user-facing answer with one citation group per atomic claim."""
    parts: list[str] = []
    for claim in claims:
        text = claim.text.strip()
        if claim.evidence_ids:
            citations = " ".join(f"[{evidence_id}]" for evidence_id in claim.evidence_ids)
            if text.endswith((".", "!", "?")):
                text = f"{text[:-1]} {citations}{text[-1]}"
            else:
                text = f"{text} {citations}"
        if not text.endswith((".", "!", "?", ";", ":")):
            text += "."
        parts.append(text)
    return " ".join(parts)


def _resolve_uncertainty(
    data: dict[str, Any],
    answer_text: str,
    cited_evidence_ids: list[str],
) -> bool:
    """Resolve the uncertainty flag with the existing safety fallback."""
    raw_uncertain = data.get("is_uncertain")
    lower_text = answer_text.lower()
    if raw_uncertain is not None:
        is_uncertain = bool(raw_uncertain)
        if (
            not is_uncertain
            and not cited_evidence_ids
            and (
                "not disclosed in the ingested document" in lower_text
                or "does not contain sufficient information" in lower_text
            )
        ):
            is_uncertain = True
        return is_uncertain
    return (
        not cited_evidence_ids
        or "not disclosed in the ingested document" in lower_text
        or "does not contain sufficient information" in lower_text
    )


def parse_grounded_answer(
    raw_response: str,
    valid_evidence_ids: set[str] | None = None,
) -> GroundedAnswer:
    """
    Deterministically parse raw LLM JSON completion into a GroundedAnswer.

    Reconciles evidence citations between the structured array and the body text,
    validates cited evidence IDs against the active context, and flags uncertainty.

    Args:
        raw_response: Raw text returned by LLM chat completion.
        valid_evidence_ids: Optional set of allowed evidence IDs (e.g. {'E1', 'E2'}).

    Returns:
        Validated GroundedAnswer instance.

    Raises:
        GeneratorModelError: If raw_response cannot be parsed into the expected JSON structure.
    """
    if not raw_response or not raw_response.strip():
        raise GeneratorModelError("Received empty response from generation model.")

    cleaned = raw_response.strip()

    # Strip markdown code blocks if present
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Attempt direct JSON deserialization
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: look for embedded JSON object {...}
        json_obj_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if json_obj_match:
            try:
                data = json.loads(json_obj_match.group(1).strip())
            except json.JSONDecodeError as exc:
                raise GeneratorModelError(
                    f"Malformed model response; could not parse JSON: {exc}"
                ) from exc
        else:
            raise GeneratorModelError(
                f"Model response does not contain valid JSON: {cleaned[:150]}"
            )

    if not isinstance(data, dict):
        raise GeneratorModelError(
            f"Expected JSON object from model, got {type(data).__name__}."
        )

    # Preferred schema: structured atomic claims with explicit evidence sets.
    raw_claims = data.get("claims")
    if isinstance(raw_claims, list) and raw_claims:
        claims = _parse_claims(raw_claims, valid_evidence_ids)
        answer_text = _assemble_answer_text(claims)

        ordered_cites: list[str] = []
        seen_cites: set[str] = set()
        for claim in claims:
            for cite in claim.evidence_ids:
                if cite not in seen_cites:
                    seen_cites.add(cite)
                    ordered_cites.append(cite)

        return GroundedAnswer(
            answer_text=answer_text,
            cited_evidence_ids=ordered_cites,
            claims=claims,
            is_uncertain=_resolve_uncertainty(data, answer_text, ordered_cites),
        )

    # Backward-compatible fallback: free-form answer_text with inline tokens.
    answer_text = str(data.get("answer_text") or data.get("answer") or "").strip()
    if not answer_text:
        raise GeneratorModelError("Parsed response contains empty 'answer_text'.")

    # Extract explicit citations from JSON array
    raw_citations: Any = data.get("cited_evidence_ids") or data.get("citations") or []
    if not isinstance(raw_citations, list):
        raw_citations = []
    explicit_cites = _coerce_evidence_ids(raw_citations)

    # Extract citations embedded directly in the answer text (e.g. "[E1]")
    text_cites = [match.group(1) for match in _EVIDENCE_TOKEN_PATTERN.finditer(answer_text)]

    # Check for invalid citation tokens
    all_cited = set(text_cites).union(explicit_cites)
    _raise_for_invalid_evidence_ids(all_cited, valid_evidence_ids)

    # Reconcile citations: preserve text occurrence order, followed by explicit array
    ordered_cites = []
    seen_cites = set()

    for cite in text_cites + explicit_cites:
        if cite not in seen_cites:
            seen_cites.add(cite)
            ordered_cites.append(cite)

    return GroundedAnswer(
        answer_text=answer_text,
        cited_evidence_ids=ordered_cites,
        is_uncertain=_resolve_uncertainty(data, answer_text, ordered_cites),
    )


# ── Grounded Generator Component ────────────────────────────────────────────


class GroundedGenerator:
    """
    Evidence-grounded answer generation engine for Phase B7.

    Guarantees:
    - Answers the user's original query using exclusively the provided EvidenceContext.
    - Zero additional retrieval or Weaviate calls.
    - Zero financial hallucinations or uncited metrics.
    - Explicit uncertainty whenever evidence is missing or ambiguous.
    - Strict [EX] citation tokens matching active context evidence items.
    - Untrusted data boundary: treats evidence text as data, ignoring embedded instructions.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: openai.OpenAI | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        """
        Initialize the Grounded Generator.

        Args:
            settings: Optional application settings. Defaults to get_settings().
            client: Optional injected OpenAI client (for unit testing without API keys).
            model_name: Optional LLM model identifier. Defaults to settings.generation_model.
            api_key: Optional explicit API key. Defaults to settings.openai_api_key or OPENAI_API_KEY env.
            temperature: Sampling temperature (default 0.0 for deterministic factual synthesis).
            max_tokens: Maximum completion tokens (default 1024).
        """
        self._settings = settings or get_settings()
        self._model_name = model_name or self._settings.generation_model
        self._temperature = temperature
        self._max_tokens = max_tokens

        self._client = client
        if self._client is None:
            resolved_key = (
                api_key or self._settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
            )
            if not resolved_key or not resolved_key.strip():
                raise GeneratorConfigurationError(
                    "OpenAI API key must be provided via settings, api_key argument, "
                    "or OPENAI_API_KEY environment variable."
                )
            self._client = openai.OpenAI(
                api_key=resolved_key,
                timeout=45.0,
                max_retries=2,
            )

        logger.info(
            "Initialized GroundedGenerator (model='%s', temp=%.2f, max_tokens=%d)",
            self._model_name,
            self._temperature,
            self._max_tokens,
        )

    @property
    def model_name(self) -> str:
        """Configured generation LLM model name."""
        return self._model_name

    @property
    def temperature(self) -> float:
        """Sampling temperature."""
        return self._temperature

    @property
    def max_tokens(self) -> int:
        """Maximum completion token limit."""
        return self._max_tokens

    def generate(
        self,
        query: str | None = None,
        context: EvidenceContext | None = None,
        *,
        fallback_on_error: bool = False,
        trace: TraceRecorder | None = None,
    ) -> GroundedAnswer:
        """
        Generate a factually grounded diligence answer from B6 EvidenceContext.

        Args:
            query: The user's original query. If None, falls back to context.original_query.
            context: The EvidenceContext emitted by Context Builder (Phase B6).
            fallback_on_error: If True, returns a safe uncertain GroundedAnswer on failure.
            trace: Optional observability recorder (no behavior change when None).

        Returns:
            GroundedAnswer containing the answer text, cited evidence IDs, and uncertainty flag.

        Raises:
            GeneratorInputError: If query or context is missing or invalid.
            GeneratorModelError: If LLM call or response parsing fails and fallback_on_error is False.
        """
        # Validate context
        if context is None or not isinstance(context, EvidenceContext):
            raise GeneratorInputError(
                f"Expected context to be an EvidenceContext instance, got {type(context).__name__}."
            )

        # Resolve and validate query
        effective_query = query if query is not None else context.original_query
        if not isinstance(effective_query, str):
            raise GeneratorInputError(
                f"Expected query to be str, got {type(effective_query).__name__}."
            )
        cleaned_query = effective_query.strip()
        if not cleaned_query:
            raise GeneratorInputError("User query cannot be empty or whitespace-only.")

        # Fast path: If context contains zero evidence passages, return explicit uncertainty immediately
        if not context.evidence_items or "<no_evidence>" in context.formatted_context:
            logger.info("Context contains no active evidence passages. Returning explicit uncertainty.")
            if trace is not None:
                trace.note("b7", "fast_path", True)
                trace.note("b7", "llm_called", False)
                trace.note("b7", "model", self._model_name)
            return GroundedAnswer(
                answer_text="Not disclosed in the ingested document. No relevant evidence passages were found to answer the query.",
                cited_evidence_ids=[],
                is_uncertain=True,
            )

        valid_evidence_ids = set(context.evidence_map.keys())

        # Format user prompt
        user_prompt = format_generation_user_prompt(cleaned_query, context.formatted_context)
        if trace is not None:
            trace.note("b7", "fast_path", False)
            trace.note("b7", "llm_called", True)
            trace.note("b7", "model", self._model_name)
            trace.note("b7", "temperature", self._temperature)
            trace.note("b7", "max_tokens", self._max_tokens)
            trace.note("b7", "evidence_ids", sorted(valid_evidence_ids))
            trace.note("b7", "system_prompt", GENERATION_SYSTEM_PROMPT)
            trace.note("b7", "user_prompt", user_prompt)

        # Execute LLM generation call
        try:
            assert self._client is not None
            if trace is not None:
                with trace.time("b7"):
                    response = self._client.chat.completions.create(
                        model=self._model_name,
                        messages=[
                            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                        response_format={"type": "json_object"},
                    )
            else:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    response_format={"type": "json_object"},
                )
            raw_content = response.choices[0].message.content or ""
            if trace is not None:
                trace.note("b7", "raw_response", raw_content)
                trace.note("b7", "usage", TokenUsage.from_openai_response(response))
            return parse_grounded_answer(
                raw_content,
                valid_evidence_ids=valid_evidence_ids,
            )

        except Exception as exc:
            if fallback_on_error:
                logger.warning(
                    "Grounded generation invocation failed (%s). Returning fallback uncertain answer.",
                    exc,
                )
                if trace is not None:
                    trace.note("b7", "error", f"{type(exc).__name__}: {exc}")
                return GroundedAnswer(
                    answer_text="An error occurred while generating the answer from the ingested document.",
                    cited_evidence_ids=[],
                    is_uncertain=True,
                )
            if isinstance(exc, GeneratorModelError):
                raise
            raise GeneratorModelError(
                f"Grounded generation LLM invocation failed: {exc}"
            ) from exc


# ── Convenience Wrapper ─────────────────────────────────────────────────────


def generate_grounded_answer(
    query: str | None = None,
    context: EvidenceContext | None = None,
    settings: Settings | None = None,
    client: openai.OpenAI | None = None,
    model_name: str | None = None,
    *,
    fallback_on_error: bool = False,
) -> GroundedAnswer:
    """
    Functional convenience wrapper for grounded answer generation.

    Instantiates a temporary ``GroundedGenerator`` or uses injected dependencies.
    """
    generator = GroundedGenerator(
        settings=settings,
        client=client,
        model_name=model_name,
    )
    return generator.generate(
        query=query,
        context=context,
        fallback_on_error=fallback_on_error,
    )
