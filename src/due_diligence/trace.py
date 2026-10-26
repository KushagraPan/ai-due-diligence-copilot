"""
Execution trace data model and renderers for RAG pipeline observability.

1. Purpose:
   Captures what actually flows through the pipeline (queries, prompts,
   candidates, scores, evidence, answers, citations, latencies, token usage)
   so learners can follow a query from input to verified answer. Two
   presentation formats: compact (educational) and full (forensic).

2. Role in Overall RAG Pipeline:
   Pure observability side-channel. Components accept an optional
   ``TraceRecorder`` and record inputs/outputs/timings; a ``None`` recorder
   means zero behavior change. Renderers turn a finished
   ``DiligenceResponse`` plus its recorder into terminal text. Never alters
   retrieval, ranking, generation, or verification.

3. Expected Inputs:
   - ``TraceRecorder`` (created per query by the caller, e.g. CLI).
   - ``DiligenceResponse`` + ``Settings`` at render time.

4. Outputs:
   - ``render_compact_trace`` / ``render_full_trace`` strings.
   - Never includes API keys, secrets, or environment variables.

5. Dependencies:
   - Standard library only (time, dataclasses, re).

6. Non-Responsibilities:
   - Does NOT call LLMs, Weaviate, or embeddings.
   - Does NOT change pipeline results with or without a recorder.
   - Does NOT reconstruct unavailable data; renderers print
     "Not captured by current pipeline" for genuine gaps.
"""

import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

NOT_CAPTURED = "Not captured by current pipeline"


@dataclass
class TokenUsage:
    """Token consumption for one LLM call (all fields optional)."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def from_openai_response(cls, response: Any) -> "TokenUsage | None":
        """Best-effort usage extraction; returns None when unavailable."""
        try:
            usage = getattr(response, "usage", None)
            if usage is None:
                return None
            prompt = getattr(usage, "prompt_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
            total = getattr(usage, "total_tokens", None)
            if prompt is None and completion is None and total is None:
                return None
            return cls(input_tokens=prompt, output_tokens=completion, total_tokens=total)
        except (AttributeError, TypeError, ValueError):
            return None

    def render(self) -> str:
        """Single-line rendering with graceful fallbacks."""

        def fmt(value: int | None) -> str:
            return str(value) if value is not None else "—"

        return f"input: {fmt(self.input_tokens)} / output: {fmt(self.output_tokens)} / total: {fmt(self.total_tokens)}"


@dataclass
class StageTrace:
    """Captured facts for one pipeline stage."""

    name: str
    latency_ms: float | None = None
    data: dict[str, Any] = field(default_factory=dict)


class TraceRecorder:
    """
    Per-query observability recorder.

    Components write inputs, outputs, prompts, raw responses, latencies, and
    token usage here. A fresh recorder per query keeps memory bounded.
    """

    def __init__(self) -> None:
        self.stages: dict[str, StageTrace] = {}

    def stage(self, name: str) -> StageTrace:
        """Get-or-create the trace section for a stage."""
        if name not in self.stages:
            self.stages[name] = StageTrace(name=name)
        return self.stages[name]

    def note(self, stage: str, key: str, value: Any) -> None:
        """Record one fact about a stage (no-op safe for any value)."""
        self.stage(stage).data[key] = value

    @contextmanager
    def time(self, stage: str) -> Iterator[None]:
        """Time a block and store its latency under the stage."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stage(stage).latency_ms = (time.perf_counter() - start) * 1000.0


def short_chunk_id(chunk_id: str) -> str:
    """Compact chunk identifier for dense listings (full IDs in full trace)."""
    match = re.search(r"_c(\d+)", chunk_id)
    if match:
        return f"c{match.group(1)}"
    return chunk_id[-12:] if len(chunk_id) > 12 else chunk_id


def _fmt_ms(value: float | None) -> str:
    return f"{value:.0f} ms" if value is not None else NOT_CAPTURED


def _fmt_usage(usage: TokenUsage | None) -> str:
    return usage.render() if usage is not None else NOT_CAPTURED


def _pages(page_numbers: list[int]) -> str:
    return ", ".join(str(p) for p in page_numbers) if page_numbers else "Unknown"


def _support_status(evidence_id: str, warnings: list[str]) -> str:
    """PASS unless a hard support-failure warning names this evidence ID."""
    for warning in warnings:
        if f"[{evidence_id}]" in warning and (
            "absent from evidence text" in warning or "unsupported by evidence text" in warning
        ):
            return "FAIL"
    return "PASS"


def _citing_sentences(raw_answer: str, token: str) -> list[str]:
    """Sentences of the captured B7 answer that carry a citation token."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_answer) if f"[{token}]" in s]


def _order_changed(before: list[str], after: list[str]) -> str:
    if before == after:
        return "No — identical order"
    moved = sum(1 for i, cid in enumerate(after) if i >= len(before) or before[i] != cid)
    return f"Yes — {moved} of {len(after)} positions differ from B4 order"


def render_compact_trace(
    response: Any,
    trace: TraceRecorder | None = None,
    settings: Any = None,
) -> str:
    """1–2 screen educational trace: queries, counts, IDs, scores, citations."""
    data = trace.stages if trace is not None else {}
    result = response.pipeline_result
    verified = response.verified_answer
    context = response.evidence_context

    lines = ["", "🔍 PIPELINE TRACE", ""]
    lines.append("Query")
    lines.append(f"  Original query:   {response.original_query}")
    lines.append(f"  Retrieval query:  {response.retrieval_query}")
    lines.append(
        "  Changed:        "
        + ("Yes" if response.original_query.strip() != response.retrieval_query.strip() else "No")
    )

    rw = data.get("rewriter")
    lines += ["", "🔄 QUERY REWRITER", ""]
    lines.append(f"  Model:          {(rw.data.get('model') if rw else None) or NOT_CAPTURED}")
    if rw and rw.data.get("fast_path") is True:
        lines.append("  Fast path:      Yes (empty history — no LLM call, query unchanged)")
    else:
        turns = (rw.data.get("history_turns") if rw else None) or 0
        lines.append(f"  History used:   {'Yes' if turns else 'No'} ({turns} turns)")
        lines.append(f"  LLM called:     {(rw.data.get('llm_called') if rw else None)}")
    lines.append(f"  Latency:        {_fmt_ms(rw.latency_ms if rw else None)}")
    if rw and rw.data.get("usage") is not None:
        lines.append(f"  Tokens:         {_fmt_usage(rw.data['usage'])}")

    b4 = data.get("b4")
    lines += ["", "🔎 B4 — HYBRID RETRIEVAL", ""]
    lines.append(f"  Query:          {response.retrieval_query}")
    lines.append(f"  Embedding:      {(b4.data.get('embedding_model') if b4 else None) or NOT_CAPTURED}"
                 + (f" ({b4.data['embedding_dimension']}d)" if b4 and b4.data.get("embedding_dimension") else ""))
    lines.append(f"  Alpha / Top-N:  {(b4.data.get('alpha') if b4 else None)} / {(b4.data.get('top_n') if b4 else None)}")
    lines.append("  Vector + BM25:  single fused Weaviate hybrid call (no separate candidate lists in this architecture)")
    lines.append(f"  Returned:       {result.raw_retrieval_count} candidates")
    lines.append(f"  B4 candidates:  {result.raw_retrieval_count}")
    for rank, cand in enumerate(result.raw_candidates, start=1):
        section = (cand.section_heading or "-")[:44]
        lines.append(
            f"  #{rank:<3} {short_chunk_id(cand.chunk_id):<8} pages {_pages(cand.page_numbers):<10} "
            f"score {cand.score:.4f}  {section}"
        )
    lines.append(f"  Latency:        {_fmt_ms(b4.latency_ms if b4 else None)}")

    b5 = data.get("b5")
    b4_ids = [c.chunk_id for c in result.raw_candidates]
    b5_ids = [c.chunk_id for c in result.candidates]
    lines += ["", "🎯 B5 — LLM RERANKER", ""]
    lines.append(f"  Model:          {(b5.data.get('model') if b5 else None) or NOT_CAPTURED}")
    lines.append(f"  Input / output: {len(b4_ids)} → {len(b5_ids)}")
    lines.append(f"  Order changed:  {_order_changed(b4_ids, b5_ids)}")
    for rank, cand in enumerate(result.candidates, start=1):
        section = (cand.section_heading or "-")[:36]
        lines.append(
            f"  Rank {rank:<2} → {short_chunk_id(cand.chunk_id):<8} pages {_pages(cand.page_numbers):<10} "
            f"rerank {cand.rerank_score:.4f} (B4 {cand.score:.4f})  {section}"
        )
    lines.append(f"  Latency:        {_fmt_ms(b5.latency_ms if b5 else None)}")
    if b5 and b5.data.get("usage") is not None:
        lines.append(f"  Tokens:         {_fmt_usage(b5.data['usage'])}")

    lines += ["", "📦 B6 — EVIDENCE CONTEXT", ""]
    lines.append(f"  Evidence items: {len(context.evidence_items)}")
    for item in context.evidence_items:
        section = (item.section_heading or "-")[:40]
        lines.append(
            f"  {item.evidence_id} → {short_chunk_id(item.chunk_id):<8} "
            f"pages {_pages(item.page_numbers):<10} {section}"
        )
    omitted = list(context.omitted_items)
    if omitted:
        lines.append(f"  Omitted:        {len(omitted)} ({'; '.join(o.reason for o in omitted)[:160]})")
    lines.append(f"  Context size:   {len(context.formatted_context)} chars")

    lines += ["", "✍️ B7 — GROUNDED GENERATOR", ""]
    b7 = data.get("b7")
    lines.append(f"  Model:          {(b7.data.get('model') if b7 else None) or NOT_CAPTURED}")
    lines.append(f"  Evidence:       {', '.join(context.evidence_map) or '—'}")
    lines.append(f"  Answer:         {verified.raw_answer_text}")
    lines.append(f"  Cited:          {', '.join(sorted({c.evidence_id for c in verified.citations})) or '—'}")
    lines.append(f"  Uncertain:      {verified.is_uncertain}")
    lines.append(f"  Latency:        {_fmt_ms(b7.latency_ms if b7 else None)}")
    if b7 and b7.data.get("usage") is not None:
        lines.append(f"  Tokens:         {_fmt_usage(b7.data['usage'])}")

    lines += ["", "🛡️ B8 — CITATION VERIFIER", ""]
    warnings = [str(w) for w in verified.grounding_warnings]
    for cit in verified.citations:
        lines.append(f"  {cit.evidence_id}: exists=YES, chunk={short_chunk_id(cit.chunk_id)}, "
                     f"pages={_pages(cit.page_numbers)}, provenance=VALID, "
                     f"claim-support={_support_status(cit.evidence_id, warnings)}")
    if verified.unverified_citations:
        lines.append(f"  Unverified tokens: {', '.join(f'[{t}]' for t in verified.unverified_citations)}")
    lines.append(f"  Summary:        {len(verified.citations)} verified / "
                 f"{len(verified.unverified_citations)} unverified / {len(warnings)} grounding warnings")

    lines += ["", "✅ FINAL ANSWER", "", verified.answer_text, ""]
    total: float = sum(
        s.latency_ms for s in (trace.stages.values() if trace else []) if s.latency_ms is not None
    )
    lines.append(f"Total latency (sum of captured stages): {total:.0f} ms" if total else "Total latency: " + NOT_CAPTURED)
    lines.append("")
    return "\n".join(lines)


def _block(title: str) -> list[str]:
    return ["", "━" * 70, title, "━" * 70]


def render_full_trace(
    response: Any,
    trace: TraceRecorder | None = None,
    settings: Any = None,
) -> str:
    """Exhaustive forensic trace: every captured artifact, stage by stage."""
    data = trace.stages if trace is not None else {}
    result = response.pipeline_result
    verified = response.verified_answer
    context = response.evidence_context
    warnings = [str(w) for w in verified.grounding_warnings]

    def get(stage: str, key: str, default: Any = None) -> Any:
        section = data.get(stage)
        if section is None:
            return default
        return section.data.get(key, default)

    lines = ["", "🔍 FULL PIPELINE TRACE (FORENSIC)", ""]

    # ── Query Rewriter ──
    lines += _block("QUERY REWRITER")
    lines.append(f"Original user message:\n  {response.original_query}")
    history = get("rewriter", "history") or []
    lines.append(f"History turns used: {len(history)}")
    for i, turn in enumerate(history, start=1):
        role = turn.get("role", "?") if isinstance(turn, dict) else "?"
        content = turn.get("content", "") if isinstance(turn, dict) else str(turn)
        lines.append(f"  Turn {i} [{role}]: {content}")
    lines.append(f"Empty-history fast path: {get('rewriter', 'fast_path', NOT_CAPTURED)}")
    lines.append(f"LLM call made: {get('rewriter', 'llm_called', NOT_CAPTURED)}")
    lines.append(f"Model: {get('rewriter', 'model', NOT_CAPTURED)}")
    lines.append(f"System prompt:\n{get('rewriter', 'system_prompt', NOT_CAPTURED)}")
    lines.append(f"User prompt:\n{get('rewriter', 'user_prompt', NOT_CAPTURED)}")
    lines.append(f"Raw model response:\n{get('rewriter', 'raw_response', NOT_CAPTURED)}")
    lines.append(f"Parsed retrieval query: {response.retrieval_query}")
    lines.append(f"Query changed: {'Yes' if response.original_query.strip() != response.retrieval_query.strip() else 'No'}")
    lines.append(f"Latency: {_fmt_ms(data['rewriter'].latency_ms if 'rewriter' in data else None)}")
    lines.append(f"Tokens: {_fmt_usage(get('rewriter', 'usage'))}")
    if get("rewriter", "error") is not None:
        lines.append(f"Error/fallback: {get('rewriter', 'error')}")

    # ── B4 ──
    lines += _block("B4 — HYBRID RETRIEVAL")
    lines.append(f"Retrieval query: {response.retrieval_query}")
    lines.append(f"Embedding model: {get('b4', 'embedding_model', NOT_CAPTURED)}"
                 + (f" ({get('b4', 'embedding_dimension')}d)" if get("b4", "embedding_dimension") else ""))
    lines.append(f"Embedding latency: {_fmt_ms(get('b4', 'embedding_latency_ms'))}")
    lines.append("Vector search: Weaviate native hybrid (fused call — no separate vector candidate list in this architecture)")
    lines.append("BM25: Weaviate native hybrid (fused call — no separate BM25 candidate list in this architecture)")
    lines.append(f"Hybrid alpha: {get('b4', 'alpha', NOT_CAPTURED)}")
    lines.append(f"Metadata filters: {get('b4', 'filters', NOT_CAPTURED)}")
    lines.append(f"Entity-scope filtering decisions: {get('entity_filter', 'summary', NOT_CAPTURED)}")
    if get("entity_filter", "removed_chunk_ids"):
        lines.append(f"  Removed by entity filter: {get('entity_filter', 'removed_chunk_ids')}")
    lines.append(f"Candidate pool size: {result.raw_retrieval_count}")
    for rank, cand in enumerate(result.raw_candidates, start=1):
        lines += _block(f"B4 CANDIDATE #{rank}")
        lines.append(f"chunk_id: {cand.chunk_id}")
        lines.append(f"document: {cand.document_name}")
        lines.append(f"page: {_pages(cand.page_numbers)}")
        lines.append(f"section: {cand.section_heading or '-'}")
        lines.append(f"heading_path: {' > '.join(cand.heading_path) if cand.heading_path else '-'}")
        lines.append(f"chunk_type: {cand.chunk_type}")
        lines.append(f"company: {cand.company or '-'}")
        lines.append(f"entity_scope: {cand.entity_scope}")
        lines.append(f"financial_period: {cand.financial_period or '-'}")
        lines.append("vector_score: " + NOT_CAPTURED + " (pipeline retains only the fused hybrid score)")
        lines.append("bm25_score: " + NOT_CAPTURED + " (pipeline retains only the fused hybrid score)")
        lines.append(f"hybrid_score: {cand.score:.6f}")
        lines.append(f"TEXT:\n{cand.text}")
    lines.append(f"B4 latency: {_fmt_ms(data['b4'].latency_ms if 'b4' in data else None)}")
    lines.append("B4 embedding tokens: " + NOT_CAPTURED + " (provider interface returns vectors only)")

    # ── B5 ──
    lines += _block("B5 — LLM RERANKER")
    lines.append(f"Model: {get('b5', 'model', NOT_CAPTURED)}")
    lines.append(f"Input candidates: {len(result.raw_candidates)} / Output: {len(result.candidates)}")
    lines.append(f"Retrieval query supplied: {response.retrieval_query}")
    lines.append(f"Original query supplied: {get('b5', 'original_query', NOT_CAPTURED)}")
    lines.append(f"Candidate IDs supplied: {[c.chunk_id for c in result.raw_candidates]}")
    lines.append(f"Exact reranker prompt:\n{get('b5', 'prompt', NOT_CAPTURED)}")
    lines.append(f"Raw model response:\n{get('b5', 'raw_response', NOT_CAPTURED)}")
    lines.append(f"Parsed ranked IDs: {get('b5', 'parsed_ids', NOT_CAPTURED)}")
    lines.append(f"Invalid IDs rejected: {get('b5', 'invalid_ids', NOT_CAPTURED)}")
    lines.append(f"Duplicates removed: {get('b5', 'duplicate_count', NOT_CAPTURED)}")
    lines.append(f"Fallback to B4 order: {get('b5', 'fallback', NOT_CAPTURED)}")
    b4_ids = [c.chunk_id for c in result.raw_candidates]
    for rank, cand in enumerate(result.candidates, start=1):
        b4_rank: int | str
        try:
            b4_rank = b4_ids.index(cand.chunk_id) + 1
        except ValueError:
            b4_rank = "—"
        lines.append(
            f"  B4 #{b4_rank} → B5 #{rank}: {cand.chunk_id} | pages {_pages(cand.page_numbers)} | "
            f"{(cand.section_heading or '-')[:50]} | B4 {cand.score:.4f} | B5 pseudo {cand.rerank_score:.4f}"
        )
    lines.append("B4 order:")
    lines.append("  " + "\n  ".join(f"{i}. {short_chunk_id(c)}" for i, c in enumerate(b4_ids, start=1)))
    lines.append("B5 order:")
    lines.append("  " + "\n  ".join(f"{i}. {short_chunk_id(c.chunk_id)}" for i, c in enumerate(result.candidates, start=1)))
    for cand in result.candidates:
        lines += _block(f"B5 SELECTED: {short_chunk_id(cand.chunk_id)}")
        lines.append(f"chunk_id: {cand.chunk_id}")
        lines.append(f"page: {_pages(cand.page_numbers)}")
        lines.append(f"section: {cand.section_heading or '-'}")
        lines.append(f"FULL TEXT:\n{cand.text}")
    lines.append(f"B5 latency: {_fmt_ms(data['b5'].latency_ms if 'b5' in data else None)}")
    lines.append(f"B5 tokens: {_fmt_usage(get('b5', 'usage'))}")

    # ── B6 ──
    lines += _block("B6 — EVIDENCE CONTEXT BUILDER")
    for item in context.evidence_items:
        lines += _block(f"EVIDENCE {item.evidence_id}")
        lines.append(f"Evidence ID: {item.evidence_id}")
        lines.append(f"Source chunk: {item.chunk_id}")
        lines.append(f"B5 rank: {next((i for i, c in enumerate(result.candidates, start=1) if c.chunk_id == item.chunk_id), '—')}")
        lines.append(f"Document: {item.document_name}")
        lines.append(f"Page: {_pages(item.page_numbers)}")
        lines.append(f"Section: {item.section_heading or '-'}")
        lines.append(f"Heading path: {' > '.join(item.heading_path) if item.heading_path else '-'}")
        lines.append(f"Chunk type: {item.chunk_type}")
        lines.append(f"Entity scope: {item.entity_scope}")
        lines.append(f"B4 score: {item.retrieval_score:.4f} / B5 pseudo: {item.rerank_score:.4f}")
        lines.append(f"Full evidence text:\n{item.text}")
        lines.append(f"Table metadata: {item.table_metadata if item.table_metadata else '—'}")
    lines.append("Evidence map:")
    for eid, item in context.evidence_map.items():
        lines.append(f"  {eid} → {item.chunk_id} (pages {_pages(item.page_numbers)})")
    omitted = list(context.omitted_items)
    lines.append(f"Deduplication/omissions: {len(omitted)}")
    for entry in omitted:
        lines.append(f"  {entry.chunk_id}: {entry.reason}")
    lines.append(f"Final context size: {len(context.formatted_context)} chars")
    lines.append("Context XML actually supplied to B7:")
    lines.append(context.formatted_context)

    # ── B7 ──
    lines += _block("B7 — GROUNDED GENERATOR")
    lines.append(f"Original question: {response.original_query}")
    lines.append(f"Retrieval query: {response.retrieval_query}")
    lines.append(f"Evidence IDs supplied: {', '.join(context.evidence_map) or '—'}")
    lines.append(f"Model: {get('b7', 'model', NOT_CAPTURED)}")
    lines.append(f"Temperature: {get('b7', 'temperature', NOT_CAPTURED)}")
    lines.append(f"Max tokens: {get('b7', 'max_tokens', NOT_CAPTURED)}")
    lines.append(f"System prompt:\n{get('b7', 'system_prompt', NOT_CAPTURED)}")
    lines.append(f"User prompt (exact serialized context sent to model):\n{get('b7', 'user_prompt', NOT_CAPTURED)}")
    lines.append(f"RAW MODEL OUTPUT:\n{get('b7', 'raw_response', NOT_CAPTURED)}")
    lines.append(f"PARSED answer: {verified.answer_text}")
    lines.append(f"PARSED cited_evidence_ids: {sorted({c.evidence_id for c in verified.citations}) or '—'}")
    lines.append(f"PARSED is_uncertain: {verified.is_uncertain}")
    for token in sorted({c.evidence_id for c in verified.citations}):
        for sentence in _citing_sentences(verified.raw_answer_text, token):
            lines.append(f"  Citing sentence for [{token}] (recomputed from captured answer): {sentence}")
    lines.append(f"B7 latency: {_fmt_ms(data['b7'].latency_ms if 'b7' in data else None)}")
    lines.append(f"B7 tokens: {_fmt_usage(get('b7', 'usage'))}")

    # ── B8 ──
    lines += _block("B8 — CITATION VERIFIER")
    lines.append(f"Answer text consumed:\n{verified.raw_answer_text}")
    lines.append(f"Citation tokens consumed: {sorted({c.evidence_id for c in verified.citations}) or '—'}")
    lines.append(f"Uncertainty flag consumed: {verified.is_uncertain}")
    for cit in verified.citations:
        lines += _block(f"CITATION {cit.evidence_id}")
        lines.append(f"B7 token: [{cit.evidence_id}]")
        lines.append("Evidence exists: YES")
        lines.append(f"Chunk: {cit.chunk_id}")
        lines.append(f"Document: {cit.document_name}")
        lines.append(f"Page: {_pages(cit.page_numbers)}")
        lines.append(f"Section: {cit.section_heading or '-'}")
        for sentence in _citing_sentences(verified.raw_answer_text, cit.evidence_id):
            lines.append(f"Claim span (recomputed from captured answer): {sentence}")
        item = context.evidence_map.get(cit.evidence_id)
        excerpt = (item.text[:600] + "…") if item and len(item.text) > 600 else (item.text if item else NOT_CAPTURED)
        lines.append(f"Evidence text (first 600 chars):\n{excerpt}")
        lines.append(f"Claim support: {_support_status(cit.evidence_id, warnings)}")
        lines.append("Provenance: PASS (resolved exclusively via B6 evidence map)")
        lines.append("Final verification: VERIFIED")
    if verified.unverified_citations:
        lines.append(f"Unverified tokens: {', '.join(f'[{t}]' for t in verified.unverified_citations)}")
    lines.append(f"B8 summary: {len(verified.citations)} verified / "
                 f"{len(verified.unverified_citations)} unverified / {len(warnings)} grounding warnings")
    for warning in warnings:
        lines.append(f"  Warning: {warning}")

    # ── Final output ──
    lines += _block("FINAL OUTPUT")
    lines.append(f"B7 raw answer (captured):\n{get('b7', 'raw_response', verified.raw_answer_text)}")
    lines.append(f"B8 verified answer:\n{verified.answer_text}")
    lines.append(f"Answer text modified B7 → final: "
                 f"{'No' if get('b7', 'raw_response') is None else 'see diff above (citation tokens resolved to footnotes)'}")
    lines.append("Stage latencies:")
    total: float = 0.0
    for name in ("rewriter", "b4", "b5", "b6", "b7", "b8"):
        section = data.get(name)
        latency: float | None = section.latency_ms if section else None
        lines.append(f"  {name}: {_fmt_ms(latency)}")
        total += latency or 0.0
    lines.append(f"Total (sum of captured stages): {total:.0f} ms")
    lines.append("Token usage:")
    for name in ("rewriter", "b4", "b5", "b7"):
        lines.append(f"  {name}: {_fmt_usage(get(name, 'usage'))}")
    lines.append("")
    return "\n".join(lines)
