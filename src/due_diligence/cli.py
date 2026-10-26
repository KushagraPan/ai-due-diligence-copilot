"""
Interactive CLI product for the AI Due Diligence Co-pilot.

1. Purpose:
   Terminal product layer over the frozen production RAG pipeline:
   normal users get Question -> Answer -> Sources; learners/developers
   get on-demand /sources and /trace views of the B4 -> B8 pipeline.

2. Role in Overall RAG Pipeline:
   Presentation only. Calls ``DiligencePipeline.run`` (Phases B4-B8) and
   formats verified output. Never reimplements retrieval, reranking,
   generation, or citation verification.

3. Expected Inputs:
   - CLI flags: ``--query`` / positional question, ``--company``,
     ``--period``, ``--top-n``, ``--top-k``, ``--rerank-candidates``,
     ``--no-evidence``, ``--trace``.
   - Interactive commands: /help /sources /trace /clear /exit.

4. Outputs:
   - Grounded answer with [1], [2] footnotes, friendly source list,
     optional evidence excerpts and stage diagnostics.

5. Dependencies:
   - ``due_diligence.pipeline.DiligencePipeline``
   - ``due_diligence.retrieval.query_rewriter.ConversationTurn``

6. Non-Responsibilities:
   - Does NOT implement retrieval, reranking, generation, or verification.
   - Does NOT modify indexes, embeddings, documents, or model parameters.
"""

import argparse
import logging
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import wraps
from typing import Any

from due_diligence.core.config import get_settings
from due_diligence.pipeline import DiligencePipeline, DiligenceResponse
from due_diligence.retrieval.pipeline import RetrievalPipeline
from due_diligence.retrieval.query_rewriter import ConversationTurn
from due_diligence.retrieval.schemas import PipelineQueryResult
from due_diligence.retrieval.weaviate_store import WeaviateConnectionError
from due_diligence.trace import TraceRecorder, render_compact_trace, render_full_trace

logger = logging.getLogger(__name__)

# Emoji markers with plain-ASCII fallbacks for limited terminal encodings.
_EMOJI_FALLBACK = {
    "🔎": "[search]",
    "🤖": "[answer]",
    "📚": "[sources]",
    "📄": "[doc]",
    "⏳": "[...]",
    "⚠️": "[warning]",
    "ℹ️": "[info]",
    "❌": "[error]",
    "✓": "[ok]",
    "👋": "[bye]",
    "🔧": "[trace]",
}


def _configure_console() -> None:
    """Allow UTF-8 output where the terminal supports it (Windows-safe)."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError, AttributeError):
        logger.debug("Could not reconfigure stdout encoding.", exc_info=True)


_QUIET_LOGGERS = ("due_diligence", "httpx", "httpcore", "openai", "weaviate")


@contextmanager
def _quiet_cli_logging() -> Iterator[None]:
    """
    Mute pipeline and SDK logs for the duration of a CLI run.

    Logger levels are restored on exit so calling ``main`` from tests or another
    host process does not permanently alter global logging behaviour.
    """
    previous = {name: logging.getLogger(name).level for name in _QUIET_LOGGERS}
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        yield
    finally:
        for name, level in previous.items():
            logging.getLogger(name).setLevel(level)


def _quiet_cli(func: Callable[..., Any]) -> Callable[..., Any]:
    """Run a CLI entrypoint with project and SDK loggers muted."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _quiet_cli_logging():
            return func(*args, **kwargs)

    return wrapper


def say(text: str = "") -> None:
    """Print with graceful emoji fallback on limited terminal encodings."""
    try:
        print(text)
    except UnicodeEncodeError:
        for emoji, fallback in _EMOJI_FALLBACK.items():
            text = text.replace(emoji, fallback)
        print(text.encode("ascii", errors="replace").decode("ascii"))


def _friendly_doc_label(response: DiligenceResponse) -> str:
    """Short human document label derived from evidence metadata."""
    for item in response.evidence_context.evidence_items:
        if item.company:
            period = f" {item.financial_period}" if item.financial_period else ""
            return f"{item.company} Annual Report{period}"
    return "Annual Report"


def _format_citation_line(response: DiligenceResponse, number: int) -> str:
    """Render one verified citation as '[n] Doc — Page X — Section'."""
    for cit in response.verified_answer.citations:
        if cit.citation_number != number:
            continue
        item = response.evidence_context.evidence_map.get(cit.evidence_id)
        doc = _friendly_doc_label(response)
        pages = ", ".join(str(p) for p in cit.page_numbers) if cit.page_numbers else "Unknown"
        section = ""
        if item and item.section_heading:
            section = f" — {item.section_heading}"
        elif cit.section_heading:
            section = f" — {cit.section_heading}"
        return f"[{number}] {doc} — Page {pages}{section}"
    return f"[{number}] (citation unavailable)"


def format_answer_view(response: DiligenceResponse, show_sources: bool = True) -> str:
    """Normal-user view: grounded answer plus concise citation list (shown once)."""
    lines = ["", "🤖 Copilot", "", response.verified_answer.answer_text, ""]
    if not show_sources:
        lines.append("")
        return "\n".join(lines)
    lines.extend(["📚 Sources", ""])
    if response.verified_answer.citations:
        numbers = sorted(c.citation_number for c in response.verified_answer.citations)
        lines.extend(_format_citation_line(response, n) for n in numbers)
    else:
        lines.append("None.")
    if response.verified_answer.unverified_citations:
        tokens = ", ".join(f"[{t}]" for t in response.verified_answer.unverified_citations)
        lines.append(f"\n⚠️ Citation verification warning: {tokens}\nUse /trace for details.")
    lines.append("")
    return "\n".join(lines)


def format_sources_view(response: DiligenceResponse) -> str:
    """Detailed evidence excerpts behind the last answer's verified citations."""
    lines = ["", "📚 Evidence for previous answer", ""]
    if not response.verified_answer.citations:
        return "\n".join(lines + ["None.", ""])
    for cit in response.verified_answer.citations:
        item = response.evidence_context.evidence_map.get(cit.evidence_id)
        pages = ", ".join(str(p) for p in cit.page_numbers) if cit.page_numbers else "Unknown"
        section = (item.section_heading if item and item.section_heading else None) or "General"
        lines.append(f"[{cit.citation_number}] Page {pages} — {section}")
        if item:
            snippet = item.text.strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."
            lines.append(f'    "{snippet}"')
        lines.append("")
    return "\n".join(lines)


def format_stage_trace(response: DiligenceResponse, elapsed_ms: float | None = None) -> str:
    """Developer view: per-stage B4 -> B5 -> B6 -> B7 -> B8 diagnostics."""
    result = response.pipeline_result
    b4_pages = sorted({p for c in result.raw_candidates for p in c.page_numbers})
    b5_pages = sorted({p for c in result.candidates for p in c.page_numbers})
    verified = response.verified_answer
    b6_ids = ", ".join(response.evidence_context.evidence_map) or "—"
    b7_cited = ", ".join(sorted({c.evidence_id for c in verified.citations})) or "—"
    if verified.is_uncertain:
        b7_cited += " (uncertain)"
    lines = [
        "",
        "🔧 Trace (B4 → B8)",
        f"Original query:   {response.original_query}",
        f"Retrieval query:  {response.retrieval_query}",
        f"B4 candidates:    {result.raw_retrieval_count} (pages: {', '.join(map(str, b4_pages[:12])) or '—'})",
        f"B5 selected:      {len(result.candidates)} (pages: {', '.join(map(str, b5_pages)) or '—'})",
        f"B6 evidence:      {len(response.evidence_context.evidence_items)} ({b6_ids})",
        f"B7 cited:         {b7_cited}",
        f"B8 verified:      {len(verified.citations)} ok / {len(verified.unverified_citations)} unverified / {len(verified.grounding_warnings)} grounding warnings",
    ]
    if elapsed_ms is not None:
        lines.append(f"Latency (total):  {elapsed_ms:.0f} ms")
    lines.append("")
    return "\n".join(lines)


def format_cli_answer(
    response: DiligenceResponse,
    show_evidence: bool = True,
    show_trace: bool = False,
) -> str:
    """
    Format the complete diligence response for terminal display.

    Normal view is answer + concise sources (shown exactly once). Pass
    ``show_trace=True`` for the developer stage-trace block. Detailed
    excerpts are available separately via ``format_sources_view``.
    """
    parts = [format_answer_view(response, show_sources=show_evidence)]
    if show_trace:
        parts.append(format_stage_trace(response))
    return "\n".join(parts)


def format_evidence_display(result: PipelineQueryResult | DiligenceResponse) -> str:
    """
    Format pipeline diagnostics and top evidence passages for clean CLI output.

    Provided for backward compatibility with existing unit tests and callers.
    """
    if isinstance(result, DiligenceResponse):
        return format_cli_answer(result, show_evidence=True, show_trace=True)

    lines: list[str] = [
        "",
        "=" * 80,
        "DILIGENCE RETRIEVAL TRACE",
        "=" * 80,
        f"Original query:   {result.original_query}",
        f"Retrieval query:  {result.retrieval_query}",
        f"B4 Candidates:    {result.raw_retrieval_count} chunks retrieved",
        f"B5 Evidence:      {len(result.candidates)} passages reranked",
        "-" * 80,
    ]

    if not result.candidates:
        lines.append("No matching candidates found.")
        lines.append("=" * 80)
        return "\n".join(lines)

    lines.append("TOP EVIDENCE PASSAGES:")
    for rank, cand in enumerate(result.candidates, start=1):
        pages_str = ", ".join(str(p) for p in cand.page_numbers)
        section = cand.section_heading or "General"
        lines.append(
            f"\n[{rank}] Page(s): {pages_str} | Section: {section} | "
            f"Score: {cand.rerank_score:.4f} (Hybrid: {cand.score:.4f}) | Type: {cand.chunk_type}"
        )
        text_snippet = cand.text.strip().replace("\n", " ")
        if len(text_snippet) > 200:
            text_snippet = text_snippet[:197] + "..."
        lines.append(f"    Excerpt: {text_snippet}")

    lines.append("=" * 80)
    lines.append("")
    return "\n".join(lines)


def _friendly_error(exc: Exception) -> str:
    """Map common failures to user-facing messages without stack traces."""
    if isinstance(exc, WeaviateConnectionError):
        return "⚠️ Unable to reach the retrieval service.\nDetails: Weaviate connection failed."
    message = str(exc).strip()
    lowered = message.lower()
    if "openai" in lowered and ("api key" in lowered or "unauthorized" in lowered or "401" in lowered):
        return "⚠️ Language-model service rejected the request.\nDetails: check OPENAI_API_KEY."
    if "weaviate" in lowered:
        return f"⚠️ Unable to reach the retrieval service.\nDetails: {message or type(exc).__name__}."
    return f"❌ Something went wrong.\nDetails: {message or type(exc).__name__}."


def _print_help() -> None:
    say(
        "\nAvailable commands:\n"
        "  /help        Show commands\n"
        "  /sources     Show sources for the last answer\n"
        "  /trace       Show compact pipeline trace for the last answer\n"
        "  /trace full  Show exhaustive forensic trace for the last answer\n"
        "  /trace all   Show exhaustive forensic trace (alias for /trace full)\n"
        "  /trace on    Auto-show compact trace after each answer\n"
        "  /trace off   Stop auto-showing traces\n"
        "  /clear       Clear conversation history\n"
        "  /exit        Exit\n"
    )


def _show_trace(
    last_response: DiligenceResponse | None,
    last_trace: TraceRecorder | None,
    full: bool,
) -> None:
    """Render compact or full trace for the most recent query, if any."""
    if last_response is None:
        say("ℹ️ No trace available. Run a query first.")
        return
    settings = get_settings()
    try:
        if full:
            say(render_full_trace(last_response, last_trace, settings))
        else:
            say(render_compact_trace(last_response, last_trace, settings))
    except Exception as exc:
        logger.exception("Trace rendering failed")
        say(f"⚠️ Could not render trace. Details: {exc}")


# ── Interactive Session ──────────────────────────────────────────────────


@_quiet_cli
def run_interactive_session(
    pipeline: DiligencePipeline,
    company: str | None = None,
    financial_period: str | None = None,
    top_n: int | None = None,
    top_k: int | None = None,
    rerank_candidate_count: int | None = None,
    max_history_turns: int = 4,
    show_evidence: bool = True,
    show_trace: bool = False,
    trace_mode: bool = False,
) -> None:
    """
    Run an interactive multi-turn diligence dialogue in the terminal.

    Reuses the injected DiligencePipeline instance across turns to keep models
    and connections alive. Maintains a bounded window of recent turns, passed
    unchanged to the pipeline's own QueryRewriter.
    """
    history: list[ConversationTurn] = []
    last_response: DiligenceResponse | None = None
    last_trace: TraceRecorder | None = None

    period = f" {financial_period}" if financial_period else " FY2025-26"
    say("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    say("        🔎 AI DUE DILIGENCE CO-PILOT")
    say("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    say(f"📄 Document: {company or 'HDFC Bank'}{period} Annual Report")
    say("🤖 Grounded RAG • Weaviate • GPT")
    say("")
    say("Type your question below.")
    say("Commands: /help  /sources  /trace  /trace full  /trace all  /clear  /exit")
    say("AI DUE DILIGENCE CO-PILOT — INTERACTIVE DILIGENCE CLI")
    say("")

    while True:
        try:
            user_input = input("You:\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            say("\n👋 Session ended.")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in ("exit", "quit", "q", "/exit"):
            say("👋 Session ended.")
            break
        if lowered in ("/help", "help"):
            _print_help()
            continue
        if lowered == "/clear":
            history = []
            say("✓ Conversation cleared.")
            continue
        if lowered == "/trace" or lowered.startswith("/trace "):
            parts = lowered.split()
            if len(parts) == 1:
                _show_trace(last_response, last_trace, full=False)
            elif parts[1] in ("full", "all"):
                _show_trace(last_response, last_trace, full=True)
            elif parts[1] == "on":
                trace_mode = True
                say("🔧 Trace mode: ON")
            elif parts[1] == "off":
                trace_mode = False
                say("🔧 Trace mode: OFF")
            else:
                say("ℹ️ Usage: /trace  •  /trace full  •  /trace all  •  /trace on|off")
            continue
        if lowered == "/sources":
            if last_response is None:
                say("ℹ️ No answer yet — ask a question first.")
            else:
                say(format_sources_view(last_response))
            continue
        if lowered.startswith("/"):
            say(f"ℹ️ Unknown command '{user_input}'. Type /help for commands.")
            continue

        try:
            say("⏳ Analyzing...")
            recorder = TraceRecorder()
            response = pipeline.run(
                user_query=user_input,
                conversation_history=history,
                company=company,
                financial_period=financial_period,
                top_n=top_n,
                top_k=top_k,
                rerank_candidate_count=rerank_candidate_count,
                trace=recorder,
            )
            last_response = response
            last_trace = recorder

            # Normal-user view: answer + concise sources (shown exactly once).
            # Detailed excerpts are on-demand via /sources only.
            say(format_answer_view(response, show_sources=show_evidence))
            if show_trace or trace_mode:
                say(render_compact_trace(response, recorder, get_settings()))

            # Maintain bounded history: preserve original user query and assistant answer
            history.append(ConversationTurn(role="user", content=response.original_query))
            history.append(ConversationTurn(role="assistant", content=response.answer_text))

            if len(history) > max_history_turns:
                history = history[-max_history_turns:]

        except Exception as exc:
            logger.exception("Error executing diligence pipeline for query")
            say(_friendly_error(exc) + "\n")


# ── Argument Parser & Main Entrypoint ───────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="AI Due Diligence Co-pilot — Terminal RAG Pipeline",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Optional question to answer directly without entering interactive loop.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional single query to execute without entering interactive loop.",
    )
    parser.add_argument(
        "--company",
        type=str,
        default="HDFC Bank",
        help="Target company name for metadata filtering (default: 'HDFC Bank').",
    )
    parser.add_argument(
        "--period",
        type=str,
        default=None,
        help="Target financial period filter (e.g. 'FY2025-26').",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of candidates to retrieve in B4 hybrid search (default: from settings).",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=None,
        dest="rerank_candidates",
        help="Number of B4 candidates to forward to B5 reranking (default: from settings).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of candidates to retain after B5 reranking (default: from settings).",
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="Hide the concise 📚 Sources list (display only; pipeline stays evidence-grounded).",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Show B4 → B8 pipeline diagnostics after each answer (developer view).",
    )
    return parser


@_quiet_cli
def main(argv: Sequence[str] | None = None) -> int:
    """CLI application entrypoint."""
    _configure_console()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    settings = get_settings()

    try:
        # Reuse single pipeline instance across executions
        retrieval_pipeline = RetrievalPipeline(settings=settings)
        pipeline = DiligencePipeline(
            settings=settings,
            retrieval_pipeline=retrieval_pipeline,
        )
    except Exception as exc:
        logger.exception("CLI startup failed")
        say(_friendly_error(exc))
        return 1

    show_evidence = not args.no_evidence
    one_shot_query = args.question or args.query

    if one_shot_query:
        # One-shot query mode
        try:
            recorder = TraceRecorder()
            response = pipeline.run(
                user_query=one_shot_query,
                conversation_history=None,
                company=args.company,
                financial_period=args.period,
                top_n=args.top_n,
                top_k=args.top_k,
                rerank_candidate_count=args.rerank_candidates,
                trace=recorder,
            )
        except Exception as exc:
            logger.exception("CLI one-shot query failed")
            say(_friendly_error(exc))
            return 1
        say(format_answer_view(response, show_sources=show_evidence))
        if args.trace:
            say(render_compact_trace(response, recorder, settings))
        return 0

    # Interactive session mode
    run_interactive_session(
        pipeline=pipeline,
        company=args.company,
        financial_period=args.period,
        top_n=args.top_n,
        top_k=args.top_k,
        rerank_candidate_count=args.rerank_candidates,
        show_evidence=show_evidence,
        show_trace=args.trace,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
