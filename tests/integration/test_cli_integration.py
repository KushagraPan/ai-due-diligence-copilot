"""
Integration tests for the end-to-end Diligence Pipeline and CLI integration.

Mocks all external services (OpenAI, Weaviate, Hugging Face models) strictly.
Verifies the complete 6-stage online pipeline:
User -> Conversation History -> Query Rewriter -> retrieval_query ->
B4 Hybrid Retrieval -> B5 LLM Reranker -> B6 ContextBuilder ->
B7 Grounded Generation -> B8 Citation Verification -> CLI Answer.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from due_diligence.agents.citation_verifier import CitationVerifier
from due_diligence.agents.generator import GroundedAnswer, GroundedGenerator
from due_diligence.cli import (
    format_answer_view,
    format_cli_answer,
    format_sources_view,
    format_stage_trace,
    main,
    run_interactive_session,
)
from due_diligence.core.config import Settings
from due_diligence.pipeline import DiligencePipeline
from due_diligence.retrieval.context_builder import EvidenceContextBuilder
from due_diligence.retrieval.pipeline import RetrievalPipeline
from due_diligence.retrieval.query_rewriter import ConversationTurn, QueryRewriter
from due_diligence.retrieval.reranker import BaseReranker
from due_diligence.retrieval.retriever import WeaviateHybridRetriever
from due_diligence.retrieval.schemas import (
    RerankedResult,
    RetrievalResult,
)
from due_diligence.retrieval.weaviate_store import WeaviateConnectionError

# ── Test Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing dummy test settings."""
    return Settings(
        weaviate_url="https://mock-cluster.weaviate.cloud",
        weaviate_api_key="mock-weaviate-key",
        openai_api_key="mock-openai-key",
        generation_model="gpt-4o",
        llm_reranker_model="gpt-4o-mini",
        rerank_candidate_count=25,
        rerank_top_k=5,
    )


def _make_retrieval_result(
    chunk_id: str,
    score: float,
    page: int = 402,
    section: str = "Capital Adequacy",
    text: str = "CET1 ratio was 16.4% as of March 31, 2026.",
) -> RetrievalResult:
    """Helper to create a RetrievalResult candidate."""
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        company="HDFC Bank",
        document_name="hdfc_annual_report_fy26.pdf",
        document_type="annual_report",
        financial_period="FY2025-26",
        page_numbers=[page],
        section_heading=section,
        heading_path=["Financial Statements", "Notes", section],
        chunk_type="narrative",
        source_element_ids=[f"elem_{chunk_id}"],
        table_metadata=None,
    )


def _make_reranked_result(
    chunk_id: str,
    hybrid_score: float,
    rerank_score: float,
    page: int = 402,
    section: str = "Capital Adequacy",
    text: str = "CET1 ratio was 16.4% as of March 31, 2026.",
) -> RerankedResult:
    """Helper to create a RerankedResult candidate."""
    cand = _make_retrieval_result(chunk_id, hybrid_score, page, section, text)
    return RerankedResult.from_retrieval_result(cand, rerank_score=rerank_score)


@pytest.fixture
def mock_retriever() -> MagicMock:
    """Fixture providing a mock WeaviateHybridRetriever (B4)."""
    retriever = MagicMock(spec=WeaviateHybridRetriever)
    retriever.alpha = 0.60
    retriever.top_n = 25
    retriever.retrieve.return_value = [
        _make_retrieval_result("chunk_1", 0.88, 402, "Capital Adequacy", "HDFC Bank reported a CET1 ratio of 16.4%."),
        _make_retrieval_result("chunk_2", 0.76, 378, "Balance Sheet", "Total deposits grew 15.2%."),
    ]
    return retriever


@pytest.fixture
def mock_reranker() -> MagicMock:
    """Fixture providing a mock BaseReranker (B5)."""
    reranker = MagicMock(spec=BaseReranker)
    reranker.top_k = 5
    reranker.rerank.return_value = [
        _make_reranked_result("chunk_1", 0.88, 0.95, 402, "Capital Adequacy", "HDFC Bank reported a CET1 ratio of 16.4%."),
        _make_reranked_result("chunk_2", 0.76, 0.82, 378, "Balance Sheet", "Total deposits grew 15.2%."),
    ]
    return reranker


@pytest.fixture
def mock_query_rewriter() -> MagicMock:
    """Fixture providing a mock QueryRewriter."""
    rewriter = MagicMock(spec=QueryRewriter)
    rewriter.model_name = "gpt-4o"

    def _mock_rewrite(
        user_query: str, conversation_history: Any = None, **kwargs: Any
    ) -> str:
        if conversation_history:
            return "What was HDFC Bank's CET1 ratio in FY2025-26?"
        return user_query

    rewriter.rewrite.side_effect = _mock_rewrite
    return rewriter


@pytest.fixture
def mock_generator() -> MagicMock:
    """Fixture providing a mock GroundedGenerator (B7)."""
    generator = MagicMock(spec=GroundedGenerator)
    generator.model_name = "gpt-4o"
    generator.generate.return_value = GroundedAnswer(
        answer_text="HDFC Bank reported a CET1 ratio of 16.4% as of March 31, 2026 [E1].",
        cited_evidence_ids=["E1"],
        is_uncertain=False,
    )
    return generator


@pytest.fixture
def mock_pipeline(
    mock_retriever: MagicMock,
    mock_reranker: MagicMock,
    mock_query_rewriter: MagicMock,
    mock_generator: MagicMock,
    test_settings: Settings,
) -> DiligencePipeline:
    """Fixture providing a DiligencePipeline with mocked components."""
    retrieval_pipe = RetrievalPipeline(
        retriever=mock_retriever,
        reranker=mock_reranker,
        query_rewriter=mock_query_rewriter,
        settings=test_settings,
    )
    context_builder = EvidenceContextBuilder(settings=test_settings)
    citation_verifier = CitationVerifier()

    return DiligencePipeline(
        settings=test_settings,
        retrieval_pipeline=retrieval_pipe,
        context_builder=context_builder,
        generator=mock_generator,
        citation_verifier=citation_verifier,
    )


# ── Pipeline & CLI Integration Tests ────────────────────────────────────────


def test_diligence_pipeline_end_to_end(
    mock_pipeline: DiligencePipeline,
    mock_retriever: MagicMock,
    mock_reranker: MagicMock,
    mock_generator: MagicMock,
) -> None:
    """
    Test the full 6-stage diligence pipeline execution:
    1. Preserves original_query separately from retrieval_query.
    2. Uses retrieval_query for B4 and B5.
    3. Provides both queries to B6 Context Builder.
    4. Generates from original_query + B6 EvidenceContext in B7.
    5. Verifies B7 citations against B6 evidence_map in B8.
    """
    user_query = "What is the bank's CET1 ratio?"
    history = [ConversationTurn(role="user", content="Tell me about HDFC Bank.")]

    response = mock_pipeline.run(
        user_query=user_query,
        conversation_history=history,
        company="HDFC Bank",
    )

    # 1. Query preservation verification
    assert response.original_query == user_query
    assert response.retrieval_query == "What was HDFC Bank's CET1 ratio in FY2025-26?"

    # 2. B4 and B5 called with retrieval_query
    mock_retriever.retrieve.assert_called_once()
    retriever_q = (
        mock_retriever.retrieve.call_args.kwargs.get("query")
        or mock_retriever.retrieve.call_args[0][0]
    )
    assert retriever_q == response.retrieval_query

    mock_reranker.rerank.assert_called_once()
    reranker_q = (
        mock_reranker.rerank.call_args.kwargs.get("query")
        or mock_reranker.rerank.call_args[0][0]
    )
    assert reranker_q == response.retrieval_query

    # 3. B6 EvidenceContext received both queries
    assert response.evidence_context.original_query == user_query
    assert response.evidence_context.retrieval_query == response.retrieval_query
    assert len(response.evidence_context.evidence_items) == 2
    assert "E1" in response.evidence_context.evidence_map

    # 4. B7 Generator called with original_query
    mock_generator.generate.assert_called_once()
    call_kwargs = mock_generator.generate.call_args[1]
    assert call_kwargs["query"] == user_query
    assert call_kwargs["context"] == response.evidence_context

    # 5. B8 Citation Verification resolved [E1] to [1]
    assert "16.4% as of March 31, 2026 [1]" in response.answer_text
    assert len(response.verified_answer.citations) == 1
    cit = response.verified_answer.citations[0]
    assert cit.citation_number == 1
    assert cit.evidence_id == "E1"
    assert cit.page_numbers == [402]
    assert cit.section_heading == "Capital Adequacy"
    assert cit.format_citation() == "[hdfc_annual_report_fy26.pdf, Page 402, Capital Adequacy]"


def test_cli_format_answer_display(mock_pipeline: DiligencePipeline) -> None:
    """Test clean terminal formatting of DiligenceResponse."""
    response = mock_pipeline.run(user_query="What is the CET1 ratio?")
    formatted = format_cli_answer(response, show_evidence=True, show_trace=True)

    # Must contain Answer section
    assert "🤖 Copilot" in formatted
    assert "16.4% as of March 31, 2026 [1]" in formatted

    # Must contain Verified Citations section exactly once (no duplicated excerpts)
    assert "📚 Sources" in formatted
    assert "[1] HDFC Bank Annual Report FY2025-26 — Page 402 — Capital Adequacy" in formatted
    assert formatted.count("📚 Sources") == 1
    assert "Evidence for previous answer" not in formatted

    # Must contain Pipeline Trace
    assert "Trace (B4 → B8)" in formatted
    assert "Original query:   What is the CET1 ratio?" in formatted


def test_cli_normal_answer_sources_exactly_once(mock_pipeline: DiligencePipeline) -> None:
    """Normal answer view shows the concise Sources list exactly once."""
    response = mock_pipeline.run(user_query="What is the CET1 ratio?")
    formatted = format_cli_answer(response)
    assert formatted.count("📚 Sources") == 1
    assert "[1] HDFC Bank Annual Report FY2025-26 — Page 402 — Capital Adequacy" in formatted


def test_cli_sources_command_shows_excerpts_not_citation_list(
    mock_pipeline: DiligencePipeline,
) -> None:
    """/sources shows detailed excerpts without repeating the concise citation list."""
    response = mock_pipeline.run(user_query="What is the CET1 ratio?")
    sources = format_sources_view(response)
    assert "📚 Evidence for previous answer" in sources
    assert '"HDFC Bank reported a CET1 ratio of 16.4%."' in sources
    assert "[1] HDFC Bank Annual Report FY2025-26 — Page 402 — Capital Adequacy" not in sources


def test_cli_format_answer_no_evidence_flag(mock_pipeline: DiligencePipeline) -> None:
    """Test --no-evidence flag hides the concise Sources list (display only)."""
    response = mock_pipeline.run(user_query="What is the CET1 ratio?")
    formatted = format_cli_answer(response, show_evidence=False, show_trace=False)

    assert "🤖 Copilot" in formatted
    assert "16.4% as of March 31, 2026 [1]" in formatted
    assert "📚 Sources" not in formatted
    assert "Trace (B4 → B8)" not in formatted


def test_cli_one_shot_execution(
    mock_pipeline: DiligencePipeline,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test executing a one-shot query via CLI --query flag."""
    monkeypatch.setattr("due_diligence.cli.DiligencePipeline", lambda **kwargs: mock_pipeline)
    monkeypatch.setattr("due_diligence.cli.get_settings", lambda: mock_pipeline.settings)

    exit_code = main(["--query", "What is the CET1 ratio?", "--company", "HDFC Bank"])
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "🤖 Copilot" in captured
    assert "📚 Sources" in captured
    assert "[1] HDFC Bank Annual Report FY2025-26 — Page 402 — Capital Adequacy" in captured


def test_cli_interactive_session_multi_turn(
    mock_pipeline: DiligencePipeline,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Test multi-turn interactive session:
    1. Reuses same pipeline instance across turns.
    2. Maintains bounded conversation history across turns.
    3. Terminates cleanly on 'exit'.
    """
    inputs = iter([
        "What is the CET1 ratio?",
        "What about total deposits?",
        "exit",
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    run_interactive_session(
        pipeline=mock_pipeline,
        company="HDFC Bank",
        max_history_turns=4,
    )

    captured = capsys.readouterr().out
    assert "AI DUE DILIGENCE CO-PILOT" in captured
    assert "👋 Session ended." in captured
    assert "📚 Sources" in captured


def test_cli_unverified_citation_warning(
    mock_pipeline: DiligencePipeline,
    mock_generator: MagicMock,
) -> None:
    """Test that hallucinated citation tokens produce explicit warnings in CLI display."""
    # Generator emits an unverified token [E9]
    mock_generator.generate.return_value = GroundedAnswer(
        answer_text="CET1 was 16.4% [E1]. Burn rate was 50M [E9].",
        cited_evidence_ids=["E1", "E9"],
        is_uncertain=False,
    )

    response = mock_pipeline.run(user_query="What is the burn rate?")
    formatted = format_cli_answer(response)

    assert "[Unverified: E9]" in formatted
    assert "Citation verification warning" in formatted
    assert "[E9]" in formatted
    assert "Use /trace for details." in formatted


def test_cli_uncertain_refusal_display(
    mock_pipeline: DiligencePipeline,
    mock_generator: MagicMock,
) -> None:
    """Test CLI formatting when information is not disclosed in the document."""
    mock_generator.generate.return_value = GroundedAnswer(
        answer_text="Not disclosed in the ingested document. The document contains no data on customer churn.",
        cited_evidence_ids=[],
        is_uncertain=True,
    )

    response = mock_pipeline.run(user_query="What is the customer churn?")
    formatted = format_cli_answer(response)

    assert "Not disclosed in the ingested document." in formatted
    assert "📚 Sources" in formatted
    assert "None." in formatted


def _run_session(mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], inputs: list[str], **kwargs: object) -> str:
    """Drive an interactive session with canned inputs, returning stdout."""
    stream = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(stream))
    run_interactive_session(pipeline=mock_pipeline, company="HDFC Bank", **kwargs)  # type: ignore[arg-type]
    return capsys.readouterr().out


def test_cli_startup_banner(
    mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Startup shows product banner, document line, and command hints."""
    out = _run_session(mock_pipeline, monkeypatch, capsys, ["exit"])
    assert "🔎 AI DUE DILIGENCE CO-PILOT" in out
    assert "📄 Document:" in out
    assert "/help" in out
    assert "👋 Session ended." in out


def test_cli_normal_view_answer_and_sources(
    mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Default view shows answer + sources without the developer trace."""
    out = _run_session(mock_pipeline, monkeypatch, capsys, ["What is the CET1 ratio?", "exit"])
    assert "🤖 Copilot" in out
    assert "16.4%" in out
    assert "📚 Sources" in out
    assert "Trace (B4 → B8)" not in out


def test_cli_answer_view_and_sources_view_unit(mock_pipeline: DiligencePipeline) -> None:
    """Unit-level: answer view and sources view render citations and excerpts."""
    response = mock_pipeline.run(user_query="What is the CET1 ratio?")
    answer = format_answer_view(response)
    assert "🤖 Copilot" in answer
    assert "[1] HDFC Bank Annual Report FY2025-26 — Page 402 — Capital Adequacy" in answer
    sources = format_sources_view(response)
    assert "Evidence for previous answer" in sources
    assert "Page 402 — Capital Adequacy" in sources


def test_cli_multiple_questions_one_session(
    mock_pipeline: DiligencePipeline, mock_retriever: MagicMock,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Two questions in one session each produce an answer."""
    out = _run_session(
        mock_pipeline, monkeypatch, capsys,
        ["What is the CET1 ratio?", "What about total deposits?", "exit"],
    )
    assert out.count("🤖 Copilot") == 2
    assert mock_retriever.retrieve.call_count == 2


def test_cli_followup_uses_history(
    mock_pipeline: DiligencePipeline, mock_query_rewriter: MagicMock,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Follow-up turns pass accumulated conversation history to the pipeline."""
    _run_session(
        mock_pipeline, monkeypatch, capsys,
        ["What is the CET1 ratio?", "What about total deposits?", "exit"],
    )
    second_call_history = mock_query_rewriter.rewrite.call_args_list[1].kwargs["conversation_history"]
    assert len(second_call_history) >= 2


def test_cli_help_command(
    mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test /help lists commands."""
    out = _run_session(mock_pipeline, monkeypatch, capsys, ["/help", "exit"])
    assert "Available commands:" in out
    assert "/sources" in out and "/trace" in out and "/clear" in out


def test_cli_sources_command(
    mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test /sources shows evidence for the last answer."""
    out = _run_session(mock_pipeline, monkeypatch, capsys, ["What is the CET1 ratio?", "/sources", "exit"])
    assert "Evidence for previous answer" in out
    assert "Page 402 — Capital Adequacy" in out


def test_cli_sources_before_any_answer(
    mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test /sources with no prior answer shows a hint, not an error."""
    out = _run_session(mock_pipeline, monkeypatch, capsys, ["/sources", "exit"])
    assert "ask a question first" in out


def test_cli_trace_toggle_and_stage_block(
    mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """/trace shows the compact trace; /trace on enables auto-trace."""
    out = _run_session(
        mock_pipeline, monkeypatch, capsys,
        ["What is the CET1 ratio?", "/trace", "/trace on", "What about deposits?", "exit"],
    )
    assert "🔍 PIPELINE TRACE" in out
    assert "B4 — HYBRID RETRIEVAL" in out
    assert "🔧 Trace mode: ON" in out
    # Auto-trace renders compact trace after the second answer
    assert out.count("🔍 PIPELINE TRACE") >= 2


def test_cli_trace_full_command(
    mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """/trace full shows the forensic trace."""
    out = _run_session(
        mock_pipeline, monkeypatch, capsys, ["What is the CET1 ratio?", "/trace full", "exit"]
    )
    assert "FULL PIPELINE TRACE" in out
    assert "B4 CANDIDATE #1" in out
    assert "CITATION" in out


def test_cli_trace_all_command(
    mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """/trace all shows the forensic trace identically to /trace full."""
    out = _run_session(
        mock_pipeline, monkeypatch, capsys, ["What is the CET1 ratio?", "/trace all", "exit"]
    )
    assert "FULL PIPELINE TRACE" in out
    assert "B4 CANDIDATE #1" in out
    assert "CITATION" in out


def test_cli_trace_no_previous_query(
    mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """/trace and /trace full without a prior query show a hint, not an error."""
    out = _run_session(mock_pipeline, monkeypatch, capsys, ["/trace", "/trace full", "exit"])
    assert out.count("No trace available. Run a query first.") == 2
    assert "Traceback" not in out


def test_cli_trace_render_failure_stays_in_session(
    mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rendering failure shows a friendly message and keeps the session alive."""
    monkeypatch.setattr(
        "due_diligence.cli.render_full_trace", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    out = _run_session(
        mock_pipeline, monkeypatch, capsys, ["What is the CET1 ratio?", "/trace full", "exit"]
    )
    assert "Could not render trace." in out
    assert "👋 Session ended." in out


def test_cli_trace_off_command(
    mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """/trace off disables auto-trace."""
    out = _run_session(
        mock_pipeline, monkeypatch, capsys,
        ["/trace on", "/trace off", "What is the CET1 ratio?", "exit"],
    )
    assert "🔧 Trace mode: OFF" in out
    assert "🔍 PIPELINE TRACE" not in out


def test_cli_stage_trace_unit(mock_pipeline: DiligencePipeline) -> None:
    """Unit-level: stage trace exposes queries, counts, pages, and citations."""
    response = mock_pipeline.run(user_query="What is the CET1 ratio?")
    trace = format_stage_trace(response, elapsed_ms=1234.0)
    assert "Original query:" in trace
    assert "B4 candidates:" in trace and "B5 selected:" in trace
    assert "B6 evidence:" in trace and "B7 cited:" in trace and "B8 verified:" in trace
    assert "1234 ms" in trace


def test_cli_clear_command(
    mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test /clear resets conversation state only."""
    out = _run_session(mock_pipeline, monkeypatch, capsys, ["/clear", "exit"])
    assert "Conversation cleared." in out


def test_cli_quit_alias(
    mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test quit exits cleanly like /exit."""
    out = _run_session(mock_pipeline, monkeypatch, capsys, ["quit"])
    assert "👋 Session ended." in out


def test_cli_empty_input_is_noop(
    mock_pipeline: DiligencePipeline, mock_retriever: MagicMock,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Empty input triggers no pipeline call and no error."""
    calls_before = mock_retriever.retrieve.call_count
    out = _run_session(mock_pipeline, monkeypatch, capsys, ["", "   ", "exit"])
    assert mock_retriever.retrieve.call_count == calls_before
    assert "Traceback" not in out
    assert "👋 Session ended." in out


def test_cli_unknown_slash_command(
    mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unknown /commands show a hint instead of hitting the pipeline."""
    out = _run_session(mock_pipeline, monkeypatch, capsys, ["/frobnicate", "exit"])
    assert "Unknown command" in out


def test_cli_no_evidence_refusal_view(
    mock_pipeline: DiligencePipeline, mock_generator: MagicMock,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Refusal answers display cleanly with an empty source list."""
    mock_generator.generate.return_value = GroundedAnswer(
        answer_text="Not disclosed in the ingested document.",
        cited_evidence_ids=[],
        is_uncertain=True,
    )
    out = _run_session(mock_pipeline, monkeypatch, capsys, ["What is customer churn?", "exit"])
    assert "Not disclosed in the ingested document." in out
    assert "📚 Sources" in out
    assert "None." in out


def test_cli_service_failure_friendly(
    mock_pipeline: DiligencePipeline, mock_retriever: MagicMock,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Weaviate outages show a friendly message without a stack trace."""
    mock_retriever.retrieve.side_effect = WeaviateConnectionError("boom")
    out = _run_session(mock_pipeline, monkeypatch, capsys, ["What is the CET1 ratio?", "exit"])
    assert "Unable to reach the retrieval service." in out
    assert "Traceback" not in out


def test_cli_positional_one_shot(
    mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Positional question arg works like --query for one-shot mode."""
    monkeypatch.setattr("due_diligence.cli.DiligencePipeline", lambda **kwargs: mock_pipeline)
    monkeypatch.setattr("due_diligence.cli.get_settings", lambda: mock_pipeline.settings)
    from due_diligence.cli import main as cli_main

    assert cli_main(["What is the CET1 ratio?"]) == 0
    out = capsys.readouterr().out
    assert "🤖 Copilot" in out
    assert "Trace (B4 → B8)" not in out


def test_cli_one_shot_trace_flag(
    mock_pipeline: DiligencePipeline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--trace adds the stage block to one-shot output."""
    monkeypatch.setattr("due_diligence.cli.DiligencePipeline", lambda **kwargs: mock_pipeline)
    monkeypatch.setattr("due_diligence.cli.get_settings", lambda: mock_pipeline.settings)
    from due_diligence.cli import main as cli_main

    assert cli_main(["--trace", "--query", "What is the CET1 ratio?"]) == 0
    assert "🔍 PIPELINE TRACE" in capsys.readouterr().out


def test_cli_normal_query_suppresses_internal_trace_and_diagnostic_logs(
    mock_pipeline: DiligencePipeline,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Normal query answering suppresses internal trace/diagnostic logging output."""
    import logging

    orig_run = mock_pipeline.run

    def _pipeline_run_with_logging(*args: Any, **kwargs: Any) -> Any:
        # Simulate internal diagnostic logs emitted during pipeline stages
        log = logging.getLogger("due_diligence.agents.citation_verifier")
        log.warning("Claim support: Headline number '16.4%' not found in chunk text")
        log.warning("Numeric grounding: Metric 'CET1 ratio' claimed 16.4%")
        log.warning("Citation token 'E9' in answer could not be resolved against active evidence map ['E1']")
        return orig_run(*args, **kwargs)

    monkeypatch.setattr(mock_pipeline, "run", _pipeline_run_with_logging)

    stream = iter(["What is the CET1 ratio?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(stream))
    run_interactive_session(pipeline=mock_pipeline, company="HDFC Bank")
    captured = capsys.readouterr()

    # Out contains normal answer and sources
    assert "🤖 Copilot" in captured.out
    assert "📚 Sources" in captured.out

    # Internal diagnostics must NOT appear in stdout or stderr
    assert "Claim support:" not in captured.out
    assert "Claim support:" not in captured.err
    assert "Numeric grounding:" not in captured.out
    assert "Numeric grounding:" not in captured.err
    assert "could not be resolved against active evidence map" not in captured.out
    assert "could not be resolved against active evidence map" not in captured.err
