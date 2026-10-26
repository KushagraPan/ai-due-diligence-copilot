"""
Unit tests for pipeline execution tracing (trace.py) and recorder capture.

Verifies:
1. TraceRecorder notes timings and stage facts.
2. Compact and full renderers on real DiligenceResponse objects.
3. Renders with no recorder (graceful "Not captured" paths).
4. No-citation, one-citation, multi-citation, invalid-citation renders.
5. Uncertain answers and missing optional metadata.
6. QueryRewriter fast path vs actual LLM rewrite capture.
7. LLMReranker prompt/raw/parsed/usage capture.
8. GroundedGenerator prompt/raw capture.
9. Tracing never alters pipeline results (answer, chunks, IDs, verification).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from due_diligence.agents.citation_verifier import CitationVerifier
from due_diligence.agents.generator import GroundedAnswer, GroundedGenerator
from due_diligence.pipeline import DiligenceResponse
from due_diligence.retrieval.context_builder import EvidenceContextBuilder
from due_diligence.retrieval.query_rewriter import QueryRewriter
from due_diligence.retrieval.reranker import LLMReranker
from due_diligence.retrieval.schemas import (
    EvidenceContext,
    EvidenceItem,
    PipelineQueryResult,
    RerankedResult,
    RetrievalResult,
)
from due_diligence.trace import (
    NOT_CAPTURED,
    TokenUsage,
    TraceRecorder,
    render_compact_trace,
    render_full_trace,
    short_chunk_id,
)


def _make_candidate(chunk_id: str, text: str = "Disclosure text.", score: float = 0.7) -> RetrievalResult:
    """Build a valid RetrievalResult test candidate."""
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        company="HDFC Bank",
        document_name="hdfc_fy26.pdf",
        document_type="pdf",
        financial_period="FY2025-26",
        page_numbers=[10],
        section_heading="Notes",
        heading_path=["Statements"],
        chunk_type="narrative",
        source_element_ids=[f"elem_{chunk_id}"],
        table_metadata=None,
    )


def _make_response(
    answer_text: str = "CET1 was 16.4% [E1].",
    cited_ids: list[str] | None = None,
    uncertain: bool = False,
    evidence_text: str = "CET1 was 16.4% in FY2025-26.",
) -> DiligenceResponse:
    """Build a real DiligenceResponse through B6 context construction + B8 verification."""
    cand = _make_candidate("doc_c00001_abc", text=evidence_text)
    reranked = RerankedResult.from_retrieval_result(cand, rerank_score=0.95)
    context = EvidenceContextBuilder().build_context(
        [reranked], original_query="What was CET1?", retrieval_query="What was CET1?"
    )
    grounded = GroundedAnswer(
        answer_text=answer_text,
        cited_evidence_ids=cited_ids if cited_ids is not None else ["E1"],
        is_uncertain=uncertain,
    )
    verified = CitationVerifier().verify(grounded, context)
    return DiligenceResponse(
        verified_answer=verified,
        evidence_context=context,
        pipeline_result=PipelineQueryResult(
            original_query="What was CET1?",
            retrieval_query="What was CET1?",
            candidates=[reranked],
            raw_retrieval_count=1,
            raw_candidates=[cand],
        ),
    )


def _mock_chat_client(content: str, usage: SimpleNamespace | None = None) -> MagicMock:
    """Mock OpenAI client returning fixed content with optional usage."""
    client = MagicMock()
    completion = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    completion.choices = [choice]
    completion.usage = usage or SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    client.chat.completions.create.return_value = completion
    return client


class TestTraceRecorder:
    """Recorder stores stage facts and timings without affecting callers."""

    def test_note_and_stage_creation(self) -> None:
        rec = TraceRecorder()
        assert rec.stages == {}
        rec.note("b4", "alpha", 0.5)
        assert rec.stages["b4"].data["alpha"] == 0.5

    def test_time_context_manager_records_latency(self) -> None:
        rec = TraceRecorder()
        with rec.time("b5"):
            pass
        assert rec.stages["b5"].latency_ms is not None
        assert rec.stages["b5"].latency_ms >= 0.0

    def test_token_usage_from_openai_response(self) -> None:
        usage = TokenUsage.from_openai_response(
            SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10))
        )
        assert usage is not None
        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (7, 3, 10)

    def test_token_usage_none_when_absent(self) -> None:
        assert TokenUsage.from_openai_response(object()) is None
        assert TokenUsage.from_openai_response(SimpleNamespace(usage=None)) is None

    def test_short_chunk_id(self) -> None:
        assert short_chunk_id("hdfc-bank__annual-report__fy2025-26.pdf_c00089_abc123") == "c00089"
        assert short_chunk_id("tiny") == "tiny"


class TestCompactRenderer:
    """Compact trace renders all stages from response + optional recorder."""

    def test_compact_full_response(self) -> None:
        out = render_compact_trace(_make_response(), TraceRecorder())
        for marker in (
            "🔍 PIPELINE TRACE", "🔄 QUERY REWRITER", "🔎 B4", "🎯 B5",
            "📦 B6", "✍️ B7", "🛡️ B8", "✅ FINAL ANSWER",
            "E1 →", "16.4%",
        ):
            assert marker in out, f"missing: {marker}"

    def test_compact_without_recorder_shows_not_captured(self) -> None:
        out = render_compact_trace(_make_response(), None)
        assert NOT_CAPTURED in out
        assert "E1 →" in out  # post-hoc evidence map needs no recorder

    def test_compact_no_citations(self) -> None:
        resp = _make_response(
            answer_text="Not disclosed in the ingested document.",
            cited_ids=[],
            uncertain=True,
        )
        out = render_compact_trace(resp, TraceRecorder())
        assert "Uncertain:      True" in out
        assert "Cited:          —" in out

    def test_compact_multiple_citations(self) -> None:
        cand2 = _make_candidate("doc_c00002_def", text="Deposits grew 15.2%.")
        r2 = RerankedResult.from_retrieval_result(cand2, rerank_score=0.80)
        # rebuild with two evidence items for a faithful map
        cand1 = _make_candidate("doc_c00001_abc", text="CET1 was 16.4% in FY2025-26.")
        r1 = RerankedResult.from_retrieval_result(cand1, rerank_score=0.95)
        context = EvidenceContextBuilder().build_context(
            [r1, r2], original_query="Q?", retrieval_query="Q?"
        )
        grounded = GroundedAnswer(
            answer_text="CET1 was 16.4% [E1] and deposits grew 15.2% [E2].",
            cited_evidence_ids=["E1", "E2"],
        )
        verified = CitationVerifier().verify(grounded, context)
        resp2 = DiligenceResponse(
            verified_answer=verified,
            evidence_context=context,
            pipeline_result=PipelineQueryResult(
                original_query="Q?", retrieval_query="Q?", candidates=[r1, r2],
                raw_retrieval_count=2, raw_candidates=[cand1, cand2],
            ),
        )
        out = render_compact_trace(resp2, TraceRecorder())
        assert "E1 →" in out and "E2 →" in out

    def test_compact_invalid_citation(self) -> None:
        resp = _make_response(answer_text="Known [E1], unknown [E9].", cited_ids=["E1", "E9"])
        out = render_compact_trace(resp, TraceRecorder())
        assert "Unverified tokens: [E9]" in out


class TestFullRenderer:
    """Full trace exposes forensic detail with honest gaps."""

    def test_full_sections_present(self) -> None:
        out = render_full_trace(_make_response(), TraceRecorder())
        for marker in (
            "FULL PIPELINE TRACE", "QUERY REWRITER", "B4 — HYBRID RETRIEVAL",
            "B5 — LLM RERANKER", "B6 — EVIDENCE CONTEXT", "B7 — GROUNDED GENERATOR",
            "B8 — CITATION VERIFIER", "FINAL OUTPUT", "B4 CANDIDATE #1",
            "EVIDENCE E1", "CITATION E1",
        ):
            assert marker in out, f"missing: {marker}"

    def test_full_marks_uncaptured_vector_split(self) -> None:
        out = render_full_trace(_make_response(), TraceRecorder())
        assert "pipeline retains only the fused hybrid score" in out

    def test_full_shows_b6_xml_and_evidence_map(self) -> None:
        out = render_full_trace(_make_response(), TraceRecorder())
        assert "E1 → doc_c00001_abc" in out
        assert "CET1 was 16.4% in FY2025-26." in out

    def test_full_no_secrets(self) -> None:
        out = render_full_trace(_make_response(), TraceRecorder())
        assert "OPENAI_API_KEY" not in out
        assert "sk-" not in out


class TestRecorderCapture:
    """Components record prompts, raw responses, usage, and fast paths."""

    def test_rewriter_fast_path_recorded(self) -> None:
        rec = TraceRecorder()
        rewriter = QueryRewriter(client=_mock_chat_client("unused"))
        out = rewriter.rewrite("Standalone question?", conversation_history=[], trace=rec)
        assert out == "Standalone question?"
        assert rec.stages["rewriter"].data["fast_path"] is True
        assert rec.stages["rewriter"].data["llm_called"] is False

    def test_rewriter_llm_path_recorded(self) -> None:
        rec = TraceRecorder()
        rewriter = QueryRewriter(client=_mock_chat_client("Rewritten standalone query?"))
        out = rewriter.rewrite(
            "What about it?",
            conversation_history=[{"role": "user", "content": "What was NII?"}, {"role": "assistant", "content": "NII was X."}],
            trace=rec,
        )
        assert out == "Rewritten standalone query?"
        data = rec.stages["rewriter"].data
        assert data["fast_path"] is False
        assert data["history_turns"] == 2
        assert "What about it?" in data["user_prompt"]
        assert data["raw_response"] == "Rewritten standalone query?"
        assert data["usage"] is not None

    def test_reranker_capture(self) -> None:
        rec = TraceRecorder()
        reranker = LLMReranker(client=_mock_chat_client('{"ranked_chunk_ids": ["c1", "c0"]}'))
        results = reranker.rerank(
            "Which chunk?",
            [_make_candidate("c0"), _make_candidate("c1")],
            original_query="Original?",
            trace=rec,
        )
        assert [r.chunk_id for r in results] == ["c1", "c0"]
        data = rec.stages["b5"].data
        assert data["model"] == "gpt-4o-mini"
        assert "c0" in data["prompt"] and "c1" in data["prompt"]
        assert data["parsed_ids"] == ["c1", "c0"]
        assert data["invalid_ids"] == []
        assert data["duplicate_count"] == 0
        assert data["fallback"] is False
        assert data["usage"] is not None
        assert rec.stages["b5"].latency_ms is not None

    def test_generator_capture(self) -> None:
        rec = TraceRecorder()
        cand = _make_candidate("doc_c00001_abc", text="CET1 was 16.4% in FY2025-26.")
        reranked = RerankedResult.from_retrieval_result(cand, rerank_score=0.95)
        context = EvidenceContextBuilder().build_context(
            [reranked], original_query="What was CET1?", retrieval_query="What was CET1?"
        )
        gen = GroundedGenerator(
            client=_mock_chat_client('{"answer_text": "CET1 was 16.4% [E1].", "cited_evidence_ids": ["E1"], "is_uncertain": false}')
        )
        answer = gen.generate(query="What was CET1?", context=context, trace=rec)
        assert answer.answer_text == "CET1 was 16.4% [E1]."
        data = rec.stages["b7"].data
        assert data["model"] == "gpt-4o"
        assert data["temperature"] == 0.0
        assert "CET1 was 16.4% [E1]" in data["raw_response"]
        assert "E1" in data["user_prompt"]
        assert data["usage"] is not None

    def test_tracing_does_not_change_results(self) -> None:
        """Identical outputs with and without a recorder across components."""
        history = [{"role": "user", "content": "What was NII?"}]
        mock_content = '{"ranked_chunk_ids": ["c1", "c0"]}'

        plain_rewriter = QueryRewriter(client=_mock_chat_client("R?"))
        traced_rewriter = QueryRewriter(client=_mock_chat_client("R?"))
        assert (
            plain_rewriter.rewrite("What about it?", history)
            == traced_rewriter.rewrite("What about it?", history, trace=TraceRecorder())
        )

        cands = [_make_candidate("c0"), _make_candidate("c1")]
        plain_reranker = LLMReranker(client=_mock_chat_client(mock_content))
        traced_reranker = LLMReranker(client=_mock_chat_client(mock_content))
        plain_out = [r.chunk_id for r in plain_reranker.rerank("Q", cands)]
        traced_out = [r.chunk_id for r in traced_reranker.rerank("Q", cands, trace=TraceRecorder())]
        assert plain_out == traced_out

    def test_missing_metadata_renders_gracefully(self) -> None:
        """Sparse evidence items render without crashing either format."""
        item = EvidenceItem(
            evidence_id="E1",
            chunk_id="c_no_meta",
            text="Sparse text.",
            page_numbers=[],
            section_heading=None,
            heading_path=[],
            chunk_type="narrative",
            company="",
            document_name="d.pdf",
            financial_period=None,
            source_element_ids=["e"],
            table_metadata=None,
            rerank_score=0.5,
            retrieval_score=0.5,
        )
        context = EvidenceContext(
            original_query="Q?",
            retrieval_query="Q?",
            evidence_items=[item],
            formatted_context="<diligence_context>Sparse text.</diligence_context>",
            evidence_map={"E1": item},
            omitted_items=[],
            total_context_chars=10,
        )
        cand = _make_candidate("c_no_meta", text="Sparse text.")
        reranked = RerankedResult.from_retrieval_result(cand, rerank_score=0.5)
        grounded = GroundedAnswer(answer_text="Sparse text [E1].", cited_evidence_ids=["E1"])
        verified = CitationVerifier().verify(grounded, context)
        resp = DiligenceResponse(
            verified_answer=verified,
            evidence_context=context,
            pipeline_result=PipelineQueryResult(
                original_query="Q?", retrieval_query="Q?", candidates=[reranked],
                raw_retrieval_count=1, raw_candidates=[cand],
            ),
        )
        assert "Unknown" in render_compact_trace(resp, TraceRecorder())
        assert "Unknown" in render_full_trace(resp, TraceRecorder())
