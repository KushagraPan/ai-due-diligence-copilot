"""
Unit tests for the Phase B5 LLM Reranker layer.

Verifies:
1. Correct JSON parsing of ranked_chunk_ids responses.
2. Only supplied B4 chunk IDs accepted (unknown IDs rejected).
3. Duplicate IDs deduplicated safely preserving model order.
4. Invalid IDs rejected without breaking ranking.
5. Fewer-than-7 valid candidates returns fewer results.
6. Empty B4 candidate set returns [].
7. LLM API failure falls back to deterministic B4 input order.
8. Malformed LLM responses fall back safely.
9. Timeout configuration and retry-safe fallback behavior.
10. Deterministic candidate ordering, pruning, and top_k limiting.
11. Standalone vs consolidated discrimination wiring.
12. Subsidiary vs HDFC Bank discrimination wiring.
13. Table vs narrative relevance wiring.
14. Primary financial statement vs descriptive commentary wiring.
15. Negative/unanswerable queries.
16. Conversational retrieval queries (original + retrieval query).
All tests run offline with mocked LLM clients (NO network calls).
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from due_diligence.prompts.reranker import (
    LLM_RERANKER_SYSTEM_PROMPT,
    format_llm_reranker_prompt,
)
from due_diligence.retrieval.reranker import (
    LLMReranker,
    RerankerConfigurationError,
    RerankerInputError,
    RerankerModelError,
    parse_ranked_chunk_ids,
)
from due_diligence.retrieval.schemas import RerankedResult, RetrievalResult


def _make_candidate(
    chunk_id: str,
    text: str = "Corporate disclosure passage text.",
    score: float = 0.75,
    company: str = "HDFC Bank",
    document_name: str = "hdfc_fy26.pdf",
    page_numbers: list[int] | None = None,
    section_heading: str | None = "Capital Adequacy",
    chunk_type: str = "narrative",
    entity_scope: str = "hdfc_bank_general",
    table_metadata: dict[str, Any] | None = None,
) -> RetrievalResult:
    """Build a valid RetrievalResult test candidate."""
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        company=company,
        document_name=document_name,
        document_type="pdf",
        financial_period="FY2025-26",
        page_numbers=page_numbers or [10],
        section_heading=section_heading,
        heading_path=["Financial Statements", "Notes"],
        chunk_type=chunk_type,
        source_element_ids=[f"elem_{chunk_id}"],
        table_metadata=table_metadata,
        entity_scope=entity_scope,
    )


def _mock_client(content: str) -> MagicMock:
    """Build a mock OpenAI client returning the given completion content."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


def _ranked(ids: list[str]) -> str:
    """Serialize a ranked_chunk_ids payload."""
    return json.dumps({"ranked_chunk_ids": ids})


# ── Initialization Tests ────────────────────────────────────────────────


class TestLLMInitialization:
    """Verify LLMReranker defaults and configuration validation."""

    def test_default_initialization(self) -> None:
        reranker = LLMReranker(client=MagicMock())
        assert reranker.model_name == "gpt-4o-mini"
        assert reranker.top_k == 7
        assert reranker.candidate_count == 40

    def test_explicit_model_and_limits(self) -> None:
        reranker = LLMReranker(client=MagicMock(), model_name="gpt-4o", top_k=5, candidate_count=25)
        assert reranker.model_name == "gpt-4o"
        assert reranker.top_k == 5
        assert reranker.candidate_count == 25

    def test_invalid_top_k_raises(self) -> None:
        with pytest.raises(RerankerConfigurationError, match="top_k must be a positive integer"):
            LLMReranker(client=MagicMock(), top_k=0)

    def test_invalid_candidate_count_raises(self) -> None:
        with pytest.raises(RerankerConfigurationError, match="candidate_count must be a positive integer"):
            LLMReranker(client=MagicMock(), candidate_count=0)

    def test_invalid_timeout_raises(self) -> None:
        with pytest.raises(RerankerConfigurationError, match="timeout must be positive"):
            LLMReranker(client=MagicMock(), timeout=0)

    def test_missing_api_key_raises(self) -> None:
        # Whitespace-only explicit key fails validation regardless of environment
        with pytest.raises(RerankerConfigurationError, match="OpenAI API key must be provided"):
            LLMReranker(api_key="  ")

    def test_timeout_passed_to_openai_client(self) -> None:
        with (
            patch("due_diligence.retrieval.reranker.openai.OpenAI") as mock_openai,
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        ):
            LLMReranker(timeout=33.0)
        _, kwargs = mock_openai.call_args
        assert kwargs["timeout"] == 33.0


# ── Input Validation Tests ──────────────────────────────────────────────


class TestLLMInputValidation:
    """Verify error handling on empty or invalid inputs."""

    def test_empty_query_raises_input_error(self) -> None:
        reranker = LLMReranker(client=MagicMock())
        with pytest.raises(RerankerInputError, match="Query cannot be empty or whitespace-only"):
            reranker.rerank("", [_make_candidate("c1")])

    def test_whitespace_query_raises_input_error(self) -> None:
        reranker = LLMReranker(client=MagicMock())
        with pytest.raises(RerankerInputError, match="Query cannot be empty or whitespace-only"):
            reranker.rerank("   \t\n  ", [_make_candidate("c1")])

    def test_non_string_query_raises_input_error(self) -> None:
        reranker = LLMReranker(client=MagicMock())
        with pytest.raises(RerankerInputError, match="Query cannot be empty or whitespace-only"):
            reranker.rerank(None, [_make_candidate("c1")])  # type: ignore[arg-type]

    def test_non_list_candidates_raises_input_error(self) -> None:
        reranker = LLMReranker(client=MagicMock())
        with pytest.raises(RerankerInputError, match="Expected list of RetrievalResult"):
            reranker.rerank("Query", "not_a_list")  # type: ignore[arg-type]

    def test_empty_candidates_returns_empty(self) -> None:
        reranker = LLMReranker(client=MagicMock())
        assert reranker.rerank("Valid query", []) == []

    def test_invalid_top_k_override_raises(self) -> None:
        reranker = LLMReranker(client=MagicMock())
        with pytest.raises(RerankerConfigurationError, match="top_k override must be a positive integer"):
            reranker.rerank("Query", [_make_candidate("c1")], top_k=0)

    def test_invalid_candidate_count_override_raises(self) -> None:
        reranker = LLMReranker(client=MagicMock())
        with pytest.raises(RerankerConfigurationError, match="candidate_count override must be a positive integer"):
            reranker.rerank("Query", [_make_candidate("c1")], candidate_count=0)


# ── Chunk-ID Contract Tests ─────────────────────────────────────────────


class TestChunkIdContract:
    """Verify strict ranked_chunk_ids parsing, validation, and fallback."""

    def test_happy_path_orders_by_chunk_ids(self) -> None:
        reranker = LLMReranker(client=_mock_client(_ranked(["c1", "c0"])), top_k=2)
        results = reranker.rerank(query="What is CET1?", candidates=[_make_candidate("c0"), _make_candidate("c1")])
        assert [r.chunk_id for r in results] == ["c1", "c0"]
        assert results[0].rerank_score > results[1].rerank_score

    def test_unknown_ids_rejected(self) -> None:
        reranker = LLMReranker(client=_mock_client(_ranked(["c_ghost", "c1", "c0"])), top_k=3)
        results = reranker.rerank(query="Q", candidates=[_make_candidate("c0"), _make_candidate("c1")])
        assert [r.chunk_id for r in results] == ["c1", "c0"]

    def test_duplicate_ids_deduplicated(self) -> None:
        reranker = LLMReranker(client=_mock_client(_ranked(["c1", "c1", "c0", "c1"])), top_k=3)
        results = reranker.rerank(query="Q", candidates=[_make_candidate("c0"), _make_candidate("c1")])
        assert [r.chunk_id for r in results] == ["c1", "c0"]

    def test_non_string_ids_rejected(self) -> None:
        reranker = LLMReranker(client=_mock_client('{"ranked_chunk_ids": [1, null, "c0"]}'), top_k=2)
        results = reranker.rerank(query="Q", candidates=[_make_candidate("c0"), _make_candidate("c1")])
        assert results[0].chunk_id == "c0"

    def test_fewer_than_top_k_returns_fewer(self) -> None:
        reranker = LLMReranker(client=_mock_client(_ranked(["c0"])), top_k=7)
        results = reranker.rerank(query="Q", candidates=[_make_candidate("c0")])
        assert len(results) == 1
        assert results[0].chunk_id == "c0"

    def test_malformed_json_falls_back_to_input_order(self) -> None:
        reranker = LLMReranker(client=_mock_client("not json at all {{{"))
        candidates = [_make_candidate("c0"), _make_candidate("c1")]
        results = reranker.rerank(query="Q", candidates=candidates)
        assert [r.chunk_id for r in results] == ["c0", "c1"]

    def test_missing_key_falls_back_to_input_order(self) -> None:
        reranker = LLMReranker(client=_mock_client('{"wrong_key": ["c1"]}'))
        candidates = [_make_candidate("c0"), _make_candidate("c1")]
        results = reranker.rerank(query="Q", candidates=candidates)
        assert [r.chunk_id for r in results] == ["c0", "c1"]

    def test_non_list_value_falls_back_to_input_order(self) -> None:
        reranker = LLMReranker(client=_mock_client('{"ranked_chunk_ids": "c1"}'))
        candidates = [_make_candidate("c0"), _make_candidate("c1")]
        results = reranker.rerank(query="Q", candidates=candidates)
        assert [r.chunk_id for r in results] == ["c0", "c1"]

    def test_parse_helper_rejects_invalid_payload(self) -> None:
        with pytest.raises(RerankerModelError, match="invalid JSON"):
            parse_ranked_chunk_ids("{{{", ["c0"])
        with pytest.raises(RerankerModelError, match="ranked_chunk_ids"):
            parse_ranked_chunk_ids('{"other": []}', ["c0"])

    def test_parse_helper_filters_and_dedupes(self) -> None:
        out = parse_ranked_chunk_ids(_ranked(["c9", "c1", "c1", "c0", 5]), ["c0", "c1"])
        assert out == ["c1", "c0"]


# ── Determinism, Pruning & Preservation Tests ───────────────────────────


class TestDeterminismAndPreservation:
    """Verify deterministic ordering, candidate pruning, and metadata preservation."""

    def test_candidate_count_limits_scored_passages(self) -> None:
        reranker = LLMReranker(
            client=_mock_client(_ranked(["c_2", "c_1", "c_0"])), candidate_count=3, top_k=5
        )
        candidates = [_make_candidate(f"c_{i}") for i in range(6)]
        results = reranker.rerank("Query", candidates)
        assert [r.chunk_id for r in results] == ["c_2", "c_1", "c_0"]

    def test_top_k_limits_output(self) -> None:
        reranker = LLMReranker(client=_mock_client(_ranked(["c1", "c0"])), top_k=1)
        results = reranker.rerank("Q", [_make_candidate("c0"), _make_candidate("c1")])
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_top_k_override(self) -> None:
        reranker = LLMReranker(client=_mock_client(_ranked(["c0", "c1", "c2"])), top_k=5)
        results = reranker.rerank("Q", [_make_candidate(f"c{i}") for i in range(3)], top_k=2)
        assert len(results) == 2

    def test_unranked_candidates_appended_in_input_order(self) -> None:
        reranker = LLMReranker(client=_mock_client(_ranked(["c2"])), top_k=3)
        candidates = [_make_candidate("c0"), _make_candidate("c1"), _make_candidate("c2")]
        results = reranker.rerank("Q", candidates)
        assert [r.chunk_id for r in results] == ["c2", "c0", "c1"]

    def test_pseudo_scores_descend_and_b4_scores_preserved(self) -> None:
        reranker = LLMReranker(client=_mock_client(_ranked(["c1", "c0"])), top_k=2)
        c0 = _make_candidate("c0", score=0.91)
        c1 = _make_candidate("c1", score=0.42)
        results = reranker.rerank("Q", [c0, c1])
        assert results[0].rerank_score > results[1].rerank_score
        assert results[0].score == pytest.approx(0.42)
        assert results[1].score == pytest.approx(0.91)

    def test_full_provenance_and_table_metadata_preserved(self) -> None:
        meta = {"table_group_id": "tbl", "headers": ["A"], "caption": "Cap"}
        cand = _make_candidate(
            "tbl1", text="| A |\n|---|\n| 1 |", chunk_type="table",
            page_numbers=[42], section_heading="Sched", table_metadata=meta,
            entity_scope="hdfc_bank_standalone",
        )
        reranker = LLMReranker(client=_mock_client(_ranked(["tbl1"])), top_k=1)
        (res,) = reranker.rerank("Q", [cand])
        assert isinstance(res, RerankedResult)
        assert res.text == cand.text
        assert res.table_metadata == meta
        assert res.entity_scope == "hdfc_bank_standalone"
        assert res.page_numbers == [42]


# ── Resilience Tests ────────────────────────────────────────────────────


class TestLLMResilience:
    """Verify safe fallback on API, timeout, and content failures."""

    def test_api_error_falls_back_to_input_order(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("rate limit")
        reranker = LLMReranker(client=mock_client, top_k=2)
        results = reranker.rerank(query="Q", candidates=[_make_candidate("c0"), _make_candidate("c1")])
        assert [r.chunk_id for r in results] == ["c0", "c1"]

    def test_none_content_falls_back_to_input_order(self) -> None:
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        reranker = LLMReranker(client=mock_client, top_k=2)
        results = reranker.rerank(query="Q", candidates=[_make_candidate("c0"), _make_candidate("c1")])
        assert [r.chunk_id for r in results] == ["c0", "c1"]

    def test_empty_ranked_list_falls_back_to_input_order(self) -> None:
        reranker = LLMReranker(client=_mock_client(_ranked([])), top_k=2)
        results = reranker.rerank(query="Q", candidates=[_make_candidate("c0"), _make_candidate("c1")])
        assert [r.chunk_id for r in results] == ["c0", "c1"]

    def test_all_unknown_ids_falls_back_to_input_order(self) -> None:
        reranker = LLMReranker(client=_mock_client(_ranked(["ghost1", "ghost2"])), top_k=2)
        results = reranker.rerank(query="Q", candidates=[_make_candidate("c0"), _make_candidate("c1")])
        assert [r.chunk_id for r in results] == ["c0", "c1"]


# ── Discrimination Wiring Tests (mocked model order honored) ────────────


class TestDiscriminationWiring:
    """Verify the model-specified order wins for scope/table/statement contrasts."""

    def _ranked_first(self, first_id: str, candidates: list[RetrievalResult]) -> list[RerankedResult]:
        others = [c.chunk_id for c in candidates if c.chunk_id != first_id]
        reranker = LLMReranker(client=_mock_client(_ranked([first_id, *others])), top_k=7)
        return reranker.rerank("Q", candidates)

    def test_standalone_beats_consolidated(self) -> None:
        sa = _make_candidate("c_sa", text="Basic EPS 48.62.", entity_scope="hdfc_bank_standalone")
        con = _make_candidate("c_con", text="Basic EPS 49.50.", entity_scope="hdfc_bank_consolidated")
        results = self._ranked_first("c_sa", [con, sa])
        assert results[0].chunk_id == "c_sa"
        assert results[0].entity_scope == "hdfc_bank_standalone"

    def test_parent_beats_subsidiary(self) -> None:
        sub = _make_candidate("c_hdb", text="HDBFSL PAT 2,544 cr.", entity_scope="hdb_financial_services")
        parent = _make_candidate("c_bank", text="Bank PAT 74,671 cr.", entity_scope="hdfc_bank_standalone")
        results = self._ranked_first("c_bank", [sub, parent])
        assert results[0].chunk_id == "c_bank"

    def test_table_evidence_honored(self) -> None:
        narr = _make_candidate("c_narr", text="Advances grew strongly this year.", chunk_type="narrative")
        tbl = _make_candidate(
            "c_tbl", text="| Sch | FY26 |\n|---|---|\n| Total | 34,179,458,972 |",
            chunk_type="table",
        )
        results = self._ranked_first("c_tbl", [narr, tbl])
        assert results[0].chunk_id == "c_tbl"
        assert results[0].chunk_type == "table"

    def test_primary_statement_beats_commentary(self) -> None:
        comm = _make_candidate("c_comm", text="Deposits saw healthy growth.", section_heading="MD&A Overview")
        stmt = _make_candidate(
            "c_stmt", text="| Sch | FY26 |\n|---|---|\n| Deposits | 31,052,504,740 |",
            section_heading="SCHEDULE 3 - DEPOSITS", chunk_type="table",
        )
        results = self._ranked_first("c_stmt", [comm, stmt])
        assert results[0].chunk_id == "c_stmt"

    def test_negative_query_returns_fewer(self) -> None:
        reranker = LLMReranker(client=_mock_client(_ranked(["c0"])), top_k=7)
        candidates = [_make_candidate(f"c{i}") for i in range(5)]
        results = reranker.rerank("Valuation cap for Series C acquisition of FinTech Corp?", candidates)
        assert len(results) == 5  # ranked 1 + 4 appended deterministically
        assert results[0].chunk_id == "c0"


# ── Conversational Query Tests ──────────────────────────────────────────


class TestConversationalReranking:
    """Verify original + retrieval queries reach the ranking prompt."""

    def test_original_query_forwarded_to_prompt(self) -> None:
        reranker = LLMReranker(client=_mock_client(_ranked(["c0"])), top_k=1)
        reranker.rerank(
            query="What was HDB Financial Services' profit after tax as of March 31, 2026?",
            candidates=[_make_candidate("c0")],
            original_query="What was its profit after tax for the same period?",
        )
        sent_prompt = reranker._client.chat.completions.create.call_args[1]["messages"][1]["content"]
        assert "What was its profit after tax for the same period?" in sent_prompt
        assert "What was HDB Financial Services' profit after tax as of March 31, 2026?" in sent_prompt

    def test_prompt_without_original_query(self) -> None:
        reranker = LLMReranker(client=_mock_client(_ranked(["c0"])), top_k=1)
        results = reranker.rerank(query="Standalone query", candidates=[_make_candidate("c0")])
        assert results[0].chunk_id == "c0"
        sent_prompt = reranker._client.chat.completions.create.call_args[1]["messages"][1]["content"]
        assert "Standalone query" in sent_prompt


# ── Prompt Contract Tests ───────────────────────────────────────────────


class TestRerankerPromptContract:
    """Verify the ranking prompt carries queries, IDs, scopes, and full text."""

    def test_prompt_contains_queries_ids_scopes_and_text(self) -> None:
        cand = _make_candidate(
            "doc_c00001_abc", text="Basic EPS stood at 48.62.",
            section_heading="SCHEDULE 12", entity_scope="hdfc_bank_standalone",
        )
        prompt = format_llm_reranker_prompt(
            retrieval_query="HDFC Bank basic EPS FY2025-26?",
            candidates=[cand],
            original_query="What was its EPS?",
        )
        assert "What was its EPS?" in prompt
        assert "HDFC Bank basic EPS FY2025-26?" in prompt
        assert "doc_c00001_abc" in prompt
        assert "hdfc_bank_standalone" in prompt
        assert "Basic EPS stood at 48.62." in prompt
        assert "SCHEDULE 12" in prompt

    def test_long_text_truncated_with_marker(self) -> None:
        cand = _make_candidate("c_long", text="X" * 5000)
        prompt = format_llm_reranker_prompt(retrieval_query="Q", candidates=[cand], max_chars_per_candidate=100)
        assert "[TRUNCATED]" in prompt
        assert "X" * 5000 not in prompt

    def test_system_prompt_establishes_ranking_role_and_rules(self) -> None:
        required = [
            "ranking component",
            "do NOT answer",
            "ranked_chunk_ids",
            "ENTITY SCOPE",
            "TABLES ARE FIRST-CLASS",
            "PRIMARY STATEMENTS",
            "METRIC SCOPE",
            "NEGATIVE",
            "CONVERSATIONAL",
        ]
        for phrase in required:
            assert phrase in LLM_RERANKER_SYSTEM_PROMPT, f"missing prompt rule: {phrase}"
