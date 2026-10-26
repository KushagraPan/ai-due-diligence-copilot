"""
Unit tests for Phase B8 Citation Verification and Attribution subsystem.

Verifies:
1. Exact citation string formatting according to AGENTS.md Section 4.5.
2. Graceful fallback when page numbers are missing ('Page: Unknown').
3. Deterministic mapping of [EX] evidence tokens to numbered footnotes [1], [2].
4. Detection and flagging of unverified/hallucinated citation tokens.
5. Handling of duplicate citation tokens in answer text.
6. Preservation of order of appearance for footnote numbering.
7. Optional disabling of token replacement (retaining [EX] in text).
8. Uncertainty flag propagation from GroundedAnswer.
9. Functional convenience wrapper verify_citations.
10. Backward-compatible CitationEngine alias.
11. Immutability of Citation and VerifiedAnswer models.
"""

import pytest
from pydantic import ValidationError

from due_diligence.agents.citation_engine import CitationEngine
from due_diligence.agents.citation_verifier import (
    Citation,
    CitationVerifier,
    VerifiedAnswer,
    resolve_citation_heading,
    verify_citations,
)
from due_diligence.agents.generator import GroundedAnswer
from due_diligence.retrieval.schemas import EvidenceContext, EvidenceItem


def _make_evidence_item(
    evidence_id: str,
    text: str = "Common Equity Tier 1 (CET1) ratio was 16.4% under Basel III regulations.",
    chunk_id: str = "chunk_1",
    page_numbers: list[int] | None = None,
    section_heading: str | None = "Note 14: Capital Adequacy",
    document_name: str = "hdfc_annual_report_fy26.pdf",
    chunk_type: str = "narrative",
) -> EvidenceItem:
    """Helper to build an EvidenceItem with realistic metadata."""
    return EvidenceItem(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        text=text,
        page_numbers=page_numbers or [402],
        section_heading=section_heading,
        heading_path=["Financial Statements", "Notes", section_heading] if section_heading else [],
        chunk_type=chunk_type,
        company="HDFC Bank",
        document_name=document_name,
        financial_period="FY2025-26",
        source_element_ids=[f"elem_{chunk_id}"],
        table_metadata=None,
        rerank_score=0.92,
        retrieval_score=0.81,
    )


def _make_context(
    items: list[EvidenceItem],
    original_query: str = "What was HDFC Bank's CET1 ratio?",
    retrieval_query: str = "What was HDFC Bank's CET1 ratio in FY2025-26?",
) -> EvidenceContext:
    """Helper to construct an EvidenceContext from evidence items."""
    evidence_map = {item.evidence_id: item for item in items}
    return EvidenceContext(
        original_query=original_query,
        retrieval_query=retrieval_query,
        evidence_items=items,
        formatted_context="<diligence_context>...</diligence_context>",
        evidence_map=evidence_map,
        omitted_items=[],
        total_context_chars=100,
    )


def test_citation_formatting() -> None:
    """Test format_citation output format under various metadata states."""
    # Complete metadata
    cit1 = Citation(
        citation_number=1,
        evidence_id="E1",
        chunk_id="chunk_1",
        document_name="annual_report_2026.pdf",
        page_numbers=[42],
        section_heading="Note 7: Long-Term Debt",
    )
    assert cit1.format_citation() == "[annual_report_2026.pdf, Page 42, Note 7: Long-Term Debt]"

    # Multi-page chunk
    cit2 = Citation(
        citation_number=2,
        evidence_id="E2",
        chunk_id="chunk_2",
        document_name="annual_report_2026.pdf",
        page_numbers=[378, 379],
        section_heading="Balance Sheet",
    )
    assert cit2.format_citation() == "[annual_report_2026.pdf, Page 378, Page 379, Balance Sheet]"

    # Missing section heading
    cit3 = Citation(
        citation_number=3,
        evidence_id="E3",
        chunk_id="chunk_3",
        document_name="board_deck.pdf",
        page_numbers=[15],
        section_heading=None,
    )
    assert cit3.format_citation() == "[board_deck.pdf, Page 15]"

    # Missing page numbers: reports 'Page: Unknown' per AGENTS.md Section 4.5
    cit4 = Citation(
        citation_number=4,
        evidence_id="E4",
        chunk_id="chunk_4",
        document_name="web_disclosure.pdf",
        page_numbers=[],
        section_heading="Overview",
    )
    assert cit4.format_citation() == "[web_disclosure.pdf, Page: Unknown, Overview]"


def test_verify_citations_success() -> None:
    """Test successful verification and numeric footnote replacement."""
    item1 = _make_evidence_item("E1", "HDFC Bank reported CET1 ratio was 16.4%.", "c1", [402], "Capital Adequacy")
    item2 = _make_evidence_item("E2", "Deposits grew 15.2%.", "c2", [378], "Balance Sheet")
    ctx = _make_context([item1, item2])

    answer = GroundedAnswer(
        answer_text="HDFC Bank reported CET1 of 16.4% [E1], while deposits grew 15.2% [E2].",
        cited_evidence_ids=["E1", "E2"],
        is_uncertain=False,
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert res.answer_text == "HDFC Bank reported CET1 of 16.4% [1], while deposits grew 15.2% [2]."
    assert res.raw_answer_text == answer.answer_text
    assert len(res.citations) == 2
    assert res.citations[0].citation_number == 1
    assert res.citations[0].evidence_id == "E1"
    assert res.citations[0].chunk_id == "c1"
    assert res.citations[0].page_numbers == [402]
    assert res.citations[1].citation_number == 2
    assert res.citations[1].evidence_id == "E2"
    assert len(res.unverified_citations) == 0
    assert not res.is_uncertain


def test_verify_citations_duplicate_in_text() -> None:
    """Test that repeated citations in text share the same footnote number."""
    item1 = _make_evidence_item("E1", "CET1 was 16.4%. Regulatory requirement is 8.0%.", "c1", [402])
    ctx = _make_context([item1])

    answer = GroundedAnswer(
        answer_text="CET1 was 16.4% [E1]. Regulatory requirement is 8.0% [E1].",
        cited_evidence_ids=["E1"],
        is_uncertain=False,
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert res.answer_text == "CET1 was 16.4% [1]. Regulatory requirement is 8.0% [1]."
    assert len(res.citations) == 1
    assert res.citations[0].citation_number == 1


def test_verify_citations_unverified_token() -> None:
    """Test detection and explicit tagging of hallucinated/unverified citation tokens."""
    item1 = _make_evidence_item("E1", "CET1 was 16.4%. Regulatory requirement is 8.0%.", "c1", [402])
    ctx = _make_context([item1])

    answer = GroundedAnswer(
        answer_text="CET1 was 16.4% [E1]. Burn rate was 10M [E9].",
        cited_evidence_ids=["E1", "E9"],
        is_uncertain=False,
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert "CET1 was 16.4% [1]." in res.answer_text
    assert "[Unverified: E9]" in res.answer_text
    assert len(res.citations) == 1
    assert res.unverified_citations == ["E9"]


def test_verify_citations_order_of_appearance() -> None:
    """Test that footnote numbers follow appearance order in text."""
    item1 = _make_evidence_item("E1", "Note 1", "c1", [10])
    item2 = _make_evidence_item("E2", "Note 2", "c2", [20])
    ctx = _make_context([item1, item2])

    # E2 appears before E1 in the text
    answer = GroundedAnswer(
        answer_text="First mentioned [E2], then mentioned [E1].",
        cited_evidence_ids=["E1", "E2"],
        is_uncertain=False,
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert res.answer_text == "First mentioned [1], then mentioned [2]."
    assert res.citations[0].evidence_id == "E2"
    assert res.citations[0].citation_number == 1
    assert res.citations[1].evidence_id == "E1"
    assert res.citations[1].citation_number == 2


def test_verify_citations_no_replacement() -> None:
    """Test that setting replace_tokens_with_numbers=False keeps [EX] tokens in text."""
    item1 = _make_evidence_item("E1", "CET1 was 16.4%. Regulatory requirement is 8.0%.", "c1", [402])
    ctx = _make_context([item1])

    answer = GroundedAnswer(
        answer_text="CET1 was 16.4% [E1].",
        cited_evidence_ids=["E1"],
        is_uncertain=False,
    )

    verifier = CitationVerifier(replace_tokens_with_numbers=False)
    res = verifier.verify(answer, ctx)

    assert res.answer_text == "CET1 was 16.4% [E1]."
    assert len(res.citations) == 1
    assert res.citations[0].evidence_id == "E1"


def test_verify_uncertain_answer_without_citations() -> None:
    """Test that uncertain/refusal answers without citations are handled cleanly."""
    ctx = _make_context([])

    answer = GroundedAnswer(
        answer_text="Not disclosed in the ingested document.",
        cited_evidence_ids=[],
        is_uncertain=True,
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert res.answer_text == "Not disclosed in the ingested document."
    assert len(res.citations) == 0
    assert len(res.unverified_citations) == 0
    assert res.is_uncertain is True


def test_verify_citations_convenience_wrapper() -> None:
    """Test verify_citations function wrapper."""
    item1 = _make_evidence_item("E1", "Excerpt", "c1", [100])
    ctx = _make_context([item1])
    answer = GroundedAnswer(
        answer_text="Fact [E1].",
        cited_evidence_ids=["E1"],
        is_uncertain=False,
    )

    res = verify_citations(answer, ctx)
    assert res.answer_text == "Fact [1]."
    assert len(res.citations) == 1


def test_citation_engine_alias() -> None:
    """Test that CitationEngine is an alias of CitationVerifier."""
    assert CitationEngine is CitationVerifier


def test_models_immutability() -> None:
    """Test that Citation and VerifiedAnswer models are immutable."""
    cit = Citation(
        citation_number=1,
        evidence_id="E1",
        chunk_id="c1",
        document_name="doc.pdf",
        page_numbers=[1],
    )
    with pytest.raises(ValidationError):
        cit.citation_number = 2

    ans = VerifiedAnswer(
        answer_text="Text",
        raw_answer_text="Text",
        citations=[cit],
    )
    with pytest.raises(ValidationError):
        ans.answer_text = "Modified"


# ── Explicit Phase B8 Specification Tests ───────────────────────────────────


def test_phase_b8_valid_e1_citation() -> None:
    """TEST: [E1] valid citation translation."""
    item = _make_evidence_item("E1", "Advances grew 15%.", "c1", [42], "Lending", "hdfc_fy26.pdf")
    ctx = _make_context([item])
    answer = GroundedAnswer(
        answer_text="Total advances grew 15% year-over-year [E1].",
        cited_evidence_ids=["E1"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert res.answer_text == "Total advances grew 15% year-over-year [1]."
    assert len(res.citations) == 1
    assert res.citations[0].evidence_id == "E1"
    assert res.citations[0].document_name == "hdfc_fy26.pdf"
    assert res.citations[0].page_numbers == [42]
    assert res.citations[0].section_heading == "Lending"
    assert res.citations[0].format_citation() == "[hdfc_fy26.pdf, Page 42, Lending]"
    assert res.unverified_citations == []


def test_phase_b8_multiple_valid_citations() -> None:
    """TEST: multiple valid citations across statements."""
    item1 = _make_evidence_item("E1", "Deposits grew strongly.", "c1", [100], "Deposits")
    item2 = _make_evidence_item("E2", "NIM expanded 10 bps.", "c2", [105], "Margins")
    ctx = _make_context([item1, item2])
    answer = GroundedAnswer(
        answer_text="Deposits grew strongly [E1] and NIM expanded 10 bps [E2].",
        cited_evidence_ids=["E1", "E2"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert res.answer_text == "Deposits grew strongly [1] and NIM expanded 10 bps [2]."
    assert len(res.citations) == 2
    assert res.citations[0].evidence_id == "E1"
    assert res.citations[1].evidence_id == "E2"
    assert res.unverified_citations == []


def test_phase_b8_invalid_citation_e99() -> None:
    """TEST: invalid [E99] citation detection and tagging."""
    item1 = _make_evidence_item("E1", "Real fact.", "c1", [10])
    ctx = _make_context([item1])
    answer = GroundedAnswer(
        answer_text="Fact is true [E1], but unsupported projection is 50% [E99].",
        cited_evidence_ids=["E1", "E99"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert "[1]" in res.answer_text
    assert "[Unverified: E99]" in res.answer_text
    assert len(res.citations) == 1
    assert res.citations[0].evidence_id == "E1"
    assert res.unverified_citations == ["E99"]


def test_phase_b8_duplicate_tags() -> None:
    """TEST: duplicate tags in answer text share identical footnote numbering."""
    item1 = _make_evidence_item("E1", "Fact text.", "c1", [25])
    ctx = _make_context([item1])
    answer = GroundedAnswer(
        answer_text="Revenue reached $100M [E1]. As noted, revenue was $100M [E1].",
        cited_evidence_ids=["E1"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert res.answer_text == "Revenue reached $100M [1]. As noted, revenue was $100M [1]."
    assert len(res.citations) == 1
    assert res.citations[0].citation_number == 1
    assert res.unverified_citations == []


def test_phase_b8_malformed_tags() -> None:
    """TEST: malformed tags like [EX], [Eabc], [E] are detected and flagged."""
    item1 = _make_evidence_item("E1", "Fact.", "c1", [10])
    ctx = _make_context([item1])
    answer = GroundedAnswer(
        answer_text="Valid claim [E1], placeholder token [EX], and bad token [Eabc].",
        cited_evidence_ids=["E1"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert "[1]" in res.answer_text
    assert "[Unverified: EX]" in res.answer_text
    assert "[Unverified: Eabc]" in res.answer_text
    assert len(res.citations) == 1
    assert "EX" in res.unverified_citations
    assert "Eabc" in res.unverified_citations


def test_phase_b8_no_citations() -> None:
    """TEST: answer with no citations handled cleanly."""
    ctx = _make_context([])
    answer = GroundedAnswer(
        answer_text="The requested information is not disclosed in the ingested document.",
        cited_evidence_ids=[],
        is_uncertain=True,
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert res.answer_text == "The requested information is not disclosed in the ingested document."
    assert len(res.citations) == 0
    assert len(res.unverified_citations) == 0
    assert res.is_uncertain is True


def test_phase_b8_empty_answer() -> None:
    """TEST: empty answer text handled gracefully without exception."""
    item1 = _make_evidence_item("E1")
    ctx = _make_context([item1])

    verifier = CitationVerifier()

    # Empty string via GroundedAnswer
    res1 = verifier.verify(GroundedAnswer(answer_text="", cited_evidence_ids=[]), ctx)
    assert res1.answer_text == ""
    assert len(res1.citations) == 0
    assert res1.is_uncertain is True

    # Empty string directly
    res2 = verifier.verify("", ctx)
    assert res2.answer_text == ""
    assert len(res2.citations) == 0
    assert res2.is_uncertain is True

    # Whitespace only
    res3 = verifier.verify("   \n\t  ", ctx)
    assert len(res3.citations) == 0
    assert res3.is_uncertain is True


def test_phase_b8_exact_page_document_provenance_zero_llm_trust() -> None:
    """
    TEST: exact page/document provenance.
    Verifies that even if the LLM hallucinated page coordinates in prose ('Page 888'),
    provenance is resolved ONLY from EvidenceContext (Page 402).
    """
    item1 = _make_evidence_item(
        "E1",
        text="CET1 ratio was 16.4%.",
        chunk_id="chunk_real_1",
        page_numbers=[402],
        section_heading="Capital Adequacy",
        document_name="hdfc_annual_report_fy26.pdf",
    )
    ctx = _make_context([item1])

    answer = GroundedAnswer(
        answer_text="According to Page 888, the CET1 ratio was 16.4% [E1].",
        cited_evidence_ids=["E1"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    citation = res.citations[0]
    assert citation.document_name == "hdfc_annual_report_fy26.pdf"
    assert citation.page_numbers == [402]
    assert 888 not in citation.page_numbers
    assert citation.section_heading == "Capital Adequacy"
    assert citation.format_citation() == "[hdfc_annual_report_fy26.pdf, Page 402, Capital Adequacy]"


def test_phase_b8_string_and_dict_inputs() -> None:
    """TEST: verify accepts string and dictionary inputs in addition to GroundedAnswer."""
    item1 = _make_evidence_item("E1", "Deposits grew 15%.", "c1", [50], "Deposits")
    ctx = _make_context([item1])

    # String input
    res_str = verify_citations("Deposits grew 15% [E1].", ctx)
    assert res_str.answer_text == "Deposits grew 15% [1]."
    assert len(res_str.citations) == 1

    # Dict input
    res_dict = verify_citations(
        {"answer_text": "Deposits grew 15% [E1].", "cited_evidence_ids": ["E1"]},
        ctx,
    )
    assert res_dict.answer_text == "Deposits grew 15% [1]."
    assert len(res_dict.citations) == 1


class TestNearestHeadingResolution:
    """Citations use the nearest heading above the content, not the chunk's first heading."""

    def test_table_chunk_uses_section_heading(self) -> None:
        table_text = (
            "| Schedule | FY26 |\n|---|---|\n"
            "| I Claims | 50,640,067 |\n| Total | 34,179,458,972 |"
        )
        item = _make_evidence_item(
            "E1", table_text, "c1", [389],
            "SCHEDULE  12 - CONTINGENT LIABILITIES", chunk_type="table",
        )
        # Total row and table rows must never be mistaken for headings
        assert resolve_citation_heading(item) == "SCHEDULE  12 - CONTINGENT LIABILITIES"

    def test_mid_chunk_schedule_line_wins_over_first_heading(self) -> None:
        text = (
            "SCHEDULE 11  -  OTHER ASSETS\n\n"
            "Interest accrued was 34,741.20.\n\n"
            "SCHEDULE  12 - CONTINGENT LIABILITIES\n\n"
            "Claims against the bank not acknowledged as debts - taxation 50,640,067."
        )
        item = _make_evidence_item("E1", text, "c1", [508], "SCHEDULE 11  -  OTHER ASSETS")
        assert resolve_citation_heading(item) == "SCHEDULE  12 - CONTINGENT LIABILITIES"

    def test_no_heading_lines_falls_back_to_section(self) -> None:
        item = _make_evidence_item("E1", "Deposits grew 15% year over year.", "c1", [50], "Deposits")
        assert resolve_citation_heading(item) == "Deposits"

    def test_verify_uses_nearest_heading_in_citation(self) -> None:
        text = (
            "SCHEDULE 11  -  OTHER ASSETS\n\n"
            "SCHEDULE  12 - CONTINGENT LIABILITIES\n\n"
            "Total 34,179,458,972."
        )
        item = _make_evidence_item("E1", text, "c1", [389], "SCHEDULE 11  -  OTHER ASSETS")
        ctx = _make_context([item])
        answer = GroundedAnswer(
            answer_text="Contingent liabilities totalled 34,179,458,972 [E1].",
            cited_evidence_ids=["E1"],
        )
        res = CitationVerifier().verify(answer, ctx)
        assert res.citations[0].section_heading == "SCHEDULE  12 - CONTINGENT LIABILITIES"


def _make_table_item(
    evidence_id: str,
    text: str,
    chunk_id: str = "chunk_t",
    pages: list[int] | None = None,
    section: str | None = "SCHEDULE  12 - CONTINGENT LIABILITIES",
    caption: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        text=text,
        page_numbers=pages or [389],
        section_heading=section,
        heading_path=["Financial Statements", section] if section else [],
        chunk_type="table",
        company="HDFC Bank",
        document_name="hdfc_annual_report_fy26.pdf",
        financial_period="FY2025-26",
        source_element_ids=[f"elem_{chunk_id}"],
        table_metadata={"caption": caption} if caption else None,
        rerank_score=0.92,
        retrieval_score=0.81,
    )


class TestNumericGroundingTripwire:
    """B8 headline-figure check: numbers must sit under subject-matching labels."""

    QUESTION_Q06 = "What was the total amount of contingent liabilities as of March 31, 2026?"

    def test_misattached_caption_total_raises_warning(self) -> None:
        """Reconstructed old-Q06 shape: 248,623.68 only in caption above Schedule-12 rows."""
        text = (
            "[Table: Total 248,623.68 215,800.53]\n\n"
            "| Schedule | FY26 |\n|---|---|\n"
            "| I Claims - taxation | 8,265.49 |\n"
            "| VIII Other items | 11,701.60 |"
        )
        item = _make_table_item("E1", text, "c_old", [508], caption="Total 248,623.68 215,800.53")
        ctx = _make_context([item], self.QUESTION_Q06, self.QUESTION_Q06)
        answer = GroundedAnswer(
            answer_text="Total contingent liabilities were 248,623.68 [E1].",
            cited_evidence_ids=["E1"],
        )
        res = CitationVerifier().verify(answer, ctx)
        # Citation itself still verifies (provenance intact) but tripwire fires
        assert len(res.citations) == 1
        assert len(res.grounding_warnings) >= 1
        assert any("248,623.68" in w for w in res.grounding_warnings)

    def test_correct_total_row_is_silent(self) -> None:
        """Fixed chunk: Total row inside its own Schedule 12 table raises no warning."""
        text = (
            "| Schedule | FY26 |\n|---|---|\n"
            "| I Claims - taxation | 50,640,067 |\n"
            "| Total | 34,179,458,972 |"
        )
        item = _make_table_item("E1", text, "c_new", [389])
        ctx = _make_context([item], self.QUESTION_Q06, self.QUESTION_Q06)
        answer = GroundedAnswer(
            answer_text="Total contingent liabilities were 34,179,458,972 [E1].",
            cited_evidence_ids=["E1"],
        )
        res = CitationVerifier().verify(answer, ctx)
        assert len(res.citations) == 1
        assert res.grounding_warnings == []

    def test_calculated_figures_skipped(self) -> None:
        """Derived (Calculated: ...) numbers need not appear verbatim in evidence."""
        text = "| Schedule | FY26 |\n|---|---|\n| Total | 34,179,458,972 |"
        item = _make_table_item("E1", text, "c_new", [389])
        ctx = _make_context([item], self.QUESTION_Q06, self.QUESTION_Q06)
        answer = GroundedAnswer(
            answer_text="Growth was 10.9% (Calculated: (74671.3 - 67347.4) / 67347.4 from [E1]) [E1].",
            cited_evidence_ids=["E1"],
        )
        res = CitationVerifier().verify(answer, ctx)
        assert res.grounding_warnings == []

    def test_number_absent_from_evidence_raises_warning(self) -> None:
        """A cited number nowhere in the chunk text fails content support and is flagged unverified."""
        item = _make_evidence_item("E1", "Deposits grew steadily.", "c1", [50], "Deposits")
        ctx = _make_context(
            [item],
            "What was the bank's total deposit base as of March 31, 2026?",
            "What was the bank's total deposit base as of March 31, 2026?",
        )
        answer = GroundedAnswer(
            answer_text="Deposits totalled 9,999,999 [E1].",
            cited_evidence_ids=["E1"],
        )
        res = CitationVerifier().verify(answer, ctx)
        assert len(res.citations) == 0
        assert res.unverified_citations == ["E1"]
        assert "[Unverified: E1]" in res.answer_text
        assert any("9,999,999" in w for w in res.grounding_warnings)

    def test_footnote_label_not_used_as_heading(self) -> None:
        """A 'Footnotes:' line inside chunk text never becomes the citation heading."""
        text = (
            "| Schedule | FY26 |\n|---|---|\n"
            "| A (i) Bills purchased | 491,258,030 |\n"
            "Footnotes:\n- Figures in brackets relate to previous year."
        )
        item = _make_table_item("E1", text, "c1", [386], section="SCHEDULE  9  -  ADVANCES")
        assert resolve_citation_heading(item) == "SCHEDULE  9  -  ADVANCES"

    def test_multi_citation_sentence_attributes_numbers_to_preceding_token(self) -> None:
        """Q06 shape: each number is checked against its own token's chunk only."""
        standalone_text = "| Schedule | FY26 |\n|---|---|\n| Total | 34,179,458,972 |"
        consolidated_text = "| Schedule | FY26 |\n|---|---|\n| Total | 3,479,487.31 |"
        item1 = _make_table_item("E1", standalone_text, "c_sa", [389])
        item2 = _make_table_item("E2", consolidated_text, "c_con", [508])
        ctx = _make_context(
            [item1, item2],
            TestNumericGroundingTripwire.QUESTION_Q06,
            TestNumericGroundingTripwire.QUESTION_Q06,
        )
        answer = GroundedAnswer(
            answer_text=(
                "Contingent liabilities were 34,179,458,972 on a standalone basis [E1] "
                "and 3,479,487.31 crore on a consolidated basis [E2]."
            ),
            cited_evidence_ids=["E1", "E2"],
        )
        res = CitationVerifier().verify(answer, ctx)
        assert res.grounding_warnings == []

    def test_date_fragments_and_small_bare_numbers_skipped(self) -> None:
        """FY2025-26 fragments and small bare counts are not headline figures."""
        item = _make_evidence_item("E1", "The board met 17 times in FY2025-26.", "c1", [41], "Board")
        ctx = _make_context([item], "How many meetings were held?", "How many meetings were held?")
        answer = GroundedAnswer(
            answer_text="The board met 17 times in FY2025-26 [E1].",
            cited_evidence_ids=["E1"],
        )
        res = CitationVerifier().verify(answer, ctx)
        assert res.grounding_warnings == []


class TestClaimSupportCheck:
    """B8 content-support gate: citations require claim text in the chunk, not just chunk existence."""

    def test_unsupported_entity_names_flagged_unverified(self) -> None:
        """Q10 shape: firm names absent from the cited chunk fail verification."""
        item = _make_evidence_item(
            "E1",
            "We have jointly audited the Standalone Financial Statements of the Bank.",
            "c1",
            [369],
            "Opinion",
        )
        ctx = _make_context([item])
        answer = GroundedAnswer(
            answer_text="The joint statutory auditors were B S R & Co. LLP and Batliboi & Purohit [E1].",
            cited_evidence_ids=["E1"],
        )
        res = CitationVerifier().verify(answer, ctx)
        assert res.citations == []
        assert res.unverified_citations == ["E1"]
        assert "[Unverified: E1]" in res.answer_text
        assert any("B S R" in w or "Batliboi" in w for w in res.grounding_warnings)

    def test_supported_entities_and_numbers_verify(self) -> None:
        """Names and numbers present verbatim in the chunk verify normally."""
        item = _make_evidence_item(
            "E1",
            "For B S R & Co. LLP and Batliboi & Purohit, audit fees were 50,000.",
            "c1",
            [378],
            "Auditors",
        )
        ctx = _make_context(
            [item],
            "Who were the joint statutory auditors and what were their fees?",
            "Who were the joint statutory auditors and what were their fees?",
        )
        answer = GroundedAnswer(
            answer_text="Auditors B S R & Co. LLP and Batliboi & Purohit charged 50,000 [E1].",
            cited_evidence_ids=["E1"],
        )
        res = CitationVerifier().verify(answer, ctx)
        assert len(res.citations) == 1
        assert res.unverified_citations == []
        assert res.grounding_warnings == []

    def test_claim_without_anchors_passes(self) -> None:
        """Pure-prose claims with no numbers or named entities cannot be anchor-checked."""
        item = _make_evidence_item("E1", "The framework was reviewed.", "c1", [100], "Risk")
        ctx = _make_context([item])
        answer = GroundedAnswer(
            answer_text="The framework was reviewed [E1].",
            cited_evidence_ids=["E1"],
        )
        res = CitationVerifier().verify(answer, ctx)
        assert len(res.citations) == 1

    def test_bracket_annotations_ignored_for_anchors(self) -> None:
        """[NIL/NOT DISCLOSED] markers must not seed false entity anchors."""
        item = _make_evidence_item("E1", "The value was reported.", "c1", [100], "Notes")
        ctx = _make_context([item])
        answer = GroundedAnswer(
            answer_text="The value was [NIL/NOT DISCLOSED] as reported [E1].",
            cited_evidence_ids=["E1"],
        )
        res = CitationVerifier().verify(answer, ctx)
        assert len(res.citations) == 1
        assert res.unverified_citations == []

    def test_abbreviation_spanning_entities_all_checked(self) -> None:
        """Periods inside entity names must not fracture the claim into sentences."""
        item = _make_evidence_item("E1", "The audit was completed.", "c1", [100], "Notes")
        ctx = _make_context([item])
        answer = GroundedAnswer(
            answer_text="Auditors B S R & Co. LLP and Batliboi & Purohit signed [E1].",
            cited_evidence_ids=["E1"],
        )
        res = CitationVerifier().verify(answer, ctx)
        assert res.citations == []
        assert res.unverified_citations == ["E1"]
        warned = " ".join(res.grounding_warnings)
        assert "B S R" in warned and "Batliboi" in warned

    def test_inline_equation_result_exempt_inputs_checked(self) -> None:
        """An explicit `A - B = C` derivation exempts the result; inputs must be present."""
        item = _make_evidence_item(
            "E1", "Interest earned 3,075,220,673. Interest expended 1,788,360,400.",
            "c1", [379], "P&L",
        )
        ctx = _make_context([item])
        answer = GroundedAnswer(
            answer_text="The difference is 3,075,220,673 - 1,788,360,400 = 1,286,860,273 [E1].",
            cited_evidence_ids=["E1"],
        )
        res = CitationVerifier().verify(answer, ctx)
        assert len(res.citations) == 1
        assert res.unverified_citations == []

