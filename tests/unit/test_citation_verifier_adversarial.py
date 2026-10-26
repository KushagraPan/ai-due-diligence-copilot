"""
Adversarial test suite for Phase B8 Citation Verification and Attribution.

Exhaustively verifies all 18 security, grounding, and provenance boundary cases:
1. Valid [E1]
2. Invalid [E99]
3. Mixed [E1, E99]
4. Malformed [EX], [E], [Eabc], [E_1]
5. Repeated [E1] appearing multiple times
6. Grouped [E1, E2]
7. "[Calculated: ...]" must not be mistaken for a citation
8. "[NIL/NOT DISCLOSED]" must not be mistaken for a citation
9. LLM text containing a fabricated "Page 999"
10. Missing page_numbers
11. Empty answer
12. Whitespace-only answer
13. Answer with no citations
14. Refusal/uncertain answer
15. Evidence ID existing in text but absent from evidence_map
16. Evidence map containing E1 but answer referencing E2
17. Multiple citations referring to the same underlying chunk
18. Verify that B8 never derives citation metadata from LLM prose.
"""

from due_diligence.agents.citation_verifier import CitationVerifier
from due_diligence.agents.generator import GroundedAnswer
from due_diligence.retrieval.schemas import EvidenceContext, EvidenceItem


def _make_item(
    evidence_id: str,
    text: str = "Quarterly net profit rose 12.5% to ₹16,821 crore.",
    chunk_id: str = "chunk_p378_c0",
    page_numbers: list[int] | None = None,
    section_heading: str | None = "Financial Performance",
    document_name: str = "hdfc_annual_report_fy26.pdf",
) -> EvidenceItem:
    """Helper to assemble a typed EvidenceItem."""
    return EvidenceItem(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        text=text,
        page_numbers=page_numbers if page_numbers is not None else [378],
        section_heading=section_heading,
        heading_path=["Financial Statements", section_heading] if section_heading else [],
        chunk_type="narrative",
        company="HDFC Bank",
        document_name=document_name,
        financial_period="FY2025-26",
        source_element_ids=[f"elem_{chunk_id}"],
        rerank_score=0.95,
        retrieval_score=0.82,
    )


def _make_context(items: list[EvidenceItem]) -> EvidenceContext:
    """Helper to assemble a typed EvidenceContext."""
    return EvidenceContext(
        original_query="What was the net profit and CET1 ratio?",
        retrieval_query="What was the net profit and CET1 ratio?",
        evidence_items=items,
        formatted_context="<diligence_context>context_data</diligence_context>",
        evidence_map={item.evidence_id: item for item in items},
        omitted_items=[],
        total_context_chars=100,
    )


# ── Case 1: Valid [E1] ──────────────────────────────────────────────────────


def test_adv_01_valid_single_e1() -> None:
    """Case 1: Standard single valid [E1] translates to numbered footnote [1]."""
    item1 = _make_item(
        "E1",
        text="Total debt was ₹50,000 Cr.",
        page_numbers=[42],
        section_heading="Note 7: Debt",
    )
    ctx = _make_context([item1])
    answer = GroundedAnswer(
        answer_text="Total debt was ₹50,000 Cr [E1].",
        cited_evidence_ids=["E1"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert res.answer_text == "Total debt was ₹50,000 Cr [1]."
    assert len(res.citations) == 1
    assert res.citations[0].citation_number == 1
    assert res.citations[0].evidence_id == "E1"
    assert res.citations[0].page_numbers == [42]
    assert res.citations[0].document_name == "hdfc_annual_report_fy26.pdf"
    assert res.citations[0].section_heading == "Note 7: Debt"
    assert res.citations[0].format_citation() == "[hdfc_annual_report_fy26.pdf, Page 42, Note 7: Debt]"
    assert res.unverified_citations == []


# ── Case 2: Invalid [E99] ───────────────────────────────────────────────────


def test_adv_02_invalid_e99() -> None:
    """Case 2: Nonexistent citation [E99] is detected, tagged [Unverified: E99], and omitted from citations."""
    item1 = _make_item("E1")
    ctx = _make_context([item1])
    answer = GroundedAnswer(
        answer_text="Future projected revenue is ₹1,00,000 Cr [E99].",
        cited_evidence_ids=["E99"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert "[Unverified: E99]" in res.answer_text
    assert "[1]" not in res.answer_text
    assert len(res.citations) == 0
    assert res.unverified_citations == ["E99"]


# ── Case 3: Mixed [E1, E99] ─────────────────────────────────────────────────


def test_adv_03_mixed_e1_and_e99() -> None:
    """Case 3: Mixed valid and invalid citations: E1 is resolved to [1], E99 is flagged as unverified."""
    item1 = _make_item(
        "E1",
        text="Deposits grew 15%. Deposits reached ₹23,45,600 Cr.",
        page_numbers=[100],
    )
    ctx = _make_context([item1])

    # Test grouped bracket [E1, E99]
    answer_grouped = GroundedAnswer(
        answer_text="Deposits grew 15% [E1, E99].",
        cited_evidence_ids=["E1", "E99"],
    )
    verifier = CitationVerifier()
    res_grouped = verifier.verify(answer_grouped, ctx)

    assert "[1]" in res_grouped.answer_text
    assert "[Unverified: E99]" in res_grouped.answer_text
    assert len(res_grouped.citations) == 1
    assert res_grouped.citations[0].evidence_id == "E1"
    assert res_grouped.unverified_citations == ["E99"]

    # Test separated brackets [E1] and [E99]
    answer_separated = GroundedAnswer(
        answer_text="Deposits reached ₹23,45,600 Cr [E1], while projected growth was 20% [E99].",
        cited_evidence_ids=["E1", "E99"],
    )
    res_sep = verifier.verify(answer_separated, ctx)
    assert "[1]" in res_sep.answer_text
    assert "[Unverified: E99]" in res_sep.answer_text
    assert len(res_sep.citations) == 1
    assert res_sep.unverified_citations == ["E99"]


# ── Case 4: Malformed [EX], [E], [Eabc], [E_1] ──────────────────────────────


def test_adv_04_malformed_tokens_ex_e_eabc_e_1() -> None:
    """Case 4: Malformed citation tokens [EX], [E], [Eabc], [E_1] are flagged as unverified."""
    item1 = _make_item("E1")
    ctx = _make_context([item1])
    answer = GroundedAnswer(
        answer_text="Valid [E1], placeholder [EX], bare [E], letters [Eabc], underscore [E_1].",
        cited_evidence_ids=["E1"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert "[1]" in res.answer_text
    assert "[Unverified: EX]" in res.answer_text
    assert "[Unverified: E]" in res.answer_text
    assert "[Unverified: Eabc]" in res.answer_text
    assert "[Unverified: E_1]" in res.answer_text
    assert len(res.citations) == 1
    assert set(res.unverified_citations) == {"EX", "E", "Eabc", "E_1"}


# ── Case 5: Repeated [E1] Appearing Multiple Times ──────────────────────────


def test_adv_05_repeated_e1_multiple_times() -> None:
    """Case 5: Repeated mentions of [E1] across sentences map to the identical footnote number [1]."""
    item1 = _make_item(
        "E1",
        text="CET1 was 16.4%. Minimum requirement is 8.0%. Buffer was 8.4%.",
        page_numbers=[402],
        section_heading="Capital Adequacy",
    )
    ctx = _make_context([item1])
    answer = GroundedAnswer(
        answer_text="CET1 was 16.4% [E1]. Minimum requirement is 8.0% [E1]. Buffer was 8.4% [E1].",
        cited_evidence_ids=["E1"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    expected = "CET1 was 16.4% [1]. Minimum requirement is 8.0% [1]. Buffer was 8.4% [1]."
    assert res.answer_text == expected
    assert len(res.citations) == 1
    assert res.citations[0].citation_number == 1
    assert res.unverified_citations == []


# ── Case 6: Grouped [E1, E2] ────────────────────────────────────────────────


def test_adv_06_grouped_e1_e2() -> None:
    """Case 6: Grouped bracketed citations [E1, E2] are resolved into [1] [2]."""
    item1 = _make_item("E1", page_numbers=[10])
    item2 = _make_item("E2", page_numbers=[20])
    ctx = _make_context([item1, item2])
    answer = GroundedAnswer(
        answer_text="Both statements were confirmed by filings [E1, E2].",
        cited_evidence_ids=["E1", "E2"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert res.answer_text == "Both statements were confirmed by filings [1] [2]."
    assert len(res.citations) == 2
    assert res.citations[0].citation_number == 1
    assert res.citations[1].citation_number == 2
    assert res.unverified_citations == []


# ── Case 7: "[Calculated: ...]" Must Not Be Mistaken for Citation ───────────


def test_adv_07_calculated_bracket_not_mistaken_for_citation() -> None:
    """Case 7: Analytical calculation brackets [Calculated: ...] must not be flagged or replaced."""
    item1 = _make_item("E1", page_numbers=[50])
    ctx = _make_context([item1])
    answer = GroundedAnswer(
        answer_text="Operating margin expanded 150 bps [Calculated: (24.5% - 23.0%)] [E1].",
        cited_evidence_ids=["E1"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert "[Calculated: (24.5% - 23.0%)]" in res.answer_text
    assert "[1]" in res.answer_text
    assert len(res.citations) == 1
    assert res.unverified_citations == []


# ── Case 8: "[NIL/NOT DISCLOSED]" Must Not Be Mistaken for Citation ─────────


def test_adv_08_nil_not_disclosed_bracket_not_mistaken() -> None:
    """Case 8: Special symbol font placeholder [NIL/NOT DISCLOSED] must remain intact."""
    item1 = _make_item("E1")
    ctx = _make_context([item1])
    answer = GroundedAnswer(
        answer_text="Contingent liabilities were [NIL/NOT DISCLOSED] as of the reporting date [E1].",
        cited_evidence_ids=["E1"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert "[NIL/NOT DISCLOSED]" in res.answer_text
    assert "[1]" in res.answer_text
    assert len(res.citations) == 1
    assert res.unverified_citations == []


# ── Case 9: LLM Text Containing a Fabricated "Page 999" ─────────────────────


def test_adv_09_fabricated_page_999_in_prose() -> None:
    """Case 9: Fabricated 'Page 999' in prose is never adopted; the unsupported claim is flagged."""
    item1 = _make_item(
        "E1",
        text="Deposits stood at ₹23,45,600 Cr.",
        page_numbers=[378],
        section_heading="Balance Sheet",
    )
    ctx = _make_context([item1])
    answer = GroundedAnswer(
        answer_text="According to Page 999 of the report, deposits stood at ₹23,45,600 Cr [E1].",
        cited_evidence_ids=["E1"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    # The fabricated page number is unsupported by evidence, so the citation is
    # flagged rather than verified — and no citation metadata ever adopts it.
    assert res.citations == []
    assert res.unverified_citations == ["E1"]
    assert "[Unverified: E1]" in res.answer_text
    for cit in res.citations:
        assert 999 not in cit.page_numbers


# ── Case 10: Missing page_numbers ───────────────────────────────────────────


def test_adv_10_missing_page_numbers_reports_unknown() -> None:
    """Case 10: Evidence chunk with empty page_numbers reports 'Page: Unknown' per AGENTS.md §4.5."""
    item_no_page = _make_item("E1", page_numbers=[], section_heading="Schedule 12")
    ctx = _make_context([item_no_page])
    answer = GroundedAnswer(
        answer_text="Contingent liabilities were reported [E1].",
        cited_evidence_ids=["E1"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    citation = res.citations[0]
    assert citation.page_numbers == []
    assert citation.format_citation() == "[hdfc_annual_report_fy26.pdf, Page: Unknown, Schedule 12]"


# ── Case 11: Empty Answer ───────────────────────────────────────────────────


def test_adv_11_empty_answer() -> None:
    """Case 11: Empty answer string is handled cleanly without exceptions."""
    item1 = _make_item("E1")
    ctx = _make_context([item1])

    verifier = CitationVerifier()

    res_obj = verifier.verify(GroundedAnswer(answer_text="", cited_evidence_ids=[]), ctx)
    assert res_obj.answer_text == ""
    assert len(res_obj.citations) == 0
    assert res_obj.is_uncertain is True

    res_str = verifier.verify("", ctx)
    assert res_str.answer_text == ""
    assert len(res_str.citations) == 0
    assert res_str.is_uncertain is True


# ── Case 12: Whitespace-Only Answer ─────────────────────────────────────────


def test_adv_12_whitespace_only_answer() -> None:
    """Case 12: Whitespace-only answer text handled safely with is_uncertain=True."""
    item1 = _make_item("E1")
    ctx = _make_context([item1])

    verifier = CitationVerifier()
    res = verifier.verify("   \n\t  ", ctx)

    assert len(res.citations) == 0
    assert res.is_uncertain is True


# ── Case 13: Answer with No Citations ───────────────────────────────────────


def test_adv_13_answer_with_no_citations() -> None:
    """Case 13: Prose answer without citations returns unmodified text and empty citations list."""
    item1 = _make_item("E1")
    ctx = _make_context([item1])
    raw = "The bank management discussed economic headwinds in the global economy."
    answer = GroundedAnswer(
        answer_text=raw,
        cited_evidence_ids=[],
        is_uncertain=False,
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert res.answer_text == raw
    assert len(res.citations) == 0
    assert res.unverified_citations == []


# ── Case 14: Refusal / Uncertain Answer ─────────────────────────────────────


def test_adv_14_refusal_uncertain_answer() -> None:
    """Case 14: Refusal answer declaring absent information preserves is_uncertain=True."""
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
    assert res.is_uncertain is True


# ── Case 15: Evidence ID in Text but Absent from evidence_map ────────────────


def test_adv_15_evidence_id_in_text_absent_from_evidence_map() -> None:
    """Case 15: Evidence token [E5] in text is absent from context evidence_map containing only E1-E3."""
    item1 = _make_item("E1")
    item2 = _make_item("E2")
    item3 = _make_item("E3")
    ctx = _make_context([item1, item2, item3])

    answer = GroundedAnswer(
        answer_text="Known fact [E1] and unverified claim [E5].",
        cited_evidence_ids=["E1"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert "[1]" in res.answer_text
    assert "[Unverified: E5]" in res.answer_text
    assert len(res.citations) == 1
    assert res.unverified_citations == ["E5"]


# ── Case 16: Evidence Map Has E1 but Answer References E2 ───────────────────


def test_adv_16_evidence_map_has_e1_answer_references_e2() -> None:
    """Case 16: Context only contains E1, but answer cites [E2]."""
    item1 = _make_item("E1")
    ctx = _make_context([item1])

    answer = GroundedAnswer(
        answer_text="The corporate tax rate was 25% [E2].",
        cited_evidence_ids=["E2"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert "[Unverified: E2]" in res.answer_text
    assert len(res.citations) == 0
    assert res.unverified_citations == ["E2"]


# ── Case 17: Multiple Citations Referring to the Same Underlying Chunk ───────


def test_adv_17_multiple_citations_same_underlying_chunk() -> None:
    """Case 17: Multiple distinct evidence items pointing to the same underlying chunk."""
    common_chunk_id = "hdfc_chunk_p100_shared"
    item1 = _make_item("E1", text="Operating revenue rose.", chunk_id=common_chunk_id, page_numbers=[100])
    item2 = _make_item("E2", text="Net income rose.", chunk_id=common_chunk_id, page_numbers=[100])
    ctx = _make_context([item1, item2])

    answer = GroundedAnswer(
        answer_text="Operating revenue grew [E1], and net income rose [E2].",
        cited_evidence_ids=["E1", "E2"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    assert res.answer_text == "Operating revenue grew [1], and net income rose [2]."
    assert len(res.citations) == 2
    assert res.citations[0].citation_number == 1
    assert res.citations[0].chunk_id == common_chunk_id
    assert res.citations[1].citation_number == 2
    assert res.citations[1].chunk_id == common_chunk_id
    assert res.unverified_citations == []


# ── Case 18: B8 Never Derives Citation Metadata from LLM Prose ───────────────


def test_adv_18_never_derives_metadata_from_llm_prose() -> None:
    """
    Case 18: Critical security check: Even when LLM prose contains extensive fake coordinates
    (fake document name, fake pages, fake section headings), B8 never presents the claim
    as verified and never derives citation metadata from prose. Under the content-support
    contract, the unsupported claim is flagged unverified.
    """
    real_doc = "canonical_hdfc_report_fy2026.pdf"
    real_pages = [378, 379]
    real_section = "Standalone Balance Sheet and Notes"
    real_chunk = "hdfc_c378_real_chunk_id"

    item = _make_item(
        evidence_id="E1",
        text="Total deposits were ₹23,45,600 Cr.",
        chunk_id=real_chunk,
        page_numbers=real_pages,
        section_heading=real_section,
        document_name=real_doc,
    )
    ctx = _make_context([item])

    adversarial_prose = (
        "In file 'competitor_deck_2020.pdf' on Page 777 and Page 888 under "
        "Section 'Fabricated Overview', deposits were ₹23,45,600 Cr [E1]."
    )
    answer = GroundedAnswer(
        answer_text=adversarial_prose,
        cited_evidence_ids=["E1"],
    )

    verifier = CitationVerifier()
    res = verifier.verify(answer, ctx)

    # The fabricated coordinates/numbers are unsupported by evidence, so no
    # verified citation is issued — and no fake metadata leaks into any citation.
    assert len(res.citations) == 0
    assert res.unverified_citations == ["E1"]
    assert "[Unverified: E1]" in res.answer_text
