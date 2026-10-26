"""
Unit tests for entity_scope resolution and filtering module.

Tests deterministic scope assignment and parent vs subsidiary query filtering.
"""


from due_diligence.core.entity_scope import (
    SCOPE_HDB_FINANCIAL_SERVICES,
    SCOPE_HDFC_AMC,
    SCOPE_HDFC_BANK_CONSOLIDATED,
    SCOPE_HDFC_BANK_GENERAL,
    SCOPE_HDFC_BANK_STANDALONE,
    SCOPE_HDFC_ERGO,
    SCOPE_HDFC_LIFE,
    SCOPE_HDFC_SECURITIES,
    filter_subsidiary_candidates,
    is_subsidiary_scope,
    query_targets_subsidiary,
    resolve_entity_scope,
)
from due_diligence.retrieval.schemas import RetrievalResult


def _make_dummy_candidate(
    chunk_id: str,
    scope: str | None = None,
    heading_path: list[str] | None = None,
    section_heading: str | None = None,
    text: str = "Dummy text",
    page_numbers: list[int] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=0.85,
        company="HDFC Bank Limited",
        document_name="annual_report.pdf",
        document_type="pdf",
        page_numbers=page_numbers or [100],
        section_heading=section_heading,
        heading_path=heading_path or [],
        chunk_type="narrative",
        source_element_ids=["elem1"],
        entity_scope=scope or "hdfc_bank_general",
    )


class TestEntityScopeResolution:
    """Test resolution heuristics from headings, narrative text, and page ranges."""

    def test_resolve_subsidiary_from_heading(self) -> None:
        assert (
            resolve_entity_scope(
                heading_path=["Statutory Reports", "Performance of Subsidiary Companies"],
                section_heading="HDB Financial Services Limited (HDBFSL)",
            )
            == SCOPE_HDB_FINANCIAL_SERVICES
        )
        assert (
            resolve_entity_scope(
                section_heading="HDFC Life Insurance Company Limited (HDFC Life)",
            )
            == SCOPE_HDFC_LIFE
        )
        assert (
            resolve_entity_scope(
                section_heading="HDFC ERGO General Insurance Company Limited",
            )
            == SCOPE_HDFC_ERGO
        )
        assert (
            resolve_entity_scope(
                section_heading="HDFC Securities Limited",
            )
            == SCOPE_HDFC_SECURITIES
        )
        assert (
            resolve_entity_scope(
                section_heading="HDFC Asset Management Company Limited (HDFC AMC)",
            )
            == SCOPE_HDFC_AMC
        )

    def test_resolve_standalone_and_consolidated_financials(self) -> None:
        assert (
            resolve_entity_scope(
                section_heading="SCHEDULES TO THE STANDALONE FINANCIAL STATEMENTS",
            )
            == SCOPE_HDFC_BANK_STANDALONE
        )
        assert (
            resolve_entity_scope(
                section_heading="SCHEDULES TO THE CONSOLIDATED FINANCIAL STATEMENTS",
            )
            == SCOPE_HDFC_BANK_CONSOLIDATED
        )
        # By physical page range
        assert (
            resolve_entity_scope(page_numbers=[378])
            == SCOPE_HDFC_BANK_STANDALONE
        )
        assert (
            resolve_entity_scope(page_numbers=[500])
            == SCOPE_HDFC_BANK_CONSOLIDATED
        )

    def test_resolve_subsidiary_from_page_305_narrative(self) -> None:
        text_hdb = (
            "The company's Profit After Tax stood at `2,544 crore as compared to `2,176 crore. "
            "The Total Loan Book stood at `1,18,493 crore as on March 31, 2026. HDBFSL is a leading NBFC."
        )
        assert (
            resolve_entity_scope(
                page_numbers=[305],
                text=text_hdb,
            )
            == SCOPE_HDB_FINANCIAL_SERVICES
        )

    def test_resolve_fallback_to_general(self) -> None:
        assert (
            resolve_entity_scope(
                heading_path=["Overview", "Corporate Profile"],
                section_heading="Board of Directors",
                page_numbers=[50],
            )
            == SCOPE_HDFC_BANK_GENERAL
        )

    def test_resolve_consolidated_overrides_leaked_standalone_breadcrumb(self) -> None:
        """Physical pages 494-580 are consolidated even if heading_path carries standalone breadcrumbs."""
        assert (
            resolve_entity_scope(
                heading_path=["SCHEDULES TO THE STANDALONE FINANCIAL STATEMENTS"],
                section_heading="18. Earnings per equity share",
                page_numbers=[571],
            )
            == SCOPE_HDFC_BANK_CONSOLIDATED
        )
        assert (
            resolve_entity_scope(
                heading_path=["SCHEDULES TO THE STANDALONE FINANCIAL STATEMENTS"],
                section_heading="SCHEDULE 12 - CONTINGENT LIABILITIES",
                page_numbers=[508],
            )
            == SCOPE_HDFC_BANK_CONSOLIDATED
        )


class TestEntityScopeFiltering:
    """Test filtering of subsidiary candidates when query targets parent bank."""

    def test_is_subsidiary_scope(self) -> None:
        assert is_subsidiary_scope(SCOPE_HDB_FINANCIAL_SERVICES) is True
        assert is_subsidiary_scope(SCOPE_HDFC_LIFE) is True
        assert is_subsidiary_scope(SCOPE_HDFC_BANK_STANDALONE) is False
        assert is_subsidiary_scope(SCOPE_HDFC_BANK_CONSOLIDATED) is False
        assert is_subsidiary_scope(SCOPE_HDFC_BANK_GENERAL) is False
        assert is_subsidiary_scope(None) is False

    def test_query_targets_subsidiary(self) -> None:
        assert query_targets_subsidiary("What was HDB Financial's loan book?", SCOPE_HDB_FINANCIAL_SERVICES) is True
        assert query_targets_subsidiary("What was HDBFSL profit?", SCOPE_HDB_FINANCIAL_SERVICES) is True
        assert query_targets_subsidiary("What was the bank's net profit?", SCOPE_HDB_FINANCIAL_SERVICES) is False
        assert query_targets_subsidiary("What was HDFC Bank's net profit?", SCOPE_HDB_FINANCIAL_SERVICES) is False

    def test_filter_removes_subsidiary_on_parent_query(self) -> None:
        parent_cand = _make_dummy_candidate("p1", scope=SCOPE_HDFC_BANK_STANDALONE, text="HDFC Bank net profit was 74,671 cr")
        sub_cand = _make_dummy_candidate("s1", scope=SCOPE_HDB_FINANCIAL_SERVICES, text="HDBFSL profit was 2,544 cr")

        query = "What was the percentage growth in net profit for HDFC Bank?"
        filtered = filter_subsidiary_candidates(query, [parent_cand, sub_cand])

        assert len(filtered) == 1
        assert filtered[0].chunk_id == "p1"
        assert filtered[0].entity_scope == SCOPE_HDFC_BANK_STANDALONE

    def test_filter_preserves_subsidiary_when_explicitly_requested(self) -> None:
        parent_cand = _make_dummy_candidate("p1", scope=SCOPE_HDFC_BANK_STANDALONE, text="HDFC Bank profit")
        sub_cand = _make_dummy_candidate("s1", scope=SCOPE_HDB_FINANCIAL_SERVICES, text="HDBFSL profit was 2,544 cr")

        query = "What was HDB Financial Services' profit after tax?"
        filtered = filter_subsidiary_candidates(query, [parent_cand, sub_cand])

        assert len(filtered) == 2
        assert any(c.chunk_id == "s1" for c in filtered)

    def test_filter_all_subsidiaries_fallback(self) -> None:
        sub_cand = _make_dummy_candidate("s1", scope=SCOPE_HDB_FINANCIAL_SERVICES)
        # If pool has only subsidiaries, do not return empty list
        filtered = filter_subsidiary_candidates("What is the net profit?", [sub_cand])
        assert len(filtered) == 1
