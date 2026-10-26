"""
Unit and regression tests for layout reconciliation layer (layout.py).
"""

from pathlib import Path

import pymupdf
import pytest

from due_diligence.ingestion.elements import Heading, NarrativeText, Table
from due_diligence.ingestion.layout import (
    LayoutConfig,
    classify_heading_block,
    classify_list_item_block,
    detect_multi_column_layout,
    is_valid_table_caption,
    propagate_section_headings,
    reconcile_page_elements,
)
from due_diligence.ingestion.table_extractor import extract_tables_from_page

CORPUS_PDF = Path("data/raw/hdfc-bank__annual-report__fy2025-26.pdf")


class TestLayoutReconciliation:
    """Test suite for layout.py spatial partitioning, classification, and reconciliation."""

    @pytest.mark.skipif(not CORPUS_PDF.exists(), reason="Corpus PDF not present on disk")
    def test_character_reconciliation_real_pages(self) -> None:
        """1. Character reconciliation audit on real pages (343 and 80)."""
        doc = pymupdf.open(str(CORPUS_PDF))
        try:
            # Test Page 343
            page343 = doc[342]
            _, rep343, _ = reconcile_page_elements(page343, 343, "hdfc.pdf")
            assert rep343.raw_character_count > 0
            assert rep343.structured_character_count > 0
            assert rep343.reconciliation_delta_pct <= 5.0
            assert rep343.unowned_block_count == 0
            assert rep343.multiply_owned_block_count == 0

            # Test Page 80
            page80 = doc[79]
            _, rep80, _ = reconcile_page_elements(page80, 80, "hdfc.pdf")
            assert rep80.raw_character_count > 0
            assert rep80.structured_character_count > 0
            assert rep80.unowned_block_count == 0
            assert rep80.multiply_owned_block_count == 0
        finally:
            doc.close()

    def test_spatial_table_exclusion(self) -> None:
        """2. Text inside a table bbox must belong to the table and not become NarrativeText."""
        table_bbox = (40.0, 100.0, 550.0, 300.0)
        table = Table(
            element_id="test_tab0",
            element_type="table",
            page_number=1,
            bbox=table_bbox,
            document_name="test.pdf",
            headers=["Name", "Amount"],
            rows=[["Revenue", "100"], ["Net Profit", "20"]],
            markdown="| Name | Amount |\n|---|---|\n| Revenue | 100 |\n| Net Profit | 20 |",
        )

        # Create a synthetic PyMuPDF document with text inside and outside the table bbox
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        # Narrative before table
        page.insert_text((50, 70), "Introductory paragraph above table.")
        # Table text inside table bbox
        page.insert_text((50, 150), "Revenue 100")
        page.insert_text((50, 200), "Net Profit 20")
        # Narrative after table
        page.insert_text((50, 350), "Concluding paragraph below table.")

        elems, report, _ = reconcile_page_elements(page, 1, "test.pdf", tables=[table])

        narratives = [e for e in elems if isinstance(e, NarrativeText) and not e.metadata.get("is_header_footer")]
        narr_texts = [n.text for n in narratives]

        # The table cell text must NOT appear in narrative blocks
        for t in narr_texts:
            assert "Revenue 100" not in t
            assert "Net Profit 20" not in t

        # Narratives outside must be present
        assert any("Introductory paragraph" in t for t in narr_texts)
        assert any("Concluding paragraph" in t for t in narr_texts)
        assert report.unowned_block_count == 0
        assert report.multiply_owned_block_count == 0

    @pytest.mark.skipif(not CORPUS_PDF.exists(), reason="Corpus PDF not present on disk")
    def test_multi_column_detection_real_and_synthetic(self) -> None:
        """3. Multi-column detection on real corpus pages (Page 80 is 2-col, Page 343 is 1-col)."""
        doc = pymupdf.open(str(CORPUS_PDF))
        try:
            cfg = LayoutConfig()

            # Page 80 is 2-column
            page80 = doc[79]
            pdict80 = page80.get_text("dict")
            blocks80 = [b for b in pdict80.get("blocks", []) if b.get("type") == 0]
            is_mc80, cols80, xr80 = detect_multi_column_layout(blocks80, page80.rect.width, page80.rect.height, [], cfg)
            assert is_mc80 is True
            assert cols80 == 2
            assert len(xr80) == 2

            # Page 343 is 1-column
            page343 = doc[342]
            pdict343 = page343.get_text("dict")
            blocks343 = [b for b in pdict343.get("blocks", []) if b.get("type") == 0]
            is_mc343, cols343, _ = detect_multi_column_layout(blocks343, page343.rect.width, page343.rect.height, [], cfg)
            assert is_mc343 is False
            assert cols343 == 1
        finally:
            doc.close()

    def test_heading_false_positive_rejection(self) -> None:
        """4. Bold inline emphasis inside a paragraph must NOT become a Heading."""
        cfg = LayoutConfig()
        # Synthetic block with bold emphasized word inside a sentence
        paragraph_block = {
            "bbox": (50.0, 100.0, 500.0, 140.0),
            "lines": [
                {
                    "bbox": (50.0, 100.0, 500.0, 115.0),
                    "spans": [
                        {"text": "The Bank has experienced ", "size": 9.0, "font": "Arial", "flags": 4},
                        {"text": "SIGNIFICANT ", "size": 11.0, "font": "Arial-Bold", "flags": 20},
                        {"text": "growth across retail segments.", "size": 9.0, "font": "Arial", "flags": 4},
                    ],
                },
                {
                    "bbox": (50.0, 120.0, 480.0, 135.0),
                    "spans": [
                        {"text": "This performance was driven by digital initiatives.", "size": 9.0, "font": "Arial", "flags": 4}
                    ],
                },
            ],
        }
        is_h, _, _ = classify_heading_block(paragraph_block, median_body_size=9.0, page_h=842.0, config=cfg)
        assert is_h is False

    def test_heading_detection_legitimate(self) -> None:
        """5. Legitimate standalone heading must be classified as Heading with correct level."""
        cfg = LayoutConfig()
        heading_block = {
            "bbox": (50.0, 100.0, 350.0, 120.0),
            "lines": [
                {
                    "bbox": (50.0, 100.0, 350.0, 120.0),
                    "spans": [
                        {
                            "text": "Note 18: Contingent Liabilities",
                            "size": 13.0,
                            "font": "Helvetica-Bold",
                            "flags": 20,
                        }
                    ],
                }
            ],
        }
        is_h, lvl, text = classify_heading_block(heading_block, median_body_size=9.0, page_h=842.0, config=cfg)
        assert is_h is True
        assert lvl <= 3
        assert "Note 18" in text

    def test_list_item_detection(self) -> None:
        """6. Bullet and numbered items become ListItem, while normal prose does not."""
        # Bullet item
        is_li1, m1, t1 = classify_list_item_block("• Robust capital adequacy ratio of 19.3%.")
        assert is_li1 is True
        assert m1 == "•"
        assert t1 == "Robust capital adequacy ratio of 19.3%."

        # Numbered item
        is_li2, m2, t2 = classify_list_item_block("1. Approval of financial statements.")
        assert is_li2 is True
        assert m2 == "1."
        assert t2 == "Approval of financial statements."

        # Sub-clause marker
        is_li3, m3, _ = classify_list_item_block("(a) Applicable statutory disclosures.")
        assert is_li3 is True
        assert m3 == "(a)"

        # Normal prose without bullet marker
        is_li4, _, _ = classify_list_item_block("Non-performing assets decreased during the quarter.")
        assert is_li4 is False

    def test_section_heading_propagation(self) -> None:
        """7. Hierarchical section heading propagation across H1 -> H2 -> H3 transitions."""
        h1 = Heading(
            element_id="e1", element_type="heading", page_number=1, bbox=(0, 0, 10, 10),
            document_name="doc.pdf", text="Annual Report", level=1,
        )
        h2 = Heading(
            element_id="e2", element_type="heading", page_number=1, bbox=(0, 0, 10, 10),
            document_name="doc.pdf", text="Corporate Governance", level=2,
        )
        h3 = Heading(
            element_id="e3", element_type="heading", page_number=1, bbox=(0, 0, 10, 10),
            document_name="doc.pdf", text="Board of Directors", level=3,
        )
        narr = NarrativeText(
            element_id="e4", element_type="narrative", page_number=1, bbox=(0, 0, 10, 10),
            document_name="doc.pdf", text="The Board consists of 12 directors.",
        )
        h2_new = Heading(
            element_id="e5", element_type="heading", page_number=1, bbox=(0, 0, 10, 10),
            document_name="doc.pdf", text="Financial Statements", level=2,
        )
        narr2 = NarrativeText(
            element_id="e6", element_type="narrative", page_number=1, bbox=(0, 0, 10, 10),
            document_name="doc.pdf", text="Balance sheet as of March 31, 2026.",
        )

        elements = [h1, h2, h3, narr, h2_new, narr2]
        propagated, _ = propagate_section_headings(elements)

        # Check narrative after H3
        p_narr = propagated[3]
        assert p_narr.section_heading == "Board of Directors"
        assert p_narr.metadata["heading_path"] == ["Annual Report", "Corporate Governance", "Board of Directors"]

        # Check narrative after new H2
        p_narr2 = propagated[5]
        assert p_narr2.section_heading == "Financial Statements"
        assert p_narr2.metadata["heading_path"] == ["Annual Report", "Financial Statements"]

    def test_unknown_ambiguous_content_fallback(self) -> None:
        """8. Ambiguous or low-confidence content becomes NarrativeText with review metadata, never dropped."""
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 100), "Ambiguous stray metric: XYZ / 99.4 % ?")

        elems, report, _ = reconcile_page_elements(page, 1, "test.pdf")
        assert len(elems) >= 1
        assert report.unowned_block_count == 0
        assert report.owned_block_count == report.source_block_count

    @pytest.mark.skipif(not CORPUS_PDF.exists(), reason="Corpus PDF not present on disk")
    def test_multiple_tables_on_one_page_page470(self) -> None:
        """9. Multiple tables on one page (Page 470) remain separate and correctly ordered."""
        doc = pymupdf.open(str(CORPUS_PDF))
        try:
            page470 = doc[469]
            tables = extract_tables_from_page(page470, 470, "hdfc.pdf", enable_tier2=False)
            assert len(tables) == 2

            elems, rep470, _ = reconcile_page_elements(page470, 470, "hdfc.pdf", tables=tables)
            assert rep470.table_count == 2

            extracted_tables = [e for e in elems if isinstance(e, Table)]
            assert len(extracted_tables) == 2

            # Table 0: Business segments
            assert extracted_tables[0].caption == "Business segments:"
            assert any("Digital Banking Segment" in fn for fn in extracted_tables[0].footnotes)

            # Table 1: Geographic segments
            assert extracted_tables[1].caption == "Geographic segments:"

            # Tables must be vertically ordered: Table 0 above Table 1
            assert extracted_tables[0].bbox[1] < extracted_tables[1].bbox[1]
        finally:
            doc.close()

    @pytest.mark.skipif(not CORPUS_PDF.exists(), reason="Corpus PDF not present on disk")
    def test_regression_pages_343_and_continuation_380_381(self) -> None:
        """10. Regression testing for Page 343 footnotes and Page 380-381 continuation."""
        doc = pymupdf.open(str(CORPUS_PDF))
        try:
            # Page 343: Board Meeting Attendance table + 3 footnotes
            page343 = doc[342]
            t343 = extract_tables_from_page(page343, 343, "hdfc.pdf", enable_tier2=False)
            elems343, _, _ = reconcile_page_elements(page343, 343, "hdfc.pdf", tables=t343)
            tables343 = [e for e in elems343 if isinstance(e, Table)]
            assert len(tables343) == 1
            assert len(tables343[0].footnotes) == 3
            assert any("maximum number of meetings" in fn for fn in tables343[0].footnotes)
            assert any("Atanu Chakraborty" in fn for fn in tables343[0].footnotes)
            assert any("Bhavesh Zaveri" in fn for fn in tables343[0].footnotes)

            # Pages 380-381: Cash Flow Statement continuation
            p380 = doc[379]
            p381 = doc[380]
            t380 = extract_tables_from_page(p380, 380, "hdfc.pdf", enable_tier2=False)
            t381 = extract_tables_from_page(p381, 381, "hdfc.pdf", prev_page_tables=t380, enable_tier2=False)

            _, _, stack = reconcile_page_elements(p380, 380, "hdfc.pdf", tables=t380)
            elems381, _, _ = reconcile_page_elements(
                p381, 381, "hdfc.pdf", tables=t381, prev_page_tables=t380, initial_heading_stack=stack
            )

            t381_extracted = next(e for e in elems381 if isinstance(e, Table))
            assert t381_extracted.is_continuation is True
            assert t381_extracted.continuation_from_page == 380
        finally:
            doc.close()

    def test_total_row_never_claimed_as_caption(self) -> None:
        """11. Total summary rows and numeric lines are rejected as table captions."""
        assert is_valid_table_caption("Total 248,623.68 215,800.53") is False
        assert is_valid_table_caption("Total 34,179,458,972 26,474,170,018") is False
        assert is_valid_table_caption("34,179,458,972 26,474,170,018") is False
        assert is_valid_table_caption("") is False
        # Genuine captions still accepted
        assert is_valid_table_caption("Business segments:") is True
        assert is_valid_table_caption("Geographic segments:") is True
        assert is_valid_table_caption("(C in '000)") is True

    @pytest.mark.skipif(not CORPUS_PDF.exists(), reason="Corpus PDF not present on disk")
    def test_page508_schedule12_caption_not_schedule11_total(self) -> None:
        """12. Page 508 Schedule 12 caption must not be Schedule 11's Total line."""
        doc = pymupdf.open(str(CORPUS_PDF))
        try:
            page508 = doc[507]
            tables = extract_tables_from_page(page508, 508, "hdfc.pdf", enable_tier2=False)
            assert len(tables) == 2
            elems, _, _ = reconcile_page_elements(page508, 508, "hdfc.pdf", tables=tables)
            extracted = [e for e in elems if isinstance(e, Table)]
            assert len(extracted) == 2
            sched12 = next(e for e in extracted if "CONTINGENT" in (e.section_heading or "").upper()
                           or any("contingently liable" in " ".join(r).lower() for r in e.rows))
            assert sched12.caption is None or "248,623.68" not in sched12.caption
            # Schedule 11's Total now lives in Schedule 11's own rows
            sched11 = next(e for e in extracted if e is not sched12)
            assert any(r[0] == "Total" and "248,623.68" in " ".join(r) for r in sched11.rows)
        finally:
            doc.close()
