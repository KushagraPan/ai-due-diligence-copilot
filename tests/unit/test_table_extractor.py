"""
Unit tests for table extraction engine and quality gate validation.
"""

from pathlib import Path

import pymupdf
import pytest

from due_diligence.ingestion.elements import Table
from due_diligence.ingestion.table_extractor import (
    detect_all_table_regions,
    detect_merged_cell_anomalies,
    evaluate_quality_gate,
    extract_table_tier1,
    extract_tables_from_page,
    stitch_continuation_tables,
)

PDF_PATH = Path("data/raw/hdfc-bank__annual-report__fy2025-26.pdf")


@pytest.fixture(scope="module")
def hdfc_doc():
    if not PDF_PATH.exists():
        pytest.skip(f"Source PDF not present at {PDF_PATH}")
    doc = pymupdf.open(PDF_PATH)
    yield doc
    doc.close()


class TestTableExtractor:
    def test_multi_region_detection_on_page_470(self, hdfc_doc):
        page = hdfc_doc[469]  # Page 470
        boxes = detect_all_table_regions(page)
        assert len(boxes) == 2, f"Expected 2 table regions on Page 470, found {len(boxes)}"
        # Region 0: Business segments (upper half)
        assert boxes[0][1] < 200 and boxes[0][3] > 400
        # Region 1: Geographic segments (lower half)
        assert boxes[1][1] > 450 and boxes[1][3] > 550

    def test_single_region_detection_on_page_378(self, hdfc_doc):
        page = hdfc_doc[377]  # Page 378 (Balance Sheet)
        boxes = detect_all_table_regions(page)
        assert len(boxes) == 1

    def test_tier1_extraction_balance_sheet(self, hdfc_doc):
        page = hdfc_doc[377]  # Page 378
        boxes = detect_all_table_regions(page)
        t_res = extract_table_tier1(page, boxes[0])
        assert t_res["col_count"] == 4
        assert t_res["row_count"] >= 15
        gate_res = evaluate_quality_gate(t_res, page)
        assert gate_res["gate_passed"] is True
        assert gate_res["is_continuation"] is False

    def test_merged_cell_anomaly_detection(self):
        rows = [
            ["Mr. Atanu Chakraborty", "17", "17 Present"],
            ["Mr. M. D. Ranganath", "21", "21 Present"],
        ]
        anomalies = detect_merged_cell_anomalies(rows)
        assert len(anomalies) == 2
        assert "17 Present" in anomalies
        assert "21 Present" in anomalies

    def test_quality_gate_fails_on_merged_cells(self, hdfc_doc):
        page = hdfc_doc[342]  # Page 343 (Board Attendance)
        boxes = detect_all_table_regions(page)
        t_res = extract_table_tier1(page, boxes[0])
        gate_res = evaluate_quality_gate(t_res, page)
        assert gate_res["gate_passed"] is False
        assert any("merged_cell" in f for f in gate_res["failures"])

    def test_continuation_detection_on_page_381(self, hdfc_doc):
        page_380 = hdfc_doc[379]  # Page 380
        page_381 = hdfc_doc[380]  # Page 381
        box_380 = detect_all_table_regions(page_380)[0]
        box_381 = detect_all_table_regions(page_381)[0]

        t_res_380 = extract_table_tier1(page_380, box_380)
        t_res_381 = extract_table_tier1(page_381, box_381)

        gate_res = evaluate_quality_gate(t_res_381, page_381, prev_table=t_res_380)
        assert gate_res["gate_passed"] is False
        assert gate_res["is_continuation"] is True
        assert any("continuation_detected" in f for f in gate_res["failures"])

    def test_stitch_continuation_tables(self):
        t1 = Table(
            element_id="t1",
            element_type="table",
            page_number=380,
            bbox=(40.0, 50.0, 550.0, 700.0),
            document_name="report.pdf",
            headers=["Particulars", "FY26", "FY25"],
            rows=[["Operating Cash Flow", "100", "90"]],
            markdown="| Particulars | FY26 | FY25 |\n|---|---|---|\n| Operating Cash Flow | 100 | 90 |",
            caption="Cash Flow Statement",
            footnotes=["Note A"],
        )
        t2 = Table(
            element_id="t2",
            element_type="table",
            page_number=381,
            bbox=(40.0, 50.0, 550.0, 300.0),
            document_name="report.pdf",
            headers=["Particulars", "FY26", "FY25"],
            rows=[["Financing Cash Flow", "(50)", "(40)"]],
            markdown="| Particulars | FY26 | FY25 |\n|---|---|---|\n| Financing Cash Flow | (50) | (40) |",
            is_continuation=True,
            continuation_from_page=380,
            footnotes=["Note B"],
        )

        stitched = stitch_continuation_tables(t1, t2)
        assert stitched.is_continuation is True
        assert stitched.continuation_from_page == 380
        assert stitched.row_count == 2
        assert "Operating Cash Flow" in stitched.rows[0]
        assert "Financing Cash Flow" in stitched.rows[1]
        assert "Note A" in stitched.footnotes
        assert "Note B" in stitched.footnotes

    def test_extract_tables_from_page_orchestration(self, hdfc_doc):
        # On Page 378, Tier 1 clean pass produces a Table element
        page = hdfc_doc[377]
        tables = extract_tables_from_page(page, page_number=378, document_name="hdfc.pdf", enable_tier2=False)
        assert len(tables) == 1
        assert tables[0].extraction_method == "tier1"
        assert tables[0].extraction_status == "successful"
        assert tables[0].needs_review is False
        assert tables[0].confidence_score >= 0.95

    def test_extract_tables_from_page_multi_table_orchestration(self, hdfc_doc):
        # On Page 470 without API key, Table 0 fails and degrades gracefully to Tier 3
        page = hdfc_doc[469]
        tables = extract_tables_from_page(page, page_number=470, document_name="hdfc.pdf", enable_tier2=False)
        assert len(tables) == 2
        # Table 0 is the complex 8-column segment table that failed quality gate
        assert tables[0].extraction_method == "tier3"
        assert tables[0].extraction_status == "needs_review"
        assert tables[0].needs_review is True
        # Table 1 is the 4-column geographic segment table that passed Tier 1
        assert tables[1].extraction_method == "tier1"
        assert tables[1].extraction_status == "successful"
        assert tables[1].needs_review is False


class TestTrailingTotalRowPreservation:
    """Regression tests: trailing Total rows must stay inside their own table.

    Covers the Q06 failure where Schedule 11's Total (248,623.68) was clipped
    from its table and misattached as Schedule 12's caption, while both
    Schedule 12 Totals (34,179,458,972 standalone, 3,479,487.31 consolidated)
    were dropped entirely.
    """

    def test_page389_standalone_schedule12_keeps_total(self, hdfc_doc):
        page = hdfc_doc[388]  # Page 389
        boxes = detect_all_table_regions(page)
        assert len(boxes) == 1
        t_res = extract_table_tier1(page, boxes[0])
        assert t_res["rows"][-1] == ["Total", "34,179,458,972", "26,474,170,018"]

    def test_page508_schedule11_keeps_total(self, hdfc_doc):
        page = hdfc_doc[507]  # Page 508, first table (Schedule 11 Other Assets)
        boxes = detect_all_table_regions(page)
        assert len(boxes) == 2
        t_res = extract_table_tier1(page, boxes[0])
        assert t_res["rows"][-1] == ["Total", "248,623.68", "215,800.53"]

    def test_page508_schedule12_keeps_own_total(self, hdfc_doc):
        page = hdfc_doc[507]  # Page 508, second table (Schedule 12 Contingent Liabilities)
        boxes = detect_all_table_regions(page)
        assert len(boxes) == 2
        t_res = extract_table_tier1(page, boxes[1])
        assert t_res["rows"][-1] == ["Total", "3,479,487.31", "2,709,921.85"]
        # Schedule 11's Total must not leak into Schedule 12's rows
        assert all("248,623.68" not in " ".join(r) for r in t_res["rows"])

    def test_extend_box_for_trailing_total_noop_without_total(self, hdfc_doc):
        from due_diligence.ingestion.table_extractor import extend_box_for_trailing_total

        page = hdfc_doc[377]  # Page 378 balance sheet (Total rows mid-table, none trailing)
        boxes = detect_all_table_regions(page)
        words = page.get_text("words")
        visual_lines: list[dict] = []
        for w in words:
            if not (40 < w[1] < page.rect.height - 40):
                continue
            y_mid = (w[1] + w[3]) / 2
            matched = next((vl for vl in visual_lines if abs(y_mid - vl["y_mid"]) < 4.0), None)
            if matched is None:
                visual_lines.append({"y_mid": y_mid, "words": [w]})
            else:
                matched["words"].append(w)
        box = boxes[0]
        assert extend_box_for_trailing_total(box, visual_lines, page.rect.height) == box
