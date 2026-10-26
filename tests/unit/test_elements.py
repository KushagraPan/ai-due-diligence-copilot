"""
Unit tests for DocumentElement hierarchy (NarrativeText, Heading, ListItem, Table).
"""

import pytest
from pydantic import ValidationError

from due_diligence.ingestion.elements import (
    Heading,
    ListItem,
    NarrativeText,
    Table,
)


class TestDocumentElements:
    def test_narrative_text_creation(self):
        elem = NarrativeText(
            element_id="doc1_p1_e0",
            page_number=1,
            bbox=(40.0, 50.0, 500.0, 100.0),
            document_name="annual_report.pdf",
            text="HDFC Bank is India's leading private bank.",
            section_heading="Corporate Overview",
        )
        assert elem.element_type == "narrative"
        assert elem.page_number == 1
        assert elem.text == "HDFC Bank is India's leading private bank."
        assert elem.section_heading == "Corporate Overview"

    def test_heading_creation_and_level_validation(self):
        heading = Heading(
            element_id="doc1_p1_h0",
            page_number=1,
            bbox=(40.0, 20.0, 400.0, 45.0),
            document_name="annual_report.pdf",
            text="Financial Highlights",
            level=2,
        )
        assert heading.element_type == "heading"
        assert heading.level == 2

        with pytest.raises(ValidationError):
            Heading(
                element_id="doc1_p1_h1",
                page_number=1,
                bbox=(40.0, 20.0, 400.0, 45.0),
                document_name="annual_report.pdf",
                text="Invalid Level",
                level=7,  # max is 6
            )

    def test_list_item_creation(self):
        item = ListItem(
            element_id="doc1_p2_l0",
            page_number=2,
            bbox=(50.0, 100.0, 500.0, 120.0),
            document_name="annual_report.pdf",
            text="Robust capital adequacy of 19.3%",
            marker="•",
        )
        assert item.element_type == "list_item"
        assert item.marker == "•"

    def test_table_creation_and_properties(self):
        headers = ["Particulars", "FY26", "FY25"]
        rows = [
            ["Deposits", "2,500,000", "2,200,000"],
            ["Advances", "2,100,000", "1,850,000"],
        ]
        md = Table.render_markdown(headers, rows)
        table = Table(
            element_id="doc1_p10_t0",
            page_number=10,
            bbox=(40.0, 150.0, 550.0, 350.0),
            document_name="annual_report.pdf",
            headers=headers,
            rows=rows,
            markdown=md,
            caption="Key Balance Sheet Items",
            footnotes=["* Unaudited provisional figures"],
            confidence_score=0.98,
            needs_review=False,
            extraction_status="successful",
            extraction_method="tier1",
        )
        assert table.element_type == "table"
        assert table.row_count == 2
        assert table.col_count == 3
        assert "| Particulars" in table.markdown
        assert "| Deposits" in table.markdown
        assert table.confidence_score == 0.98
        assert not table.needs_review

    def test_table_markdown_rendering_escapes_pipes(self):
        headers = ["Item | Type", "Amount"]
        rows = [["Cash | Equivalents", "100,000"]]
        md = Table.render_markdown(headers, rows)
        assert "Item \\| Type" in md
        assert "Cash \\| Equivalents" in md

    def test_table_validation_constraints(self):
        with pytest.raises(ValidationError):
            Table(
                element_id="invalid",
                page_number=0,  # page_number must be >= 1
                bbox=(0.0, 0.0, 10.0, 10.0),
                document_name="doc.pdf",
            )

        with pytest.raises(ValidationError):
            Table(
                element_id="invalid",
                page_number=1,
                bbox=(0.0, 0.0, 10.0, 10.0),
                document_name="doc.pdf",
                confidence_score=1.5,  # max is 1.0
            )

    def test_element_immutability(self):
        elem = NarrativeText(
            element_id="frozen_1",
            page_number=1,
            bbox=(0.0, 0.0, 10.0, 10.0),
            document_name="doc.pdf",
            text="Initial text",
        )
        with pytest.raises(ValidationError):
            elem.text = "Modified text"
