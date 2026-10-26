"""
Unit tests for the batch preflight symbol font scanner.

Verifies preflight_scan, assert_preflight_clean, and report generation
across clean and unmapped PDF documents.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest

from due_diligence.ingestion import (
    DocumentLoader,
    PDFParser,
    PreflightReviewRequiredError,
    assert_preflight_clean,
    partition_preflight_batch,
    preflight_scan,
    save_preflight_report,
)
from scripts.preflight_scan import main as cli_main


def _create_mock_pdf_with_font(
    file_path: Path,
    pages_data: list[list[tuple[str, str]]],
) -> Path:
    """Helper to create a PDF with specific font spans on each page."""
    doc = pymupdf.open()
    for spans in pages_data:
        page = doc.new_page()
        y_pos = 72
        for font_name, text in spans:
            page.insert_text((50, y_pos), text, fontname="Helvetica")  # Draw visible placeholder
            y_pos += 25
    doc.save(file_path)
    doc.close()
    return file_path


def test_preflight_scan_clean_pdf(tmp_path: Path) -> None:
    """Verifies that a PDF with only mapped symbol glyphs reports clean status."""
    pdf_path = tmp_path / "clean_doc.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Standard paragraph text without symbol fonts.")
    doc.save(pdf_path)
    doc.close()

    results = preflight_scan([pdf_path])
    summary = results["batch_summary"]

    assert summary["total_documents"] == 1
    assert summary["clean_documents"] == 1
    assert summary["review_documents"] == 0
    assert summary["total_unmapped_codepoints"] == 0

    doc_info = results["documents"][str(pdf_path)]
    assert doc_info["status"] == "clean"
    assert doc_info["total_pages"] == 1
    assert doc_info["unmapped_codepoints"] == []


def test_preflight_scan_unmapped_codepoints(tmp_path: Path) -> None:
    """Verifies that unmapped symbol font codepoints are correctly detected and detailed."""
    pdf_path = tmp_path / "unmapped_doc.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Heading")
    doc.save(pdf_path)
    doc.close()

    # Mock page.get_text("dict") to return an unmapped symbol font span
    with patch("pymupdf.open") as mock_open:
        mock_doc = mock_open.return_value
        mock_doc.__len__.return_value = 1
        mock_page = mock_doc.__getitem__.return_value
        mock_page.number = 0
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                # Wingdings3 with unmapped 0xE999
                                {"font": "Wingdings3", "text": "\ue999"},
                                {"font": "Helvetica", "text": " Some metric description"},
                            ]
                        }
                    ]
                }
            ]
        }

        results = preflight_scan([pdf_path])

    summary = results["batch_summary"]
    assert summary["total_documents"] == 1
    assert summary["clean_documents"] == 0
    assert summary["review_documents"] == 1
    assert summary["total_unmapped_codepoints"] == 1

    doc_info = results["documents"][str(pdf_path)]
    assert doc_info["status"] == "needs_review"
    unmapped = doc_info["unmapped_codepoints"]
    assert len(unmapped) == 1
    assert unmapped[0]["font_name"] == "Wingdings3"
    assert unmapped[0]["codepoint_hex"] == "0xE999"
    assert unmapped[0]["codepoint_dec"] == 0xE999
    assert unmapped[0]["count"] == 1
    assert unmapped[0]["page_numbers"] == [1]


def test_preflight_page_numbers_capped_at_five(tmp_path: Path) -> None:
    """Verifies that page_numbers list is capped at 5 and total_pages tracks full count."""
    pdf_path = tmp_path / "multi_page_unmapped.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    # Simulate 8 pages all containing unmapped glyph 0xE111 in Wingdings
    with patch("pymupdf.open") as mock_open:
        mock_doc = mock_open.return_value
        mock_doc.__len__.return_value = 8

        def get_mock_page(idx: int) -> pymupdf.Page:
            page = pymupdf.open().new_page()
            page.get_text = lambda fmt: {"blocks": [{"lines": [{"spans": [{"font": "Wingdings", "text": "\ue111"}]}]}]}  # type: ignore[method-assign,misc]
            return page

        mock_doc.__getitem__.side_effect = get_mock_page

        results = preflight_scan([pdf_path])

    unmapped = results["documents"][str(pdf_path)]["unmapped_codepoints"][0]
    assert unmapped["page_numbers"] == [1, 2, 3, 4, 5]
    assert unmapped["total_pages"] == 8
    assert unmapped["count"] == 8


def test_save_preflight_report(tmp_path: Path) -> None:
    """Verifies report JSON serialization to disk."""
    report_target = tmp_path / "custom_report.json"
    scan_data = {
        "documents": {"doc.pdf": {"status": "needs_review"}},
        "batch_summary": {"review_documents": 1},
    }

    saved_path = save_preflight_report(scan_data, output_path=report_target)
    assert saved_path == report_target
    assert saved_path.exists()

    with open(saved_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == scan_data


def test_assert_preflight_clean_enforcement(tmp_path: Path) -> None:
    """Verifies exception raising on needs_review when allow_unmapped is False."""
    report_target = tmp_path / "review.json"
    dirty_results = {
        "documents": {"bad.pdf": {"status": "needs_review"}},
        "batch_summary": {"review_documents": 1},
    }

    with pytest.raises(PreflightReviewRequiredError) as exc_info:
        assert_preflight_clean(dirty_results, allow_unmapped=False, report_path=report_target)

    assert "1 document(s) require review" in str(exc_info.value)
    assert exc_info.value.review_documents == 1
    assert report_target.exists()

    # Bypassed when allow_unmapped=True
    assert_preflight_clean(dirty_results, allow_unmapped=True, report_path=report_target)


def test_cli_preflight_scan_clean(tmp_path: Path) -> None:
    """Verifies CLI exits with 0 on clean PDFs."""
    clean_pdf = tmp_path / "clean.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(clean_pdf)
    doc.close()

    with patch("sys.argv", ["preflight_scan.py", str(clean_pdf)]):
        with pytest.raises(SystemExit) as exc:
            cli_main()
        assert exc.value.code == 0


def test_cli_preflight_scan_unmapped_exit_codes(tmp_path: Path) -> None:
    """Verifies CLI exits with code 1 without --allow-unmapped and 0 with --allow-unmapped."""
    dirty_pdf = tmp_path / "dirty.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(dirty_pdf)
    doc.close()

    mock_results = {
        "documents": {
            str(dirty_pdf): {
                "status": "needs_review",
                "total_pages": 1,
                "unmapped_codepoints": [
                    {
                        "font_name": "Wingdings",
                        "codepoint_hex": "0xE999",
                        "codepoint_dec": 59801,
                        "count": 1,
                        "page_numbers": [1],
                        "total_pages": 1,
                        "example_context": "\ue999 Context",
                    }
                ],
            }
        },
        "batch_summary": {
            "total_documents": 1,
            "clean_documents": 0,
            "review_documents": 1,
            "total_unmapped_codepoints": 1,
        },
    }

    report_path = tmp_path / "test_report.json"

    with patch("scripts.preflight_scan.preflight_scan", return_value=mock_results):
        # 1. Without --allow-unmapped -> exits 1
        with patch(
            "sys.argv",
            ["preflight_scan.py", str(dirty_pdf), "--output", str(report_path)],
        ):
            with pytest.raises(SystemExit) as exc:
                cli_main()
            assert exc.value.code == 1

        # 2. With --allow-unmapped -> exits 0
        with patch(
            "sys.argv",
            ["preflight_scan.py", str(dirty_pdf), "--allow-unmapped", "--output", str(report_path)],
        ):
            with pytest.raises(SystemExit) as exc:
                cli_main()
            assert exc.value.code == 0


def test_partition_preflight_batch_mixed_behavior(tmp_path: Path) -> None:
    """Verifies that partition_preflight_batch separates clean from flagged documents."""
    doc_clean1 = tmp_path / "clean1.pdf"
    doc_clean2 = tmp_path / "clean2.pdf"
    doc_flagged = tmp_path / "flagged.pdf"

    mock_results = {
        "documents": {
            str(doc_clean1): {"status": "clean"},
            str(doc_clean2): {"status": "clean"},
            str(doc_flagged): {"status": "needs_review", "unmapped_codepoints": [{"cp": 1}]},
        },
        "batch_summary": {
            "total_documents": 3,
            "clean_documents": 2,
            "review_documents": 1,
            "total_unmapped_codepoints": 1,
        },
    }

    # 1. Default (allow_unmapped=False): clean proceed, flagged excluded
    approved, skipped = partition_preflight_batch(mock_results, allow_unmapped=False)
    assert len(approved) == 2
    assert doc_clean1 in approved
    assert doc_clean2 in approved
    assert len(skipped) == 1
    assert doc_flagged in skipped

    # 2. With allow_unmapped=True: all 3 approved
    approved_all, skipped_none = partition_preflight_batch(mock_results, allow_unmapped=True)
    assert len(approved_all) == 3
    assert len(skipped_none) == 0


def test_mixed_batch_ingestion_excludes_flagged_without_allow_unmapped(tmp_path: Path) -> None:
    """Verifies that PDFParser.parse_batch ingests 2 clean documents and skips 1 flagged document."""
    # Create 2 clean PDFs
    clean_pdf_1 = tmp_path / "clean_1.pdf"
    d1 = pymupdf.open()
    p1 = d1.new_page()
    p1.insert_text((50, 72), "Note 1: Clean performance data.")
    d1.save(clean_pdf_1)
    d1.close()

    clean_pdf_2 = tmp_path / "clean_2.pdf"
    d2 = pymupdf.open()
    p2 = d2.new_page()
    p2.insert_text((50, 72), "Note 2: Clean risk factors.")
    d2.save(clean_pdf_2)
    d2.close()

    # Create 1 flagged PDF with unmapped symbol font glyph
    flagged_pdf = tmp_path / "flagged.pdf"
    df = pymupdf.open()
    pf = df.new_page()
    pf.insert_text((50, 72), "Unmapped symbol item")
    df.save(flagged_pdf)
    df.close()

    loader = DocumentLoader()
    val_clean_1 = loader.validate_and_load(clean_pdf_1)
    val_clean_2 = loader.validate_and_load(clean_pdf_2)
    val_flagged = loader.validate_and_load(flagged_pdf)

    # Mock preflight_scan to report clean_1 and clean_2 as clean, and flagged as needs_review
    mock_scan_results = {
        "documents": {
            str(clean_pdf_1.resolve()): {"status": "clean"},
            str(clean_pdf_2.resolve()): {"status": "clean"},
            str(flagged_pdf.resolve()): {
                "status": "needs_review",
                "unmapped_codepoints": [
                    {
                        "font_name": "Wingdings4",
                        "codepoint_hex": "0x0079",
                        "codepoint_dec": 121,
                        "count": 1,
                        "page_numbers": [1],
                        "total_pages": 1,
                        "example_context": "Sample",
                    }
                ],
            },
        },
        "batch_summary": {
            "total_documents": 3,
            "clean_documents": 2,
            "review_documents": 1,
            "total_unmapped_codepoints": 1,
        },
    }

    parser = PDFParser()

    with patch("due_diligence.ingestion.parser.preflight_scan", return_value=mock_scan_results):
        # 1. Ingest without allow_unmapped: 2 clean documents ingest normally, 1 flagged document is skipped
        extracted, skipped = parser.parse_batch(
            [val_clean_1, val_clean_2, val_flagged],
            allow_unmapped=False,
        )

        assert len(extracted) == 2
        assert extracted[0].document.document_name == "clean_1.pdf"
        assert extracted[1].document.document_name == "clean_2.pdf"
        assert "Clean performance data" in extracted[0].pages[0].text
        assert "Clean risk factors" in extracted[1].pages[0].text

        assert len(skipped) == 1
        assert skipped[0].document_name == "flagged.pdf"

        # 2. Ingest with allow_unmapped=True: all 3 documents proceed to ingestion
        extracted_all, skipped_none = parser.parse_batch(
            [val_clean_1, val_clean_2, val_flagged],
            allow_unmapped=True,
        )
        assert len(extracted_all) == 3
        assert len(skipped_none) == 0


def test_assert_preflight_clean_per_document(tmp_path: Path) -> None:
    """Verifies that assert_preflight_clean passes on clean document and raises on flagged document."""
    doc_clean = tmp_path / "clean.pdf"
    doc_flagged = tmp_path / "flagged.pdf"

    scan_data = {
        "documents": {
            str(doc_clean): {"status": "clean"},
            str(doc_flagged): {"status": "needs_review", "unmapped_codepoints": [{"cp": 1}]},
        },
        "batch_summary": {
            "total_documents": 2,
            "clean_documents": 1,
            "review_documents": 1,
            "total_unmapped_codepoints": 1,
        },
    }

    # Clean document passes
    assert_preflight_clean(scan_data, document_path=doc_clean, allow_unmapped=False)

    # Flagged document raises PreflightReviewRequiredError
    with pytest.raises(PreflightReviewRequiredError) as exc_info:
        assert_preflight_clean(scan_data, document_path=doc_flagged, allow_unmapped=False)
    assert f"Document '{doc_flagged!s}' requires review" in str(exc_info.value)

    # Flagged document passes when allow_unmapped=True
    assert_preflight_clean(scan_data, document_path=doc_flagged, allow_unmapped=True)

