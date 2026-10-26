"""
Unit tests for PDF page-level text extraction.

Tests PDFParser and ExtractedDocument schemas with real PyMuPDF-generated fixtures.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from pydantic import ValidationError

from due_diligence.ingestion import (
    DEFAULT_CONFIG_PATH,
    NIL_NOT_DISCLOSED_MARKER,
    CorruptedPDFParseError,
    DocumentLoader,
    EncryptedDocumentError,
    ExtractedDocument,
    PDFParser,
    clear_symbol_mappings_cache,
    extract_page_text_with_font_remapping,
    format_unmapped_placeholder,
    get_symbol_font_category,
    is_symbol_font,
    load_symbol_mappings,
    normalize_standalone_nil_dashes,
    strip_subset_prefix,
)


@pytest.fixture
def multi_page_pdf(tmp_path: Path) -> Path:
    """Generate a real 3-page PDF with distinct text on each page."""
    pdf_path = tmp_path / "multi_page_diligence.pdf"
    doc = pymupdf.open()

    # Page 1
    p1 = doc.new_page()
    p1.insert_text((50, 72), "Note 1: Executive Overview\nTarget corporation was founded in 2018.")

    # Page 2
    p2 = doc.new_page()
    p2.insert_text(
        (50, 72), "Note 2: Financial Performance\nTotal revenue reached $45.2M in FY2023."
    )

    # Page 3
    p3 = doc.new_page()
    p3.insert_text(
        (50, 72), "Note 3: Long-Term Debt Obligations\nCredit facility matures in June 2027."
    )

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def pdf_with_blank_page(tmp_path: Path) -> Path:
    """Generate a 2-page PDF where the second page has no text."""
    pdf_path = tmp_path / "blank_page_doc.pdf"
    doc = pymupdf.open()

    p1 = doc.new_page()
    p1.insert_text((50, 72), "Page 1 Content: Diligence Cover Page.")

    # Page 2 remains intentionally empty
    doc.new_page()

    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_parse_multi_page_pdf_preserves_1_indexed_pages(multi_page_pdf: Path) -> None:
    """Verifies that PDFParser extracts text with accurate 1-indexed page numbering."""
    loader = DocumentLoader()
    validated_doc = loader.validate_and_load(
        multi_page_pdf, company="Acme Capital", financial_period="FY2023"
    )

    parser = PDFParser()
    extracted = parser.parse(validated_doc)

    assert isinstance(extracted, ExtractedDocument)
    assert extracted.document == validated_doc
    assert extracted.total_pages == 3
    assert len(extracted.pages) == 3

    # Provenance verification: exact 1-indexed pages
    assert extracted.pages[0].page_number == 1
    assert "Executive Overview" in extracted.pages[0].text
    assert not extracted.pages[0].is_blank
    assert extracted.pages[0].character_count > 0

    assert extracted.pages[1].page_number == 2
    assert "Financial Performance" in extracted.pages[1].text
    assert "$45.2M" in extracted.pages[1].text

    assert extracted.pages[2].page_number == 3
    assert "Long-Term Debt" in extracted.pages[2].text

    # Total characters must equal the sum of each page
    expected_chars = sum(p.character_count for p in extracted.pages)
    assert extracted.total_characters == expected_chars

    # Helper get_page retrieval
    assert extracted.get_page(1) == extracted.pages[0]
    assert extracted.get_page(3) == extracted.pages[2]
    assert extracted.get_page(4) is None
    assert extracted.get_page(0) is None


def test_parse_pdf_with_blank_page(pdf_with_blank_page: Path) -> None:
    """Verifies that blank pages are flagged with is_blank=True and zero characters."""
    loader = DocumentLoader()
    validated_doc = loader.validate_and_load(pdf_with_blank_page)

    parser = PDFParser()
    extracted = parser.parse(validated_doc)

    assert extracted.total_pages == 2
    assert not extracted.pages[0].is_blank
    assert extracted.pages[0].character_count > 0

    # Second page is empty
    blank_page = extracted.pages[1]
    assert blank_page.page_number == 2
    assert blank_page.is_blank is True
    assert blank_page.character_count == 0
    assert blank_page.text == ""


def test_clean_page_text_normalizes_whitespace_and_null_bytes() -> None:
    """Verifies that excessive blank lines, trailing spaces, and null bytes are stripped."""
    raw = "Section 1.1 Heading   \n\n\n\n\nBody text with \x00 null byte   \n\n\n\nNext Section"
    cleaned = PDFParser._clean_page_text(raw)

    assert "\x00" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned == "Section 1.1 Heading\n\nBody text with  null byte\n\nNext Section"


def test_parse_pdf_text_cleaning_on_page(tmp_path: Path) -> None:
    """Verifies that text extraction from a generated PDF normalizes line ends."""
    pdf_path = tmp_path / "page_text.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Line One with trailing spaces   ")
    page.insert_text((50, 100), "Line Two after gap")
    doc.save(pdf_path)
    doc.close()

    loader = DocumentLoader()
    validated_doc = loader.validate_and_load(pdf_path)

    parser = PDFParser()
    extracted = parser.parse(validated_doc)

    assert "Line One with trailing spaces" in extracted.pages[0].text
    assert "Line Two after gap" in extracted.pages[0].text
    assert not extracted.pages[0].text.endswith(" ")


def test_parse_encrypted_pdf_raises_error(tmp_path: Path) -> None:
    """Verifies that encrypted/password-protected PDFs raise EncryptedDocumentError."""
    pdf_path = tmp_path / "locked.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Confidential Board Minutes")
    doc.save(
        pdf_path,
        encryption=getattr(pymupdf, "PDF_ENCRYPT_AES_256", 2),
        user_pw="secret123",
        owner_pw="admin123",
    )
    doc.close()

    loader = DocumentLoader()
    validated_doc = loader.validate_and_load(pdf_path)

    parser = PDFParser()
    with pytest.raises(EncryptedDocumentError) as exc_info:
        parser.parse(validated_doc)

    assert "encrypted or password-protected" in str(exc_info.value)


def test_parse_corrupted_pdf_raises_error(tmp_path: Path) -> None:
    """Verifies that unreadable PDF bytes raise CorruptedPDFParseError."""
    corrupted_path = tmp_path / "broken.pdf"
    corrupted_path.write_bytes(b"%PDF-1.4\nTRUNCATED_AND_MALFORMED_GARBAGE_STREAM")

    loader = DocumentLoader()
    validated_doc = loader.validate_and_load(corrupted_path)

    parser = PDFParser()
    with pytest.raises(CorruptedPDFParseError) as exc_info:
        parser.parse(validated_doc)

    assert "Failed to open PDF" in str(exc_info.value)


def test_extracted_models_are_frozen(multi_page_pdf: Path) -> None:
    """Verifies that ExtractedDocument and ExtractedPage cannot be mutated."""
    loader = DocumentLoader()
    validated_doc = loader.validate_and_load(multi_page_pdf)

    parser = PDFParser()
    extracted = parser.parse(validated_doc)

    with pytest.raises(ValidationError):
        extracted.total_pages = 99

    with pytest.raises(ValidationError):
        extracted.pages[0].text = "Mutated Text"


def test_font_aware_remapping_on_real_hdfc_page() -> None:
    """Verifies symbol font bullet and CP1252/smart quote remapping on real HDFC annual report page 87."""
    raw_pdf = Path("data/raw/hdfc-bank__annual-report__fy2025-26.pdf")
    if not raw_pdf.exists():
        pytest.skip(f"Real document not found at {raw_pdf}")

    doc = pymupdf.open(raw_pdf)
    page_86 = doc[86]
    text = extract_page_text_with_font_remapping(page_86)
    doc.close()

    # 1. Contains no stray "\ty" sequences
    assert "\ty" not in text

    # 2. "Quarterly financial reports" and "Investor conferences, Analyst day"
    #    appear correctly with a bullet character, not a literal "y"
    assert "Quarterly financial reports" in text
    assert "Investor conferences, Analyst day" in text
    assert "•" in text

    quarterly_idx = text.find("Quarterly financial reports")
    investor_idx = text.find("Investor conferences, Analyst day")
    assert quarterly_idx != -1 and investor_idx != -1

    # Check bullet appears between Means of Engagement and Quarterly financial reports
    means_idx = text.find("Means of Engagement")
    assert means_idx < quarterly_idx
    segment_before_quarterly = text[means_idx:quarterly_idx]
    assert "•" in segment_before_quarterly
    assert "y" not in segment_before_quarterly.split()

    # Check bullet appears between Quarterly and Investor conferences
    segment_between = text[quarterly_idx:investor_idx]
    assert "•" in segment_between

    # 3. "Bank's" appears with a correct apostrophe, not a raw 0x92 byte or mangled character
    assert "Bank's Response" in text
    assert "\x92" not in text
    assert "\ufffd" not in text

    # 4. No word from a normal-text-font span was altered
    normal_words = [
        "Quarterly",
        "governance",
        "oversight",
        "Transparent",
        "shareholders",
        "Ordinary",
        "priorities",
        "Compliance",
        "Community",
    ]
    for word in normal_words:
        assert word in text, f"Normal word '{word}' was incorrectly altered"


def test_unmapped_symbol_codepoint_behavior(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies unmapped symbol font codepoints log warnings with provenance and emit placeholders."""
    mock_page = MagicMock()
    # Unmapped codepoint 0xE999 in Wingdings3
    mock_page.number = 41  # 0-indexed page 41 -> physical page 42
    mock_page.get_text.return_value = {
        "blocks": [
            {
                "lines": [
                    {
                        "spans": [
                            {"font": "Wingdings3", "text": "\ue999"},
                            {
                                "font": "HelveticaNeueLTStd-Roman",
                                "text": "Unmapped Metric Headline",
                            },
                        ]
                    }
                ]
            }
        ]
    }

    with caplog.at_level("WARNING"):
        text = extract_page_text_with_font_remapping(
            mock_page,
            document_id="test_filing.pdf",
            page_number=42,
        )

    # 1. Raw unmapped character must NEVER appear unchanged in output
    assert "\ue999" not in text

    # 2. Traceable placeholder must be present, with whitespace padding
    expected_placeholder = format_unmapped_placeholder("Wingdings3", 0xE999)
    assert "[UNMAPPED:Wingdings3:0xE999]" in text
    assert expected_placeholder in text

    # 3. Adjacent normal text words must not be corrupted or merged
    assert "Unmapped Metric Headline" in text

    # 4. Structured warning must be logged with document, page, font, and hex codepoint
    assert "Unmapped codepoint in symbol font" in caplog.text
    assert "test_filing.pdf" in caplog.text
    assert "42" in caplog.text
    assert "Wingdings3" in caplog.text
    assert "0xE999" in caplog.text


def test_strip_subset_prefix_normalization() -> None:
    """Verifies 6-character uppercase subset prefixes are stripped and resolve to same config entry."""
    assert strip_subset_prefix("ABCDEF+Wingdings-Regular") == "Wingdings-Regular"
    assert strip_subset_prefix("XYZUVW+Wingdings3") == "Wingdings3"
    assert strip_subset_prefix("Wingdings-Regular") == "Wingdings-Regular"
    assert strip_subset_prefix("HelveticaNeueLTStd-Roman") == "HelveticaNeueLTStd-Roman"

    # Both subset-prefixed and non-prefixed spans must resolve identically
    mock_page_subset = MagicMock()
    mock_page_subset.number = 0
    mock_page_subset.get_text.return_value = {
        "blocks": [
            {
                "lines": [
                    {
                        "spans": [
                            {"font": "ABCDEF+Wingdings-Regular", "text": "y"},
                            {"font": "Helvetica", "text": "Item text"},
                        ]
                    }
                ]
            }
        ]
    }

    mock_page_clean = MagicMock()
    mock_page_clean.number = 0
    mock_page_clean.get_text.return_value = {
        "blocks": [
            {
                "lines": [
                    {
                        "spans": [
                            {"font": "Wingdings-Regular", "text": "y"},
                            {"font": "Helvetica", "text": "Item text"},
                        ]
                    }
                ]
            }
        ]
    }

    text_subset = extract_page_text_with_font_remapping(mock_page_subset)
    text_clean = extract_page_text_with_font_remapping(mock_page_clean)

    assert text_subset == text_clean
    assert "• Item text" in text_subset


def test_symbol_font_mappings_yaml_regression() -> None:
    """Regression test: verifies every entry in symbol_font_mappings.yaml resolves to its configured glyph."""
    clear_symbol_mappings_cache()
    symbol_map, text_punct_map = load_symbol_mappings(DEFAULT_CONFIG_PATH)

    assert len(symbol_map) > 0, "symbol_fonts section in YAML is empty"
    assert len(text_punct_map) > 0, "text_punctuation section in YAML is empty"

    # Test each symbol font mapping entry
    for (category, cp), expected_glyph in symbol_map.items():
        font_name = (
            "Wingdings3"
            if category == "wingdings3"
            else ("Symbol" if category == "symbol" else "Wingdings-Regular")
        )
        mock_page = MagicMock()
        mock_page.number = 0
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {"font": font_name, "text": chr(cp)},
                                {"font": "Helvetica", "text": " Test"},
                            ]
                        }
                    ]
                }
            ]
        }

        output_text = extract_page_text_with_font_remapping(mock_page)
        assert expected_glyph in output_text, (
            f"Failed regression for font={font_name}, cp=0x{cp:04X}"
        )

    # Test each text punctuation entry
    for cp, expected_glyph in text_punct_map.items():
        mock_page = MagicMock()
        mock_page.number = 0
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {"font": "HelveticaNeueLTStd-Roman", "text": f"Pre{chr(cp)}Post"},
                            ]
                        }
                    ]
                }
            ]
        }

        output_text = extract_page_text_with_font_remapping(mock_page)
        assert expected_glyph in output_text, f"Failed regression for punctuation cp=0x{cp:04X}"


def test_symbol_font_category_isolation() -> None:
    """Verifies Wingdings, Wingdings2, Wingdings3, Webdings, and Symbol resolve to distinct categories."""
    # Canonical families resolve distinctly
    assert get_symbol_font_category("Wingdings") == "wingdings"
    assert get_symbol_font_category("Wingdings-Regular") == "wingdings"
    assert get_symbol_font_category("WingdingsMT") == "wingdings"
    assert get_symbol_font_category("ABCDEF+Wingdings-Regular") == "wingdings"

    assert get_symbol_font_category("Wingdings2") == "wingdings2"
    assert get_symbol_font_category("Wingdings-2") == "wingdings2"
    assert get_symbol_font_category("Wingdings 2 Regular") == "wingdings2"

    assert get_symbol_font_category("Wingdings3") == "wingdings3"
    assert get_symbol_font_category("Wingdings-3") == "wingdings3"
    assert get_symbol_font_category("XYZUVW+Wingdings3") == "wingdings3"

    assert get_symbol_font_category("Webdings") == "webdings"
    assert get_symbol_font_category("Webdings-Regular") == "webdings"

    assert get_symbol_font_category("Symbol") == "symbol"
    assert get_symbol_font_category("SymbolMT") == "symbol"
    assert get_symbol_font_category("Symbol-Regular") == "symbol"

    # Unrecognized or variant families MUST NOT conflate with Wingdings 1 or other families
    unrecognized = [
        "Wingdings4",
        "Wingdings-4",
        "WingdingsPro",
        "CustomWingdings",
        "LatinSymbols",
        "Symbols",
        "MyCustomFont",
    ]
    for font_name in unrecognized:
        assert get_symbol_font_category(font_name) == "unknown", (
            f"Font '{font_name}' should return 'unknown', not match an existing category"
        )

    # Broad symbol detector catches variants to prevent raw text leak
    assert is_symbol_font("Wingdings4") is True
    assert is_symbol_font("Wingdings-4") is True
    assert is_symbol_font("ZapfDingbats") is True
    assert is_symbol_font("Helvetica") is False


def test_unrecognized_symbol_font_falls_through_to_unmapped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies unknown symbol font (e.g. Wingdings4) with codepoint 0x79 does NOT map to bullet."""
    mock_page = MagicMock()
    mock_page.number = 0
    mock_page.get_text.return_value = {
        "blocks": [
            {
                "lines": [
                    {
                        "spans": [
                            # 0x79 is 'y', which is a bullet in Wingdings 1, but unknown in Wingdings4
                            {"font": "Wingdings4", "text": "y"},
                            {"font": "Helvetica", "text": " Unknown Item"},
                        ]
                    }
                ]
            }
        ]
    }

    with caplog.at_level("WARNING"):
        text = extract_page_text_with_font_remapping(mock_page, document_id="doc_test.pdf")

    # Must NOT become a bullet
    assert "•" not in text
    # Must NOT pass through raw ASCII 'y'
    assert " y " not in text
    # Must emit unmapped placeholder
    assert "[UNMAPPED:Wingdings4:0x0079]" in text
    assert "Unknown Item" in text
    assert "Unmapped codepoint in symbol font" in caplog.text


def test_standalone_nil_dash_heuristic() -> None:
    """Verifies that standalone en-dashes/em-dashes surrounded by whitespace map to [NIL/NOT DISCLOSED]."""
    mock_page = MagicMock()
    mock_page.number = 0
    mock_page.get_text.return_value = {
        "blocks": [
            {
                "lines": [
                    {
                        "spans": [
                            # Synthetic table line with standalone en-dash and em-dash
                            {
                                "font": "Helvetica",
                                "text": "Non-performing Assets\t–\t1,420\nCash Flow\t—\t500",
                            },
                        ]
                    }
                ]
            }
        ]
    }

    text = extract_page_text_with_font_remapping(mock_page)
    assert f"Non-performing Assets\t{NIL_NOT_DISCLOSED_MARKER}\t1,420" in text
    assert f"Cash Flow\t{NIL_NOT_DISCLOSED_MARKER}\t500" in text
    assert "–" not in text
    assert "—" not in text

    # Direct unit test for the standalone normalization helper
    direct_result = normalize_standalone_nil_dashes("Non-performing Assets\t–\t1,420")
    assert direct_result == f"Non-performing Assets\t{NIL_NOT_DISCLOSED_MARKER}\t1,420"


def test_dash_preservation_mid_word_and_numeric_prefix() -> None:
    """Verifies that mid-word and numeric prefix dashes are preserved as plain dashes, not converted to nil marker."""
    mock_page = MagicMock()
    mock_page.number = 0
    mock_page.get_text.return_value = {
        "blocks": [
            {
                "lines": [
                    {
                        "spans": [
                            # Mid-word em-dash and en-dash
                            {"font": "Helvetica", "text": "facility—maturing in 2028"},
                            {"font": "Helvetica", "text": "Debt–to–equity ratio"},
                            # Prefix to number (negative values)
                            {"font": "Helvetica", "text": "Contraction of -12% observed"},
                            {"font": "Helvetica", "text": "Net loss of –1,420 recorded"},
                        ]
                    }
                ]
            }
        ]
    }

    text = extract_page_text_with_font_remapping(mock_page)
    # Mid-word dashes preserved
    assert "facility—maturing" in text
    assert "Debt–to–equity" in text
    # Negative prefixes preserved
    assert "-12%" in text
    assert "–1,420" in text
    # Nil marker MUST NOT be present
    assert NIL_NOT_DISCLOSED_MARKER not in text


def test_extracted_document_symbol_font_status_provenance(tmp_path: Path) -> None:
    """Verifies that ExtractedDocument carries symbol_font_status and unmapped_symbol_count."""
    # 1. Clean document
    clean_pdf = tmp_path / "clean_doc.pdf"
    doc1 = pymupdf.open()
    p1 = doc1.new_page()
    p1.insert_text((50, 72), "Clean document text")
    doc1.save(clean_pdf)
    doc1.close()

    loader = DocumentLoader()
    val_clean = loader.validate_and_load(clean_pdf)
    parser = PDFParser()
    extracted_clean = parser.parse(val_clean)

    assert extracted_clean.symbol_font_status == "clean"
    assert extracted_clean.unmapped_symbol_count == 0
    assert extracted_clean.pages[0].has_unmapped_symbols is False

    # 2. Document with unmapped symbol placeholders
    dirty_pdf = tmp_path / "dirty_doc.pdf"
    doc2 = pymupdf.open()
    p2 = doc2.new_page()
    p2.insert_text((50, 72), "Sample text")
    doc2.save(dirty_pdf)
    doc2.close()

    val_dirty = loader.validate_and_load(dirty_pdf)

    # Mock extract_page_text_with_font_remapping to simulate unmapped symbol glyph
    with patch(
        "due_diligence.ingestion.parser.extract_page_text_with_font_remapping",
        return_value="Headline [UNMAPPED:Wingdings4:0x0079] Metric",
    ):
        extracted_dirty = parser.parse(val_dirty)

    assert extracted_dirty.symbol_font_status.startswith("unmapped_allowed_")
    assert extracted_dirty.unmapped_symbol_count == 1
    assert extracted_dirty.pages[0].has_unmapped_symbols is True

    # 3. Explicit reviewed status
    with patch(
        "due_diligence.ingestion.parser.extract_page_text_with_font_remapping",
        return_value="Headline [UNMAPPED:Wingdings4:0x0079] Metric",
    ):
        extracted_custom = parser.parse(val_dirty, symbol_font_status="reviewed_2026-09-25")

    assert extracted_custom.symbol_font_status == "reviewed_2026-09-25"
    assert extracted_custom.unmapped_symbol_count == 1



