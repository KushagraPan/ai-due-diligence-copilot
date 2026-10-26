"""
Layout reconciliation and spatial partitioning of document pages.

1. Purpose:
   Unifies narrative text, headings, list items, and extracted tables into an
   ordered stream of typed DocumentElements per page, resolving spatial ownership
   and propagating hierarchical section context.

2. Role in Overall RAG Pipeline:
   Sits between page-level text extraction (parser.py) / table extraction (table_extractor.py)
   and downstream chunking. Reconciles raw text blocks against table bounding boxes,
   eliminates duplicate table ingestion, detects multi-column reading orders, and
   enriches elements with section heading provenance before semantic chunking.

3. Expected Inputs:
   - PyMuPDF Page object, 1-indexed page number, and document name.
   - Optional pre-extracted Table objects (from table_extractor.py).
   - Optional LayoutConfig for tuning overlap, column, and heading heuristics.
   - Optional active heading stack for multi-page section context propagation.

4. Outputs:
   - Ordered list of DocumentElement instances (Heading, NarrativeText, ListItem, Table).
   - Machine-readable PageLayoutReport tracking character reconciliation, block ownership,
     column geometry, and review flags.
   - Updated heading stack for subsequent pages.

5. Dependencies:
   - pymupdf (fitz)
   - pydantic
   - due_diligence.ingestion.elements (DocumentElement, NarrativeText, Heading, ListItem, Table)
   - due_diligence.ingestion.table_extractor (detect_all_table_regions, extract_tables_from_page)
   - due_diligence.ingestion.font_mappings (load_symbol_mappings, is_symbol_font, get_symbol_font_category, strip_subset_prefix, format_unmapped_placeholder)
   - due_diligence.ingestion.parser (normalize_standalone_nil_dashes)

6. Outside Scope:
   - Does not perform vector embedding generation or vector database operations.
   - Does not perform semantic chunk splitting or text chunk windowing.
   - Does not mutate or redefine the DocumentElement Pydantic schemas in elements.py.
"""

import logging
import re
from pathlib import Path
from typing import Any

import pymupdf
from pydantic import BaseModel, ConfigDict, Field

from due_diligence.ingestion.elements import (
    TOTAL_ROW_PATTERN,
    DocumentElement,
    Heading,
    ListItem,
    NarrativeText,
    Table,
)
from due_diligence.ingestion.font_mappings import (
    format_unmapped_placeholder,
    get_symbol_font_category,
    is_symbol_font,
    load_symbol_mappings,
    strip_subset_prefix,
)
from due_diligence.ingestion.parser import normalize_standalone_nil_dashes
from due_diligence.ingestion.table_extractor import extract_tables_from_page

logger = logging.getLogger(__name__)

# List marker pattern recognizing standard bullet characters, numbered items, and roman numerals
LIST_MARKER_PATTERN = re.compile(
    r"^(\s*(?:[•●▪◦‣\*\-\–\—]|\d+\.|\([a-zA-Z0-9]+\)|[a-zA-Z]\.|\([ivxLCDM]+\)|[ivxLCDM]+\.))\s+(.+)$",
    re.DOTALL,
)

# Numbering pattern for section and note titles
SECTION_NUMBER_PATTERN = re.compile(
    r"^(note\s+\d+|item\s+\d+|schedule\s+[a-z0-9]+|\d+(\.\d+)*\s+[a-z])",
    re.IGNORECASE,
)

# Footnote indicator pattern
FOOTNOTE_START_PATTERN = re.compile(
    r"^(\*+|\^|#|†|‡|Note:|Notes:|\d+\s+[A-Z]|\([a-z0-9]+\)|\[[a-z0-9]+\])",
    re.IGNORECASE,
)

# Standalone date pattern rejected from becoming headings
DATE_ONLY_PATTERN = re.compile(
    r"^(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)?,?\s*"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}$",
    re.IGNORECASE,
)

# Corporate signatory and governance title patterns rejected from becoming headings
SIGNATORY_PATTERN = re.compile(
    r"\b(?:independent director|executive director|managing director|non-executive director|"
    r"company secretary|chief financial officer|chartered accountants|membership no|partner)\b",
    re.IGNORECASE,
)

# Table summary-row pattern is canonical in elements.py (imported above).
# A "Total ..." line is a data row of the table above it, never a caption for
# the table below it. Claiming it as a caption (via istitle()) caused Schedule
# 11's Total to surface as Schedule 12's table title.


def is_valid_table_caption(text: str) -> bool:
    """
    Guard deciding whether a nearby text line may serve as a table caption.

    Rejects Total summary rows and pure-numeric lines, which carry a table's
    figures rather than titling the adjacent table.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if TOTAL_ROW_PATTERN.match(stripped):
        return False
    if not re.search(r"[A-Za-z]", stripped):
        return False
    tokens = stripped.split()
    digit_tokens = sum(1 for tok in tokens if any(c.isdigit() for c in tok))
    return not (tokens and digit_tokens / len(tokens) > 0.6)


class LayoutConfig(BaseModel):
    """Configurable heuristics and thresholds for layout extraction and reconciliation."""

    model_config = ConfigDict(frozen=True)

    table_overlap_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum bbox overlap ratio to claim block as table"
    )
    reconciliation_tolerance_pct: float = Field(
        default=5.0, ge=0.0, description="Maximum allowed percentage difference between raw and structured chars"
    )
    heading_font_ratio: float = Field(
        default=1.15, gt=0.0, description="Minimum font size ratio vs median body font for heading candidate"
    )
    header_margin_pt: float = Field(
        default=45.0, ge=0.0, description="Top margin pt below which content is treated as page body"
    )
    footer_margin_pt: float = Field(
        default=45.0, ge=0.0, description="Bottom margin pt above which content is treated as page body"
    )
    multi_column_gutter_pt: float = Field(
        default=15.0, ge=0.0, description="Minimum gutter width between multi-column bands"
    )
    multi_column_min_blocks: int = Field(
        default=2, ge=1, description="Minimum blocks per column cluster to establish multi-column layout"
    )


class PageLayoutReport(BaseModel):
    """Machine-readable diagnostic and reconciliation record for a single page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(..., ge=1, description="1-indexed physical page number")
    raw_character_count: int = Field(..., ge=0, description="Raw character count from page.get_text('text')")
    structured_character_count: int = Field(..., ge=0, description="Sum of characters across all structured elements")
    reconciliation_delta_pct: float = Field(..., ge=0.0, description="Absolute percentage delta between raw and structured")
    table_count: int = Field(..., ge=0, description="Total table elements on the page")
    heading_count: int = Field(..., ge=0, description="Total heading elements on the page")
    list_item_count: int = Field(..., ge=0, description="Total list item elements on the page")
    narrative_count: int = Field(..., ge=0, description="Total narrative text elements on the page")
    multi_column: bool = Field(..., description="True if page layout was classified as multi-column")
    column_count: int = Field(..., ge=1, description="Detected column count (1, 2, or 3)")
    source_block_count: int = Field(..., ge=0, description="Total source text blocks in PyMuPDF dict")
    owned_block_count: int = Field(..., ge=0, description="Total blocks claimed by exactly one category")
    unowned_block_count: int = Field(..., ge=0, description="Blocks unassigned to any category (must be 0)")
    multiply_owned_block_count: int = Field(..., ge=0, description="Blocks assigned to multiple categories (must be 0)")
    needs_review: bool = Field(..., description="True if any audit check failed")
    review_reasons: list[str] = Field(default_factory=list, description="List of diagnostic reasons flagged during audit")


def compute_bbox_overlap_ratio(
    bbox1: tuple[float, float, float, float],
    bbox2: tuple[float, float, float, float],
) -> float:
    """
    Computes intersection area divided by bbox1 area.
    Returns 0.0 if there is no intersection or if bbox1 area is zero.
    """
    ix0 = max(bbox1[0], bbox2[0])
    iy0 = max(bbox1[1], bbox2[1])
    ix1 = min(bbox1[2], bbox2[2])
    iy1 = min(bbox1[3], bbox2[3])

    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    intersection_area = (ix1 - ix0) * (iy1 - iy0)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

    if bbox1_area <= 0:
        return 0.0

    return intersection_area / bbox1_area


def clean_block_spans_with_font_remapping(
    lines: list[dict[str, Any]],
    symbol_map: dict[tuple[str, int], str],
    text_punct_map: dict[int, str],
    document_id: str,
    page_number: int,
) -> str:
    """
    Assembles normalized text from block lines and spans, applying font remapping
    for symbol fonts, character normalizations, and nil-dash replacement.
    """
    lines_text: list[str] = []
    for line in lines:
        spans_text: list[str] = []
        for span in line.get("spans", []):
            text = span.get("text", "")
            font = span.get("font", "")
            if not text:
                continue

            clean_font = strip_subset_prefix(font)

            if is_symbol_font(font):
                category = get_symbol_font_category(font)
                remapped_chars: list[str] = []
                for char in text:
                    cp = ord(char)
                    key = (category, cp)
                    if key in symbol_map:
                        remapped_chars.append(symbol_map[key])
                    else:
                        remapped_chars.append(format_unmapped_placeholder(clean_font, cp))
                spans_text.append("".join(remapped_chars))
            else:
                remapped_chars = [text_punct_map.get(ord(char), char) for char in text]
                spans_text.append("".join(remapped_chars))

        line_str = "".join(spans_text).strip()
        if line_str:
            lines_text.append(line_str)

    if not lines_text:
        return ""

    joined = " ".join(lines_text)
    # Strip residual ASCII control characters (e.g. \x07, \x00) while keeping standard whitespace
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", joined)
    return normalize_standalone_nil_dashes(cleaned.strip())


def detect_multi_column_layout(
    blocks: list[dict[str, Any]],
    page_w: float,
    page_h: float,
    table_bboxes: list[tuple[float, float, float, float]],
    config: LayoutConfig,
) -> tuple[bool, int, list[tuple[float, float]]]:
    """
    Detects whether non-table body blocks form a genuine 2-column or 3-column layout.
    Returns (is_multi_column, column_count, column_x_ranges).
    """
    # Filter out table blocks
    non_table_blocks: list[dict[str, Any]] = []
    for b in blocks:
        if b.get("type") != 0:
            continue
        bbox = tuple(b["bbox"])
        overlaps_table = any(
            compute_bbox_overlap_ratio(bbox, tb) >= config.table_overlap_threshold
            for tb in table_bboxes
        )
        if not overlaps_table:
            non_table_blocks.append(b)

    # Filter for body blocks with substantial text
    body_blocks = [
        b
        for b in non_table_blocks
        if config.header_margin_pt < b["bbox"][1] and b["bbox"][3] < page_h - config.footer_margin_pt
    ]
    substantive_blocks = [
        b
        for b in body_blocks
        if len(" ".join(" ".join(s["text"] for s in l["spans"]) for l in b.get("lines", [])).strip()) > 20
    ]

    if len(substantive_blocks) < 4:
        return False, 1, [(0.0, page_w)]

    content_w = page_w - 80.0
    narrow_blocks = [b for b in substantive_blocks if (b["bbox"][2] - b["bbox"][0]) < content_w * 0.65]

    if len(narrow_blocks) / len(substantive_blocks) < 0.40:
        return False, 1, [(0.0, page_w)]

    # 1. Check for 3-column layout
    w3 = content_w / 3.0
    c1_3 = [b for b in substantive_blocks if b["bbox"][2] <= 40.0 + w3 + 20.0]
    c2_3 = [
        b
        for b in substantive_blocks
        if 40.0 + w3 - 20.0 <= b["bbox"][0] and b["bbox"][2] <= 40.0 + 2 * w3 + 20.0
    ]
    c3_3 = [b for b in substantive_blocks if b["bbox"][0] >= 40.0 + 2 * w3 - 20.0]

    min_b = config.multi_column_min_blocks
    if len(c1_3) >= min_b and len(c2_3) >= min_b and len(c3_3) >= min_b:
        x_ranges = [
            (min(b["bbox"][0] for b in c1_3), max(b["bbox"][2] for b in c1_3)),
            (min(b["bbox"][0] for b in c2_3), max(b["bbox"][2] for b in c2_3)),
            (min(b["bbox"][0] for b in c3_3), max(b["bbox"][2] for b in c3_3)),
        ]
        return True, 3, x_ranges

    # 2. Check for 2-column layout
    mid_x = page_w / 2.0
    col1 = [b for b in substantive_blocks if b["bbox"][2] <= mid_x + 30.0]
    col2 = [b for b in substantive_blocks if b["bbox"][0] >= mid_x - 30.0]

    if len(col1) >= min_b and len(col2) >= min_b:
        col1_y = (min(b["bbox"][1] for b in col1), max(b["bbox"][3] for b in col1))
        col2_y = (min(b["bbox"][1] for b in col2), max(b["bbox"][3] for b in col2))
        v_overlap = max(0.0, min(col1_y[1], col2_y[1]) - max(col1_y[0], col2_y[0]))
        if v_overlap > 60.0:
            x_ranges = [
                (min(b["bbox"][0] for b in col1), max(b["bbox"][2] for b in col1)),
                (min(b["bbox"][0] for b in col2), max(b["bbox"][2] for b in col2)),
            ]
            return True, 2, x_ranges

    return False, 1, [(0.0, page_w)]


def classify_heading_block(
    block: dict[str, Any],
    median_body_size: float,
    page_h: float,
    config: LayoutConfig,
) -> tuple[bool, int, str]:
    """
    Classifies a text block as a Heading or non-heading.
    Requires at least two structural/layout signals beyond raw font size.
    Returns (is_heading, level, cleaned_text).
    """
    bbox = block["bbox"]
    # Reject running headers and footers
    if bbox[1] < config.header_margin_pt or bbox[3] > page_h - config.footer_margin_pt:
        return False, 0, ""

    spans = [s for l in block.get("lines", []) for s in l.get("spans", []) if s.get("text", "").strip()]
    if not spans:
        return False, 0, ""

    raw_text = " ".join(s["text"].strip() for s in spans).strip()
    words = raw_text.split()

    if len(words) > 16 or len(raw_text) > 140 or len(block.get("lines", [])) > 3:
        return False, 0, ""

    # Reject lowercase leading lines or ordinary prose ending with periods
    if raw_text and raw_text[0].islower():
        return False, 0, ""

    if raw_text.endswith(".") and not SECTION_NUMBER_PATTERN.match(raw_text) and len(words) > 5:
        return False, 0, ""

    # Reject standalone date lines
    if DATE_ONLY_PATTERN.match(raw_text):
        return False, 0, ""

    # Reject corporate signatory designations from signature blocks
    if SIGNATORY_PATTERN.search(raw_text):
        return False, 0, ""

    max_size = max(s.get("size", 0.0) for s in spans)
    is_bold = any(
        (
            "bold" in s.get("font", "").lower()
            or s.get("font", "").lower().endswith("-bd")
            or "-bd" in s.get("font", "").lower()
            or bool(s.get("flags", 0) & 2)
            or bool(s.get("flags", 0) & 16)
            or "heavy" in s.get("font", "").lower()
            or "black" in s.get("font", "").lower()
        )
        for s in spans
    )
    is_large = max_size >= median_body_size * config.heading_font_ratio
    is_all_caps = raw_text.isupper() and len(raw_text) > 4
    has_numbering = bool(SECTION_NUMBER_PATTERN.match(raw_text))
    is_short_title = len(words) <= 8 and (raw_text.istitle() or raw_text.endswith(":"))

    accepted = False
    if (is_large and (is_bold or is_all_caps or has_numbering)) or (
        is_bold and (has_numbering or is_short_title or is_all_caps)
    ) or (has_numbering and is_large):
        accepted = True

    if not accepted:
        return False, 0, ""

    # Determine hierarchical level (1 to 6)
    if max_size >= median_body_size * 1.8:
        level = 1
    elif max_size >= median_body_size * 1.4:
        level = 2
    elif max_size >= median_body_size * 1.2 or (is_bold and is_all_caps):
        level = 3
    elif is_bold:
        level = 4
    elif has_numbering:
        level = 5
    else:
        level = 6

    return True, level, raw_text


def classify_list_item_block(text: str) -> tuple[bool, str | None, str]:
    """
    Determines if cleaned block text starts with a bullet or list item marker.
    Returns (is_list, marker, remaining_text).
    """
    match = LIST_MARKER_PATTERN.match(text)
    if not match:
        return False, None, text

    marker = match.group(1).strip()
    remaining = match.group(2).strip()

    # Reject single punctuation without text
    if not remaining:
        return False, None, text

    return True, marker, remaining


def reconcile_table_captions_and_footnotes(
    tables: list[Table],
    blocks: list[dict[str, Any]],
    symbol_map: dict[tuple[str, int], str],
    text_punct_map: dict[int, str],
    document_id: str,
    page_number: int,
    page_h: float,
    prev_page_tables: list[Table] | None = None,
) -> tuple[list[Table], set[int]]:
    """
    Identifies captions immediately above, footnotes immediately below, and cross-page continuation.
    Enriches Table elements using model_copy and returns the updated tables and claimed block indices.
    """
    claimed_block_indices: set[int] = set()
    enriched_tables: list[Table] = []

    for table in tables:
        t_bbox = table.bbox
        new_caption = table.caption
        new_footnotes = list(table.footnotes)

        # Check cross-page continuation if unclosed statement preceded on prior page
        is_cont = table.is_continuation
        cont_from = table.continuation_from_page
        if not is_cont and prev_page_tables:
            prev_t = prev_page_tables[-1]
            if prev_t.page_number == page_number - 1:
                prev_text = " ".join(" ".join(r) for r in prev_t.rows[-3:]).lower()
                if any(
                    kw in prev_text
                    for kw in ["investing activities", "operating activities", "carried forward", "contd", "continued"]
                ):
                    is_cont = True
                    cont_from = prev_t.page_number

        # 1. Look for Caption above table
        if not new_caption:
            for idx, b in enumerate(blocks):
                if idx in claimed_block_indices or b.get("type") != 0:
                    continue
                # Skip header and footer margins
                if b["bbox"][1] < 40.0 or b["bbox"][3] > page_h - 40.0:
                    continue
                # A caption above the table must start above the table top boundary
                if b["bbox"][1] >= t_bbox[1] - 1.0:
                    continue
                by0, by1 = b["bbox"][1], b["bbox"][3]
                if t_bbox[1] - 35.0 <= by1 <= t_bbox[1] + 10.0 or (t_bbox[1] - 35.0 <= by0 < t_bbox[1]):
                    text = clean_block_spans_with_font_remapping(
                        b.get("lines", []), symbol_map, text_punct_map, document_id, page_number
                    )
                    words = text.split()
                    # Captions are concise labels, schedules, or unit indicators.
                    # Total summary rows and numeric lines are never captions:
                    # they carry the previous table's figures (see is_valid_table_caption).
                    is_caption = (
                        len(words) <= 10
                        and is_valid_table_caption(text)
                        and (
                            text.endswith(":")
                            or text.startswith(("Table", "Schedule", "("))
                            or text.istitle()
                        )
                    )
                    if is_caption:
                        new_caption = text
                        claimed_block_indices.add(idx)
                        break

        # 2. Look for Footnotes below table
        for idx, b in enumerate(blocks):
            if idx in claimed_block_indices or b.get("type") != 0:
                continue
            # Skip header and footer margins
            if b["bbox"][1] < 40.0 or b["bbox"][3] > page_h - 40.0:
                continue
            by0 = b["bbox"][1]
            if t_bbox[3] - 10.0 <= by0 <= t_bbox[3] + 45.0:
                text = clean_block_spans_with_font_remapping(
                    b.get("lines", []), symbol_map, text_punct_map, document_id, page_number
                )
                if FOOTNOTE_START_PATTERN.match(text):
                    if text not in new_footnotes:
                        new_footnotes.append(text)
                    claimed_block_indices.add(idx)

        # Apply enrichment via frozen model_copy
        if (
            new_caption != table.caption
            or new_footnotes != table.footnotes
            or is_cont != table.is_continuation
            or cont_from != table.continuation_from_page
        ):
            enriched_tables.append(
                table.model_copy(
                    update={
                        "caption": new_caption,
                        "footnotes": new_footnotes,
                        "is_continuation": is_cont,
                        "continuation_from_page": cont_from,
                    }
                )
            )
        else:
            enriched_tables.append(table)

    return enriched_tables, claimed_block_indices


def sort_elements_reading_order(
    text_elements: list[DocumentElement],
    tables: list[Table],
    is_multi_column: bool,
    column_count: int,
    col_ranges: list[tuple[float, float]],
    page_w: float,
    page_h: float,
) -> list[DocumentElement]:
    """
    Combines text elements and table elements into visual reading order.
    For single-column: sorts by y0 (with x0 tie-breaking).
    For multi-column: groups elements into columns, sorting by y0 within each column,
    while placing spanning elements (width >= 0.65 content width) at their vertical positions.
    """
    all_elements: list[DocumentElement] = list(text_elements) + list(tables)
    if not is_multi_column or column_count <= 1:
        return sorted(all_elements, key=lambda e: (e.bbox[1], e.bbox[0]))

    content_w = page_w - 80.0
    spanning = [e for e in all_elements if (e.bbox[2] - e.bbox[0]) >= content_w * 0.65]
    spanning.sort(key=lambda e: e.bbox[1])

    non_spanning = [e for e in all_elements if e not in spanning]

    def get_column_index(elem: DocumentElement) -> int:
        mid_x = (elem.bbox[0] + elem.bbox[2]) / 2.0
        best_col = 0
        min_dist = float("inf")
        for i, (cx0, cx1) in enumerate(col_ranges):
            col_mid = (cx0 + cx1) / 2.0
            dist = abs(mid_x - col_mid)
            if dist < min_dist:
                min_dist = dist
                best_col = i
        return best_col

    cuts = [0.0] + [s.bbox[1] for s in spanning] + [page_h]
    ordered: list[DocumentElement] = []

    for z in range(len(cuts) - 1):
        z_top = cuts[z]
        z_bot = cuts[z + 1]

        z_elems = [e for e in non_spanning if z_top <= e.bbox[1] < z_bot]

        for c_idx in range(column_count):
            c_elems = [e for e in z_elems if get_column_index(e) == c_idx]
            c_elems.sort(key=lambda e: e.bbox[1])
            ordered.extend(c_elems)

        if z < len(spanning):
            ordered.append(spanning[z])

    return ordered


def propagate_section_headings(
    elements: list[DocumentElement],
    initial_heading_stack: list[tuple[int, str]] | None = None,
) -> tuple[list[DocumentElement], list[tuple[int, str]]]:
    """
    Propagates hierarchical section heading context to all document elements.
    Maintains a stack of (level, heading_text).
    Updates element.section_heading with the nearest applicable heading and
    records element.metadata['heading_path'] containing the complete hierarchy.
    """
    heading_stack: list[tuple[int, str]] = list(initial_heading_stack or [])
    propagated: list[DocumentElement] = []

    for elem in elements:
        if isinstance(elem, Heading):
            # Equal or higher-level heading terminates deeper nested scope
            while heading_stack and heading_stack[-1][0] >= elem.level:
                heading_stack.pop()
            heading_stack.append((elem.level, elem.text))

            parent_heading = heading_stack[-2][1] if len(heading_stack) > 1 else None
            current_path = [h[1] for h in heading_stack]

            new_meta = dict(elem.metadata)
            new_meta["heading_path"] = current_path
            propagated.append(
                elem.model_copy(
                    update={
                        "section_heading": parent_heading,
                        "metadata": new_meta,
                    }
                )
            )
        else:
            current_heading = heading_stack[-1][1] if heading_stack else None
            current_path = [h[1] for h in heading_stack]
            new_meta = dict(elem.metadata)
            if current_path:
                new_meta["heading_path"] = current_path
            propagated.append(
                elem.model_copy(
                    update={
                        "section_heading": current_heading,
                        "metadata": new_meta,
                    }
                )
            )

    return propagated, heading_stack


def reconcile_page_elements(
    page: pymupdf.Page,
    page_number: int,
    document_name: str,
    tables: list[Table] | None = None,
    prev_page_tables: list[Table] | None = None,
    initial_heading_stack: list[tuple[int, str]] | None = None,
    config: LayoutConfig | None = None,
    config_path: Path | str | None = None,
) -> tuple[list[DocumentElement], PageLayoutReport, list[tuple[int, str]]]:
    """
    Reconciles all textual blocks and tabular data on a single PDF page.

    Performs spatial partitioning, caption/footnote attachment, multi-column reading order
    sorting, and section heading propagation. Evaluates an audit-grade reconciliation report.

    Args:
        page: PyMuPDF Page object.
        page_number: 1-indexed physical page number.
        document_name: Source document filename.
        tables: Optional pre-extracted Table elements. If None, extracted locally via table_extractor.
        initial_heading_stack: Optional active heading hierarchy from previous page.
        config: Optional LayoutConfig heuristics overrides.
        config_path: Optional path to symbol font mappings YAML.

    Returns:
        tuple of (ordered_elements, layout_report, updated_heading_stack).
    """
    cfg = config or LayoutConfig()
    page_w = page.rect.width
    page_h = page.rect.height

    raw_text = page.get_text("text")  # type: ignore[no-untyped-call]
    raw_character_count = len(raw_text)

    # 1. Obtain tables if not provided
    if tables is None:
        tables = extract_tables_from_page(
            page,
            page_number=page_number,
            document_name=document_name,
            enable_tier2=False,
        )

    # 2. Extract layout blocks
    page_dict = page.get_text("dict")  # type: ignore[no-untyped-call]
    raw_blocks = page_dict.get("blocks", [])
    text_blocks = [b for b in raw_blocks if b.get("type") == 0]
    source_block_count = len(text_blocks)

    symbol_map, text_punct_map = load_symbol_mappings(config_path)

    # Calculate median body text size
    all_sizes = [
        s.get("size", 0.0)
        for b in text_blocks
        for l in b.get("lines", [])
        for s in l.get("spans", [])
        if len(s.get("text", "").strip()) > 3
    ]
    median_body_size = (
        sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else 9.0
    )

    # 3. Reconcile Table Captions and Footnotes
    enriched_tables, claimed_by_tables = reconcile_table_captions_and_footnotes(
        tables=tables,
        blocks=text_blocks,
        symbol_map=symbol_map,
        text_punct_map=text_punct_map,
        document_id=document_name,
        page_number=page_number,
        page_h=page_h,
        prev_page_tables=prev_page_tables,
    )

    # 4. Spatial Partitioning against Table Bounding Boxes
    table_bboxes = [t.bbox for t in enriched_tables]
    owned_by_table: set[int] = set(claimed_by_tables)

    for idx, block in enumerate(text_blocks):
        if idx in owned_by_table:
            continue
        bbox = tuple(block["bbox"])
        for tb in table_bboxes:
            if compute_bbox_overlap_ratio(bbox, tb) >= cfg.table_overlap_threshold:
                owned_by_table.add(idx)
                break

    # 5. Detect Multi-Column Layout
    is_multi_column, col_count, col_ranges = detect_multi_column_layout(
        blocks=text_blocks,
        page_w=page_w,
        page_h=page_h,
        table_bboxes=table_bboxes,
        config=cfg,
    )

    # 6. Classify Non-Table Blocks
    owned_by_heading: set[int] = set()
    owned_by_list: set[int] = set()
    owned_by_narrative: set[int] = set()

    text_elements: list[DocumentElement] = []

    for idx, block in enumerate(text_blocks):
        if idx in owned_by_table:
            continue

        bbox = tuple(block["bbox"])
        # Filter out lines that fall inside any table bbox to prevent duplicate table text
        outside_lines: list[dict[str, Any]] = []
        for line in block.get("lines", []):
            line_bbox = tuple(line["bbox"])
            line_in_table = any(
                compute_bbox_overlap_ratio(line_bbox, tb) >= cfg.table_overlap_threshold
                for tb in table_bboxes
            )
            if not line_in_table:
                outside_lines.append(line)

        if not outside_lines:
            owned_by_table.add(idx)
            continue

        if len(outside_lines) < len(block.get("lines", [])):
            bbox = (
                min(l["bbox"][0] for l in outside_lines),
                min(l["bbox"][1] for l in outside_lines),
                max(l["bbox"][2] for l in outside_lines),
                max(l["bbox"][3] for l in outside_lines),
            )

        cleaned_text = clean_block_spans_with_font_remapping(
            outside_lines, symbol_map, text_punct_map, document_name, page_number
        )

        if not cleaned_text:
            owned_by_narrative.add(idx)
            continue

        sub_block = {"bbox": bbox, "lines": outside_lines}

        # Check for running headers or footers
        is_running_header = bbox[1] < cfg.header_margin_pt
        is_running_footer = bbox[3] > page_h - cfg.footer_margin_pt

        if is_running_header or is_running_footer:
            narr = NarrativeText(
                element_id=f"{document_name}_p{page_number}_hdr_ftr_{idx}",
                element_type="narrative",
                page_number=page_number,
                bbox=bbox,
                document_name=document_name,
                text=cleaned_text,
                metadata={"is_header_footer": True},
            )
            text_elements.append(narr)
            owned_by_narrative.add(idx)
            continue

        # Check for Heading
        is_h, h_lvl, _ = classify_heading_block(sub_block, median_body_size, page_h, cfg)
        if is_h:
            heading = Heading(
                element_id=f"{document_name}_p{page_number}_h_{idx}",
                element_type="heading",
                page_number=page_number,
                bbox=bbox,
                document_name=document_name,
                text=cleaned_text,
                level=h_lvl,
            )
            text_elements.append(heading)
            owned_by_heading.add(idx)
            continue

        # Check for ListItem
        is_li, li_marker, li_text = classify_list_item_block(cleaned_text)
        if is_li:
            list_item = ListItem(
                element_id=f"{document_name}_p{page_number}_li_{idx}",
                element_type="list_item",
                page_number=page_number,
                bbox=bbox,
                document_name=document_name,
                text=li_text,
                marker=li_marker,
            )
            text_elements.append(list_item)
            owned_by_list.add(idx)
            continue

        # Default to NarrativeText
        narr = NarrativeText(
            element_id=f"{document_name}_p{page_number}_narr_{idx}",
            element_type="narrative",
            page_number=page_number,
            bbox=bbox,
            document_name=document_name,
            text=cleaned_text,
        )
        text_elements.append(narr)
        owned_by_narrative.add(idx)

    # 7. Block Ownership Verification
    all_owned = owned_by_table | owned_by_heading | owned_by_list | owned_by_narrative
    unowned_count = source_block_count - len(all_owned)

    # Multiply owned count audit
    multiply_owned_count = 0
    for idx in range(source_block_count):
        categories_count = (
            (1 if idx in owned_by_table else 0)
            + (1 if idx in owned_by_heading else 0)
            + (1 if idx in owned_by_list else 0)
            + (1 if idx in owned_by_narrative else 0)
        )
        if categories_count > 1:
            multiply_owned_count += 1

    # 8. Sort Elements into Visual Reading Order
    ordered_elements = sort_elements_reading_order(
        text_elements=text_elements,
        tables=enriched_tables,
        is_multi_column=is_multi_column,
        column_count=col_count,
        col_ranges=col_ranges,
        page_w=page_w,
        page_h=page_h,
    )

    # 9. Propagate Section Heading Context
    final_elements, updated_stack = propagate_section_headings(
        ordered_elements, initial_heading_stack=initial_heading_stack
    )

    # 10. Structured Character Count Reconciliation
    struct_char_count = 0
    for elem in final_elements:
        if isinstance(elem, Table):
            struct_char_count += sum(len(c) for r in elem.rows for c in r)
            struct_char_count += sum(len(h) for h in elem.headers)
            struct_char_count += sum(len(f) for f in elem.footnotes)
            if elem.caption:
                struct_char_count += len(elem.caption)
        elif isinstance(elem, Heading):
            struct_char_count += len(elem.text)
        elif isinstance(elem, ListItem):
            struct_char_count += len(elem.text) + (len(elem.marker) if elem.marker else 0)
        elif isinstance(elem, NarrativeText):
            struct_char_count += len(elem.text)

    reconciliation_delta_pct = (
        abs(struct_char_count - raw_character_count) / max(raw_character_count, 1) * 100.0
    )

    # Audit & Review Reasons
    review_reasons: list[str] = []
    if reconciliation_delta_pct > cfg.reconciliation_tolerance_pct:
        review_reasons.append(
            f"character_reconciliation_delta_exceeded ({reconciliation_delta_pct:.2f}% > {cfg.reconciliation_tolerance_pct:.2f}%)"
        )
    if unowned_count > 0:
        review_reasons.append(f"unowned_source_blocks ({unowned_count})")
    if multiply_owned_count > 0:
        review_reasons.append(f"multiply_owned_source_blocks ({multiply_owned_count})")

    # Propagate table review flags if any table requires human review
    for t in enriched_tables:
        if t.needs_review:
            review_reasons.extend(t.review_reasons)

    needs_review = len(review_reasons) > 0

    report = PageLayoutReport(
        page_number=page_number,
        raw_character_count=raw_character_count,
        structured_character_count=struct_char_count,
        reconciliation_delta_pct=round(reconciliation_delta_pct, 2),
        table_count=len(enriched_tables),
        heading_count=len([e for e in final_elements if isinstance(e, Heading)]),
        list_item_count=len([e for e in final_elements if isinstance(e, ListItem)]),
        narrative_count=len([e for e in final_elements if isinstance(e, NarrativeText)]),
        multi_column=is_multi_column,
        column_count=col_count,
        source_block_count=source_block_count,
        owned_block_count=len(all_owned),
        unowned_block_count=unowned_count,
        multiply_owned_block_count=multiply_owned_count,
        needs_review=needs_review,
        review_reasons=review_reasons,
    )

    return final_elements, report, updated_stack
