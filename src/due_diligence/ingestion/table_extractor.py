"""
Three-tier financial table extraction pipeline with quality gate validation.

1. Purpose:
   Provides robust, audit-grade table extraction for corporate diligence documents.
   Implements a 3-tier architecture:
   - Tier 1: Local PyMuPDF column-projection extraction (fast, deterministic, zero-cost).
   - Tier 2: Multimodal LLM vision fallback (Google Gemini Flash) for degraded, borderless,
             dense multi-column, or multi-page continuation tables.
   - Tier 3: Graceful degradation with audit flags ('needs_review') when both tiers fail.

2. Role in Overall RAG Pipeline:
   Acts as the table extraction engine during document ingestion. Converts raw page
   layouts into structured `Table` elements with exact Markdown representations,
   capturing captions, footnotes, and provenance before chunking and vector indexing.

3. Expected Inputs:
   - PyMuPDF Page object, 1-indexed page number, document name, and prior page tables.
   - Optional GEMINI_API_KEY environment variable for Tier 2 multimodal vision fallback.

4. Outputs:
   - List of strongly typed, validated `Table` elements (from `due_diligence.ingestion.elements`)
     carrying confidence scores, extraction methods, status flags, and provenance bounding boxes.

5. Dependencies:
   - pymupdf (fitz)
   - pydantic
   - requests (for Tier 2 Gemini API calls)
   - python-dotenv
   - due_diligence.ingestion.elements (Table)

6. Non-Responsibilities:
   - Does not perform vector embedding generation or store chunks into Weaviate.
   - Does not extract narrative text paragraphs or document headings (handled in parser.py).
   - Does not perform ad-hoc question answering or due diligence synthesis.
"""

import base64
import json
import logging
import os
import re
import time
from typing import Any

import pymupdf
import requests
from dotenv import load_dotenv

from due_diligence.ingestion.elements import TOTAL_ROW_PATTERN, Table
from due_diligence.ingestion.font_mappings import load_symbol_mappings

load_dotenv()
logger = logging.getLogger(__name__)

# Valid 1-2 letter lowercase tokens in English & corporate reporting
VALID_LOWER_SHORT = {
    "a", "i", "an", "am", "in", "on", "at", "by", "of", "or", "as", "is", "it", "to",
    "if", "do", "so", "up", "us", "we", "my", "he", "me", "no", "be", "go",
    "sr", "dr", "mr", "ms", "rs", "cr", "fy", "q1", "q2", "q3", "q4", "co", "bs", "pl",
    "re", "ex", "id", "na", "st", "nd", "rd", "th", "yr", "kg", "sc", "ca", "cs", "cx",
    "ai", "pg", "rp", "vc", "gj", "al", "ge", "pm", "dg", "rm", "fd", "fx", "dp", "md",
    "ir", "qr", "hr", "vp", "ed", "gm", "ag", "nb", "fc", "lc", "bg", "od", "cc",
    "td", "cd", "cp", "dd", "po", "ro", "ho", "bo", "zo", "mo", "yo", "eu", "uk",
    "un", "v", "x", "l", "c", "d", "m", "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii"
}

KNOWN_SPLINTERS = {
    "sto", "ck", "tio", "ns", "anc", "ial", "tat", "eme", "nt", "ect", "ion",
    "pro", "vis", "sta", "tem", "par", "tic", "ula", "rs"
}

# A trailing summary row (e.g. "Total 34,179,458,972 26,474,170,018").
# Such rows frequently sit below the table's last horizontal rule with no rule
# beneath them, so region detection clips them and Tier-1's strict word
# containment then drops them entirely. Canonical pattern lives in elements.py
# (TOTAL_ROW_PATTERN) and is used by extend_box_for_trailing_total below.


def detect_all_table_regions(page: pymupdf.Page) -> list[tuple[float, float, float, float]]:
    """
    Detect all distinct tabular regions on a page.
    Handles both bordered tables (via merged line rules) and borderless tables
    (via vertical numeric column alignment and row clustering).
    """
    page_w = page.rect.width
    page_h = page.rect.height

    # 1. Identify vector line drawings and thin rects
    drawings = page.get_drawings()
    h_lines = []
    for d in drawings:
        for item in d["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 1.0 and abs(p1.x - p2.x) > 15:
                    h_lines.append((min(p1.x, p2.x), p1.y, max(p1.x, p2.x), p2.y))
            elif item[0] == "re":
                r = item[1]
                if r.height < 3.0 and r.width > 15:
                    h_lines.append((r.x0, r.y0, r.x1, r.y1))

    # Merge collinear segments at the same y into logical rules
    y_groups: list[dict[str, Any]] = []
    for l in sorted(h_lines, key=lambda x: x[1]):
        matched = False
        for g in y_groups:
            if abs(l[1] - g["y"]) < 1.5:
                g["lines"].append(l)
                g["min_x"] = min(g["min_x"], l[0])
                g["max_x"] = max(g["max_x"], l[2])
                g["y"] = sum(x[1] for x in g["lines"]) / len(g["lines"])
                matched = True
                break
        if not matched:
            y_groups.append({"y": l[1], "lines": [l], "min_x": l[0], "max_x": l[2]})

    # Wide rules: total horizontal span > 200 pt and within page margins
    wide_rules = [g for g in y_groups if (g["max_x"] - g["min_x"]) > 200 and 35 < g["y"] < page_h - 35]

    # Cluster wide rules into vertical regions (gap > 45 pt splits into separate tables)
    line_regions: list[list[dict[str, Any]]] = []
    if wide_rules:
        curr = [wide_rules[0]]
        for r in wide_rules[1:]:
            if (r["y"] - curr[-1]["y"]) <= 45:
                curr.append(r)
            else:
                line_regions.append(curr)
                curr = [r]
        if curr:
            line_regions.append(curr)

    # 2. Extract words and group into visual lines
    words = page.get_text("words")  # type: ignore[no-untyped-call]
    visual_lines: list[dict[str, Any]] = []
    for w in [w for w in words if 40 < w[1] < page_h - 40]:
        y_mid = (w[1] + w[3]) / 2
        matched = False
        for vl in visual_lines:
            if abs(y_mid - vl["y_mid"]) < 4.0:
                vl["words"].append(w)
                vl["y_mid"] = sum((x[1] + x[3]) / 2 for x in vl["words"]) / len(vl["words"])
                matched = True
                break
        if not matched:
            visual_lines.append({"y_mid": y_mid, "words": [w]})
    visual_lines.sort(key=lambda l: l["y_mid"])

    # Convert line regions to candidate table bboxes
    candidate_boxes: list[tuple[float, float, float, float]] = []
    for lr in line_regions:
        if len(lr) >= 2:
            y_top = max(40.0, lr[0]["y"] - 35.0)
            y_bot = min(page_h - 40.0, lr[-1]["y"] + 10.0)
            candidate_boxes.append((40.0, y_top, page_w - 40.0, y_bot))
        elif len(lr) == 1:
            rule_y = lr[0]["y"]
            subseq_multi_num = [
                vl for vl in visual_lines
                if rule_y <= vl["y_mid"] <= rule_y + 250
                and sum(1 for w in vl["words"] if any(c.isdigit() for c in w[4])) >= 2
            ]
            if len(subseq_multi_num) >= 3:
                y_top = max(40.0, rule_y - 30.0)
                y_bot = min(page_h - 40.0, max(w[3] for vl in subseq_multi_num for w in vl["words"]) + 15.0)
                candidate_boxes.append((40.0, y_top, page_w - 40.0, y_bot))

    # 3. Check for borderless tables not covered by line regions
    multi_num_lines = [vl for vl in visual_lines if sum(1 for w in vl["words"] if any(c.isdigit() for c in w[4])) >= 2]
    if len(multi_num_lines) >= 3:
        num_line_clusters: list[list[dict[str, Any]]] = []
        curr_cl = [multi_num_lines[0]]
        for nl in multi_num_lines[1:]:
            if (nl["y_mid"] - curr_cl[-1]["y_mid"]) <= 35:
                curr_cl.append(nl)
            else:
                num_line_clusters.append(curr_cl)
                curr_cl = [nl]
        if curr_cl:
            num_line_clusters.append(curr_cl)

        for cl in num_line_clusters:
            if len(cl) >= 3:
                cl_top = max(40.0, cl[0]["y_mid"] - 30.0)
                cl_bot = min(page_h - 40.0, cl[-1]["y_mid"] + 20.0)
                overlaps = False
                for b in candidate_boxes:
                    inter_top = max(cl_top, b[1])
                    inter_bot = min(cl_bot, b[3])
                    if inter_bot > inter_top and (inter_bot - inter_top) > 0.5 * (cl_bot - cl_top):
                        overlaps = True
                        break
                if not overlaps:
                    candidate_boxes.append((40.0, cl_top, page_w - 40.0, cl_bot))

    # Sort boxes by y_top and merge overlapping boxes
    candidate_boxes.sort(key=lambda b: b[1])
    merged_boxes: list[tuple[float, float, float, float]] = []
    for b in candidate_boxes:
        if not merged_boxes:
            merged_boxes.append(b)
        else:
            prev = merged_boxes[-1]
            if b[1] <= prev[3] - 10:
                merged = (min(prev[0], b[0]), min(prev[1], b[1]), max(prev[2], b[2]), max(prev[3], b[3]))
                merged_boxes[-1] = merged
            else:
                merged_boxes.append(b)

    # Extend each box downward to swallow a trailing Total row that sits below
    # the last horizontal rule (otherwise Tier-1 clips it and the orphaned Total
    # line gets misattached as the NEXT table's caption downstream).
    merged_boxes = [
        extend_box_for_trailing_total(b, visual_lines, page_h) for b in merged_boxes
    ]
    # Clamp extensions so no box bleeds into the next box's top edge
    clamped: list[tuple[float, float, float, float]] = []
    for i, b in enumerate(merged_boxes):
        y_bot = b[3]
        if i + 1 < len(merged_boxes):
            y_bot = min(y_bot, merged_boxes[i + 1][1] - 1.0)
        clamped.append((b[0], b[1], b[2], max(y_bot, b[1] + 5.0)))

    return clamped


def extend_box_for_trailing_total(
    box: tuple[float, float, float, float],
    visual_lines: list[dict[str, Any]],
    page_h: float,
) -> tuple[float, float, float, float]:
    """
    Extend a table bounding box downward to include a trailing Total row.

    Scans visual lines just below the box bottom for a Total summary line with
    numeric content. If found, the box bottom is stretched to cover it so that
    Tier-1 column-projection keeps the Total row inside the extracted table.

    Returns the (possibly unchanged) box. Logs a warning when a Total-like line
    is visible further below but beyond safe reach, so the gap stays observable.
    """
    _, _, _, y_bot = box
    nearest_total: dict[str, Any] | None = None
    nearest_bottom = 0.0

    for vl in visual_lines:
        y_mid = vl["y_mid"]
        if not (y_bot - 12.0 < y_mid <= y_bot + 50.0):
            continue
        words = sorted(vl["words"], key=lambda w: w[0])
        text = " ".join(w[4] for w in words)
        if TOTAL_ROW_PATTERN.match(text) and any(any(c.isdigit() for c in w[4]) for w in words):
            line_bottom = max(w[3] for w in words)
            if nearest_total is None or y_mid < nearest_total["y_mid"]:
                nearest_total = vl
                nearest_bottom = line_bottom

    if nearest_total is not None:
        new_bot = min(page_h - 40.0, nearest_bottom + 8.0)
        if new_bot > y_bot:
            return (box[0], box[1], box[2], new_bot)
        return box

    # Observability: a Total line further below likely belongs to this table
    for vl in visual_lines:
        y_mid = vl["y_mid"]
        if not (y_bot + 50.0 < y_mid <= y_bot + 150.0):
            continue
        text = " ".join(w[4] for w in sorted(vl["words"], key=lambda w: w[0]))
        if TOTAL_ROW_PATTERN.match(text) and any(c.isdigit() for c in text):
            logger.warning(
                "Trailing Total line beyond safe reach (box bottom %.1f, line y %.1f): %s",
                y_bot,
                y_mid,
                text[:80],
            )
            break

    return box


def extract_table_tier1(page: pymupdf.Page, bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    """
    Tier 1 Column-Projection Table Extractor.
    Derives column dividers from vertical number clusters and lines up word baselines.
    """
    x0_t, y0_t, x1_t, y1_t = bbox
    all_words = page.get_text("words")  # type: ignore[no-untyped-call]
    words = [w for w in all_words if x0_t <= w[0] and w[2] <= x1_t and y0_t <= w[1] and w[3] <= y1_t]

    if not words:
        return {"col_count": 0, "row_count": 0, "headers": [], "rows": [], "words": [], "col_dividers": [], "bbox": bbox}

    # Truncate signature blocks at bottom if detected
    sig_words = [w for w in words if any(kw in w[4] for kw in ["Chartered", "Accountants", "Batliboi", "BSR", "Managing", "Director", "Company", "Secretary"])]
    sig_header = [w for w in words if "report" in w[4].lower() and w[1] > y0_t + 150]
    if sig_header:
        min_sig_y = min(w[1] for w in sig_header)
        words = [w for w in words if w[3] <= min_sig_y - 5]
    elif len(sig_words) >= 4:
        bottom_sigs = [w for w in sig_words if w[1] > y0_t + (y1_t - y0_t) * 0.65]
        if len(bottom_sigs) >= 3:
            min_sig_y = min(w[1] for w in bottom_sigs)
            words = [w for w in words if w[3] <= min_sig_y - 5]

    min_y = min(w[1] for w in words)
    number_words = [w for w in words if any(c.isdigit() for c in w[4]) and w[1] > min_y + 20]

    num_x_centers = sorted([(w[0] + w[2]) / 2 for w in number_words])
    col_clusters: list[dict[str, Any]] = []
    for x in num_x_centers:
        if not col_clusters or (x - col_clusters[-1]["mean"]) > 25:
            col_clusters.append({"mean": x, "count": 1, "points": [x]})
        else:
            col_clusters[-1]["points"].append(x)
            col_clusters[-1]["mean"] = sum(col_clusters[-1]["points"]) / len(col_clusters[-1]["points"])
            col_clusters[-1]["count"] += 1

    valid_num_cols = [c for c in col_clusters if c["count"] >= 3]

    words.sort(key=lambda w: (w[1], w[0]))
    lines: list[dict[str, Any]] = []
    for w in words:
        y_mid = (w[1] + w[3]) / 2
        matched = False
        for line in lines:
            if abs(y_mid - line["y_mid"]) < 4.0:
                line["words"].append(w)
                line["y_mid"] = sum((x[1] + x[3]) / 2 for x in line["words"]) / len(line["words"])
                matched = True
                break
        if not matched:
            lines.append({"y_mid": y_mid, "words": [w]})

    lines.sort(key=lambda l: l["y_mid"])

    # Robust phrase-segment and whitespace gutter column projection
    col_dividers = [x0_t]
    if valid_num_cols:
        num_means = [c["mean"] for c in valid_num_cols]
        M = len(num_means)

        # Break lines into phrase segments (words with gap < 7.0 pt belong to same cell)
        segments: list[dict[str, float]] = []
        for line in lines:
            lw = sorted(line["words"], key=lambda w: w[0])
            curr = [lw[0]]
            for w in lw[1:]:
                if w[0] - curr[-1][2] < 7.0:
                    curr.append(w)
                else:
                    segments.append({"x0": curr[0][0], "x1": curr[-1][2], "mid": (curr[0][0] + curr[-1][2]) / 2.0})
                    curr = [w]
            if curr:
                segments.append({"x0": curr[0][0], "x1": curr[-1][2], "mid": (curr[0][0] + curr[-1][2]) / 2.0})

        # Track spans for data columns vs label column
        data_col_spans = [{"x0": 9999.0, "x1": 0.0} for _ in range(M)]
        label_max_x = x0_t

        for seg in segments:
            seg_mid = seg["mid"]
            closest_k = min(range(M), key=lambda k: abs(seg_mid - num_means[k]))
            dist = abs(seg_mid - num_means[closest_k])

            if seg["x1"] < num_means[0] - 25 and dist > 45:
                label_max_x = max(label_max_x, seg["x1"])
            else:
                data_col_spans[closest_k]["x0"] = min(data_col_spans[closest_k]["x0"], seg["x0"])
                data_col_spans[closest_k]["x1"] = max(data_col_spans[closest_k]["x1"], seg["x1"])

        first_data_x0 = data_col_spans[0]["x0"] if data_col_spans[0]["x0"] < 9000 else num_means[0] - 20
        if first_data_x0 > label_max_x + 6.0:
            div_0 = (label_max_x + first_data_x0) / 2.0
        else:
            div_0 = max(x0_t + 20, min(first_data_x0 - 5.0, num_means[0] - 25))
        col_dividers.append(div_0)

        for k in range(M - 1):
            right_k = data_col_spans[k]["x1"] if data_col_spans[k]["x1"] > 0 else num_means[k] + 10
            left_k1 = data_col_spans[k + 1]["x0"] if data_col_spans[k + 1]["x0"] < 9000 else num_means[k + 1] - 10
            if left_k1 > right_k + 4.0:
                div_k = (right_k + left_k1) / 2.0
            else:
                div_k = (num_means[k] + num_means[k + 1]) / 2.0
            col_dividers.append(div_k)

    col_dividers.append(x1_t)
    col_dividers = sorted(set(col_dividers))

    # Anti-collision check: ensure NO divider cuts through any word bounding box
    safe_dividers = [x0_t]
    for div in col_dividers[1:-1]:
        cutting = [w for w in words if w[0] + 0.5 < div < w[2] - 0.5]
        if cutting:
            w_cut = cutting[0]
            if w_cut[0] - 2.0 > safe_dividers[-1] + 5.0:
                safe_dividers.append(w_cut[0] - 2.0)
            else:
                safe_dividers.append(w_cut[2] + 2.0)
        else:
            safe_dividers.append(div)
    safe_dividers.append(x1_t)
    col_dividers = sorted(set(safe_dividers))

    num_cols = max(1, len(col_dividers) - 1)

    _, punct_map = load_symbol_mappings()

    def clean_cell_text(cell: str) -> str:
        return "".join(punct_map.get(ord(c), c) for c in cell).strip()

    raw_matrix = []
    for line in lines:
        row = ["" for _ in range(num_cols)]
        for w in line["words"]:
            x_mid = (w[0] + w[2]) / 2
            col_idx = 0
            for i in range(num_cols):
                if col_dividers[i] <= x_mid < col_dividers[i + 1]:
                    col_idx = i
                    break
            if row[col_idx]:
                row[col_idx] += " " + w[4]
            else:
                row[col_idx] = w[4]
        raw_matrix.append([clean_cell_text(c) for c in row])

    # Multi-line header vs data rows detection
    def is_header_or_unit_line(row: list[str]) -> bool:
        full_text = " ".join(row).lower()
        if re.search(r"\b(in\s+['`’]?000|in\s+crore|in\s+lakh|%\s+of\s+rwa)\b", full_text):
            return True
        data_text = " ".join(row[1:]).lower()
        if re.search(r"\b(as at|year ended|schedule|particulars|march \d+|march, \d+)\b", data_text):
            return True
        has_digits = any(any(c.isdigit() for c in cell) for cell in row[1:])
        if not has_digits:
            return True
        tokens = [tok for c in row[1:] for tok in re.findall(r"\b\d+[\d,.]*\b", c)]
        return bool(
            all(tok in {"2024", "2025", "2026", "2027", "31", "30", "1", "2", "3"} for tok in tokens)
            and any(kw in data_text for kw in ["march", "as at", "ended"])
        )

    first_data_idx = -1
    for idx, row in enumerate(raw_matrix):
        if not is_header_or_unit_line(row):
            first_data_idx = idx
            break

    if first_data_idx == -1:
        first_data_idx = 1 if len(raw_matrix) > 1 else 0

    header_lines = raw_matrix[:first_data_idx] if first_data_idx > 0 else (raw_matrix[:1] if raw_matrix else [])
    data_lines = raw_matrix[first_data_idx:] if first_data_idx > 0 else (raw_matrix[1:] if len(raw_matrix) > 1 else [])

    merged_headers = ["" for _ in range(num_cols)]
    for hline in header_lines:
        for c in range(num_cols):
            val = hline[c].strip()
            if val:
                merged_headers[c] = (merged_headers[c] + " " + val).strip()

    merged_data_lines = []
    i = 0
    while i < len(data_lines):
        curr_row = data_lines[i]
        while i + 1 < len(data_lines):
            next_row = data_lines[i + 1]
            next_has_data = any(next_row[c] for c in range(1, num_cols))
            curr_has_data = any(curr_row[c] for c in range(1, num_cols))
            if curr_has_data and not next_has_data and next_row[0]:
                curr_row[0] += " " + next_row[0]
                i += 1
            elif not curr_has_data and next_has_data and curr_row[0] and not any(curr_row[c] for c in range(1, num_cols)):
                next_row[0] = curr_row[0] + " " + next_row[0]
                curr_row = next_row
                i += 1
            else:
                break
        merged_data_lines.append(curr_row)
        i += 1

    headers = [clean_cell_text(c) for c in merged_headers]
    rows = [[clean_cell_text(c) for c in r] for r in merged_data_lines]

    return {
        "col_count": num_cols,
        "row_count": len(rows),
        "headers": headers,
        "rows": rows,
        "words": words,
        "col_dividers": col_dividers,
        "bbox": bbox,
    }


def detect_fragmented_words(rows: list[list[str]]) -> list[str]:
    """
    Identifies genuine word splinters caused by improper column splitting.
    Excludes valid English short words, corporate acronyms, single-letter column headers,
    and possessives.
    """
    frags = []
    for r in rows:
        for cell in r:
            cleaned = re.sub(r"['’\u2018\u2019\ufffd]s\b", "", cell)
            cleaned = re.sub(r"\([a-zA-Z0-9]{1,3}\)", " ", cleaned)
            cleaned = re.sub(r"\b[A-Za-z]\.", " ", cleaned)

            tokens = re.findall(r"\b[A-Za-z]+\b", cleaned)
            for tok in tokens:
                if len(tok) == 1 and tok.isupper():
                    continue
                if len(tok) == 2 and tok.isupper():
                    continue
                low = tok.lower()
                if low in KNOWN_SPLINTERS:
                    frags.append(tok)
                elif len(low) == 1:
                    if low not in {"a", "i", "v", "x", "l", "c", "d", "m"}:
                        frags.append(tok)
                elif len(low) == 2 and low not in VALID_LOWER_SHORT:
                    frags.append(tok)
    return frags


def detect_merged_cell_anomalies(rows: list[list[str]]) -> list[str]:
    """
    Detects merged column artifacts within data cells (e.g. number immediately
    followed by a text word, such as '17 Present' on Page 343).
    """
    anomalies = []
    if len(rows) <= 1:
        return []
    for r in rows:
        for c in r[1:]:
            matches = re.findall(r"\b\d+\s+[A-Za-z]{3,}\b", c)
            anomalies.extend(matches)
    return anomalies


def evaluate_quality_gate(
    table_result: dict[str, Any],
    page: pymupdf.Page,
    prev_table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Evaluates the 5 Quality Gate checks against Tier 1 output:
    1. col_count >= 2
    2. row_count >= 2
    3. row_consistency >= 85%
    4. zero fragmented words & zero merged cell anomalies
    5. page continuation detection
    """
    col_count = table_result.get("col_count", 0)
    rows = table_result.get("rows", [])
    row_count = len(rows)

    failures: list[str] = []

    # Check 1: Col count >= 2
    if col_count < 2:
        failures.append(f"col_count ({col_count}) < 2")

    # Check 2: Row count >= 2
    if row_count < 2:
        failures.append(f"row_count ({row_count}) < 2")

    # Check 3: Row consistency >= 85%
    if rows and col_count >= 2:
        valid_rows = 0
        for r in rows:
            has_label = bool(r[0].strip() or (len(r) > 1 and r[1].strip())) if r else False
            has_data = any(bool(c.strip()) for c in r[1:]) if len(r) > 1 else False
            label_len = len(r[0] + (r[1] if len(r) > 1 else "")) if r else 0
            if (has_label and has_data) or (has_label and not has_data and label_len >= 3):
                valid_rows += 1
        consistency = (valid_rows / len(rows)) * 100 if rows else 0.0
        if consistency < 85.0:
            failures.append(f"row_consistency ({consistency:.1f}%) < 85%")
    else:
        consistency = 0.0
        if col_count >= 2:
            failures.append("row_consistency (0.0%) < 85%")

    # Check 4a: Fragmented words
    frags = detect_fragmented_words(rows)
    if frags:
        failures.append(f"fragmented_words ({len(frags)} splinters: {frags[:2]})")

    # Check 4b: Merged cell anomaly
    merged_cells = detect_merged_cell_anomalies(rows)
    if merged_cells:
        failures.append(f"merged_cell_anomalies ({len(merged_cells)} instances: {merged_cells[:2]})")

    # Check 5: Page Continuation Detection
    is_continuation = False
    continuation_reason = None
    if prev_table and prev_table.get("row_count", 0) > 0:
        prev_rows = prev_table.get("rows", [])
        prev_last_row = prev_rows[-1] if prev_rows else []
        prev_text = " ".join(prev_last_row).lower()
        curr_header_text = " ".join(table_result.get("headers", [])).lower()
        page_top_text = page.get_text("text")[:400].lower()  # type: ignore[no-untyped-call]

        prev_is_closed = any(kw in prev_text for kw in [
            "integral part", "report of even date", "for and on behalf", "total capital and liabilities", "total assets"
        ])
        curr_is_new = any(kw in page_top_text for kw in [
            "balance sheet", "profit and loss", "cash flow statement", "schedule 1", "schedule 2", "schedule 3",
            "attendance of each director", "segment reporting"
        ]) or any(kw in curr_header_text for kw in ["schedule", "particulars"])

        if not prev_is_closed:
            if any(kw in prev_text for kw in ["investing activities", "operating activities", "carried forward", "contd", "continued"]):
                is_continuation = True
                continuation_reason = "prior_page_unclosed_financial_section"
            elif "contd" in curr_header_text or "(continued)" in curr_header_text:
                is_continuation = True
                continuation_reason = "table_has_continued_marker"
            elif not curr_is_new and abs(col_count - prev_table.get("col_count", 0)) == 0 and len(rows) > 5:
                prev_divs = prev_table.get("col_dividers", [])
                curr_divs = table_result.get("col_dividers", [])
                if len(prev_divs) == len(curr_divs) and all(abs(a - b) < 15 for a, b in zip(prev_divs, curr_divs)):
                    is_continuation = True
                    continuation_reason = "column_geometry_matches_prior_page_table"

    if is_continuation:
        failures.append(f"continuation_detected ({continuation_reason})")

    # Check 6: Divider cuts words
    divs = table_result.get("col_dividers", [])
    words = table_result.get("words", [])
    cut_words = [w[4] for div in divs[1:-1] for w in words if w[0] + 0.5 < div < w[2] - 0.5]
    if cut_words:
        failures.append(f"divider_cuts_words ({len(cut_words)} words: {cut_words[:3]})")

    # Check 7: Header drift in data rows
    if rows:
        header_rows = [r for r in rows if bool(re.search(r"\b(as at|year ended|for the year ended)\b", " ".join(r[1:]).lower()))]
        if header_rows:
            failures.append(f"header_drift_in_data_rows ({len(header_rows)} rows contain header phrases in data columns)")

    # Check 8: Unit only header (missing column 0 header in multi-column table)
    headers = table_result.get("headers", [])
    if col_count >= 2 and headers and not headers[0].strip() and all(not bool(h.strip()) for h in headers[:-1]):
        failures.append("unit_only_header (column 0 and other data columns lack headers)")

    gate_passed = (len(failures) == 0)
    return {
        "gate_passed": gate_passed,
        "is_continuation": is_continuation,
        "consistency": round(consistency, 1),
        "fragmented_word_count": len(frags),
        "merged_cell_count": len(merged_cells),
        "failures": failures,
    }


def extract_table_tier2_gemini(
    page: pymupdf.Page,
    bbox: tuple[float, float, float, float] | None = None,
    api_key: str | None = None,
    dpi: int = 150,
    model_name: str = "gemini-3.5-flash-lite",
) -> dict[str, Any]:
    """
    Tier 2: Calls Google Gemini Flash vision model on rendered page image.
    Determines whether region contains an actual table and extracts with complete fidelity.
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_key:
        logger.warning("GEMINI_API_KEY not configured. Tier 2 fallback unavailable.")
        return {"success": False, "error": "missing_api_key"}

    # Render target bounding box or full page to PNG bytes
    clip_rect = pymupdf.Rect(*bbox) if bbox else page.rect  # type: ignore[no-untyped-call]
    pix = page.get_pixmap(dpi=dpi, clip=clip_rect)
    img_bytes = pix.tobytes("png")  # type: ignore[no-untyped-call]
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    prompt = (
        "Extract the financial/data table from this image with complete fidelity.\n"
        "First determine whether this region is an actual data table (with >= 2 columns of data), "
        "or non-tabular narrative text, legal disclosures, AGM notices, or boxed prose with decorative borders.\n"
        "Return ONLY a JSON object with this exact schema:\n"
        "{\n"
        '  "table_detected": true or false,\n'
        '  "rejection_reason": "narrative_prose" or "boxed_callout" or "decorative_border" or null,\n'
        '  "caption": "Table title if present, else null",\n'
        '  "headers": ["Col 1", "Col 2", ...],\n'
        '  "rows": [["Cell 1", "Cell 2", ...], ...],\n'
        '  "footnotes": ["Footnote 1", ...],\n'
        '  "is_borderless": true or false\n'
        "}\n"
        "RULES:\n"
        "1. If this is narrative text or an AGM notice rather than a data table, set table_detected to false.\n"
        "2. Never truncate or concatenate numbers across columns.\n"
        "3. Preserve en/em-dashes (— or -) for nil or zero values.\n"
        "4. Preserve full row labels without abbreviation.\n"
        "5. Do not include markdown code block syntax (like ```json), return raw JSON only."
    )

    _, punct_map = load_symbol_mappings()

    def clean_cell_text(cell: str) -> str:
        return "".join(punct_map.get(ord(c), c) for c in cell).strip()

    models_to_try = [model_name, "gemini-flash-lite-latest", "gemini-3.8-flash", "gemini-3.5-flash"]
    for current_model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={resolved_key}"
        payload: dict[str, Any] = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": img_b64}}
                ]
            }],
            "generationConfig": {"temperature": 0.0}
        }

        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    # Strip markdown code blocks if model wrapped JSON
                    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                    raw_text = re.sub(r"\s*```$", "", raw_text)
                    parsed = json.loads(raw_text)
                    return {
                        "success": True,
                        "table_detected": parsed.get("table_detected", True),
                        "rejection_reason": parsed.get("rejection_reason"),
                        "model": current_model,
                        "caption": parsed.get("caption"),
                        "headers": [clean_cell_text(c) for c in parsed.get("headers", [])],
                        "rows": [[clean_cell_text(c) for c in r] for r in parsed.get("rows", [])],
                        "footnotes": parsed.get("footnotes", []),
                        "is_borderless": parsed.get("is_borderless", False),
                    }
                elif resp.status_code in {429, 503}:
                    time.sleep(2 ** attempt)
                else:
                    logger.debug("Gemini %s returned status %d: %s", current_model, resp.status_code, resp.text[:100])
                    break
            except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                logger.debug("Gemini request exception on attempt %d: %s", attempt, str(e))
                time.sleep(1)

    return {"success": False, "error": "all_models_failed"}


def stitch_continuation_tables(prev_table: Table, curr_table: Table) -> Table:
    """
    Stitches a multi-page continuation table with its predecessor table.
    Inherits headers from predecessor and combines rows and footnotes.
    """
    combined_rows = list(prev_table.rows) + list(curr_table.rows)
    combined_footnotes = list(prev_table.footnotes) + [
        f for f in curr_table.footnotes if f not in prev_table.footnotes
    ]
    markdown = Table.render_markdown(prev_table.headers, combined_rows)

    return Table(
        element_id=f"{prev_table.element_id}_stitched_p{curr_table.page_number}",
        element_type="table",
        page_number=prev_table.page_number,
        bbox=prev_table.bbox,
        document_name=prev_table.document_name,
        section_heading=prev_table.section_heading,
        headers=prev_table.headers,
        rows=combined_rows,
        markdown=markdown,
        caption=prev_table.caption or curr_table.caption,
        footnotes=combined_footnotes,
        is_borderless=prev_table.is_borderless or curr_table.is_borderless,
        is_continuation=True,
        continuation_from_page=prev_table.page_number,
        confidence_score=min(prev_table.confidence_score, curr_table.confidence_score),
        needs_review=prev_table.needs_review or curr_table.needs_review,
        extraction_status="successful" if not (prev_table.needs_review or curr_table.needs_review) else "needs_review",
        extraction_method="tier2_gemini" if "tier2" in (prev_table.extraction_method + curr_table.extraction_method) else "tier1",
        review_reasons=prev_table.review_reasons + curr_table.review_reasons,
    )


def extract_tables_from_page(
    page: pymupdf.Page,
    page_number: int,
    document_name: str,
    prev_page_tables: list[Table] | None = None,
    api_key: str | None = None,
    enable_tier2: bool = True,
) -> list[Table]:
    """
    High-level orchestrator extracting all table elements from a single page.
    Applies Tier 1 column-projection, quality gate checks, Tier 2 vision fallback,
    and cross-page continuation stitching.
    """
    boxes = detect_all_table_regions(page)
    if not boxes:
        return []

    tables: list[Table] = []

    for reg_idx, bbox in enumerate(boxes):
        element_id = f"{document_name}_p{page_number}_tab{reg_idx}"
        t1_res = extract_table_tier1(page, bbox)

        # 1. Early Rejection: Single-column boxed text or phantom-column narrative callouts are NEVER tables.
        # Check both raw column count and populated column count (columns with actual non-empty text).
        raw_cols = t1_res.get("col_count", 0)
        t1_headers = t1_res.get("headers", [])
        t1_rows = t1_res.get("rows", [])
        populated_cols = sum(
            1 for c in range(raw_cols)
            if (c < len(t1_headers) and bool(t1_headers[c].strip()))
            or any(c < len(r) and bool(r[c].strip()) for r in t1_rows)
        )
        if raw_cols < 2 or populated_cols < 2:
            logger.debug(
                "Rejecting candidate region %s on page %d: populated cols (%d) < 2 (single-column callout)",
                bbox,
                page_number,
                populated_cols,
            )
            continue

        prev_t_dict = None
        if reg_idx == 0 and prev_page_tables:
            prev_t = prev_page_tables[-1]
            prev_t_dict = {
                "row_count": prev_t.row_count,
                "col_count": prev_t.col_count,
                "headers": prev_t.headers,
                "rows": prev_t.rows,
                "col_dividers": [],
            }

        gate_res = evaluate_quality_gate(t1_res, page, prev_table=prev_t_dict)

        if gate_res["gate_passed"]:
            # Tier 1 Clean Pass
            headers = t1_res["headers"]
            rows = t1_res["rows"]
            md = Table.render_markdown(headers, rows)
            table_elem = Table(
                element_id=element_id,
                element_type="table",
                page_number=page_number,
                bbox=bbox,
                document_name=document_name,
                headers=headers,
                rows=rows,
                markdown=md,
                confidence_score=0.98,
                needs_review=False,
                extraction_status="successful",
                extraction_method="tier1",
                review_reasons=[],
            )
            tables.append(table_elem)
        elif enable_tier2 and (os.getenv("GEMINI_API_KEY") or api_key):
            # Tier 2: Gemini Flash Vision Fallback
            tier2_res = extract_table_tier2_gemini(page, bbox=bbox, api_key=api_key)
            if tier2_res.get("success"):
                # Check if Tier 2 explicitly confirmed this region is NOT a table
                if not tier2_res.get("table_detected", True) or not tier2_res.get("rows"):
                    logger.info(
                        "Tier 2 rejected candidate region %s on page %d as table (%s). Reclassifying as narrative.",
                        bbox,
                        page_number,
                        tier2_res.get("rejection_reason"),
                    )
                    continue

                headers = tier2_res.get("headers", [])
                rows = tier2_res.get("rows", [])
                md = Table.render_markdown(headers, rows)
                table_elem = Table(
                    element_id=element_id,
                    element_type="table",
                    page_number=page_number,
                    bbox=bbox,
                    document_name=document_name,
                    headers=headers,
                    rows=rows,
                    markdown=md,
                    caption=tier2_res.get("caption"),
                    footnotes=tier2_res.get("footnotes", []),
                    is_borderless=tier2_res.get("is_borderless", False),
                    is_continuation=gate_res["is_continuation"],
                    continuation_from_page=prev_page_tables[-1].page_number if gate_res["is_continuation"] and prev_page_tables else None,
                    confidence_score=0.92,
                    needs_review=False,
                    extraction_status="successful",
                    extraction_method="tier2_gemini",
                    review_reasons=[],
                )
                tables.append(table_elem)
            else:
                # Tier 3 Fallback: degraded Tier 1 output flagged for review
                headers = t1_res.get("headers", [])
                rows = t1_res.get("rows", [])
                md = Table.render_markdown(headers, rows)
                table_elem = Table(
                    element_id=element_id,
                    element_type="table",
                    page_number=page_number,
                    bbox=bbox,
                    document_name=document_name,
                    headers=headers,
                    rows=rows,
                    markdown=md,
                    confidence_score=0.45,
                    needs_review=True,
                    extraction_status="needs_review",
                    extraction_method="tier3",
                    review_reasons=gate_res["failures"] + [f"tier2_fallback_failed: {tier2_res.get('error')}"],
                )
                tables.append(table_elem)
        else:
            # Tier 3: Quality gate failed and Tier 2 not active
            headers = t1_res.get("headers", [])
            rows = t1_res.get("rows", [])
            md = Table.render_markdown(headers, rows)
            table_elem = Table(
                element_id=element_id,
                element_type="table",
                page_number=page_number,
                bbox=bbox,
                document_name=document_name,
                headers=headers,
                rows=rows,
                markdown=md,
                confidence_score=0.50,
                needs_review=True,
                extraction_status="needs_review",
                extraction_method="tier3",
                review_reasons=gate_res["failures"],
            )
            tables.append(table_elem)

    return tables
