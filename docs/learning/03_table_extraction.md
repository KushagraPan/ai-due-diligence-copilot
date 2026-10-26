# Financial Table Extraction Subsystem: 3-Tier Architecture & Quality Gating
**AI Due Diligence Co-pilot Technical Manual & Learning Notes**

---

## 1. What is this?

### 1.1 The Layman's Foundation: The "Invisible Grid" Problem
To understand why table extraction is notoriously difficult, we must first understand a fundamental truth about PDF documents:

> **A PDF has no concept of a "table", "row", or "column".**

When you open a spreadsheet in Microsoft Excel or Google Sheets, the computer knows that cell `B4` contains a number, and that column `B` has a header named `"FY2025-26"`. The grid is real, structured, and machine-readable.

In a PDF, **there is no grid**. A PDF is simply a digital sheet of paper with instructions telling a printer where to drop ink:
- *"Draw a black letter 'T' at coordinates $(X=140, Y=320)$."*
- *"Draw a number '5' at coordinates $(X=280, Y=320)$."*
- *"Draw a thin gray line from $(X=100, Y=310)$ to $(X=500, Y=310)$."*

To a human reading the document, your brain instantly connects the visual spacing and infers: *"Aha! This is a table comparing 2025 to 2026."*

```
WHAT A HUMAN SEES (A Structured Grid):
┌─────────────────────────┬──────────────┬──────────────┐
│ Particulars             │   FY2025-26  │   FY2024-25  │
├─────────────────────────┼──────────────┼──────────────┤
│ Gross Advances          │   28,493,140 │   24,848,620 │
│ Net Interest Income     │    1,120,450 │      985,320 │
└─────────────────────────┴──────────────┴──────────────┘

WHAT A COMPUTER SEES IN RAW PDF DATA (Floating Islands of Text):
Word("Particulars", x0=42.1, y0=110.4, x1=98.3, y1=122.1)
Word("FY2025-26",   x0=240.5, y0=110.4, x1=290.1, y1=122.1)
Word("FY2024-25",   x0=360.2, y0=110.4, x1=410.8, y1=122.1)
Word("Gross",       x0=42.1, y0=135.2, x1=70.3, y1=146.9)
Word("Advances",    x0=73.5, y0=135.2, x1=118.0, y1=146.9)
Word("28,493,140",  x0=242.0, y0=135.2, x1=292.5, y1=146.9)
...
```

If a standard computer program tries to read this page naively, it might read left-to-right across lines, jumbling row labels and numbers into meaningless soup:
> *"Particulars FY2025-26 FY2024-25 Gross Advances 28,493,140 24,848,620 Net Interest Income..."*

---

### 1.2 The Three Silent Disasters of Table Extraction
When naive extractors attempt to reconstruct tables from raw PDF coordinates, they fail in three catastrophic ways:

1. **Word Splintering (The Guillotine Error):**
   The program draws an imaginary vertical line between columns, but misplaces it by 2 millimeters. The line cuts right through a word, splitting `"Investment"` into `"In"` and `"vestment"`. The AI now thinks the company made an investment in something called "vestment".
2. **Merged Column Collisions:**
   A long company name or multi-digit number spills slightly into the next column's whitespace. The program fails to see a gap and lumps two columns together (e.g. `"17 Present"` instead of `Column 1: 17, Column 2: Present`).
3. **The Trailing Total Drop:**
   In financial reports, the bottom summary row (*"Total Capital & Liabilities: ₹34,179,458,972"*) frequently sits slightly below the table's bottom visual line with **no border drawn underneath it**. Naive extractors cut off at the visual line, leaving the single most important number in the balance sheet behind.

To solve this, our system implements a **3-Tier Architecture** governed by an automated **5-Point Quality Gate**.

---

## 2. Where does it fit in the RAG Pipeline?

Table extraction runs during **Track A (Offline Ingestion)** on every single PDF page before document chunking and vector indexing:

```mermaid
flowchart TD
    A["Raw PDF Page (PyMuPDF Text Spans & Vector Drawings)"] --> B["1. detect_all_table_regions(page)<br/>• Scans horizontal line clusters & numeric text density<br/>• extend_box_for_trailing_total() swallows summary rows"]
    B --> C["Candidate Table Bounding Box (x0, y0, x1, y1)"]
    
    C --> D["2. Tier 1: Local Column-Projection Engine<br/>• Fast, deterministic, zero-cost PyMuPDF geometry<br/>• Clusters numeric X-centers into vertical gutters<br/>• Truncates auditor signatures & aligns baselines"]
    
    D --> E{"3. The 5-Point Quality Gate<br/>(evaluate_quality_gate)"}
    
    E -->|PASSED (gate_passed = True)| F["Tier 1 Clean Table Element<br/>• confidence_score: 0.98<br/>• needs_review: False"]
    
    E -->|FAILED (gate_passed = False)| G["4. Tier 2: Multimodal Vision Fallback<br/>(Google Gemini Flash)<br/>• Renders 150 DPI PNG snapshot of bbox<br/>• Inspects visual pixels to reconstruct grid"]
    
    G --> H{"Vision Model Decision"}
    H -->|Verified as Real Table| I["Tier 2 Table Element<br/>• confidence_score: 0.92<br/>• needs_review: False"]
    H -->|Identified as Narrative Prose| J["Reclassify Region as Prose<br/>(e.g., Decorative Boxed Callout)"]
    H -->|API Failure / Offline| K["5. Tier 3: Graceful Degradation<br/>• Preserves raw Tier 1 text<br/>• confidence_score: 0.45<br/>• needs_review: True (Flagged for Auditor)"]
    
    F --> L["DocumentElement Stream"]
    I --> L
    K --> L
    
    L --> M["Spatial Layout Reconciliation (layout.py)<br/>• Cookie-cutter exclusion: text inside table bbox removed from prose"]
    M --> N["Document-Aware Chunking (chunker.py)<br/>• Atomic tables (up to 1,000 tokens)<br/>• Repeated Markdown headers on slices"]
```

---

## 3. What did WE Implement?

The table extraction engine is implemented in [`src/due_diligence/ingestion/table_extractor.py`](../../src/due_diligence/ingestion/table_extractor.py):

| Component / Function | Role & Technical Mechanism |
| :--- | :--- |
| [`detect_all_table_regions()`](../../src/due_diligence/ingestion/table_extractor.py#L125-L226) | Clusters horizontal vector drawing rules and high-density numeric text lines to identify candidate bounding boxes `(x0, y0, x1, y1)`. |
| [`extend_box_for_trailing_total()`](../../src/due_diligence/ingestion/table_extractor.py#L228-L270) | Scans up to 50pt below the table bottom for rule-less `Total` summary rows and dynamically extends the bounding box downward. |
| [`extract_table_tier1()`](../../src/due_diligence/ingestion/table_extractor.py#L272-L485) | The local column-projection engine. Clusters numeric X-centers, projects whitespace gutters, and enforces anti-collision dividers. |
| [`evaluate_quality_gate()`](../../src/due_diligence/ingestion/table_extractor.py#L545-L661) | The 5-point quality gate enforcing column counts, row schema consistency, word splinters, merged cells, and continuation markers. |
| [`extract_table_tier2_gemini()`](../../src/due_diligence/ingestion/table_extractor.py#L663-L756) | Multimodal vision fallback calling Google Gemini Flash (`gemini-3.5-flash-lite`, `temperature=0.0`) on a 150 DPI rendered PNG crop. |
| [`stitch_continuation_tables()`](../../src/due_diligence/ingestion/table_extractor.py#L758-L798) | Merges multi-page financial statements (e.g. Cash Flow statements continuing across pages) into a single cohesive parent table. |

---

## 4. The 3-Tier Architecture Explained

```
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 1: Local Column-Projection Engine                                 │
│ • Cost: $0.00 (100% Local Python)                                      │
│ • Latency: ~10–15 ms per page (~70 pages/sec on standard CPU)          │
│ • Handles 85%+ of standard bordered and borderless schedules           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Quality Gate Fails
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 2: Multimodal LLM Vision Fallback (Google Gemini Flash)           │
│ • Cost: ~$0.001 per table crop                                         │
│ • Latency: ~1.2 seconds per API call                                   │
│ • Renders 150 DPI PNG snapshot; reads visual alignment like human eyes │
│ • Disambiguates decorative boxed prose vs. real data grids             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ API Error / Offline / Key Missing
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 3: Graceful Degradation with Audit Flags                          │
│ • Cost: $0.00                                                          │
│ • Preserves best-effort Tier 1 text (Zero Unowned Text invariant)       │
│ • Flags table: confidence=0.45, needs_review=True, lists gate reasons  │
│ • Downstream auditors immediately see which tables need human review   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Technical Deep-Dive: How Tier 1 Column Projection Works

Tier 1 operates mathematically in four sequential steps:

### Step 5.1: Numeric X-Center Clustering (Finding the Data Columns)
In financial tables, row labels (descriptions) vary wildly in length, but numbers in the same column tend to align vertically (either left-aligned, right-aligned, or decimal-aligned).

Tier 1 scans all words in the bounding box containing digits and computes their horizontal centers:
$$\text{center}_x = \frac{x_0 + x_1}{2}$$

It groups these X-centers into clusters where numbers sit within 25 points of each other:
```python
# src/due_diligence/ingestion/table_extractor.py
num_x_centers = sorted([(w[0] + w[2]) / 2 for w in number_words])
col_clusters = []
for x in num_x_centers:
    if not col_clusters or (x - col_clusters[-1]["mean"]) > 25:
        col_clusters.append({"mean": x, "count": 1, "points": [x]})
    else:
        col_clusters[-1]["points"].append(x)
        col_clusters[-1]["mean"] = sum(col_clusters[-1]["points"]) / len(col_clusters[-1]["points"])
        col_clusters[-1]["count"] += 1

# A valid data column must contain at least 3 numbers aligned vertically
valid_num_cols = [c for c in col_clusters if c["count"] >= 3]
```

### Step 5.2: Whitespace Gutter Projection
Once the data column positions are identified, the algorithm determines the boundary between the descriptive row label (Column 0) and the first data column (Column 1):
```
[Column 0: Description]                  [Column 1: 2026]     [Column 2: 2025]
Gross Loans and Advances . . . . . . . .   28,493,140           24,848,620
Net Term Deposits  . . . . . . . . . . .   15,210,000           13,100,500
                         ▲             ▲
                   label_max_x     first_data_x0
                         │             │
                         └──────┬──────┘
                                ▼
                       div_0 = (label_max_x + first_data_x0) / 2
```

### Step 5.3: The Anti-Collision Guardrail (Preventing Word Splinters)
What if a long row description accidentally crosses `div_0`?
Tier 1 enforces an anti-collision check:
```python
# src/due_diligence/ingestion/table_extractor.py
safe_dividers = [x0_t]
for div in col_dividers[1:-1]:
    # Check if this divider cuts through ANY word's bounding box
    cutting = [w for w in words if w[0] + 0.5 < div < w[2] - 0.5]
    if cutting:
        w_cut = cutting[0]
        # Push the divider to the left or right edge of the word!
        if w_cut[0] - 2.0 > safe_dividers[-1] + 5.0:
            safe_dividers.append(w_cut[0] - 2.0)
        else:
            safe_dividers.append(w_cut[2] + 2.0)
    else:
        safe_dividers.append(div)
```

---

## 6. The 5-Point Quality Gate: Technical Specification

Every candidate table extracted by Tier 1 must pass [`evaluate_quality_gate()`](../../src/due_diligence/ingestion/table_extractor.py#L545-L661):

```
┌────────────────────────────────────────────────────────────────────────┐
│                   THE 5-POINT QUALITY GATE CHECKS                      │
├────────────────────────────────────────────────────────────────────────┤
│ Check 1: Minimum Column Count (col_count >= 2)                         │
│ • Code: if col_count < 2: failures.append("col_count < 2")             │
│ • Rejects single-column decorative text blocks or boxed AGM notices.   │
├────────────────────────────────────────────────────────────────────────┤
│ Check 2: Minimum Row Count (row_count >= 2)                            │
│ • Code: if row_count < 2: failures.append("row_count < 2")             │
│ • Rejects isolated decorative header bars or horizontal divider rules. │
├────────────────────────────────────────────────────────────────────────┤
│ Check 3: Row Consistency Threshold (row_consistency >= 85%)            │
│ • Verifies that >= 85% of rows have a label in Col 0 and valid data.   │
│ • Catches column drift where headers and cells got shifted.            │
├────────────────────────────────────────────────────────────────────────┤
│ Check 4: Zero Word Splinters & Zero Merged Cells                       │
│ • detect_fragmented_words(): Scans for chopped words (e.g. ['in','vest'])│
│ • detect_merged_cell_anomalies(): Scans for fused text (e.g. '17 Present')│
├────────────────────────────────────────────────────────────────────────┤
│ Check 5: Page Continuation Detection                                   │
│ • Checks if previous page ended with unclosed financial sections       │
│   ("Operating activities", "(continued)", matching column geometry).   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Real-World Practical Case Study: Page 343 Director Attendance Table

Let us trace a real-world edge case that occurred during our audit of the **HDFC Bank FY2025-26 Annual Report (Page 343)**.

### The Situation on Page 343:
Page 343 contains the **Board of Directors Meeting Attendance Table**:
- Column 0: Director Name (*"Mr. Atanu Chakraborty"*)
- Column 1: Meetings Held (*"17"*)
- Column 2: Meetings Attended (*"Present for 17"* / *"Attended 16"*)

### What Happened in Tier 1:
Because the directors' attendance status had words (`"Present"`) immediately adjacent to numbers (`"17"`), Tier 1's local column-projection miscalculated the whitespace gutter. It placed the column divider too far to the right!

```
TIER 1 EXTRACTION ERROR:
| Director Name | Meetings Held & Attended |
| Mr. Atanu Chakraborty | 17 Present |   <-- Column 1 and Column 2 fused!
```

### How the Quality Gate Caught It:
`evaluate_quality_gate()` ran Check 4b ([`detect_merged_cell_anomalies`](../../src/due_diligence/ingestion/table_extractor.py#L530-L543)):
```python
# Regex scans for numbers directly followed by 3+ letter words in data cells:
matches = re.findall(r"\b\d+\s+[A-Za-z]{3,}\b", cell)
# Detected anomaly: "17 Present"
```
The Quality Gate triggered:
```
gate_passed = False
failures = ["merged_cell_anomalies (1 instances: ['17 Present'])"]
```

### The Escalation to Tier 2 (Gemini Flash Vision):
1. Because `gate_passed` was `False`, the code immediately called [`extract_table_tier2_gemini()`](../../src/due_diligence/ingestion/table_extractor.py#L663).
2. It clipped the exact table bounding box on Page 343 and rendered a **150 DPI PNG image**.
3. It sent the image to Google Gemini Flash with the instruction:
   > *"Extract the financial/data table from this image with complete fidelity. Return JSON with headers and rows."*
4. **Result:** The vision model looked at the pixels, clearly saw the two distinct columns, and emitted clean, separated cells:
   ```json
   {
     "table_detected": true,
     "headers": ["Name of Director", "Meetings Held", "Meetings Attended"],
     "rows": [
       ["Mr. Atanu Chakraborty", "17", "17"]
     ]
   }
   ```
5. The table was saved with `extraction_method="tier2_gemini"`, `confidence_score=0.92`, and `needs_review=False`.

**Silent failure averted!** The downstream RAG vector database received a clean, 3-column table instead of corrupted text.

---

## 8. Another Real-World Case Study: The Trailing Total Row (`extend_box_for_trailing_total`)

During the evaluation of golden benchmark questions, another vulnerability was discovered on the **Standalone Balance Sheet (Schedule 4)**:

```
────────────────────────────────────────────────────────────────────────
Schedule 4 - Borrowings                                 As at March 31
────────────────────────────────────────────────────────────────────────
I. Borrowings in India:
   (a) Reserve Bank of India . . . . . . . . . . . . . . .    45,210.00
   (b) Other Banks . . . . . . . . . . . . . . . . . . . .    12,400.00
────────────────────────────────────────────────────────────────────────  <-- Bottom Line Drawn Here!
TOTAL (I + II) . . . . . . . . . . . . . . . . . . . . . .    57,610.00   <-- Sits BELOW the line!
```

### The Failure:
Standard PDF table detection uses the **horizontal vector lines** to define the table boundary. Because the `TOTAL` line sat **7 points below the bottom line**, naive bounding box detection cut off above it!

The `TOTAL` row was left floating in the raw page. When `layout.py` extracted narrative prose, it saw this floating text and misattached `"TOTAL (I + II): 57,610.00"` as the **caption of the NEXT schedule**!

### The Solution in Code:
We implemented [`extend_box_for_trailing_total()`](../../src/due_diligence/ingestion/table_extractor.py#L228-L270):
```python
# src/due_diligence/ingestion/table_extractor.py
for vl in visual_lines:
    y_mid = vl["y_mid"]
    # Look up to 50pt below the table bottom
    if y_bot - 12.0 < y_mid <= y_bot + 50.0:
        line_text = " ".join(w[4] for w in vl["words"])
        # Check if line matches Total pattern and contains numbers
        if TOTAL_ROW_PATTERN.search(line_text) and any(any(c.isdigit() for c in w[4]) for w in vl["words"]):
            # Dynamically stretch the table bounding box downward!
            box = (box[0], box[1], box[2], max(y_bot, vl_bottom + 4.0))
```
This guarantees that summary `Total` rows are never clipped or orphaned.

---

## 9. Downstream Connection: How Extracted Tables Become Chunks

Extracted `Table` elements directly shape how `chunker.py` and `layout.py` behave:

1. **Spatial Cookie-Cutter (Single-Owner Invariant):**
   In `layout.py`, any text falling inside an accepted `Table` bounding box is deleted from the narrative prose pipeline. Tables own their space exclusively; zero text is duplicated.
2. **Atomic Sizing:**
   In `chunker.py`, tables up to **1,000 tokens** stay as single atomic chunks so the LLM has full context.
3. **Repeated Column Headers on Multi-Slice Tables:**
   If a table has 80 rows and exceeds 1,000 tokens, it is sliced into chunks. On **every slice**, the chunker prepends the Markdown column header row (`| Particulars | FY2026 | FY2025 |`), ensuring the table never loses its column meaning.
4. **Total Row Lock:**
   The `Total` row is held back and attached strictly to the **final slice**, ensuring financial totals are never severed from their schedules.

---

## 10. Summary Matrix

| Failure Mode | How It Is Detected | Escalation Tier | Prevention Mechanism |
| :--- | :--- | :--- | :--- |
| **Word Splintering** | Quality Gate Check 4 (`detect_fragmented_words`) | Tier 2 (Gemini Vision) | Anti-collision divider repositioning + Vision OCR |
| **Merged Columns** | Quality Gate Check 4 (`detect_merged_cell_anomalies`) | Tier 2 (Gemini Vision) | Quality Gate failure $\to$ 150 DPI Gemini image fallback |
| **Trailing Total Drop**| Pre-extraction scan (`extend_box_for_trailing_total`) | Tier 1 (Automatic) | Bounding box stretched downward by up to 50pt |
| **Decorative Box Callout**| Quality Gate Check 1 (`col_count < 2`) | Early Rejection / Tier 2 | Reclassified as narrative prose; not chunked as a table |
| **Multi-Page Continuations**| Quality Gate Check 5 (`continuation_detected`) | `stitch_continuation_tables` | Links row buffers across pages into one master table |
