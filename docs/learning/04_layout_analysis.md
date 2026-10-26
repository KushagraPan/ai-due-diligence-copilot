# Layout Analysis & Spatial Reconciliation

---

## 1. Why Layout Analysis Exists

When extracting text from complex corporate annual reports, naive PDF extractors create fatal corruptions:
1. **Multi-Column Text Interleaving (Broken Reading Order):** Corporate annual reports are dominated by 2-column and 3-column layouts. A naive parser reads horizontally straight across the physical page, interleaving Line 1 of Column 1 with Line 1 of Column 2 into garbled, unreadable nonsense.
2. **Text Collision & Double-Counting:** When tables and prose paragraphs coexist on the same page, naive extractors either double-count text (ingesting financial numbers once inside a table and a second time as messy text) or accidentally clip adjacent narrative paragraphs when carving out table bounding boxes.
3. **Loss of Structural Hierarchy:** Paragraphs and tables in financial reports derive their legal and accounting meaning from their enclosing section headings (e.g., distinguishing *Standalone* from *Consolidated* schedules).

**Layout Analysis & Spatial Reconciliation** ([`src/due_diligence/ingestion/layout.py`](../../src/due_diligence/ingestion/layout.py)) transforms the raw, unstructured visual soup of PyMuPDF text spans and extracted tables into an ordered, provenance-tagged, mutually exclusive stream of [`DocumentElement`](../../src/due_diligence/ingestion/elements.py) objects.

---

## 2. Where does it fit in the Ingestion Pipeline?

```
Raw PDF File (PyMuPDF Text Spans & Visual Rectangles)
                       │
                       ▼
       Tier 1/2 Table Extraction (table_extractor.py)
                       │
                       ▼
   Extracted Table Bounding Boxes + Page Text Blocks
                       │
                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│            LAYOUT ANALYSIS ENGINE (layout.py)                          │
│ 1. Cookie-Cutter Table BBox Claiming & Line-Level Trimming             │
│ 2. Caption & Footnote Association (is_valid_table_caption)             │
│ 3. Multi-Column Gutter Histogram Detection                             │
│ 4. Human Reading Order Sorting                                         │
│ 5. Structural Classification (Heading, ListItem, NarrativeText)        │
│ 6. Section Heading Stack Propagation (heading_path, section_heading)   │
│ 7. Single-Owner Conservation Verification (unowned == 0, multi == 0)   │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
                             ▼
  Reconciled Stream of 11,008 DocumentElements (Current Rebuilt Baseline)
                             │
                             ▼
         Document-Aware Chunking Engine (chunker.py)
```

---

## 3. The Single-Owner Spatial Invariant vs. Character Reconciliation

Our layout engine enforces a strict mathematical distinction between **Block Ownership Conservation** and **Character Reconciliation Diagnostics**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│        THE SINGLE-OWNER INVARIANT (Hard Correctness Requirement)        │
├─────────────────────────────────────────────────────────────────────────┤
│ Every source visual block on the PDF page must belong to EXACTLY ONE    │
│ structural owner:                                                       │
│                                                                         │
│   source_block_count == owned_by_table + owned_by_heading +             │
│                         owned_by_list + owned_by_narrative              │
│                                                                         │
│ - unowned_block_count == 0         (Zero dropped blocks)                │
│ - multiply_owned_block_count == 0  (Zero duplicate/conflicted blocks)   │
│                                                                         │
│ Measured Result across 678 pages: 100% compliance (0 unowned, 0 multi). │
└─────────────────────────────────────────────────────────────────────────┘
                                     vs.
┌─────────────────────────────────────────────────────────────────────────┐
│     CHARACTER RECONCILIATION DELTA (Diagnostic Quality Indicator)       │
├─────────────────────────────────────────────────────────────────────────┤
│ Measures character count variance between raw stream and structured:    │
│                                                                         │
│   delta_pct = (|structured_chars - raw_stream_chars| / raw_chars) * 100│
│                                                                         │
│ - Tolerance Threshold: <= 5.0%                                          │
│ - Why Table Pages Exceed 5%: Converting multi-line table cells into     │
│   Markdown table syntax collapses embedded newlines (\n) into spaces.   │
│   A ~8% delta on financial statement pages (e.g. Page 380) reflects     │
│   benign newline collapse with 100% vocabulary/word retention.          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. How the Spatial Reconciliation Pipeline Works

### 4.1 "Cookie-Cutter" Table Partitioning
1. **Tables Claim Their Territory:** Bounding boxes of extracted `Table` elements act as cookie cutters. Any PyMuPDF text block whose bounding box has $\ge 50\%$ spatial overlap with a table is claimed by that table.
2. **Partial Overlap Line-Level Trimming:** If a narrative block partially intersects a table edge, individual text lines inside the table box are trimmed away, while lines outside remain in the narrative block.
3. **Table Exclusion:** Claimed table text is excluded from narrative extraction, eliminating the risk of financial tables being ingested twice.

### 4.2 Table Caption & Footnote Association (`is_valid_table_caption`)
Near-table text blocks are inspected to identify captions above and footnotes below:
- **Caption Guard (`is_valid_table_caption`):**
  ```python
  def is_valid_table_caption(text: str) -> bool:
      stripped = text.strip()
      if not stripped or TOTAL_ROW_PATTERN.match(stripped):
          return False
      if not re.search(r"[A-Za-z]", stripped):
          return False
      tokens = stripped.split()
      digit_tokens = sum(1 for tok in tokens if any(c.isdigit() for c in tok))
      return not (tokens and digit_tokens / len(tokens) > 0.6)
  ```
- **Why this guard is critical:** A line beginning with `"Total"` (such as Schedule 11's Total row) is a summary data row of the table above it. Without `is_valid_table_caption()`, the layout engine previously mistook Schedule 11's Total line for Schedule 12's table caption (the root cause of the Q06 issue!).

### 4.3 Multi-Column Gutter Detection & Human Reading Order
1. **Histogram Gutter Analysis:** On remaining non-table blocks, X-coordinates are projected onto a horizontal density histogram. Valleys of whitespace wider than 15 pt establish vertical column gutters.
2. **Reading Order Sorting:** Blocks are sorted top-to-bottom in Column 1, then top-to-bottom in Column 2.
3. **Full-Width Table Splitters:** Full-width tables act as horizontal barriers: blocks above the table are read first, the table is read second, and blocks below the table are read third.

### 4.4 Section Heading Detection & Stack Propagation
1. **Heading Classification:** Text with font size significantly above the page median (`heading_font_ratio >= 1.15`) or bold/uppercase formatting is classified as `Heading(level=1..6)`.
2. **Active Heading Stack:** An active heading stack is maintained across all 678 pages:
   - `# Financial Statements` (H1)
   - `## Standalone Financial Statements` (H2)
   - `### Schedule 12 - Contingent Liabilities` (H3)
3. Every narrative paragraph, bullet point, and table inherits:
   - `section_heading`: Innermost enclosing heading.
   - `heading_path`: Full hierarchical breadcrumb path.

---

## 5. Why Layout Mistakes Become Downstream RAG Mistakes

1. **Broken Reading Order Scrambles Legal Meaning:**
   If a 2-column contract has:
   - Column 1: *"The borrower shall repay all principal..."*
   - Column 2: *"...unless an event of default occurs."*
   Reading horizontally creates: *"The borrower shall repay all unless an event of default principal occurs."* Downstream embeddings become gibberish.
2. **Caption Errors Cause Metric Inversion:**
   If Schedule 11's Total is misidentified as Schedule 12's caption, BM25 matches Schedule 12 queries to Schedule 11 figures.
3. **Missing Heading Propagation Destroys Disambiguation:**
   Without `heading_path`, Standalone and Consolidated schedules have identical table names (*"Balance Sheet"*), leading to random selection between parent and subsidiary numbers.

---

## 6. Reconciled Corpus Baseline Statistics

From the rebuilt production baseline ([`data/processed/layout_preflight_2026-09-25.json`](../../data/processed/layout_preflight_2026-09-25.json)):

| Metric | Measured Value | Meaning / Impact | Status |
| :--- | :--- | :--- | :--- |
| **Total Pages Reconciled** | 678 pages | Complete HDFC Bank FY2025-26 Annual Report | Verified |
| **Reconciled Elements** | **11,008 elements** | Total structured intermediate elements | Verified |
| **Ownership Conflicts** | **0 blocks (0.00%)** | `multiply_owned_block_count == 0` | 100% compliant |
| **Unowned Blocks** | **0 blocks (0.00%)** | `unowned_block_count == 0` | 100% compliant |
| **Multi-Column Pages** | 348 pages (51.3%) | Correctly sorted Column 1 $\to$ Column 2 | Verified |
| **Table Captions Validated**| Zero Total-as-caption | Guarded by `is_valid_table_caption` | Production |

---

## 7. Things to Remember

1. **Single-Owner Invariant:** Every block has exactly one owner (`unowned == 0`, `duplicate == 0`).
2. **Reconciliation Delta is Diagnostic:** ~8% delta on table pages reflects newline collapse in Markdown cells, not lost words.
3. **Total Rows are NOT Captions:** Guarded by `is_valid_table_caption(text)` to prevent cross-schedule contamination.
4. **Column 1 Before Column 2:** Over 51% of pages are multi-column; reading order must be spatially reconstructed.
5. **Heading Propagation is Mandatory:** Every element carries `section_heading` and `heading_path` to disambiguate corporate scope.
