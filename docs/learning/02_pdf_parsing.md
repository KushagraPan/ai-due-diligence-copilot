# PDF Parsing & Font-Aware Normalization

## 1. What is this?

**PDF Parsing & Font-Aware Normalization** extracts raw, page-level text streams and character spans from PDF files while repairing font encoding artifacts and preserving exact physical page provenance.

Corporate filings—especially annual reports and audited financial statements—are rarely formatted using standard web typography. Instead, they embed specialized visual fonts (such as Wingdings, Symbol, or proprietary corporate icon sets) to render visual disclosure elements: bullet points, checkboxes, arrow indicators, and standalone dashes (`–`, `—`) representing zero liabilities or non-disclosed metrics. Standard PDF extractors (like `pypdf` or naive `pdfminer`) extract these glyphs as non-printable private-use Unicode characters (e.g., `\uf0b7`, `0xf0fc`), unreadable mojibake, or drop them entirely.

Our parser combines PyMuPDF's low-level C-rendering engine with a deterministic glyph translation layer backed by a living YAML configuration file ([`config/symbol_font_mappings.yaml`](../../config/symbol_font_mappings.yaml)). This guarantees that visual bullets, ticks, and financial nil-dashes are translated into canonical Unicode or explicit semantic markers (`[NIL/NOT DISCLOSED]`) without losing character fidelity.

---

## 2. Where does it fit?

The PDF Parser operates directly downstream of Document Ingestion and feeds both Table Extraction and Layout Reconciliation:

```
ValidatedDocument
       │
       ▼
   parser.py (PDFParser) <─── font_mappings.py (SymbolFontMapper)
       │                              ▲
       │                              │ (Loads config)
       │                     config/symbol_font_mappings.yaml
       ▼
ExtractedDocument
 ├── Page-level Text Streams
 ├── Normalized Unicode Glyphs
 ├── 1-Indexed Physical Page References
 └── Font Provenance Metadata
       │
       ▼
Downstream Layout & Table Processing (table_extractor.py, layout.py)
```

---

## 3. What did WE implement?

- [`src/due_diligence/ingestion/parser.py`](../../src/due_diligence/ingestion/parser.py):
  - [`PDFParser`](../../src/due_diligence/ingestion/parser.py#L35-L160): Extracts page text streams, raw character counts, and font metadata per page using PyMuPDF (`fitz`). Enforces strict 1-indexed physical page numbering.
- [`src/due_diligence/ingestion/font_mappings.py`](../../src/due_diligence/ingestion/font_mappings.py):
  - [`SymbolFontMapper`](../../src/due_diligence/ingestion/font_mappings.py#L25-L95): Translates raw character codepoints and font names against configuration rules. Normalizes bullet points to `•`, checkmarks to `✓`, and handles financial dashes. Emits structured `[UNMAPPED:Font:0xHEX]` tokens for unrecognized glyphs.
- [`config/symbol_font_mappings.yaml`](../../config/symbol_font_mappings.yaml):
  - Living configuration mapping specific font families (Wingdings, Wingdings 2, Wingdings 3, Symbol, Zapf Dingbats, Webdings) and hexadecimal codepoints to canonical Unicode equivalents.

---

## 4. Data flow

```
[PyMuPDF Page i (0-indexed in C API)]
                 │
                 ▼ (page.get_text("rawdict"))
[Raw Text Spans: text, font, size, flags, codepoints]
                 │
                 ▼ (SymbolFontMapper.map_text)
1. Check font name against symbol font families.
2. If symbol font: translate codepoint (e.g. 0xF0B7 -> "•").
3. If unmapped symbol: insert "[UNMAPPED:Wingdings:0xF0A1]".
4. If standalone en-dash/em-dash in financial cell: normalize to "–" or "[NIL/NOT DISCLOSED]".
                 │
                 ▼
[PageTextContainer]
  - page_number: i + 1 (1-indexed physical page)
  - text: Clean normalized text stream
  - raw_character_count: len(text) (Ground truth baseline for layout audit)
  - symbol_font_status: "clean" | "mapped" | "unmapped_detected"
```

---

## 5. Important implementation details

- **Strict 1-Indexed Page Numbering:** In PyMuPDF, page indexing is 0-based (`doc[0]`). In our system, `page_number = i + 1`. This invariant ensures that citations like `[Annual Report, Page 378]` match the physical page in Acrobat Reader or printed copies, eliminating off-by-one citation errors.
- **Living Mapping Architecture:** Rather than hardcoding glyph translations in Python source code, mappings are maintained in `config/symbol_font_mappings.yaml`. When a new corporate report introduces a custom bullet or icon font, developers add the mapping without modifying or redeploying Python code.
- **Financial Dash Preservation:** In financial statements, a standalone dash (`–` or `—`) often denotes `0`, `NIL`, or `Not Disclosed`. Standard NLP tokenizers often discard dashes as punctuation. Our parser explicitly preserves standalone dashes or maps them to `[NIL/NOT DISCLOSED]` to prevent zero-value debt or revenue lines from disappearing.
- **Raw Character Count as Ground Truth:** For every page, the parser computes `raw_character_count = len(page.get_text("text"))`. This raw count serves as the ground-truth benchmark for the downstream Layout Preflight reconciliation audit.

---

## 6. Concepts I learned

### Python concepts
- **PyMuPDF (`fitz`) Text Extraction Hierarchy:** Understanding the difference between `page.get_text("text")` (plain text string), `page.get_text("blocks")` (bounding boxes with text blocks), and `page.get_text("rawdict")` (low-level span dictionary with font names, glyph codes, and font flags).
- **YAML Configuration Loading with `yaml.safe_load`:** Safely deserializing user-configurable mapping files into Python dictionaries without arbitrary code execution vulnerabilities.
- **Regex Hexadecimal Codepoint Matching:** Matching Unicode escape sequences (`[\uF000-\uF0FF]`) using Python's `re` module to locate private-use glyphs.

### RAG concepts
- **Character Stream Fidelity:** Vector databases and LLMs are only as good as the raw text stream. If extraction produces corrupted characters (`\uf0b7`) instead of bullets (`•`), BM25 exact matching fails and embedding similarity degrades.
- **Financial Semantic Preservation:** Recognizing that symbols in financial tables (dashes, asterisks, brackets indicating negative numbers) carry critical semantic meaning and must never be stripped during text cleaning.

---

## 7. Problems we encountered

### Problem: Wingdings and Symbol Font Artifacts
- **Problem:** PyMuPDF extracted bullet points and disclosure symbols from the HDFC Bank report as unprintable characters (`\uf0b7`, `\uf02d`), corrupting sentences and headers.
- **Cause:** Microsoft Word and desktop publishing tools map custom symbols to the Private Use Area (PUA) of Unicode (`0xE000`–`0xF8FF`). PDF text extractors have no default translation table for these proprietary fonts.
- **Fix:** Built `config/symbol_font_mappings.yaml` mapping Wingdings and Symbol PUA hex codes to canonical Unicode bullets (`•`), arrows (`→`), and checkmarks (`✓`), wrapped behind `SymbolFontMapper`.
- **Lesson:** PDF text extraction is not equivalent to semantic text extraction; font metadata must be actively inspected and normalized.

---

## 8. Important measured results

| Metric | Measured Value | Workload / Context | Status |
| :--- | :--- | :--- | :--- |
| **Total Pages Processed** | 678 pages | HDFC Bank FY2025-26 Annual Report | Measured |
| **Parsing Throughput** | ~70–80 pages/second | PyMuPDF text stream extraction | Measured |
| **Mapped Symbol Fonts** | 6 font families | Wingdings, Wingdings 2/3, Symbol, Webdings, Zapf | Configured |
| **Symbol Preflight Scan** | 0 unmapped fatal glyphs | Full scan of primary HDFC report | Measured |

---

## 9. Why this design?

1. **Why PyMuPDF (`fitz`) over `pypdf` or `pdfminer.six`?** PyMuPDF wraps the MuPDF C-rendering library. It is 15–25x faster than pure-Python libraries, extracts accurate spatial bounding boxes at line and word level, and provides direct access to embedded font tables and raw character spans.
2. **Why a YAML Living Mapping File?** Corporate annual reports from different companies (e.g. Reliance, Apple, HDFC) use different visual fonts for bullets and tables. A YAML mapping enables rapid operational adaptation without code refactoring.
3. **Why Track `raw_character_count` per Page?** To detect silent text omission. If downstream layout extraction extracts 1,500 characters on a page where the raw stream had 2,500 characters, the discrepancy is immediately caught by preflight audits.

---

## 10. Things to remember

1. **1-Indexed Always:** Physical PDF page 1 must be `page_number=1`.
2. **Don't Strip Dashes:** Standalone dashes in financial tables mean `0` or `NIL`; never strip them as whitespace.
3. **PUA Glyphs Corrupt Retrieval:** Characters like `\uf0b7` break BM25 tokenization and degrade vector search; normalize them to canonical Unicode.
4. **PyMuPDF is Ground Truth:** The raw text stream from PyMuPDF defines the character baseline for all downstream audits.
