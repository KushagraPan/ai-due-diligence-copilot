# Document Ingestion Engine & Document Validation

## 1. What is this?

The **Document Ingestion Engine** is the initial gatekeeper of the AI Due Diligence Co-pilot. It is responsible for accepting raw corporate diligence files (primarily PDFs), validating their physical integrity and MIME types, extracting authoritative file metadata, computing cryptographic content hashes, and instantiating strongly typed document representations that anchor provenance across the entire RAG pipeline.

In corporate diligence deal rooms, files are frequently misnamed, partially corrupted during upload, password-protected, or formatted with non-standard internal specifications. If corrupted or unreadable documents slip past the ingestion gate into chunking and vectorization, downstream subsystems will either fail with cryptic runtime exceptions or silently index empty content.

The ingestion engine enforces a "fail-loud, fail-early" policy. Before any PDF parsing, table extraction, or embedding occurs, the file must be cryptographically hashed (SHA-256), verified on the filesystem, checked for structural integrity, and represented by an immutable [`ValidatedDocument`](../../src/due_diligence/ingestion/document.py#L25-L50) instance.

---

## 2. Where does it fit?

The Ingestion Engine sits at the very front of the Track A pipeline:

```
[Local File / Deal Room Upload]
              │
              ▼
    loaders.py (PDFLoader)
   ├── Magic Bytes / MIME Sniffing
   ├── File Integrity & Readability Checks
   └── SHA-256 Checksum Computation
              │
              ▼
   document.py (ValidatedDocument)
              │
              ▼
   preflight.py (Scan 1: Font Glyph Audit)
              │
              ▼
       Downstream Ingestion
    (parser.py, table_extractor.py)
```

---

## 3. What did WE implement?

- [`src/due_diligence/ingestion/loaders.py`](../../src/due_diligence/ingestion/loaders.py):
  - [`PDFLoader`](../../src/due_diligence/ingestion/loaders.py#L38-L120): Validates filesystem paths, checks file existence and permissions, reads raw file headers to verify PDF magic bytes (`%PDF-`), calculates SHA-256 checksums, and extracts page counts using PyMuPDF.
  - Typed exceptions: `DocumentLoadError`, `UnsupportedFormatError`, `CorruptDocumentError`.
- [`src/due_diligence/ingestion/document.py`](../../src/due_diligence/ingestion/document.py):
  - [`ValidatedDocument`](../../src/due_diligence/ingestion/document.py#L25-L50): Immutable Pydantic model encapsulating `file_path`, `document_name`, `company`, `document_type`, `financial_period`, `content_hash`, `file_size_bytes`, and `page_count`.
  - [`ExtractedDocument`](../../src/due_diligence/ingestion/document.py#L53-L95): In-memory container holding page-level text streams, font provenance, and extracted document elements.
- [`src/due_diligence/ingestion/preflight.py`](../../src/due_diligence/ingestion/preflight.py):
  - [`preflight_scan`](../../src/due_diligence/ingestion/preflight.py#L49-L150): Embedded font preflight scanner inspecting embedded font tables before parsing.

---

## 4. Data flow

```
[Raw PDF Path: "data/raw/hdfc-bank__annual-report__fy2025-26.pdf"]
                           │
                           ▼ (PDFLoader.load)
1. Verify path exists, is file, and is readable.
2. Read header: verify b"%PDF-".
3. Compute SHA-256 hash across full byte stream.
4. Inspect size in bytes.
5. Open via PyMuPDF: verify page count > 0, check encryption flags.
                           │
                           ▼
[ValidatedDocument]
  - file_path: Path("data/raw/hdfc-bank__annual-report__fy2025-26.pdf")
  - document_name: "hdfc-bank__annual-report__fy2025-26.pdf"
  - company: "HDFC Bank"
  - document_type: "annual_report"
  - financial_period: "FY2025-26"
  - content_hash: "92de3c75b6fc05afe8731e97705c9d3534cc192e677af02f4b325e3d82eafca6"
  - file_size_bytes: 12,667,766
  - page_count: 678
                           │
                           ▼ (preflight_scan)
[Font Preflight Audit Report]
  - Verifies embedded fonts against config/symbol_font_mappings.yaml
                           │
                           ▼
Passed to parser.py & layout.py
```

---

## 5. Important implementation details

- **The Two Preflight Concepts:** The repository contains two distinct preflight mechanisms that must never be confused:
  1. *Embedded Font Glyph Preflight ([`preflight.py`](../../src/due_diligence/ingestion/preflight.py)):* Runs *before* text extraction to identify non-standard font codepoints (Wingdings, symbol fonts) so missing glyphs can be mapped in `config/symbol_font_mappings.yaml`.
  2. *Full-Corpus Audit Runners ([`scripts/run_*_preflight.py`](../../scripts/)):* Run *after* table and layout extraction across all 678 pages to verify block conservation (zero unowned blocks) and table quality before spending money on chunking and vector indexing.
- **Cryptographic Provenance:** The `content_hash` (SHA-256) is attached to every downstream chunk and Weaviate object. If an analyst uploads an updated version of a document, the hash difference triggers atomic invalidation and re-indexing.
- **Magic Bytes Validation:** Rather than blindly trusting file extensions (`.pdf`), the loader inspects the initial bytes for the `%PDF-` signature.
- **1-Indexed Page Number Invariant:** Page numbers in `ValidatedDocument` and downstream components are strictly 1-indexed (Page 1 is physical page 1). This ensures 100% fidelity with human visual page numbers in PDF viewers.

---

## 6. Concepts I learned

### Python concepts
- **`pathlib.Path` Object Model:** Object-oriented filesystem path manipulation replacing fragile string concatenations.
- **`hashlib.sha256` Streaming:** Computing cryptographic digests in chunked buffers (e.g. 64KB blocks) to hash multi-hundred-megabyte files without exhausting RAM.
- **Pydantic v2 Model Invariants:** Leveraging field validators and frozen model configurations (`ConfigDict(frozen=True)`) to ensure metadata integrity across processing stages.

### RAG concepts
- **Provenance Anchoring:** Establishing immutable document identity at ingestion time so that every downstream retrieved chunk can be definitively traced back to its physical origin.
- **Fail-Loud Data Ingestion:** Catching corruption, unreadable text streams, or unsupported formats before allocating expensive vector database storage and compute.

---

## 7. Problems we encountered

### Problem: Misleading File Extensions and Unchecked Payloads
- **Problem:** Deal room files can have corrupted headers or incorrect `.pdf` extensions (e.g. HTML login redirect pages saved as `.pdf`).
- **Cause:** Naive ingestion scripts open files directly with PDF rendering libraries, leading to unhandled segmentation faults or cryptic C-level crashes.
- **Fix:** Implemented header byte inspection in `PDFLoader` checking for `%PDF-` and inspecting PyMuPDF's `is_encrypted` and `is_repaired` flags.
- **Lesson:** Never trust file extensions; always inspect magic numbers and file headers before invoking third-party document parsers.

---

## 8. Important measured results

All numbers measured on the primary validation corpus:

| Metric | Measured Value | Workload / Context | Status |
| :--- | :--- | :--- | :--- |
| **Corpus File Size** | 12,667,766 bytes (~12.08 MB) | `hdfc-bank__annual-report__fy2025-26.pdf` | Measured |
| **Physical Page Count** | 678 pages | PyMuPDF page count check | Measured |
| **Content Hash (SHA-256)** | `92de3c75b6fc05afe8731e97705c9d3534cc192e677af02f4b325e3d82eafca6` | Full byte stream hash | Measured |
| **Loader Execution Latency** | < 15 ms | Validation, hashing, and PyMuPDF init | Measured |

---

## 9. Why this design?

1. **Why Compute SHA-256 Hashes?** In due diligence deal rooms, files are frequently re-uploaded with identical filenames but revised numbers. A SHA-256 content hash allows the vector store to detect whether a document has truly changed, preventing duplicate indexing while enabling atomic invalidation when contents differ.
2. **Why Enforce `ValidatedDocument` as an Immutable Pydantic Model?** Downstream extraction and chunking stages run concurrently or sequentially. An immutable model prevents accidental in-place mutation of core document provenance.
3. **Why Preflight Embedded Fonts Before Parsing?** Symbol fonts use private-use Unicode blocks. Discovering unmapped symbols after indexing 2,000 chunks requires a costly re-indexing pass. Discovering them at ingestion time takes milliseconds.

---

## 10. Things to remember

1. **Validate Before Processing:** Check magic bytes, encryption, and physical page count before calling layout extractors.
2. **Provenance Starts Here:** `content_hash`, `document_name`, and `company` originate in `ValidatedDocument`.
3. **1-Indexed Pages:** Always maintain physical 1-indexed page numbering.
4. **Two Preflights:** Font preflight scans glyphs *before* parsing; corpus preflight audits layout conservation *after* extraction.
