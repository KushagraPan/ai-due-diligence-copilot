# AGENTS.md: AI Due Diligence Co-pilot Operating Manual

Welcome, Agent. This repository houses the **AI Due Diligence Co-pilot**, an enterprise-grade, source-grounded Retrieval-Augmented Generation (RAG) system engineered to analyze corporate diligence documents (annual reports, 10-K/10-Q filings, financial statements, investor presentations, and M&A contracts).

This project is built as a **real portfolio-quality, production-oriented product**, not as a quick course demo or proof-of-concept hack. Every module, test, and prompt must reflect high professional standards.

This document serves as your binding operating manual when reading, developing, testing, or refactoring code in this repository.

---

## 1. Project Goal & Current Priority

### 1.1 Project Goal
The AI Due Diligence Co-pilot is a source-grounded RAG application that ingests corporate diligence documents, extracts text and tables, structures chunks with rich provenance metadata, embeds and indexes them into Weaviate Cloud, retrieves relevant candidates using hybrid search, reranks candidate passages, constructs evidence-grounded contexts, and synthesizes analytical answers with exact page and section citations.

### 1.2 Current Development Priority (Single-Document End-to-End Vertical Slice)
The primary foundation is a production-hardened **single-document end-to-end RAG vertical slice**:
```
ONE REAL DOCUMENT (HDFC Bank FY2025-26 Annual Report - 678 Pages)
  └── Text/Table Extraction (3-tier extractor, 5-point quality gate, layout single-owner invariant)
      └── Document-Aware Chunking (1,984 canonical chunks, repeated table headers)
          └── Metadata Extraction & Attachment (heading_path, pages, element IDs)
              └── Dense Embeddings Generation (text-embedding-3-large, 1,536 dims)
                  └── Weaviate Cloud Indexing (DiligenceChunk BYOV, HNSW + BM25)
                      └── Query Rewriting (gpt-4o-mini, 4-turn bounded history, fast-path)
                          └── Hybrid Retrieval (Vector + BM25, alpha=0.50, Top-N=40)
                              └── Entity Scope Filtering (Parent vs. Subsidiary isolation)
                                  └── LLM Reranking (gpt-4o-mini, 10 principles, Top-K=7)
                                      └── Evidence Context Builder (XML wire format, statement labels)
                                          └── Grounded LLM Generation (gpt-4o, 8 grounding rules)
                                              └── Citation Verification (Deterministic Python, check_claim_support)
```
Only after this vertical slice is thoroughly tested and reliable end-to-end should advanced agents, multi-company indexing, UI layers, or automated evaluation frameworks be introduced.

---

## 2. Core Principles & Hard Constraints

### 2.1 Strict Grounding and Uncertainty Handling
Due diligence directly impacts investment and legal decisions. Hallucinations or unsupported extrapolation can lead to critical misjudgments.
- **Strict Evidence Support:** Every factual claim, number, date, or contractual clause in generated analysis must be directly supported by retrieved source evidence.
- **Explicit Unknowns and Ambiguity:** If evidence is missing, insufficient, ambiguous, or contradictory, explicitly state that (e.g., *"Not disclosed in the ingested document"* or *"Section 4.1 contradicts Exhibit B regarding indemnification caps"*). Never make assumptions or educated guesses.
- **Never Synthesize Uncited Financial Data:** Never invent or extrapolate financial metrics (EBITDA, ARR, burn rate, debt maturities, valuation caps) unless the source explicitly supports the calculation.
- **Fact vs. Interpretation Separation:** Clearly distinguish between verbatim source facts and model-generated analytical interpretation.
- **Disclaimer of Advice:** Never present system outputs as definitive legal opinions or investment recommendations.

### 2.2 MVP-First Development & Anti-Patterns
Build strictly through incremental vertical slices. Validate each slice before moving to the next.
- **Do NOT prematurely implement:**
  - Multiple specialized agents (financial agent, legal agent, tax agent, etc.)
  - Complex multi-agent orchestrators or state graphs
  - Production authentication and user management
  - Elaborate FastAPI routing matrices or WebSocket streaming
  - Advanced evaluation frameworks (Ragas pipelines) before basic retrieval works
  - Premature abstractions and framework bloat
- Focus exclusively on making the single-document RAG pipeline robust, tested, and verifiable.

### 2.3 Single Document First
- The first working iteration must operate on **ONE real-world corporate document** (e.g., an annual report or 10-K).
- Do not build or test across 10 companies or dozens of files simultaneously before the single-document ingestion, retrieval, and citation pipeline is proven.
- Design modular interfaces so that expanding to multi-document and multi-company deal rooms requires no core architectural rewrite.

### 2.4 Data Confidentiality & Security
Deal rooms handle material non-public information (MNPI), proprietary financials, and sensitive terms.
- **Zero Hardcoded Secrets:** All credentials, endpoints, and API keys must be loaded exclusively via environment variables (`.env`). Never commit `.env` files.
- **No Unnecessary Data Exposure:** Do not log raw document text or sensitive customer data to external tracing systems or console outputs.
- **Authorized API Destinations Only:** Do not send document contents to third-party endpoints other than explicitly configured LLM/embedding/Weaviate services.

---

## 3. RAG Architecture & Subsystem Responsibilities

Maintain clean, decoupled separation across the core pipeline stages:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  INGESTION   │ ──> │  RETRIEVAL   │ ──> │  RERANKING   │ ──> │  GENERATION  │ ──> │ CITATION VER │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

1. **Ingestion Subsystem:**
   - Raw Document $\to$ Text & Table Extraction (3-tier) $\to$ Layout Spatial Reconciliation $\to$ Document-Aware Chunking (1,984 canonical chunks) $\to$ Metadata Attachment $\to$ Dense Embedding Generation (`text-embedding-3-large`, 1,536 dims) $\to$ Weaviate Cloud Storage (`DiligenceChunk`).
   - Ingestion is purely offline data preparation; it does not perform ad-hoc retrieval or generation.

2. **Retrieval Subsystem (B3 & B4):**
   - User Query $\to$ Conditional Query Rewriting (`gpt-4o-mini`, 4-turn bounded window, fast-path for standalone queries) $\to$ Weaviate Native Hybrid Retrieval (Dense Vector + BM25 Lexical, $\alpha=0.50$, Top-$N=40$) $\to$ Entity Scope Filtering (`filter_subsidiary_candidates`).
   - Retrieves and filters a high-recall candidate pool.

3. **Reranking Subsystem (B5):**
   - Query + Retrieved Candidates ($N=40$) $\to$ LLM Reranker (`gpt-4o-mini`, JSON mode, 10 financial ranking principles) $\to$ Top Evidence ($K=7$ chunk IDs).
   - Conceptually distinct from retrieval. The system previously tested local cross-encoders, which were completely removed due to 50s CPU latency and 512-token table truncation. B5 is now exclusively an LLM-based reranker.

4. **Context Construction Subsystem (B6):**
   - Reranked Chunks ($K=7$) $\to$ Deduplication $\to$ Statement Scope Label Injection (`[STANDALONE ...]` / `[CONSOLIDATED ...]`) $\to$ U-shaped attention ordering $\to$ Sanitized XML serialization (`<diligence_context>`) with deterministic evidence tokens (`E1`...`E7`).

5. **Generation Subsystem (B7):**
   - Original Query + Reranked Evidence Context $\to$ Grounded LLM Generator (`gpt-4o`, temperature=0.0, JSON object mode) $\to$ Structured Answer with `[EX]` citations under 8 strict grounding rules.

6. **Citation & Claim-Support Verification Subsystem (B8):**
   - Generated Answer + Evidence Map $\to$ Deterministic Python Verifier (`CitationVerifier`).
   - Executes physical lexical and entity/numeric claim-support checks (`check_claim_support()`), resolves nearest section headings (`resolve_citation_heading()`), and translates tokens into user-facing citations `[1]`, `[2]`.
   - Never fabricates coordinates; unverified citations are converted to `[Unverified: EX]` with diagnostic warnings.

---

## 4. Technical Specifications & Standards

### 4.1 Vector Database: Weaviate Cloud
- **Primary Vector Database:** Weaviate Cloud is the primary vector store for this project.
- **Capabilities Used:** Vector search, BM25 keyword search, native hybrid retrieval, and tenant/metadata filtering.
- **No Redundant Stores:** Do not introduce secondary vector databases (Chroma, FAISS, Pinecone, pgvector) unless explicitly mandated.
- **No Fake Local Stores:** Do not maintain synthetic or mocked local vector store engines unless strictly designated for lightweight unit testing.

### 4.2 Document-Aware Chunking & Provenance
Chunking must be document-aware, respecting headings, sections, and tabular structures, rather than naive fixed-character splitting.
- Every chunk must preserve structured metadata:
  - `company`: Name or ticker of the entity
  - `document_name`: Filename or title
  - `document_type`: Annual Report, 10-K, Presentation, Agreement, etc.
  - `financial_period`: FY2023, Q3 2024, etc. (where applicable)
  - `page_number`: Exact source page number (1-indexed)
  - `section_heading`: Article, section, or chapter header
  - `source_url` or `source_path`: Original document reference
  - `chunk_id`: Unique deterministic chunk identifier
  - `chunk_text`: Extracted content (text or Markdown table)
- Chunks must enable seamless trace-back to the exact page and section in the original file.

### 4.3 Query Rewriting Guidelines
- Query rewriting is handled by `QueryRewriter` using **`gpt-4o-mini`** (configured via `Settings.query_rewriter_model`).
- Fast-Path Invariant: If conversation history is empty or whitespace-only, the rewriter immediately returns `user_query` verbatim with **zero LLM API calls**.
- Bounded Window: When history is present, it is strictly capped to the most recent 4 turns (messages, not full dialogues).
- Prompt Injection Defense: History and query inputs are sanitized against XML delimiters (`<conversation_history>`, `<current_query>`, `<rewritten_query>`).
- Dual Query Separation:
  - `retrieval_query`: Emitted by the rewriter to optimize Weaviate hybrid search and B5 candidate scoring.
  - `original_query`: Preserved verbatim and passed to B7 Generation to ensure the model answers the user's literal inquiry without conversational drift.
- Zero Fact Injection: The rewriter must **never introduce external assumptions, company names, or facts** absent from the conversation history.

### 4.4 Reranking Strategy
- Retrieval fetches a broader candidate pool (Top $N=40$).
- The production reranker is `LLMReranker` powered by **`gpt-4o-mini`** (configured via `Settings.llm_reranker_model`), operating at `temperature=0.0` with `response_format={"type": "json_object"}`.
- It evaluates all Top-40 candidates against 10 strict financial ranking principles (answerability, exact evidence, primary statements, entity/metric scope, tables as first-class citizens, multi-page notes, citation usefulness, conversational context, and negative queries).
- It emits at most Top-$K=7$ chunk IDs (`{"ranked_chunk_ids": [...]}`). Unknown/duplicate IDs are rejected, and API errors fall back deterministically to B4 input rank order.
- Pipeline flow:
  $$\text{Weaviate Hybrid Search} \xrightarrow{\text{Top } N = 40} \text{LLM Reranker (gpt-4o-mini)} \xrightarrow{\text{Top } K = 7} \text{Context Assembly}$$
- **Historical Note:** The repository previously explored local cross-encoders. They were removed because they suffered from 50s CPU latency, 512-token context truncation, and systematic bias against tabular markdown schedules.

### 4.5 Citation & Provenance Rules
- Every statement or extracted metric presented to the user must carry a verifiable citation:
  - Format example: `[HDFC Bank Annual Report FY2025-26 — Page 378 — Standalone Balance Sheet]`
- **Deterministic Token Indirection:** B7 LLMs are strictly forbidden from typing page numbers or document names from memory; they emit only abstract tokens (`[E1]`, `[E2]`). B8 translates these into formatted footnote citations.
- **Active Claim-Support Verification (`check_claim_support`):** Every headline number ($\ge 4$ digits, %, currency) and multi-word capitalized entity in a citing sentence must be present in the cited chunk text. Violations convert the token to `[Unverified: EX]`.
- **Nearest Heading Resolution (`resolve_citation_heading`):** Citations resolve the nearest true section heading, filtering out table data rows, Total lines, and footnote markers.
- If page numbers cannot be reliably extracted from a given file, the system reports `[Page: Unknown]` rather than guessing.

---

## 5. Engineering Standards & Code Style

1. **Python Environment:**
   - Python 3.11+.
   - Dependency management via `pyproject.toml` and `uv`.

2. **Clean Python over Abstraction Bloat:**
   - Write clear, idiomatic, readable Python.
   - Avoid creating deep inheritance hierarchies, excessive design patterns, or wrapper classes around standard libraries until real necessity emerges.
   - Use **Pydantic v2** where structured domain validation (schemas, chunk metadata, response formats) is genuinely useful.

3. **Pragmatic Concurrency (No Async Cargo-Culting):**
   - Do not introduce async complexity merely for the sake of being "async-first."
   - Use `async` when the selected library (e.g., FastAPI, async HTTP clients) and application I/O genuinely benefit from it. Keep data processing and parsing synchronous when naturally CPU/file-bound.

4. **Resilience & Explicit Error Handling:**
   - External API calls (Weaviate, LLM, Embeddings) must include appropriate retry logic with exponential backoff and descriptive error messages.
   - Never use bare `except:` or silently swallow exceptions.
   - Log errors with contextual metadata (`document_name`, `chunk_id`, `query`).

5. **Prompt Management:**
   - Store system and task prompts in `src/due_diligence/prompts/` once the corresponding component exists.
   - Do not build an over-engineered prompt orchestration framework ahead of time. Plain Python modules or templated files are preferred.

6. **Educational Code Documentation:**
   Because this repository is also being developed as a learning project, every new program file must contain a clear module-level docstring explaining:
   1. The purpose of the module.
   2. Its role in the overall RAG pipeline.
   3. Its expected inputs.
   4. Its outputs.
   5. Its dependencies on other project components.
   6. What responsibilities are explicitly outside the module.

   Important classes and functions must have concise docstrings explaining their purpose, inputs, outputs, and important behavior.

   Use inline comments only where they explain non-obvious implementation or architectural decisions. Do not add comments that merely restate the code.

   Documentation should use simple, technically accurate language so that a developer learning the system can understand why the code exists and how it connects to the rest of the RAG pipeline.

7. **Documentation Style:**
   Keep module docstrings, function docstrings, and comments concise.
   Documentation must explain the essential purpose, inputs, outputs, and important design decisions without repeating information that is already obvious from the code.
   Prefer a few precise lines over long explanations.
   Use comments to explain WHY something is done when it is not obvious, not WHAT the code literally does.

   - **Avoid:**
     - Verbose essays inside source files
     - Comments that restate code
     - Redundant documentation
     - Obvious comments such as `# loop through chunks`

   - **Good Example:**
     ```python
     """
     Extract page-level text from a PDF while preserving page metadata.
     Used by ingestion before chunking and indexing.
     """
     ```

   - **Avoid Example:**
     ```python
     """
     This module is responsible for extracting text from PDF documents.
     It does this by opening the PDF, reading each page, and extracting the text.
     The extracted text is then returned for further processing...
     """
     ```

---

## 6. Testing & Quality Assurance

Every completed component must be accompanied by focused unit or integration tests. Do not declare a milestone done without verification.

### 6.1 Priority Test Areas
- **Parsers:** Extracting text and tables accurately from sample PDF pages.
- **Chunkers:** Preserving document structure, table rows, and correct metadata fields.
- **Metadata Preservation:** Ensuring zero metadata loss between raw document and chunk.
- **Retrieval:** Validating that hybrid search correctly returns expected candidate passages.
- **Citation Provenance:** Ensuring generated answers accurately cite the source chunks.
- **End-to-End RAG:** Running full query-to-answer pipelines on real sample documents.

### 6.2 Negative & Edge Case Testing
Test systems against realistic failure modes:
- Empty or corrupted documents.
- Scanned/malformed PDFs lacking clean text streams.
- Missing page or section metadata.
- Duplicate or near-identical chunks.
- Queries where the document contains zero relevant evidence (must produce explicit uncertainty).
- Ingested passages containing conflicting statements.

---

## 7. Development Workflow Protocol

Always implement features **one milestone at a time**. Strictly follow this 8-step protocol:

1. **Read `AGENTS.md`** to verify constraints and standards.
2. **Read `_docs/architecture.md`** to maintain architectural consistency.
3. **Read `_docs/plan.md`** to determine the current milestone.
4. **Identify the single active task** from the plan.
5. **Implement only that task** without speculative future additions.
6. **Add or update tests** covering positive and edge cases.
7. **Run the tests** to prove functionality.
8. **Update `_docs/plan.md`** to record progress.

Do not skip steps. Do not implement future phases speculatively.
