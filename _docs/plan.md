# Implementation Plan: AI Due Diligence Co-pilot

This roadmap outlines the phased development plan for the AI Due Diligence Co-pilot. Each phase includes clear deliverables, verification criteria, and acceptance criteria.

---

## Phase Status Summary

| Phase | Focus Area | Status | Target Deliverables |
|---|---|---|---|
| **Phase 0** | Project Setup & Architecture Scaffolding | **Completed** | Repository skeleton, docs, configs, package layout |
| **Phase 1** | Ingestion & Document Processing Engine | **Completed** | Multi-format parsing, table extraction, chunking |
| **Phase 2** | Dual Indexing & Hybrid Retrieval Engine (Track B) | **Completed** | Weaviate Cloud, Dense + BM25, LLM reranker, schemas, golden benchmarks |
| **Phase 3** | Domain Diligence Agents & Workflows | Not Started | Financial, Legal, Operational & Synthesis agents |
| **Phase 4** | Grounding Guardrails & Attribution Engine | Not Started | Exact citations, hallucination grader, confidence scoring |
| **Phase 5** | API Layer & Deal Room Session Management | Not Started | FastAPI async endpoints, SSE streaming, deal isolation |
| **Phase 6** | Automated Evaluation & Benchmarking | Not Started | Golden dataset tests, Ragas evaluation pipeline |
| **Phase 7** | Diligence Memorandum Generation & Export | Not Started | Markdown/PDF report generator, executive summaries |

---

## Detailed Milestones & Tasks

### Phase 0: Project Setup & Architecture Scaffolding
- [x] Define directory structure adhering to modular boundaries.
- [x] Draft `AGENTS.md` guidelines for AI agents working in this repository.
- [x] Draft `_docs/architecture.md` detailing system design and data flow.
- [x] Draft `_docs/plan.md` defining implementation roadmap.
- [x] Initialize dependency configuration (`pyproject.toml`, `.env.example`, `.gitignore`, `README.md`).
- **Definition of Done:** Project repo is structured, documented, and ready for incremental feature development without any ungrounded code.

---

### Phase 1: Ingestion & Document Processing Engine
- [x] Implement file type detector and validation in `src/due_diligence/ingestion/loaders.py`.
- [x] Implement PDF page-level text extraction with 1-indexed page tracking in `src/due_diligence/ingestion/parser.py`.
- [x] Harden symbol-font text extraction: living mapping config (`config/symbol_font_mappings.yaml`), dash preservation with interim standalone `[NIL/NOT DISCLOSED]` heuristic, batch preflight scanner (`src/due_diligence/ingestion/preflight.py`), and document/page-level `symbol_font_status` provenance tracking in `ExtractedDocument` (Part D).
- [x] Implement financial table extractor (preserving rows, columns, and Markdown headers) with 3-tier architecture (Tier 1 column-projection, Tier 2 Gemini Flash multimodal vision fallback, Tier 3 graceful degradation) and DocumentElement hierarchy in `src/due_diligence/ingestion/elements.py` and `table_extractor.py`.
- [x] Implement layout-aware text extraction and spatial reconciliation layer in `src/due_diligence/ingestion/layout.py` unifying narrative text, headings, list items, and tables with spatial table exclusion, multi-column detection, and hierarchical section heading propagation.
- [x] Implement document-aware chunking with metadata tagging (`section_heading`, `heading_path`, `page_numbers`, `source_element_ids`, `overlap_tokens`) in `src/due_diligence/ingestion/chunks.py` and `chunker.py`. Full-corpus validation against 678-page HDFC FY2025-26 Annual Report: 10,851 DocumentElements -> 2,005 Chunks. Provenance invariant: valid (0 errors). 20 unit tests (all pass). Ruff + mypy strict: 0 errors.
- [x] Add unit tests for document loaders, table extraction, layout reconciliation, and chunking in `tests/unit/`.
- **Definition of Done:** Able to ingest sample 10-K and M&A contract PDFs, correctly extract tabular data into Markdown, and produce structured chunks with complete metadata.

---

### Phase 2: Dual Indexing & Hybrid Retrieval Engine (Track B)
The original engineering implementation plan is superseded by [`architecture.md`](architecture.md) and [`rag_architecture.md`](rag_architecture.md).

- [x] **Phase B1: Core Configuration & Retrieval Schemas**
  - Implemented centralized settings in `src/due_diligence/core/config.py` using `pydantic-settings` `BaseSettings` (`weaviate_url`, `weaviate_api_key`, `embedding_provider`, `embedding_model`, `embedding_dimension`, `hybrid_alpha=0.60`, `retrieval_top_n=25`, `reranker_model`, `rerank_top_k=5`, `generation_model`) loaded from environment with zero hardcoded secrets.
  - Implemented system boundary contract and retrieval models in `src/due_diligence/retrieval/schemas.py`:
    - `IndexingPayload`: Immutable Pydantic model bridging `Chunk` and `ValidatedDocument` metadata for Weaviate indexing, including automated table metadata unpacking (`is_table`, `table_headers`, `table_caption`).
    - `RetrievalResult`: Immutable typed candidate representation emitted by hybrid retrieval for consumption by reranker and context builder.
  - Updated `.env.example` with Weaviate Cloud variables replacing legacy stores.
  - Added 63 comprehensive unit tests (`tests/unit/test_config.py`, `tests/unit/test_retrieval_schemas.py`) covering defaults, overrides, validation, serialization round-trips, immutability, and edge cases. Total test suite: 142 tests passing. Strict mypy & ruff: 0 errors.
- [x] **Phase B2: Dense Embedding Provider Abstraction**
  - Implemented `BaseEmbeddingProvider` abstract interface in `src/due_diligence/retrieval/embeddings.py` defining the contract for `embed_documents(texts: list[str]) -> list[list[float]]` and `embed_query(query: str) -> list[float]` with strict input-to-output order preservation and dimension invariants.
  - Implemented `OpenAIEmbeddingProvider` using modern OpenAI SDK with Matryoshka dimension reduction (1536 dims), batch slicing (default 64), response index sorting, and custom exponential backoff retry with full jitter on transient rate limits / 5xx errors.
  - Implemented `LocalBGEEmbeddingProvider` enforcing native model dimensionality (1024 dims for `bge-large-en-v1.5`), asymmetric retrieval query prefix injection (`Represent this sentence for searching relevant passages: `), and unit L2 vector normalization.
  - Implemented authoritative factory `get_embedding_provider()` enforcing provider-owned dimensionality: validates configured `embedding_dimension` against provider native capabilities and raises `EmbeddingConfigurationError` on mismatch without silent overrides.
  - Added 33 unit tests in `tests/unit/test_embeddings.py` covering validation, offline mocking, scrambled index re-sorting, retry backoff, exhausted retry errors, native dimension lookups, and factory enforcement. Total test suite: 175 tests passing. Strict mypy & ruff: 0 errors.
- [x] **Phase B3: Weaviate Cloud Indexer & Collection Manager**
  - Implemented `src/due_diligence/retrieval/weaviate_store.py` providing complete client lifecycle management, typed exception hierarchy (`WeaviateStoreError`, `WeaviateConnectionError`, `WeaviateSchemaError`, `WeaviateAlignmentError`, `WeaviateBatchIndexError`, `WeaviateVerificationError`), pre-indexing invariant validation (enforcing all 10 pre-flight checks), property mapping (`build_weaviate_properties`), and idempotent batch indexing (`index_canonical_corpus`).
  - Configured Bring Your Own Vectors (`vector_config=Configure.Vectors.self_provided`) with native `hfresh` vector indexing (with graceful fallback to `hnsw` cosine) and BM25 inverted indexing (`bm25_b=0.75, bm25_k1=1.2`) across 21 schema properties, safely defaulting to `recreate=False`.
  - Implemented deterministic UUIDv5 identifier generation from `chunk_id` for idempotent, non-duplicating upserts.
  - Implemented fail-loud fixed-size batching (`batch_size=100`) inspecting `failed_objects` on completion.
  - Implemented post-indexing health verification (`verify_collection_health`) checking aggregate cluster counts and vector dimension integrity on representative probe chunks.
  - Created 37 comprehensive unit tests in `tests/unit/test_weaviate_store.py` covering all 10 invariant failures, property mapping, lifecycle, safe recreate defaults, batch execution, and health verification. Total test suite: 215 tests passing.
  - Created operational execution script `scripts/run_hdfc_weaviate_indexing.py` and executed live against Weaviate Cloud cluster (`eu-central-1`): successfully indexed all 1,988 canonical HDFC Bank chunks with 1536-dimensional pre-computed OpenAI embeddings in 8.87 seconds (224.2 obj/sec). Zero API calls to OpenAI; verified 100% cluster health and 5 diverse probe chunks. Report persisted at `data/processed/hdfc_weaviate_indexing_report.json`.
- [x] **Phase B4: Native Weaviate Hybrid Retrieval Engine**
  - Implemented `src/due_diligence/retrieval/retriever.py` providing `WeaviateHybridRetriever` combining dense vector search and BM25 lexical matching in Weaviate Cloud.
  - Initial configuration: default hybrid alpha = 0.60, retrieval top_n = 25, dense embeddings via `BaseEmbeddingProvider` (`text-embedding-3-large`).
  - Implemented typed exceptions: `RetrievalError`, `RetrievalConfigurationError`, and `RetrievalQueryError`.
  - Added query pre-validation and graceful zero-candidate handling: empty or whitespace queries return `[]` without wasting embedding or database calls.
  - Implemented safe metadata filtering supporting company isolation (`company`), `document_name`, `document_type`, `financial_period`, `is_table`, and arbitrary Weaviate `_Filters` combined with `Filter.all_of`.
  - Preserved complete provenance: mapped stored `chunk_text` to schema `text`, extracted Weaviate metadata hybrid score, reconstructed `table_metadata`, and attached source page numbers and section heading breadcrumbs into typed `RetrievalResult` objects.
  - Enforced strict deterministic ordering: sorted by descending hybrid score with `chunk_id` ascending tie-breaking.
  - Created compatibility re-export in `src/due_diligence/retrieval/config.py` and exported public retrieval APIs in `src/due_diligence/retrieval/__init__.py`.
  - Added 22 mock-based unit tests in `tests/unit/test_retriever.py` covering initialization, empty/whitespace queries, pipeline execution, table reconstruction, deterministic ordering, tie-breaking, metadata filtering, parameter overrides, empty results, and upstream error propagation.
  - Total test suite: 238 unit tests passing. Strict mypy and ruff: 0 errors. Live Weaviate retrieval deliberately deferred per instructions.
- [x] **Phase B5: Cross-Encoder Reranker Layer & Interactive Optimization**
  - Implemented `src/due_diligence/retrieval/reranker.py` providing `BaseReranker`, `CrossEncoderReranker`, and aliases `BGEReranker` and `EttinReranker` using Hugging Face / Sentence Transformers.
  - Implemented two production profiles configurable via `Settings`:
    - **Interactive Demo Profile (Default):** `cross-encoder/ettin-reranker-17m-v1` (~70 MB, 17M parameters, ModernBERT architecture). Scores pruned B4 candidates (`candidate_count=10`, `batch_size=16`) in **612.35 ms / query** on CPU with **18.26 s** first-time load time.
    - **Reference Quality Baseline:** `BAAI/bge-reranker-large` (2.24 GB, 560M parameters). Scores full candidate pool (`candidate_count=25`) in **50.63 s / query** on CPU with **1,157.01 s** load time.
  - Added typed data model `RerankedResult` in `src/due_diligence/retrieval/schemas.py` preserving the original Weaviate B4 hybrid score in `score` while attaching the cross-encoder relevance score in `rerank_score`.
  - Added configurable settings in `src/due_diligence/core/config.py`: `rerank_candidate_count` (default: 10), `reranker_batch_size` (default: 16), `reranker_max_length` (default: 512), and `reranker_device` ("auto", "cpu", "cuda").
  - Enforced lazy model loading, CPU/CUDA auto-detection, `torch.inference_mode()`, and deterministic sorting (`-rerank_score, chunk_id`).
  - Integrated candidate count pruning into `RetrievalPipeline` and `cli.py` (`--rerank-candidates`).
  - Completed real model validation across all 6 standard HDFC queries: average Top-5 overlap with BGE reference baseline is **53.3%** (reaching 80.0% on financial queries). Weaviate collection verified unchanged at 1,988 objects. Report persisted to `data/processed/hdfc_b5_ettin_validation_2026-09-26.json`.
  - Total test suite: 350 unit tests passing. Strict mypy and ruff: 0 errors across 52 source files.
- [x] **Phase B6: Evidence Context Builder**
  - Implemented `EvidenceContextBuilder` (alias `ContextBuilder`) in `src/due_diligence/retrieval/context_builder.py` transforming B5 Top-K reranked candidates into structured, sanitized XML context blocks for Phase B7 Grounded Generation.
  - Implemented typed models in `src/due_diligence/retrieval/schemas.py`: `EvidenceItem`, `OmittedEvidenceItem`, and `EvidenceContext`.
  - Added deterministic context-local token ID assignment (`E1`, `E2`, ...) and 1-to-1 evidence lookup mapping (`evidence_map`) decoupling raw chunk IDs from generation prompts while preserving complete source provenance for citation attribution.
  - Implemented multi-tier deduplication (exact `chunk_id`, exact content text, and >85% `source_element_ids` overlap) and strict budget enforcement (`max_context_chars`, `max_evidence_items`) with non-destructive clean omission.
  - Implemented security sanitization (`sanitize_evidence_text`) preventing document content from breaking out of XML evidence boundaries (`<evidence>`, `</evidence>`, `<diligence_context>`, `</diligence_context>`).
  - Implemented both descending B5 rerank score ordering and U-shaped attention ordering ("lost_in_the_middle") via `apply_lost_in_the_middle_ordering`.
  - Added backward-compatible re-export module in `src/due_diligence/agents/context_builder.py` and exported public APIs in `src/due_diligence/retrieval/__init__.py`.
  - Added 23 comprehensive unit tests in `tests/unit/test_context_builder.py` covering ordering, deterministic IDs, evidence map, query preservation, score preservation, page fallbacks, tables, budget limits, XML delimiter sanitization, deduplication, attention ordering, and immutability.
  - Total test suite: 329 unit tests passing. Strict mypy & ruff: 0 errors across 52 source files.
- [x] **Phase B7: Grounded Generation**
  - Implemented `GroundedGenerator` in `src/due_diligence/agents/generator.py` synthesizing factual, citation-grounded diligence answers strictly from B6 `EvidenceContext` and the original user query.
  - Implemented immutable Pydantic response model `GroundedAnswer` (`answer_text`, `cited_evidence_ids`, `is_uncertain`).
  - Implemented structured system and user prompts in `src/due_diligence/prompts/generation.py` enforcing 5 non-negotiable grounding rules: strict evidence boundaries, mandatory `[EX]` citations, zero financial inventions, explicit fact vs. calculation separation, and explicit uncertainty declarations (`is_uncertain=True`).
  - Implemented deterministic response parser `parse_grounded_answer()` with markdown code block stripping, citation reconciliation between body text and JSON arrays, active evidence ID validation, and heuristic uncertainty safeguards.
  - Implemented fast-path zero-evidence handling returning explicit uncertainty (`is_uncertain=True`, `cited_evidence_ids=[]`) without consuming LLM API calls.
  - Added functional convenience wrapper `generate_grounded_answer()` and exported generation components in `src/due_diligence/agents/__init__.py` and `src/due_diligence/prompts/__init__.py`.
  - Added 20 comprehensive unit tests in `tests/unit/test_generator.py` covering fully grounded answers, insufficient evidence, numerical questions, multiple citations, prompt-injection defense in evidence, empty evidence fast paths, malformed JSON handling, input validation, citation filtering, and immutability.
  - Total test suite: 370 unit tests passing. Strict mypy & ruff: 0 errors across 55 source files.
- [x] **Phase B8: Citation Attribution & CLI Pipeline Integration**
  - Implemented `CitationVerifier` (alias `CitationEngine`) in `src/due_diligence/agents/citation_verifier.py` and `src/due_diligence/agents/citation_engine.py` translating B7 evidence tokens (`[EX]`) into structured `Citation` instances with exact physical document and page coordinates (`format_citation() -> [Document, Page(s), Section]`) and resolving text footnote markers (`[1]`, `[2]`).
  - Added typed immutable models `Citation` and `VerifiedAnswer` in `src/due_diligence/agents/citation_verifier.py`.
  - Implemented unverified/hallucinated citation token detection and explicit tagging (`[Unverified: EX]`) without guessing page numbers.
  - Implemented end-to-end `DiligencePipeline` in `src/due_diligence/pipeline.py` orchestrating the complete 6-stage online flow: Query Rewriter (`gpt-4o-mini`) -> B4 Hybrid Retrieval -> B5 LLM Reranker (`gpt-4o-mini`) -> B6 EvidenceContext -> B7 Grounded Generation (`gpt-4o`) -> B8 Citation Verification.
  - Updated CLI (`src/due_diligence/cli.py`) to execute `DiligencePipeline`, providing bounded multi-turn conversation memory, single pipeline/model instance reuse across turns, and structured terminal output (Answer, Verified Citations, Source Evidence excerpts, and Diagnostic Retrieval Trace).
  - Added 19 unit tests in `tests/unit/test_citation_verifier.py`, 18 exhaustive adversarial tests in `tests/unit/test_citation_verifier_adversarial.py`, and 7 comprehensive integration tests in `tests/integration/test_cli_integration.py` covering all 18 security and provenance boundary cases: valid [E1], invalid [E99], mixed [E1, E99], malformed tags ([EX], [E], [Eabc], [E_1]), duplicate tokens, grouped brackets, non-citation brackets (`[Calculated: ...]`, `[NIL/NOT DISCLOSED]`), prose page hallucinations (Page 999), missing page numbers, empty/whitespace answers, refusal handling, and strict metadata isolation.
  - Total test suite: 418 unit and integration tests passing. Strict mypy & ruff: 0 errors across 61 source files.
- [x] **Phase B9: Golden Evaluation Dataset & Automated Benchmark Runner**
  - Implemented evaluation data contracts in `src/due_diligence/evaluation/schemas.py`: `GoldenQuestion`, `GoldenDataset`, `RetrievalStageEval`, `RerankStageEval`, `GenerationStageEval`, `PipelineQueryEval`, `BenchmarkSummary`, `StageLatencies`, `CitationSummary`, `QuestionEvaluationRecord`, and four-pillar summaries (`RetrievalMetricsSummary`, `RerankerMetricsSummary`, `CitationMetricsSummary`, `AnswerQualitySummary`, `LatencyMetricsSummary`, `BenchmarkRunReport`).
  - Created 25-question golden benchmark dataset (`tests/eval/data/golden_qa_25.json` and `data/golden_qa_25.json`) covering 16 diligence failure modes for HDFC Bank FY2025-26 Annual Report.
  - **Audit Note & Dataset Caveat (2026-09-26):** Golden dataset was originally AI-authored from processed canonical chunks (`hdfc_canonical_chunks.jsonl`), not human-verified from the raw source PDF. Direct verification against the primary PDF (`data/raw/hdfc-bank__annual-report__fy2025-26.pdf`) uncovered:
    1. **Spurious/Bogus Expected Pages in Golden Dataset:** Q08 expected pages 57, 58, 563 (Executive Profiles and Gratuity notes) instead of actual retail digital banking pages (114-121, 280-281). Q20 erroneously included page 57 and 563 alongside genuine segment pages 470 and 582.
    2. **Entity Conflation in Retrieval (Parent vs. Subsidiary):** In Q04, the pipeline retrieved Page 304/305 (HDB Financial Services Limited - HDBFSL) and reported its PAT (₹2,544 cr / ₹2,176 cr) as HDFC Bank's net profit. The golden dataset correctly targeted the parent bank (Pages 273/379: ₹74,671 cr, 10.9%), confirming the need for `entity_scope` metadata filtering.
    3. **Unit Formatting Discrepancies:** The PDF balance sheet and schedules report figures in `₹ in ('000)` (thousands). Golden expected facts appended string labels like `"31,052,504,740 thousand"`, causing strict substring matching misses despite exact numerical extraction.
    4. **Uncertainty Traps Confirmed:** Q24 (commercial cards market share) and Q25 (Series C acquisition of FinTech Corp) verified as 100% genuine negative controls correctly absent from the PDF.
  - Implemented evaluation engine `DiligenceEvaluator` in `src/due_diligence/evaluation/evaluator.py` supporting stage-wise retrieval recall@N, cross-encoder rerank recall@K and Top-1 precision, normalized fact extraction and coverage, uncertainty/refusal compliance, and citation precision.
  - Implemented automated benchmark runner `BenchmarkRunner` in `src/due_diligence/evaluation/runner.py` coordinating all six production stages (`Query Rewriter (gpt-4o-mini) -> B4 Hybrid Retrieval -> B5 LLM Reranker (gpt-4o-mini) -> B6 Context Assembly -> B7 Grounded Generation (gpt-4o) -> B8 Citation Verification`) without bypassing any component, recording stage latencies (`time.perf_counter()`), intermediate artifacts, and deterministic metrics.
  - Created operational CLI benchmark script `scripts/run_b9_evaluation.py` supporting `--limit`, `--output-json`, and `--output-md`.
  - Executed live 25-question golden benchmark against Weaviate Cloud and GPT-4o in **91.57s** (mean latency: 3,662 ms/query; B4 Retrieval: 993 ms, B5 Rerank: 1,398 ms, B7 Generation: 1,232 ms).
  - Persisted results to `data/processed/b9_golden_evaluation_report.json` and `data/processed/b9_golden_evaluation_report.md`.
  - Core pipeline components B4–B8 remained 100% untouched.
  - Total test suite: 444 unit, eval, and integration tests passing in 13.33s. Strict mypy & ruff: 0 errors across 69 source files.
- [x] **Phase B9 follow-up: B6 explicit statement-scope labels (Standalone vs. Consolidated fix)**
  - (Previous session work; retained and exercised by the new B5 below.)
- [x] **Phase B5 re-architecture: cross-encoder removed, dedicated LLM reranker (2026-09-29)**
  - Deleted `CrossEncoderReranker`, `BGEReranker`/`EttinReranker` aliases, `TwoStageReranker`, and all `sentence-transformers`/`torch`/`transformers` production paths from `retrieval/reranker.py` (torch/transformers stay for LocalBGE embeddings). Removed `sentence-transformers` from `pyproject.toml`; `uv lock` regenerated.
  - New chunk-ID output contract (`{"ranked_chunk_ids": [...]}` with strict validation, dedup, deterministic fallback) replacing integer indices; `original_query` threaded through `BaseReranker` → `LLMReranker` → `RetrievalPipeline.run()`.
  - Rewrote `prompts/reranker.py` with 10 ranking principles (answerability, exact evidence, primary statements, entity/metric scope, tables first-class, multi-page, citation usefulness, conversational, negative-query) and full chunk text (4000-char cap with marker).
  - Config: `reranker_type` narrowed to `"llm"`; removed `reranker_model`, `reranker_batch_size`, `reranker_max_length`, `reranker_device`. `RetrievalPipeline` always builds `LLMReranker`. Fixed stale Ettin references in `pipeline.py`, `cli.py`, `run_golden_benchmark.py`, evaluator B5 latency label, and `_docs/rag_architecture.md` §7.
  - Tests: rewrote `tests/unit/test_reranker.py` (44 LLM-only tests, mocked client); updated `test_config.py`, `test_retrieval_pipeline.py`, `test_cli_integration.py`. Full suite: 484 unit+integration pass; ruff + mypy strict clean. Pre-existing set-1 golden eval failures (version string, stale chunk IDs) untouched.
  - Live validation `data/processed/b9_llm_reranker_validation_20260929.{json,md}` (Q01–Q25, unchanged goldens/B4/B6/B7/B8): 14/25 evaluator pass vs 17/25 prior run — NOT claimed as improvement (see analysis: B4 outputs byte-identical; flips are B5-composition + B7 variance + metric thresholds; Q12 net-over-gross is a genuine B7 selection flip with correct evidence present).
- [x] **Steps 1–6: Q06 Total-row fix, citation heading, B8 tripwire, golden v1.0.1, full re-run (2026-09-28)**
  - Step 1: Printed c01493/c01136. Confirmed: both Total rows dropped by Tier-1 (bbox clipped rule-less Total line; strict word containment), and Schedule 11's Total misattached as Schedule 12's caption via `istitle()`. Correction to initial report: c01493's body did not span schedules — contamination was caption-only.
  - Step 2: `extend_box_for_trailing_total()` (bbox swallows trailing Total, clamped, warns beyond reach) + `is_valid_table_caption()` guard + canonical Total/schedule patterns in `elements.py` + Total-reserved-for-final-slice in `_chunk_table`. Rebuilt: 11,008 elements → 1,984 chunks → 1,984 vectors → Weaviate `--recreate` (IDs embed index+hash). Q06 re-run: B7 correct (34,179,458,972 standalone [389] + 3,479,487.31cr consolidated [508]).
  - Step 3: `resolve_citation_heading()` in B8 (nearest heading-like line wins; table/Total/caption/footnote lines excluded). Fixed follow-up: "Footnotes:" lines excluded after Q23 cited a Schedule-9 chunk as "Footnotes:".
  - Step 4: `check_numeric_grounding()` tripwire (numbers must sit under subject-matching row/sentence; skips Calculated derivations; warnings-only in `VerifiedAnswer.grounding_warnings`). Refined after full run: per-token segment attribution, date-fragment masking, headline-magnitude gate (killed dozens of false positives on years/date fragments).
  - Step 5: Golden v1.0.1 (both copies byte-identical): Q04 pages +272/275/280; Q08 pages → 114–121, 280–281; Q12 facts → gross 2,960,018.17cr, pages → [426], gross-vs-net notes; Q17 facts OR-joined. Q12 and Q17 now genuinely PASS.
  - Step 6: Full re-run `b9_full_rerun_20260928.json`: 17/25 pass. 4 artifact fails with correct answers (Q03/Q14/Q15 unit-expansion representation; Q06 golden lacks 'thousand'-less + consolidated facts → 33% ceiling by construction). 4 real fails (Q04 thin answer; Q05 consolidated-instead-of-standalone despite labels; Q08 missing DBUs/branches; Q23 p378 recall miss with correct answer). Caveats: `target_chunk_ids` stale corpus-wide (metrics run on pages only); runner hardcodes 1,988-object warning (actual 1,984).
- [x] **B8 content-support gate for citations (2026-09-29, highest priority)**
  - Q10 root cause traced deterministically: B6 held E1 (pp495-496 consolidated report, HAS both auditor names) through E6 (p369 Opinion, nameless); B7 named both firms but cited only [E6]. Names were context-grounded, attribution false. B8 passed it (existence-only); tripwire silent (name claims, no numbers).
  - Fix in `agents/citation_verifier.py`: new `check_claim_support()` — every headline number and multi-word named entity in a citing segment must be supported by chunk text (word-fraction rule for entity runs; skips Calculated derivations, inline `A-B=C` results, date fragments, bracket annotations, primary-entity-only runs via B6 scope path, scope-consistent subsidiary names). Failures route the token to `unverified_citations` (`[Unverified: EX]`) with explicit warnings. Plus abbreviation-safe sentence splitting (found live: "Co." fractured claims).
  - Corpus rebuild record confirmed (not inferred): embeddings manifest 2026-09-28T07:14:14Z + indexing report 07:14:35Z, 1984 chunks/vectors, model text-embedding-3-large; pre-rebuild backup `.bak20260928` present. No rebuild performed.
  - Q10 re-run: B7 cited E5→pp378-379 chunk verified to contain both firms AND both partners → clean [1], zero warnings (outcome (a), legitimately). Original Q10 answer+chunk replayed offline through new gate: 0 citations, `[Unverified: E6]`, both firms named in warnings.
  - Silent-gap audit over all 50 stored answers: only Q10 is a true failure. Precision flags (not failures): Q15/Q22 derived forms, Q26/Q28/Q31 multi-citation spans where sibling tokens cover the gap.
  - Tests: 490 pass (11 fixtures updated to the stricter contract, 5 new support tests); ruff + mypy strict clean.
- [x] **Q06-chunking-bug closure verification + Set-1 post-fix re-run (2026-09-29)**
  - Task's c01493/c01136 IDs are stale pre-rechunk indices; current same-index chunks are unrelated (p509 Sched-15 / p390 Sched-14) — diagnosing by raw index would hit wrong chunks. True successors verified in current 1984-chunk corpus: c01133 (p389 Sched-12, Total 34,179,458,972 present, caption None), c01489 (p508 Sched-12, Total 3,479,487.31, caption None), c01488 (p508 Sched-11, own Total). No re-chunk/re-embed/re-index performed (all Step-1 fixes already live and verified in code + corpus + Weaviate 1984).
  - p386 Schedule-9 note confirms task's IBPC detail: 29,371,662,693 sits beside "Total Net of Bills Rediscounted" + IBPC 46,680cr — net figure; compatible with existing Q12 notes (p378/p273).
  - Q06 CLOSED: standalone 34,179,458,972 [E1→c01133 p389] + consolidated 3,479,487.31cr [E2→c01489 p508], citations support claims under correct labels; evaluator FAIL is golden-string artifact only.
  - Full Set-1 `data/processed/b9_set1_postfix_20260929.{json,md}`: 13/25 evaluator. Substantive: 17 fully correct + 3 partial (Q08/Q18/Q19) + 5 genuine failures — Q03/Q14 missing scale qualifiers (1000x understatement as written), Q04 thin answer, Q10 FALSE PROVENANCE (auditor names cited to nameless p369 chunk; most dangerous find), Q12 B7 net-over-gross despite p426 evidence present. New positive: Q05 now answers BOTH scopes with legitimate p477 citation (metric-only B5-recall FAIL).
- [x] **CLI productization: two-view terminal product (2026-09-29, presentation layer only)**
- [x] **B4→B8 execution tracing: `/trace` + `/trace full` observability (2026-09-29, no pipeline-semantics change)**
  - New `src/due_diligence/trace.py`: `TraceRecorder` (per-query notes/timings/token usage), `render_compact_trace()` (educational), `render_full_trace()` (forensic: all 40 B4 texts+scores, exact B5 prompt/raw/parsed, B6 XML+map, B7 prompts/raw/parsed, per-citation B8 blocks, latencies, usage). Never records secrets; honest "Not captured by current pipeline" gaps (vector/BM25 split, embedding tokens).
  - Optional `trace=` parameter threaded through QueryRewriter/Retriever/RetrievalPipeline/LLMReranker/Generator/DiligencePipeline (default None = byte-identical behavior); B6/B8 rendered post-hoc from response objects. B6+B8 latencies timed in `DiligencePipeline.run`.
  - CLI: `/trace` (compact), `/trace full` (forensic), `/trace on|off`, `--trace` flag; per-query recorder, no extra model/Weaviate calls.
  - Tests: 20 new in `tests/unit/test_trace.py` (recorder, renderers, capture, non-interference); full suite 534 pass, ruff + mypy strict clean.
  - Live validation: cheeseburger `/trace` (fast path, 40→7, clean refusal) and CET1 `/trace full` (219KB forensic record; surfaced a real diagnostic — Basel CET1 table missed top-7, B7 honestly fell back to Tier 1 17.7% uncertain).
  - Reworked `cli.py` display only; calls real `DiligencePipeline.run` (no parallel RAG, no logic moved, no B7/B8 terminal awareness). Normal view: 🤖 answer + 📚 sources; learner view: `/sources` evidence excerpts, `/trace` B4→B8 stage block (counts, pages, evidence IDs, citations, warnings, total latency).
  - Interactive `/help /sources /trace /clear /exit` (+quit/exit aliases), empty-input no-op, bounded history reused by existing QueryRewriter, friendly errors without tracebacks, Windows-safe emoji output, startup pipeline check.
  - One-shot positional query + `--query` preserved; `--trace` (default off) and `--no-evidence` (display-only) preserved; `--top-n/--top-k/--rerank-candidates` now default to settings instead of silently overriding production values.
  - Entry points: `uv run python -m due_diligence.cli` and `uv run diligence` (`[project.scripts]`). README gained a product-oriented "Running the CLI" section.
  - Tests: 508 unit+integration pass (18 new CLI view/command tests; legacy chrome assertions updated); ruff + mypy strict clean. Live one-shot synthesis query verified end-to-end against real Weaviate + OpenAI.
- [x] **Second evaluation set Q26–Q50 (evaluation design only, 2026-09-29, no app-code changes)**
  - `tests/eval/data/golden_qa_25_additional.json` + identical mirror `data/golden_qa_25_additional.json` (v2.0.0, existing GoldenQuestion schema; no schema/evaluator/pipeline modifications).
  - 25 new questions: operations (branches/ATMs/DBUs/PayZapp/SmartHub), energy + Scope-3 + water + waste tables, 6 subsidiary-isolation items (HDB/Loan-book+PAT, Life VNB, ERGO PAT, HSL MTF, AMC AUM, Capital negative), AGM + PoSH governance, 10-item risk taxonomy + 3-item material-environment lists, 3 reworder-dependent follow-ups (rural branches, renewable share, HDB 'its' PAT), ambiguous digital-share (98% vs 97%/90% definitions), 3 negatives (SmartUp metrics, Capital fund size, NovaPay fiction).
  - Every fact/page verified against the raw PDF (physical pages); absence verified by full-text search for negatives; chunk IDs mapped into the current 1,984-chunk corpus by exact figure-text match.
  - `tests/eval/test_golden_dataset_additional.py`: 8 integrity tests pass. Pre-existing set-1 test failures (version 1.0.0 assertion; stale pre-rechunk chunk IDs) left untouched.
- [x] **Query Rewriter model cost optimization (2026-09-29, config-only)**
  - Query Rewriter default model changed from GPT-4o to GPT-4o-mini via new dedicated `Settings.query_rewriter_model` (default `"gpt-4o-mini"`, env `QUERY_REWRITER_MODEL`); previously it silently shared `generation_model` with B7. B5 (`llm_reranker_model=gpt-4o-mini`) and B7 (`generation_model=gpt-4o`) unchanged. Prompt, history handling, fast path, and parsing untouched.
  - Tests: `test_query_rewriter.py` default + call-model assertions updated to `gpt-4o-mini`; new `test_default_query_rewriter_model` in `test_config.py`. Full suite: 524 pass + 2 pre-existing golden-dataset failures (stale chunk IDs, version string). Ruff + mypy strict clean on touched files.

---

### Phase 3: Domain Diligence Agents & Reasoning Workflows
- [ ] Implement agent base state, memory, and message schemas in `src/due_diligence/schemas/`.
- [ ] Build **Query Router Agent** to categorize diligence queries (Financial, Legal, Commercial, General).
- [ ] Build **Financial Diligence Agent** (balance sheet, revenue trends, EBITDA adjustments, debt schedules).
- [ ] Build **Legal & Contract Agent** (change of control, IP assignment, exclusivity, liabilities).
- [ ] Build **Commercial & Operational Agent** (customer concentration, churn, supplier dependencies).
- [ ] Build **Red Flag Synthesis Agent** (combines findings, rates severity: Low/Medium/High/Deal-Breaker).
- [ ] Add integration tests in `tests/integration/test_agents.py` testing end-to-end multi-agent flow.
- **Definition of Done:** Multi-agent pipeline accepts a deal room query, routes to appropriate specialists, and produces structured risk findings with reasoning steps.

---

### Phase 4: Grounding Guardrails & Attribution Engine
- [ ] Implement Citation Verifier in `src/due_diligence/evaluation/citation_verifier.py` (verifies claims against retrieved chunk text).
- [ ] Implement Hallucination / Faithfulness Grader (detects unsupported numerical or contractual statements).
- [ ] Implement explicit uncertainty fallback mechanism for missing/ambiguous terms.
- [ ] Add unit tests in `tests/unit/test_guardrails.py` testing detection of ungrounded statements.
- **Definition of Done:** Any synthesized insight without direct chunk evidence is flagged or rejected, ensuring zero unsupported conclusions.

---

### Phase 5: API Layer & Deal Room Session Management
- [ ] Set up FastAPI application with CORS and security middleware in `src/due_diligence/api/app.py`.
- [ ] Implement Deal Room CRUD endpoints (`POST /api/v1/deals`, `GET /api/v1/deals/{id}`).
- [ ] Implement Document Upload & Ingestion triggers (`POST /api/v1/deals/{id}/documents`).
- [ ] Implement Diligence Query & Streaming endpoints (`POST /api/v1/deals/{id}/query` with SSE/WebSockets).
- [ ] Implement audit logging middleware for compliance.
- [ ] Add integration tests in `tests/integration/test_api.py`.
- **Definition of Done:** API can handle document uploads, run queries against isolated deal rooms, and stream structured answers with citations.

---

### Phase 6: Automated Evaluation & Golden Dataset Benchmarking
- [ ] Build synthetic and curated golden due diligence Q&A test set in `tests/eval/data/golden_qa.json`.
- [ ] Implement automated RAG evaluation pipeline (Context Precision, Context Recall, Faithfulness, Answer Relevance).
- [ ] Set up regression benchmark script in `scripts/run_eval.py`.
- **Definition of Done:** Automated evaluation suite executes against golden dataset with minimum target thresholds (Faithfulness > 0.95, Citation Precision > 0.90).

---

### Phase 7: Diligence Memorandum Generation & Export
- [ ] Create structured Diligence Memorandum template (Executive Summary, Financial Analysis, Legal Findings, Red Flag Matrix).
- [ ] Implement markdown to PDF/DOCX exporter in `src/due_diligence/agents/memo_generator.py`.
- [ ] Add comprehensive end-to-end test in `tests/integration/test_e2e_memo.py`.
- **Definition of Done:** Co-pilot can generate a complete due diligence investment memo for an ingested deal room with full source citations.

---

## Risk Register & Mitigations

| Risk | Impact | Likelihood | Mitigation Strategy |
|---|---|---|---|
| Inaccurate financial extraction from multi-column PDFs | High | High | Use specialized table extractors + markdown conversion; validate table row/column sums. |
| Hallucinated legal terms or dollar amounts | Critical | Medium | Strict citation verifier step; explicit fallback when confidence < threshold. |
| Cross-deal data contamination in shared vector index | Critical | Low | Hard metadata filtering by `deal_id` enforced in vector DB client wrapper. |
| Excessive LLM latency in multi-agent orchestration | Medium | Medium | Parallelize independent specialist agent queries; stream intermediate findings via SSE. |
