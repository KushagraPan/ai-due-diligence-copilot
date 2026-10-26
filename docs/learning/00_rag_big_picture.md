# RAG Big Picture: AI Due Diligence Co-pilot Architecture

---

## 1. What is this?

The **AI Due Diligence Co-pilot** is an enterprise-grade, source-grounded Retrieval-Augmented Generation (RAG) system engineered to analyze complex corporate diligence documents: annual reports, SEC filings (10-K, 10-Q), financial statements, regulatory notes, and M&A contracts.

In corporate due diligence, the stakes are asymmetric. A generic AI assistant making a minor hallucination—such as misstating an EBITDA figure, inventing a debt maturity year, or hallucinating an intellectual property assignment clause—can lead to catastrophic financial or legal liability. Traditional RAG systems built on naive text-splitting and standard semantic search routinely fail in this environment because financial documents are dominated by multi-column layouts, borderless numerical tables, dense footnote disclosures, and strict regulatory terminology.

This project rejects course-demo shortcuts and "black-box" framework bloat. Instead, it implements a disciplined, audit-verifiable pipeline operating under the **Single-Document MVP Vertical Slice** priority: proving complete, error-free text and table extraction, document-aware chunking, dense vector and BM25 hybrid indexing, LLM-based reranking, structured context construction, and source-grounded synthesis with exact page-level citations on a real-world 678-page corporate filing (HDFC Bank FY2025-26 Annual Report).

The architecture is divided into two distinct tracks governed by an immutable system boundary contract:
1. **Track A (Document Ingestion Engine):** Offline processing transforming raw PDFs into structured, layout-reconciled, provenance-tagged chunks (1,984 chunks).
2. **Track B (Dual Indexing, Retrieval & Synthesis Engine):** Online and nearline systems handling embedding generation, Weaviate Cloud storage, query rewriting, hybrid retrieval, entity scope filtering, LLM reranking, context assembly, grounded generation, and citation verification.

---

## 2. Where does it fit?

The high-level pipeline spans decoupled subsystems connected across the Track A / Track B boundary:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TRACK A: OFFLINE DOCUMENT INGESTION ENGINE                                             │
│                                                                                        │
│   Raw PDF ──> PDF Parser ──> 3-Tier Table Engine ──> Spatial Layout ──> Chunker        │
│                                                            │              │            │
│                                                     (Preflight Audits)    │            │
└────────────────────────────────────────────────────────────┼──────────────┼────────────┘
                                                             │              │
                                   THE SYSTEM BOUNDARY CONTRACT             │
                                   [IndexingPayload: Chunk + Metadata]      │
                                                             │              │
┌────────────────────────────────────────────────────────────▼──────────────▼────────────┐
│ TRACK B: RETRIEVAL, RERANKING & GROUNDED GENERATION (DiligencePipeline)                │
│                                                                                        │
│   Indexing:   IndexingPayload ──> Dense Embeddings ──> Weaviate Cloud (hfresh + BM25)  │
│                                                                                        │
│   Querying:   User Query ──> Query Rewriter (B3) ─────────────────┐                    │
│                                    │                              │                    │
│                                    ▼                              │ (Original Query)   │
│                              Retrieval Query                      │                    │
│                                    │                              │                    │
│                                    ▼                              │                    │
│                              Hybrid Retrieval (B4)                │                    │
│                                    │ (Top-40, alpha=0.50)         │                    │
│                                    ▼                              │                    │
│                              Entity Scope Filtering               │                    │
│                                    │ (Parent vs Subsidiary)       │                    │
│                                    ▼                              │                    │
│                              LLM Reranker (B5: gpt-4o-mini)       │                    │
│                                    │ (Top-7 Evidence Passages)    │                    │
│                                    ▼                              │                    │
│                              Evidence Context Builder (B6)        │                    │
│                                    │ (XML Context + E1..E7 Map)   │                    │
│                                    ▼                              ▼                    │
│                              Grounded LLM Generation (B7: gpt-4o) ─┘                    │
│                                    │ (GroundedAnswer with [EX] tokens)                 │
│                                    ▼                                                   │
│                              Citation Verifier (B8)                                    │
│                                    │ (Claim support check & heading resolution)        │
│                                    ▼                                                   │
│                              Audit-Verified Diligence Answer with Footnotes [1]        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. What did WE implement?

- [`src/due_diligence/core/config.py`](../../src/due_diligence/core/config.py): Centralized settings managing credentials, retrieval budgets ($N=40, K=7$, $\alpha=0.50$), and model selections (`gpt-4o-mini` query rewriter, `gpt-4o-mini` reranker, `gpt-4o` generation, `text-embedding-3-large`).
- [`src/due_diligence/ingestion/`](../../src/due_diligence/ingestion/):
  - [`loaders.py`](../../src/due_diligence/ingestion/loaders.py): Document loading and file validation (`PDFLoader`).
  - [`parser.py`](../../src/due_diligence/ingestion/parser.py): PyMuPDF page extraction preserving 1-indexed physical pages.
  - [`table_extractor.py`](../../src/due_diligence/ingestion/table_extractor.py): 3-tier extraction engine with trailing Total row preservation (`extend_box_for_trailing_total`).
  - [`layout.py`](../../src/due_diligence/ingestion/layout.py): Spatial partitioning, single-owner conservation invariant, caption validation, and heading propagation.
  - [`chunker.py`](../../src/due_diligence/ingestion/chunker.py) & [`chunks.py`](../../src/due_diligence/ingestion/chunks.py): Document-aware chunker producing 1,984 canonical [`Chunk`](../../src/due_diligence/ingestion/chunks.py#L34-L83) objects.
- [`src/due_diligence/retrieval/`](../../src/due_diligence/retrieval/):
  - [`embeddings.py`](../../src/due_diligence/retrieval/embeddings.py): Production `OpenAIEmbeddingProvider` (`text-embedding-3-large`, 1,536 dims).
  - [`weaviate_store.py`](../../src/due_diligence/retrieval/weaviate_store.py): Cloud collection manager for `DiligenceChunk` (1,984 objects, 21 properties).
  - [`retriever.py`](../../src/due_diligence/retrieval/retriever.py): `WeaviateHybridRetriever` ($N=40, \alpha=0.50$).
  - [`core/entity_scope.py`](../../src/due_diligence/core/entity_scope.py): Entity scope filtering isolating parent bank from subsidiaries.
  - [`reranker.py`](../../src/due_diligence/retrieval/reranker.py): Production `LLMReranker` (`gpt-4o-mini`, 10 principles, strict JSON chunk-ID output).
  - [`query_rewriter.py`](../../src/due_diligence/retrieval/query_rewriter.py): Reference disambiguation across bounded 4-turn history powered by `gpt-4o-mini` with zero-LLM fast path.
  - [`context_builder.py`](../../src/due_diligence/retrieval/context_builder.py): `EvidenceContextBuilder` ($K=7$, statement-scope labels, XML wire formatting).
- [`src/due_diligence/agents/`](../../src/due_diligence/agents/):
  - [`generator.py`](../../src/due_diligence/agents/generator.py): `GroundedGenerator` (`gpt-4o`, 8 grounding rules, JSON output).
  - [`citation_verifier.py`](../../src/due_diligence/agents/citation_verifier.py): `CitationVerifier` with `check_claim_support()`, `check_numeric_grounding()`, and `resolve_citation_heading()`.
- [`src/due_diligence/pipeline.py`](../../src/due_diligence/pipeline.py): End-to-end `DiligencePipeline` orchestrator.
- [`src/due_diligence/evaluation/`](../../src/due_diligence/evaluation/): `BenchmarkRunner` and `DiligenceEvaluator` evaluating 50 golden benchmark questions.

---

## 4. Subsystem Status & Maturity Matrix

| Pipeline Stage | Active Component | Implementation Status | Verification Status |
| :--- | :--- | :--- | :--- |
| **Document Loading** | `loaders.PDFLoader` | **IMPLEMENTED** | **VALIDATED** (`test_ingestion.py`) |
| **PDF Text Parsing** | `parser.PDFParser` | **IMPLEMENTED** | **VALIDATED** (`test_parser.py`) |
| **Table Extraction Engine** | `table_extractor.py` | **IMPLEMENTED** | **VALIDATED** (678-page audit report) |
| **Spatial Layout Reconciliation** | `layout.py` | **IMPLEMENTED** | **VALIDATED** (11,008 elements, 0 unowned) |
| **Document-Aware Chunking** | `chunker.DocumentChunker` | **IMPLEMENTED** | **VALIDATED** (1,984 canonical chunks) |
| **Dense Embeddings** | `embeddings.OpenAIEmbeddingProvider` | **IMPLEMENTED** | **VALIDATED** (1,984 precomputed vectors) |
| **Weaviate Indexer** | `weaviate_store.DiligenceWeaviateStore`| **IMPLEMENTED** | **VALIDATED** (Live Weaviate Cloud index) |
| **Query Rewriting (B3)** | `query_rewriter.QueryRewriter` | **IMPLEMENTED** | **VALIDATED** (`gpt-4o-mini`, 4 turns) |
| **Hybrid Retrieval (B4)** | `retriever.WeaviateHybridRetriever` | **IMPLEMENTED** | **VALIDATED** ($N=40, \alpha=0.50$) |
| **Entity Scope Filtering** | `entity_scope.py` | **IMPLEMENTED** | **VALIDATED** (Subsidiary isolation) |
| **LLM Reranker (B5)** | `reranker.LLMReranker` | **IMPLEMENTED** | **VALIDATED** (`gpt-4o-mini`, $K=7$) |
| **Evidence Context Builder (B6)** | `context_builder.ContextBuilder` | **IMPLEMENTED** | **VALIDATED** (Statement labels, XML) |
| **Grounded LLM Generator (B7)** | `generator.GroundedGenerator` | **IMPLEMENTED** | **VALIDATED** (`gpt-4o`, 8 rules) |
| **Citation Verifier (B8)** | `citation_verifier.CitationVerifier` | **IMPLEMENTED** | **VALIDATED** (Claim support & headings) |
| **End-to-End Pipeline** | `pipeline.DiligencePipeline` | **IMPLEMENTED** | **VALIDATED** (6-stage orchestration) |
| **Golden Benchmark Suite (B9)** | `evaluation.BenchmarkRunner` | **IMPLEMENTED** | **VALIDATED** (50 golden questions) |

---

## 5. Things to Remember

1. **Current Pipeline is 6-Stage:** Query Rewriter $\to$ B4 Hybrid Retrieval $\to$ Entity Scope $\to$ B5 LLM Reranker $\to$ B6 Context Builder $\to$ B7 Grounded Generation $\to$ B8 Citation Verification.
2. **Production Reranker is an LLM:** `gpt-4o-mini` with 10 principles; cross-encoders are deprecated.
3. **Citation Verification Checks Content:** `check_claim_support()` ensures claimed numbers and entities appear in the cited chunk text.
4. **Rebuilt Production Baseline:** 11,008 elements, 1,984 chunks, 1,984 vectors.
