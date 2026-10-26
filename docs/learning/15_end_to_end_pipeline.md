# End-to-End Pipeline Orchestration & Subsystem Integration

---

## 1. What is this?

**End-to-End Pipeline Orchestration & Subsystem Integration** coordinates the decoupled subsystems of the AI Due Diligence Co-pilot into a seamless, verifiable diligence workflow.

In complex RAG applications, the greatest engineering risk is **architectural spaghetti**: modules directly importing and calling each other, data schemas being modified on-the-fly, configuration secrets scattered across files, and pipeline stages tightly coupled to specific third-party SDKs. When an error occurs, debugging becomes impossible because state transitions are undocumented and untyped.

Our system enforces clean separation of concerns:
- **Track A (Document Ingestion Engine):** Operates offline, transforming raw multi-page corporate PDFs into structured, layout-reconciled, provenance-tagged [`Chunk`](../../src/due_diligence/ingestion/chunks.py) units (1,984 chunks).
- **The System Boundary Contract:** Bridges Track A and Track B via [`IndexingPayload`](../../src/due_diligence/retrieval/schemas.py#L42-L115), attaching document-level metadata without mutating chunk models.
- **Track B (Dual Indexing, Retrieval & Synthesis):** Coordinates embedding generation, Weaviate Cloud storage, query reference resolution, hybrid retrieval ($N=40$), entity scope filtering, LLM reranking ($K=7$), XML context assembly, grounded LLM synthesis, and citation verification.

Online execution is unified behind [`DiligencePipeline`](../../src/due_diligence/pipeline.py), which orchestrates the complete 6-stage flow and returns a strongly typed [`VerifiedAnswer`](../../src/due_diligence/agents/citation_verifier.py#L467-L498).

---

## 2. Where does it fit?

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TRACK A: OFFLINE DOCUMENT INGESTION ENGINE                                             │
│                                                                                        │
│   Raw PDF ──> loaders.py ──> parser.py ──> table_extractor.py ──> layout.py ──> chunker│
│                                                   │                    │               │
│                                            (table_preflight)    (layout_preflight)     │
└───────────────────────────────────────────────────┼────────────────────┼───────────────┘
                                                    │                    │
                                     THE SYSTEM BOUNDARY CONTRACT        │
                                     [IndexingPayload: Chunk + Metadata] │
                                                    │                    │
┌───────────────────────────────────────────────────▼────────────────────▼───────────────┐
│ TRACK B: STORAGE & ONLINE RETRIEVAL PIPELINE (pipeline.py)                             │
│                                                                                        │
│   Indexing: IndexingPayload ──> embeddings.py ──> weaviate_store.py (Weaviate Cloud)   │
│                                                                                        │
│   Online:   User Query ──> query_rewriter.py (gpt-4o-mini) ────────────┐               │
│                                  │                                     │               │
│                                  ▼                                     │               │
│                            retrieval_query                             │               │
│                                  │                                     │               │
│                                  ▼                                     │ (original_q)  │
│                            retriever.py (Weaviate Hybrid Search)       │               │
│                                  │ (Top-40 Candidates, alpha=0.50)     │               │
│                                  ▼                                     │               │
│                            entity_scope.py (Subsidiary Filtering)      │               │
│                                  │                                     │               │
│                                  ▼                                     │               │
│                            reranker.py (LLM Reranker: gpt-4o-mini)     │               │
│                                  │ (Top-7 Evidence Passages)           │               │
│                                  ▼                                     │               │
│                            context_builder.py (EvidenceContextBuilder) │               │
│                                  │ (XML <diligence_context> + Map)     │               │
│                                  ▼                                     ▼               │
│                            generator.py (GroundedGenerator: gpt-4o) ───┘               │
│                                  │ (GroundedAnswer with [EX] tokens)                   │
│                                  ▼                                                     │
│                            citation_verifier.py (Phase B8 Citation Verification)       │
│                                  │                                                     │
│                                  ▼                                                     │
│                            Verified Diligence Finding with Exact Footnotes [1]         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. What did WE implement?

- [`src/due_diligence/pipeline.py`](../../src/due_diligence/pipeline.py):
  - [`DiligencePipeline`](../../src/due_diligence/pipeline.py): Master orchestrator coordinating all 6 online stages:
    1. Phase B3: Query Rewriter (`QueryRewriter`, `gpt-4o-mini`)
    2. Phase B4: Hybrid Retrieval (`WeaviateHybridRetriever`, $N=40, \alpha=0.50$)
    3. Entity Scope Filtering (`filter_subsidiary_candidates`)
    4. Phase B5: LLM Reranking (`LLMReranker`, `gpt-4o-mini`, $K=7$)
    5. Phase B6: Context Assembly (`EvidenceContextBuilder`)
    6. Phase B7: Grounded Generation (`GroundedGenerator`, `gpt-4o`)
    7. Phase B8: Citation Verification (`CitationVerifier`)
- [`src/due_diligence/cli.py`](../../src/due_diligence/cli.py):
  - Production CLI executing interactive diligence sessions, providing multi-turn conversation memory, and rendering verified footnotes, cited excerpts, and retrieval diagnostics.
- [`src/due_diligence/evaluation/`](../../src/due_diligence/evaluation/):
  - [`runner.py`](../../src/due_diligence/evaluation/runner.py) & [`evaluator.py`](../../src/due_diligence/evaluation/evaluator.py): Automated benchmark runner executing the complete pipeline against golden evaluation datasets (Q01–Q25, Q26–Q50).

---

## 4. Subsystem Status & Maturity Matrix

| Pipeline Stage | Active Component | Implementation Status | Verification Status |
| :--- | :--- | :--- | :--- |
| **Document Loading** | `loaders.PDFLoader` | **IMPLEMENTED** | **VALIDATED** (`test_ingestion.py`) |
| **PDF Text Parsing** | `parser.PDFParser` | **IMPLEMENTED** | **VALIDATED** (`test_parser.py`) |
| **Symbol Font Normalization** | `font_mappings.SymbolFontMapper` | **IMPLEMENTED** | **VALIDATED** (`test_preflight.py`) |
| **Table Extraction & Quality Gate**| `table_extractor.py` | **IMPLEMENTED** | **VALIDATED** (678-page audit report) |
| **Spatial Layout Reconciliation** | `layout.py` | **IMPLEMENTED** | **VALIDATED** (11,008 elements, 0 unowned) |
| **Document-Aware Chunking** | `chunker.DocumentChunker` | **IMPLEMENTED** | **VALIDATED** (1,984 canonical chunks) |
| **Embedding Subsystem** | `embeddings.OpenAIEmbeddingProvider` | **IMPLEMENTED** | **VALIDATED** (1,984 precomputed vectors) |
| **Weaviate Indexer** | `weaviate_store.DiligenceWeaviateStore`| **IMPLEMENTED** | **VALIDATED** (Live Weaviate Cloud index) |
| **Query Rewriting Engine (B3)** | `query_rewriter.QueryRewriter` | **IMPLEMENTED** | **VALIDATED** (`gpt-4o-mini`, 4 turns) |
| **Hybrid Retrieval Engine (B4)** | `retriever.WeaviateHybridRetriever` | **IMPLEMENTED** | **VALIDATED** ($N=40, \alpha=0.50$) |
| **Entity Scope Filtering** | `entity_scope.py` | **IMPLEMENTED** | **VALIDATED** (Parent vs. Subsidiary) |
| **LLM Reranker (B5)** | `reranker.LLMReranker` | **IMPLEMENTED** | **VALIDATED** (`gpt-4o-mini`, $K=7$) |
| **Evidence Context Builder (B6)** | `context_builder.ContextBuilder` | **IMPLEMENTED** | **VALIDATED** (Statement labels, XML) |
| **Grounded LLM Generator (B7)** | `generator.GroundedGenerator` | **IMPLEMENTED** | **VALIDATED** (`gpt-4o`, 8 rules) |
| **Citation Verifier (B8)** | `citation_verifier.CitationVerifier` | **IMPLEMENTED** | **VALIDATED** (Claim support & headings) |
| **Golden Benchmark Suite (B9)** | `evaluation.BenchmarkRunner` | **IMPLEMENTED** | **VALIDATED** (50 golden questions) |

---

## 5. Things to Remember

1. **Clean Architectural Decoupling:** Ingestion runs once offline; querying runs repeatedly online.
2. **Current Budgets:** B4 hybrid retrieves 40 candidates; B5 scores 40 and emits Top-7 evidence items.
3. **No Cross-Encoders in Production:** B5 is exclusively powered by `gpt-4o-mini`.
4. **Active Verification:** B8 actively checks claim support, refusing citations to chunks that lack the claimed entities or numbers.
5. **Rebuilt Corpus:** 11,008 elements $\to$ 1,984 chunks $\to$ 1,984 vectors in Weaviate Cloud.
