# Performance Optimization, Benchmarks & Resource Profiling

---

## 1. What is this?

**Performance Optimization, Benchmarks & Resource Profiling** documents the empirical latency, throughput, memory, and computational efficiency measurements observed across the AI Due Diligence Co-pilot pipeline.

In corporate diligence, performance must be balanced against extreme accuracy:
- Ingestion must process 600+ page filings in minutes rather than hours.
- Retrieval must surface relevant disclosures in sub-second timeframes ($N=40$ candidates).
- Reranking must maximize precision without creating a multi-minute user latency bottleneck (solved by moving from CPU cross-encoders to fast LLM reranking).
- Embedding and vector storage must minimize operational cloud API costs while maintaining exact dimension contracts.

---

## 2. Where does it fit?

```
[Phase 1 Ingestion (Track A)]
  PDF Parsing & Table Preflight (72.3 pages/sec)
  Layout Preflight Reconciliation (30.5 pages/sec)
  Layout-Aware Chunking (11,008 elements -> 1,984 chunks in <3s)
         │
         ▼
[Phase 2 Storage & Indexing (Track B)]
  OpenAI Batch Embeddings (27.9 chunks/sec, 71.24s total for 1,984 chunks)
  Weaviate Cloud Batch Indexing (224.2 objects/sec, 8.87s total)
         │
         ▼
[Online Query Pipeline (Production Today)]
  Query Rewriter Fast-Path (0.0 ms standalone / ~800 ms LLM)
  Weaviate Hybrid Retrieval B4 (~900–1,100 ms warm, N=40, alpha=0.50)
  Entity Scope Filtering (< 2 ms in-memory)
  LLM Reranker B5 (~1,300–1,500 ms gpt-4o-mini scoring 40 candidates)
  Evidence Context Builder B6 (< 5 ms deduplication & XML serialization)
  Grounded LLM Generation B7 (~1,200–2,500 ms GPT-4o synthesis)
  Citation Verifier B8 (~10–20 ms content support & footnote resolution)
  Total End-to-End Latency: ~3.5–5.5 seconds per interactive query
```

---

## 3. Master Performance & Benchmark Registry

Every recorded number distinguishes its origin:
- **Measured Value:** Directly timed and recorded in an empirical validation log.
- **Configured Value:** Explicit setting in configuration.
- **Design Target:** Architectural specification goal.

| Pipeline Component | Metric Description | Value | Workload / Context | Environment | Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Document Loader** | File Read & Hash Latency | < 15 ms | 12.08 MB PDF (SHA-256) | Windows 11, NVMe SSD | Measured |
| **Table Preflight** | Scan Duration | 9.37 seconds | Full 678 pages | 4 CPU threads | Measured |
| **Table Preflight** | Processing Throughput | 72.3 pages/sec | 486 tables detected | 4 CPU threads | Measured |
| **Layout Preflight** | Scan Duration | 22.24 seconds | Full 678 pages | 4 CPU threads | Measured |
| **Layout Preflight** | Processing Throughput | 30.5 pages/sec | 11,008 elements sorted | 4 CPU threads | Measured |
| **Chunking Engine** | Execution Duration | < 3.0 seconds | 11,008 elements $\to$ 1,984 chunks | Python 3.12 | Measured |
| **Chunking Engine** | Output Chunk Count | **1,984 chunks** | Rebuilt baseline | Python 3.12 | Measured |
| **Chunk Token Limits**| Chunks > 1,000 Tokens | 0 chunks (0.0%) | 100% compliance | Tokenizer: cl100k_base | Measured |
| **Chunk Median Size**| Median Chunk Size | 187.5 tokens | 1,984 canonical chunks | Tokenizer: cl100k_base | Measured |
| **Embeddings API** | Batch Size | 64 chunks / call | OpenAI `text-embedding-3-large`| Cloud API | Configured |
| **Embeddings API** | Generation Throughput | 27.9 chunks/sec | OpenAI `text-embedding-3-large`| Cloud API (HTTPS) | Measured |
| **Embeddings Vector**| Dimensionality | 1,536 dimensions | Matryoshka reduced from 3072 | Memory: 12.2 MB array | Configured |
| **Weaviate Indexing**| Total Objects Indexed | **1,984 objects** | Rebuilt baseline | Weaviate Cloud | Measured |
| **Weaviate Indexing**| Ingestion Rate | 224.2 objects/sec | BYOV (Zero cloud API embedding)| Weaviate Cloud | Measured |
| **B4 Retrieval** | Hybrid Alpha ($\alpha$) | **0.50** | 50% Vector, 50% BM25 | Weaviate Cloud | Configured |
| **B4 Retrieval** | Candidate Pool Size ($N$)| **40 candidates** | Wide candidate pool | Weaviate Cloud | Configured |
| **B4 Retrieval** | Retrieval Latency | ~900–1,100 ms | Hybrid query against 1,984 vectors| Weaviate Cloud | Measured |
| **B5 Reranker (Current)**| Model | `gpt-4o-mini` | LLMReranker (temperature=0.0) | Cloud API | Production |
| **B5 Reranker (Current)**| Candidate Count / Top-K | 40 in $\to$ 7 out | Configured candidate & evidence limits | Cloud API | Configured |
| **B5 Reranker (Current)**| Inference Latency | **~1,300–1,500 ms** | 40 candidates evaluated in parallel | Cloud API | Measured |
| **B6 Context Builder** | Max Context Budget | 24,000 characters | Upper ceiling for generation | In-memory XML format | Configured |
| **B6 Context Builder** | Assembly Latency | < 5 ms | Deduplication + XML format | Python 3.12 | Measured |
| **B7 Generation** | Model & Synthesis Latency | `gpt-4o` (~1.5–2.5s) | JSON mode grounded answer | Cloud API | Measured |
| **B8 Citation Verifier**| Verification Latency | < 20 ms | Per-claim evidence support & footnote resolution | Python 3.12 | Measured |

---

## 4. Historical / Deprecated Benchmark: Cross-Encoder Latency

> [!NOTE]
> **HISTORICAL ARCHIVE:**  
> The section below documents the benchmark results of the legacy cross-encoder architecture tested during Phase B5 before its deprecation on **2026-09-29**.

```
CPU INFERENCE LATENCY COMPARISON (25 Candidates on 4 CPU Threads):

bge-reranker-large (Deprecated):
  ████████████████████████████████████████ 50.63 seconds / query

ettin-reranker-17m-v1 (Deprecated Experiment):
  █ 1.01 seconds / query

LLMReranker (gpt-4o-mini) (CURRENT PRODUCTION):
  █ 1.35 seconds / query (40 candidates, 128k context, zero table truncation)
```

### Why the LLM Reranker Won:
1. **Eliminated the 50.6-Second CPU Bottleneck:** Running BGE-large on CPU made interactive queries unusable (~50.6s per query).
2. **Eliminated 512-Token Truncation:** 28% of financial table chunks exceeded the 512-token ceiling of cross-encoders, causing the bottom rows (and Total rows) to be cut off before scoring.
3. **Native Table Reasoning:** LLMs understand two-dimensional Markdown grids natively, avoiding the prose bias of models pre-trained on MS-MARCO.

---

## 5. Things to Remember

1. **Production Pipeline Latency is ~3.5–5.5s:** Fast hybrid retrieval (<1s) + LLM reranking (~1.4s) + GPT-4o synthesis (~2.0s).
2. **Rebuilt Corpus is 1,984 Chunks:** 1,984 chunks vectorized at 1,536 dimensions.
3. **Retrieval Budget is $N=40, K=7$:** 40 candidates retrieved; 7 top evidence passages passed to generation.
4. **Zero Cloud Embedding Cost at Indexing:** Precomputed `.npy` arrays allow fast, reproducible Weaviate indexing in 8.87 seconds.
