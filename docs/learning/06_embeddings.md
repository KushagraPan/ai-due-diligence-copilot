# Dense Embedding Subsystem & Provider Contracts

---

## 1. What is this?

The **Dense Embedding Subsystem** transforms text chunks and natural language search queries into high-dimensional numerical vectors (dense embeddings) that capture semantic meaning.

In corporate due diligence, vector embeddings bridge the vocabulary gap between analyst questions and source filings:
- When an analyst asks about *"credit loss provisioning"*, an embedding model maps the query close to document chunks discussing *"allowance for impairment of stage 3 financial assets"*, even though the phrasing shares few exact words.

Our embedding subsystem is built on three core engineering principles:
1. **Bring Your Own Vectors (BYOV):** We generate embeddings in Python via an abstract provider interface rather than allowing the cloud vector database to vectorize text automatically. This guarantees mathematical parity between document vectors and query vectors.
2. **Strict Dimension Contracts:** Model dimensions are enforced at the factory layer; mismatches raise explicit configuration errors rather than corrupting vector index dimensions.
3. **API Resilience:** External embedding calls feature automated batching (batch size 64), response index re-sorting, and exponential backoff retry with full jitter to handle API rate limits gracefully.

---

## 2. Where does it fit?

The embedding engine operates both at offline indexing time and online query time:

```
[At Ingestion / Indexing Time]
Rebuilt Canonical Chunks (1,984 Chunks)
              │
              ▼
   BaseEmbeddingProvider.embed_documents(texts)
              │ (OpenAI text-embedding-3-large, dims=1536)
              ▼
   Numpy Vector Array: hdfc_embeddings.npy (1984, 1536)
   + Manifest: hdfc_embeddings_manifest.json
              │
              ▼
   weaviate_store.py (Indexed into Weaviate Cloud)

[At Online Query Time]
User Search Query: "What was HDFC Bank's CET1 ratio?"
              │
              ▼
   BaseEmbeddingProvider.embed_query(query)
              │
              ▼
   Query Vector (1,536-dimensional float list)
              │
              ▼
   Phase B4 Hybrid Retrieval (near_vector search)
```

---

## 3. What did WE implement?

- [`src/due_diligence/retrieval/embeddings.py`](../../src/due_diligence/retrieval/embeddings.py):
  - [`BaseEmbeddingProvider`](../../src/due_diligence/retrieval/embeddings.py#L56-L95): Abstract base class defining `@property dimension -> int`, `embed_documents(texts: list[str]) -> list[list[float]]`, and `embed_query(query: str) -> list[float]`.
  - [`OpenAIEmbeddingProvider`](../../src/due_diligence/retrieval/embeddings.py#L98-L230): **Active Production Provider** using OpenAI's `text-embedding-3-large`. Employs Matryoshka dimension truncation (1,536 dims), batch slicing (default 64), response index re-sorting, and exponential backoff with full jitter.
  - [`LocalBGEEmbeddingProvider`](../../src/due_diligence/retrieval/embeddings.py#L233-L335): **Optional Local Provider** for `BAAI/bge-large-en-v1.5` (1,024 dims). Injects mandatory asymmetric retrieval query prefix and normalizes vectors to unit L2 norm.
  - [`get_embedding_provider(settings)`](../../src/due_diligence/retrieval/embeddings.py#L340-L390): Authoritative factory validating that configured `embedding_dimension` matches provider capabilities.
  - Typed exceptions: `EmbeddingError`, `EmbeddingConfigurationError`, `EmbeddingRateLimitError`, `EmbeddingServiceUnavailableError`.
- Persistent Precomputed Artifacts (Current Production Baseline):
  - [`data/processed/hdfc_embeddings.npy`](../../data/processed/hdfc_embeddings.npy): Binary serialized NumPy array of shape `(1984, 1536)` (~12.2 MB).
  - [`data/processed/hdfc_embeddings_manifest.json`](../../data/processed/hdfc_embeddings_manifest.json): Audit manifest tracking model name, dimension, L2 norm statistics, generation timestamp (2026-09-28T07:14:14Z), and `chunk_id_to_index` mapping dictionary.

---

## 4. Production vs. Optional Local Provider

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ PRODUCTION PROVIDER (Default)        │ OPTIONAL LOCAL PROVIDER (Fallback)   │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ Provider: OpenAIEmbeddingProvider    │ Provider: LocalBGEEmbeddingProvider  │
│ Model: text-embedding-3-large        │ Model: BAAI/bge-large-en-v1.5        │
│ Dimension: 1,536 (Matryoshka)        │ Dimension: 1,024 (Native)            │
│ Deployment: OpenAI Cloud API         │ Deployment: Local PyTorch CPU / CUDA │
│ Query Asymmetry: None required       │ Query Asymmetry: Mandatory prefix    │
│ Production Corpus: 1,984 vectors     │ Evaluation only / Offline sandbox    │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

> [!IMPORTANT]
> The active production embedding model is **OpenAI `text-embedding-3-large`** at 1,536 dimensions. Local BGE is an optional fallback provider for offline development environments and is NOT the production default.

---

## 5. Important Implementation Details

### Strict Dimension Contract Enforcement
Different embedding models operate on immutable native dimensionalities:
- `BAAI/bge-large-en-v1.5`: Strictly 1,024 dimensions.
- `text-embedding-3-large`: Native 3,072 dimensions, supporting Matryoshka dimension reduction to 1,536.

In `get_embedding_provider()`, the factory validates:
```python
if configured_dimension != provider_native_dimension:
    raise EmbeddingConfigurationError(...)
```
It rejects invalid configurations immediately without silently overriding them or corrupting database index structures.

### Matryoshka Representation Learning (1,536 Dimensions)
We configure OpenAI's `text-embedding-3-large` with `dimensions=1536` rather than 3,072:
- **Index Efficiency:** Reduces Weaviate vector memory footprint and network payload sizes by **50%**.
- **Accuracy Retention:** Matryoshka training ensures that 1,536-dimensional truncated vectors retain >99% of the retrieval accuracy of the full 3,072-dimensional vector.

### Response Index Sorting (Scrambled API Batches)
When sending a list of 64 texts to OpenAI's embeddings API, the API returns a list of objects with an `index` attribute. In rare circumstances under high load, objects can return out of order. Our provider explicitly executes:
```python
response.data.sort(key=lambda item: item.index)
```
This guarantees an immutable invariant: `vectors[i]` corresponds to `texts[i]`.

---

## 6. Corpus Rebuild Baseline Statistics (2026-09-28)

Real measured results from [`data/processed/hdfc_embeddings_manifest.json`](../../data/processed/hdfc_embeddings_manifest.json):

| Metric | Measured Value | Workload / Context | Status |
| :--- | :--- | :--- | :--- |
| **Total Chunks Vectorized** | **1,984 chunks** | Rebuilt canonical HDFC Bank chunks | Measured |
| **Active Provider** | `OpenAIEmbeddingProvider` | Model: `text-embedding-3-large` | Production |
| **Vector Dimensionality** | 1,536 dimensions | Matryoshka reduced from 3,072 | Production |
| **Batch Size** | 64 chunks / request | OpenAI API batching | Production |
| **Manifest Timestamp** | 2026-09-28T07:14:14Z | Rebuilt post-Q06 boundary fixes | Verified |
| **Failed Vectors** | 0 (0.00%) | Zero errors across all 1,984 chunks | Verified |
| **L2 Norm Range** | [0.99999988, 1.00000012] | Verified unit normalization | Verified |
| **Binary Artifact Size** | 12,189,696 bytes (~12.2 MB) | `data/processed/hdfc_embeddings.npy` | Verified |

---

## 7. Things to Remember

1. **Production is OpenAI 1,536 Dims:** `text-embedding-3-large` is the active production model.
2. **Local BGE is Optional:** 1,024 dimensions, requires asymmetric prefix, used only for offline testing.
3. **Always Re-Sort API Batches:** Re-sort response vectors by `item.index` to prevent vector-chunk transposition.
4. **BYOV Guarantees Parity:** Embedding documents and queries with the same Python provider prevents mathematical drift.
5. **Rebuilt Corpus is 1,984 Chunks:** 1,984 canonical chunks and vectors form the current production baseline.
