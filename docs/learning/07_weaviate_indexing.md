# Weaviate Cloud Indexer, Collection Schema & Invariant Audits

---

## 1. What is this?

The **Weaviate Cloud Indexer & Collection Manager** creates, configures, populates, and verifies the primary hybrid vector and lexical database for the AI Due Diligence Co-pilot.

In corporate diligence, the vector database is the search engine of the entire system. If chunk vectors are transposed, if table metadata is flattened incorrectly, or if document replacement produces duplicate records, retrieval performance degrades silently.

Our indexer implements strict operational guarantees:
- **Bring Your Own Vectors (BYOV):** Configured with `Configure.Vectorizer.none()`, ensuring Weaviate Cloud stores and indexes precomputed Python vectors without modifying them.
- **Dual Vector + BM25 Indexing:** Configures cosine vector indexing (`hfresh` with fallback to `hnsw`) alongside BM25 inverted indexing across 21 schema properties (`bm25_b=0.75, bm25_k1=1.2`).
- **Deterministic Idempotency:** Derives Weaviate UUIDs deterministically via UUIDv5 from the canonical `chunk_id`. Re-indexing a document updates records in-place without duplicating objects.
- **The 10-Point Pre-Indexing Invariant Gate:** Before uploading a single byte to the cloud, the indexer verifies 10 mathematical and structural invariants across the local chunks JSONL, NumPy embeddings array, and manifest JSON.

---

## 2. Where does it fit?

The Weaviate Indexer bridges offline data preparation (Track A + Embeddings) with online query retrieval (Track B):

```
Canonical Chunks (.jsonl) + Embeddings (.npy) + Manifest (.json)
                              │
                              ▼
           weaviate_store.py (validate_pre_indexing_invariants)
                              │ (10 Invariant Checks Must Pass)
                              ▼
           DiligenceWeaviateStore.create_collection()
                              │ (Collection: 'DiligenceChunk', BYOV, hfresh + BM25)
                              ▼
           DiligenceWeaviateStore.index_canonical_corpus()
                              │ (Idempotent UUIDv5, Batch Size 100)
                              ▼
           verify_collection_health()
                              │ (Aggregate Count 1,984 + 5 Probe Vector Checks)
                              ▼
             Ready for Online Hybrid Retrieval (retriever.py)
```

---

## 3. What did WE implement?

- [`src/due_diligence/retrieval/weaviate_store.py`](../../src/due_diligence/retrieval/weaviate_store.py):
  - [`DiligenceWeaviateStore`](../../src/due_diligence/retrieval/weaviate_store.py#L350-L600): Comprehensive collection lifecycle manager using the modern Weaviate v4 Python client (`weaviate.connect_to_weaviate_cloud`).
  - [`validate_pre_indexing_invariants()`](../../src/due_diligence/retrieval/weaviate_store.py#L125-L220): The 10-point pre-indexing audit gate.
  - [`build_weaviate_properties()`](../../src/due_diligence/retrieval/weaviate_store.py#L223-L280): Maps [`IndexingPayload`](../../src/due_diligence/retrieval/schemas.py#L42-L115) into 21 flat Weaviate schema properties, unpacking table attributes (`is_table`, `table_headers`, `table_caption`, `entity_scope`).
  - [`verify_collection_health()`](../../src/due_diligence/retrieval/weaviate_store.py#L285-L345): Post-indexing health audit verifying cluster counts and vector dimensionality across 5 representative probe chunks.
  - Typed exceptions: `WeaviateStoreError`, `WeaviateConnectionError`, `WeaviateSchemaError`, `WeaviateAlignmentError`, `WeaviateBatchIndexError`, `WeaviateVerificationError`.
- [`scripts/run_hdfc_weaviate_indexing.py`](../../scripts/run_hdfc_weaviate_indexing.py): Operational deployment script executing live indexing against Weaviate Cloud.
- Persistent Indexing Audit Report:
  - [`data/processed/hdfc_weaviate_indexing_report.json`](../../data/processed/hdfc_weaviate_indexing_report.json): Complete machine-readable audit of the live cluster indexing run.

---

## 4. The 10-Point Pre-Indexing Invariant Gate

Before modifying Weaviate Cloud, `validate_pre_indexing_invariants()` enforces:
1. **Chunks File Existence:** Confirms `.jsonl` exists.
2. **Embeddings File Existence:** Confirms `.npy` exists.
3. **Manifest File Existence:** Confirms `manifest.json` exists.
4. **Length Equality:** `len(chunks) == len(embeddings) == manifest['chunk_count']` (1,984 objects).
5. **Dimensionality Invariant:** `embeddings.ndim == 2` and `embeddings.shape[1] == expected_dimension` (1,536).
6. **No NaN Values:** `not np.isnan(embeddings).any()`.
7. **No Infinite Values:** `not np.isinf(embeddings).any()`.
8. **No Duplicate Chunk IDs:** Unique set length equals total list length.
9. **Manifest Mapping Correspondence:** Every chunk ID exists in manifest's `chunk_id_to_index`.
10. **Strict Index Alignment:** For every chunk $i$, `manifest["chunk_id_to_index"][chunk.chunk_id] == i`. Vector row $i$ is guaranteed to belong to chunk $i$.

---

## 5. Important Implementation Details

### Offline Embedding vs. Online Search
A fundamental architectural principle:
- **Embedding Happens Offline:** Document chunks are vectorized in Python before indexing. Vectors are loaded directly into Weaviate alongside metadata properties.
- **BM25 Happens Dynamically:** The BM25 inverted index is built automatically by Weaviate on text properties. At retrieval time, BM25 keyword matching and dense vector similarity are evaluated concurrently.

### Deterministic UUIDv5 for Idempotency
Weaviate requires a UUID for every object. Using random UUIDs (`uuid4()`) causes re-indexing to duplicate objects. We derive a deterministic UUIDv5 using a project namespace:
```python
WEAVIATE_NAMESPACE = uuid.UUID("e3b0c442-98fc-1c14-9afbf4c8996fb924")
chunk_uuid = uuid.uuid5(WEAVIATE_NAMESPACE, chunk.chunk_id)
```
If indexing is rerun, Weaviate updates objects in-place rather than creating duplicates.

### The 21 Weaviate Schema Properties
Weaviate performs BM25 inverted indexing and exact filtering on properties. The property mapper flattens nested table metadata into 21 properties:
- Core content: `chunk_text`, `chunk_id`, `chunk_type`, `page_numbers`, `document_name`, `company`.
- Provenance: `section_heading`, `heading_path`, `entity_scope`, `source_element_ids`.
- Table metadata: `is_table`, `table_headers`, `table_caption`, `table_group_id`, `table_slice_index`, `table_slice_count`, `table_footnotes`, `confidence_score`, `needs_review`, `extraction_method`.

---

## 6. Current Production Baseline Statistics (2026-09-28 Rebuild)

Real measured results from Weaviate Cloud index verification:

| Metric | Measured Value | Workload / Context | Status |
| :--- | :--- | :--- | :--- |
| **Collection Name** | `DiligenceChunk` | Weaviate Cloud cluster | Production |
| **Total Objects Indexed** | **1,984 objects** | Rebuilt post-Q06 boundary fixes | Verified |
| **Pre-Indexing Invariants** | 10 of 10 passed (100%) | Verification before upload | Verified |
| **Vector Dimension** | 1,536 dimensions | OpenAI `text-embedding-3-large` | Production |
| **Vector Index Type** | `hfresh` (fallback `hnsw`) | Cosine distance metric | Production |
| **BM25 Inverted Index** | $b=0.75, k1=1.2$ | 21 properties indexed | Production |
| **Health Verification** | `"healthy"` | 100% object count & probe checks | Verified |
| **Probe Chunks Verified** | 5 probe chunks | First, last, middle, table, narrative | Verified |

---

## 7. Things to Remember

1. **BYOV Always:** Vectorizer is `None`; Python controls embedding generation.
2. **Deterministic UUIDv5:** Always derive UUIDs from `chunk_id` for idempotent upserts.
3. **Safe Recreate Default:** `recreate=False` prevents accidental index wiping.
4. **10 Invariants Protect the Database:** Never bypass `validate_pre_indexing_invariants()`.
5. **Rebuilt Baseline is 1,984 Objects:** Matches the 1,984 canonical chunks and vectors.
