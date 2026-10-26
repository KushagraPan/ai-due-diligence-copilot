# Native Weaviate Hybrid Retrieval Engine (Phase B4) & Entity Scope Filtering

---

## 1. What is this?

### 1.1 The Non-Technical Intuition (The Dual-Detective Search)
Imagine searching a 678-page banking annual report for the answer to:
> *"What were the total gross advances in FY2025-26?"*

If you employ only a **Semantic Detective** (Dense Vector Search):
- The Semantic Detective understands concepts and meanings (*"advances"*, *"loans"*, *"credit facilities"*).
- But the Semantic Detective is blind to exact alphanumeric codes: it cannot easily distinguish between `"FY2024-25"` and `"FY2025-26"`, or between `"Schedule 9"` and `"Schedule 11"`. To an embedding model, these passages look almost identical in 1,536-dimensional space.

If you employ only a **Keyword Detective** (BM25 Lexical Search):
- The Keyword Detective searches for exact words (*"Schedule 9"*, *"FY2025-26"*).
- But if an analyst asks about *"provisions for bad debts"*, and the document authors titled the table *"Impairment allowance on financial instruments under Stage 3"*, the Keyword Detective misses it completely because the exact word *"bad debts"* never appears!

**Phase B4 Hybrid Retrieval** pairs both detectives together inside Weaviate Cloud. It runs dense vector search and BM25 lexical search simultaneously, merging their outputs via **Relative Score Fusion** with an exact balance ($\alpha = 0.50$) to produce a broad candidate pool of **40 high-recall candidates**.

---

## 2. Where does it fit in the RAG Pipeline?

Hybrid Retrieval executes immediately after Phase B3 Query Rewriting and feeds into Entity Scope Filtering before Phase B5 LLM Reranking:

```
[Inbound User Query / Bounded Conversational Context]
                         │
                         ▼
           Phase B3: Query Rewriter (gpt-4o-mini)
                         │
                         ▼
     [Disambiguated Standalone Retrieval Query]
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│            PHASE B4: HYBRID RETRIEVAL (retriever.py)                   │
│ 1. Dense Query Embedding (OpenAI text-embedding-3-large, 1,536 dims)   │
│ 2. Weaviate Cloud Native Hybrid Search:                                │
│    - alpha = 0.50 (50% Dense Vector, 50% BM25 Lexical)                 │
│    - limit = 40 (Top-40 Candidate Pool)                                │
│    - Hard Metadata Pre-Filters (company == "HDFC Bank")                │
│ 3. Score Extraction & Table Metadata Reconstruction                   │
│ 4. Deterministic Ordering: (-score, chunk_id)                          │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
                             ▼
         [Raw Top-40 Candidate Pool (RetrievalResult[])]
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│          ENTITY SCOPE FILTERING (core/entity_scope.py)                 │
│ - Inspects entity_scope metadata on candidate chunks                   │
│ - Detects if query targets Parent Bank or a Subsidiary                 │
│ - filter_subsidiary_candidates() suppresses subsidiary contamination    │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
                             ▼
      [Cleaned 40 Candidates Forwarded to Phase B5 LLMReranker]
```

---

## 3. What did WE implement?

We implemented the retrieval layer across two core modules:

- [`src/due_diligence/retrieval/retriever.py`](../../src/due_diligence/retrieval/retriever.py):
  - [`WeaviateHybridRetriever`](../../src/due_diligence/retrieval/retriever.py#L40-L240): Connects to Weaviate Cloud, embeds queries via `OpenAIEmbeddingProvider`, constructs metadata filters, invokes `collection.query.hybrid()`, unpacks table metadata, and enforces deterministic sorting.
  - Pre-validation guard: Fast-paths whitespace/empty queries to `[]` without wasting OpenAI or Weaviate API calls.
- [`src/due_diligence/core/entity_scope.py`](../../src/due_diligence/core/entity_scope.py):
  - Taxonomy of canonical entity scopes:
    - Parent Bank: `hdfc_bank_standalone`, `hdfc_bank_consolidated`, `hdfc_bank_general`.
    - Operating Subsidiaries: `hdb_financial_services`, `hdfc_life`, `hdfc_ergo`, `hdfc_securities`, `hdfc_amc`, `hdfc_capital`.
  - [`filter_subsidiary_candidates(query, candidates)`](../../src/due_diligence/core/entity_scope.py#L187-L245): Algorithmic filter that strips subsidiary chunks when queries target the parent bank.
- [`src/due_diligence/retrieval/schemas.py`](../../src/due_diligence/retrieval/schemas.py):
  - [`RetrievalResult`](../../src/due_diligence/retrieval/schemas.py#L180-L245): Immutable candidate container preserving `chunk_id`, `text`, `score`, `entity_scope`, `page_numbers`, `heading_path`, `source_element_ids`, and `table_metadata`.

---

## 4. The Production Retrieval Configuration

Configured in [`src/due_diligence/core/config.py`](../../src/due_diligence/core/config.py):

| Parameter | Configuration Key | Current Production Value | Purpose |
| :--- | :--- | :--- | :--- |
| **Hybrid Balance ($\alpha$)** | `hybrid_alpha` | **`0.50`** | Equal 50/50 balance between vector & lexical search |
| **Candidate Budget ($N$)** | `retrieval_top_n` | **`40`** | Expansive candidate pool maximizing multi-page recall |
| **Embedding Model** | `embedding_model` | **`text-embedding-3-large`** | 1,536-dimensional Matryoshka embeddings |
| **Weaviate Index** | `weaviate_index_type` | **`hfresh`** (fallback `hnsw`) | Low-latency vector indexing |
| **BM25 Parameters** | Collection schema | **$b=0.75, k1=1.2$** | Standard Robertson-Spärck Jones BM25 tuning |

---

## 5. Relative Score Fusion Explained ($\alpha = 0.50$)

Weaviate blends dense vector search and BM25 using **Relative Score Fusion**:
1. Vector cosine similarities and BM25 lexical scores live on fundamentally different scales:
   - Cosine similarities range from $-1.0$ to $+1.0$.
   - BM25 scores are unbounded positive numbers ($0.0$ to $50.0+$).
2. Relative Score Fusion normalizes both candidate sets onto a standard $[0.0, 1.0]$ relative scale.
3. The final hybrid score is computed as:
   $$\text{Score}_{\text{hybrid}} = \alpha \cdot \text{Score}_{\text{vector}} + (1 - \alpha) \cdot \text{Score}_{\text{BM25}}$$
4. With **$\alpha = 0.50$**, dense semantic understanding and exact keyword presence contribute equally:
   - If a chunk contains the exact keyword *"Schedule 12"*, its BM25 score boosts it into the top pool.
   - If a chunk expresses the semantic meaning of *"liquidity buffer"* without using that exact string, its vector score boosts it into the top pool.

> [!CAUTION]
> **Score Interpretation Rule:**  
> The Weaviate hybrid score returned in `RetrievalResult.score` is a **relative retrieval ranking signal**, NOT a calibrated probability or confidence score. The system must never tell a user: *"The model is 92% confident in this answer"* based on a retrieval score of 0.92.

---

## 6. Entity Scope Filtering: Preventing Cross-Entity Leakage

During benchmark evaluation, question **Q04** (*"What was the bank's net profit?"*) exposed an insidious RAG failure mode:

### 6.1 The Failure Mode: Subsidiary Profit Ingestion
In a massive banking group, subsidiaries publish their own mini-financials inside the annual report. On Page 304 of the HDFC Bank report, the directors discuss subsidiary **HDB Financial Services Limited (HDBFSL)**:
- HDBFSL Net Profit: ₹2,544 Crore.
- HDFC Bank Parent Net Profit: ₹74,671 Crore.

Because both sections contain the words *"Net Profit"* and *"HDFC"*, hybrid search retrieved the HDBFSL disclosure on Page 304. Downstream generation read the chunk and reported ₹2,544 Crore as HDFC Bank's net profit—an error of over ₹72,000 Crore!

### 6.2 The Solution: Algorithmic Scope Isolation
In [`src/due_diligence/core/entity_scope.py`](../../src/due_diligence/core/entity_scope.py), every chunk carries an `entity_scope` tag assigned during ingestion:
- Pages 369–493 $\to$ `hdfc_bank_standalone`
- Pages 494–580 $\to$ `hdfc_bank_consolidated`
- Pages 304–308 $\to$ `hdb_financial_services`, `hdfc_life`, etc.

When Phase B4 returns the 40 candidates, `filter_subsidiary_candidates()` evaluates the query:
1. If the query asks about the parent bank (*"HDFC Bank"*, *"the bank"*, *"net profit"*), any candidate tagged with a `SUBSIDIARY_SCOPES` tag is **filtered out of the candidate pool**.
2. If the user explicitly asks: *"What was HDB Financial Services' loan book?"*, `query_targets_subsidiary()` detects the subsidiary keyword and retains the subsidiary chunks.

This guarantees that parent bank questions are never contaminated by subsidiary financial disclosures.

---

## 7. Safe Hard Filters vs. Dangerous Over-Filters

Metadata filters are evaluated inside Weaviate before candidates are returned:

```
                            Metadata Filter Decision
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
    SAFE HARD FILTERS                                   DANGEROUS OVER-FILTERS
  (Apply at Weaviate Level)                           (Do NOT Apply as Hard Gates)
  
  • company == "HDFC Bank"                            • financial_period == "FY2025-26"
    (Prevents cross-company pollution)                  (Drops comparative prior-year baselines)
    
  • document_name == "hdfc_report.pdf"                • section_heading == "Risk Factors"
    (Isolates specific target document)                 (Misses critical risks in Notes to Accounts)
```

1. **`company` (Safe Hard Filter):** When an analyst queries a specific deal room, filtering on `company == "HDFC Bank"` prevents multi-tenant data leakage.
2. **`financial_period` (Dangerous Over-Filter):** Financial tables always compare current year (FY26) against prior year (FY25). Hard-filtering on `financial_period == "FY2025-26"` excludes prior-year baseline chunks, destroying growth rate calculations.
3. **`section_heading` (Dangerous Over-Filter):** Hard-filtering on `section_heading == "Risk Factors"` misses critical risks disclosed in *"Note 14: Asset Quality"*.

---

## 8. Concepts I Learned

### Python Concepts
- **Weaviate v4 Filter Expressions:** Composing type-safe filters using `Filter.by_property()` and `Filter.all_of([...])`.
- **Fast-Path Input Pre-Validation:** Checking for empty/whitespace queries before allocating external vectorizer resources.
- **Deterministic Multi-Key Sorting:** Using `(-result.score, result.chunk_id)` to ensure identical rankings across repeated test runs.

### RAG Concepts
- **Relative Score Fusion:** Normalizing dense vectors and BM25 scores before linear combination to eliminate score scale discrepancies.
- **Candidate Pool Expansion ($N=40$):** Using a large candidate pool ($N=40$) to maximize multi-page recall, delegating precision filtering to the downstream LLM reranker.
- **Entity Scope Protection:** Isolating corporate subsidiaries at retrieval time to prevent catastrophic cross-entity number confusion.

---

## 9. Things to Remember

1. **$\alpha = 0.50$ Balanced Fusion:** 50% dense vector semantic search, 50% BM25 keyword matching.
2. **Budget is $N = 40$ Candidates:** Retrieves 40 candidates for Phase B5 scoring.
3. **Hybrid Scores are Relative Signals:** Never call retrieval scores "probabilities" or "confidence".
4. **Entity Scope Filters Prevent Leakage:** Parent bank queries filter out subsidiary candidates.
5. **Deterministic Ordering:** Results are deterministically sorted by descending score and ascending chunk ID.
