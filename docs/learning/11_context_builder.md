# Evidence Context Builder (Phase B6)

---

## 1. What is this?

The **Evidence Context Builder** transforms Phase B5 Top-$K$ ($K=7$) reranked passages into structured, sanitized, audit-tagged XML context blocks for Phase B7 Grounded Generation.

In naive RAG pipelines, retrieved candidate chunks are simply concatenated into a giant string separated by arbitrary newlines and fed to the LLM. In corporate due diligence, this causes three major failures:
1. **Citation Confusion:** Chunks lack standardized identifiers, forcing the LLM to either invent citations from memory or regurgitate long, messy chunk IDs (`hdfc-bank__annual-report__fy2025-26.pdf_c00519_12b09d8829d8`).
2. **Context Window Bloat & Redundancy:** Multiple retrieved chunks from adjacent sections or table slices often contain 80–90% identical text, wasting token budgets and confusing the model.
3. **"Lost in the Middle" Recall Degradation:** LLM attention fidelity is highest at the very start and very end of the prompt context, while information buried in the middle is frequently ignored.

The Evidence Context Builder solves these challenges deterministically:
- It deduplicates chunks based on source element overlap.
- It enforces strict character and item budgets ($K=7$ active passages).
- It injects explicit **statement-scope labels** to disambiguate Standalone from Consolidated statements.
- It maps complex chunk IDs to short context-local tokens (`E1`, `E2`, ...).
- It maintains an authoritative 1-to-1 lookup map (`evidence_map`).
- It applies prompt-injection sanitization to prevent document content from breaking out of XML tags.

---

## 2. Where does it fit in the RAG Pipeline?

The Context Builder sits between Phase B5 LLM Reranking and Phase B7 Grounded Generation:

```
Phase B5: Top-K RerankedResult Passages (K=7 from LLMReranker)
                   │
                   ▼
     context_builder.py (EvidenceContextBuilder)
  ├── 1. Multi-Tier Deduplication (ID, Text, >85% Element Overlap)
  ├── 2. Strict Budget Enforcement (max_context_chars, max_evidence_items=7)
  ├── 3. XML Delimiter Sanitization (Escapes <evidence>, <diligence_context>)
  ├── 4. Statement-Scope Label Injection ([STANDALONE ...], [CONSOLIDATED ...])
  ├── 5. Token ID Assignment (E1, E2, ..., Ek)
  ├── 6. Evidence Map Construction (E1 -> EvidenceItem Provenance)
  └── 7. Attention Ordering (Descending Rerank Score OR Lost-in-the-Middle)
                   │
                   ▼
     EvidenceContext (Immutable Pydantic Model)
      ├── formatted_context: <diligence_context>...<evidence id="E1">...
      ├── evidence_items: list[EvidenceItem]
      ├── evidence_map: dict[str, EvidenceItem]
      └── omitted_items: list[OmittedEvidenceItem]
                   │
                   ▼
     Phase B7: Grounded LLM Generation (generator.py)
```

---

## 3. What did WE implement?

- [`src/due_diligence/retrieval/context_builder.py`](../../src/due_diligence/retrieval/context_builder.py):
  - [`EvidenceContextBuilder`](../../src/due_diligence/retrieval/context_builder.py#L75-L350): Master builder managing character budgets, item limits ($K=7$), deduplication thresholds, and attention ordering.
  - [`statement_label_for_scope(scope)`](../../src/due_diligence/retrieval/context_builder.py): Injects prominent bracketed statement labels into chunk text to prevent Standalone vs. Consolidated confusion.
  - [`apply_lost_in_the_middle_ordering(items)`](../../src/due_diligence/retrieval/context_builder.py#L355-L390): Re-orders candidates into a U-shaped attention distribution.
  - [`sanitize_evidence_text(text)`](../../src/due_diligence/retrieval/context_builder.py#L50-L72): Escapes prompt-delimiting XML tags to prevent context breakout.
- [`src/due_diligence/retrieval/schemas.py`](../../src/due_diligence/retrieval/schemas.py):
  - [`EvidenceItem`](../../src/due_diligence/retrieval/schemas.py#L318-L380): Strongly typed active evidence snippet containing `evidence_id: str` (`"E1"`, `"E2"`), `chunk_id`, `text`, `score`, `rerank_score`, `page_numbers`, `section_heading`, and `heading_path`.
  - [`OmittedEvidenceItem`](../../src/due_diligence/retrieval/schemas.py#L385-L435): Audit record tracking chunks excluded due to budget ceilings or near-duplicate overlap.
  - [`EvidenceContext`](../../src/due_diligence/retrieval/schemas.py#L438-L485): Frozen container holding `formatted_context: str`, `evidence_items: list[EvidenceItem]`, `evidence_map: dict[str, EvidenceItem]`, and `omitted_items: list[OmittedEvidenceItem]`.

---

## 4. Key Implementation Features

### 4.1 Explicit Statement-Scope Labels (Standalone vs. Consolidated Fix)
In financial reports, Standalone and Consolidated statements use identical table titles (*"Balance Sheet"* or *"Profit and Loss Account"*). In B6, the builder inspects `entity_scope` and prepends an unmistakable header to the evidence text:
```
[STANDALONE FINANCIAL STATEMENT - HDFC BANK PARENT ONLY]
or
[CONSOLIDATED FINANCIAL STATEMENT - GROUP INCLUDING SUBSIDIARIES]
or
[SUBSIDIARY DISCLOSURE - HDB FINANCIAL SERVICES]
```
Rule 6 of the generation prompt instructs the LLM to trust only these labels when reporting financial metrics.

### 4.2 Deterministic Token IDs (`E1`, `E2`, ...) and `evidence_map`
Rather than forcing the LLM to handle 60-character internal chunk IDs, the Context Builder assigns short tokens: `E1`, `E2`, ..., `E7`.
Crucially, it constructs an authoritative Python lookup dictionary:
```python
evidence_map: dict[str, EvidenceItem] = {
    "E1": EvidenceItem(
        chunk_id="hdfc_c01133_a8f9",
        page_numbers=[389],
        section_heading="Schedule 12 - Contingent Liabilities",
        ...
    ),
    "E2": EvidenceItem(...)
}
```
This decouples generation tokens from physical metadata and enables Phase B8 Citation Verification to resolve exact page numbers deterministically.

### 4.3 Multi-Tier Deduplication (>85% Overlap Pruning)
When adjacent table slices or narrative chunks are retrieved, they may share substantial content. The builder computes the Jaccard overlap of `source_element_ids`:
$$\text{Overlap} = \frac{|\text{elem\_ids}_A \cap \text{elem\_ids}_B|}{\min(|\text{elem\_ids}_A|, |\text{elem\_ids}_B|)}$$
If overlap exceeds **85%**, the lower-ranked chunk is omitted and logged to `omitted_items`, preserving token budget for diverse evidence.

### 4.4 XML Delimiter Sanitization
To prevent prompt injection from malicious or adversarial document text, `sanitize_evidence_text()` escapes tags:
- `<evidence` $\to$ `&lt;evidence`
- `</evidence>` $\to$ `&lt;/evidence&gt;`
- `<diligence_context` $\to$ `&lt;diligence_context`
Document text is rendered as passive XML data that cannot break out of container tags.

---

## 5. Things to Remember

1. **Top-$K = 7$ Evidence Passages:** Context Builder passes up to 7 reranked passages to B7.
2. **Statement-Scope Labels Prevent Conflation:** Prominent bracketed labels inform B7 whether metrics are Standalone or Consolidated.
3. **Decoupled Token Indirection:** `E1`..`Ek` tokens decouple LLM generation from physical file coordinates.
4. **Authoritative `evidence_map`:** Drives Phase B8 citation verification and attribution.
5. **Deduplication Saves Budget:** Drops candidate chunks sharing $>85\%$ source elements.
