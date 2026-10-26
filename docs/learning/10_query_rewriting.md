# Query Rewriting & Reference Disambiguation (Phase B3)

---

## 1. What is this?

**Query Rewriting & Reference Disambiguation** reformulates conversational, pronoun-heavy, or elliptical user queries into standalone search strings optimized for hybrid retrieval against corporate filings.

In conversational due diligence workflows, users rarely ask self-contained questions in every interaction:
1. Turn 1: *"What was HDFC Bank's net interest income in FY2025-26?"*
2. Turn 2: *"How did it change from the previous year?"*
3. Turn 3: *"What about debt maturity?"*

If Turn 2 (*"How did it change from the previous year?"*) is sent directly to Weaviate, hybrid search fails completely: the query contains no company name, no metric name, and no fiscal year.

The Query Rewriter resolves conversational pronouns (*"it"*, *"they"*, *"the company"*), comparative requests, and ellipses using recent conversation history to produce a fully resolved retrieval query (e.g. *"How did HDFC Bank's net interest income change from FY2024-25 to FY2025-26?"*).

Crucially, query rewriting is **conditional**: standalone queries are left completely untouched, and the system strictly preserves the user's `original_query` separately from `retrieval_query` so that downstream answer generation does not suffer from conversational drift.

---

## 2. Where does it fit in the RAG Pipeline?

The Query Rewriter operates at the very beginning of the Online Retrieval Pipeline:

```
User Query ("How did it change from the previous year?")
+ Conversation History (Turn 1: "What was HDFC Bank's net interest income?")
                         │
                         ▼
        query_rewriter.py (QueryRewriter)
  ├── 1. Empty History Check (Fast-Path to verbatim query)
  ├── 2. Bounded Window Slicing (Max 4 recent turns)
  ├── 3. XML Delimiter Sanitization (Prompt injection defense)
  ├── 4. Zero-Fact-Injection Prompt Completion (gpt-4o-mini)
  └── 5. Deterministic Response Parser & Prefix Stripper
                         │
                         ├─────────────────────────────────────────┐
                         ▼                                         ▼
                 retrieval_query                            original_query
       ("How did HDFC Bank's net interest              ("How did it change from
        income change from the previous year?")          the previous year?")
                         │                                         │
                         ▼                                         │
        Phase B4: Weaviate Hybrid Retrieval                        │
                         │                                         │
                         ▼                                         │
        Entity Scope Filtering (entity_scope.py)                   │
                         │                                         │
                         ▼                                         │
        Phase B5: LLM Reranker (gpt-4o-mini)                       │
                         │                                         │
                         ▼                                         ▼
        Phase B6: Evidence Context Builder ─────────► Phase B7: Generation
```

---

## 3. What did WE implement?

- [`src/due_diligence/retrieval/query_rewriter.py`](../../src/due_diligence/retrieval/query_rewriter.py):
  - [`QueryRewriter`](../../src/due_diligence/retrieval/query_rewriter.py#L215-L360): Coordinates history normalization, prompt formatting, LLM execution, deterministic parsing, and fallback recovery.
  - [`normalize_history()`](../../src/due_diligence/retrieval/query_rewriter.py#L95-L155): Normalizes diverse history formats (tuples, dicts, `ConversationTurn` models) into typed instances.
  - [`parse_rewritten_query()`](../../src/due_diligence/retrieval/query_rewriter.py#L156-L210): Extracts content from `<rewritten_query>` tags, strips markdown code fences, removes common prefixes (*"Rewritten query:"*), and strips quotes with safe fallback.
  - [`rewrite_query()`](../../src/due_diligence/retrieval/query_rewriter.py#L360-L386): Functional convenience wrapper.
  - Typed exceptions: `QueryRewriterError`, `QueryRewriterConfigurationError`, `QueryRewriterInputError`, `QueryRewriterModelError`.
- [`src/due_diligence/prompts/query_rewriter.py`](../../src/due_diligence/prompts/query_rewriter.py):
  - `QUERY_REWRITER_SYSTEM_PROMPT`: Enforces strict anti-hallucination, zero fact injection, untrusted data boundaries, and XML tag enclosure.
  - `_sanitize_xml_content()`: Escapes prompt-delimiting XML tags to prevent prompt injection breakouts.
  - `format_query_rewriter_user_prompt()`: Fences conversation history and current query strictly as passive XML data.

---

## 4. Data Flow & Security Guarantees

```
[Inbound user_query: "How did it change?"] + [History: 3 Turns]
                           │
                           ▼ (QueryRewriter.rewrite)
1. Query Validation: Check user_query is non-empty string.
2. Fast-Path Check: If history is empty, return user_query immediately (Zero LLM calls).
3. Bounded History Window: Retain only the most recent 4 turns (messages, not full dialogues).
4. XML Sanitization: Escape <conversation_history>, <current_query>, <turn>, <rewritten_query>.
5. LLM Chat Completion: Invoke gpt-4o-mini (temperature=0.0, max_tokens=256).
6. Deterministic Parse: Extract string between <rewritten_query> tags; fall back to original query on failure.
                           │
                           ▼
[Output retrieval_query: "How did HDFC Bank's net interest income change from FY24 to FY25?"]
```

### Key Security & Correctness Safeguards:
1. **Zero Fact Injection Rule:** The rewriter is strictly instructed NEVER to add facts or answer the question; it only resolves pronouns and contextual references.
2. **Decoupled Dual Queries:** `retrieval_query` is used for Weaviate search and B5 ranking; `original_query` is passed to B7 Generation to ensure the LLM answers the user's exact phrasing without drift.
3. **XML Tag Sanitization:** Strips external XML delimiters from history inputs to prevent jailbreak or instruction override.

---

## 5. Things to Remember

1. **Standalone Queries Fast-Path:** If conversation history is empty, the rewriter returns the query verbatim without calling any LLM API.
2. **Bounded Window:** Only the last 4 turns are included in the prompt context.
3. **Dual Query Separation:** Downstream components receive both `retrieval_query` and `original_query`.
4. **Zero Answer Generation:** The rewriter formulates a search query; it never provides an answer.
