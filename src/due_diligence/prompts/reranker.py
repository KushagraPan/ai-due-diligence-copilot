"""
Candidate reranking prompt templates for the Phase B5 LLM Reranker.

1. Purpose:
   Defines the system prompt and candidate formatting for ranking B4 hybrid
   retrieval candidates by direct evidence value using a fast LLM. The model
   ranks passages — it never answers the diligence question itself.

2. Role in Overall RAG Pipeline:
   Used by ``LLMReranker`` in Phase B5. Converts the retrieval query, the
   original user question, and up to 40 B4 ``RetrievalResult`` candidates into
   a single JSON ranking completion over candidate chunk IDs.

3. Expected Inputs:
   - ``original_query: str``: The user's original question wording (intent/context).
   - ``retrieval_query: str``: The disambiguated standalone query used for retrieval.
   - ``candidates: Sequence[RetrievalResult]``: B4 Top-N candidates.
   - ``max_chars_per_candidate: int``: Full-text character budget per candidate.

4. Outputs:
   - System prompt string and formatted user prompt string.
   - The LLM must return ``{"ranked_chunk_ids": [...]}`` containing only
     supplied chunk IDs, most relevant first.

5. Dependencies:
   - ``due_diligence.retrieval.schemas.RetrievalResult`` (typing only)

6. Non-Responsibilities:
   - Does not invoke LLMs or external APIs.
   - Does not perform document retrieval or vector search.
   - Does not parse or validate the model response (see ``reranker.py``).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from due_diligence.retrieval.schemas import RetrievalResult

LLM_RERANKER_SYSTEM_PROMPT = """You are a relevance-ranking component in a financial due-diligence RAG system.

You rank evidence passages. You do NOT answer the user's question, explain anything,
invent evidence, generate citations, compute financial values, modify chunk text, create
new evidence, or retrieve additional documents. Your ONLY output is a JSON object listing
the supplied chunk IDs from most to least useful as evidence for answering the query.

RANKING PRINCIPLES — apply in order:

1. DIRECT ANSWERABILITY: Does the chunk actually contain the information needed to
   answer the query? Rank a chunk with the requested fact/value above one that merely
   discusses the topic.

2. EXACT EVIDENCE: Prefer the chunk containing the requested number, ratio, date, name,
   or clause over descriptive commentary. Example: for "total gross advances", a
   standalone Balance Sheet containing total gross advances outranks a sectoral advances
   table that merely repeats the word "advances" many times.

3. PRIMARY STATEMENTS OVER COMMENTARY: For quantitative financial questions, prefer the
   actual primary financial statement or statutory schedule (Balance Sheet, Profit and
   Loss, Cash Flow, Schedules 1-18) holding the requested value over narrative summaries,
   MD&A mentions, or secondary disclosures.

4. ENTITY SCOPE: Distinguish HDFC Bank standalone (parent bank only) from HDFC Bank
   consolidated (group including subsidiaries), HDB Financial Services, HDFC Life,
   HDFC ERGO, HDFC AMC, HDFC Securities, and other entities. A correct-entity chunk with
   the figure outranks a wrong-entity chunk even when numbers look similar. Use each
   candidate's entity scope label; never infer entity from a page heading alone.

5. METRIC SCOPE: Distinguish actual reported values from regulatory minimums/floors;
   standalone from consolidated; bank-level from subsidiary, segment, or unit-level
   figures; headline results from component breakdowns; gross from net where applicable.

6. TABLES ARE FIRST-CLASS EVIDENCE: Never penalize a chunk merely for being a table.
   Financial tables, schedules, balance sheets, P&L statements, and structured
   disclosures are often the strongest evidence. Read all rows and columns, including
   Total rows.

7. MULTI-PAGE EVIDENCE: If several chunks together answer the question (e.g. a table
   plus its Total row, or narrative plus schedule), rank the complementary chunks
   adjacently near the top.

8. CITATION USEFULNESS: Prefer chunks with reliable page/section provenance that directly
   support a citable claim.

9. CONVERSATIONAL QUERIES: The retrieval query is the primary semantic query; the
   original user question preserves intent and reference context (e.g. what "it" or
   "those" refers to). Resolve references from the original question, then judge
   candidates against the retrieval query.

10. NEGATIVE / UNANSWERABLE QUERIES: Do NOT rank a vaguely related chunk highly merely
    for keyword overlap. If no candidate actually supports the requested fact, rank
    conservatively — fewer strong candidates outrank a long list of weak ones.

OUTPUT CONTRACT (strict):
- Respond with a single valid JSON object: {"ranked_chunk_ids": ["<chunk_id>", ...]}
- List supplied chunk IDs ONLY, most relevant first. No invented, duplicate, or
  external IDs. No page numbers instead of IDs. At most 7 IDs; fewer is acceptable
  when fewer candidates genuinely support the query. No other keys, no prose.
"""


def format_llm_reranker_prompt(
    retrieval_query: str,
    candidates: Sequence["RetrievalResult"],
    original_query: str | None = None,
    max_chars_per_candidate: int = 4000,
) -> str:
    """
    Format B4 candidates into a chunk-ID ranking prompt for the LLM reranker.

    Args:
        retrieval_query: Disambiguated standalone query used for B4 retrieval.
        candidates: B4 Top-N candidates to rank.
        original_query: Optional original user wording (intent/reference context).
        max_chars_per_candidate: Full-text budget per candidate; longer texts are
            truncated with an explicit marker (financial tables fit in full).

    Returns:
        Formatted user prompt string ready for LLM submission.
    """
    items: list[str] = []
    for c in candidates:
        text = (c.text or "").strip().replace("\r", " ")
        if len(text) > max_chars_per_candidate:
            text = text[:max_chars_per_candidate] + "\n[TRUNCATED]"
        heading_path = " > ".join(c.heading_path) if c.heading_path else "-"
        pages = ", ".join(str(p) for p in c.page_numbers) if c.page_numbers else "Unknown"
        items.append(
            f"chunk_id: {c.chunk_id}\n"
            f"type: {c.chunk_type} | pages: {pages} | section: {c.section_heading or '-'} | "
            f"entity_scope: {c.entity_scope} | document: {c.document_name}\n"
            f"heading_path: {heading_path}\n"
            f"text:\n{text}"
        )

    header_lines = [f'Original user question: "{(original_query or retrieval_query).strip()}"']
    if original_query and original_query.strip() != retrieval_query.strip():
        header_lines.append(f'Retrieval query (primary semantic query): "{retrieval_query.strip()}"')
    else:
        header_lines.append(f'Retrieval query: "{retrieval_query.strip()}"')
    header_lines.append(
        f"\nRank these {len(candidates)} candidates by direct evidence value for the query. "
        'Return JSON: {"ranked_chunk_ids": ["<chunk_id>", ...]} with at most 7 IDs, most relevant first.'
    )
    return "\n".join(header_lines) + "\n\nCandidates:\n\n" + "\n\n---\n\n".join(items)
