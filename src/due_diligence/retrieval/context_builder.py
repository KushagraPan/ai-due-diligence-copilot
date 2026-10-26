"""
Evidence Context Builder for Phase B6 of the RAG pipeline.

1. Purpose:
   Transforms B5 Top-K reranked candidates into a structured, sanitized XML
   evidence block with deterministic context-local evidence IDs (E1, E2, ...)
   and complete provenance mappings for Phase B7 Grounded Generation and Phase B8 Citations.

2. Role in Overall RAG Pipeline:
   Acts as the Phase B6 boundary between Retrieval/Reranking and Generation.
   Consumes ``PipelineQueryResult`` or ``list[RerankedResult]``, prunes near-duplicates,
   enforces context budgets without corrupting tables, sanitizes document text against
   prompt-injection breakouts, and emits an immutable ``EvidenceContext``.

3. Expected Inputs:
   - ``candidates: list[RerankedResult] | PipelineQueryResult``
   - Optional query strings: ``original_query: str``, ``retrieval_query: str``
   - Configurable budgets: ``max_context_chars``, ``max_evidence_items``
   - Configurable ordering: ``ordering_strategy`` ("rerank_score" or "lost_in_the_middle")

4. Outputs:
   - ``EvidenceContext``: Frozen Pydantic container holding ``formatted_context``,
     ``evidence_items``, ``evidence_map``, and ``omitted_items``.

5. Dependencies:
   - ``pydantic``
   - ``due_diligence.core.config.Settings``, ``get_settings``
   - ``due_diligence.retrieval.schemas`` (EvidenceContext, EvidenceItem, OmittedEvidenceItem,
     PipelineQueryResult, RerankedResult)

6. Non-Responsibilities:
   - Does NOT perform document retrieval or query Weaviate.
   - Does NOT call LLMs or generate answers (Phase B7).
   - Does NOT rerank candidate passages (Phase B5).
   - Does NOT rewrite user queries (QueryRewriter).
   - Does NOT invent or fabricate citations or page numbers.
   - Does NOT mutate source candidates.
"""

import html
import logging
from collections.abc import Sequence
from typing import Literal

from due_diligence.core.config import Settings, get_settings
from due_diligence.core.entity_scope import resolve_entity_scope
from due_diligence.retrieval.schemas import (
    EvidenceContext,
    EvidenceItem,
    OmittedEvidenceItem,
    PipelineQueryResult,
    RerankedResult,
)

logger = logging.getLogger(__name__)

OrderingStrategy = Literal["rerank_score", "lost_in_the_middle"]


def sanitize_evidence_text(text: str) -> str:
    """
    Sanitize raw document evidence text to prevent XML structural breakout.

    Replaces delimiters that could prematurely close or forge context tags,
    while preserving the raw factual content, numbers, and Markdown table syntax.
    """
    return (
        text.replace("<diligence_context>", "&lt;diligence_context&gt;")
        .replace("<diligence_context", "&lt;diligence_context")
        .replace("</diligence_context>", "&lt;/diligence_context&gt;")
        .replace("<evidence>", "&lt;evidence&gt;")
        .replace("<evidence", "&lt;evidence")
        .replace("</evidence>", "&lt;/evidence&gt;")
    )


# Explicit statement-scope labels injected into each evidence block's visible text.
# The generator (B7) cannot reliably infer standalone vs. consolidated provenance
# from native page headings (e.g. "Opinion" vs "18. Earnings per equity share"),
# so B6 derives this signal from the chunk's entity_scope metadata instead.
ENTITY_SCOPE_STATEMENT_LABELS: dict[str, str] = {
    "hdfc_bank_standalone": (
        "[STANDALONE FINANCIAL STATEMENT — HDFC Bank parent bank only; excludes subsidiaries]"
    ),
    "hdfc_bank_consolidated": (
        "[CONSOLIDATED FINANCIAL STATEMENT — HDFC Bank Group including subsidiaries]"
    ),
    "hdfc_bank_general": (
        "[GENERAL HDFC BANK INFORMATION — not a standalone/consolidated statement figure]"
    ),
    "hdb_financial_services": (
        "[SUBSIDIARY — HDB Financial Services; NOT HDFC Bank parent bank figures]"
    ),
    "hdfc_life": (
        "[SUBSIDIARY — HDFC Life; NOT HDFC Bank parent bank figures]"
    ),
    "hdfc_ergo": (
        "[SUBSIDIARY — HDFC ERGO; NOT HDFC Bank parent bank figures]"
    ),
    "hdfc_securities": (
        "[SUBSIDIARY — HDFC Securities; NOT HDFC Bank parent bank figures]"
    ),
    "hdfc_amc": (
        "[SUBSIDIARY — HDFC AMC; NOT HDFC Bank parent bank figures]"
    ),
    "hdfc_capital": (
        "[SUBSIDIARY — HDFC Capital; NOT HDFC Bank parent bank figures]"
    ),
}


def statement_label_for_scope(scope: str | None) -> str | None:
    """
    Return the explicit statement-scope label for an entity_scope value.

    Returns None for unknown or empty scopes so no misleading label is injected.
    """
    if not scope:
        return None
    return ENTITY_SCOPE_STATEMENT_LABELS.get(scope.strip().lower())


def apply_lost_in_the_middle_ordering(items: list[RerankedResult]) -> list[RerankedResult]:
    """
    Reorder items using U-shaped attention distribution (Rank 1 at start, Rank 2 at end).

    Strategy documented in project architecture (_docs/rag_architecture.md Section 9 & 15):
    For 5 items (indices 0, 1, 2, 3, 4 by score):
    Position 1: Rank #1 (index 0)
    Position 2: Rank #3 (index 2)
    Position 3: Rank #5 (index 4)
    Position 4: Rank #4 (index 3)
    Position 5: Rank #2 (index 1)
    """
    if len(items) <= 2:
        return list(items)

    # General U-shaped interleaving for arbitrary length N:
    # Places items alternately at the front and back of the result list
    result: list[RerankedResult | None] = [None] * len(items)
    left = 0
    right = len(items) - 1

    for idx, item in enumerate(items):
        if idx % 2 == 0:
            result[left] = item
            left += 1
        else:
            result[right] = item
            right -= 1

    return [item for item in result if item is not None]


class EvidenceContextBuilder:
    """
    Context assembly engine formatting B5 reranked candidates into structured XML evidence.

    Guarantees:
    - Provenance preservation: chunk_id, pages, headings, table metadata, B4/B5 scores.
    - Deterministic context-local IDs: E1, E2, E3, ... mapping 1-to-1 to evidence entries.
    - Zero document text corruption: uncorrupted tables and narrative blocks.
    - Budget compliance: excludes oversized candidates with explicit reasons.
    - Delimiter security: document text cannot escape evidence boundaries.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        max_context_chars: int = 16000,
        max_evidence_items: int | None = None,
        ordering_strategy: OrderingStrategy = "rerank_score",
        deduplicate_overlap_threshold: float = 0.85,
    ) -> None:
        """
        Initialize the context builder.

        Args:
            settings: Application settings. Defaults to get_settings().
            max_context_chars: Maximum character budget for the generated XML context string.
            max_evidence_items: Maximum number of evidence items to include (defaults to settings.rerank_top_k).
            ordering_strategy: "rerank_score" (default, descending relevance) or
                               "lost_in_the_middle" (U-shaped attention distribution).
            deduplicate_overlap_threshold: Threshold for pruning overlapping source_element_ids.
        """
        self._settings = settings or get_settings()
        self._max_context_chars = max_context_chars
        self._max_evidence_items = (
            max_evidence_items
            if max_evidence_items is not None
            else self._settings.rerank_top_k
        )
        self._ordering_strategy = ordering_strategy
        self._deduplicate_overlap_threshold = deduplicate_overlap_threshold

        logger.info(
            "Initialized EvidenceContextBuilder (max_chars=%d, max_items=%d, ordering='%s')",
            self._max_context_chars,
            self._max_evidence_items,
            self._ordering_strategy,
        )

    @property
    def max_context_chars(self) -> int:
        """Maximum character budget for the formatted context."""
        return self._max_context_chars

    @property
    def max_evidence_items(self) -> int:
        """Maximum evidence passages allowed in context."""
        return self._max_evidence_items

    @property
    def ordering_strategy(self) -> OrderingStrategy:
        """Active passage ordering strategy."""
        return self._ordering_strategy

    def _calculate_element_overlap(
        self, elements_a: Sequence[str], elements_b: Sequence[str]
    ) -> float:
        """Calculate Jaccard-like overlap fraction between two sets of DocumentElement IDs."""
        if not elements_a or not elements_b:
            return 0.0
        set_a = set(elements_a)
        set_b = set(elements_b)
        intersection = set_a.intersection(set_b)
        smaller_len = min(len(set_a), len(set_b))
        if smaller_len == 0:
            return 0.0
        return len(intersection) / smaller_len

    def _format_evidence_block(self, item: EvidenceItem) -> str:
        """Format a single EvidenceItem into an XML block."""
        pages_str = ", ".join(str(p) for p in item.page_numbers) if item.page_numbers else "Unknown"
        section_attr = html.escape(item.section_heading or "General", quote=True)
        doc_attr = html.escape(item.document_name, quote=True)
        type_attr = html.escape(item.chunk_type, quote=True)

        effective_scope = item.entity_scope
        if not effective_scope or effective_scope == "hdfc_bank_general":
            resolved = resolve_entity_scope(
                heading_path=item.heading_path,
                section_heading=item.section_heading,
                text=item.text,
                page_numbers=item.page_numbers,
            )
            if resolved != "hdfc_bank_general":
                effective_scope = resolved

        scope_attr = html.escape(effective_scope or "hdfc_bank_general", quote=True)

        attrs = [
            f'id="{item.evidence_id}"',
            f'page="{pages_str}"',
            f'section="{section_attr}"',
            f'document="{doc_attr}"',
            f'chunk_type="{type_attr}"',
            f'entity_scope="{scope_attr}"',
        ]

        if item.chunk_type == "table":
            attrs.append('is_table="true"')
            if item.table_metadata and item.table_metadata.get("caption"):
                caption = str(item.table_metadata["caption"])
                attrs.append(f'caption="{html.escape(caption, quote=True)}"')

        opening_tag = f"  <evidence {' '.join(attrs)}>"
        sanitized_content = sanitize_evidence_text(item.text.strip())
        # Prepend the explicit statement-scope label so the generator can tell
        # standalone apart from consolidated/subsidiary figures without relying
        # on native page headings. The stored EvidenceItem.text stays pristine.
        scope_label = statement_label_for_scope(effective_scope)
        body = f"{scope_label}\n{sanitized_content}" if scope_label else sanitized_content
        closing_tag = "  </evidence>"

        return f"{opening_tag}\n{body}\n{closing_tag}"

    def build_context(
        self,
        candidates: Sequence[RerankedResult] | PipelineQueryResult,
        original_query: str | None = None,
        retrieval_query: str | None = None,
    ) -> EvidenceContext:
        """
        Assemble the structured EvidenceContext from reranked candidates.

        Args:
            candidates: Sequence of B5 RerankedResult objects or a PipelineQueryResult.
            original_query: Optional original user question (extracted from PipelineQueryResult if provided).
            retrieval_query: Optional disambiguated retrieval query (extracted from PipelineQueryResult if provided).

        Returns:
            EvidenceContext ready for consumption by Phase B7 Generation.
        """
        # Unpack PipelineQueryResult if provided
        raw_candidates: list[RerankedResult] = []
        if isinstance(candidates, PipelineQueryResult):
            raw_candidates = list(candidates.candidates)
            orig_q = original_query or candidates.original_query
            ret_q = retrieval_query or candidates.retrieval_query
        else:
            raw_candidates = list(candidates)
            orig_q = original_query or ""
            ret_q = retrieval_query or ""

        # Step 1: Empty candidates fast path
        if not raw_candidates:
            empty_xml = "<diligence_context>\n  <no_evidence>No relevant evidence passages found.</no_evidence>\n</diligence_context>"
            return EvidenceContext(
                original_query=orig_q,
                retrieval_query=ret_q,
                evidence_items=[],
                formatted_context=empty_xml,
                evidence_map={},
                omitted_items=[],
                total_context_chars=len(empty_xml),
            )

        # Step 2: Deduplication and redundancy filtering
        accepted_candidates: list[RerankedResult] = []
        omitted_records: list[OmittedEvidenceItem] = []
        seen_chunk_ids: set[str] = set()
        seen_texts: set[str] = set()

        for cand in raw_candidates:
            # Check exact chunk_id duplicate
            if cand.chunk_id in seen_chunk_ids:
                omitted_records.append(
                    OmittedEvidenceItem(
                        chunk_id=cand.chunk_id,
                        reason="duplicate_chunk_id",
                        rerank_score=cand.rerank_score,
                        chunk_type=cand.chunk_type,
                    )
                )
                continue

            # Check exact text duplicate
            norm_text = cand.text.strip()
            if norm_text in seen_texts:
                omitted_records.append(
                    OmittedEvidenceItem(
                        chunk_id=cand.chunk_id,
                        reason="duplicate_content_text",
                        rerank_score=cand.rerank_score,
                        chunk_type=cand.chunk_type,
                    )
                )
                continue

            # Check element overlap against accepted candidates
            is_redundant = False
            for accepted in accepted_candidates:
                overlap = self._calculate_element_overlap(
                    cand.source_element_ids, accepted.source_element_ids
                )
                if overlap >= self._deduplicate_overlap_threshold:
                    omitted_records.append(
                        OmittedEvidenceItem(
                            chunk_id=cand.chunk_id,
                            reason=f"redundant_element_overlap ({overlap:.2f} with {accepted.chunk_id})",
                            rerank_score=cand.rerank_score,
                            chunk_type=cand.chunk_type,
                        )
                    )
                    is_redundant = True
                    break

            if is_redundant:
                continue

            seen_chunk_ids.add(cand.chunk_id)
            seen_texts.add(norm_text)
            accepted_candidates.append(cand)

        # Step 3: Count limit enforcement
        if len(accepted_candidates) > self._max_evidence_items:
            for extra in accepted_candidates[self._max_evidence_items:]:
                omitted_records.append(
                    OmittedEvidenceItem(
                        chunk_id=extra.chunk_id,
                        reason="max_evidence_items_limit_reached",
                        rerank_score=extra.rerank_score,
                        chunk_type=extra.chunk_type,
                    )
                )
            accepted_candidates = accepted_candidates[:self._max_evidence_items]

        # Step 4: Optional attention ordering strategy (Lost-in-the-Middle)
        if self._ordering_strategy == "lost_in_the_middle":
            ordered_candidates = apply_lost_in_the_middle_ordering(accepted_candidates)
        else:
            ordered_candidates = accepted_candidates

        # Step 5: Assign deterministic evidence IDs and build EvidenceItems
        final_evidence_items: list[EvidenceItem] = []
        evidence_map: dict[str, EvidenceItem] = {}
        formatted_blocks: list[str] = []

        # Derive document and company metadata from the highest-ranked candidate
        primary_company = html.escape(ordered_candidates[0].company, quote=True) if ordered_candidates else ""
        primary_doc = html.escape(ordered_candidates[0].document_name, quote=True) if ordered_candidates else ""

        root_open = f'<diligence_context company="{primary_company}" document="{primary_doc}">'
        root_close = "</diligence_context>"
        base_overhead = len(root_open) + len(root_close) + 2  # newlines

        current_total_chars = base_overhead

        for idx, cand in enumerate(ordered_candidates, start=1):
            evidence_id = f"E{idx}"
            item = EvidenceItem.from_reranked_result(evidence_id=evidence_id, candidate=cand)
            block_xml = self._format_evidence_block(item)
            block_length = len(block_xml) + 1  # includes newline

            # Budget check: If adding this block exceeds max_context_chars, omit it cleanly
            if current_total_chars + block_length > self._max_context_chars:
                remaining_budget = max(0, self._max_context_chars - current_total_chars)
                omitted_records.append(
                    OmittedEvidenceItem(
                        chunk_id=cand.chunk_id,
                        reason=(
                            f"budget_exceeded (block requires {block_length} chars, "
                            f"remaining budget is {remaining_budget})"
                        ),
                        rerank_score=cand.rerank_score,
                        chunk_type=cand.chunk_type,
                    )
                )
                continue

            current_total_chars += block_length
            final_evidence_items.append(item)
            evidence_map[evidence_id] = item
            formatted_blocks.append(block_xml)

        # Assemble final XML document
        if formatted_blocks:
            body_xml = "\n\n".join(formatted_blocks)
            final_xml = f"{root_open}\n{body_xml}\n{root_close}"
        else:
            final_xml = f"{root_open}\n  <no_evidence>No candidate chunks fit within the context budget.</no_evidence>\n{root_close}"

        return EvidenceContext(
            original_query=orig_q,
            retrieval_query=ret_q,
            evidence_items=final_evidence_items,
            formatted_context=final_xml,
            evidence_map=evidence_map,
            omitted_items=omitted_records,
            total_context_chars=len(final_xml),
        )


# Backward compatibility alias
ContextBuilder = EvidenceContextBuilder
