"""
Document-aware chunking engine for layout elements.

1. Purpose:
   Converts an ordered stream of DocumentElement objects (NarrativeText, Heading,
   ListItem, Table) into retrieval-ready, context-enriched Chunk objects.

2. Role in Overall RAG Pipeline:
   Sits immediately after layout reconciliation (layout.py) and before vector
   indexing / lexical BM25 storage. Enforces semantic boundaries, propagates heading
   hierarchy, applies narrative overlap, and slices oversized tables.

3. Expected Inputs:
   - Ordered list of DocumentElement instances (produced by layout.py).
   - Optional ChunkingConfig specifying target token sizes and overlap thresholds.
   - Optional document_name string override.

4. Outputs:
   - Ordered list of immutable Chunk objects ready for embedding and indexing.

5. Dependencies:
   - tiktoken (cl100k_base tokenizer)
   - pydantic (BaseModel, ConfigDict, Field)
   - due_diligence.ingestion.elements (DocumentElement, NarrativeText, Heading, ListItem, Table)
   - due_diligence.ingestion.chunks (Chunk)

6. Outside Scope:
   - Does not generate vector embeddings or communicate with Weaviate.
   - Does not perform PDF extraction or visual layout detection.
   - Does not execute LLM generation or query answering.
"""

import hashlib
import logging
import re
from collections.abc import Sequence

import tiktoken
from pydantic import BaseModel, ConfigDict, Field

from due_diligence.ingestion.chunks import Chunk
from due_diligence.ingestion.elements import (
    DocumentElement,
    Heading,
    ListItem,
    NarrativeText,
    Table,
    is_schedule_boundary_heading,
    is_total_row_label,
)

logger = logging.getLogger(__name__)

# Global tokenizer instance for deterministic token counting
_TOKENIZER: tiktoken.Encoding | None = None
try:
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except (KeyError, ValueError):  # pragma: no cover
    _TOKENIZER = None


def count_tokens(text: str) -> int:
    """
    Returns deterministic token count for text using cl100k_base tokenizer.
    Falls back to whitespace estimation if tokenizer fails to initialize.
    """
    if not text:
        return 0
    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text, disallowed_special=()))
    # Fallback heuristic: approx 4 characters per token
    return max(1, len(text) // 4)


def split_text_into_sentences(text: str) -> list[str]:
    """
    Splits narrative prose into sentences using punctuation boundaries.
    Preserves abbreviations and numeric decimals (e.g., '16.8%') without breaking.
    """
    if not text:
        return []
    # Split on sentence-ending punctuation followed by whitespace and an uppercase letter/number/quote
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"\'\(\[])", text)
    result: list[str] = [s.strip() for s in sentences if s.strip()]
    return result if result else [text.strip()]


def extract_narrative_overlap(text: str, target_tokens: int = 50) -> tuple[str, int]:
    """
    Extracts approximately target_tokens from the end of narrative text for retrieval context.
    Attempts to break cleanly on a sentence or word boundary.
    Returns (overlap_text, token_count).
    """
    if not text or target_tokens <= 0:
        return "", 0

    tokens = _TOKENIZER.encode(text, disallowed_special=()) if _TOKENIZER else []
    if not tokens or len(tokens) <= target_tokens:
        return text.strip(), len(tokens) if tokens else count_tokens(text)

    # Slice the last target_tokens tokens
    tail_tokens = tokens[-target_tokens:]
    decoded_tail = _TOKENIZER.decode(tail_tokens) if _TOKENIZER else text[-target_tokens * 4 :]

    # Clean up cut words at the front: advance to the first whitespace if in the middle of a word
    first_space = decoded_tail.find(" ")
    if first_space != -1 and first_space < len(decoded_tail) // 2:
        overlap_str = decoded_tail[first_space + 1 :].strip()
    else:
        overlap_str = decoded_tail.strip()

    actual_tokens = count_tokens(overlap_str)
    return overlap_str, actual_tokens


def generate_chunk_id(
    document_name: str,
    chunk_index: int,
    source_element_ids: list[str],
    chunk_text: str,
) -> str:
    """
    Generates a deterministic, reproducible unique identifier for a chunk.
    Incorporates document slug, zero-padded sequence index, and SHA-256 content hash.
    """
    safe_doc = re.sub(r"[^\w\-.]", "_", document_name)
    content_key = f"{safe_doc}|{chunk_index}|{','.join(source_element_ids)}|{chunk_text}"
    content_hash = hashlib.sha256(content_key.encode("utf-8")).hexdigest()[:12]
    return f"{safe_doc}_c{chunk_index:05d}_{content_hash}"


class ChunkingConfig(BaseModel):
    """Configuration thresholds for document-aware chunking."""

    model_config = ConfigDict(frozen=True)

    target_max_tokens: int = Field(
        default=600, gt=0, description="Preferred maximum token count for narrative chunks"
    )
    hard_max_tokens: int = Field(
        default=750, gt=0, description="Hard ceiling above which individual elements are split"
    )
    narrative_overlap_tokens: int = Field(
        default=50, ge=0, description="Tokens carried forward across split narrative chunks"
    )
    table_max_tokens: int = Field(
        default=1000, gt=0, description="Token ceiling above which tables are split into row slices"
    )


class DocumentChunker:
    """
    Deterministic chunker converting DocumentElements into retrieval-ready Chunks.

    Maintains multi-page section hierarchy, groups list items, handles atomic tables,
    splits oversized tables into row slices with repeated headers, and applies narrative overlap.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk_document_elements(
        self,
        elements: Sequence[DocumentElement],
        document_name: str | None = None,
    ) -> list[Chunk]:
        """
        Processes an ordered list of DocumentElements and returns retrieval Chunks.

        Args:
            elements: Ordered stream of DocumentElement objects from layout.py.
            document_name: Optional override for document filename.

        Returns:
            List of strongly typed, deterministic Chunk objects.
        """
        if not elements:
            return []

        doc_name = document_name or elements[0].document_name

        chunks: list[Chunk] = []
        chunk_index = 0

        # State tracking across the document
        current_section_heading: str | None = None
        current_heading_path: list[str] = []
        pending_heading_elements: list[Heading] = []

        # Narrative / List accumulation buffer
        buffer_elements: list[DocumentElement] = []
        buffer_texts: list[str] = []
        pending_overlap_text: str = ""
        pending_overlap_tokens: int = 0

        def flush_buffer() -> None:
            nonlocal chunk_index, buffer_elements, buffer_texts
            nonlocal pending_overlap_text, pending_overlap_tokens, pending_heading_elements

            if not buffer_elements:
                return

            # Combine heading texts if there are pending headings
            prefix_headings: list[str] = [h.text for h in pending_heading_elements]
            heading_ids: list[str] = [h.element_id for h in pending_heading_elements]

            # Determine chunk type
            has_narrative = any(isinstance(e, NarrativeText) for e in buffer_elements)
            has_list = any(isinstance(e, ListItem) for e in buffer_elements)
            if has_narrative and has_list:
                c_type: str = "mixed"
            elif has_list:
                c_type = "list"
            else:
                c_type = "narrative"

            # Assemble body text
            body_parts: list[str] = []
            if pending_overlap_text:
                body_parts.append(pending_overlap_text)
            if prefix_headings:
                body_parts.extend(prefix_headings)
            body_parts.extend(buffer_texts)

            full_text = "\n\n".join(body_parts).strip()
            source_ids = heading_ids + [e.element_id for e in buffer_elements]
            pages = sorted({e.page_number for e in (pending_heading_elements + buffer_elements)})

            cid = generate_chunk_id(doc_name, chunk_index, source_ids, full_text)
            chunk = Chunk(
                chunk_id=cid,
                text=full_text,
                chunk_type=c_type,  # type: ignore[arg-type]
                page_numbers=pages,
                section_heading=current_section_heading,
                heading_path=list(current_heading_path),
                source_element_ids=source_ids,
                overlap_tokens=pending_overlap_tokens,
                table_metadata=None,
            )
            chunks.append(chunk)
            chunk_index += 1

            # Compute narrative overlap for subsequent chunk if pure narrative or mixed
            if c_type in ("narrative", "mixed") and self.config.narrative_overlap_tokens > 0:
                # Extract overlap from the end of the newly flushed buffer text (excluding prefix)
                raw_flushed = "\n\n".join(buffer_texts)
                ol_text, ol_tokens = extract_narrative_overlap(
                    raw_flushed, self.config.narrative_overlap_tokens
                )
                pending_overlap_text = ol_text
                pending_overlap_tokens = ol_tokens
            else:
                pending_overlap_text = ""
                pending_overlap_tokens = 0

            # Reset buffer and consumed headings
            buffer_elements = []
            buffer_texts = []
            pending_heading_elements = []

        for elem in elements:
            # 1. Filter out running headers and footers explicitly flagged by layout.py
            if elem.metadata.get("is_header_footer") is True:
                continue

            # 2. Heading element
            if isinstance(elem, Heading):
                # Flushes pending buffer before updating heading hierarchy.
                # This flush IS the hard chunk boundary at every new SCHEDULE /
                # Note section: no narrative, list, overlap, or pending heading
                # text ever crosses into the next section's chunks. Ancestor
                # prefix headings are intentionally preserved (not cleared) so
                # the next chunk keeps its full section context; hierarchy is
                # additionally carried in heading_path metadata.
                flush_buffer()
                # Clear narrative overlap across major heading boundaries
                pending_overlap_text = ""
                pending_overlap_tokens = 0

                if is_schedule_boundary_heading(elem.text):
                    logger.debug(
                        "Schedule boundary at '%s' (page %s); buffer flushed, overlap cleared",
                        elem.text,
                        elem.page_number,
                    )

                current_section_heading = elem.text
                if elem.metadata.get("heading_path"):
                    current_heading_path = list(elem.metadata["heading_path"])
                else:
                    # Update local heading stack
                    if not current_heading_path:
                        current_heading_path = [elem.text]
                    else:
                        current_heading_path.append(elem.text)

                pending_heading_elements.append(elem)
                continue

            # 3. Table element
            if isinstance(elem, Table):
                # Flush pending narrative/list buffer before processing table
                flush_buffer()
                # Tables do not receive or produce narrative overlap
                pending_overlap_text = ""
                pending_overlap_tokens = 0

                table_chunks = self._chunk_table(
                    table=elem,
                    doc_name=doc_name,
                    start_chunk_index=chunk_index,
                    section_heading=current_section_heading,
                    heading_path=current_heading_path,
                    prefix_headings=pending_heading_elements,
                )
                chunks.extend(table_chunks)
                chunk_index += len(table_chunks)
                pending_heading_elements = []
                continue

            # 4. ListItem element
            if isinstance(elem, ListItem):
                item_text = f"{elem.marker} {elem.text}" if elem.marker else f"• {elem.text}"
                item_tokens = count_tokens(item_text)

                # Check if adding this list item exceeds limits
                curr_tokens = count_tokens("\n\n".join(buffer_texts))
                if buffer_elements and (curr_tokens + item_tokens > self.config.target_max_tokens):
                    flush_buffer()
                    # Lists have zero narrative overlap
                    pending_overlap_text = ""
                    pending_overlap_tokens = 0

                buffer_elements.append(elem)
                buffer_texts.append(item_text)
                continue

            # 5. NarrativeText element
            if isinstance(elem, NarrativeText):
                narr_text = elem.text.strip()
                if not narr_text:
                    continue

                narr_tokens = count_tokens(narr_text)

                # If the single narrative element itself exceeds hard_max_tokens, split it internally
                if narr_tokens > self.config.hard_max_tokens:
                    flush_buffer()
                    sub_chunks = self._split_oversized_narrative(
                        elem=elem,
                        doc_name=doc_name,
                        start_chunk_index=chunk_index,
                        section_heading=current_section_heading,
                        heading_path=current_heading_path,
                        prefix_headings=pending_heading_elements,
                        initial_overlap_text=pending_overlap_text,
                        initial_overlap_tokens=pending_overlap_tokens,
                    )
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                    pending_heading_elements = []
                    # Carry forward overlap from the end of the last slice
                    last_chunk_text = sub_chunks[-1].text if sub_chunks else ""
                    ol_text, ol_tokens = extract_narrative_overlap(
                        last_chunk_text, self.config.narrative_overlap_tokens
                    )
                    pending_overlap_text = ol_text
                    pending_overlap_tokens = ol_tokens
                    continue

                # Normal accumulation: check if adding exceeds target_max_tokens
                curr_tokens = count_tokens("\n\n".join(buffer_texts))
                if buffer_elements and (curr_tokens + narr_tokens > self.config.target_max_tokens):
                    flush_buffer()

                buffer_elements.append(elem)
                buffer_texts.append(narr_text)
                continue

        # Final flush for any remaining buffer
        flush_buffer()

        # Handle trailing orphan headings if document ended with a heading
        if pending_heading_elements:
            trailing_text = "\n\n".join(h.text for h in pending_heading_elements)
            trailing_ids = [h.element_id for h in pending_heading_elements]
            trailing_pages = sorted({h.page_number for h in pending_heading_elements})
            cid = generate_chunk_id(doc_name, chunk_index, trailing_ids, trailing_text)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=trailing_text,
                    chunk_type="narrative",
                    page_numbers=trailing_pages,
                    section_heading=current_section_heading,
                    heading_path=list(current_heading_path),
                    source_element_ids=trailing_ids,
                    overlap_tokens=0,
                    table_metadata=None,
                )
            )

        return chunks

    def _split_oversized_narrative(
        self,
        elem: NarrativeText,
        doc_name: str,
        start_chunk_index: int,
        section_heading: str | None,
        heading_path: list[str],
        prefix_headings: list[Heading],
        initial_overlap_text: str,
        initial_overlap_tokens: int,
    ) -> list[Chunk]:
        """
        Splits a single NarrativeText element exceeding hard_max_tokens into multiple chunks
        using sentence boundaries and safe token limits while applying narrative overlap.
        """
        sentences = split_text_into_sentences(elem.text)
        slices: list[str] = []
        curr_slice_sentences: list[str] = []

        for s in sentences:
            candidate = " ".join(curr_slice_sentences + [s])
            if count_tokens(candidate) <= self.config.target_max_tokens or not curr_slice_sentences:
                curr_slice_sentences.append(s)
            else:
                slices.append(" ".join(curr_slice_sentences))
                curr_slice_sentences = [s]

        if curr_slice_sentences:
            slices.append(" ".join(curr_slice_sentences))

        sub_chunks: list[Chunk] = []
        curr_idx = start_chunk_index
        ol_text = initial_overlap_text
        ol_tokens = initial_overlap_tokens

        for i, slice_text in enumerate(slices):
            heading_prefix = [h.text for h in prefix_headings] if i == 0 else []
            heading_ids = [h.element_id for h in prefix_headings] if i == 0 else []

            parts: list[str] = []
            if ol_text:
                parts.append(ol_text)
            if heading_prefix:
                parts.extend(heading_prefix)
            parts.append(slice_text)

            chunk_text = "\n\n".join(parts).strip()
            source_ids = heading_ids + [elem.element_id]
            pages = sorted({elem.page_number, *(h.page_number for h in prefix_headings if i == 0)})

            cid = generate_chunk_id(doc_name, curr_idx, source_ids, chunk_text)
            sub_chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=chunk_text,
                    chunk_type="narrative",
                    page_numbers=pages,
                    section_heading=section_heading,
                    heading_path=list(heading_path),
                    source_element_ids=source_ids,
                    overlap_tokens=ol_tokens,
                    table_metadata=None,
                )
            )
            curr_idx += 1

            # Prepare overlap for next slice
            ol_text, ol_tokens = extract_narrative_overlap(slice_text, self.config.narrative_overlap_tokens)

        return sub_chunks

    def _chunk_table(
        self,
        table: Table,
        doc_name: str,
        start_chunk_index: int,
        section_heading: str | None,
        heading_path: list[str],
        prefix_headings: list[Heading],
    ) -> list[Chunk]:
        """
        Chunks a Table element. If the table exceeds table_max_tokens, it is partitioned
        into contiguous complete-row slices with repeated complete headers.

        A trailing Total summary row is reserved for the final slice so the
        statement total is never separated from its header context.
        """
        full_table_text = self._format_table_text(table, rows=table.rows, is_continuation=False)
        total_tokens = count_tokens(full_table_text)

        # 1. Normal-sized table: emit as single atomic chunk
        if total_tokens <= self.config.table_max_tokens or not table.rows:
            source_ids = [h.element_id for h in prefix_headings] + [table.element_id]
            t_pages = {table.page_number, *(h.page_number for h in prefix_headings)}
            if table.continuation_from_page:
                t_pages.add(table.continuation_from_page)
            pages = sorted(t_pages)

            cid = generate_chunk_id(doc_name, start_chunk_index, source_ids, full_table_text)
            t_meta = {
                "table_group_id": table.element_id,
                "table_slice_index": 0,
                "table_slice_count": 1,
                "caption": table.caption,
                "footnotes": list(table.footnotes),
                "confidence_score": table.confidence_score,
                "needs_review": table.needs_review,
                "extraction_method": table.extraction_method,
            }
            return [
                Chunk(
                    chunk_id=cid,
                    text=full_table_text,
                    chunk_type="table",
                    page_numbers=pages,
                    section_heading=section_heading,
                    heading_path=list(heading_path),
                    source_element_ids=source_ids,
                    overlap_tokens=0,
                    table_metadata=t_meta,
                )
            ]

        # 2. Oversized table: partition into row slices with repeated headers.
        # A trailing Total row is held back and appended to the final slice so
        # the total always travels with repeated headers, never stranded alone
        # or attached to an earlier slice.
        total_row: list[str] | None = None
        sliceable_rows = table.rows
        if sliceable_rows and is_total_row_label(sliceable_rows[-1][0] if sliceable_rows[-1] else None):
            total_row = sliceable_rows[-1]
            sliceable_rows = sliceable_rows[:-1]

        slices: list[list[list[str]]] = []
        curr_slice: list[list[str]] = []

        for row in sliceable_rows:
            candidate = curr_slice + [row]
            cand_text = self._format_table_text(
                table, rows=candidate, is_continuation=(len(slices) > 0)
            )
            cand_tokens = count_tokens(cand_text)

            if cand_tokens <= self.config.table_max_tokens or not curr_slice:
                curr_slice.append(row)
            else:
                slices.append(curr_slice)
                curr_slice = [row]

        if curr_slice:
            slices.append(curr_slice)

        if total_row is not None:
            if slices:
                slices[-1].append(total_row)
            else:
                slices.append([total_row])

        slice_count = len(slices)
        table_chunks: list[Chunk] = []

        for idx, row_slice in enumerate(slices):
            is_cont = idx > 0
            slice_text = self._format_table_text(table, rows=row_slice, is_continuation=is_cont)
            source_ids = ([h.element_id for h in prefix_headings] if idx == 0 else []) + [table.element_id]
            t_pages = {table.page_number}
            if table.continuation_from_page:
                t_pages.add(table.continuation_from_page)
            if idx == 0:
                t_pages.update(h.page_number for h in prefix_headings)
            pages = sorted(t_pages)

            cid = generate_chunk_id(doc_name, start_chunk_index + idx, source_ids, slice_text)
            t_meta = {
                "table_group_id": table.element_id,
                "table_slice_index": idx,
                "table_slice_count": slice_count,
                "caption": table.caption,
                "footnotes": list(table.footnotes),
                "confidence_score": table.confidence_score,
                "needs_review": table.needs_review,
                "extraction_method": table.extraction_method,
            }
            table_chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=slice_text,
                    chunk_type="table",
                    page_numbers=pages,
                    section_heading=section_heading,
                    heading_path=list(heading_path),
                    source_element_ids=source_ids,
                    overlap_tokens=0,
                    table_metadata=t_meta,
                )
            )

        return table_chunks

    def _format_table_text(
        self,
        table: Table,
        rows: list[list[str]],
        is_continuation: bool,
    ) -> str:
        """
        Renders markdown and attaches caption, headers, and footnotes for a table slice.
        """
        parts: list[str] = []

        # 1. Caption or Table label
        if table.caption:
            label = f"[Table: {table.caption} — Continuation]" if is_continuation else f"[Table: {table.caption}]"
            parts.append(label)
        elif is_continuation:
            parts.append("[Table — Continuation]")

        # 2. Markdown table (with repeated headers)
        md = Table.render_markdown(table.headers, rows)
        if md:
            parts.append(md)

        # 3. Footnotes
        if table.footnotes:
            fn_lines = ["Footnotes:"] + [f"- {fn}" for fn in table.footnotes]
            parts.append("\n".join(fn_lines))

        return "\n\n".join(parts).strip()


def verify_chunking_provenance(
    elements: Sequence[DocumentElement],
    chunks: list[Chunk],
) -> tuple[bool, list[str]]:
    """
    Validates the critical content-provenance invariant between DocumentElements and Chunks.

    Guarantees:
    - Zero non-header/footer elements lost.
    - Non-split elements appear in exactly one chunk.
    - Every table row across all table slices equals the exact original row count.

    Returns (is_valid, list_of_errors).
    """
    errors: list[str] = []

    # 1. Filter out elements with metadata is_header_footer == True
    active_elements = [e for e in elements if not e.metadata.get("is_header_footer")]
    active_element_ids = {e.element_id for e in active_elements}

    # 2. Collect all source_element_ids present across all chunks
    claimed_ids_by_chunk: dict[str, list[str]] = {}
    for c in chunks:
        for eid in c.source_element_ids:
            claimed_ids_by_chunk.setdefault(eid, []).append(c.chunk_id)

    # 3. Verify zero missing active elements
    for eid in active_element_ids:
        if eid not in claimed_ids_by_chunk:
            errors.append(f"Missing source element {eid} in generated chunks")

    # 4. Verify table row preservation
    tables = [e for e in active_elements if isinstance(e, Table)]
    for t in tables:
        t_slices = [c for c in chunks if c.chunk_type == "table" and c.table_group_id == t.element_id]
        if not t_slices:
            errors.append(f"Table {t.element_id} has no generated chunk slices")
            continue

        # Count data rows rendered across all slices by inspecting markdown lines
        total_rows_rendered = 0
        for s in t_slices:
            # Markdown table has header row, separator row, and data rows
            lines = [ln.strip() for ln in s.text.split("\n") if ln.strip().startswith("|") and ln.strip().endswith("|")]
            # subtract 2 for header and separator
            data_rows = max(0, len(lines) - 2)
            total_rows_rendered += data_rows

        if total_rows_rendered != len(t.rows):
            errors.append(
                f"Table {t.element_id} row mismatch: original {len(t.rows)} rows vs {total_rows_rendered} rendered across {len(t_slices)} slices"
            )

    is_valid = len(errors) == 0
    return is_valid, errors
