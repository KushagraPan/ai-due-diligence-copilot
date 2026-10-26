"""
Chunk schema definitions representing retrieval-ready document units.

1. Purpose:
   Defines the immutable Chunk model representing layout-aware, context-enriched
   retrieval units constructed from ordered DocumentElements.

2. Role in Overall RAG Pipeline:
   Acts as the data boundary between ingestion chunking (chunker.py) and downstream
   indexing (vector embedding, BM25 sparse indexing, LLM reranking).

3. Expected Inputs:
   - Chunk attributes emitted by DocumentChunker: text, chunk_type, page_numbers,
     heading hierarchy, source element IDs, overlap tokens, and table metadata.

4. Outputs:
   - Strongly typed, frozen Pydantic Chunk instances with deterministic IDs.

5. Dependencies:
   - pydantic (BaseModel, ConfigDict, Field)
   - typing (Literal, Any)

6. Non-Responsibilities:
   - Does not perform tokenization or text splitting (handled in chunker.py).
   - Does not generate vector embeddings or communicate with Weaviate.
   - Does not perform layout reconciliation (handled in layout.py).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """
    Immutable retrieval-ready document chunk.

    Encapsulates text, hierarchical section provenance, page tracking,
    and source DocumentElement references for exact audit citations.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        ...,
        description="Deterministic unique identifier for the chunk",
    )
    text: str = Field(
        ...,
        description="Retrieval-ready text for embedding and lexical search",
    )
    chunk_type: Literal["narrative", "table", "list", "mixed"] = Field(
        ...,
        description="Classification of content within the chunk",
    )
    page_numbers: list[int] = Field(
        ...,
        min_length=1,
        description="Unique sorted physical 1-indexed PDF page numbers represented in this chunk",
    )
    section_heading: str | None = Field(
        default=None,
        description="Immediate or leaf section heading enclosing this chunk",
    )
    heading_path: list[str] = Field(
        default_factory=list,
        description="Full hierarchical heading ancestry (e.g. ['Financial Statements', 'Note 18'])",
    )
    entity_scope: str = Field(
        default="hdfc_bank_general",
        description="Entity or statement scope (e.g. 'hdfc_bank_standalone', 'hdfc_bank_consolidated', 'hdb_financial_services')",
    )
    source_element_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of DocumentElement IDs contributing to this chunk",
    )
    overlap_tokens: int = Field(
        default=0,
        ge=0,
        description="Approximate narrative retrieval tokens carried over from previous chunk",
    )
    table_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata for table chunks (table_group_id, slice indexes, caption, etc.)",
    )

    @property
    def table_group_id(self) -> str | None:
        """Returns the parent table group ID if this chunk is a table slice."""
        if self.table_metadata:
            return self.table_metadata.get("table_group_id")
        return None

    @property
    def table_slice_index(self) -> int | None:
        """Returns the 0-indexed slice index if this chunk is a table slice."""
        if self.table_metadata:
            return self.table_metadata.get("table_slice_index")
        return None

    @property
    def table_slice_count(self) -> int | None:
        """Returns total slice count if this chunk is a table slice."""
        if self.table_metadata:
            return self.table_metadata.get("table_slice_count")
        return None

    @property
    def confidence_score(self) -> float | None:
        """Returns extraction confidence score if this chunk represents a table."""
        if self.table_metadata:
            val = self.table_metadata.get("confidence_score")
            return float(val) if val is not None else None
        return None

    @property
    def needs_review(self) -> bool | None:
        """Returns human review flag if this chunk represents a table."""
        if self.table_metadata:
            val = self.table_metadata.get("needs_review")
            return bool(val) if val is not None else None
        return None

    @property
    def extraction_method(self) -> str | None:
        """Returns extraction tier (tier1, tier2_gemini, tier3) if this chunk represents a table."""
        if self.table_metadata:
            return self.table_metadata.get("extraction_method")
        return None
