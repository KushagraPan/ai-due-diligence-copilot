"""
Typed data models bridging ingestion chunks to retrieval results.

1. Purpose:
   Defines ``IndexingPayload`` (the system boundary contract between
   Track A ingestion and Track B indexing) and ``RetrievalResult`` (the
   normalized output of Weaviate hybrid search consumed by the reranker).

2. Role in Overall RAG Pipeline:
   - ``IndexingPayload`` merges an immutable ``Chunk`` with its parent
     ``ValidatedDocument`` metadata so every Weaviate object carries full
     provenance (company, document name, financial period, content hash).
   - ``RetrievalResult`` is the typed contract between hybrid retrieval
     and the reranking / context-building stages.

3. Expected Inputs:
   - ``IndexingPayload``: receives a ``Chunk`` and document-level fields
     from ``ValidatedDocument`` at indexing time.
   - ``RetrievalResult``: populated from Weaviate query response objects
     at query time.

4. Outputs:
   - Validated, frozen Pydantic model instances ready for Weaviate
     upsert (``IndexingPayload``) or downstream reranking / context
     construction (``RetrievalResult``).

5. Dependencies:
   - ``pydantic`` (BaseModel, ConfigDict, Field)
   - ``due_diligence.ingestion.chunks.Chunk`` (read-only reference)
   - ``due_diligence.ingestion.document.ValidatedDocument`` (read-only reference)

6. Non-Responsibilities:
   - Does not call Weaviate, embedding APIs, or LLMs.
   - Does not mutate ``Chunk`` or ``ValidatedDocument``.
   - Does not perform retrieval, reranking, or generation logic.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from due_diligence.core.entity_scope import resolve_entity_scope
from due_diligence.ingestion.chunks import Chunk
from due_diligence.ingestion.document import ValidatedDocument


class IndexingPayload(BaseModel):
    """
    System boundary contract: packages a ``Chunk`` with document-level
    provenance for Weaviate indexing.

    Track B must never modify ``Chunk`` or ``ValidatedDocument``.
    Instead, this model flattens the fields that Weaviate needs into a
    single typed object suitable for batch insertion.
    """

    model_config = ConfigDict(frozen=True)

    # ── Chunk-level fields (from Chunk) ──────────────────────────────
    chunk_id: str = Field(
        ...,
        description="Deterministic unique identifier from the Chunk",
    )
    text: str = Field(
        ...,
        description="Retrieval-ready text (narrative prose or Markdown table)",
    )
    chunk_type: str = Field(
        ...,
        description="Content classification: narrative, table, list, or mixed",
    )
    page_numbers: list[int] = Field(
        ...,
        min_length=1,
        description="1-indexed PDF page numbers spanned by this chunk",
    )
    section_heading: str | None = Field(
        default=None,
        description="Innermost section heading enclosing the chunk",
    )
    heading_path: list[str] = Field(
        default_factory=list,
        description="Full hierarchical heading ancestry breadcrumbs",
    )
    source_element_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Ordered IDs of contributing DocumentElements",
    )
    table_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Unpacked table properties (group_id, slice info, headers, caption)",
    )

    # ── Document-level fields (from ValidatedDocument) ───────────────
    company: str = Field(
        ...,
        description="Target company or entity name",
    )
    document_name: str = Field(
        ...,
        description="Original document filename with extension",
    )
    document_type: str = Field(
        ...,
        description="Document format classification (e.g. 'pdf')",
    )
    financial_period: str | None = Field(
        default=None,
        description="Fiscal year or reporting period (e.g. 'FY2025-26')",
    )
    content_hash: str = Field(
        ...,
        description="SHA-256 hash of the source document for deduplication",
    )

    # ── Derived convenience fields ───────────────────────────────────
    is_table: bool = Field(
        default=False,
        description="True when the chunk represents tabular content",
    )
    table_headers: list[str] = Field(
        default_factory=list,
        description="Column headers extracted from table_metadata",
    )
    table_caption: str | None = Field(
        default=None,
        description="Table title/caption from table_metadata",
    )
    entity_scope: str = Field(
        default="hdfc_bank_general",
        description="Entity or statement scope (e.g. 'hdfc_bank_standalone', 'hdb_financial_services')",
    )

    @classmethod
    def from_chunk_and_document(
        cls,
        chunk: Chunk,
        document: ValidatedDocument,
    ) -> "IndexingPayload":
        """
        Construct an ``IndexingPayload`` by combining a ``Chunk`` with its
        parent ``ValidatedDocument`` metadata.

        Table-specific derived fields (``is_table``, ``table_headers``,
        ``table_caption``) are unpacked from ``chunk.table_metadata`` so
        they become first-class filterable properties in Weaviate.
        """
        table_meta = chunk.table_metadata
        is_table = chunk.chunk_type == "table"
        table_headers: list[str] = []
        table_caption: str | None = None

        if table_meta:
            raw_headers = table_meta.get("headers", [])
            # Headers may arrive as strings or nested structures
            table_headers = [str(h) for h in raw_headers] if raw_headers else []
            table_caption = table_meta.get("caption")

        return cls(
            # Chunk fields
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            chunk_type=chunk.chunk_type,
            page_numbers=list(chunk.page_numbers),
            section_heading=chunk.section_heading,
            heading_path=list(chunk.heading_path),
            source_element_ids=list(chunk.source_element_ids),
            table_metadata=chunk.table_metadata,
            # Document fields
            company=document.company or "",
            document_name=document.document_name,
            document_type=document.document_type.value,
            financial_period=document.financial_period,
            content_hash=document.content_hash,
            # Derived table fields
            is_table=is_table,
            table_headers=table_headers,
            table_caption=table_caption,
            entity_scope=getattr(chunk, "entity_scope", None)
            or resolve_entity_scope(
                heading_path=chunk.heading_path,
                section_heading=chunk.section_heading,
                text=chunk.text,
                page_numbers=chunk.page_numbers,
            ),
        )


class RetrievalResult(BaseModel):
    """
    Normalized retrieval candidate emitted by Weaviate hybrid search.

    Carries full provenance metadata so downstream stages (reranker,
    context builder, citation engine) can operate without re-querying
    Weaviate.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        ...,
        description="Deterministic chunk identifier",
    )
    text: str = Field(
        ...,
        description="Chunk content (narrative prose or Markdown table)",
    )
    score: float = Field(
        ...,
        description="Hybrid retrieval relevance score from Weaviate",
    )
    company: str = Field(
        ...,
        description="Target entity name",
    )
    document_name: str = Field(
        ...,
        description="Source document filename",
    )
    document_type: str = Field(
        ...,
        description="Document format classification",
    )
    financial_period: str | None = Field(
        default=None,
        description="Reporting period if specified",
    )
    page_numbers: list[int] = Field(
        ...,
        description="Physical 1-indexed PDF page numbers",
    )
    section_heading: str | None = Field(
        default=None,
        description="Immediate section heading enclosing the chunk",
    )
    heading_path: list[str] = Field(
        default_factory=list,
        description="Section hierarchy breadcrumbs",
    )
    chunk_type: str = Field(
        ...,
        description="Classification: narrative, table, list, mixed",
    )
    source_element_ids: list[str] = Field(
        ...,
        description="IDs of source DocumentElements",
    )
    table_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Unpacked table properties if applicable",
    )
    entity_scope: str = Field(
        default="hdfc_bank_general",
        description="Entity or statement scope (e.g. 'hdfc_bank_standalone', 'hdb_financial_services')",
    )

    @model_validator(mode="before")
    @classmethod
    def _populate_entity_scope(cls, data: Any) -> Any:
        if isinstance(data, dict):
            scope = data.get("entity_scope")
            if not scope or scope == "hdfc_bank_general":
                resolved = resolve_entity_scope(
                    heading_path=data.get("heading_path"),
                    section_heading=data.get("section_heading"),
                    text=data.get("text"),
                    page_numbers=data.get("page_numbers"),
                )
                if resolved != "hdfc_bank_general":
                    data["entity_scope"] = resolved
        return data


class RerankedResult(RetrievalResult):
    """
    Candidate passage scored by the LLM reranker (Phase B5).

    Preserves all original RetrievalResult metadata and the Weaviate B4 hybrid score
    in ``score``, while adding the reranker relevance score in ``rerank_score``.
    """

    model_config = ConfigDict(frozen=True)

    rerank_score: float = Field(
        ...,
        description="Reranker relevance score from Phase B5 reranker",
    )

    @classmethod
    def from_retrieval_result(
        cls,
        candidate: RetrievalResult,
        rerank_score: float,
    ) -> "RerankedResult":
        """
        Construct a RerankedResult by preserving all original candidate attributes
        and attaching the new reranker rerank_score.
        """
        return cls(
            chunk_id=candidate.chunk_id,
            text=candidate.text,
            score=candidate.score,
            company=candidate.company,
            document_name=candidate.document_name,
            document_type=candidate.document_type,
            financial_period=candidate.financial_period,
            page_numbers=list(candidate.page_numbers),
            section_heading=candidate.section_heading,
            heading_path=list(candidate.heading_path),
            chunk_type=candidate.chunk_type,
            source_element_ids=list(candidate.source_element_ids),
            table_metadata=candidate.table_metadata,
            entity_scope=candidate.entity_scope,
            rerank_score=rerank_score,
        )


class PipelineQueryResult(BaseModel):
    """
    Normalized result emitted by the retrieval pipeline (Rewriter -> B4 -> B5).

    Preserves both the original user query and the disambiguated retrieval query,
    alongside the top-K reranked evidence passages from Phase B5, providing a
    clean handoff contract for B6 Context Builder and B7 Grounded Generation.
    """

    model_config = ConfigDict(frozen=True)

    original_query: str = Field(
        ...,
        description="The user's original query wording",
    )
    retrieval_query: str = Field(
        ...,
        description="Disambiguated standalone query used for retrieval and reranking",
    )
    candidates: list[RerankedResult] = Field(
        default_factory=list,
        description="Top-K reranked evidence passages from Phase B5",
    )
    raw_retrieval_count: int = Field(
        default=0,
        ge=0,
        description="Number of candidate chunks retrieved by B4 before reranking",
    )
    raw_candidates: list[RetrievalResult] = Field(
        default_factory=list,
        description="Top-N raw candidate chunks retrieved by B4 hybrid search",
    )


class EvidenceItem(BaseModel):
    """
    Context-local evidence passage mapped from a B5 RerankedResult.

    Preserves complete chunk provenance (chunk_id, pages, headings, table metadata)
    alongside context-local identifier ('E1', 'E2') for citation resolution in B7/B8.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(
        ...,
        description="Deterministic context-local identifier (e.g. 'E1', 'E2')",
    )
    chunk_id: str = Field(
        ...,
        description="Deterministic chunk identifier from source document",
    )
    text: str = Field(
        ...,
        description="Exact source chunk text (narrative prose or Markdown table)",
    )
    page_numbers: list[int] = Field(
        default_factory=list,
        description="Physical 1-indexed PDF page numbers",
    )
    section_heading: str | None = Field(
        default=None,
        description="Immediate section heading enclosing the chunk",
    )
    heading_path: list[str] = Field(
        default_factory=list,
        description="Section hierarchy ancestry breadcrumbs",
    )
    chunk_type: str = Field(
        ...,
        description="Classification: narrative, table, list, mixed",
    )
    company: str = Field(
        ...,
        description="Target company or entity name",
    )
    document_name: str = Field(
        ...,
        description="Source document filename",
    )
    financial_period: str | None = Field(
        default=None,
        description="Reporting fiscal period if specified",
    )
    source_element_ids: list[str] = Field(
        default_factory=list,
        description="IDs of source DocumentElements",
    )
    table_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Unpacked table properties if applicable",
    )
    rerank_score: float = Field(
        ...,
        description="Reranker relevance score from Phase B5 reranker",
    )
    retrieval_score: float = Field(
        ...,
        description="Original Weaviate hybrid score from Phase B4",
    )
    entity_scope: str = Field(
        default="hdfc_bank_general",
        description="Entity or statement scope (e.g. 'hdfc_bank_standalone', 'hdb_financial_services')",
    )

    @classmethod
    def from_reranked_result(
        cls,
        evidence_id: str,
        candidate: RerankedResult,
    ) -> "EvidenceItem":
        """
        Construct an EvidenceItem by assigning a context-local evidence ID
        and preserving all underlying RerankedResult provenance.
        """
        return cls(
            evidence_id=evidence_id,
            chunk_id=candidate.chunk_id,
            text=candidate.text,
            page_numbers=list(candidate.page_numbers),
            section_heading=candidate.section_heading,
            heading_path=list(candidate.heading_path),
            chunk_type=candidate.chunk_type,
            company=candidate.company,
            document_name=candidate.document_name,
            financial_period=candidate.financial_period,
            source_element_ids=list(candidate.source_element_ids),
            table_metadata=candidate.table_metadata,
            entity_scope=candidate.entity_scope,
            rerank_score=candidate.rerank_score,
            retrieval_score=candidate.score,
        )


class OmittedEvidenceItem(BaseModel):
    """
    Record of a candidate chunk excluded by the Context Builder with explicit rationale.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        ...,
        description="Identifier of omitted chunk",
    )
    reason: str = Field(
        ...,
        description="Explicit rationale for omission (e.g. budget_exceeded, duplicate)",
    )
    rerank_score: float = Field(
        ...,
        description="B5 reranker score of omitted candidate",
    )
    chunk_type: str = Field(
        ...,
        description="Content classification of omitted candidate",
    )


class EvidenceContext(BaseModel):
    """
    Structured evidence package emitted by Context Builder (Phase B6) for Generation (Phase B7).

    Preserves both original and retrieval queries, formatted XML context string,
    evidence items, lookup map for citation attribution, and omission records.
    """

    model_config = ConfigDict(frozen=True)

    original_query: str = Field(
        ...,
        description="The user's original query wording",
    )
    retrieval_query: str = Field(
        ...,
        description="Disambiguated standalone query used for retrieval and reranking",
    )
    evidence_items: list[EvidenceItem] = Field(
        default_factory=list,
        description="Ordered active evidence items included in formatted context",
    )
    formatted_context: str = Field(
        ...,
        description="Sanitized, structured XML context string for LLM generation",
    )
    evidence_map: dict[str, EvidenceItem] = Field(
        default_factory=dict,
        description="Lookup map from evidence_id ('E1') to EvidenceItem for citation resolution",
    )
    omitted_items: list[OmittedEvidenceItem] = Field(
        default_factory=list,
        description="Items excluded due to budget, limits, or deduplication with explicit reasons",
    )
    total_context_chars: int = Field(
        default=0,
        ge=0,
        description="Total character length of formatted_context",
    )
