"""
Native Weaviate hybrid candidate retrieval layer.

1. Purpose:
   Executes hybrid (dense vector + BM25 lexical) candidate search against
   Weaviate Cloud, producing typed, deterministic, and metadata-rich
   ``RetrievalResult`` instances for corporate due diligence workflows.

2. Role in Overall RAG Pipeline:
   Acts as Phase B4 in the post-ingestion pipeline. Connects user query
   processing to LLM reranking (Phase B5). Generates query dense
   vectors via ``BaseEmbeddingProvider``, executes Weaviate native hybrid
   search (alpha=0.50, top_n=40), applies optional metadata filters, and
   returns candidate evidence chunks with full page and section provenance.

3. Expected Inputs:
   - ``query: str``: Natural language diligence question.
   - Optional metadata filters: ``company``, ``document_name``, ``document_type``,
     ``financial_period``, ``is_table``, or a raw Weaviate ``_Filters`` object.
   - Optional parameter overrides: ``top_n``, ``alpha``.

4. Outputs:
   - ``list[RetrievalResult]``: Typed, validated candidates ordered
     deterministically by descending hybrid score with chunk_id tie-breaking.

5. Dependencies:
   - ``weaviate-client`` (v4 SDK: Filter, MetadataQuery)
   - ``due_diligence.core.config`` (Settings, get_settings)
   - ``due_diligence.retrieval.schemas`` (RetrievalResult)
   - ``due_diligence.retrieval.embeddings`` (BaseEmbeddingProvider, get_embedding_provider)
   - ``due_diligence.retrieval.weaviate_store`` (WeaviateStore, DEFAULT_COLLECTION_NAME)

6. Non-Responsibilities:
   - Does NOT perform LLM reranking (Phase B5).
   - Does NOT rewrite user queries (conditional query rewriter).
   - Does NOT synthesize answers or invoke LLMs (Phase B7).
   - Does NOT manage alternative or local vector databases (Weaviate is sole store).
   - Does NOT make raw OpenAI API calls (uses BaseEmbeddingProvider abstraction).
"""

import logging
import time
from types import TracebackType
from typing import Any, Self

from pydantic import ValidationError
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.client import WeaviateClient
from weaviate.collections import Collection
from weaviate.collections.classes.filters import _Filters

from due_diligence.core.config import Settings, get_settings
from due_diligence.core.entity_scope import resolve_entity_scope
from due_diligence.retrieval.embeddings import (
    BaseEmbeddingProvider,
    EmbeddingError,
    get_embedding_provider,
)
from due_diligence.retrieval.schemas import RetrievalResult
from due_diligence.retrieval.weaviate_store import (
    DEFAULT_COLLECTION_NAME,
    WeaviateStore,
)
from due_diligence.trace import TraceRecorder

logger = logging.getLogger(__name__)


# ── Typed Exceptions ────────────────────────────────────────────────────────


class RetrievalError(Exception):
    """Base exception for all retrieval operations."""


class RetrievalConfigurationError(RetrievalError):
    """Raised when retriever configuration or initialization parameters are invalid."""


class RetrievalQueryError(RetrievalError):
    """Raised when an error occurs during query embedding or Weaviate search execution."""


# ── Native Weaviate Hybrid Retriever ────────────────────────────────────────


class WeaviateHybridRetriever:
    """
    Retrieval engine combining dense vector search and BM25 lexical matching in Weaviate Cloud.

    Follows the Single-Document MVP specification:
    - Dense vectors via authoritative BaseEmbeddingProvider (default: text-embedding-3-large).
    - Native hybrid fusion: Score = alpha * VectorScore + (1 - alpha) * BM25Score.
    - Default configuration: alpha = 0.50, top_n = 40.
    - Deterministic ordering: descending score with chunk_id tie-breaking.
    - Full provenance retention: page numbers, headings, source element IDs, table metadata.
    """

    def __init__(
        self,
        weaviate_store: WeaviateStore | None = None,
        client: WeaviateClient | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
        settings: Settings | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        alpha: float | None = None,
        top_n: int | None = None,
    ) -> None:
        """
        Initialize the hybrid retriever.

        Args:
            weaviate_store: Optional injected WeaviateStore instance.
            client: Optional injected WeaviateClient instance (e.g. for unit testing).
            embedding_provider: Optional injected BaseEmbeddingProvider.
            settings: Optional Settings instance. Defaults to get_settings().
            collection_name: Target Weaviate collection name. Defaults to 'DiligenceChunk'.
            alpha: Hybrid search weighting (0.0=pure BM25, 1.0=pure dense vector).
                   Defaults to settings.hybrid_alpha (0.50).
            top_n: Number of candidates to retrieve. Defaults to settings.retrieval_top_n (40).
        """
        self._settings = settings or get_settings()
        self._collection_name = collection_name

        # Resolve alpha
        self._alpha = alpha if alpha is not None else self._settings.hybrid_alpha
        if not (0.0 <= self._alpha <= 1.0):
            raise RetrievalConfigurationError(
                f"Hybrid alpha must be between 0.0 and 1.0, got {self._alpha}."
            )

        # Resolve top_n
        self._top_n = top_n if top_n is not None else self._settings.retrieval_top_n
        if self._top_n < 1:
            raise RetrievalConfigurationError(
                f"Retrieval top_n must be a positive integer, got {self._top_n}."
            )

        # Resolve embedding provider
        self._embedding_provider = embedding_provider or get_embedding_provider(self._settings)

        # Resolve Weaviate storage / client
        self._client = client
        self._weaviate_store = weaviate_store
        self._owns_store = False

        if self._client is None and self._weaviate_store is None:
            self._weaviate_store = WeaviateStore(settings=self._settings)
            self._owns_store = True

        logger.info(
            "Initialized WeaviateHybridRetriever (collection='%s', alpha=%.2f, top_n=%d, provider='%s')",
            self._collection_name,
            self._alpha,
            self._top_n,
            self._embedding_provider.model_name,
        )

    @property
    def alpha(self) -> float:
        """Default hybrid search alpha weighting."""
        return self._alpha

    @property
    def top_n(self) -> int:
        """Default candidate retrieval pool size."""
        return self._top_n

    @property
    def collection_name(self) -> str:
        """Target Weaviate collection name."""
        return self._collection_name

    @property
    def embedding_provider(self) -> BaseEmbeddingProvider:
        """Active embedding provider."""
        return self._embedding_provider

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close any owned underlying Weaviate resources."""
        if self._owns_store and self._weaviate_store is not None:
            self._weaviate_store.close()
            self._weaviate_store = None

    def _get_collection(self) -> Collection[Any, Any]:
        """Obtain the Weaviate Collection object from client or store."""
        if self._client is not None:
            return self._client.collections.get(self._collection_name)
        if self._weaviate_store is not None:
            return self._weaviate_store.client.collections.get(self._collection_name)
        raise RetrievalConfigurationError("No active Weaviate client or store available.")

    def _build_filter(
        self,
        company: str | None = None,
        document_name: str | None = None,
        document_type: str | None = None,
        financial_period: str | None = None,
        is_table: bool | None = None,
        filters: _Filters | None = None,
    ) -> _Filters | None:
        """
        Assemble optional metadata filters into a single Weaviate filter expression.

        Supports standard property equality filters as well as custom _Filters objects.
        """
        conditions: list[_Filters] = []

        if company is not None and company.strip():
            conditions.append(Filter.by_property("company").equal(company.strip()))

        if document_name is not None and document_name.strip():
            conditions.append(Filter.by_property("document_name").equal(document_name.strip()))

        if document_type is not None and document_type.strip():
            conditions.append(Filter.by_property("document_type").equal(document_type.strip()))

        if financial_period is not None and financial_period.strip():
            conditions.append(Filter.by_property("financial_period").equal(financial_period.strip()))

        if is_table is not None:
            conditions.append(Filter.by_property("is_table").equal(is_table))

        if filters is not None:
            conditions.append(filters)

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return Filter.all_of(conditions)

    def retrieve(
        self,
        query: str,
        *,
        company: str | None = None,
        document_name: str | None = None,
        document_type: str | None = None,
        financial_period: str | None = None,
        is_table: bool | None = None,
        filters: _Filters | None = None,
        top_n: int | None = None,
        alpha: float | None = None,
        trace: TraceRecorder | None = None,
    ) -> list[RetrievalResult]:
        """
        Execute hybrid search in Weaviate Cloud and return typed candidate passages.

        Pipeline:
        1. Validate query string. If empty or whitespace-only, safely return [].
        2. Generate dense query embedding vector via BaseEmbeddingProvider.
        3. Assemble optional metadata filters (e.g. company isolation).
        4. Execute Weaviate collection.query.hybrid with vector and BM25 text.
        5. Map response objects into typed RetrievalResult models.
        6. Apply deterministic sorting: descending score, then ascending chunk_id.

        Args:
            query: User's natural language search query.
            company: Optional company name for deal-room tenant isolation.
            document_name: Optional document filename filter.
            document_type: Optional document format classification.
            financial_period: Optional reporting period filter.
            is_table: Optional boolean to filter specifically for tabular or text chunks.
            filters: Optional custom Weaviate _Filters expression.
            top_n: Optional candidate limit override.
            alpha: Optional hybrid balance override (0.0 to 1.0).

        Returns:
            List of typed RetrievalResult objects in deterministic rank order.
        """
        # 1. Handle empty / whitespace queries safely
        if not isinstance(query, str) or not query.strip():
            logger.debug("Empty or whitespace-only query provided; returning 0 results.")
            return []

        cleaned_query = query.strip()

        b4_start = time.perf_counter() if trace is not None else 0.0

        # Resolve parameter overrides
        effective_top_n = top_n if top_n is not None else self._top_n
        if effective_top_n < 1:
            raise RetrievalConfigurationError(
                f"top_n override must be a positive integer, got {effective_top_n}."
            )

        effective_alpha = alpha if alpha is not None else self._alpha
        if not (0.0 <= effective_alpha <= 1.0):
            raise RetrievalConfigurationError(
                f"alpha override must be between 0.0 and 1.0, got {effective_alpha}."
            )

        if trace is not None:
            trace.note("b4", "embedding_model", self._embedding_provider.model_name)
            trace.note("b4", "embedding_dimension", self._embedding_provider.dimension)
            trace.note("b4", "alpha", effective_alpha)
            trace.note("b4", "top_n", effective_top_n)
            trace.note("b4", "filters", {
                "company": company,
                "document_name": document_name,
                "document_type": document_type,
                "financial_period": financial_period,
                "is_table": is_table,
                "custom_filters": "present" if filters is not None else None,
            })

        # 2. Dense embedding generation via existing provider abstraction
        try:
            embed_start = time.perf_counter() if trace is not None else 0.0
            query_vector = self._embedding_provider.embed_query(cleaned_query)
            if trace is not None:
                trace.note("b4", "embedding_latency_ms", (time.perf_counter() - embed_start) * 1000.0)
        except EmbeddingError as err:
            logger.error("Embedding generation failed for query: %s", err)
            raise RetrievalQueryError(f"Embedding generation failed: {err}") from err
        except Exception as err:
            logger.error("Unexpected error during query embedding: %s", err)
            raise RetrievalQueryError(f"Unexpected embedding error: {err}") from err

        # 3. Assemble metadata filters
        combined_filter = self._build_filter(
            company=company,
            document_name=document_name,
            document_type=document_type,
            financial_period=financial_period,
            is_table=is_table,
            filters=filters,
        )

        # 4. Execute Weaviate native hybrid search
        try:
            collection = self._get_collection()
            response = collection.query.hybrid(
                query=cleaned_query,
                vector=query_vector,
                alpha=effective_alpha,
                limit=effective_top_n,
                filters=combined_filter,
                return_metadata=MetadataQuery(score=True, explain_score=True),
            )
        except RetrievalConfigurationError:
            raise
        except Exception as err:
            logger.error("Weaviate hybrid query failed: %s", err)
            raise RetrievalQueryError(f"Weaviate hybrid search failed: {err}") from err

        # 5. Handle no results
        if not response or not hasattr(response, "objects") or not response.objects:
            logger.debug("Weaviate hybrid search returned 0 objects for query: '%s'", cleaned_query)
            if trace is not None:
                trace.stage("b4").latency_ms = (time.perf_counter() - b4_start) * 1000.0
                trace.note("b4", "returned_count", 0)
            return []

        # 6. Transform response objects to RetrievalResult
        results: list[RetrievalResult] = []
        for obj in response.objects:
            result = self._map_to_retrieval_result(obj)
            if result is not None:
                results.append(result)

        # 7. Deterministic ordering: descending score, then ascending chunk_id for tie-breaking
        results.sort(key=lambda r: (-r.score, r.chunk_id))

        # Enforce candidate pool limit
        limited = results[:effective_top_n]
        if trace is not None:
            trace.stage("b4").latency_ms = (time.perf_counter() - b4_start) * 1000.0
            trace.note("b4", "returned_count", len(limited))
        return limited

    @staticmethod
    def _map_to_retrieval_result(obj: Any) -> RetrievalResult | None:
        """
        Map a Weaviate query result object into a typed RetrievalResult.

        Extracts properties, maps stored 'chunk_text' to schema 'text', unpacks
        table metadata, and extracts the hybrid score from Weaviate metadata.
        """
        try:
            props: dict[str, Any] = getattr(obj, "properties", {}) or {}

            # Hybrid score extraction
            score = 0.0
            metadata = getattr(obj, "metadata", None)
            if metadata is not None:
                raw_score = getattr(metadata, "score", None)
                if raw_score is not None:
                    score = float(raw_score)

            # Text mapping: Weaviate stores chunk_text
            text = str(props.get("chunk_text") or props.get("text") or "")

            # Chunk ID
            chunk_id = str(props.get("chunk_id", ""))

            # Table metadata reconstruction
            table_metadata: dict[str, Any] | None = None
            if isinstance(props.get("table_metadata"), dict):
                table_metadata = props["table_metadata"]
            elif props.get("is_table") or props.get("table_headers") or props.get("table_group_id"):
                raw_headers = props.get("table_headers") or []
                table_metadata = {
                    "table_group_id": props.get("table_group_id"),
                    "table_slice_index": props.get("table_slice_index"),
                    "table_slice_count": props.get("table_slice_count"),
                    "headers": [str(h) for h in raw_headers] if isinstance(raw_headers, list) else [],
                    "caption": props.get("table_caption"),
                    "confidence_score": props.get("confidence_score"),
                    "needs_review": bool(props.get("needs_review", False)),
                    "extraction_method": props.get("extraction_method"),
                }

            # Page numbers & element IDs guarantees
            raw_pages = props.get("page_numbers")
            page_numbers = [int(p) for p in raw_pages] if isinstance(raw_pages, list) and raw_pages else [1]

            raw_elements = props.get("source_element_ids")
            source_elements = [str(e) for e in raw_elements] if isinstance(raw_elements, list) and raw_elements else [chunk_id or "unknown"]

            raw_heading_path = props.get("heading_path")
            heading_path = [str(h) for h in raw_heading_path] if isinstance(raw_heading_path, list) else []
            section_heading = props.get("section_heading")

            entity_scope = str(
                props.get("entity_scope")
                or resolve_entity_scope(heading_path, section_heading, text, page_numbers)
            )

            return RetrievalResult(
                chunk_id=chunk_id,
                text=text,
                score=score,
                company=str(props.get("company", "")),
                document_name=str(props.get("document_name", "")),
                document_type=str(props.get("document_type", "pdf")),
                financial_period=props.get("financial_period"),
                page_numbers=page_numbers,
                section_heading=section_heading,
                heading_path=heading_path,
                chunk_type=str(props.get("chunk_type", "narrative")),
                source_element_ids=source_elements,
                table_metadata=table_metadata,
                entity_scope=entity_scope,
            )
        except (ValidationError, KeyError, ValueError, TypeError, AttributeError) as err:
            logger.warning("Failed to parse Weaviate object into RetrievalResult: %s", err)
            return None
