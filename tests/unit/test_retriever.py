"""
Unit tests for the WeaviateHybridRetriever native hybrid search layer.

Verifies:
1. Retriever initialization defaults, configuration validation, and context management.
2. Safe handling of empty, whitespace-only, and non-string queries.
3. Hybrid query pipeline: embedding generation -> Weaviate native hybrid search -> typed RetrievalResult.
4. Metadata preservation: text mapping, page numbers, heading path, and table metadata reconstruction.
5. Deterministic ordering: descending score with chunk_id tie-breaking.
6. Safe metadata filtering: company, document_name, financial_period, is_table, and custom Filter objects.
7. Query-time parameter overrides (top_n, alpha) and validation.
8. Empty results handling (0 candidates returned).
9. Upstream error wrapping (EmbeddingError, Weaviate exceptions -> RetrievalQueryError).
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from weaviate.classes.query import Filter, MetadataQuery

from due_diligence.core.config import Settings
from due_diligence.retrieval.embeddings import (
    BaseEmbeddingProvider,
    EmbeddingProviderError,
)
from due_diligence.retrieval.retriever import (
    RetrievalConfigurationError,
    RetrievalQueryError,
    WeaviateHybridRetriever,
)
from due_diligence.retrieval.schemas import RetrievalResult

# ── Test Doubles & Helpers ──────────────────────────────────────────────────


class DummyEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic dummy embedding provider for offline testing."""

    def __init__(self, dimension: int = 1536) -> None:
        self._dim = dimension

    @property
    def model_name(self) -> str:
        return "text-embedding-3-large"

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dim for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        self._validate_text(query, "Query")
        return [0.2] * self._dim


def _make_mock_weaviate_object(
    chunk_id: str,
    text: str,
    score: float,
    company: str = "HDFC Bank",
    document_name: str = "hdfc_fy26.pdf",
    document_type: str = "pdf",
    financial_period: str = "FY2025-26",
    page_numbers: list[int] | None = None,
    section_heading: str | None = "Capital Adequacy",
    heading_path: list[str] | None = None,
    chunk_type: str = "narrative",
    source_element_ids: list[str] | None = None,
    is_table: bool = False,
    table_metadata: dict[str, Any] | None = None,
    table_group_id: str | None = None,
    table_headers: list[str] | None = None,
    table_caption: str | None = None,
    confidence_score: float | None = None,
) -> MagicMock:
    """Construct a mock Weaviate v4 query return object."""
    mock_obj = MagicMock()
    mock_obj.metadata.score = score
    mock_obj.metadata.explain_score = f"Score explanation for {chunk_id}"

    props: dict[str, Any] = {
        "chunk_id": chunk_id,
        "chunk_text": text,
        "company": company,
        "document_name": document_name,
        "document_type": document_type,
        "financial_period": financial_period,
        "page_numbers": page_numbers or [10],
        "section_heading": section_heading,
        "heading_path": heading_path or ["Financial Statements", "Notes"],
        "chunk_type": chunk_type,
        "source_element_ids": source_element_ids or ["elem_01"],
        "is_table": is_table,
        "table_headers": table_headers or [],
        "table_caption": table_caption,
        "table_group_id": table_group_id,
        "table_slice_index": 0 if is_table else None,
        "table_slice_count": 1 if is_table else None,
        "confidence_score": confidence_score,
        "needs_review": False,
        "extraction_method": "column_projection" if is_table else None,
    }
    if table_metadata:
        props["table_metadata"] = table_metadata

    mock_obj.properties = props
    return mock_obj


def _build_mock_collection(objects: list[Any]) -> MagicMock:
    """Construct a mock collection that returns the specified objects on hybrid query."""
    mock_collection = MagicMock()
    mock_response = MagicMock()
    mock_response.objects = objects
    mock_collection.query.hybrid.return_value = mock_response
    return mock_collection


# ── Initialization Tests ────────────────────────────────────────────────────


class TestRetrieverInitialization:
    """Verify WeaviateHybridRetriever initialization, settings, and validation."""

    def test_default_initialization(self) -> None:
        mock_client = MagicMock()
        provider = DummyEmbeddingProvider()
        settings = Settings(
            weaviate_url="https://test.weaviate.cloud",
            weaviate_api_key="key",
            openai_api_key="test-key",
            hybrid_alpha=0.60,
            retrieval_top_n=25,
        )

        retriever = WeaviateHybridRetriever(
            client=mock_client,
            embedding_provider=provider,
            settings=settings,
        )

        assert retriever.alpha == 0.60
        assert retriever.top_n == 25
        assert retriever.collection_name == "DiligenceChunk"
        assert retriever.embedding_provider == provider

    def test_custom_alpha_and_top_n(self) -> None:
        mock_client = MagicMock()
        provider = DummyEmbeddingProvider()

        retriever = WeaviateHybridRetriever(
            client=mock_client,
            embedding_provider=provider,
            alpha=0.75,
            top_n=10,
            collection_name="CustomCollection",
        )

        assert retriever.alpha == 0.75
        assert retriever.top_n == 10
        assert retriever.collection_name == "CustomCollection"

    def test_invalid_alpha_raises(self) -> None:
        mock_client = MagicMock()
        provider = DummyEmbeddingProvider()

        with pytest.raises(RetrievalConfigurationError, match="Hybrid alpha must be between"):
            WeaviateHybridRetriever(client=mock_client, embedding_provider=provider, alpha=-0.1)

        with pytest.raises(RetrievalConfigurationError, match="Hybrid alpha must be between"):
            WeaviateHybridRetriever(client=mock_client, embedding_provider=provider, alpha=1.05)

    def test_invalid_top_n_raises(self) -> None:
        mock_client = MagicMock()
        provider = DummyEmbeddingProvider()

        with pytest.raises(RetrievalConfigurationError, match="top_n must be a positive integer"):
            WeaviateHybridRetriever(client=mock_client, embedding_provider=provider, top_n=0)

    def test_context_manager_lifecycle(self) -> None:
        mock_client = MagicMock()
        provider = DummyEmbeddingProvider()

        with WeaviateHybridRetriever(
            client=mock_client,
            embedding_provider=provider,
        ) as retriever:
            assert retriever.top_n == 40

        retriever.close()


# ── Empty & No-Result Query Handling ────────────────────────────────────────


class TestEmptyAndNoResultHandling:
    """Verify that empty queries and zero results are handled cleanly."""

    def test_empty_string_returns_empty_list(self) -> None:
        mock_client = MagicMock()
        provider = MagicMock(spec=BaseEmbeddingProvider)
        retriever = WeaviateHybridRetriever(client=mock_client, embedding_provider=provider)

        results = retriever.retrieve("")
        assert results == []
        provider.embed_query.assert_not_called()
        mock_client.collections.get.assert_not_called()

    def test_whitespace_string_returns_empty_list(self) -> None:
        mock_client = MagicMock()
        provider = MagicMock(spec=BaseEmbeddingProvider)
        retriever = WeaviateHybridRetriever(client=mock_client, embedding_provider=provider)

        results = retriever.retrieve("   \t\n   ")
        assert results == []
        provider.embed_query.assert_not_called()

    def test_weaviate_empty_results_returns_empty_list(self) -> None:
        mock_client = MagicMock()
        mock_collection = _build_mock_collection([])
        mock_client.collections.get.return_value = mock_collection
        provider = DummyEmbeddingProvider()

        retriever = WeaviateHybridRetriever(client=mock_client, embedding_provider=provider)
        results = retriever.retrieve("Non-existent topic")

        assert results == []
        mock_collection.query.hybrid.assert_called_once()

    def test_weaviate_none_objects_returns_empty_list(self) -> None:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.hybrid.return_value = None
        mock_client.collections.get.return_value = mock_collection
        provider = DummyEmbeddingProvider()

        retriever = WeaviateHybridRetriever(client=mock_client, embedding_provider=provider)
        results = retriever.retrieve("query")
        assert results == []


# ── Hybrid Retrieval Pipeline Tests ─────────────────────────────────────────


class TestHybridRetrievalPipeline:
    """Verify query embedding -> hybrid search -> RetrievalResult transformation."""

    def test_retrieval_returns_typed_results_with_metadata(self) -> None:
        mock_client = MagicMock()
        obj1 = _make_mock_weaviate_object(
            chunk_id="chunk_001",
            text="CET1 ratio was 16.5% as of March 31, 2026.",
            score=0.92,
            company="HDFC Bank",
            page_numbers=[378],
            section_heading="Capital Adequacy",
        )
        mock_collection = _build_mock_collection([obj1])
        mock_client.collections.get.return_value = mock_collection
        provider = DummyEmbeddingProvider(dimension=1536)

        retriever = WeaviateHybridRetriever(
            client=mock_client,
            embedding_provider=provider,
            alpha=0.60,
            top_n=25,
        )

        results = retriever.retrieve("What was the CET1 ratio?")

        assert len(results) == 1
        res = results[0]
        assert isinstance(res, RetrievalResult)
        assert res.chunk_id == "chunk_001"
        assert res.text == "CET1 ratio was 16.5% as of March 31, 2026."
        assert res.score == pytest.approx(0.92)
        assert res.company == "HDFC Bank"
        assert res.page_numbers == [378]
        assert res.section_heading == "Capital Adequacy"
        assert res.heading_path == ["Financial Statements", "Notes"]

        # Verify call arguments to Weaviate hybrid query
        mock_collection.query.hybrid.assert_called_once()
        kwargs = mock_collection.query.hybrid.call_args.kwargs
        assert kwargs["query"] == "What was the CET1 ratio?"
        assert len(kwargs["vector"]) == 1536
        assert kwargs["alpha"] == 0.60
        assert kwargs["limit"] == 25
        assert isinstance(kwargs["return_metadata"], MetadataQuery)

    def test_reconstructs_table_metadata(self) -> None:
        mock_client = MagicMock()
        obj_table = _make_mock_weaviate_object(
            chunk_id="chunk_table_01",
            text="| Category | Amount |\n| --- | --- |\n| Advances | 1000 |",
            score=0.88,
            is_table=True,
            table_group_id="tbl_advances",
            table_headers=["Category", "Amount"],
            table_caption="Breakdown of Advances",
            confidence_score=0.98,
        )
        mock_collection = _build_mock_collection([obj_table])
        mock_client.collections.get.return_value = mock_collection
        provider = DummyEmbeddingProvider()

        retriever = WeaviateHybridRetriever(client=mock_client, embedding_provider=provider)
        results = retriever.retrieve("Show advances table")

        assert len(results) == 1
        res = results[0]
        assert res.table_metadata is not None
        assert res.table_metadata["table_group_id"] == "tbl_advances"
        assert res.table_metadata["headers"] == ["Category", "Amount"]
        assert res.table_metadata["caption"] == "Breakdown of Advances"
        assert res.table_metadata["confidence_score"] == pytest.approx(0.98)


# ── Deterministic Ordering & Tie-Breaking Tests ─────────────────────────────


class TestDeterministicOrdering:
    """Verify that results are ordered by score descending and chunk_id ascending for ties."""

    def test_sorts_by_score_descending(self) -> None:
        mock_client = MagicMock()
        # Returned out of order by mock
        obj_low = _make_mock_weaviate_object("chunk_low", "Low relevance", score=0.60)
        obj_high = _make_mock_weaviate_object("chunk_high", "High relevance", score=0.95)
        obj_mid = _make_mock_weaviate_object("chunk_mid", "Mid relevance", score=0.80)

        mock_collection = _build_mock_collection([obj_low, obj_high, obj_mid])
        mock_client.collections.get.return_value = mock_collection
        provider = DummyEmbeddingProvider()

        retriever = WeaviateHybridRetriever(client=mock_client, embedding_provider=provider)
        results = retriever.retrieve("test query")

        assert [r.chunk_id for r in results] == ["chunk_high", "chunk_mid", "chunk_low"]
        assert [r.score for r in results] == [0.95, 0.80, 0.60]

    def test_tie_breaking_by_chunk_id(self) -> None:
        mock_client = MagicMock()
        # Chunks with identical score 0.85
        obj_b = _make_mock_weaviate_object("chunk_b", "Content B", score=0.85)
        obj_c = _make_mock_weaviate_object("chunk_c", "Content C", score=0.85)
        obj_a = _make_mock_weaviate_object("chunk_a", "Content A", score=0.85)

        mock_collection = _build_mock_collection([obj_b, obj_c, obj_a])
        mock_client.collections.get.return_value = mock_collection
        provider = DummyEmbeddingProvider()

        retriever = WeaviateHybridRetriever(client=mock_client, embedding_provider=provider)
        results = retriever.retrieve("tie break test")

        assert [r.chunk_id for r in results] == ["chunk_a", "chunk_b", "chunk_c"]

    def test_limit_enforced_strictly(self) -> None:
        mock_client = MagicMock()
        objs = [
            _make_mock_weaviate_object(f"chunk_{i:02d}", f"Text {i}", score=0.90 - i * 0.01)
            for i in range(10)
        ]
        mock_collection = _build_mock_collection(objs)
        mock_client.collections.get.return_value = mock_collection
        provider = DummyEmbeddingProvider()

        retriever = WeaviateHybridRetriever(client=mock_client, embedding_provider=provider, top_n=5)
        results = retriever.retrieve("limit test")

        assert len(results) == 5
        assert results[0].chunk_id == "chunk_00"
        assert results[4].chunk_id == "chunk_04"


# ── Metadata Filtering Tests ────────────────────────────────────────────────


class TestMetadataFiltering:
    """Verify single and multi-property metadata filtering."""

    def test_company_filter_applied(self) -> None:
        mock_client = MagicMock()
        mock_collection = _build_mock_collection([])
        mock_client.collections.get.return_value = mock_collection
        provider = DummyEmbeddingProvider()

        retriever = WeaviateHybridRetriever(client=mock_client, embedding_provider=provider)
        retriever.retrieve("Deposits", company="HDFC Bank")

        kwargs = mock_collection.query.hybrid.call_args.kwargs
        filt = kwargs["filters"]
        assert filt is not None
        assert filt.target == "company"
        assert filt.value == "HDFC Bank"

    def test_multiple_filters_combined_with_all_of(self) -> None:
        mock_client = MagicMock()
        mock_collection = _build_mock_collection([])
        mock_client.collections.get.return_value = mock_collection
        provider = DummyEmbeddingProvider()

        retriever = WeaviateHybridRetriever(client=mock_client, embedding_provider=provider)
        retriever.retrieve(
            "Advances",
            company="HDFC Bank",
            financial_period="FY2025-26",
            is_table=True,
        )

        kwargs = mock_collection.query.hybrid.call_args.kwargs
        filt = kwargs["filters"]
        assert filt is not None
        # Should be an AND combination of 3 filters
        assert hasattr(filt, "filters")
        assert len(filt.filters) == 3

    def test_custom_weaviate_filter_passthrough(self) -> None:
        mock_client = MagicMock()
        mock_collection = _build_mock_collection([])
        mock_client.collections.get.return_value = mock_collection
        provider = DummyEmbeddingProvider()

        custom_filter = Filter.by_property("document_type").equal("annual_report")

        retriever = WeaviateHybridRetriever(client=mock_client, embedding_provider=provider)
        retriever.retrieve("Summary", filters=custom_filter)

        kwargs = mock_collection.query.hybrid.call_args.kwargs
        assert kwargs["filters"] == custom_filter


# ── Search Overrides & Error Handling Tests ─────────────────────────────────


class TestOverridesAndErrors:
    """Verify runtime parameter overrides and error propagation."""

    def test_runtime_alpha_and_top_n_overrides(self) -> None:
        mock_client = MagicMock()
        mock_collection = _build_mock_collection([])
        mock_client.collections.get.return_value = mock_collection
        provider = DummyEmbeddingProvider()

        retriever = WeaviateHybridRetriever(
            client=mock_client,
            embedding_provider=provider,
            alpha=0.60,
            top_n=25,
        )

        retriever.retrieve("query", alpha=0.90, top_n=5)

        kwargs = mock_collection.query.hybrid.call_args.kwargs
        assert kwargs["alpha"] == 0.90
        assert kwargs["limit"] == 5

    def test_invalid_top_n_override_raises(self) -> None:
        mock_client = MagicMock()
        provider = DummyEmbeddingProvider()
        retriever = WeaviateHybridRetriever(client=mock_client, embedding_provider=provider)

        with pytest.raises(RetrievalConfigurationError, match="top_n override must be a positive integer"):
            retriever.retrieve("query", top_n=-5)

    def test_invalid_alpha_override_raises(self) -> None:
        mock_client = MagicMock()
        provider = DummyEmbeddingProvider()
        retriever = WeaviateHybridRetriever(client=mock_client, embedding_provider=provider)

        with pytest.raises(RetrievalConfigurationError, match="alpha override must be between"):
            retriever.retrieve("query", alpha=2.0)

    def test_embedding_error_raises_retrieval_query_error(self) -> None:
        mock_client = MagicMock()
        provider = MagicMock(spec=BaseEmbeddingProvider)
        provider.embed_query.side_effect = EmbeddingProviderError("API connection timeout")

        retriever = WeaviateHybridRetriever(client=mock_client, embedding_provider=provider)

        with pytest.raises(RetrievalQueryError, match="Embedding generation failed"):
            retriever.retrieve("Failed embedding query")

    def test_weaviate_error_raises_retrieval_query_error(self) -> None:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.hybrid.side_effect = RuntimeError("Weaviate cluster unavailable")
        mock_client.collections.get.return_value = mock_collection
        provider = DummyEmbeddingProvider()

        retriever = WeaviateHybridRetriever(client=mock_client, embedding_provider=provider)

        with pytest.raises(RetrievalQueryError, match="Weaviate hybrid search failed"):
            retriever.retrieve("Failing query")
