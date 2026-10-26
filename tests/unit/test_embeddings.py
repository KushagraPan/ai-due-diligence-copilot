"""
Unit tests for dense embedding provider abstraction and implementations.

Verifies BaseEmbeddingProvider validation, OpenAIEmbeddingProvider with mocked SDK,
retry behaviors, dimension invariant enforcement, and the provider factory.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import openai
import pytest

from due_diligence.core.config import Settings
from due_diligence.retrieval.embeddings import (
    BaseEmbeddingProvider,
    EmbeddingConfigurationError,
    EmbeddingDimensionError,
    EmbeddingInputError,
    EmbeddingProviderError,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)

# ── Helpers & Test Doubles ──────────────────────────────────────────────


def _make_embedding_item(index: int, dimension: int = 1536) -> MagicMock:
    """Create a mock OpenAI Embedding item with deterministic vector values."""
    item = MagicMock()
    item.index = index
    # Deterministic vector: index is encoded in first element
    item.embedding = [float(index)] + [0.0] * (dimension - 1)
    return item


def _make_create_response(items: list[MagicMock]) -> MagicMock:
    """Create a mock CreateEmbeddingResponse containing data items."""
    response = MagicMock()
    response.data = items
    return response


# ── Base Class & Validation Tests ───────────────────────────────────────


class TestBaseEmbeddingProviderValidation:
    """Verify input validation rules enforced on the abstract base class."""

    def test_cannot_instantiate_abstract_base_class(self) -> None:
        with pytest.raises(TypeError):
            BaseEmbeddingProvider()  # type: ignore[abstract]

    def test_validate_text_valid_string_returned(self) -> None:
        text = "Valid corporate diligence clause."
        assert BaseEmbeddingProvider._validate_text(text, "test") == text

    def test_validate_text_empty_string_raises(self) -> None:
        with pytest.raises(EmbeddingInputError, match="cannot be empty"):
            BaseEmbeddingProvider._validate_text("", "test")

    def test_validate_text_whitespace_only_raises(self) -> None:
        with pytest.raises(EmbeddingInputError, match="cannot be empty or whitespace"):
            BaseEmbeddingProvider._validate_text("   \n\t  ", "test")

    def test_validate_text_non_string_raises(self) -> None:
        with pytest.raises(EmbeddingInputError, match="expected str, got int"):
            BaseEmbeddingProvider._validate_text(12345, "test")  # type: ignore[arg-type]

    def test_validate_text_none_raises(self) -> None:
        with pytest.raises(EmbeddingInputError, match="expected str, got NoneType"):
            BaseEmbeddingProvider._validate_text(None, "test")  # type: ignore[arg-type]


# ── OpenAI Provider Tests ───────────────────────────────────────────────


class TestOpenAIEmbeddingProvider:
    """Verify OpenAIEmbeddingProvider behavior, batching, and error handling."""

    def test_init_validation_empty_api_key_raises(self) -> None:
        with pytest.raises(EmbeddingConfigurationError, match="API key must be a non-empty string"):
            OpenAIEmbeddingProvider(api_key="")

    def test_init_validation_whitespace_api_key_raises(self) -> None:
        with pytest.raises(EmbeddingConfigurationError, match="API key must be a non-empty string"):
            OpenAIEmbeddingProvider(api_key="   ")

    def test_init_validation_non_positive_dimension_raises(self) -> None:
        with pytest.raises(EmbeddingConfigurationError, match="dimension must be positive"):
            OpenAIEmbeddingProvider(api_key="sk-test", dimension=0)

    def test_init_validation_non_positive_batch_size_raises(self) -> None:
        with pytest.raises(EmbeddingConfigurationError, match="Batch size must be positive"):
            OpenAIEmbeddingProvider(api_key="sk-test", batch_size=-1)

    def test_properties(self) -> None:
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test",
            model_name="text-embedding-3-large",
            dimension=1536,
            batch_size=32,
        )
        assert provider.model_name == "text-embedding-3-large"
        assert provider.dimension == 1536
        assert provider.batch_size == 32

    def test_embed_documents_empty_list_returns_empty_immediately(self) -> None:
        mock_client = MagicMock()
        provider = OpenAIEmbeddingProvider(api_key="sk-test", client=mock_client)
        result = provider.embed_documents([])
        assert result == []
        mock_client.embeddings.create.assert_not_called()

    def test_embed_documents_rejects_empty_string_inside_batch(self) -> None:
        mock_client = MagicMock()
        provider = OpenAIEmbeddingProvider(api_key="sk-test", client=mock_client)
        with pytest.raises(EmbeddingInputError, match="Input text at index 1"):
            provider.embed_documents(["Valid text", "", "Another valid text"])
        mock_client.embeddings.create.assert_not_called()

    def test_embed_documents_single_batch(self) -> None:
        mock_client = MagicMock()
        items = [_make_embedding_item(0, 1536), _make_embedding_item(1, 1536)]
        mock_client.embeddings.create.return_value = _make_create_response(items)

        provider = OpenAIEmbeddingProvider(api_key="sk-test", client=mock_client)
        texts = ["First chunk", "Second chunk"]
        vectors = provider.embed_documents(texts)

        assert len(vectors) == 2
        assert len(vectors[0]) == 1536
        assert len(vectors[1]) == 1536
        mock_client.embeddings.create.assert_called_once_with(
            input=texts,
            model="text-embedding-3-large",
            dimensions=1536,
        )

    def test_embed_documents_ordering_preservation_with_scrambled_response(self) -> None:
        """Verify that provider re-sorts API response by item.index to preserve input order."""
        mock_client = MagicMock()
        # API returns index 1 first, then index 0
        item_0 = _make_embedding_item(0, 1536)
        item_1 = _make_embedding_item(1, 1536)
        mock_client.embeddings.create.return_value = _make_create_response([item_1, item_0])

        provider = OpenAIEmbeddingProvider(api_key="sk-test", client=mock_client)
        vectors = provider.embed_documents(["chunk_0", "chunk_1"])

        # First element of vector encodes original index in our helper
        assert vectors[0][0] == 0.0
        assert vectors[1][0] == 1.0

    def test_embed_documents_multi_batch_slicing(self) -> None:
        """Verify that batch_size=2 slices 5 texts into 3 calls (2, 2, 1)."""
        mock_client = MagicMock()

        def mock_create(**kwargs: Any) -> MagicMock:
            batch_input = kwargs["input"]
            items = [_make_embedding_item(i, 1536) for i in range(len(batch_input))]
            return _make_create_response(items)

        mock_client.embeddings.create.side_effect = mock_create

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test",
            batch_size=2,
            client=mock_client,
        )
        texts = ["t1", "t2", "t3", "t4", "t5"]
        vectors = provider.embed_documents(texts)

        assert len(vectors) == 5
        assert mock_client.embeddings.create.call_count == 3

    def test_embed_query_single_vector(self) -> None:
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _make_create_response(
            [_make_embedding_item(0, 1536)]
        )

        provider = OpenAIEmbeddingProvider(api_key="sk-test", client=mock_client)
        vector = provider.embed_query("What was HDFC Bank's net profit?")

        assert len(vector) == 1536
        assert isinstance(vector, list)
        mock_client.embeddings.create.assert_called_once_with(
            input=["What was HDFC Bank's net profit?"],
            model="text-embedding-3-large",
            dimensions=1536,
        )

    def test_embed_query_empty_raises(self) -> None:
        provider = OpenAIEmbeddingProvider(api_key="sk-test", client=MagicMock())
        with pytest.raises(EmbeddingInputError, match="Search query: text cannot be empty"):
            provider.embed_query("")

    def test_dimension_mismatch_raises_embedding_dimension_error(self) -> None:
        """When API returns vectors with incorrect dimension, must raise EmbeddingDimensionError."""
        mock_client = MagicMock()
        # Mock returns 768 dimensions when provider expects 1536
        mock_client.embeddings.create.return_value = _make_create_response(
            [_make_embedding_item(0, 768)]
        )

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test",
            dimension=1536,
            client=mock_client,
        )
        with pytest.raises(EmbeddingDimensionError, match="has 768 dimensions; expected 1536"):
            provider.embed_query("test query")

    @patch("time.sleep", return_value=None)
    def test_retry_on_transient_rate_limit(self, mock_sleep: MagicMock) -> None:
        """Verify that RateLimitError is retried and succeeds on attempt 2."""
        mock_client = MagicMock()
        rate_limit_error = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        success_response = _make_create_response([_make_embedding_item(0, 1536)])

        mock_client.embeddings.create.side_effect = [rate_limit_error, success_response]

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test",
            max_retries=3,
            client=mock_client,
        )
        vector = provider.embed_query("valid query")

        assert len(vector) == 1536
        assert mock_client.embeddings.create.call_count == 2
        mock_sleep.assert_called_once()

    @patch("time.sleep", return_value=None)
    def test_retry_exhausted_raises_provider_error(self, mock_sleep: MagicMock) -> None:
        """Verify that exhausting retries raises EmbeddingProviderError."""
        mock_client = MagicMock()
        timeout_error = openai.APITimeoutError(request=MagicMock())
        mock_client.embeddings.create.side_effect = timeout_error

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test",
            max_retries=2,
            client=mock_client,
        )
        with pytest.raises(EmbeddingProviderError, match="failed after 2 retries"):
            provider.embed_query("query")

        # 1 initial attempt + 2 retries = 3 attempts total
        assert mock_client.embeddings.create.call_count == 3

    def test_non_retryable_auth_error_fails_immediately(self) -> None:
        """Verify that AuthenticationError fails fast without retrying."""
        mock_client = MagicMock()
        auth_error = openai.AuthenticationError(
            message="Incorrect API key provided",
            response=MagicMock(status_code=401, headers={}),
            body=None,
        )
        mock_client.embeddings.create.side_effect = auth_error

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test",
            max_retries=3,
            client=mock_client,
        )
        with pytest.raises(EmbeddingConfigurationError, match="OpenAI credential error"):
            provider.embed_query("query")

        # Must not retry: call count exactly 1
        assert mock_client.embeddings.create.call_count == 1


# ── Provider Factory Tests (get_embedding_provider) ─────────────────────


class TestGetEmbeddingProviderFactory:
    """Verify the OpenAI provider factory and credential validation."""

    def test_openai_provider_created_with_authoritative_1536(self) -> None:
        settings = Settings(
            weaviate_url="https://test.weaviate.network",
            weaviate_api_key="test-key",
            embedding_model="text-embedding-3-large",
            embedding_dimension=1536,
            openai_api_key="sk-valid-key",
        )
        provider = get_embedding_provider(settings)
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.dimension == 1536
        assert provider.model_name == "text-embedding-3-large"

    def test_openai_provider_missing_key_raises(self) -> None:
        settings = Settings(
            weaviate_url="https://test.weaviate.network",
            weaviate_api_key="test-key",
            openai_api_key=None,
        )
        with pytest.raises(EmbeddingConfigurationError, match="OpenAI API key is required"):
            get_embedding_provider(settings)


class TestVectorDiagnostics:
    """Verify validation logic for real embedding vector arrays."""

    def test_clean_unit_vectors(self) -> None:
        from scripts.run_hdfc_embedding_validation import compute_vector_diagnostics

        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        diag = compute_vector_diagnostics(vectors, expected_dim=3)
        assert diag["count"] == 2
        assert diag["wrong_dim_count"] == 0
        assert not diag["has_nan"]
        assert not diag["has_inf"]
        assert not diag["has_empty"]
        assert diag["mean_l2_norm"] == 1.0

    def test_detects_wrong_dimension(self) -> None:
        from scripts.run_hdfc_embedding_validation import compute_vector_diagnostics

        vectors = [[1.0, 0.0], [0.0, 1.0, 0.0]]
        diag = compute_vector_diagnostics(vectors, expected_dim=3)
        assert diag["wrong_dim_count"] == 1

    def test_detects_nan_and_inf(self) -> None:
        from scripts.run_hdfc_embedding_validation import compute_vector_diagnostics

        vectors = [[float("nan"), 0.0], [float("inf"), 1.0]]
        diag = compute_vector_diagnostics(vectors, expected_dim=2)
        assert diag["has_nan"] is True
        assert diag["has_inf"] is True

    def test_detects_empty_vector(self) -> None:
        from scripts.run_hdfc_embedding_validation import compute_vector_diagnostics

        vectors = [[], [1.0, 0.0]]
        diag = compute_vector_diagnostics(vectors, expected_dim=2)
        assert diag["has_empty"] is True
