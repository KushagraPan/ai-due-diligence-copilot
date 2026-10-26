"""
Dense vector embedding provider abstractions and implementations.

1. Purpose:
   Transforms text chunks and natural language queries into dense
   floating-point vector representations for Weaviate indexing and
   hybrid search retrieval.

2. Role in Overall RAG Pipeline:
   Acts as the vector generation engine between chunking (chunker.py) and
   Weaviate vector indexing (weaviate_store.py) during offline ingestion,
   and between query processing and hybrid retrieval during online querying.

3. Expected Inputs:
   - Document indexing: list[str] representing chunk texts.
   - Query retrieval: str representing user question or rewritten query.

4. Outputs:
   - Document vectors: list[list[float]] of exact length N, aligned with inputs.
   - Query vector: list[float] of exact dimensionality matching the index.

5. Dependencies:
   - ``openai`` for OpenAI API embeddings.
   - ``due_diligence.core.config.Settings`` for credential/model configuration.

6. Non-Responsibilities:
   - Does not store vectors in Weaviate or execute search queries.
   - Does not perform text chunking or token counting.
   - Does not perform reranking.
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any

import openai

from due_diligence.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# ── Typed Exceptions ────────────────────────────────────────────────────────


class EmbeddingError(Exception):
    """Base exception for all embedding generation and configuration errors."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when embedding configuration or credentials are invalid or mismatched."""


class EmbeddingInputError(EmbeddingError):
    """Raised when input text is empty, whitespace-only, None, or structurally invalid."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when generated vector dimensionality does not match expected dimension."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when an upstream provider (API or model) permanently fails."""


# ── Abstract Base Interface ─────────────────────────────────────────────────


class BaseEmbeddingProvider(ABC):
    """
    Abstract interface for dense vector embedding models.

    Defines the contract for transforming document texts and query strings
    into dense vectors with guaranteed dimensionality, L2 normalization,
    and strict input-to-output order preservation.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Canonical model identifier (e.g. 'text-embedding-3-large')."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Authoritative vector dimensionality produced by this provider."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate dense vectors for an ordered list of document texts.

        Guarantees:
        - len(output) == len(texts)
        - output[i] corresponds strictly to texts[i]
        - len(output[i]) == self.dimension
        """

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """
        Generate a dense vector for a single search query.

        Guarantees:
        - len(output) == self.dimension
        - Applies model-specific asymmetric query instructions where required.
        """

    @staticmethod
    def _validate_text(text: str, context: str) -> str:
        """
        Validate that input text is non-empty, non-whitespace, and a string.

        Raises EmbeddingInputError on invalid inputs.
        """
        if not isinstance(text, str):
            raise EmbeddingInputError(f"{context}: expected str, got {type(text).__name__}.")
        cleaned = text.strip()
        if not cleaned:
            raise EmbeddingInputError(f"{context}: text cannot be empty or whitespace-only.")
        return text

    def _validate_batch_inputs(self, texts: list[str]) -> list[str]:
        """Validate an entire batch of document texts before processing."""
        if not isinstance(texts, list):
            raise EmbeddingInputError(f"Expected list of strings, got {type(texts).__name__}.")
        for idx, text in enumerate(texts):
            self._validate_text(text, f"Input text at index {idx}")
        return texts


# ── OpenAI Provider ─────────────────────────────────────────────────────────


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI embeddings provider using the modern OpenAI Python SDK.

    Supports Matryoshka dimension reduction (e.g. 1536 on text-embedding-3-large),
    batching, exponential backoff with full jitter on transient rate limits,
    and strict index sorting to preserve batch ordering.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-large",
        dimension: int = 1536,
        batch_size: int = 64,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        client: openai.OpenAI | None = None,
    ) -> None:
        """
        Initialize the OpenAI embedding provider.

        Args:
            api_key: OpenAI API key.
            model_name: Embedding model identifier.
            dimension: Target vector dimensionality (authoritative).
            batch_size: Maximum texts per API batch call.
            max_retries: Maximum transient retry attempts.
            base_delay: Initial retry backoff in seconds.
            max_delay: Maximum retry backoff in seconds.
            client: Optional injected OpenAI client (useful for unit testing).
        """
        if not api_key or not api_key.strip():
            raise EmbeddingConfigurationError("OpenAI API key must be a non-empty string.")
        if dimension <= 0:
            raise EmbeddingConfigurationError(f"Embedding dimension must be positive, got {dimension}.")
        if batch_size <= 0:
            raise EmbeddingConfigurationError(f"Batch size must be positive, got {batch_size}.")

        self._model_name = model_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

        self._client = client or openai.OpenAI(
            api_key=api_key,
            timeout=30.0,
            max_retries=0,  # We manage retries explicitly for granular logging and backoff
        )
        logger.info(
            "Initialized OpenAIEmbeddingProvider (model=%s, dimension=%d, batch_size=%d)",
            self._model_name,
            self._dimension,
            self._batch_size,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Batch-embed an ordered sequence of document texts.

        Splits inputs into slices of ``batch_size``, executes API requests with
        retry handling, verifies dimensional consistency, and preserves input order.
        """
        if not texts:
            return []

        self._validate_batch_inputs(texts)

        all_vectors: list[list[float]] = []
        total_batches = (len(texts) + self._batch_size - 1) // self._batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * self._batch_size
            end = min(start + self._batch_size, len(texts))
            batch_texts = texts[start:end]

            logger.debug(
                "Embedding document batch %d/%d (%d texts)",
                batch_idx + 1,
                total_batches,
                len(batch_texts),
            )
            batch_vectors = self._embed_batch_with_retry(batch_texts, batch_idx=batch_idx + 1)
            all_vectors.extend(batch_vectors)

        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single search query string.

        Validates non-empty input and calls OpenAI API with retry handling.
        """
        valid_query = self._validate_text(query, "Search query")
        vectors = self._embed_batch_with_retry([valid_query], batch_idx=1)
        return vectors[0]

    def _embed_batch_with_retry(
        self,
        batch_texts: list[str],
        batch_idx: int,
    ) -> list[list[float]]:
        """
        Execute embedding request for a single batch with exponential backoff retry.

        Permanent errors (401, 400, 403) fail immediately without retrying.
        Transient errors (429, 500, 503, timeouts) are retried up to ``max_retries``.
        """
        attempts = 0
        while True:
            attempts += 1
            try:
                # Pass dimensions parameter for models that support Matryoshka reduction
                kwargs: dict[str, Any] = {
                    "input": batch_texts,
                    "model": self._model_name,
                }
                if "text-embedding-3" in self._model_name:
                    kwargs["dimensions"] = self._dimension

                response = self._client.embeddings.create(**kwargs)

                # Ensure strict index ordering regardless of API return order
                sorted_data = sorted(response.data, key=lambda item: item.index)
                vectors: list[list[float]] = [item.embedding for item in sorted_data]

                # Dimensionality invariant verification
                if len(vectors) != len(batch_texts):
                    raise EmbeddingDimensionError(
                        f"Expected {len(batch_texts)} vectors from API, got {len(vectors)}."
                    )
                for v_idx, vec in enumerate(vectors):
                    if len(vec) != self._dimension:
                        raise EmbeddingDimensionError(
                            f"Vector at batch index {v_idx} has {len(vec)} dimensions; "
                            f"expected {self._dimension}."
                        )

                return vectors

            except (
                openai.AuthenticationError,
                openai.PermissionDeniedError,
            ) as err:
                logger.error("OpenAI authentication/permission error: %s", err)
                raise EmbeddingConfigurationError(f"OpenAI credential error: {err}") from err

            except openai.BadRequestError as err:
                logger.error("OpenAI bad request error: %s", err)
                raise EmbeddingInputError(f"OpenAI bad request: {err}") from err

            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ) as err:
                if attempts > self._max_retries:
                    logger.error(
                        "Exhausted %d retries for embedding batch %d: %s",
                        self._max_retries,
                        batch_idx,
                        err,
                    )
                    raise EmbeddingProviderError(
                        f"OpenAI embedding request failed after {self._max_retries} retries: {err}"
                    ) from err

                delay = self._calculate_backoff(attempts)
                logger.warning(
                    "Transient OpenAI error on batch %d (attempt %d/%d): %s. Retrying in %.2fs",
                    batch_idx,
                    attempts,
                    self._max_retries,
                    err,
                    delay,
                )
                time.sleep(delay)

            except EmbeddingDimensionError:
                raise

            except Exception as err:
                logger.error("Unexpected error in OpenAI embedding request: %s", err)
                raise EmbeddingProviderError(f"Unexpected embedding error: {err}") from err

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with full random jitter."""
        exponential = self._base_delay * (2.0 ** (attempt - 1))
        capped = min(exponential, self._max_delay)
        jitter = random.uniform(0.85, 1.15)
        return capped * jitter


# ── Provider Factory ────────────────────────────────────────────────────────


def get_embedding_provider(
    settings: Settings | None = None,
) -> BaseEmbeddingProvider:
    """
    Factory function instantiating the OpenAI embedding provider from configuration.

    The configured ``embedding_dimension`` is authoritative (e.g. 1536 for
    text-embedding-3-large). Phase B3 (Weaviate Indexer) reads
    ``provider.dimension`` to configure collection vector dimensions.
    """
    cfg = settings or get_settings()

    if not cfg.openai_api_key or not cfg.openai_api_key.strip():
        raise EmbeddingConfigurationError(
            "OpenAI API key is required for embeddings. "
            "Set OPENAI_API_KEY environment variable or in .env file."
        )

    return OpenAIEmbeddingProvider(
        api_key=cfg.openai_api_key,
        model_name=cfg.embedding_model,
        dimension=cfg.embedding_dimension,
    )
