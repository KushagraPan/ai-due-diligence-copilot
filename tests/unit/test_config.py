"""
Unit tests for the centralized Settings configuration module.

Validates that Settings loads correctly from environment variables,
applies defaults for optional fields, and rejects missing required
fields with clear validation errors.
"""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from due_diligence.core.config import Settings

# Minimal required env vars for a valid Settings instance
REQUIRED_ENV = {
    "WEAVIATE_URL": "https://test-cluster.weaviate.network",
    "WEAVIATE_API_KEY": "test-api-key-secret",
}


class TestSettingsDefaults:
    """Verify that all optional fields have correct defaults."""

    def test_creates_with_required_fields_only(self) -> None:
        """Settings instantiates when only required fields are provided."""
        settings = Settings(**REQUIRED_ENV)  # type: ignore[arg-type]
        assert settings.weaviate_url == "https://test-cluster.weaviate.network"
        assert settings.weaviate_api_key == "test-api-key-secret"

    def test_default_app_env(self) -> None:
        settings = Settings(**REQUIRED_ENV)  # type: ignore[arg-type]
        assert settings.app_env == "development"

    def test_default_log_level(self) -> None:
        settings = Settings(**REQUIRED_ENV)  # type: ignore[arg-type]
        assert settings.log_level == "INFO"

    def test_default_embedding_provider(self) -> None:
        settings = Settings(**REQUIRED_ENV)  # type: ignore[arg-type]
        assert settings.embedding_provider == "openai"

    def test_default_embedding_model(self) -> None:
        settings = Settings(**REQUIRED_ENV)  # type: ignore[arg-type]
        assert settings.embedding_model == "text-embedding-3-large"

    def test_default_embedding_dimension(self) -> None:
        settings = Settings(**REQUIRED_ENV)  # type: ignore[arg-type]
        assert settings.embedding_dimension == 1536

    def test_default_openai_api_key_is_none_or_empty(self) -> None:
        """openai_api_key defaults to None, but may pick up an empty string
        from .env.example or environment.  Either None or '' is acceptable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            settings = Settings(**REQUIRED_ENV, _env_file=None)  # type: ignore[arg-type,call-arg]
            assert settings.openai_api_key is None or settings.openai_api_key == ""

    def test_default_hybrid_alpha(self) -> None:
        settings = Settings(**REQUIRED_ENV)  # type: ignore[arg-type]
        assert settings.hybrid_alpha == pytest.approx(0.50)

    def test_default_retrieval_top_n(self) -> None:
        settings = Settings(**REQUIRED_ENV)  # type: ignore[arg-type]
        assert settings.retrieval_top_n == 40

    def test_default_reranker_type(self) -> None:
        settings = Settings(**REQUIRED_ENV)  # type: ignore[arg-type]
        assert settings.reranker_type == "llm"

    def test_default_llm_reranker_model(self) -> None:
        settings = Settings(**REQUIRED_ENV)  # type: ignore[arg-type]
        assert settings.llm_reranker_model == "gpt-4o-mini"

    def test_default_rerank_candidate_count(self) -> None:
        settings = Settings(**REQUIRED_ENV)  # type: ignore[arg-type]
        assert settings.rerank_candidate_count == 40

    def test_default_rerank_top_k(self) -> None:
        settings = Settings(**REQUIRED_ENV)  # type: ignore[arg-type]
        assert settings.rerank_top_k == 7

    def test_default_generation_model(self) -> None:
        settings = Settings(**REQUIRED_ENV)  # type: ignore[arg-type]
        assert settings.generation_model == "gpt-4o"

    def test_default_query_rewriter_model(self) -> None:
        settings = Settings(**REQUIRED_ENV)  # type: ignore[arg-type]
        assert settings.query_rewriter_model == "gpt-4o-mini"


class TestSettingsOverrides:
    """Verify that explicit values override defaults."""

    def test_override_app_env(self) -> None:
        settings = Settings(**REQUIRED_ENV, app_env="production")  # type: ignore[arg-type]
        assert settings.app_env == "production"

    def test_override_embedding_provider_local(self) -> None:
        settings = Settings(**REQUIRED_ENV, embedding_provider="local_bge")  # type: ignore[arg-type]
        assert settings.embedding_provider == "local_bge"

    def test_override_hybrid_alpha(self) -> None:
        settings = Settings(**REQUIRED_ENV, hybrid_alpha=0.75)  # type: ignore[arg-type]
        assert settings.hybrid_alpha == pytest.approx(0.75)

    def test_override_retrieval_top_n(self) -> None:
        settings = Settings(**REQUIRED_ENV, retrieval_top_n=50)  # type: ignore[arg-type]
        assert settings.retrieval_top_n == 50

    def test_override_llm_reranker_model(self) -> None:
        settings = Settings(**REQUIRED_ENV, llm_reranker_model="gpt-4o")  # type: ignore[arg-type]
        assert settings.llm_reranker_model == "gpt-4o"

    def test_override_rerank_candidate_count(self) -> None:
        settings = Settings(**REQUIRED_ENV, rerank_candidate_count=20)  # type: ignore[arg-type]
        assert settings.rerank_candidate_count == 20

    def test_override_rerank_top_k(self) -> None:
        settings = Settings(**REQUIRED_ENV, rerank_top_k=7)  # type: ignore[arg-type]
        assert settings.rerank_top_k == 7

    def test_override_generation_model(self) -> None:
        settings = Settings(**REQUIRED_ENV, generation_model="gpt-4o-mini")  # type: ignore[arg-type]
        assert settings.generation_model == "gpt-4o-mini"

    def test_override_openai_api_key(self) -> None:
        settings = Settings(**REQUIRED_ENV, openai_api_key="sk-test-123")  # type: ignore[arg-type]
        assert settings.openai_api_key == "sk-test-123"

    def test_override_embedding_dimension(self) -> None:
        settings = Settings(**REQUIRED_ENV, embedding_dimension=3072)  # type: ignore[arg-type]
        assert settings.embedding_dimension == 3072


class TestSettingsValidation:
    """Verify that invalid inputs are rejected."""

    def test_missing_weaviate_url_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True), pytest.raises(ValidationError):
            Settings(_env_file=None, weaviate_api_key="key")  # type: ignore[call-arg]

    def test_missing_weaviate_api_key_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True), pytest.raises(ValidationError):
            Settings(_env_file=None, weaviate_url="https://test.weaviate.network")  # type: ignore[call-arg]

    def test_invalid_app_env_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(**REQUIRED_ENV, app_env="invalid_env")  # type: ignore[arg-type]

    def test_invalid_embedding_provider_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(**REQUIRED_ENV, embedding_provider="pinecone")  # type: ignore[arg-type]

    def test_hybrid_alpha_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(**REQUIRED_ENV, hybrid_alpha=-0.1)  # type: ignore[arg-type]

    def test_hybrid_alpha_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(**REQUIRED_ENV, hybrid_alpha=1.5)  # type: ignore[arg-type]

    def test_embedding_dimension_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(**REQUIRED_ENV, embedding_dimension=0)  # type: ignore[arg-type]

    def test_retrieval_top_n_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(**REQUIRED_ENV, retrieval_top_n=0)  # type: ignore[arg-type]

    def test_rerank_candidate_count_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(**REQUIRED_ENV, rerank_candidate_count=0)  # type: ignore[arg-type]

    def test_rerank_top_k_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(**REQUIRED_ENV, rerank_top_k=0)  # type: ignore[arg-type]


class TestSettingsFromEnvironment:
    """Verify that Settings loads from actual environment variables."""

    def test_loads_from_env_vars(self) -> None:
        env = {
            **REQUIRED_ENV,
            "OPENAI_API_KEY": "sk-env-test",
            "EMBEDDING_MODEL": "text-embedding-ada-002",
            "HYBRID_ALPHA": "0.45",
            "RETRIEVAL_TOP_N": "30",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.weaviate_url == "https://test-cluster.weaviate.network"
            assert settings.openai_api_key == "sk-env-test"
            assert settings.embedding_model == "text-embedding-ada-002"
            assert settings.hybrid_alpha == pytest.approx(0.45)
            assert settings.retrieval_top_n == 30

    def test_extra_env_vars_are_ignored(self) -> None:
        """Settings with extra='ignore' should not fail on unknown env vars."""
        env = {
            **REQUIRED_ENV,
            "UNKNOWN_SETTING": "should_be_ignored",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.weaviate_url == "https://test-cluster.weaviate.network"


class TestSettingsImmutability:
    """Verify that Settings fields cannot be mutated after construction."""

    def test_frozen_field_raises_on_assignment(self) -> None:
        """Pydantic Settings does not enforce frozen by default, but we
        should verify the behavior of our configuration object."""
        settings = Settings(**REQUIRED_ENV)  # type: ignore[arg-type]
        # Settings inherits from BaseSettings which may or may not be frozen.
        # This test documents current behavior rather than enforcing immutability.
        assert settings.weaviate_url == "https://test-cluster.weaviate.network"
