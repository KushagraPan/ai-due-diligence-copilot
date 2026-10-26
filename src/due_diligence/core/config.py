"""
Centralized application settings loaded from environment variables.

1. Purpose:
   Provides a single source of truth for all configuration: Weaviate Cloud
   connection, embedding model selection, retrieval/reranking parameters,
   and LLM generation model — all loaded from environment variables or a
   local `.env` file with zero hardcoded secrets.

2. Role in Overall RAG Pipeline:
   Every Track B component (embeddings, Weaviate indexer, hybrid retrieval,
   reranker, context builder, generator) imports ``get_settings()`` to obtain
   connection strings, model identifiers, and tuning parameters.

3. Expected Inputs:
   Environment variables (or `.env` file entries) such as
   ``WEAVIATE_URL``, ``WEAVIATE_API_KEY``, ``OPENAI_API_KEY``, etc.

4. Outputs:
   A validated, immutable ``Settings`` instance accessible via
   ``get_settings()`` with typed fields and sensible defaults.

5. Dependencies:
   - ``pydantic-settings`` (BaseSettings, SettingsConfigDict)
   - ``pydantic`` (Field)

6. Non-Responsibilities:
   - Does not perform Weaviate connections or API calls.
   - Does not validate that API keys are functionally correct (only that they
     are present when required).
   - Does not manage per-request or per-user configuration.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide configuration for the AI Due Diligence Co-pilot.

    All credentials and endpoints are loaded strictly from environment
    variables or a local ``.env`` file.  No secrets are hardcoded.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment identifier",
    )
    log_level: str = Field(
        default="INFO",
        description="Python logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    # ── Weaviate Cloud (sole vector & lexical store) ─────────────────
    weaviate_url: str = Field(
        ...,
        description="Weaviate Cloud cluster endpoint URL",
    )
    weaviate_api_key: str = Field(
        ...,
        description="Weaviate Cloud API authentication key",
    )

    # ── Embedding Provider ───────────────────────────────────────────
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="Model identifier used for both document and query embeddings",
    )
    embedding_dimension: int = Field(
        default=1536,
        ge=1,
        description="Vector dimensionality produced by the embedding model",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (required for embeddings and LLM calls)",
    )

    # ── Retrieval & Reranking ────────────────────────────────────────
    hybrid_alpha: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description=(
            "Weaviate hybrid search weight: "
            "1.0 = pure vector, 0.0 = pure BM25, 0.50 = 50%% vector / 50%% BM25"
        ),
    )
    retrieval_top_n: int = Field(
        default=40,
        ge=1,
        description="Number of candidate passages returned by hybrid retrieval",
    )
    reranker_type: Literal["llm"] = Field(
        default="llm",
        description="Reranker architecture: 'llm' (single-stage fast LLM reranker, e.g. gpt-4o-mini)",
    )
    llm_reranker_model: str = Field(
        default="gpt-4o-mini",
        description="Fast LLM model used for candidate reranking",
    )
    rerank_candidate_count: int = Field(
        default=40,
        ge=1,
        description="Number of B4 hybrid candidates forwarded to LLM reranking (default: 40)",
    )
    rerank_top_k: int = Field(
        default=7,
        ge=1,
        description="Number of evidence passages selected after reranking",
    )

    # ── Query Rewriting ─────────────────────────────────────────────
    query_rewriter_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model used for conversational query rewriting",
    )

    # ── LLM Generation ──────────────────────────────────────────────
    generation_model: str = Field(
        default="gpt-4o",
        description="LLM model used for grounded answer synthesis",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton application settings instance.

    Cached so repeated imports across modules share a single validated
    ``Settings`` object without re-reading environment variables.
    """
    return Settings()  # type: ignore[call-arg]
