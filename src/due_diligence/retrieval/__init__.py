"""
Post-ingestion retrieval subsystem for AI Due Diligence Co-pilot.

Exports core retrieval components:
- WeaviateStore: Vector and lexical collection manager and batch indexer.
- WeaviateHybridRetriever: Hybrid dense + BM25 retrieval engine.
- LLMReranker: LLM-based passage reranker (Phase B5).
- Schemas: IndexingPayload, RetrievalResult, and RerankedResult.
- Embeddings: BaseEmbeddingProvider and provider factories.
"""

from due_diligence.retrieval.context_builder import (
    ContextBuilder,
    EvidenceContextBuilder,
    apply_lost_in_the_middle_ordering,
    sanitize_evidence_text,
)
from due_diligence.retrieval.embeddings import (
    BaseEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)
from due_diligence.retrieval.pipeline import RetrievalPipeline
from due_diligence.retrieval.query_rewriter import (
    ConversationTurn,
    QueryRewriter,
    QueryRewriterConfigurationError,
    QueryRewriterError,
    QueryRewriterInputError,
    QueryRewriterModelError,
    rewrite_query,
)
from due_diligence.retrieval.reranker import (
    BaseReranker,
    LLMReranker,
    RerankerConfigurationError,
    RerankerError,
    RerankerInputError,
    RerankerModelError,
)
from due_diligence.retrieval.retriever import (
    RetrievalConfigurationError,
    RetrievalError,
    RetrievalQueryError,
    WeaviateHybridRetriever,
)
from due_diligence.retrieval.schemas import (
    EvidenceContext,
    EvidenceItem,
    IndexingPayload,
    OmittedEvidenceItem,
    PipelineQueryResult,
    RerankedResult,
    RetrievalResult,
)
from due_diligence.retrieval.weaviate_store import WeaviateStore

__all__ = [
    "BaseEmbeddingProvider",
    "BaseReranker",
    "ContextBuilder",
    "ConversationTurn",
    "EvidenceContext",
    "EvidenceContextBuilder",
    "EvidenceItem",
    "IndexingPayload",
    "LLMReranker",
    "OmittedEvidenceItem",
    "OpenAIEmbeddingProvider",
    "PipelineQueryResult",
    "QueryRewriter",
    "QueryRewriterConfigurationError",
    "QueryRewriterError",
    "QueryRewriterInputError",
    "QueryRewriterModelError",
    "RerankedResult",
    "RerankerConfigurationError",
    "RerankerError",
    "RerankerInputError",
    "RerankerModelError",
    "RetrievalConfigurationError",
    "RetrievalError",
    "RetrievalPipeline",
    "RetrievalQueryError",
    "RetrievalResult",
    "WeaviateHybridRetriever",
    "WeaviateStore",
    "apply_lost_in_the_middle_ordering",
    "get_embedding_provider",
    "rewrite_query",
    "sanitize_evidence_text",
]

