"""
Context Builder module alias for Phase B6 of the RAG pipeline.

Re-exports EvidenceContextBuilder and supporting types from
``due_diligence.retrieval.context_builder`` to maintain backward
compatibility with the agent layer specification in _docs/rag_architecture.md.
"""

from due_diligence.retrieval.context_builder import (
    ContextBuilder,
    EvidenceContextBuilder,
    OrderingStrategy,
    apply_lost_in_the_middle_ordering,
    sanitize_evidence_text,
)

__all__ = [
    "ContextBuilder",
    "EvidenceContextBuilder",
    "OrderingStrategy",
    "apply_lost_in_the_middle_ordering",
    "sanitize_evidence_text",
]
