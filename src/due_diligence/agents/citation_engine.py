"""
Citation Engine module alias for Phase B8 of the RAG pipeline.

Re-exports CitationVerifier, Citation, and VerifiedAnswer from
``due_diligence.agents.citation_verifier`` to maintain backward compatibility
with the citation engine specification in _docs/rag_implementation_plan.md.
"""

from due_diligence.agents.citation_verifier import (
    Citation,
    CitationVerifier,
    VerifiedAnswer,
    verify_citations,
)

# Alias CitationEngine to CitationVerifier for interface consistency
CitationEngine = CitationVerifier

__all__ = [
    "Citation",
    "CitationEngine",
    "CitationVerifier",
    "VerifiedAnswer",
    "verify_citations",
]
