"""
Agent and synthesis subsystem for AI Due Diligence Co-pilot.

Exports:
- GroundedGenerator: Synthesizes grounded answers strictly from EvidenceContext (Phase B7).
- GroundedAnswer: Structured response model with answer_text, cited_evidence_ids, is_uncertain.
- generate_grounded_answer: Functional wrapper for grounded generation.
- ContextBuilder / EvidenceContextBuilder: Formats reranked candidates into structured context (Phase B6).
- CitationVerifier / CitationEngine: Verifies and attributes citations against EvidenceContext (Phase B8).
- Citation, VerifiedAnswer: Structured models for audit-grounded citations.
- verify_citations: Functional wrapper for citation verification.
"""

from due_diligence.agents.citation_engine import CitationEngine
from due_diligence.agents.citation_verifier import (
    Citation,
    CitationVerifier,
    VerifiedAnswer,
    verify_citations,
)
from due_diligence.agents.context_builder import (
    ContextBuilder,
    EvidenceContextBuilder,
    apply_lost_in_the_middle_ordering,
    sanitize_evidence_text,
)
from due_diligence.agents.generator import (
    GeneratorConfigurationError,
    GeneratorError,
    GeneratorInputError,
    GeneratorModelError,
    GroundedAnswer,
    GroundedGenerator,
    generate_grounded_answer,
    parse_grounded_answer,
)

__all__ = [
    "Citation",
    "CitationEngine",
    "CitationVerifier",
    "ContextBuilder",
    "EvidenceContextBuilder",
    "GeneratorConfigurationError",
    "GeneratorError",
    "GeneratorInputError",
    "GeneratorModelError",
    "GroundedAnswer",
    "GroundedGenerator",
    "VerifiedAnswer",
    "apply_lost_in_the_middle_ordering",
    "generate_grounded_answer",
    "parse_grounded_answer",
    "sanitize_evidence_text",
    "verify_citations",
]
