"""
Entity and statement scope resolution and filtering for corporate diligence.

1. Purpose:
   Defines the taxonomy, deterministic resolution heuristics, and query-time filtering
   logic for distinguishing the parent bank (HDFC Bank standalone and consolidated)
   from its operating subsidiaries (HDB Financial Services, HDFC Life, HDFC ERGO,
   HDFC Securities, HDFC AMC, HDFC Capital).

2. Role in Overall RAG Pipeline:
   Operates across both ingestion and retrieval:
   - Ingestion: Attaches an immutable `entity_scope` tag to DocumentElements and Chunks.
   - Retrieval/Reranking: Prevents cross-entity conflation by filtering or deprioritizing
     subsidiary candidates when queries target the parent bank.

3. Expected Inputs:
   - Heading breadcrumbs (`heading_path`), immediate section title (`section_heading`),
     prose text excerpt, physical page numbers, and user/retrieval query strings.

4. Outputs:
   - Deterministic entity scope strings (e.g. 'hdfc_bank_standalone', 'hdb_financial_services').
   - Cleaned, entity-consistent candidate sequences for downstream reranking and generation.

5. Dependencies:
   - standard library (re, typing, logging).

6. Outside Scope:
   - Does not perform vector search or Weaviate query compilation.
   - Does not perform LLM answer generation or hallucination verification.
"""

import logging
import re
from collections.abc import Sequence
from typing import TypeVar

logger = logging.getLogger(__name__)

# Canonical Entity Scopes
SCOPE_HDFC_BANK_STANDALONE = "hdfc_bank_standalone"
SCOPE_HDFC_BANK_CONSOLIDATED = "hdfc_bank_consolidated"
SCOPE_HDFC_BANK_GENERAL = "hdfc_bank_general"

SCOPE_HDB_FINANCIAL_SERVICES = "hdb_financial_services"
SCOPE_HDFC_LIFE = "hdfc_life"
SCOPE_HDFC_ERGO = "hdfc_ergo"
SCOPE_HDFC_SECURITIES = "hdfc_securities"
SCOPE_HDFC_AMC = "hdfc_amc"
SCOPE_HDFC_CAPITAL = "hdfc_capital"

SUBSIDIARY_SCOPES: frozenset[str] = frozenset({
    SCOPE_HDB_FINANCIAL_SERVICES,
    SCOPE_HDFC_LIFE,
    SCOPE_HDFC_ERGO,
    SCOPE_HDFC_SECURITIES,
    SCOPE_HDFC_AMC,
    SCOPE_HDFC_CAPITAL,
})

# Patterns matching subsidiary names in queries or headings
SUBSIDIARY_QUERY_PATTERNS: dict[str, re.Pattern[str]] = {
    SCOPE_HDB_FINANCIAL_SERVICES: re.compile(
        r"\b(hdb|hdbfsl|hdb\s+financial)\b", re.IGNORECASE
    ),
    SCOPE_HDFC_LIFE: re.compile(
        r"\b(hdfc\s+life|life\s+insurance)\b", re.IGNORECASE
    ),
    SCOPE_HDFC_ERGO: re.compile(
        r"\b(hdfc\s+ergo|ergo|general\s+insurance)\b", re.IGNORECASE
    ),
    SCOPE_HDFC_SECURITIES: re.compile(
        r"\b(hdfc\s+securities|hsl|securities\s+broking)\b", re.IGNORECASE
    ),
    SCOPE_HDFC_AMC: re.compile(
        r"\b(hdfc\s+amc|hdfc\s+asset\s+management|mutual\s+fund)\b", re.IGNORECASE
    ),
    SCOPE_HDFC_CAPITAL: re.compile(
        r"\b(hdfc\s+capital|capital\s+advisors)\b", re.IGNORECASE
    ),
}

T = TypeVar("T")


def is_subsidiary_scope(scope: str | None) -> bool:
    """Returns True if the entity scope corresponds to a named operating subsidiary."""
    return bool(scope and scope in SUBSIDIARY_SCOPES)


def query_targets_subsidiary(query: str, subsidiary_scope: str) -> bool:
    """Check if the user/retrieval query explicitly names or targets a specific subsidiary."""
    pattern = SUBSIDIARY_QUERY_PATTERNS.get(subsidiary_scope)
    return bool(pattern and pattern.search(query))


def resolve_entity_scope(
    heading_path: Sequence[str] | None = None,
    section_heading: str | None = None,
    text: str | None = None,
    page_numbers: Sequence[int] | None = None,
) -> str:
    """
    Deterministically resolve the entity and statement scope for a chunk or document element.

    Hierarchical evaluation order:
    1. Heading hierarchy (checks for explicit subsidiary sections).
    2. Dedicated subsidiary report section page range (Physical Pages 304 to 308).
    3. Narrative named-entity focus (identifies subsidiary financial metrics).
    4. Primary Financial Statement schedules (Standalone vs. Consolidated).
    5. Fallback to general parent bank scope.
    """
    headings = " > ".join(heading_path or [])
    if section_heading:
        headings += f" > {section_heading}"
    headings_lower = headings.lower()
    text_lower = (text or "").lower()
    first_400 = text_lower[:400]

    # 1. Heading Hierarchy
    if any(k in headings_lower for k in ["hdb financial", "hdbfsl"]):
        return SCOPE_HDB_FINANCIAL_SERVICES
    if any(k in headings_lower for k in ["hdfc life", "hdfc standard life"]):
        return SCOPE_HDFC_LIFE
    if any(k in headings_lower for k in ["hdfc ergo"]):
        return SCOPE_HDFC_ERGO
    if any(k in headings_lower for k in ["hdfc securities", "hdfc sec"]):
        return SCOPE_HDFC_SECURITIES
    if any(k in headings_lower for k in ["hdfc asset management", "hdfc amc"]):
        return SCOPE_HDFC_AMC
    if any(k in headings_lower for k in ["hdfc capital"]):
        return SCOPE_HDFC_CAPITAL

    # 2. Dedicated Subsidiary Section Page Range (Directors' Report Physical Pages 304 to 308)
    p = page_numbers[0] if page_numbers else None
    if p and 304 <= p <= 308:
        if any(k in text_lower for k in ["hdbfsl", "hdb financial", "2,544 crore", "tractor loans", "micro lending", "consumer loans", "enterprise loans", "asset finance"]):
            return SCOPE_HDB_FINANCIAL_SERVICES
        if any(k in text_lower for k in ["hdfc life", "vnb", "embedded value", "solvency ratio", "pension fund"]):
            return SCOPE_HDFC_LIFE
        if any(k in text_lower for k in ["hdfc ergo", "general insurance", "motor insurance", "crop insurance"]):
            return SCOPE_HDFC_ERGO
        if any(k in text_lower for k in ["hdfc amc", "mutual fund", "qaaum", "portfolio management"]):
            return SCOPE_HDFC_AMC
        if any(k in text_lower for k in ["hdfc securities", "broking", "hsl"]):
            return SCOPE_HDFC_SECURITIES
        return SCOPE_HDB_FINANCIAL_SERVICES

    # 3. Narrative Named-Entity Focus (Outside P.304-308)
    if re.search(r"\b(hdb financial services|hdbfsl)\b", first_400) and any(k in text_lower for k in ["loan book", "pat", "nbfc", "bpo", "profit after tax"]):
        return SCOPE_HDB_FINANCIAL_SERVICES
    if re.search(r"\bhdfc life\b", first_400) and any(k in text_lower for k in ["vnb", "embedded value", "solvency ratio", "persistency"]):
        return SCOPE_HDFC_LIFE
    if re.search(r"\bhdfc ergo\b", first_400) and any(k in text_lower for k in ["general insurance", "gwp", "combined ratio"]):
        return SCOPE_HDFC_ERGO
    if re.search(r"\bhdfc securities\b", first_400) and any(k in text_lower for k in ["broking", "active clients", "demat"]):
        return SCOPE_HDFC_SECURITIES
    if re.search(r"\bhdfc amc\b|\bhdfc asset management\b", first_400) and any(k in text_lower for k in ["mutual fund", "qaaum", "aum"]):
        return SCOPE_HDFC_AMC

    # 4. Standalone vs. Consolidated Financial Statements
    # Physical page boundaries in HDFC Bank Annual Report:
    # Pages 369-493: Standalone Financial Statements & Notes
    # Pages 494-580: Consolidated Financial Statements & Notes
    if p:
        if 494 <= p <= 580:
            return SCOPE_HDFC_BANK_CONSOLIDATED
        if 369 <= p <= 493:
            return SCOPE_HDFC_BANK_STANDALONE

    # Immediate section heading takes precedence over inherited ancestry breadcrumbs
    sec_lower = (section_heading or "").lower()
    if "consolidated" in sec_lower:
        return SCOPE_HDFC_BANK_CONSOLIDATED
    if "standalone" in sec_lower:
        return SCOPE_HDFC_BANK_STANDALONE

    # Full ancestry check: check consolidated first to guard against carryover standalone headers
    if "consolidated" in headings_lower:
        return SCOPE_HDFC_BANK_CONSOLIDATED
    if "standalone" in headings_lower:
        return SCOPE_HDFC_BANK_STANDALONE

    # 5. Default Parent Bank General Scope
    return SCOPE_HDFC_BANK_GENERAL


def filter_subsidiary_candidates(
    query: str,
    candidates: Sequence[T],
) -> list[T]:
    """
    Filters out candidates from named subsidiaries when the query is asking about
    the parent bank ('HDFC Bank', 'the bank', 'bank's', or general corporate diligence),
    unless the query explicitly names that specific subsidiary.

    Preserves parent candidates (standalone, consolidated, general) while eliminating
    cross-entity financial conflation.
    """
    if not candidates:
        return []

    filtered: list[T] = []
    deprioritized: list[T] = []

    for cand in candidates:
        # Check if candidate exposes entity_scope attribute
        scope: str | None = getattr(cand, "entity_scope", None)
        if not scope:
            # Fallback to dynamic resolution if not already tagged
            heading_path = getattr(cand, "heading_path", None)
            section_heading = getattr(cand, "section_heading", None)
            text = getattr(cand, "text", "") or getattr(cand, "chunk_text", "")
            page_numbers = getattr(cand, "page_numbers", None)
            if page_numbers is None and hasattr(cand, "page_number"):
                page_numbers = [cand.page_number]
            scope = resolve_entity_scope(heading_path, section_heading, text, page_numbers)

        if is_subsidiary_scope(scope):
            # If the user's query explicitly asks about this subsidiary, retain it
            if query_targets_subsidiary(query, scope):
                filtered.append(cand)
            else:
                deprioritized.append(cand)
        else:
            filtered.append(cand)

    # If all candidates were filtered out (edge case), return original to avoid empty results
    if not filtered:
        logger.warning(
            "Entity filtering eliminated all %d candidates for query '%s'; falling back to unfiltered.",
            len(candidates),
            query,
        )
        return list(candidates)

    if deprioritized:
        logger.info(
            "Entity filtering removed %d subsidiary candidates from pool of %d for parent query '%s'.",
            len(deprioritized),
            len(candidates),
            query,
        )

    return filtered
