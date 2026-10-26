"""
Citation verification and attribution subsystem for Phase B8 of the RAG pipeline.

1. Purpose:
   Verifies that evidence tokens ([EX]) emitted by Phase B7 Grounded Generation
   exist in the active EvidenceContext (Phase B6) evidence map AND that the cited
   evidence set actually supports the attributed claim (content-support check),
   translates valid tokens into human-readable footnotes and structured Citation
   instances with exact document and page coordinates, and detects
   unverified/hallucinated citations. Structured claims are checked as a set, so
   one claim may be supported by several chunks and one token may support several
   claims.

2. Role in Overall RAG Pipeline:
   Acts as Phase B8 Citation Verification between Phase B7 (Grounded Generation)
   and the user presentation layer (CLI / API). Bridges model-generated internal
   evidence IDs ('E1', 'E2') to immutable source document provenance.

3. Expected Inputs:
   - grounded_answer: GroundedAnswer | str | dict[str, Any] (from Phase B7 with
     structured claims and/or answer_text plus cited_evidence_ids)
   - context: EvidenceContext (from Phase B6 containing evidence_map with complete provenance)

4. Outputs:
   - VerifiedAnswer: Container with resolved answer_text ([1], [2]), structured
     Citation objects, and any flagged unverified citations.

5. Dependencies:
   - pydantic (BaseModel, ConfigDict, Field)
   - due_diligence.agents.generator (GroundedAnswer)
   - due_diligence.retrieval.schemas (EvidenceContext, EvidenceItem)

6. Non-Responsibilities:
   - Does NOT retrieve documents or execute vector search (Phase B4).
   - Does NOT rerank candidate passages (Phase B5).
   - Does NOT assemble context blocks (Phase B6).
   - Does NOT generate or synthesize text via LLMs (Phase B7).
   - Does NOT fabricate page numbers or source metadata for unverified tokens.
"""

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from due_diligence.agents.generator import Claim, GroundedAnswer
from due_diligence.ingestion.elements import (
    TOTAL_ROW_PATTERN,
    is_schedule_boundary_heading,
)
from due_diligence.retrieval.schemas import EvidenceContext, EvidenceItem

logger = logging.getLogger(__name__)


# ── Numeric Grounding Check (headline-figure tripwire) ───────────────────────


# Tokens carrying no subject meaning for label/question keyword overlap.
_GROUNDING_STOPWORDS = frozenset({
    "what", "was", "were", "is", "are", "the", "a", "an", "of", "as", "on",
    "in", "for", "to", "by", "and", "or", "with", "from", "about", "how",
    "much", "many", "does", "did", "bank", "hdfc", "limited", "year", "ended",
    "march", "april", "february", "january", "december", "per", "its",
})
_NUMBER_PATTERN = re.compile(r"\d[\d,]*\.?\d*\s?%?")
_YEAR_PATTERN = re.compile(r"^(19|20)\d{2}$")
# Date expressions whose fragments must not be treated as headline numbers
_DATE_MASK_PATTERN = re.compile(
    r"(FY\s?(19|20)\d{2}\s*-\s*\d{2,4})|((19|20)\d{2}\s*-\s*\d{1,4})|"
    r"((January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+(19|20)\d{2})",
    re.IGNORECASE,
)
_UNIT_OR_CURRENCY_PATTERN = re.compile(
    r"(crore|thousand|lakh|million|billion|trillion|bps|percent|per\s+cent|₹|\brs\.?|\binr\b)",
    re.IGNORECASE,
)


def _question_keywords(query: str) -> set[str]:
    """Extract subject-bearing keyword tokens from a diligence question."""
    keywords: set[str] = set()
    for tok in re.findall(r"[A-Za-z]+", query.lower()):
        if len(tok) >= 3 and tok not in _GROUNDING_STOPWORDS:
            keywords.add(tok)
    return keywords


def _normalize_number(num: str) -> str:
    """Normalize a numeric string for comparison (strip commas/spaces/percent)."""
    return num.replace(",", "").replace(" ", "").rstrip("%")


# Periods that do not end a sentence (single initials, common abbreviations).
_ABBR_PERIOD_PATTERN = re.compile(
    r"\b(?:[A-Z]\.|Co\.|Ltd\.|Inc\.|No\.|Mr\.|Mrs\.|Ms\.|Dr\.|St\.|Jr\.|Sr\.|vs\.)\s"
)


def _sentences_citing_token(answer_text: str, token: str) -> list[str]:
    """Split answer into sentences, returning those citing the given token."""
    # Protect abbreviation periods ("B.", "Co.", "Ltd.") so entity names
    # spanning them are not fractured into separate sentences.
    protected = _ABBR_PERIOD_PATTERN.sub(
        lambda m: m.group(0).replace(".", "\u0001"), answer_text
    )
    sentences = [
        s.replace("\u0001", ".")
        for s in re.split(r"(?<=[.!?])\s+", protected)
    ]
    return [s for s in sentences if f"[{token}]" in s]


def _segments_preceding_token(sentence: str, token: str) -> list[str]:
    """
    Return the text segments a citation token directly supports: the span since
    the previous citation token (or sentence start) up to each occurrence of the
    token. A citation supports the claim immediately preceding it, not every
    number elsewhere in a multi-citation sentence.
    """
    segments: list[str] = []
    marker = f"[{token}]"
    other_markers = re.compile(r"\[[A-Za-z]*\d+\]")
    start = 0
    idx = sentence.find(marker)
    while idx != -1:
        preceding = sentence[start:idx]
        # Narrow to text since the previous citation token, if any
        others = list(other_markers.finditer(preceding))
        if others:
            preceding = preceding[others[-1].end():]
        segments.append(preceding)
        start = idx + len(marker)
        idx = sentence.find(marker, start)
    return segments


def _is_headline_number(raw_num: str, segment: str) -> bool:
    """Decide whether a numeric mention qualifies as a headline figure worth checking."""
    norm = _normalize_number(raw_num)
    if not norm:
        return False
    if raw_num.strip().endswith("%"):
        return True
    digits = re.sub(r"\D", "", norm)
    if len(digits) >= 4:
        return True
    return bool(_UNIT_OR_CURRENCY_PATTERN.search(segment))


# Runs of 2+ capitalized words (allowing '&' connectors and single letters),
# e.g. "Batliboi & Purohit", "B S R & Co. LLP", "HDFC Bank".
_ENTITY_RUN_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z.]*\.?(?:\s+(?:&\s+)?[A-Z][A-Za-z.]*\.?)+)"
)


def _normalize_entity(text: str) -> str:
    """Normalize an entity mention for comparison (case, periods, whitespace)."""
    lowered = text.lower().replace(".", " ")
    return re.sub(r"\s+", " ", lowered).strip()


# Words that make up the primary-entity name only. B7 is required to attribute
# answers to HDFC Bank while table chunks rarely repeat the name; primary-entity
# attribution is governed by the B6 statement-scope labels and entity_scope
# metadata, not by chunk-text presence, so such runs are not checkable anchors.
_PRIMARY_ENTITY_WORDS = frozenset({"hdfc", "bank", "limited"})

# Distinctive name keywords per subsidiary scope. An entity anchor consistent
# with the chunk's own scope is scope-attributed (trusted via the same metadata
# path as B6 labels); anything else must appear in the chunk text.
_SCOPE_ENTITY_KEYWORDS: dict[str, frozenset[str]] = {
    "hdb_financial_services": frozenset({"hdb", "hdbfsl", "hdb financial services"}),
    "hdfc_life": frozenset({"hdfc life"}),
    "hdfc_ergo": frozenset({"hdfc ergo", "ergo"}),
    "hdfc_securities": frozenset({"hdfc securities", "hsl"}),
    "hdfc_amc": frozenset({"hdfc amc"}),
    "hdfc_capital": frozenset({"hdfc capital"}),
}


def _is_primary_entity_run(normalized_run: str) -> bool:
    """True when an entity run names only the primary entity."""
    words = set(re.findall(r"[A-Za-z0-9]+", normalized_run))
    return bool(words) and words <= _PRIMARY_ENTITY_WORDS


def _matches_chunk_scope(normalized_run: str, entity_scope: str | None) -> bool:
    """True when an entity anchor is consistent with the chunk's own scope."""
    keywords = _SCOPE_ENTITY_KEYWORDS.get((entity_scope or "").strip().lower())
    if not keywords:
        return False
    return any(
        re.search(rf"\b{re.escape(kw)}\b", normalized_run) for kw in keywords
    )


def _equation_results(segment: str) -> set[str]:
    """Numbers appearing as the result of an explicit inline equation (``A - B = C``)."""
    results: set[str] = set()
    for match in re.finditer(r"=\s*(\d[\d,]*\.?\d*\s?%?)", segment):
        norm = _normalize_number(match.group(1).strip())
        if norm:
            results.add(norm)
    return results


def _extract_claim_entities(masked_segment: str) -> list[str]:
    """Extract multi-word capitalized entity mentions from a claim segment."""
    return [m.group(1) for m in _ENTITY_RUN_PATTERN.finditer(masked_segment)]


def _entity_supported(entity: str, item_entities: str) -> bool:
    """
    Decide whether a multi-word entity run is supported by chunk text.

    Requires at least half of the run's alphanumeric words to appear (case and
    punctuation normalized). This tolerates leading descriptors ("Auditors B S R
    ...") while still rejecting runs with no foothold ("Batliboi & Purohit"
    against a chunk naming neither).
    """
    words = re.findall(r"[A-Za-z0-9]+", _normalize_entity(entity))
    if not words:
        return True
    chunk_words = set(re.findall(r"[A-Za-z0-9]+", item_entities))
    hits = sum(1 for w in words if w in chunk_words)
    return hits * 2 >= len(words)


def _claim_anchors(segment: str) -> tuple[list[str], list[str]]:
    """Return the headline numbers and named entities in a claim segment."""
    # Non-citation bracket markers ([NIL/NOT DISCLOSED], [Calculated: ...])
    # are annotations, not claim content.
    masked = re.sub(r"\[[^\]]*\]", " ", segment)
    masked = _DATE_MASK_PATTERN.sub(" ", masked)

    numbers: list[str] = []
    if "calculat" not in segment.lower():
        equation_results = _equation_results(segment)
        for match in _NUMBER_PATTERN.finditer(masked):
            raw_num = match.group(0).strip()
            norm_num = _normalize_number(raw_num)
            if not norm_num or _YEAR_PATTERN.match(norm_num):
                continue
            if norm_num in equation_results:
                continue
            if not _is_headline_number(raw_num, segment):
                continue
            numbers.append(raw_num)

    entities: list[str] = []
    for entity in _extract_claim_entities(masked):
        normalized_run = _normalize_entity(entity)
        if _is_primary_entity_run(normalized_run):
            continue
        entities.append(entity)
    return numbers, entities


def _claim_content_words(text: str) -> set[str]:
    """Extract meaningful words for a conservative lexical-support fallback."""
    words = re.findall(r"[A-Za-z]{4,}", text.lower())
    return {word for word in words if word not in _GROUNDING_STOPWORDS}


def _claim_text_support_reasons(
    segment: str,
    token: str,
    item: EvidenceItem,
) -> list[str]:
    """Check one claim segment's numbers and entities against a cited chunk."""
    reasons: list[str] = []
    item_norm = _normalize_number(item.text)
    item_entities = _normalize_entity(item.text)
    numbers, entities = _claim_anchors(segment)

    for raw_num in numbers:
        norm_num = _normalize_number(raw_num)
        if norm_num not in item_norm:
            reasons.append(
                f"[{token}] cited claim includes number {raw_num} absent "
                f"from evidence text (chunk {item.chunk_id})"
            )
    for entity in entities:
        normalized_run = _normalize_entity(entity)
        if _matches_chunk_scope(normalized_run, item.entity_scope):
            continue
        if not _entity_supported(entity, item_entities):
            reasons.append(
                f"[{token}] cited claim names entity '{entity}' unsupported "
                f"by evidence text (chunk {item.chunk_id})"
            )
    return reasons


def _item_contributes_anchor(
    item: EvidenceItem,
    numbers: list[str],
    entities: list[str],
) -> bool:
    """Whether a cited chunk supports at least one checkable claim anchor."""
    if not numbers and not entities:
        return True
    item_norm = _normalize_number(item.text)
    item_entities = _normalize_entity(item.text)
    if any(_normalize_number(raw_num) in item_norm for raw_num in numbers):
        return True
    for entity in entities:
        normalized_run = _normalize_entity(entity)
        if _matches_chunk_scope(normalized_run, item.entity_scope):
            return True
        if _entity_supported(entity, item_entities):
            return True
    return False


def check_claim_evidence_support(
    claim_text: str,
    evidence_ids: list[str],
    items: list[EvidenceItem],
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Verify a structured claim against its complete evidence set.

    The claim's numbers and entities may be spread across several cited chunks;
    the union must contain every anchor. Each cited chunk must also contribute at
    least one checkable anchor, so irrelevant citations are not silently accepted.

    Returns:
        A pair ``(claim_failures, token_failures)``. Claim-level failures apply to
        every evidence ID in the claim because the union is incomplete. Token-level
        failures apply only to the specific evidence ID.
    """
    claim_failures: list[str] = []
    token_failures: dict[str, list[str]] = {}
    item_by_id = {item.evidence_id: item for item in items}

    for token in evidence_ids:
        if token not in item_by_id:
            token_failures.setdefault(token, []).append(
                f"[{token}] cited by generator but absent from EvidenceContext."
            )

    resolved_items = [item_by_id[token] for token in evidence_ids if token in item_by_id]
    if not resolved_items:
        claim_failures.append(
            f"claim cites no resolvable evidence: {', '.join(evidence_ids) or '(none)'}"
        )
        return claim_failures, token_failures

    token_label = "/".join(evidence_ids)
    union_text = "\n".join(item.text for item in resolved_items)
    union_item = resolved_items[0].model_copy(
        update={"text": union_text, "entity_scope": None}
    )
    union_reasons = _claim_text_support_reasons(claim_text, token_label, union_item)
    if union_reasons:
        claim_failures.extend(union_reasons)
        return claim_failures, token_failures

    numbers, entities = _claim_anchors(claim_text)
    if not numbers and not entities:
        claim_words = _claim_content_words(claim_text)
        for item in resolved_items:
            if not claim_words:
                token_failures.setdefault(item.evidence_id, []).append(
                    f"[{item.evidence_id}] claim has no checkable content to verify "
                    f"(chunk {item.chunk_id})"
                )
            elif not (claim_words & _claim_content_words(item.text)):
                token_failures.setdefault(item.evidence_id, []).append(
                    f"[{item.evidence_id}] cited evidence contributes no lexical "
                    f"support to the claim (chunk {item.chunk_id})"
                )
        return claim_failures, token_failures

    for item in resolved_items:
        if not _item_contributes_anchor(item, numbers, entities):
            token_failures.setdefault(item.evidence_id, []).append(
                f"[{item.evidence_id}] cited evidence contributes no checkable element "
                f"of the claim (chunk {item.chunk_id})"
            )
    return claim_failures, token_failures


def _claim_with_token(claim_text: str, token: str) -> str:
    """Attach one token to a claim for the advisory numeric-grounding check."""
    text = claim_text.rstrip()
    if text.endswith((".", "!", "?")):
        return f"{text[:-1]} [{token}]{text[-1]}"
    return f"{text} [{token}]"


def check_claim_support(
    answer_text: str,
    token: str,
    item: EvidenceItem,
) -> list[str]:
    """
    Content-support check for free-form answers: the cited chunk must actually
    contain the specific claim attributed to it, not merely exist as a real chunk.

    For each claim segment preceding the citation token, every headline number
    and every multi-word named entity must appear (normalized) in the chunk
    text. Calculated derivations (``(Calculated ...)``) are exempt for numbers
    since they need not appear verbatim. Segments with no checkable anchors
    pass. Returns human-readable failure reasons (empty = supported).
    """
    reasons: list[str] = []
    for sentence in _sentences_citing_token(answer_text, token):
        segments = [
            segment
            for segment in _segments_preceding_token(sentence, token)
            if segment.strip()
        ]
        if not segments:
            reasons.append(
                f"[{token}] citation has no preceding claim text to verify."
            )
            continue
        for segment in segments:
            reasons.extend(_claim_text_support_reasons(segment, token, item))
    return reasons


def check_claim_text_support(
    claim_text: str,
    token: str,
    item: EvidenceItem,
) -> list[str]:
    """
    Content-support check for one structured claim.

    Unlike the free-form path, this does not infer claim boundaries from citation
    placement. Every number and entity in the claim must be present in the cited
    chunk. Returns human-readable failure reasons (empty = supported).
    """
    return _claim_text_support_reasons(claim_text, token, item)


def check_numeric_grounding(
    answer_text: str,
    token: str,
    item: EvidenceItem,
    question: str,
) -> list[str]:
    """
    Tripwire check: headline numbers cited from an evidence chunk should appear
    in a table row or sentence whose label matches the question's subject.

    Returns warning strings (never raises). Skips model-calculated figures
    (``(Calculated ...)`` derivations need not appear verbatim) and never
    fails verification — mismatches are logged for analyst review.
    """
    warnings: list[str] = []
    keywords = _question_keywords(question)
    if not keywords:
        return warnings

    item_text = item.text
    item_norm = _normalize_number(item_text)
    caption = ""
    if item.table_metadata and item.table_metadata.get("caption"):
        caption = str(item.table_metadata["caption"])
    section = item.section_heading or ""
    heading_blob = " ".join([caption, section, *item.heading_path]).lower()

    for sentence in _sentences_citing_token(answer_text, token):
        if "(calculated" in sentence.lower():
            continue
        for segment in _segments_preceding_token(sentence, token):
            masked = _DATE_MASK_PATTERN.sub(" ", segment)
            for match in _NUMBER_PATTERN.finditer(masked):
                raw_num = match.group(0).strip()
                norm_num = _normalize_number(raw_num)
                if not norm_num or _YEAR_PATTERN.match(norm_num):
                    continue
                if not _is_headline_number(raw_num, segment):
                    continue
                if norm_num not in item_norm:
                    warnings.append(
                        f"[{token}] number {raw_num} cited but not found in evidence text "
                        f"(chunk {item.chunk_id}, pages {item.page_numbers})"
                    )
                    continue
                # Locate the line carrying the number to identify its row label
                label = ""
                line_kind = "sentence"
                for line in item_text.split("\n"):
                    if norm_num in _normalize_number(line):
                        stripped = line.strip()
                        if stripped.startswith("|"):
                            cells = [c.strip() for c in stripped.strip("|").split("|")]
                            label = cells[0] if cells else ""
                            line_kind = "row"
                        elif stripped.startswith("[Table:") or TOTAL_ROW_PATTERN.match(stripped):
                            label = stripped[:80]
                            line_kind = "caption"
                        else:
                            label = stripped[:160]
                            line_kind = "sentence"
                        break
                if line_kind == "caption":
                    warnings.append(
                        f"[{token}] number {raw_num} appears only in a table caption/Total line, "
                        f"not in a labeled data row (chunk {item.chunk_id})"
                    )
                    continue
                label_words = set(re.findall(r"[A-Za-z]+", label.lower()))
                if not (label_words & keywords) and not any(k in heading_blob for k in keywords):
                    warnings.append(
                        f"[{token}] number {raw_num} sits under label '{label[:60]}' sharing no "
                        f"subject keyword with the question (chunk {item.chunk_id})"
                    )
    return warnings


def resolve_citation_heading(item: EvidenceItem) -> str | None:
    """
    Resolve the heading nearest above the cited content within a chunk.

    A chunk's ``section_heading`` reflects the active section when the chunk
    was assembled, but chunk text can contain a later heading-like line (a
    schedule/note title the heading detector classified as narrative, or a
    section label inside accumulated prose). Citing the chunk's first heading
    in that case misattributes the number, so the nearest heading-like line
    above the content wins. Table rows, Total summary rows, table labels, and
    footnotes are never headings. Falls back to the chunk section heading.
    """
    fallback = item.section_heading or (item.heading_path[-1] if item.heading_path else None)
    last_heading: str | None = None

    for raw_line in item.text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("|", "[", "-", "*")):
            continue
        if TOTAL_ROW_PATTERN.match(line):
            continue
        # Structural footnote labels are not subject headings
        if re.match(r"(?i)^footnotes?\s*:", line):
            continue
        if is_schedule_boundary_heading(line):
            last_heading = line
            continue
        if len(line) <= 80 and (line.endswith(":") or line.isupper()):
            last_heading = line

    return last_heading or fallback


# ── Domain Models ───────────────────────────────────────────────────────────


class Citation(BaseModel):
    """
    Audit-grounded source citation resolved from an evidence token ([EX]).

    Directly links generated statements to physical document coordinates.
    """

    model_config = ConfigDict(frozen=True)

    citation_number: int = Field(
        ...,
        ge=1,
        description="1-indexed footnote index (e.g. 1 for [1])",
    )
    evidence_id: str = Field(
        ...,
        description="Context-local evidence tag (e.g. 'E1')",
    )
    chunk_id: str = Field(
        ...,
        description="Deterministic source chunk identifier",
    )
    document_name: str = Field(
        ...,
        description="Source document filename",
    )
    page_numbers: list[int] = Field(
        default_factory=list,
        description="Physical 1-indexed PDF page numbers",
    )
    section_heading: str | None = Field(
        default=None,
        description="Immediate section heading enclosing chunk",
    )
    heading_path: list[str] = Field(
        default_factory=list,
        description="Section hierarchy ancestry breadcrumbs",
    )
    source_element_ids: list[str] = Field(
        default_factory=list,
        description="IDs of source DocumentElements",
    )
    chunk_type: str = Field(
        default="narrative",
        description="Chunk classification: narrative, table, list, mixed",
    )
    snippet: str = Field(
        default="",
        description="Brief excerpt of source chunk text",
    )

    def format_citation(self) -> str:
        """
        Format citation into readable audit string: [Document, Page(s), Section].

        Conforms to project citation standards documented in AGENTS.md Section 4.5.
        """
        if self.page_numbers:
            pages_str = ", ".join(f"Page {p}" for p in self.page_numbers)
        else:
            pages_str = "Page: Unknown"

        sec_str = f", {self.section_heading}" if self.section_heading else ""
        return f"[{self.document_name}, {pages_str}{sec_str}]"


ClaimVerificationStatus = Literal["verified", "partially_supported", "unverified"]


class ClaimVerification(BaseModel):
    """Verification outcome for one structured claim and its evidence set."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(..., description="Claim text that was verified")
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence IDs the claim cited",
    )
    status: ClaimVerificationStatus = Field(
        ...,
        description="verified, partially_supported, or unverified",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable failure reasons; empty when verified",
    )


class VerifiedAnswer(BaseModel):
    """
    Diligence answer with verified source citations and resolved footnote markers.
    """

    model_config = ConfigDict(frozen=True)

    answer_text: str = Field(
        ...,
        description="Resolved answer text with footnote markers ([1], [2])",
    )
    raw_answer_text: str = Field(
        ...,
        description="Original answer text with evidence tokens ([E1], [E2])",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Verified citations mapped to evidence items",
    )
    unverified_citations: list[str] = Field(
        default_factory=list,
        description="Invalid or hallucinated citation tokens not found in evidence context",
    )
    is_uncertain: bool = Field(
        default=False,
        description="Whether answer indicates uncertainty or missing disclosure",
    )
    grounding_warnings: list[str] = Field(
        default_factory=list,
        description="Headline-figure grounding tripwire warnings (advisory only, never fail verification)",
    )
    claims: list[Claim] = Field(
        default_factory=list,
        description="Structured claims supplied by B7",
    )
    claim_verifications: list[ClaimVerification] = Field(
        default_factory=list,
        description="Per-claim verification outcomes",
    )


# ── Citation Verifier Component ─────────────────────────────────────────────


class CitationVerifier:
    """
    Verification and attribution engine for Phase B8.

    Validates citations against the EvidenceContext lookup map, assigns
    deterministic 1-indexed footnote numbers, replaces evidence tokens with
    clean numeric markers, and detects any hallucinated tokens.
    """

    def __init__(self, replace_tokens_with_numbers: bool = True) -> None:
        """
        Initialize the citation verifier.

        Args:
            replace_tokens_with_numbers: Whether to replace [EX] tokens in the answer
                                         text with numeric footnotes [1], [2].
        """
        self._replace_tokens_with_numbers = replace_tokens_with_numbers

    @property
    def replace_tokens_with_numbers(self) -> bool:
        """Whether token replacement with numbers is enabled."""
        return self._replace_tokens_with_numbers

    def verify(
        self,
        grounded_answer: GroundedAnswer | str | dict[str, Any],
        context: EvidenceContext,
    ) -> VerifiedAnswer:
        """
        Verify citations emitted by GroundedGenerator against EvidenceContext.

        Args:
            grounded_answer: GroundedAnswer instance, raw string, or dictionary.
            context: The EvidenceContext emitted by EvidenceContextBuilder (Phase B6).

        Returns:
            VerifiedAnswer containing the resolved text, structured citations,
            and any flagged unverified tokens.
        """
        # Unpack answer
        claims: list[Claim] = []
        if isinstance(grounded_answer, GroundedAnswer):
            raw_text = grounded_answer.answer_text
            is_uncertain = grounded_answer.is_uncertain
            explicit_cites = list(grounded_answer.cited_evidence_ids)
            claims = list(grounded_answer.claims)
        elif isinstance(grounded_answer, dict):
            raw_text = str(grounded_answer.get("answer_text") or grounded_answer.get("answer") or "")
            is_uncertain = bool(grounded_answer.get("is_uncertain", False))
            raw_cites = grounded_answer.get("cited_evidence_ids") or grounded_answer.get("citations") or []
            explicit_cites = [str(c).strip().strip("[]") for c in raw_cites if str(c).strip()]
        elif isinstance(grounded_answer, str):
            raw_text = grounded_answer
            lower_text = raw_text.lower()
            is_uncertain = (
                "not disclosed in the ingested document" in lower_text
                or "does not contain sufficient information" in lower_text
            )
            explicit_cites = []
        else:
            raise TypeError(
                f"Expected grounded_answer to be GroundedAnswer, str, or dict, got {type(grounded_answer).__name__}."
            )

        # Fast path: Empty or whitespace-only answer
        if not raw_text or not raw_text.strip():
            logger.info("Answer text is empty. Returning verified answer with empty citations.")
            return VerifiedAnswer(
                answer_text=raw_text,
                raw_answer_text=raw_text,
                citations=[],
                unverified_citations=[],
                is_uncertain=True,
                claims=claims,
            )

        evidence_map: dict[str, EvidenceItem] = context.evidence_map

        # Step 1/2: Determine candidate order and per-token support failures.
        # Structured claims are verified as a set: a claim's facts may be spread
        # across several cited chunks, but every chunk must contribute.
        token_failures: dict[str, list[str]] = {}
        claim_verifications: list[ClaimVerification] = []
        if claims:
            ordered_candidates: list[str] = []
            seen: set[str] = set()
            for claim in claims:
                for token in claim.evidence_ids:
                    clean_token = token.strip().strip("[]")
                    if clean_token and clean_token not in seen:
                        seen.add(clean_token)
                        ordered_candidates.append(clean_token)

            for claim in claims:
                if not claim.evidence_ids:
                    claim_verifications.append(
                        ClaimVerification(
                            text=claim.text,
                            evidence_ids=[],
                            status="unverified",
                            reasons=["claim cites no evidence IDs."],
                        )
                    )
                    continue

                resolved_items = [
                    evidence_map[token]
                    for token in claim.evidence_ids
                    if token in evidence_map
                ]
                claim_failures, token_specific = check_claim_evidence_support(
                    claim.text,
                    claim.evidence_ids,
                    resolved_items,
                )

                failed_tokens: list[str] = []
                for token in claim.evidence_ids:
                    reasons = list(claim_failures)
                    reasons.extend(token_specific.get(token, []))
                    if reasons:
                        failed_tokens.append(token)
                        token_failures.setdefault(token, []).extend(reasons)

                if not failed_tokens:
                    status: ClaimVerificationStatus = "verified"
                elif len(failed_tokens) == len(claim.evidence_ids):
                    status = "unverified"
                else:
                    status = "partially_supported"

                claim_reasons = list(claim_failures)
                for token in claim.evidence_ids:
                    claim_reasons.extend(token_specific.get(token, []))
                claim_verifications.append(
                    ClaimVerification(
                        text=claim.text,
                        evidence_ids=list(claim.evidence_ids),
                        status=status,
                        reasons=list(dict.fromkeys(claim_reasons)),
                    )
                )
        else:
            # Backward-compatible free-form path: scan answer_text for tokens.
            text_tokens: list[str] = []
            for match in re.finditer(r"\[([^\]]+)\]", raw_text):
                inner = match.group(1).strip()
                # Split comma-separated tokens inside brackets (e.g. "E1, E2")
                parts = [p.strip() for p in re.split(r"[,;]", inner) if p.strip()]
                for part in parts:
                    if re.fullmatch(r"E\d+|[eE][xX]|[eE][a-zA-Z]{1,4}|[eE][_\-]\d+|[eE]", part):
                        text_tokens.append(part)

            ordered_candidates = []
            seen = set()
            for token in text_tokens + explicit_cites:
                clean_token = token.strip().strip("[]")
                if clean_token and clean_token not in seen:
                    seen.add(clean_token)
                    ordered_candidates.append(clean_token)

            for token in ordered_candidates:
                item = evidence_map.get(token)
                if item is None:
                    token_failures[token] = [
                        f"[{token}] cited by generator but absent from EvidenceContext."
                    ]
                else:
                    token_failures[token] = check_claim_support(raw_text, token, item)

        # Step 3: Partition candidates into verified and unverified
        verified_citations: list[Citation] = []
        unverified_citations: list[str] = []
        all_grounding_warnings: list[str] = []
        token_to_number: dict[str, int] = {}

        for token in ordered_candidates:
            item = evidence_map.get(token)
            support_failures = token_failures.get(token, [])
            if item is None or support_failures:
                unverified_citations.append(token)
                for reason in support_failures:
                    logger.warning("Claim support: %s", reason)
                    all_grounding_warnings.append(reason)
                continue

            cit_num = len(verified_citations) + 1
            token_to_number[token] = cit_num

            # Build excerpt snippet
            clean_excerpt = item.text.strip().replace("\n", " ")
            if len(clean_excerpt) > 200:
                clean_excerpt = clean_excerpt[:197] + "..."

            # Headline-figure grounding tripwire (advisory warnings only)
            question = context.original_query or context.retrieval_query
            if claims:
                for claim in claims:
                    if token in claim.evidence_ids:
                        grounding_notes = check_numeric_grounding(
                            _claim_with_token(claim.text, token),
                            token,
                            item,
                            question,
                        )
                        for note in grounding_notes:
                            logger.warning("Numeric grounding: %s", note)
                            all_grounding_warnings.append(note)
            else:
                grounding_notes = check_numeric_grounding(raw_text, token, item, question)
                for note in grounding_notes:
                    logger.warning("Numeric grounding: %s", note)
                    all_grounding_warnings.append(note)

            citation = Citation(
                citation_number=cit_num,
                evidence_id=token,
                chunk_id=item.chunk_id,
                document_name=item.document_name,
                page_numbers=list(item.page_numbers),
                section_heading=resolve_citation_heading(item),
                heading_path=list(item.heading_path),
                source_element_ids=list(item.source_element_ids),
                chunk_type=item.chunk_type,
                snippet=clean_excerpt,
            )
            verified_citations.append(citation)

        all_grounding_warnings = list(dict.fromkeys(all_grounding_warnings))

        # Step 4: Resolve text markers
        resolved_text = raw_text
        if self._replace_tokens_with_numbers:
            # First replace any grouped brackets like [E1, E2]
            def _replace_grouped_bracket(m: re.Match[str]) -> str:
                inner = m.group(1).strip()
                if "," not in inner and ";" not in inner:
                    return m.group(0)
                sub_parts = [p.strip() for p in re.split(r"[,;]", inner) if p.strip()]
                new_parts: list[str] = []
                for p in sub_parts:
                    if p in token_to_number:
                        new_parts.append(f"[{token_to_number[p]}]")
                    elif p in unverified_citations:
                        new_parts.append(f"[Unverified: {p}]")
                    else:
                        new_parts.append(p)
                return " ".join(new_parts)

            resolved_text = re.sub(r"\[([^\]]+)\]", _replace_grouped_bracket, resolved_text)

            # Replace individual verified tokens [EX] with [N]
            for token, number in token_to_number.items():
                resolved_text = re.sub(
                    rf"\[{re.escape(token)}\]",
                    f"[{number}]",
                    resolved_text,
                )
            # Flag unverified tokens in the text explicitly
            for token in unverified_citations:
                resolved_text = re.sub(
                    rf"\[{re.escape(token)}\]",
                    f"[Unverified: {token}]",
                    resolved_text,
                )

        return VerifiedAnswer(
            answer_text=resolved_text,
            raw_answer_text=raw_text,
            citations=verified_citations,
            unverified_citations=unverified_citations,
            is_uncertain=is_uncertain,
            grounding_warnings=all_grounding_warnings,
            claims=claims,
            claim_verifications=claim_verifications,
        )


# Backward-compatible alias
CitationEngine = CitationVerifier


# ── Functional Convenience Wrapper ──────────────────────────────────────────


def verify_citations(
    grounded_answer: GroundedAnswer | str | dict[str, Any],
    context: EvidenceContext,
    replace_tokens_with_numbers: bool = True,
) -> VerifiedAnswer:
    """
    Functional convenience wrapper for citation verification.

    Instantiates a temporary ``CitationVerifier`` and executes verification.
    """
    verifier = CitationVerifier(replace_tokens_with_numbers=replace_tokens_with_numbers)
    return verifier.verify(grounded_answer, context)
