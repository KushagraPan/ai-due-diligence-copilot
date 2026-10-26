"""
Document element hierarchy representing structured content blocks.

1. Purpose:
   Defines the core document element object model (DocumentElement, NarrativeText,
   Heading, ListItem, Table) representing layout-aware, parsed content with rich
   spatial provenance, confidence scoring, and audit flags.

2. Role in Overall RAG Pipeline:
   Acts as the universal structured output layer for document parsing and table
   extraction. Downstream chunkers consume these typed elements to assemble
   context-aware chunks, preserving table boundaries, footnotes, and exact citations.

3. Expected Inputs:
   - Extracted text spans, bounding boxes, table matrices, headers, and metadata
     produced by PDF parsers and table extraction engines.

4. Outputs:
   - Strongly typed, validated Pydantic models (NarrativeText, Heading, ListItem, Table)
     with serialization and Markdown rendering capabilities.

5. Dependencies:
   - pydantic (BaseModel, Field, ConfigDict)
   - typing (Literal, Any)

6. Non-Responsibilities:
   - Does not perform PDF parsing or text extraction (handled in parser.py).
   - Does not execute table extraction or quality gating (handled in table_extractor.py).
   - Does not split elements into chunks or generate embeddings.
"""

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Table summary-row pattern: a "Total ..." line is a data row of the table above
# it, never a caption for the table below it.
TOTAL_ROW_PATTERN = re.compile(r"^\s*total\b", re.IGNORECASE)

# Financial-statement schedule/note headings always open a new statement section
# (e.g. "SCHEDULE 12 - CONTINGENT LIABILITIES", "Note 18: Related Party").
# Chunks must never merge content across such boundaries.
SCHEDULE_HEADING_PATTERN = re.compile(
    r"^\s*(SCHEDULE\s+[A-Z0-9]+|NOTE\s+\d+|NOTES\b|ANNEXURE\b|EXHIBIT\b)",
    re.IGNORECASE,
)


def is_total_row_label(label: str | None) -> bool:
    """Returns True when a table row label is a trailing summary (Total) row."""
    return bool(label and TOTAL_ROW_PATTERN.match(label))


def is_schedule_boundary_heading(text: str | None) -> bool:
    """Returns True when a heading opens a new schedule/note statement section."""
    return bool(text and SCHEDULE_HEADING_PATTERN.match(text))


class DocumentElement(BaseModel):
    """
    Abstract base model for all layout-aware document elements.

    Preserves 1-indexed page numbering, spatial bounding boxes, and document provenance.
    """

    model_config = ConfigDict(frozen=True)

    element_id: str = Field(..., description="Deterministic unique identifier for the element")
    element_type: Literal["narrative", "heading", "list_item", "table"] = Field(
        ..., description="Discriminator tag for element type"
    )
    page_number: int = Field(..., ge=1, description="1-indexed physical page number in source document")
    bbox: tuple[float, float, float, float] = Field(
        ..., description="Bounding box coordinates (x0, y0, x1, y1) in PDF points"
    )
    document_name: str = Field(..., description="Source document filename or title")
    section_heading: str | None = Field(
        default=None, description="Active section or chapter heading enclosing this element"
    )
    entity_scope: str = Field(
        default="hdfc_bank_general",
        description="Entity or statement scope (e.g. 'hdfc_bank_standalone', 'hdfc_bank_consolidated', 'hdb_financial_services')",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary extra metadata (e.g. font info, style tags)"
    )


class NarrativeText(DocumentElement):
    """Prose paragraph or body text block."""

    element_type: Literal["narrative"] = "narrative"
    text: str = Field(..., description="Cleaned narrative paragraph content")


class Heading(DocumentElement):
    """Document section, chapter, or sub-heading."""

    element_type: Literal["heading"] = "heading"
    text: str = Field(..., description="Heading label or title")
    level: int = Field(default=1, ge=1, le=6, description="Heading hierarchy level (1 to 6)")


class ListItem(DocumentElement):
    """Bullet point or numbered list item."""

    element_type: Literal["list_item"] = "list_item"
    text: str = Field(..., description="Text content of the list item")
    marker: str | None = Field(
        default=None, description="List bullet or numeral marker (e.g., '•', '-', '1.', '(a)')"
    )


class Table(DocumentElement):
    """
    Structured tabular data element.

    Preserves rows, columns, Markdown representation, attached footnotes,
    extraction method, confidence scoring, and quality review flags.
    """

    element_type: Literal["table"] = "table"
    headers: list[str] = Field(
        default_factory=list, description="Column header labels"
    )
    rows: list[list[str]] = Field(
        default_factory=list, description="Matrix of table body cell values"
    )
    markdown: str = Field(
        default="", description="Rendered GitHub-Flavored Markdown representation of the table"
    )
    caption: str | None = Field(
        default=None, description="Table title, caption, or schedule label"
    )
    footnotes: list[str] = Field(
        default_factory=list, description="Footnotes and explanatory notes attached to the table"
    )
    is_borderless: bool = Field(
        default=False, description="True if table lacked visible outer/inner borders"
    )
    is_continuation: bool = Field(
        default=False, description="True if table continues from the prior page"
    )
    continuation_from_page: int | None = Field(
        default=None, description="Predecessor page number if this table continues an unclosed statement"
    )
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence score (0.0 to 1.0)"
    )
    needs_review: bool = Field(
        default=False, description="True if quality checks flagged this table for human verification"
    )
    extraction_status: Literal["successful", "needs_review"] = Field(
        default="successful", description="Final operational status of the extraction"
    )
    extraction_method: Literal["tier1", "tier2_gemini", "tier3"] = Field(
        default="tier1", description="Extraction engine pipeline tier that produced this element"
    )
    review_reasons: list[str] = Field(
        default_factory=list, description="Diagnostic tags explaining quality gate failures or warnings"
    )

    @property
    def row_count(self) -> int:
        """Total number of data rows in the table."""
        return len(self.rows)

    @property
    def col_count(self) -> int:
        """Number of columns in the table (based on headers or widest row)."""
        if self.headers:
            return len(self.headers)
        if self.rows:
            return max(len(r) for r in self.rows)
        return 0

    @staticmethod
    def render_markdown(headers: list[str], rows: list[list[str]]) -> str:
        """
        Formats header and data rows into standard GitHub-Flavored Markdown.
        Escapes internal pipe characters and aligns column widths.
        """
        if not headers and not rows:
            return ""

        num_cols = len(headers) if headers else (max(len(r) for r in rows) if rows else 0)
        if num_cols == 0:
            return ""

        norm_headers = [h.replace("|", "\\|").replace("\n", " ").strip() for h in headers]
        while len(norm_headers) < num_cols:
            norm_headers.append(f"Col {len(norm_headers) + 1}")

        norm_rows = []
        for r in rows:
            norm_r = [c.replace("|", "\\|").replace("\n", " ").strip() for c in r]
            while len(norm_r) < num_cols:
                norm_r.append("")
            norm_rows.append(norm_r[:num_cols])

        # Compute column widths
        col_widths = [len(norm_headers[c]) for c in range(num_cols)]
        for r in norm_rows:
            for c in range(num_cols):
                col_widths[c] = max(col_widths[c], len(r[c]))
        col_widths = [max(w, 3) for w in col_widths]

        header_line = "| " + " | ".join(norm_headers[c].ljust(col_widths[c]) for c in range(num_cols)) + " |"
        sep_line = "|-" + "-|-".join("-" * col_widths[c] for c in range(num_cols)) + "-|"
        row_lines = [
            "| " + " | ".join(r[c].ljust(col_widths[c]) for c in range(num_cols)) + " |"
            for r in norm_rows
        ]

        return "\n".join([header_line, sep_line] + row_lines)
