"""
Ingestion subsystem package.

1. Purpose: Exposes document loading, validation, metadata schemas, and PDF page text parsing.
2. Pipeline Role: Public entry point for ingesting raw diligence documents into the RAG pipeline.
3. Expected Inputs: Raw file references, metadata, and validated document objects.
4. Outputs: ValidatedDocument and ExtractedDocument instances with 1-indexed page text.
5. Dependencies: `document`, `loaders`, `parser`, and `font_mappings` modules within this package.
6. Outside Scope: Semantic chunking, table extraction, and vector indexing.
"""

from due_diligence.ingestion.document import (
    DocumentType,
    ExtractedDocument,
    ExtractedPage,
    ValidatedDocument,
)
from due_diligence.ingestion.elements import (
    DocumentElement,
    Heading,
    ListItem,
    NarrativeText,
    Table,
)
from due_diligence.ingestion.font_mappings import (
    DEFAULT_CONFIG_PATH,
    clear_symbol_mappings_cache,
    format_unmapped_placeholder,
    get_symbol_font_category,
    is_symbol_font,
    load_symbol_mappings,
    strip_subset_prefix,
)
from due_diligence.ingestion.loaders import (
    CorruptedDocumentError,
    DocumentLoader,
    DocumentLoadError,
    DocumentNotFoundError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)
from due_diligence.ingestion.parser import (
    NIL_NOT_DISCLOSED_MARKER,
    CorruptedPDFParseError,
    DocumentParseError,
    EncryptedDocumentError,
    PDFParser,
    extract_page_text_with_font_remapping,
    normalize_standalone_nil_dashes,
)
from due_diligence.ingestion.preflight import (
    PreflightReviewRequiredError,
    assert_preflight_clean,
    partition_preflight_batch,
    preflight_scan,
    save_preflight_report,
)
from due_diligence.ingestion.table_extractor import (
    detect_all_table_regions,
    evaluate_quality_gate,
    extract_table_tier1,
    extract_table_tier2_gemini,
    extract_tables_from_page,
    stitch_continuation_tables,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "NIL_NOT_DISCLOSED_MARKER",
    "CorruptedDocumentError",
    "CorruptedPDFParseError",
    "DocumentElement",
    "DocumentLoadError",
    "DocumentLoader",
    "DocumentNotFoundError",
    "DocumentParseError",
    "DocumentType",
    "EmptyDocumentError",
    "EncryptedDocumentError",
    "ExtractedDocument",
    "ExtractedPage",
    "Heading",
    "ListItem",
    "NarrativeText",
    "PDFParser",
    "PreflightReviewRequiredError",
    "Table",
    "UnsupportedDocumentTypeError",
    "ValidatedDocument",
    "assert_preflight_clean",
    "clear_symbol_mappings_cache",
    "detect_all_table_regions",
    "evaluate_quality_gate",
    "extract_page_text_with_font_remapping",
    "extract_table_tier1",
    "extract_table_tier2_gemini",
    "extract_tables_from_page",
    "format_unmapped_placeholder",
    "get_symbol_font_category",
    "is_symbol_font",
    "load_symbol_mappings",
    "normalize_standalone_nil_dashes",
    "partition_preflight_batch",
    "preflight_scan",
    "save_preflight_report",
    "stitch_continuation_tables",
    "strip_subset_prefix",
]
