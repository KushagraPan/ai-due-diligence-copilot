"""
Unit tests for retrieval schema models: IndexingPayload and RetrievalResult.

Validates construction, factory method behavior, serialization round-tripping,
frozen immutability, and edge cases (missing metadata, empty fields).
"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from due_diligence.ingestion.chunks import Chunk
from due_diligence.ingestion.document import DocumentType, ValidatedDocument
from due_diligence.retrieval.schemas import IndexingPayload, RetrievalResult

# ── Fixtures ────────────────────────────────────────────────────────────


def _make_chunk(
    *,
    chunk_id: str = "hdfc_c00012_a1b2",
    text: str = "Total deposits were ₹23,45,600 Cr as of March 31, 2026.",
    chunk_type: str = "narrative",
    page_numbers: list[int] | None = None,
    section_heading: str | None = "Balance Sheet",
    heading_path: list[str] | None = None,
    source_element_ids: list[str] | None = None,
    overlap_tokens: int = 0,
    table_metadata: dict[str, Any] | None = None,
) -> Chunk:
    """Build a test Chunk with sensible defaults."""
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        chunk_type=chunk_type,  # type: ignore[arg-type]
        page_numbers=page_numbers if page_numbers is not None else [378],
        section_heading=section_heading,
        heading_path=heading_path if heading_path is not None else ["Financial Statements", "Balance Sheet"],
        source_element_ids=source_element_ids if source_element_ids is not None else ["elem_001", "elem_002"],
        overlap_tokens=overlap_tokens,
        table_metadata=table_metadata,
    )


def _make_document(
    *,
    document_name: str = "hdfc-bank__annual-report__fy2025-26.pdf",
    company: str | None = "HDFC Bank",
    financial_period: str | None = "FY2025-26",
    content_hash: str | None = None,
) -> ValidatedDocument:
    """Build a test ValidatedDocument with sensible defaults."""
    return ValidatedDocument(
        document_name=document_name,
        document_type=DocumentType.PDF,
        file_path=Path("/data/uploads/hdfc-bank__annual-report__fy2025-26.pdf"),
        file_size_bytes=42_000_000,
        content_hash=content_hash or ("a" * 64),
        company=company,
        financial_period=financial_period,
    )


# ── IndexingPayload Tests ──────────────────────────────────────────────


class TestIndexingPayloadFactory:
    """Test the from_chunk_and_document factory method."""

    def test_creates_from_narrative_chunk(self) -> None:
        chunk = _make_chunk()
        doc = _make_document()
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        assert payload.chunk_id == "hdfc_c00012_a1b2"
        assert payload.text == chunk.text
        assert payload.chunk_type == "narrative"
        assert payload.page_numbers == [378]
        assert payload.section_heading == "Balance Sheet"
        assert payload.heading_path == ["Financial Statements", "Balance Sheet"]
        assert payload.source_element_ids == ["elem_001", "elem_002"]
        assert payload.company == "HDFC Bank"
        assert payload.document_name == "hdfc-bank__annual-report__fy2025-26.pdf"
        assert payload.document_type == "pdf"
        assert payload.financial_period == "FY2025-26"
        assert payload.content_hash == "a" * 64

    def test_narrative_chunk_sets_is_table_false(self) -> None:
        chunk = _make_chunk(chunk_type="narrative")
        doc = _make_document()
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        assert payload.is_table is False
        assert payload.table_headers == []
        assert payload.table_caption is None

    def test_table_chunk_unpacks_metadata(self) -> None:
        table_meta = {
            "table_group_id": "tbl_001",
            "table_slice_index": 0,
            "table_slice_count": 2,
            "headers": ["Schedule", "FY2026 (₹ Cr)", "FY2025 (₹ Cr)"],
            "caption": "Schedule 3: Deposits",
            "footnotes": [],
        }
        chunk = _make_chunk(chunk_type="table", table_metadata=table_meta)
        doc = _make_document()
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        assert payload.is_table is True
        assert payload.table_headers == ["Schedule", "FY2026 (₹ Cr)", "FY2025 (₹ Cr)"]
        assert payload.table_caption == "Schedule 3: Deposits"
        assert payload.table_metadata == table_meta

    def test_table_chunk_with_empty_headers(self) -> None:
        table_meta: dict[str, Any] = {
            "table_group_id": "tbl_002",
            "headers": [],
            "caption": None,
        }
        chunk = _make_chunk(chunk_type="table", table_metadata=table_meta)
        doc = _make_document()
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        assert payload.is_table is True
        assert payload.table_headers == []
        assert payload.table_caption is None

    def test_table_chunk_with_no_headers_key(self) -> None:
        """Table metadata missing the 'headers' key should produce empty list."""
        table_meta = {"table_group_id": "tbl_003"}
        chunk = _make_chunk(chunk_type="table", table_metadata=table_meta)
        doc = _make_document()
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        assert payload.table_headers == []

    def test_document_with_no_company_defaults_to_empty_string(self) -> None:
        """ValidatedDocument.company=None should become empty string on payload."""
        chunk = _make_chunk()
        doc = _make_document(company=None)
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        assert payload.company == ""

    def test_document_with_no_financial_period(self) -> None:
        chunk = _make_chunk()
        doc = _make_document(financial_period=None)
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        assert payload.financial_period is None

    def test_multiple_page_numbers_preserved(self) -> None:
        chunk = _make_chunk(page_numbers=[42, 43, 44])
        doc = _make_document()
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        assert payload.page_numbers == [42, 43, 44]

    def test_heading_path_empty_preserved(self) -> None:
        chunk = _make_chunk(heading_path=[])
        doc = _make_document()
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        assert payload.heading_path == []

    def test_mixed_chunk_type(self) -> None:
        chunk = _make_chunk(chunk_type="mixed")
        doc = _make_document()
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        assert payload.chunk_type == "mixed"
        assert payload.is_table is False

    def test_list_chunk_type(self) -> None:
        chunk = _make_chunk(chunk_type="list")
        doc = _make_document()
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        assert payload.chunk_type == "list"
        assert payload.is_table is False


class TestIndexingPayloadSerialization:
    """Test JSON serialization and deserialization round-trips."""

    def test_round_trip_narrative(self) -> None:
        chunk = _make_chunk()
        doc = _make_document()
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        data = payload.model_dump()
        restored = IndexingPayload.model_validate(data)
        assert restored == payload

    def test_round_trip_table(self) -> None:
        table_meta = {
            "table_group_id": "tbl_001",
            "headers": ["Col A", "Col B"],
            "caption": "Test Table",
        }
        chunk = _make_chunk(chunk_type="table", table_metadata=table_meta)
        doc = _make_document()
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        json_str = payload.model_dump_json()
        restored = IndexingPayload.model_validate_json(json_str)
        assert restored == payload

    def test_model_dump_contains_all_fields(self) -> None:
        chunk = _make_chunk()
        doc = _make_document()
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)
        data = payload.model_dump()

        expected_keys = {
            "chunk_id", "text", "chunk_type", "page_numbers",
            "section_heading", "heading_path", "source_element_ids",
            "table_metadata", "company", "document_name", "document_type",
            "financial_period", "content_hash", "is_table",
            "table_headers", "table_caption", "entity_scope",
        }
        assert set(data.keys()) == expected_keys


class TestIndexingPayloadImmutability:
    """Verify that IndexingPayload instances are frozen."""

    def test_cannot_mutate_text(self) -> None:
        chunk = _make_chunk()
        doc = _make_document()
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        with pytest.raises(ValidationError):
            payload.text = "mutated text"

    def test_cannot_mutate_company(self) -> None:
        chunk = _make_chunk()
        doc = _make_document()
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        with pytest.raises(ValidationError):
            payload.company = "Other Bank"


class TestIndexingPayloadValidation:
    """Test that invalid inputs are rejected."""

    def test_missing_chunk_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            IndexingPayload(
                text="some text",
                chunk_type="narrative",
                page_numbers=[1],
                source_element_ids=["e1"],
                company="Test",
                document_name="test.pdf",
                document_type="pdf",
                content_hash="a" * 64,
            )  # type: ignore[call-arg]

    def test_empty_page_numbers_raises(self) -> None:
        with pytest.raises(ValidationError):
            IndexingPayload(
                chunk_id="test_001",
                text="some text",
                chunk_type="narrative",
                page_numbers=[],
                source_element_ids=["e1"],
                company="Test",
                document_name="test.pdf",
                document_type="pdf",
                content_hash="a" * 64,
            )

    def test_empty_source_element_ids_raises(self) -> None:
        with pytest.raises(ValidationError):
            IndexingPayload(
                chunk_id="test_001",
                text="some text",
                chunk_type="narrative",
                page_numbers=[1],
                source_element_ids=[],
                company="Test",
                document_name="test.pdf",
                document_type="pdf",
                content_hash="a" * 64,
            )


# ── RetrievalResult Tests ──────────────────────────────────────────────


def _make_retrieval_result(**overrides: Any) -> RetrievalResult:
    """Build a test RetrievalResult with sensible defaults."""
    defaults: dict[str, Any] = {
        "chunk_id": "hdfc_c00012_a1b2",
        "text": "Total deposits were ₹23,45,600 Cr.",
        "score": 0.87,
        "company": "HDFC Bank",
        "document_name": "hdfc-bank__annual-report__fy2025-26.pdf",
        "document_type": "pdf",
        "financial_period": "FY2025-26",
        "page_numbers": [378],
        "section_heading": "Balance Sheet",
        "heading_path": ["Financial Statements", "Balance Sheet"],
        "chunk_type": "narrative",
        "source_element_ids": ["elem_001"],
    }
    defaults.update(overrides)
    return RetrievalResult(**defaults)


class TestRetrievalResultConstruction:
    """Test basic construction and field access."""

    def test_creates_with_all_fields(self) -> None:
        result = _make_retrieval_result()
        assert result.chunk_id == "hdfc_c00012_a1b2"
        assert result.score == pytest.approx(0.87)
        assert result.company == "HDFC Bank"
        assert result.page_numbers == [378]

    def test_financial_period_optional(self) -> None:
        result = _make_retrieval_result(financial_period=None)
        assert result.financial_period is None

    def test_section_heading_optional(self) -> None:
        result = _make_retrieval_result(section_heading=None)
        assert result.section_heading is None

    def test_heading_path_defaults_to_empty(self) -> None:
        result = RetrievalResult(
            chunk_id="test",
            text="test",
            score=0.5,
            company="Test Corp",
            document_name="test.pdf",
            document_type="pdf",
            page_numbers=[1],
            chunk_type="narrative",
            source_element_ids=["e1"],
        )
        assert result.heading_path == []

    def test_table_metadata_optional(self) -> None:
        result = _make_retrieval_result(table_metadata=None)
        assert result.table_metadata is None

    def test_table_metadata_with_values(self) -> None:
        meta = {"table_group_id": "tbl_001", "headers": ["A", "B"]}
        result = _make_retrieval_result(table_metadata=meta)
        assert result.table_metadata == meta


class TestRetrievalResultSerialization:
    """Test JSON serialization round-trips."""

    def test_round_trip(self) -> None:
        result = _make_retrieval_result()
        data = result.model_dump()
        restored = RetrievalResult.model_validate(data)
        assert restored == result

    def test_json_round_trip(self) -> None:
        result = _make_retrieval_result(
            table_metadata={"headers": ["X"], "caption": "Test"},
        )
        json_str = result.model_dump_json()
        restored = RetrievalResult.model_validate_json(json_str)
        assert restored == result

    def test_model_dump_keys(self) -> None:
        result = _make_retrieval_result()
        data = result.model_dump()
        expected_keys = {
            "chunk_id", "text", "score", "company", "document_name",
            "document_type", "financial_period", "page_numbers",
            "section_heading", "heading_path", "chunk_type",
            "source_element_ids", "table_metadata", "entity_scope",
        }
        assert set(data.keys()) == expected_keys


class TestRetrievalResultImmutability:
    """Verify that RetrievalResult instances are frozen."""

    def test_cannot_mutate_score(self) -> None:
        result = _make_retrieval_result()
        with pytest.raises(ValidationError):
            result.score = 0.99

    def test_cannot_mutate_text(self) -> None:
        result = _make_retrieval_result()
        with pytest.raises(ValidationError):
            result.text = "changed"


class TestRetrievalResultSorting:
    """Verify that results can be sorted by score for downstream use."""

    def test_sort_by_score_descending(self) -> None:
        results = [
            _make_retrieval_result(chunk_id="c1", score=0.50),
            _make_retrieval_result(chunk_id="c2", score=0.95),
            _make_retrieval_result(chunk_id="c3", score=0.72),
        ]
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        assert [r.chunk_id for r in sorted_results] == ["c2", "c3", "c1"]
