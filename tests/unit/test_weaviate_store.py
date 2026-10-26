"""
Unit tests for Weaviate Cloud vector store connector and batch indexer.

Verifies:
1. Invariant pre-flight checks (validate_pre_indexing_artifacts) on chunks, embeddings,
   and manifest files.
2. Property builder (build_weaviate_properties) mapping and table provenance unpacking.
3. Connection lifecycle and context manager semantics in WeaviateStore.
4. Collection management (ensure_collection) with BYOV Vectorizer.none, HNSW cosine,
   BM25 index, and safe default recreate=False.
5. Batch indexing (index_canonical_corpus) with deterministic UUID5, fixed-size batching,
   and fail-loud failed_objects verification.
6. Post-indexing verification (verify_collection_health) with aggregate checks and probe inspections.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from due_diligence.core.config import Settings
from due_diligence.ingestion.chunks import Chunk
from due_diligence.ingestion.document import DocumentType, ValidatedDocument
from due_diligence.retrieval.schemas import IndexingPayload
from due_diligence.retrieval.weaviate_store import (
    DEFAULT_COLLECTION_NAME,
    WeaviateAlignmentError,
    WeaviateBatchIndexError,
    WeaviateConnectionError,
    WeaviateSchemaError,
    WeaviateStore,
    WeaviateVerificationError,
    build_weaviate_properties,
    validate_pre_indexing_artifacts,
)

# ── Helpers & Test Doubles ──────────────────────────────────────────────────


def _make_test_chunk(
    chunk_id: str,
    text: str = "Test corporate disclosure text.",
    chunk_type: str = "narrative",
    page_numbers: list[int] | None = None,
    section_heading: str | None = "Financial Overview",
    table_metadata: dict[str, Any] | None = None,
) -> Chunk:
    """Build a valid Chunk instance for testing."""
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        chunk_type=chunk_type,  # type: ignore[arg-type]
        page_numbers=page_numbers if page_numbers is not None else [10],
        section_heading=section_heading,
        heading_path=["Item 7", "MD&A"],
        source_element_ids=["elem_01", "elem_02"],
        overlap_tokens=0,
        table_metadata=table_metadata,
    )


def _make_test_document() -> ValidatedDocument:
    """Build a valid ValidatedDocument instance for testing."""
    return ValidatedDocument(
        file_path=Path("tests/data/sample.pdf"),
        company="HDFC Bank",
        document_type=DocumentType.PDF,
        financial_period="FY2025-26",
        document_name="hdfc-bank__annual-report__fy2025-26.pdf",
        content_hash="a" * 64,
        file_size_bytes=1024,
    )


def _create_test_artifacts(
    tmp_path: Path,
    num_chunks: int = 5,
    dimension: int = 1536,
    scramble_manifest: bool = False,
    missing_manifest_id: bool = False,
    include_nan: bool = False,
    include_inf: bool = False,
    duplicate_chunk_id: bool = False,
) -> tuple[Path, Path, Path]:
    """Helper to generate aligned test chunks, embeddings, and manifest on disk."""
    chunks: list[Chunk] = []
    chunk_id_to_index: dict[str, int] = {}

    for i in range(num_chunks):
        cid = f"chunk_{i:04d}"
        if duplicate_chunk_id and i == num_chunks - 1:
            cid = "chunk_0000"
        chunks.append(_make_test_chunk(cid, text=f"Chunk {i} text."))
        chunk_id_to_index[cid] = i

    if scramble_manifest and num_chunks >= 2:
        # Swap indices 0 and 1
        chunk_id_to_index["chunk_0000"] = 1
        chunk_id_to_index["chunk_0001"] = 0

    if missing_manifest_id and num_chunks >= 1:
        del chunk_id_to_index["chunk_0000"]

    chunks_path = tmp_path / "test_chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        f.writelines(c.model_dump_json() + "\n" for c in chunks)

    vectors = np.ones((num_chunks, dimension), dtype=np.float32)
    if include_nan:
        vectors[0, 0] = np.nan
    if include_inf:
        vectors[0, 0] = np.inf

    embeddings_path = tmp_path / "test_embeddings.npy"
    np.save(embeddings_path, vectors)

    manifest_path = tmp_path / "test_manifest.json"
    manifest = {
        "document_name": "sample.pdf",
        "model_name": "text-embedding-3-large",
        "dimension": dimension,
        "chunk_count": num_chunks,
        "chunk_id_to_index": chunk_id_to_index,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return chunks_path, embeddings_path, manifest_path


# ── Invariant Validation Tests ──────────────────────────────────────────────


class TestValidatePreIndexingArtifacts:
    """Verify all 10 pre-flight invariant checks in validate_pre_indexing_artifacts."""

    def test_all_invariants_pass_cleanly(self, tmp_path: Path) -> None:
        c_path, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=5, dimension=1536)
        chunks, embeddings, manifest = validate_pre_indexing_artifacts(
            chunks_path=c_path,
            embeddings_path=e_path,
            manifest_path=m_path,
            expected_dimension=1536,
        )
        assert len(chunks) == 5
        assert embeddings.shape == (5, 1536)
        assert manifest["chunk_count"] == 5

    def test_invariant_1_missing_chunks_file(self, tmp_path: Path) -> None:
        _, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=2)
        bad_chunks_path = tmp_path / "nonexistent.jsonl"

        with pytest.raises(WeaviateAlignmentError, match="Invariant 1 failed: Chunks file not found"):
            validate_pre_indexing_artifacts(bad_chunks_path, e_path, m_path, 1536)

    def test_invariant_2_missing_embeddings_file(self, tmp_path: Path) -> None:
        c_path, _, m_path = _create_test_artifacts(tmp_path, num_chunks=2)
        bad_e_path = tmp_path / "nonexistent.npy"

        with pytest.raises(WeaviateAlignmentError, match="Invariant 2 failed: Embeddings file not found"):
            validate_pre_indexing_artifacts(c_path, bad_e_path, m_path, 1536)

    def test_invariant_3_missing_manifest_file(self, tmp_path: Path) -> None:
        c_path, e_path, _ = _create_test_artifacts(tmp_path, num_chunks=2)
        bad_m_path = tmp_path / "nonexistent.json"

        with pytest.raises(WeaviateAlignmentError, match="Invariant 3 failed: Manifest file not found"):
            validate_pre_indexing_artifacts(c_path, e_path, bad_m_path, 1536)

    def test_corrupted_chunks_file_raises(self, tmp_path: Path) -> None:
        c_path, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=2)
        c_path.write_text("{bad json line\n", encoding="utf-8")

        with pytest.raises(WeaviateAlignmentError, match="Failed to read/parse chunks file"):
            validate_pre_indexing_artifacts(c_path, e_path, m_path, 1536)

    def test_corrupted_embeddings_file_raises(self, tmp_path: Path) -> None:
        c_path, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=2)
        e_path.write_bytes(b"not a valid npy header")

        with pytest.raises(WeaviateAlignmentError, match="Failed to load embeddings array"):
            validate_pre_indexing_artifacts(c_path, e_path, m_path, 1536)

    def test_corrupted_manifest_file_raises(self, tmp_path: Path) -> None:
        c_path, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=2)
        m_path.write_text("{invalid json", encoding="utf-8")

        with pytest.raises(WeaviateAlignmentError, match="Failed to load manifest JSON"):
            validate_pre_indexing_artifacts(c_path, e_path, m_path, 1536)

    def test_invariant_4_count_mismatch(self, tmp_path: Path) -> None:
        c_path, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=5)
        # Overwrite embeddings with 4 vectors
        np.save(e_path, np.ones((4, 1536), dtype=np.float32))

        with pytest.raises(WeaviateAlignmentError, match="Invariant 4 failed: Count mismatch"):
            validate_pre_indexing_artifacts(c_path, e_path, m_path, 1536)

    def test_invariant_5_dimension_mismatch(self, tmp_path: Path) -> None:
        c_path, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=5, dimension=768)

        with pytest.raises(WeaviateAlignmentError, match="Invariant 5 failed: Embeddings dimension"):
            validate_pre_indexing_artifacts(c_path, e_path, m_path, expected_dimension=1536)

    def test_invariant_5_non_2d_embeddings(self, tmp_path: Path) -> None:
        c_path, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=5)
        # Overwrite with 3D array (length is 5, but ndim is 3)
        np.save(e_path, np.ones((5, 1, 1536), dtype=np.float32))

        with pytest.raises(WeaviateAlignmentError, match="Invariant 5 failed: Embeddings array must be 2D"):
            validate_pre_indexing_artifacts(c_path, e_path, m_path, expected_dimension=1536)

    def test_invariant_6_nan_in_embeddings(self, tmp_path: Path) -> None:
        c_path, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=5, include_nan=True)

        with pytest.raises(WeaviateAlignmentError, match="Invariant 6 failed: Embeddings array contains NaN"):
            validate_pre_indexing_artifacts(c_path, e_path, m_path, 1536)

    def test_invariant_7_inf_in_embeddings(self, tmp_path: Path) -> None:
        c_path, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=5, include_inf=True)

        with pytest.raises(WeaviateAlignmentError, match="Invariant 7 failed: Embeddings array contains Inf"):
            validate_pre_indexing_artifacts(c_path, e_path, m_path, 1536)

    def test_invariant_8_duplicate_chunk_ids(self, tmp_path: Path) -> None:
        c_path, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=5, duplicate_chunk_id=True)

        with pytest.raises(WeaviateAlignmentError, match="Invariant 8 failed: Duplicate chunk IDs"):
            validate_pre_indexing_artifacts(c_path, e_path, m_path, 1536)

    def test_invariant_9_missing_chunk_id_in_manifest(self, tmp_path: Path) -> None:
        c_path, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=5, missing_manifest_id=True)

        with pytest.raises(WeaviateAlignmentError, match="Invariant 9 failed: Chunk ID .* not found in manifest"):
            validate_pre_indexing_artifacts(c_path, e_path, m_path, 1536)

    def test_invariant_9_manifest_missing_chunk_id_to_index_dict(self, tmp_path: Path) -> None:
        c_path, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=2)
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump({"chunk_count": 2}, f)

        with pytest.raises(WeaviateAlignmentError, match="Invariant 9 failed: Manifest missing 'chunk_id_to_index'"):
            validate_pre_indexing_artifacts(c_path, e_path, m_path, 1536)

    def test_invariant_10_index_alignment_mismatch(self, tmp_path: Path) -> None:
        c_path, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=5, scramble_manifest=True)

        with pytest.raises(WeaviateAlignmentError, match="Invariant 10 failed: Index alignment mismatch"):
            validate_pre_indexing_artifacts(c_path, e_path, m_path, 1536)


# ── Property Builder Tests ──────────────────────────────────────────────────


class TestBuildWeaviateProperties:
    """Verify build_weaviate_properties property dictionary creation and unpacking."""

    def test_narrative_chunk_properties(self) -> None:
        chunk = _make_test_chunk("c_narrative", text="General disclosure.", chunk_type="narrative")
        doc = _make_test_document()
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        props = build_weaviate_properties(payload)

        assert props["chunk_id"] == "c_narrative"
        assert props["chunk_text"] == "General disclosure."
        assert props["company"] == "HDFC Bank"
        assert props["document_name"] == "hdfc-bank__annual-report__fy2025-26.pdf"
        assert props["document_type"] == "pdf"
        assert props["financial_period"] == "FY2025-26"
        assert props["is_table"] is False
        assert props["table_headers"] == []
        assert props["table_caption"] is None
        assert props["table_group_id"] is None
        assert props["table_slice_index"] is None
        assert props["table_slice_count"] is None
        assert props["confidence_score"] is None
        assert props["needs_review"] is False
        assert props["extraction_method"] is None

    def test_table_chunk_properties_unpacked(self) -> None:
        table_meta = {
            "table_group_id": "tbl_group_01",
            "table_slice_index": 0,
            "table_slice_count": 2,
            "confidence_score": 0.95,
            "needs_review": True,
            "extraction_method": "pdfplumber_lattice",
            "headers": ["Line Item", "FY26 (Cr)"],
            "caption": "Balance Sheet Highlights",
        }
        chunk = _make_test_chunk("c_tbl", text="| Item | Value |", chunk_type="table", table_metadata=table_meta)
        doc = _make_test_document()
        payload = IndexingPayload.from_chunk_and_document(chunk, doc)

        props = build_weaviate_properties(payload, table_metadata=chunk.table_metadata)

        assert props["chunk_id"] == "c_tbl"
        assert props["is_table"] is True
        assert props["table_headers"] == ["Line Item", "FY26 (Cr)"]
        assert props["table_caption"] == "Balance Sheet Highlights"
        assert props["table_group_id"] == "tbl_group_01"
        assert props["table_slice_index"] == 0
        assert props["table_slice_count"] == 2
        assert props["confidence_score"] == 0.95
        assert props["needs_review"] is True
        assert props["extraction_method"] == "pdfplumber_lattice"


# ── Store Client Lifecycle Tests ────────────────────────────────────────────


class TestWeaviateStoreLifecycle:
    """Verify WeaviateStore client connection, close, and context manager lifecycle."""

    def test_connect_raises_when_missing_url(self) -> None:
        settings = Settings(
            weaviate_url="",
            weaviate_api_key="test-key",
            openai_api_key="test-key",
        )
        store = WeaviateStore(settings=settings)
        with pytest.raises(WeaviateConnectionError, match="Weaviate URL is required"):
            store.connect()

    def test_connect_raises_when_missing_api_key(self) -> None:
        settings = Settings(
            weaviate_url="https://test.weaviate.cloud",
            weaviate_api_key="",
            openai_api_key="test-key",
        )
        store = WeaviateStore(settings=settings)
        with pytest.raises(WeaviateConnectionError, match="Weaviate API key is required"):
            store.connect()

    @patch("weaviate.connect_to_weaviate_cloud")
    def test_connect_success(self, mock_connect: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.is_live.return_value = True
        mock_client.is_connected.return_value = True
        mock_connect.return_value = mock_client

        settings = Settings(
            weaviate_url="https://test.weaviate.cloud",
            weaviate_api_key="secret-key",
            openai_api_key="test-key",
        )
        store = WeaviateStore(settings=settings)
        store.connect()

        assert store.client == mock_client
        mock_connect.assert_called_once()

    @patch("weaviate.connect_to_weaviate_cloud")
    def test_connect_raises_when_not_live(self, mock_connect: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.is_live.return_value = False
        mock_connect.return_value = mock_client

        settings = Settings(
            weaviate_url="https://test.weaviate.cloud",
            weaviate_api_key="secret-key",
            openai_api_key="test-key",
        )
        store = WeaviateStore(settings=settings)
        with pytest.raises(WeaviateConnectionError, match="cluster is reachable but reported not live"):
            store.connect()

    def test_close_owned_client(self) -> None:
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        store = WeaviateStore(client=mock_client)
        # Injected client has _owns_client = False
        store.close()
        mock_client.close.assert_not_called()

        # Now test owned client
        store._owns_client = True
        store._client = mock_client
        store.close()
        mock_client.close.assert_called_once()
        assert store._client is None

    @patch("weaviate.connect_to_weaviate_cloud")
    def test_context_manager_connects_and_closes(self, mock_connect: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.is_live.return_value = True
        mock_client.is_connected.return_value = True
        mock_connect.return_value = mock_client

        settings = Settings(
            weaviate_url="https://test.weaviate.cloud",
            weaviate_api_key="secret-key",
            openai_api_key="test-key",
        )
        with WeaviateStore(settings=settings) as store:
            assert store.client == mock_client

        mock_client.close.assert_called_once()


# ── Collection Management Tests ─────────────────────────────────────────────


class TestEnsureCollection:
    """Verify collection creation, BYOV Vectorizer.none, HNSW, BM25, and recreate safety."""

    def test_recreate_false_reuses_existing_collection(self) -> None:
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.collections.exists.return_value = True
        mock_coll = MagicMock()
        mock_client.collections.get.return_value = mock_coll

        store = WeaviateStore(client=mock_client)
        result = store.ensure_collection(collection_name="DiligenceChunk", recreate=False)

        assert result == mock_coll
        mock_client.collections.delete.assert_not_called()
        mock_client.collections.get.assert_called_once_with("DiligenceChunk")
        mock_client.collections.create.assert_not_called()

    def test_recreate_true_deletes_and_creates(self) -> None:
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.collections.exists.return_value = False
        mock_created_coll = MagicMock()
        mock_client.collections.create.return_value = mock_created_coll

        store = WeaviateStore(client=mock_client)
        result = store.ensure_collection(collection_name="DiligenceChunk", recreate=True)

        assert result == mock_created_coll
        mock_client.collections.delete.assert_called_once_with("DiligenceChunk")
        mock_client.collections.create.assert_called_once()

    def test_creates_collection_with_byov_and_cosine(self) -> None:
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.collections.exists.return_value = False
        mock_created_coll = MagicMock()
        mock_client.collections.create.return_value = mock_created_coll

        store = WeaviateStore(client=mock_client)
        store.ensure_collection(collection_name="DiligenceChunk", recreate=False)

        mock_client.collections.create.assert_called_once()
        create_kwargs = mock_client.collections.create.call_args.kwargs
        assert create_kwargs["name"] == "DiligenceChunk"
        assert "vector_config" in create_kwargs

        # Verify properties count (21 properties defined)
        properties = create_kwargs["properties"]
        assert len(properties) == 21
        prop_names = {p.name for p in properties}
        assert "chunk_id" in prop_names
        assert "chunk_text" in prop_names
        assert "company" in prop_names
        assert "document_name" in prop_names
        assert "table_group_id" in prop_names
        assert "confidence_score" in prop_names

    def test_creates_collection_fallback_to_hnsw_when_hfresh_not_allowed(self) -> None:
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.collections.exists.return_value = False
        mock_created_coll = MagicMock()
        mock_client.collections.create.side_effect = [
            RuntimeError("hfresh is not allowed for vector_index_type. Allowed values: hnsw."),
            mock_created_coll,
        ]

        store = WeaviateStore(client=mock_client)
        result = store.ensure_collection(collection_name="DiligenceChunk", recreate=False)

        assert result == mock_created_coll
        assert mock_client.collections.create.call_count == 2

    def test_collection_creation_failure_raises_schema_error(self) -> None:
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.collections.exists.return_value = False
        mock_client.collections.create.side_effect = RuntimeError("Weaviate cluster quota exceeded")

        store = WeaviateStore(client=mock_client)
        with pytest.raises(WeaviateSchemaError, match="Collection creation error for 'DiligenceChunk'"):
            store.ensure_collection(collection_name="DiligenceChunk")


# ── Batch Indexing Tests ────────────────────────────────────────────────────


class TestIndexCanonicalCorpus:
    """Verify batch indexing execution, UUID5 idempotency, and error handling."""

    def test_batch_indexing_success(self, tmp_path: Path) -> None:
        c_path, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=3, dimension=1536)
        doc = _make_test_document()

        # Mock collection batch context manager
        mock_batch = MagicMock()
        mock_coll = MagicMock()
        mock_coll.batch.fixed_size.return_value.__enter__.return_value = mock_batch
        mock_coll.batch.failed_objects = []

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.collections.exists.return_value = True
        mock_client.collections.get.return_value = mock_coll

        settings = Settings(
            weaviate_url="https://test.weaviate.cloud",
            weaviate_api_key="secret-key",
            openai_api_key="test-key",
            embedding_dimension=1536,
        )
        store = WeaviateStore(settings=settings, client=mock_client)
        report = store.index_canonical_corpus(
            chunks_path=c_path,
            embeddings_path=e_path,
            manifest_path=m_path,
            document=doc,
            collection_name=DEFAULT_COLLECTION_NAME,
            batch_size=2,
            recreate_collection=False,
        )

        assert report["total_objects_indexed"] == 3
        assert report["failed_objects_count"] == 0
        assert report["batch_size"] == 2
        assert report["total_batches"] == 2  # ceil(3 / 2)
        assert mock_batch.add_object.call_count == 3

    def test_batch_indexing_fails_loudly_on_failed_objects(self, tmp_path: Path) -> None:
        c_path, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=3, dimension=1536)
        doc = _make_test_document()

        # Create mock failed object
        mock_err_obj = MagicMock()
        mock_err_obj.original_uuid = "uuid-123"
        mock_err_obj.message = "Failed to index object: vector dimension error"

        mock_batch = MagicMock()
        mock_coll = MagicMock()
        mock_coll.batch.fixed_size.return_value.__enter__.return_value = mock_batch
        mock_coll.batch.failed_objects = [mock_err_obj]

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.collections.exists.return_value = True
        mock_client.collections.get.return_value = mock_coll

        settings = Settings(
            weaviate_url="https://test.weaviate.cloud",
            weaviate_api_key="secret-key",
            openai_api_key="test-key",
            embedding_dimension=1536,
        )
        store = WeaviateStore(settings=settings, client=mock_client)

        with pytest.raises(WeaviateBatchIndexError, match="Batch indexing failed: 1 of 3 objects failed"):
            store.index_canonical_corpus(
                chunks_path=c_path,
                embeddings_path=e_path,
                manifest_path=m_path,
                document=doc,
            )

    def test_batch_indexing_aborts_before_weaviate_if_invariants_fail(self, tmp_path: Path) -> None:
        c_path, e_path, m_path = _create_test_artifacts(tmp_path, num_chunks=3, dimension=768)
        doc = _make_test_document()

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True

        settings = Settings(
            weaviate_url="https://test.weaviate.cloud",
            weaviate_api_key="secret-key",
            openai_api_key="test-key",
            embedding_dimension=1536,
        )
        store = WeaviateStore(settings=settings, client=mock_client)

        # Expected dimension is 1536, but embeddings are 768
        with pytest.raises(WeaviateAlignmentError, match="Invariant 5 failed"):
            store.index_canonical_corpus(
                chunks_path=c_path,
                embeddings_path=e_path,
                manifest_path=m_path,
                document=doc,
            )

        # Verify no collection access was ever attempted
        mock_client.collections.exists.assert_not_called()
        mock_client.collections.get.assert_not_called()


# ── Health Verification Tests ───────────────────────────────────────────────


class TestVerifyCollectionHealth:
    """Verify post-indexing count checks and probe inspections."""

    def test_health_check_succeeds_with_matching_count_and_probes(self) -> None:
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.collections.exists.return_value = True

        mock_coll = MagicMock()
        mock_aggregate = MagicMock()
        mock_aggregate.total_count = 1988
        mock_coll.aggregate.over_all.return_value = mock_aggregate

        # Mock probe fetch
        mock_probe_obj = MagicMock()
        mock_probe_obj.vector = [0.1] * 1536
        mock_probe_obj.properties = {"is_table": False}
        mock_coll.query.fetch_object_by_id.return_value = mock_probe_obj

        mock_client.collections.get.return_value = mock_coll

        settings = Settings(
            weaviate_url="https://test.weaviate.cloud",
            weaviate_api_key="secret-key",
            openai_api_key="test-key",
            embedding_dimension=1536,
        )
        store = WeaviateStore(settings=settings, client=mock_client)
        report = store.verify_collection_health(
            collection_name=DEFAULT_COLLECTION_NAME,
            expected_count=1988,
            probe_chunk_ids=["probe_001"],
        )

        assert report["status"] == "healthy"
        assert report["total_objects"] == 1988
        assert report["probes_verified_count"] == 1
        assert report["probes"][0]["verified"] is True
        assert report["probes"][0]["vector_dimension"] == 1536

    def test_health_check_raises_when_collection_missing(self) -> None:
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.collections.exists.return_value = False

        store = WeaviateStore(client=mock_client)
        with pytest.raises(WeaviateVerificationError, match="Collection 'DiligenceChunk' does not exist"):
            store.verify_collection_health("DiligenceChunk")

    def test_health_check_raises_when_count_mismatch(self) -> None:
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.collections.exists.return_value = True

        mock_coll = MagicMock()
        mock_aggregate = MagicMock()
        mock_aggregate.total_count = 1980  # expected 1988
        mock_coll.aggregate.over_all.return_value = mock_aggregate
        mock_client.collections.get.return_value = mock_coll

        store = WeaviateStore(client=mock_client)
        with pytest.raises(WeaviateVerificationError, match="Object count mismatch.*expected 1988, got 1980"):
            store.verify_collection_health("DiligenceChunk", expected_count=1988)

    def test_health_check_raises_when_probe_missing(self) -> None:
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.collections.exists.return_value = True

        mock_coll = MagicMock()
        mock_aggregate = MagicMock()
        mock_aggregate.total_count = 5
        mock_coll.aggregate.over_all.return_value = mock_aggregate
        mock_coll.query.fetch_object_by_id.return_value = None
        mock_client.collections.get.return_value = mock_coll

        store = WeaviateStore(client=mock_client)
        with pytest.raises(WeaviateVerificationError, match="Probe chunk 'missing_probe' .* not found"):
            store.verify_collection_health("DiligenceChunk", expected_count=5, probe_chunk_ids=["missing_probe"])

    def test_health_check_raises_when_probe_vector_dimension_mismatch(self) -> None:
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.collections.exists.return_value = True

        mock_coll = MagicMock()
        mock_aggregate = MagicMock()
        mock_aggregate.total_count = 5
        mock_coll.aggregate.over_all.return_value = mock_aggregate

        mock_probe_obj = MagicMock()
        mock_probe_obj.vector = [0.1] * 768  # 768 vs 1536
        mock_probe_obj.properties = {"is_table": False}
        mock_coll.query.fetch_object_by_id.return_value = mock_probe_obj
        mock_client.collections.get.return_value = mock_coll

        settings = Settings(
            weaviate_url="https://test.weaviate.cloud",
            weaviate_api_key="secret-key",
            openai_api_key="test-key",
            embedding_dimension=1536,
        )
        store = WeaviateStore(settings=settings, client=mock_client)
        with pytest.raises(WeaviateVerificationError, match="Probe chunk 'probe_001' vector has 768 dimensions"):
            store.verify_collection_health("DiligenceChunk", expected_count=5, probe_chunk_ids=["probe_001"])
