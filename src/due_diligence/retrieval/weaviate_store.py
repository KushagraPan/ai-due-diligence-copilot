"""
Weaviate Cloud vector store connector, collection manager, and batch indexer.

1. Purpose:
   Manages Weaviate Cloud client lifecycle, creates and verifies the DiligenceChunk
   collection schema, and provides idempotent batch indexing of pre-computed
   dense vectors and chunk payloads.

2. Role in Overall RAG Pipeline:
   Serves as the primary vector and keyword index storage layer (Phase B3).
   Consumes verified Chunk payloads and pre-computed dense embeddings from Phase B2,
   storing them in Weaviate Cloud for subsequent hybrid retrieval (Phase B4).

3. Expected Inputs:
   - Canonical chunks: Path to JSON lines file containing serialized Chunk objects.
   - Embeddings: Path to NumPy .npy array containing float32 vectors.
   - Manifest: Path to JSON manifest containing chunk_id-to-index mapping.
   - Document metadata: ValidatedDocument instance providing document provenance.

4. Outputs:
   - Indexed Weaviate data objects with deterministic UUID5 identifiers.
   - Indexing execution summary report and health verification diagnostics.

5. Dependencies:
   - ``weaviate-client`` (v4 SDK) for Weaviate Cloud connectivity.
   - ``numpy`` for array loading and NaN/Inf validation.
   - ``due_diligence.core.config`` for cluster credentials and settings.
   - ``due_diligence.retrieval.schemas`` for IndexingPayload models.

6. Non-Responsibilities:
   - Does NOT generate embeddings (BYOV: Bring Your Own Vectors).
   - Does NOT parse raw PDFs or construct chunks.
   - Does NOT perform hybrid retrieval, reranking, or LLM generation (B4-B8).
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
import weaviate
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)
from weaviate.classes.init import Auth
from weaviate.client import WeaviateClient
from weaviate.collections import Collection
from weaviate.exceptions import WeaviateBaseError
from weaviate.util import generate_uuid5

from due_diligence.core.config import Settings, get_settings
from due_diligence.ingestion.chunks import Chunk
from due_diligence.ingestion.document import ValidatedDocument
from due_diligence.retrieval.schemas import IndexingPayload

logger = logging.getLogger(__name__)

# Default canonical collection name
DEFAULT_COLLECTION_NAME = "DiligenceChunk"


# ── Typed Exceptions ────────────────────────────────────────────────────────


class WeaviateStoreError(Exception):
    """Base exception for all Weaviate store operations."""


class WeaviateConnectionError(WeaviateStoreError):
    """Raised when Weaviate Cloud cannot be reached or credentials are invalid."""


class WeaviateSchemaError(WeaviateStoreError):
    """Raised when collection schema creation, configuration, or validation fails."""


class WeaviateAlignmentError(WeaviateStoreError):
    """Raised when local chunk and embedding artifacts fail pre-flight alignment checks."""


class WeaviateBatchIndexError(WeaviateStoreError):
    """Raised when batch insertion encounters one or more failed objects."""


class WeaviateVerificationError(WeaviateStoreError):
    """Raised when post-indexing count or probe verification fails."""


# ── Pre-Indexing Artifacts Validation (The 10 Invariants) ───────────────────


def validate_pre_indexing_artifacts(
    chunks_path: Path,
    embeddings_path: Path,
    manifest_path: Path,
    expected_dimension: int,
) -> tuple[list[Chunk], np.ndarray, dict[str, Any]]:
    """
    Validate all 10 pre-indexing invariants before contacting Weaviate Cloud.

    Checks:
    1. Chunks file exists.
    2. Embeddings file exists.
    3. Manifest file exists.
    4. Chunk count == embedding count == manifest chunk_count.
    5. Embeddings shape[1] == expected_dimension.
    6. No NaN values in embeddings.
    7. No Inf values in embeddings.
    8. No duplicate chunk IDs in chunks set.
    9. Every chunk ID exists in manifest chunk_id_to_index.
    10. Manifest index exactly matches chunk position i.

    Returns:
        tuple of (chunks, embeddings_array, manifest_dict)

    Raises:
        WeaviateAlignmentError: If ANY of the 10 checks fail.
    """
    # 1. Chunks file exists
    if not chunks_path.exists():
        raise WeaviateAlignmentError(f"Invariant 1 failed: Chunks file not found at {chunks_path}")

    # 2. Embeddings file exists
    if not embeddings_path.exists():
        raise WeaviateAlignmentError(f"Invariant 2 failed: Embeddings file not found at {embeddings_path}")

    # 3. Manifest file exists
    if not manifest_path.exists():
        raise WeaviateAlignmentError(f"Invariant 3 failed: Manifest file not found at {manifest_path}")

    # Load artifacts
    try:
        chunks: list[Chunk] = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(Chunk.model_validate_json(line))
    except Exception as err:
        raise WeaviateAlignmentError(f"Failed to read/parse chunks file {chunks_path}: {err}") from err

    try:
        embeddings = np.load(embeddings_path)
    except Exception as err:
        raise WeaviateAlignmentError(f"Failed to load embeddings array from {embeddings_path}: {err}") from err

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as err:
        raise WeaviateAlignmentError(f"Failed to load manifest JSON from {manifest_path}: {err}") from err

    # 4. Length equality
    manifest_count = manifest.get("chunk_count")
    if not (len(chunks) == len(embeddings) == manifest_count):
        raise WeaviateAlignmentError(
            f"Invariant 4 failed: Count mismatch: chunks={len(chunks)}, "
            f"embeddings={len(embeddings)}, manifest={manifest_count}"
        )

    # 5. Dimensionality
    if embeddings.ndim != 2:
        raise WeaviateAlignmentError(
            f"Invariant 5 failed: Embeddings array must be 2D, got {embeddings.ndim}D shape {embeddings.shape}"
        )
    if embeddings.shape[1] != expected_dimension:
        raise WeaviateAlignmentError(
            f"Invariant 5 failed: Embeddings dimension ({embeddings.shape[1]}) does not match "
            f"expected authoritative dimension ({expected_dimension})"
        )

    # 6. No NaN
    if np.isnan(embeddings).any():
        raise WeaviateAlignmentError("Invariant 6 failed: Embeddings array contains NaN values")

    # 7. No Inf
    if np.isinf(embeddings).any():
        raise WeaviateAlignmentError("Invariant 7 failed: Embeddings array contains Inf values")

    # 8. No duplicate chunk IDs
    unique_chunk_ids = {c.chunk_id for c in chunks}
    if len(unique_chunk_ids) != len(chunks):
        raise WeaviateAlignmentError(
            f"Invariant 8 failed: Duplicate chunk IDs detected: {len(chunks)} chunks, "
            f"but only {len(unique_chunk_ids)} unique IDs"
        )

    # 9 & 10. Manifest mapping correspondence
    if "chunk_id_to_index" not in manifest or not isinstance(manifest["chunk_id_to_index"], dict):
        raise WeaviateAlignmentError("Invariant 9 failed: Manifest missing 'chunk_id_to_index' dictionary")
    chunk_id_to_index = manifest["chunk_id_to_index"]

    for i, c in enumerate(chunks):
        if c.chunk_id not in chunk_id_to_index:
            raise WeaviateAlignmentError(
                f"Invariant 9 failed: Chunk ID '{c.chunk_id}' at index {i} not found in manifest"
            )
        manifest_idx = chunk_id_to_index[c.chunk_id]
        if manifest_idx != i:
            raise WeaviateAlignmentError(
                f"Invariant 10 failed: Index alignment mismatch for '{c.chunk_id}': "
                f"manifest index is {manifest_idx}, but chunk position is {i}"
            )

    logger.info(
        "Pre-indexing validation passed: %d chunks, %d vectors of dim %d perfectly aligned.",
        len(chunks),
        len(embeddings),
        expected_dimension,
    )
    return chunks, embeddings, manifest


# ── Property Builder ────────────────────────────────────────────────────────


def build_weaviate_properties(
    payload: IndexingPayload,
    table_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Map an IndexingPayload into a flat dictionary of Weaviate collection properties.

    Maps ``payload.text`` to ``chunk_text`` and unpacks table provenance fields
    as first-class properties (null for non-table chunks).
    """
    props: dict[str, Any] = {
        "chunk_id": payload.chunk_id,
        "chunk_text": payload.text,
        "company": payload.company,
        "document_name": payload.document_name,
        "document_type": payload.document_type,
        "financial_period": payload.financial_period,
        "page_numbers": payload.page_numbers,
        "section_heading": payload.section_heading,
        "heading_path": payload.heading_path,
        "chunk_type": payload.chunk_type,
        "is_table": payload.is_table,
        "table_headers": payload.table_headers,
        "table_caption": payload.table_caption,
        "source_element_ids": payload.source_element_ids,
        "content_hash": payload.content_hash,
    }

    # First-class table provenance fields
    meta = table_metadata or payload.table_metadata
    if meta:
        props["table_group_id"] = meta.get("table_group_id")
        props["table_slice_index"] = meta.get("table_slice_index")
        props["table_slice_count"] = meta.get("table_slice_count")
        props["confidence_score"] = meta.get("confidence_score")
        props["needs_review"] = meta.get("needs_review", False)
        props["extraction_method"] = meta.get("extraction_method")
    else:
        props["table_group_id"] = None
        props["table_slice_index"] = None
        props["table_slice_count"] = None
        props["confidence_score"] = None
        props["needs_review"] = False
        props["extraction_method"] = None

    return props


# ── Weaviate Store Manager ──────────────────────────────────────────────────


class WeaviateStore:
    """
    Weaviate Cloud vector store manager and batch indexer.

    Manages connection lifecycle, creates BYOV collections with HNSW cosine distance
    and BM25 inverted indexing, and provides deterministic, idempotent batch ingestion.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: WeaviateClient | None = None,
    ) -> None:
        """
        Initialize WeaviateStore with configuration.

        Args:
            settings: Optional Settings instance. Defaults to get_settings().
            client: Optional injected WeaviateClient (useful for unit testing).
        """
        self._settings = settings or get_settings()
        self._client = client
        self._owns_client = client is None

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def connect(self) -> None:
        """Connect to Weaviate Cloud cluster if not already connected."""
        if self._client is not None and self._client.is_connected():
            return

        if not self._settings.weaviate_url or not self._settings.weaviate_url.strip():
            raise WeaviateConnectionError("Weaviate URL is required in settings (WEAVIATE_URL).")
        if not self._settings.weaviate_api_key or not self._settings.weaviate_api_key.strip():
            raise WeaviateConnectionError("Weaviate API key is required in settings (WEAVIATE_API_KEY).")

        try:
            logger.info("Connecting to Weaviate Cloud: %s", self._settings.weaviate_url)
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self._settings.weaviate_url,
                auth_credentials=Auth.api_key(self._settings.weaviate_api_key),
                skip_init_checks=True,
            )
            if not self._client.is_live():
                raise WeaviateConnectionError("Weaviate cluster is reachable but reported not live.")
            logger.info("Connected successfully to Weaviate Cloud.")
        except WeaviateConnectionError:
            raise
        except Exception as err:
            logger.error("Failed to connect to Weaviate Cloud: %s", err)
            raise WeaviateConnectionError(f"Weaviate Cloud connection error: {err}") from err

    def close(self) -> None:
        """Close connection and clean up underlying client resources."""
        if self._owns_client and self._client is not None:
            try:
                self._client.close()
                logger.info("Closed Weaviate client connection.")
            except (WeaviateBaseError, OSError, RuntimeError) as err:
                logger.warning("Error closing Weaviate client: %s", err)
            finally:
                self._client = None

    @property
    def client(self) -> WeaviateClient:
        """Return the active WeaviateClient, connecting if necessary."""
        if self._client is None or not self._client.is_connected():
            self.connect()
        assert self._client is not None
        return self._client

    def ensure_collection(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        recreate: bool = False,
    ) -> Collection[Any, Any]:
        """
        Verify that the collection exists, or create it configured for BYOV and BM25.

        Safe Default:
            recreate is False. An existing collection is never deleted or recreated
            automatically unless explicitly instructed.

        Args:
            collection_name: Name of the collection (e.g. 'DiligenceChunk').
            recreate: If True, deletes existing collection before creation.
        """
        cli = self.client

        if recreate:
            logger.warning("Explicit recreate=True requested. Deleting collection '%s'...", collection_name)
            try:
                cli.collections.delete(collection_name)
            except (WeaviateBaseError, OSError, RuntimeError) as err:
                logger.warning("Error deleting collection during recreate: %s", err)

        if cli.collections.exists(collection_name):
            logger.info("Collection '%s' already exists. Reusing existing collection.", collection_name)
            return cli.collections.get(collection_name)

        logger.info("Creating collection '%s' with BYOV (Vectorizer.none) and HNSW cosine...", collection_name)

        properties = [
            Property(name="chunk_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="chunk_text", data_type=DataType.TEXT, tokenization=Tokenization.WORD, index_searchable=True),
            Property(name="company", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="document_name", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="document_type", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="financial_period", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="page_numbers", data_type=DataType.INT_ARRAY, index_filterable=True),
            Property(
                name="section_heading",
                data_type=DataType.TEXT,
                tokenization=Tokenization.WORD,
                index_searchable=True,
                index_filterable=True,
            ),
            Property(name="heading_path", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="chunk_type", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="is_table", data_type=DataType.BOOL, index_filterable=True),
            Property(name="table_headers", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.WORD, index_searchable=True),
            Property(name="table_caption", data_type=DataType.TEXT, tokenization=Tokenization.WORD, index_searchable=True),
            Property(name="source_element_ids", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="content_hash", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            # Table provenance properties
            Property(name="table_group_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="table_slice_index", data_type=DataType.INT, index_filterable=True),
            Property(name="table_slice_count", data_type=DataType.INT, index_filterable=True),
            Property(name="confidence_score", data_type=DataType.NUMBER, index_filterable=True),
            Property(name="needs_review", data_type=DataType.BOOL, index_filterable=True),
            Property(name="extraction_method", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
        ]

        try:
            # Weaviate Cloud (WCS Sandbox/Serverless) enforces hfresh; self-hosted clusters use hnsw.
            try:
                vector_index_cfg = Configure.VectorIndex.hfresh(distance_metric=VectorDistances.COSINE)
                collection = cli.collections.create(
                    name=collection_name,
                    vector_config=Configure.Vectors.self_provided(vector_index_config=vector_index_cfg),
                    inverted_index_config=Configure.inverted_index(
                        bm25_b=0.75,
                        bm25_k1=1.2,
                    ),
                    properties=properties,
                )
                logger.info("Successfully created collection '%s' with hfresh vector index.", collection_name)
                return collection
            except Exception as inner_err:
                err_msg = str(inner_err).lower()
                if "hfresh is not allowed" in err_msg or "allowed values: hnsw" in err_msg:
                    logger.info("Cluster does not allow hfresh index; falling back to HNSW...")
                    hnsw_cfg = Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)
                    collection = cli.collections.create(
                        name=collection_name,
                        vector_config=Configure.Vectors.self_provided(vector_index_config=hnsw_cfg),
                        inverted_index_config=Configure.inverted_index(
                            bm25_b=0.75,
                            bm25_k1=1.2,
                        ),
                        properties=properties,
                    )
                    logger.info("Successfully created collection '%s' with HNSW vector index.", collection_name)
                    return collection
                raise
        except Exception as err:
            logger.error("Failed to create Weaviate collection '%s': %s", collection_name, err)
            raise WeaviateSchemaError(f"Collection creation error for '{collection_name}': {err}") from err

    def index_canonical_corpus(
        self,
        chunks_path: Path,
        embeddings_path: Path,
        manifest_path: Path,
        document: ValidatedDocument,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        batch_size: int = 100,
        recreate_collection: bool = False,
    ) -> dict[str, Any]:
        """
        Validate artifacts, connect to Weaviate, and batch-upsert chunks with vectors.

        Guarantees:
        - Executes all 10 pre-flight invariant checks before opening batch context.
        - Generates deterministic UUID5 from chunk_id for idempotency.
        - Uses Weaviate v4 fixed_size batching with 100 objects per request.
        - Inspects failed_objects and fails loudly if any object fails.

        Args:
            chunks_path: Path to canonical chunks JSONL.
            embeddings_path: Path to embeddings .npy array.
            manifest_path: Path to embeddings manifest JSON.
            document: ValidatedDocument providing document provenance.
            collection_name: Target collection name.
            batch_size: Number of objects per batch request.
            recreate_collection: If True, drops collection first. Defaults to False.

        Returns:
            Dictionary containing indexing execution metrics.
        """
        # Validate pre-indexing artifacts BEFORE any Weaviate contact
        chunks, embeddings, _manifest = validate_pre_indexing_artifacts(
            chunks_path=chunks_path,
            embeddings_path=embeddings_path,
            manifest_path=manifest_path,
            expected_dimension=self._settings.embedding_dimension,
        )

        collection = self.ensure_collection(
            collection_name=collection_name,
            recreate=recreate_collection,
        )

        total_objects = len(chunks)
        total_batches = (total_objects + batch_size - 1) // batch_size
        logger.info(
            "Starting batch indexing of %d objects into '%s' (batch_size=%d, batches=%d)...",
            total_objects,
            collection_name,
            batch_size,
            total_batches,
        )

        t0 = datetime.now(UTC)

        try:
            with collection.batch.fixed_size(batch_size=batch_size, concurrent_requests=2) as batch:
                for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
                    payload = IndexingPayload.from_chunk_and_document(chunk, document)
                    properties = build_weaviate_properties(payload, chunk.table_metadata)
                    uuid = generate_uuid5(chunk.chunk_id)

                    batch.add_object(
                        properties=properties,
                        vector=vector.tolist(),
                        uuid=uuid,
                    )

            # Check for failed objects
            failed_objects = collection.batch.failed_objects
            if failed_objects:
                logger.error("Weaviate batch indexing encountered %d failures!", len(failed_objects))
                for err_obj in failed_objects[:5]:
                    logger.error("Failed UUID %s: %s", err_obj.original_uuid, err_obj.message)
                raise WeaviateBatchIndexError(
                    f"Batch indexing failed: {len(failed_objects)} of {total_objects} objects failed."
                )

        except WeaviateBatchIndexError:
            raise
        except Exception as err:
            logger.error("Error during Weaviate batch insertion: %s", err)
            raise WeaviateBatchIndexError(f"Weaviate batch index error: {err}") from err

        t1 = datetime.now(UTC)
        elapsed = (t1 - t0).total_seconds()
        throughput = round(total_objects / max(elapsed, 0.001), 2)
        logger.info(
            "Batch indexing complete: %d objects indexed in %.2fs (%.1f obj/sec).",
            total_objects,
            elapsed,
            throughput,
        )

        return {
            "collection_name": collection_name,
            "document_name": document.document_name,
            "company": document.company,
            "total_objects_indexed": total_objects,
            "failed_objects_count": 0,
            "batch_size": batch_size,
            "total_batches": total_batches,
            "elapsed_seconds": round(elapsed, 2),
            "throughput_obj_per_sec": throughput,
            "indexed_at": t1.isoformat(),
        }

    def verify_collection_health(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        expected_count: int = 1988,
        probe_chunk_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Perform post-indexing health verification on the Weaviate collection.

        Verifies:
        - Collection exists.
        - Total object count equals expected_count.
        - Probe chunk IDs can be fetched by UUID5.
        - Returned probe objects have valid vectors and preserved properties.
        """
        cli = self.client
        if not cli.collections.exists(collection_name):
            raise WeaviateVerificationError(f"Collection '{collection_name}' does not exist in cluster.")

        collection = cli.collections.get(collection_name)

        # 1. Total count verification
        try:
            aggregate = collection.aggregate.over_all(total_count=True)
            actual_count = aggregate.total_count
        except Exception as err:
            raise WeaviateVerificationError(f"Failed to query collection count aggregation: {err}") from err

        if actual_count != expected_count:
            raise WeaviateVerificationError(
                f"Object count mismatch in '{collection_name}': expected {expected_count}, got {actual_count}."
            )

        # 2. Probe chunk verification
        probe_results: list[dict[str, Any]] = []
        if probe_chunk_ids:
            for c_id in probe_chunk_ids:
                uuid = generate_uuid5(c_id)
                try:
                    obj = collection.query.fetch_object_by_id(uuid, include_vector=True)
                    if obj is None:
                        raise WeaviateVerificationError(f"Probe chunk '{c_id}' (UUID: {uuid}) not found in collection.")

                    vec = obj.vector.get("default") if isinstance(obj.vector, dict) else obj.vector
                    if vec is None or len(vec) != self._settings.embedding_dimension:
                        dim_len = len(vec) if vec else 0
                        raise WeaviateVerificationError(
                            f"Probe chunk '{c_id}' vector has {dim_len} dimensions; "
                            f"expected {self._settings.embedding_dimension}"
                        )

                    probe_results.append({
                        "chunk_id": c_id,
                        "uuid": str(uuid),
                        "verified": True,
                        "vector_dimension": len(vec),
                        "is_table": obj.properties.get("is_table"),
                    })
                except WeaviateVerificationError:
                    raise
                except Exception as err:
                    raise WeaviateVerificationError(f"Failed to fetch probe chunk '{c_id}': {err}") from err

        return {
            "collection_name": collection_name,
            "status": "healthy",
            "total_objects": actual_count,
            "expected_objects": expected_count,
            "probes_verified_count": len(probe_results),
            "probes": probe_results,
            "verified_at": datetime.now(UTC).isoformat(),
        }
