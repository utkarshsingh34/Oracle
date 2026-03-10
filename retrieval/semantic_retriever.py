"""
Semantic retriever for Oracle.

Uses all-MiniLM-L6-v2 sentence-transformer for embeddings and ChromaDB for
vector storage. Four collections (one per level). All ChunkMetadata fields
are stored in collection metadata for zero-lookup reconstruction.

Usage:
    from Oracle.retrieval.semantic_retriever import semantic_retrieve, build_semantic_index

    build_semantic_index(chunks, "function", persist_dir=".oracle_db")
    results = semantic_retrieve("create a new user", "function", top_k=10)
"""

from __future__ import annotations

import os
import time
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from Oracle.config import CHROMA_PERSIST_DIR, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
from Oracle.ingestion.ast_chunker import ChunkMetadata
from Oracle.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Collection naming
# ---------------------------------------------------------------------------

_COLLECTION_PREFIX = "oracle_"
_LEVELS = ("directory", "file", "class", "function")

_NONE_SENTINEL = "__none__"


def _collection_name(level: str) -> str:
    return f"{_COLLECTION_PREFIX}{level}"


# ---------------------------------------------------------------------------
# Lazy singletons — avoids heavy loads at import time
# ---------------------------------------------------------------------------

_embedding_model: SentenceTransformer | None = None
_chroma_clients: dict[str, chromadb.ClientAPI] = {}


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_chroma_client(persist_dir: str) -> chromadb.ClientAPI:
    if persist_dir not in _chroma_clients:
        os.makedirs(persist_dir, exist_ok=True)
        _chroma_clients[persist_dir] = chromadb.PersistentClient(path=persist_dir)
    return _chroma_clients[persist_dir]


# ---------------------------------------------------------------------------
# Metadata serialization — ChromaDB does not support None values
# ---------------------------------------------------------------------------

def _serialize_metadata(chunk: ChunkMetadata) -> dict[str, Any]:
    """Convert ChunkMetadata to a ChromaDB-safe metadata dict.

    Excludes ``content`` (stored as the ChromaDB document).
    Converts None values to a sentinel string.
    """
    data = chunk.model_dump()
    del data["content"]  # goes into documents, not metadata

    for key, value in data.items():
        if value is None:
            data[key] = _NONE_SENTINEL

    return data


def _deserialize_metadata(metadata: dict[str, Any], document: str) -> ChunkMetadata:
    """Reconstruct a ChunkMetadata from ChromaDB metadata + document."""
    fields = dict(metadata)
    fields["content"] = document

    for key, value in fields.items():
        if value == _NONE_SENTINEL:
            fields[key] = None

    return ChunkMetadata(**fields)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_semantic_index(
    chunks: list[ChunkMetadata],
    level: str,
    persist_dir: str,
) -> None:
    """Embed chunks at *level* and store in the ChromaDB collection."""
    level_chunks = [c for c in chunks if c.level == level]

    if not level_chunks:
        logger.warning("semantic_index_empty", level=level)
        return

    client = _get_chroma_client(persist_dir)
    col_name = _collection_name(level)
    collection = client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

    model = _get_embedding_model()
    total_inserted = 0
    batch_number = 0

    for i in range(0, len(level_chunks), EMBEDDING_BATCH_SIZE):
        batch = level_chunks[i : i + EMBEDDING_BATCH_SIZE]
        batch_number += 1
        batch_start = time.perf_counter()

        texts = [c.content for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        ids = [c.chunk_id for c in batch]
        metadatas = [_serialize_metadata(c) for c in batch]

        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        batch_ms = (time.perf_counter() - batch_start) * 1000.0
        total_inserted += len(batch)

        logger.info(
            "embedding_batch",
            batch_number=batch_number,
            batch_size=len(batch),
            level=level,
            latency_ms=round(batch_ms, 2),
        )

    logger.info(
        "chromadb_collection_updated",
        collection_name=col_name,
        documents_inserted=total_inserted,
        total_collection_size=collection.count(),
    )


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

def semantic_retrieve(
    query: str,
    level: str,
    top_k: int,
    persist_dir: str | None = None,
) -> list[ChunkMetadata]:
    """Retrieve the top-k semantically similar chunks at *level*."""
    use_dir = persist_dir or CHROMA_PERSIST_DIR
    client = _get_chroma_client(use_dir)
    col_name = _collection_name(level)

    try:
        collection = client.get_collection(name=col_name)
    except Exception as exc:
        logger.warning(
            "semantic_collection_not_found",
            collection_name=col_name,
            error=str(exc),
            exc_info=True,
        )
        return []

    model = _get_embedding_model()
    query_embedding = model.encode(query, show_progress_bar=False).tolist()

    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    results: list[ChunkMetadata] = []
    top_results_debug: list[dict] = []

    if raw["ids"] and raw["ids"][0]:
        for idx, chunk_id in enumerate(raw["ids"][0]):
            metadata = raw["metadatas"][0][idx]
            document = raw["documents"][0][idx]
            distance = raw["distances"][0][idx]

            chunk = _deserialize_metadata(metadata, document)
            results.append(chunk)

            if len(top_results_debug) < 3:
                top_results_debug.append(
                    {"chunk_id": chunk_id, "distance": round(distance, 4)}
                )

    logger.info(
        "semantic_retrieve",
        query_text_length=len(query),
        level=level,
        top_k=top_k,
        results_returned=len(results),
    )

    logger.debug("semantic_top_results", results=top_results_debug)

    return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_collection_count(level: str, persist_dir: str | None = None) -> int:
    """Return the number of documents in the collection for *level*."""
    use_dir = persist_dir or CHROMA_PERSIST_DIR
    client = _get_chroma_client(use_dir)
    col_name = _collection_name(level)
    try:
        collection = client.get_collection(name=col_name)
        return collection.count()
    except Exception:
        return 0
