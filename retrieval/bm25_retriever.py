"""
BM25 retriever for Oracle.

Four separate BM25Okapi indexes — one per structural level (directory, file,
class, function). Persisted to disk as pickle files alongside vocabulary.

Usage:
    from Oracle.retrieval.bm25_retriever import bm25_retrieve, build_bm25_index

    build_bm25_index(chunks, "function", persist_dir=".oracle_db")
    results = bm25_retrieve("create user", "function", top_k=10)
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from Oracle.config import CHROMA_PERSIST_DIR
from Oracle.ingestion.ast_chunker import ChunkMetadata
from Oracle.logging_config import get_logger
from Oracle.retrieval.technicality_scorer import camel_snake_tokenize

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level state — populated by build or load
# ---------------------------------------------------------------------------

_bm25_indexes: dict[str, BM25Okapi] = {}
_bm25_corpus_ids: dict[str, list[str]] = {}
_bm25_chunk_map: dict[str, ChunkMetadata] = {}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_bm25_index(
    chunks: list[ChunkMetadata],
    level: str,
    persist_dir: str | None = None,
) -> None:
    """Build a BM25Okapi index for *level* from *chunks* and optionally persist."""
    start = time.perf_counter()

    level_chunks = [c for c in chunks if c.level == level]

    if not level_chunks:
        logger.warning("bm25_index_empty", level=level)
        return

    tokenized_corpus: list[list[str]] = []
    corpus_ids: list[str] = []
    all_tokens: set[str] = set()

    for chunk in level_chunks:
        tokens = camel_snake_tokenize(chunk.content)
        tokenized_corpus.append(tokens)
        corpus_ids.append(chunk.chunk_id)
        _bm25_chunk_map[chunk.chunk_id] = chunk
        all_tokens.update(tokens)

    index = BM25Okapi(tokenized_corpus)

    _bm25_indexes[level] = index
    _bm25_corpus_ids[level] = corpus_ids

    duration_ms = (time.perf_counter() - start) * 1000.0

    logger.info(
        "bm25_index_built",
        level=level,
        corpus_size=len(level_chunks),
        vocabulary_size=len(all_tokens),
        duration_ms=round(duration_ms, 2),
    )

    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        level_chunk_map = {cid: _bm25_chunk_map[cid] for cid in corpus_ids}
        pkl_path = os.path.join(persist_dir, f"bm25_{level}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump((index, corpus_ids, level_chunk_map), f)


def load_bm25_index(level: str, persist_dir: str) -> bool:
    """Load a persisted BM25 index for *level*. Returns True on success."""
    pkl_path = os.path.join(persist_dir, f"bm25_{level}.pkl")
    if not os.path.exists(pkl_path):
        logger.warning("bm25_index_not_found", level=level, path=pkl_path)
        return False

    try:
        with open(pkl_path, "rb") as f:
            index, corpus_ids, level_chunk_map = pickle.load(f)
        _bm25_indexes[level] = index
        _bm25_corpus_ids[level] = corpus_ids
        _bm25_chunk_map.update(level_chunk_map)
        return True
    except Exception as exc:
        logger.warning(
            "bm25_index_load_failed",
            level=level,
            path=pkl_path,
            error=str(exc),
            exc_info=True,
        )
        return False


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

def bm25_retrieve(
    query: str,
    level: str,
    top_k: int,
    persist_dir: str | None = None,
) -> list[ChunkMetadata]:
    """Retrieve the top-k chunks for *query* at *level* using BM25."""
    # Auto-load from disk if not in memory
    if level not in _bm25_indexes:
        load_dir = persist_dir or CHROMA_PERSIST_DIR
        if not load_bm25_index(level, load_dir):
            logger.warning("bm25_retrieve_no_index", level=level)
            return []

    query_tokens = camel_snake_tokenize(query)

    index = _bm25_indexes[level]
    corpus_ids = _bm25_corpus_ids[level]

    scores = index.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results: list[ChunkMetadata] = []
    top_results_debug: list[dict] = []

    for i in top_indices:
        score = float(scores[i])
        if score <= 0.0:
            break
        chunk_id = corpus_ids[i]
        chunk = _bm25_chunk_map.get(chunk_id)
        if chunk is not None:
            results.append(chunk)
            if len(top_results_debug) < 3:
                top_results_debug.append({"chunk_id": chunk_id, "score": round(score, 4)})

    logger.info(
        "bm25_retrieve",
        query_tokens=query_tokens,
        level=level,
        top_k=top_k,
        results_returned=len(results),
    )

    logger.debug("bm25_top_results", results=top_results_debug)

    return results


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocabulary(
    chunks: list[ChunkMetadata],
    persist_dir: str | None = None,
) -> set[str]:
    """Build and optionally persist the vocabulary set from all chunk contents."""
    vocabulary: set[str] = set()
    for chunk in chunks:
        tokens = camel_snake_tokenize(chunk.content)
        vocabulary.update(tokens)

    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        vocab_path = os.path.join(persist_dir, "vocabulary.pkl")
        with open(vocab_path, "wb") as f:
            pickle.dump(vocabulary, f)

        logger.info(
            "vocabulary_persisted",
            vocabulary_size=len(vocabulary),
            file_path=vocab_path,
        )

    return vocabulary


def load_vocabulary(persist_dir: str) -> set[str]:
    """Load the persisted vocabulary set. Returns empty set if missing."""
    vocab_path = os.path.join(persist_dir, "vocabulary.pkl")
    if not os.path.exists(vocab_path):
        logger.warning("vocabulary_not_found", path=vocab_path)
        return set()

    try:
        with open(vocab_path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning(
            "vocabulary_load_failed",
            path=vocab_path,
            error=str(exc),
            exc_info=True,
        )
        return set()


# ---------------------------------------------------------------------------
# Dynamic top_k — will move to hybrid_retriever when it's built
# ---------------------------------------------------------------------------

def compute_top_k(total_chunks: int) -> int:
    """~1% of corpus, floored at 5, capped at 20."""
    return max(5, min(20, total_chunks // 100))
