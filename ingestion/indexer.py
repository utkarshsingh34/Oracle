"""
Oracle indexing pipeline.

Wires: repo file walk → AST chunking → LLM summarization for long functions
→ directory chunk generation → semantic embedding → BM25 index build
→ vocabulary and symbols persistence.

This is a pipeline entrypoint — binds a trace_id at start, unbinds at end.

Usage:
    from Oracle.ingestion.indexer import index_repo
    from Oracle.llm.client import LLMClient

    summary = index_repo("/path/to/repo")
    summary = index_repo("/path/to/repo", llm_client=LLMClient())
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path, PurePosixPath

from Oracle.config import (
    CHROMA_PERSIST_DIR,
    SUMMARIZE_MIN_LINES,
    SUPPORTED_LANGUAGES,
)
from Oracle.ingestion.ast_chunker import (
    ChunkMetadata,
    build_directory_chunk,
    chunk_file,
)
from Oracle.llm.client import LLMClient
from Oracle.llm.schemas import FunctionSummary
from Oracle.logging_config import bind_trace_id, get_logger, unbind_trace_id
from Oracle.retrieval.bm25_retriever import build_bm25_index, build_vocabulary
from Oracle.retrieval.semantic_retriever import build_semantic_index

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# File walker (inline until repo_loader.py is built)
# ---------------------------------------------------------------------------

_SKIP_DIRS: set[str] = {
    ".git", "node_modules", "__pycache__", "dist", "build",
    ".oracle_db", ".venv", "venv",
}

_SKIP_EXTENSIONS: set[str] = {
    ".min.js", ".lock", ".json", ".yaml", ".yml",
    ".md", ".txt", ".png", ".jpg", ".svg", ".gif", ".ico",
}

_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
}


def _walk_repo(repo_path: str) -> list[tuple[str, str, str]]:
    """Walk *repo_path* and return (relative_path, source_code, language) tuples.

    Skips directories and extensions listed above. Skips files that fail
    UTF-8 decode (binary detection). Normalizes paths to forward slashes.
    """
    root = Path(repo_path).resolve()
    results: list[tuple[str, str, str]] = []
    skipped = 0
    language_counts: dict[str, int] = {}

    logger.info("repo_walk_started", repo_path=repo_path)
    start = time.perf_counter()

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        # Check if any parent directory should be skipped
        parts = path.relative_to(root).parts
        if any(part in _SKIP_DIRS for part in parts):
            logger.debug("file_skipped", file_path=str(path), reason="path_pattern")
            skipped += 1
            continue

        # Check extension
        ext = path.suffix.lower()

        # Skip known non-code extensions
        if ext in _SKIP_EXTENSIONS:
            logger.debug("file_skipped", file_path=str(path), reason="extension")
            skipped += 1
            continue

        # Only process supported language files
        language = _LANGUAGE_MAP.get(ext)
        if language is None:
            logger.debug("file_skipped", file_path=str(path), reason="extension")
            skipped += 1
            continue

        if language not in SUPPORTED_LANGUAGES:
            logger.debug("file_skipped", file_path=str(path), reason="extension")
            skipped += 1
            continue

        # Try UTF-8 decode
        try:
            source_code = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            logger.debug("file_skipped", file_path=str(path), reason="utf8_decode_failed")
            skipped += 1
            continue

        # Normalize to forward-slash relative path
        rel_path = PurePosixPath(path.relative_to(root)).as_posix()

        results.append((rel_path, source_code, language))
        language_counts[language] = language_counts.get(language, 0) + 1

    duration_ms = (time.perf_counter() - start) * 1000.0

    logger.info(
        "repo_walk_completed",
        repo_path=repo_path,
        total_files=len(results),
        skipped_files=skipped,
        languages=language_counts,
        duration_ms=round(duration_ms, 2),
    )

    return results


# ---------------------------------------------------------------------------
# Summarization prompt
# ---------------------------------------------------------------------------

_SUMMARIZATION_PROMPT = """Summarize this function precisely.

Include: what it does, all parameters with types, return value and type,
exceptions raised, key algorithms, external dependencies, and side effects
(DB writes, API calls, file I/O).

Write 5-10 technically precise sentences. Do not omit parameters or return types.

Function:
{function_code}
"""


# ---------------------------------------------------------------------------
# Pipeline entrypoint
# ---------------------------------------------------------------------------

def index_repo(
    repo_path: str,
    llm_client: LLMClient | None = None,
) -> dict:
    """Index a repository: chunk, summarize, embed, build BM25.

    Parameters
    ----------
    repo_path:
        Path to the local repository root.
    llm_client:
        LLM client for function summarization. If None, a default is created
        only when summarization is needed.

    Returns
    -------
    dict with indexing summary stats.
    """
    trace_id = bind_trace_id()
    pipeline_start = time.perf_counter()

    logger.info("indexing_started", repo_path=repo_path, trace_id=trace_id)

    persist_dir = CHROMA_PERSIST_DIR
    os.makedirs(persist_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Walk the repository
    # ------------------------------------------------------------------
    files = _walk_repo(repo_path)

    # ------------------------------------------------------------------
    # Step 2: Chunk each file
    # ------------------------------------------------------------------
    all_chunks: list[ChunkMetadata] = []
    level_counts: dict[str, int] = {"directory": 0, "file": 0, "class": 0, "function": 0}

    for rel_path, source_code, language in files:
        try:
            file_chunks = chunk_file(rel_path, source_code, language)
            all_chunks.extend(file_chunks)
            for c in file_chunks:
                level_counts[c.level] = level_counts.get(c.level, 0) + 1
        except Exception as exc:
            logger.error(
                "indexing_file_failed",
                file_path=rel_path,
                phase="chunking",
                error=str(exc),
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Step 3: Summarize long functions
    # ------------------------------------------------------------------
    total_llm_calls = 0

    long_functions = [
        c for c in all_chunks
        if c.level == "function"
        and c.original_line_count is not None
        and c.original_line_count >= SUMMARIZE_MIN_LINES
    ]

    if long_functions:
        if llm_client is None:
            llm_client = LLMClient()

        for chunk in long_functions:
            logger.info(
                "summarization_call",
                file_path=chunk.file_path,
                function_name=chunk.function_name,
                line_count=chunk.original_line_count,
                llm_backend=llm_client.backend,
                llm_model=llm_client.model,
            )

            call_start = time.perf_counter()
            try:
                prompt = _SUMMARIZATION_PROMPT.format(function_code=chunk.content)
                result = llm_client.complete(prompt, FunctionSummary)

                latency_ms = (time.perf_counter() - call_start) * 1000.0
                total_llm_calls += 1

                # Mutate chunk: store summary in content, original in full_content
                chunk.full_content = chunk.content
                chunk.content = result.summary
                chunk.is_summarized = True

                logger.info(
                    "summarization_completed",
                    file_path=chunk.file_path,
                    function_name=chunk.function_name,
                    latency_ms=round(latency_ms, 2),
                    summary_length=len(result.summary),
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - call_start) * 1000.0
                logger.warning(
                    "summarization_failed",
                    file_path=chunk.file_path,
                    function_name=chunk.function_name,
                    error=str(exc),
                    latency_ms=round(latency_ms, 2),
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Step 4: Build directory chunks
    # ------------------------------------------------------------------
    dir_paths = {c.directory_path for c in all_chunks if c.level == "file"}

    for dir_path in sorted(dir_paths):
        dir_file_chunks = [c for c in all_chunks if c.level == "file"]
        dir_chunk = build_directory_chunk(dir_path, dir_file_chunks)
        if dir_chunk is not None:
            all_chunks.append(dir_chunk)
            level_counts["directory"] = level_counts.get("directory", 0) + 1

    # ------------------------------------------------------------------
    # Step 5: Build BM25 indexes (4 levels)
    # ------------------------------------------------------------------
    for level in ("directory", "file", "class", "function"):
        build_bm25_index(all_chunks, level, persist_dir=persist_dir)

    # ------------------------------------------------------------------
    # Step 6: Build semantic indexes (4 levels)
    # ------------------------------------------------------------------
    total_embedding_batches = 0
    for level in ("directory", "file", "class", "function"):
        level_chunk_count = len([c for c in all_chunks if c.level == level])
        if level_chunk_count > 0:
            build_semantic_index(all_chunks, level, persist_dir=persist_dir)
            # Approximate batch count for logging
            from Oracle.config import EMBEDDING_BATCH_SIZE
            total_embedding_batches += (level_chunk_count + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    # ------------------------------------------------------------------
    # Step 7: Persist vocabulary
    # ------------------------------------------------------------------
    vocabulary = build_vocabulary(all_chunks, persist_dir=persist_dir)

    # ------------------------------------------------------------------
    # Step 8: Persist symbols
    # ------------------------------------------------------------------
    symbols: set[str] = set()
    for c in all_chunks:
        if c.function_name:
            symbols.add(c.function_name)
        if c.class_name:
            symbols.add(c.class_name)

    symbols_path = os.path.join(persist_dir, "symbols.pkl")
    with open(symbols_path, "wb") as f:
        pickle.dump(symbols, f)

    logger.info(
        "symbols_persisted",
        symbols_count=len(symbols),
        file_path=symbols_path,
    )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    duration_ms = (time.perf_counter() - pipeline_start) * 1000.0

    summary = {
        "repo_path": repo_path,
        "trace_id": trace_id,
        "total_chunks": level_counts,
        "total_llm_calls": total_llm_calls,
        "total_embedding_batches": total_embedding_batches,
        "vocabulary_size": len(vocabulary),
        "symbols_count": len(symbols),
        "duration_ms": round(duration_ms, 2),
    }

    logger.info(
        "indexing_completed",
        repo_path=repo_path,
        trace_id=trace_id,
        total_chunks=level_counts,
        total_llm_calls=total_llm_calls,
        total_embedding_batches=total_embedding_batches,
        vocabulary_size=len(vocabulary),
        symbols_count=len(symbols),
        duration_ms=round(duration_ms, 2),
    )

    unbind_trace_id()
    return summary
