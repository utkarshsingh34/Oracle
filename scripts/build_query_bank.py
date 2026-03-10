"""
Build the static query bank for Oracle layer-2 query routing.

Downloads CodeSearchNet (Python split), CoSQA, and StaQC. Filters to natural
language description/question side. Embeds with all-MiniLM-L6-v2. For each
level (directory, file, class, function): filters to level-appropriate
descriptions, runs k-means sweeping k=50–1000, picks k at the elbow, takes
centroids as representative embeddings. Deduplicates across sources (cosine
similarity > 0.95 → keep higher quality source). Saves data/query_bank.npz.

This script runs standalone. It is NOT called at query time.

Usage:
    python -m scripts.build_query_bank
    python scripts/build_query_bank.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Logging — use structlog if available, else basic stderr
# ---------------------------------------------------------------------------
try:
    from Oracle.logging_config import get_logger
    logger = get_logger(__name__)

    def log_info(event: str, **kw: object) -> None:
        logger.info(event, **kw)

    def log_warning(event: str, **kw: object) -> None:
        logger.warning(event, **kw)

    def log_error(event: str, **kw: object) -> None:
        logger.error(event, **kw)

except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    _logger = logging.getLogger(__name__)

    def log_info(event: str, **kw: object) -> None:
        _logger.info("%s %s", event, kw)

    def log_warning(event: str, **kw: object) -> None:
        _logger.warning("%s %s", event, kw)

    def log_error(event: str, **kw: object) -> None:
        _logger.error("%s %s", event, kw)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "query_bank.npz"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
SOURCE_PRIORITY = {"cosqa": 0, "staqc": 1, "codesearchnet": 2}  # lower = higher quality
K_MIN = 50
K_MAX = 1000
K_STEP = 50
DEDUP_THRESHOLD = 0.95
LEVELS = ["directory", "file", "class", "function"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_codesearchnet_python() -> list[dict]:
    """
    Load CodeSearchNet Python split.
    Returns list of dicts with keys: text (NL description), source, level_hint.
    """
    log_info("dataset_download_started", dataset="codesearchnet_python")
    t0 = time.perf_counter()

    try:
        from datasets import load_dataset
        ds = load_dataset("code_search_net", "python", split="train", trust_remote_code=True)
    except Exception as exc:
        log_error("dataset_download_failed", dataset="codesearchnet_python", error=str(exc), exc_info=True)
        return []

    results: list[dict] = []
    for row in ds:
        docstring = (row.get("func_documentation_string") or "").strip()
        if len(docstring) < 20:
            continue

        # Heuristic level classification based on function context
        func_name = row.get("func_name", "")
        whole_func = row.get("whole_func_string", "")

        level = _classify_codesearchnet_level(func_name, whole_func, docstring)
        results.append({
            "text": docstring,
            "source": "codesearchnet",
            "level": level,
        })

    duration_ms = (time.perf_counter() - t0) * 1000
    log_info("dataset_loaded", dataset="codesearchnet_python",
             total_rows=len(ds), filtered_rows=len(results), duration_ms=round(duration_ms, 2))
    return results


def _classify_codesearchnet_level(func_name: str, whole_func: str, docstring: str) -> str:
    """Classify a CodeSearchNet entry to a structural level."""
    doc_lower = docstring.lower()
    func_lower = func_name.lower()

    # Class-level indicators
    if any(kw in doc_lower for kw in ["class for", "base class", "mixin", "abstract class",
                                        "class that", "class representing"]):
        return "class"

    # File/module-level indicators
    if any(kw in doc_lower for kw in ["module", "this file", "utilities for",
                                        "helper functions", "collection of"]):
        return "file"

    # Directory-level indicators (rare in CodeSearchNet)
    if any(kw in doc_lower for kw in ["package", "subsystem", "this directory"]):
        return "directory"

    # Method inside class
    if "self" in whole_func[:200] or "." in func_name:
        return "function"

    return "function"


def _load_cosqa() -> list[dict]:
    """
    Load CoSQA dataset (Microsoft — real Bing developer search queries).
    Returns list of dicts with keys: text, source, level.
    """
    log_info("dataset_download_started", dataset="cosqa")
    t0 = time.perf_counter()

    try:
        from datasets import load_dataset
        ds = load_dataset("neulab/cosqa", split="train", trust_remote_code=True)
    except Exception as exc:
        log_error("dataset_download_failed", dataset="cosqa", error=str(exc), exc_info=True)
        return []

    results: list[dict] = []
    for row in ds:
        query = (row.get("doc") or row.get("query") or "").strip()
        if len(query) < 10:
            continue

        level = _classify_query_level(query)
        results.append({
            "text": query,
            "source": "cosqa",
            "level": level,
        })

    duration_ms = (time.perf_counter() - t0) * 1000
    log_info("dataset_loaded", dataset="cosqa",
             total_rows=len(ds), filtered_rows=len(results), duration_ms=round(duration_ms, 2))
    return results


def _load_staqc() -> list[dict]:
    """
    Load StaQC dataset (Stack Overflow Q&A pairs).
    Returns list of dicts with keys: text, source, level.
    """
    log_info("dataset_download_started", dataset="staqc")
    t0 = time.perf_counter()

    try:
        from datasets import load_dataset
        ds = load_dataset("koutch/staqc", split="train", trust_remote_code=True)
    except Exception as exc:
        log_warning("dataset_download_failed_trying_alt", dataset="staqc", error=str(exc), exc_info=True)
        # StaQC may be under a different name; try alternate
        try:
            ds = load_dataset("thegenerality/staqc-py", split="train", trust_remote_code=True)
        except Exception as exc2:
            log_error("dataset_download_failed", dataset="staqc", error=str(exc2), exc_info=True)
            return []

    results: list[dict] = []
    for row in ds:
        question = (row.get("question") or row.get("title") or "").strip()
        if len(question) < 10:
            continue

        level = _classify_query_level(question)
        results.append({
            "text": question,
            "source": "staqc",
            "level": level,
        })

    duration_ms = (time.perf_counter() - t0) * 1000
    log_info("dataset_loaded", dataset="staqc",
             total_rows=len(ds), filtered_rows=len(results), duration_ms=round(duration_ms, 2))
    return results


def _classify_query_level(text: str) -> str:
    """Classify a natural language query/description to a structural level."""
    t = text.lower()

    if any(kw in t for kw in ["directory", "folder", "package", "module structure",
                                "project structure", "architecture"]):
        return "directory"
    if any(kw in t for kw in ["file", "module", "script", "import", "what is in"]):
        return "file"
    if any(kw in t for kw in ["class", "object", "instance", "inherit", "method of",
                                "attributes of", "properties of"]):
        return "class"
    return "function"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_texts(texts: list[str], batch_size: int = 256) -> NDArray[np.float32]:
    """Embed texts with all-MiniLM-L6-v2. Returns (N, 384) float32 array."""
    from sentence_transformers import SentenceTransformer

    log_info("embedding_started", total_texts=len(texts), batch_size=batch_size)
    t0 = time.perf_counter()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    duration_ms = (time.perf_counter() - t0) * 1000
    log_info("embedding_completed", total_texts=len(texts),
             shape=list(embeddings.shape), duration_ms=round(duration_ms, 2))

    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# K-means with elbow detection
# ---------------------------------------------------------------------------

def _find_elbow_k(embeddings: NDArray[np.float32], level: str) -> int:
    """
    Sweep k-means from K_MIN to K_MAX (or max feasible) and pick k at the elbow.

    The elbow is the point of maximum second derivative (largest decrease in
    the rate of inertia reduction).
    """
    from sklearn.cluster import MiniBatchKMeans

    n_samples = embeddings.shape[0]
    if n_samples < K_MIN:
        log_warning("too_few_samples_for_kmeans", level=level, n_samples=n_samples)
        return max(1, n_samples // 5)

    k_values = list(range(K_MIN, min(K_MAX, n_samples) + 1, K_STEP))
    if not k_values:
        k_values = [max(1, n_samples // 5)]

    inertias: list[float] = []
    log_info("kmeans_sweep_started", level=level, k_range=f"{k_values[0]}-{k_values[-1]}",
             n_samples=n_samples)

    for k in k_values:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024, n_init=3)
        km.fit(embeddings)
        inertias.append(km.inertia_)

    # Find elbow: point of maximum second derivative
    if len(inertias) < 3:
        best_k = k_values[0]
    else:
        second_deriv = []
        for i in range(1, len(inertias) - 1):
            d2 = inertias[i - 1] - 2 * inertias[i] + inertias[i + 1]
            second_deriv.append(d2)
        elbow_idx = int(np.argmax(second_deriv)) + 1  # +1 because second_deriv starts at index 1
        best_k = k_values[elbow_idx]

    log_info("kmeans_elbow_found", level=level, best_k=best_k,
             inertia_at_elbow=round(inertias[k_values.index(best_k)], 2))
    return best_k


def _run_kmeans_for_level(
    embeddings: NDArray[np.float32],
    texts: list[str],
    sources: list[str],
    level: str,
) -> tuple[NDArray[np.float32], list[str], list[str]]:
    """
    Run k-means for a single level.
    Returns (centroids, closest_phrases, centroid_sources).
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics.pairwise import cosine_similarity

    t0 = time.perf_counter()
    best_k = _find_elbow_k(embeddings, level)

    km = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=1024, n_init=3)
    km.fit(embeddings)
    centroids = km.cluster_centers_.astype(np.float32)

    # Normalize centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centroids = centroids / norms

    # Find closest actual phrase to each centroid (for debugging)
    sims = cosine_similarity(centroids, embeddings)  # (best_k, n_samples)
    closest_indices = sims.argmax(axis=1)
    closest_phrases = [texts[i] for i in closest_indices]
    centroid_sources = [sources[i] for i in closest_indices]

    duration_ms = (time.perf_counter() - t0) * 1000
    log_info("kmeans_completed", level=level, n_centroids=best_k,
             duration_ms=round(duration_ms, 2))

    return centroids, closest_phrases, centroid_sources


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate_across_levels(
    all_embeddings: NDArray[np.float32],
    all_labels: list[str],
    all_phrases: list[str],
    all_sources: list[str],
) -> tuple[NDArray[np.float32], list[str], list[str]]:
    """
    Remove near-duplicate centroids across levels.
    When cosine similarity > DEDUP_THRESHOLD, keep the one from the
    higher-quality source (CoSQA > StaQC > CodeSearchNet).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    t0 = time.perf_counter()
    n = all_embeddings.shape[0]

    if n == 0:
        return all_embeddings, all_labels, all_phrases

    # Compute pairwise similarities in batches to avoid memory issues
    keep = [True] * n
    batch_size = 5000

    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        for j in range(i, n, batch_size):
            end_j = min(j + batch_size, n)
            if j < i:
                continue
            sims = cosine_similarity(
                all_embeddings[i:end_i],
                all_embeddings[j:end_j],
            )
            for ri in range(sims.shape[0]):
                for rj in range(sims.shape[1]):
                    abs_i = i + ri
                    abs_j = j + rj
                    if abs_i >= abs_j:
                        continue
                    if not keep[abs_i] or not keep[abs_j]:
                        continue
                    if sims[ri, rj] > DEDUP_THRESHOLD:
                        # Keep higher quality source (lower priority number)
                        pri_i = SOURCE_PRIORITY.get(all_sources[abs_i], 99)
                        pri_j = SOURCE_PRIORITY.get(all_sources[abs_j], 99)
                        if pri_i <= pri_j:
                            keep[abs_j] = False
                        else:
                            keep[abs_i] = False

    mask = np.array(keep)
    deduped_embeddings = all_embeddings[mask]
    deduped_labels = [l for l, k in zip(all_labels, keep) if k]
    deduped_phrases = [p for p, k in zip(all_phrases, keep) if k]

    duration_ms = (time.perf_counter() - t0) * 1000
    log_info("deduplication_completed",
             before=n, after=int(mask.sum()),
             removed=n - int(mask.sum()),
             duration_ms=round(duration_ms, 2))

    return deduped_embeddings, deduped_labels, deduped_phrases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_query_bank() -> None:
    """Build the query bank end-to-end and save to data/query_bank.npz."""
    total_start = time.perf_counter()
    log_info("build_started", output_path=str(OUTPUT_PATH))

    # Step 1: Load datasets
    csn = _load_codesearchnet_python()
    cosqa = _load_cosqa()
    staqc = _load_staqc()

    all_entries = csn + cosqa + staqc
    if not all_entries:
        log_error("no_data_loaded", message="All dataset downloads failed. Cannot build query bank.")
        sys.exit(1)

    log_info("datasets_combined", total_entries=len(all_entries),
             codesearchnet=len(csn), cosqa=len(cosqa), staqc=len(staqc))

    # Step 2: Embed all texts
    texts = [e["text"] for e in all_entries]
    embeddings = _embed_texts(texts)

    # Step 3: Per-level k-means
    all_centroids: list[NDArray[np.float32]] = []
    all_labels: list[str] = []
    all_phrases: list[str] = []
    all_sources: list[str] = []

    for level in LEVELS:
        # Filter entries for this level
        level_mask = [e["level"] == level for e in all_entries]
        level_indices = [i for i, m in enumerate(level_mask) if m]

        if len(level_indices) < 10:
            log_warning("insufficient_data_for_level", level=level, count=len(level_indices))
            continue

        level_embeddings = embeddings[level_indices]
        level_texts = [texts[i] for i in level_indices]
        level_sources = [all_entries[i]["source"] for i in level_indices]

        log_info("level_processing", level=level, entries=len(level_indices))

        centroids, phrases, sources = _run_kmeans_for_level(
            level_embeddings, level_texts, level_sources, level,
        )

        all_centroids.append(centroids)
        all_labels.extend([level] * centroids.shape[0])
        all_phrases.extend(phrases)
        all_sources.extend(sources)

    if not all_centroids:
        log_error("no_centroids_produced", message="K-means produced no centroids for any level.")
        sys.exit(1)

    combined_embeddings = np.vstack(all_centroids)

    # Step 4: Deduplicate across levels
    final_embeddings, final_labels, final_phrases = _deduplicate_across_levels(
        combined_embeddings, all_labels, all_phrases, all_sources,
    )

    # Step 5: Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(OUTPUT_PATH),
        embeddings=final_embeddings,
        labels=np.array(final_labels),
        phrases=np.array(final_phrases),
    )

    total_duration_ms = (time.perf_counter() - total_start) * 1000
    level_counts = {}
    for level in LEVELS:
        level_counts[level] = sum(1 for l in final_labels if l == level)

    log_info("build_completed",
             output_path=str(OUTPUT_PATH),
             total_centroids=final_embeddings.shape[0],
             embedding_dim=final_embeddings.shape[1],
             level_counts=level_counts,
             file_size_mb=round(OUTPUT_PATH.stat().st_size / (1024 * 1024), 2),
             total_duration_ms=round(total_duration_ms, 2))


if __name__ == "__main__":
    build_query_bank()
