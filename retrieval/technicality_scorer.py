"""
Technicality scorer for Oracle query routing.

Produces a float 0.0–1.0 representing how technical (vs semantic) the query is.
This score determines BM25/semantic weighting for all retrieval calls.

Also contains camel_snake_tokenize — the shared tokenizer used by BM25 retrieval
and vocabulary overlap scoring.
"""

from __future__ import annotations

import re

from Oracle.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer (shared with BM25 retriever)
# ---------------------------------------------------------------------------

def camel_snake_tokenize(text: str) -> list[str]:
    """
    Split text on camelCase, snake_case, and non-alphanumeric boundaries.

    Examples:
        getUserById   → ["get", "user", "by", "id"]
        get_user_by_id → ["get", "user", "by", "id"]
        UserService   → ["user", "service"]
        create_user   → ["create", "user"]
    """
    # Step 1: Insert space before uppercase preceded by lowercase (camelCase split)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    # Step 2: Replace underscores and all non-alphanumeric chars with spaces
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    # Step 3: Lowercase, split, filter empty strings and single chars
    return [t.lower() for t in text.split() if len(t) > 1]


# ---------------------------------------------------------------------------
# Technicality scoring
# ---------------------------------------------------------------------------

def score_technicality(
    parsed_query: object,
    index_vocabulary: set[str],
) -> float:
    """
    Score how technical a parsed query is on a 0.0–1.0 scale.

    Three signals weighted 0.35 / 0.45 / 0.20:
      s1 — Code identifier present in extracted_identifiers
      s2 — File path or file attachment present
      s3 — Vocabulary overlap between query tokens and index vocabulary

    Parameters
    ----------
    parsed_query:
        A ParsedQuery instance (or any object with extracted_identifiers,
        extracted_file_paths, attached_file_path, natural_language attributes).
    index_vocabulary:
        Set of all tokens from camel_snake_tokenize across all indexed chunks.
    """
    # Signal 1: Code identifier present (weight 0.35)
    has_identifier = len(parsed_query.extracted_identifiers) > 0
    s1 = 1.0 if has_identifier else 0.0

    # Signal 2: File path or file attachment present (weight 0.45)
    has_file = (
        len(parsed_query.extracted_file_paths) > 0
        or parsed_query.attached_file_path is not None
    )
    s2 = 1.0 if has_file else 0.0

    # Signal 3: Vocabulary overlap with index (weight 0.20)
    all_tokens = camel_snake_tokenize(parsed_query.natural_language)
    if all_tokens:
        overlap = len(set(all_tokens) & index_vocabulary) / len(all_tokens)
    else:
        overlap = 0.0
    s3 = overlap

    score = (s1 * 0.35) + (s2 * 0.45) + (s3 * 0.20)

    bm25_weight = 0.3 + (0.5 * score)
    semantic_weight = 1.0 - bm25_weight

    logger.info(
        "technicality_scored",
        s1_identifier=s1,
        s2_file_path=s2,
        s3_vocab_overlap=round(s3, 3),
        final_score=round(score, 3),
        bm25_weight=round(bm25_weight, 3),
        semantic_weight=round(semantic_weight, 3),
    )

    return score


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def compute_retrieval_weights(technicality: float) -> tuple[float, float]:
    """
    Convert technicality score to (bm25_weight, semantic_weight).

    bm25_weight ranges from 0.30 (technicality=0) to 0.80 (technicality=1).
    semantic_weight = 1.0 - bm25_weight.
    """
    bm25_weight = 0.3 + (0.5 * technicality)
    semantic_weight = 1.0 - bm25_weight
    return bm25_weight, semantic_weight
