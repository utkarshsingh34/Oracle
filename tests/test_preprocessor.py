"""
Tests for query_preprocessor and technicality_scorer modules.

Organized in four groups:
  1. Core extraction (identifiers, file paths, code blocks)
  2. Two-stage identifier filter
  3. Technicality scorer
  4. Log assertions
"""

from __future__ import annotations

import structlog
import structlog.testing
import pytest

from Oracle.retrieval.query_preprocessor import preprocess_query, ParsedQuery
from Oracle.retrieval.technicality_scorer import (
    score_technicality,
    compute_retrieval_weights,
    camel_snake_tokenize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_structlog():
    """Reset structlog between tests so capture_logs works cleanly."""
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


# ===========================================================================
# Group 1: Core extraction
# ===========================================================================

class TestCoreExtraction:

    def test_identifier_extraction_camelcase(self) -> None:
        p = preprocess_query("what does getUserById return?")
        assert "getUserById" in p.extracted_identifiers

    def test_identifier_extraction_snake_case(self) -> None:
        p = preprocess_query("what does create_user return?")
        assert "create_user" in p.extracted_identifiers

    def test_file_path_extraction(self) -> None:
        p = preprocess_query("in auth.py what does login do?")
        assert "auth.py" in p.extracted_file_paths

    def test_code_block_detection(self) -> None:
        query = "explain this:\n```python\ndef foo(): pass\n```"
        p = preprocess_query(query)
        assert p.has_code_block
        assert p.pasted_code is not None
        assert "foo" in p.extracted_identifiers

    def test_code_block_language_hint(self) -> None:
        query = "look:\n```typescript\nconst bar = 1;\n```"
        p = preprocess_query(query)
        assert p.pasted_code_language == "typescript"

    def test_file_path_with_directories(self) -> None:
        p = preprocess_query("in services/auth.py what does login do?")
        assert "services/auth.py" in p.extracted_file_paths

    def test_structural_keywords_extracted(self) -> None:
        p = preprocess_query("what methods does the class expose?")
        assert "method" in p.extracted_keywords or "class" in p.extracted_keywords

    def test_is_mixed_true_for_code_and_nl(self) -> None:
        query = "explain this code:\n```python\nx = 1\n```"
        p = preprocess_query(query)
        assert p.is_mixed is True

    def test_is_mixed_false_for_pure_nl(self) -> None:
        p = preprocess_query("how does authentication work?")
        assert p.is_mixed is False

    def test_empty_query(self) -> None:
        p = preprocess_query("")
        assert p.extracted_identifiers == []
        assert p.extracted_file_paths == []
        assert p.has_code_block is False


# ===========================================================================
# Group 2: Two-stage identifier filter
# ===========================================================================

class TestTwoStageFilter:

    def test_plain_english_not_extracted_without_index(self) -> None:
        p = preprocess_query(
            "where is the thing that generates accounts?",
            index_symbols=None,
        )
        assert len(p.extracted_identifiers) == 0

    def test_single_word_recognized_via_index(self) -> None:
        p = preprocess_query(
            "what does login do?",
            index_symbols={"login", "dispatch", "render"},
        )
        assert "login" in p.extracted_identifiers

    def test_single_word_ignored_without_index(self) -> None:
        p = preprocess_query(
            "what does login do?",
            index_symbols=None,
        )
        assert "login" not in p.extracted_identifiers

    def test_structural_patterns_all_forms(self) -> None:
        query = "getUserById create_user UserService MAX_RETRIES parseJSON"
        p = preprocess_query(query, index_symbols=None)
        for name in ["getUserById", "create_user", "UserService", "MAX_RETRIES", "parseJSON"]:
            assert name in p.extracted_identifiers, f"Missing {name}"

    def test_code_block_tokens_bypass_structural_check(self) -> None:
        query = "check:\n```python\ndispatch = render(value)\n```"
        p = preprocess_query(query, index_symbols=None)
        assert "dispatch" in p.extracted_identifiers
        assert "render" in p.extracted_identifiers
        assert "value" in p.extracted_identifiers

    def test_mixed_query_separates_code_and_nl(self) -> None:
        query = "explain dispatch in\n```python\nresult = dispatch(event)\n```"
        p = preprocess_query(query, index_symbols=None)
        # "dispatch" from NL fails structural check (no index),
        # but "dispatch" from code block passes → still in identifiers
        assert "dispatch" in p.extracted_identifiers

    def test_stop_words_always_filtered(self) -> None:
        query = "```python\ndef return import class\n```"
        p = preprocess_query(query)
        for word in ["def", "return", "import", "class"]:
            assert word not in p.extracted_identifiers


# ===========================================================================
# Group 3: Technicality scorer
# ===========================================================================

class TestTechnicalityScorer:

    def test_camel_case_tokenized(self) -> None:
        assert set(camel_snake_tokenize("getUserById")) >= {"get", "user", "by", "id"}

    def test_snake_case_tokenized(self) -> None:
        assert set(camel_snake_tokenize("get_user_by_id")) >= {"get", "user", "by", "id"}

    def test_both_forms_same_tokens(self) -> None:
        assert camel_snake_tokenize("getUserById") == camel_snake_tokenize("get_user_by_id")

    def test_single_char_tokens_filtered(self) -> None:
        tokens = camel_snake_tokenize("a_b_c_longword")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "longword" in tokens

    def test_technicality_high_for_identifier_query(self) -> None:
        p = preprocess_query("what does create_user return?")
        # Vocabulary must cover most query tokens for s3 to push score above 0.5
        score = score_technicality(p, {"what", "does", "create", "user", "return"})
        assert score > 0.5

    def test_technicality_low_for_semantic_query(self) -> None:
        p = preprocess_query(
            "where is the thing that generates accounts?",
            index_symbols=None,
        )
        score = score_technicality(p, set())
        assert score < 0.3

    def test_technicality_maxes_with_file_path(self) -> None:
        p = preprocess_query(
            "in services/auth.py what does login do?",
            index_symbols={"login"},
        )
        score = score_technicality(p, {"login", "auth"})
        assert score > 0.8

    def test_compute_retrieval_weights_sum_to_one(self) -> None:
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            bm25_w, sem_w = compute_retrieval_weights(t)
            assert abs(bm25_w + sem_w - 1.0) < 0.001

    def test_compute_retrieval_weights_range(self) -> None:
        bm25_low, _ = compute_retrieval_weights(0.0)
        bm25_high, _ = compute_retrieval_weights(1.0)
        assert abs(bm25_low - 0.3) < 0.001
        assert abs(bm25_high - 0.8) < 0.001


# ===========================================================================
# Group 4: Log assertions
# ===========================================================================

class TestLogEvents:

    def test_preprocessor_logs_extracted_identifiers(self) -> None:
        with structlog.testing.capture_logs() as logs:
            preprocess_query("what does getUserById return?")
        events = [l for l in logs if l.get("event") == "query_preprocessed"]
        assert len(events) == 1
        assert "identifiers_extracted" in events[0]
        assert "getUserById" in events[0]["identifiers_extracted"]

    def test_preprocessor_logs_index_symbols_usage(self) -> None:
        with structlog.testing.capture_logs() as logs:
            preprocess_query("what does login do?", index_symbols={"login"})
        events = [l for l in logs if l.get("event") == "query_preprocessed"]
        assert len(events) == 1
        assert events[0]["used_index_symbols"] is True

    def test_preprocessor_logs_no_index_symbols(self) -> None:
        with structlog.testing.capture_logs() as logs:
            preprocess_query("what does login do?")
        events = [l for l in logs if l.get("event") == "query_preprocessed"]
        assert len(events) == 1
        assert events[0]["used_index_symbols"] is False

    def test_technicality_logs_all_signals(self) -> None:
        p = preprocess_query("what does create_user return?")
        with structlog.testing.capture_logs() as logs:
            score_technicality(p, {"create", "user"})
        events = [l for l in logs if l.get("event") == "technicality_scored"]
        assert len(events) == 1
        for field in ["s1_identifier", "s2_file_path", "s3_vocab_overlap",
                       "final_score", "bm25_weight", "semantic_weight"]:
            assert field in events[0], f"Missing log field: {field}"
