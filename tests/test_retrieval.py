"""
Tests for BM25 retriever, semantic retriever, indexer, and supporting utilities.

Groups:
  1. Tokenization (camel_snake_tokenize from technicality_scorer)
  2. Dynamic top_k
  3. BM25 retriever (build, retrieve, persistence, logging)
  4. Vocabulary and symbols
  5. Semantic retriever (build, retrieve, metadata roundtrip)
  6. ChunkMetadata fields
  7. Indexer integration
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import structlog
import structlog.testing

from Oracle.ingestion.ast_chunker import ChunkMetadata, chunk_file
from Oracle.llm.schemas import FunctionSummary
from Oracle.retrieval.technicality_scorer import camel_snake_tokenize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_structlog():
    """Reset structlog between tests so capture_logs works cleanly."""
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


@pytest.fixture(autouse=True)
def _reset_bm25_state():
    """Clear BM25 module-level state between tests."""
    from Oracle.retrieval import bm25_retriever
    bm25_retriever._bm25_indexes.clear()
    bm25_retriever._bm25_corpus_ids.clear()
    bm25_retriever._bm25_chunk_map.clear()
    yield
    bm25_retriever._bm25_indexes.clear()
    bm25_retriever._bm25_corpus_ids.clear()
    bm25_retriever._bm25_chunk_map.clear()


@pytest.fixture
def tmp_persist_dir(tmp_path):
    """Provide a temp directory for BM25/ChromaDB persistence."""
    d = tmp_path / ".oracle_db"
    d.mkdir()
    return str(d)


def _make_chunk(
    chunk_id: str,
    level: str,
    content: str,
    file_path: str = "test.py",
    directory_path: str = "",
    class_name: str | None = None,
    function_name: str | None = None,
) -> ChunkMetadata:
    """Helper to build a ChunkMetadata with minimal boilerplate."""
    return ChunkMetadata(
        chunk_id=chunk_id,
        level=level,
        file_path=file_path,
        directory_path=directory_path,
        language="python",
        class_name=class_name,
        function_name=function_name,
        start_line=1,
        end_line=10,
        parent_directory_chunk_id=None,
        parent_file_chunk_id=f"{file_path}::None::None::file" if level != "file" and level != "directory" else None,
        parent_class_chunk_id=None,
        content=content,
    )


@pytest.fixture
def sample_chunks() -> list[ChunkMetadata]:
    """A minimal set of chunks representing a tiny repo."""
    return [
        _make_chunk(
            "services/::None::None::directory", "directory",
            "Directory: services/\nFiles: [auth.py, user.py]\ndefines login, UserService, create_user",
            file_path="None", directory_path="services",
        ),
        _make_chunk(
            "services/auth.py::None::None::file", "file",
            "from passlib import hash\n\ndef login(username: str, password: str) -> bool",
            file_path="services/auth.py", directory_path="services",
        ),
        _make_chunk(
            "services/user.py::None::None::file", "file",
            "from sqlalchemy import Column\n\nclass UserService\ndef get_user(user_id: int) -> dict",
            file_path="services/user.py", directory_path="services",
        ),
        _make_chunk(
            "services/user.py::UserService::None::class", "class",
            "class UserService:\n    def create_user(self, name: str) -> dict\n    def delete_user(self, user_id: int) -> None",
            file_path="services/user.py", directory_path="services",
            class_name="UserService",
        ),
        _make_chunk(
            "services/auth.py::None::login::function", "function",
            "def login(username: str, password: str) -> bool:\n    return verify_password(username, password)",
            file_path="services/auth.py", directory_path="services",
            function_name="login",
        ),
        _make_chunk(
            "services/user.py::UserService::create_user::function", "function",
            "def create_user(self, name: str) -> dict:\n    user = User(name=name)\n    db.session.add(user)\n    return user.to_dict()",
            file_path="services/user.py", directory_path="services",
            class_name="UserService", function_name="create_user",
        ),
        _make_chunk(
            "services/user.py::UserService::delete_user::function", "function",
            "def delete_user(self, user_id: int) -> None:\n    user = db.session.get(User, user_id)\n    db.session.delete(user)",
            file_path="services/user.py", directory_path="services",
            class_name="UserService", function_name="delete_user",
        ),
        _make_chunk(
            "services/user.py::None::get_user::function", "function",
            "def get_user(user_id: int) -> dict:\n    return db.session.get(User, user_id).to_dict()",
            file_path="services/user.py", directory_path="services",
            function_name="get_user",
        ),
    ]


@pytest.fixture
def tiny_repo(tmp_path):
    """Create a minimal Python repo for indexer integration tests."""
    svc = tmp_path / "services"
    svc.mkdir()
    (svc / "__init__.py").write_text("")
    (svc / "auth.py").write_text(
        "def login(username: str, password: str) -> bool:\n"
        "    return username == 'admin'\n"
    )
    (svc / "user.py").write_text(
        "class UserService:\n"
        "    def create_user(self, name: str) -> dict:\n"
        "        return {'name': name}\n"
        "\n"
        "    def delete_user(self, user_id: int) -> None:\n"
        "        pass\n"
    )
    return str(tmp_path)


# ===========================================================================
# Group 1: Tokenization
# ===========================================================================

class TestTokenization:

    def test_camel_case_tokenized(self) -> None:
        assert set(camel_snake_tokenize("getUserById")) >= {"get", "user", "by", "id"}

    def test_snake_case_tokenized(self) -> None:
        assert set(camel_snake_tokenize("get_user_by_id")) >= {"get", "user", "by", "id"}

    def test_both_forms_same_tokens(self) -> None:
        assert camel_snake_tokenize("getUserById") == camel_snake_tokenize("get_user_by_id")


# ===========================================================================
# Group 2: Dynamic top_k
# ===========================================================================

class TestDynamicTopK:

    def test_dynamic_top_k(self) -> None:
        from Oracle.retrieval.bm25_retriever import compute_top_k

        assert compute_top_k(100) == 5      # floor
        assert compute_top_k(1000) == 10    # 1%
        assert compute_top_k(5000) == 20    # cap
        assert compute_top_k(2500) == 20    # cap
        assert compute_top_k(50) == 5       # floor


# ===========================================================================
# Group 3: BM25 retriever
# ===========================================================================

class TestBM25Retriever:

    def test_build_and_retrieve_function_level(self, sample_chunks, tmp_persist_dir) -> None:
        from Oracle.retrieval.bm25_retriever import build_bm25_index, bm25_retrieve

        build_bm25_index(sample_chunks, "function", persist_dir=tmp_persist_dir)
        results = bm25_retrieve("create user", "function", top_k=5, persist_dir=tmp_persist_dir)

        assert len(results) > 0
        chunk_ids = [r.chunk_id for r in results]
        assert "services/user.py::UserService::create_user::function" in chunk_ids

    def test_bm25_retrieve_empty_for_missing_level(self, sample_chunks, tmp_persist_dir) -> None:
        from Oracle.retrieval.bm25_retriever import build_bm25_index, bm25_retrieve

        build_bm25_index(sample_chunks, "function", persist_dir=tmp_persist_dir)
        # Query at class level — no index built for it
        results = bm25_retrieve("create user", "class", top_k=5, persist_dir=tmp_persist_dir)
        # Should still work (auto-loads from disk if available, returns empty if not)
        # Since we only built function, class should have been auto-loaded or empty
        # Class index exists in sample_chunks, but we didn't build it
        assert isinstance(results, list)

    def test_bm25_index_persistence(self, sample_chunks, tmp_persist_dir) -> None:
        from Oracle.retrieval.bm25_retriever import (
            build_bm25_index, bm25_retrieve, load_bm25_index,
            _bm25_indexes, _bm25_corpus_ids, _bm25_chunk_map,
        )

        build_bm25_index(sample_chunks, "function", persist_dir=tmp_persist_dir)

        # Verify pkl file exists
        pkl_path = os.path.join(tmp_persist_dir, "bm25_function.pkl")
        assert os.path.exists(pkl_path)

        # Clear module state
        _bm25_indexes.clear()
        _bm25_corpus_ids.clear()
        _bm25_chunk_map.clear()

        # Reload from disk
        loaded = load_bm25_index("function", tmp_persist_dir)
        assert loaded is True

        # Verify retrieval still works
        results = bm25_retrieve("create user", "function", top_k=5, persist_dir=tmp_persist_dir)
        assert len(results) > 0

    def test_bm25_retrieve_logs_emitted(self, sample_chunks, tmp_persist_dir) -> None:
        from Oracle.retrieval.bm25_retriever import build_bm25_index, bm25_retrieve

        build_bm25_index(sample_chunks, "function", persist_dir=tmp_persist_dir)

        with structlog.testing.capture_logs() as logs:
            bm25_retrieve("login password", "function", top_k=5, persist_dir=tmp_persist_dir)

        events = [l for l in logs if l.get("event") == "bm25_retrieve"]
        assert len(events) == 1
        assert "query_tokens" in events[0]
        assert events[0]["level"] == "function"
        assert "results_returned" in events[0]

    def test_bm25_index_built_logs(self, sample_chunks, tmp_persist_dir) -> None:
        from Oracle.retrieval.bm25_retriever import build_bm25_index

        with structlog.testing.capture_logs() as logs:
            build_bm25_index(sample_chunks, "function", persist_dir=tmp_persist_dir)

        events = [l for l in logs if l.get("event") == "bm25_index_built"]
        assert len(events) == 1
        assert events[0]["level"] == "function"
        assert "corpus_size" in events[0]
        assert "vocabulary_size" in events[0]
        assert "duration_ms" in events[0]

    def test_bm25_imports_camel_snake_tokenize(self) -> None:
        """Verify bm25_retriever imports camel_snake_tokenize, not defines it."""
        import inspect
        import Oracle.retrieval.bm25_retriever as bm25
        source = inspect.getsource(bm25)
        assert "def camel_snake_tokenize" not in source


# ===========================================================================
# Group 4: Vocabulary and symbols
# ===========================================================================

class TestVocabularyAndSymbols:

    def test_vocabulary_contains_tokenized_content(self, sample_chunks, tmp_persist_dir) -> None:
        from Oracle.retrieval.bm25_retriever import build_vocabulary

        vocab = build_vocabulary(sample_chunks, persist_dir=tmp_persist_dir)
        # "create_user" content should produce "create" and "user" tokens
        assert "create" in vocab
        assert "user" in vocab

    def test_vocabulary_persistence(self, sample_chunks, tmp_persist_dir) -> None:
        from Oracle.retrieval.bm25_retriever import build_vocabulary, load_vocabulary

        original = build_vocabulary(sample_chunks, persist_dir=tmp_persist_dir)

        vocab_path = os.path.join(tmp_persist_dir, "vocabulary.pkl")
        assert os.path.exists(vocab_path)

        loaded = load_vocabulary(tmp_persist_dir)
        assert loaded == original

    def test_symbols_collection(self, sample_chunks) -> None:
        """Verify function_name and class_name extracted, None excluded."""
        symbols: set[str] = set()
        for c in sample_chunks:
            if c.function_name:
                symbols.add(c.function_name)
            if c.class_name:
                symbols.add(c.class_name)

        assert "login" in symbols
        assert "create_user" in symbols
        assert "delete_user" in symbols
        assert "get_user" in symbols
        assert "UserService" in symbols
        # None should not be in the set
        assert None not in symbols

    def test_symbols_includes_methods_and_classes(self, sample_chunks) -> None:
        symbols: set[str] = set()
        for c in sample_chunks:
            if c.function_name:
                symbols.add(c.function_name)
            if c.class_name:
                symbols.add(c.class_name)

        # Both class names and method names
        assert {"UserService", "create_user", "delete_user"} <= symbols


# ===========================================================================
# Group 5: Semantic retriever
# ===========================================================================

class TestSemanticRetriever:

    @pytest.mark.slow
    def test_build_and_retrieve_semantic(self, sample_chunks, tmp_persist_dir) -> None:
        from Oracle.retrieval.semantic_retriever import build_semantic_index, semantic_retrieve

        build_semantic_index(sample_chunks, "function", persist_dir=tmp_persist_dir)
        results = semantic_retrieve("create a new user", "function", top_k=5, persist_dir=tmp_persist_dir)

        assert len(results) > 0
        # create_user should be among top results for this query
        chunk_ids = [r.chunk_id for r in results]
        assert "services/user.py::UserService::create_user::function" in chunk_ids

    @pytest.mark.slow
    def test_semantic_metadata_roundtrip(self, sample_chunks, tmp_persist_dir) -> None:
        from Oracle.retrieval.semantic_retriever import build_semantic_index, semantic_retrieve

        build_semantic_index(sample_chunks, "function", persist_dir=tmp_persist_dir)
        results = semantic_retrieve("login password", "function", top_k=5, persist_dir=tmp_persist_dir)

        assert len(results) > 0
        result = results[0]
        # Verify ChunkMetadata fields survived serialization
        assert result.level == "function"
        assert result.language == "python"
        assert isinstance(result.chunk_id, str)
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    @pytest.mark.slow
    def test_none_fields_survive_chromadb(self, sample_chunks, tmp_persist_dir) -> None:
        from Oracle.retrieval.semantic_retriever import build_semantic_index, semantic_retrieve

        # The login function has class_name=None
        build_semantic_index(sample_chunks, "function", persist_dir=tmp_persist_dir)
        results = semantic_retrieve("login", "function", top_k=5, persist_dir=tmp_persist_dir)

        login_results = [r for r in results if r.function_name == "login"]
        assert len(login_results) > 0
        assert login_results[0].class_name is None

    @pytest.mark.slow
    def test_semantic_retrieve_logs(self, sample_chunks, tmp_persist_dir) -> None:
        from Oracle.retrieval.semantic_retriever import build_semantic_index, semantic_retrieve

        build_semantic_index(sample_chunks, "function", persist_dir=tmp_persist_dir)

        with structlog.testing.capture_logs() as logs:
            semantic_retrieve("login", "function", top_k=5, persist_dir=tmp_persist_dir)

        events = [l for l in logs if l.get("event") == "semantic_retrieve"]
        assert len(events) == 1
        assert "query_text_length" in events[0]
        assert events[0]["level"] == "function"
        assert "results_returned" in events[0]


# ===========================================================================
# Group 6: ChunkMetadata fields
# ===========================================================================

class TestChunkMetadataFields:

    def test_chunk_metadata_has_summarization_fields(self) -> None:
        chunk = _make_chunk("test::None::None::file", "file", "test content")
        assert chunk.is_summarized is False
        assert chunk.full_content is None
        assert chunk.original_line_count is None

    def test_short_function_not_flagged(self) -> None:
        code = (
            "def get_user(user_id: int) -> dict:\n"
            "    return db.query(user_id)\n"
        )
        chunks = chunk_file("test.py", code, "python")
        func_chunks = [c for c in chunks if c.level == "function"]
        assert len(func_chunks) == 1
        assert func_chunks[0].original_line_count is None
        assert func_chunks[0].is_summarized is False

    def test_long_function_flagged_for_summarization(self) -> None:
        long_func = "def big_function():\n" + "    x = 1\n" * 200
        chunks = chunk_file("big.py", long_func, "python")
        func_chunks = [c for c in chunks if c.level == "function"]
        assert len(func_chunks) == 1
        assert func_chunks[0].original_line_count is not None
        assert func_chunks[0].original_line_count >= 150
        # Content is full body, not truncated
        assert "def big_function" in func_chunks[0].content
        assert "[TRUNCATED" not in func_chunks[0].content
        # Not summarized yet — that's the indexer's job
        assert func_chunks[0].is_summarized is False
        assert func_chunks[0].full_content is None


# ===========================================================================
# Group 7: Indexer integration
# ===========================================================================

class TestIndexerIntegration:

    def test_indexer_walks_and_chunks(self, tiny_repo, tmp_persist_dir, monkeypatch) -> None:
        monkeypatch.setattr("Oracle.ingestion.indexer.CHROMA_PERSIST_DIR", tmp_persist_dir)

        from Oracle.ingestion.indexer import index_repo

        mock_llm = MagicMock()
        summary = index_repo(tiny_repo, llm_client=mock_llm)

        assert summary["total_chunks"]["file"] >= 2
        assert summary["total_chunks"]["function"] >= 3  # login, create_user, delete_user
        assert summary["vocabulary_size"] > 0
        assert summary["symbols_count"] > 0

        # BM25 indexes exist
        assert os.path.exists(os.path.join(tmp_persist_dir, "bm25_function.pkl"))
        assert os.path.exists(os.path.join(tmp_persist_dir, "vocabulary.pkl"))
        assert os.path.exists(os.path.join(tmp_persist_dir, "symbols.pkl"))

    def test_symbols_persistence(self, tiny_repo, tmp_persist_dir, monkeypatch) -> None:
        monkeypatch.setattr("Oracle.ingestion.indexer.CHROMA_PERSIST_DIR", tmp_persist_dir)

        from Oracle.ingestion.indexer import index_repo

        mock_llm = MagicMock()
        index_repo(tiny_repo, llm_client=mock_llm)

        symbols_path = os.path.join(tmp_persist_dir, "symbols.pkl")
        assert os.path.exists(symbols_path)

        with open(symbols_path, "rb") as f:
            symbols = pickle.load(f)

        assert isinstance(symbols, set)
        assert {"login", "UserService"} <= symbols

    def test_long_function_gets_summarized(self, tmp_path, tmp_persist_dir, monkeypatch) -> None:
        monkeypatch.setattr("Oracle.ingestion.indexer.CHROMA_PERSIST_DIR", tmp_persist_dir)

        # Create a file with a 200-line function
        long_func = "def big_function():\n" + "    x = 1\n" * 200
        (tmp_path / "big.py").write_text(long_func)

        from Oracle.ingestion.indexer import index_repo

        mock_summary = FunctionSummary(summary="This function assigns x = 1 repeatedly for 200 lines.")
        mock_llm = MagicMock()
        mock_llm.complete.return_value = mock_summary
        mock_llm.backend = "ollama"
        mock_llm.model = "test-model"

        summary = index_repo(str(tmp_path), llm_client=mock_llm)

        assert summary["total_llm_calls"] == 1
        mock_llm.complete.assert_called_once()

    def test_indexing_logs_completion_summary(self, tiny_repo, tmp_persist_dir, monkeypatch) -> None:
        monkeypatch.setattr("Oracle.ingestion.indexer.CHROMA_PERSIST_DIR", tmp_persist_dir)

        from Oracle.ingestion.indexer import index_repo

        mock_llm = MagicMock()

        with structlog.testing.capture_logs() as logs:
            index_repo(tiny_repo, llm_client=mock_llm)

        completed = [l for l in logs if l.get("event") == "indexing_completed"]
        assert len(completed) == 1
        assert "total_chunks" in completed[0]
        assert "duration_ms" in completed[0]
        assert "vocabulary_size" in completed[0]
        assert "symbols_count" in completed[0]
