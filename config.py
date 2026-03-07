"""
Oracle configuration.

All settings with sensible defaults. Override via environment variables.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_BACKEND: str = os.environ.get("LLM_BACKEND", "ollama")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
CLAUDE_MODEL: str = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# ---------------------------------------------------------------------------
# Embeddings + Storage
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR: str = os.environ.get("CHROMA_PERSIST_DIR", ".oracle_db")
EMBEDDING_BATCH_SIZE: int = int(os.environ.get("EMBEDDING_BATCH_SIZE", "64"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
BM25_WEIGHT_DEFAULT: float = float(os.environ.get("BM25_WEIGHT_DEFAULT", "0.6"))
SEMANTIC_WEIGHT_DEFAULT: float = float(os.environ.get("SEMANTIC_WEIGHT_DEFAULT", "0.4"))
RRF_K: int = int(os.environ.get("RRF_K", "60"))
TOP_K_MIN: int = int(os.environ.get("TOP_K_MIN", "5"))
TOP_K_MAX: int = int(os.environ.get("TOP_K_MAX", "20"))

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
SUMMARIZE_MIN_LINES: int = int(os.environ.get("SUMMARIZE_MIN_LINES", "150"))
MIN_DIRECTORY_FILES: int = int(os.environ.get("MIN_DIRECTORY_FILES", "3"))
MAX_DIRECTORY_FILES: int = int(os.environ.get("MAX_DIRECTORY_FILES", "20"))

# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------
WRITE_MODE_ENABLED: bool = os.environ.get("WRITE_MODE_ENABLED", "False").lower() == "true"
TEST_TIMEOUT_SECONDS: int = int(os.environ.get("TEST_TIMEOUT_SECONDS", "120"))

# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES: list[str] = ["python", "javascript", "typescript"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT: str = os.environ.get("LOG_FORMAT", "pretty")
LOG_FILE: str | None = os.environ.get("LOG_FILE", None)
LOG_LLM_PROMPTS: bool = os.environ.get("LOG_LLM_PROMPTS", "False").lower() == "true"
