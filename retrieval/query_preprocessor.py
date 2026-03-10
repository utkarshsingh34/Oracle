"""
Query preprocessor for Oracle retrieval pipeline.

Every query enters this preprocessor before any routing or retrieval.
No raw string ever enters the pipeline directly.

Usage:
    from Oracle.retrieval.query_preprocessor import preprocess_query, ParsedQuery

    parsed = preprocess_query("what does getUserById return?")
    parsed = preprocess_query("what does login do?", index_symbols={"login"})
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from Oracle.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ParsedQuery schema
# ---------------------------------------------------------------------------

class ParsedQuery(BaseModel):
    """Structured representation of a user query after preprocessing."""

    raw_input: str
    natural_language: str             # Query with code blocks removed
    attached_file_path: str | None    # If user attached a file (set by UI layer)
    attached_file_content: str | None
    pasted_code: str | None           # Detected code block content
    pasted_code_language: str | None
    extracted_identifiers: list[str]  # camelCase, snake_case, PascalCase names
    extracted_file_paths: list[str]   # anything matching *.py / *.js / *.ts etc
    extracted_keywords: list[str]     # "class", "function", "method", "file" etc
    has_code_block: bool
    is_mixed: bool                    # natural language + code both present


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOP_WORDS: set[str] = {
    "the", "and", "for", "with", "this", "that", "what", "how",
    "does", "from", "into", "about", "which", "when", "where",
    "def", "class", "return", "import", "function", "var", "let",
    "const", "type", "interface", "public", "private",
}

IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")

FILE_PATH_PATTERN = re.compile(r"\b[\w/]+\.(py|js|ts|go|java|rs)\b")

CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

STRUCTURAL_KEYWORDS: list[str] = [
    "class", "function", "method", "file", "module", "directory",
    "folder", "package", "struct", "interface", "import", "decorator",
    "async", "constructor", "property", "attribute", "variable",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_code_identifier(token: str) -> bool:
    """
    Check whether a token has code-like structural markers that plain
    English words lack.

    Catches: getUserById (camelCase), create_user (snake_case),
    UserService (PascalCase), MAX_RETRIES (SCREAMING_SNAKE), parseJSON.

    Does NOT catch: "generates", "accounts", "thing", "dispatcher".
    """
    # snake_case or SCREAMING_SNAKE_CASE
    if "_" in token:
        return True
    # camelCase: starts lowercase, has uppercase later
    if token[0].islower() and any(c.isupper() for c in token[1:]):
        return True
    # PascalCase: starts uppercase, has 2+ uppercase total (check token[2:])
    if token[0].isupper() and any(c.isupper() for c in token[2:]):
        return True
    return False


def _extract_code_blocks(query: str) -> tuple[str, str | None, str | None]:
    """
    Extract triple-backtick code blocks from the query.

    Returns (natural_language, pasted_code, pasted_code_language).
    Multiple code blocks are concatenated with newlines.
    """
    matches = list(CODE_BLOCK_PATTERN.finditer(query))
    if not matches:
        return query, None, None

    language: str | None = None
    code_parts: list[str] = []

    for m in matches:
        lang_hint = m.group(1).strip()
        code_content = m.group(2).strip()
        if lang_hint and language is None:
            language = lang_hint
        code_parts.append(code_content)

    # Remove code blocks from the query to get natural language portion
    nl = CODE_BLOCK_PATTERN.sub("", query).strip()
    pasted_code = "\n".join(code_parts)

    return nl, pasted_code, language


def _extract_identifiers_from_tokens(
    text: str,
    *,
    is_code_block: bool,
    index_symbols: set[str] | None,
) -> list[str]:
    """
    Extract identifier tokens from text using the two-stage filter.

    If is_code_block is True, all non-stop-word tokens are kept (bypass).
    Otherwise, tokens must pass Stage 1 (structural check) or Stage 2
    (index cross-reference).
    """
    raw_matches = IDENTIFIER_PATTERN.findall(text)
    identifiers: list[str] = []
    seen: set[str] = set()

    for token in raw_matches:
        if token.lower() in STOP_WORDS:
            continue
        if token in seen:
            continue

        if is_code_block:
            # Code block tokens bypass both stages
            identifiers.append(token)
            seen.add(token)
        elif _is_code_identifier(token):
            # Stage 1: structural pattern
            identifiers.append(token)
            seen.add(token)
        elif index_symbols is not None and token in index_symbols:
            # Stage 2: index cross-reference
            identifiers.append(token)
            seen.add(token)
        # else: plain English word, discard

    return identifiers


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess_query(
    query: str,
    index_symbols: set[str] | None = None,
) -> ParsedQuery:
    """
    Preprocess a raw user query into a structured ParsedQuery.

    Parameters
    ----------
    query:
        Raw user query string, possibly containing code blocks.
    index_symbols:
        Optional set of raw function/class names from the indexed codebase.
        When provided, single-word function names like "login" can be
        recognized via cross-reference even without structural code markers.
    """
    # Step 1: Extract code blocks
    natural_language, pasted_code, pasted_code_language = _extract_code_blocks(query)
    has_code_block = pasted_code is not None

    # Step 2: Extract file paths from full raw input
    extracted_file_paths = FILE_PATH_PATTERN.findall(query)
    # findall returns the capture group (extension) — re-extract full matches
    extracted_file_paths = [m.group() for m in FILE_PATH_PATTERN.finditer(query)]

    # Step 3: Extract structural keywords from natural language
    nl_lower = natural_language.lower()
    extracted_keywords = [kw for kw in STRUCTURAL_KEYWORDS if kw in nl_lower]

    # Step 4: Extract identifiers — two pools
    # Pool A: NL tokens (two-stage filter)
    nl_identifiers = _extract_identifiers_from_tokens(
        natural_language,
        is_code_block=False,
        index_symbols=index_symbols,
    )

    # Pool B: Code block tokens (bypass filter)
    code_identifiers: list[str] = []
    if pasted_code:
        code_identifiers = _extract_identifiers_from_tokens(
            pasted_code,
            is_code_block=True,
            index_symbols=index_symbols,
        )

    # Merge and deduplicate (preserving order)
    seen: set[str] = set()
    extracted_identifiers: list[str] = []
    for ident in nl_identifiers + code_identifiers:
        if ident not in seen:
            extracted_identifiers.append(ident)
            seen.add(ident)

    # Step 5: Determine if mixed
    is_mixed = bool(natural_language.strip()) and has_code_block

    parsed = ParsedQuery(
        raw_input=query,
        natural_language=natural_language,
        attached_file_path=None,
        attached_file_content=None,
        pasted_code=pasted_code,
        pasted_code_language=pasted_code_language,
        extracted_identifiers=extracted_identifiers,
        extracted_file_paths=extracted_file_paths,
        extracted_keywords=extracted_keywords,
        has_code_block=has_code_block,
        is_mixed=is_mixed,
    )

    logger.info(
        "query_preprocessed",
        raw_input_length=len(query),
        has_code_block=has_code_block,
        is_mixed=is_mixed,
        identifiers_extracted=extracted_identifiers,
        file_paths_extracted=extracted_file_paths,
        keywords_extracted=extracted_keywords,
        used_index_symbols=index_symbols is not None,
    )

    logger.debug("parsed_query_full", **parsed.model_dump())

    return parsed
