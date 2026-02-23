# Oracle — Codebase Q&A + Autonomous Issue Resolver

## What Oracle Is

Oracle ingests a GitHub repository, builds a four-level hierarchical AST-based
index, answers natural language questions about the codebase with cited
references to exact lines of code, and autonomously resolves GitHub issues by
generating a code patch, running the test suite, and opening a draft PR if
tests pass.

Oracle answers "what is in this codebase" and "how is it implemented."
It does not answer "why was this decision made" — that lives in commit history,
PR descriptions, and design documents which are not indexed.

Oracle runs entirely locally. No code, no embeddings, no queries leave the
machine. This is a deliberate design constraint, not a limitation.

---

## Lessons Carried Forward From Argus (Non-Negotiable)

Apply these everywhere without being told:

- **LangGraph state is always plain dicts.** Pydantic models written to
  LangGraph state must be `.model_dump()`'d. Nodes that read state reconstruct
  with `Model(**state["key"])` before passing to typed functions.

- **Sync functions stay sync.** Never rewrite tool functions as async. Wrap in
  `loop.run_in_executor(None, fn, *args)` with `asyncio.gather` for
  parallelism. Keeps tools simple and independently testable.

- **LLM clients are always injected as parameters.** Every function that calls
  an LLM accepts `llm_client: LLMClient = None` and instantiates a default
  only if None. No function patches globals to mock LLM behavior in tests.

- **Mock LLM calls return real Pydantic instances.** In tests, mock `.complete()`
  returns an instantiated Pydantic model, not a dict or MagicMock. Downstream
  code that accesses `.field_name` must work identically to production.

- **SSE streaming bypasses the orchestrator.** If streaming is added, call
  pipeline stages directly in an async generator. `app.invoke()` returns only
  final state with no mid-graph hooks.

- **Use `ddgs` not `duckduckgo-search`.** Old package returns empty results.
  `from ddgs import DDGS`. (Oracle doesn't search the web, but note if added.)

---

## Hard Rules (Never Violate These)

- NEVER chunk code by fixed token windows. ALL chunking uses tree-sitter AST
  parsing. This is the single most important architectural constraint.
- NEVER write to the repository being analyzed unless `WRITE_MODE_ENABLED=True`
  in config. Default is False.
- NEVER open a real PR without dry_run=False explicitly set by the caller.
  dry_run=True is the function signature default. Cannot be changed.
- NEVER skip the test suite before opening a PR. If no tests exist, report
  this and halt. Do not open a PR without test verification.
- ALL retrieval uses hybrid BM25 + semantic search merged with weighted RRF.
  Never use only one retrieval method.
- ALL LLM calls use structured Pydantic output via the `instructor` library.
  No free-form string parsing anywhere in the codebase.
- The Q&A layer must work without GitHub credentials (read-only local mode).
- Embeddings use `sentence-transformers` locally. No paid embedding APIs ever.
- NEVER raise exceptions from tool functions. Catch and return None or an
  error model. Exceptions propagate only from the orchestrator to the caller.
- ALL queries go through the preprocessor before touching retrieval. No raw
  string ever enters the routing or retrieval layer directly.
- NEVER use `print()` for diagnostic output. ALL diagnostic output goes through
  structlog. No exceptions.
- NEVER swallow exceptions silently. Every `except` block logs at minimum
  WARNING level with `logger.warning(..., exc_info=True)` before returning
  None or an error model. A bare `except: pass` is a build-failing offense.

---

## Logging Rules (Never Violate These)

Logging is not optional infrastructure — it is a first-class requirement of
every module. The ingestion and querying pipelines can break in dozens of
subtle ways (tree-sitter partial parses, LLM schema validation failures,
BM25 tokenization mismatches, ChromaDB collection drift) and without
structured logging, failures are invisible.

### Core Principles

- **All logging uses `structlog`** via `from oracle.logging_config import get_logger`.
  No stdlib `logging`, no `print()`, no `sys.stderr.write()` anywhere in the
  codebase. `structlog` produces key-value pairs that are machine-parseable
  (JSON in production) and human-readable (pretty-print in development).

- **Every pipeline entrypoint binds a `trace_id`.** The three entrypoints are
  `index_repo`, `answer_question`, and `resolve_issue`. Each generates a
  12-character hex trace ID at the start and binds it to structlog's context
  vars. Every downstream log event automatically includes this trace_id,
  allowing full reconstruction of any pipeline run from log output.

  ```python
  import structlog
  trace_id = uuid4().hex[:12]
  structlog.contextvars.bind_contextvars(trace_id=trace_id)
  logger.info("pipeline_started", pipeline="ingestion", repo_path=repo_path)
  # ... all downstream logger calls automatically carry trace_id
  structlog.contextvars.unbind_contextvars("trace_id")
  ```

- **Every LLM call logs**: backend, model, output schema name, latency_ms,
  retry_count, success boolean. On validation failure, log the full instructor
  validation error (tells you exactly which field the LLM got wrong). On final
  failure after retries, log at ERROR with the last validation error.

- **Every retrieval call logs**: query tokens (after camel_snake_tokenize),
  level queried, top_k used, result count returned, top-3 chunk IDs with
  their scores.

- **Tree-sitter parse anomalies are logged as warnings.** ERROR and MISSING
  nodes in the AST mean tree-sitter produced a partial parse — the tree built
  but contains garbage subtrees that produce bad chunks. Log file path, byte
  offset, and node type for every ERROR/MISSING node found. Never silently
  swallow partial parses.

- **Tool functions that return None on error** must log the caught exception
  at WARNING level before returning None. The caller sees None but the log
  shows why.

- **Full LLM prompts are logged only when `LOG_LLM_PROMPTS=True`** at DEBUG
  level. Default is False. Never log prompt text at INFO or above — prompts
  are large and clutter operational logs.

- **Timing is measured with `time.perf_counter()`** and logged as
  `duration_ms` (float, milliseconds). Not `time.time()` (wall clock, affected
  by NTP drift). Not `datetime.now()` deltas (insufficient precision).

### Log Level Policy

```
DEBUG   Detailed diagnostic data: full prompts, raw retrieval scores,
        intermediate parse trees, all chunk IDs in a result set.
        Extremely verbose. Only enable for targeted debugging.

INFO    Pipeline milestones and operational summaries: pipeline started/completed,
        indexing phase transitions, query received, answer returned,
        LLM call completed, retrieval completed.
        Default level for development. Every INFO log should be useful
        without context — it should tell you what happened.

WARNING Parse anomalies (tree-sitter ERROR nodes), zero retrieval results,
        LLM retry (not yet failed), chunk not found during parent expansion,
        original_code not found in apply_patch, missing test framework.
        Anything that is not an error YET but signals degraded behavior.

ERROR   LLM call failed after all retries, ChromaDB collection missing,
        indexing failed for a file, patch application failed,
        test runner subprocess crashed (not test failures — those are data).
        Something broke and the pipeline cannot produce correct output
        for this input.
```

### Per-Module Logging Requirements

What follows are the minimum required log events per module. Implementations
may add more. Omitting any of these is a build-failing offense.

**`ingestion/repo_loader.py`**
```
INFO  repo_walk_started     repo_path
INFO  repo_walk_completed   repo_path, total_files, skipped_files, languages={py: N, js: N, ts: N}, duration_ms
DEBUG file_skipped           file_path, reason (binary | extension | path_pattern | utf8_decode_failed)
```

**`ingestion/ast_chunker.py`**
```
INFO  file_parse_started     file_path, language
INFO  file_parse_completed   file_path, language, chunks_produced={directory: N, file: N, class: N, function: N}, error_node_count, duration_ms
WARN  tree_sitter_error_node file_path, byte_offset, node_type, surrounding_text (first 80 chars of parent node)
WARN  tree_sitter_missing_node file_path, byte_offset, expected_type
INFO  function_flagged_for_summarization file_path, function_name, line_count
DEBUG chunk_created           chunk_id, level, file_path, function_name, class_name, start_line, end_line, content_length
ERROR file_parse_failed       file_path, language, error (exception message), exc_info=True
```

**`ingestion/indexer.py`**
```
INFO  indexing_started        repo_path, trace_id
INFO  summarization_call      file_path, function_name, line_count, llm_backend, llm_model
INFO  summarization_completed file_path, function_name, latency_ms, token_count, summary_length
WARN  summarization_failed    file_path, function_name, error, retry_count
INFO  embedding_batch         batch_number, batch_size, level, latency_ms
INFO  bm25_index_built        level, corpus_size, vocabulary_size, duration_ms
INFO  chromadb_collection_updated collection_name, documents_inserted, total_collection_size
INFO  vocabulary_persisted     vocabulary_size, file_path
INFO  indexing_completed       repo_path, trace_id, total_chunks={directory: N, file: N, class: N, function: N}, total_llm_calls, total_embedding_batches, vocabulary_size, duration_ms
ERROR indexing_file_failed     file_path, phase (chunking | embedding | bm25 | chromadb), error, exc_info=True
```

**`retrieval/query_preprocessor.py`**
```
INFO  query_preprocessed      raw_input_length, has_code_block, is_mixed, identifiers_extracted (list), file_paths_extracted (list), keywords_extracted (list)
DEBUG parsed_query_full        (all ParsedQuery fields — only at DEBUG because raw_input may be long)
```

**`retrieval/technicality_scorer.py`**
```
INFO  technicality_scored     s1_identifier, s2_file_path, s3_vocab_overlap, final_score, bm25_weight, semantic_weight
```

**`retrieval/query_router.py`**
```
INFO  routing_started          mode (fast | thinking)
INFO  layer1_result            matched (bool), matched_level (str | None), matched_pattern (str | None), duration_ms
INFO  layer2_result            level_scores={directory: float, file: float, class: float, function: float}, best_level, best_score, confidence_gap, confident (bool), duration_ms
INFO  layer3_result            classified_level, confidence, reasoning, duration_ms
INFO  routing_completed        final_levels (list), decided_by_layer (1 | 2 | 3 | fallback), total_duration_ms
WARN  routing_fallback_triggered reason (layer3_low_confidence | exception | multi_level_query)
```

**`retrieval/bm25_retriever.py`**
```
INFO  bm25_retrieve           query_tokens (list), level, top_k, results_returned
DEBUG bm25_top_results         results (list of {chunk_id, score} for top 3)
```

**`retrieval/semantic_retriever.py`**
```
INFO  semantic_retrieve        query_text_length, level, top_k, results_returned
DEBUG semantic_top_results      results (list of {chunk_id, distance} for top 3)
```

**`retrieval/hybrid_retriever.py`**
```
INFO  hybrid_retrieve_started  levels (list), bm25_weight, semantic_weight, top_k
INFO  hybrid_retrieve_completed levels, total_results_before_merge, total_results_after_merge, consensus_hits (int — chunks in both BM25 and semantic), duration_ms
DEBUG rrf_top_results           results (list of {chunk_id, rrf_score, in_bm25, in_semantic} for top 5)
```

**`qa/qa_engine.py`**
```
INFO  qa_started               raw_query_length, mode (fast | thinking), trace_id
INFO  context_expansion        chunks_before_expansion, chunks_after_expansion, chunks_after_dedup, estimated_context_tokens
WARN  silent_escalation        reason (zero_results_fast_mode), original_levels, escalated_levels
INFO  deep_query_triggered     chunk_id, file_path, function_name, full_content_tokens
INFO  llm_synthesis_call       context_chunks, estimated_prompt_tokens
INFO  qa_completed             trace_id, mode, answer_confidence, citations_count, follow_up_count, total_duration_ms
WARN  qa_no_results            trace_id, mode, query_summary (first 100 chars)
```

**`resolver/issue_analyzer.py`**
```
INFO  issue_analysis_started   issue_number, issue_title_length, issue_body_length
INFO  identifiers_extracted    raw_identifiers (count), matched_in_index (count), matched_identifiers (list)
INFO  issue_analysis_completed issue_number, relevant_files (count), relevant_chunks (count), suggested_fix_scope, duration_ms
```

**`resolver/patch_generator.py`**
```
INFO  patch_generation_started  issue_number, relevant_chunks (count)
INFO  patch_generated           file_path, original_code_length, new_code_length, confidence, affects_other_files (list), duration_ms
WARN  patch_original_not_found  file_path, original_code_first_80_chars
ERROR patch_generation_failed   issue_number, error, exc_info=True
```

**`resolver/test_runner.py`**
```
INFO  test_run_started          framework, command, timeout_seconds
INFO  test_run_completed        framework, passed, total_tests, failed_tests (count), duration_ms, timed_out
WARN  test_framework_unknown    repo_path, files_checked (list of config files looked for)
DEBUG test_output               output_summary (first 500 chars of stdout + stderr)
```

**`resolver/pr_creator.py`**
```
INFO  pr_creation_started       issue_number, repo_full_name, dry_run, tests_passed
INFO  pr_creation_completed     issue_number, dry_run, pr_url (str | None), branch_name, duration_ms
WARN  pr_skipped_no_tests       issue_number, test_framework, test_output_summary
WARN  pr_skipped_tests_failed   issue_number, failed_tests (count)
```

**`llm/client.py`**
```
INFO  llm_call_started          backend, model, output_schema, prompt_token_estimate
INFO  llm_call_completed        backend, model, output_schema, latency_ms, retry_count, success=True
WARN  llm_call_retry            backend, model, output_schema, attempt, validation_error (str), latency_ms
ERROR llm_call_failed           backend, model, output_schema, total_attempts, last_validation_error (str), total_latency_ms, exc_info=True
DEBUG llm_prompt_text           prompt (full text — only when LOG_LLM_PROMPTS=True)
```

### `logging_config.py` Implementation

```python
"""
Oracle structured logging configuration.

Usage in every module:
    from oracle.logging_config import get_logger
    logger = get_logger(__name__)

Usage at pipeline entrypoints:
    from oracle.logging_config import get_logger, bind_trace_id, unbind_trace_id
    trace_id = bind_trace_id()
    logger.info("pipeline_started", pipeline="ingestion")
    # ... all downstream logs carry trace_id automatically
    unbind_trace_id()
"""

import structlog
import logging
import sys
from uuid import uuid4
from oracle.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE


def setup_logging() -> None:
    """Call once at application startup (main.py / server.py)."""

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if LOG_FORMAT == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(
            file=open(LOG_FILE, "a") if LOG_FILE else sys.stderr
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger bound to the given module name."""
    return structlog.get_logger(module=name)


def bind_trace_id(trace_id: str | None = None) -> str:
    """
    Generate and bind a trace_id to structlog context vars.
    Returns the trace_id for passing to API responses / SSE events.
    """
    if trace_id is None:
        trace_id = uuid4().hex[:12]
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
    return trace_id


def unbind_trace_id() -> None:
    """Clear trace_id from structlog context vars after pipeline completes."""
    structlog.contextvars.unbind_contextvars("trace_id")
```

### Debug Query Mode

The CLI supports `--debug-query` which runs a single query with LOG_LEVEL
forced to DEBUG and LOG_FORMAT forced to pretty, regardless of config.
Dumps the full pipeline trace: preprocessor output, technicality score,
routing decision (all layers attempted), retrieval results with scores,
parent expansion chain, final context assembly, and the answer with citations.

```bash
# Normal query
python -m oracle.main query "what does create_user return?"

# Debug trace — see everything that happened
python -m oracle.main query "what does create_user return?" --debug-query

# Filter by trace_id in production JSON logs
cat oracle.log | jq 'select(.trace_id == "a1b2c3d4e5f6")'

# Show only warnings and errors for a specific indexing run
cat oracle.log | jq 'select(.trace_id == "a1b2c3d4e5f6" and (.log_level == "warning" or .log_level == "error"))'
```

### Logging in Tests

Tests use `structlog.testing.capture_logs()` to assert that specific log
events were emitted with expected fields. This validates that logging is
actually happening, not just that the code runs.

```python
import structlog

def test_llm_call_logs_on_success(mock_ollama):
    with structlog.testing.capture_logs() as logs:
        client = LLMClient(backend="ollama")
        client.complete("test prompt", FunctionSummary)

    llm_logs = [l for l in logs if l.get("event") == "llm_call_completed"]
    assert len(llm_logs) == 1
    assert llm_logs[0]["backend"] == "ollama"
    assert llm_logs[0]["success"] is True
    assert "latency_ms" in llm_logs[0]

def test_tree_sitter_error_node_logged():
    code_with_syntax_error = "def foo(\n    x = 1"
    with structlog.testing.capture_logs() as logs:
        chunks = chunk_file("bad.py", code_with_syntax_error, "python")

    warnings = [l for l in logs if l.get("event") == "tree_sitter_error_node"]
    assert len(warnings) > 0
    assert warnings[0]["file_path"] == "bad.py"

def test_silent_escalation_logged():
    # Mock retrieval to return empty for fast mode
    with structlog.testing.capture_logs() as logs:
        answer = answer_question("obscure query", mode="fast")

    escalation_logs = [l for l in logs if l.get("event") == "silent_escalation"]
    assert len(escalation_logs) == 1
    assert escalation_logs[0]["reason"] == "zero_results_fast_mode"
```

---

## Directory Structure

```
oracle/
├── CLAUDE.md
├── config.py
├── logging_config.py                  # structlog setup, get_logger, trace ID binding
├── main.py                            # CLI entrypoint — calls setup_logging() once
├── scripts/
│   └── build_query_bank.py            # Run once: builds static query bank
│                                      # from CodeSearchNet + CoSQA + StaQC
├── data/
│   └── query_bank.npz                 # Committed artifact: ~1500 embeddings
│                                      # + phrases + level labels
├── ingestion/
│   ├── __init__.py
│   ├── repo_loader.py                 # Clone or load local repo, walk files
│   ├── ast_chunker.py                 # tree-sitter chunker — the core
│   └── indexer.py                     # chunk → embed → BM25 → ChromaDB
├── retrieval/
│   ├── __init__.py
│   ├── bm25_retriever.py              # BM25Okapi + camelCase tokenization
│   ├── semantic_retriever.py          # sentence-transformers + ChromaDB
│   ├── hybrid_retriever.py            # Weighted RRF fusion
│   ├── query_preprocessor.py          # ParsedQuery: identifiers, files, code
│   ├── technicality_scorer.py         # 0.0–1.0 score → dynamic BM25 weight
│   └── query_router.py               # 3-layer routing: regex → embed → LLM
├── qa/
│   ├── __init__.py
│   ├── qa_engine.py                   # Route → retrieve → expand → LLM
│   └── citation_builder.py           # Cited chunk references for answers
├── resolver/
│   ├── __init__.py
│   ├── issue_analyzer.py              # Parse issue → relevant chunks
│   ├── patch_generator.py             # LLM patch with verification
│   ├── test_runner.py                 # Run test suite via subprocess
│   └── pr_creator.py                 # Draft PR via PyGithub
├── llm/
│   ├── __init__.py
│   ├── client.py                      # LLMClient abstraction
│   └── schemas.py                     # All Pydantic output schemas
├── ui/
│   └── app.jsx                        # React: repo explorer + Q&A + resolver
└── tests/
    ├── __init__.py
    ├── test_chunker.py
    ├── test_preprocessor.py
    ├── test_retrieval.py
    ├── test_router.py
    ├── test_resolver.py
    └── test_logging.py                # Validates log events emitted by key operations
```

---

## The Four-Level Index Hierarchy

This is the foundational architectural decision. Code has natural structural
units defined by the language itself. We use those units as chunk boundaries
instead of arbitrary token windows.

```
Level 0 — Directory
  Content: directory path + all contained file summaries (one sentence each)
           + internal import relationships between files in the directory
           + most frequently exported identifiers
  Purpose: "What does this group of files do together?"
  Answers: system-level questions about modules and subsystems

Level 1 — File
  Content: file path + all import statements + all class/function signatures
           (NOT full bodies — signatures and names only)
  Purpose: "What is in this file?"
  Answers: overview questions, module structure questions

Level 2 — Class
  Content: class name + docstring + all method signatures + class variables
           (NOT method bodies)
  Purpose: "What does this class expose?"
  Answers: interface questions, "what methods does X have"

Level 3 — Function
  Content: full function/method body including docstring
  Purpose: "How is this implemented?"
  Answers: implementation questions, return values, parameters, logic
```

**Parent references — stored in every chunk's metadata:**
- Every chunk has `parent_file_chunk_id` (always set for class/function chunks)
- Function chunks inside a class have `parent_class_chunk_id`
- File chunks have `parent_directory_chunk_id`

**Context expansion on retrieval (implemented in `qa_engine.py`):**
When any chunk is retrieved, automatically fetch its full parent chain:
```
function → class → file → directory
```
Deduplicate before assembling context. The same chunk may appear via multiple
retrieval paths — keep it once. This ensures the LLM always has the full
structural context of whatever code it's looking at.

Log chunk counts at each stage: before expansion, after expansion, after dedup.

---

## Chunk Metadata Schema

Every chunk at every level stores this exact schema:

```python
class ChunkMetadata(BaseModel):
    chunk_id: str
    # Format: "{file_path}::{class_name}::{function_name}::{level}"
    # Use "None" string for absent class/function names

    level: str              # "directory" | "file" | "class" | "function"
    file_path: str          # Relative to repo root. "None" for directory chunks
    directory_path: str     # Relative to repo root
    language: str           # "python" | "javascript" | "typescript"
    class_name: str | None
    function_name: str | None
    start_line: int         # 0 for directory and file chunks
    end_line: int
    parent_directory_chunk_id: str | None
    parent_file_chunk_id: str | None    # Always set for class/function chunks
    parent_class_chunk_id: str | None   # Set only for methods
    content: str            # The actual text content of this chunk.
                            # For long functions: the LLM-generated summary.
                            # For all others: the raw code.
    full_content: str | None  # Full function body for summarized chunks.
                               # None for all non-summarized chunks.
    is_summarized: bool     # True if content is a summary, not raw code
    original_line_count: int | None  # Set when is_summarized=True
```

---

## AST Chunker (`ingestion/ast_chunker.py`)

Uses `tree-sitter` with language-specific parsers:
```bash
pip install tree-sitter tree-sitter-python tree-sitter-javascript tree-sitter-typescript
```

Supported languages: Python first, then JavaScript, TypeScript.

```python
CLASS_NODE_TYPES = {
    "python": ["class_definition"],
    "javascript": ["class_declaration", "class_expression"],
    "typescript": ["class_declaration"],
}

FUNCTION_NODE_TYPES = {
    "python": ["function_definition", "async_function_definition"],
    "javascript": ["function_declaration", "function_expression",
                   "arrow_function", "method_definition"],
    "typescript": ["function_declaration", "method_definition"],
}
```

Walk the AST recursively:
- For each **class node**: build class-level chunk (signature + docstring +
  method signatures, not bodies). For each method inside: build function-level
  chunk with `parent_class_chunk_id` set.
- For each **top-level function**: build function-level chunk with no class parent.
- For the **file**: collect all import nodes + all top-level definition
  signatures only.
- For each **directory**: after all file chunks exist, aggregate them into a
  directory chunk (see Directory Chunking section below).

**Tree-sitter error detection (mandatory):**
After parsing, walk the tree and check every node for `node.type == "ERROR"`
or `node.is_missing`. These indicate partial parses — the tree built but
contains garbage subtrees. Log each one as a warning:
```python
logger.warning("tree_sitter_error_node",
    file_path=file_path,
    byte_offset=node.start_byte,
    node_type=node.type,
    surrounding_text=source[node.parent.start_byte:node.parent.end_byte][:80]
)
```
Do NOT skip the file — extract what you can from the valid subtrees. But the
warnings let us know that chunk quality for this file is degraded.

**Files to skip entirely:**
`.git/`, `node_modules/`, `__pycache__/`, `dist/`, `build/`, `*.min.js`,
`*.lock`, `*.json`, `*.yaml`, `*.yml`, `*.md`, `*.txt`, `*.png`, `*.jpg`,
and any file that fails UTF-8 decode (binary detection via try/except).
Log every skipped file at DEBUG with the skip reason.

**Long function handling — summarize at ingestion, never truncate:**

Functions under `SUMMARIZE_MIN_LINES` (150): store full body as `content`.
Set `is_summarized=False`, `full_content=None`.

Functions at or over `SUMMARIZE_MIN_LINES`: the chunker flags them by setting
`original_line_count=N` and storing the raw body in `content`. The chunker
does NOT call the LLM — the indexer handles summarization. Log each flagged
function at INFO: file_path, function_name, line_count.

This cost is paid once at index time. Query time is unaffected.

Sub-chunking (splitting long functions into overlapping pieces) improves
retrieval quality by producing more focused embeddings. But sub-chunking
does NOT solve the context window problem. Only summarization does.
Do not conflate the two — they solve different problems.

```python
class FunctionSummary(BaseModel):
    summary: str   # 5-10 sentences. Must include: what the function does,
                   # all parameters with types, return value and type,
                   # exceptions raised, key algorithms, external dependencies,
                   # and side effects (DB writes, API calls, file I/O).
```

The summarization prompt must instruct the LLM to be technically precise
and not omit parameters or return types. A vague summary causes the retriever
to surface this chunk for unrelated queries and the synthesizer to give
confidently wrong answers.

On-demand deep query: if a user asks something the summary cannot answer
(e.g. "walk me through the exact loop logic"), the Q&A engine detects
`is_summarized=True` on the retrieved chunk, fetches `full_content`, and
makes a second focused LLM call with just that function in context.
This is an explicit escalation path, not a silent fallback. Log when triggered:
chunk_id, file_path, function_name, full_content token count.

---

## Directory-Level Chunking

Directory chunks are synthetic — not produced by tree-sitter, generated after
all file chunks exist for a directory.

**Depth limit:** Only generate directory chunks for directories containing
3–20 files. Directories with fewer than 3 files don't need aggregation.
Directories with more than 20 files are too large to summarize meaningfully —
recurse into subdirectories and generate chunks per subdirectory instead.

**Content of a directory chunk:**
```
Directory: services/
Files: [auth.py, session.py, tokens.py, user.py]

File summaries:
  auth.py: imports jwt, bcrypt. defines AuthService, authenticate_user, verify_token
  session.py: imports redis. defines SessionManager, create_session, expire_session
  tokens.py: imports secrets. defines generate_token, validate_token, refresh_token
  user.py: imports sqlalchemy. defines User, UserRepository, get_user, create_user

Internal import relationships:
  auth.py imports from session.py (SessionManager)
  auth.py imports from tokens.py (generate_token, validate_token)
  session.py imports from tokens.py (validate_token)

Most exported identifiers:
  AuthService, authenticate_user, SessionManager, generate_token, User
```

**Generation logic (`ingestion/indexer.py`):**
```python
def build_directory_chunk(dir_path: str, file_chunks: list[ChunkMetadata]) -> ChunkMetadata:
    # Extract import statements from each file chunk's content
    # Cross-reference: if file A imports something defined in file B → internal import
    # Collect all function/class names across file chunks → exported identifiers
    # Format as structured text as shown above
```

---

## Query Preprocessor (`retrieval/query_preprocessor.py`)

Every query enters this preprocessor before any routing or retrieval.
No raw string ever enters the pipeline directly.

```python
class ParsedQuery(BaseModel):
    raw_input: str
    natural_language: str             # Query with code blocks removed
    attached_file_path: str | None    # If user attached a file
    attached_file_content: str | None
    pasted_code: str | None           # Detected code block content
    pasted_code_language: str | None
    extracted_identifiers: list[str]  # camelCase, snake_case, PascalCase names
    extracted_file_paths: list[str]   # anything matching *.py / *.js / *.ts etc
    extracted_keywords: list[str]     # "class", "function", "method", "file" etc
    has_code_block: bool
    is_mixed: bool                    # natural language + code both present
```

Log the parsed result at INFO (field counts and key extracted values) and
at DEBUG (full ParsedQuery dump). This is the first debugging touchpoint for
"why didn't it find X" — usually the preprocessor stripped or missed a
critical identifier.

**Code block detection:**
Detect triple backtick blocks. Extract content and language hint if present.
Replace the block with a placeholder in `natural_language`. Set
`has_code_block=True`. Do NOT route code blocks specially — the technicality
scorer will naturally score them at 0.9+ because of identifier density.
A half-working special case is worse than letting the standard pipeline handle it.

**Identifier extraction:**
```python
IDENTIFIER_PATTERN = r'\b[A-Za-z_][A-Za-z0-9_]{2,}\b'
# Run on both natural language and pasted code
# Filter out common English stop words and Python/JS keywords
STOP_WORDS = {"the", "and", "for", "with", "this", "that", "what", "how",
              "does", "from", "into", "about", "which", "when", "where",
              "def", "class", "return", "import", "function", "var", "let",
              "const", "type", "interface", "public", "private"}
```

**File path extraction:**
```python
FILE_PATH_PATTERN = r'\b[\w/]+\.(py|js|ts|go|java|rs)\b'
```

**Structural keyword extraction:**
```python
STRUCTURAL_KEYWORDS = [
    "class", "function", "method", "file", "module", "directory",
    "folder", "package", "struct", "interface", "import", "decorator",
    "async", "constructor", "property", "attribute", "variable",
]
```

**File attachment support (UI → preprocessor):**
The UI allows users to drag a file into the query box or use a file picker.
File content is read client-side and sent alongside the query in the request
body. The preprocessor sets `attached_file_path` and `attached_file_content`.
A file attachment bypasses retrieval for that file — use attached content
directly as file-level context in the LLM prompt.

---

## Syntax Highlighting in the Query Box (UI)

Detect triple backtick blocks in the query textarea and render them with
syntax highlighting. Use the transparent overlay technique:

```jsx
// Transparent textarea on top — receives all input events
// Highlighted div underneath — purely visual, pointer-events: none
// Both must have IDENTICAL font, padding, line-height, word-wrap
// or text alignment breaks and the effect looks wrong
```

Inline backtick content gets a subtle background color.
Triple backtick blocks get a darker background with monospace font.
This is purely cosmetic — it does not affect the pipeline at all.
If CSS alignment is fiddly to get right, implement it last.

---

## Technicality Scorer (`retrieval/technicality_scorer.py`)

Produces a float 0.0–1.0 representing how technical (vs semantic) the query is.
This score determines BM25/semantic weighting for all retrieval calls.

Log all three signal values individually plus the final score and computed
weights. Without this you'll never understand why a query got routed wrong.

```python
def score_technicality(parsed_query: ParsedQuery, index_vocabulary: set[str]) -> float:

    # Signal 1: Code identifier present in query (weight 0.35)
    has_identifier = len(parsed_query.extracted_identifiers) > 0
    s1 = 1.0 if has_identifier else 0.0

    # Signal 2: File path or file attachment present (weight 0.45)
    # Strongest signal — user is pointing at specific known location
    has_file = (
        len(parsed_query.extracted_file_paths) > 0
        or parsed_query.attached_file_path is not None
    )
    s2 = 1.0 if has_file else 0.0

    # Signal 3: Vocabulary overlap with index (weight 0.20)
    # Noisiest signal — common English words appear in both queries and docstrings
    # index_vocabulary: set of all tokens from all chunks, persisted at index time
    all_tokens = camel_snake_tokenize(parsed_query.natural_language)
    if all_tokens:
        overlap = len(set(all_tokens) & index_vocabulary) / len(all_tokens)
    else:
        overlap = 0.0
    s3 = overlap

    score = (s1 * 0.35) + (s2 * 0.45) + (s3 * 0.20)

    logger.info("technicality_scored",
        s1_identifier=s1, s2_file_path=s2, s3_vocab_overlap=round(s3, 3),
        final_score=round(score, 3),
        bm25_weight=round(0.3 + (0.5 * score), 3),
        semantic_weight=round(1.0 - (0.3 + (0.5 * score)), 3),
    )

    return score
```

**How vocabulary overlap works:**
`index_vocabulary` contains every token produced by `camel_snake_tokenize`
across every chunk in the indexed codebase. It is persisted to disk at
`{CHROMA_PERSIST_DIR}/vocabulary.pkl` when indexing completes and loaded
at query time. If the query uses words that appear in the codebase (function
names, class names, module names), overlap is high → technical. If the query
uses descriptive English with no codebase-specific terms, overlap is low →
semantic.

**Dynamic weight computation:**
```python
def compute_retrieval_weights(technicality: float) -> tuple[float, float]:
    bm25_weight = 0.3 + (0.5 * technicality)   # 0.30 → 0.80
    semantic_weight = 1.0 - bm25_weight          # 0.70 → 0.20
    return bm25_weight, semantic_weight
```

Reference table:

| Query type | Example | Technicality | BM25 | Semantic |
|---|---|---|---|---|
| Exact identifier + file | "in auth.py what does create_user do?" | 1.0 | 0.80 | 0.20 |
| Identifier only | "what does getUserById return?" | 0.7 | 0.65 | 0.35 |
| Vague technical | "where is user creation handled?" | 0.4 | 0.50 | 0.50 |
| Fully semantic | "what does the account generator do?" | 0.0 | 0.30 | 0.70 |

---

## Static Query Bank (`data/query_bank.npz`)

**What it is:** A committed numpy archive containing ~1,500 pre-embedded
phrase vectors (number determined by k-means elbow analysis per level) used
for layer 2 query routing. Built once via `scripts/build_query_bank.py`,
committed to the repo, never regenerated at runtime. Completely deterministic —
same phrase bank used for every query, no randomness.

**Sources (all three used together):**

1. **CodeSearchNet** (Python split, ~500k pairs) — mine description side for
   file/class/function patterns. Largest and most diverse source.
2. **CoSQA** (Microsoft, 20k pairs) — real Bing developer search queries
   paired with Python code. Highest quality for actual developer phrasing.
   Prioritized over CodeSearchNet in deduplication.
3. **StaQC** — Stack Overflow question/answer pairs mapped to code snippets.
   Captures "how do I" and "what is the best way to" phrasing natural to SO.

Why all three: CoSQA is highest quality but small. StaQC adds SO phrasing
patterns. CodeSearchNet adds volume and diversity. Together they cover the
high-frequency query space comprehensively. The LLM fallback (layer 3)
handles the tail that none of them cover.

**Build process (`scripts/build_query_bank.py`):**
```
1. Download all three datasets
2. Filter to natural language description / question side only
3. Embed all descriptions with all-MiniLM-L6-v2
4. For each level (directory, file, class, function):
   a. Filter to descriptions whose source code is level-appropriate
      - File-level: module/file docstrings
      - Class-level: class docstrings
      - Function-level: function docstrings and CoSQA function queries
      - Directory-level: synthesize from file-level descriptions grouped by directory
   b. Run k-means sweeping k from 50 to 1000
   c. Plot within-cluster variance (the elbow curve)
   d. Pick k at the elbow — empirically ~300–600 per level for code queries
   e. Take the centroid of each cluster as the representative embedding
      (centroid = mean of all embeddings in cluster — generalizes better
       than any single sampled phrase)
   f. Store centroid embedding + closest actual phrase to centroid (for debugging)
5. Deduplicate across sources:
   If two centroids have cosine similarity > 0.95, keep the one from the
   higher-quality source (CoSQA > StaQC > CodeSearchNet)
6. Save as data/query_bank.npz:
   - embeddings: float32 array of shape (n_total, 384)
   - labels: string array of shape (n_total,) — level for each embedding
   - phrases: string array of shape (n_total,) — human-readable representative phrase
```

**At query time — single matrix multiplication:**
```python
# Loaded once at module level, never reloaded
QUERY_BANK = np.load("data/query_bank.npz")
QUERY_BANK_EMBEDDINGS = QUERY_BANK["embeddings"]   # (n_total, 384) float32
QUERY_BANK_LABELS = QUERY_BANK["labels"]

def route_by_embedding(parsed_query: ParsedQuery) -> tuple[str, float]:
    query_vec = embedding_model.encode(parsed_query.natural_language)  # (384,)
    similarities = QUERY_BANK_EMBEDDINGS @ query_vec                   # (n_total,)

    level_scores = {}
    for level in ["directory", "file", "class", "function"]:
        mask = QUERY_BANK_LABELS == level
        level_scores[level] = float(similarities[mask].max())

    best_level = max(level_scores, key=level_scores.get)
    best_score = level_scores[best_level]
    second_score = sorted(level_scores.values())[-2]
    confidence_gap = best_score - second_score

    return best_level, best_score, confidence_gap
```

Under 1ms on CPU for the matrix multiply. Deterministic.

---

## Query Router (`retrieval/query_router.py`)

Three sequential layers. Stop at the first layer that produces a confident answer.
Log every layer's result regardless of whether it was the deciding layer —
this is critical for debugging routing misclassifications.

### Layer 1: Regex Pattern Matching

Fast, free, handles obvious explicit cases. Catches ~40% of real queries.

```python
DIRECTORY_PATTERNS = [
    r"what does (the )?[\w/]+ (folder|directory|module|package) do",
    r"how do .+ (work|fit) together",
    r"(overview|architecture|structure) of [\w/]+",
    r"what (files|modules) are in",
    r"purpose of (the )?[\w/]+ (directory|folder|package)",
]
FILE_PATTERNS = [
    r"what (does|is) (this|the) (file|module)",
    r"what (is in|imports|exports) [\w.]+\.(py|js|ts)",
    r"overview of [\w.]+\.(py|js|ts)",
    r"(structure|contents) of [\w.]+\.(py|js|ts)",
    r"tell me about [\w.]+\.(py|js|ts)",
]
CLASS_PATTERNS = [
    r"what (does|is) (the|this) \w+ class",
    r"how (does|is) \w+ (class|object) (work|structured|used)",
    r"(methods|attributes|properties) (of|in|on) \w+",
    r"what (can|does) \w+ (do|provide|expose)",
]
FUNCTION_PATTERNS = [
    r"what does (the|this)?( )?\w+ (function|method|procedure) do",
    r"how (does|is) \w+ (called|used|implemented|work)",
    r"(return|returns|return value|output) of \w+",
    r"(parameters?|arguments?|inputs?) (of|for|to) \w+",
    r"how to (use|call|invoke) \w+",
    r"what (happens|occurs) when \w+ is called",
]
```

All patterns are case-insensitive. If multiple patterns match different levels,
prefer most specific: function > class > file > directory.

Log: matched (bool), matched_level, matched_pattern (the regex string that hit),
duration_ms.

### Layer 2: Embedding Similarity Against Query Bank

Triggered when layer 1 produces no match.

Confident if: `best_score > 0.65` AND `confidence_gap > 0.10` (gap between
best and second-best level score). If not confident, escalate to layer 3.

Log: all four level_scores, best_level, best_score, confidence_gap,
confident (bool), duration_ms.

### Layer 3: LLM Classification

Only triggered when layer 2 is ambiguous. Uses `LLMClient` with schema:

```python
class QueryClassification(BaseModel):
    level: str          # "directory" | "file" | "class" | "function" | "all"
    confidence: float   # 0.0–1.0
    reasoning: str      # one sentence
```

Log: classified_level, confidence, reasoning, LLM latency_ms.

### Fallback: Search All Levels

Triggered when:
- Layer 3 confidence < 0.6
- Any exception occurs anywhere in routing
- Query explicitly asks about multiple levels

Searching all levels is always correct — just slower and noisier.
A wrong level restriction is a silent failure (user gets no answer without
knowing why). When in doubt, search everything.

Log fallback triggers as WARNING with the reason (layer3_low_confidence,
exception, multi_level_query).

Log final routing decision at INFO: final_levels, decided_by_layer,
total_duration_ms.

---

## Fast vs Thinking Mode

**Fast mode (default):**
- Runs routing layer 1 + layer 2 only
- Confident level assignment → search that level only
- Total routing overhead: ~5–10ms
- Suitable for: specific questions with clear structural intent

**Thinking mode:**
- Runs all three routing layers regardless of layer 2 confidence
- Always searches all four levels simultaneously
- Merges all results with RRF across levels
- Total overhead: ~500ms–2s (LLM classification + 4× retrieval)
- Suitable for: vague questions, architectural questions, cross-cutting concerns

**Silent escalation rule:**
If Fast mode returns zero results after retrieval, automatically retry the
full query in Thinking mode before returning anything to the user. No UI
change, no notification, no asking the user to switch modes. Only surface
results (or "nothing found") after the Thinking retry completes.

Log every silent escalation as WARNING with reason=zero_results_fast_mode,
original routing levels, and escalated levels.

**UI implementation:**
Dropdown at bottom right of query input, next to submit arrow:
```
[  Ask about your codebase...                    ⚡▾  →  ]
```
Dropdown opens upward on click, closes on selection or click-outside.
Mode persists per session in React useState — not localStorage.

```
┌─────────────────────────────────────────────────┐
│ ⚡ Fast                                           │
│    Quick response, suitable for most queries     │
├─────────────────────────────────────────────────┤
│ 🧠 Thinking                                      │
│    Thorough search across all levels,            │
│    suitable for complex questions                │
└─────────────────────────────────────────────────┘
```

---

## BM25 Retriever (`retrieval/bm25_retriever.py`)

```python
def camel_snake_tokenize(text: str) -> list[str]:
    # Step 1: Insert space before uppercase preceded by lowercase (camelCase split)
    text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', text)
    # Step 2: Replace underscores and all non-alphanumeric chars with spaces
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
    # Step 3: Lowercase, split, filter empty strings and single chars
    return [t.lower() for t in text.split() if len(t) > 1]

# getUserById   → ["get", "user", "by", "id"]
# get_user_by_id → ["get", "user", "by", "id"]
# UserService   → ["user", "service"]
# create_user   → ["create", "user"]
```

Four separate BM25Okapi indexes — one per level (directory, file, class,
function). Persisted to disk at `{CHROMA_PERSIST_DIR}/bm25_{level}.pkl`
using pickle. Rebuilt only when the repository is re-indexed.

The index vocabulary (all unique tokens across all chunks) is persisted
separately at `{CHROMA_PERSIST_DIR}/vocabulary.pkl` as a Python set.
Loaded by the technicality scorer at query time.

```python
def bm25_retrieve(query: str, level: str, top_k: int) -> list[ChunkMetadata]:
    # Load persisted index for level
    # Tokenize query with camel_snake_tokenize (identical to corpus tokenization)
    # Log: query_tokens, level, top_k, results_returned
    # Log at DEBUG: top 3 chunk_ids with BM25 scores
    # Return top_k as ChunkMetadata objects
```

---

## Semantic Retriever (`retrieval/semantic_retriever.py`)

- Model: `all-MiniLM-L6-v2` — loaded **once at module level**, never reloaded
- Four ChromaDB collections: `oracle_directory`, `oracle_file`, `oracle_class`,
  `oracle_function`
- Persisted at `CHROMA_PERSIST_DIR`
- Indexing uses batch embedding: `EMBEDDING_BATCH_SIZE = 64`
- Collection metadata stores all ChunkMetadata fields verbatim so results
  can be reconstructed as `ChunkMetadata` objects without any secondary lookup

```python
def semantic_retrieve(query: str, level: str, top_k: int) -> list[ChunkMetadata]:
    # Log: query_text_length, level, top_k, results_returned
    # Log at DEBUG: top 3 chunk_ids with ChromaDB distances
```

---

## Hybrid Retriever (`retrieval/hybrid_retriever.py`)

**Dynamic top_k — scales with corpus size:**
```python
def compute_top_k(total_chunks: int) -> int:
    # ~1% of corpus, floored at 5, capped at 20
    return max(5, min(20, total_chunks // 100))
```
The RRF constant k=60 is NOT about corpus size. k=60 controls how much rank 1
differs from rank 2 — it is a smoothing constant from Cormack et al. (2009)
empirically robust across many different corpora. Do not make it dynamic.
`top_k` is what scales with corpus size.

**Weighted RRF:**
```python
def rrf_merge(
    bm25_results: list[ChunkMetadata],
    semantic_results: list[ChunkMetadata],
    bm25_weight: float,
    semantic_weight: float,
    k: int = 60,
) -> list[ChunkMetadata]:
    """
    score(chunk) =
        bm25_weight    × (1 / (k + rank_in_bm25_list))
      + semantic_weight × (1 / (k + rank_in_semantic_list))

    If chunk appears in only one list, its rank in the absent list = len(list) + 1.
    Deduplicate by chunk_id. Sort descending by score.
    """
```

Log at INFO: levels, bm25_weight, semantic_weight, top_k,
total_results_before_merge, total_results_after_merge, consensus_hits
(chunks appearing in both BM25 and semantic results — these are your
highest quality results). Log at DEBUG: top 5 RRF results with scores
and which source lists each appeared in.

```python
def hybrid_retrieve(
    parsed_query: ParsedQuery,
    levels: list[str],
    bm25_weight: float,
    semantic_weight: float,
) -> list[ChunkMetadata]:
    top_k = compute_top_k(get_total_chunk_count())
    all_results = []
    for level in levels:
        bm25 = bm25_retrieve(parsed_query.natural_language, level, top_k)
        semantic = semantic_retrieve(parsed_query.natural_language, level, top_k)
        merged = rrf_merge(bm25, semantic, bm25_weight, semantic_weight)
        all_results.extend(merged)
    return deduplicate_and_rerank(all_results)[:top_k]
```

---

## Q&A Engine (`qa/qa_engine.py`)

```python
def answer_question(
    query: str,
    mode: str = "fast",
    llm_client: LLMClient = None,
) -> QAAnswer:
    trace_id = bind_trace_id()
    logger.info("qa_started", raw_query_length=len(query), mode=mode)

    parsed = preprocess_query(query)
    technicality = score_technicality(parsed, load_vocabulary())
    bm25_weight, semantic_weight = compute_retrieval_weights(technicality)

    levels = route_query(parsed, mode)
    retrieved = hybrid_retrieve(parsed, levels, bm25_weight, semantic_weight)

    if not retrieved and mode == "fast":
        # Silent escalation: retry in thinking mode before giving up
        logger.warning("silent_escalation",
            reason="zero_results_fast_mode",
            original_levels=levels,
            escalated_levels=["directory", "file", "class", "function"],
        )
        levels = ["directory", "file", "class", "function"]
        retrieved = hybrid_retrieve(parsed, levels, bm25_weight, semantic_weight)

    expanded = expand_with_parents(retrieved)   # fetch full parent chain
    unique = deduplicate_chunks(expanded)        # deduplicate by chunk_id

    logger.info("context_expansion",
        chunks_before_expansion=len(retrieved),
        chunks_after_expansion=len(expanded),
        chunks_after_dedup=len(unique),
        estimated_context_tokens=estimate_tokens(unique),
    )

    answer = llm_answer(parsed.natural_language, unique, llm_client)

    logger.info("qa_completed",
        mode=mode,
        answer_confidence=answer.confidence,
        citations_count=len(answer.cited_chunks),
        follow_up_count=len(answer.follow_up_suggestions),
        total_duration_ms=elapsed_ms(),
    )
    unbind_trace_id()
    return answer
```

**LLM output schema:**
```python
class ChunkCitation(BaseModel):
    chunk_id: str
    file_path: str
    line_range: str         # "lines 45–78"
    relevance: str          # one sentence explaining why this chunk is relevant

class QAAnswer(BaseModel):
    answer: str             # Direct answer, 1–3 paragraphs
    confidence: float       # 0.0–1.0
    cited_chunks: list[ChunkCitation]
    follow_up_suggestions: list[str]   # 2–3 related questions the user might ask
```

---

## Issue Analyzer (`resolver/issue_analyzer.py`)

```python
def analyze_issue(issue_title: str, issue_body: str) -> IssueAnalysis:
```

1. Extract candidate identifiers from issue text using:
   `re.findall(r'\b[A-Za-z_][A-Za-z0-9_]{2,}\b', text)`
2. Cross-reference against index — keep only identifiers that appear as
   `function_name` or `class_name` in indexed chunks. This filters natural
   English words and retains only real codebase symbols. Log: raw identifiers
   count, matched identifiers count, matched identifiers list.
3. Build search query: `issue_title + " " + " ".join(matched_identifiers)`
4. Score technicality of this combined query — will be high because of
   extracted identifiers. Run `hybrid_retrieve` across all levels.
5. Weight function-level results 2× by duplicating them before RRF merge —
   bugs are almost always in function bodies, not class or file headers.
6. Return top 10 chunks with full parent context expanded.

```python
class IssueAnalysis(BaseModel):
    relevant_files: list[str]
    relevant_chunks: list[ChunkCitation]
    identified_identifiers: list[str]
    suggested_fix_scope: str    # "single_function" | "multiple_files" | "unknown"
```

---

## Patch Generator (`resolver/patch_generator.py`)

```python
class CodePatch(BaseModel):
    file_path: str
    original_code: str      # Must be exact verbatim substring of the current file
    new_code: str
    explanation: str
    confidence: float
    affects_other_files: list[str]
```

**Patch format is search-and-replace, not diff.** The LLM outputs the exact
text it wants to find and the exact text to replace it with. Simpler than diff
format and more reliably produced by LLMs.

**Verification before any write — mandatory:**
```python
def apply_patch(patch: CodePatch, repo_path: str) -> bool:
    content = Path(repo_path, patch.file_path).read_text()
    if patch.original_code not in content:
        logger.warning("patch_original_not_found",
            file_path=patch.file_path,
            original_code_preview=patch.original_code[:80],
        )
        raise ValueError(f"original_code not found in {patch.file_path}")
    new_content = content.replace(patch.original_code, patch.new_code, 1)
    if WRITE_MODE_ENABLED:
        Path(repo_path, patch.file_path).write_text(new_content)
    return True
```

LLMs frequently hallucinate code that is almost-but-not-exactly what's in the
file: wrong whitespace, slightly different variable names, missing decorators.
If `original_code` is not an exact substring, the patch is invalid. Never
apply to an ambiguous location. Raise ValueError and stop.

---

## Test Runner (`resolver/test_runner.py`)

```python
def detect_test_framework(repo_path: str) -> str:
    # Check: pytest.ini, setup.cfg [tool:pytest], pyproject.toml [tool.pytest.ini_options]
    # Check: jest.config.js, package.json with "jest" key
    # Return: "pytest" | "jest" | "unknown"
    # If "unknown": log WARNING with list of config files checked

class TestResult(BaseModel):
    passed: bool
    framework: str
    total_tests: int | None
    failed_tests: list[str]
    output_summary: str     # First 500 chars of stdout + stderr
    timed_out: bool
```

If framework is "unknown" or no test files found:
- Set `passed=False`
- Do not open PR
- Report: "No test suite found — cannot verify patch safety"

Log: framework detected, command executed, exit code, pass/fail, timeout hit,
duration_ms. Log test output at DEBUG.

---

## PR Creator (`resolver/pr_creator.py`)

```python
def create_pr(
    patch: CodePatch,
    issue_title: str,
    issue_number: int,
    repo_full_name: str,        # "owner/repo"
    test_result: TestResult,
    dry_run: bool = True,       # ALWAYS True by default. Must explicitly set False.
) -> PRResult:

class PRResult(BaseModel):
    dry_run: bool
    pr_url: str | None          # None when dry_run=True
    pr_title: str               # "[Auto] Fix: {issue_title}"
    pr_body: str                # Full markdown preview generated regardless of dry_run
    branch_name: str            # "oracle/fix-issue-{issue_number}"
```

All PRs are DRAFT: `repo.create_pull(..., draft=True)`.
`GITHUB_TOKEN` read from environment only. Never hardcoded anywhere.
PR body must include: `Closes #{issue_number}`, patch explanation,
confidence score, test results summary.

Log: issue_number, repo_full_name, dry_run flag, tests_passed, pr_url
(if created), branch_name, duration_ms. Log skipped PR creation as WARNING
with the reason (no tests, tests failed).

---

## LLM Client (`llm/client.py`)

```python
class LLMClient:
    def __init__(self, backend: str = None, model: str = None):
        # backend defaults to LLM_BACKEND from config
        # model defaults to OLLAMA_MODEL or CLAUDE_MODEL per backend

    def complete(self, prompt: str, output_schema: type[BaseModel]) -> BaseModel:
        # instructor structured output — enforces schema via retry
        # Retry up to 3 times on validation failure, exponential backoff
        #
        # LOGGING (mandatory):
        # INFO  llm_call_started:   backend, model, output_schema.__name__, prompt_token_estimate
        # INFO  llm_call_completed: backend, model, output_schema.__name__, latency_ms, retry_count, success=True
        # WARN  llm_call_retry:     backend, model, output_schema.__name__, attempt, validation_error (str), latency_ms
        # ERROR llm_call_failed:    backend, model, output_schema.__name__, total_attempts, last_validation_error, total_latency_ms, exc_info=True
        # DEBUG llm_prompt_text:    prompt (full text — only when LOG_LLM_PROMPTS=True)
        #
        # Raise RuntimeError after 3 failed attempts
```

Ollama backend:
```python
instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON
)
```

Claude backend:
```python
instructor.from_anthropic(anthropic.Anthropic())
```

Default model: `qwen2.5-coder:7b` — the code-specific fine-tuned variant.
Meaningfully better than the general model for patch generation and code Q&A.

---

## Config (`config.py`)

```python
# LLM
LLM_BACKEND = "ollama"
OLLAMA_MODEL = "qwen2.5-coder:7b"
CLAUDE_MODEL = "claude-sonnet-4-6"

# Embeddings + Storage
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = ".oracle_db"
EMBEDDING_BATCH_SIZE = 64

# Retrieval
BM25_WEIGHT_DEFAULT = 0.6       # At technicality = 0.5
SEMANTIC_WEIGHT_DEFAULT = 0.4
RRF_K = 60                      # Cormack et al. 2009 — do not make dynamic
TOP_K_MIN = 5
TOP_K_MAX = 20

# Chunking
SUMMARIZE_MIN_LINES = 150       # Functions at or over this get LLM summary at index time
                                 # Functions under this store full body directly
MIN_DIRECTORY_FILES = 3
MAX_DIRECTORY_FILES = 20

# Resolver
WRITE_MODE_ENABLED = False
TEST_TIMEOUT_SECONDS = 120

# Languages
SUPPORTED_LANGUAGES = ["python", "javascript", "typescript"]

# Logging
LOG_LEVEL = "INFO"              # DEBUG for development, INFO for production
LOG_FORMAT = "pretty"           # "pretty" for dev terminal, "json" for production/piping
LOG_FILE = None                 # None = stderr only, path string = also write to file
LOG_LLM_PROMPTS = False         # True = log full LLM prompts at DEBUG level (very verbose)
```

---

## Loading UI + Minigame

Indexing can take 30s–3min depending on codebase size. Progress must be
communicated. The minigame appears only if indexing takes longer than 5 seconds.

**Phase 1 (0–5s): SSE-driven progress bar**
Each pipeline stage emits an SSE event. UI updates in real time:
```
⠙ Parsing repository...          [██░░░░░░░░] 20%
⠙ Building AST chunks...         [████░░░░░░] 40%
⠙ Embedding chunks...            [██████░░░░] 60%
⠙ Building BM25 index...         [████████░░] 80%
✓ Index ready                    [██████████] 100%
```

**Phase 2 (>5s elapsed wall time): Endless runner minigame**
A 400×120px canvas appears below the progress bar.
Both the canvas and the progress bar are visible simultaneously —
the game is entertainment, not a replacement for status information.

```
Theme:    coding / terminal aesthetic
Character: blinking cursor  |
Obstacles: 🐛 bugs, red × syntax errors — scroll in from the right
Collectibles: ✓ green checkmarks (passing tests)
Background: faint scrolling terminal text
Controls: spacebar or single click/tap to jump — no instructions needed
Speed: increases gradually over time. No win state, no levels.
Score: "Tests passed: N" — thematic, not a raw number
Game over: "Build failed." — press space or click to restart immediately
```

Implementation: pure HTML5 canvas, zero dependencies, under 150 lines.

The 5-second threshold uses `Date.now()` elapsed time — never display
estimated remaining time. Wrong estimates are worse than no estimates.

When indexing completes while game is active:
- Show banner: "Index ready — results available"
- Do not interrupt mid-run
- After 2 seconds OR when the current run ends (whichever comes first),
  fade out canvas and fade in the query interface

---

## Acceptance Tests

```python
# test_chunker.py

def test_python_function_chunked_correctly():
    code = '''
def get_user(user_id: int) -> User:
    """Fetch user by ID."""
    return db.query(User).filter_by(id=user_id).first()
'''
    chunks = chunk_file("models/user.py", code, "python")
    func_chunks = [c for c in chunks if c.level == "function"]
    assert len(func_chunks) == 1
    assert func_chunks[0].function_name == "get_user"
    assert func_chunks[0].start_line > 0
    assert func_chunks[0].parent_file_chunk_id is not None

def test_class_methods_have_parent_metadata():
    code = '''
class UserService:
    def create_user(self, name: str): pass
    def delete_user(self, user_id: int): pass
'''
    chunks = chunk_file("services/user.py", code, "python")
    methods = [c for c in chunks if c.level == "function"]
    assert len(methods) == 2
    assert all(c.class_name == "UserService" for c in methods)
    assert all(c.parent_class_chunk_id is not None for c in methods)
    assert all(c.parent_file_chunk_id is not None for c in methods)

def test_no_fixed_token_chunking():
    # Every function-level chunk must contain a real function signature
    chunks = chunk_file("big.py", LONG_CODE, "python")
    for chunk in chunks:
        if chunk.level == "function":
            assert "def " in chunk.content or "async def " in chunk.content

def test_short_function_not_summarized():
    code = '''
def get_user(user_id: int) -> User:
    """Fetch user by ID."""
    return db.query(User).filter_by(id=user_id).first()
'''
    chunks = chunk_file("models/user.py", code, "python")
    func_chunks = [c for c in chunks if c.level == "function"]
    assert len(func_chunks) == 1
    assert func_chunks[0].is_summarized is False
    assert func_chunks[0].full_content is None

def test_long_function_flagged_for_summarization():
    # The chunker flags long functions but does NOT call the LLM itself.
    # Summarization is the indexer's responsibility (it has the LLM client).
    # The chunker sets is_summarized=False and full_content=raw body.
    # The indexer reads original_line_count and triggers summarization.
    long_func = "def big_function():\n" + "    x = 1\n" * 200
    chunks = chunk_file("big.py", long_func, "python")
    func_chunks = [c for c in chunks if c.level == "function"]
    assert len(func_chunks) == 1
    assert func_chunks[0].original_line_count >= 150
    # content is still the raw body at this stage — indexer will summarize it
    assert "def big_function" in func_chunks[0].content

def test_directory_chunk_depth_limit():
    # Directory with 2 files → no directory chunk generated
    # Directory with 5 files → directory chunk generated
    # Directory with 25 files → no single directory chunk (split by subdir)
    pass  # implement with mock file trees

def test_tree_sitter_error_nodes_logged():
    code_with_syntax_error = "def foo(\n    x = 1"
    with structlog.testing.capture_logs() as logs:
        chunks = chunk_file("bad.py", code_with_syntax_error, "python")
    warnings = [l for l in logs if l.get("event") == "tree_sitter_error_node"]
    assert len(warnings) > 0
    assert warnings[0]["file_path"] == "bad.py"


# test_preprocessor.py

def test_identifier_extraction():
    parsed = preprocess_query("what does getUserById return?")
    assert "getUserById" in parsed.extracted_identifiers

def test_file_path_extraction():
    parsed = preprocess_query("in auth.py what does login do?")
    assert "auth.py" in parsed.extracted_file_paths

def test_code_block_detection():
    parsed = preprocess_query("explain this:\n```python\ndef foo(): pass\n```")
    assert parsed.has_code_block
    assert parsed.pasted_code is not None
    assert "foo" in parsed.extracted_identifiers

def test_technicality_high_for_identifier_query():
    parsed = preprocess_query("what does create_user return?")
    score = score_technicality(parsed, {"create", "user", "return"})
    assert score > 0.5

def test_technicality_low_for_semantic_query():
    parsed = preprocess_query("where is the thing that generates accounts?")
    score = score_technicality(parsed, set())
    assert score < 0.3

def test_technicality_maxes_with_file_path():
    parsed = preprocess_query("in services/auth.py what does login do?")
    score = score_technicality(parsed, {"login", "auth"})
    assert score > 0.8


# test_retrieval.py

def test_camel_case_tokenized():
    assert set(camel_snake_tokenize("getUserById")) >= {"get", "user", "by", "id"}

def test_snake_case_tokenized():
    assert set(camel_snake_tokenize("get_user_by_id")) >= {"get", "user", "by", "id"}

def test_both_forms_same_tokens():
    assert camel_snake_tokenize("getUserById") == camel_snake_tokenize("get_user_by_id")

def test_dynamic_top_k():
    assert compute_top_k(100) == 5     # floor
    assert compute_top_k(1000) == 10   # 1%
    assert compute_top_k(5000) == 20   # cap
    assert compute_top_k(2500) == 20   # cap
    assert compute_top_k(50) == 5      # floor

def test_rrf_rewards_consensus():
    # Chunk in both lists scores higher than chunk in only one
    chunk_both = mock_chunk("both")
    chunk_bm25_only = mock_chunk("bm25_only")
    chunk_semantic_only = mock_chunk("semantic_only")
    results = rrf_merge(
        [chunk_both, chunk_bm25_only],
        [chunk_both, chunk_semantic_only],
        bm25_weight=0.6, semantic_weight=0.4
    )
    assert results[0].chunk_id == "both"

def test_rrf_deduplicates_by_chunk_id():
    chunk = mock_chunk("same_id")
    results = rrf_merge([chunk, chunk], [chunk], bm25_weight=0.6, semantic_weight=0.4)
    assert len([r for r in results if r.chunk_id == "same_id"]) == 1


# test_resolver.py

def test_patch_original_code_must_exist():
    patch = CodePatch(
        file_path="src/auth.py",
        original_code="def login():\n    pass",
        new_code="def login():\n    return True",
        explanation="fix", confidence=0.9, affects_other_files=[]
    )
    with pytest.raises(ValueError, match="original_code not found"):
        apply_patch(patch, "/tmp/fake_repo_that_does_not_exist")

def test_dry_run_does_not_call_github():
    with patch("resolver.pr_creator.Github") as mock_gh:
        result = create_pr(
            mock_patch, "Fix bug", 42, "owner/repo",
            mock_test_result, dry_run=True
        )
        mock_gh.assert_not_called()
        assert result.dry_run is True
        assert result.pr_url is None
        assert result.pr_body  # preview still generated even in dry_run

def test_no_pr_when_no_tests():
    no_test_result = TestResult(
        passed=False, framework="unknown",
        total_tests=None, failed_tests=[],
        output_summary="No tests found", timed_out=False
    )
    with patch("resolver.pr_creator.Github") as mock_gh:
        result = create_pr(
            mock_patch, "Fix bug", 1, "owner/repo",
            no_test_result, dry_run=False
        )
        mock_gh.assert_not_called()
        assert result.pr_url is None

def test_write_mode_disabled_by_default():
    # apply_patch should not write to disk when WRITE_MODE_ENABLED=False
    import config
    assert config.WRITE_MODE_ENABLED is False


# test_logging.py

def test_llm_call_logs_on_success(mock_ollama):
    """Verify LLM client emits structured log on successful call."""
    with structlog.testing.capture_logs() as logs:
        client = LLMClient(backend="ollama")
        client.complete("test prompt", FunctionSummary)
    llm_logs = [l for l in logs if l.get("event") == "llm_call_completed"]
    assert len(llm_logs) == 1
    assert llm_logs[0]["backend"] == "ollama"
    assert llm_logs[0]["success"] is True
    assert "latency_ms" in llm_logs[0]
    assert "retry_count" in llm_logs[0]

def test_llm_call_logs_retry_on_validation_failure(mock_ollama_bad_then_good):
    """Verify LLM client logs WARNING on retry and INFO on eventual success."""
    with structlog.testing.capture_logs() as logs:
        client = LLMClient(backend="ollama")
        client.complete("test prompt", FunctionSummary)
    retry_logs = [l for l in logs if l.get("event") == "llm_call_retry"]
    assert len(retry_logs) >= 1
    assert "validation_error" in retry_logs[0]
    success_logs = [l for l in logs if l.get("event") == "llm_call_completed"]
    assert len(success_logs) == 1

def test_tree_sitter_error_nodes_logged_in_chunker():
    """Verify partial parse warnings are emitted."""
    code_with_syntax_error = "def foo(\n    x = 1"
    with structlog.testing.capture_logs() as logs:
        chunks = chunk_file("bad.py", code_with_syntax_error, "python")
    warnings = [l for l in logs if l.get("event") == "tree_sitter_error_node"]
    assert len(warnings) > 0
    assert warnings[0]["file_path"] == "bad.py"
    assert "byte_offset" in warnings[0]

def test_silent_escalation_logged_in_qa():
    """Verify silent escalation from fast → thinking emits WARNING."""
    # Requires mock retrieval that returns empty for fast, non-empty for thinking
    with structlog.testing.capture_logs() as logs:
        answer = answer_question("obscure query nobody would ask", mode="fast")
    escalation_logs = [l for l in logs if l.get("event") == "silent_escalation"]
    assert len(escalation_logs) == 1
    assert escalation_logs[0]["reason"] == "zero_results_fast_mode"

def test_trace_id_propagates_through_pipeline():
    """Verify all log events in a pipeline run share the same trace_id."""
    with structlog.testing.capture_logs() as logs:
        answer = answer_question("what does create_user do?", mode="fast")
    trace_ids = {l.get("trace_id") for l in logs if l.get("trace_id")}
    assert len(trace_ids) == 1  # all events share one trace_id

def test_indexing_logs_summary_stats():
    """Verify indexing emits a final summary with chunk counts."""
    with structlog.testing.capture_logs() as logs:
        index_repo("/tmp/test_repo")
    completed = [l for l in logs if l.get("event") == "indexing_completed"]
    assert len(completed) == 1
    assert "total_chunks" in completed[0]
    assert "duration_ms" in completed[0]
    assert "vocabulary_size" in completed[0]

def test_routing_logs_all_layers():
    """Verify query router logs results from each layer attempted."""
    with structlog.testing.capture_logs() as logs:
        route_query(parsed_query, mode="thinking")
    layer1 = [l for l in logs if l.get("event") == "layer1_result"]
    layer2 = [l for l in logs if l.get("event") == "layer2_result"]
    layer3 = [l for l in logs if l.get("event") == "layer3_result"]
    assert len(layer1) == 1
    assert len(layer2) == 1
    assert len(layer3) == 1  # thinking mode runs all layers

def test_patch_not_found_logged_as_warning():
    """Verify apply_patch logs WARNING before raising ValueError."""
    patch = CodePatch(
        file_path="nonexistent.py",
        original_code="not here",
        new_code="replacement",
        explanation="test", confidence=0.5, affects_other_files=[]
    )
    with structlog.testing.capture_logs() as logs:
        with pytest.raises(ValueError):
            apply_patch(patch, "/tmp/fake_repo")
    warnings = [l for l in logs if l.get("event") == "patch_original_not_found"]
    assert len(warnings) == 1
