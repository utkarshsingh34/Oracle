# Oracle

**Local codebase Q&A and autonomous issue resolution, powered by AST-aware indexing and hybrid retrieval.**

Oracle ingests a GitHub repository, builds a hierarchical index that respects
the natural structure of code (directories, files, classes, functions), and
answers natural language questions about the codebase with cited references
to exact lines. It can also autonomously resolve GitHub issues by generating
patches, verifying them against the test suite, and opening draft PRs.

Everything runs locally. No code, embeddings, or queries leave your machine.

> **Project Status:** Oracle is under active development following an 8-phase
> build plan. The architectural design is complete and documented in
> [CLAUDE.md](CLAUDE.md). Current progress:
>
> | Phase | Scope | Status |
> |-------|-------|--------|
> | 1 | AST chunker (Python) | Done |
> | 2 | Structured logging + LLM client | Done |
> | 3 | Query preprocessor + technicality scorer + query bank | In progress |
> | 4 | BM25 + semantic retriever + indexer | Planned |
> | 5 | Hybrid retriever + query router | Planned |
> | 6 | Q&A engine + citation builder + debug CLI | Planned |
> | 7 | Issue resolver (analyzer, patch, tests, PR) | Planned |
> | 8 | API server + React UI | Planned |

---

## What Oracle Does

**Codebase Q&A** — Ask questions in plain English, get answers with citations
pointing to specific files and line ranges. Oracle understands questions at
every structural level: "what does this directory do?", "what's in auth.py?",
"what methods does UserService expose?", "how is create_user implemented?"

**Autonomous Issue Resolution** — Point Oracle at a GitHub issue. It analyzes
the issue text, identifies relevant code, generates a search-and-replace patch,
runs your test suite, and opens a draft PR if tests pass. Everything defaults
to dry-run mode — nothing writes to your repo or opens a PR without explicit
opt-in.

**Privacy by Design** — Oracle runs entirely on local infrastructure. Code
is parsed and embedded locally using open-source models. LLM inference runs
through a local Ollama instance by default. There are no external API calls,
no telemetry, and no data exfiltration.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                         User Query                           │
│              "what does getUserById return?"                  │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Query Preprocessor                         │
│  Extract identifiers, file paths, code blocks, keywords      │
│  Produce ParsedQuery with structural metadata                │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Technicality Scorer                         │
│  Score 0.0 (semantic) → 1.0 (technical)                      │
│  Dynamically weight BM25 vs semantic retrieval               │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              Three-Layer Query Router                         │
│  Layer 1: Regex patterns     (~40% of queries, <1ms)         │
│  Layer 2: Embedding similarity against query bank (<5ms)     │
│  Layer 3: LLM classification (fallback, ~500ms)              │
│  → Determines which index level(s) to search                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Hybrid Retriever                            │
│  BM25 (exact token matching, camelCase-aware tokenization)   │
│  + Semantic (sentence-transformers + ChromaDB)               │
│  → Merged via Weighted Reciprocal Rank Fusion                │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              Context Expansion + LLM Synthesis                │
│  Fetch parent chain: function → class → file → directory     │
│  Deduplicate, assemble context, generate cited answer        │
└──────────────────────────────────────────────────────────────┘
```

---

## The Four-Level Index

Oracle's foundational design decision is that code has natural structural
units defined by the language itself. These units — not arbitrary token
windows — serve as chunk boundaries.

| Level | What It Stores | What It Answers |
|-------|----------------|-----------------|
| **Directory** | File summaries, internal import relationships, top exported identifiers | "What does this module do?" System-level, architectural questions |
| **File** | File path, all imports, all class/function signatures (no bodies) | "What's in this file?" Overview and structure questions |
| **Class** | Class name, docstring, method signatures, class variables (no method bodies) | "What does this class expose?" Interface and capability questions |
| **Function** | Full function/method body including docstring | "How is this implemented?" Logic, parameters, return values |

Every chunk carries parent references in its metadata, enabling automatic
context expansion: when a function chunk is retrieved, Oracle also fetches
its parent class, file, and directory chunks to give the LLM full structural
context.

### AST-Based Chunking

All chunking uses **tree-sitter** for language-aware AST parsing. Oracle never
splits code by fixed token windows — every chunk boundary corresponds to a
real syntactic unit in the source code. This means function chunks contain
complete functions, class chunks contain complete class signatures, and file
chunks contain complete import blocks.

Tree-sitter parse anomalies (ERROR and MISSING nodes) are detected and logged
as warnings rather than silently ignored, so you always know when chunk quality
is degraded for a particular file.

### Long Function Summarization

Functions exceeding 150 lines are summarized by the LLM at index time rather
than truncated. The summary is stored as the chunk's searchable content, while
the full body is preserved for on-demand deep queries. This solves the context
window problem without sacrificing retrieval quality — the summary is what gets
embedded and searched, but the full implementation is always available when
needed.

---

## Retrieval System

Oracle uses hybrid retrieval combining BM25 (lexical) and semantic search,
merged via Weighted Reciprocal Rank Fusion (RRF).

### Dynamic Retrieval Weighting

The **technicality scorer** analyzes each query to determine how technical it
is on a 0.0–1.0 scale using three signals: presence of code identifiers
(camelCase, snake_case patterns), presence of file paths, and vocabulary
overlap with the indexed codebase.

This score dynamically adjusts retrieval weights:

| Query Type | Example | BM25 Weight | Semantic Weight |
|------------|---------|-------------|-----------------|
| Exact identifier + file | "in auth.py what does create_user do?" | 0.80 | 0.20 |
| Identifier only | "what does getUserById return?" | 0.65 | 0.35 |
| Vague technical | "where is user creation handled?" | 0.50 | 0.50 |
| Fully semantic | "what does the account generator do?" | 0.30 | 0.70 |

### BM25 with Code-Aware Tokenization

The BM25 retriever uses custom `camelCase`/`snake_case` tokenization so that
`getUserById` and `get_user_by_id` produce identical token sets
(`["get", "user", "by", "id"]`). Four separate BM25 indexes are maintained,
one per hierarchy level.

### Query Router

A three-layer routing system determines which hierarchy levels to search:

1. **Regex patterns** — fast, deterministic, handles ~40% of queries in <1ms
2. **Embedding similarity** — compares the query against a pre-built bank of
   ~1,500 code query embeddings (built from CodeSearchNet, CoSQA, and StaQC),
   routes by nearest-neighbor level in <5ms
3. **LLM classification** — fallback for ambiguous queries, ~500ms

If routing is uncertain, Oracle searches all levels. A wrong restriction
produces silent failures; searching everything is always correct, just slower.

### Fast and Thinking Modes

**Fast mode** (default): runs routing layers 1–2 only, searches the
single most confident level. Routing overhead ~5–10ms.

**Thinking mode**: runs all three routing layers, searches all four levels
simultaneously, merges results with cross-level RRF. Overhead ~500ms–2s.
Best for vague, architectural, or cross-cutting questions.

If Fast mode returns zero results, Oracle silently escalates to Thinking mode
before responding — the user never sees an empty result without a thorough
search.

---

## Issue Resolution Pipeline

```
GitHub Issue
     │
     ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Analyze    │────▶│  Generate   │────▶│  Run Tests  │────▶│  Create PR  │
│   Issue      │     │  Patch      │     │             │     │  (Draft)    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
 Extract IDs,        Search-and-         pytest / jest        dry_run=True
 find relevant       replace format,     with timeout         by default
 code chunks         exact substring
                     match required
```

The resolver extracts identifiers from the issue text, cross-references them
against the index to find real codebase symbols, retrieves relevant code with
function-level results weighted 2× (bugs live in function bodies), generates a
search-and-replace patch, runs the test suite, and opens a draft PR if tests
pass.

Safety constraints are non-negotiable: patches require exact substring matching
(no fuzzy application), writing to the repo requires explicit opt-in
(`WRITE_MODE_ENABLED=True`), PRs default to dry-run mode, and the test suite
must pass before any PR is opened. If no tests exist, Oracle halts and reports
rather than opening an unverified PR.

---

## Observability

Oracle uses **structlog** for comprehensive structured logging throughout every
module. All logging produces key-value pairs that are machine-parseable (JSON)
or human-readable (pretty-print), controlled by a single config variable.

Every pipeline run (indexing, querying, issue resolution) is tagged with a
**trace ID** — a 12-character hex identifier that propagates through every
downstream log event, enabling full reconstruction of any pipeline run from
log output.

Key things Oracle logs: tree-sitter parse anomalies, LLM call latencies and
validation failures, retrieval scores and consensus hits, routing decisions
across all layers, silent escalation events, and end-to-end pipeline timing.

A `--debug-query` CLI flag forces DEBUG-level pretty-printed output for a
single query, showing the full pipeline trace from preprocessing through
routing, retrieval, context expansion, and LLM synthesis.

```bash
# Normal query
python -m oracle.main query "what does create_user return?"

# Full debug trace
python -m oracle.main query "what does create_user return?" --debug-query

# Filter production logs by trace ID
cat oracle.log | jq 'select(.trace_id == "a1b2c3d4e5f6")'
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| AST Parsing | tree-sitter | Language-aware code chunking (Python, JavaScript, TypeScript) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) | Local semantic embeddings, no paid APIs |
| Vector Store | ChromaDB | Persistent vector storage, one collection per hierarchy level |
| Lexical Search | BM25Okapi | Token-level retrieval with code-aware tokenization |
| LLM (default) | Ollama + qwen2.5-coder:7b | Local structured inference for summaries, Q&A, patches |
| LLM (optional) | Claude via Anthropic API | Higher quality when local resources are limited |
| Structured Output | instructor | Pydantic-enforced LLM output schemas, retry on validation failure |
| Logging | structlog | Structured key-value logging with trace ID propagation |
| GitHub Integration | PyGithub | Draft PR creation (optional, requires GITHUB_TOKEN) |
| API Server | FastAPI | SSE streaming for indexing progress, REST endpoints for Q&A |
| UI | React (single file) | Query input with syntax highlighting, progress bar, citations |

---

## Project Structure

```
oracle/
├── config.py                 # All configuration: LLM, retrieval, chunking, logging
├── logging_config.py         # structlog setup, trace ID binding
├── main.py                   # CLI entrypoint
├── scripts/
│   └── build_query_bank.py   # One-time: build query routing embeddings
├── data/
│   └── query_bank.npz        # Pre-built query routing bank (~1500 embeddings)
├── ingestion/
│   ├── repo_loader.py        # Clone/load repo, walk and filter files
│   ├── ast_chunker.py        # tree-sitter chunker (the architectural core)
│   └── indexer.py            # Orchestrates: chunk → summarize → embed → store
├── retrieval/
│   ├── query_preprocessor.py # Parse queries: identifiers, paths, code blocks
│   ├── technicality_scorer.py# Score query technicality, compute retrieval weights
│   ├── query_router.py       # 3-layer routing: regex → embedding → LLM
│   ├── bm25_retriever.py     # BM25 with camelCase/snake_case tokenization
│   ├── semantic_retriever.py # sentence-transformers + ChromaDB
│   └── hybrid_retriever.py   # Weighted RRF fusion
├── qa/
│   ├── qa_engine.py          # Full Q&A pipeline: route → retrieve → expand → answer
│   └── citation_builder.py   # Map chunks to file/line citations
├── resolver/
│   ├── issue_analyzer.py     # Extract symbols from issue, find relevant code
│   ├── patch_generator.py    # LLM-generated search-and-replace patches
│   ├── test_runner.py        # pytest/jest detection and execution
│   └── pr_creator.py         # Draft PR via GitHub API (dry-run default)
├── llm/
│   ├── client.py             # Backend-agnostic LLM client (Ollama/Claude)
│   └── schemas.py            # All Pydantic output schemas
├── api/
│   └── server.py             # FastAPI with SSE streaming
├── ui/
│   └── app.jsx               # React UI: query input, progress, citations
└── tests/
    ├── test_chunker.py
    ├── test_preprocessor.py
    ├── test_retrieval.py
    ├── test_router.py
    ├── test_resolver.py
    ├── test_qa.py
    ├── test_api.py
    └── test_logging.py
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running with `qwen2.5-coder:7b`

```bash
ollama pull qwen2.5-coder:7b
```

### Installation

```bash
git clone https://github.com/your-username/oracle.git
cd oracle
pip install -r requirements.txt
```

### Build the Query Bank (One-Time)

```bash
python -m oracle.scripts.build_query_bank
```

This downloads CodeSearchNet, CoSQA, and StaQC datasets, builds clustered
embeddings for query routing, and saves `data/query_bank.npz`. Takes a few
minutes, runs once.

### Index a Repository

```bash
python -m oracle.main index /path/to/repo
```

### Ask Questions

```bash
# Fast mode (default)
python -m oracle.main query "what does create_user return?"

# Thinking mode (thorough, searches all levels)
python -m oracle.main query "how does authentication work?" --mode thinking

# Debug trace (see full pipeline)
python -m oracle.main query "what does create_user return?" --debug-query
```

### Resolve an Issue

```bash
# Dry run (default) — generates patch and PR body without writing anything
python -m oracle.main resolve \
  --repo owner/repo \
  --issue 42

# Live run — applies patch, runs tests, opens draft PR
python -m oracle.main resolve \
  --repo owner/repo \
  --issue 42 \
  --execute
```

### Start the Web UI

```bash
python -m oracle.api.server
```

---

## Configuration

All configuration lives in `config.py`. Key settings:

```python
# LLM Backend
LLM_BACKEND = "ollama"              # "ollama" or "claude"
OLLAMA_MODEL = "qwen2.5-coder:7b"   # Code-specific fine-tuned variant
CLAUDE_MODEL = "claude-sonnet-4-6"  # Optional: higher quality, requires API key

# Retrieval
RRF_K = 60                          # RRF smoothing constant (do not change)
SUMMARIZE_MIN_LINES = 150           # Functions at/over this get LLM summaries

# Safety
WRITE_MODE_ENABLED = False           # Must be True to write patches to disk
TEST_TIMEOUT_SECONDS = 120           # Test suite timeout

# Logging
LOG_LEVEL = "INFO"                   # DEBUG | INFO | WARNING | ERROR
LOG_FORMAT = "pretty"                # "pretty" for terminal, "json" for production
LOG_LLM_PROMPTS = False              # True to log full LLM prompts at DEBUG
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `GITHUB_TOKEN` | Required only for PR creation. Never hardcoded. |
| `ANTHROPIC_API_KEY` | Required only when `LLM_BACKEND="claude"`. |

---

## Design Decisions

**Why AST chunking over token windows?** Token windows split code at arbitrary
points — mid-function, mid-class, mid-expression. The chunks are meaningless
as retrieval units. AST chunking produces chunks that correspond to real
syntactic units developers think in: functions, classes, files. Every chunk is
a complete, self-contained unit of code.

**Why hybrid retrieval?** BM25 excels at exact identifier matches
(`getUserById` → finds the function immediately) but fails on semantic queries
("where is the thing that generates accounts?"). Semantic search handles
rephrased concepts but misses exact names. Hybrid retrieval with dynamic
weighting gives you both, favoring the right method for each query.

**Why summarize long functions instead of sub-chunking?** Sub-chunking (splitting
a 300-line function into overlapping pieces) helps retrieval by producing more
focused embeddings. But sub-chunking doesn't help the LLM — all pieces still
need to fit in the context window. Only summarization solves the context window
problem. Oracle stores both the summary (for search) and the full body (for
on-demand deep queries) so nothing is lost.

**Why three routing layers?** Regex handles the obvious 40% of queries for
free. Embedding similarity handles the next 50% in <5ms. LLM classification
handles the ambiguous remainder at ~500ms. Layering avoids paying LLM latency
for every query while ensuring nothing falls through the cracks.

**Why local-only?** Codebases are proprietary. Sending code to external
embedding APIs or LLM providers is a non-starter for many teams. Oracle
proves that local models with smart retrieval architecture can deliver useful
codebase Q&A without any external dependencies.

---

## Supported Languages

Oracle currently supports Python, JavaScript, and TypeScript via tree-sitter
parsers. Adding a new language requires defining its class and function node
types in the AST chunker — the rest of the pipeline is language-agnostic.

---

## Limitations

Oracle answers "what is in this codebase" and "how is it implemented." It
does not answer "why was this decision made" — that context lives in commit
messages, PR descriptions, and design documents, which Oracle does not index.

The issue resolver generates patches based on the code it can see. Complex
multi-file refactors, changes requiring new dependencies, or issues that need
context beyond the codebase (external API changes, infrastructure issues) are
outside its scope.
