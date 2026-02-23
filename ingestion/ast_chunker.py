"""
AST-based chunker for Oracle.

Parses Python source files using tree-sitter and produces ChunkMetadata
objects at four structural levels: directory, file, class, and function.
No fixed-token-window chunking — every boundary comes from the AST.
"""

from __future__ import annotations

import re
from pathlib import Path
from pydantic import BaseModel
import tree_sitter_python as tspython
from tree_sitter import Language, Parser


# ---------------------------------------------------------------------------
# Tree-sitter setup — done once at module import, reused for every parse call
# ---------------------------------------------------------------------------

PY_LANGUAGE = Language(tspython.language())  # Load the compiled Python grammar
PARSER = Parser(PY_LANGUAGE)                 # Parser bound to that grammar


# ---------------------------------------------------------------------------
# Chunk size limits — pulled from CLAUDE.md config spec
# ---------------------------------------------------------------------------

MAX_FUNCTION_LINES = 150          # Functions longer than this get truncated
MAX_FUNCTION_TRUNCATED_HEAD = 50  # Keep the first 50 lines …
MAX_FUNCTION_TRUNCATED_TAIL = 20  # … and the last 20 lines

MIN_DIRECTORY_FILES = 3   # Directories with fewer files don't get a chunk
MAX_DIRECTORY_FILES = 20  # Directories with more files get split by subdir


# ---------------------------------------------------------------------------
# ChunkMetadata — the universal schema every chunk conforms to
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    chunk_id: str
    # Format: "{file_path}::{class_name}::{function_name}::{level}"
    # "None" string for absent class/function names.
    # This gives every chunk a globally unique, human-readable ID.

    level: str              # "directory" | "file" | "class" | "function"
    file_path: str          # Relative to repo root. "None" for directory chunks.
    directory_path: str     # Relative to repo root. Always set.
    language: str           # "python" for now. JS/TS added later.
    class_name: str | None
    function_name: str | None
    start_line: int         # 0 for directory and file level chunks
    end_line: int           # 0 for directory and file level chunks
    parent_directory_chunk_id: str | None
    parent_file_chunk_id: str | None    # Always set for class and function chunks
    parent_class_chunk_id: str | None   # Set only for methods inside a class
    content: str            # The actual text stored for this chunk


# ---------------------------------------------------------------------------
# ID builders — one consistent format for all levels
# ---------------------------------------------------------------------------

def _make_chunk_id(
    file_path: str,
    class_name: str | None,
    function_name: str | None,
    level: str,
) -> str:
    """Build a chunk_id in the format file::class::function::level.

    Uses the literal string "None" for absent parts so the ID is always
    four colon-separated segments — easy to split and guaranteed unique
    within one repository.
    """
    return f"{file_path}::{class_name or 'None'}::{function_name or 'None'}::{level}"


def _make_directory_chunk_id(dir_path: str) -> str:
    """Directory chunks have no file/class/function — just the path and level."""
    return f"{dir_path}::None::None::directory"


# ---------------------------------------------------------------------------
# AST helpers — extract specific pieces from tree-sitter nodes
# ---------------------------------------------------------------------------

def _node_text(node, source_bytes: bytes) -> str:
    """Extract the source text that a tree-sitter node spans.

    tree-sitter works with byte offsets, not character offsets.
    source_bytes must be the same bytes object passed to parser.parse().
    """
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8")


def _get_docstring(body_node, source_bytes: bytes) -> str | None:
    """Extract the docstring from a class or function body if present.

    In Python's CST, a docstring is the first child of the body block,
    and that child is an expression_statement containing a single string node.
    """
    if body_node is None or body_node.child_count == 0:
        return None

    first_stmt = body_node.children[0]  # First statement in the body

    # Must be an expression_statement (bare expression, not assignment etc.)
    if first_stmt.type != "expression_statement":
        return None

    # The expression_statement must contain exactly one string child
    if first_stmt.child_count == 1 and first_stmt.children[0].type == "string":
        return _node_text(first_stmt.children[0], source_bytes)

    return None


def _get_body_node(node):
    """Find the 'block' (body) child of a class or function node.

    In tree-sitter-python, class_definition and function_definition nodes
    have a child named 'body' which is a 'block' node containing all the
    statements. We look it up by field name.
    """
    return node.child_by_field_name("body")


def _get_function_signature(node, source_bytes: bytes) -> str:
    """Extract just the 'def name(params) -> return:' line, without the body.

    Strategy: take everything from the start of the node up to (but not
    including) the colon that begins the body block. This captures the
    def keyword, name, parameters, return annotation, and decorators if
    they're part of the node.
    """
    body = _get_body_node(node)
    if body is None:
        # Degenerate case (shouldn't happen for valid Python) — return whole node
        return _node_text(node, source_bytes)

    # Everything from the node's start up to the body's start is the signature.
    # We strip trailing whitespace and the colon.
    sig_bytes = source_bytes[node.start_byte:body.start_byte]
    sig = sig_bytes.decode("utf-8").rstrip().rstrip(":")
    return sig.strip()


def _get_class_signature(node, source_bytes: bytes) -> str:
    """Extract the 'class Name(bases):' line without the body.

    Same approach as function signature: everything before the body block.
    """
    body = _get_body_node(node)
    if body is None:
        return _node_text(node, source_bytes)

    sig_bytes = source_bytes[node.start_byte:body.start_byte]
    sig = sig_bytes.decode("utf-8").rstrip().rstrip(":")
    return sig.strip()


def _collect_decorators(node, source_bytes: bytes) -> str:
    """Collect all decorator lines above a class or function definition.

    tree-sitter-python wraps decorated definitions in a 'decorated_definition'
    node, but when we walk the AST we already handle that. This helper works
    on the definition node itself — decorators are children with type 'decorator'.
    """
    decorators = []
    for child in node.children:
        if child.type == "decorator":
            decorators.append(_node_text(child, source_bytes))
    return "\n".join(decorators)


# ---------------------------------------------------------------------------
# Truncation — enforces the max function body size from CLAUDE.md
# ---------------------------------------------------------------------------

def _maybe_truncate(content: str) -> str:
    """If a function body exceeds MAX_FUNCTION_LINES, keep the head and tail
    with a clear marker showing how many lines were omitted.

    This prevents enormous functions from bloating the index while still
    capturing the signature/setup (head) and the return/cleanup (tail).
    """
    lines = content.split("\n")
    if len(lines) <= MAX_FUNCTION_LINES:
        return content  # Under the limit — keep as-is

    head = lines[:MAX_FUNCTION_TRUNCATED_HEAD]
    tail = lines[-MAX_FUNCTION_TRUNCATED_TAIL:]
    omitted = len(lines) - MAX_FUNCTION_TRUNCATED_HEAD - MAX_FUNCTION_TRUNCATED_TAIL
    marker = f"    [TRUNCATED: {omitted} lines omitted]"

    return "\n".join(head + [marker] + tail)


# ---------------------------------------------------------------------------
# Import extraction — used for file-level chunks
# ---------------------------------------------------------------------------

def _collect_imports(root_node, source_bytes: bytes) -> list[str]:
    """Walk top-level children and collect all import statements.

    Only looks at direct children of the module node — nested imports
    (inside functions or if-blocks) are intentionally ignored because
    they're implementation details, not part of the file's interface.
    """
    imports = []
    for child in root_node.children:
        if child.type in ("import_statement", "import_from_statement"):
            imports.append(_node_text(child, source_bytes))
    return imports


# ---------------------------------------------------------------------------
# Core: chunk_file — parse one file, produce all file/class/function chunks
# ---------------------------------------------------------------------------

def chunk_file(
    file_path: str,
    source_code: str,
    language: str,
) -> list[ChunkMetadata]:
    """Parse a single source file and return chunks at file, class, and function levels.

    This is a single-pass walk over the tree-sitter AST:
    1. Walk top-level children of the module node
    2. When we hit a class: create a class chunk + a function chunk per method
    3. When we hit a function: create a function chunk
    4. After the walk: assemble a file-level chunk from collected signatures

    Returns a flat list of ChunkMetadata — caller doesn't need to know the
    tree structure because parent IDs encode it.
    """
    if language != "python":
        raise ValueError(f"Unsupported language: {language}. Only 'python' is supported.")

    source_bytes = source_code.encode("utf-8")  # tree-sitter needs bytes
    tree = PARSER.parse(source_bytes)            # Parse into a concrete syntax tree
    root = tree.root_node                        # Module-level node

    # The directory this file lives in (for parent_directory_chunk_id)
    dir_path = str(Path(file_path).parent)
    if dir_path == ".":
        dir_path = ""  # Root-level files have no directory prefix

    # The file-level chunk ID — every class/function chunk will reference this
    file_chunk_id = _make_chunk_id(file_path, None, None, "file")
    dir_chunk_id = _make_directory_chunk_id(dir_path) if dir_path else None

    chunks: list[ChunkMetadata] = []

    # Accumulators for the file-level chunk content
    file_signatures: list[str] = []  # One-line signatures of all top-level definitions
    imports = _collect_imports(root, source_bytes)

    # --- Walk top-level children of the module ---
    for child in root.children:
        # Handle decorated definitions by unwrapping them.
        # tree-sitter-python wraps @decorator + def/class in a
        # 'decorated_definition' node. We need the inner node for type
        # checking, but the outer node for correct line numbers.
        actual_node = child  # The node we inspect for type
        outer_node = child   # The node we use for line numbers (includes decorators)
        decorators_text = ""

        if child.type == "decorated_definition":
            # The last child is the actual class or function definition
            actual_node = child.children[-1]
            decorators_text = _collect_decorators(child, source_bytes)

        # --- Top-level function ---
        if actual_node.type in ("function_definition", "async_function_definition"):
            func_name = actual_node.child_by_field_name("name")
            func_name_str = _node_text(func_name, source_bytes) if func_name else "unknown"

            full_text = _node_text(outer_node, source_bytes)  # Includes decorators
            truncated = _maybe_truncate(full_text)

            sig = _get_function_signature(actual_node, source_bytes)
            if decorators_text:
                sig = decorators_text + "\n" + sig

            file_signatures.append(sig)  # Add to file-level summary

            chunks.append(ChunkMetadata(
                chunk_id=_make_chunk_id(file_path, None, func_name_str, "function"),
                level="function",
                file_path=file_path,
                directory_path=dir_path,
                language=language,
                class_name=None,               # Top-level function, no class
                function_name=func_name_str,
                start_line=outer_node.start_point[0] + 1,  # tree-sitter is 0-indexed, we want 1-indexed
                end_line=outer_node.end_point[0] + 1,
                parent_directory_chunk_id=dir_chunk_id,
                parent_file_chunk_id=file_chunk_id,
                parent_class_chunk_id=None,    # Not inside a class
                content=truncated,
            ))

        # --- Top-level class ---
        elif actual_node.type == "class_definition":
            class_name_node = actual_node.child_by_field_name("name")
            class_name_str = _node_text(class_name_node, source_bytes) if class_name_node else "unknown"

            class_chunk_id = _make_chunk_id(file_path, class_name_str, None, "class")

            class_sig = _get_class_signature(actual_node, source_bytes)
            if decorators_text:
                class_sig = decorators_text + "\n" + class_sig

            file_signatures.append(class_sig)  # Add to file-level summary

            # --- Build the class-level chunk content ---
            # Contains: signature + docstring + method signatures (NOT bodies)
            body = _get_body_node(actual_node)
            docstring = _get_docstring(body, source_bytes) if body else None

            class_content_parts = [class_sig + ":"]
            if docstring:
                class_content_parts.append(f"    {docstring}")

            # Track class-level variables for the class summary
            method_sigs: list[str] = []

            # --- Walk class body for methods and nested decorated defs ---
            if body:
                for member in body.children:
                    member_actual = member
                    member_outer = member
                    member_decorators = ""

                    if member.type == "decorated_definition":
                        member_actual = member.children[-1]
                        member_decorators = _collect_decorators(member, source_bytes)

                    if member_actual.type in ("function_definition", "async_function_definition"):
                        method_name_node = member_actual.child_by_field_name("name")
                        method_name_str = _node_text(method_name_node, source_bytes) if method_name_node else "unknown"

                        # Method signature for the class-level chunk
                        msig = _get_function_signature(member_actual, source_bytes)
                        if member_decorators:
                            msig = member_decorators + "\n" + msig
                        method_sigs.append(msig)

                        # Full method body for the function-level chunk
                        full_method = _node_text(member_outer, source_bytes)
                        truncated_method = _maybe_truncate(full_method)

                        chunks.append(ChunkMetadata(
                            chunk_id=_make_chunk_id(file_path, class_name_str, method_name_str, "function"),
                            level="function",
                            file_path=file_path,
                            directory_path=dir_path,
                            language=language,
                            class_name=class_name_str,         # This method belongs to this class
                            function_name=method_name_str,
                            start_line=member_outer.start_point[0] + 1,
                            end_line=member_outer.end_point[0] + 1,
                            parent_directory_chunk_id=dir_chunk_id,
                            parent_file_chunk_id=file_chunk_id,
                            parent_class_chunk_id=class_chunk_id,  # Key link: method → class
                            content=truncated_method,
                        ))

            # Add method signatures to class chunk content
            for msig in method_sigs:
                # Indent method signatures under the class, matching Python style
                indented = "\n".join("    " + line for line in msig.split("\n"))
                class_content_parts.append(indented)

            class_content = "\n".join(class_content_parts)

            chunks.append(ChunkMetadata(
                chunk_id=class_chunk_id,
                level="class",
                file_path=file_path,
                directory_path=dir_path,
                language=language,
                class_name=class_name_str,
                function_name=None,           # Class-level, no function
                start_line=outer_node.start_point[0] + 1,
                end_line=outer_node.end_point[0] + 1,
                parent_directory_chunk_id=dir_chunk_id,
                parent_file_chunk_id=file_chunk_id,
                parent_class_chunk_id=None,   # Classes don't nest in our model
                content=class_content,
            ))

    # --- Build the file-level chunk ---
    # Combines all imports + all top-level definition signatures.
    # This is the "table of contents" for the file.
    file_content_parts = []
    if imports:
        file_content_parts.append("\n".join(imports))
        file_content_parts.append("")  # Blank line separating imports from defs
    if file_signatures:
        file_content_parts.append("\n".join(file_signatures))

    chunks.append(ChunkMetadata(
        chunk_id=file_chunk_id,
        level="file",
        file_path=file_path,
        directory_path=dir_path,
        language=language,
        class_name=None,
        function_name=None,
        start_line=0,                        # File-level spans the whole file
        end_line=0,                          # Convention from CLAUDE.md: 0 for file/dir
        parent_directory_chunk_id=dir_chunk_id,
        parent_file_chunk_id=None,           # File IS the file level — no parent file
        parent_class_chunk_id=None,
        content="\n".join(file_content_parts),
    ))

    return chunks


# ---------------------------------------------------------------------------
# Directory-level chunk builder — synthetic, not from tree-sitter
# ---------------------------------------------------------------------------

def build_directory_chunk(
    dir_path: str,
    file_chunks: list[ChunkMetadata],
) -> ChunkMetadata | None:
    """Build a synthetic directory-level chunk from existing file-level chunks.

    Only generates a chunk if the directory contains between MIN_DIRECTORY_FILES
    and MAX_DIRECTORY_FILES files. Below that threshold, aggregation adds no
    value. Above it, the summary would be too large and vague — the caller
    should recurse into subdirectories instead.

    The content follows the structured format from CLAUDE.md:
    - File list
    - Per-file summaries (imports + defined names)
    - Internal import relationships (file A imports from file B)
    - Most exported identifiers
    """
    # Filter to only file-level chunks in this specific directory
    dir_file_chunks = [
        c for c in file_chunks
        if c.level == "file" and c.directory_path == dir_path
    ]

    file_count = len(dir_file_chunks)

    # Enforce the depth limits from CLAUDE.md
    if file_count < MIN_DIRECTORY_FILES or file_count > MAX_DIRECTORY_FILES:
        return None  # Caller must handle: skip or recurse into subdirs

    chunk_id = _make_directory_chunk_id(dir_path)

    # --- File list ---
    file_names = [Path(c.file_path).name for c in dir_file_chunks]

    # --- Per-file summaries ---
    # Parse each file chunk's content to extract imports and defined names.
    file_summaries: list[str] = []
    # Map: filename → set of names it defines (for cross-referencing imports)
    defined_names: dict[str, set[str]] = {}
    # Map: filename → list of raw import lines (for cross-referencing)
    file_imports: dict[str, list[str]] = {}

    for fc in dir_file_chunks:
        fname = Path(fc.file_path).name
        lines = fc.content.split("\n")

        # Separate import lines from definition signatures
        imp_lines = [l for l in lines if l.startswith("import ") or l.startswith("from ")]
        # Definition lines are everything that's not an import and not blank
        def_lines = [l for l in lines if l and not l.startswith("import ") and not l.startswith("from ")]

        # Extract just the names from definition signatures
        # Patterns: "def name(", "async def name(", "class Name("
        names = []
        for dl in def_lines:
            # Match function defs: "def name(" or "async def name("
            m = re.match(r'(?:async\s+)?def\s+(\w+)', dl)
            if m:
                names.append(m.group(1))
                continue
            # Match class defs: "class Name" (with or without parens)
            m = re.match(r'class\s+(\w+)', dl)
            if m:
                names.append(m.group(1))

        defined_names[fname] = set(names)
        file_imports[fname] = imp_lines

        # Build the summary line: "filename: imports X. defines Y, Z"
        imp_modules = []
        for il in imp_lines:
            # Extract module name from "import X" or "from X import ..."
            m = re.match(r'(?:from\s+(\S+)|import\s+(\S+))', il)
            if m:
                imp_modules.append(m.group(1) or m.group(2))

        summary_parts = []
        if imp_modules:
            summary_parts.append(f"imports {', '.join(imp_modules)}")
        if names:
            summary_parts.append(f"defines {', '.join(names)}")

        file_summaries.append(f"  {fname}: {'. '.join(summary_parts)}")

    # --- Internal import relationships ---
    # Check if file A imports names that file B defines (both in this directory)
    internal_imports: list[str] = []
    for fname, imp_lines in file_imports.items():
        for il in imp_lines:
            # Match "from .module import Name" or "from module import Name"
            m = re.match(r'from\s+\.?(\w+)\s+import\s+(.+)', il)
            if not m:
                continue
            imported_module = m.group(1) + ".py"  # Convert module name to filename
            imported_names_str = m.group(2)
            # Check if the imported module is a file in this directory
            if imported_module in defined_names:
                # Extract individual imported names
                imported_names = [n.strip() for n in imported_names_str.split(",")]
                # Keep only names actually defined in that file
                matched = [n for n in imported_names if n in defined_names[imported_module]]
                if matched:
                    internal_imports.append(
                        f"  {fname} imports from {imported_module} ({', '.join(matched)})"
                    )

    # --- Most exported identifiers ---
    # Flatten all defined names across files, sorted for determinism
    all_names = sorted(
        name
        for names_set in defined_names.values()
        for name in names_set
    )

    # --- Assemble final content ---
    content_parts = [
        f"Directory: {dir_path}/",
        f"Files: [{', '.join(sorted(file_names))}]",
        "",
        "File summaries:",
        *file_summaries,
    ]

    if internal_imports:
        content_parts.append("")
        content_parts.append("Internal import relationships:")
        content_parts.extend(internal_imports)

    if all_names:
        content_parts.append("")
        content_parts.append(f"Most exported identifiers:")
        content_parts.append(f"  {', '.join(all_names)}")

    return ChunkMetadata(
        chunk_id=chunk_id,
        level="directory",
        file_path="None",         # Directory chunks have no single file
        directory_path=dir_path,
        language="python",
        class_name=None,
        function_name=None,
        start_line=0,             # Convention: 0 for directory
        end_line=0,
        parent_directory_chunk_id=None,  # Could be a parent dir — not implemented yet
        parent_file_chunk_id=None,
        parent_class_chunk_id=None,
        content="\n".join(content_parts),
    )
