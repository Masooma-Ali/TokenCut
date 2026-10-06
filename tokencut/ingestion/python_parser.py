"""Split one Python file into MODULE / CLASS / FUNCTION / METHOD chunks with tree-sitter.

The exact meaning of each chunk is fixed in docs/IMPLEMENTATION_PLAN.md section 2
("Precise meaning of a chunk"); this file implements those rules.
"""

import ast
import hashlib
import logging

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from tokencut.contracts import Chunk, ChunkKind
from tokencut.tokens import count_tokens

log = logging.getLogger(__name__)

PY_LANGUAGE = Language(tspython.language())
MODULE_NAME = "<module>"


def _decode(source: bytes) -> tuple[str, str]:
    """Return (text, encoding). latin-1 never fails and keeps one char per byte, so lines match."""
    try:
        return source.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return source.decode("latin-1"), "latin-1"


def _line_range(node: Node) -> tuple[int, int]:
    """1-based inclusive line range of a node."""
    start = node.start_point[0] + 1
    end_row, end_col = node.end_point
    # A node that ends exactly at the start of a line does not really occupy that line.
    if end_col == 0 and end_row > node.start_point[0]:
        end_row -= 1
    return start, end_row + 1


def _unwrap(node: Node) -> Node:
    """decorated_definition -> the function/class it decorates (otherwise the node itself)."""
    if node.type == "decorated_definition":
        inner = node.child_by_field_name("definition")
        if inner is not None:
            return inner
    return node


def _docstring(definition: Node) -> str:
    body = definition.child_by_field_name("body")
    if body is None:
        return ""
    first = None
    for child in body.named_children:
        if child.type != "comment":
            first = child
            break
    if first is None or first.type != "expression_statement" or first.named_child_count != 1:
        return ""
    string = first.named_children[0]
    if string.type != "string":
        return ""
    if any(c.type == "interpolation" for c in string.children):
        return ""  # an f-string is not a docstring
    raw = string.text.decode("utf-8", errors="replace")
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw.lstrip("rRbBuU").strip("\"'")
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else ""


class PythonParser:
    """LanguageParser for Python files (DECISIONS #6)."""

    file_suffixes = (".py",)

    def __init__(self) -> None:
        self._parser = Parser(PY_LANGUAGE)

    def parse_file(self, rel_path: str, source: bytes) -> list[Chunk]:
        file_hash = hashlib.sha256(source).hexdigest()
        text, encoding = _decode(source)
        # Strip '\r' so a Windows (CRLF) checkout gives the same chunk code as Linux.
        lines = [line.rstrip("\r") for line in text.split("\n")]
        try:
            return _FileChunker(rel_path, source, encoding, lines, file_hash).run(self._parser)
        except Exception:  # never let one bad file stop the whole repo
            log.warning(
                "could not chunk %s; keeping it as one module chunk", rel_path, exc_info=True
            )
            code = "\n".join(lines).strip("\n")
            return [
                Chunk(
                    id=f"{rel_path}::{MODULE_NAME}",
                    file_path=rel_path,
                    qualified_name=MODULE_NAME,
                    kind=ChunkKind.MODULE,
                    start_line=1,
                    end_line=max(1, len(lines)),
                    code=code,
                    token_count=count_tokens(code),
                    file_hash=file_hash,
                )
            ]


class _FileChunker:
    """Holds the state needed while chunking one file."""

    def __init__(
        self, rel_path: str, source: bytes, encoding: str, lines: list[str], file_hash: str
    ):
        self.rel_path = rel_path
        self.source = source
        self.encoding = encoding
        self.lines = lines
        self.file_hash = file_hash
        self.name_counts: dict[str, int] = {}  # qualified_name -> times seen (for '#2' suffixes)

    # ---------- helpers ----------

    def _code(self, line_numbers: list[int]) -> str:
        return "\n".join(self.lines[n - 1] for n in line_numbers if n - 1 < len(self.lines))

    def _signature(self, outer: Node, definition: Node) -> str:
        """Decorators + header up to and including the ':' (everything before the body)."""
        body = definition.child_by_field_name("body")
        end = body.start_byte if body is not None else definition.end_byte
        raw = self.source[outer.start_byte : end].decode(self.encoding, errors="replace")
        if body is None:
            raw = raw.split("\n")[0]
        return raw.replace("\r", "").rstrip()

    def _new_id(self, qualified_name: str) -> str:
        # DECISIONS #2: a repeated qualified_name in the same file gets '#2', '#3', ... in line order.
        count = self.name_counts.get(qualified_name, 0) + 1
        self.name_counts[qualified_name] = count
        suffix = "" if count == 1 else f"#{count}"
        return f"{self.rel_path}::{qualified_name}{suffix}"

    def _make(
        self,
        kind: ChunkKind,
        qualified_name: str,
        start: int,
        end: int,
        code: str,
        signature: str = "",
        docstring: str = "",
        parent_id: str | None = None,
        chunk_id: str | None = None,
    ) -> Chunk:
        return Chunk(
            id=chunk_id or self._new_id(qualified_name),
            file_path=self.rel_path,
            qualified_name=qualified_name,
            kind=kind,
            parent_id=parent_id,
            start_line=start,
            end_line=end,
            code=code,
            signature=signature,
            docstring=docstring,
            token_count=count_tokens(code),
            file_hash=self.file_hash,
        )

    # ---------- main algorithm ----------

    def run(self, parser: Parser) -> list[Chunk]:
        tree = parser.parse(self.source)
        chunks: list[Chunk] = []
        module_lines: set[int] = set()
        def_lines: set[int] = set()

        for node in tree.root_node.children:
            definition = _unwrap(node)
            start, end = _line_range(node)  # starts at the decorator, if any
            if definition.type == "function_definition":
                name = self._name(definition)
                chunks.append(
                    self._function(node, definition, name, ChunkKind.FUNCTION, start, end)
                )
                def_lines.update(range(start, end + 1))
            elif definition.type == "class_definition":
                chunks.extend(self._class(node, definition, start, end))
                def_lines.update(range(start, end + 1))
            else:
                # imports, assignments, docstring, comments, ERROR regions, stray statements
                module_lines.update(range(start, end + 1))

        kept = sorted(module_lines - def_lines)
        while kept and not self.lines[kept[-1] - 1].strip():
            kept.pop()
        module_code = self._code(kept)
        module = self._make(
            ChunkKind.MODULE,
            MODULE_NAME,
            kept[0] if kept else 1,
            kept[-1] if kept else 1,
            module_code,
            chunk_id=f"{self.rel_path}::{MODULE_NAME}",
        )
        return [module] + chunks

    def _name(self, definition: Node) -> str:
        name_node = definition.child_by_field_name("name")
        if name_node is None:
            return "<anonymous>"
        return name_node.text.decode("utf-8", errors="replace")

    def _function(
        self,
        outer: Node,
        definition: Node,
        qualified_name: str,
        kind: ChunkKind,
        start: int,
        end: int,
        parent_id: str | None = None,
    ) -> Chunk:
        return self._make(
            kind,
            qualified_name,
            start,
            end,
            self._code(list(range(start, end + 1))),
            signature=self._signature(outer, definition),
            docstring=_docstring(definition),
            parent_id=parent_id,
        )

    def _class(self, outer: Node, definition: Node, start: int, end: int) -> list[Chunk]:
        class_name = self._name(definition)
        class_id = self._new_id(class_name)

        methods: list[tuple[Node, Node, int, int]] = []
        body = definition.child_by_field_name("body")
        if body is not None:
            for child in body.children:
                inner = _unwrap(child)
                if inner.type == "function_definition":
                    m_start, m_end = _line_range(child)
                    methods.append((child, inner, m_start, m_end))

        # Class chunk = whole class minus its methods (DECISIONS #2: no double counting).
        method_lines: set[int] = set()
        for _, _, m_start, m_end in methods:
            method_lines.update(range(m_start, m_end + 1))
        class_lines = [n for n in range(start, end + 1) if n not in method_lines]
        while class_lines and not self.lines[class_lines[-1] - 1].strip():
            class_lines.pop()

        chunks = [
            self._make(
                ChunkKind.CLASS,
                class_name,
                start,
                end,
                self._code(class_lines),
                signature=self._signature(outer, definition),
                docstring=_docstring(definition),
                chunk_id=class_id,
            )
        ]
        for child, inner, m_start, m_end in methods:
            qualified = f"{class_name}.{self._name(inner)}"
            chunks.append(
                self._function(child, inner, qualified, ChunkKind.METHOD, m_start, m_end, class_id)
            )
        return chunks
