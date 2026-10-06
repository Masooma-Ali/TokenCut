"""Find which files and symbols the query mentions (IMPLEMENTATION_PLAN C1)."""

import re
from dataclasses import dataclass, field

from tokencut.contracts import Chunk, ChunkKind

SYMBOL_WEIGHT = 1.0  # query names a function/class/method (or a traceback points into it)
FILE_WEIGHT = 0.5  # query only names the file

_FILE_RE = re.compile(r"[\w./-]+\.py\b")
_TRACEBACK_RE = re.compile(r'File "([^"]+\.py)", line (\d+)(?:, in (\w+))?')
_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
MIN_SYMBOL_LENGTH = 3

STOP_WORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "any",
        "can",
        "had",
        "has",
        "her",
        "was",
        "one",
        "our",
        "out",
        "day",
        "get",
        "him",
        "his",
        "how",
        "man",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "its",
        "did",
        "let",
        "put",
        "say",
        "she",
        "too",
        "use",
        "with",
        "that",
        "this",
        "from",
        "have",
        "they",
        "will",
        "what",
        "when",
        "where",
        "which",
        "while",
        "would",
        "there",
        "their",
        "them",
        "then",
        "than",
        "been",
        "were",
        "into",
        "some",
        "such",
        "only",
        "also",
        "just",
        "like",
        "more",
        "most",
        "other",
        "very",
        "should",
        "could",
        "about",
        "after",
        "before",
        "because",
        "does",
        "doing",
        "done",
        "each",
        "here",
        "same",
        "these",
        "those",
        "under",
        "until",
        "upon",
        "your",
        "yours",
        "error",
        "errors",
        "issue",
        "bug",
        "fix",
        "fixed",
        "wrong",
        "instead",
        "expected",
        "actual",
        "should",
        "none",
        "true",
        "false",
        "self",
        "cls",
        "return",
        "returns",
        "raise",
        "raises",
        "import",
        "def",
        "class",
        "pass",
        "python",
        "file",
        "line",
        "code",
        "test",
        "tests",
        "value",
        "values",
    ]
)


@dataclass
class QueryMentions:
    file_paths: list[str] = field(default_factory=list)  # repo files the query names, sorted
    symbols: list[str] = field(
        default_factory=list
    )  # query words that matched a chunk name, sorted
    chunk_ids: list[str] = field(default_factory=list)  # every matched chunk, sorted
    weights: dict[str, float] = field(
        default_factory=dict
    )  # chunk id -> SYMBOL_WEIGHT / FILE_WEIGHT


def _path_matches(mention: str, file_path: str) -> bool:
    """True if one path is a suffix of the other on a '/' boundary ('cart.py' ~ 'shop/cart.py')."""
    if mention == file_path:
        return True
    return file_path.endswith("/" + mention) or mention.endswith("/" + file_path)


def analyse_query(query: str, chunks: list[Chunk]) -> QueryMentions:
    text = query.replace("\\", "/")  # Windows paths in tracebacks
    files = sorted({c.file_path for c in chunks})
    chunks_by_file: dict[str, list[Chunk]] = {}
    for c in chunks:
        chunks_by_file.setdefault(c.file_path, []).append(c)

    weights: dict[str, float] = {}
    matched_files: set[str] = set()
    matched_symbols: set[str] = set()

    def give(chunk_id: str, weight: float) -> None:
        weights[chunk_id] = max(weights.get(chunk_id, 0.0), weight)

    # 1. Traceback lines: the innermost chunk around the line is a strong (symbol-level) signal.
    for match in _TRACEBACK_RE.finditer(text):
        path, line, func = match.group(1), int(match.group(2)), match.group(3)
        for file_path in files:
            if not _path_matches(path.lstrip("./"), file_path):
                continue
            matched_files.add(file_path)
            target = _innermost(chunks_by_file[file_path], line)
            if target is None and func:
                target = next(
                    (
                        c
                        for c in chunks_by_file[file_path]
                        if c.qualified_name.split(".")[-1] == func
                    ),
                    None,
                )
            if target is not None:
                give(target.id, SYMBOL_WEIGHT)

    # 2. Plain file mentions: every chunk of the file gets the weaker file weight.
    for match in _FILE_RE.finditer(text):
        mention = match.group(0).lstrip("./")
        for file_path in files:
            if _path_matches(mention, file_path):
                matched_files.add(file_path)
    for file_path in sorted(matched_files):
        for c in chunks_by_file[file_path]:
            give(c.id, FILE_WEIGHT)

    # 3. Symbols: identifiers / dotted names matched against qualified names.
    by_qualified: dict[str, list[str]] = {}
    by_last: dict[str, list[str]] = {}
    for c in chunks:
        if c.kind == ChunkKind.MODULE:
            continue
        by_qualified.setdefault(c.qualified_name, []).append(c.id)
        by_last.setdefault(c.qualified_name.split(".")[-1], []).append(c.id)

    for match in _SYMBOL_RE.finditer(_FILE_RE.sub(" ", text)):
        symbol = match.group(0).strip(".")
        if len(symbol) < MIN_SYMBOL_LENGTH or symbol.lower() in STOP_WORDS:
            continue
        ids = _match_symbol(symbol, by_qualified, by_last)
        if ids:
            matched_symbols.add(symbol)
            for chunk_id in ids:
                give(chunk_id, SYMBOL_WEIGHT)

    return QueryMentions(
        file_paths=sorted(matched_files),
        symbols=sorted(matched_symbols),
        chunk_ids=sorted(weights),
        weights=dict(sorted(weights.items())),
    )


def _match_symbol(
    symbol: str, by_qualified: dict[str, list[str]], by_last: dict[str, list[str]]
) -> list[str]:
    """'Cart.total' -> exact qualified name; 'total' -> any chunk whose last name is 'total'.

    For a dotted symbol that is not an exact qualified name ('self.subtotal', 'shop.cart.Cart')
    try its longest suffix that is one, then fall back to its last component.
    """
    parts = symbol.split(".")
    for i in range(len(parts) - 1):  # suffixes with at least two parts, longest first
        suffix = ".".join(parts[i:])
        if suffix in by_qualified:
            return by_qualified[suffix]
    last = parts[-1]
    if len(last) < MIN_SYMBOL_LENGTH or last.lower() in STOP_WORDS:
        return []
    return by_last.get(last, [])


def _innermost(chunks: list[Chunk], line: int) -> Chunk | None:
    """Smallest non-module chunk whose line range contains the line (METHOD beats CLASS)."""
    best = None
    for c in chunks:
        if c.kind == ChunkKind.MODULE or not (c.start_line <= line <= c.end_line):
            continue
        if best is None or (c.end_line - c.start_line) < (best.end_line - best.start_line):
            best = c
    return best
