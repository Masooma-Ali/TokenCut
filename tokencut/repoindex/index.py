"""Repository index for coding agents: symbol map, hybrid code search, exact symbols.

Runs NEXT TO the agent (as a local MCP server, see mcp.py) on the developer's working tree,
so uncommitted edits are included and no repository leaves the machine. The gateway still
only trims what is inside requests (team decision); this is a separate, optional component.

    repo_map(prefix)   directory layout + class / function signatures of the most-used files
    find_code(query)   chunks (whole functions / classes) ranked by graph distance from
                       what the query names + BM25 + file BM25 (+ embeddings if computed)
    symbol(name)       exact source of a symbol with line numbers, its callers and callees

Incremental: every file is remembered with its size, mtime and sha256; a refresh re-parses
only files whose content changed. BM25 is updated per file; the code graph (whole repo,
~5 s on django) is rebuilt in a background thread and the last one is used meanwhile.
"""

import hashlib
import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from tokencut.contracts import Chunk, ChunkKind, EdgeKind
from tokencut.graph.builder import build_graph
from tokencut.graph.queries import GraphIndex, file_of
from tokencut.ingestion.python_parser import PythonParser
from tokencut.ingestion.walker import list_python_files
from tokencut.repoindex.bm25 import Bm25Index, normalised
from tokencut.tokens import count_tokens
from tokencut.trimming.query import SYMBOL_WEIGHT, analyse_query
from tokencut.trimming.scoring import structural_scores, tokenize

log = logging.getLogger(__name__)

INDEX_VERSION = 1
DEFAULT_CACHE_DIR = Path(
    os.environ.get("TOKENCUT_INDEX_DIR", Path.home() / ".cache" / "tokencut" / "repoindex"))
# Score = weighted sum of signals in [0, 1], times TEST_FACTOR for chunks in test files.
# (tunable) Chosen on the 40 dev tasks only (eval/repoindex_eval.py; DECISIONS change log).
W_STRUCTURAL = 0.2
W_TEXT = 0.4
W_FILE = 0.0
TEST_FACTOR = 0.7  # bug fixes rarely edit tests; they still show up when clearly relevant
W_EMBED = 0.3  # (not tuned) only when embeddings were computed for the repository
PSEUDO_MENTIONS = 3  # query names nothing: its best text matches act as the mentions
MAP_DIR_LINES = 40
MAP_METHODS_SHOWN = 12
MAP_LINES_PER_FILE = 12  # classes / functions listed per file (more files fit the budget)
MAX_SYMBOL_MATCHES = 4
MAX_RELATED = 12


@dataclass
class FileEntry:
    size: int
    mtime_ns: int
    sha256: str
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    structural: float = 0.0
    text: float = 0.0
    file: float = 0.0
    embed: float = 0.0


def breadcrumb(chunk: Chunk) -> str:
    """'django/db/models/query.py › QuerySet › only'"""
    if chunk.kind == ChunkKind.MODULE:
        return f"{chunk.file_path} › (module level)"
    return " › ".join([chunk.file_path, *chunk.qualified_name.split(".")])


def numbered_code(chunk: Chunk) -> str:
    """The chunk's source with line numbers, in the agent's Read format ('  12→code')."""
    return "\n".join(f"{n:>6}→{line}"
                     for n, line in enumerate(chunk.code.split("\n"), chunk.start_line))


def _signature_line(chunk: Chunk) -> str:
    sig = " ".join(part.strip() for part in chunk.signature.split("\n"))
    return sig.rstrip(":").strip() or chunk.qualified_name.split(".")[-1]


def _cache_file(root: Path, cache_dir: Path) -> Path:
    key = hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:16]
    return cache_dir / f"{root.resolve().name}-{key}.json"


class RepoIndex:
    def __init__(self, root: Path | None = None, cache_file: Path | None = None):
        self.root = root
        self.cache_file = cache_file
        self.files: dict[str, FileEntry] = {}
        self.text = Bm25Index()  # chunk documents: path + qualified name + code
        self.file_text = Bm25Index()  # file documents: path + all its code
        self.by_id: dict[str, Chunk] = {}
        self.embedder = None  # optional: object with similarities(query, chunks) -> {id: cos}
        self._version = 0
        self._cache_dirty = False
        self._graph: GraphIndex | None = None
        self._callers: dict[str, list[str]] = {}
        self._graph_version = -1
        self._rebuild: threading.Thread | None = None
        self._lock = threading.Lock()

    # ---------- building ----------

    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> "RepoIndex":
        """An index over already-parsed chunks (offline evaluation; no files, no refresh)."""
        index = cls()
        by_file: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_file.setdefault(c.file_path, []).append(c)
        for path in sorted(by_file):
            index._set_file(path, FileEntry(0, 0, by_file[path][0].file_hash, by_file[path]))
        return index

    @classmethod
    def open(cls, root: str | Path, cache_dir: Path = DEFAULT_CACHE_DIR) -> "RepoIndex":
        """Load the cached index of this working tree (if any) and bring it up to date."""
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"repository folder not found: {root}")
        index = cls(root, _cache_file(root, cache_dir))
        index._load()
        index.refresh()
        return index

    def _load(self) -> None:
        if self.cache_file is None or not self.cache_file.is_file():
            return
        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            if data.get("version") != INDEX_VERSION:
                return
            for path, entry in sorted(data["files"].items()):
                chunks = [Chunk.model_validate(c) for c in entry["chunks"]]
                self._set_file(path, FileEntry(entry["size"], entry["mtime_ns"],
                                               entry["sha256"], chunks))
        except (OSError, ValueError, KeyError) as exc:
            log.warning("index cache unreadable, rebuilding: %s", exc)
            self.files.clear()
            self.text, self.file_text, self.by_id = Bm25Index(), Bm25Index(), {}

    def save(self) -> None:
        if self.cache_file is None or not self._cache_dirty:
            return
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": INDEX_VERSION, "files": {
            path: {"size": e.size, "mtime_ns": e.mtime_ns, "sha256": e.sha256,
                   "chunks": [c.model_dump(mode="json") for c in e.chunks]}
            for path, e in sorted(self.files.items())}}
        tmp = self.cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.cache_file)
        self._cache_dirty = False

    def refresh(self) -> list[str]:
        """Re-parse files whose content changed in the working tree. Returns changed paths."""
        if self.root is None:
            return []
        current = list_python_files(self.root)
        changed = []
        parser = None
        for rel in current:
            try:
                st = (self.root / rel).stat()
                entry = self.files.get(rel)
                if entry and entry.size == st.st_size and entry.mtime_ns == st.st_mtime_ns:
                    continue
                source = (self.root / rel).read_bytes()
            except OSError as exc:
                log.warning("cannot read %s: %s", rel, exc)
                continue
            sha = hashlib.sha256(source).hexdigest()
            self._cache_dirty = True
            if entry and entry.sha256 == sha:  # touched, not changed
                entry.size, entry.mtime_ns = st.st_size, st.st_mtime_ns
                continue
            parser = parser or PythonParser()
            self._set_file(rel, FileEntry(st.st_size, st.st_mtime_ns, sha,
                                          parser.parse_file(rel, source)))
            changed.append(rel)
        for rel in sorted(set(self.files) - set(current)):
            self._drop_file(rel)
            changed.append(rel)
        if changed:
            self._version += 1
            self._cache_dirty = True
        return sorted(changed)

    def _drop_file(self, path: str) -> None:
        entry = self.files.pop(path, None)
        if entry is None:
            return
        for c in entry.chunks:
            self.text.remove(c.id)
            self.by_id.pop(c.id, None)
        self.file_text.remove(path)

    def _set_file(self, path: str, entry: FileEntry) -> None:
        self._drop_file(path)
        self.files[path] = entry
        words = tokenize(path)
        for c in entry.chunks:
            self.by_id[c.id] = c
            self.text.add(c.id, tokenize(f"{c.file_path} {c.qualified_name} {c.code}"))
            words.extend(tokenize(c.code))
        self.file_text.add(path, words)
        self._version += 1

    def chunks(self) -> list[Chunk]:
        return [c for path in sorted(self.files) for c in self.files[path].chunks]

    # ---------- code graph ----------

    def _build_graph(self, version: int) -> None:
        graph = build_graph(self.chunks())
        callers: dict[str, list[str]] = {}
        for edge in graph.edges:
            if edge.kind == EdgeKind.CALLS:
                callers.setdefault(edge.target, []).append(edge.source)
        with self._lock:
            self._graph, self._callers, self._graph_version = GraphIndex(graph), callers, version

    def graph(self, wait: bool = False) -> GraphIndex | None:
        """The code graph; after a change the previous graph is returned while a new one is
        built in the background (wait=True builds it now)."""
        version = self._version
        if self._graph_version == version:
            return self._graph
        if self._graph is None or wait:
            self._build_graph(version)
            return self._graph
        current = self._graph
        if self._rebuild is None or not self._rebuild.is_alive():
            self._rebuild = threading.Thread(target=self._build_graph, args=(version,),
                                             daemon=True)
            self._rebuild.start()
        return current

    # ---------- search ----------

    def search(self, query: str, k: int = 10, weights: tuple[float, ...] | None = None,
               chunks: list[Chunk] | None = None) -> list[Hit]:
        """Best k chunks for the query. `weights`: (structural, text, file, embed,
        test factor). `chunks`: precomputed self.chunks() (speed)."""
        w_struct, w_text, w_file, w_embed, test_factor = weights or (
            W_STRUCTURAL, W_TEXT, W_FILE, W_EMBED, TEST_FACTOR)
        chunks = chunks if chunks is not None else self.chunks()
        words = tokenize(query)
        text = normalised(self.text.scores(words))
        file_text = normalised(self.file_text.scores(words))
        mentions = analyse_query(query, chunks).weights
        if not mentions:
            best = sorted(text, key=lambda cid: (-text[cid], cid))[:PSEUDO_MENTIONS]
            mentions = {cid: SYMBOL_WEIGHT for cid in best}
        graph = self.graph()
        if graph is not None:
            structural = structural_scores(graph, mentions)
        else:
            structural = dict(mentions)
        top = max(structural.values(), default=0.0)
        structural = {cid: s / top for cid, s in structural.items()} if top > 0 else {}
        embed: dict[str, float] = {}
        if self.embedder is not None and w_embed > 0:
            sims = self.embedder.similarities(query, chunks)
            if sims:
                lo, hi = min(sims.values()), max(sims.values())
                embed = {cid: (s - lo) / (hi - lo) for cid, s in sims.items()} if hi > lo else {}
        candidates = set(text) | set(structural) | {c.id for c in chunks
                                                    if c.file_path in file_text}
        candidates |= set(sorted(embed, key=lambda cid: -embed[cid])[:500])
        hits = []
        for cid in candidates:
            chunk = self.by_id.get(cid)
            if chunk is None:
                continue
            s, t = structural.get(cid, 0.0), text.get(cid, 0.0)
            f, e = file_text.get(chunk.file_path, 0.0), embed.get(cid, 0.0)
            score = w_struct * s + w_text * t + w_file * f + (w_embed * e if embed else 0.0)
            if is_test_file(chunk.file_path):
                score *= test_factor
            hits.append(Hit(chunk, score, s, t, f, e))
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits[:k]

    # ---------- what the agent sees ----------

    def find_code(self, query: str, k: int = 8, max_tokens: int = 3000) -> str:
        hits = self.search(query, k)
        if not hits:
            return "No matching code found."
        out = [(f"{len(hits)} best matches (best first). Code is shown while it fits "
                f"{max_tokens} tokens; use `symbol` for the rest.")]
        used = count_tokens(out[0])
        for i, hit in enumerate(hits, 1):
            c = hit.chunk
            head = f"\n{i}. {breadcrumb(c)}  (lines {c.start_line}-{c.end_line})"
            if c.kind != ChunkKind.MODULE and c.signature:
                head += f"\n   {_signature_line(c)}"
            code = numbered_code(c)
            cost = count_tokens(head) + count_tokens(code)
            if used + cost <= max_tokens:
                out += [head, code]
                used += cost
            else:
                out.append(head)
                used += count_tokens(head)
        return "\n".join(out)

    def _file_importance(self) -> dict[str, int]:
        """Number of other non-test files whose code points at each file (calls / imports /
        bases). Tests are left out: they would make test helpers look most important."""
        graph = self.graph()
        if graph is None:
            return {}
        users: dict[str, set[str]] = {}
        for source, targets in graph.undirected.items():
            src_file = file_of(source)
            if is_test_file(src_file):
                continue
            for target in targets:
                tgt_file = file_of(target)
                if tgt_file != src_file:
                    users.setdefault(tgt_file, set()).add(src_file)
        return {path: len(files) for path, files in users.items()}

    def repo_map(self, prefix: str = "", max_tokens: int = 2000) -> str:
        """Directory layout, then class / function signatures of the most-used files."""
        prefix = prefix.strip("/")
        paths = [p for p in sorted(self.files)
                 if not prefix or p == prefix or p.startswith(prefix + "/")]
        if not paths:
            return f"No Python files under '{prefix}'."
        out = [f"# {len(paths)} Python files" + (f" under {prefix}/" if prefix else ""),
               "## Directories (Python files below each)"]
        out += _directory_lines(paths)
        out.append("## Most-used files: classes and functions")
        importance = self._file_importance()
        ranked = sorted(paths, key=lambda p: (is_test_file(p), -importance.get(p, 0), p))
        used = count_tokens("\n".join(out))
        shown = 0
        for path in ranked:
            block = self._file_outline(path, importance.get(path, 0))
            cost = count_tokens(block)
            if used + cost > max_tokens:
                continue
            out.append(block)
            used += cost
            shown += 1
        if shown < len(paths):
            out.append(f"... {len(paths) - shown} more files not shown (pass a prefix, "
                       "or use find_code)")
        return "\n".join(out)

    def _file_outline(self, path: str, users: int) -> str:
        chunks = self.files[path].chunks
        lines = [f"{path}" + (f"  (used by {users} files)" if users else "")]
        methods: dict[str, list[str]] = {}
        for c in chunks:
            if c.kind == ChunkKind.METHOD and c.parent_id:
                methods.setdefault(c.parent_id, []).append(c.qualified_name.split(".")[-1])
        for c in chunks:
            if c.kind == ChunkKind.CLASS and "." not in c.qualified_name:
                names = methods.get(c.id, [])
                listed = ", ".join(names[:MAP_METHODS_SHOWN])
                more = f", +{len(names) - MAP_METHODS_SHOWN}" if len(names) > MAP_METHODS_SHOWN else ""
                lines.append(f"  {_signature_line(c)}" + (f": {listed}{more}" if names else ""))
            elif c.kind == ChunkKind.FUNCTION:
                lines.append(f"  {_signature_line(c)}")
        if len(lines) > MAP_LINES_PER_FILE + 1:
            hidden = len(lines) - 1 - MAP_LINES_PER_FILE
            lines = lines[: MAP_LINES_PER_FILE + 1] + [f"  ... +{hidden} more (use symbol / find_code)"]
        return "\n".join(lines)

    def _match(self, name: str) -> list[Chunk]:
        name = name.strip().replace("::", ".")
        exact = [c for c in self.by_id.values()
                 if c.kind != ChunkKind.MODULE and (c.qualified_name == name or c.id == name
                                                    or c.qualified_name.endswith("." + name))]
        if not exact:
            last = name.split(".")[-1].lower()
            exact = [c for c in self.by_id.values() if c.kind != ChunkKind.MODULE
                     and c.qualified_name.split(".")[-1].lower() == last]
        return sorted(exact, key=lambda c: (c.file_path, c.start_line, c.id))

    def symbol(self, name: str) -> str:
        matches = self._match(name)
        if not matches:
            return f"No symbol named '{name}'. Try find_code."
        if len(matches) > MAX_SYMBOL_MATCHES:
            listed = "\n".join(f"- {breadcrumb(c)}  (lines {c.start_line}-{c.end_line})"
                               for c in matches[:40])
            more = f"\n... {len(matches) - 40} more" if len(matches) > 40 else ""
            return (f"{len(matches)} symbols match '{name}'; ask again with a qualified name "
                    f"(e.g. Class.method):\n{listed}{more}")
        graph = self.graph()
        blocks = []
        for c in matches:
            block = [f"{breadcrumb(c)}  (lines {c.start_line}-{c.end_line})", numbered_code(c)]
            callers = sorted(set(self._callers.get(c.id, [])))
            if callers:
                block.append(_related("called by", callers))
            if graph is not None and graph.depends_on.get(c.id):
                block.append(_related("calls / inherits", sorted(set(graph.depends_on[c.id]))))
            blocks.append("\n".join(block))
        return "\n\n".join(blocks)


def is_test_file(path: str) -> bool:
    parts = PurePosixPath(path).parts
    name = parts[-1] if parts else ""
    return (name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"
            or any(part in ("tests", "test", "testing") for part in parts[:-1]))


def _directory_lines(paths: list[str]) -> list[str]:
    """Directories with the number of Python files below them, at the deepest level that
    fits MAP_DIR_LINES lines (deep trees such as 100 locale folders collapse into one)."""
    lines: list[str] = []
    for depth in range(4, 0, -1):
        counts: dict[str, int] = {}
        for p in paths:
            parts = PurePosixPath(p).parts[:-1]
            key = "/".join(parts[:depth]) if parts else "."
            counts[key] = counts.get(key, 0) + 1
        lines = [f"{d}/  ({counts[d]})" for d in sorted(counts)]
        if len(lines) <= MAP_DIR_LINES:
            return lines
    return lines[:MAP_DIR_LINES] + [f"... {len(lines) - MAP_DIR_LINES} more (pass a prefix)"]


def _related(label: str, ids: list[str]) -> str:
    shown = ", ".join(i.replace("::", ":") for i in ids[:MAX_RELATED])
    more = f" (+{len(ids) - MAX_RELATED} more)" if len(ids) > MAX_RELATED else ""
    return f"{label}: {shown}{more}"


def timed(fn: Callable[[], str]) -> str:
    """Run a tool function; any failure becomes a short message instead of a crash."""
    try:
        return fn()
    except Exception as exc:  # the agent gets an answer, the server keeps running
        log.exception("index tool failed")
        return f"TokenCut index error: {type(exc).__name__}: {exc}"
