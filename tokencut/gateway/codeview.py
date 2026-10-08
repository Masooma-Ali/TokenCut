"""Trim a Python file that arrived inside a tool result (file read, `cat` output).

Uses the Iteration 1 engine (chunker + graph + scorer + tier packer) to decide which
functions / classes matter for the query, then keeps those **original lines exactly**
(including the tool's line-number prefixes, e.g. Claude Code's "    12→"). Every gap
becomes one marker line naming the omitted line range, so the model can read it again.
Kept lines are never rewritten, so an edit tool's exact-match `old_string` still works.
"""

import re
from dataclasses import dataclass

from tokencut.contracts import Chunk, ChunkKind, Tier
from tokencut.graph.builder import build_graph
from tokencut.ingestion.python_parser import PythonParser
from tokencut.tokens import count_tokens
from tokencut.trimming.variants import generate_variants

# "     12→code", "12\tcode", "  12 | code" (prefix kept verbatim, code is what follows)
_NUMBERED = re.compile(r"^(\s*\d+(?:→|\t| \| |│))(.*)$")
_NUMBERED_SHARE = 0.8
MIN_GAP_TOKENS = 60  # a smaller gap is kept as-is: a marker (~25 tokens) would save too little


@dataclass
class FileView:
    prefixes: list[str]  # per line: the tool's line-number prefix ("" if none)
    code: list[str]  # per line: the source text
    first_line: int  # source line number of code[0] (a read with an offset starts later)


def parse_listing(text: str) -> FileView:
    """Split tool output into line-number prefixes and source lines."""
    lines = text.split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    matches = [_NUMBERED.match(line) for line in lines]
    numbered = sum(1 for m in matches if m)
    if lines and numbered / len(lines) >= _NUMBERED_SHARE:
        prefixes = [m.group(1) if m else "" for m in matches]
        code = [m.group(2) if m else line for m, line in zip(matches, lines)]
        first = next((m for m in matches if m), None)
        first_line = int(re.sub(r"\D", "", first.group(1))) if first else 1
        return FileView(prefixes, code, first_line)
    return FileView([""] * len(lines), lines, 1)


def _own_lines(chunk: Chunk, chunks: list[Chunk], n_lines: int) -> set[int]:
    """1-based line numbers (within the listing) that belong to this chunk alone."""
    if chunk.kind == ChunkKind.MODULE:
        covered = set()
        for c in chunks:
            if c.kind in (ChunkKind.FUNCTION, ChunkKind.CLASS):
                covered.update(range(c.start_line, c.end_line + 1))
        return set(range(1, n_lines + 1)) - covered
    lines = set(range(chunk.start_line, chunk.end_line + 1))
    if chunk.kind == ChunkKind.CLASS:
        for c in chunks:
            if c.parent_id == chunk.id:
                lines -= set(range(c.start_line, c.end_line + 1))
    return lines


def select_lines(code: str, path: str, query: str, tier: Tier) -> set[int] | None:
    """Line numbers (1-based within `code`) to keep for this tier; None = keep everything."""
    chunks = PythonParser().parse_file(path or "file.py", code.encode("utf-8"))
    if len([c for c in chunks if c.kind != ChunkKind.MODULE]) < 2:
        return None  # nothing to choose between
    variants = generate_variants(chunks, build_graph(chunks), query)
    variant = next(v for v in variants if v.tier == tier)
    by_id = {c.id: c for c in chunks}
    n_lines = len(code.split("\n"))
    keep: set[int] = set()
    for cid in variant.kept_chunk_ids:
        keep |= _own_lines(by_id[cid], chunks, n_lines)
    for cid in variant.skeleton_chunk_ids:  # signature lines only; the body becomes a marker
        chunk = by_id[cid]
        keep |= set(range(chunk.start_line, chunk.start_line + chunk.signature.count("\n") + 1))
    return keep


def render(view: FileView, keep: set[int], path: str) -> str:
    """Kept lines verbatim; each run of dropped lines -> one marker (blank-only runs kept)."""
    out: list[str] = []
    name = path or "this file"
    i, n = 0, len(view.code)
    while i < n:
        if (i + 1) in keep:
            out.append(view.prefixes[i] + view.code[i])
            i += 1
            continue
        j = i
        while j < n and (j + 1) not in keep:
            j += 1
        gap = view.code[i:j]
        if all(not line.strip() for line in gap) or count_tokens("\n".join(gap)) < MIN_GAP_TOKENS:
            out.extend(view.prefixes[k] + view.code[k] for k in range(i, j))
        else:
            first, last = view.first_line + i, view.first_line + j - 1
            indent = re.match(r"\s*", next(line for line in gap if line.strip())).group(0)
            out.append(
                f"{indent}[TokenCut: lines {first}-{last} of {name} omitted "
                f"({count_tokens(chr(10).join(gap))} tokens); read them again if needed]"
            )
        i = j
    return "\n".join(out)


def trim_file_listing(text: str, path: str, query: str, tier: Tier) -> str | None:
    """Trimmed listing, or None if trimming would not help (not Python, too small, no gain)."""
    view = parse_listing(text)
    keep = select_lines("\n".join(view.code), path, query, tier)
    if keep is None:
        return None
    result = render(view, keep, path)
    return result if count_tokens(result) < count_tokens(text) else None
