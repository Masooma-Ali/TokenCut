"""Turn a set of kept / skeleton chunks into the text sent to the model (IMPLEMENTATION_PLAN C5).

# File: shop/cart.py
<chunks of this file in start_line order: full code, or skeleton>

# File: shop/utils/pricing.py
...
"""

from tokencut.contracts import Chunk

INDENT = "    "


def file_header(file_path: str) -> str:
    return f"# File: {file_path}"


def _leading_whitespace(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def skeleton_text(chunk: Chunk) -> str:
    """Signature line(s) + docstring + '...', indented as in the source."""
    code_lines = chunk.code.split("\n")
    indent = _leading_whitespace(code_lines[0]) if code_lines else ""

    # Use the real body indentation when we can see it (first code line after the header).
    signature_lines = chunk.signature.split("\n") if chunk.signature else []
    body_indent = indent + INDENT
    for line in code_lines[len(signature_lines) :]:
        if line.strip():
            candidate = _leading_whitespace(line)
            if len(candidate) > len(indent):
                body_indent = candidate
            break

    lines = [indent + chunk.signature] if chunk.signature else []
    if chunk.docstring:
        doc = chunk.docstring.replace('"""', '\\"\\"\\"')
        lines.append(f'{body_indent}"""{doc}"""')
    lines.append(f"{body_indent}...")
    return "\n".join(lines)


def render_variant(chunks: list[Chunk], kept_ids: set[str], skeleton_ids: set[str]) -> str:
    """Render the chosen chunks. `chunks` must contain every id in kept_ids and skeleton_ids."""
    by_file: dict[str, list[Chunk]] = {}
    for c in chunks:
        if c.id in kept_ids or c.id in skeleton_ids:
            by_file.setdefault(c.file_path, []).append(c)

    blocks = []
    for file_path in sorted(by_file):
        parts = [file_header(file_path)]
        file_chunks = sorted(by_file[file_path], key=lambda c: (c.start_line, c.id))
        texts = []
        for c in file_chunks:
            text = c.code if c.id in kept_ids else skeleton_text(c)
            if text.strip():
                texts.append(text)
        if texts:
            parts.append("\n\n".join(texts))
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)
