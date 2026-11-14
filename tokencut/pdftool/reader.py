"""Read selected PDF pages: text pages as text, the other pages as PNG pictures."""

import base64
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from tokencut.gateway.media import pdf_page_kind

MAX_PAGES = 20  # pages per call, as Claude Code's Read
PICTURE_PX = 1568  # long edge: what standard-tier Claude models receive anyway
_PDF_ERRORS = (PyPdfError, ValueError, KeyError, TypeError, AttributeError, OSError)
_RANGE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")


def parse_pages(spec: str, total: int) -> list[int]:
    """1-based page numbers from "3", "1-5", "1-3,8"; empty = from page 1. At most MAX_PAGES."""
    if not spec.strip():
        return list(range(1, min(total, MAX_PAGES) + 1))
    pages: list[int] = []
    for part in spec.split(","):
        m = _RANGE.match(part)
        if not m:
            raise ValueError(f"bad page range {part.strip()!r} (use e.g. '3', '1-5', '1-3,8')")
        first, last = int(m.group(1)), int(m.group(2) or m.group(1))
        pages += [n for n in range(first, last + 1) if 1 <= n <= total and n not in pages]
    if not pages:
        raise ValueError(f"no pages in {spec!r}; the PDF has {total} pages")
    return pages[:MAX_PAGES]


def render_page(path: Path, number: int) -> bytes:
    """One page as PNG, long edge PICTURE_PX (Poppler's pdftoppm)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "page"
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-f", str(number), "-l", str(number),
                        "-scale-to", str(PICTURE_PX), str(path), str(out)],
                       check=True, capture_output=True, timeout=60)
        return out.with_suffix(".png").read_bytes()


def read_pdf(path: str, pages: str = "", as_images: bool = False) -> tuple[list[dict], bool]:
    """(MCP content blocks, is_error). Pages in order; consecutive text pages share one block."""
    file = Path(path).expanduser()
    try:
        reader = PdfReader(file)
        if reader.is_encrypted:
            return [_text(f"{file.name} is encrypted: use the Read tool.")], True
        numbers = parse_pages(pages, len(reader.pages))
        kinds = [("visual", "") if as_images else pdf_page_kind(reader.pages[n - 1])
                 for n in numbers]
    except (*_PDF_ERRORS, FileNotFoundError) as e:
        return [_text(f"Cannot read {path}: {e}")], True
    if any(kind == "visual" for kind, _ in kinds) and shutil.which("pdftoppm") is None:
        return [_text("Poppler's pdftoppm is not installed, so pages with charts or tables "
                      "cannot be shown. Install poppler-utils.")], True

    text_pages = [n for n, (kind, _) in zip(numbers, kinds, strict=True) if kind == "text"]
    picture_pages = [n for n in numbers if n not in text_pages]
    content = [_text(_header(file.name, len(reader.pages), numbers, text_pages, picture_pages))]
    for n, (kind, text) in zip(numbers, kinds, strict=True):
        if kind == "text":
            section = f"--- page {n} ---\n{text}"
            if content[-1]["type"] == "text" and content[-1]["text"].startswith("--- page"):
                content[-1]["text"] += "\n\n" + section
            else:
                content.append(_text(section))
        else:
            try:
                png = render_page(file, n)
            except (subprocess.SubprocessError, OSError) as e:
                return [_text(f"Cannot render page {n} of {path}: {e}")], True
            content.append(_text(f"--- page {n} (picture) ---"))
            content.append({"type": "image", "data": base64.b64encode(png).decode(),
                            "mimeType": "image/png"})
    return content, False


def _header(name: str, total: int, numbers: list[int], text_pages: list[int],
            picture_pages: list[int]) -> str:
    def join(ns: list[int]) -> str:
        return ", ".join(map(str, ns)) or "none"
    more = (f" Pages after {numbers[-1]}: call again with pages='{numbers[-1] + 1}-...'."
            if numbers[-1] < total else "")
    return (f"[read_pdf: {name}, {total} pages. Shown: {join(numbers)}. As text: "
            f"{join(text_pages)}. As pictures (charts, tables, images or scans): "
            f"{join(picture_pages)}. If how a text page looks matters, call again with "
            f"as_images=true for that page.{more}]")


def _text(text: str) -> dict:
    return {"type": "text", "text": text}
