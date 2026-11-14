"""Attachment token estimates: images and documents (docs/ATTACHMENTS_PLAN.md, step 1).

Attachments are forwarded unchanged (`SegmentKind.MEDIA`). Their text is empty, so until now
they counted as 0 tokens. This module estimates what the provider bills for them, from the
formulas the providers publish (read Nov 2026, see the plan's sources), so stats and later
cost decisions see them. Estimates are in the provider's own tokens, not proxy tokens.

Only headers are read (no image library). Unknown sizes (an image behind a URL or a file id)
count as 0 and are reported as `unknown`.
"""

import base64
import binascii
import copy
import hashlib
import io
import math
import re
import struct
from typing import Any

from PIL import Image, ImageChops
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PyPdfError

from tokencut.contracts import Segment, SegmentKind
from tokencut.gateway.adapters.common import lookup
from tokencut.gateway.routing import DEFAULT_GPT_PREFIXES
from tokencut.tokens import count_tokens

HEAD_B64_CHARS = 1 << 18  # 256k base64 characters (~192 KB): enough to reach an image header
PDF_TEXT_TOKENS_PER_PAGE = 2_250  # Claude: "typically 1,500-3,000" text tokens per page (midpoint)
_PDF_CACHE_MAX = 64
_pdf_cache: dict[tuple[int, str, str], int | None] = {}

# ---------- reading sizes ----------

_JPEG_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def image_size(raw: bytes) -> tuple[int, int] | None:
    """(width, height) from a PNG / JPEG / GIF / WebP header, or None."""
    try:
        if raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR":
            return struct.unpack(">II", raw[16:24])
        if raw[:6] in (b"GIF87a", b"GIF89a"):
            return struct.unpack("<HH", raw[6:10])
        if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            return _webp_size(raw)
        if raw[:2] == b"\xff\xd8":
            return _jpeg_size(raw)
    except struct.error:
        return None
    return None


def _webp_size(raw: bytes) -> tuple[int, int] | None:
    chunk = raw[12:16]
    if chunk == b"VP8X":
        return 1 + int.from_bytes(raw[24:27], "little"), 1 + int.from_bytes(raw[27:30], "little")
    if chunk == b"VP8L":
        bits = int.from_bytes(raw[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8 ":
        w, h = struct.unpack("<HH", raw[26:30])
        return w & 0x3FFF, h & 0x3FFF
    return None


def _jpeg_size(raw: bytes) -> tuple[int, int] | None:
    i = 2
    while i + 9 <= len(raw):
        if raw[i] != 0xFF:
            i += 1
            continue
        marker = raw[i + 1]
        if marker in _JPEG_SOF:
            h, w = struct.unpack(">HH", raw[i + 5:i + 9])
            return w, h
        if marker == 0xFF:  # fill byte
            i += 1
        elif marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:  # markers without a length
            i += 2
        else:
            i += 2 + struct.unpack(">H", raw[i + 2:i + 4])[0]
    return None


def pdf_pages(raw: bytes) -> int | None:
    """Page count from the page objects, else from the page tree's /Count, else (page objects
    inside compressed object streams, PDF 1.5+) from the PDF parser."""
    if not raw.startswith(b"%PDF"):
        return None
    pages = len(re.findall(rb"/Type\s*/Page(?![A-Za-z])", raw))
    if pages:
        return pages
    counts = [int(c) for c in re.findall(rb"/Type\s*/Pages\b[^>]*?/Count\s+(\d+)", raw)]
    if counts:
        return max(counts)
    try:
        return len(PdfReader(io.BytesIO(raw)).pages) or None
    except _PDF_ERRORS:
        return None


def _decode(data: str, limit: int | None = None) -> bytes:
    part = data if limit is None else data[:limit - limit % 4]
    try:
        return base64.b64decode(part, validate=False)
    except (binascii.Error, ValueError):
        return b""


def _pdf_pages_b64(data: str) -> int | None:
    """Whole document decoded once per distinct PDF (the same PDF is re-sent every turn)."""
    key = (len(data), data[:2048], data[-2048:])
    if key not in _pdf_cache:
        if len(_pdf_cache) >= _PDF_CACHE_MAX:
            _pdf_cache.pop(next(iter(_pdf_cache)))
        _pdf_cache[key] = pdf_pages(_decode(data))
    return _pdf_cache[key]


# ---------- provider formulas ----------


def claude_high_res(model: str) -> bool:
    """Claude 4.7 and later use the high-resolution image tier."""
    m = re.search(r"claude-(?:[a-z]+-)?(\d+)-(\d{1,2})(?!\d)", model) or re.search(
        r"claude-(?:[a-z]+-)?(\d+)", model)
    if not m:
        return False
    major = int(m.group(1))
    minor = int(m.group(2)) if m.lastindex and m.lastindex >= 2 else 0
    return (major, minor) >= (4, 7)


def _claude_tokens(w: int, h: int) -> int:
    return math.ceil(w / 28) * math.ceil(h / 28)


def claude_image_tokens(w: int, h: int, model: str) -> int:
    """ceil(w/28) x ceil(h/28); larger images are shrunk (aspect kept) to the tier's limits."""
    edge, cap = (2576, 4784) if claude_high_res(model) else (1568, 1568)
    if max(w, h) <= edge and _claude_tokens(w, h) <= cap:
        return _claude_tokens(w, h)
    long_side, short_side = max(w, h), min(w, h)
    side = min(edge, long_side)
    while side > 1:
        other = max(1, round(side * short_side / long_side))
        if _claude_tokens(side, other) <= cap:
            return _claude_tokens(side, other)
        side -= 1
    return 1


# (base, per tile) for tile-priced OpenAI models; first matching prefix wins
_OPENAI_TILES = (("gpt-4o-mini", (2833, 5667)), ("gpt-4.1-mini", None), ("gpt-4.1-nano", None),
                 ("gpt-4o", (85, 170)), ("gpt-4.1", (85, 170)), ("gpt-5.1", (70, 140)))
_OPENAI_PATCH_MULTIPLIER = (("gpt-4.1-mini", 1.62), ("gpt-5-nano", 1.5))
_OPENAI_DEFAULT_MULTIPLIER = 1.2  # gpt-5.2 / 5.4 / 5.5 in the guide
_OPENAI_MAX_SIDE, _OPENAI_MAX_PATCHES = 2048, 2500


def _fit(w: int, h: int, box: int) -> tuple[int, int]:
    scale = min(1.0, box / max(w, h))
    return max(1, round(w * scale)), max(1, round(h * scale))


def openai_image_tokens(w: int, h: int, model: str, detail: str = "auto") -> int:
    """Tile models: base + per-tile x 512-px tiles. Patch models: 32-px patches x multiplier."""
    name = model.lower()
    tiles = next((t for prefix, t in _OPENAI_TILES if name.startswith(prefix)), None)
    if tiles is not None:
        base, per_tile = tiles
        if detail == "low":
            return base
        w, h = _fit(w, h, 2048)
        if min(w, h) > 768:
            scale = 768 / min(w, h)
            w, h = round(w * scale), round(h * scale)
        return base + per_tile * math.ceil(w / 512) * math.ceil(h / 512)
    mult = next((m for prefix, m in _OPENAI_PATCH_MULTIPLIER if name.startswith(prefix)),
                _OPENAI_DEFAULT_MULTIPLIER)
    if detail == "low":
        w, h = _fit(w, h, 512)
    elif detail != "original":
        w, h = _fit(w, h, _OPENAI_MAX_SIDE)
        while math.ceil(w / 32) * math.ceil(h / 32) > _OPENAI_MAX_PATCHES:
            w, h = _fit(w, h, max(w, h) - 1)
    return math.ceil(math.ceil(w / 32) * math.ceil(h / 32) * mult)


_GEMINI_IMAGE = {"LOW": 280, "MEDIUM": 560, "HIGH": 1120, "ULTRA_HIGH": 2240}
_GEMINI_PDF_PAGE = {"LOW": 280, "MEDIUM": 560, "HIGH": 1120}


def _gemini_new(model: str) -> bool:
    m = re.search(r"gemini-(\d+)", model)
    return m is None or int(m.group(1)) >= 3  # unknown: assume a current model


def _resolution(value: str) -> str:
    return value.upper().removeprefix("MEDIA_RESOLUTION_")


def gemini_image_tokens(w: int, h: int, model: str, resolution: str = "") -> int:
    if _gemini_new(model):
        return _GEMINI_IMAGE.get(_resolution(resolution), 1120)
    if w <= 384 and h <= 384:
        return 258
    unit = max(1, math.floor(min(w, h) / 1.5))
    return 258 * math.ceil(w / unit) * math.ceil(h / unit)


def gemini_pdf_page_tokens(model: str, resolution: str = "") -> int:
    if _gemini_new(model):
        return _GEMINI_PDF_PAGE.get(_resolution(resolution), 560)
    return 258


# ---------- attachments inside requests ----------


def _data_url(url: str) -> tuple[str, str] | None:
    m = re.match(r"data:([^;,]+)[^,]*;base64,", url)
    return (m.group(1), url[m.end():]) if m else None


def _estimate(provider: str, model: str, block: Any, body: dict) -> tuple[str, int | None] | None:
    """(kind "image"/"document", tokens or None if unknown) for an attachment block, else None."""
    if not isinstance(block, dict):
        return None
    if provider == "anthropic" and block.get("type") in ("image", "document"):
        kind = block["type"]
        source = block.get("source") or {}
        if source.get("type") == "text":
            return kind, count_tokens(str(source.get("data", "")))
        if source.get("type") != "base64":
            return kind, None
        return kind, _from_bytes(provider, model, str(source.get("media_type", "")),
                                 str(source.get("data", "")), "")
    if provider == "openai":
        if block.get("type") == "image_url":
            image = block.get("image_url") or {}
            parsed = _data_url(str(image.get("url", "")))
            if parsed is None:
                return "image", None
            return "image", _from_bytes(provider, model, parsed[0], parsed[1],
                                        str(image.get("detail", "auto")))
        if block.get("type") == "file":
            parsed = _data_url(str((block.get("file") or {}).get("file_data", "")))
            if parsed is None:
                return "document", None
            return "document", _from_bytes(provider, model, parsed[0], parsed[1], "")
        return None
    if provider == "gemini":
        inline = block.get("inlineData") or block.get("inline_data")
        file_data = block.get("fileData") or block.get("file_data")
        if isinstance(file_data, dict):
            mime = str(file_data.get("mimeType") or file_data.get("mime_type") or "")
            return ("image" if mime.startswith("image/") else "document"), None
        if not isinstance(inline, dict):
            return None
        mime = str(inline.get("mimeType") or inline.get("mime_type") or "")
        config = body.get("generationConfig") or body.get("generation_config") or {}
        resolution = str(config.get("mediaResolution") or config.get("media_resolution") or "")
        kind = "image" if mime.startswith("image/") else "document"
        return kind, _from_bytes(provider, model, mime, str(inline.get("data", "")), resolution)
    return None


def pricing_provider(provider: str, model: str) -> str:
    """Whose image / PDF prices apply. Claude Code sends GPT models in Claude's format when
    TokenCut routes them through LiteLLM to the ChatGPT plan: those are priced as OpenAI."""
    if provider == "anthropic" and model.lower().startswith(DEFAULT_GPT_PREFIXES):
        return "openai"
    return provider


def _from_bytes(provider: str, model: str, mime: str, data: str, option: str) -> int | None:
    provider = pricing_provider(provider, model)
    if mime.startswith("image/"):
        size = image_size(_decode(data, HEAD_B64_CHARS))
        if size is None:
            return None
        w, h = size
        if provider == "anthropic":
            return claude_image_tokens(w, h, model)
        if provider == "openai":
            return openai_image_tokens(w, h, model, option or "auto")
        return gemini_image_tokens(w, h, model, option)
    if mime == "application/pdf":
        pages = _pdf_pages_b64(data)
        if pages is None:
            return None
        if provider == "gemini":
            return pages * gemini_pdf_page_tokens(model, option)
        return pages * PDF_TEXT_TOKENS_PER_PAGE  # text side only: page images come on top
    if mime.startswith("text/"):
        return count_tokens(_decode(data).decode("utf-8", errors="replace"))
    return None


def attachment_key(block: Any) -> str | None:
    """Content hash of an inline attachment (Anthropic base64 block, Gemini inlineData)."""
    if not isinstance(block, dict):
        return None
    data = None
    if block.get("type") in ("image", "document"):
        source = block.get("source") or {}
        if source.get("type") == "base64":
            data = source.get("data")
    else:
        inline = block.get("inlineData") or block.get("inline_data")
        if isinstance(inline, dict):
            data = inline.get("data")
    if not data:
        return None
    return hashlib.sha1(str(data).encode("ascii", errors="replace")).hexdigest()


def _has_function_response(content: Any) -> bool:
    return isinstance(content, dict) and any(
        isinstance(part, dict) and ("functionResponse" in part or "function_response" in part)
        for part in content.get("parts") or [])


def replaceable_tool_attachment(body: dict, path: list[str | int]) -> bool:
    """An attachment a tool returned, at a place where a text note is valid:
    Anthropic `messages[i].content[j].content[k]` (inside a tool_result); Gemini
    `contents[i].parts[j]` next to a functionResponse (Gemini CLI's sibling parts). OpenAI chat
    tool messages carry text only, so tool images arrive as user messages: never changed."""
    if len(path) == 6 and path[0] == "messages" and path[2] == "content" and path[4] == "content":
        return True
    if len(path) == 4 and path[0] == "contents" and path[2] == "parts":
        return _has_function_response(body["contents"][path[1]])
    return False


def note_block(path: list[str | int], note: str) -> dict:
    """A text block in the provider's own format."""
    return {"text": note} if path[0] == "contents" else {"type": "text", "text": note}


def duplicate_note(number: int) -> str:
    return (f"[TokenCut: identical to attachment #{number} earlier in this conversation "
            "(same bytes), not sent again]")


# ---------- step 5: shrink large images where the provider bills by size ----------

TARGET_LONG_EDGE = 1568  # what standard-tier Claude models always receive
MIN_SAVING = 0.10  # resize only if the provider's own formula saves at least this share
_RESIZE_FORMATS = {"PNG", "JPEG", "WEBP"}
_KEEP_DETAIL = re.compile(
    r"\bzoom|\bpixel|\btiny\b|\bsmall (?:text|font|print)|\bfine print|\bread (?:the )?exact",
    re.IGNORECASE)
_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_RESIZE_CACHE_MAX = 32
_resize_cache: dict[tuple, str | None] = {}  # base64 of the cropped / resized image
_box_cache: dict[tuple[int, str, str], tuple[int, int, int, int] | None] = {}


def tool_attachment(body: dict, path: list[str | int]) -> bool:
    """An attachment a tool returned (any provider format TokenCut can recognise)."""
    if replaceable_tool_attachment(body, path):
        return True
    # Gemini CLI on multimodal models: contents[i].parts[j].functionResponse.parts[m]
    return len(path) == 7 and path[0] == "contents" and path[5] == "parts"


def _tool_name(body: dict, path: list[str | int]) -> str:
    if path[0] == "messages":
        call_id = body["messages"][path[1]]["content"][path[3]].get("tool_use_id")
        for message in body["messages"][:path[1]]:
            for block in message.get("content") or [] if isinstance(message, dict) else []:
                if isinstance(block, dict) and block.get("id") == call_id:
                    return str(block.get("name", ""))
        return ""
    for part in body["contents"][path[1]].get("parts") or []:
        response = part.get("functionResponse") or part.get("function_response") if isinstance(
            part, dict) else None
        if isinstance(response, dict):
            return str(response.get("name", ""))
    return ""


def _computer_use(body: dict) -> bool:
    return any(isinstance(t, dict) and (str(t.get("type", "")).startswith("computer")
                                        or "computer" in str(t.get("name", "")).lower())
               for t in body.get("tools") or [])


def attachment_questions(segments: list[Segment]) -> dict[str, str]:
    """Per attachment segment: the user's last message before it plus the user text sent
    with it (same message). Never a later message, so every decision based on it is the same
    on every turn (prompt cache). Only role "user": agents such as Claude Code send their own
    context (environment, skills list) as role "system" messages inside the conversation."""
    with_message: dict[int, list[str]] = {}
    for seg in segments:
        if seg.kind == SegmentKind.USER_TEXT and seg.role == "user":
            text = _REMINDER.sub("", seg.text).strip()
            if text:
                with_message.setdefault(seg.message_index, []).append(text)
    questions, last = {}, ""
    for seg in segments:
        if seg.kind == SegmentKind.USER_TEXT and seg.role == "user":
            text = _REMINDER.sub("", seg.text).strip()
            if text:
                last = text
        elif seg.kind == SegmentKind.MEDIA:
            questions[seg.id] = "\n".join([last, *with_message.get(seg.message_index, [])])
    return questions


def _image_data(block: dict) -> tuple[str, str] | None:
    """(mime, base64 data) of an inline image block in any provider format."""
    if block.get("type") == "image":
        source = block.get("source") or {}
        if source.get("type") == "base64":
            return str(source.get("media_type", "")), str(source.get("data", ""))
        return None
    if block.get("type") == "image_url":
        return _data_url(str((block.get("image_url") or {}).get("url", "")))
    inline = block.get("inlineData") or block.get("inline_data")
    if isinstance(inline, dict):
        return str(inline.get("mimeType") or inline.get("mime_type") or ""), str(inline.get("data", ""))
    return None


def _with_image_data(block: dict, data: str) -> dict:
    new = copy.deepcopy(block)
    if new.get("type") == "image":
        new["source"]["data"] = data
    elif new.get("type") == "image_url":
        url = new["image_url"]["url"]
        new["image_url"]["url"] = url[:url.index(",") + 1] + data
    else:
        key = "inlineData" if "inlineData" in new else "inline_data"
        new[key]["data"] = data
    return new


def _image_tokens(provider: str, model: str, w: int, h: int, block: dict, body: dict) -> int:
    provider = pricing_provider(provider, model)
    if provider == "anthropic":
        return claude_image_tokens(w, h, model)
    if provider == "openai":
        return openai_image_tokens(w, h, model, str((block.get("image_url") or {}).get("detail", "auto")))
    config = body.get("generationConfig") or body.get("generation_config") or {}
    return gemini_image_tokens(w, h, model, str(config.get("mediaResolution")
                                                or config.get("media_resolution") or ""))


def _adjusted_b64(data: str, box: tuple[int, int, int, int] | None,
                  size: tuple[int, int] | None) -> str | None:
    """Crop to `box`, then resize to `size`. Same input -> same output bytes (fixed filter and
    settings), cached per image and operation."""
    key = (len(data), data[:2048], data[-2048:], box, size)
    if key not in _resize_cache:
        if len(_resize_cache) >= _RESIZE_CACHE_MAX:
            _resize_cache.pop(next(iter(_resize_cache)))
        raw = _transform(_decode(data), box, size)
        _resize_cache[key] = base64.b64encode(raw).decode() if raw is not None else None
    return _resize_cache[key]


def _open_image(raw: bytes) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(raw))
        if img.format not in _RESIZE_FORMATS or getattr(img, "is_animated", False):
            return None
        img.load()
        return img
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def _transform(raw: bytes, box: tuple[int, int, int, int] | None,
               size: tuple[int, int] | None) -> bytes | None:
    img = _open_image(raw)
    if img is None:
        return None
    fmt = img.format
    out = img.crop(box) if box is not None else img
    if size is not None and size != out.size:
        out = out.resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    if fmt == "JPEG":
        out.convert("RGB").save(buf, "JPEG", quality=90)
    elif fmt == "WEBP":
        out.save(buf, "WEBP", quality=90)
    else:
        out.save(buf, "PNG")
    return buf.getvalue()


# ---------- step 6: plain outer borders are cut off ----------

BORDER_TOLERANCE = 16  # max per-channel difference still counted as background
BORDER_PAD = 8  # px of the border kept around the content
BORDER_MAX_REMOVED = 0.75  # cutting more than this share: a near-empty image, left alone
BORDER_MIN_CONTENT = 32  # px; smaller content boxes are left alone
_LAYOUT_QUESTION = re.compile(
    r"\b(?:layout|align\w*|spacing|spaced|margins?|padding|cent(?:er|re)d?|position\w*|"
    r"whitespace|white space|gaps?|offset|overflow\w*|cut off|borders?)\b", re.IGNORECASE)


def content_box(img: Image.Image) -> tuple[int, int, int, int] | None:
    """Box (left, top, right, bottom) around everything that differs from a plain background,
    plus BORDER_PAD px; None if the corners do not share one colour or nothing differs."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    corners = [rgb.getpixel(p) for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
    background = corners[0]
    if any(max(abs(a - b) for a, b in zip(c, background, strict=True)) > BORDER_TOLERANCE
           for c in corners):
        return None
    red, green, blue = ImageChops.difference(rgb, Image.new("RGB", rgb.size, background)).split()
    strongest = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    box = strongest.point(lambda v: 255 if v > BORDER_TOLERANCE else 0).getbbox()
    if box is None:
        return None
    left, top, right, bottom = box
    return (max(0, left - BORDER_PAD), max(0, top - BORDER_PAD),
            min(w, right + BORDER_PAD), min(h, bottom + BORDER_PAD))


def _border_box(data: str, w: int, h: int) -> tuple[int, int, int, int] | None:
    """The content box worth cropping to, cached per image."""
    key = (len(data), data[:2048], data[-2048:])
    if key not in _box_cache:
        if len(_box_cache) >= _RESIZE_CACHE_MAX:
            _box_cache.pop(next(iter(_box_cache)))
        img = _open_image(_decode(data))
        box = content_box(img) if img is not None else None
        if box is not None:
            bw, bh = box[2] - box[0], box[3] - box[1]
            if (bw, bh) == (w, h) or min(bw, bh) < BORDER_MIN_CONTENT \
                    or 1 - (bw * bh) / (w * h) > BORDER_MAX_REMOVED:
                box = None
        _box_cache[key] = box
    return _box_cache[key]


def _image_note(w: int, h: int, box: tuple[int, int, int, int], size: tuple[int, int]) -> str:
    bw, bh = box[2] - box[0], box[3] - box[1]
    scaled = f", then scaled to {size[0]}x{size[1]}" if size != (bw, bh) else ""
    return (f"[TokenCut: plain border cut from this image: original {w}x{h}, kept the "
            f"{bw}x{bh} area at offset ({box[0]}, {box[1]}){scaled}; positions refer to the "
            "image as sent]")


def adjust_images(provider: str, model: str, body: dict, segments: list[Segment],
                  include_user: bool = False, resize: bool = True, trim: bool = True
                  ) -> tuple[list[tuple[list[str | int], list[dict]]], list[dict]]:
    """(path -> replacement blocks, report entries): plain borders cut (step 6) and large images
    shrunk (step 5), only where the provider's own formula saves >= MIN_SAVING. The body is not
    changed; a cropped image is followed by a note, so insert in descending path order."""
    if _computer_use(body) or not (resize or trim):
        return [], []
    questions = attachment_questions(segments)
    changes, entries = [], []
    for seg in segments:
        if seg.kind != SegmentKind.MEDIA:
            continue
        block = lookup(body, seg.path)
        image = _image_data(block) if isinstance(block, dict) else None
        if image is None or not image[0].startswith("image/"):
            continue
        is_tool = tool_attachment(body, seg.path)
        if not (is_tool or include_user):
            continue
        if is_tool and "computer" in _tool_name(body, seg.path).lower():
            continue
        size = image_size(_decode(image[1], HEAD_B64_CHARS))
        if size is None:
            continue
        w, h = size
        # a note can follow the image except inside Gemini's functionResponse.parts
        can_note = not (len(seg.path) == 7 and seg.path[0] == "contents")
        question = questions.get(seg.id, "")
        trim_here = trim and can_note and not _LAYOUT_QUESTION.search(question)
        resize_here = resize and not _KEEP_DETAIL.search(question)
        box = _border_box(image[1], w, h) if trim_here else None
        cw, ch = (box[2] - box[0], box[3] - box[1]) if box is not None else (w, h)
        target = (cw, ch)
        if resize_here and max(cw, ch) > TARGET_LONG_EDGE:
            scale = TARGET_LONG_EDGE / max(cw, ch)
            target = (max(1, round(cw * scale)), max(1, round(ch * scale)))
        if box is None and target == (w, h):
            continue
        before = _image_tokens(provider, model, w, h, block, body)
        after = _image_tokens(provider, model, *target, block, body)
        if before - after < MIN_SAVING * before:
            continue
        encoded = _adjusted_b64(image[1], box, target)
        if encoded is None:
            continue
        blocks = [_with_image_data(block, encoded)]
        if box is not None:
            blocks.append(note_block(seg.path, _image_note(w, h, box, target)))
        how = ("border_trimmed_resized" if target != (cw, ch) else "border_trimmed") \
            if box is not None else "resized"
        changes.append((seg.path, blocks))
        entries.append({"segment": seg.id, "how": how, "tool": "", "kind": "image",
                        "from_size": [w, h], "to_size": list(target),
                        "crop_box": list(box) if box is not None else None,
                        "before": before,
                        "after": after})
    return changes, entries


# ---------- step 7: PDF pages without images or drawings are sent as text ----------

PDF_MIN_TEXT_CHARS = 200  # less text than this: probably a scan or a figure page
PDF_MAX_DRAW_OPS = 20  # more line / box / curve operators: probably a table or a chart
PDF_TABLE_ROWS = 3  # this many number-table rows: a table (columns are lost as plain text)
PDF_MAX_PAGES = 50  # longer PDFs are left as they are (~115 ms per page the first time)
_DRAW_OP = re.compile(rb"(?<![A-Za-z])(?:re|l|c|v|y)(?=\s)")
_NUMBER_CELL = re.compile(r"[-+]?\d+(?:[.,]\d+)*%?")
# Intent safety: a question about how pages look keeps the whole PDF (EcoDoc, ACL 2025, also
# chooses text or image per page from page characteristics and query intent).
_VISUAL_QUESTION = re.compile(
    r"\b(?:charts?|graphs?|plots?|figures?|fig\.|diagrams?|images?|pictures?|photos?|"
    r"logos?|signatures?|signed|stamps?|seals?|handwrit\w*|colou?rs?|layout|formatting|"
    r"(?<!answer )formats?(?!:|\s+(?:your|the|this)\s+(?:final\s+)?(?:answer|reply|response))|"
    r"fonts?|scan(?:ned|s)?|tables?|maps?|drawings?|screenshots?|visual\w*|"
    r"looks? like|appearance|highlight\w*)", re.IGNORECASE)
_PDF_ERRORS = (PyPdfError, ValueError, KeyError, TypeError, AttributeError, OSError)
_SPLIT_CACHE_MAX = 32
_split_cache: dict[tuple[int, str, str], tuple[str, str | None, list[int], int] | None] = {}


def _draw_ops(page: Any) -> int:
    """Line / box / curve operators on the page and in its form XObjects (vector charts)."""
    contents = page.get_contents()
    total = len(_DRAW_OP.findall(contents.get_data())) if contents is not None else 0
    resources = page.get("/Resources")
    xobjects = resources.get_object().get("/XObject") if resources is not None else None
    for ref in (xobjects.get_object().values() if xobjects is not None else []):
        obj = ref.get_object()
        if obj.get("/Subtype") == "/Form":
            total += len(_DRAW_OP.findall(obj.get_data()))
    return total


def _number_table_rows(page: Any) -> int:
    """Rows with >= 4 columns of which >= 3 are numbers, in layout-preserving text."""
    rows = 0
    for line in (page.extract_text(extraction_mode="layout") or "").splitlines():
        cells = [c for c in re.split(r" {2,}", line.strip()) if c]
        if len(cells) >= 4 and sum(1 for c in cells if _NUMBER_CELL.fullmatch(c)) >= 3:
            rows += 1
    return rows


def pdf_page_kind(page: Any) -> tuple[str, str]:
    """("text" / "visual", page text). A page is "text" only with enough text, no raster
    images, almost no drawings and no number-table rows; scans, figures, charts and tables
    stay "visual". Raises one of _PDF_ERRORS on an unreadable page."""
    text = (page.extract_text() or "").strip()
    visual = (len(page.images) > 0 or len(text) < PDF_MIN_TEXT_CHARS
              or _draw_ops(page) > PDF_MAX_DRAW_OPS
              or _number_table_rows(page) >= PDF_TABLE_ROWS)
    return ("visual" if visual else "text"), text


def _analyse_pdf(raw: bytes) -> tuple[Any, list[str], list[str]] | None:
    """(reader, per-page "text" / "visual", per-page text), or None if unreadable / too long."""
    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted or len(reader.pages) > PDF_MAX_PAGES:
            return None
        kinds, texts = [], []
        for page in reader.pages:
            kind, text = pdf_page_kind(page)
            kinds.append(kind)
            texts.append(text)
        return reader, kinds, texts
    except _PDF_ERRORS:
        return None


def pdf_page_kinds(raw: bytes) -> list[str] | None:
    analysed = _analyse_pdf(raw)
    return analysed[1] if analysed else None


def _split_pdf(data: str) -> tuple[str, str | None, list[int], int] | None:
    """(text of the text pages, base64 PDF of the other pages or None, text page numbers,
    page count), or None if there is nothing to send as text. Same bytes every time."""
    key = (len(data), data[:2048], data[-2048:])
    if key in _split_cache:
        return _split_cache[key]
    result = None
    analysed = _analyse_pdf(_decode(data))
    if analysed and "text" in analysed[1]:
        reader, kinds, texts = analysed
        try:
            text_pages = [n for n, kind in enumerate(kinds) if kind == "text"]
            text = "\n\n".join(f"--- page {n + 1} ---\n{texts[n]}" for n in text_pages)
            visual = [n for n, kind in enumerate(kinds) if kind == "visual"]
            subset = None
            if visual:
                writer = PdfWriter()
                for n in visual:
                    writer.add_page(reader.pages[n])
                buf = io.BytesIO()
                writer.write(buf)
                subset = base64.b64encode(buf.getvalue()).decode()
            result = (text, subset, [n + 1 for n in text_pages], len(kinds))
        except _PDF_ERRORS:
            result = None
    if len(_split_cache) >= _SPLIT_CACHE_MAX:
        _split_cache.pop(next(iter(_split_cache)))
    _split_cache[key] = result
    return result


def _pdf_note(name: str, text_pages: list[int], total: int, visual_left: bool) -> str:
    pages = ", ".join(str(n) for n in text_pages)
    rest = "the other pages follow as PDF" if visual_left else "no page images sent"
    label = f" '{name}'" if name else ""
    return (f"[TokenCut: PDF{label}, pages {pages} of {total} sent as text (no images or "
            f"drawings found on them); {rest}]")


def split_pdfs(provider: str, body: dict, segments: list[Segment], include_user: bool = False
               ) -> tuple[list[tuple[list[str | int], list[dict]]], list[dict]]:
    """(path -> replacement blocks, report entries) for PDFs with text pages (Claude, OpenAI).
    Insert the replacements in descending path order: one block may become two."""
    if provider not in ("anthropic", "openai"):
        return [], []  # Gemini: its docs conflict on whether PDF text is billed
    changes, entries = [], []
    questions = attachment_questions(segments)
    for seg in segments:
        if seg.kind != SegmentKind.MEDIA:
            continue
        block = lookup(body, seg.path)
        if not isinstance(block, dict):
            continue
        if not (tool_attachment(body, seg.path) or include_user):
            continue
        if _VISUAL_QUESTION.search(questions.get(seg.id, "")):
            continue  # the question is about how pages look: keep every page picture
        if provider == "anthropic" and block.get("type") == "document":
            source = block.get("source") or {}
            if source.get("type") != "base64" or source.get("media_type") != "application/pdf":
                continue
            data, name = str(source.get("data", "")), str(block.get("title", ""))
        elif provider == "openai" and block.get("type") == "file":
            parsed = _data_url(str((block.get("file") or {}).get("file_data", "")))
            if parsed is None or parsed[0] != "application/pdf":
                continue
            data, name = parsed[1], str((block.get("file") or {}).get("filename", ""))
        else:
            continue
        split = _split_pdf(data)
        if split is None:
            continue
        text, subset, text_pages, total = split
        note = _pdf_note(name, text_pages, total, subset is not None)
        blocks = [{"type": "text", "text": f"{note}\n\n{text}"}]
        if subset is not None:
            rest = copy.deepcopy(block)
            if provider == "anthropic":
                rest["source"]["data"] = subset
            else:
                rest["file"]["file_data"] = "data:application/pdf;base64," + subset
            blocks.append(rest)
        elif "cache_control" in block:
            blocks[0]["cache_control"] = block["cache_control"]
        changes.append((seg.path, blocks))
        entries.append({"segment": seg.id, "how": "pdf_text_pages", "tool": "",
                        "before": seg.token_count, "after": seg.token_count, "pages": total,
                        "page_images_removed": len(text_pages), "kind": "pdf",
                        "text_pages": text_pages,
                        "visual_pages": [n for n in range(1, total + 1) if n not in text_pages]})
    return changes, entries


def with_media_tokens(provider: str, model: str, body: dict,
                      segments: list[Segment]) -> tuple[list[Segment], dict]:
    """Segments with estimated attachment tokens, and counts for the analysis."""
    stats = {"images": 0, "documents": 0, "unknown": 0, "tokens": 0}
    out = []
    for seg in segments:
        found = None
        if seg.kind == SegmentKind.MEDIA:
            found = _estimate(provider, model, lookup(body, seg.path), body)
        if found is None:
            out.append(seg)
            continue
        kind, tokens = found
        stats["images" if kind == "image" else "documents"] += 1
        if tokens is None:
            stats["unknown"] += 1
            tokens = 0
        stats["tokens"] += tokens
        out.append(seg.model_copy(update={"token_count": tokens}))
    return out, stats
