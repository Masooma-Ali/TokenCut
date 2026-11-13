"""Attachment token estimates (images, documents) — docs/ATTACHMENTS_PLAN.md step 1.

Expected values are the providers' own published examples where they exist.
"""

import base64
import json
import struct

from tokencut.contracts import SegmentKind
from tokencut.gateway import media
from tokencut.gateway.adapters import anthropic, gemini, openai
from tokencut.gateway.records import RequestLog, analyse
from tokencut.gateway.trimmer import TrimConfig, trim_request

# ---------- image headers (no image library needed) ----------


def png(w: int, h: int) -> bytes:
    ihdr = struct.pack(">II", w, h) + b"\x08\x02\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00" * 4 + b"IEND"


def jpeg(w: int, h: int) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, h, w, 3) + b"\x01\x22\x00" * 3
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"


def gif(w: int, h: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", w, h) + b"\x00" * 8


def webp_vp8x(w: int, h: int) -> bytes:
    chunk = b"VP8X" + struct.pack("<I", 10) + b"\x00" * 4 + (w - 1).to_bytes(3, "little") + (
        h - 1).to_bytes(3, "little")
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def webp_vp8l(w: int, h: int) -> bytes:
    bits = (w - 1) | ((h - 1) << 14)
    chunk = b"VP8L" + struct.pack("<I", 5) + b"\x2f" + bits.to_bytes(4, "little")
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def test_image_size_from_headers():
    assert media.image_size(png(1920, 1080)) == (1920, 1080)
    assert media.image_size(jpeg(1024, 768)) == (1024, 768)
    assert media.image_size(gif(300, 200)) == (300, 200)
    assert media.image_size(webp_vp8x(2000, 1000)) == (2000, 1000)
    assert media.image_size(webp_vp8l(640, 480)) == (640, 480)
    assert media.image_size(b"not an image") is None
    assert media.image_size(b"") is None


# ---------- Claude: ceil(w/28) x ceil(h/28), tiers (Anthropic vision docs) ----------


def test_claude_examples_from_the_docs():
    assert media.claude_image_tokens(1000, 1000, "claude-sonnet-4-5") == 1296
    assert media.claude_image_tokens(1000, 1000, "claude-opus-4-7") == 1296
    # standard tier: 1920x1080 is downsized to 1456x819 = 1,560 tokens
    assert media.claude_image_tokens(1920, 1080, "claude-sonnet-4-5") == 1560
    # high-resolution tier (Claude 4.7+): 1920x1080 is sent as is
    assert media.claude_image_tokens(1920, 1080, "claude-haiku-5-5") == 69 * 39
    # 3840x2160 on the high-resolution tier is downsized to 2576x1449 = 4,784 tokens
    assert media.claude_image_tokens(3840, 2160, "claude-opus-4-7") == 4784


def test_claude_tier_from_model_name():
    assert media.claude_high_res("claude-opus-4-7")
    assert media.claude_high_res("claude-sonnet-5-5")
    assert media.claude_high_res("claude-fable-5-1")
    assert not media.claude_high_res("claude-sonnet-4-5-20250929")
    assert not media.claude_high_res("claude-sonnet-4-20250514")
    assert not media.claude_high_res("claude-3-7-sonnet-20250219")
    assert not media.claude_high_res("")  # unknown: standard tier (the lower estimate)


# ---------- OpenAI (images and vision guide) ----------


def test_openai_tile_models():
    assert media.openai_image_tokens(1024, 1024, "gpt-4o", "high") == 765
    assert media.openai_image_tokens(1024, 1024, "gpt-4o", "low") == 85
    assert media.openai_image_tokens(4000, 3000, "gpt-4.1", "low") == 85


def test_openai_patch_models():
    # the guide's worked example: 1024x1024 -> ceil(1024 x 1.2); 2048x2048 -> 1600x1600 -> 3000
    assert media.openai_image_tokens(1024, 1024, "gpt-5.5", "high") == 1229
    assert media.openai_image_tokens(2048, 2048, "gpt-5.5", "high") == 3000


# ---------- Gemini (media resolution page, Gemini 3 / 4) ----------


def test_gemini_media_resolution():
    assert media.gemini_image_tokens(1920, 1080, "gemini-3-pro", "") == 1120
    assert media.gemini_image_tokens(1920, 1080, "gemini-3-pro", "MEDIA_RESOLUTION_LOW") == 280
    assert media.gemini_pdf_page_tokens("gemini-3-pro", "") == 560
    # older models: 258 per tile; small images are one tile
    assert media.gemini_image_tokens(300, 300, "gemini-2.5-flash", "") == 258


# ---------- PDFs ----------


def tiny_pdf(pages: int) -> bytes:
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Count %d >>" % pages]
    objs += [b"<< /Type /Page /Parent 2 0 R >>"] * pages
    return b"%PDF-1.4\n" + b"\n".join(objs) + b"\n%%EOF"


def test_pdf_pages():
    assert media.pdf_pages(tiny_pdf(3)) == 3
    assert media.pdf_pages(b"%PDF-1.7\n<< /Type /Pages /Count 12 >>") == 12  # object streams
    assert media.pdf_pages(b"no pdf") is None


# ---------- requests: estimates reach the segments and the analysis ----------


def claude_request(blocks: list) -> dict:
    return {
        "model": "claude-sonnet-4-5",
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": "Look at the screen"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Screenshot",
                                               "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                          "content": blocks}]},
        ],
    }


def image_block(raw: bytes, mime: str = "image/png") -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64(raw)}}


def test_segments_get_attachment_tokens():
    body = claude_request([{"type": "text", "text": "screenshot taken"}, image_block(png(1920, 1080))])
    segments, stats = media.with_media_tokens("anthropic", body["model"], body, anthropic.parse(body))
    shot = next(s for s in segments if s.kind == SegmentKind.MEDIA)
    assert shot.token_count == 1560
    assert stats == {"images": 1, "documents": 0, "unknown": 0, "tokens": 1560}


def test_documents_and_unknown_attachments():
    pdf = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                          "data": b64(tiny_pdf(2))}}
    text_doc = {"type": "document", "source": {"type": "text", "media_type": "text/plain",
                                               "data": "alpha beta gamma"}}
    url_image = {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}
    body = claude_request([pdf, text_doc, url_image])
    segments, stats = media.with_media_tokens("anthropic", body["model"], body, anthropic.parse(body))
    tokens = [s.token_count for s in segments if s.kind == SegmentKind.MEDIA]
    assert tokens[0] == 2 * media.PDF_TEXT_TOKENS_PER_PAGE  # text side only: a lower bound
    assert tokens[1] > 0  # plain-text document: counted as text
    assert tokens[2] == 0  # image behind a URL: size unknown
    assert stats["documents"] == 2 and stats["images"] == 1 and stats["unknown"] == 1


def test_non_attachment_media_keeps_its_count():
    body = {"model": "claude-sonnet-4-5",
            "messages": [{"role": "assistant", "content": [{"type": "server_tool_use", "id": "x"}]}]}
    before = anthropic.parse(body)
    after, stats = media.with_media_tokens("anthropic", body["model"], body, before)
    assert after == before and stats["tokens"] == 0


def test_stats_add_up_attachments():
    log = RequestLog(None)
    for raw in (png(1920, 1080), jpeg(1000, 1000)):
        analysis, _ = analyse("anthropic", "v1/messages",
                              json.dumps(claude_request([image_block(raw)])).encode())
        log.add({"provider": "anthropic", "method": "POST", "analysis": analysis})
    log.add({"provider": "anthropic", "method": "GET", "analysis": None})
    assert log.summary()["attachments"] == {"images": 2, "documents": 0, "unknown": 0,
                                            "tokens": 1560 + 1296}


def test_analysis_reports_attachments_and_sends_them_unchanged():
    body = claude_request([image_block(jpeg(1000, 1000), "image/jpeg")])
    analysis, new_body = analyse("anthropic", "v1/messages", json.dumps(body).encode())
    assert new_body is None
    assert analysis["media"] == {"images": 1, "documents": 0, "unknown": 0, "tokens": 1296}
    assert analysis["tokens_by_kind"]["media"] == 1296


# ---------- step 4: an identical attachment returned again by a tool becomes a note ----------


def screenshot_turns(*images: bytes, user_image: bytes | None = None) -> dict:
    """A user message (optionally with a pasted image), then one screenshot tool call per image."""
    first = [{"type": "text", "text": "Fix the layout bug"}]
    if user_image is not None:
        first.append(image_block(user_image))
    messages = [{"role": "user", "content": first}]
    for n, raw in enumerate(images):
        messages += [
            {"role": "assistant", "content": [{"type": "tool_use", "id": f"s{n}", "name": "Screenshot",
                                               "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"s{n}",
                                          "content": [image_block(raw)]}]},
        ]
    return {"model": "claude-sonnet-4-5", "max_tokens": 10, "messages": messages}


def trim(body: dict):
    segments, _ = media.with_media_tokens("anthropic", body["model"], body, anthropic.parse(body))
    return trim_request(body, segments, TrimConfig())


def tool_image(body: dict, message: int) -> dict:
    return body["messages"][message]["content"][0]["content"][0]


def test_identical_screenshot_again_becomes_a_note():
    shot = png(1920, 1080)
    body = screenshot_turns(shot, shot)
    new_body, report = trim(body)
    assert tool_image(new_body, 2) == tool_image(body, 2)  # first copy kept whole
    note = tool_image(new_body, 4)
    assert note["type"] == "text" and "attachment #1" in note["text"]
    assert [(d["how"], d["before"]) for d in report.trimmed] == [("duplicate_attachment", 1560)]


def test_different_screenshots_are_all_kept():
    new_body, report = trim(screenshot_turns(png(1920, 1080), png(1280, 720)))
    assert new_body is None and report.trimmed == []


def test_pasted_image_is_never_touched_but_counts_as_seen():
    shot = png(800, 600)
    body = screenshot_turns(shot, user_image=shot)
    new_body, report = trim(body)
    assert new_body["messages"][0] == body["messages"][0]  # the user's own message is unchanged
    assert tool_image(new_body, 2)["type"] == "text"  # the tool's copy of the same bytes is a note
    assert len(report.trimmed) == 1


def test_duplicate_note_is_identical_on_every_turn():
    shot = png(1920, 1080)
    turn = screenshot_turns(shot, shot)
    later = screenshot_turns(shot, shot, png(1280, 720))
    first, _ = trim(turn)
    second, _ = trim(later)
    assert second["messages"][:5] == first["messages"][:5]  # cached prefix stays byte-identical


# ---------- other providers: Gemini (Gemini CLI) and OpenAI ----------

def inline(raw: bytes, mime: str = "image/png") -> dict:
    return {"inlineData": {"mimeType": mime, "data": b64(raw)}}


def gemini_turns(*tool_images: bytes, nested: bool = False, user_image: bytes | None = None) -> dict:
    """Gemini CLI: a tool's image is nested in functionResponse.parts or sent as a sibling part."""
    first = [{"text": "Fix the layout bug"}] + ([inline(user_image)] if user_image else [])
    contents = [{"role": "user", "parts": first}]
    for n, raw in enumerate(tool_images):
        response = {"name": "read_file", "id": f"c{n}", "response": {"output": "Binary content"}}
        parts = [{"functionResponse": response}]
        if nested:
            response["parts"] = [inline(raw)]
        else:
            parts.append(inline(raw))
        contents += [{"role": "model", "parts": [{"functionCall": {"name": "read_file", "id": f"c{n}",
                                                                   "args": {}}}]},
                     {"role": "user", "parts": parts}]
    return {"contents": contents}


def gemini_trim(body: dict, model: str = "gemini-2.5-flash"):
    segments, stats = media.with_media_tokens("gemini", model, body, gemini.parse(body))
    return trim_request(body, segments, TrimConfig()), stats


def test_gemini_nested_tool_images_are_counted():
    body = gemini_turns(png(300, 300), nested=True)
    _, stats = gemini_trim(body)
    assert stats["images"] == 1 and stats["tokens"] == 258


def test_gemini_duplicate_sibling_tool_image_becomes_a_note():
    shot = png(1920, 1080)
    body = gemini_turns(shot, shot)
    (new_body, report), _ = gemini_trim(body)
    assert new_body["contents"][2] == body["contents"][2]  # first copy kept
    assert "attachment #1" in new_body["contents"][4]["parts"][1]["text"]
    assert [d["how"] for d in report.trimmed] == ["duplicate_attachment"]


def test_gemini_user_image_is_never_touched():
    shot = png(800, 600)
    body = gemini_turns(user_image=shot)
    body["contents"].append({"role": "user", "parts": [{"text": "again"}, inline(shot)]})
    (new_body, report), _ = gemini_trim(body)
    assert new_body is None and report.trimmed == []


def test_openai_attachments_are_counted_but_never_changed():
    url = "data:image/png;base64," + b64(png(1024, 1024))
    body = {"model": "gpt-4o", "messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]},
    ]}
    segments, stats = media.with_media_tokens("openai", "gpt-4o", body, openai.parse(body))
    assert stats == {"images": 2, "documents": 0, "unknown": 0, "tokens": 2 * 765}
    assert trim_request(body, segments, TrimConfig())[0] is None  # user messages: never changed


# ---------- dual-plan routing: a GPT model in a Claude-format request is priced as OpenAI ----------


def test_gpt_model_in_claude_format_uses_the_openai_formula():
    body = claude_request([image_block(png(1024, 1024))])
    body["model"] = "gpt-6-luna"  # Claude Code -> TokenCut -> LiteLLM -> ChatGPT plan
    _, stats = media.with_media_tokens("anthropic", body["model"], body, anthropic.parse(body))
    assert stats["tokens"] == media.openai_image_tokens(1024, 1024, "gpt-6-luna", "auto") == 1229


def objstm_pdf(pages: int) -> bytes:
    """A PDF whose page objects sit in a compressed object stream (PDF 1.5+), as many modern
    PDFs are written: the page dictionaries are not visible as plain text."""
    import zlib
    kids = " ".join(f"{3 + i} 0 R" for i in range(pages))
    objects = [b"<</Type/Catalog/Pages 2 0 R>>", f"<</Type/Pages/Kids[{kids}]/Count {pages}>>".encode()]
    objects += [b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>"] * pages
    offsets, body = [], b""
    for obj in objects:
        offsets.append(len(body))
        body += obj + b" "
    header = " ".join(f"{n + 1} {off}" for n, off in enumerate(offsets)).encode() + b" "
    stream = zlib.compress(header + body)
    stm_num, xref_num = len(objects) + 1, len(objects) + 2
    out = b"%PDF-1.7\n"
    stm_offset = len(out)
    out += (f"{stm_num} 0 obj<</Type/ObjStm/N {len(objects)}/First {len(header)}/Filter/FlateDecode"
            f"/Length {len(stream)}>>stream\n").encode() + stream + b"\nendstream endobj\n"
    xref_offset = len(out)
    rows = [b"\x00\x00\x00\x00\x00\xff\xff"]
    rows += [b"\x02" + stm_num.to_bytes(4, "big") + n.to_bytes(2, "big") for n in range(len(objects))]
    rows += [b"\x01" + stm_offset.to_bytes(4, "big") + b"\x00\x00",
             b"\x01" + xref_offset.to_bytes(4, "big") + b"\x00\x00"]
    xref = zlib.compress(b"".join(rows))
    out += (f"{xref_num} 0 obj<</Type/XRef/Size {xref_num + 1}/W[1 4 2]/Root 1 0 R/Filter/FlateDecode"
            f"/Length {len(xref)}>>stream\n").encode() + xref + b"\nendstream endobj\n"
    return out + f"startxref\n{xref_offset}\n%%EOF\n".encode()


def test_pdf_pages_in_compressed_object_streams():
    # Found on MMLongBench-Doc: such a 17-page PDF was counted as 0 pages / 0 tokens.
    assert b"/Type/Page" not in objstm_pdf(4)
    assert media.pdf_pages(objstm_pdf(4)) == 4
