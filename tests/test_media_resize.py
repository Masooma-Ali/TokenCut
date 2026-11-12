"""Attachments step 5: shrink large images where the provider bills by size.

Rule: target long edge 1,568 px (what standard-tier Claude models always receive), only if
the provider's own formula saves at least 10 %; never upscale; never for computer use or a
request to zoom / read small text; tool images by default, user attachments only on request.
"""

import base64
import io
import json

from PIL import Image

from tokencut.gateway import media
from tokencut.gateway.records import analyse
from tokencut.gateway.trimmer import TrimConfig


def picture(w: int, h: int, fmt: str = "PNG") -> bytes:
    img = Image.new("RGB", (w, h), (30, 120, 200))
    for x in range(0, w, 97):  # some detail so encoders have work to do
        img.putpixel((x, h // 2), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def claude_tool_image(raw: bytes, model: str, user_text: str = "Fix the layout",
                      tool: str = "Screenshot", tools: list | None = None) -> dict:
    body = {
        "model": model,
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "s1", "name": tool,
                                               "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "s1", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": b64(raw)}}]}]},
        ],
    }
    if tools is not None:
        body["tools"] = tools
    return body


def run(provider: str, body: dict, path: str = "v1/messages", **config) -> tuple[dict, dict | None]:
    analysis, new_bytes = analyse(provider, path, json.dumps(body).encode(), TrimConfig(**config))
    return analysis, (json.loads(new_bytes) if new_bytes else None)


def resized(analysis: dict) -> list[dict]:
    return [d for d in analysis["trim"]["details"] if d["how"] == "resized"]


def sent_size(block: dict) -> tuple[int, int]:
    return media.image_size(base64.b64decode(block["source"]["data"]))


def tool_block(body: dict) -> dict:
    return body["messages"][2]["content"][0]["content"][0]


def test_high_res_claude_screenshot_is_shrunk_to_the_standard_size():
    analysis, sent = run("anthropic", claude_tool_image(picture(1920, 1080), "claude-haiku-5-5"))
    assert sent_size(tool_block(sent)) == (1568, 882)
    assert [(d["before"], d["after"]) for d in resized(analysis)] == [(69 * 39, 56 * 32)]


def test_standard_claude_resizes_itself_so_nothing_is_saved_or_changed():
    analysis, sent = run("anthropic", claude_tool_image(picture(1920, 1080), "claude-sonnet-4-5"))
    assert sent is None and resized(analysis) == []


def test_small_images_are_never_touched():
    analysis, sent = run("anthropic", claude_tool_image(picture(1024, 768), "claude-haiku-5-5"))
    assert sent is None and resized(analysis) == []


def test_computer_use_screenshots_keep_their_size():
    computer = [{"type": "computer_20250124", "name": "computer", "display_width_px": 1920,
                 "display_height_px": 1080}]
    for body in (claude_tool_image(picture(1920, 1080), "claude-haiku-5-5", tools=computer),
                 claude_tool_image(picture(1920, 1080), "claude-haiku-5-5", tool="computer")):
        analysis, sent = run("anthropic", body)
        assert sent is None and resized(analysis) == []


def test_request_to_read_small_text_keeps_the_size():
    body = claude_tool_image(picture(1920, 1080), "claude-haiku-5-5",
                             user_text="Zoom in and read the small text in the corner")
    analysis, sent = run("anthropic", body)
    assert sent is None and resized(analysis) == []


def test_resizing_is_identical_on_every_turn():
    body = claude_tool_image(picture(1920, 1080), "claude-haiku-5-5")
    _, first = run("anthropic", body)
    _, second = run("anthropic", body)
    assert first == second


def test_jpeg_stays_jpeg():
    raw = picture(2400, 1600, "JPEG")
    body = claude_tool_image(raw, "claude-haiku-5-5")
    tool_block(body)["source"]["media_type"] = "image/jpeg"
    _, sent = run("anthropic", body)
    assert base64.b64decode(tool_block(sent)["source"]["data"])[:2] == b"\xff\xd8"
    assert sent_size(tool_block(sent)) == (1568, 1045)


def test_user_attachments_only_on_request():
    body = {"model": "claude-haiku-5-5", "max_tokens": 10, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is wrong here?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": b64(picture(1920, 1080))}}]}]}
    analysis, sent = run("anthropic", body)
    assert sent is None and resized(analysis) == []
    analysis, sent = run("anthropic", body, media_user=True)
    assert media.image_size(base64.b64decode(sent["messages"][0]["content"][1]["source"]["data"])) \
        == (1568, 882)


def test_openai_patch_model_user_image_on_request():
    url = "data:image/png;base64," + b64(picture(1920, 1080))
    body = {"model": "gpt-5.5", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is wrong here?"},
        {"type": "image_url", "image_url": {"url": url}}]}]}
    analysis, sent = run("openai", body, path="v1/chat/completions", media_user=True)
    new_url = sent["messages"][0]["content"][1]["image_url"]["url"]
    assert new_url.startswith("data:image/png;base64,")
    assert media.image_size(base64.b64decode(new_url.split(",", 1)[1])) == (1568, 882)
    assert [(d["before"], d["after"]) for d in resized(analysis)] == [(2448, 1647)]


def test_openai_tile_model_saves_nothing_so_nothing_changes():
    url = "data:image/png;base64," + b64(picture(1920, 1080))
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}}]}]}
    analysis, sent = run("openai", body, path="v1/chat/completions", media_user=True)
    assert sent is None and resized(analysis) == []


def test_gemini_3_bills_per_image_so_nothing_changes():
    body = {"contents": [
        {"role": "user", "parts": [{"text": "check"}]},
        {"role": "model", "parts": [{"functionCall": {"name": "read_file", "args": {}}}]},
        {"role": "user", "parts": [{"functionResponse": {"name": "read_file", "response": {}}},
                                   {"inlineData": {"mimeType": "image/png",
                                                   "data": b64(picture(1920, 1080))}}]}]}
    analysis, sent = run("gemini", body, path="v1beta/models/gemini-3-pro:generateContent")
    assert sent is None and resized(analysis) == []


def test_switched_off():
    analysis, sent = run("anthropic", claude_tool_image(picture(1920, 1080), "claude-haiku-5-5"),
                         resize_images=False)
    assert sent is None and resized(analysis) == []


def test_settings_from_environment(monkeypatch):
    from tokencut.gateway.app import GatewayConfig

    default = GatewayConfig.from_env().trim
    assert default.resize_images and not default.media_user
    monkeypatch.setenv("TOKENCUT_RESIZE_IMAGES", "0")
    monkeypatch.setenv("TOKENCUT_MEDIA_USER", "1")
    changed = GatewayConfig.from_env().trim
    assert not changed.resize_images and changed.media_user
