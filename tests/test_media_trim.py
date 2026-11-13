"""Attachments step 6: plain borders around an image are cut off (outer margins only).

Research (PixelPrune arXiv 2604.00886; arXiv 2509.06415) shows blank areas are a large share
of document / GUI images and removing them keeps accuracy. Through an API TokenCut can only
send a smaller picture, so it removes the outer plain margin, keeps an 8-px gap, adds a note
with the original size, and skips layout questions and computer use.
"""

import base64
import io
import json

from PIL import Image, ImageDraw

from tokencut.gateway import media
from tokencut.gateway.records import analyse
from tokencut.gateway.trimmer import TrimConfig


def framed(canvas: tuple[int, int], content: tuple[int, int, int, int], fmt: str = "PNG") -> bytes:
    """A white canvas with a busy coloured block at content = (left, top, right, bottom)."""
    img = Image.new("RGB", canvas, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = content
    draw.rectangle(content, fill=(40, 90, 160))
    for x in range(left + 2, right - 1, 40):  # 3-px lines stay inside the block
        draw.line([(x, top), (x, bottom)], fill=(250, 200, 30), width=3)
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


CHART = framed((1600, 1000), (200, 150, 1399, 849))  # content 1200 x 700, centred


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def tool_image(raw: bytes, model: str = "claude-haiku-5-5", question: str = "Explain this chart",
               tools: list | None = None) -> dict:
    body = {"model": model, "max_tokens": 10, "messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "v1", "name": "ViewImage",
                                           "input": {"path": "chart.png"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "v1", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": b64(raw)}}]}]},
    ]}
    if tools is not None:
        body["tools"] = tools
    return body


def run(provider: str, body: dict, path: str = "v1/messages", **config):
    analysis, new_bytes = analyse(provider, path, json.dumps(body).encode(), TrimConfig(**config))
    return analysis, (json.loads(new_bytes) if new_bytes else None)


def image_entries(analysis: dict) -> list[tuple]:
    return [(d["how"], d["before"], d["after"]) for d in analysis["trim"]["details"]
            if d["how"] in ("border_trimmed", "border_trimmed_resized", "resized")]


def size_of(block: dict) -> tuple[int, int]:
    return media.image_size(base64.b64decode(block["source"]["data"]))


def test_content_box():
    assert media.content_box(Image.open(io.BytesIO(CHART))) == (192, 142, 1408, 858)
    plain = Image.new("RGB", (800, 600), (0, 0, 0))
    assert media.content_box(plain) is None  # nothing but background


def test_border_is_cut_and_a_note_follows_the_image():
    analysis, sent = run("anthropic", tool_image(CHART))
    blocks = sent["messages"][2]["content"][0]["content"]
    assert [b["type"] for b in blocks] == ["image", "text"]
    assert size_of(blocks[0]) == (1216, 716)
    assert "1600x1000" in blocks[1]["text"] and "1216x716" in blocks[1]["text"]
    assert image_entries(analysis) == [("border_trimmed", 58 * 36, 44 * 26)]


def test_standard_claude_also_saves():
    analysis, sent = run("anthropic", tool_image(CHART, model="claude-sonnet-4-5"))
    assert size_of(sent["messages"][2]["content"][0]["content"][0]) == (1216, 716)
    assert image_entries(analysis) == [("border_trimmed", 1550, 44 * 26)]  # 1550: tier downsize


def test_large_image_is_cut_then_shrunk():
    big = framed((3000, 2000), (300, 250, 2699, 1749))
    analysis, sent = run("anthropic", tool_image(big))
    assert size_of(sent["messages"][2]["content"][0]["content"][0]) == (1568, 984)
    assert [e[0] for e in image_entries(analysis)] == ["border_trimmed_resized"]


def test_no_plain_border_no_change():
    edge_to_edge = framed((1600, 1000), (0, 0, 1599, 999))
    for raw in (edge_to_edge, framed((1600, 1000), (20, 15, 1579, 984))):  # tiny border: < 10 %
        analysis, sent = run("anthropic", tool_image(raw))
        assert sent is None and image_entries(analysis) == []


def test_nearly_empty_image_is_left_alone():
    dot = framed((1600, 1000), (790, 490, 810, 510))  # would remove > 75 % of the area
    analysis, sent = run("anthropic", tool_image(dot))
    assert sent is None and image_entries(analysis) == []


def test_layout_questions_keep_the_margins():
    for question in ("Is the chart centred on the page?", "Why is there so much spacing on the left?",
                     "Check the alignment and padding"):
        analysis, sent = run("anthropic", tool_image(CHART, question=question))
        assert sent is None and image_entries(analysis) == []


def test_computer_use_keeps_the_margins():
    computer = [{"type": "computer_20250124", "name": "computer", "display_width_px": 1600,
                 "display_height_px": 1000}]
    analysis, sent = run("anthropic", tool_image(CHART, tools=computer))
    assert sent is None and image_entries(analysis) == []


def test_gemini_3_bills_per_image_so_nothing_changes():
    body = {"contents": [
        {"role": "user", "parts": [{"text": "explain the chart"}]},
        {"role": "model", "parts": [{"functionCall": {"name": "read_file", "args": {}}}]},
        {"role": "user", "parts": [{"functionResponse": {"name": "read_file", "response": {}}},
                                   {"inlineData": {"mimeType": "image/png", "data": b64(CHART)}}]}]}
    analysis, sent = run("gemini", body, path="v1beta/models/gemini-3-pro:generateContent")
    assert sent is None and image_entries(analysis) == []


def test_openai_user_image_on_request_gets_a_note_part():
    url = "data:image/png;base64," + b64(CHART)
    body = {"model": "gpt-5.5", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Explain this chart"},
        {"type": "image_url", "image_url": {"url": url}}]}]}
    assert run("openai", body, path="v1/chat/completions")[1] is None  # user images: opt-in
    _, sent = run("openai", body, path="v1/chat/completions", media_user=True)
    parts = sent["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "text"]
    assert "1600x1000" in parts[2]["text"]


def test_same_bytes_on_every_turn():
    assert run("anthropic", tool_image(CHART))[1] == run("anthropic", tool_image(CHART))[1]


def test_switched_off(monkeypatch):
    analysis, sent = run("anthropic", tool_image(CHART), trim_borders=False)
    assert sent is None and image_entries(analysis) == []
    from tokencut.gateway.app import GatewayConfig

    assert GatewayConfig.from_env().trim.trim_borders
    monkeypatch.setenv("TOKENCUT_TRIM_BORDERS", "0")
    assert not GatewayConfig.from_env().trim.trim_borders


def test_a_later_layout_question_does_not_change_an_earlier_image():
    _, first = run("anthropic", tool_image(CHART))
    later = tool_image(CHART)
    later["messages"] += [{"role": "assistant", "content": "It is a bar chart."},
                          {"role": "user", "content": "Is it centred on the page?"}]
    _, second = run("anthropic", later)
    assert second["messages"][:3] == first["messages"][:3]
