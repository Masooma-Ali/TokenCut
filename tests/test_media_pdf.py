"""Attachments step 7: PDF pages without images or drawings are sent as text.

Claude and OpenAI bill every PDF page twice (extracted text + a page image). A page with only
text loses nothing when sent as text; pages with images, drawings (tables, charts) or almost
no text (scans) stay a PDF. Gemini is left unchanged (its docs conflict on billing PDF text).
"""

import base64
import io
import json

from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

from tokencut.gateway import media
from tokencut.gateway.records import analyse
from tokencut.gateway.trimmer import TrimConfig

TEXT = [f"Clause {i}: the supplier delivers the goods within thirty days of the order." for i in range(30)]


def _font(writer: PdfWriter) -> DictionaryObject:
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    return DictionaryObject({NameObject("/Font"): DictionaryObject(
        {NameObject("/F1"): writer._add_object(font)})})


def _number_table_ops() -> list[str]:
    ops = ["BT /F1 10 Tf 60 740 Td (Results by model) Tj ET"]
    for row in range(8):
        for col, value in enumerate(["model_" + str(row), str(70 + row), str(12.5 + row), str(3 * row)]):
            ops.append(f"BT /F1 10 Tf {60 + col * 120} {700 - row * 16} Td ({value}) Tj ET")
    return ops


def make_pdf(*kinds: str) -> bytes:
    """Pages: "text", "table" (text + 120 drawn boxes), "image" (a picture, no text),
    "chart" (text + a chart drawn inside a form XObject), "numbers" (text + a number table)."""
    writer = PdfWriter()
    for n, kind in enumerate(kinds, 1):
        if kind == "image":
            img, buf = Image.new("RGB", (400, 500), (200, 60, 60)), io.BytesIO()
            img.save(buf, "PDF")
            writer.add_page(PdfReader(io.BytesIO(buf.getvalue())).pages[0])
            continue
        page = writer.add_blank_page(612, 792)
        page[NameObject("/Resources")] = _font(writer)
        lines = [f"Page {n}."] + TEXT
        ops = ["BT /F1 10 Tf 60 740 Td 13 TL"] + [f"({line}) Tj T*" for line in lines] + ["ET"]
        if kind == "table":
            ops += [f"{60 + i % 10 * 50} {100 + i // 10 * 20} 50 20 re S" for i in range(120)]
        if kind == "numbers":
            ops = ops[:1] + [f"(Clause {i}: short line) Tj T*" for i in range(12)] + ["ET"]
            ops += [op.replace("740", "300") for op in _number_table_ops()]
        if kind == "chart":  # a vector chart drawn inside a form XObject
            chart = DecodedStreamObject()
            chart.set_data("\n".join(f"{100 + i} {200 + i * 7 % 90} m {101 + i} {200 + i * 11 % 90} l S"
                                     for i in range(150)).encode())
            chart.update({NameObject("/Type"): NameObject("/XObject"),
                          NameObject("/Subtype"): NameObject("/Form"),
                          NameObject("/BBox"): ArrayObject([NumberObject(v) for v in (0, 0, 612, 792)])})
            page[NameObject("/Resources")][NameObject("/XObject")] = DictionaryObject(
                {NameObject("/Fm1"): writer._add_object(chart)})
            ops.append("/Fm1 Do")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(ops).encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def pdf_block(raw: bytes) -> dict:
    return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                           "data": b64(raw)}}


def tool_pdf_request(*blocks: dict, model: str = "claude-sonnet-4-5") -> dict:
    return {"model": model, "max_tokens": 10, "messages": [
        {"role": "user", "content": "Summarise the contract"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "r1", "name": "Read",
                                           "input": {"file_path": "contract.pdf"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "r1",
                                      "content": list(blocks)}]},
    ]}


def run(provider: str, body: dict, path: str = "v1/messages", **config):
    analysis, new_bytes = analyse(provider, path, json.dumps(body).encode(), TrimConfig(**config))
    return analysis, (json.loads(new_bytes) if new_bytes else None)


def pdf_pages(block: dict) -> int:
    return len(PdfReader(io.BytesIO(base64.b64decode(block["source"]["data"]))).pages)


def split_entries(analysis: dict) -> list[dict]:
    return [d for d in analysis["trim"]["details"] if d["how"] == "pdf_text_pages"]


def test_page_kinds():
    kinds = media.pdf_page_kinds(make_pdf("text", "table", "image", "text"))
    assert kinds == ["text", "visual", "visual", "text"]


def test_charts_in_form_objects_and_number_tables_stay_visual():
    # found on a real 26-page paper: charts drawn inside form XObjects and booktabs-style
    # number tables (few ruling lines) were first classified as text
    assert media.pdf_page_kinds(make_pdf("chart", "numbers", "text")) == ["visual", "visual", "text"]


def test_text_pages_become_text_and_visual_pages_stay_pdf():
    _, sent = run("anthropic", tool_pdf_request(pdf_block(make_pdf("text", "table", "text"))))
    content = sent["messages"][2]["content"][0]["content"]
    assert [b["type"] for b in content] == ["text", "document"]
    assert "pages 1, 3" in content[0]["text"] and "Clause 29" in content[0]["text"]
    assert "--- page 1 ---" in content[0]["text"] and "--- page 3 ---" in content[0]["text"]
    assert pdf_pages(content[1]) == 1  # only the table page is still a PDF


def test_report_counts_page_images_removed():
    analysis, _ = run("anthropic", tool_pdf_request(pdf_block(make_pdf("text", "table", "text"))))
    entry = split_entries(analysis)[0]
    assert entry["page_images_removed"] == 2 and entry["pages"] == 3
    assert entry["before"] == entry["after"]  # page-image tokens are not documented: not claimed


def test_text_only_pdf_becomes_one_text_block():
    _, sent = run("anthropic", tool_pdf_request(pdf_block(make_pdf("text", "text"))))
    content = sent["messages"][2]["content"][0]["content"]
    assert [b["type"] for b in content] == ["text"]


def test_pdf_without_text_pages_is_unchanged():
    analysis, sent = run("anthropic", tool_pdf_request(pdf_block(make_pdf("table", "image"))))
    assert sent is None and split_entries(analysis) == []


def test_two_pdfs_and_a_following_block_keep_their_places():
    note = {"type": "text", "text": "two files read"}
    body = tool_pdf_request(pdf_block(make_pdf("text", "image")), pdf_block(make_pdf("text")), note)
    _, sent = run("anthropic", body)
    content = sent["messages"][2]["content"][0]["content"]
    assert [b["type"] for b in content] == ["text", "document", "text", "text"]
    assert content[-1] == note


def test_same_bytes_on_every_turn():
    body = tool_pdf_request(pdf_block(make_pdf("text", "table")))
    assert run("anthropic", body)[1] == run("anthropic", body)[1]


def test_user_pdf_only_on_request():
    body = {"model": "claude-sonnet-4-5", "max_tokens": 10, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Summarise"}, pdf_block(make_pdf("text", "table"))]}]}
    assert run("anthropic", body)[1] is None
    _, sent = run("anthropic", body, media_user=True)
    assert [b["type"] for b in sent["messages"][0]["content"]] == ["text", "text", "document"]


def test_openai_file_on_request():
    url = "data:application/pdf;base64," + b64(make_pdf("text", "table"))
    body = {"model": "gpt-5.5", "messages": [{"role": "user", "content": [
        {"type": "file", "file": {"filename": "contract.pdf", "file_data": url}}]}]}
    _, sent = run("openai", body, path="v1/chat/completions", media_user=True)
    parts = sent["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "file"]
    assert "contract.pdf" in parts[0]["text"]
    assert parts[1]["file"]["file_data"].startswith("data:application/pdf;base64,")


def test_gemini_pdf_is_unchanged():
    body = {"contents": [
        {"role": "user", "parts": [{"text": "summarise"}]},
        {"role": "model", "parts": [{"functionCall": {"name": "read_file", "args": {}}}]},
        {"role": "user", "parts": [{"functionResponse": {"name": "read_file", "response": {}}},
                                   {"inlineData": {"mimeType": "application/pdf",
                                                   "data": b64(make_pdf("text", "table"))}}]}]}
    analysis, sent = run("gemini", body, path="v1beta/models/gemini-3-pro:generateContent")
    assert sent is None and split_entries(analysis) == []


def test_broken_pdf_is_unchanged():
    _, sent = run("anthropic", tool_pdf_request(pdf_block(b"%PDF-1.4 not really a pdf")))
    assert sent is None


def test_switched_off_with_the_media_setting():
    body = tool_pdf_request(pdf_block(make_pdf("text", "table")))
    assert run("anthropic", body, pdf_text_pages=False)[1] is None


def test_pdf_setting_from_environment(monkeypatch):
    from tokencut.gateway.app import GatewayConfig

    assert GatewayConfig.from_env().trim.pdf_text_pages
    monkeypatch.setenv("TOKENCUT_PDF_TEXT_PAGES", "0")
    assert not GatewayConfig.from_env().trim.pdf_text_pages
