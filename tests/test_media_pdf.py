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


# ---------- intent safety: a question about how pages look keeps the whole PDF ----------


def asked(question: str, *kinds: str) -> dict:
    body = tool_pdf_request(pdf_block(make_pdf(*kinds)))
    body["messages"][0]["content"] = question
    return body


def test_visual_questions_keep_the_whole_pdf():
    for question in ("What does the chart on page 2 show?", "Is the contract signed?",
                     "What colour is the logo?", "Describe the layout of the first page",
                     "Summarise the table", "Read the scanned form"):
        analysis, sent = run("anthropic", asked(question, "text", "table", "text"))
        assert sent is None and split_entries(analysis) == [], question


def test_text_questions_still_split():
    _, sent = run("anthropic", asked("Summarise the payment terms", "text", "table", "text"))
    assert [b["type"] for b in sent["messages"][2]["content"][0]["content"]] == ["text", "document"]


def test_question_attached_with_the_pdf_counts():
    body = {"model": "claude-sonnet-4-5", "max_tokens": 10, "messages": [{"role": "user", "content": [
        pdf_block(make_pdf("text", "table")), {"type": "text", "text": "Is it signed?"}]}]}
    assert run("anthropic", body, media_user=True)[1] is None


def test_a_later_visual_question_does_not_change_an_earlier_pdf():
    body = asked("Summarise the payment terms", "text", "table", "text")
    _, first = run("anthropic", body)
    later = json.loads(json.dumps(body))
    later["messages"] += [{"role": "assistant", "content": "Payment is due in 30 days."},
                          {"role": "user", "content": "What does the chart show?"}]
    _, second = run("anthropic", later)
    assert second["messages"][:3] == first["messages"][:3]


def test_log_entry_lists_text_and_picture_pages():
    analysis, _ = run("anthropic", tool_pdf_request(pdf_block(make_pdf("text", "table", "text"))))
    entry = split_entries(analysis)[0]
    assert entry["kind"] == "pdf" and entry["text_pages"] == [1, 3] and entry["visual_pages"] == [2]


def test_answer_format_instructions_are_not_visual_questions():
    # Found on MMLongBench-Doc: "Format your answer as YYYY-MM-DD" kept the whole PDF.
    for question in ("What is the date of the document? Format your answer as YYYY-MM-DD.",
                     "How many staff are listed? Answer format: a number"):
        _, sent = run("anthropic", asked(question, "text", "table", "text"))
        assert sent is not None, question
    _, sent = run("anthropic", asked("Check the formatting of page 1", "text", "table", "text"))
    assert sent is None  # real formatting questions still keep the whole PDF


def test_cache_control_kept_on_a_text_only_split():
    # A PDF with no visual pages left splits into exactly one text block; the provider's
    # prompt-cache breakpoint on the original block must survive onto it.
    block = pdf_block(make_pdf("text", "text"))
    block["cache_control"] = {"type": "ephemeral"}
    _, sent = run("anthropic", tool_pdf_request(block))
    content = sent["messages"][2]["content"][0]["content"]
    assert [b["type"] for b in content] == ["text"]
    assert content[0].get("cache_control") == {"type": "ephemeral"}


def test_cache_control_kept_on_a_text_plus_visual_split():
    # The common case: text pages plus a visual-pages subset (two blocks). The cache-control
    # breakpoint on the original block must still reach the request, on whichever block the
    # provider expects to carry it (the last block, matching Anthropic's own cache placement:
    # the breakpoint marks everything up to and including that block as cacheable).
    block = pdf_block(make_pdf("text", "table", "text"))
    block["cache_control"] = {"type": "ephemeral"}
    _, sent = run("anthropic", tool_pdf_request(block))
    content = sent["messages"][2]["content"][0]["content"]
    assert [b["type"] for b in content] == ["text", "document"]
    assert content[-1].get("cache_control") == {"type": "ephemeral"}, (
        "the PDF split must not silently drop the provider's prompt-cache breakpoint")


def test_agent_context_messages_are_not_the_users_question():
    # Found live: Claude Code sends its environment / skills context as role "system" messages
    # in the conversation ("... whenever you create any chart, graph, plot, image ..."). That is
    # not what the user asked, so it must not keep the whole PDF.
    body = asked("What are the payment terms?", "text", "table", "text")
    body["messages"].insert(1, {"role": "system", "content": [{"type": "text", "text": (
        "# Environment\n- dataviz: use whenever you create any chart, graph, plot or image; "
        "check the layout and formatting of every screenshot.")}]})
    _, sent = run("anthropic", body)
    assert sent is not None
    assert [b["type"] for b in sent["messages"][3]["content"][0]["content"]] == ["text", "document"]
