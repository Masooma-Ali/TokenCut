"""read_pdf (local MCP tool) and its Claude Code hook."""

import base64
import io
import json
import shutil

import pytest

from eval.attachments_bench import _report_pdf
from tokencut.pdftool.__main__ import handle, hook, serve
from tokencut.pdftool.reader import MAX_PAGES, parse_pages, read_pdf

needs_poppler = pytest.mark.skipif(shutil.which("pdftoppm") is None, reason="needs Poppler")


@pytest.fixture
def agreement(tmp_path):
    # Pages 1, 2, 5 text ("Payment terms: net 45 days" on 2); 3 a vector chart; 4 a number table.
    path = tmp_path / "agreement.pdf"
    _report_pdf(path)
    return path


def kinds(content: list[dict]) -> list[str]:
    return [b["text"].split("\n")[0] if b["type"] == "text" else "<image>" for b in content[1:]]


@needs_poppler
def test_text_pages_as_text_and_chart_and_table_pages_as_pictures(agreement):
    content, is_error = read_pdf(str(agreement))
    assert not is_error
    assert "As text: 1, 2, 5. As pictures (charts, tables, images or scans): 3, 4." in content[0]["text"]
    assert kinds(content) == ["--- page 1 ---", "--- page 3 (picture) ---", "<image>",
                              "--- page 4 (picture) ---", "<image>", "--- page 5 ---"]
    assert "net 45 days" in content[1]["text"]  # pages 1-2 share one text block
    png = base64.b64decode(content[3]["data"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


@needs_poppler
def test_page_range_and_as_images(agreement):
    content, _ = read_pdf(str(agreement), pages="2-3")
    assert kinds(content) == ["--- page 2 ---", "--- page 3 (picture) ---", "<image>"]
    content, _ = read_pdf(str(agreement), pages="2", as_images=True)
    assert kinds(content) == ["--- page 2 (picture) ---", "<image>"]


def test_text_only_pages_need_no_poppler(agreement, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    content, is_error = read_pdf(str(agreement), pages="1-2")
    assert not is_error and kinds(content) == ["--- page 1 ---"]


def test_parse_pages():
    assert parse_pages("", 5) == [1, 2, 3, 4, 5]
    assert parse_pages("1-3, 8", 10) == [1, 2, 3, 8]
    assert parse_pages("4-9", 5) == [4, 5]
    assert parse_pages("", 100) == list(range(1, MAX_PAGES + 1))
    with pytest.raises(ValueError):
        parse_pages("first", 5)


def test_errors_are_reported_not_raised(tmp_path):
    content, is_error = read_pdf(str(tmp_path / "missing.pdf"))
    assert is_error and "Cannot read" in content[0]["text"]
    content, is_error = read_pdf(str(tmp_path), pages="x")
    assert is_error


def run_hook(event: dict) -> tuple[int, str]:
    err = io.StringIO()
    return hook(io.StringIO(json.dumps(event)), err), err.getvalue()


def test_hook_sends_pdf_reads_to_read_pdf():
    code, message = run_hook({"tool_name": "Read", "tool_input": {"file_path": "/d/Report.PDF"}})
    assert code == 2 and "read_pdf" in message
    assert run_hook({"tool_name": "Read", "tool_input": {"file_path": "/d/app.py"}})[0] == 0
    assert run_hook({"tool_name": "Bash", "tool_input": {"command": "ls x.pdf"}})[0] == 0
    assert hook(io.StringIO("not json"), io.StringIO()) == 0  # never block on a bad event


def test_mcp_server_lists_and_calls_the_tool(agreement):
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "read_pdf", "arguments": {"file_path": str(agreement), "pages": "2"}}},
    ]
    out = io.StringIO()
    serve(io.StringIO("\n".join(json.dumps(m) for m in messages) + "\n"), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [1, 2, 3]  # no reply to the notification
    assert replies[1]["result"]["tools"][0]["name"] == "read_pdf"
    assert "net 45 days" in replies[2]["result"]["content"][1]["text"]
    assert handle({"jsonrpc": "2.0", "id": 4, "method": "nope"})["error"]["code"] == -32601
