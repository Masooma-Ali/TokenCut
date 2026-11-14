"""Command line for read_pdf.

    python -m tokencut.pdftool serve    MCP server (stdio) with the read_pdf tool
    python -m tokencut.pdftool hook     Claude Code PreToolUse hook: Read of a .pdf -> read_pdf
    python -m tokencut.pdftool read FILE [PAGES]   print what the agent would get (no images)

Claude Code, for one session (nothing saved in its settings):
    claude --mcp-config pdf_mcp.json --settings pdf_hook.json
with pdf_mcp.json {"mcpServers": {"pdf": {"command": "python", "args": ["-m",
"tokencut.pdftool", "serve"]}}} and pdf_hook.json {"hooks": {"PreToolUse": [{"matcher": "Read",
"hooks": [{"type": "command", "command": "python -m tokencut.pdftool hook"}]}]}}.
"""

import json
import sys
from typing import IO

from tokencut.pdftool.reader import read_pdf

SERVER_INFO = {"name": "tokencut-pdf", "version": "0.1.0"}
DEFAULT_PROTOCOL = "2025-06-18"
TOOL = {
    "name": "read_pdf",
    "description": (
        "Read a PDF file (use this instead of Read for every .pdf). Pages with only text come "
        "back as text; pages with charts, tables, images or scans come back as pictures. Up to "
        "20 pages per call."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path of the PDF"},
            "pages": {"type": "string",
                      "description": "Pages, e.g. '3', '1-5', '1-3,8' (default: from page 1)"},
            "as_images": {"type": "boolean",
                          "description": "All selected pages as pictures (when looks matter)"},
        },
        "required": ["file_path"],
    },
}
HOOK_MESSAGE = ("Read is not used for PDFs in this session. Use the read_pdf tool with the same "
                "file_path (and pages, if you need a range).")


def handle(message: dict) -> dict | None:
    """Answer one JSON-RPC message (None for notifications)."""
    method, msg_id = message.get("method"), message.get("id")
    params = message.get("params") or {}
    if msg_id is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": params.get("protocolVersion") or DEFAULT_PROTOCOL,
                  "capabilities": {"tools": {"listChanged": False}}, "serverInfo": SERVER_INFO}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": [TOOL]}
    elif method == "tools/call" and params.get("name") == "read_pdf":
        args = params.get("arguments") or {}
        content, is_error = read_pdf(str(args.get("file_path", "")), str(args.get("pages") or ""),
                                     bool(args.get("as_images", False)))
        result = {"content": content, "isError": is_error}
    else:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def serve(stdin: IO[str] = sys.stdin, stdout: IO[str] = sys.stdout) -> None:
    for line in stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None,
                     "error": {"code": -32700, "message": "Parse error"}}
        else:
            reply = handle(message) if isinstance(message, dict) else None
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()


def hook(stdin: IO[str] = sys.stdin, stderr: IO[str] = sys.stderr) -> int:
    """Exit code 2 blocks the tool call and shows stderr to the model; 0 lets it run."""
    try:
        event = json.load(stdin)
    except ValueError:
        return 0  # not our business: never block on a malformed event
    path = str((event.get("tool_input") or {}).get("file_path", ""))
    if event.get("tool_name") == "Read" and path.lower().endswith(".pdf"):
        stderr.write(HOOK_MESSAGE + "\n")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["serve"]:
        serve()
        return 0
    if args[:1] == ["hook"]:
        return hook()
    if args[:1] == ["read"] and len(args) >= 2:
        content, is_error = read_pdf(args[1], args[2] if len(args) > 2 else "")
        for block in content:
            print(block["text"] if block["type"] == "text" else f"<{block['mimeType']} picture>")
        return 1 if is_error else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
