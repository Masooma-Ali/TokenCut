"""The repo index as a local MCP server (Model Context Protocol, stdio transport).

    claude mcp add tokencut -- python -m tokencut.repoindex serve --repo .

The agent then has three tools (Claude Code names them mcp__tokencut__<name>). Every call
first refreshes the index from the working tree, so answers include uncommitted edits.
Messages are JSON-RPC 2.0, one per line on stdin / stdout; logs go to stderr.
"""

import json
import logging
import sys
import time
from typing import IO

from tokencut.repoindex.index import RepoIndex, timed

log = logging.getLogger(__name__)

SERVER_INFO = {"name": "tokencut-repoindex", "version": "0.1.0"}
DEFAULT_PROTOCOL = "2025-06-18"
SAVE_EVERY_SECONDS = 60

TOOLS = [
    {
        "name": "repo_map",
        "description": (
            "Overview of the repository: directories, then the classes and function "
            "signatures of its most-used files. Call it first in an unfamiliar repository, "
            "or with a directory prefix to zoom in. Cheaper than listing and reading files."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prefix": {"type": "string", "description": "Only files under this directory"},
                "max_tokens": {"type": "integer", "description": "Size limit (default 2000)"},
            },
        },
    },
    {
        "name": "find_code",
        "description": (
            "Find the functions / classes relevant to a question, bug report, error message "
            "or traceback. Returns the best matches with file, line range and code (with line "
            "numbers). Use it instead of several grep + read rounds."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Issue text, error, or symbol names"},
                "k": {"type": "integer", "description": "Number of matches (default 8)"},
                "max_tokens": {"type": "integer", "description": "Code shown (default 3000)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "symbol",
        "description": (
            "Exact current source of a function, method or class by name (e.g. "
            "'QuerySet.only'), with line numbers, plus what calls it and what it calls."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Name or Class.method"}},
            "required": ["name"],
        },
    },
]


def _int(args: dict, key: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(args.get(key, default))))
    except (TypeError, ValueError):
        return default


def call_tool(index: RepoIndex, name: str, args: dict) -> tuple[str, bool]:
    """(text, is_error) for one tool call."""
    index.refresh()
    if name == "repo_map":
        return timed(lambda: index.repo_map(str(args.get("prefix", "")),
                                            _int(args, "max_tokens", 2000, 200, 20_000))), False
    if name == "find_code":
        query = str(args.get("query", "")).strip()
        if not query:
            return "find_code needs a query.", True
        return timed(lambda: index.find_code(query, _int(args, "k", 8, 1, 50),
                                             _int(args, "max_tokens", 3000, 200, 20_000))), False
    if name == "symbol":
        symbol = str(args.get("name", "")).strip()
        if not symbol:
            return "symbol needs a name.", True
        return timed(lambda: index.symbol(symbol)), False
    return f"Unknown tool {name!r}.", True


def handle(index: RepoIndex, message: dict) -> dict | None:
    """Answer one JSON-RPC message (None for notifications)."""
    method = message.get("method")
    msg_id = message.get("id")
    params = message.get("params") or {}
    if msg_id is None:
        return None  # notification (e.g. notifications/initialized)
    if method == "initialize":
        result = {
            "protocolVersion": params.get("protocolVersion") or DEFAULT_PROTOCOL,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        text, is_error = call_tool(index, str(params.get("name", "")),
                                   params.get("arguments") or {})
        result = {"content": [{"type": "text", "text": text}], "isError": is_error}
    else:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def serve(index: RepoIndex, stdin: IO[str] = sys.stdin, stdout: IO[str] = sys.stdout) -> None:
    last_save = time.monotonic()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None,
                     "error": {"code": -32700, "message": "Parse error"}}
        else:
            reply = handle(index, message) if isinstance(message, dict) else None
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()
        if time.monotonic() - last_save > SAVE_EVERY_SECONDS:
            index.save()
            last_save = time.monotonic()
    index.save()
