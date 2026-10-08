"""Guess what a tool result contains, from the tool's name and the text itself."""

import re

from tokencut.contracts import ToolResultKind

# Checked in this order; first match wins. Matched against the whole words of the tool name
# ("read_many_files" -> read, many, files), so "notification" never matches "cat".
# Names seen in Claude Code, Gemini CLI, Cline, Codex...
_NAME_RULES: list[tuple[ToolResultKind, frozenset[str]]] = [
    (ToolResultKind.SEARCH, frozenset({"grep", "glob", "search", "find", "ls", "list"})),
    (ToolResultKind.FILE_READ, frozenset({"read", "view", "open", "cat"})),
    (
        ToolResultKind.COMMAND_OUTPUT,
        frozenset({"bash", "shell", "exec", "execute", "run", "terminal", "command", "cmd"}),
    ),
]
_NAME_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
# "  12→code", "12\tcode", "  12 | code", "12: code" (line-numbered file listings)
_NUMBERED_LINE = re.compile(r"^\s*\d+\s*(?:→|\t|\||│|:)")
_SAMPLE_LINES = 20
_NUMBERED_SHARE = 0.6


def looks_like_numbered_file(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()][:_SAMPLE_LINES]
    if len(lines) < 2:
        return False
    numbered = sum(1 for line in lines if _NUMBERED_LINE.match(line))
    return numbered / len(lines) >= _NUMBERED_SHARE


def classify_tool_result(tool_name: str, text: str) -> ToolResultKind:
    words = {w.lower() for w in _NAME_WORD.findall(tool_name)}
    for kind, rule_words in _NAME_RULES:
        if words & rule_words:
            return kind
    # Unknown tool (e.g. an editor tool's "view" command): fall back to the content.
    if looks_like_numbered_file(text):
        return ToolResultKind.FILE_READ
    return ToolResultKind.OTHER
