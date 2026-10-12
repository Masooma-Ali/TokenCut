"""Cache-safe trimming of a provider request (Iteration 2, phase 3).

Only large tool results are trimmed: file contents, command output, search results.
System prompt, tool definitions, user and assistant text, thinking and tool calls are
never touched, and every tool call keeps its result (only the result's text gets shorter).

Cache safety: providers bill a repeated prompt prefix at ~10% of the price, and changing
anything inside that prefix forces a full-price rewrite. So the trim of a tool result
depends ONLY on that result and on what came before it in the conversation (the user's
request, the tool call that produced it) — never on later messages or on the current
turn. A result is therefore trimmed identically on every later turn and the cached
prefix stays byte-for-byte the same. (tests/test_gateway_trimming.py checks this.)
"""

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from tokencut.contracts import TIER_BUDGET_FRACTION, Segment, SegmentKind, Tier, ToolResultKind
from tokencut.gateway.adapters.common import lookup, set_text
from tokencut.gateway.codeview import trim_file_listing
from tokencut.gateway.textcut import trim_output
from tokencut.tokens import count_tokens

_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_EDIT_WORDS = frozenset({"edit", "write", "replace", "create", "patch", "insert"})
_PATH_KEYS = ("file_path", "path", "absolute_path", "filename", "target_file", "file")
# `cat foo.py`, `head -n 50 foo.py`, `sed -n '1,80p' foo.py`, `nl foo.py` — one file only
_SHOW_FILE = re.compile(
    r"^\s*(?:cat|head|tail|nl|bat|sed\s+-n\s+\S+)\b(?:\s+-\S+)*(?:\s+\d+)?\s+"
    r"([^\s|;&<>]+\.py)\s*$"
)
# R8: a diff is code, not a log (every line matters), so it is never trimmed on arrival.
_DIFF = re.compile(r"^(?:diff --git |@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@)", re.MULTILINE)
# R8: "show me the full diff / file / output ..." -> the user wants that output whole.
# "the whole test suite" does not match: the word must refer to an output-like noun.
_FULL_REQUEST = re.compile(
    r"\b(?:full|entire|whole|complete)\b(?:\W+\w+){0,2}?\W+"
    r"(?:diffs?|files?|output|contents?|code|logs?|listing)\b",
    re.IGNORECASE,
)


@dataclass
class TrimConfig:
    tier: Tier = Tier.MODERATE
    min_tokens: int = 2_000  # tool results smaller than this are never trimmed
    min_keep_tokens: int = 500  # never cut a result below this


@dataclass
class TrimReport:
    tokens_before: int = 0
    tokens_after: int = 0
    trimmed: list[dict] = field(default_factory=list)  # one entry per trimmed segment

    @property
    def saved(self) -> int:
        return self.tokens_before - self.tokens_after

    def as_dict(self) -> dict:
        return {
            "segments_trimmed": len(self.trimmed),
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.saved,
            "details": self.trimmed,
        }


@dataclass
class _Context:
    """What is known at a given point of the conversation (only from earlier segments)."""

    last_user_text: str = ""
    user_texts: list[str] = field(default_factory=list)
    calls: dict[str, tuple[str, str]] = field(default_factory=dict)  # call id -> (name, input)
    last_call_by_name: dict[str, str] = field(default_factory=dict)  # for Gemini (no ids)
    edited: set[str] = field(default_factory=set)  # paths an edit tool has touched


def _words(name: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(name)}


def _call_path(call_input: str) -> str:
    """The file a tool call is about, if any (from its JSON input)."""
    try:
        data = json.loads(call_input)
    except ValueError:
        return ""
    if not isinstance(data, dict):
        return ""
    for key in _PATH_KEYS:
        if isinstance(data.get(key), str):
            return data[key]
    command = data.get("command") or data.get("cmd")
    if isinstance(command, list):
        command = " ".join(str(part) for part in command)
    if isinstance(command, str):
        match = _SHOW_FILE.match(command)
        if match:
            return match.group(1)
    return ""


def _is_edit_call(name: str, call_input: str) -> bool:
    if '"command": "view"' in call_input:  # editor tools that also have a read-only command
        return False
    return bool(_words(name) & _EDIT_WORDS)


def _mentioned(path: str, texts: list[str]) -> bool:
    """Did the user name this file? (then it is probably about to be edited: keep it whole)"""
    p = PurePosixPath(path.replace("\\", "/"))
    tail = "/".join(p.parts[-2:]) if len(p.parts) >= 2 else p.name
    pattern = re.compile(rf"(?<![\w.]){re.escape(p.name)}(?![\w])")
    return any(tail in text or pattern.search(text) for text in texts)


def _same_file(a: str, b: str) -> bool:
    a, b = a.replace("\\", "/"), b.replace("\\", "/")
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def trim_request(
    body: dict, segments: list[Segment], config: TrimConfig
) -> tuple[dict | None, TrimReport]:
    """Return (new body or None if nothing changed, report)."""
    report = TrimReport()
    ctx = _Context()
    new_body: dict | None = None

    for seg in segments:  # adapters emit segments in request order
        if seg.kind == SegmentKind.USER_TEXT:
            text = _REMINDER.sub("", seg.text).strip()
            if text:
                ctx.last_user_text = text
                ctx.user_texts.append(text)
        elif seg.kind == SegmentKind.TOOL_CALL:
            ctx.calls[seg.tool_call_id] = (seg.tool_name, seg.text)
            ctx.last_call_by_name[seg.tool_name] = seg.tool_call_id
            if _is_edit_call(seg.tool_name, seg.text):
                path = _call_path(seg.text)
                if path:
                    ctx.edited.add(path)
        elif seg.kind == SegmentKind.TOOL_RESULT and seg.token_count >= config.min_tokens:
            new_text, how = _trim_result(seg, ctx, config)
            if new_text is None or not isinstance(lookup(body, seg.path), str):
                continue
            after = count_tokens(new_text)
            if after >= seg.token_count:
                continue
            if new_body is None:
                new_body = copy.deepcopy(body)
            set_text(new_body, seg.path, new_text)
            report.tokens_before += seg.token_count
            report.tokens_after += after
            report.trimmed.append(
                {
                    "segment": seg.id,
                    "how": how,
                    "tool": seg.tool_name,
                    "before": seg.token_count,
                    "after": after,
                }
            )
    return new_body, report


def _trim_result(seg: Segment, ctx: _Context, config: TrimConfig) -> tuple[str | None, str]:
    call_id = seg.tool_call_id or ctx.last_call_by_name.get(seg.tool_name, "")
    _, call_input = ctx.calls.get(call_id, (seg.tool_name, ""))
    path = _call_path(call_input)
    budget = max(config.min_keep_tokens, round(TIER_BUDGET_FRACTION[config.tier] * seg.token_count))
    query = f"{ctx.last_user_text}\n{call_input}"

    # R8: trimming a diff, or output the user asked for in full, leaves fragments the agent
    # fetches again (live run L1: 3 extra requests, a net loss). Leave them whole.
    if _DIFF.search(seg.text) or _FULL_REQUEST.search(ctx.last_user_text):
        return None, "protected"

    if path.endswith(".py") and seg.result_kind in (
        ToolResultKind.FILE_READ,
        ToolResultKind.COMMAND_OUTPUT,
    ):
        # Edit safety: files the user named or the agent already edited stay whole.
        if any(_same_file(path, e) for e in ctx.edited) or _mentioned(path, ctx.user_texts):
            return None, "protected"
        return trim_file_listing(seg.text, path, query, config.tier), "code"
    if seg.result_kind == ToolResultKind.FILE_READ:
        return trim_output(seg.text, budget, "file content"), "file_head_tail"
    if seg.result_kind == ToolResultKind.COMMAND_OUTPUT:
        return trim_output(seg.text, budget, "command output"), "command_output"
    if seg.result_kind == ToolResultKind.SEARCH:
        return trim_output(seg.text, budget, "search results"), "search"
    return None, "unknown_tool"
