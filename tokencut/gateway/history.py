"""Cache-event trimming: re-trim the whole history when the prompt cache is lost anyway.

Normal turns: only newly arrived tool results are trimmed (trimmer.trim_request), and
every earlier decision is re-applied unchanged, so the cached prefix stays identical.

Cache events (model switch, cache expired, context nearly full): the provider re-reads
the whole prompt at full price anyway, so TokenCut also shrinks old history:
  * old command / test / search output -> one-line note + its error lines
  * an old file read -> a note (newer read later / file edited since / read again)
  * optional: tool definitions -> only tools used so far + core tools
The most recent turns are never touched. Decisions are remembered (session.py) and sent
identically until the next event. A file the agent re-reads after we removed an old read
of it is protected from then on.
"""

import copy
import json
import re
import time
from dataclasses import dataclass

from tokencut.contracts import Segment, SegmentKind, Tier, ToolResultKind
from tokencut.gateway.adapters.common import lookup, set_text
from tokencut.gateway.session import ConversationState, SessionStore, conversation_key, fingerprint
from tokencut.gateway.textcut import _ERROR
from tokencut.gateway.trimmer import (
    TrimConfig,
    TrimReport,
    _call_path,
    _is_edit_call,
    _same_file,
    trim_request,
)
from tokencut.tokens import count_tokens

# Recent turns (assistant messages, counted from the end) kept exactly as they are.
KEEP_RECENT_TURNS = {Tier.AGGRESSIVE: 2, Tier.MODERATE: 4, Tier.CONSERVATIVE: 8, Tier.FULL: 10**9}
MAX_KEY_LINES = 10  # error lines kept in a note about removed output
# Claude Code / Gemini CLI / common agent tools always offered when pruning tool definitions.
DEFAULT_CORE_TOOLS = (
    "Read", "Edit", "MultiEdit", "Write", "Bash", "Grep", "Glob", "LS", "TodoWrite", "Task",
    "read_file", "write_file", "replace", "run_shell_command", "search_file_content", "glob",
    "list_directory", "apply_patch", "str_replace_editor", "str_replace_based_edit_tool",
)
_ONE_HOUR_TTL = re.compile(r'"ttl"\s*:\s*"1h"')


@dataclass
class CacheEventConfig:
    cache_ttl_seconds: int = 300  # provider default; requests marked ttl "1h" use 3600
    compact_at_tokens: int = 110_000  # proxy tokens (~165k Claude tokens): context nearly full
    compact_growth_tokens: int = 20_000  # re-trigger "nearly full" only after this much growth
    history_min_tokens: int = 200  # older results smaller than this are left alone
    prune_tools: bool = False  # opt-in: fewer tool definitions at events
    core_tools: tuple[str, ...] = DEFAULT_CORE_TOOLS


# ---------- event detection ----------


def detect_event(state: ConversationState | None, model: str, now: float, total_tokens: int,
                 body: dict, config: CacheEventConfig) -> str | None:
    if state is None:
        return None  # first time we see this conversation: its cache may still be valid
    if model and state.model and model != state.model:
        return "model_switch"
    ttl = 3600 if _ONE_HOUR_TTL.search(json.dumps(body.get("system", ""))) else config.cache_ttl_seconds
    if now - state.last_seen > ttl:
        return "cache_expired"
    if total_tokens >= config.compact_at_tokens and (
        state.last_event_tokens is None
        or total_tokens - state.last_event_tokens >= config.compact_growth_tokens
    ):
        return "context_full"
    return None


# ---------- what each tool result is ----------


@dataclass
class _ResultInfo:
    seg: Segment
    position: int  # index in the segment list
    path: str  # file the producing call was about ("" if none)
    call_input: str
    call_name: str


def _scan(segments: list[Segment]) -> tuple[list[_ResultInfo], list[tuple[int, str]]]:
    """Tool results with their producing call, and (position, path) of every edit call."""
    calls: dict[str, tuple[str, str]] = {}
    last_by_name: dict[str, str] = {}
    results: list[_ResultInfo] = []
    edits: list[tuple[int, str]] = []
    for pos, seg in enumerate(segments):
        if seg.kind == SegmentKind.TOOL_CALL:
            calls[seg.tool_call_id] = (seg.tool_name, seg.text)
            last_by_name[seg.tool_name] = seg.tool_call_id
            if _is_edit_call(seg.tool_name, seg.text):
                path = _call_path(seg.text)
                if path:
                    edits.append((pos, path))
        elif seg.kind == SegmentKind.TOOL_RESULT:
            call_id = seg.tool_call_id or last_by_name.get(seg.tool_name, "")
            name, call_input = calls.get(call_id, (seg.tool_name, ""))
            # An edit call's result ("file updated") is not a read of the file.
            path = "" if _is_edit_call(name, call_input) else _call_path(call_input)
            results.append(_ResultInfo(seg, pos, path, call_input, name))
    return results, edits


def _recent_from(segments: list[Segment], keep_turns: int) -> int:
    """Message index from which messages count as recent (never re-trimmed)."""
    assistant_messages = sorted({s.message_index for s in segments
                                 if s.role in ("assistant", "model") and s.message_index >= 0})
    if len(assistant_messages) <= keep_turns:
        return 0
    return assistant_messages[-keep_turns]


def _command_label(call_input: str) -> str:
    try:
        data = json.loads(call_input)
    except ValueError:
        return ""
    if isinstance(data, dict):
        for key in ("command", "cmd", "pattern", "query", "path"):
            value = data.get(key)
            if isinstance(value, list):
                value = " ".join(map(str, value))
            if isinstance(value, str) and value:
                return value if len(value) <= 80 else value[:77] + "..."
    return ""


def _output_note(info: _ResultInfo) -> str:
    what = {ToolResultKind.SEARCH: "search results", ToolResultKind.FILE_READ: "file content"}.get(
        info.seg.result_kind, "output")
    label = _command_label(info.call_input)
    head = f"[TokenCut: earlier {what}"
    head += f" of `{label}`" if label else (f" of {info.call_name}" if info.call_name else "")
    head += f" removed ({info.seg.token_count} tokens)"
    lines = [line for line in info.seg.text.split("\n") if line.strip()]
    key = [line for line in lines if _ERROR.search(line)][:MAX_KEY_LINES]
    if lines and lines[-1] not in key:
        key.append(lines[-1])  # usually the summary ("3 failed, 40 passed")
    if not key:
        return head + "]"
    return head + ". Key lines:]\n" + "\n".join(line[:300] for line in key)


def _history_replacements(segments: list[Segment], state: ConversationState,
                          config: CacheEventConfig, keep_turns: int) -> dict[str, tuple[str, str]]:
    """fingerprint -> (replacement text, how) for old tool results, decided at an event."""
    results, edits = _scan(segments)
    recent_from = _recent_from(segments, keep_turns)
    decided: dict[str, tuple[str, str]] = {}
    for i, info in enumerate(results):
        seg = info.seg
        if seg.message_index >= recent_from or seg.token_count < config.history_min_tokens:
            continue
        fp = fingerprint(seg)
        if fp in state.pinned:
            continue  # already decided at an earlier event: keep it identical
        path = info.path
        if path and any(_same_file(path, p) for p in state.protected_paths):
            continue
        if path:
            newer = any(r.path and _same_file(r.path, path) for r in results[i + 1:])
            edited = any(pos > info.position and _same_file(p, path) for pos, p in edits)
            if newer:
                note = (f"[TokenCut: earlier read of {path} removed ({seg.token_count} tokens); "
                        "a newer read of this file appears later in the conversation]")
                how = "history_duplicate_read"
            elif edited:
                note = (f"[TokenCut: earlier read of {path} removed ({seg.token_count} tokens); "
                        "the file was edited afterwards, read it again for the current version]")
                how = "history_stale_read"
            elif seg.result_kind == ToolResultKind.FILE_READ:
                note = (f"[TokenCut: earlier read of {path} removed ({seg.token_count} tokens); "
                        "read it again if needed]")
                how = "history_old_read"
            else:
                note, how = _output_note(info), "history_old_output"
            state.masked_paths.add(path)
        else:
            note, how = _output_note(info), "history_old_output"
        if count_tokens(note) < seg.token_count:
            decided[fp] = (note, how)
    return decided


# ---------- tool definitions ----------


def _tool_lists(body: dict) -> list[tuple[list, str]]:
    """(list of tool definitions, how to read a name) for each provider format."""
    tools = body.get("tools")
    if isinstance(tools, list) and tools and all(isinstance(t, dict) for t in tools):
        if any("functionDeclarations" in t or "function_declarations" in t for t in tools):
            out = []
            for t in tools:
                decls = t.get("functionDeclarations", t.get("function_declarations"))
                if isinstance(decls, list):
                    out.append((decls, "name"))
            return out
        if any("function" in t for t in tools):
            return [(tools, "function.name")]
        return [(tools, "name")]
    if isinstance(body.get("functions"), list):
        return [(body["functions"], "name")]
    return []


def _tool_name(tool: dict, how: str) -> str:
    if how == "function.name":
        return str((tool.get("function") or {}).get("name", ""))
    return str(tool.get("name", ""))


def _forced_tool(body: dict) -> str:
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        return str(choice.get("name") or (choice.get("function") or {}).get("name", ""))
    return ""


def _prune_tools(body: dict, keep: set[str]) -> int:
    """Remove tool definitions not in `keep` (in place). Returns tokens removed."""
    removed = 0
    for tool_list, how in _tool_lists(body):
        kept = []
        for tool in tool_list:
            name = _tool_name(tool, how)
            if not name or name in keep:
                kept.append(tool)
            else:
                removed += count_tokens(json.dumps(tool, ensure_ascii=False, sort_keys=True))
        tool_list[:] = kept
    return removed


# ---------- the whole request ----------


def trim_with_cache_events(provider: str, model: str, body: dict, segments: list[Segment],
                           config: TrimConfig, events_config: CacheEventConfig,
                           store: SessionStore, now: float | None = None,
                           total_tokens: int | None = None) -> tuple[dict | None, TrimReport, str | None]:
    """Returns (new body or None, report, cache event or None)."""
    now = time.time() if now is None else now
    total = total_tokens if total_tokens is not None else sum(s.token_count for s in segments)
    new_body, report = trim_request(body, segments, config)  # on-arrival trims (cache-safe)
    key = conversation_key(provider, segments)
    if key is None:
        return new_body, report, None

    with store.lock:
        state = store.get(key)
        event = detect_event(state, model, now, total, body, events_config)
        if state is None:
            state = ConversationState(model=model, last_seen=now)

        # A new read of a file we removed earlier: the agent needs it, protect it from now on.
        results, _ = _scan(segments)
        for info in results:
            if info.path and fingerprint(info.seg) not in state.pinned and any(
                _same_file(info.path, p) for p in state.masked_paths
            ):
                state.protected_paths.add(info.path)

        if event is not None:
            decided = _history_replacements(segments, state, events_config,
                                            KEEP_RECENT_TURNS[config.tier])
            for fp, (text, _) in decided.items():
                state.pinned[fp] = text
            if events_config.prune_tools:
                used = {s.tool_name for s in segments if s.kind == SegmentKind.TOOL_CALL}
                state.kept_tools = sorted(set(events_config.core_tools) | used)
            state.events += 1
            state.last_event_tokens = total
        else:
            decided = {}

        # Re-apply every remembered decision (identical text every turn = cache stays valid).
        on_arrival = {d["segment"]: d for d in report.trimmed}
        for seg in segments:
            fp = fingerprint(seg)
            if fp not in state.pinned or not isinstance(lookup(body, seg.path), str):
                continue
            text = state.pinned[fp]
            if new_body is None:
                new_body = copy.deepcopy(body)
            set_text(new_body, seg.path, text)
            after = count_tokens(text)
            how = decided[fp][1] if fp in decided else "history_pinned"
            earlier = on_arrival.pop(seg.id, None)
            if earlier is not None:  # the history decision replaces the on-arrival trim
                report.trimmed.remove(earlier)
                report.tokens_before -= earlier["before"]
                report.tokens_after -= earlier["after"]
            report.tokens_before += seg.token_count
            report.tokens_after += after
            report.trimmed.append({"segment": seg.id, "how": how, "tool": seg.tool_name,
                                   "before": seg.token_count, "after": after})

        if state.kept_tools is not None:
            keep = set(state.kept_tools) | {_forced_tool(body)}
            if new_body is None:
                new_body = copy.deepcopy(body)
            removed = _prune_tools(new_body, keep)
            if removed:
                report.tokens_before += removed
                report.trimmed.append({"segment": "tools", "how": "tools_pruned",
                                       "tool": "", "before": removed, "after": 0})
            elif new_body == body:
                new_body = None

        state.model = model or state.model
        state.last_seen = now
        store.put(key, state)
    return new_body, report, event
