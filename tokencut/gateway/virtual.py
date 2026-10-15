"""Gateway-side tools: tools the gateway offers to the model and answers itself.

    tokencut_retrieve       return the exact original of an output TokenCut shortened (CCR)
    tokencut_enable_tools   switch on tool definitions that lazy tool loading hid

The agent does not know these tools. When the model calls one, the agent answers with an
error ("no such tool") in its next request; the gateway replaces that error with the real
answer before forwarding. The answer depends only on the call and the original content, so
it is the same on every later turn and the cached prefix stays byte-for-byte identical.

Lazy tool loading: every request carries only the core tools, the tools already used or
enabled in this conversation, and a short catalog of the rest (inside
tokencut_enable_tools). Decided from the request alone, so it survives gateway restarts.

Anthropic and OpenAI formats only (Gemini function calls have no ids to match answers).
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field

from tokencut.gateway.ccr import RETRIEVE_TOOL, CCRStore, ref_for, slice_lines
from tokencut.gateway.history import DEFAULT_CORE_TOOLS, _forced_tool, _tool_lists, _tool_name
from tokencut.tokens import count_tokens

ENABLE_TOOL = "tokencut_enable_tools"
VIRTUAL_TOOLS = (RETRIEVE_TOOL, ENABLE_TOOL)
PROVIDERS = ("anthropic", "openai")
MIN_ORIGINAL_CHARS = 500  # smaller outputs are never shortened, so never retrieved
CATALOG_SUMMARY_CHARS = 90
CATALOG_MAX_CHARS = 4_000  # past this, hidden tools are listed by name only



@dataclass
class VirtualConfig:
    ccr: bool = False  # TOKENCUT_CCR: shortened outputs can be fetched back (tokencut_retrieve)
    lazy_tools: bool = False  # TOKENCUT_LAZY_TOOLS: only core / used tools + a catalog
    core_tools: tuple[str, ...] = DEFAULT_CORE_TOOLS
    store: CCRStore = field(default_factory=CCRStore)


RETRIEVE_DESCRIPTION = (
    "Return the exact original text of a tool output that TokenCut shortened (it ends with "
    "'[TokenCut ref tc_...]'). Pass start_line / end_line (inclusive, numbered as in the "
    "original output or in the omitted-lines note) to fetch only the part you need. Use this "
    "instead of reading the file again or re-running the command."
)
RETRIEVE_SCHEMA = {
    "type": "object",
    "properties": {
        "ref": {"type": "string", "description": "The reference, e.g. tc_1a2b3c4d5e6f"},
        "start_line": {"type": "integer", "description": "First line to return (optional)"},
        "end_line": {"type": "integer", "description": "Last line to return (optional)"},
    },
    "required": ["ref"],
}
ENABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "names": {"type": "array", "items": {"type": "string"},
                  "description": "Names of the tools to switch on"},
    },
    "required": ["names"],
}


# ---------- reading calls and answers from a raw body ----------


def _calls(provider: str, body: dict) -> Iterator[tuple[str, str, dict]]:
    """(call id, tool name, parsed input) for every tool call in the request."""
    for message in body.get("messages", []):
        if not isinstance(message, dict):
            continue
        if provider == "anthropic":
            content = message.get("content")
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    data = block.get("input")
                    yield str(block.get("id", "")), str(block.get("name", "")), \
                        data if isinstance(data, dict) else {}
        else:
            for call in message.get("tool_calls") or []:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                try:
                    data = json.loads(function.get("arguments") or "{}")
                except (TypeError, ValueError):
                    data = {}
                yield str(call.get("id", "")), str(function.get("name", "")), \
                    data if isinstance(data, dict) else {}


def _result_texts(provider: str, body: dict) -> Iterator[str]:
    """Text of every tool result in the request."""
    for message in body.get("messages", []):
        if not isinstance(message, dict):
            continue
        if provider == "anthropic":
            content = message.get("content")
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                inner = block.get("content")
                if isinstance(inner, str):
                    yield inner
                elif isinstance(inner, list):
                    for sub in inner:
                        if isinstance(sub, dict) and sub.get("type") == "text":
                            yield str(sub.get("text", ""))
        elif message.get("role") in ("tool", "function") and isinstance(message.get("content"), str):
            yield message["content"]


def _set_answer(provider: str, body: dict, call_id: str, text: str) -> bool:
    """Replace the agent's answer to call `call_id` with `text`. False if there is none yet."""
    for message in body.get("messages", []):
        if not isinstance(message, dict):
            continue
        if provider == "anthropic":
            content = message.get("content")
            for block in content if isinstance(content, list) else []:
                if (isinstance(block, dict) and block.get("type") == "tool_result"
                        and block.get("tool_use_id") == call_id):
                    block["content"] = text
                    block.pop("is_error", None)
                    return True
        elif message.get("role") == "tool" and message.get("tool_call_id") == call_id:
            message["content"] = text
            return True
    return False


# ---------- answering the gateway's own tools ----------


def _retrieve_answer(data: dict, originals: dict[str, str], store: CCRStore | None) -> str:
    ref = str(data.get("ref", "")).strip()
    original = originals.get(ref) or (store.get(ref) if store is not None else None)
    if original is None:
        return (f"[TokenCut: {ref or 'this reference'} is no longer available; read the file "
                "again or re-run the command]")

    def as_int(key: str) -> int | None:
        try:
            return int(data[key]) if data.get(key) is not None else None
        except (TypeError, ValueError):
            return None

    text, lo, hi = slice_lines(original, as_int("start_line"), as_int("end_line"))
    if not text:
        return f"[TokenCut: {ref} has no lines in that range]"
    return f"[TokenCut: original of {ref}, lines {lo}-{hi}]\n{text}"


def _enable_answer(data: dict, known: set[str]) -> str:
    names = [str(n) for n in data.get("names") or [] if isinstance(n, str | int)]
    found = [n for n in names if n in known]
    missing = [n for n in names if n not in known]
    parts = []
    if found:
        parts.append(f"enabled {', '.join(found)}; call them directly from now on")
    if missing:
        parts.append(f"no tool named {', '.join(missing)}")
    return f"[TokenCut: {'; '.join(parts) or 'no tools named'}]"


def resolve(provider: str, body: dict, store: CCRStore | None) -> list[dict]:
    """Answer the gateway's own tool calls in place. Returns report entries."""
    if provider not in PROVIDERS:
        return []
    calls = [(cid, name, data) for cid, name, data in _calls(provider, body) if name in VIRTUAL_TOOLS]
    if not calls:
        return []
    originals: dict[str, str] = {}
    if any(name == RETRIEVE_TOOL for _, name, _ in calls):
        for text in _result_texts(provider, body):
            if len(text) >= MIN_ORIGINAL_CHARS:
                originals.setdefault(ref_for(text), text)
    known = {_tool_name(t, how) for tools, how in _tool_lists(body) for t in tools}
    entries = []
    for call_id, name, data in calls:
        answer = (_retrieve_answer(data, originals, store) if name == RETRIEVE_TOOL
                  else _enable_answer(data, known))
        if _set_answer(provider, body, call_id, answer):
            how = "ccr_retrieved" if name == RETRIEVE_TOOL else "tools_enabled"
            entries.append({"segment": call_id, "how": how, "tool": name,
                            "before": 0, "after": count_tokens(answer)})
    return entries


# ---------- tool definitions sent upstream ----------


def _summary(tool: dict, how: str) -> str:
    desc = tool.get("description") or (tool.get("function") or {}).get("description") or ""
    first = str(desc).strip().split("\n")[0].split(". ")[0]
    return first[:CATALOG_SUMMARY_CHARS]


def _definition(provider: str, name: str, description: str, schema: dict) -> dict:
    if provider == "openai":
        return {"type": "function",
                "function": {"name": name, "description": description, "parameters": schema}}
    return {"name": name, "description": description, "input_schema": schema}


def _catalog(hidden: list[tuple[str, str]]) -> str:
    head = ("Switch on tools that are hidden to keep requests small. Call this with their names "
            "when you need one; afterwards call them directly. Hidden tools:")
    lines, used = [], len(head)
    for i, (name, summary) in enumerate(hidden):
        line = f"\n- {name}: {summary}" if summary else f"\n- {name}"
        if used + len(line) > CATALOG_MAX_CHARS:
            lines.append("\n- also: " + ", ".join(n for n, _ in hidden[i:]))
            break
        lines.append(line)
        used += len(line)
    return head + "".join(lines)


def finish_tools(provider: str, body: dict, ccr: bool, lazy: bool,
                 core_tools: tuple[str, ...]) -> list[dict]:
    """Lazy-prune the tool list and add the gateway's tools, in place. Report entries."""
    if provider not in PROVIDERS or not (ccr or lazy):
        return []
    lists = _tool_lists(body)
    if len(lists) != 1 or lists[0][0] is not body.get("tools"):
        return []  # no tools (plain chat), or a legacy "functions" list: leave alone
    tools, how = lists[0]
    entries = []
    if lazy:
        called = {name for _, name, _ in _calls(provider, body)}
        enabled = {str(n) for _, name, data in _calls(provider, body) if name == ENABLE_TOOL
                   for n in data.get("names") or []}
        keep = set(core_tools) | called | enabled | {_forced_tool(body)}
        kept, hidden, removed, cache_mark = [], [], 0, None
        for tool in tools:
            name = _tool_name(tool, how)
            if not name or name in keep or name in VIRTUAL_TOOLS:
                kept.append(tool)
                continue
            hidden.append((name, _summary(tool, how)))
            removed += count_tokens(json.dumps(tool, ensure_ascii=False, sort_keys=True))
            cache_mark = tool.get("cache_control", cache_mark)
        if hidden:
            # Keep the provider's cache breakpoint if it sat on a hidden tool.
            if cache_mark is not None and kept and not any("cache_control" in t for t in kept):
                kept[-1] = {**kept[-1], "cache_control": cache_mark}
            tools[:] = kept
            catalog = _definition(provider, ENABLE_TOOL, _catalog(hidden), ENABLE_SCHEMA)
            tools.append(catalog)
            entries.append({"segment": "tools", "how": "tools_lazy", "tool": "",
                            "before": removed, "after": count_tokens(json.dumps(catalog))})
    if ccr:
        retrieve = _definition(provider, RETRIEVE_TOOL, RETRIEVE_DESCRIPTION, RETRIEVE_SCHEMA)
        tools.append(retrieve)
        entries.append({"segment": "tools", "how": "ccr_tool", "tool": "",
                        "before": 0, "after": count_tokens(json.dumps(retrieve))})
    return entries


def offers_tools(provider: str, body: dict) -> bool:
    """Will finish_tools be able to add the gateway's tools to this request?"""
    lists = _tool_lists(body)
    return provider in PROVIDERS and len(lists) == 1 and lists[0][0] is body.get("tools")
