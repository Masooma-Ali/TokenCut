"""A fixed system note that tells the model how to treat TokenCut's markers (prompt enhancement).

Measured harms it targets: after an `omitted` marker the model re-reads the whole file, or
re-runs a command to see cut log lines (live run L3). The note is the same text on every
request and sits at the end of the system prompt, never in a user message, so it is cached
with the rest of the prefix (~0.1x price after the first request). Only agent requests (two or
more client tools) get it; side requests (titles, summaries) are left alone.
"""

from tokencut.gateway.history import _tool_lists
from tokencut.gateway.virtual import MIN_CLIENT_TOOLS, _is_server_tool
from tokencut.tokens import count_tokens

SYSTEM_NOTE = (
    "TokenCut note: tool output in this conversation may be shortened to save tokens. A line "
    "like [TokenCut: lines A-B of path omitted ...] means those lines exist but were not sent; "
    "if you need them, read only lines A-B (for example Read with offset/limit, or "
    "sed -n 'A,Bp' path), not the whole file again. Shortened command output keeps the errors, "
    "failures and final summary: do not re-run a command only to see the omitted lines. Prefer "
    "reading files in line ranges, and keep test output short (for example pytest -q --tb=short)."
)
_SYSTEM_ROLES = ("system", "developer")


def add_system_note(provider: str, body: dict) -> dict | None:
    """Add SYSTEM_NOTE to an agent request's system prompt, in place. Report entry or None."""
    if not _is_agent_request(body):
        return None
    if provider == "anthropic":
        system = body.get("system")
        if isinstance(system, list):
            if any(isinstance(b, dict) and b.get("text") == SYSTEM_NOTE for b in system):
                return None
            system.append({"type": "text", "text": SYSTEM_NOTE})
        elif isinstance(system, str) and system:
            if SYSTEM_NOTE in system:
                return None
            body["system"] = f"{system}\n\n{SYSTEM_NOTE}"
        elif not system:
            body["system"] = SYSTEM_NOTE
        else:
            return None
    elif provider == "openai":
        messages = body.get("messages")
        if not isinstance(messages, list):
            return None  # not Chat Completions
        if any(isinstance(m, dict) and m.get("content") == SYSTEM_NOTE for m in messages):
            return None
        lead = 0
        while lead < len(messages) and isinstance(messages[lead], dict) \
                and messages[lead].get("role") in _SYSTEM_ROLES:
            lead += 1
        role = messages[0]["role"] if lead else "system"  # "developer" where the agent uses it
        messages.insert(lead, {"role": role, "content": SYSTEM_NOTE})
    elif provider == "gemini":
        key = "system_instruction" if "system_instruction" in body else "systemInstruction"
        system = body.setdefault(key, {})
        parts = system.setdefault("parts", []) if isinstance(system, dict) else None
        if not isinstance(parts, list) or {"text": SYSTEM_NOTE} in parts:
            return None
        parts.append({"text": SYSTEM_NOTE})
    else:
        return None
    return {"segment": "system", "how": "system_note", "tool": "", "before": 0,
            "after": count_tokens(SYSTEM_NOTE)}


def _is_agent_request(body: dict) -> bool:
    client_tools = sum(1 for tools, _ in _tool_lists(body) for t in tools
                       if isinstance(t, dict) and not _is_server_tool(t))
    return client_tools >= MIN_CLIENT_TOOLS
