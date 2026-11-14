"""Dual-plan demo routing: one Claude Code session, two subscriptions.

Claude Code sends every request in Anthropic format. Requests for a Claude model go to
Anthropic as usual (with the user's Claude plan login). Requests for a GPT model
(e.g. after `/model gpt-6.1-sol`) go to a local LiteLLM proxy instead, which translates
them to OpenAI's Responses format and calls the ChatGPT subscription backend with its own
ChatGPT login. Trimming happens before routing, so both routes are trimmed the same way.

Off unless TOKENCUT_GPT_UPSTREAM is set. Demo feature (model routing is ODE's job later).

With TOKENCUT_CLAUDE_VIA_LITELLM=1, Claude requests go through the same LiteLLM proxy
("TokenCut decides, LiteLLM delivers"): the Claude plan login stays in Authorization (LiteLLM
forwards it to Anthropic) and LiteLLM's own key travels in x-litellm-api-key. If LiteLLM is
unreachable the gateway sends the Claude request straight to Anthropic (fail open).
"""

import json
import os
from dataclasses import dataclass

DEFAULT_GPT_PREFIXES = ("gpt-", "o3", "o4", "codex")
# The Claude plan login must never be sent to the GPT side.
_CLAUDE_AUTH_HEADERS = {"authorization", "x-api-key"}


@dataclass(frozen=True)
class RouteConfig:
    gpt_upstream: str | None = None  # e.g. http://127.0.0.1:4000 (LiteLLM proxy)
    gpt_prefixes: tuple[str, ...] = DEFAULT_GPT_PREFIXES
    gpt_key: str | None = None  # LiteLLM master key, if the proxy uses one
    claude_via_litellm: bool = False  # send Claude requests through LiteLLM as well

    @classmethod
    def from_env(cls) -> "RouteConfig":
        upstream = os.environ.get("TOKENCUT_GPT_UPSTREAM", "").strip().rstrip("/")
        prefixes = os.environ.get("TOKENCUT_GPT_MODELS", "")
        return cls(
            gpt_upstream=upstream or None,
            gpt_prefixes=tuple(p.strip().lower() for p in prefixes.split(",") if p.strip())
            or DEFAULT_GPT_PREFIXES,
            gpt_key=os.environ.get("TOKENCUT_GPT_KEY") or None,
            claude_via_litellm=os.environ.get("TOKENCUT_CLAUDE_VIA_LITELLM", "") == "1",
        )


def model_from_body(body: bytes) -> str:
    """The request's model name, or "" for anything that is not a JSON object."""
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return ""
    return str(data.get("model", "")) if isinstance(data, dict) else ""


def pick_route(provider: str, model: str, config: RouteConfig) -> str:
    """"gpt" or "claude" for Claude Code traffic; other providers keep their own name."""
    if provider != "anthropic":
        return provider
    if config.gpt_upstream and model.lower().startswith(config.gpt_prefixes):
        return "gpt"
    return "claude"


def gpt_headers(headers: dict[str, str], config: RouteConfig) -> dict[str, str]:
    """Headers for the LiteLLM proxy: without the Claude login, with the proxy key if any."""
    out = {k: v for k, v in headers.items() if k.lower() not in _CLAUDE_AUTH_HEADERS}
    if config.gpt_key:
        out["authorization"] = f"Bearer {config.gpt_key}"
    return out


def claude_via_litellm(config: RouteConfig) -> bool:
    """Whether Claude inference requests are delivered by LiteLLM (needs its address)."""
    return bool(config.claude_via_litellm and config.gpt_upstream)


def claude_litellm_headers(headers: dict[str, str], config: RouteConfig) -> dict[str, str]:
    """Headers for a Claude request through LiteLLM: keep the Claude login, add the proxy key."""
    out = dict(headers)
    if config.gpt_key:
        out["x-litellm-api-key"] = f"Bearer {config.gpt_key}"
    return out


def _moved_note(block: dict) -> dict:
    name = str(block.get("title") or "the PDF")
    return {"type": "text", "text": (
        f"[TokenCut: {name} from this tool result follows after the tool results of this message "
        f"(the GPT route does not accept files inside tool results)]")}


def move_tool_documents(body: dict) -> tuple[dict, int]:
    """(body to send to the GPT route, number of documents moved).

    ChatGPT's backend rejects a file inside a tool output (400: "expected an image MIME type",
    what LiteLLM makes of an Anthropic `document` in a `tool_result`), but reads the same PDF in
    the user's message. So each document inside a tool result moves to the end of that same
    user message (after its tool results) and leaves a short note in its place. LiteLLM already
    does this for images. Same input -> same output, so it is identical on every turn (prompt
    cache). The input body is not changed."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, 0
    new_messages, moved = [], 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not (isinstance(content, list) and message.get("role") == "user"):
            new_messages.append(message)
            continue
        new_content, documents = [], []
        for block in content:
            inner = block.get("content") if isinstance(block, dict) else None
            if (isinstance(block, dict) and block.get("type") == "tool_result"
                    and isinstance(inner, list)
                    and any(isinstance(b, dict) and b.get("type") == "document" for b in inner)):
                kept = []
                for b in inner:
                    if isinstance(b, dict) and b.get("type") == "document":
                        documents.append(b)
                        kept.append(_moved_note(b))
                    else:
                        kept.append(b)
                block = {**block, "content": kept}
            new_content.append(block)
        if documents:
            moved += len(documents)
            message = {**message, "content": new_content + documents}
        new_messages.append(message)
    if not moved:
        return body, 0
    return {**body, "messages": new_messages}, moved
