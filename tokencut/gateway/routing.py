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
