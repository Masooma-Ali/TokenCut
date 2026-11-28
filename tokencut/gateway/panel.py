"""Data for the team panel (the live page): system facts and per-request extras.

Yes/no answers and numbers only. Never a key, token, header, prompt or code: the Claude login
file is read for one field (the plan name), the ChatGPT login only for its existence. Every
section is built on its own and fails on its own (an error name instead of the section), so a
broken check never takes the page or the gateway down.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger(__name__)


def claude_login() -> dict:
    """Claude Code's login: whether there is one and which plan (pro / max / ... / api_key)."""
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            return {"logged_in": True, "plan": "api_key", "source": "ANTHROPIC_API_KEY"}
        path = Path.home() / ".claude" / ".credentials.json"
        oauth = json.loads(path.read_text(encoding="utf-8")).get("claudeAiOauth") or {}
        if oauth.get("accessToken"):
            return {"logged_in": True, "plan": str(oauth.get("subscriptionType") or "unknown"),
                    "source": "claude_code_login"}
    except (OSError, ValueError, AttributeError):
        pass
    return {"logged_in": False, "plan": None, "source": None}


def _host(url: str | None) -> str | None:
    """host:port of a URL (never a path, user or password)."""
    return urlparse(url).netloc.rsplit("@", 1)[-1] if url else None


def _package_version() -> str:
    try:
        return version("tokencut-proxy")
    except PackageNotFoundError:
        return "0+checkout"


def process(started: float) -> dict:
    return {"version": _package_version(), "python": sys.version.split()[0], "pid": os.getpid(),
            "uptime_s": round(time.time() - started),
            "host": os.environ.get("TOKENCUT_HOST", "127.0.0.1"),
            "port": int(os.environ.get("TOKENCUT_PORT", "8787"))}


def gateway(config) -> dict:
    return {"mode": config.mode, "tier": config.trim.tier.value,
            "min_trim_tokens": config.trim.min_tokens, "cache_events": config.cache_events,
            "cache_ttl_s": config.events.cache_ttl_seconds,
            "compact_at_tokens": config.events.compact_at_tokens,
            "prune_tools": config.events.prune_tools, "task_masking": config.events.task_masking,
            "ccr": config.virtual.ccr, "lazy_tools": config.virtual.lazy_tools,
            "system_note": config.trim.system_note, "prompt_brief": config.trim.prompt_brief,
            "brief_threshold": config.trim.brief_threshold,
            "key_required": bool(config.client_key),
            "log_dir": str(config.log_dir) if config.log_dir else None,
            "capture": config.capture}


def routing(config) -> dict:
    route = config.route
    return {"upstreams": {name: _host(url) for name, url in config.upstreams.items()},
            "gpt_upstream": _host(route.gpt_upstream), "gpt_prefixes": list(route.gpt_prefixes),
            "claude_via_litellm": route.claude_via_litellm, "gpt_key_set": bool(route.gpt_key)}


def ode(config, chooser) -> dict:
    cfg = config.ode
    out = {"mode": cfg.mode, "choosing": chooser is not None, "force_arm": cfg.force_arm,
           "explore": cfg.explore, "alpha": cfg.alpha, "restore_model": cfg.restore_model,
           "configured_arms": list(cfg.arms), "gpt_route": bool(config.route.gpt_upstream)}
    if chooser is not None:
        out.update(chooser.summary())
        out["prior"] = Path(chooser.prior).name
    return out


def logins(config) -> dict:
    from tokencut import litellm_proxy, userconfig

    saved = userconfig.load()
    return {"claude": claude_login(),
            "chatgpt": {"logged_in": litellm_proxy.chatgpt_logged_in(),
                        "enabled_in_config": bool(saved["chatgpt"]["enabled"]),
                        "models": saved["chatgpt"]["models"]},
            "tokencut_key": bool(userconfig.key("tokencut", create=False))}


def _ping(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health/liveliness", timeout=1) as response:
            return response.status < 500
    except OSError:
        return False


def installs(config, chooser) -> dict:
    from tokencut import litellm_proxy, userconfig

    upstream = config.route.gpt_upstream
    return {"tokencut": _package_version(),
            "config_file": userconfig.config_path().exists(),
            "claude_code_on_path": shutil.which("claude") is not None,
            "connector_installed": litellm_proxy.installed(),
            "connector_version": litellm_proxy.LITELLM_VERSION,
            "connector_reachable": _ping(upstream) if upstream else None,
            "prompt_model_loaded": config.router is not None,
            "ode_prior_loaded": chooser is not None}


def system_info(config, chooser, started: float) -> dict:
    """Everything the panel shows about the setup, each section built (and failing) alone."""
    out = {}
    for name, build in (("process", lambda: process(started)),
                        ("gateway", lambda: gateway(config)),
                        ("routing", lambda: routing(config)),
                        ("ode", lambda: ode(config, chooser)),
                        ("logins", lambda: logins(config)),
                        ("installs", lambda: installs(config, chooser))):
        try:
            out[name] = build()
        except Exception as exc:  # a broken check must not hide the other sections
            log.warning("panel section %s failed", name, exc_info=True)
            out[name] = {"error": type(exc).__name__}
    return out


def ode_view(record: dict) -> dict | None:
    """What ODE did to one request, compact: None if ODE did not touch it."""
    raw = record.get("ode")
    if not isinstance(raw, dict):
        return None
    choice = raw.get("choice") or {}
    numbers = raw.get("task_numbers") or {}
    arm_name = raw.get("arm") or choice.get("arm")
    return {"mode": raw.get("mode"), "arm": arm_name, "trim": raw.get("trim"),
            "decided": bool(choice.get("decided")), "previous": choice.get("previous"),
            "explored": bool(choice.get("explored")), "greedy": choice.get("greedy"),
            "requested": raw.get("model_requested"), "sent": raw.get("model_sent"),
            "skipped": raw.get("skipped"), "restored": raw.get("model_restored"),
            "task": numbers.get("task") or numbers.get("machine"), "p": numbers.get("p"),
            "scores": {a: s.get("total") for a, s in (choice.get("scores") or {}).items()},
            "error": raw.get("error")}


def cost_usd(record: dict) -> float | None:
    """List-price cost of one request from its billed usage; None if unknown."""
    try:
        from tokencut.ode.reward import model_for, request_cost, usage_from_dict

        usage = record.get("usage")
        model = (record.get("analysis") or {}).get("model") or (
            (record.get("ode") or {}).get("model_sent"))
        if not usage or not model:
            return None
        return round(request_cost(model_for(model), usage_from_dict(usage)), 6)
    except (ValueError, TypeError, KeyError):
        return None


def request_extras(record: dict) -> dict:
    """Panel fields for one request (numbers, yes/no and names only)."""
    analysis = record.get("analysis") or {}
    trim = analysis.get("trim") or {}
    return {"by_kind": analysis.get("tokens_by_kind") or {},
            "via": record.get("via"), "auth": record.get("auth"),
            "trim_ms": trim.get("trim_ms"), "analysis_ms": analysis.get("analysis_ms"),
            "ode": ode_view(record), "cost_usd": cost_usd(record),
            "error": record.get("analysis_error") or record.get("error")
            or record.get("fallback")}
