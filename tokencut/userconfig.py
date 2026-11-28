"""The installed tool's settings: ~/.tokencut/config.json (no secrets) and two private keys.

`tokencut setup` writes the file; `tokencut` (tokencut.launch) turns it into the gateway's
TOKENCUT_* environment. A TOKENCUT_* variable the user already set always wins, so every
gateway setting in CLAUDE.md / README keeps working. TOKENCUT_HOME moves the folder (tests).
"""

import copy
import json
import os
import secrets
from pathlib import Path

DEFAULTS: dict = {
    "version": 1,
    "gateway": {
        "port": 8787,
        "mode": "trim",
        "tier": "moderate",
        "system_note": True,
        "prompt_brief": True,
    },
    # ODE: picks the model (and trim level) per task. "on" = switches, "shadow" = only logs its
    # choice, "off" = never. restore_model: replies name the model Claude Code asked for.
    "ode": {"mode": "on", "restore_model": False},
    "litellm": {"enabled": False, "port": 8788},  # model connector; Claude + GPT go through it
    "chatgpt": {"enabled": False, "models": ["gpt-6.1-sol", "gpt-6-luna", "gpt-6-astra"]},
}
ODE_MODES = ("off", "shadow", "on")
KEY_FILES = {"tokencut": "tokencut_key", "litellm": "litellm_master_key"}


def home() -> Path:
    return Path(os.environ.get("TOKENCUT_HOME") or Path.home() / ".tokencut")


def config_path() -> Path:
    return home() / "config.json"


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(
            out.get(key), dict) else value
    return out


def load() -> dict:
    """The saved settings on top of the defaults (defaults alone if there is no file)."""
    path = config_path()
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _merge(DEFAULTS, saved)


def save(config: dict) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def key(name: str, create: bool = True) -> str | None:
    """A private random key from ~/.tokencut (owner-only file); created on first use."""
    path = home() / KEY_FILES[name]
    if path.exists():
        return path.read_text(encoding="utf-8").strip() or None
    if not create:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    value = ("tc-" if name == "tokencut" else "sk-") + secrets.token_hex(32)
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)
    return value


def ode_arms(config: dict, gpt_ready: bool) -> list[str] | None:
    """The arms ODE may choose, or None for the gateway's default (every arm but Opus).

    GPT arms need a running connector, a ChatGPT login and the arm's model in the user's
    ChatGPT model list; without one of them they are left out (a GPT request would fail)."""
    from tokencut.ode.arms import ARM_NAMES, GPT, arm

    gpt_ok = gpt_ready and GPT.model_id in config["chatgpt"]["models"]
    if gpt_ok:
        return None
    return [a for a in ARM_NAMES if not a.startswith("opus") and arm(a).model.provider != "openai"]


def gateway_env(config: dict, litellm_url: str | None, environ: dict[str, str],
                gpt_ready: bool | None = None) -> dict[str, str]:
    """Environment for the gateway process: config values, never overriding the user's own.

    gpt_ready: ChatGPT is usable (default: connector running and ChatGPT enabled in the config)."""
    gw = config["gateway"]
    if gpt_ready is None:
        gpt_ready = bool(litellm_url and config["chatgpt"]["enabled"])
    wanted = {
        "TOKENCUT_PORT": str(gw["port"]),
        "TOKENCUT_MODE": gw["mode"],
        "TOKENCUT_TIER": gw["tier"],
        "TOKENCUT_SYSTEM_NOTE": "1" if gw["system_note"] else "0",
        "TOKENCUT_PROMPT_BRIEF": "1" if gw["prompt_brief"] else "0",
        "TOKENCUT_LOG_DIR": str(home() / "logs"),
    }
    ode = config.get("ode", {})
    mode = ode.get("mode", "on")
    wanted["TOKENCUT_ODE"] = mode if mode in ODE_MODES else "off"  # a typo never stops the gateway
    if wanted["TOKENCUT_ODE"] != "off":
        arms = ode_arms(config, gpt_ready)
        if arms is not None:
            wanted["TOKENCUT_ODE_ARMS"] = ",".join(arms)
        if ode.get("restore_model"):
            wanted["TOKENCUT_ODE_RESTORE_MODEL"] = "1"
    tokencut_key = key("tokencut", create=False)
    if tokencut_key:
        wanted["TOKENCUT_KEY"] = tokencut_key
    if litellm_url:
        wanted["TOKENCUT_GPT_UPSTREAM"] = litellm_url
        wanted["TOKENCUT_CLAUDE_VIA_LITELLM"] = "1"
        litellm_key = key("litellm", create=False)
        if litellm_key:
            wanted["TOKENCUT_GPT_KEY"] = litellm_key
    return {**wanted, **environ}


def agent_env(config: dict, environ: dict[str, str]) -> dict[str, str]:
    """Environment for Claude Code: talk to the gateway and send the TokenCut key."""
    port = environ.get("TOKENCUT_PORT", str(config["gateway"]["port"]))
    out = {**environ, "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}/anthropic"}
    tokencut_key = key("tokencut", create=False)
    if tokencut_key:
        headers = [line for line in environ.get("ANTHROPIC_CUSTOM_HEADERS", "").splitlines()
                   if line.strip() and not line.lower().startswith("x-tokencut-key:")]
        out["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join([*headers, f"x-tokencut-key: {tokencut_key}"])
    return out
