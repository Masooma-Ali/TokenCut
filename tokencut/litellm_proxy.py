"""The hidden model connector: a private LiteLLM install that delivers Claude and GPT requests.

The user never sees LiteLLM: `tokencut setup` installs a pinned version into
~/.tokencut/litellm-venv (its dependencies would clash with TokenCut's), the config is generated
from ~/.tokencut/config.json, and `tokencut` runs it as a child process on 127.0.0.1.
Claude requests keep the user's Claude Code login (LiteLLM forwards it to Anthropic); GPT
requests use LiteLLM's ChatGPT plan login (~/.config/litellm/chatgpt/auth.json).
If the connector is missing or down, Claude requests go straight to Anthropic (fail open).
"""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from tokencut import userconfig

LITELLM_VERSION = "1.109.1"  # keep equal to deploy/litellm/requirements.txt (tested)
PYTHON_FOR_UV = "3.11"
_BIN, _EXE = ("Scripts", ".exe") if os.name == "nt" else ("bin", "")
CHATGPT_AUTH = Path.home() / ".config" / "litellm" / "chatgpt" / "auth.json"
Runner = Callable[..., subprocess.CompletedProcess]


def venv_dir() -> Path:
    return userconfig.home() / "litellm-venv"


def _python() -> Path:
    return venv_dir() / _BIN / f"python{_EXE}"


def executable() -> Path:
    override = os.environ.get("TOKENCUT_LITELLM")
    return Path(override) if override else venv_dir() / _BIN / f"litellm{_EXE}"


def _stamp() -> Path:
    return venv_dir() / "tokencut-litellm-version"


def installed() -> bool:
    """The pinned version is installed (a TokenCut update with a new pin reinstalls it)."""
    if os.environ.get("TOKENCUT_LITELLM"):
        return executable().exists()
    stamp = _stamp()
    return (executable().exists() and stamp.exists()
            and stamp.read_text(encoding="utf-8").strip() == LITELLM_VERSION)


def install_commands(uv: str | None) -> list[list[str]]:
    target = str(venv_dir())
    package = f"litellm[proxy]=={LITELLM_VERSION}"
    if uv:  # much faster, and brings its own Python if the machine has none
        return [[uv, "venv", "--quiet", "--allow-existing", "--python", PYTHON_FOR_UV, target],
                [uv, "pip", "install", "--quiet", "--python", str(_python()), package]]
    return [[sys.executable, "-m", "venv", target],
            [str(_python()), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
            [str(_python()), "-m", "pip", "install", "--quiet", package]]


def install(run: Runner = subprocess.run) -> bool:
    """Install the pinned connector; False (with the reason printed by the tool) on failure."""
    venv_dir().parent.mkdir(parents=True, exist_ok=True)
    for command in install_commands(shutil.which("uv")):
        if run(command).returncode != 0:
            return False
    _stamp().parent.mkdir(parents=True, exist_ok=True)
    _stamp().write_text(LITELLM_VERSION, encoding="utf-8")
    return True


def config_dict(config: dict) -> dict:
    """LiteLLM proxy config: Claude always (Claude Code login forwarded), GPT if enabled."""
    models = []
    if config["chatgpt"]["enabled"]:
        models += [{"model_name": name, "model_info": {"mode": "responses"},
                    "litellm_params": {"model": f"chatgpt/{name}"}}
                   for name in config["chatgpt"]["models"]]
    # Claude Code sends plain names like claude-haiku-5-5.
    models += [{"model_name": "claude-*", "litellm_params": {"model": "anthropic/claude-*"}},
               {"model_name": "anthropic/*", "litellm_params": {"model": "anthropic/*"}}]
    return {
        "model_list": models,
        # Claude Code sends Anthropic-only fields (thinking, cache_control, ...); drop what GPT
        # rejects.
        "litellm_settings": {"drop_params": True},
        "general_settings": {
            "master_key": "os.environ/LITELLM_MASTER_KEY",  # never the key itself
            # Forward anthropic-beta and x-* headers (Claude Code's beta flags, e.g. 1h cache).
            "forward_client_headers_to_llm_api": True,
        },
    }


def write_config(config: dict) -> Path:
    """Write the generated config (JSON is valid YAML, so no YAML library is needed)."""
    path = userconfig.home() / "litellm_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_dict(config), indent=2) + "\n", encoding="utf-8")
    return path


def url(config: dict) -> str:
    return f"http://127.0.0.1:{config['litellm']['port']}"


def command(config: dict) -> list[str]:
    config_file = os.environ.get("TOKENCUT_LITELLM_CONFIG") or str(write_config(config))
    return [str(executable()), "--config", config_file, "--host", "127.0.0.1",
            "--port", str(config["litellm"]["port"])]


def process_env(environ: dict[str, str]) -> dict[str, str]:
    # UTF-8: LiteLLM's banner crashes a cp1252 Windows console otherwise.
    env = {**environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    master = userconfig.key("litellm", create=False)
    if master:
        env["LITELLM_MASTER_KEY"] = master
    return env


def chatgpt_logged_in() -> bool:
    return CHATGPT_AUTH.exists()


def chatgpt_login(run: Runner = subprocess.run) -> bool:
    """ChatGPT device-code sign-in (prints a link and a code); the token stays in LiteLLM's file."""
    code = ("from litellm.llms.chatgpt.authenticator import Authenticator; "
            "Authenticator().get_access_token()")
    return run([str(_python()), "-c", code], env=process_env(dict(os.environ))).returncode == 0
