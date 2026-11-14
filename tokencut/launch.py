"""One-command demo launcher: TokenCut gateway + LiteLLM (ChatGPT plan) + Claude Code.

    python -m tokencut.launch code [extra claude args...]

Starts what is not already running, then opens Claude Code in the current folder wired to
the gateway. Inside Claude Code:
    /model sonnet          -> Claude Pro plan (Anthropic)
    /model gpt-6.1-sol     -> ChatGPT plan (via LiteLLM)
Live page: http://127.0.0.1:8787/tokencut/

One-time LiteLLM setup: deploy/litellm/setup.ps1 (or setup.sh), then login.ps1.
Settings (environment): TOKENCUT_LITELLM (litellm executable), TOKENCUT_LITELLM_CONFIG
(default deploy/litellm/config.yaml),
TOKENCUT_GPT_PORT (default 4000), plus every TOKENCUT_* gateway setting.
Processes started here are stopped when Claude Code exits.
"""

import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

GATEWAY_PORT = int(os.environ.get("TOKENCUT_PORT", "8787"))
GPT_PORT = int(os.environ.get("TOKENCUT_GPT_PORT", "4000"))
HOME = Path.home() / ".tokencut"
_BIN = "Scripts" if os.name == "nt" else "bin"
_EXE = ".exe" if os.name == "nt" else ""


def _up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status < 500
    except OSError:
        return False


def _wait(url: str, what: str, seconds: int) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if _up(url):
            return
        time.sleep(0.5)
    sys.exit(f"tokencut: {what} did not start within {seconds}s ({url})")


def _start(cmd: list[str], env: dict[str, str], log_name: str) -> subprocess.Popen:
    HOME.mkdir(parents=True, exist_ok=True)
    log = (HOME / log_name).open("ab")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def _litellm_command() -> list[str] | None:
    exe = os.environ.get("TOKENCUT_LITELLM") or str(HOME / "litellm-venv" / _BIN / f"litellm{_EXE}")
    repo_config = Path(__file__).resolve().parents[1] / "deploy" / "litellm" / "config.yaml"
    config = os.environ.get("TOKENCUT_LITELLM_CONFIG") or str(repo_config)
    if not (Path(exe).exists() and Path(config).exists()):
        return None
    return [exe, "--config", config, "--host", "127.0.0.1", "--port", str(GPT_PORT)]


def _master_key() -> str | None:
    """LiteLLM's master key: env var, else ~/.tokencut/litellm_master_key (never printed)."""
    key = os.environ.get("LITELLM_MASTER_KEY")
    path = HOME / "litellm_master_key"
    if not key and path.exists():
        key = path.read_text(encoding="utf-8").strip()
    return key or None


def main(argv: list[str]) -> int:
    if not argv or argv[0] != "code":
        print(__doc__)
        return 2
    started: list[subprocess.Popen] = []
    env = dict(os.environ)
    gpt_url = f"http://127.0.0.1:{GPT_PORT}"
    key = _master_key()
    try:
        # 1. LiteLLM for the ChatGPT plan (optional: without it only Claude is routed).
        gpt_on = _up(f"{gpt_url}/health/liveliness")
        if not gpt_on:
            cmd = _litellm_command()
            if cmd is None:
                print("tokencut: LiteLLM not set up; GPT route off "
                      "(run deploy/litellm/setup.ps1, see docs/DEMO_DUAL_PLAN.md)")
            else:
                print(f"tokencut: starting LiteLLM (ChatGPT plan) on {gpt_url} ...")
                # UTF-8: LiteLLM's banner crashes a cp1252 Windows console otherwise.
                lenv = {**env, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
                if key:
                    lenv["LITELLM_MASTER_KEY"] = key
                started.append(_start(cmd, lenv, "litellm.log"))
                _wait(f"{gpt_url}/health/liveliness", "LiteLLM", 120)
                gpt_on = True

        # 2. The gateway.
        gateway = f"http://127.0.0.1:{GATEWAY_PORT}"
        if _up(f"{gateway}/tokencut/health"):
            print(f"tokencut: gateway already running on {gateway} (its own settings apply)")
        else:
            genv = {
                **env,
                "TOKENCUT_MODE": env.get("TOKENCUT_MODE", "trim"),
                # Logs outside the user's repo; run this checkout's gateway code.
                "TOKENCUT_LOG_DIR": env.get("TOKENCUT_LOG_DIR", str(HOME / "logs")),
                "PYTHONPATH": os.pathsep.join(
                    filter(None, [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH")])
                ),
            }
            if gpt_on:
                genv.setdefault("TOKENCUT_GPT_UPSTREAM", gpt_url)
                if key:
                    genv.setdefault("TOKENCUT_GPT_KEY", key)
            print(f"tokencut: starting gateway on {gateway} ...")
            started.append(_start([sys.executable, "-m", "tokencut.gateway"], genv, "gateway.log"))
            _wait(f"{gateway}/tokencut/health", "gateway", 60)

        # 3. Claude Code, wired to the gateway.
        claude = shutil.which("claude")
        if claude is None:
            sys.exit("tokencut: Claude Code ('claude') not found on PATH")
        print(f"tokencut: live page {gateway}/tokencut/")
        print("tokencut: in Claude Code use /model sonnet (Claude plan) or "
              "/model gpt-6.1-sol (ChatGPT plan)\n")
        cenv = {**env, "ANTHROPIC_BASE_URL": f"{gateway}/anthropic"}
        return subprocess.call([claude, *argv[1:]], env=cenv)
    finally:
        for proc in reversed(started):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
