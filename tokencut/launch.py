"""`tokencut` / `tokencut code`: model connector + gateway + Claude Code, in one command.

    tokencut [claude args...]          (python -m tokencut.launch code [claude args...])

Reads ~/.tokencut/config.json (runs `tokencut setup` first if there is none), starts what is
not already running, then opens Claude Code in the current folder wired to the gateway with
the TokenCut key. The model connector (LiteLLM) stays hidden: it delivers Claude requests
(with Claude Code's own login) and, if set up, GPT requests (`/model gpt-6.1-sol`).
Every TOKENCUT_* variable already set in the environment overrides the config.
Live page: http://127.0.0.1:8787/tokencut/   Processes started here stop when Claude Code exits.
"""

import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from tokencut import litellm_proxy, userconfig


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
    log = userconfig.home() / f"{what}.log"
    sys.exit(f"tokencut: the {what} did not start within {seconds}s (see {log})")


def _start(cmd: list[str], env: dict[str, str], log_name: str) -> subprocess.Popen:
    userconfig.home().mkdir(parents=True, exist_ok=True)
    log = (userconfig.home() / log_name).open("ab")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def _start_connector(config: dict, env: dict[str, str],
                     started: list[subprocess.Popen]) -> str | None:
    """The connector's URL once it runs, or None (Claude then goes straight to Anthropic)."""
    if not (config["litellm"]["enabled"] and litellm_proxy.installed()):
        return None
    url = litellm_proxy.url(config)
    if not _up(f"{url}/health/liveliness"):
        started.append(_start(litellm_proxy.command(config), litellm_proxy.process_env(env),
                              "connector.log"))
        _wait(f"{url}/health/liveliness", "connector", 120)
    return url


def main(argv: list[str]) -> int:
    if not argv or argv[0] != "code":
        print(__doc__)
        return 2
    if not userconfig.config_path().exists():
        from tokencut.wizard import run_setup

        if run_setup() != 0:
            return 1
    config = userconfig.load()
    claude = shutil.which("claude")
    if claude is None:
        sys.exit("tokencut: Claude Code ('claude') not found on PATH; run `tokencut setup`")
    started: list[subprocess.Popen] = []
    env = dict(os.environ)
    try:
        connector = _start_connector(config, env, started)
        genv = userconfig.gateway_env(config, connector, env)
        # Run this package's gateway code, also from a checkout without an install.
        genv["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH")]))
        gateway = f"http://127.0.0.1:{genv['TOKENCUT_PORT']}"
        if _up(f"{gateway}/tokencut/health"):
            print(f"tokencut: already running on {gateway} (its own settings apply)")
        else:
            print("tokencut: starting ...")
            started.append(_start([sys.executable, "-m", "tokencut.gateway"], genv,
                                  "gateway.log"))
            _wait(f"{gateway}/tokencut/health", "gateway", 180)  # first start loads the router
        models = "Claude" + (" + GPT (/model " + config["chatgpt"]["models"][0] + ")"
                             if connector and config["chatgpt"]["enabled"] else "")
        print(f"tokencut: ready - {models}; savings at {gateway}/tokencut/\n")
        return subprocess.call([claude, *argv[1:]], env=userconfig.agent_env(config, env))
    finally:
        for proc in reversed(started):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
