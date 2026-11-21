"""`tokencut setup`: first-run questions and one-time installs.

1. Claude Code must be installed (the user installs it; we print the official command).
2. Ask whether to add the ChatGPT plan (GPT models in Claude Code via `/model gpt-…`).
3. Create the private keys, install the hidden model connector (LiteLLM), sign in to ChatGPT.
4. Download the prompt router model once (task briefs), so the first start is fast.
5. Save ~/.tokencut/config.json.
Claude itself needs no step here: Claude Code asks for its own login on first start, and that
login travels through TokenCut unchanged. Every failure leaves a working Claude-only setup.
"""

import os
import shutil
from collections.abc import Callable

from tokencut import litellm_proxy, userconfig

CLAUDE_INSTALL = {
    "posix": "curl -fsSL https://claude.ai/install.sh | bash",
    "nt": "irm https://claude.ai/install.ps1 | iex",
}


def _yes(answer: str, default: bool) -> bool:
    answer = answer.strip().lower()
    return default if not answer else answer in ("y", "yes")


def _download_router(out: Callable[[str], None]) -> bool:
    from tokencut.gateway.task_router import load_router

    out("Downloading the prompt model (about 135 MB, one time) ...")
    return load_router() is not None


def run_setup(
    chatgpt: bool | None = None,
    ask: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
    which: Callable[[str], str | None] = shutil.which,
    os_name: str | None = None,
    install_connector: Callable[[], bool] = litellm_proxy.install,
    connector_ready: Callable[[], bool] = litellm_proxy.installed,
    chatgpt_ready: Callable[[], bool] = litellm_proxy.chatgpt_logged_in,
    chatgpt_login: Callable[[], bool] = litellm_proxy.chatgpt_login,
    download_router: Callable[[Callable[[str], None]], bool] = _download_router,
) -> int:
    if which("claude") is None:
        out("TokenCut works inside Claude Code, which is not installed yet. Install it with:")
        out(f"    {CLAUDE_INSTALL.get(os_name or os.name, CLAUDE_INSTALL['posix'])}")
        out("then run `tokencut setup` again.")
        return 1
    config = userconfig.load()
    if chatgpt is None:
        chatgpt = _yes(ask("Also use your ChatGPT plan for GPT models? [y/N] "), False)

    userconfig.key("tokencut")
    userconfig.key("litellm")

    ready = connector_ready()
    if not ready:
        out("Installing the model connector (one time, 1-3 minutes) ...")
        ready = install_connector()
        if not ready:
            out("Could not install the model connector; Claude will connect directly, "
                "GPT models are off. Run `tokencut setup` again to retry.")
    config["litellm"]["enabled"] = ready

    config["chatgpt"]["enabled"] = False
    if chatgpt and ready:
        if not chatgpt_ready():
            out("ChatGPT sign-in: open the link below, sign in, and enter the code.")
            if not chatgpt_login():
                out("ChatGPT sign-in did not finish; GPT models are off for now.")
        config["chatgpt"]["enabled"] = chatgpt_ready()
    elif chatgpt:
        out("GPT models need the model connector; they stay off for now.")

    if config["gateway"]["prompt_brief"] and not download_router(out):
        out("Prompt model unavailable; task briefs are off (everything else works).")
        config["gateway"]["prompt_brief"] = False

    if ready:
        litellm_proxy.write_config(config)
    path = userconfig.save(config)
    models = "Claude" + (" + ChatGPT (" + ", ".join(config["chatgpt"]["models"]) + ")"
                         if config["chatgpt"]["enabled"] else "")
    out(f"\nTokenCut is ready ({path}).")
    out(f"  Models: {models}")
    out(f"  Trimming: {config['gateway']['mode']} ({config['gateway']['tier']}), "
        f"system note: {'on' if config['gateway']['system_note'] else 'off'}, "
        f"task briefs: {'on' if config['gateway']['prompt_brief'] else 'off'}")
    out("Run `tokencut` in your project folder to open Claude Code through TokenCut.")
    return 0
