"""The `tokencut` command (installed with the tokencut-cli package).

    tokencut [claude args...]   open Claude Code through TokenCut (runs setup the first time)
    tokencut setup              logins and one-time installs (--chatgpt / --no-chatgpt)
    tokencut status             is it running, what did it save
    tokencut logs [gateway|connector] [-n LINES]
    tokencut gateway            the gateway alone, in the foreground
    tokencut --version
"""

import argparse
import json
import os
import sys
import urllib.request
from collections import deque
from importlib.metadata import PackageNotFoundError, version

from tokencut import userconfig

COMMANDS = ("setup", "status", "logs", "gateway", "code")


def _version() -> str:
    try:
        return version("tokencut-cli")
    except PackageNotFoundError:  # running from a checkout without an install
        return "0+checkout"


def _gateway() -> int:
    import uvicorn

    from tokencut.gateway.app import create_app

    uvicorn.run(create_app(), host=os.environ.get("TOKENCUT_HOST", "127.0.0.1"),
                port=int(os.environ.get("TOKENCUT_PORT", "8787")))
    return 0


def _get(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return json.loads(response.read())
    except (OSError, ValueError):
        return None


def status_text(health: dict | None, stats: dict | None, gateway: str) -> str:
    if health is None:
        return f"TokenCut is not running ({gateway}). Start it with `tokencut`."
    features = [name for name in ("task_masking", "prune_tools", "ccr", "lazy_tools")
                if health.get(name)]
    models = "Claude + GPT" if health.get("gpt_upstream") else "Claude"
    on = f"; on: {', '.join(features)}" if features else ""
    lines = [f"TokenCut is running on {gateway} - mode {health['mode']}, tier {health['tier']}",
             f"  models: {models}{on}"]
    if stats:
        trimming = stats["trimming"]
        saved = trimming["tokens_saved_applied"] or trimming["tokens_saved_possible"]
        total = stats["tokens_total"]
        share = f" ({saved / total:.0%} of {total:,})" if total else ""
        billed = stats["billed_usage"]
        lines.append(f"  requests: {stats['requests']}, tokens removed: {saved:,}{share}")
        lines.append(f"  billed: input {billed['input_tokens']:,}, cache read "
                     f"{billed['cache_read_tokens']:,}, cache write "
                     f"{billed['cache_write_tokens']:,}, output {billed['output_tokens']:,}")
    lines.append(f"  live page: {gateway}/tokencut/")
    return "\n".join(lines)


def _status() -> int:
    port = os.environ.get("TOKENCUT_PORT", str(userconfig.load()["gateway"]["port"]))
    gateway = f"http://127.0.0.1:{port}"
    health = _get(f"{gateway}/tokencut/health")
    print(status_text(health, _get(f"{gateway}/tokencut/stats") if health else None, gateway))
    return 0 if health else 1


def _logs(which: str, lines: int) -> int:
    path = userconfig.home() / f"{which}.log"
    if not path.exists():
        print(f"no {which} log yet ({path})")
        return 1
    with path.open(encoding="utf-8", errors="replace") as handle:
        sys.stdout.writelines(deque(handle, maxlen=lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] == "code" or argv[0] not in (*COMMANDS, "-h", "--help", "--version"):
        from tokencut import launch  # every argument belongs to Claude Code

        return launch.main(["code", *(argv[1:] if argv[:1] == ["code"] else argv)])
    parser = argparse.ArgumentParser(prog="tokencut", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=f"tokencut {_version()}")
    commands = parser.add_subparsers(dest="command")
    setup = commands.add_parser("setup", help="logins and one-time installs")
    choice = setup.add_mutually_exclusive_group()
    choice.add_argument("--chatgpt", action="store_true", default=None,
                        help="add the ChatGPT plan without asking")
    choice.add_argument("--no-chatgpt", dest="chatgpt", action="store_false",
                        help="Claude only, without asking")
    commands.add_parser("status", help="is it running, what did it save")
    logs = commands.add_parser("logs", help="last lines of a process log")
    logs.add_argument("which", nargs="?", default="gateway", choices=("gateway", "connector"))
    logs.add_argument("-n", "--lines", type=int, default=40)
    commands.add_parser("gateway", help="run the gateway in the foreground")
    commands.add_parser("code", help="open Claude Code through TokenCut (the default)")
    args = parser.parse_args(argv)
    if args.command == "setup":
        from tokencut.wizard import run_setup

        return run_setup(chatgpt=args.chatgpt)
    if args.command == "status":
        return _status()
    if args.command == "logs":
        return _logs(args.which, args.lines)
    if args.command == "gateway":
        return _gateway()
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
