"""The `tokencut` command (installed by `pip install tokencut`).

    tokencut --version
    tokencut code [claude args...]   gateway + LiteLLM + Claude Code (tokencut.launch)
    tokencut gateway                 the gateway alone, in the foreground

`setup`, `status` and `logs` follow in the next installation steps.
"""

import argparse
import os
import sys
from importlib.metadata import PackageNotFoundError, version


def _version() -> str:
    try:
        return version("tokencut")
    except PackageNotFoundError:  # running from a checkout without an install
        return "0+checkout"


def _gateway() -> int:
    import uvicorn

    from tokencut.gateway.app import create_app

    uvicorn.run(create_app(), host=os.environ.get("TOKENCUT_HOST", "127.0.0.1"),
                port=int(os.environ.get("TOKENCUT_PORT", "8787")))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="tokencut", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=f"tokencut {_version()}")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("code", help="open Claude Code through TokenCut", add_help=False)
    commands.add_parser("gateway", help="run the gateway in the foreground")
    if argv[:1] == ["code"]:  # every further argument belongs to Claude Code
        from tokencut import launch

        return launch.main(argv)
    args = parser.parse_args(argv)
    if args.command == "gateway":
        return _gateway()
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
