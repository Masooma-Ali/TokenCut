#!/bin/sh
# TokenCut installer (macOS / Linux):
#   curl -fsSL https://github.com/Masooma-Ali/TokenCut/releases/latest/download/install.sh | sh
# Installs uv (Astral's Python tool manager) if missing, then the tokencut command in its own
# environment, then runs `tokencut setup`. Re-run it to update.
# TOKENCUT_PACKAGE overrides what is installed (e.g. git+https://github.com/Masooma-Ali/TokenCut).
set -eu
pkg="${TOKENCUT_PACKAGE:-tokencut-proxy}"
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$PATH"
  export PATH
fi
if uv tool list 2>/dev/null | grep -q '^tokencut-proxy '; then
  uv tool upgrade tokencut-proxy
else
  uv tool install --python 3.12 "$pkg"
fi
uv tool update-shell >/dev/null 2>&1 || true
PATH="$(uv tool dir --bin):$PATH"
export PATH
# stdin is this script when piped from curl: answer setup's questions from the terminal.
if [ -r /dev/tty ]; then tokencut setup </dev/tty; else tokencut setup --no-chatgpt; fi
echo "Open a new terminal, go to your project folder and run: tokencut"
