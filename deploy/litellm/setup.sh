#!/usr/bin/env bash
# One-time LiteLLM setup for TokenCut (macOS / Linux). Same as setup.ps1.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tc="$HOME/.tokencut"
mkdir -p "$tc"
[ -x "$tc/litellm-venv/bin/python" ] || python3.11 -m venv "$tc/litellm-venv"
"$tc/litellm-venv/bin/python" -m pip install --quiet --upgrade pip
"$tc/litellm-venv/bin/python" -m pip install --quiet -r "$here/requirements.txt"
if [ ! -f "$tc/litellm_master_key" ]; then
  "$tc/litellm-venv/bin/python" -c "import secrets,pathlib,sys; pathlib.Path(sys.argv[1]).write_text('sk-'+secrets.token_hex(32))" "$tc/litellm_master_key"
  chmod 600 "$tc/litellm_master_key"
  echo "Created a private LiteLLM master key in $tc/litellm_master_key (never commit it)."
fi
echo "Setup done. Next: ChatGPT sign-in, then deploy/litellm/start.sh"
echo "  $tc/litellm-venv/bin/python -c \"from litellm.llms.chatgpt.authenticator import Authenticator; Authenticator().get_access_token()\""
