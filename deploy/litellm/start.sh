#!/usr/bin/env bash
# Start the LiteLLM proxy on 127.0.0.1:4000 with the repository's config (macOS / Linux).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tc="$HOME/.tokencut"
LITELLM_MASTER_KEY="$(cat "$tc/litellm_master_key")" exec "$tc/litellm-venv/bin/litellm" \
  --config "$here/config.yaml" --host 127.0.0.1 --port 4000
