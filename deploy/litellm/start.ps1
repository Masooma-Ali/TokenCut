# Start the LiteLLM proxy on 127.0.0.1:4000 with the repository's config.
# Leave this window open and do not click inside it (a click can pause a Windows console).
$ErrorActionPreference = "Stop"
$tc = Join-Path $env:USERPROFILE ".tokencut"
$env:PYTHONUTF8 = "1"          # LiteLLM's banner crashes a cp1252 console otherwise
$env:PYTHONIOENCODING = "utf-8"
$env:LITELLM_MASTER_KEY = (Get-Content (Join-Path $tc "litellm_master_key") -Raw).Trim()
& (Join-Path $tc "litellm-venv\Scripts\litellm.exe") --config (Join-Path $PSScriptRoot "config.yaml") --host 127.0.0.1 --port 4000
