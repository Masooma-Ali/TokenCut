# One-time LiteLLM setup for TokenCut's cross-provider routing (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File deploy\litellm\setup.ps1
# Creates ~/.tokencut/litellm-venv (separate from the project's .venv), installs the pinned
# LiteLLM version and creates a private master key. Secrets never go into the repository.
$ErrorActionPreference = "Stop"
$tc = Join-Path $env:USERPROFILE ".tokencut"
$venv = Join-Path $tc "litellm-venv"
New-Item -ItemType Directory -Force $tc | Out-Null

if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    Write-Host "Creating $venv (Python 3.11) ..."
    & py -3.11 -m venv $venv
}
$py = Join-Path $venv "Scripts\python.exe"
& $py -m pip install --quiet --upgrade pip
& $py -m pip install --quiet -r (Join-Path $PSScriptRoot "requirements.txt")

$keyFile = Join-Path $tc "litellm_master_key"
if (-not (Test-Path $keyFile)) {
    & $py -c "import secrets,pathlib,sys; pathlib.Path(sys.argv[1]).write_text('sk-'+secrets.token_hex(32))" $keyFile
    Write-Host "Created a private LiteLLM master key in $keyFile (never commit it)."
}
Write-Host "Setup done. Next: deploy\litellm\login.ps1 (ChatGPT sign-in), then deploy\litellm\start.ps1"
