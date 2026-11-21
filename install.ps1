# TokenCut installer (Windows PowerShell):
#   irm https://github.com/Masooma-Ali/TokenCut/releases/latest/download/install.ps1 | iex
# Installs uv (Astral's Python tool manager) if missing, then the tokencut command in its own
# environment, then runs `tokencut setup`. Re-run it to update.
# $env:TOKENCUT_PACKAGE overrides what is installed (e.g. git+https://github.com/Masooma-Ali/TokenCut).
$ErrorActionPreference = "Stop"
$pkg = if ($env:TOKENCUT_PACKAGE) { $env:TOKENCUT_PACKAGE } else { "tokencut-proxy" }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv ..."
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
if ((uv tool list 2>$null) -match '^tokencut-proxy ') {
    uv tool upgrade tokencut-proxy
} else {
    uv tool install --python 3.12 $pkg
}
uv tool update-shell *> $null
$env:Path = "$(uv tool dir --bin);$env:Path"
tokencut setup
Write-Host "Open a new terminal, go to your project folder and run: tokencut"
