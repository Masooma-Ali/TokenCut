# Teammate setup: TokenCut with your Claude plan + ChatGPT plan

Steps for anyone who pulls `feature/dual-plan-demo` and wants to run the dual-plan demo on
their own laptop (Windows PowerShell). You use **your own** subscriptions; nobody else's
logins or keys are in the repository.

**Time:** ~15 minutes the first time, ~1 minute afterwards.

## What you need first

- Git, and **Python 3.11** (`py -3.11 --version` should work)
- **Claude Code** installed and logged in with your Claude plan: run `claude`, type `/login`,
  choose your Claude account, then `/exit`
- A **ChatGPT plan that includes Codex** (Plus, Pro or Business)
- **Poppler** (`pdftoppm`), for PDFs: see "Poppler for PDFs" below. Without it, Claude Code's
  Read tool fails whenever the model asks for PDF pages (`pages: "1-20"`), with or without
  TokenCut. It is a program, not a Python package, so `pip install -r requirements.txt`
  cannot install it.

### Poppler for PDFs, one time (~44 MB download, ~120 MB unpacked)

1. Download the latest `Release-*.zip` from
   https://github.com/oschwartz10612/poppler-windows/releases (the standard Windows build).
2. Unzip it to a folder of your choice, e.g. `D:\tools\poppler`.
3. Add its `Library\bin` folder to your **user** PATH, then open a new terminal:
   ```powershell
   $bin = "D:\tools\poppler\poppler-26.09.0\Library\bin"   # adjust to your folder
   [Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";$bin", "User")
   ```
   Or, for the current terminal only: `$env:Path += ";$bin"`.
4. Check: `pdftoppm -v` prints a version. Start Claude Code from a terminal where this works.

macOS: `brew install poppler`. Linux: `sudo apt-get install poppler-utils`.

## 1. Get the branch

From your TokenCut folder:
```powershell
git fetch origin
git checkout feature/dual-plan-demo
git pull
```

## 2. Project environment (skip if you already have `.venv`)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: all tests pass except `test_repoindex::test_touching_a_file_without_changing_it_reparses_nothing`
(a known failure that also happens on `main`).

## 3. LiteLLM, one time

LiteLLM is the translator to your ChatGPT plan. It is installed **outside** the project
(in `%USERPROFILE%\.tokencut\litellm-venv`), so the project's packages stay unchanged.

```powershell
powershell -ExecutionPolicy Bypass -File deploy\litellm\setup.ps1
```
This installs LiteLLM 1.109.1 and creates your private LiteLLM key in
`%USERPROFILE%\.tokencut\litellm_master_key`. Never share or commit that file.

## 4. Sign in to your ChatGPT plan, one time

```powershell
powershell -ExecutionPolicy Bypass -File deploy\litellm\login.ps1
```
It prints a link and a code. Open the link, sign in to **your** ChatGPT account, enter the
code. Wait for `Login OK`.
If ChatGPT says device-code login is not allowed: ChatGPT → Settings → Security → enable it,
then run the script again.

## 5. Start LiteLLM and check it (every time you work)

**Window 1**, leave it open:
```powershell
powershell -ExecutionPolicy Bypass -File deploy\litellm\start.ps1
```
Ready when it prints `Uvicorn running on http://127.0.0.1:4000`.
**Do not click inside this window** while it runs; on Windows a click can freeze it.

**Window 2**, first time only, to check your GPT models:
```powershell
powershell -ExecutionPolicy Bypass -File deploy\litellm\test.ps1
```
Expect `OK  gpt-6.1-sol -> pong` (and the same for the others). If a model says
*"not supported when using Codex with a ChatGPT account"*, your plan doesn't include it:
delete its 5 lines from `deploy\litellm\config.yaml` and restart window 1.

## 6. Start the TokenCut gateway (every time)

**Window 2**, from the TokenCut folder:
```powershell
$env:TOKENCUT_MODE="trim"; $env:TOKENCUT_MIN_TRIM_TOKENS="300"; $env:TOKENCUT_GPT_UPSTREAM="http://127.0.0.1:4000"; $env:TOKENCUT_CLAUDE_VIA_LITELLM="1"; $env:TOKENCUT_GPT_KEY=(Get-Content "$env:USERPROFILE\.tokencut\litellm_master_key" -Raw).Trim(); .\.venv\Scripts\python.exe -m tokencut.gateway
```
Open <http://127.0.0.1:8787/tokencut/> in a browser: it should say **GPT route on**.

## 7. Make a throwaway practice repo (first time, or for a fresh start)

**Window 3** (replace `D:\TokenCut` with your TokenCut folder):
```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\tokencut-playground" -ErrorAction SilentlyContinue
Copy-Item -Recurse D:\TokenCut\fixtures\sample_repo "$env:USERPROFILE\tokencut-playground"
cd "$env:USERPROFILE\tokencut-playground"; git init -q; git add -A; git commit -qm start
```
Never point this at an important repository.

## 8. Run Claude Code through TokenCut

**Window 3**:
```powershell
$env:ANTHROPIC_BASE_URL="http://127.0.0.1:8787/anthropic"; claude
```
Then try:
1. `/model haiku` → *Read shop/cart.py and explain Cart.total() in 3 lines* (your Claude plan)
2. `/model gpt-6.1-sol` → *Add a method is_empty() to Cart and verify it with a python -c command* (your ChatGPT plan)
3. `/model haiku` → *Review the changes made to shop/cart.py and list any problems* (back to Claude)

Watch the live page: each request appears under **Claude** or **GPT / Codex**, with cache reads
and anything TokenCut trimmed.

## 9. When you finish

1. In Claude Code: `/model` and pick a Claude model. (`/model gpt-…` is saved as your
   default and fails next time if you start Claude Code without the gateway.)
2. `/exit`, then Ctrl+C in window 2 (gateway) and window 1 (LiteLLM).

## Shortcut (not yet live-tested)

After steps 3–4, from the practice repo, one command starts LiteLLM and the gateway, opens
Claude Code, and stops everything when you exit (replace `D:\TokenCut`):
```powershell
$env:PYTHONPATH="D:\TokenCut"; $env:TOKENCUT_CLAUDE_VIA_LITELLM="1"; & D:\TokenCut\.venv\Scripts\python.exe -m tokencut.launch code
```
Logs then go to `%USERPROFILE%\.tokencut\` (`gateway.log`, `litellm.log`). Steps 5–8 are the
tested way; use them if the shortcut misbehaves.

## Troubleshooting

| You see | Cause | Fix |
|---|---|---|
| `UnicodeEncodeError … charmap` when LiteLLM starts | Windows console encoding | Start it with `start.ps1` (sets UTF-8) |
| `refused to start: no master key is set` | `setup.ps1` not run | Run step 3 |
| `test.ps1` says *The operation has timed out*, LiteLLM window silent | Console frozen by a click | Click the LiteLLM window, press Esc; or restart it |
| `There are no healthy deployments` | LiteLLM started before ChatGPT sign-in | Step 4, then restart window 1 |
| `ChatGPT refresh token failed, re-login required` | ChatGPT login expired | Run step 4 again |
| `OAuth session expired` / `Failed to authenticate` in Claude Code | Claude Code logged out | `/login` in Claude Code |
| `429 rate_limit_error` | Claude plan limit / short rate limit | Wait, or switch to `/model gpt-6.1-sol` |
| Claude Code asks before every `python -c` | Your `blockReadsOutsideWorkingDirectories` setting | Approve with Yes (it is a safety prompt, not an error) |
| `address already in use` on port 4000 or 8787 | An old window is still running | Close it (Ctrl+C) or end the old `litellm.exe` / python process |
| *Claude Code doesn't recognize the model "gpt-6.1-sol"* | Claude Code doesn't know GPT's context size | Harmless warning |
| `pdftoppm is not installed` when the agent reads a PDF | Poppler missing; Claude Code needs it for PDF page ranges | Install Poppler ("Poppler for PDFs" above), restart the terminal and Claude Code |

## Good to know

- Switching between Claude and GPT re-sends the conversation once (prompt cache is per
  provider). Switch per task, not every message.
- The ChatGPT subscription backend is unofficial and may change; demo use only.
- More detail and the measured results: `docs/DEMO_DUAL_PLAN.md`.
