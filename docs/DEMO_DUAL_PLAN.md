# Demo: one launch, two subscriptions (Claude Pro + ChatGPT plan)

```
your repo ◄── edits / commands ── Claude Code (the harness, runs tools locally)
                                        │ every model request
                                        ▼
                              TokenCut gateway :8787  (trim → route → log)
                               │                         │
                 Claude model  ▼                         ▼  GPT model
              Anthropic (Claude Pro login)    LiteLLM :4000 (ChatGPT plan login)
```

The models never touch the repo: whichever model answers only *proposes* tool calls;
Claude Code runs them on your machine.

## One-time setup (each person, on their own machine)

Everything LiteLLM needs is in `deploy/litellm/` **except the secrets**, which every person
creates locally: their own ChatGPT login and their own LiteLLM master key.

| File | What it is | In git? |
|---|---|---|
| `deploy/litellm/config.yaml` | which model names go to which provider | yes |
| `deploy/litellm/requirements.txt` | pinned LiteLLM version (`1.109.1`, the tested one) | yes |
| `deploy/litellm/setup.ps1` / `setup.sh` | creates `~/.tokencut/litellm-venv`, installs LiteLLM, creates the master key | yes |
| `deploy/litellm/login.ps1` | ChatGPT device-code sign-in | yes |
| `deploy/litellm/start.ps1` / `start.sh`, `test.ps1` | start the proxy; check GPT models answer | yes |
| `~/.tokencut/litellm_master_key` | the proxy's password | **no, local only** |
| `~/.config/litellm/chatgpt/auth.json` | your ChatGPT login | **no, local only** |

Windows PowerShell, from the repository root:
```powershell
powershell -ExecutionPolicy Bypass -File deploy\litellm\setup.ps1   # once
powershell -ExecutionPolicy Bypass -File deploy\litellm\login.ps1   # once: sign in to YOUR ChatGPT plan
powershell -ExecutionPolicy Bypass -File deploy\litellm\start.ps1   # window 1, leave open
powershell -ExecutionPolicy Bypass -File deploy\litellm	est.ps1    # window 2: expect OK ... pong
```
If ChatGPT says device-code login is blocked, enable it in ChatGPT Settings → Security.
Remove from `config.yaml` any GPT model that `test.ps1` reports as not supported by your plan.

Claude Code must be logged in with your Claude plan as usual (`claude`, then `/login`).

Windows notes: LiteLLM needs `PYTHONUTF8=1` (start.ps1 sets it), and clicking inside a running
console window can pause the process. macOS / Linux: `setup.sh`, then the sign-in command it
prints, then `start.sh`.

## Run the demo

From a throwaway copy of a repo (never an important one):
```powershell
python -m tokencut.launch code
```
It starts LiteLLM and the gateway (trim mode) if they are not running, opens Claude Code
wired to the gateway, and stops what it started when Claude Code exits.

- `/model sonnet` (or opus / haiku) → Claude Pro plan
- `/model gpt-6.1-sol` → ChatGPT plan
- Live page: <http://127.0.0.1:8787/tokencut/> (savings, billed usage, cache reads per plan)

Logs: `~/.tokencut/gateway.log`, `~/.tokencut/litellm.log`, numbers-only records in
`~/.tokencut/logs/`.

### Manual run (three windows, what was used in the live test)

```powershell
# window 1: LiteLLM
powershell -ExecutionPolicy Bypass -File deploy\litellm\start.ps1
# window 2: gateway (from the repository root)
$env:TOKENCUT_MODE="trim"; $env:TOKENCUT_MIN_TRIM_TOKENS="300"; $env:TOKENCUT_GPT_UPSTREAM="http://127.0.0.1:4000"; $env:TOKENCUT_CLAUDE_VIA_LITELLM="1"; $env:TOKENCUT_GPT_KEY=(Get-Content "$env:USERPROFILE\.tokencut\litellm_master_key" -Raw).Trim(); python -m tokencut.gateway
# window 3: Claude Code in a throwaway repo
$env:ANTHROPIC_BASE_URL="http://127.0.0.1:8787/anthropic"; claude
```
Before closing Claude Code, `/model` back to a Claude model: `/model gpt-…` is saved as the
default and fails without the gateway.

### Live test results (2026-11-12, sample repo, Claude Pro + ChatGPT plan)

| Check | Result |
|---|---|
| Claude (Haiku 5.5) through LiteLLM | works; 94 % cache reads, ~1.2–1.5 s per step (same as direct) |
| GPT (`gpt-6.1-sol`) through LiteLLM | edits files and runs PowerShell via Claude Code's tools; 87 % cache reads, 4–12 s per step |
| GPT → Claude switch back | works (no format / thinking errors); first request still read 33k tokens from cache (prefix before the GPT turns) |
| Claude → GPT switch | first GPT request 0 cache reads (~31k re-sent), then cached |
| Trimming | 0 % on this tiny repo (files below the 300-token threshold) |
| Seen once | Anthropic `429 rate_limit_error` on Claude Code's 1-token startup request via LiteLLM (LiteLLM retried twice) |
| Seen | auto-mode command checks go to Sonnet, ~31k tokens uncached each, on the Claude plan |
| Seen | after a switch, Claude claimed GPT's edit as its own (history has no model attribution) |
| Seen | two ChatGPT cache misses mid-session (cache is best-effort) |

Not yet tested: switch back to Sonnet/Opus with extended thinking, trimming with large outputs,
fallback when LiteLLM is stopped.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `TOKENCUT_GPT_UPSTREAM` | off (launcher: `http://127.0.0.1:4000`) | where GPT-model requests go |
| `TOKENCUT_GPT_MODELS` | `gpt-,o3,o4,codex` | model-name prefixes routed to the GPT side |
| `TOKENCUT_GPT_KEY` | none | LiteLLM master key, if the proxy uses one |
| `TOKENCUT_CLAUDE_VIA_LITELLM` | off | `1` = Claude requests also go through LiteLLM (Claude login forwarded, proxy key in `x-litellm-api-key`); if LiteLLM is down they go straight to Anthropic |
| `TOKENCUT_LITELLM`, `TOKENCUT_LITELLM_CONFIG`, `TOKENCUT_GPT_PORT` | `~/.tokencut/litellm-venv/...`, `deploy/litellm/config.yaml`, `4000` | launcher only |

## Caveats

- The ChatGPT subscription backend is undocumented; it can change, and the terms for
  third-party clients are unclear. Demo use only.
- Every switch between Claude and GPT rewrites the prompt cache once; switch per task, not
  per turn.
- Claude Code's prompt and tools are tuned for Claude; GPT models may use them less smoothly.

## Variant: TokenCut decides, LiteLLM delivers (`TOKENCUT_CLAUDE_VIA_LITELLM=1`)

All `/v1/messages` requests go to LiteLLM, which routes by model name: `claude-*` to Anthropic
with the Claude Code login it receives (config has **no** Anthropic API key, so nothing can be
billed to an API account), `gpt-*` to the ChatGPT plan. Other Claude paths (token counting,
account calls) stay direct. Config entries: `model_name: "claude-*"` → `anthropic/claude-*`, and
`general_settings.forward_client_headers_to_llm_api: true` so Claude Code's `anthropic-beta`
flags reach Anthropic. Open question to measure: does LiteLLM keep the Claude request body
stable enough for prompt caching (compare cache-read share with the direct route)?
