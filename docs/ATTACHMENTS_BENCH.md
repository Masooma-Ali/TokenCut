# Attachment benchmark — how to run it and read the results

Branch: `test/attachments-litellm` (attachments work + LiteLLM dual-plan routing).
Tool: `eval/attachments_bench.py`. Everything it writes goes to `logs/attachments_bench/`
(gitignored: it contains the attachments themselves).

## 1. Build the sample items (free)

```bash
.venv\Scripts\python.exe -m eval.attachments_bench sample
```

Seven items, one per attachment rule, each with a known answer:

| Item | File | What TokenCut should do |
|---|---|---|
| big_screenshot | 1920×1080 screenshot | shrink to 1568×882 (Claude 4.7+, OpenAI patch models) |
| zoom_question | same screenshot, "zoom in … small text" | keep full size |
| small_screenshot | 1024×640 screenshot | nothing (already small) |
| chart_margins | 1600×1000 chart on a white page | cut the plain border, add a note |
| pdf_text_question | 5-page PDF, question about the text | pages 1, 2, 5 as text; 3 (chart), 4 (table) stay PDF |
| pdf_chart_question | same PDF, question about the chart | keep the whole PDF (intent check) |
| csv_middle_row | 600-row CSV, answer in row 318 | keep the table whole (never head / tail) |

Your own items: put files in a folder with an `items.jsonl` (one JSON object per line:
`id`, `file`, `question`, `answer`, `match` = `number` or `contains`, optional `expect`) and
pass `--items path/to/items.jsonl`.

## 2. Offline run: what TokenCut does, no model call (free)

```bash
.venv\Scripts\python.exe -m eval.attachments_bench offline --models claude-haiku-5-5,claude-sonnet-4-5,gpt-6-luna
```

Each item goes through the gateway's own code as a file the agent read with its Read tool.
Open `logs\attachments_bench\results\report.html`: per model and item it shows what TokenCut
did (sizes, crop box, which PDF pages became text), the estimated attachment tokens before →
after, the gateway time, and before / after copies of every attachment (thumbnails, PDFs,
text) so you can look at them.

## 3. Live run: real answers, tokens, cost, time (uses plan usage)

Close any running gateway first (the tool starts its own on port 8787, once per arm).

Claude (Pro plan):

```bash
.venv\Scripts\python.exe -m eval.attachments_bench live --model haiku
```

GPT through LiteLLM (ChatGPT plan): start LiteLLM as in `docs/TEAMMATE_SETUP.md` (window 1,
`deploy\litellm\start.ps1`), then in PowerShell:

```powershell
$env:TOKENCUT_GPT_UPSTREAM="http://127.0.0.1:4000"
$env:TOKENCUT_GPT_KEY=(Get-Content "$env:USERPROFILE\.tokencut\litellm_master_key" -Raw).Trim()
.venv\Scripts\python.exe -m eval.attachments_bench live --model gpt-6-luna --out logs\attachments_bench\results_gpt
```

Each item runs twice in a fresh Claude Code session: **A** = gateway in shadow mode (nothing
changed, everything measured), **B** = TokenCut on. Results: `live.csv`, `report.md`,
`report.html` in the output folder; Claude Code transcripts and the gateway's numbers-only
logs in `arm_A/` and `arm_B/`.

## 4. What each number means

| Column | Meaning |
|---|---|
| What TokenCut did | the attachment rule that acted: size before → after, crop box, PDF text pages / PDF pages |
| Attachment tokens (est.) | TokenCut's estimate from the provider's published formula (offline) |
| Page pictures removed | PDF pages sent as text instead of text + page picture (not counted as tokens: providers do not publish a page-picture cost) |
| Correct A / B | the answer matched the known answer without / with TokenCut |
| Input-equivalent tokens | billed by the provider: uncached input + 0.1 × cache reads + 1.25 × cache writes |
| Claude Code cost | Claude Code's own API-equivalent estimate; on a plan you pay a flat price, this shows the plan usage it stands for |
| Gateway ms | time TokenCut added (parsing, counting, trimming, image / PDF work) |
| Wall time | the whole Claude Code run, model included |

Notes: each item is a separate Claude Code session, so Claude Code's own system prompt and
tool definitions are in every request of both arms; the attachment's effect is the difference
between A and B. One run per item and arm, so small differences can be noise.

## 5. Real benchmarks (ChartQA, DocVQA, InfographicVQA, MMLongBench-Doc, WikiTableQuestions)

```bash
.venv\Scripts\python.exe -m eval.attachments_datasets --per-benchmark 10   # ~35 MB, only the sampled items
.venv\Scripts\python.exe -m eval.attachments_bench offline --bench all     # free, no model call
.venv\Scripts\python.exe -m eval.attachments_bench serve                   # results page: http://127.0.0.1:8790/
.venv\Scripts\python.exe -m eval.attachments_bench live --bench all --model haiku
.venv\Scripts\python.exe -m eval.attachments_bench live --bench all --model gpt-6-luna --limit 5   # LiteLLM on :4000
.venv\Scripts\python.exe -m eval.attachments_bench switch --bench chartqa,docvqa,mmlongbench --limit 1
```

Results: `logs/attachments_bench/real/` — `FINDINGS.md` (conclusions), `SUMMARY.md` / `index.html`
(all benchmarks), one folder per benchmark (`report.md`, `report.html`, `live.csv`, transcripts).
Live runs: a fresh gateway per run, A / B order alternating, a warm-up call when the prompt cache
may be cold, finished runs skipped on a rerun, a usage-limit reply stops the run (rerun later to
resume). Claude Code works in `D:\tokencut_bench_work` (outside the repo, so this project's
CLAUDE.md is not part of the benchmark requests); `TOKENCUT_BENCH_WORK` changes it.

## 6. Read-only snapshot of a real run (no setup needed)

`logs/attachments_bench/real/` is gitignored (it holds the attachments themselves), so pulling
this branch alone does not give you the results — you'd have to rerun section 5 yourself.

A static, read-only snapshot of one real run (the one `FINDINGS.md` above was written from) is
published here, with thumbnails instead of full files:

**<https://claude.ai/artifact/QTup77w4q6TkotBKUzR2vf>**

If you can't open it, ask Fatima to re-share it (artifacts are private until the owner shares
them). To see a live, editable copy instead of this snapshot, use option 2 or 3 above.
