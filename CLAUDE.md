# CLAUDE.md — TokenCut project guide for Claude Code

Read this whole file before doing anything in this repository. It tells you what TokenCut
is, what is already built, what the team has decided, **what your job is in the current
phase**, how to run everything, and how to report results.

---

## 0. Your job in this phase: TEST, COMPARE, MEASURE, REPORT

The team is in a **testing and evaluation phase**. Do **not** build new features, do not
start ODE / model routing, and do not prepare mid-evaluation slides on your own. The goal is
an honest, reproducible answer to:

> **Where does TokenCut's context trimming stand?** How much does it save, does the agent
> still work correctly, how does it compare with baselines, what is fast or slow, and what
> breaks?

Concretely you will (details in section 6):

1. Reproduce the existing results (tests, benchmark) on your machine.
2. Run the **offline** benchmark and extend the analysis (per repo, per budget, latency).
3. Run **live** sessions through the gateway with a real coding agent, in a fixed test
   matrix (shadow vs trim, tiers, cache events, tool pruning, with vs without TokenCut).
4. Record every number, every failure, every odd agent behaviour.
5. Write the results into `docs/RESULTS_REPORT.md` (template in section 8) and keep it up
   to date. Honest negative results are as valuable as positive ones.

**Update 2026-10-12:** at a team member's request, two improvements found by this phase's
measurements were implemented on branch `trimming/researchpapers_methodology` — **R8** (diffs
and "full" requests never trimmed) and **R10** (opt-in task masking, `TOKENCUT_TASK_MASK=1`);
see `docs/DECISIONS.md` and `docs/RESULTS_REPORT.md` §4.6. Other recommendations stay
recommendations until the team decides.

Small fixes are allowed **only** when a test or measurement exposes a real bug. Then: write
a failing test first, fix, keep all tests green, log it in `docs/DECISIONS.md`, and mention
it in the report. Anything bigger → write it down as a recommendation, do not build it.

Always tell the user what you measured, how, and what the numbers mean in plain words.
Never present a result you did not actually measure. If a run fails, say so with the output.

---

## 1. What TokenCut is

TokenCut is a final-year project (FAST NUCES Islamabad): a **gateway** that sits between an
AI coding agent (Claude Code, Cline, Gemini CLI, any OpenAI-compatible client) and the model
provider (Anthropic, OpenAI, Gemini). It removes content the model does not need from each
request — mainly **tool output** (file contents, test logs, search results), and at the
right moments old history and unused tool definitions — so every request costs less,
**without breaking the agent, its edits, or the provider's prompt cache.**

Later (after the mid evaluation, NOT now) a contextual bandit called **ODE** will choose the
context tier and the model per request.

### One-sentence definition
*TokenCut is a model-free, cache-aware gateway that cuts the tool output coding agents feed
back to the model, keeping the code and errors that matter, so every request costs less
without breaking the agent.*

### Plain-language picture
An editor stands between the agent and the model. Big file → keep the functions that matter,
leave a note for the rest ("lines 40–95 omitted, read again if needed"). Long test log → keep
failures and the summary. Never touch the user's messages or the model's replies. The model
remembers recent text cheaply (prompt cache), so old pages are never rewritten while that
memory is valid; when it is lost anyway (model switch, long pause, context nearly full), the
old pages are tidied too.

---

## 2. Team decisions you must respect

| Decision | Detail |
|---|---|
| **Phase focus** | Testing / comparison / reporting of context trimming. No new features. |
| **ODE / model routing** | Deferred until after the mid evaluation. Do not implement. Ideas parked for later: control the provider's thinking budget / reasoning effort per step; model switches cost a cache rewrite. |
| **No local repo map in the gateway** | The gateway only trims what is inside requests. Do not add whole-repo indexing to the live gateway. |
| **Embeddings** | Allowed in the **offline benchmark only** (`eval/embeddings.py`, `fastembed`). Too slow for the live gateway on a laptop CPU (~13 chunks/s → 2–5 s per new large file). Do not add them to the gateway. |
| **Cache safety is non-negotiable** | Between cache events, earlier messages must be sent byte-for-byte as before. There are tests for this; keep them green. |
| **Fail open** | Any trimming error → forward the original request. |
| **Privacy** | Logs contain numbers only (no prompts, code, headers or keys). Full bodies only with `TOKENCUT_CAPTURE=1`, local, gitignored. Never commit `logs/`, `cache/`, `swebench_repos/`. |
| **Shared files need approval of the other two members** | `tokencut/contracts.py`, `tokencut/tokens.py`, `requirements.txt`, `pyproject.toml`, `docs/DECISIONS.md` (append to the change log is fine). |

Owners (README): ingestion + SWE-bench loader — Masooma Ali; graph + API — Huma Taj;
trimming + metrics — Fatima Shakir. In practice Claude implemented all parts; the team reviews.

---

## 3. What is built (current state)

### Pipeline (offline, whole repo)
```
repo ──► ingestion.parse_repo() ──► list[Chunk]          (tree-sitter: module / class / function / method)
               │
               ▼
         graph.build_graph() ──► CodeGraph               (contains / imports / calls / inherits edges)
               │
   query ─────►▼
         trimming.generate_variants() ──► 4 × ContextVariant  (aggressive 15% / moderate 35% / conservative 60% / full)
```
All shared data formats: `tokencut/contracts.py` (Chunk, Edge, CodeGraph, ScoredChunk,
ContextVariant, Tier, Segment, SegmentKind, ToolResultKind).

### Gateway (live)
```
agent ──► TokenCut gateway (python -m tokencut.gateway, 127.0.0.1:8787) ──► provider
  1. adapter: provider JSON → list[Segment] with JSON paths      (gateway/adapters/)
  2. classify tool results: file_read / command_output / search  (gateway/classify.py)
  3. on-arrival trim of results ≥ 2,000 tokens                   (gateway/trimmer.py)
       Python code → codeview.py (Iteration 1 engine, verbatim lines + markers)
       logs / search → textcut.py (tail, error lines, head)
  4. cache events: model switch to a cold model / cache expired / context nearly full
       → re-trim old history, optional tool pruning              (gateway/history.py, session.py)
  5. write back, forward, stream reply back, read billed usage   (gateway/app.py, usage.py)
  6. log numbers, /tokencut/stats                                (gateway/records.py)
```

### Code map
| Path | What it does |
|---|---|
| `tokencut/ingestion/walker.py` | file list via `git ls-files`; skips non-.py, >1 MB, binary, `migrations/`, `vendor/` |
| `tokencut/ingestion/python_parser.py` | tree-sitter chunker; ids `path::Qualified.name`, `#2` for duplicates |
| `tokencut/graph/builder.py`, `modules.py`, `queries.py` | code graph; name-resolution rules (DECISIONS #7); `GraphIndex` closure / BFS |
| `tokencut/trimming/query.py` | file / symbol / traceback mentions in the query |
| `tokencut/trimming/scoring.py` | score = 0.5·structural (graph distance from mentions) + 0.5·BM25 |
| `tokencut/trimming/packing.py` | 32k pool (whole files, top-3 chunks + file BM25), tier budgets, nested greedy packer, skeletons; aggressive tier takes chunks without callees (`CLOSURE_HOPS`) |
| `tokencut/trimming/render.py`, `variants.py` | rendering; `generate_variants` |
| `api/main.py` | demo API: `POST /context/variants` |
| `tokencut/gateway/adapters/` | Anthropic / OpenAI / Gemini request ↔ Segments |
| `tokencut/gateway/trimmer.py` | on-arrival, cache-safe trimming; edit safety (files the user named / edited stay whole) |
| `tokencut/gateway/codeview.py` | code trimming inside tool results (kept lines verbatim incl. line numbers; `[TokenCut: lines a-b of f.py omitted …]`) |
| `tokencut/gateway/textcut.py` | command / search output trimming |
| `tokencut/gateway/history.py`, `session.py` | cache-event trimming, per-conversation memory, tool pruning |
| `tokencut/gateway/usage.py` | billed usage (incl. cache read / write) from SSE / JSON responses |
| `tokencut/gateway/records.py`, `app.py` | analysis, logs, stats, modes, settings |
| `eval/loader.py`, `metrics.py`, `run_eval.py` | SWE-bench Lite loader (HF datasets-server), gold chunks, Iteration 1 eval |
| `eval/benchmark.py`, `embeddings.py`, `prepare.py` | head-to-head benchmark vs baselines; task sets in `eval/tasks_iter2.json` |
| `tests/` | 145 tests (all must pass) |

### Gateway settings (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `TOKENCUT_MODE` | `shadow` | `shadow` = forward unchanged, report possible savings; `trim` = send trimmed request |
| `TOKENCUT_TIER` | `moderate` | `aggressive` / `moderate` / `conservative` |
| `TOKENCUT_MIN_TRIM_TOKENS` | `2000` | smaller tool results are never trimmed (use `300` on the tiny playground repo) |
| `TOKENCUT_CACHE_EVENTS` | `1` | `0` disables history re-trimming at cache events |
| `TOKENCUT_CACHE_TTL` | `300` | idle seconds after which the cache counts as expired (requests marked `"ttl": "1h"` use 3600 — Claude Code appears to use this) |
| `TOKENCUT_PRUNE_TOOLS` | off | `1` = at cache events keep only tools used so far + core tools |
| `TOKENCUT_CORE_TOOLS` | Claude Code / Gemini CLI basics | comma-separated tools always kept when pruning |
| `TOKENCUT_TASK_MASK` | **on** (P1) | at a new user task, mask earlier tasks' large outputs if the cache rewrite pays off (R10); `0` = off |
| `TOKENCUT_TASK_MASK_TOKENS` | `300` | smallest output masked at a new task |
| `TOKENCUT_CCR` | off | `1` = shortened outputs carry a `tc_…` reference; gateway offers and answers `tokencut_retrieve` (P1) |
| `TOKENCUT_LAZY_TOOLS` | off | `1` = only core / used tools sent + catalog tool `tokencut_enable_tools` (P1) |
| `TOKENCUT_SYSTEM_NOTE` | off | `1` = fixed ~125-token note on TokenCut's markers at the end of the system prompt (agent requests, trim mode only; prompt enhancement, not yet A/B tested) |
| `TOKENCUT_PROMPT_BRIEF` | off | `1` = task brief appended to confidently classified typed user messages (rules + ONNX router; agent requests, trim mode only; stateless repeat rule; fail open) |
| `TOKENCUT_BRIEF_THRESHOLD` | `0.87` | router probability needed for a brief |
| `TOKENCUT_ROUTER_DIR` | — | local router model folder (4 files, not in git) |
| `TOKENCUT_ROUTER_REPO` | `humataj/tokencut-router` | private HF repo, downloaded once to `~/.cache/tokencut/router` if no local folder |
| `TOKENCUT_INDEX_DIR` | `~/.cache/tokencut/repoindex` | repo index cache (P1, `python -m tokencut.repoindex`) |
| `TOKENCUT_CAPTURE` | off | `1` = save full request bodies locally (private data!) |
| `TOKENCUT_LOG_DIR` | `logs/gateway` | numbers-only JSONL logs |
| `TOKENCUT_{ANTHROPIC,OPENAI,GEMINI}_UPSTREAM` | official URLs | other compatible endpoints |
| `TOKENCUT_HOST` / `TOKENCUT_PORT` | `127.0.0.1` / `8787` | listen address |

Agent wiring: Claude Code `ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic`; OpenAI SDKs /
Cline base URL `http://127.0.0.1:8787/openai/v1`; Gemini base URL `http://127.0.0.1:8787/gemini`.

---

## 4. Results so far (what has actually been measured)

### Offline benchmark (`eval/results/benchmark_test.md`, 81 held-out SWE-bench Lite tasks, never tuned on)
Metric: share of the chunks edited by the real fix that survive as full code. Equal token budgets.

**Pure trimming** (all methods shrink the same neutral 32k context; share of *available* gold kept):

| Budget | TokenCut | BM25 | Embeddings | Head truncation | Random |
|---|---|---|---|---|---|
| 15% | 0.68 | 0.65 | 0.72 | 0.15 | 0.12 |
| 35% | 0.86 | 0.80 | 0.90 | 0.47 | 0.36 |
| 60% | 0.90 | 0.93 | 0.96 | 0.69 | 0.63 |

TokenCut ≫ truncation / random (p < 0.001); level with BM25 and embeddings (no significant
difference). **Whole-repo selection:** embeddings lead (0.43 / 0.57 / 0.62 vs TokenCut 0.36 /
0.49 / 0.58); TokenCut's whole-file 32k pool limits file recall (~62%). On the 40 dev tasks
TokenCut looked better than on test — part of that was tuning; report test numbers.

### Gateway, offline
On-arrival trimming of a request with 4 real source files + a long pytest log: **70% smaller**,
**~85 ms** added (first request ~330 ms warm-up). Cache-event re-trim of a synthetic 30-turn
session: 71–85% smaller event request, ~30 ms (synthetic best case).

### Gateway, live (Claude Code on the tiny playground repo, Haiku 5.5 ↔ Sonnet 5.5)
| Session | Result |
|---|---|
| Shadow mode | Tool definitions 66–80% of request tokens; 96% of input tokens are cache reads; file reads often arrive through `cat` (Bash) |
| Trim mode, on-arrival only | 22% of tool output removed, ~2% of request tokens; all tasks correct; cache intact |
| + cache-event trimming (model switch) | 6.4% of request tokens saved; post-switch summary complete and accurate |
| + tool pruning | **71% of request tokens saved** (34k tokens of tool definitions removed per request after the switch); all tasks correct; Claude used `curl` when the pruned web-fetch tool was missing |

These live sessions are **short and on a tiny repo**. Nothing yet measures **task success on
real bugs** with trimmed context — that is the biggest open evidence gap.

---

## 5. Setup and commands

### Install (Python 3.11)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                 # must be 145+ passed
ruff check .              # must be clean
```
The machine may have no `python` command; use `python3` or the venv. The first token count
downloads a small tiktoken file (needs internet once).

### Offline benchmark
Local data is NOT in git (gitignored, several GB): `swebench_repos/`, `cache/`
(parsed chunks, embeddings). Rebuild once:
```bash
python -m eval.prepare --set dev     # 40 tuning tasks (requests, flask, pytest, django)
python -m eval.prepare --set test    # 81 held-out tasks (10 repos) — clones with --filter=blob:none
python -m eval.benchmark --set test --methods tokencut,bm25,random,head,bm25_files      # minutes
python -m eval.benchmark --set test --methods tokencut,tokencut_hybrid,bm25,embed,random,head,bm25_files   # hours on a laptop CPU (embeddings), cached after
python -m eval.benchmark --merge a.csv b.csv --set test --out eval/results               # combine runs
python -m eval.run_eval              # Iteration 1 metrics on the 40 dev tasks
```
Results: `eval/results/benchmark_<set>.md` / `.csv`. **Never tune parameters on the test
set.** If you must tune, use dev, then confirm on a fresh held-out set (≈150 unused SWE-bench
Lite tasks remain; build it like `eval/loader.select_test_ids`).

### Live gateway
```bash
# terminal 1
TOKENCUT_MODE=trim TOKENCUT_MIN_TRIM_TOKENS=300 python -m tokencut.gateway
# terminal 2 (never on an important repo — use a throwaway copy)
cp -r fixtures/sample_repo ~/tokencut-playground && cd ~/tokencut-playground && git init -q && git add -A && git commit -qm start
ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic claude
```
Stats: `http://127.0.0.1:8787/tokencut/stats` (`trimming.saved_by`, `trimming.cache_events`,
`billed_usage`). Per-request log lines: `logs/gateway/requests-<date>.jsonl`, e.g.
```bash
grep -o '"model": "[^"]*"\|"cache_event": [^,]*' logs/gateway/requests-*.jsonl | paste - - | tail -25
grep -o '"how": "[a-z_]*", "tool": "[^"]*", "before": [0-9]*, "after": [0-9]*' logs/gateway/requests-*.jsonl | tail
```
Restart the gateway between test runs (it keeps per-conversation memory in RAM).

---

## 6. The testing & evaluation work program

Work through these in order. Record every run (date, machine, settings, agent, model, repo,
task script) in the report.

### A. Reproduce (do first)
1. `pytest -q`, `ruff check .` — note results.
2. `python -m eval.prepare --set test`, then the no-embedding benchmark on test; compare with
   `eval/results/benchmark_test.md`. Differences must be explained (versions, data).

### B. Offline analysis (no API cost)
1. **Per-repo breakdown** of the benchmark (django, sympy, matplotlib, …): where does TokenCut
   win or lose, and why (look at individual tasks).
2. **Skeleton value**: report `chunk_recall_with_skeletons` vs `chunk_recall` for TokenCut.
3. **Latency**: time `trim_request` / `trim_with_cache_events` on realistic request bodies of
   increasing size (10k, 50k, 100k, 200k tokens); report median and p95.
4. **Failure analysis**: list the held-out tasks where gold code is lost at 35% and categorise
   the cause (mention missing, large function, wrong file, module-level edit, …).
5. Optional: re-run the embedding baselines if time allows (they are slow; cached after).

### C. Live test matrix (real agent, throwaway playground repo, and ideally one larger real repo copy)
Use **the same scripted task list** for every configuration so results are comparable
(e.g. 8–12 tasks: run tests, read 2–3 files, grep, add a function with tests, fix a failing
test, summarise the session, …). For each configuration record `/tokencut/stats`, the log
lines above, task correctness, and any agent confusion (re-reads, mentions of TokenCut notes,
missing tools, errors).

| # | Configuration | Purpose |
|---|---|---|
| C1 | **No TokenCut** (agent talks to the provider directly) — or `TOKENCUT_MODE=shadow` | baseline cost (shadow = same numbers, nothing changed) |
| C2 | `trim`, `moderate`, cache events off (`TOKENCUT_CACHE_EVENTS=0`) | on-arrival trimming alone |
| C3 | `trim`, `moderate`, cache events on, switch model once mid-session | history re-trim at a model switch |
| C4 | C3 + `TOKENCUT_PRUNE_TOOLS=1` | tool pruning |
| C5 | C3 with `aggressive` and with `conservative` | tier comparison |
| C6 | long session (40+ turns) to reach `context_full`, and a >1 h idle gap for `cache_expired` | other cache events |
| C7 | another agent if available (Gemini CLI via `/gemini`, Cline / OpenAI-compatible via `/openai/v1`) | multi-provider check |

Key questions for each: **tokens saved (%), billed input per request, cache read / write
share, latency added, tasks correct (yes / no), extra turns or re-reads caused by trimming.**

### D. Task-success check (needs the user's go-ahead and an API budget — ask first)
Run a small set of real SWE-bench tasks with an agent through the gateway, with and without
trimming, and compare resolve rate and cost. Do not spend money without explicit approval.

### E. Report
Fill `docs/RESULTS_REPORT.md` (section 8). Lead with the honest bottom line, then tables.

---

## 7. Conventions

- Branch per piece of work: `test/<topic>` or `docs/<topic>`; small PRs; one teammate reviews.
- Before every commit: `ruff check .` and `pytest -q` (CI runs both). Ruff 0.16 enables extra
  default rules (e.g. blind `except Exception` must log, unused unpacked variables, `datetime.UTC`).
- Contracts are law: modules talk through `tokencut/contracts.py` classes only.
- Deterministic output: sort, break ties by chunk id, never iterate a `set` to build output.
- Count tokens only with `tokencut.tokens.count_tokens` (tiktoken `o200k_base` proxy; it
  undercounts Claude by ~1.5× — use billed usage for money).
- Record any change of behaviour or tuning in the `docs/DECISIONS.md` change log.
- PR descriptions end with an AI-assistance note.
- Tests that fake a provider must return **streamed** bodies (`ChunkedBody` / `reply_json` in
  `tests/test_gateway_app.py`); httpx2's `MockTransport` pre-reads plain byte bodies.
- The machine used so far could not `git push` (no credentials): commit locally and give the
  user the push command and a PR description.

---

## 8. Report template — `docs/RESULTS_REPORT.md`

```markdown
# TokenCut — Results Report (<date>)

## Bottom line (5 lines, plain words)
- How much TokenCut saves (offline / live), whether agents still work, how it compares, main weakness.

## Setup
Machine, Python, commit hash, agent + version, models, repos, settings per run.

## 1. Offline benchmark
Tables (trim / retrieve, 15/35/60%), CIs, significance, per-repo breakdown, failure categories.

## 2. Latency
Per request size: median / p95 for on-arrival and cache-event trimming.

## 3. Live sessions
Per configuration C1–C7: tokens saved, billed input per request, cache read/write share,
events, tasks correct, re-reads / confusion, notes.

## 4. Comparison with research (SWE-Pruner, SWEzze, LongCodeZip, Pichay, Complexity Trap, Anthropic context editing)
What they report vs what we measured; where we are better / worse / not comparable.

## 5. Limitations and risks
## 6. Recommendations (do NOT implement — for the team to decide)
```

---

## 9. Known limitations / open issues

1. No task-success (resolve-rate) evidence yet — offline metrics are "needed code kept".
2. Held-out benchmark: level with BM25 / embeddings, not clearly ahead; whole-repo file recall ~62%.
3. On small projects most tokens are Claude Code's tool definitions; without tool pruning the saving is small (~2–6%).
4. Code trimming is Python-only (logs / search work for any language).
5. Per-conversation memory is in RAM: a gateway restart costs one cache rewrite (never a broken request).
6. Tool pruning can block a tool with no substitute until the next cache event (hence opt-in).
7. Conversation identity = system prompt + first user message; Claude Code's automatic
   first-message text may make two sessions in the same folder look alike (seen once; a model
   switch now only counts if the new model's cache is cold, which limits the impact).
8. Live results so far are short sessions on a tiny repo.
9. Re-request protection works per file, not per line range.

## 10. Glossary
- **Chunk** — one module / class / function / method of a Python file.
- **Tier** — aggressive (15%), moderate (35%), conservative (60%), full (100%); nested.
- **Gold chunk** — a chunk edited by the real SWE-bench fix.
- **Prompt cache** — provider feature: a repeated prompt prefix is billed at ~0.1×; writing it costs ~1.25×.
- **Cache event** — model switch to a cold model, cache TTL expired, or context nearly full.
- **On-arrival trimming** — trimming a tool result the first time it appears; identical afterwards.
- **Shadow mode** — measure only, forward unchanged.
- **ODE** — the planned bandit for (tier × model) choice; out of scope now.
