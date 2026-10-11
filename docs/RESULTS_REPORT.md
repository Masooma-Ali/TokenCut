# TokenCut — Results Report (2026-10-11, in progress)

## Bottom line (5 lines, plain words)
- _To be written once steps B–D are done._
- Step A (reproduction): the held-out offline benchmark reproduces exactly on a second machine
  (Windows) — every recall number is identical to the committed results.
- Step B (offline): once the right file is in the pool TokenCut keeps the needed code well;
  74 % of whole-repo losses come from file choice. Skeletons almost never fit (budget is 99.7 %
  full code). Gateway trimming adds ~40 ms at 13k tokens up to ~0.8 s (p95 1.8 s) at 90–170k,
  because every turn re-trims the whole history.

## Setup

| Item | Value |
|---|---|
| Machine | Windows 10 Pro 10.0.19045, Intel 4-core laptop CPU (Family 6 Model 78) |
| Python | 3.11.0 (project `.venv`), packages = `requirements.txt` (tree-sitter 0.26.0, tree-sitter-python 0.25.0, tiktoken 0.14.0, rank-bm25 0.2.2, ruff 0.16.10, pytest 9.1.1) |
| Git | 2.51.0.windows.2, `core.autocrlf=true`, `core.symlinks=false` |
| Commit | `cecdc19` (main after PR #9) |
| Data | SWE-bench Lite test split via HF datasets-server; 10 repos cloned with `--filter=blob:none` (~2 GB) |

## 0. Reproduction (step A)

### A1. Tests and lint
| Command | Result |
|---|---|
| `pytest -q` | **136 passed** in 8.2 s |
| `ruff check .` | **All checks passed** |

### A2. Held-out benchmark, no embeddings
Command: `python -m eval.prepare --set test`, then
`python -m eval.benchmark --set test --methods tokencut,bm25,random,head,bm25_files --out eval/results/repro`.

- `prepare` needed three attempts: the first process was killed externally after 21 tasks (no
  error), the second hit a GitHub network drop while cloning scikit-learn (`curl 18 … early
  EOF`). Both were environment problems; cached work was reused, no code change.
- Benchmark run time: **53 min** for 81 tasks on this laptop (sum of per-task times).

**Comparison with the committed `eval/results/benchmark_test.csv`** (1,944 rows = 81 tasks ×
3 budgets × methods × settings; matched row by row):

| Check | Result |
|---|---|
| Rows | 1,944 vs 1,944, none missing or extra |
| `chunk_recall`, `chunk_recall_available`, `chunk_recall_with_skeletons`, `file_recall`, `budget`, `gold_chunks` | **identical in every row, every method** |
| `tokens` (size of the selected context) | identical except **18 rows, all pylint, all in the `retrieve` setting**: 17 `random` rows (differences of 2–267 tokens) and 1 `bm25` row (pylint-7114 at 60 %: 19,198 → 18,623). Recall in these rows is unchanged. |
| TokenCut rows | **identical in every column** (except run time) |

**Why the pylint token counts differ.** The pylint repo contains two symlinked `.py` files
(`tests/functional/s/symlink/_binding/__init__.py`, `symlink_module.py`). On Windows with
`core.symlinks=false`, git checks them out as tiny text files containing the link target
(`../symlink_module/__init__.py`) instead of the real module code, so the whole-repo chunk set
is slightly different from Linux. Methods that pick from the whole repo (random sampling,
BM25 retrieval) then pick slightly different chunks. TokenCut never selected these files, so
its results are unaffected. CRLF line endings (`core.autocrlf=true`) make no difference
because the parser strips `\r`.

**Run time.** Per-task method times were ~2–3× slower than in the committed run (TokenCut
median 0.22 s vs 0.07 s per task and budget; BM25 file retrieval 3.0 s vs 1.0 s). This is a
slower laptop CPU; selection quality is unaffected. Latency is measured properly in section 2.

**Conclusion.** The held-out offline numbers in CLAUDE.md §4 / `benchmark_test.md` are
reproducible. The embedding baselines (`embed`, `tokencut_hybrid`) were not re-run (hours on
CPU); the committed numbers for them are used unchanged.

Raw output: `eval/results/repro/benchmark_test.{md,csv}` (local, not committed).

## 1. Offline benchmark (step B, held-out test set, 81 tasks)

Overall tables: `eval/results/benchmark_test.md` (reproduced exactly, section 0). Two
settings: **trim** = shrink the same neutral 32k context (metric: share of the gold chunks
present in that context that stay as full code; 45 tasks have gold in the context);
**retrieve** = choose from the whole repo (metric: share of all gold chunks kept; 80 tasks).

> **Which setting matches the live gateway?** Only **trim**. In the gateway the agent chooses
> which files to read; TokenCut only trims what is already in the request, with the same
> scoring and packing engine (`gateway/codeview.py` → `generate_variants`). The gateway never
> selects files from a repository (team decision, CLAUDE.md §2), so the **retrieve** setting —
> and its weakness, the whole-file 32k pool (B3) — does not apply to the gateway. Retrieve is
> kept as a secondary comparison with standard retrieval methods and for the demo API.

### B1. Per-repo breakdown
Few tasks per repo (2–14), so single repos are anecdotes, not proof. Embedding numbers are
from the committed run (not re-run).

**Trim, 35 % budget** (gold kept / gold available):

| repo | tasks | TokenCut | BM25 | embed | head | random |
|---|---|---|---|---|---|---|
| astropy | 5 | 0.80 | 1.00 | 1.00 | 0.50 | 0.40 |
| django | 6 | 1.00 | 1.00 | 1.00 | 0.67 | 0.08 |
| matplotlib | 5 | 1.00 | 0.80 | 1.00 | 0.60 | 0.36 |
| seaborn | 4 | 0.75 | 0.75 | 0.75 | 0.25 | 0.00 |
| xarray | 2 | 0.50 | 0.50 | 0.50 | 0.00 | 0.00 |
| pylint | 4 | 1.00 | 1.00 | 0.75 | 0.75 | 0.25 |
| scikit-learn | 9 | 0.78 | 0.46 | 0.96 | 0.22 | 0.43 |
| sphinx | 5 | 0.90 | 0.93 | 0.93 | 0.17 | 0.60 |
| sympy | 5 | 0.80 | 0.80 | 0.80 | 1.00 | 0.80 |

**Retrieve (whole repo), 35 % budget** (gold kept / all gold):

| repo | tasks | TokenCut | BM25 | embed | BM25 files |
|---|---|---|---|---|---|
| astropy | 6 | 0.58 | 0.67 | 0.83 | 0.50 |
| django | 10 | 0.60 | 0.68 | 0.93 | 0.50 |
| matplotlib | 12 | 0.33 | 0.28 | 0.46 | 0.17 |
| seaborn | 4 | 0.75 | 0.75 | 0.25 | 0.50 |
| xarray | 5 | 0.20 | 0.40 | 0.60 | 0.20 |
| pylint | 6 | 0.58 | 0.67 | 0.33 | 0.67 |
| pytest | 1 | 1.00 | 1.00 | 0.00 | 0.00 |
| scikit-learn | 12 | 0.60 | 0.26 | 0.62 | 0.42 |
| sphinx | 10 | 0.30 | 0.57 | 0.60 | 0.40 |
| sympy | 14 | 0.52 | 0.40 | 0.43 | 0.14 |

What it shows:
- **TokenCut wins clearly over BM25 on scikit-learn** (trim 0.78 vs 0.46, retrieve 0.60 vs
  0.26) and **sympy** (retrieve 0.52 vs 0.40) — repos where issues name classes/functions
  (`RidgeCV`, `LabelEncoder`) that BM25 drowns in common words.
- **TokenCut loses most on sphinx** (retrieve 0.30 vs 0.57 BM25 / 0.60 embed) and **django /
  astropy / xarray** in retrieve, almost entirely because the right file never enters its
  32k pool (see B4).
- Embeddings are strongest on django (0.93) and weakest on seaborn / pylint / pytest.

### B2. Value of skeletons
| setting | budget | recall (full code) | recall incl. skeletons | tasks where skeletons helped |
|---|---|---|---|---|
| trim | 15 / 35 / 60 % | 0.383 / 0.481 / 0.508 | 0.383 / 0.485 / 0.508 | 0 / 1 / 0 |
| retrieve | 15 / 35 / 60 % | 0.360 / 0.494 / 0.577 | 0.360 / 0.494 / 0.577 | 0 / 0 / 0 |

**Skeletons add almost nothing**: in one task out of 480 task-budget pairs. Reason (measured
with a per-task diagnostic): skeletons are almost never created. Median skeletons per variant
= 0 in every tier; they use 0–0.5 % of the budget; 61–81 of 81 tasks have none. The full-code
packer runs first and fills **99.7 %** of every budget (median 12–56 tokens left over), so
there is no room left for skeletons. This is a design effect, not a bug (DECISIONS #5 says
skeletons are added "if they fit"). See recommendation R2.

### B3. Failure analysis — gold chunks lost at the 35 % budget
Categories assigned in this order (first match wins), from a per-task diagnostic of
TokenCut (pool membership, chunk kind, size, rank, query mentions):

| cause | trim (69 gold available) | retrieve (132 gold) |
|---|---|---|
| file not in TokenCut's 32k pool | – | **48** |
| module-level edit | 0 | 0 |
| chunk larger than half the budget | 0 | 0 |
| not mentioned in the issue, ranked too low | 8 | 15 |
| only its file mentioned, ranked too low | 0 | 1 |
| named in the issue but still dropped | 1 | 1 |
| **total lost** | **9 (9 tasks)** | **65 (46 tasks)** |

- **Whole-repo selection fails mainly at file choice: 74 % of the lost gold (48 / 65) is in a
  file that never entered the 32k pool.** Once a gold chunk is in the pool, TokenCut keeps it
  most of the time. Improving the pool (file ranking) is the biggest lever.
- Pure trimming loses few chunks; those lost are mostly not named in the issue and rank low
  (median rank of lost in-pool gold: 268 in trim, 321 in retrieve).
- **"Named but dropped" case (astropy-7746, `WCS._array_converter`, rank 1):** the method is
  629 tokens, but keeping a method also requires its whole class chunk (`WCS`, **6,775 tokens**,
  mostly a long class docstring) plus the module chunk (1,203). ~8.6k tokens did not fit the
  space left at 35 %. See recommendation R3.
- Gold chunks are small (median 224 tokens; 2 % over 2,000), so "function too big" is not a
  cause in this set.

## 2. Latency (step B3)

What was timed: `tokencut.gateway.records.analyse()` — exactly what the gateway runs before
forwarding each request in **trim** mode (JSON parse, split into segments, token counts,
trimming, re-serialise). In shadow mode this runs in parallel and adds no delay.

Requests: Claude-Code-shaped Anthropic bodies built from **real django source files** (Read
results with line numbers, 2.5k–9k tokens each), pytest logs with one failure, and grep
output, plus ~1.3k system prompt and ~7.6k tool definitions. Gateway defaults (moderate,
min 2,000 tokens). Three cases: **on-arrival** (no session memory, = `TOKENCUT_CACHE_EVENTS=0`),
**steady** (same conversation, no cache event), **model switch** (cache event). Same laptop as
above; repetitions: 15 / 15 / 10 / 8. Script: run locally, not committed (see R5).

| request size (proxy tokens) | large results trimmed | on-arrival median / p95 | steady median / p95 | model switch median / p95 |
|---|---|---|---|---|
| 12.9k | 1 | 39 / 72 ms | 41 / 165 ms | 38 / 60 ms |
| 48.9k | 6 | 279 / 552 ms | 325 / 752 ms | 322 / 626 ms |
| 91.4k | 14 | 783 / 1,067 ms | 795 / 1,031 ms | 825 / 1,274 ms |
| 171.6k | 29 | 819 / 1,739 ms | 906* / 1,792 ms | 760 / 1,514 ms |

\* at 171.6k the "steady" request is above the context-full threshold (~110k), so it is itself
a `context_full` cache event — expected behaviour.

Saving on these synthetic requests: on-arrival 17 / 41 / 49 / 53 % of request tokens; at the
model switch 17 / 63 / 80 / 90 %. These are upper-bound numbers (requests are mostly large
tool output); live Claude Code requests are dominated by tool definitions and cache reads,
where the measured on-arrival saving was ~2 % (CLAUDE.md §4).

What it means:
- **Latency grows with conversation size, because nothing is cached between requests:** every
  large tool result in the whole history is re-trimmed (tree-sitter parse + scoring) on every
  turn, even though the result is identical each time. ~40 ms at 13k tokens, ~0.3 s at 50k,
  ~0.8 s at 90–170k (p95 up to 1.8 s).
- 0.8 s per turn is small next to model response time (several seconds), but it is added to
  every turn of a long session. Memoising the trim per tool result (same input → same output,
  already guaranteed by the cache-safety design) would make steady turns almost free. See R1.
- The ~85 ms reported earlier (4 files + one log) is consistent with this curve.
- Caveat: 8–15 repetitions on a shared laptop (desktop app and a download running), so p95 is
  noisy; the 91k vs 172k medians being close is partly noise.

## 3. Live sessions (step C)

Script: `docs/LIVE_TEST_SCRIPT.md` (10 fixed tasks). Agent: Claude Code 2.1.296, Sonnet 5.5,
Claude Pro login, typed by a team member. Repo: playground copy of `fixtures/sample_repo`,
reset to the `start` tag before each run. Gateway on this laptop, `TOKENCUT_MIN_TRIM_TOKENS=300`.

### C1 — shadow (baseline), 2026-10-11
| Measure | Value |
|---|---|
| Requests (POST /v1/messages) | 28 (2 got HTTP 429 rate-limit, retried by Claude Code) |
| Proxy tokens sent, share by kind | tools **78.8 %**, user text 11.9 %, tool results 3.2 %, system 2.9 %, assistant 2.2 %, tool calls 1.1 % |
| Billed input | cache read 1,913,191 · cache write 17,270 · uncached 5,781 → **~99 % cache reads** |
| Billed output | 5,389 |
| Possible trim saving (shadow) | 13,132 proxy tokens = **1.0 %** (Bash output 8,232, Glob search 4,900) |
| Cache events | none |
| Median request latency (incl. model) | 2.4 s |
| Tasks correct | 1–5, 7–10 correct; task 6 skipped |

Notes:
- Task 6 (run tests with the venv's Python) was not given; when the agent ran the tests in
  task 5 it found no pytest and ran `pip install pytest` into the global Python 3.11 — an
  agent side effect outside the repo, unrelated to TokenCut.
- As in the earlier live runs: on a tiny repo tool definitions dominate and tool output is
  small, so on-arrival trimming has almost nothing to work on.

### Larger real repo: sympy (SWE-bench `sympy__sympy-13471`)
Setup and tasks: `docs/LIVE_TEST_SCRIPT.md` Part 2. To save Pro quota the runs use **tasks
1–5 only** (overview, three "find and explain" questions about Float, fix the real bug), the
same five in every run. Default trim threshold (2,000 tokens).

#### S1 — shadow (baseline), 2026-10-11
| Measure | Value |
|---|---|
| Requests | 25 (Sonnet 5.5; 1 extra got HTTP 429 and was retried) |
| Proxy tokens, share by kind | tools **74.2 %**, user text 11.1 %, tool results 7.5 %, assistant 3.5 %, system 2.7 %, tool calls 1.0 % |
| Tool output in the whole session (distinct) | **~6.4k tokens**: file reads 3.7k, search 1.9k, commands 0.8k; largest single result ~2.0k |
| Billed input | cache read 1,876,730 · cache write 29,199 · uncached 3,760 → **98.4 % cache reads** |
| Billed output | 8,335 |
| Possible trim saving (shadow) | **0** — no tool result was large enough / trimmable |
| Cache events | none |
| Median request latency (incl. model) | 2.9 s |
| Bug fixed? | **Yes** — official SWE-bench hidden test `test_Float_from_tuple` passes on the agent's fix (`num[1].rstrip('L')`); only the 2 known unrelated failures remain; the agent's own regression test passes |

Notes:
- **Key finding: even on a large repo, Claude Code leaves almost nothing to trim.** It searches
  first (Grep/Glob), then reads 50–100-line slices around the match instead of whole files.
  Across the session the agent never received a large file listing; `numbers.py` (3,871
  lines) was never sent whole. On-arrival trimming of tool output — TokenCut's main feature —
  had no work to do. The cost is dominated by tool definitions (74 %), billed as cheap cache
  reads.
- The agent ran `python` (global, no mpmath), so it could not run sympy and said so; it
  still produced a correct fix and test. It also merged task 6 (add a test) into task 5.

#### S4 — trim + cache events + tool pruning + model switch, 2026-10-11
Settings: `TOKENCUT_MODE=trim`, `TIER=moderate`, `PRUNE_TOOLS=1`, `CAPTURE=1`, cache events on.
Same 5 tasks; `/model haiku` before task 4, `/model sonnet` before task 5.

| Measure | Value |
|---|---|
| Requests | 25 (18 Sonnet, 7 Haiku; 1 extra HTTP 429 retried) |
| Cache events | **1** (`model_switch` at the switch to Haiku); switching back to Sonnet was correctly not an event |
| On-arrival trims | **0** — largest tool result 1,613 tokens (Read of `numbers.py` lines 941–1090, `limit: 150`) |
| Request size at the event | **53,022 → 14,255 proxy tokens (−73 %)**; every later request ~70 % smaller (55.9k → 17.1k at the end) |
| What was removed | **tool definitions: 36,578 tokens per request** (46 tools → used + core tools); old history: 2 search outputs → notes, 1 duplicate file read (1,613 → 45 tokens), 1 duplicate search |
| Saving over the session (proxy tokens) | 426k of 1,251k = **34 %**, of which **94 % from tool pruning** |
| Billed input | cache read 1,174,595 · cache write 75,814 · uncached 3,756 (S1: 1,876,730 · 29,199 · 3,760) |
| Trimming time | 1–3 ms per request before the event; 18–75 ms after it |
| Bug fixed? | **Yes** — official hidden test `test_Float_from_tuple` passes (fix `num[1].rstrip('L')`); only the 2 known unrelated failures remain. No regression test this time (agent offered one) |
| Agent confusion | none seen: no missing-tool complaints, no mention of TokenCut notes; one duplicate read was replaced by a note without the agent needing it again |

Note (S1 and S4): when the agent tried to run Python it used the global `python`, which has
no mpmath, so `import sympy` failed (S1 task 3, S4 task 4). This happens on the user's
machine before anything reaches the gateway — not a TokenCut effect — but whether the agent
tries to run code varies between runs, which makes runs slightly less comparable.

**Why the tool definitions are so large:** the 46 tools include the Claude Code built-ins plus
the claude.ai connectors enabled on this account (Canva, Google Drive, Claude Docs, …). None
of the connectors were used, so pruning dropped them.

**Cost effect — how the numbers are made.** Billed token counts (cache read / write /
uncached / output) are **measured**: the provider reports them in every response and the
gateway logs them. Turning them into cost is a **calculation** with Anthropic's price ratios
(cache read × 0.1, cache write × 1.25, uncached × 1; Haiku's lower price ignored), giving
"input-equivalent tokens". The "without TokenCut" comparison is an **estimate** of a session
that was not run (requests assumed to keep growing from their size just before the switch);
the S4-shadow run below replaces it with a measurement (−29 % measured vs −31 % estimated).
The account is on Claude Pro, so this is usage-allowance consumption, not a per-token bill.

**Cost effect (estimate, input-equivalent tokens):**
- Requests after the event (11): measured **≈ 94k**. Without TokenCut the same switch would
  have rewritten the full ~79k-token cache for Haiku and then kept reading ~80k per request:
  **≈ 190k**. So after the event the input cost is roughly **halved**.
- Whole session: measured ≈ 216k vs ≈ 315k for the same session without TokenCut (≈ −31 %).
  Compared with S1 (no model switch at all, ≈ 228k) it is only ≈ 5 % lower — the model switch
  itself is expensive, and TokenCut mostly makes the switch cheaper.
- **Side effect found:** switching **back** to Sonnet cost a 27k-token cache write. Without
  TokenCut, Sonnet's old cache (untrimmed history) would still have been valid and read
  cheaply. Once history is trimmed at an event, the earlier model's cache cannot be reused.
  This is a correct consequence of the design, but it means frequent back-and-forth model
  switching reduces the benefit. See R6.

#### S4-shadow — S4's exact steps, TokenCut only measuring, 2026-10-11
The measured "without TokenCut" counterpart of S4 (same tasks, same two model switches,
`TOKENCUT_MODE=shadow`; nothing sent to the provider was changed).

| Measure (billed, from provider usage) | S4-shadow (no trimming) | S4 (trimming) | S5 (aggressive) |
|---|---|---|---|
| Requests (before / after the first switch) | 22 (15 / 7) | 25 (14 / 11) | 23 (14 / 9) |
| Cache write at the switch to Haiku | **83,926** | **22,612** | 17,334 |
| Switch back to Sonnet | cache read 83,677 + write 4,189 (old cache reused) | write 27,247 (old cache not reusable) | write 21,398 |
| Input-equivalent per request **before** the switch | 9.1k | 8.7k | 9.0k |
| Input-equivalent per request **after** the switch | **24.1k** | **8.6k (−64 %)** | **8.4k (−65 %)** |
| Input-equivalent, whole session | 305k | **216k (−29 %)** | 202k (−34 %) |
| Bug fixed (hidden test) | yes | yes | yes |

What it shows (now measured, not estimated):
- Before the switch the runs cost the same per request (sanity check: TokenCut changes
  nothing there).
- **After the switch, each request costs ~64 % less with TokenCut**, and the switch itself
  writes ~23k instead of ~84k tokens to the cache.
- **Whole session: −29 %** with 3 *more* requests than the shadow run. The earlier estimate
  for S4 (≈ −31 %) was close.
- **R6 confirmed:** without TokenCut, switching back to Sonnet reused its old cache (4k
  write); with TokenCut it needed a 27k write. That costs ~29k input-equivalent tokens, far
  less than what pruning saved, but it would add up with frequent back-and-forth switching.
- Requests per phase differ between runs (the agent made 4 Haiku calls in shadow, 7 in S4),
  so the per-request rows are the fairest comparison. Agents never behave identically, and
  this is one run of each configuration, so treat the percentages as indicative.

#### L1 — long session, trim (moderate), no pruning, Haiku 5.5, 2026-10-11
Script: `docs/LIVE_TEST_SCRIPT.md` Part 3 (12 tasks, 3 real bugs). Settings: `TOKENCUT_MODE=trim`,
`TIER=moderate`, `CAPTURE=1`, default 2,000-token threshold, **no tool pruning**, no model switch.

| Measure | Value |
|---|---|
| Requests | **107** (Haiku 5.5; 2 extra HTTP 429 retried); session ~43 min |
| Largest request | 95,168 proxy tokens — **below the ~110k context-full threshold**, so **no cache event** happened |
| Proxy tokens, share by kind | tools 52.8 %, tool results 22.5 %, tool calls 9.1 %, user text 8.5 %, assistant 4.3 %, system 2.8 % |
| Tool outputs in the final request | 82 outputs, 28.3k tokens: 63 under 500 tokens, 12 at 500–1k, 6 at 1–2k, **1 over 2k** |
| On-arrival trims | **1**: the `git diff` in task 10, 2,652 → 983 tokens (command-output rules) |
| Saving over the session (proxy) | 21,697 tokens = **0.3 %** |
| Billed | cache read 12,681,823 · cache write 166,287 · uncached 7,519 · output 56,594 (≈ 1,484k input-equivalent) |
| Trimming time | median 2.5 ms, max 137 ms |
| Bugs fixed (official hidden tests, on the agent's source changes) | **3 / 3** — 13480, 13647, 13773: FAIL_TO_PASS pass and the full test files pass |

Findings:
- **The agent trims its own big outputs.** For the full core test suite (~77 KB of output) Haiku
  wrote the output to a file and read it with `tail -n 400`, `grep`, `sort | uniq -c`, so no
  test output reached the gateway in one big piece. 63 of 82 tool outputs were under 500 tokens.
- **The one trim backfired.** Task 10 asked for "the full git diff". TokenCut cut the diff with
  its command-output rules (keep the end, error-looking lines and the start), leaving
  fragments and `[TokenCut: N lines of command output omitted]` markers. The agent noticed
  ("The tool truncated that output, so I'll pull the diff one file at a time") and re-ran
  `git diff` per file in **3 extra requests** (each piece < 2,000 tokens, so not trimmed).
  Those 3 requests cost ≈ 52k input-equivalent tokens; the trim itself saved roughly
  ≈ 7k (1.7k tokens × 13 later requests, mostly cache reads). **Net effect: a small loss.**
  No harm to correctness — the review was complete.
- Root cause: a `git diff` is code, not a log, but it arrives as Bash output and is classified
  as `command_output`, so the log rules (tail / error lines / head) are applied. And the user
  explicitly asked for the *full* diff. See R8.
- The session did not reach the context-full event (95k < ~110k) and the cache never expired
  (Claude Code uses a 1-hour cache), so cache-event trimming was never exercised.

#### S5 — as S4 but `TOKENCUT_TIER=aggressive`, 2026-10-11
Same settings, tasks and model switches as S4; only the tier differs.

| Measure | S4 (moderate) | S5 (aggressive) |
|---|---|---|
| Requests (Sonnet / Haiku) | 25 (18 / 7) | 23 (18 / 5) |
| On-arrival trims | 0 | 0 (largest tool result ~1.4k tokens) |
| Request at the event | 53,022 → 14,255 (−73.1 %) | 54,737 → 14,827 (−72.9 %) |
| Tool definitions pruned per request | 36,578 | 36,578 |
| Old history shortened at the event (one item = one old tool output replaced by a note) | 4 items, ~2.2k tokens (1 duplicate Read, 1 duplicate Grep, 2 old Grep outputs) | **6 items, ~3.3k tokens** (2 duplicate Reads, 2 old Reads, 1 duplicate Grep, 1 old Bash output) |
| Saving over the session (proxy tokens) | 34.1 % | 30.7 % (fewer requests after the event) |
| Billed: cache read · write · uncached | 1,174,595 · 75,814 · 3,756 | 1,160,928 · 65,708 · 3,752 |
| Input-equivalent tokens (per request) | ≈ 216k (8.6k) | ≈ 202k (8.8k) |
| Trimming time | ≤ 75 ms | ≤ 48 ms |
| Bug fixed (hidden test) | yes | **yes** (fix `num[1].rstrip('L')`; no regression test added) |
| Agent confusion | none | none (no re-reads of removed content, no missing-tool complaints) |

What it shows: **aggressive cut a little more old history (it protects only the last 2 turns
instead of 4), but the difference is ~1k tokens against ~37k tokens of pruned tool
definitions**, so the two runs end up almost the same. With Claude Code on this workload the
tier choice hardly matters; tool pruning is what saves. At the switch to Haiku, 6,370 tokens
were read from cache (likely the system-prompt prefix still cached from the S4 session).

## 4. Comparison with research (SWE-Pruner, SWEzze, LongCodeZip, Pichay, Complexity Trap, Anthropic context editing)
_Pending._

## 5. Limitations and risks
- **Images and PDFs are never trimmed and are not counted.** The adapters mark image /
  document blocks (attached by the user or returned by a tool) as `MEDIA` with empty text
  (`gateway/adapters/anthropic.py:113-116`; same in the OpenAI and Gemini adapters). They are
  forwarded unchanged, history trimming at cache events keeps them too, and they count as
  0 proxy tokens, so `/tokencut/stats` token breakdowns under-state requests that contain
  media (billed usage from the provider is still correct). An image costs roughly 1–1.6k
  tokens and a PDF a lot per page, so media-heavy sessions get no saving. Not in CLAUDE.md §9
  yet.
- Windows checkouts turn symlinked files into path stubs (`core.symlinks=false`); results on
  repos with symlinked Python files differ slightly between Windows and Linux. Clone with
  `-c core.symlinks=true` (needs Windows developer mode) for byte-identical data.

## 6. Recommendations (do NOT implement — for the team to decide)
- **R1 — Cache trim results per tool result in the gateway.** A result's trim depends only on
  earlier conversation, so it can be memoised by (fingerprint, tier). Would cut steady-turn
  latency from ~0.8 s to near zero on long sessions (B3).
- **R2 — Reserve a small share of each budget for skeletons** (e.g. 5–10 %) or pack them
  before the last low-ranked full chunks; today they almost never fit (B2). Must be tested on
  dev first, then confirmed on a fresh held-out set (never tune on test).
- **R3 — A kept method should not force its whole class chunk in.** Keep only the class
  header (skeleton) when the class chunk is large (astropy-7746: 6.8k-token class blocked a
  rank-1, 629-token method).
- **R4 — File-pool recall is the main weakness of whole-repo selection** (74 % of losses).
  Options: partial files instead of whole files, or a hybrid file ranking (embeddings offline).
- **R6b — Tiers barely matter for Claude Code today** (S4 vs S5): on-arrival trimming never
  triggers, and the tier only changes how many recent turns are protected at an event
  (~1k tokens of difference against ~37k of pruned tools). Tier choice (and later ODE) only
  becomes meaningful with agents or tasks that produce large tool outputs.
- **R6 — Count the cost of switching back.** After a cache event trims history, returning to
  the previous model forces a full cache write for it (S4: 27k tokens). Option to evaluate:
  delay history trimming until the new model has been used for a few turns, or only prune
  tools (which change the start of the prompt anyway). Needs a measured comparison.
- **R7 — Tool definitions are the main cost lever for Claude Code.** In every live run they
  were 74–79 % of request tokens; tool output was 3–8 %. Tool pruning delivered 94 % of the
  S4 saving. Consider making pruning the default focus of the gateway's evaluation.
- **R8 — Don't apply log rules to diffs or to output the user asked for in full** (L1). A
  `git diff` / `git show` arriving as Bash output should be treated as code (or left whole),
  and a request like "show me the full …" should protect the next output, like files the user
  names are protected today. In L1 the only trim caused 3 extra requests and a net loss.
- **R9 — Re-check the context-full threshold (~110k proxy tokens) for long sessions.** A
  107-request Haiku session peaked at 95k, so the event never fired; Claude Code's own
  auto-compact may act before it does. Measure where Claude Code compacts.
- **R5 — Commit the analysis scripts** (per-task diagnostic, latency benchmark) under `eval/`
  so these numbers are reproducible by the team.
