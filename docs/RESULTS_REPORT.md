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

## 4. Comparison with research

Context-pruning works from the proposal's reference list, read on 2026-10-12 (full text or
main sections). Numbers are quoted from the sources; inconsistencies inside a paper are noted.
Model-routing works (PILOT, LinUCB, RouteLLM, ACRouter, Arch-Router, Workload-Router-Pool) are
out of scope for this phase.

### 4.1 What each work trims and how

| Work | What it reduces | Method | When / where | Reported result | Cost of the method |
|---|---|---|---|---|---|
| **HCP** — Zhang et al., *Hierarchical Context Pruning* (arXiv 2406.18294; arXiv page lists no venue) | repository code for a code-completion model | tree-sitter + BFS over imports; keep current file whole and only **level-1 dependency files**; remove "global context" of other files; **top-k = 5** most similar functions keep full bodies, **top-p = 0.3** keep **headers only**, the rest dropped; similarity via OpenAI ada-002 embeddings | once, before generation | input "from over 50,000 tokens to approximately 8,000"; exact match up for all 6 models (+3.20 to +6.70 points, CrossCodeEval Python); "retaining only the function header information is sufficient" | embedding calls; authors note latency grows with many files |
| **LongCodeZip** — Shi et al., ASE 2025 (arXiv 2510.00446) | long code context | **coarse**: rank functions by approximated mutual information (drop in the instruction's perplexity given the function), keep the best, **placeholders** for the rest; **fine**: split kept functions into perplexity blocks, **0/1 knapsack** per function under a budget **adapted to each function's importance** | once, before generation | ~4.3–5.3× compression with higher scores than LLMLingua / LongLLMLingua / RAG (RepoQA 75.3 vs 59.3 LongLLMLingua, DeepSeek-Coder-6.7B); generation 15.70 s → 6.59 s; input cost −77 % | 2.58 s compression with a 7B model (GPU) |
| **LLMLingua** — Jiang et al., EMNLP 2023 (arXiv 2310.05736) | general prompts | small LM deletes low-information **tokens**; budget controller; iterative token-level compression | once | "up to 20x compression with little performance loss" | small LM |
| **LongLLMLingua** — Jiang et al., ACL 2024 (arXiv 2310.06839) | long prompts (documents) | question-aware compression of documents (method details in full paper, not read) | once | ~4× fewer tokens with up to +21.4 % (NaturalQuestions, GPT-3.5); 1.4–2.6× lower end-to-end latency; −94 % cost on LooGLE | small LM |
| **ICAE** — Ge et al., ICLR 2024 (arXiv 2307.06945) | any context | LLM encodes context into short **memory slots** the LLM conditions on | once | 4× compression with ~1 % added parameters (Llama) | needs model weights — **not usable with closed API models** |
| **SWE-Pruner** — Wang et al., 2026 (arXiv 2601.16746) | file-read / search outputs in agent sessions (`cat`, `grep`); not history, not tool definitions | middleware; the agent passes an optional `context_focus_question`; a 0.6B reranker scores tokens, a line is kept if its mean score > 0.5 (line-level, structure preserved); no question → raw output | on arrival, per tool call | Sonnet 4.5 on SWE-bench Verified: tokens −23.1 %, cost −26.8 %, solved 70.6 % → 72.0 %, rounds 51.0 → 41.7; LLMLingua-2 output parsed as valid code only 0.29 % of the time | 0.6B model, ~45–102 ms; agent tool schema changes |
| **LaMR** — Wang et al., 2026 (arXiv 2605.15315) | file reads / tool outputs in agent sessions | middleware; two rubrics with separate CRFs — **semantic evidence** and **dependency support** (imports, definitions, enclosing headers, try/except siblings) — mixed by a gate; **AST repair** after pruning; the agent can still request more | on arrival | Sonnet 4.5: tokens −30.5 %, solved 70.6 % → 71.8 %, rounds 51.0 → 39.6; Opus 4.6: −3.0 %. **SWE-Pruner increased tokens on Opus 4.6 (+6.6 %, up to +22 % in 4 of 5 multi-turn settings)** because removing structurally needed code "forces the agent into repeated file reads"; LaMR itself had one +7.9 % case | 0.6B model; no latency or dollar figures |
| **Aider repo map** (tool docs) | a repo summary sent with **every** request | tree-sitter symbols + graph ranking of the most-referenced definitions with signatures; `--map-tokens` default 1k, adjusted dynamically | every request | – | none (no model) |
| GraphCodeBERT (ICLR 2021) | – | code representations from data flow | – | supporting evidence for structure-aware methods | – |

What they trim, in short: most of these papers trim **code context once**, before the model
starts (HCP, LongCodeZip, LLMLingua, LongLLMLingua, ICAE, Aider). Only **SWE-Pruner and LaMR**
act on **tool outputs during an agent session**. **None trims tool definitions or agent
history, and none considers the provider's prompt cache** — the two features that produced
TokenCut's live saving (S4 vs S4-shadow).

### 4.2 Where TokenCut stands

| Aspect | TokenCut (measured here) | Closest research |
|---|---|---|
| Architecture | transparent gateway, 3 providers, no agent change, model-free | SWE-Pruner / LaMR: middleware with a 0.6B model (SWE-Pruner also changes the tool schema); LiteLLM: gateway without context awareness |
| On-arrival trimming | whole functions + line markers (Python), tail/errors/head (logs); 0–1 trims per Claude Code session; one harmful trim (L1 git diff → 3 re-fetches) | SWE-Pruner, LaMR: line-level with structure / dependency repair |
| Offline selection | level with BM25 / embeddings (81 held-out tasks); skeletons almost never fit (B2) | HCP (headers as a fixed share), LongCodeZip (blocks inside functions, knapsack), Aider (reference-ranked repo map) |
| History | only at cache events; −64 % per request after a model switch; never fired in a 107-request session (L1) | not covered by these works (SWE-Pruner and LaMR state they do not compress history) |
| Prompt cache | **designed around it** | not discussed by any of these works |
| Re-fetch handling | per-file protection after a re-read | LaMR keeps structural lines to avoid re-reads; SWE-Pruner reports fewer repeated reads, but more on Opus 4.6 (per LaMR) |
| Tool definitions | pruning at cache events (94 % of the S4 saving) | not covered |
| Images / PDFs | not handled | not handled by any work read |

### 4.3 Methods TokenCut does not use yet — what to adopt

| # | Method (source) | What it would change in TokenCut | Tokens | Cost | Correct output | Latency | Fit |
|---|---|---|---|---|---|---|---|
| 1 | **Headers as a guaranteed share** (HCP: top-k full, top-p headers) | reserve part of each budget for signatures of the next-best functions instead of filling it with full code first | ↓ | ↓ | ↑ (HCP: accuracy up on 6/6 models) | none with BM25 instead of embeddings | **yes** — fixes the near-dead skeleton feature (B2, R2) |
| 2 | **Blocks inside large functions + knapsack** (LongCodeZip) | split big functions / classes into blocks, score blocks with BM25 (model-free), pick with a knapsack and a per-function budget | ↓ | ↓ | ↑ | small | **yes** — fixes the big-class case (astropy-7746, R3) |
| 3 | **Dependency repair** (LaMR) | when a line is kept, also keep its enclosing class / function header, the imports it uses and paired blocks (tree-sitter, no model) | ≈ | ↓ (fewer re-reads) | ↑ | small | **yes** — line-level outputs (grep, diffs, multi-file `cat`) |
| 4 | **Trim only when the intent is clear** (SWE-Pruner: no question → raw output) | leave output whole when the user asks for "full" output or for a diff | ≈ | ↓ (no re-fetch) | ↑ | none | **yes** — R8 |
| 5 | Per-item ratio by relevance (LongLLMLingua) | search results / logs: compress less-relevant items more | ↓ | ↓ | ≈ | none (BM25) | maybe |
| 6 | Rank by reference count (Aider repo map) | offline file pool ranking (the retrieve-setting weakness) | – | – | ↑ offline | none | offline only (no repo map in the gateway) |
| 7 | Only level-1 dependencies + drop other files' global context (HCP) | offline pool: fewer, closer files instead of whole high-scoring files | ↓ | ↓ | ↑ (HCP) | none | offline only |
| – | Token deletion (LLMLingua), memory slots (ICAE), 0.6B pruners (SWE-Pruner, LaMR) | – | – | – | token deletion breaks code (0.29 % valid AST) | GPU model | **no** for a model-free laptop gateway |

**Common lesson** (LaMR, SWE-Pruner and our L1 run): pruning that leaves fragments makes the
agent re-read, which can cost **more** than it saves — most of all with stronger models.
Effective methods keep **self-contained units** (relevant lines + their headers + imports),
or leave output whole when unsure.

**Evaluation method to borrow** (SWE-Pruner, LaMR): run a real agent on
SWE-bench Verified with and without the method and report **solve rate, tokens, cost and
rounds** — the evidence TokenCut still lacks at scale (CLAUDE.md step D).

### 4.4 Replay of the L1 session with the adoptable methods (free, offline)

Before implementing anything in the gateway, the candidate methods were prototyped in
`eval/replay_captures.py` and replayed on the **109 captured L1 requests** (Haiku, 12 tasks,
82 distinct tool outputs; tool outputs = 22.5 % of the session's proxy tokens). Each method
sees exactly the same session.

- **current** — TokenCut's gateway trimmer, at thresholds 2,000 (default) / 1,000 / 500
- **r8** — current, but diffs and outputs the user asked for "in full" stay whole (§4.3 row 4)
- **lamr** — r8 + model-free line-level pruning with dependency repair (enclosing
  def/class/hunk headers, used imports) for every output that is not a Python file read
  (§4.3 row 3, LaMR / SWE-Pruner idea without a model)
- **r10** — when a new user task starts, the previous task's large outputs become a ~40-token
  note (R10)

Cost unit: input-equivalent tokens (cache read × 0.1, cache write × 1.25, uncached × 1),
converted from proxy to provider tokens with the session's measured ratio (1.67). On-arrival
trims are cache-safe: they save the first cache write and every later cache read. R10 pays a
one-time cache rewrite of the changed tail at each task boundary. Baseline: the session's
**measured** billed cost, 1,483,560.

| Method | Threshold | Outputs trimmed | Of which diffs | Saving (input-equivalent) | % of session cost |
|---|---|---|---|---|---|
| current | 2,000 | 1 | 1 | 6,828 | 0.46 % — the git diff that caused 3 re-fetches |
| r8 / lamr | 2,000 | 0 | 0 | 0 | 0 % |
| current | 1,000 | 7 | 1 | 98,185 | 6.62 % |
| r8 | 1,000 | 4 | 0 | 51,914 | 3.50 % |
| lamr | 1,000 | 4 | 0 | 50,987 | 3.44 % |
| current | 500 | 19 | **5** | 125,320 | 8.45 % |
| r8 | 500 | 12 | 0 | 76,089 | 5.13 % |
| lamr | 500 | 12 | 0 | 69,073 | 4.66 % |
| **r10** (outputs ≥ 300) | – | 26 outputs at 8 task boundaries | – | **166,317** (198,386 reads saved − 32,069 rewrites) | **11.21 %** |
| r10 (outputs ≥ 1,000) | – | 7 outputs at 5 boundaries | – | 100,543 | 6.78 % |

What it shows:
- **At the default threshold nothing is left to trim** except the diff that backfired; R8 removes
  exactly that trim. With Claude Code, on-arrival savings only appear at lower thresholds
  (3.4–5.1 % of cost at 1,000–500 with R8), and lowering the threshold without R8 would have
  trimmed **5 diffs** (the re-fetched per-file diffs included) — the harmful case multiplied.
- **The LaMR-style prototype saves about the same as the current rules + R8** (slightly less,
  because repair keeps headers and imports). Its benefit is *structure* (fewer fragments, less
  re-reading), which a replay cannot measure — only a live run can. Example: for "explain how
  the Matrix classes are organised" it kept the `col_insert` and multiplication lines but
  dropped 28 `class …` lines the agent needed — exactly the re-read risk LaMR warns about.
- **R10 is the largest lever: −11.2 % of the session's input cost** (outputs ≥ 300 tokens),
  without touching the output while its task is running. Early boundaries pay off strongly
  (task 3 boundary: rewrite 409, saving ~53k over 103 later requests); late ones do not
  (output ≥ 1,000 at the last two boundaries: net −2.8k and −4.5k). A **break-even rule**
  (mask only when enough requests are expected to follow) is needed.
- Prototype time: ≤ 71 ms for the whole session (all 82 outputs).

Limits: replays show *possible* savings, not the agent's reaction (re-reads after a trim are
not modelled — the main risk for both lamr and r10); task boundaries are detected from
user-text entries (14 detected vs 12 tasks typed — Claude Code adds a few user-text entries
itself); one session, one model (Haiku). Live confirmation is needed before adoption.

Files: `eval/replay_captures.py`, results in `eval/results/replay_L1/replay.{md,json}`
(numbers only; the captures themselves stay local and gitignored).

### 4.5 Citation checks
- **HCP:** the arXiv page (2406.18294, June 2024) lists no venue — confirm "AAAI 2025" before
  citing. "50k → 8k tokens" and "better on all six models" are confirmed in the paper.
- **LaMR:** the token *increase* on stronger models is reported for **SWE-Pruner** with Opus
  4.6 (+6.6 %, up to +22 %), not for LaMR — adjust the slide wording.
- **SWE-Pruner:** its abstract range (29–54 % on SWE-QA) holds only for the GLM rows; its
  latency text says "below 100 ms" while its table shows 102 ms at 8k tokens.
- **LongCodeZip:** "up to 5.6×" comes from one model's row; most rows are 4–5×.
- Not checked here: PILOT, Arch-Router and Workload-Router-Pool numbers (model routing).

### 4.6 R8 and R10 implemented — offline check with the real gateway code (2026-10-12)

Implemented on branch `trimming/researchpapers_methodology` (tests first; 145 tests, ruff clean;
`docs/DECISIONS.md` 2026-10-12):
- **R8** (always on): a tool result containing a diff, or any result while the user's last
  message asks for the full diff / file / output / code / log / listing, is never trimmed on
  arrival. "The whole test suite" does not trigger it.
- **R10** (opt-in, `TOKENCUT_TASK_MASK=1`): at a new user message, large results (≥ 300 tokens)
  from earlier tasks become the cache-event notes (command output keeps its error and summary
  lines), remembered and re-sent identically. Done only if
  0.1 × tokens removed × expected later requests ≥ 1.15 × tokens re-cached, with expected later
  requests = max(20, requests so far). A first version that used the average requests per task
  skipped almost every boundary (the masked output is saved on *all* later requests of the
  session, not just the next task) and was replaced.

**Check:** the gateway code was run over the captured requests in order (real timestamps), and
the provider cache was modelled per request (the prefix identical to the previous request =
cache read, the rest = cache write). The model reproduces L1's measured bill within −0.5 %
(S4 −6.0 %, S5 +2.8 %: mixed models, small sessions).

| Session | Variant | Input cost vs no trimming | Tokens sent | What happened |
|---|---|---|---|---|
| L1 (Haiku, 109 requests) | R8 only | ±0 % | ±0 % | the git diff is no longer trimmed (the 3 re-fetches it caused would not happen live) |
| L1 | **R8 + R10 (≥ 300)** | **−9.9 %** | −14.0 % | 26 outputs masked at task boundaries |
| L1 | R8 + R10 (≥ 1,000) | −6.4 % | −8.9 % | 7 outputs masked |
| S4 (Sonnet / Haiku switch, 25 requests) | R8 only / + R10 (≥ 300) | −1.3 % / −1.4 % | −1.9 % / −2.7 % | event trimming does most; R10 adds 2 masks |
| S5 (aggressive, 23 requests) | R8 only / + R10 (≥ 300) | −1.2 % / −1.5 % | −2.6 % / −3.8 % | R10 adds 5 masks |

(These variants have tool pruning off; S4/S5 savings with pruning are in §3.)

What it shows: on a long session R10 cuts input cost by about a tenth, matching the prototype
(−11.2 %); on short sessions it adds a little and never made a run more expensive. R8 changes
no cost in the replay — its value is avoiding the re-fetch loop, which only a live run shows.
Masking time stays small (≤ 46 ms for the largest L1 request).

### 4.7 Live check — L3 (R8 + R10) vs L1, 2026-10-12

Same repo, same 12 tasks, same Haiku 5.5, same settings as L1; only difference: the new code
(R8 always on) and `TOKENCUT_TASK_MASK=1`. Typed by a team member
(`docs/LIVE_TEST_SCRIPT.md` Part 4).

| Measure | L1 (before) | L3 (after) |
|---|---|---|
| Requests | 107 | **86** |
| Billed: cache read · write · uncached | 12,681,823 · 166,287 · 7,519 | 7,836,922 · 200,981 · 7,477 |
| Input-equivalent cost, whole session | 1,483,560 | **1,042,395 (−29.7 %)** |
| Input-equivalent cost per request | 13,865 | **12,121 (−12.6 %)** |
| R10 masks | – | **21 outputs, 17,220 → 2,612 tokens, at 7 task boundaries** |
| `git diff` in task 10 | trimmed (2,652 → 983) → agent re-fetched per file (4 extra diffs) | **2,358 tokens, sent whole (R8)**; agent: "The diff is complete", reviewed it in one go |
| On-arrival trims | 1 (the diff) | 1: a pytest output (3,188 → 1,089, log rules, task 2) |
| Identical repeated tool calls | 0 | 0 |
| Bugs fixed (official hidden tests on the agent's source changes) | 3 / 3 | **3 / 3** (FAIL_TO_PASS pass; full test files pass) |
| Core suite before / after (task 11) | 105 failed / 1,456 passed, unchanged | same |
| Trimming time per request | median 2.5 ms, max 137 ms | median 27 ms, p95 48 ms, max 225 ms |

What it shows:
- **Per request, L3 is 12.6 % cheaper** — in line with the replay estimate for R10 (−9.9 %).
  The whole session is 29.7 % cheaper, but part of that comes from L3 needing fewer requests
  (86 vs 107): some of it is R8 (no diff re-fetch loop), the rest is normal variation between
  agent runs, so the per-request figure is the fairer one.
- **R8 worked live:** the full diff arrived whole and the re-fetch loop of L1 did not happen.
- **R10 caused no visible re-reading:** no identical repeated tool calls; for task 11 the agent
  compared with task 2 using the result files it had saved itself, not the masked output.
  Correctness was unchanged (3/3 bugs, no new core failures).
- **Same harm mechanism, new place:** in task 2 the agent's `pytest … | tail -n 200` output
  (3,188 tokens) was trimmed with the log rules into 6 fragments; the agent said "the output
  above was cut off in the middle" and re-ran the tests with a compact summary. R8 only covers
  diffs and explicit "full" requests, so on-arrival log trimming can still trigger re-runs —
  see R13.
- Trimming time rose (median 27 ms) because masked notes are recounted each turn; still small
  next to the ~3 s model latency (R1 would remove it).
- One run of each; agent behaviour varies between runs, so treat the percentages as indicative.

**In real money (Haiku 5.5 list prices, 2026-10-13):** $0.10 / $0.50 per MTok input / output
for prompts up to 100K tokens and **$0.50 / $2.50 above 100K**; cache reads 0.1×, and Claude
Code's **1-hour cache writes 2×** input (the input-equivalent figures above used 1.25×, the
5-minute write price — relative comparisons are similar, absolute ones slightly low).

| Run | Dollar cost | Requests whose prompt exceeded 100K tokens | Prompt size median / max |
|---|---|---|---|
| L1 | **$0.753** | 81 of 107 | 120,376 / 166,793 |
| L3 | **$0.405 (−46 %)** | 36 of 86 | 93,898 / 118,871 |

R10 kept most L3 requests under Haiku's 100K price step, so the dollar saving (−46 %) is much
larger than the token saving (−12.6 % per request). This effect is specific to tiered pricing
(Haiku 5.5) and long sessions.

Data: `logs/gateway/L3/` (local), agent changes kept for grading.

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
  **Status: implemented 2026-10-12 (§4.6).**
- **R9 — Re-check the context-full threshold (~110k proxy tokens) for long sessions.** A
  107-request Haiku session peaked at 95k, so the event never fired; Claude Code's own
  auto-compact may act before it does. Measure where Claude Code compacts.
- **R5 — Commit the analysis scripts** (per-task diagnostic, latency benchmark) under `eval/`
  so these numbers are reproducible by the team.
- **R10 — Mask a finished task's large outputs, cache-aware** (proposed by a team member):
  after a new user message, replace the previous task's large outputs with notes, only when
  the expected remaining session repays the one-time cache rewrite of that task's span. The
  model sees full output while it works, which would have avoided the L1 re-fetch. Simulate
  on the L1 captures first.
  **Status: implemented 2026-10-12 as opt-in `TOKENCUT_TASK_MASK=1` (§4.6); live run L3: −12.6 % per request, 3/3 bugs fixed, no re-reads (§4.7).**
- **R13 — Mark trimmed command output so agents do not re-run it** (L3): a pytest output
  trimmed into fragments made the agent re-run the tests. Options: raise the threshold for
  command output, keep one contiguous tail block instead of scattered fragments, or say in the
  note what was kept ("failures and summary kept; N passing lines omitted").
- **R11 — HCP-style header share and LongCodeZip-style blocks, model-free:** guaranteed budget
  share for signatures (top-k full, top-p headers) and BM25-scored blocks inside large
  functions with knapsack selection (§4.3 rows 1–2; tune on dev, confirm on fresh held-out).
- **R12 — LaMR-style dependency repair for line-level trims:** keep enclosing headers and used
  imports with every kept line (tree-sitter, no model) (§4.3 row 3).

## 7. Step D — task-success check (plan, agreed 2026-10-13)

**Question:** on real SWE-bench bugs, does TokenCut lower tokens and cost (and keep latency
about the same) **without lowering how many bugs the agent fixes**? This is the main open
evidence gap (CLAUDE.md §9 item 1). Branch: `test/step-d`. Status: **pilot being prepared**.

### Fixed settings
| Item | Value |
|---|---|
| Agent / model | Claude Code (headless `claude -p`), **Haiku 5.5**, default effort (medium) |
| Run A (baseline) | TokenCut `TOKENCUT_MODE=shadow` — forwards unchanged, measures everything |
| Run B (TokenCut) | `TOKENCUT_MODE=trim`, `TOKENCUT_TIER=moderate`, R8 (always on) + R10 (`TOKENCUT_TASK_MASK=1`), default 2,000-token threshold, **no tool pruning** |
| Both | `TOKENCUT_CAPTURE=1`, own log folder per run, same prompt, same turn and time limits |
| Paid by | the team's prepaid Claude plan (usage limits may pause the run; it resumes) |
| Code | no TokenCut changes; the step D script lives under `eval/` |

Why these choices: one change at a time (A vs B differ only in TokenCut); Haiku is cheap enough
for a meaningful number of bugs and fixed 3/3 bugs in L1/L3; moderate is the setting we would
recommend (the tier barely changed savings in S4 vs S5, and more aggressive cuts raise the
re-run risk seen in L3); tool pruning rarely triggers without model switches and is measured
offline from the A captures instead.

### Steps
1. **Choose and validate bugs (free).** Fresh SWE-bench Lite **sympy** tasks (not the three used
   in L1/L3). Keep a bug only if, on this machine, its official `FAIL_TO_PASS` test fails before
   the official fix and passes after it, and its `PASS_TO_PASS` tests pass. **Pilot: 5 bugs.**
2. **Automate the agent (free to build).** Per bug and run: fresh copy of sympy at the bug's
   commit → start the gateway with that run's settings → `claude -p "<official issue text>"
   --model haiku` with a fixed one-line note on which Python to use (identical in A and B),
   limited tools (no "skip all permissions"), turn and time limits → save the agent's changes.
   A/B order alternates per bug.
3. **Stop / resume safely.** Results saved after every run; finished runs skipped on restart;
   a usage-limit stop discards the interrupted run and repeats it from a fresh copy later; A/B
   stay paired.
4. **Grade automatically (free).** Agent's changes on a clean copy + official hidden tests:
   resolved = `FAIL_TO_PASS` and `PASS_TO_PASS` both pass.
5. **Log everything once** (so step D never has to be re-run for missing data):
   - `eval/results/stepd/requests.csv` (numbers only, committable): one row per request —
     timestamp, bug, run, request number, model, tokens per kind, tokens **before and after**
     trimming, every trim / R10 mask, cache events, trimming time, total latency, billed usage
     (cache read / write / uncached / output), **dollar cost** at Haiku 5.5 prices (incl. the
     100K price step and 1-hour cache writes), HTTP status.
   - `eval/results/stepd/runs.csv`: one row per run — bug, run, start / end, duration, requests,
     totals of tokens sent and billed, **dollar cost**, R8 / R10 counts, re-runs or re-reads
     after a TokenCut note, repeated tool calls, files changed, diff size, resolved (F2P / P2P
     separately), how it ended (finished / time limit / usage limit), versions and settings.
   - `eval/results/stepd/summary.md`: A vs B resolve rate, cost, tokens, requests, latency,
     with confidence intervals and per-bug pairs.
   - `logs/stepd/` (local, gitignored, private): captured requests, full agent transcripts,
     agent diffs, test output — for before/after views and offline replays (e.g. what A would
     cost with B's settings, or with tool pruning) without new runs.
6. **Pilot first:** 5 bugs × 2 runs = 10 runs → short report → the team decides on the full
   run (20–30 bugs).
7. **Analyse and report** here: paired A vs B results, failure analysis (did TokenCut ever cause
   a failure or a re-run loop), examples from the before/after viewer.

Not included: tool pruning, Gemini / GPT (Gemini blocked on the free tier — see note below),
tier or effort comparisons.

Note — other providers (2026-10-13): Codex CLI only sends OpenAI's Responses API format, which
TokenCut's OpenAI route does not parse (Chat Completions only), and ChatGPT Plus sign-in uses a
different backend — not testable without code changes. Gemini CLI connected through TokenCut
(`GOOGLE_GEMINI_BASE_URL`), and TokenCut parsed its requests with 0 errors, but Google refused
the free-tier keys (403 "project denied access" on one account; quota exhausted after automatic
retries on another), so no Gemini session could be completed.

### Changes made after checking the first 3 bugs (2026-10-13)
The run was started directly (no separate pilot) with a close check after the first 3 bugs
(sympy-24909, 24152, 23117). Two problems were found and fixed before continuing; the earlier
attempts are kept in `eval/results/stepd/attempt1_no_warmup/` and `attempt2_single_prompt/`.

1. **Cache-order effect (attempt 1).** The first request of a run carries Claude Code's system
   prompt and ~30k proxy tokens of tool definitions. The first run of a bug pays the 2× cache
   write; the second run of the same bug reads that cache. So whichever arm ran second looked
   cheaper (24909: A first $0.037 vs B second $0.019; 24152: B first $0.029 vs A second
   $0.017). A warm-up request before each run was tried (attempt 2) and dropped: Claude Code's
   first request is not byte-identical between runs (the tool list varied, 28 vs 35 tools, as
   connectors load), so a warm-up cannot reliably prepare the cache. **Fix: the summary also
   reports a "warm-start cost"** — each run's cost with its first request's cache writes billed
   as cache reads (later requests unchanged) — which removes the order effect for all runs.
2. **Single-prompt sessions give TokenCut almost nothing to do (attempt 2).** With one prompt
   per bug there is never a new user message, so **R10 can never act**, and on-arrival trimming
   acted in 1 of 3 bugs (one whole-file read in 24909). A/B differences were mostly normal
   agent variation (24152: 9 vs 16 requests with nothing trimmed). Attempt 2 result (3 bugs):
   resolved 2/3 vs 2/3; cost (warm start) −8 % for B — not attributable to TokenCut.
   **Fix (option 2, agreed):** each bug is now a short session of **three messages in one
   Claude Code session** (`--session-id`, then `--resume`), identical in A and B:
   1. "Here is an issue … First investigate it: find the relevant code and reproduce the
      problem. Do not change any files yet. Use <python> …"
   2. "Now make the code change in the repository that resolves the issue."
   3. "Run the tests relevant to your change with the same Python and summarise what you
      changed and why."
   This matches how people use Claude Code and gives R10 task boundaries; grading is the same
   official SWE-bench tests.

Other observations from the check: every run completed (no stalls, timeouts or usage-limit
stops); one transient 502 (gateway briefly could not reach Anthropic, Claude Code retried);
the computed cost matched Claude Code's own reported cost within ~1 %; runs cost about
$0.01–0.04 each; auto permission mode blocked some `cd` / `PYTHONPATH` shell commands (same in
both arms; the agent worked around it).

### Results (completed 2026-10-14)
29 SWE-bench Lite sympy bugs, each run once without (A) and once with (B) TokenCut, in
three-message Claude Code sessions (Haiku 5.5). 58 runs, all ended normally. One run
(18835/A) was redone because an internet outage made 26 of its requests fail (logged in
`runs_network_failed.csv`, not counted). Full tables: `eval/results/stepd/analysis.md`
(per task) and `runs.csv` / `requests.csv` (every run and request); script
`eval/stepd_analyse.py`.

**Correctness**
| | A (no trimming) | B (TokenCut) |
|---|---|---|
| Bugs fixed (official hidden tests) | **18 / 29 (62 %)** | **19 / 29 (66 %)** |

Fixed only in A: 18698, 20049. Fixed only in B: 21379, 21612, 22840. Sign test on these five
discordant bugs: p = 1.0 — **no detectable difference**. With 29 bugs a small change in fix
rate could still go unnoticed, but there is no sign that TokenCut lowers correctness.

**Cost, tokens, time** (per task: mean over the 29 bugs, B − A, 95 % bootstrap interval)
| Measure | A | B | Change | 95 % CI of B − A | B lower in |
|---|---|---|---|---|---|
| Cost per task, warm start | $0.0457 | $0.0391 | **−14.4 %** | −$0.025 … +$0.005 | 14 / 29 |
| Cost per task, raw | $0.0476 | $0.0416 | −12.6 % | −$0.025 … +$0.006 | 14 / 29 |
| Tokens sent per task | 1.12 M | 1.07 M | −4.4 % | −224k … +103k | 15 / 29 |
| Requests per task | 26.8 | 27.0 | +0.6 % | −3 … +3 | 10 / 29 |
| Wall time per task | 306 s | 347 s | +13.4 % | −107 … +236 s | 18 / 29 |

Per request (removes "the agent took more or fewer steps"): tokens sent **−3.7 %**, billed
prompt −4.2 %, **cost −6.9 %**. **Requests whose prompt went above Haiku's 100K price step:
A 33, B 3.**

**What TokenCut did in B:** acted in 27 / 29 bugs (R10 masks at task boundaries in 43
requests, on-arrival trims in 61); removed 1.40 M of the 32.5 M tokens the agent sent
(**4.3 %**). Trimming time per request: median 3.8 ms, p95 63 ms, max 235 ms.
**Re-reads after a trim / mask: 5** (in 21171 ×2, 18698, 20590, 21055). Identical repeated
tool calls: A 4, B 17 — mostly the agent re-running its own test script after an edit (their
earlier output had not been trimmed), not re-fetches of trimmed output.

**What it means**
- **Correctness held:** 19/29 vs 18/29, with failures in both directions.
- **Cost went down about 7 % per request and about 14 % per task in total, but the per-task
  saving is not statistically certain:** the 95 % interval includes zero and B was cheaper in
  only half the bugs, because how many steps the agent takes varies a lot between runs. One
  bug dominates the total (19254: A $0.269 after 56 requests, B $0.052 after 30).
- **The clearest effect is the price step:** R10 kept B's long sessions under Haiku's 100K
  step (3 requests above it vs 33 in A), which is where the per-request saving comes from.
  This matches L3 (−46 % in dollars).
- **Latency:** TokenCut adds a few ms per request; the +13 % wall time is noise from the
  agent's own test runs (e.g. 20322/B spent ~40 min running slow sympy tests), not trimming.
- **Re-read risk is real but small:** 5 re-reads in 29 sessions. 18698 is the one bug fixed
  only in A where a re-read also happened in B; whether it mattered cannot be told from one run.
- Compared with research: SWE-Pruner and LaMR report 23–31 % fewer tokens with open-source
  agents that read whole files; with Claude Code (which reads selectively) TokenCut removes
  4.3 % and saves ~7 % per request — consistent with the live runs (§3) and the reasons in §4.

**Limits:** one model (Haiku 5.5), one repository (sympy), one run per bug and arm; agent
variation is large, so more bugs or repeated runs would be needed to make the cost saving
statistically certain. Single-prompt runs (attempt 2) showed TokenCut has almost nothing to
do without follow-up messages.

### Which techniques matter — ranking after step D
Step D logged every trim with the rule that made it, so its 1.40 M saved tokens split by
technique (B arm, 29 sessions):

| Technique | Tokens saved | Share | Re-reads that followed it |
|---|---|---|---|
| R10 task masking (first mask + re-sent identically) | 1,271,983 | **90.5 %** | 4 |
| On-arrival code trimming | 60,063 | 4.3 % | 1 |
| On-arrival search trimming | 53,751 | 3.8 % | 0 |
| On-arrival command / log trimming | 19,094 | 1.4 % | 0 (but a test re-run in L3) |

Ranking of all techniques, combining step D with the live runs (L1/L3, S4/S5):

| Rank | Technique | Tokens / cost | Quality | Latency | When it applies |
|---|---|---|---|---|---|
| 1 | **R10 task masking** | 90.5 % of step D savings; requests above Haiku's 100K price step 33 → 3; L3 −12.6 % per request, −46 % in dollars | held (19/29 vs 18/29; 3/3 in L3); 4 re-reads in 29 sessions | ~4 ms median | every multi-message session |
| 2 | **Tool pruning** at cache events | −64 % per request after a model switch, −29 % per session (S4 vs S4-shadow) | held in S4/S5; risk of a missing tool (earlier `curl` workaround) | ms | only at cache events (model switch) — opt-in, not in step D |
| 3 | **R8 protection** (diffs, "full" requests) | saves nothing itself | prevents a measured failure (L1 diff re-fetch loop; L3 reviewed the diff in one go) | none | always on |
| 4 | On-arrival **code** trimming | 4.3 % of step D savings; often nothing to trim with Claude Code | 1 re-read | ms; up to ~0.8 s on very large requests (§2) | whole Python file reads |
| 5 | On-arrival **search** trimming | 3.8 % | no issues seen | ms | large grep / glob results |
| 6 | On-arrival **command / log** trimming | 1.4 % | worst risk per token saved (L3 test re-run after fragments) | ms | large test / command output |
| 7 | Cache-event **history** trimming | ~1–3k tokens per event (S4/S5); never triggered in L1 or step D | held | ms | model switch, expired cache, context nearly full |
| – | Tier choice | hardly any difference (S4 vs S5) | aggressive slightly raises re-read risk | – | on-arrival trims, protected recent turns |

Conclusions: R10 is TokenCut's most valuable technique with a selective agent like Claude
Code; its re-read risk is the next thing to reduce (e.g. not masking a file the agent is still
editing). Tool pruning is the strongest when it triggers and pairs naturally with the planned
model routing (ODE). On-arrival trimming adds little with Claude Code (its value is with agents
that read whole files), and log trimming has the worst risk-to-saving ratio. R8 should stay on.

## 8. P1 "big wins" (branch `feature/p1-big-wins`, 2026-10-16)

Built at a team member's request after step D showed on-arrival trimming removes only ~4 %
with Claude Code. Goal: attack where the tokens go: exploration turns, tool definitions,
finished-task history. Needs the other two members' review before merging.

| Feature | Setting | Status |
|---|---|---|
| R10 task masking on by default | `TOKENCUT_TASK_MASK` (default on) | measured in step D (−6.9 % per request) |
| CCR: every shortened output can be fetched back exactly (`tokencut_retrieve`) | `TOKENCUT_CCR=1` | unit-tested; **not measured live** |
| Lazy tool loading (core + used tools + catalog `tokencut_enable_tools`) | `TOKENCUT_LAZY_TOOLS=1` | unit-tested; **not measured live** |
| Repo index MCP server: `repo_map`, `find_code`, `symbol` | `python -m tokencut.repoindex serve` | offline retrieval measured (below); **agent use not measured** |

### Repo index search, offline (no API cost)
`python -m eval.repoindex_eval`. Query = issue text; metric = whether the code the real fix
edited is near the top. Weights chosen on the dev tasks (39 usable; grid of 144 settings),
reported once on the held-out test tasks (80 usable).

| Test set (80 tasks) | file@1 | file@3 | file@5 | chunk@5 | chunk@10 | chunk@20 |
|---|---|---|---|---|---|---|
| File BM25 (SWE-bench baseline) | 0.35 | 0.54 | 0.62 | – | – | – |
| Chunk BM25 | 0.30 | 0.51 | 0.62 | 0.29 | 0.39 | 0.49 |
| Iteration 1 scorer | 0.47 | 0.65 | 0.68 | 0.32 | 0.38 | 0.41 |
| **Repo index (dev weights)** | 0.46 | **0.70** | **0.74** | **0.41** | **0.49** | **0.54** |

- vs file BM25 on file@5: +0.11 (95 % CI +0.01 … +0.21), p = 0.049; vs chunk BM25 on chunk@10:
  +0.10 (CI +0.04 … +0.18), p = 0.013; vs Iteration 1 on chunk@10: +0.11, p = 0.013; on
  file@5: +0.06, not significant.
- Per repo, file@5 vs file BM25: better on django (0.90 vs 0.70), matplotlib (0.83 vs 0.58),
  xarray (0.80 vs 0.40), sympy (0.57 vs 0.43); level on astropy, seaborn, pylint, sklearn;
  worse on sphinx (0.60 vs 0.70).
- The dev gain was larger (file@5 0.82 vs 0.67) — part of it was selection on dev, as expected.
- Speed: a running server searches django (32k chunks) in ~0.4 s; first index of django from
  source ~3–8 s plus ~5 s for the graph, cached and incremental afterwards.
- Embeddings were **not** evaluated here (whole-repo embedding takes hours on this laptop).

### What this does and does not show
It shows the index puts the fix's code in front of the agent more often than BM25 does. It
does **not** yet show fewer turns or lower cost: that needs live runs (step D protocol, A vs
B with `claude mcp add tokencut …`, `TOKENCUT_LAZY_TOOLS=1`, `TOKENCUT_CCR=1`), which cost API
money and need the team's go-ahead.

### Step D again with the P1 features (completed 2026-10-16)
29 sympy bugs (same list and three-message sessions as §7), Claude Code 2.1.295 + Haiku,
`--effort medium`, `--strict-mcp-config`, this Linux laptop (all 29 tasks re-validated here).
Arm A = no trimming (shadow). Arm P = trim moderate + R8 + R10 + CCR + lazy tools + repo-index
MCP server. One run per bug and arm, alternating order. Script `eval/stepd_p1_run.py`; data
`eval/results/stepd_p1/` (numbers only).

| | A (no TokenCut) | P (P1 features) | change |
|---|---|---|---|
| Bugs fixed | **22 / 29** | **18 / 29** | −4 (fixed only by A: 19254, 20049, 21379, 22840; only by P: none; sign test p = 0.125) |
| Cost, total | $0.961 | $0.725 | **−24.6 %** (per bug: mean −$0.008, 95 % CI −$0.013 … −$0.002; P cheaper in 24 / 29) |
| Cost on the 18 bugs both fixed | $0.519 | $0.353 | −32.1 % |
| Cost per request | | | −17.1 % |
| Requests | 852 | 775 | −9.0 % |
| Wall time | 16,987 s | 14,038 s | −17.4 % |
| Cache reads / writes (billed tokens) | 48.8 M / 1.02 M | 18.9 M / 1.48 M | −61 % / +45 % |

What P did (gateway logs): it sent 61 % fewer proxy tokens than the agent produced, 92 % of
that from lazy tool loading (tool definitions); R10 masked 129 outputs; on-arrival trimming
fired 5 times; the agent called the repo index 45 times (≈1.5 per bug, nearly always
`find_code` first) and **never** called `tokencut_retrieve`; it enabled hidden tools 10 times.

**Bug found (not fixed yet, so all runs used the same code):** lazy tool loading also pruned
Claude Code's WebSearch side requests, whose only tool is Anthropic's server tool
`web_search_20250305`; those searches returned nothing. In 19254 (fixed only by A) the agent
enabled WebSearch, got empty results, and refused to implement an unverified formula — arm A
searched successfully. Fix: never hide provider server tools, and leave requests with only a
few tools alone. The other three A-only failures (20049, 21379, 22840) are not analysed yet.

**What it means:** P1 cuts cost clearly (−25 %, CI excludes zero), mostly through lazy tool
loading, but fixed 4 fewer bugs. With 4 discordant bugs all against P the difference is not
statistically significant (p = 0.125), yet it is a warning sign, and at least one of the four
is caused by a TokenCut bug. Do not merge with lazy tools on by default until the failures are
explained and the web-search bug is fixed and re-tested.

### Fix and re-run of the four P-only failures (2026-10-19)
The web-search bug was fixed (`virtual.py`: provider server tools are never hidden; side
requests with fewer than 2 client tools are left alone; DECISIONS 2026-10-19). Transcripts of
the other three failures show no TokenCut notes, re-reads or tool errors (21379: the agent found
`core/mod.py` itself but patched a caller in `hyperbolic.py`). Arm P re-run once on the four
bugs only (`eval/results/stepd_p1_rerun/`):

| Bug | First run (P) | Re-run (P, fixed code) | Note |
|---|---|---|---|
| 19254 | not fixed | not fixed | web search now works; the agent implemented a bound whose values differ from the official tests |
| 20049 | not fixed | **fixed** | |
| 21379 | not fixed | **fixed** | |
| 22840 | not fixed | not fixed | unexplained; failed twice |

Only the failures were re-run, so "20 / 29" would overstate P (bugs it fixed were not given a
second chance to fail). Report it as: first run 18 vs 22; of the 4 differences, 1 was a TokenCut
bug (fixed) and 2 passed on a re-run, which points to agent variation; 22840 remains open.

### Results page
`eval/demo_data.py` builds the page's data (per-bug billed costs, every recorded request replayed
through the current gateway code, tokens removed per technique, offline results, evidence files).
Terminal demo without API cost: `.venv/bin/python -m eval.demo_savings sympy__sympy-21055`.

## 9. Mini-SWE-Agent replay — the agent SWE-Pruner and LaMR used (offline, 2026-10-15)

**Question:** how much does TokenCut remove for the open-source agent SWE-Pruner and LaMR used,
compared with their results and with Claude Code? No API budget was available, so public agent
sessions were replayed through the gateway code offline: free and reproducible, but the agent
cannot react to the trims (no re-reads, no task-success measure).

**Correction (2026-10-15):** this run was first planned as the "agent that reads whole files"
test (as PROGRESS_SUMMARY / FULL_RESULTS assumed). The logs show that assumption was wrong:
Mini-SWE-Agent mostly views line ranges and caps its own output (see "Why the saving is far
below…" below). So this section does **not** answer whether TokenCut helps agents that read
whole files; that question is still open (§9.1).

### Setup
| Item | Value |
|---|---|
| Data | Public SWE-bench "bash-only" trajectories (S3 `swe-bench-submissions/bash-only/<run>/trajs/`), all **75 sympy** tasks of SWE-bench Verified per run; local copy in `cache/mini_swe_agent/` (gitignored) |
| Runs | **Mini-SWE-Agent v1.13.3 + Claude Sonnet 4.5** (70.6 % on Verified, the same baseline SWE-Pruner and LaMR report) and **v1.17.1 + GLM-4.6** (55.4 %; open-weight, GLM rows of SWE-Pruner) |
| Replay | `eval/mini_swe_replay.py`: request *k* = system + task + first *k* steps; each request goes through `records.analyse` (the gateway's own entry point: on-arrival trimming + cache events, one conversation memory per session, 10 s between requests so the cache never expires). Sonnet in Anthropic format, GLM in OpenAI format |
| Settings | `trim`, `moderate`, R8 on, cache events on, no tool pruning, R10 off (one task per session, so it cannot act); threshold 2,000 (default), plus 1,000 and 500 as a sensitivity check (not tuning) |
| Measure | proxy tokens (o200k) removed, summed over every request of a session (= tokens sent without a cache); gateway time per request |
| Overlap with step D | 10 of the 29 step D bugs are in this set (18698, 20154, 20590, 21379, 21612, 21847, 22714, 23262, 24066, 24213) |

### Finding 1 — as shipped, TokenCut removes nothing (0.0 %)
Mini-SWE-Agent v1 does not use tool calling. The model writes its command in a fenced bash
block and the output comes back as a **plain user message** (`<returncode>…<output>…`).
TokenCut only trims tool results and never touches user messages, so all 150 sessions pass
unchanged (column "as shipped" below), although command outputs are 45–57 % of the tokens sent.
The replay therefore rewrites every step as a `bash` tool call and its output as the tool
result: what TokenCut sees from an agent that uses tool calling (SWE-Pruner also changes the
agent's tool setup). TokenCut's code is unchanged by this.

### Finding 2 — a real R8 bug, fixed
The first replay still removed 0.0 % (Sonnet) and 0.1 % (GLM). Cause: Mini-SWE-Agent's standard
instructions say *"Wait for the first command to **complete, see its output**, then …"*, and
R8's "the user wants the full output" pattern matched "complete … output", so every output of
every session was protected. Fixed (small fix allowed by CLAUDE.md §0): the words between
*full / entire / whole / complete* and the output noun may now only be separated by spaces
(no punctuation, no line break). Two failing tests were written first
(`test_complete_as_a_verb_is_not_a_full_output_request`,
`test_full_output_request_across_a_line_break_is_not_matched`); 147 tests pass, ruff clean;
logged in `docs/DECISIONS.md`. Step D is not affected: none of its prompts or its 29 issue texts
match the old pattern (checked).

### Results (after the fix) — `eval/results/mini_swe/` (`summary.md`, `sessions.csv`, `requests.csv`)
| Run | Sessions / requests | Tool output share | Saved, as shipped | **Saved, threshold 2,000** | 1,000 | 500 | Sessions with a trim (2,000 / 500) |
|---|---|---|---|---|---|---|---|
| Sonnet 4.5 | 75 / 4,374 | 57.4 % | 0.0 % | **6.8 %** | 11.3 % | 16.3 % | 40 / 74 |
| GLM-4.6 | 75 / 3,874 | 45.3 % | 0.0 % | **5.7 %** | 7.9 % | 9.9 % | 37 / 63 |

Savings are shares of all proxy tokens sent over all requests (67.7 M and 69.9 M). Median per
session at 2,000: 3.3 % (Sonnet), 0.0 % (GLM). Two `context_full` cache events occurred (GLM,
requests ≥ 110k tokens) and removed 895 tokens of old output.

**Same 10 bugs as step D** (share of tokens sent):

| | Threshold 2,000 | 1,000 | 500 |
|---|---|---|---|
| Mini-SWE-Agent + Sonnet 4.5 (replay, on-arrival only) | 7.2 % | 13.6 % | 19.1 % |
| Mini-SWE-Agent + GLM-4.6 (replay, on-arrival only) | 3.1 % | 4.0 % | 5.3 % |
| Claude Code + Haiku 5.5 (step D arm B, live, on-arrival + R10) | 6.0 % | – | – |

In step D on-arrival trimming gave only 9.5 % of the saving (≈ 0.4 % of tokens sent), so for
on-arrival trimming alone Mini-SWE-Agent + Sonnet gives TokenCut far more to do. Different
models and agents, so this is indicative only.

### Why the saving is far below SWE-Pruner (−23 %) and LaMR (−31 %)
Fate of every output ≥ 2,000 tokens (first appearance):

| | Sonnet 4.5 | GLM-4.6 |
|---|---|---|
| `git diff` output — R8, never trimmed | 67 outputs, 491k tokens | 44 outputs, 477k tokens |
| already cut by the agent itself (> 10,000 characters → head + tail) | 48, 147k | 48, 137k |
| Python file view TokenCut recognises (`cat f.py`, `sed -n 'a,bp' f.py`) | 1, 2k | 1, 2k |
| Python file view **not** recognised (`cd /testbed && sed -n …`, `nl -ba f.py \| sed -n …`) | 7, 16k | 0 |
| other (tests, search, listings) | 2, 5k | 6, 17k |

- **Mini-SWE-Agent does not read whole files in practice.** It views line ranges
  (`sed -n '100,200p'`, `nl -ba … | sed -n`) and cuts any output over 10,000 characters itself.
  Median output: ~100 tokens; only ~3 % of outputs reach 2,000 tokens.
- **Most of the large outputs are diffs**, which R8 leaves whole by design (live run L1 showed a
  trimmed diff causes re-fetches).
- **Lowering the threshold helps** (Sonnet 6.8 → 16.3 % at 500), but the replay cannot show the
  re-read cost of cutting many medium-sized outputs — exactly the risk LaMR reports. A live
  check would be needed before changing the default.
- SWE-Pruner and LaMR prune **every** read output at line level with a 0.6B model guided by the
  agent's stated focus; TokenCut is model-free and only cuts large outputs.

### Gateway time (this laptop, `requests.csv`)
`trim` median 1–20 ms for requests up to 50k proxy tokens and 5–65 ms above that; p95 about
85–165 ms; whole analysis (parsing + token counting + trimming) p95 ≤ 272 ms at 50–100k tokens.
Every request re-trims the whole history (codeview runs again on each trimmed Python file),
which is why p95 is much higher than the median (see R1).

### Limits
Offline replay: the agent never sees the trimmed text, so re-reads, extra steps and task
success are not measured; tokens are the o200k proxy, not billed tokens; one repository
(sympy); the conversion to tool calls is ours (a real tool-calling Mini-SWE-Agent may format
outputs slightly differently). Gold-code survival (did the code the official fix changes stay?)
is not yet checked: it needs the SWE-bench Verified patches (a small public download).

### Recommendations (do NOT implement — for the team to decide)
- **R14 — text-protocol agents:** recognise `<returncode>…<output>…` user messages that follow an
  assistant bash block as tool output (Mini-SWE-Agent v1 and similar agents). Without it
  TokenCut saves 0 % with these agents.
- **R15 — recognise more file views:** `cd <dir> && cat|sed -n … f.py` and `nl -ba f.py | sed -n …`
  should reach the code-aware trimmer instead of the log rules.
- **R16 — threshold per agent:** agents that already cap their output (10k characters) leave
  little above 2,000 tokens; test 500–1,000 live (re-read rate!) before changing the default.

### 9.1 Which agents really send whole files? (checked in source code, 2026-10-15)
Each row was checked in the agent's current source on GitHub (main branch, read on 2026-10-15);
nothing here is taken from papers or documentation alone.

| Agent | How a file reaches the model (source) | Size limit before sending | Format TokenCut sees | Public SWE-bench logs |
|---|---|---|---|---|
| **Mini-SWE-Agent v1** | shell commands only; logs show mostly `grep` + `sed -n 'a,bp'` ranges (§9) | output > 10,000 chars → head + tail | **user message** → TokenCut trims nothing | yes (used in §9) |
| **Aider** | `get_files_content()` puts the **whole content** of every file in the chat into the prompt, no truncation (`aider/coders/base_coder.py:637–653`), every request | none in that function | **user message** (`get_chat_files_messages`, `role="user"`, l. 789–800) → TokenCut trims nothing | one run, 2024, Lite (`20240523_aider`) |
| **Cline** (VS Code v4.1.24, uses `@cline/core`) | `read_files` tool, up to 2,000 lines / 48,000 chars (`sdk/.../executors/output-limits.ts`) | then **every tool result is middle-truncated to 8,000 chars** by default (`message-builder.ts:29`, `truncateToolResultContent`) | tool result | none found |
| **Gemini CLI** | `read_file` without a range returns the **whole file up to 2,000 lines** (`packages/core/src/utils/fileUtils.ts:640–657`, `utils/constants.ts:10`) | tool output over 40,000 chars truncated (`config.ts:483`); its own output masking is under an experimental setting, default off (`settingsSchema.ts`, not every code path traced) | tool result (`functionResponse`) | none found; our free keys were blocked (§7) |
| **OpenHands** (`openhands-aci` editor) | `view` without a range returns the **whole file** with line numbers (`editor/editor.py:340–347`) | cut after 16,000 chars (`editor/config.py:1`, `results.maybe_truncate`) | tool result | yes: e.g. Claude Sonnet 4 (75 sympy logs, 14.2 MB), Claude Opus 4.5 (73, 9.2 MB), GPT-5 (75, 9.7 MB) |

**What this means:**
- Aider is the clearest whole-file agent, but it sends files as user text, so as shipped TokenCut
  does nothing with it (the same gap as R14).
- Cline cuts every tool result to 8,000 chars itself — about TokenCut's 2,000-token threshold —
  so a large whole-file read never reaches the model whole.
- **OpenHands is the best available test**: it really returns whole files (up to 16,000 chars,
  ~4,000 tokens) as tool results, which TokenCut can trim, and public logs exist for the same
  sympy tasks. Not yet checked: whether its agents actually call `view` without a range in these
  logs, and which editor version those runs used (needs one sample log).

## 10. Prompt hints and gateway code trimming (offline, 2026-11-14)

Question for prompt enhancement: does naming the needed code in the prompt help the gateway keep
it? `python -m eval.prompt_hint_eval all` (free, real `trim_request`, moderate tier; results in
`eval/results/prompt_hints/`). Gold = functions the real fix edits; kept = every line survives.

**Offline, 81 held-out tasks, 68 gold files ≥ 2,000 tokens read whole:**

| Prompt | Gold kept | Tokens sent |
|---|---|---|
| vague ("Fix the bug described in the issue.") | 29 % | 43 % |
| vague + right function names | **80 %** | 44 % |
| vague + wrong names (same file) | 25 % | 44 % |
| issue text | 69 % | 66 % |
| issue + right names | 82 % | 66 % |
| issue + file path | 100 % (file left whole: edit safety) | 100 % |

**Step D replay (real Claude Code, arm A, 29 sympy bugs):** of 9 reads that showed gold code,
8 were line ranges (`offset` / `limit`) and only 2 reached 2,000 tokens; the model wrote no text
before the call in 8 of 9 (thinking text is not readable). Both trimmable reads lost the gold
function today, with the model's words added, and with the right names added.

What it shows:
- Right names in **user** text raise gold kept 29 % → 80 % at the same size; wrong names lower it;
  a file path switches trimming off for that file. Names in system text are ignored (system-role
  fix, 2026-11-14), so they neither help nor switch protections on.
- With Claude Code the exposure is small: it reads the needed code in ranges below the threshold.
- About 20 % of gold is lost even with the right names (offline 20 functions: 9 removed,
  8 signature only, 3 partly kept; 15 of 18 files had room for it): the packer can keep a named
  function as a skeleton while keeping its callees whole (R3 family). Prompting cannot fix this.
- Research agrees that a stated focus helps pruning (SWE-Pruner goal hint; LongLLMLingua) and
  that extra instructions are followed less reliably as they pile up (IFScale, NeurIPS 2025).

Limits: right names are taken from the real fix (upper bound); moderate tier only; offline (no
agent reaction); step D has only 2 trimmable gold reads.

### 10.1 After the fix: named code is kept whole (2026-11-14)

The losses above were a packing weakness, now fixed in the gateway only (`codeview.select_lines`,
DECISIONS 2026-11-14): a function / class / method the query names, or a traceback points into,
is always kept whole. In step D the named function was not a skeleton but **dropped**: it was
bigger than the moderate budget (1,248 and 1,573 tokens vs 1,000), and the packer skips a chunk
that does not fit and fills the space with smaller, lower-ranked ones. Both functions were named
in the issue text the user gave.

| Prompt | Gold kept before | after | Tokens sent before | after |
|---|---|---|---|---|
| vague | 29 % | 29 % | 43 % | 43 % |
| vague + right names | 80 % | **91 %** | 44 % | 45 % |
| vague + wrong names | 25 % | 25 % | 44 % | 45 % |
| issue text | 69 % | **73 %** | 66 % | 69 % |
| issue + right names | 82 % | **93 %** | 66 % | 69 % |
| issue + file path | 100 % | 100 % | 100 % | 100 % |

Step D trimmable gold reads: 0 / 2 → **2 / 2** (also with today's query, because the issue names
the function). The 9 named chunks still counted as lost are all classes (e.g.
`IsolationForest`, `SessionBase`) whose edited lines are all kept; the metric counts a class only
if every method survives. So with the right names every edited line now survives, for 1-3 points
more tokens sent.

Limits: same as above; a common method name (e.g. `save`) names every method with that name in
the file, which keeps more than needed (costs tokens, never correctness). Without any name
(vague prompt) nothing changes: the packer's "skip what does not fit" rule still drops a large
top-ranked function; a fix for that needs a ranking signal, not a name.
