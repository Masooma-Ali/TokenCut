# TokenCut — Full Results With Calculations (2026-10-14)

This document collects **every test done on TokenCut so far**: the earlier work by the team
member who built the system (Iteration 1, benchmark, gateway) and the testing phase (steps A–D).
For each result it shows **where the number comes from and how it was calculated**.

**How the numbers were checked.** On 2026-10-14 every headline number below was recomputed
from the raw files by one script, `python -m eval.verify_results` (`eval/verify_results.py`).
It reads only numbers-only files: `eval/results/*.csv`, `eval/results/stepd/*.csv` and the
numbers-only gateway logs in `logs/` (local, gitignored). Each section is marked:

- ✅ **verified**: recomputed from raw data today and it matches.
- 📄 **from file**: taken from a committed result file, not re-run today.
- ⚠️ **reported only**: stated in earlier notes; the raw data is not on this machine, so it
  cannot be checked.

Corrections found during this check are listed in §11.

---

## 0. Bottom line: where we stand

1. **Offline selection quality is proven, but TokenCut is not better than standard methods.**
   On 81 held-out bugs, when shrinking a context to 35 %, TokenCut keeps 85.6 % of the code
   that the real fix edits. Cutting the start of the context keeps 47.4 % and random keeps
   35.9 % (p < 0.001). TokenCut is **level with BM25 and embeddings**: no significant difference.
2. **TokenCut does not hurt the agent.** On 29 real SWE-bench bugs Claude Code fixed
   **19/29 with TokenCut vs 18/29 without**. In the live sessions every bug that should have
   been fixed was fixed (S1, S4, S4-shadow, S5: 1/1 each; L1, L3: 3/3).
3. **The saving depends on the technique and the session.**
   - R10 task masking: about 7 % per request on 3-message bug sessions, 12.6 % per request
     (46 % in dollars) on a 12-task session.
   - Tool pruning at a model switch: 64 % per request after the switch.
   - On-arrival trimming of tool output, TokenCut's original feature: only 0–1 % with Claude
     Code, because Claude Code reads small slices itself.
4. **Not yet proven:**
   - a statistically certain cost saving per bug (the 95 % interval still includes zero);
   - results with an agent that reads whole files;
   - results with other providers;
   - the context-full and cache-expired events, which never fired.

---

## 1. Units and formulas used everywhere

| Term | Meaning |
|---|---|
| **Proxy tokens** | TokenCut's own count (`tiktoken o200k_base`). Used for "how much was removed". It undercounts Claude by about 1.5×. |
| **Billed tokens** | What Anthropic reports in each response: `cache_read`, `cache_write`, `uncached input`, `output`. The gateway logs them. They are measured, not estimated. |
| **Input-equivalent tokens** | One number for input cost, using Anthropic's price ratios: **0.1 × cache_read + 1.25 × cache_write + 1 × uncached**. Used for the Sonnet / mixed sessions run on the Pro plan. |
| **Dollar cost (Haiku 5.5)** | Per request: `prompt = cache_read + cache_write + uncached`. If `prompt ≤ 100K`: input $0.10 per million tokens, output $0.50. If `prompt > 100K`: $0.50 and $2.50. Cost = (uncached + 0.1 × cache_read + **2** × cache_write) × input price + output × output price. The 2× is the price of Claude Code's 1-hour cache writes. |
| **Warm-start cost** | A run's cost with its **first** request's cache writes billed as cache reads. This removes the "whoever runs second finds the cache already warm" effect (§9). |
| **Gold chunk** | A function, class or module chunk edited by the real SWE-bench fix. |
| **Recall** | Share of gold chunks that survive as full code. |

**Worked example of the dollar formula** (step D, bug 24909, run A, request 0, from
`requests.csv`): cache_read 45,761, cache_write 7,941, uncached 2, output 582, so the prompt is
53,704 tokens, which is under 100K.
- Input part: (2 + 0.1 × 45,761 + 2 × 7,941) × $0.10 / 1M = 20,460.1 × $0.0000001 = **$0.002046**
- Output part: 582 × $0.50 / 1M = **$0.000291**
- Total **$0.002337**, the same as the `cost_usd` logged for that request ✅.

The cost model was also checked against Claude Code's own reported cost (within about 1 %,
§9) and, for the replay, against L1's measured bill (−0.5 %, §7).

---

## 2. Earlier work (Iteration 1, benchmark, gateway)

This is the work done before the testing phase, mainly by the team member who built the
system (commits 2026-10-06 to 10-09).

### 2.1 Iteration 1 evaluation, 40 dev tasks 📄 (`eval/results/iter1_summary.md`)

- **Setup:** whole-repo selection on requests, flask, pytest and django.
- **Metric:** share of gold chunks kept as full code.

| Tier | Tokens kept (of 31.8k pool) | Reduction | Gold file recall | Gold chunk recall |
|---|---|---|---|---|
| aggressive | 4,752 | 85.1 % | 0.725 | 0.563 |
| moderate | 11,085 | 65.1 % | 0.750 | 0.678 |
| conservative | 18,996 | 40.3 % | 0.750 | 0.697 |
| full pool | 31,806 | 0 % | 0.750 | 0.769 |

**Calculation:** reduction = 1 − tokens kept / 31,806. For moderate: 1 − 11,085 / 31,806 = 0.651.

**Insight:**
- The pool already misses 25 % of the gold files (`pool_has_gold_file` 0.750). So even the
  full pool reaches only 0.769 recall.
- Moderate keeps 0.678 of 0.769, which is 88 % of what is reachable, using 35 % of the tokens.
- These 40 tasks were used for tuning, so they are optimistic.

### 2.2 Head-to-head benchmark, 81 held-out tasks ✅ (`eval/results/benchmark_test.csv`)

Two settings:
- **trim:** every method shrinks the same neutral 32k context. This is what the gateway does.
- **retrieve:** choose from the whole repository.

All methods get equal token budgets.

**Trim, share of the gold that was available in the context** (45 tasks have gold inside the
32k context). Recall = sum of per-task recall ÷ 45:

| Budget | TokenCut | BM25 | Embeddings | Head truncation | Random |
|---|---|---|---|---|---|
| 15 % | 30.67/45 = **0.681** | 29.23/45 = 0.650 | 32.27/45 = 0.717 | 6.70/45 = 0.149 | 5.60/45 = 0.124 |
| 35 % | 38.50/45 = **0.856** | 35.83/45 = 0.796 | 40.33/45 = 0.896 | 21.33/45 = 0.474 | 16.13/45 = 0.359 |
| 60 % | 40.67/45 = **0.904** | 41.67/45 = 0.926 | 43.00/45 = 0.956 | 31.17/45 = 0.693 | 28.50/45 = 0.633 |

**Significance at 35 %** (paired sign test, from `benchmark_test.md`). "Better / equal / worse"
counts tasks:

| TokenCut vs | Better / equal / worse | p |
|---|---|---|
| head truncation | 22 / 20 / 3 | < 0.001 |
| random | 28 / 13 / 4 | < 0.001 |
| BM25 | 8 / 34 / 3 | 0.227 (not significant) |
| embeddings | 3 / 36 / 6 | 0.508 (not significant) |

**Retrieve, share of all gold** (80 tasks with gold):

| Budget | TokenCut | BM25 | Embeddings | BM25 files | Random |
|---|---|---|---|---|---|
| 15 % | 28.80/80 = 0.360 | 31.87/80 = 0.398 | 34.43/80 = **0.430** | 19.00/80 = 0.237 | 0.000 |
| 35 % | 39.50/80 = 0.494 | 38.73/80 = 0.484 | 45.30/80 = **0.566** | 28.00/80 = 0.350 | 0.000 |
| 60 % | 46.17/80 = 0.577 | 43.23/80 = 0.540 | 49.92/80 = **0.624** | 33.00/80 = 0.412 | 1.00/80 = 0.013 |

TokenCut's file recall in retrieve is only about 0.62. It finds the right file in 62 % of
tasks, against 0.81–0.90 for BM25 and embeddings.

**Insight:**
- For **trimming**, the job the gateway does, TokenCut is far better than naive cutting and
  level with standard retrieval.
- For **whole-repo selection** embeddings lead, because TokenCut's pool picks the wrong file
  too often.
- On the 40 dev tasks TokenCut looked better (trim 15 %: 0.883 vs BM25 0.737), but that was
  the tuning set. The held-out numbers above are the honest ones.

### 2.3 Gateway, offline checks ⚠️ (commit messages of 2026-10-08/09)

- On-arrival trimming of one request (4 source files + a long pytest log): **70 % smaller,
  about 85 ms added**.
- Cache-event re-trim of a synthetic 30-turn session: **71–85 % smaller**, about 30 ms.

These are synthetic best cases, and the test requests are not in the repository. The latency
curve in §4.2 (about 40 ms at 13k tokens, about 280 ms at 49k) is consistent with the 85 ms
figure.

### 2.4 First live sessions, tiny playground ⚠️ (CLAUDE.md §4)

Claude Code with Haiku/Sonnet:

| Session | Saving |
|---|---|
| on-arrival only | 2 % of request tokens |
| + model-switch event | 6.4 % |
| + tool pruning | 71 % after the switch |

All tasks were correct. The logs of these sessions are not on this machine. The same effects
were later re-measured on sympy with logs (S4 and S4-shadow, §5.3), so these early numbers are
superseded.

---

## 3. Step A: reproduction ✅ (2026-10-11, re-checked 2026-10-14)

| Check | Result |
|---|---|
| `pytest -q` | 136 passed at the time of step A. **145 passed today** (9 tests added with R8/R10). |
| `ruff check .` | clean then and today |
| Held-out benchmark re-run (`eval/results/repro/`) | 1,944 rows matched row by row. **Every recall number is identical.** 18 token-count differences, all in pylint, come from 2 symlinked files that Windows checks out as text stubs. TokenCut's rows are identical in every column. |

**Insight:** the offline numbers in §2.2 are reproducible on a second machine.

---

## 4. Step B: offline analysis (no API cost)

### 4.1 Value of skeletons ✅

Skeletons are short placeholders (signatures without bodies). TokenCut tasks with gold,
recall with and without skeletons:

| Setting / budget | Recall, full code | Recall incl. skeletons | Tasks where skeletons helped |
|---|---|---|---|
| trim 15 / 35 / 60 % | 0.383 / 0.481 / 0.508 | 0.383 / 0.485 / 0.508 | 0 / 1 / 0 of 80 |
| retrieve 15 / 35 / 60 % | 0.360 / 0.494 / 0.577 | same | 0 / 0 / 0 of 80 |

Skeletons helped in 1 of 480 task-budget pairs. The per-task diagnostic
(`eval/results/repro/diag.jsonl`) shows why: full code fills **99.7 %** of every budget first,
so skeletons almost never fit. This is a design effect, not a bug (recommendation R2).

### 4.2 Latency ✅ (`eval/results/repro/latency.json`)

Trimming time on Claude-Code-shaped requests built from real django files, laptop CPU:

| Request size (proxy tokens) | Results trimmed | On-arrival median / p95 | Same conversation (steady) median / p95 | Model switch median / p95 |
|---|---|---|---|---|
| 12.9k | 1 | 39 / 72 ms | 41 / 165 ms | 38 / 60 ms |
| 48.9k | 6 | 279 / 552 ms | 325 / 752 ms | 322 / 626 ms |
| 91.4k | 14 | 783 / 1,067 ms | 795 / 1,031 ms | 825 / 1,274 ms |
| 171.6k | 29 | 819 / 1,739 ms | 906 / 1,792 ms | 760 / 1,514 ms |

**Insight:**
- Latency grows with the conversation, because every large result in history is re-trimmed
  on every turn (recommendation R1: cache the trim per result).
- On these synthetic, output-heavy requests TokenCut removed 17–53 % on arrival and 17–90 % at
  a model switch. These are upper bounds. Real Claude Code requests are mostly tool
  definitions.

### 4.3 Failure analysis at the 35 % budget 📄

Each lost gold chunk was assigned one cause, using the first rule that matched:

| Cause | trim (69 gold chunks) | retrieve (132 gold chunks) |
|---|---|---|
| file never entered TokenCut's 32k pool | – | **48** |
| not named in the issue, ranked too low | 8 | 15 |
| only its file named, ranked too low | 0 | 1 |
| named in the issue but still dropped | 1 | 1 |
| module-level edit / chunk too large | 0 | 0 |
| **total lost** | **9** | **65** |

**Calculation:** in retrieve, 48 / 65 = **74 %** of losses are a wrong file choice.

**Insight:** once the right file is in the pool, TokenCut keeps the code. The weakness is
choosing files from a whole repo, which the gateway never does.

The one "named but dropped" case is astropy-7746. A rank-1 method of 629 tokens needed its
whole 6,775-token class (recommendation R3).

### 4.4 Per-repo breakdown 📄 (RESULTS_REPORT §1 B1)

These are only 2–14 tasks per repo, so they are anecdotes, not proof.
- **TokenCut beats BM25:** on scikit-learn (trim 35 %: 0.78 vs 0.46) and sympy (retrieve:
  0.52 vs 0.40). In these repos issues name classes and functions.
- **TokenCut loses:** on sphinx (retrieve 0.30 vs 0.57 BM25), because of file choice.

---

## 5. Step C: live sessions with Claude Code ✅ (all recomputed from `logs/gateway/<run>/`)

### 5.1 C1: tiny playground, shadow (baseline), Sonnet 5.5

| Measure | Value and calculation |
|---|---|
| Requests | 28 POSTs (27 succeeded, 1 rate-limited and retried) |
| What the requests contain (proxy tokens) | tool definitions **78.8 %**, user text 11.9 %, tool results 3.2 %, system 2.9 %, assistant 2.2 %, tool calls 1.1 % |
| Billed input | cache_read 1,913,191 · cache_write 17,270 · uncached 5,781 |
| Cache-read share | 1,913,191 / (1,913,191 + 17,270 + 5,781) = **98.8 %** |
| Possible trim saving | 13,132 / 1,297,966 = **1.0 %** |
| Tasks correct | 9 of 9 given (task 6 skipped) |

**Insight:**
- On a small repo almost all tokens are tool definitions, and 99 % of input is cheap cache
  reads.
- Trimming tool output can save at most about 1 %.

### 5.2 S1: sympy bug 13471, shadow (baseline), Sonnet 5.5

| Measure | Value |
|---|---|
| Requests | 25 succeeded |
| Tool definitions share | 74.2 % (tool results only 7.5 %) |
| Billed input | cache_read 1,876,730 · cache_write 29,199 · uncached 3,760 |
| Input-equivalent | 0.1 × 1,876,730 + 1.25 × 29,199 + 3,760 = **227,932** (9,117 per request) |
| Possible trim saving | **0** (largest tool result about 2.0k tokens, nothing trimmable) |
| Bug fixed | **yes** (official hidden test passes) |

**Insight:** even on a large repo, Claude Code searches first and reads 50–100-line slices,
so on-arrival trimming has nothing to do.

### 5.3 S4 vs S4-shadow vs S5: model switch (Sonnet → Haiku → Sonnet) with tool pruning

All three runs used the same 5 tasks and the same switches:
- **S4-shadow:** TokenCut only measures (no change).
- **S4:** trimming + tool pruning, moderate tier.
- **S5:** as S4 with the aggressive tier.

| Measure | S4-shadow (no trimming) | S4 (moderate) | S5 (aggressive) |
|---|---|---|---|
| Billed cache_read · cache_write · uncached | 1,560,338 · 116,039 · 3,750 | 1,174,595 · 75,814 · 3,756 | 1,160,928 · 65,708 · 3,752 |
| Input-equivalent, whole session | 156,034 + 145,049 + 3,750 = **304,833** | 117,460 + 94,768 + 3,756 = **215,983** | 116,093 + 82,135 + 3,752 = **201,980** |
| Change vs shadow | – | 215,983 / 304,833 − 1 = **−29.1 %** | 201,980 / 304,833 − 1 = **−33.7 %** |
| Requests before / after the switch to Haiku | 15 / 7 | 14 / 11 | 14 / 9 |
| Input-equivalent per request **before** the switch | 9,069 | 8,650 | 9,003 |
| Input-equivalent per request **after** the switch | 24,113 | **8,625** | **8,437** |
| Change after the switch | – | 8,625 / 24,113 − 1 = **−64.2 %** | −65.0 % |
| Cache write at the switch | 83,926 | 22,612 | 17,334 |
| Proxy tokens removed over the session | (would have been 25.2 %) | 426,437 / 1,251,007 = **34.1 %** | 359,190 / 1,169,652 = **30.7 %** |
| Bug fixed | yes | yes | yes |
| Agent confusion | – | none | none |

**Insight:**
- Before the switch all three cost the same per request. This is the sanity check: nothing is
  changed there.
- After the switch TokenCut cuts each request by about two thirds. Most of it comes from
  **tool pruning**: 36,578 tokens of unused tool definitions removed per request (46 tools
  with the account's connectors).
- **The tier makes almost no difference.** Aggressive removed about 1k more history tokens,
  against about 37k of pruned tools.
- **Side effect (R6):** switching back to Sonnet cost a 27k cache write with TokenCut. Without
  it, Sonnet's old cache was reused (4k write). Frequent back-and-forth switching reduces the
  benefit.
- One run of each, so the percentages are indicative.

### 5.4 L1: long session, 12 tasks, 3 real bugs, Haiku 5.5, moderate trimming, no pruning

| Measure | Value |
|---|---|
| Requests | 107 succeeded (2 rate-limited) |
| Largest request | 95,168 proxy tokens, below the ~110k context-full threshold, so **no cache event** |
| Tool results share | 22.5 % (tool definitions 52.8 %) |
| On-arrival trims | **1**: a `git diff`, 2,652 → 983 tokens |
| Proxy tokens removed | 21,697 / 7,699,185 = **0.3 %** |
| Billed input | cache_read 12,681,823 · cache_write 166,287 · uncached 7,519 |
| Input-equivalent | 1,268,182 + 207,859 + 7,519 = **1,483,560** (13,865 per request) |
| Dollars (Haiku formula, §1) | **$0.753**; 81 of 107 requests had a prompt over 100K |
| Bugs fixed | **3 / 3** (official hidden tests) |

**Insight:** the only trim backfired. The user had asked for the *full* diff; TokenCut cut it
with log rules, and the agent re-fetched it in 3 extra requests:
- the re-fetch cost about 52k input-equivalent tokens;
- the trim saved about 7k;
- the net result was a **small loss**.

This finding led to **R8**.

### 5.5 Other providers (2026-10-13)

- **Codex CLI:** sends OpenAI's Responses API, which TokenCut does not parse. It cannot be
  tested without code changes.
- **Gemini CLI:** connected, and TokenCut parsed its requests with 0 errors. Google refused
  the free keys (403, then quota), so **no Gemini session finished**.

---

## 6. Comparison with research (proposal papers)

| Work | What it trims | Reported result |
|---|---|---|
| HCP | repo code for completion | 50k → about 8k tokens, accuracy up on 6 models |
| LongCodeZip | long code context | 4.3–5.3× compression |
| LLMLingua / LongLLMLingua | general prompts | up to 20× / about 4× |
| ICAE | any context | 4× (needs model weights) |
| SWE-Pruner | agent file / search outputs | Sonnet 4.5: −23.1 % tokens, solved 70.6 → 72.0 % |
| LaMR | agent file / tool outputs | −30.5 % tokens, solved 70.6 → 71.8 % |
| Aider | repo map | – |

Numbers are quoted from the papers (RESULTS_REPORT §4).

**Where TokenCut stands:**
- **None of these works trims tool definitions or history, and none considers the prompt
  cache.** Those are exactly the features that produced TokenCut's measurable live savings
  (S4, R10).
- Their agent results use **open-source agents that read whole files**. TokenCut's
  on-arrival trimming is the comparable part, and with Claude Code it removes far less: 0–4 %
  vs 23–31 %. The reason is the agent, not a weaker method. This is the gap the Mini-SWE-Agent
  test would close.
- LaMR reports that pruning which leaves fragments causes re-reads that can cost more than
  they save. L1 (diff) and L3 (pytest log) confirm this.

---

## 7. Replay and R8 / R10 offline check (free)

### 7.1 Replay of the 109 captured L1 requests 📄 (`eval/results/replay_L1/replay.md`)

- **Baseline:** L1's measured bill, 1,483,560 input-equivalent tokens.
- **Saving % =** saving ÷ 1,483,560.

| Method | Saving | % |
|---|---|---|
| current trimmer, 2,000-token threshold | 6,828 | 0.46 % (the diff that backfired) |
| + R8 | 0 | 0 % |
| R8 at 1,000 / 500 threshold | 51,914 / 76,089 | 3.50 % / 5.13 % |
| LaMR-style line pruning at 1,000 / 500 | 50,987 / 69,073 | 3.44 % / 4.66 % |
| **R10** (outputs ≥ 300 tokens) | 198,386 reads saved − 32,069 rewrites = **166,317** | **11.21 %** |

### 7.2 R8 and R10 with the real gateway code 📄

The cache model reproduced L1's real bill within −0.5 %.

| Session | R8 only | R8 + R10 (≥ 300) |
|---|---|---|
| L1 (109 requests) | ±0 % | **−9.9 % input cost** |
| S4 / S5 (short) | −1.3 % / −1.2 % | −1.4 % / −1.5 % |

**Insight:**
- R10 is the biggest lever on long sessions and never made a run more expensive.
- R8 saves nothing by itself. Its value is avoiding the re-fetch loop.

Both were implemented on 2026-10-12, tests first. 145 tests pass.

---

## 8. L3: live check of R8 + R10 (same 12 tasks as L1) ✅

| Measure | L1 (before) | L3 (after) | Calculation |
|---|---|---|---|
| Requests | 107 | 86 | |
| Billed cache_read · write · uncached | 12,681,823 · 166,287 · 7,519 | 7,836,922 · 200,981 · 7,477 | |
| Input-equivalent, session | 1,483,560 | 783,692 + 251,226 + 7,477 = **1,042,395** | 1,042,395 / 1,483,560 − 1 = **−29.7 %** |
| Input-equivalent per request | 1,483,560 / 107 = 13,865 | 1,042,395 / 86 = 12,121 | 12,121 / 13,865 − 1 = **−12.6 %** |
| Dollars (Haiku formula) | **$0.753** | **$0.405** | 0.405 / 0.753 − 1 = **−46 %** |
| Requests with prompt > 100K | 81 of 107 | 36 of 86 | |
| Proxy tokens removed | 0.3 % | 756,687 / 5,920,371 = **12.8 %** | |
| R10 masks | – | 21 outputs at 7 task boundaries | |
| `git diff` | trimmed → 3 re-fetches | sent whole (R8) → reviewed in one go | |
| Bugs fixed | 3 / 3 | **3 / 3** | |
| Trimming time | median 2.5 ms, max 137 ms | median 26.8 ms, max 225 ms | |

**Insight:**
- **The per-request figure (−12.6 %) is the fair one.** The −29.7 % for the whole session is
  partly because L3 needed fewer requests, which is normal agent variation.
- The dollar saving is much larger because R10 kept prompts under Haiku's 100K price step.
- R8 worked: the full diff arrived whole and there was no re-fetch.
- **New problem found:** a pytest output trimmed into fragments made the agent re-run the
  tests (recommendation R13).

---

## 9. Step D: task success on 29 real bugs ✅ (`eval/results/stepd/`)

### Setup

- **Bugs:** 29 SWE-bench Lite sympy bugs, each validated first. The official hidden test
  fails before the official fix and passes after it.
- **Runs:** each bug was run twice with Claude Code (Haiku 5.5), in three-message sessions:
  investigate → fix → test and summarise.
  - **A** = shadow, nothing changed.
  - **B** = moderate + R8 + R10.
- **Total:** 58 runs, 1,561 logged requests, all ended normally.
- **Redo:** one run (18835/A) was redone after an internet outage. The failed run is kept in
  `runs_network_failed.csv` and not counted.

**Fixes made before the full run:**
1. **Cache order.** Whichever run went second found the cache warm. The fix is to report
   warm-start cost as well.
2. **Single prompts.** With one prompt per bug, R10 never acts. The fix is the three-message
   sessions.

### 9.1 Correctness

| | A (no TokenCut) | B (TokenCut) |
|---|---|---|
| Bugs fixed (official FAIL_TO_PASS + PASS_TO_PASS) | **18 / 29 = 62 %** | **19 / 29 = 66 %** |

- **Fixed only in A:** 18698, 20049.
- **Fixed only in B:** 21379, 21612, 22840.
- **Sign test on the 5 different bugs:** p = 2 × P(X ≤ 2 | n = 5) = 2 × (1 + 5 + 10) / 32
  = **1.0**.

**Insight:**
- There is **no sign that TokenCut lowers correctness**.
- With 29 bugs, a small difference could still go unnoticed.

### 9.2 Cost and tokens

| Measure | A | B | Calculation | Change |
|---|---|---|---|---|
| Total cost, all 29 bugs | $1.3794 | $1.2061 | sum of `cost_usd` over successful requests | −12.6 % |
| Cost per bug, raw | $1.3794 / 29 = $0.04757 | $1.2061 / 29 = $0.04159 | | **−12.6 %** |
| Cost per bug, warm start | $0.04569 | $0.03910 | per-bug B − A, mean −$0.0066; 95 % bootstrap CI **−$0.025 … +$0.005** | **−14.4 %** |
| B cheaper in | | | 14 of 29 bugs | |
| Cost per request, **average of per-bug averages** | $0.001628 | $0.001516 | 0.001516 / 0.001628 − 1 | **−6.9 %** |
| Cost per request, **pooled** | $1.3794 / 778 = $0.001773 | $1.2061 / 782 = $0.001542 | 0.001542 / 0.001773 − 1 | **−13.0 %** |
| Tokens sent per request (per-bug average) | 40.7k | 39.2k | | −3.7 % |
| Requests with prompt > 100K (Haiku price step) | **33** | **3** | count of `billed_prompt` > 100,000 | |
| Requests per bug | 26.8 | 27.0 | | +0.6 % |

**Why two "per request" figures:**
- **Pooled** divides total cost by total requests. It is pulled by one expensive A run: bug
  19254 cost $0.269 in 56 requests.
- **Average of per-bug averages** gives each bug equal weight.

We quote **−6.9 % as the safe figure** and −13.0 % as the pooled one. The true effect is
probably between them.

**Insight:**
- **Cost went down, but the per-bug saving is not statistically certain.** The interval
  includes zero, because the agent's number of steps varies a lot between runs.
- **The clearest effect is the price step:** 33 → 3 requests above 100K.

### 9.3 What TokenCut did in B

| Measure | Value and calculation |
|---|---|
| Bugs where TokenCut acted | 27 / 29 |
| Proxy tokens removed | 1,404,891 / 32,473,738 = **4.3 %** of what the agent sent |
| Re-reads after a trim or mask | **5** (21171 ×2, 18698, 20590, 21055) in 29 sessions |
| Trimming time per request | median 3.8 ms, p95 63.1 ms, max 234.9 ms (783 requests) |
| Wall time per bug | A 306 s, B 347 s (+13 %). This is caused by the agent's own slow test runs (20322/B ran tests for about 40 min), not by trimming. |

### 9.4 Which technique saved what ✅ (from the B runs' gateway logs)

**Method:** for every request, each trim records `before` and `after` tokens. Saved =
Σ (before − after), grouped by the rule that made it.

| Technique | Tokens saved | Share = saved ÷ 1,402,631 | Re-reads after it |
|---|---|---|---|
| **R10 task masking**: first mask 105,418 + same mask re-sent in later requests 1,164,305 | **1,269,723** | **90.5 %** | 4 |
| On-arrival code trimming | 60,063 | 4.3 % | 1 |
| On-arrival search trimming | 53,751 | 3.8 % | 0 |
| On-arrival command / log trimming | 19,094 | 1.4 % | 0 (but a re-run in L3) |

"Share" means share of the *saving*, not of all tokens sent.

---

## 10. Ranking of techniques (all tests combined)

| Rank | Technique | Best measured saving | Effect on quality | Evidence |
|---|---|---|---|---|
| 1 | **R10 task masking** | 90.5 % of step D saving; L3 −12.6 % per request, −46 % in dollars | held (19 vs 18; 3/3); 4 re-reads in 29 sessions | step D, L3, replay |
| 2 | **Tool pruning** | −64 % per request after a model switch; −29 % per session | held; risk of a missing tool | S4, S4-shadow, S5 |
| 3 | **R8 protection** | none itself | removed a measured re-fetch loop | L1 → L3 |
| 4 | On-arrival code trimming | 4.3 % of step D saving | 1 re-read | step D |
| 5 | On-arrival search trimming | 3.8 % | no issues | step D |
| 6 | On-arrival command / log trimming | 1.4 % | worst risk per token (re-run in L3) | step D, L3 |
| 7 | Cache-event history trimming | about 1–3k tokens per event | held | S4, S5 |
| – | Tier choice | about 1k tokens difference | aggressive slightly riskier | S4 vs S5 |

---

## 11. Corrections found during this verification

1. **Step D cost per request:** the earlier summaries gave only −6.9 %, which is the average
   of per-bug averages. The pooled figure is −13.0 %. Both are now shown (§9.2).
2. **R10 tokens in step D:** 1,269,723, not 1,271,983. The share is unchanged at 90.5 %. The
   total technique saving (1,402,631) is 2,260 tokens below the run total (1,404,891) because
   one request in a B run failed and is excluded from the per-technique sum.
3. **C1:** 28 POSTs with **1** rate-limited request, not 2. The cache-read share is 98.8 %, a
   more precise value than the earlier "about 99 %".
4. **L3 prompt median:** 93,472 tokens, not 93,898. This does not affect any conclusion.

No other headline number changed.

---

## 12. Where we stand

| Question | Status | Evidence |
|---|---|---|
| Does TokenCut keep the code that matters when trimming? | **Yes**, far better than naive cutting, level with BM25 / embeddings | §2.2 (81 held-out tasks, p < 0.001 vs head / random) |
| Can it pick files from a whole repo? | **Weak** (file recall about 62 %); not used by the gateway | §2.2, §4.3 |
| Does the agent still fix bugs through TokenCut? | **Yes**, no drop seen | step D 19/29 vs 18/29; live runs all fixed |
| Does it save money? | **Yes, with conditions**: 7–13 % per request (step D), 12.6 % / 46 % dollars (L3), 64 % after a model switch (with pruning) | §5.3, §8, §9 |
| Is the per-bug saving statistically certain? | **No** (CI includes 0) | §9.2 |
| Does on-arrival trimming help with Claude Code? | **Barely** (0–1 % live, 9.5 % of step D saving) | §5, §9.4 |
| Does it help with agents that read whole files? | **Not tested yet** | next step: Mini-SWE-Agent |
| Other providers? | **Not tested** (Codex format unsupported, Gemini blocked) | §5.5 |
| Is it fast enough? | **Yes for normal requests** (median 3.8 ms in step D); up to about 0.8 s on very large synthetic histories | §4.2, §9.3 |
| Main risk | **Re-reads after a trim or mask** (5 in 29 sessions; L1 diff loop; L3 test re-run) | §5.4, §8, §9.3 |
| Never exercised | context-full and cache-expired events | L1 peaked at 95k < 110k |

To re-check any number: `python -m eval.verify_results`. The detailed report is
[RESULTS_REPORT.md](RESULTS_REPORT.md), and the short version is
[PROGRESS_SUMMARY.md](PROGRESS_SUMMARY.md).
