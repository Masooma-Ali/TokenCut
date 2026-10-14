# TokenCut — Testing Progress and Next Step

*As of 2026-10-14. Online copy: [Claude Doc](https://claude.ai/code/artifact/92667421-1f34-48c8-a54d-51afbdfa6212).*

## Summary

TokenCut keeps the coding agent correct while cutting cost: on 29 real SWE-bench bugs with Claude Code (Haiku 5.5), it fixed **19/29 vs 18/29 without TokenCut**, at about **7 % lower cost per request**.

- **Where the saving comes from:** R10 task masking (old outputs of a finished task become short notes) produced 90 % of the saved tokens and kept long sessions under Haiku's 100K price step (3 requests above it vs 33).
- **Longer sessions save more:** a 12-task session (L3) cost 12.6 % less per request and 46 % less in dollars than the same session without R10.
- **Main limitation:** Claude Code already reads files selectively, so TokenCut's original on-arrival trimming has little to do with it; its value should show more with agents that read whole files. That is the next step (Mini-SWE-Agent).

## What TokenCut is and what this phase was for

TokenCut is a gateway between a coding agent and the model provider that removes content the model does not need, without breaking the agent or the provider's prompt cache.

This phase was **testing, measuring and reporting**: how much TokenCut saves, whether the agent still works, how it compares with baselines and research, and what breaks. Team decisions kept: no ODE / model routing yet, no whole-repo indexing in the gateway, embeddings only offline, cache safety and fail-open always, numbers-only logs. Two improvements found by the measurements (R8 and R10) were implemented at a team member's request; everything else stays a recommendation.

## Work done

Every step below was measured; newest first.

| Date | Step | Result |
| --- | --- | --- |
| 2026-10-13/14 | **Step D** — 29 SWE-bench bugs, with and without TokenCut (Claude Code, Haiku 5.5) | fixed 19/29 vs 18/29; ~7 % cheaper per request; details below |
| 2026-10-13 | Other providers | Codex: request format not supported without code changes; Gemini CLI connected and parsed (0 errors) but Google blocked the free keys |
| 2026-10-12 | **L3** live — 12-task session with R8 + R10 | −12.6 % cost per request, −46 % in dollars vs L1; 3/3 bugs fixed; no re-reads |
| 2026-10-12 | **R8 and R10 implemented** (tests first, 145 tests) | R8: diffs and "full" requests never trimmed. R10: opt-in task masking with a cache break-even rule |
| 2026-10-12 | Replay of L1 on research methods (free) | R10 best (−11 % estimated); LaMR-style line pruning no better than R8 |
| 2026-10-11/12 | Research comparison (HCP, LongCodeZip, LLMLingua, SWE-Pruner, LaMR, ICAE, Aider) | none trims tool definitions or history, none is cache-aware; agent papers used open-source agents only |
| 2026-10-11 | **L1** live — 12-task session, moderate trimming | only 1 trim, a git diff, which the agent re-fetched in 3 extra requests (led to R8) |
| 2026-10-11 | **S1–S5** live on sympy (+ S4-shadow) | tool pruning at a model switch: −64 % per request after the switch, −29 % per session; bug still fixed |
| 2026-10-11 | **C1** live on the tiny playground | tool definitions 79 % of tokens; almost nothing to trim |
| 2026-10-11 | **Step B** offline analysis | skeletons almost never fit; 74 % of whole-repo losses come from file choice; gateway latency 40 ms → 0.8 s as history grows |
| 2026-10-11 | **Step A** reproduction | 136 tests pass; held-out benchmark reproduced exactly (every recall number identical) |

## Step D: task-success check

On 29 real bugs, TokenCut did not lower the fix rate and lowered cost per request by about 7 %.

**Setup.** 29 SWE-bench Lite sympy bugs, each checked first (official hidden test fails before the official fix, passes after). Each bug ran twice in Claude Code (Haiku 5.5): **A** = TokenCut in shadow mode (nothing changed), **B** = moderate trimming + R8 + R10. Each run is one session of three messages: investigate, fix, test and summarise. Graded with the official hidden tests. 58 runs in total.

**Fixed during the run.** (1) Whichever run went second found Claude Code's prompt already cached and looked cheaper, so costs are also reported as *warm-start cost* (first request billed as a cache read). (2) Single-prompt sessions gave R10 nothing to do, so each bug became a three-message session. One run damaged by an internet outage was redone.

| Measure | A (no TokenCut) | B (TokenCut) | Change |
| --- | --- | --- | --- |
| Bugs fixed | 18 / 29 | 19 / 29 | +1 (no significant difference) |
| Cost per request | $0.00163 | $0.00152 | −6.9 % |
| Cost per bug, warm start | $0.0457 | $0.0391 | −14.4 % (95 % CI includes 0) |
| Tokens sent per request | 40.7k | 39.2k | −3.7 % |
| Requests above Haiku's 100K price step | 33 | 3 | |
| Trimming time per request | – | median 3.8 ms, p95 63 ms | |
| Re-reads after a trim or mask | – | 5 in 29 sessions | |

The per-bug saving is not statistically certain because the agent itself varies a lot between runs; one baseline run alone cost $0.27.

## Which techniques matter

R10 task masking is TokenCut's most valuable technique with Claude Code: 90.5 % of the 1.40M tokens saved in step D. "Share" = part of the total saving, not of all tokens sent.

| Rank | Technique | Saving | Quality | When it applies |
| --- | --- | --- | --- | --- |
| 1 | **R10 task masking** — a finished task's big outputs become short notes | 90.5 % share; 33 → 3 requests over the price step; L3 −12.6 % per request | held (19 vs 18 fixed); 4 re-reads | every multi-message session |
| 2 | **Tool pruning** — drop unused tool definitions at a cache event | −64 % per request after a model switch (S4) | held; risk of a missing tool | model switches only (opt-in; not in step D) |
| 3 | **R8 protection** — never trim diffs or "full" requests | none itself | prevents a measured re-fetch loop | always on |
| 4 | On-arrival **code** trimming — relevant functions of a big Python file | 4.3 % share | 1 re-read | whole-file reads |
| 5 | On-arrival **search** trimming — shorten long grep / glob results | 3.8 % share | no issues | big search results |
| 6 | On-arrival **command / log** trimming — keep summary, errors, start | 1.4 % share | worst risk per token (test re-run in L3) | big command output |
| 7 | Cache-event **history** trimming | ~1–3k tokens per event | held | rare events only |

Tier choice (aggressive / moderate / conservative) made almost no difference. On-arrival trimming adds little with Claude Code because it reads small slices itself.

## Limitations and open questions

The results come from one agent, one model and one repository, so they show direction more than final numbers.

- **One setup:** Claude Code, Haiku 5.5, sympy, one run per bug and arm; more bugs or repeated runs are needed to make the cost saving statistically certain.
- **Re-read risk:** 5 re-reads after a trim or mask in 29 sessions; bug 18698 was fixed only without TokenCut and had one re-read, but one run cannot show cause.
- **Images and PDFs are not trimmed or counted** by TokenCut; no compared paper handles them either. Vision-model papers (GUIPruner, GUI-KV, ColPali) suggest reducing *old* screenshots and keeping only relevant PDF pages.
- **Other agents untested:** Codex sends a format TokenCut does not parse; Gemini CLI connected but Google blocked the free keys.
- **Context-full event** (~110K tokens) never fired, even in a 107-request session.

## Next step: TokenCut with Mini-SWE-Agent

Run the same kind of test with **Mini-SWE-Agent**, the open-source agent SWE-Pruner used, to test TokenCut's on-arrival trimming and compare directly with the research.

**Why this agent**

- **Comparable with the papers:** same agent and benchmark type as SWE-Pruner and LaMR (they report 23–31 % fewer tokens).
- **Tests the part Claude Code barely used:** it reads whole files with `cat` and prints full logs, which is what on-arrival code, search and log trimming are built for.
- **Supports the "any agent" claim:** a second, very different agent through the same gateway with no TokenCut changes.
- **Scriptable:** runs headless on SWE-bench tasks, so the step D scripts (bug checks, grading, logging) can be reused.

**What it needs**

- An **API key** (pay per token); the Claude Pro plan only works inside Claude Code. Haiku cost per bug was about $0.02–0.05 in step D.
- A setup check first: point its model calls (through LiteLLM) at TokenCut's `/anthropic` or `/openai/v1` route; not yet confirmed.
- R10 will rarely act (one prompt per bug), so this run tests on-arrival trimming and cache events.

**Plan**

1. Check the connection and estimate the API cost (free).
2. Team approves the budget.
3. Run the same 29 sympy bugs with and without TokenCut and compare with step D and the papers.

## Where everything is

All results and scripts are in the TokenCut repository; private logs stay local and are gitignored.

| What | Where |
| --- | --- |
| Full results report | `docs/RESULTS_REPORT.md` (step D = §7) |
| Live test scripts and task lists | `docs/LIVE_TEST_SCRIPT.md` |
| Design changes (R8, R10) | `docs/DECISIONS.md`, `tokencut/gateway/trimmer.py`, `history.py` |
| Step D scripts | `eval/stepd_validate.py`, `eval/stepd_run.py`, `eval/stepd_analyse.py` |
| Step D numbers | `eval/results/stepd/` (`analysis.md`, `runs.csv`, `requests.csv`) |
| Replay and viewer tools | `eval/replay_captures.py`, `eval/trim_viewer.py` |
| Private captures and transcripts | `logs/` (local only) |
| Branches | `trimming/researchpapers_methodology` (R8, R10; merged), `test/step-d` (step D) |
