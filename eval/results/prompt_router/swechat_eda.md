# SWE-chat user prompts — EDA and preprocessing for the prompt router

Source: SWE-chat (SALT-NLP, COLM 2026; ODC-BY), all user prompts of `conversations.parquet`. Labels: SWE-chat's own LLM annotations (`prompt_intent`, `prompt_pushback`). Numbers only; prompt texts stay local.

## Population: 229,909 prompts, 17,560 sessions, 701 repositories

- machine-generated (slash commands, interrupts, harness / CI / sub-agent / approval messages, image-only): 62,360 (27.1%)
- typed by the user: 167,549 (72.9%)

| Intent (SWE-chat) | all prompts | user-typed | of which correction / rejection |
|---|---|---|---|
| other | 63,976 (27.8%) | 45,495 | 22.2% |
| understand | 36,536 (15.9%) | 29,208 | 36.0% |
| None | 28,433 (12.4%) | 1,250 | 0.5% |
| git | 25,381 (11.0%) | 23,732 | 37.6% |
| debug | 24,444 (10.6%) | 21,978 | 38.2% |
| create new code | 24,046 (10.5%) | 21,783 | 48.3% |
| refactor | 17,017 (7.4%) | 15,343 | 74.2% |
| test | 7,906 (3.4%) | 6,781 | 29.8% |
| connect | 2,170 (0.9%) | 1,979 | 56.2% |

User-typed prompts: median length 95 chars, p90 1560, p99 10717; 16.4% are ≤ 20 chars; first prompt of a session: 6.1%.
Agents: Claude Code 70.4%, Codex 19.9%, OpenCode 4.4%, Pi 2.1%, Copilot CLI 1.0%.

## Preprocessing funnel

| Step | Prompts left |
|---|---|
| all user prompts | 229,909 |
| minus machine-generated | 167,549 |
| minus non-English | 156,335 |
| minus unlabelled | 155,085 |
| minus exact duplicates (majority label kept) | 115,590 |

Training pool: 96,107 prompts in training repositories, 75,173 after at most 3,000 per repository (no per-session cap). Dev / test: at most 20 per session and 400 per repository, so one project cannot decide the score.

## Splits (whole repositories per split; train balanced, dev / test natural)

| Label | available | train | dev | test |
|---|---|---|---|---|
| bugfix | 8,456 | 5,000 | 292 | 340 |
| explain | 10,412 | 5,000 | 378 | 336 |
| feature | 6,954 | 5,000 | 314 | 254 |
| no_brief | 47,565 | 5,000 | 1,499 | 1,640 |
| refactor | 2,395 | 2,054 | 122 | 64 |
| run_action | 6,838 | 5,000 | 280 | 284 |
| tests | 2,763 | 2,434 | 115 | 82 |
| **total** | 85,383 | 29,488 | 3,000 | 3,000 |

Repositories: train 557, dev 50, test 82.

Label mapping: create new code → feature, debug → bugfix, refactor → refactor, test → tests, understand → explain, git / connect → run_action, other → no_brief; any correction / rejection → no_brief (steering inside a running task); failure_report keeps its intent ("still failing" asks for a fix).
