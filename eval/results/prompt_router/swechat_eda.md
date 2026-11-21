# SWE-chat user prompts — EDA and preprocessing for the prompt router

Source: SWE-chat (SALT-NLP, COLM 2026; ODC-BY), all user prompts of `conversations.parquet`. Labels: SWE-chat's own LLM annotations (`prompt_intent`, `prompt_pushback`). Numbers only; prompt texts stay local.

## Population: 229,909 prompts, 17,560 sessions, 701 repositories

- machine-generated (slash commands, interrupts, harness / CI / sub-agent / approval messages, image-only): 67,493 (29.4%)
- typed by the user: 162,416 (70.6%)

| Intent (SWE-chat) | all prompts | user-typed | of which correction / rejection |
|---|---|---|---|
| other | 63,976 (27.8%) | 43,352 | 23.3% |
| understand | 36,536 (15.9%) | 27,975 | 37.5% |
| None | 28,433 (12.4%) | 1,250 | 0.5% |
| git | 25,381 (11.0%) | 22,911 | 38.8% |
| debug | 24,444 (10.6%) | 21,444 | 39.1% |
| create new code | 24,046 (10.5%) | 21,738 | 48.4% |
| refactor | 17,017 (7.4%) | 15,242 | 74.6% |
| test | 7,906 (3.4%) | 6,541 | 30.8% |
| connect | 2,170 (0.9%) | 1,963 | 56.7% |

User-typed prompts: median length 93 chars, p90 1528, p99 11107; 16.9% are ≤ 20 chars; first prompt of a session: 6.3%.
Agents: Claude Code 72.6%, Codex 17.3%, OpenCode 4.5%, Pi 2.2%, Copilot CLI 1.1%.

## Preprocessing funnel

| Step | Prompts left |
|---|---|
| all user prompts | 229,909 |
| minus machine-generated | 162,416 |
| minus non-English | 151,202 |
| minus unlabelled | 149,952 |
| minus exact duplicates (majority label kept) | 110,635 |

Training pool: 93,673 prompts in training repositories, 74,824 after at most 3,000 per repository (no per-session cap). Dev / test: at most 20 per session and 400 per repository, so one project cannot decide the score.

## Splits (whole repositories per split; train balanced, dev / test natural)

| Label | available | train | dev | test |
|---|---|---|---|---|
| bugfix | 13,585 | 5,000 | 463 | 570 |
| explain | 16,543 | 5,000 | 564 | 505 |
| feature | 13,861 | 5,000 | 502 | 563 |
| no_brief | 14,933 | 5,000 | 463 | 483 |
| refactor | 9,145 | 5,000 | 374 | 318 |
| run_action | 12,901 | 5,000 | 494 | 452 |
| tests | 4,066 | 3,611 | 140 | 109 |
| **total** | 85,034 | 33,611 | 3,000 | 3,000 |

Repositories: train 557, dev 50, test 82.

Label mapping: create new code → feature, debug → bugfix, refactor → refactor, test → tests, understand → explain, git / connect → run_action, other → no_brief. Corrections / rejections keep their intent (the text says what is asked); the gateway decides whether to repeat a brief, from the conversation.
