# SWE-chat user prompts — EDA and preprocessing for the prompt router

Source: SWE-chat (SALT-NLP, COLM 2026; ODC-BY), all user prompts of `conversations.parquet`. Labels: SWE-chat's own LLM annotations (`prompt_intent`, `prompt_pushback`). Numbers only; prompt texts stay local.

## Population: 229,909 prompts, 17,560 sessions, 701 repositories

- machine-generated (slash commands, interrupts, harness / CI / sub-agent / approval messages, image-only): 51,849 (22.6%)
- typed by the user: 178,060 (77.4%)

| Intent (SWE-chat) | all prompts | user-typed | of which correction / rejection |
|---|---|---|---|
| other | 63,976 (27.8%) | 48,465 | 20.9% |
| understand | 36,536 (15.9%) | 32,564 | 32.5% |
| None | 28,433 (12.4%) | 1,250 | 0.5% |
| git | 25,381 (11.0%) | 24,765 | 36.2% |
| debug | 24,444 (10.6%) | 23,138 | 36.5% |
| create new code | 24,046 (10.5%) | 22,465 | 47.2% |
| refactor | 17,017 (7.4%) | 15,660 | 73.5% |
| test | 7,906 (3.4%) | 7,616 | 27.0% |
| connect | 2,170 (0.9%) | 2,137 | 52.2% |

User-typed prompts: median length 103 chars, p90 1662, p99 10108; 15.4% are ≤ 20 chars; first prompt of a session: 5.7%.
Agents: Claude Code 66.2%, Codex 24.6%, OpenCode 4.1%, Pi 2.0%, Copilot CLI 1.0%.

## Preprocessing funnel

| Step | Prompts left |
|---|---|
| all user prompts | 229,909 |
| minus machine-generated | 178,060 |
| minus non-English | 166,846 |
| minus unlabelled | 165,596 |
| minus exact duplicates (majority label kept) | 125,573 |
| at most 20 per session | 79,721 |
| at most 400 per repository | 51,800 |

## Splits (whole repositories per split; train balanced, dev / test natural)

| Label | after cleaning | train | dev | test |
|---|---|---|---|---|
| bugfix | 5,440 | 4,000 | 314 | 345 |
| explain | 6,096 | 4,000 | 422 | 336 |
| feature | 4,876 | 3,904 | 303 | 283 |
| no_brief | 27,675 | 4,000 | 1,463 | 1,631 |
| refactor | 1,531 | 1,212 | 109 | 71 |
| run_action | 4,524 | 3,557 | 289 | 254 |
| tests | 1,658 | 1,342 | 100 | 80 |
| **total** | 51,800 | 22,015 | 3,000 | 3,000 |

Repositories: train 555, dev 52, test 82.

Label mapping: create new code → feature, debug → bugfix, refactor → refactor, test → tests, understand → explain, git / connect → run_action, other → no_brief; any correction / rejection → no_brief (steering inside a running task); failure_report keeps its intent ("still failing" asks for a fix).
