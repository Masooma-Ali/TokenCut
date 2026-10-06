# Design Decisions — Iteration 1

Fill this in together before writing module code. It feeds directly into report
§5.4 "Architecture Decisions and Rationale" and §6.7 "Algorithms and Technical Decisions".

| # | Decision | Options | Our choice | Why |
|---|---|---|---|---|
| 1 | Where does the code come from in the real gateway? | Repo on disk (local mode) / only code inside the request (hosted mode) / both | | |
| 2 | Chunk granularity and ID format | function + class + method + module-level; ID = `path::QualifiedName` | | |
| 3 | Tokenizer | tiktoken `o200k_base` as proxy for all providers | | |
| 4 | Tier budgets | Fraction of full context (draft: 15/35/60/100%) / fixed token budgets | | |
| 5 | What a dropped chunk becomes | Removed entirely / kept as signature + docstring "skeleton" | | |
| 6 | Language scope | Python only for Iteration 1, parser behind an interface | | |
| 7 | Call resolution | Heuristic (imports + same-module names), marked `resolved=False` | | |
| 8 | Determinism | Ties broken by chunk id; output sorted by path + line | | |
| 9 | Library versions | Pinned in `requirements.txt` | | |
| 10 | Evaluation subset | SWE-bench Lite, ~30–50 tasks from 3–4 repos | | |

## Change log
Record any later change here with date + reason (report §5.5 "Design Evolution").

| Date | Decision # | Change | Reason |
|---|---|---|---|
