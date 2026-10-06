# Design Decisions — Iteration 1

Feeds report §5.4 "Architecture Decisions and Rationale" and §6.7 "Algorithms and Technical Decisions".
Values marked (tunable) will be revisited after the first evaluation run.

| # | Decision | Our choice | Why |
|---|---|---|---|
| 1 | Source of code | Trimming core works only on chunks. Iteration 1 source = repo on disk; Iteration 2 adds chunks extracted from the agent's request. | Supports both local (individual) and hosted (team) gateway modes without rewriting the trimmer. |
| 2 | Chunk granularity and IDs | 4 kinds: module-level, class, function, method. A class chunk holds only the class header + class-level attributes; methods are separate chunks. Nested functions stay inside their parent. ID = `path::QualifiedName`; a repeated name in the same file gets `#2`, `#3`. | Prevents double-counting method tokens; IDs stay unique (e.g. property getter/setter share a name). |
| 3 | Tokenizer | tiktoken `o200k_base` as a proxy for all providers. | Tiers are relative (#4), so cross-provider count differences largely cancel out. Stated as a limitation. |
| 4 | Tier budgets (tunable) | Each tier is a fraction of the "full" context: aggressive 15%, moderate 35%, conservative 60%, full 100%. Aggressive has a 1,000-token floor. "Full" = what the agent sent (gateway) or top-ranked files up to 32k tokens (evaluation). | A whole repo exceeds any context window; the original SWE-bench paper likewise retrieved files up to fixed context limits. |
| 5 | Dropped chunks | In aggressive and moderate tiers, a dropped chunk directly connected to a kept chunk stays as a skeleton (signature + docstring); all others are removed. Skeleton tokens count toward the budget. | The model still knows which functions exist and their arguments. Close to HCP (AAAI 2025), which keeps declarations and prunes function bodies; we keep skeletons only for chunks connected to kept code. |
| 6 | Language scope | Python only; parser behind a `LanguageParser` interface. | SWE-bench is Python; other languages can be added without touching graph/trimming. |
| 7 | Call resolution | Rules in order: (1) imported names, (2) `self.method()` within the same class, (3) bare names within the same file, (4) a unique repo-wide name match with `resolved=False`; otherwise no edge. No edges to built-ins. | Simple and fast. Type-aware tools (jedi/pyright) only if recall turns out poor. |
| 8 | Determinism | Ties broken by chunk ID; output sorted by file path then line; never iterate a Python `set` when producing output. | Python's set order changes between runs; this would cause random test failures. |
| 9 | Versions | Python 3.11; exact versions pinned in `requirements.txt`. | Identical behaviour on all three laptops (Windows + Linux) and in CI. |
| 10 | Evaluation subset (tunable) | ~40 SWE-bench Lite tasks from `requests`, `flask`, `pytest`, `django`. Metrics: token reduction per tier, gold-file recall, gold-chunk recall (patch line numbers matched to chunk line ranges), runtime. | Mix of small, medium and large repos; gold-chunk recall is stronger evidence than file recall alone. |
| 11 | Query | Evaluation: issue text. Gateway: latest user message + recent tool outputs. Always a single `query: str`. | Same scorer interface in both settings. |
| 12 | Files skipped | `.gitignore` matches, non-Python files, files over 1 MB, `migrations/`, `vendor/`. `tests/` folders are kept. | Tests are sometimes needed to understand or fix the bug. |

## Change log
Record any later change here (report §5.5 "Design Evolution").

| Date | Decision # | Change | Reason |
|---|---|---|---|