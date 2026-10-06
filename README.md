# TokenCut

Cost-aware joint context-and-model routing gateway for AI coding agents (FYP, FAST NUCES Islamabad).

**Iteration 1:** repository parsing, code-graph construction, and 4-tier context-variant generation.

## Pipeline

```
repo ──► ingestion.parse_repo() ──► list[Chunk]
                                       │
                                       ▼
                         graph.build_graph() ──► CodeGraph
                                       │
                 query ───────────────►▼
                     trimming.generate_variants() ──► 4 × ContextVariant
```

All shared data formats live in `tokencut/contracts.py`. Changing that file needs a PR approved by both other members.

## Ownership

| Folder | Owner |
|---|---|
| `tokencut/ingestion/`, `eval/` (SWE-bench loader) | Member A — _name_ |
| `tokencut/graph/`, `api/` | Member B — _name_ |
| `tokencut/trimming/`, `eval/` (metrics) | Member C — _name_ |

## Setup (Windows PowerShell)

```powershell
git clone <repo-url>
cd tokencut
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

All tests in `tests/test_setup.py` must pass before you start coding.
The first run of the token counter downloads a small tokenizer file (needs internet once).

## Workflow

1. `git checkout main && git pull`
2. `git checkout -b feat/<short-name>`
3. Code + tests, run `pytest` and `ruff check .`
4. Commit small, clear messages; push; open a Pull Request
5. One teammate reviews and approves, then merge

## Test fixture

`fixtures/sample_repo/` is a tiny fake project with deliberate traps (syntax error,
circular import, name clash, binary file…). See its README.
