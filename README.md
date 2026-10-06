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

## Usage

```python
from tokencut.ingestion.parser import parse_repo
from tokencut.graph.builder import build_graph
from tokencut.trimming.variants import generate_variants

chunks = parse_repo("fixtures/sample_repo")
graph = build_graph(chunks)
for v in generate_variants(chunks, graph, "Cart.total applies the discount wrong"):
    print(v.tier.value, v.token_count, "/", v.budget_tokens)
```

API: `uvicorn api.main:app --reload`, then `POST /context/variants` with
`{"repo_path": "...", "query": "...", "include_text": true}`.

Evaluation on SWE-bench Lite (clones repos into `swebench_repos/`, caches in `cache/`):

```
python -m eval.run_eval --limit 3       # smoke run
python -m eval.run_eval                 # all tasks in eval/tasks_iter1.json
```

Results go to `eval/results/iter1_results.csv` and `eval/results/iter1_summary.md`.

## Code map

| Module | What it does |
|---|---|
| `ingestion/walker.py` | file list (`git ls-files`, skip rules from DECISIONS #12) |
| `ingestion/python_parser.py` | tree-sitter chunker (module / class / function / method) |
| `graph/modules.py` | module names, import parsing, relative imports, `src/` layout |
| `graph/builder.py` | contains / imports / inherits / calls edges (DECISIONS #7) |
| `graph/queries.py` | `GraphIndex`: dependency closure, BFS distances |
| `trimming/query.py` | file / symbol / traceback mentions in the query |
| `trimming/scoring.py` | BM25 + graph-distance score |
| `trimming/packing.py` | full pool, tier budgets, nested greedy packing, skeletons |
| `trimming/render.py` | text sent to the model |
| `eval/loader.py`, `eval/metrics.py`, `eval/run_eval.py` | SWE-bench Lite subset, gold chunks, recall metrics |

## Ownership

| Folder | Owner |
|---|---|
| `tokencut/ingestion/`, `eval/` (SWE-bench loader) | Member A — Masooma Ali |
| `tokencut/graph/`, `api/` | Member B — Huma Taj |
| `tokencut/trimming/`, `eval/` (metrics) | Member C — Fatima Shakir |

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
