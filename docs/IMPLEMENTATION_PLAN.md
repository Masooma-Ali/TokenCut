# TokenCut — Iteration 1 Implementation Plan

> **Audience:** any team member or coding agent starting work on this repository.
> Read this file, `tokencut/contracts.py` and `docs/DECISIONS.md` before writing code.
> If anything here conflicts with `contracts.py`, `contracts.py` wins — then open an issue so this file gets fixed.

---

## 0. Quick orientation (read in 2 minutes)

**Project:** TokenCut is a final-year project at FAST NUCES Islamabad. It is a gateway that sits between an AI coding agent (Claude Code, Cursor, Cline…) and the model provider. For each request it decides **how much code context to keep** and **which model to use**, to cut API cost without hurting task quality. The decision-maker (Iteration 2) is **ODE**, a LinUCB contextual bandit choosing jointly over *(context tier × model)*.

**Iteration 1 builds only the context side:**

```
repo on disk ──► [A] parse_repo() ──► list[Chunk]
                                          │
                                          ▼
                             [B] build_graph() ──► CodeGraph
                                          │
                     query ──────────────►▼
                          [C] generate_variants() ──► 4 × ContextVariant
                                                      (aggressive, moderate, conservative, full)
```

**No model API calls in Iteration 1.** Quality is measured offline on SWE-bench Lite by checking whether each tier still contains the code that the real fix edited.

**Team and ownership**

| Member | Area | Folders |
|---|---|---|
| A — Masooma Ali | Ingestion (parsing, chunking) + SWE-bench loader | `tokencut/ingestion/`, `eval/loader*`, `fixtures/expected_chunks.json` |
| B — Huma Taj | Code graph + API endpoint | `tokencut/graph/`, `api/`, `fixtures/expected_graph.json` |
| C — Fatima Shakir | Scoring, packing, rendering + evaluation metrics | `tokencut/trimming/`, `eval/metrics*` |

Shared, change only via PR approved by both others: `tokencut/contracts.py`, `tokencut/tokens.py`, `requirements.txt`, `pyproject.toml`, `docs/DECISIONS.md`.

---

## 1. Ground rules for anyone writing code here

1. **Stay in your area.** Only edit files in the folder of the task you are working on. If you need something from another area, use the hand-written fixture JSON (section 3) or open an issue.
2. **Contracts are law.** All data passed between modules uses the pydantic classes in `tokencut/contracts.py`. Never pass raw dicts or invent new fields. Proposing a change → separate PR.
3. **Python 3.11**, `pathlib` for paths, and **every stored path uses forward slashes** (`shop/cart.py`), relative to the repo root. Windows and Linux must produce identical output.
4. **Deterministic output.** Same input → byte-identical output. Sort everything; break ties by chunk id; never build output by iterating a `set` or `dict` whose order depends on hashing.
5. **Count tokens only** with `tokencut.tokens.count_tokens`.
6. **Tests with every function**, including edge cases. Use `fixtures/sample_repo/`; generate very large or weird files inside tests with pytest's `tmp_path`.
7. **Before every commit:** `ruff check . --fix` and `pytest`. Both must pass. CI runs both.
8. **Small PRs**, one task each, branch name `feat/<area>-<task>`. Put `Closes #<issue>` in the PR description when an issue exists. One teammate reviews before merge.
9. **Keep code simple.** Students must explain every line in a viva. Prefer clear loops over clever one-liners; add a short comment where a rule from `DECISIONS.md` is applied (e.g. `# DECISIONS #7 rule 2`).
10. **No new dependencies** without a PR updating `requirements.txt` (pinned).
11. **AI disclosure.** Note AI-assisted parts in the PR description (one line is enough). The report needs this.

---

## 2. Shared data contracts (summary)

Full definitions: `tokencut/contracts.py`.

| Class | Produced by | Key fields |
|---|---|---|
| `Chunk` | A | `id`, `file_path`, `qualified_name`, `kind`, `parent_id`, `start_line`, `end_line`, `code`, `signature`, `docstring`, `token_count`, `file_hash` |
| `Edge` / `CodeGraph` | B | `source`, `target`, `kind` (contains/imports/calls/inherits), `resolved` |
| `ScoredChunk` | C | `chunk_id`, `score`, `structural_score`, `text_score` |
| `ContextVariant` | C | `tier`, `budget_tokens`, `kept_chunk_ids`, `skeleton_chunk_ids`, `token_count`, `rendered_text` |

Shared settings in `contracts.py`: `EVAL_FULL_CONTEXT_CAP_TOKENS` (32k), `AGGRESSIVE_MIN_TOKENS` (1k), `MAX_FILE_BYTES` (1 MB), `SKIP_DIR_NAMES`, `TIER_BUDGET_FRACTION` (15/35/60/100 %), `SKELETON_TIERS` (aggressive, moderate).

### Precise meaning of a chunk (all members rely on this)

* **Every parsed `.py` file produces exactly one `MODULE` chunk** (`qualified_name="<module>"`), even if empty (`code=""`, `token_count=0`, `start_line=end_line=1`). The graph uses it as the target of import edges.
* `MODULE` chunk `code` = all top-level lines that are **not** inside a top-level function or class definition (imports, constants, docstring, top-level statements, and any `ERROR` regions from syntax errors), in original order, joined with `\n`. `start_line`/`end_line` = first/last such line.
* `FUNCTION` = top-level `def` / `async def` (including its decorators). Nested functions and nested classes stay inside it.
* `CLASS` = the class's decorators + header + class body **excluding its methods**. Its `start_line`/`end_line` still span the whole class. Nested classes stay inside the class chunk.
* `METHOD` = a `def` directly inside a class body (with decorators). `qualified_name="Class.method"`, `parent_id` = the class chunk id.
* **`code` contains complete original source lines, including leading indentation.** So rendering = concatenating chunk code in line order; no re-indenting needed.
* `signature` = decorators + header up to and including the `:` (e.g. `def total(self) -> float:`). Empty for module chunks.
* `docstring` = the string literal value if the body's first statement is a string, else `""`.
* `id = "<file_path>::<qualified_name>"`. If the same `qualified_name` appears again in the same file (e.g. property getter + setter), the 2nd gets `#2`, 3rd `#3`, in line order. `qualified_name` itself never gets the suffix.
* Order of `parse_repo()` output: by `file_path`, then `start_line`, then `id`.

---

## 3. Working in parallel: the fixture files

The pipeline is sequential (A → B → C) but development is parallel thanks to two **hand-written** expected-output files:

| File | Written by | Used by |
|---|---|---|
| `fixtures/expected_chunks.json` | A (first task) | A's tests (expected output), B's input until A is done |
| `fixtures/expected_graph.json` | B (first task) | B's tests (expected output), C's input until B is done |

Format: JSON list of `Chunk` objects / one `CodeGraph` object, loadable with
`[Chunk.model_validate(x) for x in json.load(f)]` and `CodeGraph.model_validate(json.load(f))`.

Add a tiny helper in `tests/conftest.py` (A or B, whoever is first) exposing pytest fixtures `sample_repo_path`, `expected_chunks`, `expected_graph`.

### Expected chunks for `fixtures/sample_repo` (30 chunks)

| File | Chunks |
|---|---|
| `shop/__init__.py` | `<module>` (empty) |
| `shop/cart.py` | `<module>`, `validate`, `Cart`, `Cart.__init__`, `Cart.add_item`, `Cart.subtotal`, `Cart.total` |
| `shop/models.py` | `<module>`, `Product`, `Product.__init__`, `Customer`, `Customer.__init__`, `Customer.discount_rate`, `PremiumCustomer`, `PremiumCustomer.discount_rate` |
| `shop/payments/__init__.py` | `<module>` (empty) |
| `shop/payments/gateway.py` | `<module>`, `validate`, `process_payment` |
| `shop/payments/refunds.py` | `<module>`, `log_failure`, `refund` |
| `shop/utils/__init__.py` | `<module>` (empty) |
| `shop/utils/pricing.py` | `<module>`, `apply_discount`, `add_tax` |
| `shop/utils/broken.py` | `<module>` (docstring + the ERROR region + stray `return`), `fine_function`, `another_fine_function` |

Not chunked: `README.md`, `config.yaml`, `shop/logo.bin`.

### Key expected edges (B must produce at least these; nothing contradicting them)

| Source | Kind | Target | resolved |
|---|---|---|---|
| `shop/models.py::PremiumCustomer` | inherits | `shop/models.py::Customer` | true |
| each class | contains | each of its methods | true |
| `shop/cart.py::<module>` | imports | `shop/models.py::<module>` | true |
| `shop/cart.py::<module>` | imports | `shop/utils/pricing.py::<module>` | true |
| `shop/payments/gateway.py::<module>` | imports | `shop/cart.py::<module>` | true |
| `shop/payments/gateway.py::<module>` | imports | `shop/payments/refunds.py::<module>` | true |
| `shop/payments/refunds.py::<module>` | imports | `shop/payments/gateway.py::<module>` | true (circular) |
| `shop/cart.py::Cart.add_item` | calls | `shop/cart.py::validate` | true (rule 3) |
| `shop/cart.py::Cart.total` | calls | `shop/cart.py::Cart.subtotal` | true (rule 2) |
| `shop/cart.py::Cart.total` | calls | `shop/utils/pricing.py::apply_discount` | true (rule 1) |
| `shop/cart.py::Cart.total` | calls | `shop/utils/pricing.py::add_tax` | true (rule 1) |
| `shop/payments/gateway.py::process_payment` | calls | `shop/payments/gateway.py::validate` | true (rule 3 — **not** cart's `validate`) |
| `shop/payments/gateway.py::process_payment` | calls | `shop/payments/refunds.py::log_failure` | true (rule 1, module alias) |
| `shop/payments/gateway.py::process_payment` | calls | `shop/cart.py::Cart.total` | **false** (rule 4, unique name) |
| `shop/payments/refunds.py::refund` | calls | `shop/payments/gateway.py::validate` | true (rule 1, module alias) |
| `shop/payments/refunds.py::refund` | calls | `shop/cart.py::Cart.total` | **false** (rule 4) |
| `shop/utils/broken.py::another_fine_function` | calls | `shop/utils/broken.py::fine_function` | true (rule 3) |

Must **not** exist: any edge to built-ins (`print`, `sum`, `len`, `ValueError`…); any edge for `self.customer.discount_rate()` (name `discount_rate` is not unique → no edge); any edge to a chunk id that does not exist.

---

## 4. Member A — Ingestion (`tokencut/ingestion/`)

### Files

```
tokencut/ingestion/
  walker.py      # list_python_files(repo_path) -> list[Path]
  python_parser.py  # PythonParser.parse_file(rel_path, source: bytes) -> list[Chunk]
  base.py        # LanguageParser Protocol (DECISIONS #6)
  parser.py      # parse_repo(repo_path) -> list[Chunk]   (public entry point, already stubbed)
```

### A1. `fixtures/expected_chunks.json` (do first — B and C wait on it)
Write the 30 chunks of section 3 by hand. Use real line numbers, real code text (full lines with indentation), real `token_count` (compute with `count_tokens` in a Python shell) and real `file_hash` (`hashlib.sha256(path.read_bytes()).hexdigest()`).

### A2. File walker (DECISIONS #12)
1. If `repo_path` is inside a git repo, get the file list with `git ls-files` run with `cwd=repo_path` (respects `.gitignore`, deterministic). Otherwise (or if git fails) fall back to `os.walk` sorted.
2. Keep only `*.py`. Drop any path containing a directory in `SKIP_DIR_NAMES`.
3. Drop files larger than `MAX_FILE_BYTES`. Drop files containing a NUL byte (binary).
4. Return paths sorted, relative to `repo_path`, converted with `.as_posix()`.

### A3. Parser (tree-sitter)
Setup (tree-sitter ≥ 0.22 API):
```python
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)
tree = parser.parse(source_bytes)
```
Algorithm for one file:
1. `file_hash = sha256(source_bytes)`. Decode text as UTF-8; on failure decode as latin-1 (keeps line numbers).
2. Iterate `tree.root_node.children`. For each node, unwrap `decorated_definition` → its `definition` field (but the chunk's line range starts at the decorator).
   * `function_definition` → FUNCTION chunk.
   * `class_definition` → CLASS chunk + one METHOD chunk per `function_definition` / `decorated_definition` directly in its `body` block.
   * anything else (imports, assignments, expression statements, `ERROR` nodes, stray statements) → its lines go to the MODULE chunk.
3. Line numbers: tree-sitter rows are 0-based → `start_line = node.start_point[0] + 1`, `end_line = node.end_point[0] + 1`.
4. `code` = the original source lines `start_line..end_line` (full lines). For CLASS chunks remove the line ranges of its methods.
5. `signature` = source text from chunk start up to the end of the header (everything before the `body` node), stripped of trailing whitespace.
6. `docstring`: if the body's first child is an `expression_statement` whose only child is a `string`, take its value (use `ast.literal_eval` inside `try`, fall back to stripping quotes).
7. Apply the duplicate-id suffix rule. `token_count = count_tokens(code)`.
8. Never raise on bad Python. A file that fails completely → only its MODULE chunk with whatever text exists; log a warning.

### A4. `parse_repo(repo_path)`
Walk → parse each file → concatenate → sort (file_path, start_line, id) → return. Optional (stretch): cache results in `cache/<repo_name>/<file_hash>.json` (folder is gitignored).

### A5. SWE-bench Lite loader (`eval/loader.py`)
* Add `datasets` to `requirements.txt` (pinned) in its own PR.
* Load `princeton-nlp/SWE-bench_Lite`, split `test`. Fields used: `instance_id`, `repo`, `base_commit`, `problem_statement`, `patch`.
* Task selection (deterministic, sorted by `instance_id`): all tasks from `psf/requests` and `pallets/flask`, then fill from `pytest-dev/pytest` and `django/django` (roughly equal) up to **40 tasks**. Save the chosen ids in `eval/tasks_iter1.json` and commit it.
* Repos: clone each once into `swebench_repos/<owner>__<name>` (gitignored); for each task `git checkout -f <base_commit>`. Process tasks grouped by repo to avoid re-cloning.
* Gold data from `patch` using `unidiff.PatchSet`:
  * `gold_files` = modified files that existed before the patch (skip newly added files).
  * `gold_lines[file]` = original-file line numbers of removed lines; for hunks that only add lines, use the hunk's source range `source_start .. source_start + max(source_length,1) - 1`.
* Output: one record per task, a pydantic model or dataclass in `eval/` with `instance_id, repo, repo_path, base_commit, query (= problem_statement), gold_files, gold_lines`.

### A tests (minimum)
* `parse_repo(sample_repo)` equals `expected_chunks.json` (compare ids, kinds, line ranges, signatures; code exact).
* broken.py: no crash, `fine_function` and `another_fine_function` present.
* empty file → one empty MODULE chunk.
* class chunk excludes method bodies; sum of tokens is not double-counted.
* duplicate names (`@property` + `@x.setter`) → `#2` suffix (write the case in `tmp_path`).
* non-UTF-8 file, file > 1 MB, `.py` file with NUL byte, file inside `migrations/` → skipped or handled as specified.
* path is forward-slash on Windows (assert `"\\" not in chunk.file_path`).
* running twice gives identical output.

---

## 5. Member B — Code graph (`tokencut/graph/`) and API (`api/`)

### Files

```
tokencut/graph/
  modules.py     # file path <-> dotted module name; import parsing & resolution
  builder.py     # build_graph(chunks) -> CodeGraph   (stubbed)
  queries.py     # dependency_closure(), neighbours_within(), to_networkx()
api/
  main.py        # FastAPI app
```

Use `networkx` internally if helpful, but the public output is always `CodeGraph`. Re-parse chunk `code` with tree-sitter to find imports and calls (robust to syntax errors).

### B1. `fixtures/expected_graph.json` (do first — C waits on it)
All edges for the sample repo following section 3 and DECISIONS #7. Sort edges by `(source, target, kind)`.

### B2. Module map
* `shop/cart.py` → `shop.cart`; `shop/payments/__init__.py` → `shop.payments`.
* Also register the name without a leading `src/` or `lib/` (pytest uses `src/_pytest/...`).
* Build `module_name -> module chunk id` and `module_name -> {top-level name -> chunk id}` (functions + classes in that file).

### B3. CONTAINS and INHERITS
* CONTAINS: for every METHOD, edge `parent_id → method id`.
* INHERITS: for every CLASS, resolve each base class name with the same rules as calls (B5, rules 1, 3, 4). Unresolvable bases (e.g. `Exception`, external libs) → no edge.

### B4. IMPORTS and the per-file name table
For each file, read import statements from its MODULE chunk:
* `import a.b` / `import a.b as z` → if `a.b` is a repo module: edge module→module; name table: `z` (or `a`) → module alias.
* `from a.b import y [as z]`: if `a.b.y` is a repo module → edge to that module, name `y/z` → module alias; else if `y` is a top-level name in module `a.b` → edge module→`a.b` module, name `y/z` → that chunk id; else if `a.b` is a repo module → edge to it only.
* Relative imports (`from . import x`, `from ..a import y`) resolve against the current file's package.
* External imports (not in repo) → ignored. `from x import *` → edge to module only.
* Imports inside functions: add them to that function's local name table (stretch; fine to skip in Iteration 1, note it as a limitation).
* Circular imports are just two edges — no recursion, so no infinite loop.

### B5. CALLS (DECISIONS #7)
For each FUNCTION/METHOD chunk, find every `call` node; look at its `function` field:
1. **Identifier `foo`:** if `foo` is in the file's import name table → that chunk (rule 1). Else if `foo` is a top-level function/class in the same file → that chunk (rule 3). Else go to rule 4.
2. **`self.foo` / `cls.foo` inside a method:** `Class.foo` in the same class (rule 2). If absent, try the class's resolved base classes (one level). Else rule 4.
3. **`alias.foo` where `alias` is an imported module:** `foo` in that module (rule 1).
4. **Any other attribute call `x.foo` and unresolved names:** if exactly one function/method/class in the whole repo has last-name `foo` → edge with `resolved=False` (rule 4). Otherwise no edge.
* Skip names in `dir(builtins)` unless resolved by rules 1–3.
* Calling a class (`Product(...)`) → edge to the class chunk.
* No self-edges; deduplicate; sort.

### B6. Queries (`queries.py`)
* `dependency_closure(graph, chunk_ids, max_hops=1) -> set[str]`: start set + everything reachable within `max_hops` along **outgoing** CALLS and INHERITS edges, plus for each METHOD its parent CLASS, plus the MODULE chunk of each included chunk's file. Callers must sort before output.
* `neighbours_within(graph, chunk_id, k) -> dict[str, int]`: undirected BFS distances (all edge kinds) up to `k` hops.
* `multi_source_distances(graph, sources) -> dict[str, int]`: undirected BFS from many sources at once (used by C's structural score).

### B7. API (`api/main.py`)
* `GET /health` → `{"status": "ok"}`.
* `POST /context/variants` body `{"repo_path": str, "query": str, "include_text": bool = true}` → `{"full_tokens": int, "variants": [ContextVariant...]}` (omit `rendered_text` when `include_text=false`).
* Errors: missing path → 400; no Python files → 422 with message.
* Run: `uvicorn api.main:app --reload`. Test with FastAPI `TestClient`.

### B tests (minimum)
* `build_graph(expected_chunks)` equals `expected_graph.json`.
* every edge's source and target exist in `chunk_ids`.
* the two `validate` functions are never confused.
* circular imports terminate and produce both edges.
* `src/` layout module resolution (build a small case in `tmp_path`).
* `dependency_closure` on `Cart.total` includes `Cart.subtotal`, `apply_discount`, `add_tax`, class `Cart`, `shop/cart.py::<module>`.
* deterministic output across two runs.
* API: `/health` 200; `/context/variants` on sample repo returns 4 variants; bad path → 400.

---

## 6. Member C — Scoring, packing, rendering (`tokencut/trimming/`) and metrics (`eval/`)

### Files

```
tokencut/trimming/
  query.py       # analyse_query(query, chunks) -> QueryMentions
  scoring.py     # score_chunks(chunks, graph, query) -> list[ScoredChunk]
  packing.py     # build_pool(), pack_tiers()
  render.py      # render_variant(...) -> str
  variants.py    # generate_variants(...)   (public entry point, already stubbed)
eval/
  metrics.py     # per-task, per-tier metrics -> CSV
  run_eval.py    # CLI: loader -> pipeline -> metrics -> eval/results/
```

Until A and B are done, load chunks and graph from the fixture JSON files.

### C1. Query analysis (`query.py`) — no dependencies, start here
Return a small dataclass `QueryMentions(file_paths: list[str], symbols: list[str], chunk_ids: list[str])`:
* File paths: regex for `[\w./-]+\.py`; match to chunks whose `file_path` ends with it.
* Traceback lines `File "…/x.py", line N, in func` → file + function name (strong signal).
* Symbols: identifiers and dotted names (`Cart.total`, `apply_discount`, `CamelCase`) of length ≥ 3, excluding a small English stop-word list; match against chunk `qualified_name` or its last component.
* `chunk_ids` = all matched chunks, sorted. Symbol matches outrank file-only matches (keep a weight: symbol = 1.0, file = 0.5).

### C2. Scoring (`scoring.py`)
* **Tokenisation for BM25:** split code and query into identifiers, then split snake_case and CamelCase into lowercase sub-words (`apply_discount` → `apply`, `discount`). Each chunk document = `file_path + qualified_name + code`.
* `text_score`: `rank_bm25.BM25Okapi` score, normalised to [0, 1] by dividing by the max (0 if max is 0).
* `structural_score`: multi-source BFS distance `d` (undirected) from the mentioned chunks → `weight / (1 + d)`; unreachable → 0. **If the query mentions nothing,** use the top 3 chunks by `text_score` as pseudo-mentions (document this fallback).
* `score = ALPHA * structural_score + (1 - ALPHA) * text_score`, with `ALPHA = 0.5` as a module constant (tunable; log changes in DECISIONS change log).
* Return sorted by `score` desc, ties by `chunk_id` asc.

### C3. The "full" pool and tier budgets (`packing.py`, DECISIONS #4)
* `build_pool(chunks, scores, full_cap_tokens)`:
  * `full_cap_tokens is None` (gateway mode) → pool = all chunks.
  * Otherwise (evaluation) rank **files** by their best chunk score; add whole files in that order while total tokens ≤ `full_cap_tokens` (skip a file that does not fit and continue with smaller ones). Pool = all chunks of selected files.
* `full_tokens` = token count of the rendered FULL variant.
* Budget per tier = `round(TIER_BUDGET_FRACTION[tier] * full_tokens)`; aggressive gets `max(budget, AGGRESSIVE_MIN_TOKENS)`; every budget capped at `full_tokens`.

### C4. Packing — nested tiers (guarantees monotonic sizes)
Process tiers in order **aggressive → moderate → conservative → full**. Each tier starts with the previous tier's kept set (so tiers are nested and token counts never decrease):
1. Walk pool chunks by score (desc, tie by id). For a candidate `c` not yet kept: `group = ({c} ∪ dependency_closure(graph, {c}, 1)) ∩ pool − kept`.
2. Cost = sum of chunk tokens + a header cost (`count_tokens(header)`) for each file not yet present.
3. If the whole group fits the remaining budget → keep all; else if `c` alone (+ its parent class and file module chunk if missing) fits → keep those; else skip `c`. **Continue** through the whole list (smaller chunks may still fit).
4. **Skeletons** (only tiers in `SKELETON_TIERS`): candidates = pool chunks not kept, of kind CLASS/FUNCTION/METHOD, directly connected (any edge, either direction) to a kept chunk. In score order, add `signature + docstring` skeleton if it fits.
5. FULL tier = the whole pool, no skeletons.
6. Render (C5), then **verify** `count_tokens(rendered_text) ≤ budget`. If it exceeds (token merging can shift counts slightly), drop the lowest-scored non-mandatory kept chunk and re-render until it fits.

### C5. Rendering (`render.py`)
```
# File: shop/cart.py
<chunks of this file in start_line order: full code, or skeleton>

# File: shop/utils/pricing.py
...
```
* Files sorted by path; chunks by `start_line`.
* Skeleton text = the chunk's `signature` line(s) + its docstring (indented as in source) + an indented `...` line.
* A kept METHOD always appears after its CLASS chunk (closure guarantees the class is present).
* `ContextVariant.token_count = count_tokens(rendered_text)`; `kept_chunk_ids` / `skeleton_chunk_ids` sorted by (file_path, start_line).

### C6. `generate_variants(chunks, graph, query, full_cap_tokens=None) -> list[ContextVariant]`
analyse → score → pool → pack (4 tiers) → render → return in tier order.

### C7. Evaluation metrics (`eval/metrics.py`, `eval/run_eval.py`)
For each task × tier, one CSV row in `eval/results/iter1_results.csv`:

| Column | Meaning |
|---|---|
| `instance_id`, `repo`, `tier` | identifiers |
| `repo_tokens` | total tokens of all chunks in the repo |
| `full_tokens`, `variant_tokens` | size of the 32k pool and of this tier |
| `reduction_vs_full` | `1 - variant_tokens / full_tokens` |
| `pool_has_gold_file` | did the 32k pool contain any gold file? (separates retrieval misses from trimming misses) |
| `gold_file_recall` | share of gold files with ≥ 1 kept (full-code) chunk |
| `gold_chunk_recall` | share of gold chunks kept as full code |
| `gold_chunk_skeleton_recall` | share of gold chunks kept as full code **or** skeleton |
| `t_parse_s`, `t_graph_s`, `t_variants_s` | runtimes in seconds |

* **Gold chunk** = for each gold line, the innermost chunk whose line range contains it (METHOD beats CLASS; a line not inside any def → that file's MODULE chunk).
* Also write `eval/results/iter1_summary.md`: mean of each metric per tier, overall and per repo. This table goes into the report.
* Cache parsed chunks per `(repo, base_commit)` under `cache/` so re-runs are fast.

### C tests (minimum)
* query analysis: file mention, function mention, traceback, empty query.
* scoring deterministic; on sample repo the query *"Cart.total applies the discount wrong"* ranks `Cart.total` first and `apply_discount` above `log_failure`.
* every variant: `token_count ≤ budget_tokens`; sizes non-decreasing across tiers; aggressive kept ⊆ moderate kept ⊆ conservative kept ⊆ full.
* budget smaller than any chunk → valid (possibly skeleton-only) variant, no crash.
* empty graph / no mentions → falls back to text score.
* a kept method's class header is always rendered.
* metrics: hand-made gold data gives the expected recall numbers.

---

## 7. Sequencing and timeline

| Phase | Masooma (A) | Huma (B) | Fatima (C) |
|---|---|---|---|
| **Days 1–2** | A1 `expected_chunks.json`, `tests/conftest.py` | B1 `expected_graph.json` | C1 query analysis |
| **Week 1** | A2 walker, A3 parser | B2 module map, B3, B4 | C2 scoring (on fixture JSON) |
| **Week 2** | A4 `parse_repo`, A tests, A5 loader | B5 calls, B6 queries, B7 API | C3–C6 packing, rendering, `generate_variants` |
| **Integration (2–3 days)** | swap fixture JSON for real `parse_repo` everywhere; end-to-end test in CI | | |
| **Week 3** | loader runs on all 40 tasks | demo script, edge-case demo | C7 metrics + full evaluation run |
| **Freeze** | tag `v0.1-iter1`; results table + figures for report chapters 6–7 | | |

Integration test (owner: whoever finishes first, all review): `tests/test_end_to_end.py` → `generate_variants(*pipeline(sample_repo), "Cart.total applies the discount wrong")` returns 4 valid variants.

---

## 8. Iteration 1 definition of done

- [ ] `parse_repo`, `build_graph`, `generate_variants` implemented and matching contracts.
- [ ] All unit tests + end-to-end test green in CI; `ruff` clean.
- [ ] Every edge case in sections 4–6 has a test.
- [ ] `POST /context/variants` works live for the demo.
- [ ] Evaluation on ~40 SWE-bench Lite tasks done; `iter1_results.csv` and `iter1_summary.md` committed.
- [ ] `docs/DECISIONS.md` change log updated with any tuning (ALPHA, tier fractions, cap).
- [ ] Every member has merged PRs in their own area; tag `v0.1-iter1` pushed.

## 9. Demo script (mid evaluation)

1. `pytest -v` live (shows tests incl. edge cases).
2. Start API; call `/context/variants` on `sample_repo` with a query → show 4 tiers and token counts.
3. Edge cases live: syntax-error file still parsed; circular import handled; query that names nothing → fallback.
4. Show one SWE-bench task: which tiers kept the gold function; then the summary table.

## 10. Out of scope for Iteration 1 (do not build yet)

Model calls / provider adapters, ODE bandit, reward function, prompt caching, redaction/security layer, dashboard, chunks extracted from live agent requests (Iteration 2), languages other than Python, incremental graph patching (only keep `file_hash` ready for it).

## 11. Known limitations to state in the report

* Token counts use a proxy tokenizer (`o200k_base`), not each provider's own.
* Call resolution is heuristic (no type inference); attribute calls on objects resolve only by unique name.
* Function-local imports are not resolved (unless implemented as stretch).
* Gold-chunk recall measures whether the edited code was kept, not whether a model would solve the task — that is measured in Iteration 2.
