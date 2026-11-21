# TokenCut

Cost-aware joint context-and-model routing gateway for AI coding agents (FYP, FAST NUCES Islamabad).

**Iteration 1:** repository parsing, code-graph construction, and 4-tier context-variant generation.

## Install and use (one command)
Needs [Claude Code](https://claude.ai/code) installed first. Then:
```bash
curl -fsSL https://raw.githubusercontent.com/Masooma-Ali/TokenCut/main/install.sh | sh     # macOS / Linux
```
```powershell
irm https://raw.githubusercontent.com/Masooma-Ali/TokenCut/main/install.ps1 | iex          # Windows
```
The installer adds [uv](https://docs.astral.sh/uv/) if missing, installs the `tokencut` command
(PyPI package `tokencut-cli`) and runs `tokencut setup`, which asks whether to add your ChatGPT
plan, installs the hidden model connector (LiteLLM) and downloads the prompt model once.
Then, in any project folder:

| Command | What it does |
|---|---|
| `tokencut` | opens Claude Code through TokenCut (trimming, system note, task briefs, routing); `/model gpt-6.1-sol` for GPT if set up |
| `tokencut setup` | change logins / re-run one-time installs (`--chatgpt`, `--no-chatgpt`) |
| `tokencut status` | running? tokens removed, billed usage |
| `tokencut logs [gateway\|connector]` | last lines of a process log |

Settings: `~/.tokencut/config.json` (no secrets); every `TOKENCUT_*` variable below overrides it.
Claude keeps Claude Code's own login; the ChatGPT token stays in LiteLLM's file; a local
TokenCut key (`~/.tokencut/tokencut_key`) stops other programs from using the gateway.
Releases: push a tag `vX.Y.Z` (`.github/workflows/release.yml` tests, publishes to PyPI and
creates a GitHub Release).

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

## Benchmark: TokenCut vs baselines

`eval/benchmark.py` compares TokenCut with BM25, dense embeddings (bge-small), naive
truncation, random and SWE-bench file retrieval at **equal token budgets** (15 / 35 / 60 %
of a 32k context), in two settings: pure trimming of a shared neutral 32k context, and
selection from the whole repository. Reports gold-chunk recall with 95% bootstrap CIs and
paired sign tests. Tasks: 40 dev (used for tuning) and 81 held-out test tasks
(`eval/tasks_iter2.json`).

```
python -m eval.prepare --set test                    # clone + parse once
python -m eval.benchmark --set test                  # no embeddings
python -m eval.benchmark --set test --methods tokencut,tokencut_hybrid,bm25,embed,random,head,bm25_files
```

Results: `eval/results/benchmark_<set>.md` / `.csv`.

## Gateway (Iteration 2)

The gateway sits between your coding agent and the provider. It streams the reply back and
records what each request contained (tokens per kind: system prompt, tools, chat, file
reads, command output…) plus the tokens the provider actually billed, including cached
tokens. In shadow mode (default) requests are forwarded unchanged and the stats show what
trimming *would* save; in trim mode the trimmed request is sent.

```
python -m tokencut.gateway          # listens on 127.0.0.1:8787
```

Point an agent at it:

| Agent / SDK | Setting |
|---|---|
| Claude Code (Anthropic) | `ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic` |
| OpenAI SDKs, Cline, OpenRouter-style clients | base URL `http://127.0.0.1:8787/openai/v1` |
| Gemini (google-genai / Gemini CLI) | base URL `http://127.0.0.1:8787/gemini` |

Your API key travels in the request headers to the provider as usual; it is never logged.
See what was measured at `http://127.0.0.1:8787/tokencut/stats`; per-request numbers go to
`logs/gateway/requests-<date>.jsonl` (gitignored).

| Variable | Default | Meaning |
|---|---|---|
| `TOKENCUT_ANTHROPIC_UPSTREAM` / `_OPENAI_` / `_GEMINI_` | official API URLs | send to another compatible endpoint (e.g. OpenRouter) |
| `TOKENCUT_LOG_DIR` | `logs/gateway` | where numbers are logged (empty = off) |
| `TOKENCUT_KEY` | none | if set, proxy requests must send it in `x-tokencut-key` (the `tokencut` command does this; the header is never forwarded) |
| `TOKENCUT_CAPTURE` | off | `1` = also save full request bodies locally, for parser test fixtures (contain your prompts and code) |
| `TOKENCUT_HOST` / `TOKENCUT_PORT` | `127.0.0.1` / `8787` | listen address |
| `TOKENCUT_MODE` | `shadow` | `shadow` = forward unchanged, report possible savings; `trim` = send the trimmed request |
| `TOKENCUT_TIER` | `moderate` | `aggressive` / `moderate` / `conservative` |
| `TOKENCUT_MIN_TRIM_TOKENS` | `2000` | tool results smaller than this are never trimmed |
| `TOKENCUT_CACHE_EVENTS` | `1` | `0` = never re-trim history at cache events |
| `TOKENCUT_CACHE_TTL` | `300` | seconds of idle after which the provider cache counts as expired |
| `TOKENCUT_PRUNE_TOOLS` | off | `1` = at cache events, send only tools used so far + core tools |
| `TOKENCUT_CORE_TOOLS` | Claude Code / Gemini CLI basics | comma-separated tools always kept when pruning |
| `TOKENCUT_TASK_MASK` | **on** | when the user starts a new task, earlier tasks' large tool outputs become notes (only if the cache rewrite pays off); `0` = off |
| `TOKENCUT_TASK_MASK_TOKENS` | `300` | smallest tool output masked at a new task |
| `TOKENCUT_CCR` | off | `1` = every shortened output gets a reference; the gateway offers `tokencut_retrieve` and answers it with the exact original lines |
| `TOKENCUT_LAZY_TOOLS` | off | `1` = every request carries only core / used tools + a catalog tool `tokencut_enable_tools` for the rest |
| `TOKENCUT_SYSTEM_NOTE` | off | `1` = trim mode adds a fixed ~125-token note to the system prompt: how to treat `[TokenCut: …]` markers (re-read only the omitted lines, do not re-run commands for cut log lines, keep test output short) |
| `TOKENCUT_PROMPT_BRIEF` | off | `1` = trim mode classifies each typed user message (rules + the ONNX prompt router) and, when confident, appends a short task brief (`<tokencut_task_brief type="bugfix">…`, 48–71 tokens); agent requests only; never repeated for the same type in a row |
| `TOKENCUT_BRIEF_THRESHOLD` | `0.87` | minimum router probability for a brief (chosen on dev, RESULTS_REPORT §11) |
| `TOKENCUT_ROUTER_DIR` | — | local folder with `router.onnx`, `router.onnx.data`, `tokenizer.json`, `router_config.json` |
| `TOKENCUT_ROUTER_REPO` | `humataj/tokencut-router` | public Hugging Face repo the model is downloaded from (once, no login, to `~/.cache/tokencut/router`) when no local folder is set |

**Trimming (phase 3).** Only large tool results are shortened: Python file contents
(through the Iteration 1 engine; kept lines stay verbatim, gaps become
`[TokenCut: lines 40-95 of x.py omitted …]` markers so the model can re-read them),
command output (failures, errors and the summary are kept) and search results.
System prompt, tools, chat text and tool calls are never touched; files the user named or
the agent already edited stay whole. A result's trim depends only on earlier conversation,
so it is identical on every later turn and the provider's prompt cache keeps working.

**Cache-event trimming.** The cache is lost anyway when the model changes, after the cache
expires, or when the context is nearly full. At those moments TokenCut also shrinks old
history: old command/test/search output becomes a note with its error lines, older reads of
a file that was read again or edited become a short note, and (opt-in) unused tool
definitions are dropped. The last few turns are never touched. The decisions are remembered
per conversation and re-sent identically until the next event; a file the agent re-reads
after its old read was removed is protected from then on.

**Reversible trimming (P1, `TOKENCUT_CCR=1`).** Every shortened output ends with
`[TokenCut ref tc_…]`. The gateway adds a `tokencut_retrieve(ref, start_line, end_line)` tool;
the agent does not know it and answers "no such tool", and the gateway replaces that answer
with the exact original lines before forwarding. References are hashes of the original, so
they are identical every turn (cache-safe) and survive restarts.

**Lazy tool loading (P1, `TOKENCUT_LAZY_TOOLS=1`).** Each request carries only the core tools
(`TOKENCUT_CORE_TOOLS`), tools already used, and one catalog tool `tokencut_enable_tools`
listing the rest; enabling one adds it from that request on (one cache rewrite).

## Repo index for the agent (P1, local MCP server)

Not part of the gateway: a local MCP server that indexes the working tree (uncommitted
edits included) and gives the agent `repo_map`, `find_code` and `symbol`, so it can find
code in one call instead of several grep + read rounds.

```bash
python -m tokencut.repoindex build --repo .            # first index (cached, incremental after)
claude mcp add tokencut -- python -m tokencut.repoindex serve --repo .
python -m tokencut.repoindex find --repo . "QuerySet.only() crashes with select_related"
```
`--embed` adds dense embeddings (computed once in the background, cached per chunk).
Offline check on SWE-bench tasks: `python -m eval.repoindex_eval --set dev|test`.

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
| `gateway/app.py` | proxy server: forward, stream back, measure (shadow mode) |
| `gateway/adapters/` | Anthropic / OpenAI / Gemini request JSON → `Segment` list |
| `gateway/classify.py`, `gateway/usage.py`, `gateway/records.py` | tool-result kinds, billed usage from responses, request log |
| `gateway/trimmer.py`, `gateway/codeview.py`, `gateway/textcut.py` | cache-safe trimming of tool results: code, command output, search |
| `gateway/history.py`, `gateway/session.py` | cache-event trimming of old history and tool definitions; per-conversation memory |

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

**For PDFs with an agent, also install Poppler** (`pdftoppm` on PATH). Claude Code's Read tool
needs it to read PDF page ranges; it is a program, not a Python package, so it is not in
`requirements.txt`. Windows steps: `docs/TEAMMATE_SETUP.md` → "Poppler for PDFs";
macOS `brew install poppler`; Linux `sudo apt-get install poppler-utils`.
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
