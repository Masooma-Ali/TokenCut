"""Member C — Iteration 1 evaluation on SWE-bench Lite (IMPLEMENTATION_PLAN C7).

    python -m eval.run_eval                 # the 40 tasks in eval/tasks_iter1.json
    python -m eval.run_eval --limit 3       # quick smoke run
    python -m eval.run_eval --repo psf/requests

Writes eval/results/iter1_results.csv and eval/results/iter1_summary.md.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from eval.loader import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REPOS_DIR,
    checkout,
    ensure_clone,
    fetch_rows,
    load_tasks,
    select_task_ids,
)
from eval.metrics import summary_markdown, task_rows, write_csv
from tokencut.contracts import EVAL_FULL_CONTEXT_CAP_TOKENS, Chunk
from tokencut.graph.builder import build_graph
from tokencut.ingestion.parser import parse_repo
from tokencut.trimming.variants import generate_variants

EVAL_DIR = Path(__file__).resolve().parent
TASKS_FILE = EVAL_DIR / "tasks_iter1.json"
RESULTS_DIR = EVAL_DIR / "results"

log = logging.getLogger("run_eval")


def cached_parse(
    repo_path: str, repo: str, commit: str, cache_dir: Path
) -> tuple[list[Chunk], float]:
    """parse_repo with a per-(repo, commit) cache. Returns (chunks, parse seconds)."""
    cache_file = cache_dir / "chunks" / f"{repo.replace('/', '__')}__{commit}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return [Chunk.model_validate(x) for x in data["chunks"]], data["seconds"]
    start = time.perf_counter()
    chunks = parse_repo(repo_path)
    seconds = time.perf_counter() - start
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"seconds": seconds, "chunks": [c.model_dump(mode="json") for c in chunks]}),
        encoding="utf-8",
    )
    return chunks, seconds


def task_ids_to_run(cache_dir: Path) -> list[str]:
    """Fixed task list, created once and committed so every run uses the same tasks."""
    if TASKS_FILE.exists():
        return json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    ids = select_task_ids(fetch_rows(cache_dir))
    TASKS_FILE.write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")
    return ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=None, help="run only the first N tasks")
    parser.add_argument("--repo", default=None, help="only tasks of this repo, e.g. psf/requests")
    parser.add_argument(
        "--cap", type=int, default=EVAL_FULL_CONTEXT_CAP_TOKENS, help="full-context cap"
    )
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--repos-dir", type=Path, default=DEFAULT_REPOS_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tasks = load_tasks(
        task_ids_to_run(args.cache_dir), cache_dir=args.cache_dir, repos_dir=args.repos_dir
    )
    if args.repo:
        tasks = [t for t in tasks if t.repo == args.repo]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    rows: list[dict] = []
    for i, task in enumerate(tasks, 1):
        log.info("[%d/%d] %s", i, len(tasks), task.instance_id)
        ensure_clone(task.repo, args.repos_dir)
        checkout(task)
        chunks, t_parse = cached_parse(task.repo_path, task.repo, task.base_commit, args.cache_dir)

        start = time.perf_counter()
        graph = build_graph(chunks)
        t_graph = time.perf_counter() - start

        start = time.perf_counter()
        variants = generate_variants(chunks, graph, task.query, full_cap_tokens=args.cap)
        t_variants = time.perf_counter() - start

        rows.extend(
            task_rows(
                task.instance_id,
                task.repo,
                chunks,
                variants,
                task.gold_files,
                task.gold_lines,
                {"parse": t_parse, "graph": t_graph, "variants": t_variants},
            )
        )

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "iter1_results.csv")
    (args.out / "iter1_summary.md").write_text(summary_markdown(rows) + "\n", encoding="utf-8")
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
