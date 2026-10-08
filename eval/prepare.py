"""Clone repos and cache parsed chunks for a task set, so benchmarks run offline and fast.

    python -m eval.prepare --set test
"""

import argparse
import json
import logging
import time

from eval.loader import DEFAULT_CACHE_DIR, DEFAULT_REPOS_DIR, checkout, ensure_clone, load_tasks
from eval.run_eval import EVAL_DIR, cached_parse

TASK_SETS_FILE = EVAL_DIR / "tasks_iter2.json"
log = logging.getLogger("prepare")


def task_set(name: str) -> list[str]:
    return json.loads(TASK_SETS_FILE.read_text(encoding="utf-8"))[name]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["dev", "test"], default="test")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tasks = load_tasks(task_set(args.set))
    for i, task in enumerate(tasks, 1):
        start = time.perf_counter()
        ensure_clone(task.repo, DEFAULT_REPOS_DIR)
        checkout(task)
        chunks, _ = cached_parse(task.repo_path, task.repo, task.base_commit, DEFAULT_CACHE_DIR)
        log.info("[%d/%d] %s: %d chunks (%.0fs)", i, len(tasks), task.instance_id, len(chunks),
                 time.perf_counter() - start)


if __name__ == "__main__":
    main()
