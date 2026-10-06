"""Member A — SWE-bench Lite loader (IMPLEMENTATION_PLAN A5).

The dataset is read from the Hugging Face datasets-server JSON API with the standard
library, so no extra dependency is needed; the rows are cached in cache/swebench_lite_test.json.
"""

import json
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from unidiff import PatchSet

DATASET = "princeton-nlp/SWE-bench_Lite"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100  # API maximum
FIRST_REPOS = ["psf/requests", "pallets/flask"]  # all of their tasks are used
FILL_REPOS = ["pytest-dev/pytest", "django/django"]  # then alternate between these
DEFAULT_TASK_COUNT = 40

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = ROOT / "cache"
DEFAULT_REPOS_DIR = ROOT / "swebench_repos"


@dataclass
class Task:
    instance_id: str
    repo: str
    repo_path: str
    base_commit: str
    query: str  # = problem_statement (DECISIONS #11)
    gold_files: list[str] = field(default_factory=list)
    gold_lines: dict[str, list[int]] = field(default_factory=dict)


def fetch_rows(cache_dir: Path = DEFAULT_CACHE_DIR) -> list[dict]:
    """All rows of the SWE-bench Lite test split (downloaded once, then read from cache)."""
    cache_file = cache_dir / "swebench_lite_test.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    rows: list[dict] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": "default",
                "split": "test",
                "offset": offset,
                "length": PAGE_SIZE,
            }
        )
        with urllib.request.urlopen(f"{ROWS_URL}?{query}", timeout=60) as response:
            page = json.load(response)
        rows.extend(item["row"] for item in page["rows"])
        offset += PAGE_SIZE
        if offset >= page["num_rows_total"]:
            break
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def select_task_ids(rows: list[dict], count: int = DEFAULT_TASK_COUNT) -> list[str]:
    """Deterministic subset (DECISIONS #10): all requests + flask, then alternate pytest / django."""
    by_repo: dict[str, list[str]] = {}
    for row in rows:
        by_repo.setdefault(row["repo"], []).append(row["instance_id"])
    for ids in by_repo.values():
        ids.sort()

    chosen = []
    for repo in FIRST_REPOS:
        chosen.extend(by_repo.get(repo, []))
    queues = [list(by_repo.get(repo, [])) for repo in FILL_REPOS]
    while len(chosen) < count and any(queues):
        for queue in queues:
            if queue and len(chosen) < count:
                chosen.append(queue.pop(0))
    return sorted(chosen[:count])


def gold_from_patch(patch: str) -> tuple[list[str], dict[str, list[int]]]:
    """Gold files and original-file line numbers touched by the reference patch.

    Only existing Python files count (we only chunk .py files). For a hunk that only adds
    lines, its whole source range is used.
    """
    gold_lines: dict[str, set[int]] = {}
    for patched in PatchSet(patch):
        if patched.is_added_file or not patched.path.endswith(".py"):
            continue
        lines = gold_lines.setdefault(patched.path, set())
        for hunk in patched:
            removed = [line.source_line_no for line in hunk if line.is_removed]
            if removed:
                lines.update(removed)
            else:
                start = hunk.source_start
                lines.update(range(start, start + max(hunk.source_length, 1)))
    files = sorted(gold_lines)
    return files, {f: sorted(gold_lines[f]) for f in files}


def repo_dir(repo: str, repos_dir: Path = DEFAULT_REPOS_DIR) -> Path:
    return repos_dir / repo.replace("/", "__")


def ensure_clone(repo: str, repos_dir: Path = DEFAULT_REPOS_DIR) -> Path:
    path = repo_dir(repo, repos_dir)
    if not (path / ".git").exists():
        repos_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--quiet", f"https://github.com/{repo}.git", str(path)], check=True
        )
    return path


def checkout(task: Task) -> None:
    subprocess.run(
        ["git", "checkout", "--quiet", "-f", task.base_commit], cwd=task.repo_path, check=True
    )
    subprocess.run(["git", "clean", "--quiet", "-fdx"], cwd=task.repo_path, check=True)


def load_tasks(
    task_ids: list[str] | None = None,
    count: int = DEFAULT_TASK_COUNT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    repos_dir: Path = DEFAULT_REPOS_DIR,
) -> list[Task]:
    """Task records sorted by (repo, instance_id) so each repo is processed in one go.
    Does not clone or check out anything; see ensure_clone() and checkout()."""
    rows = fetch_rows(cache_dir)
    wanted = set(task_ids if task_ids is not None else select_task_ids(rows, count))
    tasks = []
    for row in rows:
        if row["instance_id"] not in wanted:
            continue
        gold_files, gold_lines = gold_from_patch(row["patch"])
        tasks.append(
            Task(
                instance_id=row["instance_id"],
                repo=row["repo"],
                repo_path=str(repo_dir(row["repo"], repos_dir)),
                base_commit=row["base_commit"],
                query=row["problem_statement"],
                gold_files=gold_files,
                gold_lines=gold_lines,
            )
        )
    tasks.sort(key=lambda t: (t.repo, t.instance_id))
    return tasks
