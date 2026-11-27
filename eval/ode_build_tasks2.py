"""ODE task set 2: build eval/ode_tasks2.json (features + refactors). No model calls.

Features: FEA-Bench (microsoft/FEA-Bench, ACL 2025) sympy tasks. The change and its tests
come from the pull request in sympy's history (base commit .. PR head), the request text from
the pull request's title and description, plus the new names and signatures (as FEA-Bench
does: the hidden tests call them by name). Every chosen task was checked: the new tests fail
at the base commit and pass with the real change.

Refactors: RefactorBench (microsoft/RefactorBench, ICLR 2025) on requests and flask. Each
repository snapshot becomes a one-commit git repository; the task is the "base" instruction,
graded by RefactorBench's structure checks plus the repository's tests that passed before.

    python -m eval.ode_build_tasks2 build       # task file (needs the inputs below)
    python -m eval.ode_build_tasks2 baselines   # passing tests per refactor repository

Inputs (local, not in git): --fea feabench.json (FEA-Bench rows), --prs (GitHub pull request
JSON per number), --sympy (a sympy clone with full history), --refactorbench (clone).
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path

from eval import ode_tasks as T

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "ode_tasks2.json"
DATA = Path(os.environ.get("ODE_TASKS_DATA", "E:/FYP/ode-tasks"))

FEATURES = (18213, 18953, 20911, 21106, 21689, 22114, 22705, 23111, 17019, 16248)
REFACTORS = (
    ("requests", "rename-super-len-complex-len"),
    ("requests", "move-hooks-sessions"),
    ("requests", "add-log-parameter-get-encoding-from-headers"),
    ("requests", "combine-internal-utils-utils"),
    ("requests", "new-cookie-utils-class"),
    ("requests", "split-warnings-exceptions"),
    ("flask", "rename-send-from-directory"),
    ("flask", "render-template-str"),
    ("flask", "debughelpers-to-helpers.py"),
    ("flask", "add-log-parameter-get-flashed-messages"),
)
# problem file name -> check file name, where RefactorBench's names differ
CHECK_NAMES = {
    ("requests", "rename-super-len-complex-len"): "super-to-complex-len.py",
    ("requests", "move-hooks-sessions"): "move-hooks-sessions.py",
    ("requests", "add-log-parameter-get-encoding-from-headers"): "get-encoding-from-headers.py",
    ("requests", "combine-internal-utils-utils"): "internal-utils-utils.py",
    ("requests", "new-cookie-utils-class"): "cookie-utils.py",
    ("requests", "split-warnings-exceptions"): "requests-warnings.py",
}
GIT_ENV = {"GIT_AUTHOR_NAME": "ode", "GIT_AUTHOR_EMAIL": "ode@local",
           "GIT_COMMITTER_NAME": "ode", "GIT_COMMITTER_EMAIL": "ode@local",
           "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}


def git(repo: Path, *args: str, inp: bytes | None = None) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True, input=inp,
                          env={**os.environ, **GIT_ENV}).stdout.decode("utf-8", "replace")


# ---------- features ----------

def clean_body(body: str) -> str:
    """PR description without sympy's template: comments, empty sections, release-note markers."""
    body = re.sub(r"<!--.*?-->", "", body or "", flags=re.DOTALL)
    body = re.sub(r"(?im)^\s*(fixes|closes|resolves)\s+#\d+.*$", "", body)
    parts = re.split(r"(?m)^(#+ .*)$", body)
    out = [parts[0].strip()]
    for head, text in zip(parts[1::2], parts[2::2]):
        if text.strip() and not head.lower().startswith(("#### references", "#### other comm")):
            out.append(f"{head.strip().lstrip('#').strip()}:\n{text.strip()}")
    return "\n\n".join(p for p in out if p)


def _defs(source: str) -> dict[str, ast.AST]:
    found: dict[str, ast.AST] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                found[name] = child
                walk(child, name + ".")
    walk(ast.parse(source), "")
    return found


def new_components(repo: Path, base: str, head: str, files: list[str]) -> list[str]:
    """`path: def Class.name(args)  # first docstring line` for each name the change adds."""
    lines = []
    for path in files:
        before = _defs(git(repo, "show", f"{base}:{path}")) if git_has(repo, base, path) else {}
        after = _defs(git(repo, "show", f"{head}:{path}"))
        for name, node in after.items():
            if name in before:
                continue
            parent = name.rsplit(".", 1)[0] if "." in name else ""
            if parent and parent not in before:
                continue  # inside a new class: the class line covers it
            if isinstance(node, ast.ClassDef):
                sig = f"class {name}"
            else:
                deco = "".join(f"@{ast.unparse(d)} " for d in node.decorator_list)
                sig = f"{deco}def {name}({ast.unparse(node.args)})"
            doc = (ast.get_docstring(node) or "").strip().split("\n\n")[0].replace("\n", " ")
            lines.append(f"- {path}: `{sig}`" + (f" - {doc[:200]}" if doc else ""))
    return lines


def git_has(repo: Path, commit: str, path: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{commit}:{path}"], cwd=repo,
                          capture_output=True, check=False).returncode == 0


def feature_row(fea: dict, pr: dict, repo: Path) -> dict:
    n = fea["pull_number"]
    merges = git(repo, "log", "--all", "--merges", "--format=%H %s").splitlines()
    merge = next(m.split()[0] for m in merges if m.split(" ", 1)[1].startswith(
        f"Merge pull request #{n} "))
    base, head = fea["base_commit"], f"{merge}^2"
    files = [f for f in git(repo, "diff", "--name-only", base, head).split() if f.endswith(".py")]
    tests = [f for f in files if T.is_test_file(f)]
    src = [f for f in files if f not in tests]
    statement = f"{pr['title']}\n\n{clean_body(pr.get('body') or '')}".strip()
    comps = new_components(repo, base, head, src)
    if comps:
        statement += ("\n\nNew code to add (the tests use exactly these names):\n"
                      + "\n".join(comps))
    return {"instance_id": fea["instance_id"], "kind": "feature", "env": "sympy",
            "source": "FEA-Bench", "pull_number": n, "base_commit": base,
            "problem_statement": statement,
            "patch": git(repo, "diff", base, head, "--", *src),
            "test_patch": git(repo, "diff", base, head, "--", *tests),
            "FAIL_TO_PASS": json.dumps(fea["FAIL_TO_PASS"]),
            "PASS_TO_PASS": json.dumps(fea["PASS_TO_PASS"])}


# ---------- refactors ----------

def snapshot_repo(rb: Path, name: str, dest: Path) -> str:
    """One-commit git repository from RefactorBench's snapshot (taken from git, so line endings
    are as committed); a fixed author and date make the commit id the same on every machine."""
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        tar = subprocess.run(["git", "archive", "--format=tar", f"HEAD:repositories/{name}_refactor"],
                             cwd=rb, capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(tar)) as t:
            t.extractall(dest, filter="data")
        for junk in dest.rglob(".DS_Store"):
            junk.unlink()
        git(dest, "init", "-q")
        git(dest, "config", "core.autocrlf", "false")
        git(dest, "add", "-A")
        git(dest, "commit", "-q", "-m", f"RefactorBench {name} snapshot")
    return git(dest, "rev-parse", "HEAD").strip()


def refactor_gold(repo: Path, check: str, instruction: str) -> tuple[list[str], list[str]]:
    """Explain gold: the source files the checks read (that exist at the start), plus the
    function / class names from the instruction and the files that define them. Only code-like
    words count (snake_case or CamelCase): plain words such as "content" or "update" also name
    methods somewhere."""
    files = {p[3:] for p in re.findall(r"['\"](\.\./[\w./-]+\.py)['\"]", check)
             if not T.is_test_file(p[3:]) and (repo / p[3:]).exists()}
    words = {w for w in re.findall(r"[A-Za-z_]\w+", instruction)
             if "_" in w.strip("_") or re.search(r"[a-z][A-Z]", w)}
    names = set()
    for path in sorted((repo / "src").rglob("*.py")):
        found = words & {n.rsplit(".", 1)[-1] for n in _defs(path.read_text(encoding="utf-8"))}
        if found:
            names |= found
            files.add(path.relative_to(repo).as_posix())
    return sorted(files), sorted(names)


def refactor_row(rb: Path, name: str, task: str, repos: Path) -> dict:
    repo = repos / name
    base = snapshot_repo(rb, name, repo)
    instruction = (rb / "problems" / "base_problems" / f"{name}_refactor" /
                   f"{task}-task.txt").read_text(encoding="utf-8").strip()
    check_name = CHECK_NAMES.get((name, task), f"{task}-test.py")
    check = (rb / "tests" / f"{name}_refactor" / check_name).read_text(encoding="utf-8")
    files, funcs = refactor_gold(repo, check, instruction)
    return {"instance_id": f"refactorbench__{name}-{task.replace('.py', '')}", "kind": "refactor",
            "env": name, "source": "RefactorBench", "base_commit": base,
            "problem_statement": instruction, "check": check,
            "gold_files": files, "gold_functions": funcs}


# ---------- main ----------

def build(args) -> dict:
    fea = {r["pull_number"]: r for r in json.loads(args.fea.read_text(encoding="utf-8"))
           if r["repo"] == "sympy/sympy"}
    tasks = []
    for n in FEATURES:
        pr = json.loads((args.prs / f"{n}.json").read_text(encoding="utf-8"))
        tasks.append(feature_row(fea[n], pr, args.sympy))
    for name, task in REFACTORS:
        tasks.append(refactor_row(args.refactorbench, name, task, args.repos))
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    return {"tasks": tasks, "baselines": old.get("baselines", {})}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("build", "baselines"))
    parser.add_argument("--fea", type=Path, default=DATA / "feabench.json")
    parser.add_argument("--prs", type=Path, default=DATA / "prs")
    parser.add_argument("--sympy", type=Path, default=DATA / "sympy-full.git")
    parser.add_argument("--refactorbench", type=Path, default=DATA / "refactorbench")
    parser.add_argument("--repos", type=Path, default=DATA / "repos")
    args = parser.parse_args(argv)
    if args.step == "build":
        data = build(args)
    else:
        from eval import ode_run as R  # heavy import, only for this step
        data = json.loads(OUT.read_text(encoding="utf-8"))
        for env in sorted({t["env"] for t in data["tasks"] if t["kind"] == "refactor"}):
            row = next(t for t in data["tasks"] if t["env"] == env)
            paths = R.env_paths(env)
            R.ensure_worktree(paths["repo"], paths["work"], row["base_commit"])
            R.prepare_run_dir(paths["work"], R.RUN_DIR, row["base_commit"], paths["python"])
            outcomes, counts, _ = T.run_suite(R.RUN_DIR, paths["python"], ["tests"], 1800)
            data["baselines"][env] = sorted(t for t, s in outcomes.items() if s == "PASSED")
            print(f"{env}: {counts}", flush=True)
    OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(data['tasks'])} tasks -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
