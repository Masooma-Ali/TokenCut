"""ODE phase 4: practice sessions and their automatic checkers (no model calls here).

A session is three typed turns on one SWE-bench Lite task, in one Claude Code session:

    1 explain  "where is this handled? name the file(s) and function(s)"  -> explain_locates
    2 bugfix   "now fix it"                                              -> official SWE-bench tests
    3 tests    "now add a regression test"                               -> SWT-bench fail-to-pass

Checkers (each returns solved True / False and the numbers behind it):
  explain  (team choice, option A): the answer names a file AND a function that the real fix
           changed. It measures locating the right code, not the quality of the explanation.
  bugfix   the official FAIL_TO_PASS tests pass and PASS_TO_PASS still pass on the agent's diff
           (the task's test files are reset first, as SWE-bench does).
  tests    SWT-bench rule: at least one test in the test files the agent changed in turn 3 fails
           (or errors) on the original code and passes once the real fix is applied.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from unidiff import PatchSet

from eval.loader import gold_from_patch
from eval.stepd_validate import _RESULT, apply_patch, passed, test_files
from tokencut.contracts import ChunkKind
from tokencut.ingestion.python_parser import PythonParser

TURN_TYPES = ("explain", "bugfix", "tests")
# Task set 2 (eval/ode_tasks2.json): FEA-Bench features and RefactorBench refactors. The
# middle turn is the task itself; refactor sessions end with a run_action turn.
TURNS_BY_KIND = {"bugfix": TURN_TYPES, "feature": ("explain", "feature", "tests"),
                 "refactor": ("explain", "refactor", "run_action")}
SIGNALS = {"explain": "explain_locates", "bugfix": "swebench_tests", "tests": "swt_fail_to_pass",
           "feature": "fea_bench_tests", "refactor": "refactorbench_ast_and_suite",
           "run_action": "git_commit_matches"}
COMMIT_MESSAGE = "Apply the refactoring"
PROMPTS = (
    ("Here is an issue reported in this repository.\n\n{issue}\n\nDo not change any files yet. "
     "Explain where in the code this behaviour is handled: name the file(s) and the function(s) "
     "involved, and briefly explain how they work. Use {python} for every Python or pytest "
     "command."),
    "Now make the code change in the repository that resolves the issue. Use the same Python.",
    ("Now add a regression test for this issue to the repository's existing test suite: a test "
     "that fails without your fix and passes with it. Run it with the same Python."),
)
FEATURE_PROMPTS = (
    ("Here is a feature request for this repository.\n\n{issue}\n\nDo not change any files yet. "
     "Explain where in the code this feature belongs: name the file(s) and the existing "
     "function(s) or class(es) involved, and briefly explain how they work. Use {python} for "
     "every Python or pytest command."),
    ("Now implement the feature in the repository, with exactly the names and signatures given "
     "in the request. Use the same Python."),
    ("Now add tests for the new feature to the repository's existing test suite: tests that "
     "fail without your change and pass with it. Run them with the same Python."),
)
REFACTOR_PROMPTS = (
    ("Here is a refactoring task for this repository.\n\n{issue}\n\nDo not change any files yet. "
     "Explain which file(s) and function(s) or class(es) this refactoring involves and where "
     "they are used. Use {python} for every Python or pytest command."),
    ("Now do the refactoring in the repository. Keep the behaviour the same and update every "
     "place that uses the changed code. Use the same Python."),
    # run_action: a git commit. The prompt router labels this run_action (p 0.96); "run the
    # test suite and report the counts" was labelled tests (p 0.89-0.92) in every wording tried.
    (f"Now commit all your changes with git, with the commit message \"{COMMIT_MESSAGE}\". Do "
     "not change any files. End your answer with one line: `COMMIT: <short hash>`."),
)


def prompts(row: dict, python: str) -> list[str]:
    kind = row.get("kind", "bugfix")
    texts = {"bugfix": PROMPTS, "feature": FEATURE_PROMPTS, "refactor": REFACTOR_PROMPTS}[kind]
    return [t.format(issue=row["problem_statement"], python=python) for t in texts]


def turn_types(row: dict) -> tuple[str, ...]:
    return TURNS_BY_KIND[row.get("kind", "bugfix")]


def git(work: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=work, capture_output=True, text=True, check=check)


def run_tests(work: Path, python: str, files: list[str], timeout: int) -> tuple[dict[str, str], str]:
    """{test name: outcome} from pytest -rA (as eval/stepd_validate.run_tests), but the output
    is decoded as UTF-8: with Windows' default code page, sympy's Unicode test output (e.g.
    sympy 23191) made the reader fail and the task look ungradable."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA", *files],
                              cwd=work, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, check=False, env=env)
        out = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return {}, "timeout"
    outcomes: dict[str, str] = {}
    for status, node in _RESULT.findall(out):
        outcomes[node.split("::")[-1]] = status
    return outcomes, out[-2000:]


def reset(work: Path, commit: str) -> None:
    git(work, "checkout", "-q", "-f", commit)
    git(work, "clean", "-qfd")


# ---------- turn 1: explain ----------

@dataclass(frozen=True)
class Gold:
    files: tuple[str, ...]  # repo paths the real fix changed (Python files)
    functions: tuple[str, ...]  # their innermost changed functions / methods (short names)


_DEF = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")


def gold_locations(row: dict, work: Path) -> Gold:
    """Files and functions the real fix touched, read at the task's base commit.

    A function counts when the fix changes or removes a line inside it, or when the fix adds
    it (`def name` in an added line). Code only added between functions does not make its
    neighbours count (pilot: the fix for sympy 21055 adds `refine_arg`; the old rule credited
    the two functions next to it and marked correct answers wrong)."""
    files, _ = gold_from_patch(row["patch"])
    reset(work, row["base_commit"])
    parser = PythonParser()
    names: set[str] = set()
    for patched in PatchSet(row["patch"]):
        path = patched.path
        if path not in files:
            continue
        chunks = [c for c in parser.parse_file(path, (work / path).read_bytes())
                  if c.kind in (ChunkKind.FUNCTION, ChunkKind.METHOD, ChunkKind.CLASS)]
        for hunk in patched:
            for line in hunk:
                if line.is_added:
                    m = _DEF.match(line.value)
                    if m:
                        names.add(m.group(1))
                elif line.is_removed:
                    around = [c for c in chunks
                              if c.start_line <= line.source_line_no <= c.end_line]
                    funcs = [c for c in around if c.kind != ChunkKind.CLASS] or around
                    if funcs:
                        inner = min(funcs, key=lambda c: c.end_line - c.start_line)
                        names.add(inner.qualified_name.split(".")[-1])
    return Gold(tuple(files), tuple(sorted(names)))


def _names_file(text: str, path: str) -> bool:
    """The answer mentions this file: its repo path, a path ending in it, or its file name."""
    text = text.replace("\\", "/")
    name = path.rsplit("/", 1)[-1]
    return path in text or re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", text) is not None


def explain_locates(answer: str, gold: Gold) -> dict:
    files = [f for f in gold.files if _names_file(answer, f)]
    funcs = [f for f in gold.functions if re.search(rf"\b{re.escape(f)}\b", answer)]
    solved = bool(files) and (bool(funcs) or not gold.functions)
    return {"solved": solved, "gold_files": len(gold.files), "files_named": len(files),
            "gold_functions": len(gold.functions), "functions_named": len(funcs)}


# ---------- turn 2: bugfix ----------

def grade_bugfix(row: dict, diff: str, work: Path, python: str, timeout: int = 900) -> dict:
    """Official SWE-bench check of the agent's diff (as in eval/stepd_run.grade)."""
    f2p, p2p = json.loads(row["FAIL_TO_PASS"]), json.loads(row["PASS_TO_PASS"])
    res = {"solved": False, "f2p_pass": 0, "f2p_total": len(f2p), "p2p_pass": 0,
           "p2p_total": len(p2p), "note": ""}
    reset(work, row["base_commit"])
    try:
        if diff.strip() and not apply_patch(work, diff):
            res["note"] = "agent diff does not apply"
            return res
        files = test_files(row["test_patch"])
        git(work, "checkout", "-q", row["base_commit"], "--", *files, check=False)
        if not apply_patch(work, row["test_patch"]):
            res["note"] = "test patch does not apply after the agent diff"
            return res
        outcomes, _ = run_tests(work, python, files, timeout)
        res.update(f2p_pass=len(passed(outcomes, f2p)), p2p_pass=len(passed(outcomes, p2p)))
        res["solved"] = res["f2p_pass"] == len(f2p) and res["p2p_pass"] == len(p2p)
        return res
    finally:
        reset(work, row["base_commit"])


# ---------- turn 3: tests ----------

def is_test_file(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    return path.endswith(".py") and (
        "tests" in parts[:-1] or "test" in parts[:-1] or parts[-1].startswith("test_"))


def diff_files(diff: str) -> list[str]:
    return sorted({line.split()[3][2:] for line in diff.splitlines()
                   if line.startswith("diff --git") and len(line.split()) >= 4})


def only_files(diff: str, keep) -> str:
    """The parts of a git diff whose file passes `keep`."""
    out, take = [], False
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git"):
            parts = line.split()
            take = len(parts) >= 4 and keep(parts[3][2:])
        if take:
            out.append(line)
    return "".join(out)


def grade_tests(row: dict, test_diff: str, work: Path, python: str, timeout: int = 900) -> dict:
    """SWT-bench rule on the agent's test changes: some test fails before the real fix and
    passes after it."""
    test_diff = only_files(test_diff, is_test_file)
    files = diff_files(test_diff)
    res = {"solved": False, "test_files": len(files), "fail_to_pass": 0, "note": ""}
    if not files:
        res["note"] = "no test file changed"
        return res
    reset(work, row["base_commit"])
    try:
        if not apply_patch(work, test_diff):
            res["note"] = "agent test diff does not apply"
            return res
        before, _ = run_tests(work, python, files, timeout)
        if not apply_patch(work, row["patch"]):
            res["note"] = "real fix does not apply after the agent tests"
            return res
        after, _ = run_tests(work, python, files, timeout)
        flipped = [t for t, s in after.items() if s == "PASSED" and before.get(t) != "PASSED"]
        res.update(fail_to_pass=len(flipped), solved=bool(flipped))
        return res
    finally:
        reset(work, row["base_commit"])


# ---------- task set 2: features (FEA-Bench) ----------

def feature_gold(row: dict, work: Path) -> Gold:
    """Explain gold for a feature: the files the real change touched and the existing code it
    goes into (the class around an added method, or a function it changes). Names the change
    adds are left out: the request already gives them, so naming them proves nothing."""
    files, _ = gold_from_patch(row["patch"])
    reset(work, row["base_commit"])
    parser = PythonParser()
    names: set[str] = set()
    added: set[str] = set()
    for patched in PatchSet(row["patch"]):
        path = patched.path
        if path not in files:
            continue
        chunks = [c for c in parser.parse_file(path, (work / path).read_bytes())
                  if c.kind in (ChunkKind.FUNCTION, ChunkKind.METHOD, ChunkKind.CLASS)]

        def inner(line_no: int, insert: bool, chunks=chunks) -> None:
            # an insertion after line L belongs to a chunk that goes on after L
            around = [c for c in chunks if c.start_line <= line_no
                      and (line_no < c.end_line if insert else line_no <= c.end_line)]
            if around:
                best = min(around, key=lambda c: c.end_line - c.start_line)
                names.add(best.qualified_name.split(".")[-1])

        for hunk in patched:
            last = hunk.source_start - 1
            for line in hunk:
                if line.is_added:
                    m = _DEF.match(line.value)
                    if m:
                        added.add(m.group(1))
                    inner(last, insert=True)
                else:
                    last = line.source_line_no or last
                    if line.is_removed:
                        inner(last, insert=False)
    return Gold(tuple(files), tuple(sorted(names - added)))


# ---------- task set 2: refactors (RefactorBench) and run_action ----------

_TOOK = re.compile(r"\bin \d+(?:\.\d+)?s\b")  # pytest's final line: "... in 5.55s"
_SUMMARY = re.compile(r"(\d+) (passed|failed|errors?)\b")
# Whole node ids: they may contain spaces (a parameter, or a path such as C:\Users\Crown Tech),
# which `_RESULT` cuts off - every check in a temp folder then became one name. pytest ends a
# failure line with " - <reason>".
_NODE = re.compile(r"^(PASSED|FAILED|ERROR|XFAIL|XPASS)\s+(.+?)(?:\s+-\s.*)?$", re.MULTILINE)


def run_suite(root: Path, python: str, paths: list[str], timeout: int,
              cwd: Path | None = None) -> tuple[dict[str, str], dict[str, int], str]:
    """Whole pytest run: ({node id: outcome}, {"passed": n, "failed": n incl. errors}, tail)."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA",
                               *paths], cwd=cwd or root, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout, check=False,
                              env=env)
        out = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return {}, {"passed": 0, "failed": 0}, "timeout"
    outcomes = {node: status for status, node in _NODE.findall(out)}
    counts = {"passed": 0, "failed": 0}
    lines = [ln for ln in out.splitlines() if _SUMMARY.search(ln) and _TOOK.search(ln)]
    if lines:
        for n, word in _SUMMARY.findall(lines[-1]):
            counts["passed" if word == "passed" else "failed"] += int(n)
    return outcomes, counts, out[-2000:]


def refactor_checks(row: dict, run_dir: Path, python: str, timeout: int = 300) -> dict:
    """RefactorBench's own structure (AST) checks. They read files as `../src/...`, so they
    run from the repository's tests folder."""
    # next to the copy, not in the user's temp folder (its path may contain a space)
    with tempfile.TemporaryDirectory(dir=run_dir.parent, prefix="odecheck") as tmp:
        check = Path(tmp) / "test_refactor_check.py"  # default pytest name pattern
        check.write_text(row["check"], encoding="utf-8")
        outcomes, _, _ = run_suite(run_dir, python, [str(check)], timeout,
                                   cwd=run_dir / "tests")
    passed_n = sum(s == "PASSED" for s in outcomes.values())
    return {"checks": len(outcomes), "checks_passed": passed_n,
            "ok": bool(outcomes) and passed_n == len(outcomes)}


def grade_refactor(row: dict, run_dir: Path, python: str, baseline: list[str],
                   timeout: int = 900, keep: float = 0.98) -> dict:
    """Solved: every RefactorBench check passes, no test that passed before the refactor fails
    now, and at least `keep` of them still pass (a refactor may rename a few tests)."""
    checks = refactor_checks(row, run_dir, python)
    outcomes, counts, _ = run_suite(run_dir, python, ["tests"], timeout)
    broken = [t for t in baseline if outcomes.get(t) in ("FAILED", "ERROR")]
    still = sum(outcomes.get(t) == "PASSED" for t in baseline)
    solved = checks["ok"] and not broken and still >= keep * len(baseline)
    return {"solved": solved, "checks": checks["checks"], "checks_passed": checks["checks_passed"],
            "baseline": len(baseline), "baseline_still_pass": still, "baseline_broken": len(broken),
            "suite_passed": counts["passed"], "suite_failed": counts["failed"]}


_REPORTED = re.compile(r"COMMIT:\s*\W*([0-9a-f]{7,40})\b", re.IGNORECASE)


def staged_tree(run_dir: Path) -> str:
    """Tree id of everything in the working copy (staged first; ignored files left out)."""
    git(run_dir, "add", "-A")
    return git(run_dir, "write-tree").stdout.strip()


def grade_run_action(answer: str, run_dir: Path, expected_tree: str) -> dict:
    """The agent committed its refactoring: solved when HEAD has the requested message, holds
    exactly the files graded in turn 2 (`expected_tree`), nothing is left uncommitted, and the
    hash in the answer's last `COMMIT:` line is HEAD's."""
    found = _REPORTED.findall(answer or "")
    head = git(run_dir, "rev-parse", "HEAD", check=False).stdout.strip()
    message = git(run_dir, "log", "-1", "--format=%s", check=False).stdout.strip()
    head_tree = git(run_dir, "rev-parse", "HEAD^{tree}", check=False).stdout.strip()
    clean = not git(run_dir, "status", "--porcelain", check=False).stdout.strip()
    res = {"message_ok": message == COMMIT_MESSAGE, "same_files": head_tree == expected_tree,
           "clean": clean, "hash_ok": bool(found) and head.startswith(found[-1].lower())}
    res["solved"] = all(res.values())
    return res
