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
from dataclasses import dataclass
from pathlib import Path

from unidiff import PatchSet

from eval.loader import gold_from_patch
from eval.stepd_validate import _RESULT, apply_patch, passed, test_files
from tokencut.contracts import ChunkKind
from tokencut.ingestion.python_parser import PythonParser

TURN_TYPES = ("explain", "bugfix", "tests")
SIGNALS = {"explain": "explain_locates", "bugfix": "swebench_tests", "tests": "swt_fail_to_pass"}
PROMPTS = (
    ("Here is an issue reported in this repository.\n\n{issue}\n\nDo not change any files yet. "
     "Explain where in the code this behaviour is handled: name the file(s) and the function(s) "
     "involved, and briefly explain how they work. Use {python} for every Python or pytest "
     "command."),
    "Now make the code change in the repository that resolves the issue. Use the same Python.",
    ("Now add a regression test for this issue to the repository's existing test suite: a test "
     "that fails without your fix and passes with it. Run it with the same Python."),
)


def prompts(row: dict, python: str) -> list[str]:
    return [PROMPTS[0].format(issue=row["problem_statement"], python=python), *PROMPTS[1:]]


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
