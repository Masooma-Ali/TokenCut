"""ODE phase 4: practice-session checkers and the runner, on a tiny git repo (no model calls).

The tiny repo has a bug (add() subtracts), an official fix and official tests, like a
SWE-bench task, so every checker can be run for real with pytest.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval import ode_run as R
from eval import ode_tasks as T

PYTHON = sys.executable


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


@pytest.fixture
def task(tmp_path):
    """A one-file project with a bug, its official fix and official tests (SWE-bench shape)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "core.autocrlf", "false")
    git(repo, "config", "user.email", "t@t")
    git(repo, "config", "user.name", "t")
    write(repo / "calc.py", "def add(a, b):\n    return a - b\n\n\ndef zero():\n    return 0\n")
    write(repo / "tests" / "__init__.py", "")
    write(repo / "tests" / "test_calc.py",
          "from calc import zero\n\n\ndef test_zero():\n    assert zero() == 0\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD").strip()
    # the official fix
    write(repo / "calc.py", "def add(a, b):\n    return a + b\n\n\ndef zero():\n    return 0\n")
    patch = git(repo, "diff")
    git(repo, "checkout", "-q", "--", ".")
    # the official tests
    write(repo / "tests" / "test_calc.py",
          "from calc import add, zero\n\n\ndef test_zero():\n    assert zero() == 0\n\n\n"
          "def test_add():\n    assert add(2, 3) == 5\n")
    test_patch = git(repo, "diff")
    git(repo, "checkout", "-q", "--", ".")
    row = {"instance_id": "tiny__calc-1", "base_commit": base, "patch": patch,
           "test_patch": test_patch, "problem_statement": "add(2, 3) returns -1",
           "FAIL_TO_PASS": json.dumps(["test_add"]), "PASS_TO_PASS": json.dumps(["test_zero"])}
    work = tmp_path / "work"
    git(repo, "worktree", "add", "-q", "--detach", str(work), base)
    return row, repo, work, tmp_path


# ---------- explain ----------

GOLD = T.Gold(("sympy/core/numbers.py",), ("__eq__",))


@pytest.mark.parametrize(("answer", "solved"), [
    ("It is in sympy/core/numbers.py, in Float.__eq__, which compares ...", True),
    ("See numbers.py: the method __eq__ of Float.", True),                  # file name is enough
    (r"In sympy\core\numbers.py the __eq__ method ...", True),               # Windows path
    ("It is in sympy/core/numbers.py, somewhere in Float.", False),          # no function
    ("The comparison happens in __eq__ in sympy/core/expr.py.", False),      # wrong file
    ("mynumbers.py defines __eq__", False),                                  # not the same file
])
def test_explain_needs_the_right_file_and_function(answer, solved):
    assert T.explain_locates(answer, GOLD)["solved"] is solved


def test_explain_with_a_module_level_fix_needs_only_the_file():
    assert T.explain_locates("see setup.py", T.Gold(("setup.py",), ()))["solved"]


def test_gold_locations_read_the_real_fix(task):
    row, _, work, _ = task
    assert T.gold_locations(row, work) == T.Gold(("calc.py",), ("add",))


def test_gold_counts_an_added_function_not_its_neighbours(task):
    """Pilot (sympy 21055): the fix adds a new function between two old ones. The added one
    counts; the neighbours, whose lines did not change, do not."""
    row, _, work, _ = task
    patch = ("diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n"
             "@@ -2,4 +2,8 @@ def add(a, b):\n     return a - b\n \n \n+def mul(a, b):\n"
             "+    return a * b\n+\n+\n def zero():\n     return 0\n")
    gold = T.gold_locations({**row, "patch": patch}, work)
    assert gold == T.Gold(("calc.py",), ("mul",))
    assert T.explain_locates("calc.py needs a new mul function", gold)["solved"]
    assert not T.explain_locates("calc.py, in add", gold)["solved"]


# ---------- bugfix ----------

def test_bugfix_solved_by_the_real_fix_only(task):
    row, _, work, _ = task
    good = T.grade_bugfix(row, row["patch"], work, PYTHON)
    assert good["solved"] and (good["f2p_pass"], good["p2p_pass"]) == (1, 1)
    assert not T.grade_bugfix(row, "", work, PYTHON)["solved"]  # no change: test_add fails
    broken = T.grade_bugfix(row, "not a diff", work, PYTHON)
    assert not broken["solved"] and broken["note"] == "agent diff does not apply"


# ---------- tests (SWT rule) ----------

def test_regression_test_must_fail_before_and_pass_after(task):
    row, _, work, _ = task
    res = T.grade_tests(row, row["test_patch"], work, PYTHON)
    assert res["solved"] and res["fail_to_pass"] == 1


def test_a_test_that_always_passes_does_not_count(task):
    row, _, work, _ = task
    T.reset(work, row["base_commit"])
    write(work / "tests" / "test_calc.py",
          "from calc import zero\n\n\ndef test_zero():\n    assert zero() == 0\n\n\n"
          "def test_trivial():\n    assert True\n")
    diff = git(work, "diff")
    T.reset(work, row["base_commit"])
    res = T.grade_tests(row, diff, work, PYTHON)
    assert not res["solved"] and res["fail_to_pass"] == 0


def test_unicode_test_output_is_read(task):
    """sympy 23191 prints Unicode in its tests; reading it with the Windows code page broke
    grading. The runner decodes UTF-8."""
    row, _, work, _ = task
    T.reset(work, row["base_commit"])
    write(work / "tests" / "test_uni.py",
          "def test_pretty():\n    print('∂ ∫ √ ── π')\n    assert True\n")
    outcomes, _ = T.run_tests(work, PYTHON, ["tests/test_uni.py"], 300)
    T.reset(work, row["base_commit"])
    assert outcomes == {"test_pretty": "PASSED"}


def test_code_only_changes_are_not_regression_tests(task):
    row, _, work, _ = task
    res = T.grade_tests(row, row["patch"], work, PYTHON)
    assert not res["solved"] and res["note"] == "no test file changed"


def test_diff_helpers():
    diff = ("diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1 +1 @@\n-x\n+y\n"
            "diff --git a/tests/test_calc.py b/tests/test_calc.py\n@@ -1 +1 @@\n-a\n+b\n")
    assert T.diff_files(diff) == ["calc.py", "tests/test_calc.py"]
    kept = T.only_files(diff, T.is_test_file)
    assert kept.startswith("diff --git a/tests/test_calc.py") and "calc.py b/calc.py" not in kept
    assert T.is_test_file("sympy/core/tests/test_numbers.py")
    assert T.is_test_file("test_x.py") and not T.is_test_file("sympy/core/numbers.py")


# ---------- runner pieces ----------

def test_turn_numbers_sum_usage_price_it_and_take_the_task_numbers():
    records = [
        {"method": "POST", "ode": {"arm": "sonnet-standard", "model_sent": "claude-sonnet-5-5",
                                   "task_numbers": {"task": "bugfix", "p": 0.9,
                                                    "decision_point": True, "x": [0.5] * 13}},
         "usage": {"input_tokens": 2000, "cache_read_tokens": 30000, "cache_write_tokens": 1000,
                   "output_tokens": 500, "cache_write_1h_tokens": 1000}},
        {"method": "POST", "analysis": {"model": "claude-haiku-5-5"},  # a side request
         "usage": {"input_tokens": 1000, "output_tokens": 100}},
        {"method": "POST", "analysis": {"model": "some-other-model"},
         "usage": {"input_tokens": 5}},
    ]
    out = R.turn_numbers(records)
    assert out["requests"] == 3 and out["requests_ode"] == 1 and out["unpriced_requests"] == 1
    # Sonnet 0.016 (tests/test_ode_reward.py) + Haiku 1,000 x 0.10 + 100 x 0.50 per M
    assert out["cost_usd"] == pytest.approx(0.016 + 0.00015)
    assert out["router_task"] == "bugfix" and out["decision_point"] is True
    assert json.loads(out["x"]) == [0.5] * 13
    assert out["input"] == 3005 and out["cache_write_1h"] == 1000


def test_time_limit_kills_the_process_and_its_children(tmp_path):
    """Pilot finding: a hung turn ran 87 minutes past its limit. The runner must stop the agent
    AND the processes it started (here: a child that would outlive a plain kill)."""
    import time

    marker = tmp_path / "child_alive.txt"
    child = f"import time, pathlib\nfor i in range(60):\n    pathlib.Path(r'{marker}').write_text(str(i))\n    time.sleep(0.2)\n"
    parent = (f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', {child!r}])\n"
              "time.sleep(60)\n")
    started = time.time()
    with (tmp_path / "out.txt").open("w") as out:
        code = R.run_until([PYTHON, "-c", parent], tmp_path, None, out, time.time() + 2,
                           poll_s=0.2)
    assert code is None and time.time() - started < 20  # stopped near the 2-second limit
    time.sleep(1.0)
    seen = marker.read_text()
    time.sleep(1.0)
    assert marker.read_text() == seen  # the child stopped writing: the whole tree is gone


def test_run_until_returns_the_exit_code(tmp_path):
    import time

    with (tmp_path / "out.txt").open("w") as out:
        assert R.run_until([PYTHON, "-c", "import sys; sys.exit(3)"], tmp_path, None, out,
                           time.time() + 30, poll_s=0.1) == 3


def test_arm_order_rotates_per_task():
    arms = ["a", "b", "c"]
    assert R.arm_order(arms, 0) == ["a", "b", "c"]
    assert R.arm_order(arms, 1) == ["b", "c", "a"]
    assert R.arm_order(arms, 3) == ["a", "b", "c"]


def test_unknown_arm_is_refused(capsys):
    with pytest.raises(SystemExit):
        R.main(["tiny__calc-1", "--arms", "opus-standrad", "--fake-agent"])
    assert "opus-standrad" in capsys.readouterr().err


def test_grading_worktree_keeps_committed_line_endings(task):
    """Git for Windows may check out CRLF; SWE-bench patches are LF and must still apply."""
    row, repo, _, tmp = task
    git(repo, "config", "core.autocrlf", "true")
    crlf = tmp / "crlf-work"
    git(repo, "worktree", "add", "-q", "--detach", str(crlf), row["base_commit"])
    git(crlf, "config", "core.autocrlf", "true")
    (crlf / "calc.py").write_bytes((crlf / "calc.py").read_bytes().replace(b"\n", b"\r\n"))
    R.ensure_worktree(repo, crlf, row["base_commit"])
    assert b"\r\n" not in (crlf / "calc.py").read_bytes()
    assert T.grade_bugfix(row, row["patch"], crlf, PYTHON)["solved"]


def test_turn_3_is_graded_on_all_test_changes_since_the_start(task, monkeypatch):
    """Seen live: the agent edited the test file in turn 2 AND turn 3. A turn-3-only diff does
    not apply to the original code, so grading uses everything changed since the start."""
    row, _, work, tmp = task
    monkeypatch.setattr(R, "PRIVATE", tmp / "private")
    run = tmp / "run"
    R.prepare_run_dir(work, run, row["base_commit"])
    base = tmp / "private" / "x"
    base.mkdir(parents=True)
    args = argparse.Namespace(work=work, python=PYTHON)
    gold = T.Gold(("calc.py",), ("add",))
    # turn 2: the fix plus a first edit of the test file (official tests decide turn 2)
    assert R.apply_patch(run, row["patch"])
    write(run / "tests" / "test_calc.py",
          "from calc import add, zero\n\n\ndef test_zero():\n    assert zero() == 0\n")
    assert R.check_turn(2, row, {}, run, args, gold, base)["solved"] is True
    # turn 3: the regression test, in the same file
    write(run / "tests" / "test_calc.py",
          "from calc import add, zero\n\n\ndef test_zero():\n    assert zero() == 0\n\n\n"
          "def test_add():\n    assert add(2, 3) == 5\n")
    res = R.check_turn(3, row, {}, run, args, gold, base)
    assert res["solved"] and res["note"] == ""


def test_fake_agent_runs_the_whole_session(task, monkeypatch):
    """Fresh copy, three turns, three checkers, three rows: the pipeline without any model."""
    row, repo, work, tmp = task
    monkeypatch.setattr(R, "PRIVATE", tmp / "private")
    args = argparse.Namespace(repo=repo, work=work, run_dir=tmp / "run", python=PYTHON,
                              fake_agent=True)
    rows = R.run_one(row, "haiku-standard", args)
    assert [(r["turn"], r["task_type"], r["status"], r["solved"]) for r in rows] == [
        (1, "explain", "completed", 1), (2, "bugfix", "completed", 1),
        (3, "tests", "completed", 1)]
    assert all(r["arm"] == "haiku-standard" and r["cost_usd"] == 0.0 for r in rows)
    assert json.loads(rows[1]["check"])["f2p_pass"] == 1
    # the agent's copy has no history to look the fix up in
    assert git(tmp / "run", "rev-list", "--count", "HEAD").strip() == "4"  # start + 3 turns
    assert (tmp / "private" / "tiny__calc-1" / "haiku-standard" / "turn2.diff").exists()


# ---------- leftover processes (phase 5a: a background pytest kept the run folder open) ----------

VENV = "E:/envs/sympy/.venv/Scripts/python.exe"


def proc(pid, parent, name, cmd="", exe=""):
    return {"ProcessId": pid, "ParentProcessId": parent, "Name": name, "CommandLine": cmd,
            "ExecutablePath": exe}


def test_leftover_roots_lifts_test_python_to_topmost_shell():
    procs = [
        proc(1, 0, "explorer.exe"),
        proc(10, 999, "bash.exe", "bash -c 'source snap.sh && pytest'"),  # parent gone
        proc(11, 10, "bash.exe", "bash -c ..."),
        proc(12, 11, "python.exe", r"E:\envs\sympy\.venv\Scripts\python.exe -m pytest sympy/polys"),
        proc(13, 12, "python.exe", "C:/Python311/python.exe -m pytest sympy/polys"),
    ]
    assert R.leftover_roots(procs, VENV) == [10]


def test_leftover_roots_matches_executable_path_and_stops_at_non_shell():
    procs = [
        proc(5, 1, "claude.exe", "claude -p"),
        proc(6, 5, "python.exe", "python -m pytest", exe=r"E:\envs\sympy\.venv\Scripts\python.exe"),
    ]
    assert R.leftover_roots(procs, VENV) == [6]  # claude.exe is not a shell: not lifted to it


def test_leftover_roots_ignores_unrelated_processes():
    procs = [
        proc(20, 1, "bash.exe", "bash -c 'git status'"),
        proc(21, 20, "python.exe", "E:/FYP/iteration1/TokenCut/.venv/Scripts/python.exe -m pytest"),
        proc(22, 1, "python.exe", "python -m tokencut.gateway"),
    ]
    assert R.leftover_roots(procs, VENV) == []


def test_remove_dir_retries_after_stopping_leftovers(tmp_path, monkeypatch):
    target = tmp_path / "run"
    target.mkdir()
    calls = {"rmtree": 0, "stopped": 0}
    real = R.shutil.rmtree

    def flaky(path, onerror=None):
        calls["rmtree"] += 1
        if calls["rmtree"] == 1:
            raise PermissionError("in use")
        real(path)

    monkeypatch.setattr(R.shutil, "rmtree", flaky)
    monkeypatch.setattr(R, "stop_leftovers", lambda py: calls.__setitem__("stopped", 1) or [7])
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    R.remove_dir(target, VENV)
    assert not target.exists() and calls == {"rmtree": 2, "stopped": 1}


def test_leftover_roots_never_picks_the_runner_or_its_ancestors():
    # the runner was started as `python -m eval.ode_run --python <venv python>` from a shell
    procs = [
        proc(30, 1, "bash.exe", f"bash -c 'python -m eval.ode_run --python {VENV}'"),
        proc(31, 30, "python.exe", f"python -m eval.ode_run --python {VENV}"),
        proc(40, 999, "bash.exe", "bash -c pytest"),
        proc(41, 40, "python.exe", f"{VENV} -m pytest"),
    ]
    assert R.leftover_roots(procs, VENV, me=31) == [40]
    assert R.leftover_roots(procs, VENV) == [30, 40]  # without `me` both would be chosen
