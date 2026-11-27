"""ODE task set 2: feature / refactor / run_action prompts, graders and the task builder."""

import json
import subprocess
import sys
from pathlib import Path

from eval import ode_build_tasks2 as B
from eval import ode_run as R
from eval import ode_tasks as T

PY = sys.executable


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          check=True).stdout


def make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding="utf-8", newline="\n")
    git(repo, "init", "-q")
    git(repo, "config", "core.autocrlf", "false")
    git(repo, "config", "user.name", "t")
    git(repo, "config", "user.email", "t@t")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "start")
    return repo


# ---------- prompts ----------

def test_turn_types_and_prompts_per_kind():
    assert T.turn_types({}) == ("explain", "bugfix", "tests")
    assert T.turn_types({"kind": "feature"}) == ("explain", "feature", "tests")
    assert T.turn_types({"kind": "refactor"}) == ("explain", "refactor", "run_action")
    feature = T.prompts({"kind": "feature", "problem_statement": "Add {x} please"}, "py.exe")
    assert "Add {x} please" in feature[0] and "py.exe" in feature[0]
    assert "exactly the names and signatures" in feature[1]
    refactor = T.prompts({"kind": "refactor", "problem_statement": "Rename a to b"}, "py.exe")
    assert "Rename a to b" in refactor[0]
    assert T.COMMIT_MESSAGE in refactor[2] and "COMMIT:" in refactor[2]
    assert all(t in T.SIGNALS for t in ("feature", "refactor", "run_action"))


# ---------- feature explain gold ----------

def test_feature_gold_names_the_class_around_a_new_method_not_the_new_name(tmp_path):
    src = ("class Point:\n"
           "    def __init__(self, x):\n"
           "        self.x = x\n"
           "\n"
           "    def norm(self):\n"
           "        return abs(self.x)\n"
           "\n"
           "    def other(self):\n"
           "        return 1\n")
    repo = make_repo(tmp_path, {"pkg/point.py": src})
    new = src.replace("        return abs(self.x)\n",
                      "        return abs(self.x)\n\n    @property\n    def coordinates(self):\n"
                      "        return (self.x,)\n")
    (repo / "pkg/point.py").write_text(new, encoding="utf-8", newline="\n")
    patch = git(repo, "diff")
    git(repo, "checkout", "-q", "--", ".")
    base = git(repo, "rev-parse", "HEAD").strip()
    gold = T.feature_gold({"patch": patch, "base_commit": base}, repo)
    assert gold.files == ("pkg/point.py",)
    assert gold.functions == ("Point",)


# ---------- refactor grading ----------

CHECK = '''import unittest

class TestRename(unittest.TestCase):
    def test_new_name(self):
        text = open("../src/pkg/mod.py", encoding="utf-8").read()
        self.assertIn("def new_name", text)
        self.assertNotIn("def old_name", text)
'''


def refactor_repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path, {
        "src/pkg/__init__.py": "",
        "src/pkg/mod.py": "def old_name():\n    return 1\n",
        "tests/test_mod.py": ("import sys, pathlib\n"
                              "sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / 'src'))\n"
                              "from pkg import mod\n\n"
                              "def test_value():\n    assert getattr(mod, 'new_name', "
                              "getattr(mod, 'old_name', None))() == 1\n\n"
                              "def test_other():\n    assert True\n"),
    })


def test_run_suite_reads_outcomes_and_final_counts(tmp_path):
    repo = make_repo(tmp_path, {"tests/test_a.py": ("def test_ok():\n    assert True\n\n"
                                                    "def test_bad():\n    assert False\n")})
    outcomes, counts, _ = T.run_suite(repo, PY, ["tests"], 120)
    assert outcomes == {"tests/test_a.py::test_ok": "PASSED",
                        "tests/test_a.py::test_bad": "FAILED"}
    assert counts == {"passed": 1, "failed": 1}


def test_run_suite_keeps_whole_names_with_spaces(tmp_path):
    folder = tmp_path / "with space"
    folder.mkdir()
    repo = make_repo(folder, {"tests/test_p.py": (
        "import pytest\n\n@pytest.mark.parametrize('v', [1, 2], ids=['a b', 'c d'])\n"
        "def test_p(v):\n    assert v == 1\n")})
    check = repo / "tests" / "test_p.py"
    outcomes, counts, _ = T.run_suite(repo, PY, [str(check)], 120, cwd=repo / "tests")
    assert sorted(outcomes.values()) == ["FAILED", "PASSED"]
    assert all(" " in name for name in outcomes)
    assert counts == {"passed": 1, "failed": 1}


def test_grade_refactor_needs_checks_and_unbroken_tests(tmp_path):
    repo = refactor_repo(tmp_path)
    row = {"check": CHECK}
    baseline = ["tests/test_mod.py::test_value", "tests/test_mod.py::test_other"]
    before = T.grade_refactor(row, repo, PY, baseline, timeout=120)
    assert not before["solved"] and before["checks_passed"] == 0
    (repo / "src/pkg/mod.py").write_text("def new_name():\n    return 1\n", encoding="utf-8")
    after = T.grade_refactor(row, repo, PY, baseline, timeout=120)
    assert after["solved"], after
    assert after["checks_passed"] == after["checks"] == 1
    (repo / "src/pkg/mod.py").write_text("def new_name():\n    return 2\n", encoding="utf-8")
    broken = T.grade_refactor(row, repo, PY, baseline, timeout=120)
    assert not broken["solved"] and broken["baseline_broken"] == 1


# ---------- run_action (commit) grading ----------

def test_grade_run_action_needs_the_commit_the_hash_and_no_extra_changes(tmp_path):
    repo = make_repo(tmp_path, {"a.py": "x = 1\n"})
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    tree = T.staged_tree(repo)
    # not committed yet
    assert not T.grade_run_action("COMMIT: 1234567", repo, tree)["solved"]
    git(repo, "commit", "-q", "-m", T.COMMIT_MESSAGE)
    head = git(repo, "rev-parse", "--short", "HEAD").strip()
    ok = T.grade_run_action(f"Done.\nCOMMIT: `{head}`", repo, tree)
    assert ok["solved"]
    assert not T.grade_run_action("Done.\nCOMMIT: 0000000", repo, tree)["hash_ok"]
    assert not T.grade_run_action("Done, committed.", repo, tree)["solved"]
    (repo / "a.py").write_text("x = 3\n", encoding="utf-8")  # changed a file after turn 2
    git(repo, "commit", "-q", "-am", T.COMMIT_MESSAGE)
    head = git(repo, "rev-parse", "--short", "HEAD").strip()
    late = T.grade_run_action(f"COMMIT: {head}", repo, tree)
    assert late["message_ok"] and not late["same_files"] and not late["solved"]


# ---------- builder ----------

def test_clean_body_drops_sympy_template():
    body = ("Want point.coordinates\n\n<!-- Your title above -->\n\n#### References to other "
            "Issues or PRs\n<!-- fixes -->\nFixes #123\n\n#### Brief description of what is "
            "fixed or changed\nAdded it\n\n#### Other comments\n\n\n#### Release Notes\n"
            "<!-- BEGIN RELEASE NOTES -->\n* geometry\n    * new property\n"
            "<!-- END RELEASE NOTES -->\n")
    out = B.clean_body(body)
    assert "<!--" not in out and "References" not in out and "Fixes #123" not in out
    assert "Other comments" not in out
    assert out.startswith("Want point.coordinates")
    assert "Brief description of what is fixed or changed:\nAdded it" in out
    assert "Release Notes:\n* geometry" in out


def test_new_components_lists_only_new_names(tmp_path):
    repo = make_repo(tmp_path, {"m.py": "class A:\n    def f(self):\n        return 1\n"})
    base = git(repo, "rev-parse", "HEAD").strip()
    (repo / "m.py").write_text(
        "class A:\n    def f(self):\n        return 1\n\n    @property\n    def g(self):\n"
        "        \"\"\"The g value.\n\n        More.\"\"\"\n        return 2\n\n"
        "class B:\n    def h(self, x=1):\n        return x\n\ndef top(a, *b):\n    pass\n",
        encoding="utf-8", newline="\n")
    git(repo, "commit", "-q", "-am", "add")
    head = git(repo, "rev-parse", "HEAD").strip()
    assert B.new_components(repo, base, head, ["m.py"]) == [
        "- m.py: `@property def A.g(self)` - The g value.",
        "- m.py: `class B`",
        "- m.py: `def top(a, *b)`",
    ]


def test_refactor_gold_uses_code_words_and_their_files(tmp_path):
    repo = make_repo(tmp_path, {
        "src/pkg/utils.py": "def super_len(o):\n    pass\n\ndef update():\n    pass\n",
        "src/pkg/models.py": "from .utils import super_len\n",
        "src/pkg/hooks.py": "def content():\n    pass\n",
    })
    check = "open('../src/pkg/models.py')\nopen('../tests/test_x.py')\nopen('../src/gone.py')"
    files, funcs = B.refactor_gold(repo, check, "Rename super_len to complex_len and update "
                                                 "the content of every caller")
    assert files == ["src/pkg/models.py", "src/pkg/utils.py"]
    assert funcs == ["super_len"]


# ---------- runner ----------

def test_load_tasks_adds_kind_env_and_baseline(tmp_path, monkeypatch):
    tasks2 = tmp_path / "ode_tasks2.json"
    tasks2.write_text(json.dumps({
        "tasks": [{"instance_id": "refactorbench__flask-x", "kind": "refactor", "env": "flask"},
                  {"instance_id": "sympy__sympy-1", "kind": "feature", "env": "sympy"}],
        "baselines": {"flask": ["tests/t.py::a"]}}), encoding="utf-8")
    monkeypatch.setattr(R, "TASKS2", tasks2)
    monkeypatch.setattr(R, "load_rows", lambda ids: {"sympy__sympy-9": {"instance_id": "sympy__sympy-9"}})
    rows = R.load_tasks(["sympy__sympy-9", "refactorbench__flask-x"])
    assert rows["sympy__sympy-9"]["kind"] == "bugfix" and rows["sympy__sympy-9"]["env"] == "sympy"
    assert rows["refactorbench__flask-x"]["baseline"] == ["tests/t.py::a"]
    assert "sympy__sympy-1" not in rows


def test_env_paths_can_be_overridden(monkeypatch):
    monkeypatch.setenv("ODE_FLASK_PYTHON", "C:/py/python.exe")
    monkeypatch.setenv("ODE_FLASK_WORK", "D:/w")
    paths = R.env_paths("flask")
    assert paths["python"] == "C:/py/python.exe" and paths["work"] == Path("D:/w")
    assert paths["repo"] == R.ENVS["flask"]["repo"]
