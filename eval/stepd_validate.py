"""Step D, step 1: check which SWE-bench Lite tasks can be graded on this machine.

A task is usable when, at its own base commit:
  * its FAIL_TO_PASS tests fail before the official fix (the bug is real here),
  * they pass after the official fix (the grader can recognise a correct fix),
  * its PASS_TO_PASS tests pass after the official fix (the environment is sound).

    python -m eval.stepd_validate sympy__sympy-24909 sympy__sympy-24213 ...
        --repo swebench_repos/sympy__sympy --work C:/Users/pc/stepd-validate
        --python C:/Users/pc/tokencut-sympy/.venv/Scripts/python.exe
"""

import argparse
import csv
import json
import re
import subprocess
import tempfile
from pathlib import Path

from eval.loader import DEFAULT_CACHE_DIR

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "stepd"
_RESULT = re.compile(r"^(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\s+(\S+)", re.MULTILINE)


def load_rows(ids: list[str]) -> dict[str, dict]:
    rows = json.loads((DEFAULT_CACHE_DIR / "swebench_lite_test.json").read_text(encoding="utf-8"))
    rows = [r.get("row", r) for r in rows]
    return {r["instance_id"]: r for r in rows if r["instance_id"] in ids}


def git(work: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=work, capture_output=True, text=True, check=check)


def apply_patch(work: Path, patch: str) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8", newline="\n") as f:
        f.write(patch)
    return git(work, "apply", f.name, check=False).returncode == 0


def test_files(test_patch: str) -> list[str]:
    return sorted({line.split()[2][2:] for line in test_patch.splitlines() if line.startswith("diff --git")})


def run_tests(work: Path, python: str, files: list[str], timeout: int) -> tuple[dict[str, str], str]:
    """{test name: outcome} from pytest -rA, keyed by the function name SWE-bench uses."""
    try:
        proc = subprocess.run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA", *files],
                              cwd=work, capture_output=True, text=True, timeout=timeout, check=False)
        out = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return {}, "timeout"
    outcomes: dict[str, str] = {}
    for status, node in _RESULT.findall(out):
        outcomes[node.split("::")[-1]] = status
    return outcomes, out[-2000:]


def passed(outcomes: dict[str, str], names: list[str]) -> list[str]:
    return [n for n in names if outcomes.get(n) == "PASSED"]


def validate(row: dict, work: Path, python: str, timeout: int) -> dict:
    iid, base = row["instance_id"], row["base_commit"]
    f2p, p2p = json.loads(row["FAIL_TO_PASS"]), json.loads(row["PASS_TO_PASS"])
    result = {"instance_id": iid, "base_commit": base, "created_at": row["created_at"][:10],
              "version": row.get("version", ""), "f2p": len(f2p), "p2p": len(p2p)}
    checkout = git(work, "checkout", "-q", "-f", base, check=False)
    git(work, "clean", "-qfd", check=False)
    if checkout.returncode != 0:
        return {**result, "usable": False, "reason": "checkout failed: " + checkout.stderr[-200:]}
    files = test_files(row["test_patch"])
    if not apply_patch(work, row["test_patch"]):
        return {**result, "usable": False, "reason": "test patch does not apply"}
    before, _ = run_tests(work, python, files, timeout)
    f2p_before = passed(before, f2p)
    if not apply_patch(work, row["patch"]):
        return {**result, "usable": False, "reason": "gold patch does not apply"}
    after, log_after = run_tests(work, python, files, timeout)
    f2p_after, p2p_after = passed(after, f2p), passed(after, p2p)
    usable = not f2p_before and len(f2p_after) == len(f2p) and len(p2p_after) == len(p2p)
    reason = "ok" if usable else (
        "F2P already passes before the fix" if f2p_before else
        f"after fix: F2P {len(f2p_after)}/{len(f2p)}, P2P {len(p2p_after)}/{len(p2p)}"
        + (" (" + log_after.strip().splitlines()[-1][:120] + ")" if log_after.strip() else ""))
    git(work, "checkout", "-q", "-f", base, check=False)
    git(work, "clean", "-qfd", check=False)
    return {**result, "f2p_pass_before": len(f2p_before), "f2p_pass_after": len(f2p_after),
            "p2p_pass_after": len(p2p_after), "usable": usable, "reason": reason}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--repo", type=Path, required=True, help="clone of the task's repository")
    parser.add_argument("--work", type=Path, required=True, help="worktree used for validation")
    parser.add_argument("--python", required=True, help="interpreter with the repo's test dependencies")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)

    rows = load_rows(args.ids)
    if not args.work.exists():
        subprocess.run(["git", "worktree", "add", "-q", "--detach", str(args.work)], cwd=args.repo, check=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "validation.csv"
    done = []
    for iid in args.ids:
        if iid not in rows:
            print(f"{iid}: not in SWE-bench Lite cache")
            continue
        res = validate(rows[iid], args.work, args.python, args.timeout)
        done.append(res)
        print(f"{iid}: usable={res['usable']} ({res['reason']})", flush=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            fields = ["instance_id", "base_commit", "created_at", "version", "f2p", "p2p", "f2p_pass_before",
                      "f2p_pass_after", "p2p_pass_after", "usable", "reason"]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(done)


if __name__ == "__main__":
    main()
