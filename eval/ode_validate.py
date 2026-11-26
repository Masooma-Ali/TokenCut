"""ODE phase 5: check which practice tasks can be graded on this machine (no model calls).

Same rule as eval/stepd_validate.py (FAIL_TO_PASS fails before the official fix and passes
after it; PASS_TO_PASS passes after it), but it fetches each task's commit on its own (small
downloads), keeps the grading worktree in LF, and writes to eval/results/ode/validation.csv,
leaving the step D file alone.

    python -m eval.ode_validate $(cat eval/results/stepd/task_list.txt)
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import logging

from eval import ode_run as R
from eval import ode_tasks as T
from eval import stepd_validate
from eval.stepd_validate import load_rows, validate

log = logging.getLogger(__name__)
# The step D checker reads pytest's output with the Windows code page; use the UTF-8 reader.
stepd_validate.run_tests = T.run_tests

FIELDS = ["instance_id", "base_commit", "created_at", "version", "f2p", "p2p", "f2p_pass_before",
          "f2p_pass_after", "p2p_pass_after", "usable", "reason"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--repo", type=Path, default=Path(os.environ.get(
        "ODE_REPO", R.ROOT / "swebench_repos" / "sympy__sympy")))
    parser.add_argument("--work", type=Path, default=Path(os.environ.get(
        "ODE_WORK", "E:/FYP/swebench-envs/sympy-work")))
    parser.add_argument("--python", default=os.environ.get(
        "ODE_PYTHON", "E:/FYP/swebench-envs/sympy/.venv/Scripts/python.exe"))
    parser.add_argument("--out", type=Path, default=R.RESULTS / "validation.csv")
    args = parser.parse_args(argv)
    rows = load_rows(args.tasks)
    done = []
    for task in args.tasks:
        if task not in rows:
            print(f"{task}: not in the SWE-bench Lite cache", flush=True)
            continue
        row = rows[task]
        try:
            R.ensure_commit(args.repo, row["base_commit"])
            R.ensure_worktree(args.repo, args.work, row["base_commit"])
            res = validate(row, args.work, args.python, 900)
        except Exception as exc:  # report and go on with the next task
            log.warning("could not validate %s", task, exc_info=True)
            res = {"instance_id": task, "usable": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
        done.append(res)
        print(f"{task}: usable={res['usable']} ({res['reason']})", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(done)
    print(f"usable: {sum(bool(r['usable']) for r in done)} of {len(done)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
