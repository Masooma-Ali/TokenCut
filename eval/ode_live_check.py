"""ODE phase 6 live check: one real Claude Code session through the gateway with ODE choosing.

A fresh copy of fixtures/sample_repo, the LiteLLM connector + a gateway with
TOKENCUT_ODE=<shadow|on>, then Claude Code headless for five typed tasks in ONE session
(explain, bugfix, feature, tests, commit), so ODE decides at each typed task. Afterwards the
gateway log gives, per request: the choice (or "kept"), the model asked for and the model sent,
billed usage and list-price cost. Spends Claude / ChatGPT plan usage: run only with a go-ahead.

    python -m eval.ode_live_check shadow
    python -m eval.ode_live_check on

Private output (gitignored): logs/ode-live/<mode>/ (transcripts, gateway log). Printed: a
numbers-only summary, also written to logs/ode-live/<mode>/summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from eval import ode_run as R
from tokencut import launch, userconfig
from tokencut.ode.reward import model_for, request_cost, usage_from_dict

ROOT = R.ROOT
OUT = ROOT / "logs" / "ode-live"
REPO_COPY = Path(os.environ.get("ODE_LIVE_REPO", "E:/FYP/ode-live/repo"))
PY = sys.executable
TURN_TIMEOUT_S = 15 * 60
TASKS = (
    ("explain", ("Explain how the cart total is computed in this project: which files and "
                 "functions are involved? Do not change any files.")),
    ("bugfix", "shop/utils/broken.py has a syntax error. Fix it so the file imports cleanly."),
    ("feature", ("Add a function apply_coupon(total, percent) to shop/utils/pricing.py that "
                 "returns the total reduced by percent (0-100), rounded to 2 decimals.")),
    ("tests", (f"Add pytest tests for apply_coupon in tests/test_pricing.py and run them with "
               f"{PY} -m pytest tests.")),
    ("run_action", "Commit all changes with git, with the message \"Live check\"."),
)


def fresh_repo() -> None:
    if REPO_COPY.exists():
        R.remove_dir(REPO_COPY)
    shutil.copytree(ROOT / "fixtures" / "sample_repo", REPO_COPY)
    for args in (["init", "-q"], ["config", "user.name", "ode"], ["config", "user.email",
                 "ode@local"], ["add", "-A"], ["commit", "-q", "-m", "start"]):
        subprocess.run(["git", *args], cwd=REPO_COPY, check=True, capture_output=True)


def start(mode: str, log_dir: Path) -> tuple[list[subprocess.Popen], dict, dict]:
    if R._up(f"http://127.0.0.1:{R.PORT}/tokencut/health"):
        raise RuntimeError(f"a gateway is already running on port {R.PORT}; close it first")
    config = userconfig.load()
    env = {k: v for k, v in os.environ.items() if not k.startswith("TOKENCUT_")
           or k in ("TOKENCUT_HOME", "TOKENCUT_ROUTER_DIR", "TOKENCUT_ROUTER_REPO")}
    started: list[subprocess.Popen] = []
    connector = launch._start_connector(config, env, started)
    genv = userconfig.gateway_env(config, connector, env)
    genv.update({**R.GATEWAY_SETTINGS, "TOKENCUT_ODE": mode}, TOKENCUT_LOG_DIR=str(log_dir),
                TOKENCUT_PORT=str(R.PORT))
    genv.pop("TOKENCUT_ODE_FORCE_ARM", None)
    genv["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    out = (log_dir.parent / "gateway.out").open("ab")
    started.append(subprocess.Popen([sys.executable, "-m", "tokencut.gateway"], env=genv,
                                    stdout=out, stderr=subprocess.STDOUT))
    health_url = f"http://127.0.0.1:{R.PORT}/tokencut/health"
    for _ in range(360):
        if R._up(health_url):
            break
        time.sleep(0.5)
    else:
        R.stop(started)
        raise RuntimeError("gateway did not start (see gateway.out)")
    import urllib.request
    with urllib.request.urlopen(health_url, timeout=5) as r:
        health = json.loads(r.read())
    if health.get("ode") != mode or not health.get("ode_choosing"):
        R.stop(started)
        raise RuntimeError(f"ODE is not choosing: {health}")
    return started, userconfig.agent_env(config, env), health


def summarize(log_dir: Path) -> list[dict]:
    rows = []
    for rec in R.log_lines(log_dir):
        ode = rec.get("ode") or {}
        choice = ode.get("choice") or {}
        usage = rec.get("usage")
        model = ode.get("model_sent") or (rec.get("analysis") or {}).get("model", "")
        cost = None
        if usage:
            try:
                cost = round(request_cost(model_for(model), usage_from_dict(usage)), 6)
            except ValueError:
                cost = None
        rows.append({"agent_request": bool(choice), "decided": choice.get("decided"),
                     "arm": choice.get("arm"), "previous": choice.get("previous"),
                     "explored": choice.get("explored"),
                     "router_task": (ode.get("task_numbers") or {}).get("task"),
                     "model_requested": ode.get("model_requested"),
                     "model_sent": ode.get("model_sent"), "model": model,
                     "status": rec.get("status"), "cost_usd": cost})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("shadow", "on"))
    args = parser.parse_args(argv)
    base = OUT / args.mode
    if base.exists():
        shutil.rmtree(base)
    log_dir = base / "gateway"
    log_dir.mkdir(parents=True)
    fresh_repo()
    processes, env, health = start(args.mode, log_dir)
    turns = []
    try:
        session = str(uuid.uuid4())
        for n, (kind, prompt) in enumerate(TASKS, 1):
            seen = len(R.log_lines(log_dir))
            t0 = time.time()
            status, _ = R.run_turn(n, prompt, session, REPO_COPY, base, env,
                                        time.time() + TURN_TIMEOUT_S)
            recs = summarize(log_dir)[seen:]
            decided = [r for r in recs if r["decided"]]
            turns.append({"turn": n, "kind": kind, "status": status,
                          "wall_s": round(time.time() - t0),
                          "requests": len(recs), "agent_requests": sum(r["agent_request"] for r in recs),
                          "decisions": [{k: r[k] for k in ("arm", "previous", "explored",
                                                           "router_task")} for r in decided],
                          "models_sent": sorted({r["model"] for r in recs if r["model"]}),
                          "cost_usd": round(sum(r["cost_usd"] or 0 for r in recs), 4),
                          "errors": sum(1 for r in recs if r["status"] not in (200, None))})
            print(json.dumps(turns[-1]), flush=True)
            if status != "completed":
                break
    finally:
        R.stop(processes)
        R.stop_leftovers(PY)
    git_log = subprocess.run(["git", "log", "--oneline"], cwd=REPO_COPY, capture_output=True,
                             text=True, check=False).stdout.strip().splitlines()
    summary = {"mode": args.mode, "health": {k: health.get(k) for k in ("ode", "ode_arms",
                                                                         "gpt_upstream")},
               "turns": turns, "total_cost_usd": round(sum(t["cost_usd"] for t in turns), 4),
               "commits": len(git_log)}
    (base / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("mode", "total_cost_usd", "commits")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
