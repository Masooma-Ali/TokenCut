"""ODE phase 4: run practice sessions on forced arms and write one row per turn.

For every task and arm: a fresh copy of the repository at the task's base commit (no history),
the LiteLLM connector + a gateway with the arm forced (TOKENCUT_ODE=force), then Claude Code
headless for the three turns of eval/ode_tasks.py in ONE session (--session-id, --resume).
After each turn the turn's checker runs; the gateway log gives the turn's requests, billed
usage, list-price cost (tokencut/ode/reward.py) and the 13 task numbers.

    python -m eval.ode_run sympy__sympy-24909 --arms haiku-standard,opus-standard
    python -m eval.ode_run sympy__sympy-24909 --fake-agent        # whole pipeline, no model calls

Outputs (numbers only): eval/results/ode/turns.csv, eval/results/ode/state.json.
Private (gitignored): logs/ode/<task>/<arm>/ (transcripts, diffs, gateway log, test output).
Finished runs are skipped on restart. A usage-limit stop discards that run and exits; a file
eval/results/ode/STOP stops before the next run. This spends Claude / ChatGPT plan usage unless
--fake-agent is given: run it only with the team's go-ahead.

Paths (flags or environment): --repo (ODE_REPO, a clone with the needed commits; missing
commits are fetched one by one), --work (ODE_WORK, worktree used to grade), --run-dir
(ODE_RUN_DIR, where the agent works), --python (ODE_PYTHON, interpreter with the repo's test
dependencies).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

from eval import ode_tasks as T
from eval.stepd_validate import apply_patch, load_rows
from tokencut import launch, userconfig
from tokencut.ode.arms import ARM_NAMES
from tokencut.ode.reward import model_for, request_cost, usage_from_dict

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results" / "ode"
PRIVATE = ROOT / "logs" / "ode"
PORT = 8787
TIMEOUT_S = 45 * 60
MODEL_REQUESTED = "sonnet"  # what Claude Code asks for; the forced arm decides what is used
EFFORT = "medium"
START = "ode-start"  # tag on the agent copy's first commit: turns are graded on changes since it
LIMIT_WORDS = ("usage limit", "limit reached", "rate limit", "rate_limit", "resets")
# Gateway settings shared by every arm (the arm adds model + trim level). Task briefs and the
# system note are off: they are not A/B tested yet, and ODE should learn about the arms alone.
GATEWAY_SETTINGS = {"TOKENCUT_MODE": "trim", "TOKENCUT_TIER": "moderate",
                    "TOKENCUT_PROMPT_BRIEF": "0", "TOKENCUT_SYSTEM_NOTE": "0",
                    "TOKENCUT_ODE": "force"}

TURN_FIELDS = [
    "task", "arm", "turn", "task_type", "status", "solved", "signal", "check",
    "requests", "requests_ode", "unpriced_requests", "input", "cache_read", "cache_write",
    "cache_write_1h", "output", "cost_usd", "claude_reported_cost_usd", "router_task", "router_p",
    "decision_point", "x", "wall_s", "num_turns", "claude_version", "tokencut_commit",
    "started", "settings",
]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def append_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)


# ---------- the repository ----------

def ensure_commit(repo: Path, commit: str, tries: int = 3) -> None:
    """Fetch one commit (shallow) if the clone does not have it: small downloads survive a
    flaky connection better than a full clone."""
    if T.git(repo, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode == 0:
        return
    for attempt in range(1, tries + 1):
        if T.git(repo, "fetch", "-q", "--depth", "1", "origin", commit, check=False).returncode == 0:
            return
        time.sleep(5 * attempt)
    raise RuntimeError(f"could not fetch commit {commit}")


def ensure_worktree(repo: Path, work: Path, commit: str) -> None:
    """The grading worktree, with files exactly as committed (LF). Git for Windows converts to
    CRLF on checkout by default; SWE-bench patches are LF, so they would not apply."""
    T.git(repo, "config", "core.autocrlf", "false")
    if not (work / ".git").exists():
        T.git(repo, "worktree", "add", "-q", "--detach", str(work), commit)
        return
    T.git(work, "config", "core.autocrlf", "false")
    if T.git(work, "ls-files", "--eol", check=False).stdout.count("w/crlf"):
        T.git(work, "rm", "-q", "--cached", "-r", ".")  # rewrite every file without conversion
        T.git(work, "reset", "-q", "--hard")


def prepare_run_dir(work: Path, run_dir: Path, commit: str) -> None:
    """run_dir = the files of `commit` as a new one-commit git repo (no history to look up)."""
    T.reset(work, commit)
    if run_dir.exists():
        shutil.rmtree(run_dir, onerror=lambda f, p, _: (os.chmod(p, 0o700), f(p)))
    run_dir.mkdir(parents=True)
    names = T.git(work, "ls-files", "-z").stdout.split("\0")
    for name in filter(None, names):
        src = work / name
        if src.is_file():
            dst = run_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    T.git(run_dir, "init", "-q")
    T.git(run_dir, "config", "core.autocrlf", "false")  # diffs must apply to the grading copy
    commit_all(run_dir, "start")
    T.git(run_dir, "tag", START)


def commit_all(run_dir: Path, message: str) -> None:
    T.git(run_dir, "add", "-A")
    T.git(run_dir, "-c", "user.name=ode", "-c", "user.email=ode@local", "commit", "-q",
          "--allow-empty", "-m", message)


def diff_since(run_dir: Path, ref: str) -> str:
    T.git(run_dir, "add", "-A")
    return subprocess.run(["git", "diff", "--cached", "--binary", ref], cwd=run_dir,
                          capture_output=True, check=True).stdout.decode("utf-8", "replace")


# ---------- connector, gateway, agent ----------

def _up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status < 500
    except OSError:
        return False


def start_services(arm: str, log_dir: Path) -> tuple[list[subprocess.Popen], dict]:
    """LiteLLM connector (if set up) + a gateway with the arm forced. Returns (processes,
    environment for Claude Code)."""
    if _up(f"http://127.0.0.1:{PORT}/tokencut/health"):
        raise RuntimeError(f"a gateway is already running on port {PORT}; close it first")
    config = userconfig.load()
    env = {k: v for k, v in os.environ.items() if not k.startswith("TOKENCUT_")
           or k in ("TOKENCUT_HOME", "TOKENCUT_ROUTER_DIR", "TOKENCUT_ROUTER_REPO")}
    started: list[subprocess.Popen] = []
    connector = launch._start_connector(config, env, started)
    genv = userconfig.gateway_env(config, connector, env)
    genv.update(GATEWAY_SETTINGS, TOKENCUT_ODE_FORCE_ARM=arm, TOKENCUT_LOG_DIR=str(log_dir),
                TOKENCUT_PORT=str(PORT))
    genv["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    out = (log_dir.parent / "gateway.out").open("ab")
    started.append(subprocess.Popen([sys.executable, "-m", "tokencut.gateway"], env=genv,
                                    stdout=out, stderr=subprocess.STDOUT))
    health = f"http://127.0.0.1:{PORT}/tokencut/health"
    for _ in range(360):
        if _up(health):
            break
        time.sleep(0.5)
    else:
        stop(started)
        raise RuntimeError("gateway did not start (see gateway.out)")
    with urllib.request.urlopen(health, timeout=5) as r:
        h = json.loads(r.read())
    if h.get("ode") != "force" or h.get("ode_force_arm") != arm or h.get("mode") != "trim":
        stop(started)
        raise RuntimeError(f"gateway settings wrong: {h}")
    return started, userconfig.agent_env(config, env)


def stop(processes: list[subprocess.Popen]) -> None:
    for proc in reversed(processes):
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def claude_path() -> str:
    return shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude.exe")


def last_result(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    result: dict = {}
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result = event
    return result, raw[-3000:]


def kill_tree(proc: subprocess.Popen) -> None:
    """Stop a process and everything it started (Claude Code runs tools as child processes)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                       check=False)
    else:
        try:
            os.killpg(proc.pid, 9)
        except OSError:
            proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_until(cmd: list[str], cwd: Path, env: dict, out, deadline: float,
              poll_s: float = 1.0) -> int | None:
    """Run cmd until it ends (its exit code) or the deadline passes (None, process tree killed).

    subprocess.run(timeout=...) did not stop Claude Code on Windows (seen in the pilot: a turn
    ran 87 minutes past a 45-minute limit), so the clock is checked here and the whole process
    tree is killed."""
    kwargs = {"start_new_session": True} if os.name != "nt" else {}
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT,
                            **kwargs)
    while proc.poll() is None:
        if time.time() >= deadline:
            kill_tree(proc)
            return None
        time.sleep(poll_s)
    return proc.returncode


def run_turn(n: int, prompt: str, session: str, run_dir: Path, base: Path, env: dict,
             deadline: float) -> tuple[str, dict]:
    """One typed turn: `claude -p`, the first with --session-id, later ones --resume."""
    first = ["--session-id", session] if n == 1 else ["--resume", session]
    cmd = [claude_path(), "-p", prompt, *first, "--model", MODEL_REQUESTED, "--effort", EFFORT,
           "--output-format", "stream-json", "--verbose", "--permission-mode", "auto"]
    transcript = base / f"transcript_turn{n}.jsonl"
    if deadline - time.time() <= 0:
        return "timeout", {}
    with transcript.open("w", encoding="utf-8") as out:
        if run_until(cmd, run_dir, env, out, deadline) is None:
            result, _ = last_result(transcript)
            return "timeout", result
    result, tail = last_result(transcript)
    if (result.get("is_error") or not result) and any(w in tail.lower() for w in LIMIT_WORDS):
        return "usage_limit", result
    if result.get("is_error") or not result:
        return "agent_error", result
    return "completed", result


def fake_turn(n: int, row: dict, run_dir: Path, gold: T.Gold) -> tuple[str, dict]:
    """A pretend agent that does each turn right (pipeline check, no model calls): names the
    real fix's location, applies the real fix, adds the official tests."""
    if n == 1:
        text = f"It is handled in {', '.join(gold.files)}, in {', '.join(gold.functions)}."
        return "completed", {"result": text, "num_turns": 1}
    patch = row["patch"] if n == 2 else row["test_patch"]
    ok = apply_patch(run_dir, patch)
    return ("completed" if ok else "agent_error"), {"result": "done", "num_turns": 1}


# ---------- the gateway log ----------

def log_lines(log_dir: Path) -> list[dict]:
    lines = []
    for f in sorted(log_dir.glob("requests-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                lines.append(json.loads(line))
    return [r for r in lines if r.get("method") == "POST"]


def turn_numbers(records: list[dict]) -> dict:
    """Usage, list-price cost and task numbers of one turn's requests."""
    out = {"requests": len(records), "requests_ode": 0, "unpriced_requests": 0, "input": 0,
           "cache_read": 0, "cache_write": 0, "cache_write_1h": 0, "output": 0, "cost_usd": 0.0,
           "router_task": "", "router_p": "", "decision_point": "", "x": ""}
    for rec in records:
        ode = rec.get("ode") or {}
        out["requests_ode"] += bool(ode.get("arm"))
        usage = rec.get("usage")
        if usage:
            u = usage_from_dict(usage)
            out["input"] += u.input_tokens
            out["cache_read"] += u.cache_read_tokens
            out["cache_write"] += u.cache_write_tokens
            out["cache_write_1h"] += u.cache_write_1h_tokens
            out["output"] += u.output_tokens
            model = ode.get("model_sent") or (rec.get("analysis") or {}).get("model", "")
            try:
                out["cost_usd"] += request_cost(model_for(model), u)
            except ValueError:
                out["unpriced_requests"] += 1
        numbers = ode.get("task_numbers") or {}
        if "x" in numbers and out["x"] == "":
            out.update(router_task=numbers.get("task", ""), router_p=numbers.get("p", ""),
                       decision_point=numbers.get("decision_point", ""),
                       x=json.dumps(numbers["x"]))
    out["cost_usd"] = round(out["cost_usd"], 6)
    return out


# ---------- one run ----------

def version_info(fake: bool) -> tuple[str, str]:
    claude = "fake-agent" if fake else subprocess.run(
        [claude_path(), "--version"], capture_output=True, text=True, check=False).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=False).stdout.strip()
    return claude, commit


def check_turn(n: int, row: dict, result: dict, run_dir: Path, args, gold: T.Gold,
               base: Path) -> dict:
    turn_type = T.TURN_TYPES[n - 1]
    # Graded on everything the agent changed since the start, not only in this turn: an agent
    # may edit a test file in turn 2 and again in turn 3 (seen live), and a turn-3-only diff
    # then does not apply to the original code.
    if turn_type == "explain":
        res = T.explain_locates(result.get("result", "") or "", gold)
    elif turn_type == "bugfix":
        diff = diff_since(run_dir, START)
        (base / "turn2.diff").write_text(diff, encoding="utf-8", newline="\n")
        res = T.grade_bugfix(row, diff, args.work, args.python)
    else:
        diff = diff_since(run_dir, START)
        (base / "turn3.diff").write_text(diff, encoding="utf-8", newline="\n")
        res = T.grade_tests(row, diff, args.work, args.python)
    commit_all(run_dir, f"after turn {n}")
    return res


def run_one(row: dict, arm: str, args) -> list[dict]:
    task = row["instance_id"]
    base = PRIVATE / task / arm
    if base.exists():
        shutil.rmtree(base)  # a previous interrupted attempt
    log_dir = base / "gateway"
    log_dir.mkdir(parents=True)
    ensure_commit(args.repo, row["base_commit"])
    ensure_worktree(args.repo, args.work, row["base_commit"])
    gold = T.gold_locations(row, args.work)
    prepare_run_dir(args.work, args.run_dir, row["base_commit"])
    claude_v, commit = version_info(args.fake_agent)
    settings = json.dumps({**GATEWAY_SETTINGS, "TOKENCUT_ODE_FORCE_ARM": arm,
                           "model_requested": MODEL_REQUESTED, "effort": EFFORT,
                           "permission_mode": "auto", "timeout_s": TIMEOUT_S,
                           "fake_agent": args.fake_agent})
    processes, env = ([], {}) if args.fake_agent else start_services(arm, log_dir)
    rows: list[dict] = []
    try:
        session, deadline = str(uuid.uuid4()), time.time() + TIMEOUT_S
        for n, prompt in enumerate(T.prompts(row, args.python), 1):
            seen = len(log_lines(log_dir))
            started = time.time()
            if args.fake_agent:
                status, result = fake_turn(n, row, args.run_dir, gold)
            else:
                status, result = run_turn(n, prompt, session, args.run_dir, base, env, deadline)
            wall = round(time.time() - started)
            numbers = turn_numbers(log_lines(log_dir)[seen:])
            turn_type = T.TURN_TYPES[n - 1]
            check = (check_turn(n, row, result, args.run_dir, args, gold, base)
                     if status == "completed" else {})
            rows.append({
                "task": task, "arm": arm, "turn": n, "task_type": turn_type, "status": status,
                "solved": int(check["solved"]) if check else "", "signal": T.SIGNALS[turn_type],
                "check": json.dumps({k: v for k, v in check.items() if k != "solved"}),
                **numbers, "claude_reported_cost_usd": result.get("total_cost_usd", ""),
                "wall_s": wall, "num_turns": result.get("num_turns", ""),
                "claude_version": claude_v, "tokencut_commit": commit,
                "started": datetime.fromtimestamp(started, UTC).isoformat(timespec="seconds"),
                "settings": settings,
            })
            if status != "completed":
                break  # later turns depend on this one
    finally:
        stop(processes)
    return rows


# ---------- main ----------

def arm_order(arms: list[str], i: int) -> list[str]:
    """Rotate the arm order per task, so no arm always runs first (warm / cold cache)."""
    k = i % len(arms)
    return arms[k:] + arms[:k]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--arms", default=",".join(ARM_NAMES),
                        help="comma-separated arm names (default: all 8)")
    parser.add_argument("--repo", type=Path,
                        default=Path(os.environ.get("ODE_REPO", ROOT / "swebench_repos" /
                                                    "sympy__sympy")))
    parser.add_argument("--work", type=Path, default=Path(os.environ.get(
        "ODE_WORK", "E:/FYP/swebench-envs/sympy-work")))
    parser.add_argument("--run-dir", type=Path, default=Path(os.environ.get(
        "ODE_RUN_DIR", "E:/FYP/swebench-envs/ode-run/repo")))
    parser.add_argument("--python", default=os.environ.get(
        "ODE_PYTHON", "E:/FYP/swebench-envs/sympy/.venv/Scripts/python.exe"))
    parser.add_argument("--fake-agent", action="store_true",
                        help="pretend agent that solves every turn: checks the pipeline for free")
    parser.add_argument("--results", type=Path, default=None,
                        help="default eval/results/ode (eval/results/ode/fake with --fake-agent)")
    args = parser.parse_args(argv)
    if args.results is None:
        args.results = RESULTS / "fake" if args.fake_agent else RESULTS
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARM_NAMES]
    if unknown:
        parser.error(f"unknown arms {unknown}; arms are {ARM_NAMES}")
    args.results.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    state_path = args.results / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() \
        else {"done": []}
    rows = load_rows(args.tasks)
    for i, task in enumerate(args.tasks):
        if task not in rows:
            print(f"{task}: not in the SWE-bench Lite cache", flush=True)
            continue
        for arm in arm_order(arms, i):
            key = f"{task}/{arm}" + ("/fake" if args.fake_agent else "")
            if key in state["done"]:
                continue
            if (args.results / "STOP").exists():
                print(f"[{now()}] STOP file found; stopping before {key}", flush=True)
                return 0
            print(f"[{now()}] {key}: running", flush=True)
            out = run_one(rows[task], arm, args)
            if any(r["status"] == "usage_limit" for r in out):
                print(f"[{now()}] {key}: usage limit reached - stopping; run the same command "
                      "again after the limit resets", flush=True)
                append_csv(args.results / "turns_interrupted.csv", TURN_FIELDS, out)
                return 3
            append_csv(args.results / "turns.csv", TURN_FIELDS, out)
            state["done"].append(key)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            summary = ", ".join(f"t{r['turn']} {r['task_type']}={r['solved']}" for r in out)
            cost = sum(float(r["cost_usd"] or 0) for r in out)
            print(f"[{now()}] {key}: {summary}; cost ${cost:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
