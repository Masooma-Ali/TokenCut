"""Step D: run Claude Code on SWE-bench tasks without (A) and with (B) TokenCut trimming.

Each task is a short session of three messages in one Claude Code session (investigate /
fix / test and summarise), so R10 task masking has task boundaries to act on.

For every task and arm: fresh copy of the repository at the task's base commit (no git
history, so the real fix cannot be looked up), a gateway with the arm's settings and its own
log folder, `claude -p "<issue>"` (headless, Haiku 5.5, permission mode "auto", wall-clock
limit), then the agent's diff is graded with the task's official FAIL_TO_PASS / PASS_TO_PASS
tests. Everything is logged once (RESULTS_REPORT §7):

  eval/results/stepd/requests.csv   one row per request (numbers only)
  eval/results/stepd/runs.csv       one row per run (numbers only)
  eval/results/stepd/summary.md     A vs B
  logs/stepd/<task>/<arm>/          captures, transcript, diff, test output (private, gitignored)

Finished runs are skipped on restart. A usage-limit stop discards that run and exits; run the
same command again after the limit resets.

    python -m eval.stepd_run sympy__sympy-24909 sympy__sympy-24213 ...
    python -m eval.stepd_run --summary-only
"""

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

from eval.stepd_validate import apply_patch, load_rows, passed, run_tests, test_files
from tokencut.contracts import SegmentKind
from tokencut.gateway.adapters import anthropic

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results" / "stepd"
PRIVATE = ROOT / "logs" / "stepd"
SOURCE_REPO = ROOT / "swebench_repos" / "sympy__sympy"
GRADE_WORK = Path("C:/Users/pc/stepd-validate")  # worktree of SOURCE_REPO, used to check out / grade
RUN_DIR = Path("C:/Users/pc/stepd-work/repo")  # same path for every run: identical system prompts
PYTHON = "C:/Users/pc/tokencut-sympy/.venv/Scripts/python.exe"  # sympy deps + 3.11 shim
CLAUDE = shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude.exe")
MODEL = "haiku"
PORT = 8787
TIMEOUT_S = 45 * 60

ARMS = {
    "A": {"TOKENCUT_MODE": "shadow"},
    "B": {"TOKENCUT_MODE": "trim", "TOKENCUT_TIER": "moderate", "TOKENCUT_TASK_MASK": "1"},
}
# Each bug is a short session of three user messages (RESULTS_REPORT §7, option 2): every
# follow-up message is a new task, which is where R10 acts. Same messages in both arms.
PROMPTS = [
    ("Here is an issue reported in this repository.\n\n{issue}\n\nFirst investigate it: find the "
     "relevant code and reproduce the problem. Do not change any files yet. Use {python} for every "
     "Python or pytest command."),
    "Now make the code change in the repository that resolves the issue.",
    ("Run the tests relevant to your change with the same Python and summarise what you changed "
     "and why."),
]

# Haiku 5.5 list prices, $ per MTok (2026-10-13): prompts up to 100K tokens / above 100K.
PRICE_IN, PRICE_OUT, PRICE_IN_BIG, PRICE_OUT_BIG = 0.10, 0.50, 0.50, 2.50
CACHE_READ_X, CACHE_WRITE_1H_X = 0.1, 2.0  # Claude Code writes 1-hour cache entries
_LIMIT = re.compile(r"usage limit|limit reached|rate.?limit|reset", re.IGNORECASE)

REQUEST_FIELDS = [
    "task", "arm", "request", "ts", "model", "status", "latency_ms", "analysis_ms", "trim_ms",
    "tokens_before", "tokens_after", "tokens_saved", "system", "tools", "user_text",
    "assistant_text", "thinking", "tool_call", "tool_result", "media", "trims", "trim_hows",
    "cache_event", "cache_read", "cache_write", "input", "output", "billed_prompt", "cost_usd",
]
RUN_FIELDS = [
    "task", "arm", "order", "started", "ended", "wall_s", "status", "result_subtype", "num_turns",
    "turns_completed",
    "requests", "requests_failed", "tokens_before", "tokens_after", "tokens_saved",
    "cache_read", "cache_write", "input", "output", "cost_usd", "claude_reported_cost_usd",
    "requests_over_100k", "warmup_cache_read", "warmup_cache_write", "first_cache_read",
    "first_cache_write", "latency_ms_median", "trim_ms_median", "trim_ms_max",
    "on_arrival_trims", "task_masks", "cache_events", "tool_calls", "repeated_calls",
    "refetch_after_trim", "files_changed", "diff_lines", "f2p_pass", "f2p_total", "p2p_pass",
    "p2p_total", "resolved", "claude_version", "tokencut_commit", "settings",
]


# ---------- helpers ----------


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def cost(usage: dict) -> tuple[float, int]:
    cr, cw, i, o = (usage.get(k, 0) for k in ("cache_read_tokens", "cache_write_tokens",
                                              "input_tokens", "output_tokens"))
    prompt = cr + cw + i
    pin, pout = (PRICE_IN, PRICE_OUT) if prompt <= 100_000 else (PRICE_IN_BIG, PRICE_OUT_BIG)
    return (pin * (i + CACHE_READ_X * cr + CACHE_WRITE_1H_X * cw) + pout * o) / 1e6, prompt


def append_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def load_state() -> dict:
    path = RESULTS / "state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"done": []}


def save_state(state: dict) -> None:
    (RESULTS / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------- preparing a run ----------


def prepare_repo(row: dict) -> None:
    """RUN_DIR = files of the task's base commit, as a new git repo with one commit."""
    subprocess.run(["git", "checkout", "-q", "-f", row["base_commit"]], cwd=GRADE_WORK, check=True)
    subprocess.run(["git", "clean", "-qfd"], cwd=GRADE_WORK, check=True)
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR, onerror=lambda f, p, _: (os.chmod(p, 0o700), f(p)))
    RUN_DIR.mkdir(parents=True)
    files = subprocess.run(["git", "ls-files", "-z"], cwd=GRADE_WORK, capture_output=True,
                           check=True).stdout.decode("utf-8").split("\0")
    for name in filter(None, files):
        src = GRADE_WORK / name
        if src.is_file():
            dst = RUN_DIR / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for args in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.name=stepd", "-c", "user.email=stepd@local", "commit", "-qm", "start"]):
        subprocess.run(["git", *args], cwd=RUN_DIR, check=True, capture_output=True)


def start_gateway(arm: str, log_dir: Path) -> subprocess.Popen:
    env = {**os.environ, **ARMS[arm], "TOKENCUT_CAPTURE": "1", "TOKENCUT_LOG_DIR": str(log_dir)}
    for key in ("TOKENCUT_PRUNE_TOOLS", "TOKENCUT_MIN_TRIM_TOKENS"):
        env.pop(key, None)
    if arm == "A":
        env.pop("TOKENCUT_TASK_MASK", None)
    out = (log_dir.parent / "gateway.out").open("w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "tokencut.gateway"], cwd=ROOT, env=env,
                            stdout=out, stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/tokencut/health", timeout=2) as r:
                health = json.loads(r.read())
            want = ARMS[arm]["TOKENCUT_MODE"]
            if health["mode"] != want or health.get("task_masking") != (arm == "B"):
                proc.terminate()
                raise RuntimeError(f"gateway settings wrong: {health}")
            return proc
        except OSError:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("gateway did not start (is another gateway running on port 8787?)")



def _last_result(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    result: dict = {}
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result = event
    return result, (json.dumps(result) if result else raw[-3000:])


def run_session(prompts: list[str], base: Path) -> tuple[str, dict]:
    """Send the prompts one after another in ONE Claude Code session (--session-id, then
    --resume), like a user typing follow-up messages. Each message is a new task for R10."""
    env = {**os.environ, "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{PORT}/anthropic"}
    session = str(uuid.uuid4())
    deadline = time.time() + TIMEOUT_S
    totals = {"num_turns": 0, "total_cost_usd": 0.0, "turns_completed": 0, "subtype": ""}
    for n, prompt in enumerate(prompts, 1):
        first = ["--session-id", session] if n == 1 else ["--resume", session]
        cmd = [CLAUDE, "-p", prompt, *first, "--model", MODEL, "--output-format", "stream-json",
               "--verbose", "--permission-mode", "auto"]
        transcript = base / f"transcript_turn{n}.jsonl"
        remaining = deadline - time.time()
        if remaining <= 0:
            return "timeout", totals
        with transcript.open("w", encoding="utf-8") as out:
            try:
                subprocess.run(cmd, cwd=RUN_DIR, env=env, stdout=out, stderr=subprocess.STDOUT,
                               timeout=remaining, check=False)
            except subprocess.TimeoutExpired:
                return "timeout", totals
        result, text = _last_result(transcript)
        if (result.get("is_error") or not result) and _LIMIT.search(text):
            return "usage_limit", totals
        if result.get("is_error") or not result:
            totals["subtype"] = result.get("subtype", "no result")
            return "agent_error", totals
        totals["num_turns"] += result.get("num_turns") or 0
        totals["total_cost_usd"] += result.get("total_cost_usd") or 0.0
        totals["turns_completed"] = n
        totals["subtype"] = result.get("subtype", "")
    return "completed", totals


# ---------- grading and measuring ----------


def agent_diff(diff_path: Path) -> str:
    subprocess.run(["git", "add", "-A"], cwd=RUN_DIR, check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--binary"], cwd=RUN_DIR, capture_output=True,
                          check=True).stdout.decode("utf-8", errors="replace")
    diff_path.write_text(diff, encoding="utf-8", newline="\n")
    return diff


def grade(row: dict, diff: str, log_path: Path) -> dict:
    f2p, p2p = json.loads(row["FAIL_TO_PASS"]), json.loads(row["PASS_TO_PASS"])
    res = {"f2p_pass": 0, "f2p_total": len(f2p), "p2p_pass": 0, "p2p_total": len(p2p), "resolved": False}
    subprocess.run(["git", "checkout", "-q", "-f", row["base_commit"]], cwd=GRADE_WORK, check=True)
    subprocess.run(["git", "clean", "-qfd"], cwd=GRADE_WORK, check=True)
    if diff.strip() and not apply_patch(GRADE_WORK, diff):
        log_path.write_text("agent diff does not apply", encoding="utf-8")
        return res
    files = test_files(row["test_patch"])
    # SWE-bench rule: the task's test files are reset before the official tests are added.
    subprocess.run(["git", "checkout", "-q", row["base_commit"], "--", *files], cwd=GRADE_WORK, check=False)
    if not apply_patch(GRADE_WORK, row["test_patch"]):
        log_path.write_text("test patch does not apply after the agent diff", encoding="utf-8")
        return res
    outcomes, out = run_tests(GRADE_WORK, PYTHON, files, timeout=900)
    log_path.write_text(out, encoding="utf-8")
    res.update(f2p_pass=len(passed(outcomes, f2p)), p2p_pass=len(passed(outcomes, p2p)))
    res["resolved"] = res["f2p_pass"] == len(f2p) and res["p2p_pass"] == len(p2p)
    subprocess.run(["git", "checkout", "-q", "-f", row["base_commit"]], cwd=GRADE_WORK, check=True)
    subprocess.run(["git", "clean", "-qfd"], cwd=GRADE_WORK, check=True)
    return res


def request_rows(task: str, arm: str, log_dir: Path) -> list[dict]:
    rows = []
    lines = [json.loads(line) for f in sorted(log_dir.glob("requests-*.jsonl"))
             for line in f.read_text(encoding="utf-8").splitlines()]
    for n, rec in enumerate(r for r in lines if r.get("method") == "POST"):
        a = rec.get("analysis") or {}
        t = a.get("trim") or {}
        kinds = a.get("tokens_by_kind") or {}
        usage = rec.get("usage") or {}
        usd, prompt = cost(usage) if usage else (0.0, 0)
        before = a.get("tokens_total", 0)
        applied = arm == "B" and rec.get("trim_applied", False)
        saved = t.get("tokens_saved", 0) if applied else 0
        rows.append({
            "task": task, "arm": arm, "request": n, "ts": rec.get("ts"), "model": a.get("model", ""),
            "status": rec.get("status"), "latency_ms": rec.get("latency_ms"),
            "analysis_ms": a.get("analysis_ms"), "trim_ms": t.get("trim_ms"),
            "tokens_before": before, "tokens_after": before - saved, "tokens_saved": saved,
            **{k: kinds.get(k, 0) for k in ("system", "tools", "user_text", "assistant_text",
                                             "thinking", "tool_call", "tool_result", "media")},
            "trims": len(t.get("details", [])) if applied else 0,
            "trim_hows": json.dumps(sorted({d["how"] for d in t.get("details", [])})) if applied else "[]",
            "cache_event": t.get("cache_event") or "",
            "cache_read": usage.get("cache_read_tokens", 0), "cache_write": usage.get("cache_write_tokens", 0),
            "input": usage.get("input_tokens", 0), "output": usage.get("output_tokens", 0),
            "billed_prompt": prompt, "cost_usd": round(usd, 6),
        })
    return rows


def call_stats(log_dir: Path) -> dict:
    """Tool calls in the final request, identical repeats, and re-runs of trimmed outputs."""
    captures = sorted((log_dir / "captures").glob("*.json"))
    if not captures:
        return {"tool_calls": 0, "repeated_calls": 0, "refetch_after_trim": 0}
    lines = [json.loads(line) for f in sorted(log_dir.glob("requests-*.jsonl"))
             for line in f.read_text(encoding="utf-8").splitlines()]
    # One capture per analysed POST, in the same order (side requests included).
    posts = [r for r in lines if r.get("method") == "POST" and r.get("analysis")]
    trimmed: set[tuple[str, str]] = set()
    parsed = [anthropic.parse(json.loads(cap.read_text(encoding="utf-8"))) for cap in captures]
    for segs, rec in zip(parsed, posts):
        if not rec.get("trim_applied"):
            continue
        calls = {s.tool_call_id: (s.tool_name, s.text) for s in segs if s.kind == SegmentKind.TOOL_CALL}
        ids = {d["segment"] for d in ((rec.get("analysis") or {}).get("trim") or {}).get("details", [])}
        for s in segs:
            if s.id in ids and s.tool_call_id in calls:
                trimmed.add(calls[s.tool_call_id])
    # The main conversation is the request with the most segments (Claude Code also sends small
    # side requests, e.g. a summary at the end).
    final = max(parsed, key=len)
    seq = [(s.tool_name, s.text) for s in final if s.kind == SegmentKind.TOOL_CALL]
    seen: set[tuple[str, str]] = set()
    repeated = refetch = 0
    for call in seq:
        if call in seen:
            repeated += 1
            if call in trimmed:
                refetch += 1
        seen.add(call)
    return {"tool_calls": len(seq), "repeated_calls": repeated, "refetch_after_trim": refetch}


def version_info() -> tuple[str, str]:
    claude = subprocess.run([CLAUDE, "--version"], capture_output=True, text=True, check=False).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=False).stdout.strip()
    return claude, commit


# ---------- one run ----------


def run_one(row: dict, arm: str, order: int) -> dict:
    task = row["instance_id"]
    base = PRIVATE / task / arm
    if base.exists():
        shutil.rmtree(base)  # a previous interrupted attempt
    log_dir = base / "gateway"
    log_dir.mkdir(parents=True)
    prepare_repo(row)
    # A warm-up request was tried (attempt 2) and dropped: Claude Code's first request is not
    # byte-identical between runs (the tool list varies as connectors load), so it cannot warm
    # the cache reliably. The order effect is removed in the analysis instead (warm-start cost).
    warm = {"warmup_cache_read": "", "warmup_cache_write": ""}
    started = time.time()
    gateway = start_gateway(arm, log_dir)
    try:
        prompts = [PROMPTS[0].format(issue=row["problem_statement"], python=PYTHON), *PROMPTS[1:]]
        status, result = run_session(prompts, base)
    finally:
        gateway.terminate()
        gateway.wait(timeout=30)
    ended = time.time()
    reqs = request_rows(task, arm, log_dir)
    failed = [r for r in reqs if r["status"] not in (200, None)]
    if status != "usage_limit" and any(r["status"] == 429 for r in reqs) and not any(
            r["status"] == 200 for r in reqs[-3:]):
        status = "usage_limit"
    diff = agent_diff(base / "agent.diff")
    graded = grade(row, diff, base / "tests.log") if status in ("completed", "timeout", "agent_error") else {}
    claude_v, commit = version_info()
    billed = [r for r in reqs if r["status"] == 200]
    run = {
        "task": task, "arm": arm, "order": order,
        "started": datetime.fromtimestamp(started, UTC).isoformat(timespec="seconds"),
        "ended": datetime.fromtimestamp(ended, UTC).isoformat(timespec="seconds"),
        "wall_s": round(ended - started), "status": status,
        "result_subtype": result.get("subtype", ""), "num_turns": result.get("num_turns", ""),
        "turns_completed": result.get("turns_completed", ""),
        "requests": len(reqs), "requests_failed": len(failed),
        **{k: sum(r[k] for r in reqs) for k in ("tokens_before", "tokens_after", "tokens_saved",
                                                 "cache_read", "cache_write", "input", "output")},
        "cost_usd": round(sum(r["cost_usd"] for r in reqs), 6),
        "claude_reported_cost_usd": result.get("total_cost_usd", ""),
        "requests_over_100k": sum(1 for r in reqs if r["billed_prompt"] > 100_000),
        **warm,
        "first_cache_read": billed[0]["cache_read"] if billed else "",
        "first_cache_write": billed[0]["cache_write"] if billed else "",
        "latency_ms_median": statistics.median([r["latency_ms"] for r in billed]) if billed else "",
        "trim_ms_median": statistics.median([r["trim_ms"] or 0 for r in reqs]) if reqs else "",
        "trim_ms_max": max([r["trim_ms"] or 0 for r in reqs], default=""),
        "on_arrival_trims": sum(1 for r in reqs for h in json.loads(r["trim_hows"])
                                if h in ("code", "command_output", "search", "file_head_tail")),
        "task_masks": sum(1 for r in reqs if "task_masked" in r["trim_hows"]),
        "cache_events": sum(1 for r in reqs if r["cache_event"]),
        **call_stats(log_dir),
        "files_changed": sum(1 for line in diff.splitlines() if line.startswith("diff --git")),
        "diff_lines": sum(1 for line in diff.splitlines()
                          if line[:1] in "+-" and not line.startswith(("+++", "---"))),
        **graded, "claude_version": claude_v, "tokencut_commit": commit,
        "settings": json.dumps({**ARMS[arm], "TOKENCUT_CAPTURE": "1", "model": MODEL, "messages": len(PROMPTS),
                                "permission_mode": "auto", "timeout_s": TIMEOUT_S}),
    }
    return {"run": run, "requests": reqs}


# ---------- summary ----------


def warm_start_costs() -> dict[tuple[str, str], float]:
    """Run cost with its first billed request's cache writes billed as cache reads.

    The first request of a run carries Claude Code's system prompt and tool definitions. If an
    earlier run (of the other arm) cached them, it reads them cheaply; otherwise it writes them
    at 2x. That order effect is not caused by TokenCut, so the comparison uses the cost as if
    every run started with a warm cache. Later requests are unchanged.
    """
    path = RESULTS / "requests.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[tuple[str, str], float] = {}
    first_done: set[tuple[str, str]] = set()
    for r in rows:
        key = (r["task"], r["arm"])
        total = out.get(key, 0.0) + float(r["cost_usd"] or 0)
        if key not in first_done and r["status"] == "200":
            first_done.add(key)
            usage = {"cache_read_tokens": int(r["cache_read"]) + int(r["cache_write"]),
                     "cache_write_tokens": 0, "input_tokens": int(r["input"]),
                     "output_tokens": int(r["output"])}
            total += cost(usage)[0] - float(r["cost_usd"] or 0)
        out[key] = total
    return out


def summarise() -> None:
    path = RESULTS / "runs.csv"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        runs = [r for r in csv.DictReader(f) if r["status"] != "usage_limit"]
    by = {(r["task"], r["arm"]): r for r in runs}
    for (task, arm), warm in warm_start_costs().items():
        if (task, arm) in by:
            by[(task, arm)]["cost_warm_usd"] = warm
    tasks = sorted({r["task"] for r in runs})
    pairs = [t for t in tasks if (t, "A") in by and (t, "B") in by]

    def num(r: dict, k: str) -> float:
        try:
            return float(r[k])
        except (TypeError, ValueError):
            return 0.0

    lines = [f"# Step D summary ({now()})", "",
             (f"Runs: {len(runs)}; complete A/B pairs: {len(pairs)}. Arm A = shadow (no trimming), "
              "arm B = trim moderate + R8 + R10. Cost = Haiku 5.5 list prices incl. the 100K step "
              "and 1-hour cache writes."), "",
             ("| task | resolved A | resolved B | cost A ($) | cost B ($) | requests A | requests B | "
              "tokens sent A | tokens sent B | wall A (s) | wall B (s) | re-fetch after trim (B) |"),
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in pairs:
        a, b = by[(t, "A")], by[(t, "B")]
        lines.append(f"| {t} | {a['resolved']} | {b['resolved']} | {num(a, 'cost_usd'):.4f} | "
                     f"{num(b, 'cost_usd'):.4f} | {a['requests']} | {b['requests']} | "
                     f"{int(num(a, 'tokens_after')):,} | {int(num(b, 'tokens_after')):,} | {a['wall_s']} | "
                     f"{b['wall_s']} | {b['refetch_after_trim']} |")
    if pairs:
        def total(arm: str, k: str) -> float:
            return sum(num(by[(t, arm)], k) for t in pairs)
        res_a = sum(by[(t, "A")]["resolved"] == "True" for t in pairs)
        res_b = sum(by[(t, "B")]["resolved"] == "True" for t in pairs)
        lines += ["", "| measure | A | B | change |", "|---|---|---|---|",
                  f"| resolved | {res_a}/{len(pairs)} | {res_b}/{len(pairs)} | {res_b - res_a:+d} |"]
        for k, label in (("cost_usd", "cost ($, raw)"), ("cost_warm_usd", "cost ($, warm start)"),
                         ("tokens_after", "tokens sent"), ("requests", "requests"),
                         ("wall_s", "wall time (s)")):
            ta, tb = total("A", k), total("B", k)
            change = f"{100 * (tb - ta) / ta:+.1f} %" if ta else "–"
            lines.append(f"| {label} | {ta:,.4g} | {tb:,.4g} | {change} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def refresh_call_stats() -> None:
    """Rewrite the call-stat columns of runs.csv from each run's private logs."""
    path = RESULTS / "runs.csv"
    with path.open(encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    for run in runs:
        log_dir = PRIVATE / run["task"] / run["arm"] / "gateway"
        if log_dir.exists():
            run.update(call_stats(log_dir))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)


# ---------- main ----------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="*")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--refresh-call-stats", action="store_true",
                        help="recompute tool_calls / repeated_calls / refetch_after_trim from the logs")
    args = parser.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    if args.refresh_call_stats:
        refresh_call_stats()
    if args.summary_only or args.refresh_call_stats:
        summarise()
        return
    rows = load_rows(args.tasks)
    state = load_state()
    for i, task in enumerate(args.tasks):
        order = ["A", "B"] if i % 2 == 0 else ["B", "A"]  # alternate so cache effects cancel out
        for k, arm in enumerate(order):
            key = f"{task}/{arm}"
            if key in state["done"]:
                continue
            print(f"[{now()}] {key}: running", flush=True)
            out = run_one(rows[task], arm, k)
            if out["run"]["status"] == "usage_limit":
                print(f"[{now()}] {key}: usage limit reached - stopping; run the same command again "
                      "after the limit resets", flush=True)
                append_csv(RESULTS / "runs_interrupted.csv", RUN_FIELDS, [out["run"]])
                return
            append_csv(RESULTS / "requests.csv", REQUEST_FIELDS, out["requests"])
            append_csv(RESULTS / "runs.csv", RUN_FIELDS, [out["run"]])
            state["done"].append(key)
            save_state(state)
            r = out["run"]
            print(f"[{now()}] {key}: {r['status']}, resolved={r.get('resolved')}, "
                  f"requests={r['requests']}, cost=${r['cost_usd']:.4f}, {r['wall_s']}s", flush=True)
    summarise()


if __name__ == "__main__":
    main()
