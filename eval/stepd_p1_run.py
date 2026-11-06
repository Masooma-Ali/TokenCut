"""Step D again, for the P1 features (RESULTS_REPORT §8), on this Linux machine.

Same 29 sympy bugs and three-message sessions as step D (eval/stepd_run.py, reused here),
Claude Code + Haiku with `--effort medium`, two arms in alternating order:

    A  no trimming (gateway in shadow mode, measuring only), no MCP servers
    P  trim moderate + R8 + R10 + CCR (TOKENCUT_CCR=1) + lazy tools (TOKENCUT_LAZY_TOOLS=1)
       + the repo index MCP server (repo_map / find_code / symbol)

Both arms use --strict-mcp-config, so no other MCP server changes the tool list.
Results: eval/results/stepd_p1/ (numbers only); private logs: logs/stepd_p1/.
A file eval/results/stepd_p1/STOP makes the runner stop before the next run (usage limit
safety); finished runs are skipped on restart.

    python -m eval.stepd_p1_run $(cat eval/results/stepd/task_list.txt)
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from eval import stepd_run as sd
from tokencut.contracts import SegmentKind
from tokencut.gateway.adapters import anthropic

HOME = Path.home()
# STEPD_P1_OUT: results folder name (re-runs go to their own folder); STEPD_P1_ARMS: e.g. "P"
OUT = os.environ.get("STEPD_P1_OUT", "stepd_p1")
ONLY_ARMS = [a for a in os.environ.get("STEPD_P1_ARMS", "A,P").split(",") if a]
sd.RESULTS = sd.ROOT / "eval" / "results" / OUT
sd.PRIVATE = sd.ROOT / "logs" / OUT
sd.GRADE_WORK = HOME / "stepd-p1" / "grade"
sd.RUN_DIR = HOME / "stepd-p1" / "work" / "repo"
sd.PYTHON = str(HOME / "tokencut-sympy" / ".venv" / "bin" / "python")
EFFORT = "medium"
GATEWAY_PYTHON = str(sd.ROOT / ".venv" / "bin" / "python")
MCP_TOOLS = ["mcp__tokencut__repo_map", "mcp__tokencut__find_code", "mcp__tokencut__symbol"]
STOP_FILE = sd.RESULTS / "STOP"

sd.ARMS = {
    "A": {"TOKENCUT_MODE": "shadow", "TOKENCUT_TASK_MASK": "0"},
    "P": {"TOKENCUT_MODE": "trim", "TOKENCUT_TIER": "moderate", "TOKENCUT_TASK_MASK": "1",
          "TOKENCUT_CCR": "1", "TOKENCUT_LAZY_TOOLS": "1"},
}
_EXPECTED_HEALTH = {
    "A": {"mode": "shadow", "task_masking": False, "ccr": False, "lazy_tools": False},
    "P": {"mode": "trim", "task_masking": True, "ccr": True, "lazy_tools": True},
}


def mcp_config(arm: str) -> str:
    if arm != "P":
        return json.dumps({"mcpServers": {}})
    return json.dumps({"mcpServers": {"tokencut": {
        "command": GATEWAY_PYTHON,
        "args": ["-m", "tokencut.repoindex", "serve", "--repo", str(sd.RUN_DIR)],
        "env": {"PYTHONPATH": str(sd.ROOT)},
    }}})


def start_gateway(arm: str, log_dir: Path) -> subprocess.Popen:
    env = {**os.environ, **sd.ARMS[arm], "TOKENCUT_CAPTURE": "1", "TOKENCUT_LOG_DIR": str(log_dir)}
    for key in ("TOKENCUT_PRUNE_TOOLS", "TOKENCUT_MIN_TRIM_TOKENS", "TOKENCUT_CORE_TOOLS"):
        env.pop(key, None)
    out = (log_dir.parent / "gateway.out").open("w", encoding="utf-8")
    proc = subprocess.Popen([GATEWAY_PYTHON, "-m", "tokencut.gateway"], cwd=sd.ROOT, env=env,
                            stdout=out, stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{sd.PORT}/tokencut/health", timeout=2) as r:
                health = json.loads(r.read())
        except OSError:
            time.sleep(1)
            continue
        if any(health.get(k) != v for k, v in _EXPECTED_HEALTH[arm].items()):
            proc.terminate()
            raise RuntimeError(f"gateway settings wrong: {health}")
        return proc
    proc.terminate()
    raise RuntimeError("gateway did not start (is another gateway running on port 8787?)")


_current_arm = {"arm": "A"}


def run_session(prompts: list[str], base: Path) -> tuple[str, dict]:
    """As step D, plus --effort, --strict-mcp-config and (arm P) the repo index server."""
    arm = _current_arm["arm"]
    if arm == "P":  # index the checkout before the session (a running server is incremental)
        subprocess.run([GATEWAY_PYTHON, "-m", "tokencut.repoindex", "build", "--repo",
                        str(sd.RUN_DIR)], cwd=sd.ROOT, capture_output=True, check=False)
    env = {**os.environ, "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{sd.PORT}/anthropic"}
    session = str(sd.uuid.uuid4())
    deadline = time.time() + sd.TIMEOUT_S
    totals = {"num_turns": 0, "total_cost_usd": 0.0, "turns_completed": 0, "subtype": ""}
    extra = ["--effort", EFFORT, "--strict-mcp-config", "--mcp-config", mcp_config(arm)]
    if arm == "P":
        extra += ["--allowedTools", *MCP_TOOLS]
    for n, prompt in enumerate(prompts, 1):
        first = ["--session-id", session] if n == 1 else ["--resume", session]
        cmd = [sd.CLAUDE, "-p", prompt, *first, "--model", sd.MODEL, "--output-format",
               "stream-json", "--verbose", "--permission-mode", "auto", *extra]
        transcript = base / f"transcript_turn{n}.jsonl"
        remaining = deadline - time.time()
        if remaining <= 0:
            return "timeout", totals
        with transcript.open("w", encoding="utf-8") as out:
            try:
                subprocess.run(cmd, cwd=sd.RUN_DIR, env=env, stdout=out, stderr=subprocess.STDOUT,
                               timeout=remaining, check=False)
            except subprocess.TimeoutExpired:
                return "timeout", totals
        result, text = sd._last_result(transcript)
        if (result.get("is_error") or not result) and sd._LIMIT.search(text):
            return "usage_limit", totals
        if result.get("is_error") or not result:
            totals["subtype"] = result.get("subtype", "no result")
            return "agent_error", totals
        totals["num_turns"] += result.get("num_turns") or 0
        totals["total_cost_usd"] += result.get("total_cost_usd") or 0.0
        totals["turns_completed"] = n
        totals["subtype"] = result.get("subtype", "")
    return "completed", totals


def p1_stats(log_dir: Path) -> dict:
    """Calls of the repo index tools and of the gateway's own tools, from the last capture."""
    captures = sorted((log_dir / "captures").glob("*.json"))
    if not captures:
        return {}
    parsed = [anthropic.parse(json.loads(c.read_text(encoding="utf-8"))) for c in captures]
    final = max(parsed, key=len)
    names = [s.tool_name for s in final if s.kind == SegmentKind.TOOL_CALL]
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return {"tool_counts": json.dumps(dict(sorted(counts.items()))),
            "mcp_calls": sum(v for k, v in counts.items() if k.startswith("mcp__tokencut__")),
            "ccr_calls": counts.get("tokencut_retrieve", 0),
            "enable_calls": counts.get("tokencut_enable_tools", 0)}


sd.start_gateway = start_gateway
sd.run_session = run_session
sd.RUN_FIELDS = [*sd.RUN_FIELDS, "effort", "tool_counts", "mcp_calls", "ccr_calls", "enable_calls"]
_run_one = sd.run_one


def run_one(row: dict, arm: str, order: int) -> dict:
    _current_arm["arm"] = arm
    out = _run_one(row, arm, order)
    out["run"].update(effort=EFFORT, **p1_stats(sd.PRIVATE / row["instance_id"] / arm / "gateway"))
    settings = json.loads(out["run"]["settings"])
    settings.update(effort=EFFORT, mcp="tokencut repo index" if arm == "P" else "none")
    out["run"]["settings"] = json.dumps(settings)
    return out


sd.run_one = run_one


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    sd.RESULTS.mkdir(parents=True, exist_ok=True)
    sd.PRIVATE.mkdir(parents=True, exist_ok=True)
    rows = sd.load_rows(argv)
    state = sd.load_state()
    for i, task in enumerate(argv):
        order = ["A", "P"] if i % 2 == 0 else ["P", "A"]  # alternate so cache effects cancel out
        order = [arm for arm in order if arm in ONLY_ARMS]
        for k, arm in enumerate(order):
            key = f"{task}/{arm}"
            if key in state["done"]:
                continue
            if STOP_FILE.exists():
                print(f"[{sd.now()}] STOP file found - stopping before {key}", flush=True)
                return
            print(f"[{sd.now()}] {key}: running", flush=True)
            out = run_one(rows[task], arm, k)
            r = out["run"]
            if r["status"] == "usage_limit":
                print(f"[{sd.now()}] {key}: USAGE LIMIT - stopping; run again after the reset",
                      flush=True)
                sd.append_csv(sd.RESULTS / "runs_interrupted.csv", sd.RUN_FIELDS, [r])
                return
            sd.append_csv(sd.RESULTS / "requests.csv", sd.REQUEST_FIELDS, out["requests"])
            sd.append_csv(sd.RESULTS / "runs.csv", sd.RUN_FIELDS, [r])
            state["done"].append(key)
            sd.save_state(state)
            print(f"[{sd.now()}] {key}: DONE {r['status']}, resolved={r.get('resolved')}, "
                  f"requests={r['requests']}, cost=${r['cost_usd']:.4f}, {r['wall_s']}s, "
                  f"mcp={r.get('mcp_calls', 0)}, ccr={r.get('ccr_calls', 0)}", flush=True)
    print(f"[{sd.now()}] ALL RUNS FINISHED", flush=True)


if __name__ == "__main__":
    main()
