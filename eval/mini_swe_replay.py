"""Replay public Mini-SWE-Agent trajectories through the TokenCut gateway, offline.

Question: how much does TokenCut remove for an agent that reads whole files? The SWE-bench
team publishes Mini-SWE-Agent runs (every model message and every command output); SWE-Pruner
and LaMR used the same agent and the same Sonnet 4.5 baseline (70.6 % on Verified).

Mini-SWE-Agent v1 does not use tool calling: the model writes its command in a ```bash block
and the output comes back as a plain user message (<returncode>…<output>…). TokenCut never
touches user messages, so as shipped it trims nothing (measured: `raw_saved`). This replay
therefore rewrites each step as a tool call (`bash`) and its output as the tool result — what
TokenCut would see if the agent used tool calling (as SWE-Pruner's setup does). The gateway
code is unchanged: every request goes through `records.analyse`, the gateway's own entry point
(on-arrival trimming + cache events, one conversation memory per session).

Request k of a session = system + task + the first k steps (what the agent sent before its
(k+1)-th reply). Sonnet is replayed in Anthropic format, GLM in OpenAI format (its provider).

    python -m eval.mini_swe_replay cache/mini_swe_agent/<run> [...] --out eval/results/mini_swe

Inputs: cache/mini_swe_agent/<run>/*.traj.json (public S3 bucket swe-bench-submissions,
bash-only/<run>/trajs/, gitignored). Outputs are numbers only.
"""

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tokencut.contracts import Tier
from tokencut.gateway.history import CacheEventConfig
from tokencut.gateway.records import analyse
from tokencut.gateway.session import SessionStore
from tokencut.gateway.trimmer import TrimConfig

_CMD = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)
_OUT = re.compile(r"^<returncode>(-?\d+)</returncode>\n<output>\n?(.*)</output>\s*$", re.DOTALL)
_SECONDS_PER_REQUEST = 10  # replay clock: well inside the cache TTL (no artificial expiry)
_BASH = {"name": "bash", "description": "Execute a bash command in the repository.",
         "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                        "required": ["command"]}}


def provider_of(run: str) -> str:
    return "anthropic" if ("sonnet" in run or "claude" in run) else "openai"


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def load_steps(path: Path) -> tuple[str, str, list[tuple[str, str | None, str]]]:
    """(system, task, [(thought, command or None, observation)])."""
    messages = json.loads(path.read_text(encoding="utf-8"))["messages"]
    system, task = _text(messages[0]["content"]), _text(messages[1]["content"])
    steps = []
    for i in range(2, len(messages)):
        if messages[i]["role"] != "assistant":
            continue
        reply = _text(messages[i]["content"])
        match = _CMD.search(reply)
        thought = (reply[:match.start()] + reply[match.end():]).strip() if match else reply.strip()
        obs = ""
        if i + 1 < len(messages) and messages[i + 1]["role"] == "user":
            obs = _text(messages[i + 1]["content"])
            out = _OUT.match(obs)
            if out:  # a tool result is the plain output; a failure keeps its exit code
                obs = out.group(2) if out.group(1) == "0" else f"{out.group(2)}\n[exit code {out.group(1)}]"
        steps.append((thought, match.group(1) if match else None, obs))
    return system, task, steps


def anthropic_body(model: str, system: str, task: str, steps) -> dict:
    tool = {"name": _BASH["name"], "description": _BASH["description"], "input_schema": _BASH["parameters"]}
    messages = [{"role": "user", "content": task}]
    for k, (thought, command, obs) in enumerate(steps):
        if command is None:  # no command in this reply: the agent's feedback is a user message
            messages += [{"role": "assistant", "content": thought or "."}, {"role": "user", "content": obs}]
            continue
        call_id = f"toolu_{k:04d}"
        blocks = ([{"type": "text", "text": thought}] if thought else []) + [
            {"type": "tool_use", "id": call_id, "name": "bash", "input": {"command": command}}]
        messages += [{"role": "assistant", "content": blocks},
                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": obs}]}]
    return {"model": model, "max_tokens": 4096, "system": system, "tools": [tool], "messages": messages}


def openai_body(model: str, system: str, task: str, steps) -> dict:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    for k, (thought, command, obs) in enumerate(steps):
        if command is None:
            messages += [{"role": "assistant", "content": thought or "."}, {"role": "user", "content": obs}]
            continue
        call_id = f"call_{k:04d}"
        messages += [{"role": "assistant", "content": thought or None, "tool_calls": [
                         {"id": call_id, "type": "function",
                          "function": {"name": "bash", "arguments": json.dumps({"command": command})}}]},
                     {"role": "tool", "tool_call_id": call_id, "content": obs}]
    return {"model": model, "messages": messages, "tools": [{"type": "function", "function": _BASH}]}


def raw_body(provider: str, model: str, path: Path) -> dict:
    """The final request exactly as Mini-SWE-Agent v1 sends it (outputs as user messages)."""
    messages = [{"role": m["role"], "content": m["content"]}
                for m in json.loads(path.read_text(encoding="utf-8"))["messages"]]
    if provider == "anthropic":
        return {"model": model, "system": _text(messages[0]["content"]), "messages": messages[1:]}
    return {"model": model, "messages": messages}


def replay_session(args: tuple[str, str, int]) -> tuple[list[dict], dict]:
    run, file, min_tokens = args
    path = Path(file)
    provider = provider_of(run)
    model = run.split("_", 2)[-1]
    system, task, steps = load_steps(path)
    build = anthropic_body if provider == "anthropic" else openai_body
    store = SessionStore()
    trim = TrimConfig(tier=Tier.MODERATE, min_tokens=min_tokens)
    events = CacheEventConfig(task_masking=False)  # one task per session: R10 cannot apply
    instance = path.name.removesuffix(".traj.json")
    rows = []
    hows: Counter = Counter()
    hows_tokens: Counter = Counter()
    first_saved: dict[str, int] = {}
    for k in range(len(steps)):
        body = build(model, system, task, steps[:k])
        analysis, _ = analyse(provider, "/v1/messages", json.dumps(body).encode(),
                              trim, (store, events), now=1_000_000.0 + _SECONDS_PER_REQUEST * k)
        t = analysis["trim"]
        for d in t["details"]:
            if d["segment"] not in first_saved:  # first time this output is trimmed
                first_saved[d["segment"]] = d["before"] - d["after"]
                hows[d["how"]] += 1
                hows_tokens[d["how"]] += d["before"] - d["after"]
        rows.append({"run": run, "instance": instance, "request": k,
                     "tokens_total": analysis["tokens_total"],
                     "tool_result_tokens": analysis["tokens_by_kind"]["tool_result"],
                     "tokens_saved": t["tokens_saved"], "segments_trimmed": t["segments_trimmed"],
                     "cache_event": t["cache_event"] or "", "trim_ms": t["trim_ms"],
                     "analysis_ms": analysis["analysis_ms"]})
    raw, _ = analyse(provider, "/v1/messages", json.dumps(raw_body(provider, model, path)).encode(),
                     trim, None)
    sent = sum(r["tokens_total"] for r in rows)
    saved = sum(r["tokens_saved"] for r in rows)
    session = {"run": run, "instance": instance, "requests": len(rows),
               "commands": sum(1 for s in steps if s[1] is not None),
               "agent_truncated_outputs": sum(1 for s in steps if "<elided_chars>" in s[2]),
               "tokens_sent": sent, "tool_result_tokens": sum(r["tool_result_tokens"] for r in rows),
               "tokens_saved": saved, "pct_saved": round(100 * saved / sent, 2) if sent else 0.0,
               "max_request_tokens": max((r["tokens_total"] for r in rows), default=0),
               "outputs_trimmed": len(first_saved),
               "cache_events": sum(1 for r in rows if r["cache_event"]),
               "raw_saved": raw["trim"]["tokens_saved"],
               "hows": json.dumps({h: [hows[h], hows_tokens[h]] for h in sorted(hows)})}
    return rows, session


def _pct(a: float, b: float) -> float:
    return round(100 * a / b, 1) if b else 0.0


def _p95(values: list[float]) -> float:
    return sorted(values)[max(0, round(0.95 * len(values)) - 1)] if values else 0.0


def summarise(sessions: list[dict], requests: list[dict], min_tokens: int) -> str:
    lines = ["# Mini-SWE-Agent replay through TokenCut (offline)", "",
             (f"Settings: `trim`, `moderate`, threshold {min_tokens:,} proxy tokens, R8 on, cache events on "
              "(defaults), R10 off (one task per session), no tool pruning. Outputs rewritten as "
              "`bash` tool results (see `eval/mini_swe_replay.py`). Proxy tokens (o200k)."), "",
             ("| Run | Sessions | Requests | Tokens sent | Tool output share | Saved (all requests) "
              "| Saved, as shipped (text protocol) | Median session saving | Sessions with any trim "
              "| Cache events |"),
             "|---|---|---|---|---|---|---|---|---|---|"]
    for run in sorted({s["run"] for s in sessions}):
        ss = [s for s in sessions if s["run"] == run]
        sent = sum(s["tokens_sent"] for s in ss)
        tool = sum(s["tool_result_tokens"] for s in ss)
        saved = sum(s["tokens_saved"] for s in ss)
        lines.append(
            f"| {run} | {len(ss)} | {sum(s['requests'] for s in ss):,} | {sent:,} | {_pct(tool, sent)} % "
            f"| **{_pct(saved, sent)} %** ({saved:,}) | {_pct(sum(s['raw_saved'] for s in ss), sent)} % "
            f"| {statistics.median(s['pct_saved'] for s in ss):.1f} % "
            f"| {sum(1 for s in ss if s['outputs_trimmed'])} / {len(ss)} "
            f"| {sum(s['cache_events'] for s in ss)} |")
    lines += ["", "## Where the saving comes from (tokens removed at each output's first trim)", "",
              "| Run | Rule | Outputs | Tokens removed (first appearance) |", "|---|---|---|---|"]
    for run in sorted({s["run"] for s in sessions}):
        tot: Counter = Counter()
        cnt: Counter = Counter()
        for s in (s for s in sessions if s["run"] == run):
            for how, (n, tokens) in json.loads(s["hows"]).items():
                cnt[how] += n
                tot[how] += tokens
        for how in sorted(tot, key=lambda h: -tot[h]):
            lines.append(f"| {run} | {how} | {cnt[how]} | {tot[how]:,} |")
    lines += ["", "## Gateway time per request (ms, this laptop)", "",
              "| Run | Request size (proxy tokens) | Requests | trim median | trim p95 | whole analysis median | p95 |",
              "|---|---|---|---|---|---|---|"]
    bins = [(0, 10_000), (10_000, 25_000), (25_000, 50_000), (50_000, 100_000), (100_000, 10**9)]
    for run in sorted({r["run"] for r in requests}):
        for lo, hi in bins:
            rs = [r for r in requests if r["run"] == run and lo <= r["tokens_total"] < hi]
            if not rs:
                continue
            tm = [r["trim_ms"] for r in rs]
            am = [r["analysis_ms"] for r in rs]
            label = f"{lo // 1000}k-{hi // 1000}k" if hi < 10**9 else f">={lo // 1000}k"
            lines.append(f"| {run} | {label} | {len(rs)} | {statistics.median(tm):.1f} | {_p95(tm):.1f} "
                         f"| {statistics.median(am):.1f} | {_p95(am):.1f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="sessions per run (0 = all)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-tokens", type=int, default=2_000, help="TOKENCUT_MIN_TRIM_TOKENS")
    args = parser.parse_args(argv)

    jobs = []
    for run_dir in args.runs:
        files = sorted(run_dir.glob("*.traj.json"))
        jobs += [(run_dir.name, str(f), args.min_tokens) for f in (files[:args.limit] if args.limit else files)]
    requests: list[dict] = []
    sessions: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows, session in pool.map(replay_session, jobs):
            requests += rows
            sessions.append(session)
            print(f"{session['run'][:30]} {session['instance']}: {session['requests']} requests, "
                  f"saved {session['pct_saved']} %", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("requests.csv", requests), ("sessions.csv", sessions)):
        with (args.out / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = summarise(sessions, requests, args.min_tokens)
    (args.out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
