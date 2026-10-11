"""Step B3: gateway trimming latency on realistic Anthropic request bodies.

Times tokencut.gateway.records.analyse() (exactly what the gateway runs per request in trim
mode: JSON parse, segmenting, token breakdown, trimming, re-serialising) for request sizes
10k / 50k / 100k / 200k proxy tokens built from REAL django source files, pytest-style logs
and grep output.

Scenarios per size:
  on_arrival    trim_request only (no session memory)               -- TOKENCUT_CACHE_EVENTS=0
  steady        session memory, same conversation, no cache event    -- normal turn
  model_switch  session memory, model changes to a cold model        -- cache event
Run from repo root: .venv/Scripts/python.exe <this> <out.json>
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from tokencut.contracts import Tier
from tokencut.gateway.history import CacheEventConfig
from tokencut.gateway.records import analyse
from tokencut.gateway.session import SessionStore
from tokencut.gateway.trimmer import TrimConfig
from tokencut.tokens import count_tokens

REPO = Path("swebench_repos/django__django")
SIZES = [10_000, 50_000, 100_000, 200_000]
REPS = {10_000: 15, 50_000: 15, 100_000: 10, 200_000: 8}


def numbered(source: str) -> str:
    return "\n".join(f"{i:>6}→{line}" for i, line in enumerate(source.split("\n"), 1))


def source_files() -> list[tuple[str, str]]:
    files = []
    for path in sorted((REPO / "django").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        n = count_tokens(text)
        if 2_500 <= n <= 9_000:
            files.append((path.relative_to(REPO).as_posix(), text))
    return files


def pytest_log(i: int, n_pass: int = 260) -> str:
    lines = ["============================= test session starts =============================",
             "platform linux -- Python 3.11.4, pytest-7.4.0", f"collected {n_pass + 3} items", ""]
    lines += [f"tests/queries/test_q{i}.py::QueryTests::test_case_{k} PASSED" for k in range(n_pass)]
    lines += ["", "=================================== FAILURES ===================================",
              f"__________________________ QueryTests.test_filter_{i} __________________________",
              "    def test_filter(self):",
              ">       self.assertEqual(qs.count(), 3)",
              "E       AssertionError: 2 != 3",
              f"tests/queries/test_q{i}.py:{40 + i}: AssertionError",
              f"=========================== 1 failed, {n_pass} passed in 4.12s ==========================="]
    return "\n".join(lines)


def grep_output(files: list[tuple[str, str]], word: str, limit: int = 220) -> str:
    out = []
    for path, text in files:
        for n, line in enumerate(text.split("\n"), 1):
            if word in line:
                out.append(f"{path}:{n}:{line.strip()}")
                if len(out) >= limit:
                    return "\n".join(out)
    return "\n".join(out)


TOOLS = [{"name": n, "description": (f"{n}: " + "Performs the {n} operation on the workspace. " * 60),
          "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}}
         for n in ("Read", "Bash", "Edit", "Write", "Grep", "Glob", "WebFetch", "WebSearch",
                   "NotebookEdit", "TodoWrite", "Task", "MultiEdit")]


def build(target: int, files, model="claude-sonnet-5-5") -> dict:
    messages = [{"role": "user", "content":
                 "QuerySet.filter() returns the wrong count when combined with exclude(); "
                 "tests in tests/queries fail. Please find and fix the bug."}]
    body = {"model": model, "max_tokens": 4096, "stream": True,
            "system": "You are Claude Code, a coding agent. " * 150, "tools": TOOLS, "messages": messages}
    i = 0
    while count_tokens(json.dumps(body)) < target:
        kind = ("read", "bash", "read", "grep")[i % 4]
        if kind == "read":
            path, text = files[i % len(files)]
            call = {"name": "Read", "input": {"file_path": f"/repo/{path}"}}
            result = numbered(text)
        elif kind == "bash":
            call = {"name": "Bash", "input": {"command": f"python -m pytest tests/queries -k q{i}"}}
            result = pytest_log(i)
        else:
            word = ("filter", "exclude", "count", "query")[i % 4]
            call = {"name": "Grep", "input": {"pattern": word}}
            result = grep_output(files, word)
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": f"Step {i}: checking."},
            {"type": "tool_use", "id": f"t{i}", **call}]})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": result}]})
        i += 1
    return body


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(q * len(s) + 0.5) - 1))]


def main(out_path: str) -> None:
    files = source_files()
    config = TrimConfig(tier=Tier.MODERATE)  # gateway defaults: min 2,000 tokens
    events = CacheEventConfig()
    # warm-up (first call loads tiktoken / tree-sitter): measured separately
    warm_body = json.dumps(build(10_000, files)).encode()
    t = time.perf_counter()
    analyse("anthropic", "v1/messages", warm_body, config)
    first_call_ms = (time.perf_counter() - t) * 1000
    results = {"first_call_ms": round(first_call_ms, 1), "sizes": []}

    for size in SIZES:
        body = build(size, files)
        raw = json.dumps(body).encode()
        switched = json.dumps({**body, "model": "claude-haiku-5-5"}).encode()
        meta, _ = analyse("anthropic", "v1/messages", raw, config)
        row = {"target": size, "proxy_tokens": meta["tokens_total"],
               "tool_results_trimmed": meta["trim"]["segments_trimmed"],
               "saved_pct_on_arrival": round(100 * meta["trim"]["tokens_saved"] / meta["tokens_total"], 1),
               "messages": len(body["messages"])}
        timings = {"on_arrival": [], "steady": [], "model_switch": []}
        trim_ms = {"on_arrival": [], "steady": [], "model_switch": []}
        event_seen = set()
        for _ in range(REPS[size]):
            t = time.perf_counter()
            m, _ = analyse("anthropic", "v1/messages", raw, config)
            timings["on_arrival"].append((time.perf_counter() - t) * 1000)
            trim_ms["on_arrival"].append(m["trim"]["trim_ms"])

            store = SessionStore()
            now = 1_000_000.0
            analyse("anthropic", "v1/messages", raw, config, (store, events), now=now)  # first turn
            t = time.perf_counter()
            m, _ = analyse("anthropic", "v1/messages", raw, config, (store, events), now=now + 5)
            timings["steady"].append((time.perf_counter() - t) * 1000)
            trim_ms["steady"].append(m["trim"]["trim_ms"])
            event_seen.add(("steady", m["trim"]["cache_event"]))

            t = time.perf_counter()
            m, _ = analyse("anthropic", "v1/messages", switched, config, (store, events), now=now + 10)
            timings["model_switch"].append((time.perf_counter() - t) * 1000)
            trim_ms["model_switch"].append(m["trim"]["trim_ms"])
            event_seen.add(("model_switch", m["trim"]["cache_event"]))
            row["saved_pct_model_switch"] = round(100 * m["trim"]["tokens_saved"] / m["tokens_total"], 1)
        for k, values in timings.items():
            row[k] = {"median_ms": round(statistics.median(values), 1),
                      "p95_ms": round(pct(values, 0.95), 1),
                      "trim_median_ms": round(statistics.median(trim_ms[k]), 1), "n": len(values)}
        row["events"] = sorted(f"{a}:{b}" for a, b in event_seen)
        results["sizes"].append(row)
        print(json.dumps(row), flush=True)
    Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
