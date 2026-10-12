"""Replay a captured gateway session offline with alternative trimming rules.

Experiment for the research-methods check (RESULTS_REPORT §4.3). Nothing here changes the
gateway: the rules below are prototypes, measured on requests captured with
TOKENCUT_CAPTURE=1, so every method sees exactly the same session.

On-arrival methods (each tool output is trimmed once, when it first appears):
  current   TokenCut's gateway trimmer (trim_request) at the given threshold
  r8        current, but diffs and output the user asked for "in full" stay whole (R8)
  lamr      r8 + model-free line-level pruning with dependency repair for every output that
            is not a Python file read (LaMR / SWE-Pruner idea, without a model; R12)

History method (simulated with a cache-cost model):
  r10       when the user starts a new task, the previous task's large outputs become a
            short note (R10). Costs one cache rewrite of the changed tail, saves cache reads
            on every later request.

    python -m eval.replay_captures logs/gateway/L1 --out eval/results/replay_L1
"""

import argparse
import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

from tokencut.contracts import Segment, SegmentKind, ToolResultKind
from tokencut.gateway.adapters import anthropic
from tokencut.gateway.session import fingerprint
from tokencut.gateway.trimmer import (
    _REMINDER,
    TrimConfig,
    _call_path,
    _is_edit_call,
    _mentioned,
    _same_file,
    trim_request,
)
from tokencut.tokens import count_tokens
from tokencut.trimming.scoring import tokenize

THRESHOLDS = (2000, 1000, 500)
KEEP_FRACTION = 0.35  # moderate tier
MIN_KEEP_TOKENS = 500
NOTE_TOKENS = 40  # size of a "[TokenCut: ... omitted]" note
R10_MIN_TOKENS = (300, 1000)  # only outputs at least this big are masked at a task boundary
CACHE_READ, CACHE_WRITE = 0.1, 1.25

_DIFF = re.compile(r"^(diff --git |@@ -\d+(,\d+)? \+\d+(,\d+)? @@)", re.MULTILINE)
_FULL_REQUEST = re.compile(
    r"\b(full|entire|whole|complete)\b(?:\W+\w+){0,2}?\W+"
    r"(diff|diffs|file|files|output|content|contents|code|log|logs|listing)\b",
    re.IGNORECASE,
)
_ERROR = re.compile(
    r"error|exception|traceback|failed|failure|fatal|^\s*File \"|^FAILED|^ERROR|^E\s{2,}",
    re.IGNORECASE,
)
# Prefixes some tools put before each code line: "  12→", "12\t", "path:12:", "path-12-"
_PREFIX = re.compile(r"^(\s*\d+(?:→|\t)|[^\s:]+[:-]\d+[:-])")
_HEADER = re.compile(r"^\s*(async\s+def|def|class)\s+\w+|^@@ |^diff --git |^\+\+\+ |^--- ")
_IMPORT = re.compile(r"^\s*(?:from\s+[\w.]+\s+)?import\s+(.+)$")
TAIL_LINES = 10
HEAD_LINES = 5


@dataclass
class Output:
    """One distinct tool output of the session (first appearance)."""

    fp: str
    seg_id: str  # id in the last request
    tool: str
    kind: ToolResultKind | None
    tokens: int
    text: str
    query: str
    call_input: str
    path: str
    protected: bool  # user named the file, or the agent edited it
    user_wants_full: bool
    first_request: int = 0
    present_in: list[int] = field(default_factory=list)


# ---------- loading ----------


def load_session(run_dir: Path) -> tuple[list[dict], list[list[Segment]], list[dict]]:
    captures = sorted((run_dir / "captures").glob("*.json"))
    bodies = [json.loads(p.read_text(encoding="utf-8")) for p in captures]
    segments = [anthropic.parse(b) for b in bodies]
    log = [json.loads(line) for f in sorted(run_dir.glob("requests-*.jsonl"))
           for line in f.read_text(encoding="utf-8").splitlines()]
    log = [r for r in log if r["method"] == "POST" and r.get("analysis")]
    if len(log) != len(bodies):
        raise SystemExit(f"{len(bodies)} captures but {len(log)} logged requests")
    return bodies, segments, log


def collect_outputs(segments: list[list[Segment]]) -> list[Output]:
    """Distinct tool outputs with the context known when each first arrived."""
    last = segments[-1]
    by_fp: dict[str, Output] = {}
    user_texts: list[str] = []
    last_user = ""
    calls: dict[str, tuple[str, str]] = {}
    edited: set[str] = set()
    for seg in last:  # the last request holds the whole history, in order
        if seg.kind == SegmentKind.USER_TEXT:
            text = _REMINDER.sub("", seg.text).strip()
            if text:
                last_user = text
                user_texts.append(text)
        elif seg.kind == SegmentKind.TOOL_CALL:
            calls[seg.tool_call_id] = (seg.tool_name, seg.text)
            if _is_edit_call(seg.tool_name, seg.text) and _call_path(seg.text):
                edited.add(_call_path(seg.text))
        elif seg.kind == SegmentKind.TOOL_RESULT:
            _, call_input = calls.get(seg.tool_call_id, (seg.tool_name, ""))
            path = _call_path(call_input)
            protected = bool(path) and path.endswith(".py") and (
                any(_same_file(path, e) for e in edited) or _mentioned(path, user_texts))
            by_fp[fingerprint(seg)] = Output(
                fp=fingerprint(seg), seg_id=seg.id, tool=seg.tool_name, kind=seg.result_kind,
                tokens=seg.token_count, text=seg.text, query=f"{last_user}\n{call_input}",
                call_input=call_input, path=path, protected=protected,
                user_wants_full=bool(_FULL_REQUEST.search(last_user)))
    for i, segs in enumerate(segments):
        for seg in segs:
            if seg.kind == SegmentKind.TOOL_RESULT:
                out = by_fp.get(fingerprint(seg))
                if out is not None:
                    if not out.present_in:
                        out.first_request = i
                    out.present_in.append(i)
    return sorted(by_fp.values(), key=lambda o: o.first_request)


# ---------- on-arrival methods ----------


def is_diff(text: str) -> bool:
    return len(_DIFF.findall(text)) >= 1


def current_after(body: dict, segs: list[Segment], threshold: int) -> dict[str, int]:
    """Segment id -> tokens after TokenCut's current trimmer."""
    _, report = trim_request(body, segs, TrimConfig(min_tokens=threshold))
    return {d["segment"]: d["after"] for d in report.trimmed}


def _code(line: str) -> str:
    return _PREFIX.sub("", line, count=1)


def _indent(line: str) -> int:
    code = _code(line)
    return len(code) - len(code.lstrip())


def line_prune(text: str, query: str, budget: int) -> str | None:
    """Model-free line-level pruning with dependency repair.

    Priority: summary at the end, error lines (+-1), lines most relevant to the query (BM25
    over identifier sub-words), the start. Repair: every kept line also keeps its enclosing
    def/class/diff-hunk headers and the import lines whose names it uses (LaMR's
    "dependency support", done with indentation and regexes instead of a model).
    """
    lines = text.split("\n")
    if len(lines) < 8:
        return None
    order: list[int] = []
    seen: set[int] = set()

    def push(i: int) -> None:
        if 0 <= i < len(lines) and i not in seen:
            seen.add(i)
            order.append(i)

    for i in range(len(lines) - 1, max(-1, len(lines) - 1 - TAIL_LINES), -1):
        push(i)
    for i, line in enumerate(lines):
        if _ERROR.search(line):
            for k in (i - 1, i, i + 1):
                push(k)
    query_words = tokenize(query)
    docs = [tokenize(_code(line)) or ["_"] for line in lines]
    if query_words:
        scores = BM25Okapi(docs).get_scores(query_words)
        for i in sorted(range(len(lines)), key=lambda k: (-scores[k], k)):
            if scores[i] > 0:
                push(i)
    for i in range(min(HEAD_LINES, len(lines))):
        push(i)

    keep: set[int] = set()
    used = 0
    for i in order:
        cost = count_tokens(lines[i]) + 1
        if used + cost > budget:
            continue
        keep.add(i)
        used += cost

    # dependency repair
    imports = {}
    for i, line in enumerate(lines):
        match = _IMPORT.match(_code(line))
        if match:
            for name in re.findall(r"\w+", match.group(1)):
                imports.setdefault(name, i)
    for i in sorted(keep):
        level = _indent(lines[i])
        for j in range(i - 1, -1, -1):  # enclosing headers
            if _HEADER.match(_code(lines[j])) and (_indent(lines[j]) < level or
                                                     _code(lines[j]).startswith(("@@", "diff", "+++", "---"))):
                keep.add(j)
                level = _indent(lines[j])
                if level == 0:
                    break
        for word in re.findall(r"[A-Za-z_]\w*", _code(lines[i])):
            if word in imports:
                keep.add(imports[word])

    out: list[str] = []
    i = 0
    while i < len(lines):
        if i in keep:
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and j not in keep:
            j += 1
        out.append(f"[TokenCut: {j - i} lines omitted]")
        i = j
    result = "\n".join(out)
    return result if count_tokens(result) < count_tokens(text) else None


def method_after(out: Output, method: str, current: dict[str, int], threshold: int) -> int:
    """Tokens of this output after the method (= tokens if untouched)."""
    if out.tokens < threshold:
        return out.tokens
    if method in ("r8", "lamr") and (is_diff(out.text) or out.user_wants_full):
        return out.tokens
    python_read = out.path.endswith(".py") and out.kind in (
        ToolResultKind.FILE_READ, ToolResultKind.COMMAND_OUTPUT)
    if method == "lamr" and not python_read and not out.protected:
        budget = max(MIN_KEEP_TOKENS, round(KEEP_FRACTION * out.tokens))
        pruned = line_prune(out.text, out.query, budget)
        return count_tokens(pruned) if pruned is not None else out.tokens
    return current.get(out.seg_id, out.tokens)


# ---------- R10: masking finished tasks ----------


def task_of_requests(segments: list[list[Segment]]) -> list[int]:
    """Task number of each request = number of real user messages it contains."""
    tasks = []
    for segs in segments:
        n = sum(1 for s in segs if s.kind == SegmentKind.USER_TEXT and _REMINDER.sub("", s.text).strip())
        tasks.append(n)
    return tasks


def billed_ie(usage: dict | None) -> float:
    if not usage:
        return 0.0
    return (CACHE_READ * usage.get("cache_read_tokens", 0) + CACHE_WRITE * usage.get("cache_write_tokens", 0)
            + usage.get("input_tokens", 0))


def simulate_r10(segments, log, outputs: list[Output], min_tokens: int) -> dict:
    """Input-equivalent cost with and without masking each finished task's large outputs."""
    tasks = task_of_requests(segments)
    billed = sum((r.get("usage") or {}).get(k, 0) for r in log
                 for k in ("cache_read_tokens", "cache_write_tokens", "input_tokens"))
    proxy = sum(r["analysis"]["tokens_total"] for r in log if r.get("usage"))
    ratio = billed / proxy if proxy else 1.0  # provider tokens per proxy token
    task_of_output = {o.fp: tasks[o.first_request] for o in outputs}
    big = {o.fp: o for o in outputs if o.tokens >= min_tokens}

    baseline = sum(billed_ie(r.get("usage")) for r in log)
    extra = saved = 0.0
    boundaries = []
    for i, segs in enumerate(segments):
        if not log[i].get("usage"):
            continue  # rate-limited, not billed
        masked_now = {fp for fp, o in big.items() if task_of_output[fp] < tasks[i]}
        removed = sum(big[fp].tokens - NOTE_TOKENS for fp in masked_now)
        saved += CACHE_READ * ratio * removed
        if i > 0 and tasks[i] > tasks[i - 1]:  # a new task starts: newly masked outputs
            newly = {fp for fp in masked_now if task_of_output[fp] == tasks[i] - 1}
            if newly:
                fps = [fingerprint(s) for s in segs]
                first = min(k for k, fp in enumerate(fps) if fp in newly)
                prev = {fingerprint(s) for s in segments[i - 1]}
                tail = 0
                for k in range(first, len(segs)):
                    if fps[k] in prev:  # new content is written in both variants
                        tail += NOTE_TOKENS if fps[k] in masked_now else segs[k].token_count
                cost = (CACHE_WRITE - CACHE_READ) * ratio * tail
                extra += cost
                later = sum(1 for k in range(i, len(segments)) if log[k].get("usage"))
                gain = CACHE_READ * ratio * sum(big[fp].tokens - NOTE_TOKENS for fp in newly) * later
                boundaries.append({"request": i, "task": tasks[i], "outputs": len(newly),
                                   "removed": sum(big[fp].tokens for fp in newly),
                                   "rewrite_cost": round(cost), "later_requests": later,
                                   "net_gain": round(gain - cost)})
    return {"min_tokens": min_tokens, "baseline_ie": round(baseline),
            "r10_ie": round(baseline - saved + extra), "saved": round(saved),
            "extra_rewrite": round(extra), "change_pct": round(100 * (extra - saved) / baseline, 2),
            "boundaries": boundaries}


# ---------- main ----------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    bodies, segments, log = load_session(args.run_dir)
    outputs = collect_outputs(segments)
    total_proxy = sum(r["analysis"]["tokens_total"] for r in log)
    tool_proxy = sum(r["analysis"]["tokens_by_kind"]["tool_result"] for r in log)
    result = {"run": args.run_dir.name, "requests": len(log), "distinct_outputs": len(outputs),
              "proxy_tokens": total_proxy, "tool_output_tokens": tool_proxy, "on_arrival": [],
              "r10": []}

    for threshold in THRESHOLDS:
        current = current_after(bodies[-1], segments[-1], threshold)
        for method in ("current", "r8", "lamr"):
            t0 = time.perf_counter()
            afters = {o.fp: method_after(o, method, current, threshold) for o in outputs}
            ms = (time.perf_counter() - t0) * 1000
            trimmed = [o for o in outputs if afters[o.fp] < o.tokens]
            weighted = sum((o.tokens - afters[o.fp]) * len(o.present_in) for o in outputs)
            row = {"threshold": threshold, "method": method, "outputs_trimmed": len(trimmed),
                   "diffs_trimmed": sum(1 for o in trimmed if is_diff(o.text)),
                   "first_seen_saved": sum(o.tokens - afters[o.fp] for o in outputs),
                   "request_weighted_saved": weighted,
                   "pct_of_session": round(100 * weighted / total_proxy, 2),
                   "pct_of_tool_output": round(100 * weighted / tool_proxy, 2) if tool_proxy else 0.0,
                   "ms_total": round(ms, 1),
                   "trims": [{"tool": o.tool, "kind": o.kind.value if o.kind else "", "before": o.tokens,
                              "after": afters[o.fp], "diff": is_diff(o.text), "first_request": o.first_request}
                             for o in trimmed]}
            result["on_arrival"].append(row)
    for min_tokens in R10_MIN_TOKENS:
        result["r10"].append(simulate_r10(segments, log, outputs, min_tokens))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "replay.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [(f"# Replay of {result['run']}: {result['requests']} requests, "
              f"{result['distinct_outputs']} distinct tool outputs"), "",
             (f"Session proxy tokens: {total_proxy:,}; of which tool outputs: {tool_proxy:,} "
              f"({100 * tool_proxy / total_proxy:.1f} %)."), "",
             "## On-arrival trimming (request-weighted proxy tokens saved)", "",
             "| threshold | method | outputs trimmed | diffs trimmed | saved | % of session | % of tool output |",
             "|---|---|---|---|---|---|---|"]
    for r in result["on_arrival"]:
        lines.append(f"| {r['threshold']} | {r['method']} | {r['outputs_trimmed']} | {r['diffs_trimmed']} | "
                     f"{r['request_weighted_saved']:,} | {r['pct_of_session']} | {r['pct_of_tool_output']} |")
    lines += ["", "## R10: mask a finished task's large outputs (input-equivalent tokens)", "",
              "| min output size | baseline | with R10 | cache reads saved | extra cache rewrites | change |",
              "|---|---|---|---|---|---|"]
    for r in result["r10"]:
        lines.append(f"| {r['min_tokens']} | {r['baseline_ie']:,} | {r['r10_ie']:,} | {r['saved']:,} | "
                     f"{r['extra_rewrite']:,} | {r['change_pct']} % |")
    (args.out / "replay.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    lamr_ms = [r["ms_total"] for r in result["on_arrival"] if r["method"] == "lamr"]
    print(f"\nlamr total time per session (ms): {lamr_ms}; median {statistics.median(lamr_ms)}")


if __name__ == "__main__":
    main()
