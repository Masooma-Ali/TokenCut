"""Recompute every headline number in docs/FULL_RESULTS.md from the raw result files.

Reads only numbers-only files: eval/results/*.csv, eval/results/stepd/*.csv and the
numbers-only gateway logs in logs/ (local). Prints each figure with its inputs so the
report can show how it was obtained.

    python -m eval.verify_results
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "eval" / "results"
LOGS = ROOT / "logs"


def num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def benchmark(path: Path) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    print(f"\n## Benchmark {path.name}: {len({r['instance_id'] for r in rows})} tasks, {len(rows)} rows")
    for setting, field in (("trim", "chunk_recall_available"), ("trim", "chunk_recall"),
                           ("retrieve", "chunk_recall")):
        print(f"  {setting} / {field}")
        for frac in ("0.15", "0.35", "0.6"):
            out = []
            for method in ("tokencut", "bm25", "embed", "head", "random", "bm25_files"):
                sel = [r for r in rows if r["setting"] == setting and r["method"] == method
                       and r["fraction"] == frac]
                if setting == "trim" and field == "chunk_recall_available":
                    sel = [r for r in sel if num(r["gold_chunks_available"]) > 0]
                if field == "chunk_recall":
                    sel = [r for r in sel if num(r["gold_chunks"]) > 0]
                if sel:
                    vals = [num(r[field]) for r in sel]
                    out.append(f"{method}={statistics.fmean(vals):.3f} (sum {sum(vals):.2f}/{len(vals)})")
            print(f"    {frac}: " + ", ".join(out))
    # skeletons
    for setting in ("trim", "retrieve"):
        for frac in ("0.15", "0.35", "0.6"):
            sel = [r for r in rows if r["setting"] == setting and r["method"] == "tokencut"
                   and r["fraction"] == frac and num(r["gold_chunks"]) > 0]
            helped = sum(1 for r in sel if num(r["chunk_recall_with_skeletons"]) > num(r["chunk_recall"]))
            print(f"  skeletons {setting} {frac}: full {statistics.fmean(num(r['chunk_recall']) for r in sel):.3f}"
                  f" with skel {statistics.fmean(num(r['chunk_recall_with_skeletons']) for r in sel):.3f}"
                  f" helped {helped}/{len(sel)}")


def billed(usage_rows: list[dict]) -> tuple[int, int, int, int]:
    cr = sum(u.get("cache_read_tokens") or 0 for u in usage_rows)
    cw = sum(u.get("cache_write_tokens") or 0 for u in usage_rows)
    inp = sum(u.get("input_tokens") or 0 for u in usage_rows)
    out = sum(u.get("output_tokens") or 0 for u in usage_rows)
    return cr, cw, inp, out


def live(name: str) -> None:
    folder = LOGS / "gateway" / name
    lines = [json.loads(x) for f in sorted(folder.glob("requests-*.jsonl"))
             for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    posts = [x for x in lines if x.get("method") == "POST" and x.get("analysis")]
    ok = [x for x in posts if x["status"] == 200 and x.get("usage")]
    big = [x for x in ok if x["analysis"]["tokens_total"] > 5000]  # main requests, not side calls
    cr, cw, inp, out = billed([x["usage"] for x in ok])
    ieq = 0.1 * cr + 1.25 * cw + inp
    kinds = defaultdict(int)
    for x in ok:
        for k, v in x["analysis"]["tokens_by_kind"].items():
            kinds[k] += v
    tot = sum(kinds.values()) or 1
    before = sum(x["analysis"]["trim"]["tokens_before"] for x in ok)
    saved = sum(x["analysis"]["trim"]["tokens_saved"] for x in ok)
    sent = sum(x["analysis"]["tokens_total"] for x in ok)
    events = [x["analysis"]["trim"]["cache_event"] for x in ok if x["analysis"]["trim"]["cache_event"]]
    hows = defaultdict(int)
    for x in ok:
        for d in x["analysis"]["trim"]["details"]:
            hows[d["how"]] += 1
    trim_ms = [x["analysis"]["trim"]["trim_ms"] for x in ok]
    print(f"\n## Live {name}: POST {len(posts)} (200 with usage: {len(ok)}, non-200: "
          f"{sum(1 for x in posts if x['status'] != 200)}), main requests (>5k proxy tok) {len(big)}")
    print(f"  billed: cache_read {cr:,} cache_write {cw:,} uncached {inp:,} output {out:,}")
    print(f"  cache-read share of billed input: {cr / (cr + cw + inp):.4f}")
    print(f"  input-equivalent = 0.1*{cr:,} + 1.25*{cw:,} + {inp:,} = {ieq:,.0f}; per request {ieq / len(ok):,.0f}")
    print("  proxy share by kind: " + ", ".join(f"{k} {100 * v / tot:.1f}%" for k, v in
                                               sorted(kinds.items(), key=lambda kv: -kv[1]) if v))
    # tokens_total is measured before trimming, so it is what the agent sent
    print(f"  proxy tokens the agent sent {sent:,}; removed {saved:,} = {100 * saved / sent:.1f}%"
          f" (from {before:,} tokens of trimmed items)")
    print(f"  cache events: {events}; detail rows by how: {dict(hows)}")
    print(f"  trim_ms median {statistics.median(trim_ms):.1f} max {max(trim_ms):.1f}; "
          f"max request {max(x['analysis']['tokens_total'] for x in ok):,}")


HAIKU = {"in": 0.10, "out": 0.50, "in_hi": 0.50, "out_hi": 2.50}


def haiku_cost(u: dict) -> tuple[float, bool]:
    """Dollar cost of one Haiku 5.5 request: 100K step, cache read 0.1x, 1-hour cache write 2x."""
    cr, cw = u.get("cache_read_tokens") or 0, u.get("cache_write_tokens") or 0
    inp, out = u.get("input_tokens") or 0, u.get("output_tokens") or 0
    prompt = cr + cw + inp
    hi = prompt > 100_000
    p_in, p_out = (HAIKU["in_hi"], HAIKU["out_hi"]) if hi else (HAIKU["in"], HAIKU["out"])
    return (inp + 0.1 * cr + 2 * cw) * p_in / 1e6 + out * p_out / 1e6, hi


def dollars(name: str) -> None:
    folder = LOGS / "gateway" / name
    lines = [json.loads(x) for f in sorted(folder.glob("requests-*.jsonl"))
             for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    ok = [x for x in lines if x.get("method") == "POST" and x.get("status") == 200 and x.get("usage")]
    costs = [haiku_cost(x["usage"]) for x in ok]
    prompts = [sum((x["usage"].get(k) or 0) for k in ("cache_read_tokens", "cache_write_tokens",
                                                       "input_tokens")) for x in ok]
    print(f"  {name} dollars: ${sum(c for c, _ in costs):.3f}; requests over 100K {sum(h for _, h in costs)}"
          f" of {len(ok)}; prompt median {statistics.median(prompts):,.0f} max {max(prompts):,}")


def stepd() -> None:
    d = RES / "stepd"
    runs = list(csv.DictReader((d / "runs.csv").open(encoding="utf-8")))
    reqs = list(csv.DictReader((d / "requests.csv").open(encoding="utf-8")))
    by = {(r["task"], r["arm"]): r for r in runs}
    tasks = sorted({t for t, _ in by})
    print(f"\n## Step D: {len(tasks)} tasks, {len(runs)} runs, {len(reqs)} request rows")
    for arm in "AB":
        res = [t for t in tasks if by[(t, arm)]["resolved"] == "True"]
        print(f"  {arm}: resolved {len(res)}/{len(tasks)}")
        ok = [r for r in reqs if r["arm"] == arm and r["status"] == "200"]
        cost = sum(num(r["cost_usd"]) for r in ok)
        sent = sum(num(r["tokens_after"]) for r in ok)
        over = sum(1 for r in ok if num(r["billed_prompt"]) > 100_000)
        print(f"     requests(200) {len(ok)}; total cost ${cost:.4f}; tokens sent {sent:,.0f}; "
              f"over-100K {over}; pooled cost/request ${cost / len(ok):.6f}")
        # per-task per-request mean (as analysis.md)
        per_task = []
        for t in tasks:
            rows = [r for r in ok if r["task"] == t]
            per_task.append(statistics.fmean(num(r["cost_usd"]) for r in rows))
        print(f"     mean over tasks of cost/request ${statistics.fmean(per_task):.6f}")
        print(f"     cost per task raw mean ${statistics.fmean(num(by[(t, arm)]['cost_usd']) for t in tasks):.5f}")
    only_a = [t for t in tasks if by[(t, "A")]["resolved"] == "True" and by[(t, "B")]["resolved"] != "True"]
    only_b = [t for t in tasks if by[(t, "B")]["resolved"] == "True" and by[(t, "A")]["resolved"] != "True"]
    print(f"  only A: {only_a}; only B: {only_b}")
    b_before = sum(num(by[(t, "B")]["tokens_before"]) for t in tasks)
    b_saved = sum(num(by[(t, "B")]["tokens_saved"]) for t in tasks)
    print(f"  B tokens saved {b_saved:,.0f} of {b_before:,.0f} = {100 * b_saved / b_before:.2f}%")
    # technique split from the numbers-only gateway logs of the B runs
    how_saved = defaultdict(int)
    for t in tasks:
        for f in (LOGS / "stepd" / t / "B" / "gateway").glob("requests-*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                x = json.loads(line)
                if x.get("method") != "POST" or x.get("status") != 200 or not x.get("analysis"):
                    continue
                for det in x["analysis"]["trim"]["details"]:
                    how_saved[det["how"]] += det["before"] - det["after"]
    total = sum(how_saved.values())
    print(f"  saved by technique (sum over every request it was applied to), total {total:,}:")
    for how, v in sorted(how_saved.items(), key=lambda kv: -kv[1]):
        print(f"     {how}: {v:,} = {100 * v / total:.1f}%")
    refetch = {t: by[(t, "B")]["refetch_after_trim"] for t in tasks if num(by[(t, "B")]["refetch_after_trim"])}
    print(f"  re-fetch after trim (B): {refetch}")
    trim_ms = sorted(num(r["trim_ms"]) for r in reqs if r["arm"] == "B")
    print(f"  B trim_ms median {statistics.median(trim_ms):.1f} p95 {trim_ms[int(0.95 * len(trim_ms))]:.1f}"
          f" max {max(trim_ms):.1f} (n={len(trim_ms)})")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    benchmark(RES / "benchmark_test.csv")
    for name in ("C1", "S1", "S4", "S4shadow", "S5", "L1", "L3"):
        if (LOGS / "gateway" / name).exists():
            live(name)
    print()
    for name in ("L1", "L3"):
        dollars(name)
    stepd()


if __name__ == "__main__":
    main()
