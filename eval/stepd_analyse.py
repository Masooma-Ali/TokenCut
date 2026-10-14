"""Step D analysis: paired A (no trimming) vs B (TokenCut) results over all tasks.

Reads eval/results/stepd/runs.csv and requests.csv (numbers only) and writes
eval/results/stepd/analysis.md. Paired by task: every task was run once in each arm.

    python -m eval.stepd_analyse
"""

import csv
import random
import statistics
import sys
from pathlib import Path

from eval.stepd_run import warm_start_costs

RESULTS = Path(__file__).resolve().parent / "results" / "stepd"
BOOTSTRAP = 5000


def num(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def ci_of_mean(diffs: list[float]) -> tuple[float, float]:
    """95 % bootstrap interval of the mean paired difference (fixed seed)."""
    if not diffs:
        return 0.0, 0.0
    rng = random.Random(13)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(BOOTSTRAP))
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP)]


def sign_test(better: int, worse: int) -> float:
    """Two-sided exact sign test p-value."""
    from math import comb
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    with (RESULTS / "runs.csv").open(encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    with (RESULTS / "requests.csv").open(encoding="utf-8") as f:
        requests = list(csv.DictReader(f))
    warm = warm_start_costs()
    by = {(r["task"], r["arm"]): r for r in runs}
    tasks = sorted({t for t, _ in by if (t, "A") in by and (t, "B") in by})
    for t in tasks:
        for arm in "AB":
            by[(t, arm)]["cost_warm"] = warm.get((t, arm), num(by[(t, arm)]["cost_usd"]))
    reqs_by = {}
    for r in requests:
        reqs_by.setdefault((r["task"], r["arm"]), []).append(r)

    def per_request(t: str, arm: str, field: str) -> float:
        rows = [r for r in reqs_by.get((t, arm), []) if r["status"] == "200"]
        return statistics.fmean(num(r[field]) for r in rows) if rows else 0.0

    lines = ["# Step D — paired results (A = no trimming, B = TokenCut: moderate + R8 + R10)", "",
             (f"Tasks with both arms: **{len(tasks)}** (SWE-bench Lite, sympy). Agent: Claude Code, "
             "Haiku 5.5, three-message sessions. Cost = Haiku 5.5 list prices (100K step, 1-hour "
             "cache writes). Warm-start cost removes the cache-order effect (§7)."), ""]

    # correctness
    ra = sum(by[(t, "A")]["resolved"] == "True" for t in tasks)
    rb = sum(by[(t, "B")]["resolved"] == "True" for t in tasks)
    only_a = [t for t in tasks if by[(t, "A")]["resolved"] == "True" and by[(t, "B")]["resolved"] != "True"]
    only_b = [t for t in tasks if by[(t, "B")]["resolved"] == "True" and by[(t, "A")]["resolved"] != "True"]
    lines += ["## Correctness", "",
              "| | A (no trimming) | B (TokenCut) |", "|---|---|---|",
              (f"| resolved (official hidden tests) | **{ra}/{len(tasks)}** ({100 * ra / len(tasks):.0f} %) | "
              f"**{rb}/{len(tasks)}** ({100 * rb / len(tasks):.0f} %) |"), "",
              (f"Discordant tasks: resolved only in A: {only_a or 'none'}; only in B: {only_b or 'none'}. "
              f"Exact sign test on discordant pairs: p = {sign_test(len(only_b), len(only_a)):.2f}."), ""]

    # cost / tokens / requests / time
    metrics = [("cost_warm", "cost per task, warm start ($)"), ("cost_usd", "cost per task, raw ($)"),
               ("tokens_after", "tokens sent per task"), ("requests", "requests per task"),
               ("wall_s", "wall time per task (s)")]
    lines += ["## Cost, tokens, requests, time (per task)", "",
              "| measure | A mean | B mean | B − A (mean) | 95 % CI of the difference | change | B lower in |",
              "|---|---|---|---|---|---|---|"]
    for key, label in metrics:
        a = [num(by[(t, "A")][key]) for t in tasks]
        b = [num(by[(t, "B")][key]) for t in tasks]
        diffs = [y - x for x, y in zip(a, b)]
        lo, hi = ci_of_mean(diffs)
        lower = sum(1 for d in diffs if d < 0)
        change = 100 * (sum(b) - sum(a)) / sum(a) if sum(a) else 0.0
        lines.append(f"| {label} | {statistics.fmean(a):,.4g} | {statistics.fmean(b):,.4g} | "
                     f"{statistics.fmean(diffs):+,.4g} | {lo:+,.4g} … {hi:+,.4g} | {change:+.1f} % | "
                     f"{lower}/{len(tasks)} |")
    # per request (removes "the agent took more steps" noise)
    lines += ["", "Per request (removes differences in how many steps the agent took):", "",
              "| measure | A | B | change |", "|---|---|---|---|"]
    for field, label in (("tokens_after", "tokens sent per request"), ("cost_usd", "cost per request ($)"),
                         ("billed_prompt", "billed prompt per request")):
        a = statistics.fmean(per_request(t, "A", field) for t in tasks)
        b = statistics.fmean(per_request(t, "B", field) for t in tasks)
        lines.append(f"| {label} | {a:,.4g} | {b:,.4g} | {100 * (b - a) / a:+.1f} % |")
    over_a = sum(num(by[(t, "A")]["requests_over_100k"]) for t in tasks)
    over_b = sum(num(by[(t, "B")]["requests_over_100k"]) for t in tasks)
    lines += ["", f"Requests above Haiku's 100K price step: A {int(over_a)}, B {int(over_b)}.", ""]

    # what TokenCut did in B
    masks = sum(num(by[(t, "B")]["task_masks"]) for t in tasks)
    onarr = sum(num(by[(t, "B")]["on_arrival_trims"]) for t in tasks)
    saved = sum(num(by[(t, "B")]["tokens_saved"]) for t in tasks)
    before = sum(num(by[(t, "B")]["tokens_before"]) for t in tasks)
    refetch = sum(num(by[(t, "B")]["refetch_after_trim"]) for t in tasks)
    rep_a = sum(num(by[(t, "A")]["repeated_calls"]) for t in tasks)
    rep_b = sum(num(by[(t, "B")]["repeated_calls"]) for t in tasks)
    acted = sum(1 for t in tasks if num(by[(t, "B")]["tokens_saved"]) > 0)
    trim_ms = [num(r["trim_ms"]) for t in tasks for r in reqs_by.get((t, "B"), [])]
    lines += ["## What TokenCut did in B", "",
              (f"- Acted in **{acted}/{len(tasks)}** tasks; requests carrying an R10 mask: {int(masks)}; "
              f"requests carrying an on-arrival trim: {int(onarr)}."),
              (f"- Tokens removed in B: {int(saved):,} of {int(before):,} that the agent sent "
              f"(**{100 * saved / before:.1f} %**)."),
              (f"- Re-fetches of trimmed / masked output: **{int(refetch)}**; identical repeated tool calls: "
              f"A {int(rep_a)}, B {int(rep_b)}."),
              (f"- Trimming time per request: median {statistics.median(trim_ms):.1f} ms, "
              f"p95 {sorted(trim_ms)[int(0.95 * len(trim_ms))]:.1f} ms, max {max(trim_ms):.1f} ms."), ""]

    # per task table
    lines += ["## Per task", "",
              ("| task | resolved A / B | cost warm A / B ($) | requests A / B | tokens sent A / B | "
              "wall A / B (s) | B: tokens removed | B: masks / trims | B: re-fetch |"),
              "|---|---|---|---|---|---|---|---|---|"]
    for t in tasks:
        a, b = by[(t, "A")], by[(t, "B")]
        lines.append(
            f"| {t.replace('sympy__sympy-', '')} | {'✅' if a['resolved'] == 'True' else '❌'} / "
            f"{'✅' if b['resolved'] == 'True' else '❌'} | {num(a['cost_warm']):.4f} / {num(b['cost_warm']):.4f} | "
            f"{a['requests']} / {b['requests']} | {int(num(a['tokens_after'])):,} / {int(num(b['tokens_after'])):,} | "
            f"{a['wall_s']} / {b['wall_s']} | {int(num(b['tokens_saved'])):,} | {b['task_masks']} / "
            f"{b['on_arrival_trims']} | {b['refetch_after_trim']} |")
    status = [(r["task"], r["arm"], r["status"]) for r in runs if r["status"] != "completed"]
    lines += ["", f"Runs not ending normally: {status or 'none'}."]
    out = RESULTS / "analysis.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    print("\n".join(lines))


if __name__ == "__main__":
    main()
