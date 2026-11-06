"""Live demo: TokenCut's code shortens a real recorded agent session (no API calls, $0).

Replays the request bodies Claude Code sent during a step D run (captured locally with
TOKENCUT_CAPTURE=1) through the real gateway code (records.analyse, trim mode, the P1
settings) and prints, per request, how many tokens the agent sent and how many TokenCut
forwards, and which technique removed what. Then shows the bill the provider actually
charged for that bug with and without TokenCut (eval/results/stepd_p1/runs.csv).

    python -m eval.demo_savings                       # sympy__sympy-20590
    python -m eval.demo_savings sympy__sympy-21612
"""

import csv
import sys
from collections import Counter
from pathlib import Path

from tokencut.gateway.history import CacheEventConfig
from tokencut.gateway.records import analyse
from tokencut.gateway.session import SessionStore
from tokencut.gateway.trimmer import TrimConfig
from tokencut.gateway.virtual import VirtualConfig

ROOT = Path(__file__).resolve().parent.parent
LABELS = {
    "tools_lazy": "lazy tool loading (unused tool definitions)",
    "task_masked": "task masking (finished task's outputs)",
    "history_pinned": "task masking, repeated identically (cache-safe)",
    "history_old_output": "cache event: old output",
    "code": "on-arrival trim: code",
    "command_output": "on-arrival trim: logs",
    "search": "on-arrival trim: search",
    "ccr_tool": "cost of the retrieve tool definition",
    "tools_enabled": "answer to enable-tools",
    "ccr_retrieved": "original returned on request",
}


def main(argv: list[str] | None = None) -> None:
    task = (argv if argv is not None else sys.argv[1:] or ["sympy__sympy-20590"])[0]
    captures = sorted((ROOT / "logs" / "stepd_p1" / task / "P" / "gateway" / "captures").glob("*.json"))
    if not captures:
        raise SystemExit(f"no captured requests for {task} (logs/ is local, not in git)")
    store, virtual = SessionStore(), VirtualConfig(ccr=True, lazy_tools=True)
    by_how: Counter[str] = Counter()
    sent_total = fwd_total = 0
    print(f"Replaying {len(captures)} requests Claude Code sent while fixing {task}\n")
    print(f"{'req':>3} {'agent sent':>11} {'TokenCut sends':>15} {'saved':>7}  ms")
    for i, path in enumerate(captures, 1):
        analysis, _ = analyse("anthropic", "v1/messages", path.read_bytes(), TrimConfig(),
                              (store, CacheEventConfig()), now=float(i), virtual=virtual)
        trim = analysis["trim"]
        sent = analysis["tokens_total"]
        fwd = sent - trim["tokens_saved"]
        sent_total += sent
        fwd_total += fwd
        for d in trim["details"]:
            by_how[d["how"]] += d["before"] - d["after"]
        print(f"{i:>3} {sent:>11,} {fwd:>15,} {100 * (sent - fwd) / max(sent, 1):>6.0f}%  "
              f"{analysis['analysis_ms']:.0f}")
    print(f"\nTotal: agent sent {sent_total:,} tokens, TokenCut forwarded {fwd_total:,} "
          f"(-{100 * (sent_total - fwd_total) / sent_total:.0f}%)\n\nRemoved by technique:")
    for how, n in by_how.most_common():
        print(f"  {n:>10,}  {LABELS.get(how, how)}")

    runs = {(r["task"], r["arm"]): r for r in csv.DictReader(
        (ROOT / "eval" / "results" / "stepd_p1" / "runs.csv").open(encoding="utf-8"))}
    a, p = runs.get((task, "A")), runs.get((task, "P"))
    if a and p:
        print("\nWhat the provider actually billed for this bug (Haiku list prices):")
        for label, r in (("without TokenCut", a), ("with TokenCut   ", p)):
            print(f"  {label}: ${float(r['cost_usd']):.4f}, {r['requests']} requests, "
                  f"fixed={r['resolved']}")
        print(f"  change: {100 * (float(p['cost_usd']) - float(a['cost_usd'])) / float(a['cost_usd']):+.0f}%")
        print("\nProxy tokens (tiktoken) undercount Claude ~1.5x; the bill above is the provider's own count.")


if __name__ == "__main__":
    main()
