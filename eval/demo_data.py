"""Numbers for the results page (eval/results/demo/data.json), all from files in this repo.

  * per bug: billed cost / requests / fixed, with and without TokenCut (step D with P1,
    eval/results/stepd_p1/runs.csv) and Masooma's step D (eval/results/stepd/runs.csv)
  * per bug: every recorded request replayed through the gateway code (records.analyse,
    the P1 settings): tokens the agent sent vs tokens TokenCut forwards, and by technique
  * re-runs of the failed bugs (eval/results/stepd_p1_rerun/runs.csv)

    python -m eval.demo_data            # replay + data.json, then the page's row files
    python -m eval.demo_data --rows     # only rebuild the row files from data.json
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from tokencut.gateway.history import CacheEventConfig
from tokencut.gateway.records import analyse
from tokencut.gateway.session import SessionStore
from tokencut.gateway.trimmer import TrimConfig
from tokencut.gateway.virtual import VirtualConfig

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"


def runs(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r["status"] != "usage_limit"]
    return {(r["task"], r["arm"]): r for r in rows}


def brief(r: dict | None) -> dict | None:
    if r is None:
        return None
    return {"cost": round(float(r["cost_usd"] or 0), 5), "requests": int(r["requests"] or 0),
            "fixed": r["resolved"] == "True", "wall_s": int(float(r["wall_s"] or 0))}


def replay(task: str) -> dict:
    captures = sorted((ROOT / "logs" / "stepd_p1" / task / "P" / "gateway" / "captures").glob("*.json"))
    store, virtual = SessionStore(), VirtualConfig(ccr=True, lazy_tools=True)
    reqs, by_how = [], Counter()
    for i, path in enumerate(captures, 1):
        a, _ = analyse("anthropic", "v1/messages", path.read_bytes(), TrimConfig(),
                       (store, CacheEventConfig()), now=float(i), virtual=virtual)
        t = a["trim"]
        reqs.append([a["tokens_total"], a["tokens_total"] - t["tokens_saved"],
                     round(a["analysis_ms"], 1)])
        for d in t["details"]:
            by_how[d["how"]] += d["before"] - d["after"]
    return {"requests": reqs, "by_how": dict(by_how.most_common())}


REPO = "https://github.com/Masooma-Ali/TokenCut/blob"
BRANCH = "feature/p1-big-wins"
LABELS = {
    "tools_lazy": "Lazy tool loading", "history_pinned": "Task masking (kept identical)",
    "task_masked": "Task masking (new mask)", "command_output": "On-arrival trim (logs)",
    "code": "On-arrival trim (code)", "search": "On-arrival trim (search)",
    "ccr_tool": "Retrieve tool definition (cost)", "tools_enabled": "Enable-tools answers (cost)",
    "ccr_retrieved": "Retrieved originals (cost)",
}
PROOFS = [
    ("Step D with P1: 29 real bugs, with vs without TokenCut", "eval/results/stepd_p1/runs.csv",
     "python -m eval.stepd_p1_run $(cat eval/results/stepd/task_list.txt)", BRANCH),
    ("Per-request savings replayed through the gateway code", "eval/results/demo/requests.json",
     "python -m eval.demo_savings sympy__sympy-21055", BRANCH),
    ("Masooma's step D (R8 + R10) vs no TokenCut", "eval/results/stepd/summary.md",
     "python -m eval.stepd_run --summary-only", "main"),
    ("Offline trimming benchmark, 81 held-out tasks", "eval/results/benchmark_test.md",
     "python -m eval.benchmark --set test --methods tokencut,bm25,random,head,bm25_files", "main"),
    ("Repo search vs BM25, 80 held-out tasks", "eval/results/repoindex_test.md",
     "python -m eval.repoindex_eval --set test --weights \"0.2,0.4,0,0.7\"", BRANCH),
    ("Re-runs of the 4 failed bugs after the fix", "eval/results/stepd_p1_rerun/runs.csv",
     "STEPD_P1_OUT=stepd_p1_rerun STEPD_P1_ARMS=P python -m eval.stepd_p1_run <bugs>", BRANCH),
    ("Cache safety: earlier messages sent byte-for-byte", "tests/test_gateway_trimming.py",
     "pytest -q", "main"),
    ("Every decision and bug fix, dated", "docs/DECISIONS.md", "—", BRANCH),
]


def write_rows() -> None:
    """The page's datasets as JSON row files (eval/results/demo/*.json)."""
    out = RESULTS / "demo"
    data = json.loads((out / "data.json").read_text(encoding="utf-8"))
    rows: dict[str, list[dict]] = {"bugs": [], "requests": [], "techniques": []}
    for b in data["bugs"]:
        a, p, ma, mb, rr = b["A"], b["P"], b["mA"] or {}, b["mB"] or {}, b["rerunP"] or {}
        rows["bugs"].append({
            "task": b["task"], "a_cost": a["cost"], "p_cost": p["cost"],
            "a_requests": a["requests"], "p_requests": p["requests"],
            "a_fixed": a["fixed"], "p_fixed": p["fixed"],
            "m_a_cost": ma.get("cost"), "m_b_cost": mb.get("cost"),
            "m_a_fixed": ma.get("fixed"), "m_b_fixed": mb.get("fixed"),
            "rerun_p_fixed": rr.get("fixed"), "rerun_p_cost": rr.get("cost")})
        for n, (sent, fwd, ms) in enumerate(b["replay"]["requests"], 1):
            rows["requests"].append({"task": b["task"], "req": n, "agent_tokens": sent,
                                     "tokencut_tokens": fwd, "ms": ms})
        for how, tokens in b["replay"]["by_how"].items():
            rows["techniques"].append({"task": b["task"], "technique": LABELS.get(how, how),
                                       "tokens": tokens})
    bench = []
    text = (RESULTS / "benchmark_test.md").read_text(encoding="utf-8").split("## Setting")[1]
    names = {"tokencut": "TokenCut", "bm25": "BM25", "embed": "Embeddings",
             "head": "Head truncation", "random": "Random"}
    for budget, method, recall in re.findall(r"^\| (\d+)% \| (\w+) \| ([0-9.]+) \|", text, re.MULTILINE):
        if method in names:
            bench.append({"budget": f"{budget}%", "method": names[method], "recall": float(recall)})
    rows["benchmark"] = bench
    search = []
    names = {"bm25_files": "File BM25 (SWE-bench baseline)", "bm25": "Chunk BM25",
             "tokencut_v1": "TokenCut before (Iteration 1)",
             "index 0.2,0.4,0,0.7": "Repo index (new)"}
    for line in (RESULTS / "repoindex_test.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and cells[0] in names:
            num = [float(c) if c not in ("–", "") else None for c in cells[1:7]]
            search.append({"method": names[cells[0]], "file_at_1": num[0], "file_at_3": num[1],
                           "file_at_5": num[2], "chunk_at_5": num[3], "chunk_at_10": num[4],
                           "chunk_at_20": num[5]})
    rows["repo_search"] = search
    rows["proofs"] = [{"claim": c, "evidence": f, "reproduce": cmd, "link": f"{REPO}/{ref}/{f}",
                       "branch": ref} for c, f, cmd, ref in PROOFS]
    for name, value in rows.items():
        (out / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
        print(name, len(value))


def main() -> None:
    if "--rows" in sys.argv[1:]:
        write_rows()
        return
    mine, hers = runs(RESULTS / "stepd_p1" / "runs.csv"), runs(RESULTS / "stepd" / "runs.csv")
    rerun = runs(RESULTS / "stepd_p1_rerun" / "runs.csv")
    tasks = sorted({t for t, _ in mine})
    bugs = []
    for task in tasks:
        print("replaying", task, flush=True)
        bugs.append({"task": task, "A": brief(mine.get((task, "A"))), "P": brief(mine.get((task, "P"))),
                     "mA": brief(hers.get((task, "A"))), "mB": brief(hers.get((task, "B"))),
                     "rerunP": brief(rerun.get((task, "P"))), "replay": replay(task)})
    out = RESULTS / "demo"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps({"bugs": bugs}), encoding="utf-8")
    print("wrote", out / "data.json")
    write_rows()


if __name__ == "__main__":
    main()
