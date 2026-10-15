"""Offline check of the repo index's code search (P1) on SWE-bench Lite tasks.

Question: given only the issue text, does `find_code` put the code the real fix edited
near the top? That is what would save an agent its grep + read rounds.

    file_recall@k   share of the fix's files among the first k distinct files of the ranking
    chunk_recall@k  share of the fix's chunks (functions / classes / module code) in the top k

Methods (same chunks, same query):
    bm25_files      files ranked by file-level BM25 (the standard SWE-bench retrieval baseline)
    bm25            chunks ranked by BM25 only
    tokencut_v1     the Iteration 1 scorer (0.5 graph distance + 0.5 BM25)
    index           the repo index: graph + BM25 + file BM25 (+ test-file penalty)

The index weights are chosen on the 40 dev tasks only, then reported on the 81 held-out
test tasks (never tuned on). Embeddings are not included: the cached vectors cover only the
benchmark's 32k pools, and embedding whole repositories takes hours on a laptop CPU.

    python -m eval.repoindex_eval --set dev
    python -m eval.repoindex_eval --set test --weights 0.4,0.4,0.2,1.0
"""

import argparse
import itertools
import json
import logging
import statistics
import time
from pathlib import Path

from eval.benchmark import RESULTS_DIR, bootstrap_ci, load_chunks, sign_test_p
from eval.loader import load_tasks
from eval.metrics import gold_chunk_ids
from eval.prepare import task_set
from tokencut.graph.builder import build_graph
from tokencut.repoindex.index import RepoIndex, is_test_file
from tokencut.trimming import scoring

log = logging.getLogger("repoindex_eval")

FILE_KS = (1, 3, 5)
CHUNK_KS = (5, 10, 20)
GRID = {
    "structural": (0.0, 0.2, 0.4, 0.6),
    "text": (0.2, 0.4, 0.6),
    "file": (0.0, 0.1, 0.2, 0.4),
    "test_factor": (1.0, 0.7, 0.4),  # score multiplier for chunks in test files
}


def file_recall(ranked_files: list[str], gold_files: set[str], k: int) -> float:
    return len(gold_files & set(ranked_files[:k])) / len(gold_files)


def distinct_files(chunk_files: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for f in chunk_files:
        seen.setdefault(f, None)
    return list(seen)


def metrics(ranked_ids: list[str], files_of: dict[str, str], gold: set[str],
            gold_files: set[str], ranked_files: list[str] | None = None) -> dict:
    files = ranked_files if ranked_files is not None else distinct_files(
        [files_of[c] for c in ranked_ids])
    row = {f"file@{k}": file_recall(files, gold_files, k) for k in FILE_KS}
    for k in CHUNK_KS:
        row[f"chunk@{k}"] = (len(gold & set(ranked_ids[:k])) / len(gold)) if ranked_ids else None
    return row


def rank(signals: list[tuple], w: tuple[float, float, float, float]) -> list[str]:
    """signals: (chunk id, structural, text, file, is_test) -> ids by combined score."""
    ws, wt, wf, test_factor = w
    scored = [((ws * s + wt * t + wf * f) * (test_factor if is_test else 1.0), cid)
              for cid, s, t, f, is_test in signals]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [cid for _, cid in scored]


def run_task(task, weight_sets: list[tuple]) -> dict:
    chunks = load_chunks(task)
    gold = set(gold_chunk_ids(chunks, task.gold_lines))
    gold_files = {f for f in task.gold_files if f.endswith(".py")} & {c.file_path for c in chunks}
    if not gold or not gold_files:
        return {}
    files_of = {c.id: c.file_path for c in chunks}
    start = time.perf_counter()
    index = RepoIndex.from_chunks(chunks)
    index.graph(wait=True)
    hits = index.search(task.query, k=len(chunks), weights=(1, 1, 1, 0, 1), chunks=chunks)
    search_s = time.perf_counter() - start
    signals = [(h.chunk.id, h.structural, h.text, h.file, is_test_file(h.chunk.file_path))
               for h in hits]
    out = {"task": task.instance_id, "repo": task.repo, "chunks": len(chunks),
           "index_build_and_search_s": round(search_s, 2), "methods": {}}

    by_text = [cid for cid, *_ in sorted(signals, key=lambda x: (-x[2], x[0]))]
    out["methods"]["bm25"] = metrics(by_text, files_of, gold, gold_files)
    file_scores = {h.chunk.file_path: h.file for h in hits}
    ranked_files = sorted(file_scores, key=lambda f: (-file_scores[f], f))
    out["methods"]["bm25_files"] = metrics([], files_of, gold, gold_files, ranked_files)
    v1 = scoring.score_chunks(chunks, build_graph(chunks), task.query)
    out["methods"]["tokencut_v1"] = metrics([s.chunk_id for s in v1], files_of, gold, gold_files)
    for w in weight_sets:
        out["methods"]["index " + ",".join(f"{x:g}" for x in w)] = metrics(
            rank(signals, w), files_of, gold, gold_files)
    return out


def summarise(rows: list[dict], label: str) -> str:
    keys = [f"file@{k}" for k in FILE_KS] + [f"chunk@{k}" for k in CHUNK_KS]
    methods = list(rows[0]["methods"])
    lines = [f"## {label}: {len(rows)} tasks", "",
             "| method | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for m in methods:
        cells = []
        for k in keys:
            values = [r["methods"][m][k] for r in rows if r["methods"][m].get(k) is not None]
            cells.append(f"{statistics.mean(values):.2f}" if values else "–")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compare(rows: list[dict], a: str, b: str, key: str) -> str:
    diffs = [r["methods"][a][key] - r["methods"][b][key] for r in rows
             if r["methods"][a][key] is not None and r["methods"][b][key] is not None]
    if not diffs:
        return f"{a} vs {b} on {key}: not comparable"
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    lo, hi = bootstrap_ci(diffs)
    return (f"{a} − {b} on {key}: mean {statistics.mean(diffs):+.3f} (95% CI {lo:+.3f} … "
            f"{hi:+.3f}), better in {wins}, worse in {losses}, sign test p = "
            f"{sign_test_p(wins, losses):.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["dev", "test"], default="dev")
    parser.add_argument("--weights", default="", help="structural,text,file,test_factor; "
                        "several sets separated by ';' (default: the whole grid; use the dev "
                        "choice for test)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--summary-only", action="store_true",
                        help="re-summarise the saved per-task rows without re-running")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.weights:  # one or more sets: "0.2,0.4,0,0.7;0.4,0.4,0.2,1"
        weight_sets = [tuple(float(x) for x in w.split(",")) for w in args.weights.split(";")]
    else:
        weight_sets = list(itertools.product(*GRID.values()))
    weight_sets = [w for w in weight_sets if any(w[:3])]

    rows = []
    saved = args.out / f"repoindex_{args.set}.json"
    tasks = [] if args.summary_only else load_tasks(task_set(args.set))
    if args.summary_only:
        rows = json.loads(saved.read_text())
    for i, task in enumerate(tasks, 1):
        row = run_task(task, weight_sets)
        if row:
            rows.append(row)
            log.info("[%d] %s: %d chunks, %.1fs", i, task.instance_id, row["chunks"],
                     row["index_build_and_search_s"])
    args.out.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(rows, indent=1))

    objective = ("file@5", "chunk@10")
    index_methods = [m for m in rows[0]["methods"] if m.startswith("index ")]
    best = max(index_methods, key=lambda m: (
        sum(statistics.mean(r["methods"][m][k] for r in rows) for k in objective), m))
    shown = ["bm25_files", "bm25", "tokencut_v1", best]
    if "index 0.4,0.4,0.2,1" in index_methods and best != "index 0.4,0.4,0.2,1":
        shown.append("index 0.4,0.4,0.2,1")
    table_rows = [{**r, "methods": {m: r["methods"][m] for m in shown}} for r in rows]
    report = [f"# Repo index search — {args.set} set", "", summarise(table_rows, args.set), "",
              f"Best index weights on this set (mean of {' + '.join(objective)}): `{best}`", ""]
    for other in ("bm25_files", "bm25", "tokencut_v1"):
        for key in objective:
            report.append("- " + compare(rows, best, other, key))
    times = [r["index_build_and_search_s"] for r in rows]
    report += ["", (f"Index build + graph + one search per task: median "
                    f"{statistics.median(times):.1f} s, max {max(times):.1f} s (from cached "
                    "chunks; a running server only searches).")]
    (args.out / f"repoindex_{args.set}.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
