"""Per-task TokenCut diagnostics on the held-out test set (step B2 + B4).

Writes one JSON line per (task, setting) with skeleton stats per tier and per-gold-chunk status.
Run from repo root:  .venv/Scripts/python.exe <this> <out.jsonl>
"""

import json
import sys
import time

sys.path.insert(0, ".")

from eval.benchmark import (  # noqa: E402
    FRACTIONS,
    TIERS,
    Context,
    bm25_file_pool,
    load_chunks,
    method_tokencut,
    rendered_tokens,
)
from eval.loader import load_tasks  # noqa: E402
from eval.metrics import gold_chunk_ids  # noqa: E402
from eval.prepare import task_set  # noqa: E402
from tokencut.contracts import EVAL_FULL_CONTEXT_CAP_TOKENS  # noqa: E402
from tokencut.graph.builder import build_graph  # noqa: E402
from tokencut.trimming import scoring  # noqa: E402
from tokencut.trimming.packing import build_pool  # noqa: E402
from tokencut.trimming.query import analyse_query  # noqa: E402

out = open(sys.argv[1], "w", encoding="utf-8")
tasks = load_tasks(task_set("test"))
for i, task in enumerate(tasks, 1):
    t0 = time.perf_counter()
    chunks = load_chunks(task)
    by_id = {c.id: c for c in chunks}
    graph = build_graph(chunks)
    ctx = Context(task, chunks, graph)
    gold = sorted(gold_chunk_ids(chunks, task.gold_lines))
    mentions = analyse_query(task.query, chunks)
    scores = ctx.tokencut_scores()
    rank = {s.chunk_id: n for n, s in enumerate(scores)}
    neutral = bm25_file_pool(chunks, task.query, EVAL_FULL_CONTEXT_CAP_TOKENS)
    neutral_tokens = rendered_tokens(neutral, {c.id for c in neutral})
    own = build_pool(chunks, scores, EVAL_FULL_CONTEXT_CAP_TOKENS,
                     scoring.file_text_scores(chunks, task.query))
    settings = {
        "trim": (neutral, [round(f * neutral_tokens) for f in FRACTIONS]),
        "retrieve": (own, [round(f * EVAL_FULL_CONTEXT_CAP_TOKENS) for f in FRACTIONS]),
    }
    for setting, (pool, budgets) in settings.items():
        pool_ids = {c.id for c in pool}
        sels = method_tokencut(ctx, pool, budgets)
        tiers = []
        for f, b, (kept, skel) in zip(FRACTIONS, budgets, sels):
            skel_tokens = rendered_tokens(chunks, set(), skel) if skel else 0
            tiers.append({"fraction": f, "budget": b, "kept": len(kept), "skeletons": len(skel),
                          "skeleton_tokens": skel_tokens,
                          "gold_kept": sorted(set(gold) & kept),
                          "gold_skel": sorted(set(gold) & skel)})
        gold_info = []
        for g in gold:
            c = by_id[g]
            gold_info.append({
                "id": g, "kind": c.kind.value, "tokens": c.token_count,
                "in_pool": g in pool_ids, "rank": rank.get(g),
                "mention_weight": mentions.weights.get(g, 0.0),
                "file_mentioned": c.file_path in mentions.file_paths,
            })
        out.write(json.dumps({"instance_id": task.instance_id, "repo": task.repo,
                              "setting": setting, "pool_size": len(pool),
                              "n_chunks": len(chunks), "gold": gold_info, "tiers": tiers}) + "\n")
    out.flush()
    print(f"[{i}/{len(tasks)}] {task.instance_id} ({time.perf_counter() - t0:.0f}s)", flush=True)
out.close()
