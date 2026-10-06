"""Member C — per-task, per-tier evaluation metrics (IMPLEMENTATION_PLAN C7)."""

import csv
from pathlib import Path

from tokencut.contracts import Chunk, ChunkKind, ContextVariant, Tier

CSV_COLUMNS = [
    "instance_id",
    "repo",
    "tier",
    "repo_tokens",
    "full_tokens",
    "variant_tokens",
    "reduction_vs_full",
    "pool_has_gold_file",
    "gold_file_recall",
    "gold_chunk_recall",
    "gold_chunk_skeleton_recall",
    "t_parse_s",
    "t_graph_s",
    "t_variants_s",
]
METRIC_COLUMNS = [
    "variant_tokens",
    "reduction_vs_full",
    "pool_has_gold_file",
    "gold_file_recall",
    "gold_chunk_recall",
    "gold_chunk_skeleton_recall",
    "t_parse_s",
    "t_graph_s",
    "t_variants_s",
]
TIER_ORDER = [t.value for t in (Tier.AGGRESSIVE, Tier.MODERATE, Tier.CONSERVATIVE, Tier.FULL)]


def gold_chunk_ids(chunks: list[Chunk], gold_lines: dict[str, list[int]]) -> list[str]:
    """For each gold line: the innermost chunk containing it (METHOD beats CLASS);
    a line outside every def belongs to that file's MODULE chunk. Files not parsed are ignored."""
    by_file: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_file.setdefault(c.file_path, []).append(c)
    gold: set[str] = set()
    for file_path in sorted(gold_lines):
        file_chunks = by_file.get(file_path)
        if not file_chunks:
            continue
        module_id = f"{file_path}::<module>"
        for line in gold_lines[file_path]:
            best = None
            for c in file_chunks:
                if c.kind == ChunkKind.MODULE or not (c.start_line <= line <= c.end_line):
                    continue
                if best is None or (c.end_line - c.start_line) < (best.end_line - best.start_line):
                    best = c
            gold.add(best.id if best is not None else module_id)
    return sorted(gold)


def _ratio(hits: int, total: int) -> float | None:
    return hits / total if total else None


def task_rows(
    instance_id: str,
    repo: str,
    chunks: list[Chunk],
    variants: list[ContextVariant],
    gold_files: list[str],
    gold_lines: dict[str, list[int]],
    timings: dict[str, float],
) -> list[dict]:
    """One CSV row per tier for one task."""
    gold_chunks = gold_chunk_ids(chunks, gold_lines)
    repo_tokens = sum(c.token_count for c in chunks)
    full = next(v for v in variants if v.tier == Tier.FULL)
    full_files = {cid.split("::", 1)[0] for cid in full.kept_chunk_ids}
    pool_has_gold_file = any(f in full_files for f in gold_files)

    rows = []
    for v in variants:
        kept = set(v.kept_chunk_ids)
        kept_or_skeleton = kept | set(v.skeleton_chunk_ids)
        kept_files = {cid.split("::", 1)[0] for cid in kept}
        rows.append(
            {
                "instance_id": instance_id,
                "repo": repo,
                "tier": v.tier.value,
                "repo_tokens": repo_tokens,
                "full_tokens": full.token_count,
                "variant_tokens": v.token_count,
                "reduction_vs_full": 1 - v.token_count / full.token_count
                if full.token_count
                else 0.0,
                "pool_has_gold_file": int(pool_has_gold_file),
                "gold_file_recall": _ratio(
                    sum(f in kept_files for f in gold_files), len(gold_files)
                ),
                "gold_chunk_recall": _ratio(sum(g in kept for g in gold_chunks), len(gold_chunks)),
                "gold_chunk_skeleton_recall": _ratio(
                    sum(g in kept_or_skeleton for g in gold_chunks), len(gold_chunks)
                ),
                "t_parse_s": round(timings.get("parse", 0.0), 4),
                "t_graph_s": round(timings.get("graph", 0.0), 4),
                "t_variants_s": round(timings.get("variants", 0.0), 4),
            }
        )
    return rows


def write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in CSV_COLUMNS})


def mean_by_tier(rows: list[dict]) -> dict[str, dict[str, float | None]]:
    """tier -> metric -> mean over tasks (rows with an empty value are skipped for that metric)."""
    result: dict[str, dict[str, float | None]] = {}
    for tier in TIER_ORDER:
        tier_rows = [r for r in rows if r["tier"] == tier]
        if not tier_rows:
            continue
        result[tier] = {}
        for metric in METRIC_COLUMNS:
            values = [float(r[metric]) for r in tier_rows if r.get(metric) not in (None, "")]
            result[tier][metric] = sum(values) / len(values) if values else None
    return result


def _fmt(value: float | None, metric: str) -> str:
    if value is None:
        return "–"
    if metric == "variant_tokens":
        return f"{value:,.0f}"
    if metric.startswith("t_"):
        return f"{value:.2f}"
    return f"{value:.3f}"


def summary_markdown(rows: list[dict]) -> str:
    """Mean of each metric per tier, overall and per repo (goes into the report)."""
    tasks = sorted({r["instance_id"] for r in rows})
    lines = ["# Iteration 1 evaluation summary", "", f"Tasks: {len(tasks)}", ""]
    groups = [("All repos", rows)]
    for repo in sorted({r["repo"] for r in rows}):
        groups.append((repo, [r for r in rows if r["repo"] == repo]))
    for title, group_rows in groups:
        n = len({r["instance_id"] for r in group_rows})
        lines += [
            f"## {title} ({n} tasks)",
            "",
            "| tier | " + " | ".join(METRIC_COLUMNS) + " |",
            "|---|" + "---|" * len(METRIC_COLUMNS),
        ]
        for tier, means in mean_by_tier(group_rows).items():
            lines.append(
                f"| {tier} | " + " | ".join(_fmt(means[m], m) for m in METRIC_COLUMNS) + " |"
            )
        lines.append("")
    return "\n".join(lines)
