"""Head-to-head benchmark: TokenCut vs baseline context selection, at equal token budgets.

Two settings (every method gets the same input and the same budget, uses the same renderer
and token counter, and must stay within budget):

  trim      All methods start from the SAME 32k "full" context (whole files chosen by
            file-level BM25, the SWE-bench retrieval baseline — neutral, not TokenCut's
            ranking) and must cut it to 15 / 35 / 60 % of its size. Pure trimming.
  retrieve  Methods choose from the whole repository within 4.8k / 11.2k / 19.2k tokens
            (15 / 35 / 60 % of 32k). Retrieval + trimming together.

Metric: gold chunk recall = share of the chunks edited by the real fix that are kept as
full code (also gold file recall, and for TokenCut the recall counting skeletons).

    python -m eval.benchmark --set dev
    python -m eval.benchmark --set test --methods tokencut,bm25,random
"""

import argparse
import csv
import json
import logging
import math
import random
import time
from collections.abc import Callable
from pathlib import Path

from eval.loader import DEFAULT_CACHE_DIR, Task, load_tasks
from eval.metrics import gold_chunk_ids
from eval.prepare import task_set
from tokencut.contracts import EVAL_FULL_CONTEXT_CAP_TOKENS, Chunk, CodeGraph, ScoredChunk, Tier
from tokencut.graph.builder import build_graph
from tokencut.graph.queries import GraphIndex
from tokencut.tokens import count_tokens
from tokencut.trimming import scoring
from tokencut.trimming.packing import SEPARATOR_TOKENS, TierPacker, build_pool
from tokencut.trimming.query import analyse_query
from tokencut.trimming.render import file_header, render_variant

FRACTIONS = (0.15, 0.35, 0.60)
TIERS = (Tier.AGGRESSIVE, Tier.MODERATE, Tier.CONSERVATIVE)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RANDOM_SEED = 13
BOOTSTRAP_SAMPLES = 2000

log = logging.getLogger("benchmark")


# ---------- shared helpers ----------


def load_chunks(task: Task) -> list[Chunk]:
    cache = DEFAULT_CACHE_DIR / "chunks" / f"{task.repo.replace('/', '__')}__{task.base_commit}.json"
    data = json.loads(cache.read_text(encoding="utf-8"))
    return [Chunk.model_validate(x) for x in data["chunks"]]


def rendered_tokens(chunks: list[Chunk], kept: set[str], skeletons: set[str] = frozenset()) -> int:
    return count_tokens(render_variant(chunks, kept, set(skeletons)))


def greedy_fill(candidates: list[Chunk], order: list[str], budget: int) -> set[str]:
    """Baseline packing: add chunks in `order` (full code) while they fit the budget, then
    verify with a real render and drop the last-added chunks until it fits."""
    by_id = {c.id: c for c in candidates}
    kept: list[str] = []
    files: set[str] = set()
    used = 0
    for cid in order:
        c = by_id.get(cid)
        if c is None or c.token_count == 0:
            continue
        cost = c.token_count + SEPARATOR_TOKENS
        if c.file_path not in files:
            cost += count_tokens(file_header(c.file_path))
        if used + cost > budget:
            continue
        kept.append(cid)
        files.add(c.file_path)
        used += cost
    while kept and rendered_tokens(candidates, set(kept)) > budget:
        kept.pop()
    return set(kept)


def bm25_file_pool(chunks: list[Chunk], query: str, cap: int) -> list[Chunk]:
    """Neutral 'full' context: whole files ranked by file-level BM25 up to `cap` tokens."""
    file_scores = scoring.file_text_scores(chunks, query)
    neutral = [ScoredChunk(chunk_id=c.id, score=file_scores.get(c.file_path, 0.0)) for c in chunks]
    # rank_files with no chunk signal = pure file BM25 (FILE_TEXT_WEIGHT only breaks ties)
    return build_pool(chunks, neutral, cap, file_scores)


# ---------- methods ----------
# Each method: (candidates, all_chunks, graph, query, budgets) -> list of (kept, skeletons)


class Context:
    """Per-task data shared by all methods (computed once)."""

    def __init__(self, task: Task, chunks: list[Chunk], graph: CodeGraph):
        self.task = task
        self.chunks = chunks
        self.graph = graph
        self.index = GraphIndex(graph)
        self.query = task.query
        self._tokencut_scores: list[ScoredChunk] | None = None
        self._bm25: dict[str, float] | None = None
        self.embedder = None  # set by main() when the embedding method is enabled

    def tokencut_scores(self) -> list[ScoredChunk]:
        if self._tokencut_scores is None:
            mentions = analyse_query(self.query, self.chunks)
            self._tokencut_scores = scoring.score_chunks(
                self.chunks, self.graph, self.query, mentions=mentions, index=self.index
            )
        return self._tokencut_scores

    def bm25(self) -> dict[str, float]:
        if self._bm25 is None:
            values = scoring.text_scores(self.chunks, self.query)
            self._bm25 = {c.id: v for c, v in zip(self.chunks, values)}
        return self._bm25


Selection = tuple[set[str], set[str]]


def _pack(ctx: Context, candidates: list[Chunk], budgets: list[int], scores: list[ScoredChunk],
          tier_scores: dict[Tier, list[ScoredChunk]] | None = None) -> list[Selection]:
    """TokenCut's nested tier packer at the benchmark's budgets."""
    packer = TierPacker(candidates, scores, ctx.index, tier_scores=tier_scores)
    tiers = dict(zip(TIERS, budgets))
    tiers[Tier.FULL] = max(rendered_tokens(candidates, {c.id for c in candidates}), budgets[-1])
    packed = {p.tier: p for p in packer.pack(tiers)}
    return [(packed[t].kept, packed[t].skeletons) for t in TIERS]


def method_tokencut(ctx: Context, candidates: list[Chunk], budgets: list[int]) -> list[Selection]:
    return _pack(ctx, candidates, budgets, ctx.tokencut_scores())


EMBED_WEIGHT = 0.5  # share of the text signal that comes from embeddings (tokencut_embed/hybrid)


def blended_scores(ctx: Context, candidates: list[Chunk]) -> list[ScoredChunk]:
    """TokenCut scores with a semantic signal: within the candidate set, the text part of the
    score becomes (1 - w) * BM25 + w * embedding similarity (min-max normalised)."""
    sims = ctx.embedder.similarities(ctx.query, candidates)
    lo, hi = min(sims.values(), default=0.0), max(sims.values(), default=0.0)
    span = (hi - lo) or 1.0
    rescored = []
    for s in ctx.tokencut_scores():
        if s.chunk_id not in sims:
            continue
        semantic = (sims[s.chunk_id] - lo) / span
        text = (1 - EMBED_WEIGHT) * s.text_score + EMBED_WEIGHT * semantic
        score = scoring.ALPHA * s.structural_score + (1 - scoring.ALPHA) * text
        rescored.append(ScoredChunk(chunk_id=s.chunk_id, score=score,
                                    structural_score=s.structural_score, text_score=text))
    rescored.sort(key=lambda sc: (-sc.score, sc.chunk_id))
    return rescored


def method_tokencut_embed(ctx: Context, candidates: list[Chunk], budgets: list[int]) -> list[Selection]:
    """TokenCut packing on the blended (structure + BM25 + embedding) score, every tier."""
    return _pack(ctx, candidates, budgets, blended_scores(ctx, candidates))


def method_tokencut_hybrid(ctx: Context, candidates: list[Chunk], budgets: list[int]) -> list[Selection]:
    """TokenCut's own ranking for the aggressive tier, the blended ranking for larger tiers."""
    blended = blended_scores(ctx, candidates)
    larger = {Tier.MODERATE: blended, Tier.CONSERVATIVE: blended}
    return _pack(ctx, candidates, budgets, ctx.tokencut_scores(), larger)


def _ranked(scores: dict[str, float], candidates: list[Chunk]) -> list[str]:
    return sorted((c.id for c in candidates), key=lambda cid: (-scores.get(cid, 0.0), cid))


def method_bm25(ctx: Context, candidates: list[Chunk], budgets: list[int]) -> list[Selection]:
    order = _ranked(ctx.bm25(), candidates)
    return [(greedy_fill(candidates, order, b), set()) for b in budgets]


EMBED_PREFILTER = 500  # over a whole repo: embeddings re-rank BM25's top candidates


def method_embed(ctx: Context, candidates: list[Chunk], budgets: list[int]) -> list[Selection]:
    """Dense retrieval (bge-small). Over a large candidate set (whole repo), the standard
    'BM25 then dense re-rank' pipeline: only BM25's top EMBED_PREFILTER chunks are embedded."""
    pool = candidates
    if len(candidates) > 2 * EMBED_PREFILTER:
        bm25 = ctx.bm25()
        pool = sorted(candidates, key=lambda c: (-bm25.get(c.id, 0.0), c.id))[:EMBED_PREFILTER]
    scores = ctx.embedder.similarities(ctx.query, pool)
    order = _ranked(scores, pool)
    return [(greedy_fill(candidates, order, b), set()) for b in budgets]


def method_random(ctx: Context, candidates: list[Chunk], budgets: list[int]) -> list[Selection]:
    order = sorted(c.id for c in candidates)
    random.Random(f"{RANDOM_SEED}:{ctx.task.instance_id}").shuffle(order)
    return [(greedy_fill(candidates, order, b), set()) for b in budgets]


def method_head(ctx: Context, candidates: list[Chunk], budgets: list[int]) -> list[Selection]:
    """Naive truncation: keep the context from the top until the budget runs out."""
    order = [c.id for c in sorted(candidates, key=lambda c: (c.file_path, c.start_line, c.id))]
    return [(greedy_fill(candidates, order, b), set()) for b in budgets]


def method_bm25_files(ctx: Context, candidates: list[Chunk], budgets: list[int]) -> list[Selection]:
    """SWE-bench retrieval baseline: whole files by file-level BM25 until the budget."""
    selections = []
    for b in budgets:
        pool = bm25_file_pool(candidates, ctx.query, b)
        kept = {c.id for c in pool}
        while kept and rendered_tokens(candidates, kept) > b:  # header estimate safety
            kept.discard(max(kept))
        selections.append((kept, set()))
    return selections


METHODS: dict[str, Callable[[Context, list[Chunk], list[int]], list[Selection]]] = {
    "tokencut": method_tokencut,
    "tokencut_embed": method_tokencut_embed,
    "tokencut_hybrid": method_tokencut_hybrid,
    "bm25": method_bm25,
    "embed": method_embed,
    "random": method_random,
    "head": method_head,
    "bm25_files": method_bm25_files,
}
NEEDS_EMBEDDINGS = {"embed", "tokencut_embed", "tokencut_hybrid"}
TRIM_ONLY = {"head"}  # meaningless over a whole repo
RETRIEVE_ONLY = {"bm25_files"}  # in the trim setting the pool already is BM25 files


# ---------- running ----------


def run_task(task: Task, methods: list[str], embedder) -> list[dict]:
    chunks = load_chunks(task)
    graph = build_graph(chunks)
    ctx = Context(task, chunks, graph)
    ctx.embedder = embedder
    gold = set(gold_chunk_ids(chunks, task.gold_lines))
    gold_files = set(task.gold_files)
    rows = []

    neutral_pool = bm25_file_pool(chunks, task.query, EVAL_FULL_CONTEXT_CAP_TOKENS)
    pool_ids = {c.id for c in neutral_pool}
    pool_tokens = rendered_tokens(neutral_pool, pool_ids)
    settings = {
        "trim": (neutral_pool, [round(f * pool_tokens) for f in FRACTIONS]),
        "retrieve": (chunks, [round(f * EVAL_FULL_CONTEXT_CAP_TOKENS) for f in FRACTIONS]),
    }
    for setting, (candidates, budgets) in settings.items():
        if setting == "retrieve":
            # TokenCut's own pipeline over the repo: its 32k pool, then its packer.
            tc_candidates = build_pool(chunks, ctx.tokencut_scores(), EVAL_FULL_CONTEXT_CAP_TOKENS,
                                       scoring.file_text_scores(chunks, task.query))
        for name in methods:
            if (setting == "retrieve" and name in TRIM_ONLY) or (
                setting == "trim" and name in RETRIEVE_ONLY):
                continue
            own_pool = setting == "retrieve" and name.startswith("tokencut")
            pool = tc_candidates if own_pool else candidates
            start = time.perf_counter()
            selections = METHODS[name](ctx, pool, budgets)
            seconds = (time.perf_counter() - start) / len(budgets)
            for fraction, budget, (kept, skeletons) in zip(FRACTIONS, budgets, selections):
                tokens = rendered_tokens(chunks, kept, skeletons)
                assert tokens <= budget, f"{name} over budget on {task.instance_id}"
                kept_files = {cid.split("::", 1)[0] for cid in kept}
                in_pool = gold & pool_ids if setting == "trim" else gold
                rows.append({
                    "instance_id": task.instance_id, "repo": task.repo, "setting": setting,
                    "method": name, "fraction": fraction, "budget": budget, "tokens": tokens,
                    "gold_chunks": len(gold), "gold_chunks_available": len(in_pool),
                    "chunk_recall": len(gold & kept) / len(gold) if gold else None,
                    "chunk_recall_available": (len(in_pool & kept) / len(in_pool)) if in_pool else None,
                    "chunk_recall_with_skeletons": (len(gold & (kept | skeletons)) / len(gold)) if gold else None,
                    "file_recall": (len(gold_files & kept_files) / len(gold_files)) if gold_files else None,
                    "seconds": round(seconds, 3),
                })
    return rows


# ---------- statistics ----------


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def bootstrap_ci(values: list[float], seed: int = RANDOM_SEED) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(mean([rng.choice(values) for _ in values]) for _ in range(BOOTSTRAP_SAMPLES))
    return means[int(0.025 * BOOTSTRAP_SAMPLES)], means[int(0.975 * BOOTSTRAP_SAMPLES) - 1]


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


METRIC_TITLES = {
    "chunk_recall": "gold chunks kept as full code (share of all gold chunks)",
    "chunk_recall_available": "gold chunks kept, out of those present in the 32k context "
                              "(pure trimming quality)",
}


def summarise(rows: list[dict]) -> str:
    n_tasks = len({r["instance_id"] for r in rows})
    lines = [f"# Context-selection benchmark ({n_tasks} tasks)", "",
             ("Mean with 95% bootstrap CI. Paired vs TokenCut: number of tasks where TokenCut "
              "is better / equal / worse, and the two-sided sign-test p-value."), ""]
    tables = [("trim", "chunk_recall_available"), ("trim", "chunk_recall"),
              ("retrieve", "chunk_recall")]
    for setting, metric in tables:
        lines += _table(rows, setting, metric)
    return "\n".join(lines)


def _table(rows: list[dict], setting: str, metric: str) -> list[str]:
    lines: list[str] = []
    srows = [r for r in rows if r["setting"] == setting]
    if not srows:
        return lines
    if True:
        lines += [f"## Setting: {setting} — {METRIC_TITLES[metric]}", "",
                  "| budget | method | recall | 95% CI | file recall | TokenCut better/equal/worse | p |",
                  "|---|---|---|---|---|---|---|"]
        methods = sorted({r["method"] for r in srows}, key=lambda m: (m != "tokencut", m))
        for fraction in FRACTIONS:
            frows = [r for r in srows if r["fraction"] == fraction]
            tc = {r["instance_id"]: r[metric] for r in frows if r["method"] == "tokencut"}
            for m in methods:
                mrows = [r for r in frows if r["method"] == m and r[metric] is not None]
                values = [r[metric] for r in mrows]
                files = [r["file_recall"] for r in mrows if r["file_recall"] is not None]
                lo, hi = bootstrap_ci(values)
                paired = ""
                if m != "tokencut":
                    better = equal = worse = 0
                    for r in mrows:
                        t = tc.get(r["instance_id"])
                        if t is None:
                            continue
                        if t > r[metric] + 1e-9:
                            better += 1
                        elif t < r[metric] - 1e-9:
                            worse += 1
                        else:
                            equal += 1
                    paired = f"{better} / {equal} / {worse} | {sign_test_p(better, worse):.3f}"
                else:
                    paired = "– | –"
                lines.append(f"| {fraction:.0%} | {m} | {mean(values):.3f} | {lo:.3f}–{hi:.3f} | "
                             f"{mean(files):.3f} | {paired} |")
        lines.append("")
    return lines


def read_csv(path: Path) -> list[dict]:
    """Rows written by main(), with numbers converted back from text."""
    numeric = {"fraction", "budget", "tokens", "gold_chunks", "gold_chunks_available",
               "chunk_recall", "chunk_recall_available", "chunk_recall_with_skeletons",
               "file_recall", "seconds"}
    rows = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in numeric:
                row[key] = float(row[key]) if row[key] not in ("", None) else None
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=["dev", "test"], default="dev")
    parser.add_argument("--methods", default="tokencut,bm25,random,head,bm25_files")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--merge", nargs="+", type=Path, default=None,
                        help="combine existing benchmark CSVs into one summary, without running")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.merge:
        rows = [row for path in args.merge for row in read_csv(path)]
        write_results(rows, args.out, args.set)
        return

    methods = args.methods.split(",")
    embedder = None
    if set(methods) & NEEDS_EMBEDDINGS:
        from eval.embeddings import Embedder  # optional dependency (fastembed)

        embedder = Embedder()
    tasks = load_tasks(task_set(args.set))
    if args.limit:
        tasks = tasks[: args.limit]

    rows: list[dict] = []
    for i, task in enumerate(tasks, 1):
        start = time.perf_counter()
        rows.extend(run_task(task, methods, embedder))
        log.info("[%d/%d] %s (%.0fs)", i, len(tasks), task.instance_id, time.perf_counter() - start)

    write_results(rows, args.out, args.set)


def write_results(rows: list[dict], out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"benchmark_{name}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / f"benchmark_{name}.md").write_text(summarise(rows) + "\n", encoding="utf-8")
    log.info("wrote %s", out / f"benchmark_{name}.md")


if __name__ == "__main__":
    main()
