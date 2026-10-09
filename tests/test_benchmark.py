"""Benchmark helpers: fair packing, statistics, mixed pool (no network, no embeddings)."""

from eval.benchmark import bootstrap_ci, greedy_fill, rendered_tokens, sign_test_p, summarise
from tokencut.contracts import ScoredChunk, Tier
from tokencut.graph.queries import GraphIndex
from tokencut.trimming.packing import TierPacker


def test_greedy_fill_respects_budget_and_order(expected_chunks):
    order = [c.id for c in sorted(expected_chunks, key=lambda c: -c.token_count)]
    for budget in (0, 30, 120, 400):
        kept = greedy_fill(expected_chunks, order, budget)
        assert rendered_tokens(expected_chunks, kept) <= budget
    big = greedy_fill(expected_chunks, order, 10_000)
    assert big == {c.id for c in expected_chunks if c.token_count > 0}


def test_sign_test_and_bootstrap():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(10, 0) < 0.01
    assert abs(sign_test_p(5, 5) - 1.0) < 1e-9
    lo, hi = bootstrap_ci([0.0, 1.0] * 20)
    assert 0.3 < lo < 0.5 < hi < 0.7
    assert bootstrap_ci([0.5] * 10) == (0.5, 0.5)


def test_summary_pairs_methods_against_tokencut():
    rows = []
    for task, (tc, other) in {"t1": (1.0, 0.0), "t2": (1.0, 1.0), "t3": (0.5, 0.0)}.items():
        for method, value in (("tokencut", tc), ("bm25", other)):
            rows.append({"instance_id": task, "setting": "retrieve", "method": method,
                         "fraction": 0.15, "chunk_recall": value, "file_recall": 1.0,
                         "chunk_recall_available": value})
    text = summarise(rows)
    assert "| 15% | bm25 | 0.333 |" in text
    assert "2 / 1 / 0" in text  # TokenCut better on 2 tasks, equal on 1, worse on none



def test_tier_packer_uses_per_tier_ranking(expected_chunks, expected_graph):
    ids = [c.id for c in expected_chunks if c.token_count > 0]
    forward = [ScoredChunk(chunk_id=cid, score=1.0 - i / 100) for i, cid in enumerate(ids)]
    backward = [ScoredChunk(chunk_id=cid, score=1.0 - i / 100) for i, cid in enumerate(reversed(ids))]
    budgets = {Tier.AGGRESSIVE: 60, Tier.MODERATE: 120, Tier.CONSERVATIVE: 120, Tier.FULL: 10_000}
    index = GraphIndex(expected_graph)
    plain = {p.tier: p for p in TierPacker(expected_chunks, forward, index).pack(budgets)}
    mixed = {p.tier: p for p in TierPacker(expected_chunks, forward, index,
                                           tier_scores={Tier.MODERATE: backward}).pack(budgets)}
    assert mixed[Tier.AGGRESSIVE].kept == plain[Tier.AGGRESSIVE].kept  # unchanged tier
    assert mixed[Tier.MODERATE].kept != plain[Tier.MODERATE].kept  # different ranking used
    assert plain[Tier.AGGRESSIVE].kept <= mixed[Tier.MODERATE].kept  # still nested
