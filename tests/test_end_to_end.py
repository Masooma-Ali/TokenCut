"""Integration: repo on disk -> parse_repo -> build_graph -> generate_variants."""

from tokencut.contracts import Tier
from tokencut.graph.builder import build_graph
from tokencut.ingestion.parser import parse_repo
from tokencut.trimming.variants import generate_variants


def pipeline(repo_path):
    chunks = parse_repo(repo_path)
    return chunks, build_graph(chunks)


def test_end_to_end_sample_repo(sample_repo_path):
    chunks, graph = pipeline(sample_repo_path)
    variants = generate_variants(chunks, graph, "Cart.total applies the discount wrong")
    assert [v.tier for v in variants] == [
        Tier.AGGRESSIVE,
        Tier.MODERATE,
        Tier.CONSERVATIVE,
        Tier.FULL,
    ]
    sizes = [v.token_count for v in variants]
    assert sizes == sorted(sizes)
    for v in variants:
        assert v.token_count <= v.budget_tokens
        assert "shop/cart.py::Cart.total" in v.kept_chunk_ids


def test_end_to_end_with_small_floor_trims(sample_repo_path):
    chunks, graph = pipeline(sample_repo_path)
    variants = generate_variants(
        chunks, graph, "Cart.total applies the discount wrong", aggressive_min_tokens=0
    )
    assert variants[0].token_count < variants[-1].token_count
    assert "def total(self)" in variants[0].rendered_text
