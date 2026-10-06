"""Member C — query analysis, scoring, packing, rendering, generate_variants."""

import pytest

from tokencut.contracts import Chunk, ChunkKind, CodeGraph, Tier
from tokencut.graph.builder import build_graph
from tokencut.ingestion.parser import parse_repo
from tokencut.ingestion.python_parser import PythonParser
from tokencut.tokens import count_tokens
from tokencut.trimming.packing import build_pool, tier_budgets
from tokencut.trimming.query import FILE_WEIGHT, SYMBOL_WEIGHT, analyse_query
from tokencut.trimming.render import render_variant, skeleton_text
from tokencut.trimming.scoring import split_identifier, text_scores
from tokencut.trimming.variants import generate_variants, score_chunks

QUERY = "Cart.total applies the discount wrong"
TIERS = [Tier.AGGRESSIVE, Tier.MODERATE, Tier.CONSERVATIVE, Tier.FULL]


@pytest.fixture(scope="module")
def sample(expected_chunks, expected_graph):
    return expected_chunks, expected_graph


def check_variants(variants, chunks):
    """The invariants every call of generate_variants must satisfy."""
    assert [v.tier for v in variants] == TIERS
    by_id = {c.id: c for c in chunks}
    order = {c.id: (c.file_path, c.start_line, c.id) for c in chunks}
    previous_tokens = 0
    previous_kept: set[str] = set()
    for v in variants:
        assert v.token_count <= v.budget_tokens
        assert v.token_count == count_tokens(v.rendered_text)
        assert v.token_count >= previous_tokens  # never decreases aggressive -> full
        assert previous_kept <= set(v.kept_chunk_ids)  # nested tiers
        assert not set(v.kept_chunk_ids) & set(v.skeleton_chunk_ids)
        assert v.kept_chunk_ids == sorted(v.kept_chunk_ids, key=order.__getitem__)
        assert v.skeleton_chunk_ids == sorted(v.skeleton_chunk_ids, key=order.__getitem__)
        present = set(v.kept_chunk_ids) | set(v.skeleton_chunk_ids)
        for cid in v.kept_chunk_ids:
            parent = by_id[cid].parent_id
            if parent:
                assert parent in v.kept_chunk_ids  # a kept method's class header is rendered
        for cid in v.skeleton_chunk_ids:
            parent = by_id[cid].parent_id
            if parent:
                assert parent in present
        previous_tokens, previous_kept = v.token_count, set(v.kept_chunk_ids)
    assert variants[-1].skeleton_chunk_ids == []
    assert variants[-1].token_count == variants[-1].budget_tokens


# ---------- query analysis ----------


def test_query_symbol_mention(sample):
    chunks, _ = sample
    m = analyse_query(QUERY, chunks)
    assert "Cart.total" in m.symbols
    assert m.weights["shop/cart.py::Cart.total"] == SYMBOL_WEIGHT
    assert "shop/utils/pricing.py::apply_discount" not in m.weights  # 'discount' is not a name


def test_query_function_mention_matches_all_same_names(sample):
    chunks, _ = sample
    m = analyse_query("validate rejects good input", chunks)
    assert m.chunk_ids == ["shop/cart.py::validate", "shop/payments/gateway.py::validate"]


def test_query_file_mention(sample):
    chunks, _ = sample
    m = analyse_query("something is off in shop/utils/pricing.py", chunks)
    assert m.file_paths == ["shop/utils/pricing.py"]
    assert m.weights["shop/utils/pricing.py::add_tax"] == FILE_WEIGHT
    m2 = analyse_query("see pricing.py", chunks)
    assert m2.file_paths == ["shop/utils/pricing.py"]


def test_query_traceback(sample):
    chunks, _ = sample
    tb = (
        "Traceback (most recent call last):\n"
        '  File "C:\\work\\repo\\shop\\payments\\gateway.py", line 14, in process_payment\n'
        '    raise ValueError("Invalid card")\nValueError: Invalid card'
    )
    m = analyse_query(tb, chunks)
    assert m.file_paths == ["shop/payments/gateway.py"]
    assert m.weights["shop/payments/gateway.py::process_payment"] == SYMBOL_WEIGHT
    assert m.weights["shop/payments/gateway.py::validate"] == FILE_WEIGHT


def test_query_empty(sample):
    chunks, _ = sample
    m = analyse_query("", chunks)
    assert m.chunk_ids == [] and m.symbols == [] and m.file_paths == []


# ---------- scoring ----------


def test_split_identifier():
    assert split_identifier("apply_discount") == ["apply", "discount"]
    assert split_identifier("PremiumCustomer") == ["premium", "customer"]
    assert split_identifier("HTTPServerError") == ["http", "server", "error"]


def test_scoring_ranks_mentioned_chunk_first(sample):
    chunks, graph = sample
    scores = score_chunks(chunks, graph, QUERY)
    ids = [s.chunk_id for s in scores]
    assert ids[0] == "shop/cart.py::Cart.total"
    assert ids.index("shop/utils/pricing.py::apply_discount") < ids.index(
        "shop/payments/refunds.py::log_failure"
    )
    assert len(scores) == len(chunks)
    for s in scores:
        assert 0.0 <= s.text_score <= 1.0 and 0.0 <= s.structural_score <= 1.0


def test_scoring_deterministic_and_tie_broken_by_id(sample):
    chunks, graph = sample
    first = score_chunks(chunks, graph, QUERY)
    second = score_chunks(list(reversed(chunks)), graph, QUERY)
    assert [s.model_dump() for s in first] == [s.model_dump() for s in second]
    keys = [(-s.score, s.chunk_id) for s in first]
    assert keys == sorted(keys)


def test_no_mentions_falls_back_to_text_score(sample):
    chunks, graph = sample
    scores = score_chunks(chunks, graph, "card number checks fail")
    assert scores[0].text_score > 0
    assert any(s.structural_score > 0 for s in scores)  # pseudo-mentions from top text matches


def test_empty_graph_and_unmatched_query(sample):
    chunks, _ = sample
    empty_graph = CodeGraph(chunk_ids=[c.id for c in chunks], edges=[])
    scores = score_chunks(chunks, empty_graph, "zzzz qqqq")
    assert all(s.score == 0 for s in scores)
    assert [s.chunk_id for s in scores] == sorted(c.id for c in chunks)
    assert text_scores([], "anything") == []


# ---------- rendering ----------


def test_skeleton_text_keeps_signature_docstring_and_indent():
    source = b'class A:\n    @staticmethod\n    def f(x: int) -> int:\n        """Double it."""\n        return 2 * x\n'
    chunks = {c.id: c for c in PythonParser().parse_file("m.py", source)}
    text = skeleton_text(chunks["m.py::A.f"])
    assert (
        text
        == '    @staticmethod\n    def f(x: int) -> int:\n        """Double it."""\n        ...'
    )


def test_render_orders_files_and_chunks(sample):
    chunks, _ = sample
    text = render_variant(
        chunks,
        {"shop/utils/pricing.py::add_tax", "shop/cart.py::Cart", "shop/cart.py::Cart.total"},
        {"shop/cart.py::validate"},
    )
    assert text.index("# File: shop/cart.py") < text.index("# File: shop/utils/pricing.py")
    assert text.index("def validate") < text.index("class Cart:") < text.index("def total")
    assert "return product.price > 0" not in text  # validate is only a skeleton


# ---------- pool and budgets ----------


def test_tier_budgets():
    budgets = tier_budgets(10_000)
    assert budgets == {
        Tier.AGGRESSIVE: 1500,
        Tier.MODERATE: 3500,
        Tier.CONSERVATIVE: 6000,
        Tier.FULL: 10_000,
    }
    small = tier_budgets(2_000)
    assert small[Tier.AGGRESSIVE] == 1000  # floor
    assert small[Tier.MODERATE] == 1000  # never below the previous tier
    tiny = tier_budgets(500)
    assert all(b <= 500 for b in tiny.values())  # capped at full
    values = [tiny[t] for t in TIERS]
    assert values == sorted(values)


def test_pool_respects_cap_and_keeps_whole_files(sample):
    chunks, graph = sample
    scores = score_chunks(chunks, graph, QUERY)
    assert build_pool(chunks, scores, None) == chunks
    pool = build_pool(chunks, scores, 200)
    files = {c.file_path for c in pool}
    assert "shop/cart.py" in files  # the best-scored file comes first
    for f in files:  # whole files only
        assert {c.id for c in pool if c.file_path == f} == {
            c.id for c in chunks if c.file_path == f
        }
    assert sum(c.token_count for c in pool) <= 200
    assert build_pool(chunks, scores, 0) == []


# ---------- generate_variants ----------


def test_generate_variants_sample_repo(sample):
    chunks, graph = sample
    variants = generate_variants(chunks, graph, QUERY)
    check_variants(variants, chunks)
    # the full repo is small, so the 1,000-token floor makes aggressive cover nearly everything
    assert variants[-1].kept_chunk_ids == [c.id for c in chunks]


@pytest.mark.parametrize("aggressive_min", [0, 50, 200])
def test_generate_variants_small_budgets(sample, aggressive_min):
    chunks, graph = sample
    variants = generate_variants(chunks, graph, QUERY, aggressive_min_tokens=aggressive_min)
    check_variants(variants, chunks)
    moderate = variants[1]
    assert "shop/cart.py::Cart.total" in moderate.kept_chunk_ids


def test_mentioned_chunk_and_dependencies_kept_first(sample):
    chunks, graph = sample
    aggressive = generate_variants(chunks, graph, QUERY, aggressive_min_tokens=0)[0]
    assert "shop/cart.py::Cart.total" in aggressive.kept_chunk_ids
    assert "shop/cart.py::Cart" in aggressive.kept_chunk_ids
    assert "# File: shop/cart.py" in aggressive.rendered_text


def test_budget_smaller_than_any_chunk_gives_valid_variant(sample):
    chunks, graph = sample
    # cap the pool so every tier budget is tiny
    variants = generate_variants(chunks, graph, QUERY, full_cap_tokens=40, aggressive_min_tokens=0)
    check_variants(variants, chunks)


def test_empty_input_gives_four_empty_variants():
    variants = generate_variants([], CodeGraph(chunk_ids=[], edges=[]), "anything")
    assert [v.tier for v in variants] == TIERS
    assert all(v.token_count == 0 and v.rendered_text == "" for v in variants)


def test_generate_variants_deterministic(sample):
    chunks, graph = sample
    first = [
        v.model_dump() for v in generate_variants(chunks, graph, QUERY, aggressive_min_tokens=0)
    ]
    second = [
        v.model_dump() for v in generate_variants(chunks, graph, QUERY, aggressive_min_tokens=0)
    ]
    assert first == second


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_skeletons_appear_in_low_tiers(tmp_path):
    # One small mentioned function that calls several big documented helpers.
    helpers = []
    for i in range(6):
        body = "\n".join(f"    x{j} = {j} * value" for j in range(40))
        helpers.append(
            f'def helper_{i}(value: int) -> int:\n    """Helper number {i}."""\n{body}\n    return value\n'
        )
    caller = (
        "def entry_point(v):\n    return " + " + ".join(f"helper_{i}(v)" for i in range(6)) + "\n"
    )
    _write(tmp_path / "lib.py", "\n\n".join(helpers + [caller]))
    chunks = parse_repo(tmp_path)
    graph = build_graph(chunks)
    variants = generate_variants(
        chunks, graph, "entry_point returns the wrong number", aggressive_min_tokens=0
    )
    check_variants(variants, chunks)
    aggressive = variants[0]
    assert "lib.py::entry_point" in aggressive.kept_chunk_ids
    assert aggressive.skeleton_chunk_ids  # helpers that did not fit are summarised
    assert '"""Helper number' in aggressive.rendered_text
    for cid in aggressive.skeleton_chunk_ids:
        assert next(c for c in chunks if c.id == cid).kind != ChunkKind.MODULE


def test_large_generated_repo_invariants(tmp_path):
    for f in range(12):
        lines = [f"from pkg import mod{(f + 1) % 12}", ""]
        for i in range(8):
            lines.append(f"def func_{f}_{i}(a, b):")
            lines.append(
                f"    return mod{(f + 1) % 12}.func_{(f + 1) % 12}_{i}(a, b) + {i}"
                if i % 2
                else f"    return a * b + {i}"
            )
            lines.append("")
        lines.append(f"class Service{f}:")
        for i in range(4):
            lines.append(f"    def method_{i}(self):\n        return func_{f}_{i}(1, 2)\n")
        _write(tmp_path / f"pkg/mod{f}.py", "\n".join(lines))
    _write(tmp_path / "pkg/__init__.py", "")
    chunks = parse_repo(tmp_path)
    graph = build_graph(chunks)
    for query in ["Service3.method_2 is slow", "func_7_1 overflow", "nothing relevant here"]:
        variants = generate_variants(chunks, graph, query, aggressive_min_tokens=0)
        check_variants(variants, chunks)
        assert variants[0].token_count < variants[-1].token_count


def test_chunks_without_graph_entries_are_handled():
    chunk = Chunk(
        id="a.py::f",
        file_path="a.py",
        qualified_name="f",
        kind=ChunkKind.FUNCTION,
        start_line=1,
        end_line=2,
        code="def f():\n    return 1",
        signature="def f():",
        token_count=count_tokens("def f():\n    return 1"),
        file_hash="x",
    )
    variants = generate_variants([chunk], CodeGraph(chunk_ids=[], edges=[]), "f")
    check_variants(variants, [chunk])
    assert variants[0].kept_chunk_ids == ["a.py::f"]
