"""Setup checks — if these pass on your laptop, your environment is ready."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tokencut.contracts import TIER_BUDGET_FRACTION, Chunk, ChunkKind, Tier

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_repo"


def test_tree_sitter_parses_python():
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    parser = Parser(Language(tspython.language()))
    tree = parser.parse(b"def hello():\n    pass\n")
    assert tree.root_node.children[0].type == "function_definition"


def test_tree_sitter_survives_syntax_error():
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    parser = Parser(Language(tspython.language()))
    source = (FIXTURE / "shop/utils/broken.py").read_bytes()
    tree = parser.parse(source)
    assert tree.root_node.has_error  # error is detected...
    funcs = [n for n in tree.root_node.children if n.type == "function_definition"]
    assert len(funcs) >= 1  # ...but valid functions are still recovered


def test_token_counter():
    from tokencut.tokens import count_tokens

    assert count_tokens("") == 0
    assert count_tokens("def hello(): pass") > 0


def test_chunk_contract_valid():
    chunk = Chunk(
        id="shop/cart.py::Cart.total",
        file_path="shop/cart.py",
        qualified_name="Cart.total",
        kind=ChunkKind.METHOD,
        start_line=22,
        end_line=24,
        code="def total(self) -> float: ...",
        signature="def total(self) -> float:",
        token_count=10,
        file_hash="abc",
    )
    assert chunk.kind == ChunkKind.METHOD


def test_chunk_contract_rejects_bad_line_numbers():
    with pytest.raises(ValidationError):
        Chunk(
            id="x::y", file_path="x", qualified_name="y", kind=ChunkKind.FUNCTION,
            start_line=0, end_line=1, code="", token_count=0, file_hash="",
        )


def test_tier_budgets_increase():
    order = [Tier.AGGRESSIVE, Tier.MODERATE, Tier.CONSERVATIVE, Tier.FULL]
    fractions = [TIER_BUDGET_FRACTION[t] for t in order]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0


def test_fixture_repo_present():
    assert (FIXTURE / "shop/cart.py").exists()
    assert (FIXTURE / "shop/utils/broken.py").exists()
