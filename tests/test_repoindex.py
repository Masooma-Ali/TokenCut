"""P1: repo index (symbol map, hybrid search, exact symbols) and its MCP server."""

import io
import json
import shutil

import pytest

from tokencut.repoindex.bm25 import Bm25Index
from tokencut.repoindex.index import RepoIndex, breadcrumb, is_test_file
from tokencut.repoindex.mcp import TOOLS, handle, serve
from tokencut.tokens import count_tokens


@pytest.fixture
def repo(tmp_path, sample_repo_path):
    target = tmp_path / "repo"
    shutil.copytree(sample_repo_path, target)
    return target


@pytest.fixture
def index(repo, tmp_path):
    idx = RepoIndex.open(repo, tmp_path / "cache")
    idx.graph(wait=True)
    return idx


# ---------- BM25 ----------


def test_bm25_incremental_equals_fresh():
    a = Bm25Index()
    a.add("x", ["cart", "total"])
    a.add("y", ["tax", "amount"])
    a.add("z", ["cart", "discount"])
    a.remove("y")
    a.add("x", ["cart", "price"])  # replaced
    b = Bm25Index()
    b.add("x", ["cart", "price"])
    b.add("z", ["cart", "discount"])
    assert a.scores(["cart", "discount"]) == b.scores(["cart", "discount"])
    assert "tax" not in a.postings and len(a) == 2


# ---------- index ----------


def test_open_parses_the_working_tree(index):
    assert "shop/cart.py" in index.files
    assert "shop/cart.py::Cart.total" in index.by_id


def test_refresh_reparses_only_changed_files(repo, index, tmp_path):
    assert index.refresh() == []
    cart = repo / "shop" / "cart.py"
    cart.write_text(cart.read_text() + "\n\ndef checkout(cart):\n    return cart.total()\n")
    (repo / "shop" / "utils" / "broken.py").unlink()
    assert index.refresh() == ["shop/cart.py", "shop/utils/broken.py"]
    assert "shop/cart.py::checkout" in index.by_id
    assert not any(cid.startswith("shop/utils/broken.py") for cid in index.by_id)
    assert index.search("checkout cart", 1)[0].chunk.id == "shop/cart.py::checkout"
    index.save()
    reopened = RepoIndex.open(repo, tmp_path / "cache")  # from the cache: nothing re-parsed
    assert reopened.refresh() == [] and reopened.by_id.keys() == index.by_id.keys()


def test_touching_a_file_without_changing_it_reparses_nothing(repo, index):
    # Same bytes back (write_text would turn \n into \r\n on Windows: a real change).
    cart = repo / "shop" / "cart.py"
    cart.write_bytes(cart.read_bytes())
    assert index.refresh() == []


def test_graph_is_rebuilt_in_the_background_after_a_change(repo, index):
    old = index.graph()
    (repo / "shop" / "extra.py").write_text("def extra():\n    return 1\n")
    index.refresh()
    assert index.graph() is old  # the previous graph answers while a new one is built
    assert "shop/extra.py::extra" in index.graph(wait=True).chunk_ids


def test_search_finds_the_function_the_issue_names(index):
    hits = index.search("Cart.total returns the wrong amount when a discount applies", 3)
    assert hits[0].chunk.id == "shop/cart.py::Cart.total"
    assert hits == sorted(hits, key=lambda h: (-h.score, h.chunk.id))


def test_search_is_deterministic(index):
    q = "refund fails for the card"
    assert [h.chunk.id for h in index.search(q, 5)] == [h.chunk.id for h in index.search(q, 5)]


def test_find_code_shows_code_within_budget(index):
    out = index.find_code("apply_discount rate", k=5, max_tokens=300)
    assert "shop/utils/pricing.py › apply_discount" in out
    assert "→def apply_discount" in out  # numbered like the agent's Read tool
    assert count_tokens(out) <= 300 + 60  # code stops at the budget, headers may follow


def test_repo_map_lists_directories_and_signatures(index):
    out = index.repo_map()
    assert "# 9 Python files" in out or "Python files" in out
    assert "shop/payments/  (" in out
    assert "class Cart: __init__, add_item, subtotal, total" in out
    assert "def apply_discount(amount: float, rate: float) -> float" in out
    zoom = index.repo_map("shop/payments")
    assert "shop/cart.py" not in zoom and "def refund(" in zoom
    assert "No Python files" in index.repo_map("nope")


def test_repo_map_respects_the_token_limit(index):
    out = index.repo_map(max_tokens=200)
    assert count_tokens(out) <= 200 + 30
    assert "more files not shown" in out


def test_symbol_returns_exact_source_and_relations(index):
    out = index.symbol("Cart.total")
    assert out.startswith("shop/cart.py › Cart › total  (lines 23-")
    assert "    23→    def total(self) -> float:" in out
    assert "calls / inherits:" in out
    assert "No symbol" in index.symbol("does_not_exist")


def test_helpers():
    assert is_test_file("tests/test_x.py") and is_test_file("pkg/conftest.py")
    assert not is_test_file("pkg/testcases_like.py")
    idx = RepoIndex()
    assert idx.find_code("anything") == "No matching code found."
    from tokencut.contracts import Chunk, ChunkKind
    c = Chunk(id="a.py::<module>", file_path="a.py", qualified_name="<module>",
              kind=ChunkKind.MODULE, start_line=1, end_line=1, code="x = 1", token_count=3,
              file_hash="h")
    assert breadcrumb(c) == "a.py › (module level)"


# ---------- MCP server ----------


def rpc(index, method, params=None, msg_id=1):
    return handle(index, {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}})


def test_mcp_handshake_and_tools(index):
    init = rpc(index, "initialize", {"protocolVersion": "2025-06-18"})["result"]
    assert init["protocolVersion"] == "2025-06-18" and "tools" in init["capabilities"]
    assert handle(index, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = [t["name"] for t in rpc(index, "tools/list")["result"]["tools"]]
    assert names == ["repo_map", "find_code", "symbol"]
    assert all("inputSchema" in t for t in TOOLS)


def test_mcp_tool_calls(index):
    found = rpc(index, "tools/call", {"name": "find_code", "arguments": {"query": "add_tax"}})
    assert found["result"]["isError"] is False
    assert "add_tax" in found["result"]["content"][0]["text"]
    missing = rpc(index, "tools/call", {"name": "find_code", "arguments": {}})
    assert missing["result"]["isError"] is True
    assert rpc(index, "nope")["error"]["code"] == -32601


def test_mcp_sees_uncommitted_edits(repo, index):
    (repo / "shop" / "late.py").write_text("def brand_new_helper():\n    return 2\n")
    out = rpc(index, "tools/call", {"name": "symbol", "arguments": {"name": "brand_new_helper"}})
    assert "brand_new_helper" in out["result"]["content"][0]["text"]


def test_mcp_stdio_loop(index):
    lines = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        "not json",
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "repo_map", "arguments": {"max_tokens": 500}}}),
    ]) + "\n"
    out = io.StringIO()
    serve(index, io.StringIO(lines), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r.get("id") for r in replies] == [1, None, 2]
    assert replies[1]["error"]["code"] == -32700
    assert "class Cart" in replies[2]["result"]["content"][0]["text"]
