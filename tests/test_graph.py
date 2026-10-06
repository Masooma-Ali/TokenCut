"""Member B — code graph: module map, edges, queries."""

from tokencut.contracts import EdgeKind
from tokencut.graph.builder import build_graph, dependency_closure
from tokencut.graph.modules import module_name_for
from tokencut.graph.queries import multi_source_distances, neighbours_within, to_networkx
from tokencut.ingestion.parser import parse_repo
from tokencut.ingestion.python_parser import PythonParser


def edge_set(graph):
    return {(e.source, e.kind, e.target, e.resolved) for e in graph.edges}


def edges_from(graph, source, kind=None):
    return {
        (e.target, e.resolved)
        for e in graph.edges
        if e.source == source and (kind is None or e.kind == kind)
    }


# ---------- sample repo ----------


def test_graph_matches_expected(expected_chunks, expected_graph):
    graph = build_graph(expected_chunks)
    assert graph.model_dump() == expected_graph.model_dump()


def test_key_edges_from_plan(expected_chunks):
    edges = edge_set(build_graph(expected_chunks))
    C, I, K, H = EdgeKind.CALLS, EdgeKind.IMPORTS, EdgeKind.CONTAINS, EdgeKind.INHERITS
    required = [
        ("shop/models.py::PremiumCustomer", H, "shop/models.py::Customer", True),
        ("shop/cart.py::Cart", K, "shop/cart.py::Cart.total", True),
        ("shop/models.py::Customer", K, "shop/models.py::Customer.discount_rate", True),
        ("shop/cart.py::<module>", I, "shop/models.py::<module>", True),
        ("shop/cart.py::<module>", I, "shop/utils/pricing.py::<module>", True),
        ("shop/payments/gateway.py::<module>", I, "shop/cart.py::<module>", True),
        ("shop/payments/gateway.py::<module>", I, "shop/payments/refunds.py::<module>", True),
        ("shop/payments/refunds.py::<module>", I, "shop/payments/gateway.py::<module>", True),
        ("shop/cart.py::Cart.add_item", C, "shop/cart.py::validate", True),
        ("shop/cart.py::Cart.total", C, "shop/cart.py::Cart.subtotal", True),
        ("shop/cart.py::Cart.total", C, "shop/utils/pricing.py::apply_discount", True),
        ("shop/cart.py::Cart.total", C, "shop/utils/pricing.py::add_tax", True),
        (
            "shop/payments/gateway.py::process_payment",
            C,
            "shop/payments/gateway.py::validate",
            True,
        ),
        (
            "shop/payments/gateway.py::process_payment",
            C,
            "shop/payments/refunds.py::log_failure",
            True,
        ),
        ("shop/payments/gateway.py::process_payment", C, "shop/cart.py::Cart.total", False),
        ("shop/payments/refunds.py::refund", C, "shop/payments/gateway.py::validate", True),
        ("shop/payments/refunds.py::refund", C, "shop/cart.py::Cart.total", False),
        (
            "shop/utils/broken.py::another_fine_function",
            C,
            "shop/utils/broken.py::fine_function",
            True,
        ),
    ]
    for edge in required:
        assert edge in edges, edge


def test_every_class_contains_its_methods(expected_chunks):
    graph = build_graph(expected_chunks)
    contains = {(e.source, e.target) for e in graph.edges if e.kind == EdgeKind.CONTAINS}
    methods = [c for c in expected_chunks if c.parent_id]
    assert contains == {(m.parent_id, m.id) for m in methods}


def test_forbidden_edges(expected_chunks):
    graph = build_graph(expected_chunks)
    ids = set(graph.chunk_ids)
    for e in graph.edges:
        assert e.source in ids and e.target in ids
        assert e.source != e.target
    # the two validate functions are never confused
    assert ("shop/cart.py::validate", False) not in edges_from(
        graph, "shop/payments/gateway.py::process_payment"
    )
    assert ("shop/cart.py::validate", True) not in edges_from(
        graph, "shop/payments/gateway.py::process_payment"
    )
    # self.customer.discount_rate(): name is not unique -> no edge
    targets = {t for t, _ in edges_from(graph, "shop/cart.py::Cart.total")}
    assert not any("discount_rate" in t for t in targets)


def test_graph_is_deterministic_and_sorted(expected_chunks):
    first = build_graph(expected_chunks).model_dump()
    second = build_graph(list(reversed(expected_chunks))).model_dump()
    assert first == second
    keys = [(e["source"], e["target"], e["kind"].value) for e in first["edges"]]
    assert keys == sorted(keys)


def test_dependency_closure_of_cart_total(expected_chunks):
    graph = build_graph(expected_chunks)
    closure = dependency_closure(graph, {"shop/cart.py::Cart.total"})
    for cid in [
        "shop/cart.py::Cart.total",
        "shop/cart.py::Cart.subtotal",
        "shop/utils/pricing.py::apply_discount",
        "shop/utils/pricing.py::add_tax",
        "shop/cart.py::Cart",
        "shop/cart.py::<module>",
    ]:
        assert cid in closure
    assert (
        "shop/payments/gateway.py::process_payment" not in closure
    )  # callers are not dependencies


def test_neighbour_distances(expected_chunks):
    graph = build_graph(expected_chunks)
    near = neighbours_within(graph, "shop/cart.py::Cart.total", 1)
    assert near["shop/cart.py::Cart.total"] == 0
    assert near["shop/payments/refunds.py::refund"] == 1  # undirected: callers count too
    assert all(d <= 1 for d in near.values())
    far = multi_source_distances(graph, ["shop/cart.py::Cart.total"])
    assert far["shop/payments/refunds.py::log_failure"] == 2
    assert "shop/utils/broken.py::fine_function" not in far  # unreachable


def test_to_networkx(expected_chunks):
    graph = build_graph(expected_chunks)
    g = to_networkx(graph)
    assert g.number_of_nodes() == len(graph.chunk_ids)
    assert g.number_of_edges() == len(graph.edges)


# ---------- module names and imports ----------


def test_module_names():
    assert module_name_for("shop/cart.py") == "shop.cart"
    assert module_name_for("shop/payments/__init__.py") == "shop.payments"
    assert module_name_for("setup.py") == "setup"


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_src_layout_and_relative_imports(tmp_path):
    write(tmp_path / "src/pkg/__init__.py", "from .core import run\n")
    write(
        tmp_path / "src/pkg/core.py",
        "from . import util\nfrom .util import helper as h\n\n"
        "def run():\n    h()\n    util.other()\n",
    )
    write(tmp_path / "src/pkg/util.py", "def helper():\n    pass\n\n\ndef other():\n    pass\n")
    write(
        tmp_path / "tests/test_core.py",
        "import pkg.core\nfrom pkg import core as c\n\ndef test_run():\n    c.run()\n",
    )
    graph = build_graph(parse_repo(tmp_path))
    edges = edge_set(graph)
    I, C = EdgeKind.IMPORTS, EdgeKind.CALLS
    assert ("src/pkg/__init__.py::<module>", I, "src/pkg/core.py::<module>", True) in edges
    assert ("src/pkg/core.py::<module>", I, "src/pkg/util.py::<module>", True) in edges
    assert ("src/pkg/core.py::run", C, "src/pkg/util.py::helper", True) in edges
    assert ("src/pkg/core.py::run", C, "src/pkg/util.py::other", True) in edges
    assert ("tests/test_core.py::<module>", I, "src/pkg/core.py::<module>", True) in edges
    assert ("tests/test_core.py::test_run", C, "src/pkg/core.py::run", True) in edges


def test_builtins_externals_and_ambiguous_names(tmp_path):
    write(
        tmp_path / "a.py",
        "import os\nfrom collections import OrderedDict\n\n"
        "def len_of(x):\n    print(len(x))\n    os.path.join('a')\n"
        "    OrderedDict()\n    x.save()\n",
    )
    write(tmp_path / "b.py", "class M:\n    def save(self):\n        pass\n")
    write(tmp_path / "c.py", "class N:\n    def save(self):\n        pass\n")
    graph = build_graph(parse_repo(tmp_path))
    assert (
        edges_from(graph, "a.py::len_of", EdgeKind.CALLS) == set()
    )  # builtins, externals, 2x save
    assert all(e.kind != EdgeKind.IMPORTS for e in graph.edges)  # os / collections are external


def test_class_calls_inheritance_super_and_self_in_base(tmp_path):
    write(
        tmp_path / "base.py",
        "class Base:\n    def setup(self):\n        pass\n\n    def run(self):\n        pass\n",
    )
    write(
        tmp_path / "child.py",
        "import base\nfrom base import Base\n\n\n"
        "class Child(Base):\n    def run(self):\n        super().run()\n"
        "        self.setup()\n\n\nclass Other(base.Base):\n    pass\n\n\n"
        "def make():\n    return Child()\n",
    )
    graph = build_graph(parse_repo(tmp_path))
    edges = edge_set(graph)
    assert ("child.py::Child", EdgeKind.INHERITS, "base.py::Base", True) in edges
    assert ("child.py::Other", EdgeKind.INHERITS, "base.py::Base", True) in edges
    assert ("child.py::Child.run", EdgeKind.CALLS, "base.py::Base.run", True) in edges
    assert ("child.py::Child.run", EdgeKind.CALLS, "base.py::Base.setup", True) in edges
    assert ("child.py::make", EdgeKind.CALLS, "child.py::Child", True) in edges


def test_circular_imports_terminate(tmp_path):
    write(tmp_path / "x.py", "import y\n\ndef fx():\n    y.fy()\n")
    write(tmp_path / "y.py", "import x\n\ndef fy():\n    x.fx()\n")
    edges = edge_set(build_graph(parse_repo(tmp_path)))
    assert ("x.py::<module>", EdgeKind.IMPORTS, "y.py::<module>", True) in edges
    assert ("y.py::<module>", EdgeKind.IMPORTS, "x.py::<module>", True) in edges
    assert ("x.py::fx", EdgeKind.CALLS, "y.py::fy", True) in edges


def test_empty_input():
    graph = build_graph([])
    assert graph.chunk_ids == [] and graph.edges == []


def test_graph_on_broken_file_only():
    chunks = PythonParser().parse_file("bad.py", b"def f(:\n    g(\n")
    graph = build_graph(chunks)
    assert set(graph.chunk_ids) == {c.id for c in chunks}
