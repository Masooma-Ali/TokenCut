"""Evaluation helpers: gold data from patches, gold chunks, recall metrics (no network)."""

from eval.loader import gold_from_patch, select_task_ids
from eval.metrics import gold_chunk_ids, mean_by_tier, summary_markdown, task_rows, write_csv
from tokencut.contracts import ContextVariant, Tier

PATCH = """diff --git a/shop/cart.py b/shop/cart.py
--- a/shop/cart.py
+++ b/shop/cart.py
@@ -22,4 +22,4 @@ class Cart:

     def total(self) -> float:
-        discounted = apply_discount(self.subtotal(), self.customer.discount_rate())
+        discounted = apply_discount(self.subtotal(), self.customer.discount_rate() or 0)
         return add_tax(discounted)
diff --git a/shop/utils/pricing.py b/shop/utils/pricing.py
--- a/shop/utils/pricing.py
+++ b/shop/utils/pricing.py
@@ -2,2 +2,3 @@

 TAX_RATE = 0.17
+ROUNDING = 2
diff --git a/docs/notes.md b/docs/notes.md
--- a/docs/notes.md
+++ b/docs/notes.md
@@ -1 +1 @@
-old
+new
diff --git a/shop/new.py b/shop/new.py
new file mode 100644
--- /dev/null
+++ b/shop/new.py
@@ -0,0 +1 @@
+x = 1
"""


def test_gold_from_patch():
    files, lines = gold_from_patch(PATCH)
    assert files == ["shop/cart.py", "shop/utils/pricing.py"]  # no new files, no non-Python files
    assert lines["shop/cart.py"] == [24]  # removed line, original numbering
    assert lines["shop/utils/pricing.py"] == [2, 3]  # add-only hunk -> its source range


def test_gold_chunks_innermost(expected_chunks):
    gold = gold_chunk_ids(
        expected_chunks,
        {
            "shop/cart.py": [24, 11],  # inside Cart.total; the class header line
            "shop/utils/pricing.py": [3],  # module-level constant
            "missing.py": [1],  # file not parsed -> ignored
        },
    )
    assert gold == [
        "shop/cart.py::Cart",
        "shop/cart.py::Cart.total",
        "shop/utils/pricing.py::<module>",
    ]


def variant(tier, kept, skeletons=(), tokens=0):
    return ContextVariant(
        tier=tier,
        budget_tokens=1000,
        kept_chunk_ids=list(kept),
        skeleton_chunk_ids=list(skeletons),
        token_count=tokens,
        rendered_text="",
    )


def test_task_rows_recall(expected_chunks):
    gold_files = ["shop/cart.py", "shop/utils/pricing.py"]
    gold_lines = {"shop/cart.py": [24], "shop/utils/pricing.py": [7]}  # Cart.total, apply_discount
    all_ids = [c.id for c in expected_chunks]
    variants = [
        variant(
            Tier.AGGRESSIVE,
            ["shop/cart.py::Cart", "shop/cart.py::Cart.total"],
            ["shop/utils/pricing.py::apply_discount"],
            tokens=100,
        ),
        variant(Tier.MODERATE, ["shop/cart.py::Cart", "shop/cart.py::Cart.total"], tokens=200),
        variant(
            Tier.CONSERVATIVE,
            [
                "shop/cart.py::Cart",
                "shop/cart.py::Cart.total",
                "shop/utils/pricing.py::apply_discount",
            ],
            tokens=300,
        ),
        variant(Tier.FULL, all_ids, tokens=400),
    ]
    rows = task_rows(
        "t-1",
        "o/r",
        expected_chunks,
        variants,
        gold_files,
        gold_lines,
        {"parse": 1.0, "graph": 0.5, "variants": 0.25},
    )
    by_tier = {r["tier"]: r for r in rows}
    agg = by_tier["aggressive"]
    assert agg["reduction_vs_full"] == 0.75
    assert agg["gold_file_recall"] == 0.5
    assert agg["gold_chunk_recall"] == 0.5
    assert agg["gold_chunk_skeleton_recall"] == 1.0
    assert by_tier["conservative"]["gold_chunk_recall"] == 1.0
    assert by_tier["full"]["reduction_vs_full"] == 0.0
    assert all(r["pool_has_gold_file"] == 1 for r in rows)
    assert agg["repo_tokens"] == sum(c.token_count for c in expected_chunks)


def test_csv_and_summary(tmp_path, expected_chunks):
    variants = [
        variant(t, ["shop/cart.py::Cart.total"], tokens=10)
        for t in (Tier.AGGRESSIVE, Tier.MODERATE, Tier.CONSERVATIVE, Tier.FULL)
    ]
    rows = task_rows("t-1", "o/r", expected_chunks, variants, [], {}, {})
    assert rows[0]["gold_chunk_recall"] is None  # no gold data -> not a zero
    write_csv(rows, tmp_path / "out.csv")
    text = (tmp_path / "out.csv").read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("instance_id,repo,tier")
    assert len(text.splitlines()) == 5
    means = mean_by_tier(rows)
    assert means["aggressive"]["gold_chunk_recall"] is None
    assert "| aggressive |" in summary_markdown(rows)


def test_select_task_ids_deterministic():
    rows = (
        [{"repo": "psf/requests", "instance_id": f"psf__requests-{i}"} for i in range(3)]
        + [{"repo": "pallets/flask", "instance_id": "pallets__flask-1"}]
        + [
            {"repo": "pytest-dev/pytest", "instance_id": f"pytest-dev__pytest-{i}"}
            for i in range(10)
        ]
        + [{"repo": "django/django", "instance_id": f"django__django-{i}"} for i in range(10)]
        + [{"repo": "astropy/astropy", "instance_id": "astropy__astropy-1"}]
    )
    ids = select_task_ids(rows, count=10)
    assert len(ids) == 10
    assert ids == sorted(ids)
    assert (
        sum(i.startswith("pytest") for i in ids) == 3
        and sum(i.startswith("django") for i in ids) == 3
    )
    assert not any(i.startswith("astropy") for i in ids)
    assert select_task_ids(list(reversed(rows)), count=10) == ids
