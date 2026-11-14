"""The attachment benchmark tool: sample set, offline run, scoring and report (no model call)."""

import json

from eval import attachments_bench as bench


def test_offline_run_shows_what_tokencut_does(tmp_path):
    items = bench.build_sample(tmp_path / "sample")
    rows = {r["id"]: r for r in bench.offline(items, ["claude-haiku-5-5"], tmp_path / "out")}
    assert "1920x1080 -> 1568x882" in rows["big_screenshot"]["what_tokencut_did"]
    assert rows["zoom_question"]["what_tokencut_did"] == "nothing (sent as is)"
    assert rows["small_screenshot"]["what_tokencut_did"] == "nothing (sent as is)"
    assert "border cut" in rows["chart_margins"]["what_tokencut_did"]
    assert "pages [1, 2, 5] as text, pages [3, 4] kept as PDF" in rows["pdf_text_question"]["what_tokencut_did"]
    assert rows["pdf_text_question"]["page_images_removed"] == 3
    assert rows["pdf_chart_question"]["what_tokencut_did"] == "nothing (sent as is)"
    assert rows["csv_middle_row"]["what_tokencut_did"] == "nothing (sent as is)"
    big = rows["big_screenshot"]
    assert big["attachment_tokens_before"] == 69 * 39 and big["attachment_tokens_after"] == 56 * 32
    assert (tmp_path / "out" / "files" / big["after_files"][0]).exists()


def test_report_is_written(tmp_path):
    items = bench.build_sample(tmp_path / "sample")
    bench.offline(items, ["gpt-6-luna"], tmp_path / "out")
    page = bench.report(tmp_path / "out").read_text(encoding="utf-8")
    assert "TokenCut attachment benchmark" in page and "<img" in page
    assert "big_screenshot" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")


def test_scoring():
    assert bench.score("The error code is 4,711.", "4711", "number")
    assert not bench.score("Error 4712", "4711", "number")
    assert bench.score("east", "East", "contains")


def test_sample_items_have_answers(tmp_path):
    items = bench.load_items(bench.build_sample(tmp_path / "sample"))
    assert len(items) == 7 and all(item["answer"] and item["question"] for item in items)
    assert json.loads(json.dumps(items)) == items


def test_benchmark_scoring_rules():
    assert bench.score_item("About 51.5 percent", ["53"], "relaxed")  # ChartQA: within 5 %
    assert not bench.score_item("About 40", ["53"], "relaxed")
    assert bench.score_item("It is Dallas, Texas.", ["dallas", "dallas texas"], "any")
    assert bench.score_item("Paris and Rome", ["Paris", "Rome"], "all")
    assert not bench.score_item("Paris only", ["Paris", "Rome"], "all")
    assert not bench.score_item("1450 units", ["45"], "any")  # numbers compare as numbers


def test_old_items_still_load(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text(json.dumps({"id": "x", "file": "f.png", "question": "q", "answer": "7",
                                "match": "number"}) + "\n", encoding="utf-8")
    item = bench.load_items(path)[0]
    assert item["answers"] == ["7"] and item["benchmark"] == "sample"
