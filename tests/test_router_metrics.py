"""Prompt-router evaluation values and the token head + tail cut (no torch needed)."""

import numpy as np

from eval.prompt_router.router_metrics import (
    LABELS,
    NO_BRIEF,
    bootstrap_ci,
    brief_curve,
    calibration,
    cut_token_ids,
    evaluate,
)


def test_short_inputs_are_kept_whole():
    assert cut_token_ids(list(range(100)), head=360, tail=149, marker=[7]) == list(range(100))


def test_long_inputs_keep_head_and_tail_with_a_marker():
    ids = list(range(1000))
    out = cut_token_ids(ids, head=360, tail=149, marker=[-1])
    assert len(out) == 360 + 1 + 149
    assert out[:360] == list(range(360)) and out[-149:] == list(range(851, 1000))
    assert out[360] == -1  # the "…" marker between head and tail


def _probs(pairs):
    """[(label, confidence)] -> probability rows (rest spread evenly)."""
    rows = []
    for label, conf in pairs:
        row = np.full(len(LABELS), (1 - conf) / (len(LABELS) - 1))
        row[LABELS.index(label)] = conf
        rows.append(row)
    return np.array(rows)


def test_brief_curve_counts_right_wrong_and_unwanted_briefs():
    true = ["bugfix", "bugfix", "feature", NO_BRIEF, NO_BRIEF]
    probs = _probs([("bugfix", 0.9), ("tests", 0.8), ("feature", 0.5), ("explain", 0.95),
                    (NO_BRIEF, 0.9)])
    point = next(p for p in brief_curve(probs, true, [0.7]) if p["min_prob"] == 0.7)
    # briefs at >= 0.7: bugfix (right), tests (wrong), explain on a no_brief prompt (unwanted)
    assert point["briefs"] == 3
    assert point["all_briefs_right"] == 1 / 3
    assert point["coverage"] == 2 / 3  # 2 of 3 real tasks got a brief
    assert point["no_brief_wrongly_briefed"] == 1 / 2


def test_evaluate_reports_per_label_values_and_confusion():
    true = ["bugfix", "feature", "tests", NO_BRIEF]
    probs = _probs([("bugfix", 0.9), ("bugfix", 0.6), ("tests", 0.7), (NO_BRIEF, 0.8)])
    report = evaluate(probs, true, meta=[{"text": "x" * 30, "pushback": "non_pushback",
                                          "agent": "Claude Code", "n_tokens": 10}] * 4)
    assert report["accuracy"] == 0.75
    assert report["per_label"]["bugfix"]["precision"] == 0.5
    assert report["per_label"]["feature"]["recall"] == 0.0
    assert report["confusion"]["feature"]["bugfix"] == 1
    assert 0 < report["macro_f1"] < 1
    assert "short (<= 20 chars)" in report["slices"] or "all" in report["slices"]


def test_calibration_bins_compare_confidence_with_correctness():
    probs = _probs([("bugfix", 0.95)] * 10)
    true = ["bugfix"] * 9 + ["feature"]
    bins = calibration(probs, true, n_bins=10)
    top = [b for b in bins if b["count"]][-1]
    assert top["count"] == 10 and abs(top["accuracy"] - 0.9) < 1e-9


def test_bootstrap_interval_contains_the_estimate_and_is_seeded():
    values = np.array([1] * 80 + [0] * 20)
    low, high = bootstrap_ci(values, np.mean, seed=1)
    assert low <= 0.8 <= high and high - low < 0.25
    assert bootstrap_ci(values, np.mean, seed=1) == (low, high)


def test_prior_correction_moves_probability_towards_the_real_label_mix():
    from eval.prompt_router.router_metrics import prior_correct
    probs = np.array([[0.5, 0.5]])
    out = prior_correct(probs, train_prior=np.array([0.5, 0.5]), real_prior=np.array([0.2, 0.8]))
    assert np.allclose(out.sum(axis=1), 1) and out[0, 1] > out[0, 0]
    same = prior_correct(probs, np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    assert np.allclose(same, probs)
