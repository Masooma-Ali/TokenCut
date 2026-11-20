"""Offline prompt-router evaluation helpers: data cleaning, splits, metrics."""

import numpy as np

from eval.prompt_router.data import (
    _first_prompts,
    clean,
    extra_candidates,
    is_english,
    prompt_id,
    sample,
)
from eval.prompt_router.run import (
    cascade,
    cohen_kappa,
    confusion,
    final_labels,
    macro_f1,
    per_class,
    routed_scores,
    stratified_split,
)


def test_first_prompt_of_each_conversation_only():
    data = {"Sources": [{"ChatgptSharing": [
        {"Conversations": [{"Prompt": "fix the bug", "Answer": "x"}, {"Prompt": "thanks"}]},
        {"Conversations": [{"Prompt": "add a flag"}]},
    ]}]}
    found = []
    _first_prompts(data, found)
    assert found == ["fix the bug", "add a flag"]


def test_secrets_and_emails_are_masked_and_language_checked():
    text = clean("my key sk-abcdefghijklmnopqrstuvwx and mail me at a.b@example.com please")
    assert "sk-" not in text and "@" not in text and "<secret>" in text and "<email>" in text
    assert is_english("How do I fix this error in my code?")
    assert not is_english("如何修复这个错误")
    assert prompt_id("Fix  the bug") == prompt_id("fix the bug")


def test_sample_is_seeded_and_spread_over_sources():
    rows = [{"id": str(i), "text": "t", "source": "pr" if i % 4 else "issue"} for i in range(400)]
    a, b = sample(rows, 40), sample(rows, 40)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert sum(r["source"] == "issue" for r in a) == 10


def test_final_label_replaces_unsure_and_split_is_stratified():
    rows = [{"id": "1", "label": "unsure"}, {"id": "1", "label": "bugfix"}, {"id": "2", "label": "unsure"}]
    assert final_labels(rows) == {"1": "bugfix"}
    labels = {str(i): ("bugfix" if i < 50 else "feature") for i in range(100)}
    split = stratified_split(labels)
    assert sorted(i for ids in split.values() for i in ids) == sorted(labels)
    assert sum(labels[i] == "bugfix" for i in split["test"]) == 10
    assert stratified_split(labels) == split  # fixed seed


def test_metrics():
    true = ["bugfix", "bugfix", "feature", "explain"]
    pred = ["bugfix", "feature", "feature", "ambiguous"]
    labels = ["ambiguous", "bugfix", "explain", "feature"]
    stats = per_class(true, pred, labels)
    assert stats["bugfix"] == {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2}
    assert confusion(true, pred, labels)[1].tolist() == [0, 1, 0, 1]
    assert 0 < macro_f1(stats) < 1
    assert routed_scores(true, pred) == {"coverage": 0.75, "precision": 2 / 3}
    assert cohen_kappa(true, true) == 1.0


def test_cascade_sends_unclear_prompts_to_ambiguous():
    probs = np.array([[0.9, 0.1], [0.55, 0.45]])
    assert cascade(probs, ["bugfix", "feature"], 0.6, 0.0) == ["bugfix", "ambiguous"]
    assert cascade(probs, ["bugfix", "feature"], 0.0, 0.2) == ["bugfix", "ambiguous"]


def test_extra_candidates_skip_the_random_sample_and_never_repeat():
    rows = [{"id": str(i), "text": f"please add unit tests for f{i}", "source": "pr"} for i in range(30)]
    rows += [{"id": "r", "text": "review my code", "source": "pr"}]
    picked = extra_candidates(rows, exclude={"0", "1"}, per_label={"tests": 5, "review": 3})
    ids = [r["id"] for r in picked]
    assert len(ids) == len(set(ids)) == 6 and not {"0", "1"} & set(ids)
    assert {r["searched_for"] for r in picked} == {"tests", "review"}
    assert picked == extra_candidates(rows, exclude={"0", "1"}, per_label={"tests": 5, "review": 3})
