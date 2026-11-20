"""Prompt router: rules, classifiers on frozen embeddings, confidence cascade, save / load."""

import numpy as np

from tokencut.gateway.prompt_router import (
    AMBIGUOUS,
    KNN,
    LABELS,
    STAGE1,
    PromptRouter,
    Prototypes,
    SoftmaxHead,
    TwoStage,
    rule_label,
)

# Fake embedding space: each label has its own direction, examples are noisy copies.
DIM = 16
RNG = np.random.default_rng(0)
DIRECTIONS = {label: RNG.normal(size=DIM) for label in LABELS}


def fake_examples(n_per_label: int = 12, noise: float = 0.3):
    rng = np.random.default_rng(1)
    xs, ys = [], []
    for label in LABELS:
        if label == AMBIGUOUS:
            continue  # the fallback is never a trained class
        for _ in range(n_per_label):
            xs.append(DIRECTIONS[label] + noise * rng.normal(size=DIM))
            ys.append(label)
    return np.array(xs, dtype=np.float32), ys


def test_every_label_has_a_stage1_group():
    assert set(STAGE1) == set(LABELS)
    assert STAGE1["bugfix"] == STAGE1["feature"] == "change_code"
    assert STAGE1["explain"] == STAGE1["review"] == "understand_code"


def test_rules_catch_followups_and_empty_text():
    for text in ("yes", "ok go ahead", "Continue.", "do it", "looks good", "option 2", "y"):
        assert rule_label(text) == "followup", text
    assert rule_label("   ") == AMBIGUOUS
    assert rule_label("yes, but also rename the helper and add tests for it") is None
    assert rule_label("fix the failing test in cart.py") is None


def test_classifiers_learn_separable_labels():
    x, y = fake_examples()
    for model in (Prototypes(), KNN(k=3), SoftmaxHead(epochs=300), TwoStage(SoftmaxHead)):
        model.fit(x, y)
        probs = model.predict_proba(x)
        assert probs.shape == (len(y), len(model.labels))
        assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
        predicted = [model.labels[i] for i in probs.argmax(axis=1)]
        accuracy = np.mean([p == t for p, t in zip(predicted, y)])
        assert accuracy > 0.9, (type(model).__name__, accuracy)


def test_low_confidence_goes_to_ambiguous():
    x, y = fake_examples()
    head = SoftmaxHead(epochs=300).fit(x, y)
    vectors = {"clear bug": DIRECTIONS["bugfix"], "mixed": (DIRECTIONS["bugfix"] + DIRECTIONS["design"]) / 2}
    router = PromptRouter(head, embed=lambda texts: np.array([vectors[t] for t in texts]),
                          min_prob=0.7, min_margin=0.2)
    assert router.classify("clear bug")[0] == "bugfix"
    label, confidence = router.classify("mixed")
    assert label == AMBIGUOUS and confidence < 0.7 + 1e-9
    assert router.classify("yes")[0] == "followup"  # rules first, no embedding needed


def test_router_is_deterministic_and_round_trips(tmp_path):
    x, y = fake_examples()
    head = SoftmaxHead(epochs=200).fit(x, y)
    path = tmp_path / "router.npz"
    head.save(path)
    loaded = SoftmaxHead.load(path)
    assert loaded.labels == head.labels
    assert np.array_equal(loaded.predict_proba(x), head.predict_proba(x))
    assert np.array_equal(head.predict_proba(x), head.predict_proba(x))
