"""ODE phase 2: the 13 task numbers, from Claude Code-shaped requests (no model download).

The saved step D request bodies are not on this laptop, so the requests are built here in the
same shape as tests/test_gateway_adapters.py; a fake router stands in for the ONNX model.
"""

import math

import numpy as np
import pytest

from tokencut.gateway.adapters import anthropic
from tokencut.gateway.task_router import LABELS
from tokencut.ode import features as F
from tokencut.ode.bandit import HybridLinUCB


class FakeRouter:
    """Returns fixed probabilities; equal priors make the prior correction a no-op."""

    def __init__(self, **probs):
        self.p = np.array([probs.get(label, 0.0) for label in LABELS])
        self.p = self.p / self.p.sum()
        self.train_prior = self.natural_prior = np.full(len(LABELS), 1 / len(LABELS))
        self.seen: list[str] = []

    def probs(self, text):
        self.seen.append(text)
        return self.p


TRACEBACK = '''Traceback (most recent call last):
  File "shop/cart.py", line 12, in total
    return sum(i.price for i in self.items)
TypeError: unsupported operand type(s)'''


def _body(*user_turns):
    """Claude Code-shaped request: user / assistant turns, the newest user message last."""
    messages = []
    for i, text in enumerate(user_turns):
        if i:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": "Done."}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
    return {"model": "claude-sonnet-5-5", "max_tokens": 1024, "system": "You are Claude Code.",
            "tools": [{"name": "Read", "description": "Read a file", "input_schema": {}}],
            "messages": messages}


def test_thirteen_named_features_and_shared_indices():
    assert len(F.FEATURE_NAMES) == F.N_FEATURES == 13
    assert [F.FEATURE_NAMES[i] for i in F.SHARED] == [
        "p_bugfix", "p_feature", "p_refactor", "p_tests", "p_explain", "p_run_action",
        "context_len", "context_over_100k", "turns_before", "bias"]


def test_raw_features_by_hand():
    probs = {"bugfix": 0.85, "explain": 0.05, "tests": 0.05, "no_brief": 0.05}
    msg = f"Cart.total in shop/cart.py and tests/test_cart.py fails:\n{TRACEBACK}"
    x = F.raw_features(probs, msg, context_tokens=40_000, turns_before=4)
    assert list(x[:6]) == [0.85, 0.0, 0.0, 0.05, 0.05, 0.0]  # no_brief left out
    assert x[6] == pytest.approx(math.log1p(2))  # shop/cart.py, tests/test_cart.py (deduplicated)
    assert x[7] == 1.0
    assert x[8] > 0
    assert x[9] == pytest.approx(math.log1p(40_000))
    assert x[10] == 0.0
    assert x[11] == pytest.approx(math.log1p(4))
    assert x[12] == 1.0


def test_haiku_price_step_flag():
    assert F.raw_features({}, "x", 100_000, 0)[10] == 0.0
    assert F.raw_features({}, "x", 100_001, 0)[10] == 1.0


@pytest.mark.parametrize(("text", "count"), [
    ("look at shop/cart.py", 1),
    ("compare src/app.ts with README.md and src/app.ts", 2),
    (r"open C:\repo\pkg\main.go", 1),
    ("e.g. this, i.e. that, version 1.2, example.com", 0),
])
def test_files_named(text, count):
    assert F.files_named(text) == count


@pytest.mark.parametrize(("text", "found"), [
    (TRACEBACK, True),
    ("TypeError: x\n    at Object.<anonymous> (/app/index.js:10:5)", True),
    ("Exception\n\tat com.shop.Cart.total(Cart.java:12)", True),
    ("the total is wrong when the cart is empty", False),
])
def test_traceback_detection(text, found):
    assert F.has_traceback(text) is found


def test_request_features_use_the_newest_typed_message():
    body = _body("explain how the cart works", "now fix Cart.total in shop/cart.py\n" + TRACEBACK)
    router = FakeRouter(bugfix=0.9, no_brief=0.1)
    x, probs = F.request_features(anthropic.parse(body), router, context_tokens=12_000)
    assert router.seen[-1].startswith("now fix Cart.total")
    assert probs["bugfix"] == pytest.approx(0.9)
    assert x[0] == pytest.approx(0.9)
    assert x[7] == 1.0
    assert x[9] == pytest.approx(math.log1p(12_000))
    assert x[11] == pytest.approx(math.log1p(1))  # one earlier task message


def test_follow_ups_and_reminders_are_not_tasks():
    assert F.latest_task_message(anthropic.parse(_body("fix the bug in cart.py", "yes"))) is None
    msg = F.latest_task_message(anthropic.parse(_body(
        "fix the bug", "ok",
        "<system-reminder>internal note</system-reminder>add a discount feature")))
    assert msg.text == "add a discount feature"
    assert msg.turns_before == 1  # "ok" is a follow-up, not a task
    assert F.request_features(anthropic.parse(_body("thanks")), FakeRouter(bugfix=1.0)) is None


def test_context_size_falls_back_to_the_gateway_count():
    segments = anthropic.parse(_body("fix the bug in cart.py"))
    x, _ = F.request_features(segments, FakeRouter(bugfix=1.0))
    assert x[9] == pytest.approx(math.log1p(sum(s.token_count for s in segments)))


def test_decision_point_rule():
    assert F.is_decision_point({"bugfix": 0.9, "no_brief": 0.1}, threshold=0.87)
    assert not F.is_decision_point({"bugfix": 0.8, "no_brief": 0.2}, threshold=0.87)
    assert not F.is_decision_point({"bugfix": 0.05, "no_brief": 0.95}, threshold=0.87)


def test_standardizer_keeps_bias_and_survives_save_load():
    rng = np.random.default_rng(0)
    rows = np.column_stack([rng.random((50, 12)) * 10, np.ones(50)])
    rows[:, 10] = 0.0  # a feature that never varies
    s = F.Standardizer.fit(rows)
    z = np.array([s.transform(r) for r in rows])
    assert np.allclose(z[:, :10].mean(axis=0), 0, atol=1e-9)
    assert np.allclose(z[:, :10].std(axis=0), 1)
    assert np.all(z[:, 10] == 0.0)  # centred, not divided by zero
    assert np.all(z[:, 12] == 1.0)  # bias stays 1
    restored = F.Standardizer.from_state(s.to_state())
    assert np.array_equal(restored.transform(rows[3]), s.transform(rows[3]))
    assert np.array_equal(F.Standardizer().transform(rows[3]), rows[3])  # identity before Phase 5


def test_bad_input_is_refused():
    with pytest.raises(ValueError):
        F.raw_features({"bugfix": 1.5}, "x", 0, 0)
    with pytest.raises(ValueError):
        F.raw_features({}, "x", -1, 0)
    with pytest.raises(ValueError):
        F.Standardizer.fit([[0.0] * 13])
    with pytest.raises(ValueError):
        F.Standardizer().transform([0.0] * 12)
    state = F.Standardizer().to_state()
    state["feature_names"] = np.array(["a"] * 13)
    with pytest.raises(ValueError):
        F.Standardizer.from_state(state)


def test_features_plug_into_the_bandit():
    from tokencut.ode import arms as A

    bandit = HybridLinUCB(A.ARM_NAMES, A.arm_features(), n_features=F.N_FEATURES,
                          shared=F.SHARED)
    assert bandit.k == 20  # z = 10 shared features x 2 arm numbers
    x, _ = F.request_features(anthropic.parse(_body("fix the bug in shop/cart.py")),
                              FakeRouter(bugfix=1.0), context_tokens=5_000)
    dec = bandit.choose(F.Standardizer().transform(x), allowed=A.allowed_arms(5_000))
    assert dec.arm in A.ARM_NAMES
