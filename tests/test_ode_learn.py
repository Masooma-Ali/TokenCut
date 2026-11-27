"""ODE phase 5b: learning the prior from saved turns (no model calls)."""

import csv
import json
import random

import numpy as np
import pytest

from eval import ode_learn as L
from eval import ode_run as R
from tokencut.ode.features import N_FEATURES

ARMS3 = ("haiku-aggressive", "sonnet-aggressive", "gpt-aggressive")


def x_for(kind: str, ctx: float = 9.0) -> tuple[float, ...]:
    """Raw x: router probabilities by turn type, then the other numbers."""
    p = {"explain": [0.1, 0, 0, 0, 0.9, 0], "bugfix": [0.9, 0, 0, 0, 0.1, 0],
         "tests": [0.1, 0, 0, 0.9, 0, 0]}[kind]
    return (*p, 0.0, 0.0, 5.0, ctx, 0.0, 1.0, 1.0)


def turn(task, arm, n, kind, solved, cost, ctx=9.0):
    return L.Turn(task, arm, n, kind, solved, cost, x_for(kind, ctx), kind)


def world(n_tasks=12, seed=0):
    """Haiku solves explain and tests cheaply but fails bugfix; Sonnet solves everything at 5x
    the cost; GPT is like Sonnet but fails tests. Right answer: haiku for explain and tests,
    sonnet for bugfix."""
    rng = random.Random(seed)
    out = []
    for i in range(n_tasks):
        task = f"t{i:02d}"
        for n, kind in enumerate(("explain", "bugfix", "tests"), 1):
            ctx = 9.0 + rng.random()
            out += [turn(task, "haiku-aggressive", n, kind, kind != "bugfix", 0.02, ctx),
                    turn(task, "sonnet-aggressive", n, kind, True, 0.10, ctx),
                    turn(task, "gpt-aggressive", n, kind, kind != "tests", 0.10, ctx)]
    return out


def test_load_turns_keeps_only_graded_rows(tmp_path):
    path = tmp_path / "turns.csv"
    x = json.dumps([0.5] * N_FEATURES)
    rows = [
        {"task": "b", "arm": "haiku-standard", "turn": 1, "task_type": "explain",
         "status": "completed", "solved": 1, "cost_usd": 0.01, "x": x, "router_task": "bugfix"},
        {"task": "a", "arm": "opus-standard", "turn": 2, "task_type": "bugfix",
         "status": "completed", "solved": 0, "cost_usd": 0.3, "x": x, "router_task": "refactor"},
        {"task": "a", "arm": "gpt-standard", "turn": 3, "task_type": "tests",
         "status": "timeout", "solved": "", "cost_usd": 0.2, "x": x, "router_task": "tests"},
        {"task": "a", "arm": "gpt-standard", "turn": 1, "task_type": "explain",
         "status": "completed", "solved": 1, "cost_usd": 0.1, "x": "", "router_task": "bugfix"},
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=R.TURN_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    got = L.load_turns(path)
    assert [(t.task, t.arm, t.turn, t.solved) for t in got] == [
        ("a", "opus-standard", 2, False), ("b", "haiku-standard", 1, True)]
    assert got[0].cost == 0.3 and len(got[0].x) == N_FEATURES


def test_c_ref_estimates_use_opus_p95_and_sonnet_cross_check():
    turns = [turn("t", "opus-standard", 1, "explain", True, c / 10) for c in range(1, 21)]
    turns += [turn("t", "sonnet-standard", 1, "explain", True, 0.1)] * 5
    est = L.c_ref_estimates(turns)
    assert est["opus_p95"] == pytest.approx(np.percentile([c / 10 for c in range(1, 21)], 95))
    assert est["opus_sonnet_price_ratio"] == 2.0
    assert est["sonnet_p95_x_ratio"] == pytest.approx(0.2)


def test_prior_does_not_depend_on_row_order():
    turns = world()
    a, _ = L.build_prior(turns, 0.38, 0.4)
    shuffled = turns[:]
    random.Random(3).shuffle(shuffled)
    b, _ = L.build_prior(shuffled, 0.38, 0.4)
    for key, value in a.to_state().items():
        if value.dtype.kind == "f":
            assert np.allclose(value, b.to_state()[key]), key


def test_prior_learns_the_cheapest_arm_that_solves_each_turn_type():
    turns = world()
    bandit, std = L.build_prior(turns, 0.38, 0.4, alpha=0.0)
    picks = {k: bandit.choose(std.transform(x_for(k)), allowed=ARMS3).arm
             for k in ("explain", "bugfix", "tests")}
    assert picks == {"explain": "haiku-aggressive", "bugfix": "sonnet-aggressive",
                     "tests": "haiku-aggressive"}


def test_leave_one_task_out_never_trains_on_the_held_out_task(monkeypatch):
    turns = world(n_tasks=4)
    seen = []
    real = L.build_prior

    def spy(train, *a, **k):
        seen.append({t.task for t in train})
        return real(train, *a, **k)

    monkeypatch.setattr(L, "build_prior", spy)
    out = L.leave_one_task_out(turns, 0.38, 0.4, arms=ARMS3)
    assert out["tasks"] == ["t00", "t01", "t02", "t03"]
    assert seen == [{"t01", "t02", "t03"}, {"t00", "t02", "t03"}, {"t00", "t01", "t03"},
                    {"t00", "t01", "t02"}]


def test_leave_one_task_out_matches_the_oracle_on_a_clear_world():
    turns = world()
    out = L.leave_one_task_out(turns, 0.38, 0.4, arms=ARMS3)
    ode = L.summarize(out["per_turn"], out["grid"], 0.38, 0.4)
    base = L.baselines(out["grid"], 0.38, 0.4, arms=ARMS3)
    assert ode == {**base["best per turn (oracle)"], "arms": ode["arms"]}
    assert ode["mean_reward"] > base["always sonnet-aggressive"]["mean_reward"]
    assert ode["mean_reward"] > base["always haiku-aggressive"]["mean_reward"]


def test_per_session_policy_keeps_the_turn_one_choice():
    out = L.leave_one_task_out(world(), 0.38, 0.4, arms=ARMS3)
    first = {}
    for task, n, arm in out["per_session"]:
        first.setdefault(task, arm)
        assert arm == first[task]
    assert {arm for _, _, arm in out["per_session"]} == {"haiku-aggressive"}


def test_cells_need_every_compared_arm():
    turns = world(n_tasks=2)
    turns = [t for t in turns if not (t.task == "t01" and t.turn == 3 and t.arm == "gpt-aggressive")]
    assert ("t01", 3) not in L.cells(turns, ARMS3)
    assert len(L.cells(turns, ARMS3)) == 5


def test_prior_file_round_trip(tmp_path):
    turns = world()
    bandit, std = L.build_prior(turns, 0.38, 0.4)
    path = tmp_path / "ode_prior.npz"
    L.save_prior(path, bandit, std, 0.38, 0.4, len(turns))
    b2, s2, c_ref, lam = L.load_prior(path)
    x = std.transform(x_for("bugfix"))
    assert (c_ref, lam) == (0.38, 0.4)
    assert np.allclose(s2.transform(x_for("bugfix")), x)
    assert b2.choose(x).arm == bandit.choose(x).arm
    assert np.allclose(b2.beta(), bandit.beta())


def test_retraining_refreshes_the_packaged_prior_only_from_the_default_folder(tmp_path):
    results, packaged = tmp_path / "results", tmp_path / "pkg" / "ode_prior.npz"
    bandit, std = L.build_prior(world(), 0.38, 0.4)
    assert L.write_priors(results, bandit, std, 0.38, 0.4, 36, results, packaged) == [
        results / "ode_prior.npz", packaged]
    assert packaged.read_bytes() == (results / "ode_prior.npz").read_bytes()
    packaged.unlink()
    other = tmp_path / "experiment"
    assert L.write_priors(other, bandit, std, 0.38, 0.4, 36, results, packaged) == [
        other / "ode_prior.npz"]
    assert not packaged.exists()
