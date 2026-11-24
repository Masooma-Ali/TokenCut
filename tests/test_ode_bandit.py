"""ODE phase 1: the hybrid LinUCB bandit on synthetic data (no gateway, no API calls).

Every test builds its own fake data with a fixed seed, so the right answer is known in
advance. See the ODE review doc (phase 1) for the scenario behind each test.
"""

import numpy as np
import pytest

from tokencut.ode.bandit import HybridLinUCB

ARMS3 = ["haiku", "sonnet", "opus"]
PRICE3 = {"haiku": 0.10, "sonnet": 2.0, "opus": 4.0}


def _bandit(arms=ARMS3, d=3, shared=(0, 1, 2), arm_features=None, **kw):
    if arm_features is None:
        arm_features = [[np.log(PRICE3[a])] for a in arms]
    return HybridLinUCB(arms, arm_features, n_features=d, shared=shared, **kw)


def _random_rows(rng, n, d, arms):
    xs = rng.normal(size=(n, d))
    picks = rng.integers(0, len(arms), size=n)
    rewards = rng.normal(size=n)
    return [(xs[i], arms[picks[i]], float(rewards[i])) for i in range(n)]


def _phi(bandit, x, arm):
    """The joint feature vector [z, 0 .. x (at this arm's slot) .. 0] of the full ridge problem."""
    d, k, n = bandit.n_features, bandit.k, len(bandit.arms)
    phi = np.zeros(k + n * d)
    phi[:k] = bandit.z(x, arm)
    i = bandit.arms.index(arm)
    phi[k + i * d:k + (i + 1) * d] = x
    return phi


# ---------- 1. the math matches a plain ridge regression ----------

def test_update_matches_joint_ridge_regression():
    rng = np.random.default_rng(1)
    bandit = _bandit(ridge=1.0)
    rows = _random_rows(rng, 60, 3, ARMS3)
    for x, arm, r in rows:
        bandit.update(x, arm, r)

    k, d, n = bandit.k, 3, len(ARMS3)
    big = np.eye(k + n * d)  # ridge = 1 on every weight
    rhs = np.zeros(k + n * d)
    for x, arm, r in rows:
        phi = _phi(bandit, x, arm)
        big += np.outer(phi, phi)
        rhs += r * phi
    w = np.linalg.solve(big, rhs)

    assert np.allclose(bandit.beta(), w[:k])
    for i, arm in enumerate(ARMS3):
        assert np.allclose(bandit.theta(arm), w[k + i * d:k + (i + 1) * d])

    # The uncertainty is the same quadratic form, phi' M^-1 phi, for every arm.
    x = rng.normal(size=3)
    scores = bandit.choose(x).scores
    for arm in ARMS3:
        phi = _phi(bandit, x, arm)
        expected = np.sqrt(phi @ np.linalg.solve(big, phi))
        assert scores[arm].bonus == pytest.approx(expected, rel=1e-9)
        assert scores[arm].estimate == pytest.approx(phi @ w, rel=1e-9, abs=1e-12)


# ---------- 2. with nothing shared, hybrid is plain disjoint LinUCB ----------

def _disjoint_scores(rows, x, arms, d, alpha):
    out = {}
    for arm in arms:
        a = np.eye(d)
        b = np.zeros(d)
        for xr, ar, r in rows:
            if ar == arm:
                a += np.outer(xr, xr)
                b += r * xr
        theta = np.linalg.solve(a, b)
        out[arm] = theta @ x + alpha * np.sqrt(x @ np.linalg.solve(a, x))
    return out


def test_zero_arm_features_reduce_to_disjoint_linucb():
    rng = np.random.default_rng(2)
    bandit = _bandit(arm_features=[[0.0], [0.0], [0.0]], alpha=0.7, explore=0.0)
    rows = _random_rows(rng, 40, 3, ARMS3)
    for x, arm, r in rows:
        bandit.update(x, arm, r)
    x = rng.normal(size=3)
    expected = _disjoint_scores(rows, x, ARMS3, 3, 0.7)
    scores = bandit.choose(x).scores
    for arm in ARMS3:
        assert scores[arm].total == pytest.approx(expected[arm], rel=1e-9)


# ---------- 3. it learns a known answer (the main test) ----------

TRUTH = {  # chance of solving, by task type
    "explain": {"haiku": 0.90, "sonnet": 0.90, "opus": 0.90},
    "bugfix": {"haiku": 0.30, "sonnet": 0.60, "opus": 0.85},
}
COST = {"haiku": 0.02, "sonnet": 0.30, "opus": 0.60}  # normalized cost c~
LAMBDA = 0.5
BEST = {"explain": "haiku", "bugfix": "opus"}  # 0.89 vs 0.60; 0.55 vs 0.45 vs 0.29


def _task_x(rng, kind):
    """[p_bugfix, p_explain, bias]: a confident but noisy router reading."""
    p = float(np.clip(rng.normal(0.9, 0.05), 0.6, 1.0))
    return np.array([p, 1 - p, 1.0]) if kind == "bugfix" else np.array([1 - p, p, 1.0])


def _expected_reward(kind, arm):
    return TRUTH[kind][arm] - LAMBDA * COST[arm]


def _outcome(world, kind, arm):
    return float(world.random() < TRUTH[kind][arm]) - LAMBDA * COST[arm]


def _play(rounds, seed, warm=0, explore=0.0):
    """Returns (task kind, greedy pick) per round and the greedy pick's regret per round.

    warm = practice tasks where every arm is run first (like the Phase 5 grid)."""
    world = np.random.default_rng(seed)
    bandit = _bandit(alpha=1.0, explore=explore, seed=seed)
    for _ in range(warm):
        kind = "bugfix" if world.random() < 0.5 else "explain"
        x = _task_x(world, kind)
        for arm in ARMS3:
            bandit.update(x, arm, _outcome(world, kind, arm))
    picks, regret = [], []
    for _ in range(rounds):
        kind = "bugfix" if world.random() < 0.5 else "explain"
        x = _task_x(world, kind)
        dec = bandit.choose(x)
        bandit.update(x, dec.arm, _outcome(world, kind, dec.arm))
        picks.append((kind, dec.greedy))
        regret.append(_expected_reward(kind, BEST[kind]) - _expected_reward(kind, dec.greedy))
    return picks, np.array(regret)


def _share_right(picks, kind):
    mine = [arm for k, arm in picks if k == kind]
    return sum(arm == BEST[kind] for arm in mine) / len(mine)


@pytest.mark.parametrize("seed", range(5))
def test_learns_cheap_model_for_explain_and_strong_model_for_bugfix(seed):
    """The planned setup: warm start on 20 practice tasks, then online with 5% random tries.

    On 30 seeds checked by hand, every one ends >= 88.6% right per task type, and the second
    half's regret is below 0.01 per task."""
    picks, regret = _play(2000, seed, warm=20, explore=0.05)
    for kind in ("explain", "bugfix"):
        assert _share_right(picks[-300:], kind) >= 0.85, kind
    assert regret[1000:].mean() < 0.01  # regret levels off


def test_cold_start_can_lock_out_an_arm_and_warm_start_prevents_it():
    """Known risk (and why the plan has a warm start and random tries): with no practice runs and
    no random tries, early bad luck can make LinUCB stop trying the right arm for good."""
    cold, _ = _play(2000, seed=4)
    assert _share_right(cold[-300:], "explain") < 0.5  # stuck on the wrong arm
    warm, _ = _play(2000, seed=4, warm=20, explore=0.05)
    assert _share_right(warm[-300:], "explain") > 0.9


# ---------- 4. the unsure bonus shrinks with experience ----------

def test_bonus_is_larger_for_an_untried_arm():
    rng = np.random.default_rng(4)
    bandit = _bandit(explore=0.0)
    for _ in range(200):
        bandit.update(rng.normal(size=3), "haiku", float(rng.random()))
    scores = bandit.choose(np.array([0.5, 0.5, 1.0])).scores
    assert scores["opus"].bonus > 3 * scores["haiku"].bonus


# ---------- 5. masked arms are never chosen ----------

def test_masked_arm_is_never_returned_even_when_exploring():
    bandit = _bandit(explore=0.5, seed=5)
    x = np.array([0.1, 0.9, 1.0])
    for _ in range(300):
        bandit.update(x, "haiku", 1.0)  # haiku is clearly the favourite
        bandit.update(x, "sonnet", 0.0)
        bandit.update(x, "opus", 0.0)
    assert bandit.choose(x).greedy == "haiku"
    for _ in range(1000):
        assert bandit.choose(x, allowed=["sonnet", "opus"]).arm != "haiku"


def test_empty_allowed_list_is_an_error():
    with pytest.raises(ValueError):
        _bandit().choose(np.ones(3), allowed=[])


# ---------- 6. and 7. the switch margin and the switch cost ----------

def _trained(rewards, **kw):
    """Arms with near-exact known estimates: x = [1], 1,000 identical rewards each, alpha = 0
    (estimate = r * 1000 / 1001)."""
    arms = list(rewards)
    bandit = HybridLinUCB(arms, [[0.0]] * len(arms), n_features=1, shared=(0,),
                          alpha=0.0, explore=0.0, **kw)
    for arm, r in rewards.items():
        for _ in range(1000):
            bandit.update(np.ones(1), arm, r)
    return bandit


def test_switch_margin_keeps_current_model_for_a_tiny_gain():
    stay = _trained({"sonnet": 0.600, "gpt": 0.610}, epsilon=0.02)
    assert stay.choose(np.ones(1)).arm == "gpt"  # no current model: plain best
    assert stay.choose(np.ones(1), current="sonnet").arm == "sonnet"  # +0.01 < 0.02
    move = _trained({"sonnet": 0.600, "gpt": 0.640}, epsilon=0.02)
    assert move.choose(np.ones(1), current="sonnet").arm == "gpt"  # +0.04 > 0.02


def test_switch_cost_counts_against_every_other_arm():
    bandit = _trained({"sonnet": 0.60, "haiku": 0.65}, epsilon=0.02)
    expensive = bandit.choose(np.ones(1), current="sonnet", switch_cost={"haiku": 0.15})
    assert expensive.arm == "sonnet"
    assert expensive.scores["haiku"].switch_cost == 0.15
    assert expensive.scores["sonnet"].switch_cost == 0.0  # staying is free
    free = bandit.choose(np.ones(1), current="sonnet", switch_cost={"haiku": 0.0})
    assert free.arm == "haiku"


def test_masked_current_model_does_not_block_a_switch():
    bandit = _trained({"sonnet": 0.60, "haiku": 0.55}, epsilon=0.02)
    assert bandit.choose(np.ones(1), allowed=["haiku"], current="sonnet").arm == "haiku"


# ---------- 8. random tries happen about 5% of the time and are logged honestly ----------

def test_exploration_rate_and_logged_probabilities():
    arms = ["a", "b", "c", "d"]
    bandit = HybridLinUCB(arms, [[0.0]] * 4, n_features=1, shared=(0,),
                          alpha=0.0, explore=0.05, seed=8)
    for _ in range(100):
        bandit.update(np.ones(1), "a", 1.0)
    n, explored = 10_000, 0
    for _ in range(n):
        dec = bandit.choose(np.ones(1))
        explored += dec.explored
        assert dec.greedy == "a"
        if dec.arm == "a":
            assert dec.probability == pytest.approx(0.95 + 0.05 / 4)
        else:
            assert dec.explored
            assert dec.probability == pytest.approx(0.05 / 4)
    assert 400 <= explored <= 600  # 5% of 10,000 = 500, binomial sd ~22


# ---------- 9. repeatable: same seed, same choices; save and load ----------

def test_same_seed_gives_the_same_choices():
    assert _play(300, seed=9)[0] == _play(300, seed=9)[0]


def test_save_then_load_gives_identical_scores():
    rng = np.random.default_rng(10)
    bandit = _bandit(alpha=0.5, epsilon=0.03, explore=0.05, seed=10)
    for x, arm, r in _random_rows(rng, 100, 3, ARMS3):
        bandit.update(x, arm, r)
    restored = HybridLinUCB.from_state(bandit.to_state())
    assert restored.arms == bandit.arms
    assert (restored.alpha, restored.epsilon, restored.explore) == (0.5, 0.03, 0.05)
    for _ in range(20):
        x = rng.normal(size=3)
        a, b = bandit.choose(x).scores, restored.choose(x).scores
        for arm in ARMS3:
            assert a[arm] == b[arm]


def test_state_holds_only_numpy_arrays():
    """np.savez can store it as-is (Phase 6 writes it to ~/.tokencut/ode_state.npz)."""
    state = _bandit().to_state()
    assert all(isinstance(v, np.ndarray) for v in state.values())


# ---------- 10. bad input fails loudly ----------

def test_wrong_x_length_is_refused():
    bandit = _bandit()
    with pytest.raises(ValueError, match="13|3"):
        bandit.choose(np.ones(4))
    with pytest.raises(ValueError):
        bandit.update(np.ones(2), "haiku", 1.0)


def test_unknown_arm_is_refused():
    bandit = _bandit()
    with pytest.raises(ValueError, match="haiku-standrad"):
        bandit.update(np.ones(3), "haiku-standrad", 1.0)
    with pytest.raises(ValueError):
        bandit.choose(np.ones(3), allowed=["gpt"])
    with pytest.raises(ValueError):
        bandit.choose(np.ones(3), current="gpt")


def test_non_finite_values_are_refused():
    bandit = _bandit()
    with pytest.raises(ValueError):
        bandit.update(np.ones(3), "haiku", float("nan"))
    with pytest.raises(ValueError):
        bandit.choose(np.array([1.0, np.inf, 1.0]))


def test_state_from_a_different_arm_list_is_refused():
    state = _bandit().to_state()
    other = _bandit(arms=["haiku", "sonnet", "gpt"],
                    arm_features=[[np.log(0.1)], [np.log(2.0)], [np.log(2.0)]])
    with pytest.raises(ValueError, match="arms"):
        other.load_state(state)


def test_bad_construction_is_refused():
    with pytest.raises(ValueError):
        HybridLinUCB(["a", "a"], [[0.0], [0.0]], n_features=1, shared=(0,))
    with pytest.raises(ValueError):
        HybridLinUCB(["a", "b"], [[0.0]], n_features=1, shared=(0,))
    with pytest.raises(ValueError):
        HybridLinUCB(["a"], [[0.0]], n_features=2, shared=(5,))
    with pytest.raises(ValueError):
        HybridLinUCB(["a"], [[0.0]], n_features=1, shared=(0,), explore=1.5)
