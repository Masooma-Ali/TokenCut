"""Hybrid LinUCB (Li, Chu, Langford, Schapire, WWW 2010, Algorithm 2) for choosing an arm per task.

Model: the expected reward of arm a for task x is

    z_{x,a} . beta  +  x . theta_a,        z_{x,a} = x[shared] (outer) w_a, flattened

beta is shared by all arms and learns lessons such as "aggressive trimming hurts bug fixes";
theta_a is arm a's own part; w_a are a few numbers describing the arm (trim level, log price).
Both are ridge regressions solved exactly from small matrices, so an update costs microseconds
and every choice can be explained (estimate + unsure bonus - switch cost).

Score at a decision:  total_a = estimate_a + alpha * sqrt(s_a) - switch_cost_a
  - switch_cost_a is supplied by the caller (the cache rewrite a model switch would cost, in
    reward units) and is 0 for the current arm;
  - the current arm is kept unless the best arm beats it by more than epsilon;
  - with probability `explore` a uniformly random allowed arm is tried instead, and the
    probability of the returned arm is logged (needed for off-policy evaluation later).

This module knows nothing about the gateway, prices or tasks: it takes numbers and arm names.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from tokencut.ode import settings

STATE_KIND = "hybrid_linucb"
STATE_VERSION = 1


@dataclass(frozen=True)
class ArmScore:
    estimate: float  # predicted reward
    bonus: float  # alpha * uncertainty: how unsure the bandit is about this arm here
    switch_cost: float  # subtracted when this arm is not the current one
    total: float  # estimate + bonus - switch_cost


@dataclass(frozen=True)
class Decision:
    arm: str  # the arm to use
    greedy: str  # the arm the scores (with the switch margin) pointed to
    explored: bool  # True when `arm` came from a random try
    probability: float  # chance this policy returns `arm` for this input
    scores: dict[str, ArmScore]  # every allowed arm, in arm order


class Bandit(ABC):
    """What every ODE bandit offers, so disjoint LinUCB or LinTS can be swapped in later."""

    arms: list[str]

    @abstractmethod
    def choose(self, x: Sequence[float], allowed: Iterable[str] | None = None,
               current: str | None = None,
               switch_cost: Mapping[str, float] | None = None) -> Decision: ...

    @abstractmethod
    def update(self, x: Sequence[float], arm: str, reward: float) -> None: ...

    @abstractmethod
    def to_state(self) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def load_state(self, state: Mapping[str, np.ndarray]) -> None: ...


class HybridLinUCB(Bandit):
    def __init__(self, arms: Sequence[str], arm_features: Sequence[Sequence[float]], *,
                 n_features: int, shared: Sequence[int], alpha: float = settings.ALPHA,
                 ridge: float = settings.RIDGE, epsilon: float = settings.SWITCH_MARGIN,
                 explore: float = 0.0,
                 seed: int | None = None):
        """
        arms: arm names, in a fixed order (ties go to the earlier arm).
        arm_features: one row per arm, w_a (e.g. [aggressive 0/1, log input price]).
        n_features: length d of every x.
        shared: indices of x that are combined with w_a into the shared features z.
        alpha: weight of the unsure bonus (exploration). ridge: prior strength on every weight.
        epsilon: switch margin. explore: share of random tries (0 to 1; 0 here, the live
        policy passes settings.EXPLORE). Defaults and their sources: tokencut/ode/settings.py.
        """
        self.arms = list(arms)
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError(f"arms must be non-empty and unique, got {self.arms}")
        w = np.asarray(arm_features, dtype=float)
        if w.ndim != 2 or w.shape[0] != len(self.arms):
            raise ValueError(f"arm_features needs one row per arm ({len(self.arms)}), "
                             f"got shape {w.shape}")
        if n_features < 1:
            raise ValueError("n_features must be at least 1")
        idx = np.asarray(shared, dtype=int)
        if idx.ndim != 1 or idx.size == 0 or idx.min() < 0 or idx.max() >= n_features \
                or len(set(idx.tolist())) != idx.size:
            raise ValueError(f"shared must be distinct indices of x (0..{n_features - 1}), "
                             f"got {list(shared)}")
        if not (alpha >= 0 and ridge > 0 and epsilon >= 0 and 0 <= explore <= 1):
            raise ValueError("need alpha >= 0, ridge > 0, epsilon >= 0 and 0 <= explore <= 1")
        if not np.all(np.isfinite(w)):
            raise ValueError("arm_features must be finite")

        self.arm_features = w
        self.n_features = int(n_features)
        self.shared = idx
        self.k = idx.size * w.shape[1]
        self.alpha, self.ridge = float(alpha), float(ridge)
        self.epsilon, self.explore = float(epsilon), float(explore)
        self._rng = np.random.default_rng(seed)

        d, k, n = self.n_features, self.k, len(self.arms)
        self.A0 = self.ridge * np.eye(k)
        self.b0 = np.zeros(k)
        self.A = np.stack([self.ridge * np.eye(d)] * n)
        self.B = np.zeros((n, d, k))
        self.b = np.zeros((n, d))
        self._refresh()

    # ---------- the model ----------

    def z(self, x: Sequence[float], arm: str) -> np.ndarray:
        """Shared features for (x, arm): x[shared] times w_a, flattened."""
        x = self._check_x(x)
        return np.outer(x[self.shared], self.arm_features[self._index(arm)]).ravel()

    def beta(self) -> np.ndarray:
        return self._beta.copy()

    def theta(self, arm: str) -> np.ndarray:
        i = self._index(arm)
        return self._Ainv[i] @ (self.b[i] - self.B[i] @ self._beta)

    def _refresh(self) -> None:
        """Recompute the inverses after a change (d and k are small: 13 and ~20)."""
        self._A0inv = np.linalg.inv(self.A0)
        self._beta = self._A0inv @ self.b0
        self._Ainv = np.linalg.inv(self.A)

    def _score(self, x: np.ndarray, i: int) -> tuple[float, float]:
        z = np.outer(x[self.shared], self.arm_features[i]).ravel()
        ainv, bi = self._Ainv[i], self.B[i]
        theta = ainv @ (self.b[i] - bi @ self._beta)
        estimate = float(z @ self._beta + x @ theta)
        # Li et al. 2010, Algorithm 2, line 12.
        a0z = self._A0inv @ z
        ax = ainv @ x
        s = (z @ a0z - 2 * (a0z @ (bi.T @ ax)) + x @ ax
             + (bi.T @ ax) @ self._A0inv @ (bi.T @ ax))
        return estimate, self.alpha * math.sqrt(max(float(s), 0.0))

    # ---------- deciding ----------

    def choose(self, x: Sequence[float], allowed: Iterable[str] | None = None,
               current: str | None = None,
               switch_cost: Mapping[str, float] | None = None) -> Decision:
        x = self._check_x(x)
        if allowed is None:
            names = self.arms
        else:
            wanted = set(allowed)
            unknown = wanted - set(self.arms)
            if unknown:
                raise ValueError(f"unknown arms in allowed: {sorted(unknown)}")
            names = [a for a in self.arms if a in wanted]
        if not names:
            raise ValueError("no arm is allowed")
        if current is not None:
            self._index(current)
        costs = dict(switch_cost or {})
        for arm, c in costs.items():
            self._index(arm)
            if not math.isfinite(c) or c < 0:
                raise ValueError(f"switch cost for {arm} must be finite and >= 0, got {c}")

        scores: dict[str, ArmScore] = {}
        for arm in names:
            estimate, bonus = self._score(x, self._index(arm))
            cost = 0.0 if arm == current else float(costs.get(arm, 0.0))
            scores[arm] = ArmScore(estimate, bonus, cost, estimate + bonus - cost)

        best = max(names, key=lambda a: scores[a].total)  # first in arm order on ties
        greedy = best
        if current in scores and scores[best].total - scores[current].total <= self.epsilon:
            greedy = current

        explored = self._rng.random() < self.explore
        arm = names[int(self._rng.integers(len(names)))] if explored else greedy
        share = self.explore / len(names)
        probability = share + (1 - self.explore) * (arm == greedy)
        return Decision(arm, greedy, bool(explored), float(probability), scores)

    # ---------- learning ----------

    def update(self, x: Sequence[float], arm: str, reward: float) -> None:
        """Add one finished task: the arm used and its reward (Li et al. 2010, Algorithm 2)."""
        x = self._check_x(x)
        i = self._index(arm)
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward}")
        z = np.outer(x[self.shared], self.arm_features[i]).ravel()
        ainv, bi = self._Ainv[i], self.B[i]
        self.A0 += bi.T @ ainv @ bi
        self.b0 += bi.T @ ainv @ self.b[i]
        self.A[i] += np.outer(x, x)
        self.B[i] += np.outer(x, z)
        self.b[i] += reward * x
        ainv = np.linalg.inv(self.A[i])
        self.A0 += np.outer(z, z) - self.B[i].T @ ainv @ self.B[i]
        self.b0 += reward * z - self.B[i].T @ ainv @ self.b[i]
        self._refresh()

    # ---------- saving ----------

    def to_state(self) -> dict[str, np.ndarray]:
        """Everything needed to rebuild this bandit, as numpy arrays (np.savez can store it).

        The random generator's position is not saved: a restored bandit makes the same scores,
        and its random tries restart from the seed."""
        return {
            "kind": np.array(STATE_KIND), "version": np.array(STATE_VERSION),
            "arms": np.array(self.arms), "arm_features": self.arm_features.copy(),
            "n_features": np.array(self.n_features), "shared": self.shared.copy(),
            "alpha": np.array(self.alpha), "ridge": np.array(self.ridge),
            "epsilon": np.array(self.epsilon), "explore": np.array(self.explore),
            "A0": self.A0.copy(), "b0": self.b0.copy(),
            "A": self.A.copy(), "B": self.B.copy(), "b": self.b.copy(),
        }

    def load_state(self, state: Mapping[str, np.ndarray]) -> None:
        """Load learned matrices; refused unless arms, arm features and x layout match."""
        if str(state.get("kind")) != STATE_KIND or int(state.get("version", -1)) != STATE_VERSION:
            raise ValueError(f"not a {STATE_KIND} v{STATE_VERSION} state")
        if [str(a) for a in state["arms"]] != self.arms:
            raise ValueError(f"state arms {list(state['arms'])} differ from {self.arms}")
        if (int(state["n_features"]) != self.n_features
                or not np.array_equal(state["shared"], self.shared)
                or not np.array_equal(state["arm_features"], self.arm_features)):
            raise ValueError("state features (n_features, shared or arm_features) differ")
        d, k, n = self.n_features, self.k, len(self.arms)
        shapes = {"A0": (k, k), "b0": (k,), "A": (n, d, d), "B": (n, d, k), "b": (n, d)}
        for key, shape in shapes.items():
            if np.shape(state[key]) != shape:
                raise ValueError(f"state {key} has shape {np.shape(state[key])}, need {shape}")
        self.A0 = np.array(state["A0"], dtype=float)
        self.b0 = np.array(state["b0"], dtype=float)
        self.A = np.array(state["A"], dtype=float)
        self.B = np.array(state["B"], dtype=float)
        self.b = np.array(state["b"], dtype=float)
        self._refresh()

    @classmethod
    def from_state(cls, state: Mapping[str, np.ndarray], seed: int | None = None) -> HybridLinUCB:
        bandit = cls([str(a) for a in state["arms"]], state["arm_features"],
                     n_features=int(state["n_features"]), shared=state["shared"].tolist(),
                     alpha=float(state["alpha"]), ridge=float(state["ridge"]),
                     epsilon=float(state["epsilon"]), explore=float(state["explore"]), seed=seed)
        bandit.load_state(state)
        return bandit

    # ---------- checks ----------

    def _index(self, arm: str) -> int:
        try:
            return self.arms.index(arm)
        except ValueError:
            raise ValueError(f"unknown arm {arm!r}; arms are {self.arms}") from None

    def _check_x(self, x: Sequence[float]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_features,):
            raise ValueError(f"x must have {self.n_features} numbers, got shape {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must be finite")
        return x
