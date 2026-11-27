"""ODE's learned starting point (the prior): bandit state + feature scaling in one .npz file.

Written offline by eval/ode_learn.py (phase 5b) from graded practice turns; read by the gateway
(TOKENCUT_ODE=shadow / on). Holds numbers only: matrices, means, the reward settings used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokencut.ode.bandit import HybridLinUCB
from tokencut.ode.features import Standardizer


def save_prior(path: Path, bandit: HybridLinUCB, std: Standardizer, c_ref: float, lam: float,
               rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **bandit.to_state(), **std.to_state(), c_ref=np.array(c_ref),
             lam=np.array(lam), rows=np.array(rows))


def load_prior(path: Path, seed: int | None = None
               ) -> tuple[HybridLinUCB, Standardizer, float, float]:
    """(bandit, standardizer, C_ref, lambda). Raises if the file is missing or not a prior."""
    with np.load(path) as data:
        state = {k: data[k] for k in data.files}
    return (HybridLinUCB.from_state(state, seed=seed), Standardizer.from_state(state),
            float(state["c_ref"]), float(state["lam"]))


def prior_rows(path: Path) -> int:
    with np.load(path) as data:
        return int(data["rows"]) if "rows" in data.files else 0
