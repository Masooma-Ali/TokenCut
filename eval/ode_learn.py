"""ODE phase 5b: learn the starting bandit (prior) from the saved practice turns (no model calls).

    python -m eval.ode_learn                       # prior + report with the Phase 0 values
    python -m eval.ode_learn --lam 0.4 --c-ref 0.38

Steps:
  1. Read eval/results/ode/turns.csv: one row per graded turn (task, arm, solved, cost, raw x).
     Turns without a success signal (timeout, agent error, ungraded) are left out: no reward,
     no update (reward.py).
  2. C_ref = the 95th percentile of Opus turn costs (settings.C_REF_PERCENTILE). Cross-check:
     Sonnet's 95th percentile x (Opus price / Sonnet price), since Opus costs exactly 2x Sonnet
     for every token kind.
  3. reward = solved - lambda * min(1, cost / C_ref) per turn (reward.py).
  4. Standardize x (features.Standardizer, fitted on all rows), then HybridLinUCB.update() for
     every row. The hybrid LinUCB sums do not depend on the row order.
  5. Save eval/results/ode/ode_prior.npz (bandit state + standardizer + C_ref + lambda).
  6. Check, still offline: leave-one-task-out (train on the other tasks, choose greedily for the
     held-out task, look up what the chosen arm really scored there, since every arm ran every
     task), against fixed rules ("always <arm>") and the best possible choice; a lambda sweep;
     C_ref sensitivity.

The evaluation compares arms that ran on every task (Opus ran only in the pilot). Two policies:
"per turn" (a choice at every typed message; switch costs ignored, since the data has no
mid-session switches) and "per session" (the turn-1 choice kept for the whole session).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tokencut.ode import settings
from tokencut.ode.arms import ARM_NAMES, OPUS, SONNET, arm_features
from tokencut.ode.bandit import HybridLinUCB
from tokencut.ode.features import N_FEATURES, SHARED, Standardizer
from tokencut.ode.reward import reward

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results" / "ode"
EVAL_ARMS = ("haiku-standard", "haiku-aggressive", "gpt-standard", "gpt-aggressive",
             "sonnet-standard", "sonnet-aggressive")
LAMBDAS = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0)


@dataclass(frozen=True)
class Turn:
    task: str
    arm: str
    turn: int
    task_type: str
    solved: bool
    cost: float
    x: tuple[float, ...]  # raw (not standardized)
    router_task: str


# ---------- 1. the saved turns ----------

def load_turns(path: Path) -> list[Turn]:
    """Graded turns only (status completed, solved 0/1, 13 numbers in x), in a fixed order."""
    out = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] != "completed" or r["solved"] not in ("0", "1") or not r["x"]:
                continue
            x = tuple(float(v) for v in json.loads(r["x"]))
            if len(x) != N_FEATURES or r["arm"] not in ARM_NAMES:
                continue
            out.append(Turn(r["task"], r["arm"], int(r["turn"]), r["task_type"],
                            r["solved"] == "1", float(r["cost_usd"] or 0), x, r["router_task"]))
    return sorted(out, key=lambda t: (t.task, t.arm, t.turn))


# ---------- 2. C_ref ----------

def model_of(arm: str) -> str:
    return arm.split("-")[0]


def percentile_cost(turns: list[Turn], model: str, q: float = settings.C_REF_PERCENTILE) -> float:
    costs = [t.cost for t in turns if model_of(t.arm) == model]
    if not costs:
        raise ValueError(f"no {model} turns to take a percentile of")
    return float(np.percentile(costs, q))


def c_ref_estimates(turns: list[Turn]) -> dict:
    """C_ref from Opus turns, and the cross-check from Sonnet turns x the price ratio."""
    ratio = OPUS.price.input / SONNET.price.input
    out = {"opus_turns": sum(model_of(t.arm) == "opus" for t in turns),
           "sonnet_turns": sum(model_of(t.arm) == "sonnet" for t in turns),
           "opus_sonnet_price_ratio": ratio}
    out["opus_p95"] = percentile_cost(turns, "opus") if out["opus_turns"] else None
    out["sonnet_p95_x_ratio"] = (percentile_cost(turns, "sonnet") * ratio
                                 if out["sonnet_turns"] else None)
    return out


# ---------- 3-5. rewards and the prior ----------

def rewards(turns: list[Turn], c_ref: float, lam: float) -> list[float]:
    return [reward(t.solved, t.cost, c_ref, lam) for t in turns]


def new_bandit(alpha: float = settings.ALPHA) -> HybridLinUCB:
    return HybridLinUCB(ARM_NAMES, arm_features(), n_features=N_FEATURES, shared=SHARED,
                        alpha=alpha)


def build_prior(turns: list[Turn], c_ref: float, lam: float,
                alpha: float = settings.ALPHA) -> tuple[HybridLinUCB, Standardizer]:
    std = Standardizer.fit([t.x for t in turns])
    bandit = new_bandit(alpha)
    for t, r in zip(turns, rewards(turns, c_ref, lam), strict=True):
        bandit.update(std.transform(t.x), t.arm, r)
    return bandit, std


def save_prior(path: Path, bandit: HybridLinUCB, std: Standardizer, c_ref: float, lam: float,
               rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **bandit.to_state(), **std.to_state(), c_ref=np.array(c_ref),
             lam=np.array(lam), rows=np.array(rows))


def load_prior(path: Path) -> tuple[HybridLinUCB, Standardizer, float, float]:
    with np.load(path) as data:
        state = {k: data[k] for k in data.files}
    return (HybridLinUCB.from_state(state), Standardizer.from_state(state),
            float(state["c_ref"]), float(state["lam"]))


# ---------- 6. offline checks ----------

def cells(turns: list[Turn], arms=EVAL_ARMS) -> dict[tuple[str, int], dict[str, Turn]]:
    """(task, turn) -> {arm: turn} for the cells where every arm in `arms` has a graded turn."""
    by: dict[tuple[str, int], dict[str, Turn]] = defaultdict(dict)
    for t in turns:
        if t.arm in arms:
            by[(t.task, t.turn)][t.arm] = t
    return {k: v for k, v in sorted(by.items()) if len(v) == len(arms)}


def cell_x(cell: dict[str, Turn]) -> np.ndarray:
    """The turn's x: the same prompt in every arm, so router numbers, message and turn count
    agree; context size differs a little per arm (earlier turns) -> the mean."""
    return np.mean([t.x for t in cell.values()], axis=0)


def _score(t: Turn, c_ref: float, lam: float) -> float:
    return reward(t.solved, t.cost, c_ref, lam)


def leave_one_task_out(turns: list[Turn], c_ref: float, lam: float,
                       arms=EVAL_ARMS) -> dict:
    """For each task: train on the other tasks' turns, choose greedily (alpha 0) among `arms`
    for each of its turns, and score the chosen arm's real result there."""
    grid = cells(turns, arms)
    tasks = sorted({task for task, _ in grid})
    per_turn: list[tuple[str, int, str]] = []  # (task, turn, chosen arm)
    per_session: list[tuple[str, int, str]] = []
    for task in tasks:
        train = [t for t in turns if t.task != task]
        bandit, std = build_prior(train, c_ref, lam, alpha=0.0)
        mine = sorted((n, c) for (tk, n), c in grid.items() if tk == task)
        first = None
        for n, cell in mine:
            arm = bandit.choose(std.transform(cell_x(cell)), allowed=arms).arm
            first = first or arm
            per_turn.append((task, n, arm))
            per_session.append((task, n, first))
    return {"tasks": tasks, "grid": grid, "per_turn": per_turn, "per_session": per_session}


def summarize(choices: list[tuple[str, int, str]], grid, c_ref: float, lam: float) -> dict:
    picked = [grid[(task, n)][arm] for task, n, arm in choices]
    return {"turns": len(picked), "solved": sum(t.solved for t in picked),
            "cost": round(sum(t.cost for t in picked), 4),
            "mean_reward": round(float(np.mean([_score(t, c_ref, lam) for t in picked])), 4),
            "arms": dict(Counter(arm for _, _, arm in choices).most_common())}


def baselines(grid, c_ref: float, lam: float, arms=EVAL_ARMS) -> dict:
    keys = list(grid)
    out = {f"always {a}": summarize([(k[0], k[1], a) for k in keys], grid, c_ref, lam)
           for a in arms}
    best_turn = [(k[0], k[1], max(arms, key=lambda a, k=k: _score(grid[k][a], c_ref, lam)))
                 for k in keys]
    out["best per turn (oracle)"] = summarize(best_turn, grid, c_ref, lam)
    by_task = defaultdict(list)
    for k in keys:
        by_task[k[0]].append(k)
    best_session = []
    for task_keys in by_task.values():
        a = max(arms, key=lambda a: sum(_score(grid[k][a], c_ref, lam) for k in task_keys))
        best_session += [(k[0], k[1], a) for k in task_keys]
    out["best per session (oracle)"] = summarize(best_session, grid, c_ref, lam)
    mean = float(np.mean([_score(grid[k][a], c_ref, lam) for k in keys for a in arms]))
    out["random arm (expected)"] = {
        "turns": len(keys),
        "solved": round(sum(np.mean([grid[k][a].solved for a in arms]) for k in keys), 1),
        "cost": round(sum(np.mean([grid[k][a].cost for a in arms]) for k in keys), 4),
        "mean_reward": round(mean, 4), "arms": {}}
    return out


def prior_picks(bandit: HybridLinUCB, std: Standardizer, turns: list[Turn]) -> dict:
    """What the prior predicts for a typical turn of each type (mean x of that type)."""
    out = {}
    for kind in sorted({t.task_type for t in turns}):
        x = std.transform(np.mean([t.x for t in turns if t.task_type == kind], axis=0))
        d = bandit.choose(x)
        out[kind] = {"choice": d.arm, "estimates": {a: round(s.estimate, 3)
                                                    for a, s in d.scores.items()},
                     "bonus": {a: round(s.bonus, 3) for a, s in d.scores.items()}}
    return out


# ---------- report ----------

def _table(rows: dict[str, dict]) -> list[str]:
    lines = ["| Policy | Turns solved | Cost (list price) | Mean reward | Arms chosen |",
             "|---|---|---|---|---|"]
    for name, s in rows.items():
        arms = ", ".join(f"{a} {n}" for a, n in s["arms"].items()) or "-"
        lines.append(f"| {name} | {s['solved']} / {s['turns']} | ${s['cost']:.2f} | "
                     f"{s['mean_reward']:.3f} | {arms} |")
    return lines


def report(turns: list[Turn], c_ref: float, lam: float, est: dict) -> tuple[str, dict]:
    bandit, std = build_prior(turns, c_ref, lam)
    loto = leave_one_task_out(turns, c_ref, lam)
    grid = loto["grid"]
    main = {"ODE per turn (leave-one-task-out)": summarize(loto["per_turn"], grid, c_ref, lam),
            "ODE per session (leave-one-task-out)": summarize(loto["per_session"], grid, c_ref,
                                                              lam),
            **baselines(grid, c_ref, lam)}
    sweep = {}
    for lam_i in LAMBDAS:
        lo = leave_one_task_out(turns, c_ref, lam_i)
        sweep[lam_i] = summarize(lo["per_turn"], lo["grid"], c_ref, lam_i)
    sens = {}
    for name, c in (("C_ref (Opus p95)", c_ref), ("Sonnet p95 x 2", est["sonnet_p95_x_ratio"]),
                    ("C_ref / 2", c_ref / 2), ("C_ref x 2", c_ref * 2)):
        if c:
            lo = leave_one_task_out(turns, c, lam)
            sens[f"{name} = ${c:.3f}"] = summarize(lo["per_turn"], lo["grid"], c, lam)
    over_100k = sum(t.x[10] > 0.5 for t in turns)
    picks = prior_picks(bandit, std, turns)
    r = rewards(turns, c_ref, lam)
    capped = sum(t.cost >= c_ref for t in turns)

    lines = [
        "# ODE phase 5b: prior and offline check",
        "",
        (f"Rows: {len(turns)} graded turns, {len({t.task for t in turns})} tasks, "
         f"{len({t.arm for t in turns})} arms (Opus only in the pilot)."),
        (f"C_ref = ${c_ref:.4f} (Opus p{settings.C_REF_PERCENTILE}, {est['opus_turns']} turns); "
         f"cross-check Sonnet p{settings.C_REF_PERCENTILE} x {est['opus_sonnet_price_ratio']:.0f}"
         f" = ${est['sonnet_p95_x_ratio']:.4f} ({est['sonnet_turns']} turns). "
         f"lambda = {lam}. Turns at or above C_ref (cost score capped at 1): {capped}."),
        (f"Rewards: mean {np.mean(r):.3f}, min {min(r):.3f}, max {max(r):.3f}. "
         f"Turns with context > 100k: {over_100k} of {len(turns)}."),
        "",
        (f"## Leave-one-task-out ({len(loto['tasks'])} tasks, {len(grid)} turns where all "
         f"{len(EVAL_ARMS)} compared arms were graded; greedy choice, alpha 0)"),
        "",
        *_table(main),
        "",
        "Per turn ignores switch costs (the data has no mid-session switches); per session keeps",
        "the turn-1 choice. Oracles know every arm's real result: an upper bound, not a policy.",
        "",
        "## Lambda sweep (per turn, leave-one-task-out; reward scored with that lambda)",
        "",
        *_table({f"lambda = {k}": v for k, v in sweep.items()}),
        "",
        "## C_ref sensitivity (per turn, leave-one-task-out)",
        "",
        *_table(sens),
        "",
        "## What the full prior picks for a typical turn of each type (alpha = 1)",
        "",
        "| Turn type | Choice | Estimate per arm (bonus) |",
        "|---|---|---|",
    ]
    for kind, p in picks.items():
        est_s = ", ".join(f"{a} {e:.2f} (+{p['bonus'][a]:.2f})" for a, e in p["estimates"].items())
        lines.append(f"| {kind} | {p['choice']} | {est_s} |")
    data = {"c_ref": c_ref, "lam": lam, "c_ref_estimates": est, "rows": len(turns),
            "main": main, "lambda_sweep": {str(k): v for k, v in sweep.items()},
            "c_ref_sensitivity": sens, "prior_picks": picks, "context_over_100k": over_100k}
    return "\n".join(lines) + "\n", data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--turns", type=Path, default=RESULTS / "turns.csv")
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--lam", type=float, default=settings.LAMBDA)
    parser.add_argument("--c-ref", type=float, default=None,
                        help="USD; default: Opus p95 of the saved turns")
    args = parser.parse_args(argv)
    turns = load_turns(args.turns)
    est = c_ref_estimates(turns)
    c_ref = args.c_ref or est["opus_p95"]
    if not c_ref or not math.isfinite(c_ref):
        parser.error("no Opus turns: pass --c-ref")
    bandit, std = build_prior(turns, c_ref, args.lam)
    save_prior(args.out / "ode_prior.npz", bandit, std, c_ref, args.lam, len(turns))
    text, data = report(turns, c_ref, args.lam, est)
    (args.out / "prior_report.md").write_text(text, encoding="utf-8")
    (args.out / "prior_report.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
