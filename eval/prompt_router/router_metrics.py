"""Prompt router (SWE-chat labels): token head + tail cut and every evaluation value.

Plain Python + numpy so it is tested in the TokenCut environment and also runs on Kaggle next
to train_router_kaggle.py (upload both files).

A brief is added only when the top label is a real task (not `no_brief`) AND its probability
is >= the threshold. Main values per threshold (brief_curve):
  all_briefs_right          correct briefs / all briefs given
  coverage                  real-task prompts that got a brief / all real-task prompts
  no_brief_wrongly_briefed  no_brief prompts that got a brief / all no_brief prompts
"""

import re

import numpy as np

LABELS = ["bugfix", "explain", "feature", "no_brief", "refactor", "run_action", "tests"]
NO_BRIEF = "no_brief"
GROUP = {"bugfix": "change_code", "feature": "change_code", "refactor": "change_code",
         "tests": "change_code", "explain": "understand", "run_action": "run", NO_BRIEF: NO_BRIEF}
HEAD_TOKENS, TAIL_TOKENS = 360, 149  # + [CLS], "…", [SEP] = 512, the model's limit
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_LOG = re.compile(r"Traceback|Error|error:|Exception|at .*\(.*:\d+\)|^\s*File \"", re.MULTILINE)


def cut_token_ids(ids: list[int], head: int = HEAD_TOKENS, tail: int = TAIL_TOKENS,
                  marker: list[int] | None = None) -> list[int]:
    """Keep the first `head` and last `tail` token ids (instruction + final error line)."""
    if len(ids) <= head + tail + len(marker or []):
        return list(ids)
    return list(ids[:head]) + list(marker or []) + list(ids[-tail:])


def _decide(probs: np.ndarray) -> tuple[list[str], np.ndarray]:
    return [LABELS[i] for i in probs.argmax(axis=1)], probs.max(axis=1)


def brief_curve(probs: np.ndarray, true: list[str], thresholds=THRESHOLDS) -> list[dict]:
    pred, conf = _decide(probs)
    real = np.array([t != NO_BRIEF for t in true])
    out = []
    for th in thresholds:
        brief = np.array([p != NO_BRIEF and c >= th for p, c in zip(pred, conf)])
        right = sum(b and p == t for b, p, t in zip(brief, pred, true))
        out.append({
            "min_prob": th, "briefs": int(brief.sum()),
            "all_briefs_right": right / brief.sum() if brief.any() else None,
            "coverage": float((brief & real).sum() / real.sum()) if real.any() else None,
            "no_brief_wrongly_briefed": (float((brief & ~real).sum() / (~real).sum())
                                         if (~real).any() else None)})
    return out


def calibration(probs: np.ndarray, true: list[str], n_bins: int = 10) -> list[dict]:
    """Per confidence band: mean confidence vs share actually correct."""
    pred, conf = _decide(probs)
    correct = np.array([p == t for p, t in zip(pred, true)], dtype=float)
    bins = []
    for k in range(n_bins):
        lo, hi = k / n_bins, (k + 1) / n_bins
        mask = (conf >= lo) & ((conf < hi) if k < n_bins - 1 else (conf <= hi))
        bins.append({"from": lo, "to": hi, "count": int(mask.sum()),
                     "confidence": float(conf[mask].mean()) if mask.any() else None,
                     "accuracy": float(correct[mask].mean()) if mask.any() else None})
    return bins


def bootstrap_ci(values: np.ndarray, stat, n: int = 1000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    stats = [stat(values[rng.integers(0, len(values), len(values))]) for _ in range(n)]
    low, high = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(low), float(high)


def _per_label(true: list[str], pred: list[str]) -> dict:
    out = {}
    for lab in LABELS:
        tp = sum(t == lab and p == lab for t, p in zip(true, pred))
        n_pred, n_true = pred.count(lab), true.count(lab)
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_true if n_true else 0.0
        out[lab] = {"precision": prec, "recall": rec, "support": n_true,
                    "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    return out


def _slices(meta: list[dict]) -> dict[str, np.ndarray]:
    def mask(f):
        return np.array([bool(f(m)) for m in meta])
    out = {"all": mask(lambda m: True),
           "long (> 510 tokens, cut)": mask(lambda m: m.get("n_tokens", 0) > HEAD_TOKENS + TAIL_TOKENS),
           "error log / traceback": mask(lambda m: _LOG.search(m["text"])),
           "short (<= 20 chars)": mask(lambda m: len(m["text"]) <= 20),
           "correction / rejection": mask(lambda m: m.get("pushback") in ("correction", "rejection"))}
    for agent in sorted({m.get("agent") for m in meta if m.get("agent")}):
        out[f"agent: {agent}"] = mask(lambda m, a=agent: m.get("agent") == a)
    return out


def evaluate(probs: np.ndarray, true: list[str], meta: list[dict], ci: bool = False) -> dict:
    """Every value of the plan: overall, per label, confusion, groups, curve, calibration, slices."""
    pred, _ = _decide(probs)
    per = _per_label(true, pred)
    present = [lab for lab in LABELS if per[lab]["support"]]
    report = {
        "n": len(true),
        "accuracy": float(np.mean([p == t for p, t in zip(pred, true)])),
        "macro_f1": float(np.mean([per[lab]["f1"] for lab in present])),
        "broad_group_accuracy": float(np.mean([GROUP[p] == GROUP[t] for p, t in zip(pred, true)])),
        "per_label": per,
        "confusion": {t: {p: sum(1 for a, b in zip(true, pred) if a == t and b == p)
                          for p in LABELS} for t in LABELS},
        "brief_curve": brief_curve(probs, true),
        "calibration": calibration(probs, true),
        "slices": {},
    }
    for name, mask in _slices(meta).items():
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        sub_true = [true[i] for i in idx]
        report["slices"][name] = {
            "n": int(mask.sum()),
            "accuracy": float(np.mean([pred[i] == true[i] for i in idx])),
            "curve_at_0.7": next(p for p in brief_curve(probs[idx], sub_true, [0.7]))}
    if ci:
        correct = np.array([p == t for p, t in zip(pred, true)], dtype=float)
        report["accuracy_ci95"] = bootstrap_ci(correct, np.mean)
        idx = np.arange(len(true))
        for point in report["brief_curve"]:
            th = point["min_prob"]

            def right_share(sample, th=th):
                pts = brief_curve(probs[sample], [true[i] for i in sample], [th])[0]
                return pts["all_briefs_right"] if pts["all_briefs_right"] is not None else 0.0
            point["all_briefs_right_ci95"] = bootstrap_ci(idx, right_share, n=300)
    return report


def prior_correct(probs: np.ndarray, train_prior: np.ndarray, real_prior: np.ndarray) -> np.ndarray:
    """Re-weight probabilities from the training label mix to the real one (Bayes' rule).

    Training is balanced by label; real traffic is not. p_real(y|x) ∝ p_train(y|x) · real(y) / train(y).
    """
    scaled = probs * (np.asarray(real_prior) / np.maximum(np.asarray(train_prior), 1e-12))
    return scaled / scaled.sum(axis=1, keepdims=True)
