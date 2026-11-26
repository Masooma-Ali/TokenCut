"""The 13 numbers (x) that describe a task to ODE, read from the request and the router.

    0-5   router probability: bugfix, feature, refactor, tests, explain, run_action
          (no_brief is left out: the 7 probabilities sum to 1, so it adds nothing next to the bias)
    6     files named in the message, log(1 + count)
    7     a traceback or stack trace in the message (0/1)
    8     message length, log(1 + tokens)
    9     conversation size, log(1 + tokens)
    10    conversation over 100,000 tokens (0/1): Haiku's price step
    11    earlier typed task messages in the conversation, log(1 + count)
    12    bias, always 1

z (shared by all arms) uses 0-5, 9-11 and 12: task type and session size times the arm numbers.
Everything is computed from the request itself (no disk, no network); the router is the one
already loaded for task briefs. Numbers only: no text from the request is kept or logged.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from tokencut.contracts import Segment, SegmentKind
from tokencut.gateway.task_router import LABELS, NO_BRIEF, prepare, prior_correct
from tokencut.ode import settings
from tokencut.tokens import count_tokens

CLASSES = ("bugfix", "feature", "refactor", "tests", "explain", "run_action")
FEATURE_NAMES = (*[f"p_{c}" for c in CLASSES], "files_named", "traceback", "message_len",
                 "context_len", "context_over_100k", "turns_before", "bias")
N_FEATURES = len(FEATURE_NAMES)  # 13
SHARED = (0, 1, 2, 3, 4, 5, 9, 10, 11, 12)  # features combined with the arm numbers into z
BIAS = N_FEATURES - 1
HAIKU_STEP = 100_000

_EXT = (r"py|pyi|ipynb|js|jsx|mjs|ts|tsx|java|kt|go|rs|rb|php|c|cc|cpp|h|hpp|cs|swift|scala|sh|"
        r"ps1|sql|html|css|md|rst|txt|json|ya?ml|toml|ini|cfg|lock")
_FILE_RE = re.compile(rf"(?<![\w/.-])((?:[A-Za-z]:/)?(?:[\w.-]+/)*[\w-]+\.(?:{_EXT}))\b",
                      re.IGNORECASE)
_TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\)"  # Python
    r'|File "[^"\n]+", line \d+'  # Python frame
    r"|^\s+at \S.*:\d+:\d+\)?\s*$"  # JavaScript / TypeScript
    r"|^\s+at [\w.$<>]+\([\w.]+:\d+\)",  # Java / Kotlin
    re.MULTILINE)


def files_named(text: str) -> int:
    return len({m.group(1).replace("\\", "/").lower() for m in _FILE_RE.finditer(
        text.replace("\\", "/"))})


def has_traceback(text: str) -> bool:
    return bool(_TRACEBACK_RE.search(text))


def raw_features(probs: Mapping[str, float], message: str, context_tokens: int,
                 turns_before: int) -> np.ndarray:
    """x before standardization. probs: router label -> probability (extra labels are ignored)."""
    if context_tokens < 0 or turns_before < 0:
        raise ValueError("context_tokens and turns_before must be >= 0")
    p = [float(probs.get(c, 0.0)) for c in CLASSES]
    if any(not (0.0 <= v <= 1.0) for v in p):
        raise ValueError(f"probabilities must be in [0, 1], got {p}")
    return np.array([
        *p,
        math.log1p(files_named(message)),
        1.0 if has_traceback(message) else 0.0,
        math.log1p(count_tokens(message)),
        math.log1p(context_tokens),
        1.0 if context_tokens > HAIKU_STEP else 0.0,
        math.log1p(turns_before),
        1.0,
    ])


# ---------- from a request ----------

@dataclass(frozen=True)
class TaskMessage:
    text: str  # the newest typed message, prepared for the router (no reminders or markup)
    turns_before: int  # earlier typed messages in the conversation that were real tasks


def latest_task_message(segments: Sequence[Segment]) -> TaskMessage | None:
    """The newest typed user message if it can be a task; None for a follow-up ("yes"), a
    machine message (slash command, harness notice) or a request with no typed message."""
    groups: dict[int, list[str]] = {}
    for seg in segments:
        if seg.kind == SegmentKind.USER_TEXT and seg.message_index >= 0:
            groups.setdefault(seg.message_index, []).append(seg.text)
    prepared = [prepare("\n".join(texts))[0] for _, texts in sorted(groups.items())]
    if not prepared or prepared[-1] is None:
        return None
    return TaskMessage(prepared[-1], sum(t is not None for t in prepared[:-1]))


def estimate_context_tokens(segments: Sequence[Segment]) -> int:
    """Request size from the gateway's own token counts (tiktoken proxy, undercounts Claude by
    ~1.5x). Prefer the billed prompt size of the previous request when the gateway has it."""
    return sum(seg.token_count for seg in segments)


def router_probs(router, text: str) -> dict[str, float]:
    """All 7 router probabilities (prior-corrected, as task briefs use them)."""
    probs = prior_correct(router.probs(text)[None, :], router.train_prior, router.natural_prior)[0]
    return {label: float(p) for label, p in zip(LABELS, probs, strict=True)}


def is_decision_point(probs: Mapping[str, float],
                      threshold: float = settings.DECISION_THRESHOLD) -> bool:
    """ODE decides at a real task: the router's top label is not `no_brief` and reaches the
    threshold (0 since the pilot: see settings.py). Separate from R10."""
    label = max(probs, key=probs.get)
    return label != NO_BRIEF and probs[label] >= threshold


def request_features(segments: Sequence[Segment], router,
                     context_tokens: int | None = None) -> tuple[np.ndarray, dict[str, float]] | None:
    """(raw x, router probabilities) for the newest typed message, or None if it is no task."""
    msg = latest_task_message(segments)
    if msg is None:
        return None
    probs = router_probs(router, msg.text)
    ctx = estimate_context_tokens(segments) if context_tokens is None else context_tokens
    return raw_features(probs, msg.text, ctx, msg.turns_before), probs


# ---------- standardization ----------

class Standardizer:
    """Zero mean, unit variance per feature, fitted once on the warm-up rows (Phase 5) and stored
    with the bandit. The bias stays 1; a feature that never varies is only centred."""

    def __init__(self, mean: Sequence[float] | None = None, std: Sequence[float] | None = None):
        self.mean = np.zeros(N_FEATURES) if mean is None else np.asarray(mean, dtype=float)
        self.std = np.ones(N_FEATURES) if std is None else np.asarray(std, dtype=float)
        if self.mean.shape != (N_FEATURES,) or self.std.shape != (N_FEATURES,) \
                or np.any(self.std <= 0):
            raise ValueError(f"mean and std need {N_FEATURES} values, std > 0")

    @classmethod
    def fit(cls, rows: Sequence[Sequence[float]]) -> Standardizer:
        x = np.asarray(rows, dtype=float)
        if x.ndim != 2 or x.shape[1] != N_FEATURES or len(x) < 2:
            raise ValueError(f"need at least 2 rows of {N_FEATURES} features, got {x.shape}")
        mean, std = x.mean(axis=0), x.std(axis=0)
        std[std < 1e-9] = 1.0
        mean[BIAS], std[BIAS] = 0.0, 1.0
        return cls(mean, std)

    def transform(self, x: Sequence[float]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape != (N_FEATURES,):
            raise ValueError(f"x must have {N_FEATURES} numbers, got shape {x.shape}")
        return (x - self.mean) / self.std

    def to_state(self) -> dict[str, np.ndarray]:
        return {"feature_names": np.array(FEATURE_NAMES), "feature_mean": self.mean.copy(),
                "feature_std": self.std.copy()}

    @classmethod
    def from_state(cls, state: Mapping[str, np.ndarray]) -> Standardizer:
        if tuple(str(n) for n in state["feature_names"]) != FEATURE_NAMES:
            raise ValueError("state was saved with a different feature list")
        return cls(state["feature_mean"], state["feature_std"])
