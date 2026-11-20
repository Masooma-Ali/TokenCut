"""Classify a user's request into a task type (prompt enhancement, docs/PROMPT_ROUTER_LABELS.md).

Cascade: cheap rules first (follow-ups like "yes", empty text), then a classifier on frozen
sentence embeddings (local, fastembed / ONNX on CPU). A prediction that is not confident
enough goes to `ambiguous`, and the user's message is then left unchanged: a wrong label
would give the agent wrong advice. Everything here is deterministic for the same text and
the same saved model, so a message gets the same label on every turn (prompt cache safe).
"""

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

AMBIGUOUS = "ambiguous"
STAGE1 = {
    "bugfix": "change_code",
    "feature": "change_code",
    "refactor": "change_code",
    "tests": "change_code",
    "explain": "understand_code",
    "review": "understand_code",
    "design": "plan",
    "run_action": "run_action",
    "write_text": "write_text",
    "followup": "followup",
    AMBIGUOUS: AMBIGUOUS,
}
LABELS = tuple(STAGE1)
MODEL = "BAAI/bge-small-en-v1.5"  # same model as the offline benchmark and the repo index
MAX_CHARS = 2_000  # the start of a request says what is wanted; long pastes follow it

_FOLLOWUP_WORDS = frozenset(
    ["y", "yes", "yep", "yeah", "yup", "ok", "okay", "k", "sure", "no", "nope", "continue", "go", "ahead", "on", "proceed", "do", "it", "that", "this", "looks", "look", "good", "great", "fine", "lgtm", "thanks", "thank", "you", "please", "option", "use", "the", "first", "second", "third", "one", "right", "correct", "agreed", "next", "sounds", "perfect"]
)
_WORD_RE = re.compile(r"[a-z]+|\d+")
MAX_FOLLOWUP_WORDS = 6


def rule_label(text: str) -> str | None:
    """Label decided without a model, or None (then the classifier decides)."""
    words = _WORD_RE.findall(text.lower())
    if not words:
        return AMBIGUOUS
    if len(words) <= MAX_FOLLOWUP_WORDS and all(w in _FOLLOWUP_WORDS or w.isdigit() for w in words):
        return "followup"
    return None


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Prototypes:
    """One mean vector per label; probability = softmax of scaled cosine similarity."""

    def __init__(self, temperature: float = 20.0):
        self.temperature = temperature

    def fit(self, x: np.ndarray, y: list[str]) -> "Prototypes":
        x = _unit(x)
        self.labels = sorted(set(y))
        self.centroids = _unit(np.stack([x[[t == lab for t in y]].mean(axis=0) for lab in self.labels]))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return _softmax(self.temperature * _unit(x) @ self.centroids.T)


class KNN:
    """Similarity-weighted vote of the k nearest training prompts."""

    def __init__(self, k: int = 5, temperature: float = 20.0):
        self.k, self.temperature = k, temperature

    def fit(self, x: np.ndarray, y: list[str]) -> "KNN":
        self.x = _unit(x)
        self.labels = sorted(set(y))
        self.y = np.array([self.labels.index(t) for t in y])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        sims = _unit(x) @ self.x.T
        k = min(self.k, sims.shape[1])
        top = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        out = np.full((len(x), len(self.labels)), 1e-6, dtype=np.float64)
        for i, row in enumerate(top):
            weights = np.exp(self.temperature * (sims[i, row] - sims[i, row].max()))
            np.add.at(out[i], self.y[row], weights)
        return out / out.sum(axis=1, keepdims=True)


class SoftmaxHead:
    """Multinomial logistic regression on frozen embeddings (full-batch Adam, L2).

    class_weight="balanced": each label counts as much as any other in the loss, however many
    examples it has (weight = n / (labels * count)), so rare labels are not drowned out."""

    def __init__(self, l2: float = 1e-3, epochs: int = 500, lr: float = 0.05,
                 class_weight: str | None = None):
        self.l2, self.epochs, self.lr, self.class_weight = l2, epochs, lr, class_weight

    def fit(self, x: np.ndarray, y: list[str]) -> "SoftmaxHead":
        x = _unit(x).astype(np.float64)
        self.labels = sorted(set(y))
        target = np.zeros((len(y), len(self.labels)))
        target[np.arange(len(y)), [self.labels.index(t) for t in y]] = 1
        weight = np.ones((len(y), 1))
        if self.class_weight == "balanced":
            counts = target.sum(axis=0)
            weight = (len(y) / (len(self.labels) * counts))[[self.labels.index(t) for t in y]][:, None]
        w = np.zeros((x.shape[1], len(self.labels)))
        b = np.zeros(len(self.labels))
        m = [np.zeros_like(w), np.zeros_like(b)]
        v = [np.zeros_like(w), np.zeros_like(b)]
        for step in range(1, self.epochs + 1):
            grad = weight * (_softmax(x @ w + b) - target) / len(y)
            grads = [x.T @ grad + self.l2 * w, grad.sum(axis=0)]
            for j, (param, g) in enumerate(zip((w, b), grads)):
                m[j] = 0.9 * m[j] + 0.1 * g
                v[j] = 0.999 * v[j] + 0.001 * g * g
                param -= self.lr * (m[j] / (1 - 0.9**step)) / (np.sqrt(v[j] / (1 - 0.999**step)) + 1e-8)
        self.w, self.b = w.astype(np.float32), b.astype(np.float32)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return _softmax(_unit(x) @ self.w + self.b)

    def save(self, path: Path) -> None:
        np.savez(path, w=self.w, b=self.b, labels=np.array(self.labels))

    @classmethod
    def load(cls, path: Path) -> "SoftmaxHead":
        data = np.load(path)
        head = cls()
        head.w, head.b, head.labels = data["w"], data["b"], [str(s) for s in data["labels"]]
        return head


class TwoStage:
    """Broad group first (STAGE1), then the label inside the group: p = p(group) * p(label|group)."""

    def __init__(self, make: Callable[[], object] = SoftmaxHead):
        self.make = make

    def fit(self, x: np.ndarray, y: list[str]) -> "TwoStage":
        self.labels = sorted(set(y))
        groups = [STAGE1[t] for t in y]
        self.stage1 = self.make().fit(x, groups)
        self.stage2 = {}
        for group in self.stage1.labels:
            rows = [i for i, g in enumerate(groups) if g == group]
            if len({y[i] for i in rows}) > 1:
                self.stage2[group] = self.make().fit(x[rows], [y[i] for i in rows])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p1 = self.stage1.predict_proba(x)
        out = np.zeros((len(x), len(self.labels)))
        for g, group in enumerate(self.stage1.labels):
            if group in self.stage2:
                inner = self.stage2[group]
                p2 = inner.predict_proba(x)
                for j, label in enumerate(inner.labels):
                    out[:, self.labels.index(label)] = p1[:, g] * p2[:, j]
            else:
                label = next(t for t in self.labels if STAGE1[t] == group)
                out[:, self.labels.index(label)] = p1[:, g]
        return out


def fastembed_embedder(cache_dir: Path, model: str = MODEL) -> Callable[[list[str]], np.ndarray]:
    from fastembed import TextEmbedding  # optional dependency, imported only when used

    encoder = TextEmbedding(model, cache_dir=str(cache_dir / "models"))
    return lambda texts: np.stack([np.asarray(v, dtype=np.float32) for v in encoder.embed(texts)])


class PromptRouter:
    """rules -> classifier -> `ambiguous` when the top label is not clearly ahead."""

    def __init__(self, model, embed: Callable[[list[str]], np.ndarray],
                 min_prob: float = 0.6, min_margin: float = 0.15):
        self.model, self.embed = model, embed
        self.min_prob, self.min_margin = min_prob, min_margin
        self._cache: dict[str, tuple[str, float]] = {}

    def classify(self, text: str) -> tuple[str, float]:
        ruled = rule_label(text)
        if ruled is not None:
            return ruled, 1.0
        key = text[:MAX_CHARS]
        if key not in self._cache:
            probs = self.model.predict_proba(self.embed([key]))[0]
            order = np.argsort(-probs, kind="stable")
            top = float(probs[order[0]])
            second = float(probs[order[1]]) if len(order) > 1 else 0.0
            label = self.model.labels[order[0]]
            if top < self.min_prob or top - second < self.min_margin:
                label = AMBIGUOUS
            self._cache[key] = (label, top)
        return self._cache[key]
