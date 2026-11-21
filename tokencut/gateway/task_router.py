"""Task type of a typed user message (prompt enhancement): rules first, then the trained router.

Rules: `<system-reminder>` text is removed; machine-generated messages (slash commands,
interrupts, harness / CI / sub-agent / approval messages, tool logs, image-only) and short
follow-ups ("yes", "continue") get no brief. Everything else goes to the router: bge-small
fine-tuned on SWE-chat (RESULTS_REPORT §11), run as ONNX on CPU with the training input (first
360 + "…" + last 149 tokens) and probabilities re-weighted to the natural label mix.

The model files (router.onnx, router.onnx.data, tokenizer.json, router_config.json) are not in
git: TOKENCUT_ROUTER_DIR, or downloaded once (no login) from the public Hugging Face repo
TOKENCUT_ROUTER_REPO. onnxruntime, tokenizers and huggingface_hub are package dependencies and
are imported only when the router is used.
"""

import hashlib
import json
import logging
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

from tokencut.gateway.prompt_router import rule_label

log = logging.getLogger(__name__)

# Must stay identical to eval/prompt_router/router_metrics.py (the standalone Kaggle copy).
LABELS = ["bugfix", "explain", "feature", "no_brief", "refactor", "run_action", "tests"]
NO_BRIEF = "no_brief"
HEAD_TOKENS, TAIL_TOKENS = 360, 149
MODEL_FILES = ("router.onnx", "router.onnx.data", "tokenizer.json", "router_config.json")
DEFAULT_REPO = "humataj/tokencut-router"
DEFAULT_CACHE = Path.home() / ".cache" / "tokencut" / "router"
CACHE_SIZE = 10_000  # classified messages kept (text hash -> label, prob): ~1 MB

_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_MACHINE_START = re.compile(
    r"^[\s\"'`]*(<(command-name|command-message|command-args|local-command-stdout|environment_context|"
    r"subagent_notification|turn_aborted|skill|skills|codex_internal_context|recommended_plugins|"
    r"timestamp|task-notification|teammate-message|ci-monitor-event|objective|state_management|"
    r"verification|team_[a-z_]+|system-reminder|user_instructions|permissions)\b"
    r"|\[\d+\] tool [\w.-]+ (call|result):|\[\d+\] (assistant|user|system):"
    r"|Base directory for this skill:|Current state: \w+ \(for "  # Claude Code status summary
    r"|\[Request interrupted|\[delegate worktree\]|/[A-Za-z][\w:-]*(\s|$)|>>> )",
    re.IGNORECASE)
_MACHINE_ANYWHERE = re.compile(
    r"APPROVAL REQUEST START|The Codex agent has requested|TRANSCRIPT DELTA START|"
    r"^Execution instructions:|^# AGENTS\.md instructions|Another Claude session sent a message|"
    r"<teammate-message|<task-notification|<usage>|<tool_uses>", re.IGNORECASE | re.MULTILINE)
_IMAGE = re.compile(r"<image[^>]*>|</image>|\[Image[^\]]*\]", re.IGNORECASE)


def strip_markup(text: str) -> str:
    return " ".join(_IMAGE.sub(" ", text).split()) if _IMAGE.search(text) else text.strip()


def is_machine(text: str) -> bool:
    """Not typed by the user: harness, slash command, CI / sub-agent notice, approval request."""
    rest = strip_markup(text)
    return (not rest or bool(_MACHINE_START.match(text)) or bool(_MACHINE_ANYWHERE.search(text)))


def prepare(text: str) -> tuple[str | None, str]:
    """(text for the router, "") or (None, reason the message gets no brief)."""
    text = _REMINDER.sub("", text).strip()
    if not text:
        return None, "empty"
    if is_machine(text):
        return None, "machine"
    if rule_label(text) == "followup":
        return None, "followup"
    return strip_markup(text), ""


def cut_token_ids(ids: list[int], head: int = HEAD_TOKENS, tail: int = TAIL_TOKENS,
                  marker: list[int] | None = None) -> list[int]:
    """Keep the first `head` and last `tail` token ids (instruction + final error line)."""
    if len(ids) <= head + tail + len(marker or []):
        return list(ids)
    return list(ids[:head]) + list(marker or []) + list(ids[-tail:])


def prior_correct(probs: np.ndarray, train_prior: np.ndarray, real_prior: np.ndarray) -> np.ndarray:
    """Re-weight probabilities from the training label mix to the real one (Bayes' rule)."""
    scaled = probs * (np.asarray(real_prior) / np.maximum(np.asarray(train_prior), 1e-12))
    return scaled / scaled.sum(axis=1, keepdims=True)


class OnnxRouter:
    """Exactly the training input: head 6,000 + tail 2,000 characters for huge pastes, then the
    first 360 + "…" + last 149 tokens between [CLS] and [SEP]."""

    def __init__(self, folder: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.config = json.loads((folder / "router_config.json").read_text(encoding="utf-8"))
        if self.config["labels"] != LABELS:
            raise ValueError(f"router labels {self.config['labels']} != {LABELS}")
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.marker = [self.tokenizer.token_to_id("…")]
        self.cls, self.sep = self.tokenizer.token_to_id("[CLS]"), self.tokenizer.token_to_id("[SEP]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(folder / "router.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.train_prior = np.array(self.config["train_prior"])
        self.natural_prior = np.array(self.config["natural_prior"])
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def ids(self, text: str) -> list[int]:
        text = text[:6000] + ("\n" + text[-2000:] if len(text) > 8000 else "")
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        cut = cut_token_ids(ids, self.config["head_tokens"], self.config["tail_tokens"], self.marker)
        return [self.cls, *cut, self.sep]

    def probs(self, text: str) -> np.ndarray:
        """Raw model probabilities (no prior correction)."""
        ids = np.array([self.ids(text)], dtype=np.int64)
        logits = self.session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True))[0]

    def classify(self, text: str) -> tuple[str, float]:
        """(label, probability) for a prepared text, prior-corrected; cached by text hash."""
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        probs = prior_correct(self.probs(text)[None, :], self.train_prior, self.natural_prior)[0]
        best = int(probs.argmax())
        result = (LABELS[best], float(probs[best]))
        self._cache[key] = result
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return result


def load_router(folder: str | None = None, repo: str | None = None) -> OnnxRouter | None:
    """The router from a local folder, else from the Hugging Face repo; None if unavailable
    (the gateway then adds no briefs: fail open)."""
    try:
        path = Path(folder).expanduser() if folder else None
        if path is None or not all((path / f).exists() for f in MODEL_FILES):
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(repo_id=repo or DEFAULT_REPO, allow_patterns=list(MODEL_FILES),
                                          local_dir=str(DEFAULT_CACHE)))
        router = OnnxRouter(path)
        log.info("prompt router loaded from %s", path)
        return router
    except Exception:
        log.warning("prompt router unavailable; no task briefs", exc_info=True)
        return None
