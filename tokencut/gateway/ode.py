"""ODE in the gateway: choose and carry out an arm (model x trim level) per task.

    TOKENCUT_ODE=force                  every agent request uses TOKENCUT_ODE_FORCE_ARM (practice
                                        runs, phases 3-5)
    TOKENCUT_ODE=shadow                 ODE chooses at each new typed task and LOGS its choice;
                                        requests go out unchanged (phase 6, step 1)
    TOKENCUT_ODE=on                     ODE chooses and the task's requests use its arm (step 2)
    TOKENCUT_ODE_FORCE_ARM=opus-standard    the forced arm (names in tokencut/ode/arms.py)
    TOKENCUT_ODE_PRIOR=path.npz         learned prior (default: eval/results/ode/ode_prior.npz)
    TOKENCUT_ODE_ARMS=a,b,...           arms ODE may choose (default: every arm except Opus,
                                        which had only 12 practice turns; GPT arms are left out
                                        when no LiteLLM route is set up)
    TOKENCUT_ODE_EXPLORE=0.05           share of random tries (probability logged)
    TOKENCUT_ODE_ALPHA=0                unsure bonus; 0 while there is no live learning
    TOKENCUT_ODE_RESTORE_MODEL=1        reply says the model the agent asked for (see below)

Choosing (shadow / on): at a request that starts a typed task (the router does not call it a
follow-up or a machine message), the 13 task numbers are scaled with the prior's standardizer
and the bandit picks among the allowed arms, keeping the conversation's current arm unless
another wins by more than the switch margin. Every later request of that task (tool loops)
uses the same arm: a decision per task, never mid-task (team decision; a switch costs a cache
rewrite). No learning live yet: the prior is fixed (a live reward is still to be designed and
checked against graded runs). A missing or broken prior turns choosing off (fail open).

For an agent request (Claude Code's main loop: Anthropic format, at least two client tools):
  1. the `model` field is swapped to the arm's model BEFORE trimming and routing, so the
     cache-event memory sees the real model (a switch is a cache event) and a GPT arm goes to
     the LiteLLM route (Claude Code itself never sees a GPT name);
  2. the arm's trim level replaces the gateway-wide P1 settings for this request
     (aggressive = lazy tools + tool pruning at cache events + CCR; standard = none of them);
  3. at the start of a typed task the 13 task numbers are logged (numbers only).
Side requests (titles, summaries) are left alone. A GPT arm without a LiteLLM route set up is
skipped (request unchanged). Any error: the request goes out unchanged (fail open).

Reply model name: the reply names the model that answered. Whether Claude Code minds a reply
from another model than it asked for is not known yet (to check live in Phase 3);
TOKENCUT_ODE_RESTORE_MODEL=1 writes the requested name back into the reply.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from tokencut.gateway.adapters import anthropic
from tokencut.gateway.history import CacheEventConfig
from tokencut.gateway.note import _is_agent_request
from tokencut.gateway.routing import RouteConfig
from tokencut.gateway.session import conversation_key
from tokencut.gateway.virtual import VirtualConfig
from tokencut.ode import features as F
from tokencut.ode import settings
from tokencut.ode.arms import ARM_NAMES, Arm, arm
from tokencut.ode.bandit import HybridLinUCB
from tokencut.ode.prior import load_prior, prior_rows

MODES = ("off", "force", "shadow", "on")
DEFAULT_PRIOR = Path(__file__).resolve().parents[2] / "eval" / "results" / "ode" / "ode_prior.npz"
# Opus is left out until it has practice data (team decision 2026-11-27: only 12 pilot turns,
# so its unsure bonus made the prior pick it for every task type).
DEFAULT_ARMS = tuple(a for a in ARM_NAMES if not a.startswith("opus"))


@dataclass(frozen=True)
class OdeConfig:
    mode: str = "off"
    force_arm: str | None = None
    restore_model: bool = False
    prior_path: str | None = None  # None: DEFAULT_PRIOR
    arms: tuple[str, ...] = DEFAULT_ARMS  # arms ODE may choose (shadow / on)
    explore: float = settings.EXPLORE
    # Unsure bonus. 0 while ODE does not learn live: the bonus is for trying arms in order to
    # learn from them; without updates it would only add cost, forever. 0 is also the policy
    # the phase 5b leave-one-task-out check measured. Set settings.ALPHA (1) with live learning.
    alpha: float = 0.0
    seed: int | None = None  # random tries; fixed only in tests

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"TOKENCUT_ODE must be one of {MODES}, not {self.mode!r}")
        if self.mode == "force":
            if not self.force_arm:
                raise ValueError("TOKENCUT_ODE=force needs TOKENCUT_ODE_FORCE_ARM")
            arm(self.force_arm)  # raises for an unknown arm name
        for name in self.arms:
            arm(name)
        if not self.arms:
            raise ValueError("TOKENCUT_ODE_ARMS needs at least one arm")
        if self.alpha < 0:
            raise ValueError(f"TOKENCUT_ODE_ALPHA must be >= 0, not {self.alpha}")
        if not 0 <= self.explore <= 1:
            raise ValueError(f"TOKENCUT_ODE_EXPLORE must be between 0 and 1, not {self.explore}")

    @property
    def on(self) -> bool:
        return self.mode != "off"

    @property
    def chooses(self) -> bool:
        return self.mode in ("shadow", "on")

    @classmethod
    def from_env(cls) -> OdeConfig:
        arms = tuple(a.strip() for a in os.environ.get("TOKENCUT_ODE_ARMS", "").split(",")
                     if a.strip()) or DEFAULT_ARMS
        explore = os.environ.get("TOKENCUT_ODE_EXPLORE", "").strip()
        alpha = os.environ.get("TOKENCUT_ODE_ALPHA", "").strip()
        return cls(mode=os.environ.get("TOKENCUT_ODE", "off").strip() or "off",
                   force_arm=os.environ.get("TOKENCUT_ODE_FORCE_ARM", "").strip() or None,
                   restore_model=os.environ.get("TOKENCUT_ODE_RESTORE_MODEL", "") == "1",
                   prior_path=os.environ.get("TOKENCUT_ODE_PRIOR", "").strip() or None,
                   arms=arms, explore=float(explore) if explore else settings.EXPLORE,
                   alpha=float(alpha) if alpha else 0.0)


@dataclass
class Chooser:
    """ODE's live choices: the prior's bandit (fixed, no learning yet) and, per conversation,
    the arm of the task in progress."""

    bandit: HybridLinUCB
    std: F.Standardizer
    arms: tuple[str, ...]
    prior: str = ""
    rows: int = 0
    max_conversations: int = 1000
    current: OrderedDict = field(default_factory=OrderedDict)
    counts: Counter = field(default_factory=Counter)
    explored: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def load(cls, config: OdeConfig, gpt_route: bool) -> Chooser:
        path = Path(config.prior_path) if config.prior_path else DEFAULT_PRIOR
        bandit, std, _, _ = load_prior(path, seed=config.seed)
        bandit.explore, bandit.alpha = config.explore, config.alpha
        arms = tuple(a for a in config.arms if a in bandit.arms
                     and (gpt_route or arm(a).model.provider != "openai"))
        if not arms:
            raise ValueError(f"none of the arms {config.arms} can be used")
        return cls(bandit, std, arms, prior=str(path), rows=prior_rows(path))

    def decide(self, key: str | None, numbers: dict | None) -> tuple[str | None, dict]:
        """(arm for this request or None, what happened). A new decision only where a typed
        task starts; otherwise the conversation's current arm (None before its first task)."""
        with self.lock:  # requests of different conversations run in parallel threads
            return self._decide(key, numbers)

    def _decide(self, key: str | None, numbers: dict | None) -> tuple[str | None, dict]:
        current = self.current.get(key) if key else None
        if (numbers is None or numbers.get("task") is None or not numbers.get("decision_point")
                or "x" not in numbers):
            return current, {"decided": False, "arm": current}
        d = self.bandit.choose(self.std.transform(numbers["x"]), allowed=self.arms,
                               current=current if current in self.arms else None)
        if key:
            self.current[key] = d.arm
            self.current.move_to_end(key)
            while len(self.current) > self.max_conversations:
                self.current.popitem(last=False)
        self.counts[d.arm] += 1
        self.explored += d.explored
        return d.arm, {
            "decided": True, "arm": d.arm, "previous": current, "greedy": d.greedy,
            "explored": d.explored, "probability": round(d.probability, 6),
            "scores": {a: {"estimate": round(s.estimate, 4), "bonus": round(s.bonus, 4),
                           "total": round(s.total, 4)} for a, s in d.scores.items()}}

    def summary(self) -> dict:
        return {"prior": self.prior, "prior_rows": self.rows, "arms": list(self.arms),
                "decisions": sum(self.counts.values()),
                "by_arm": dict(sorted(self.counts.items())),
                "random_tries": self.explored, "conversations": len(self.current)}


def choose(provider: str, body: bytes, chooser: Chooser,
           router) -> tuple[str | None, dict | None, dict | None]:
    """For an agent request: (arm or None, choice log, task numbers). (None, None, None) for
    any other request (side requests are never touched). Runs the router: call in a thread."""
    if provider != "anthropic":
        return None, None, None
    data = json.loads(body)
    if not isinstance(data, dict) or not _is_agent_request(data):
        return None, None, None
    numbers = task_numbers(body, router)
    key = conversation_key(provider, anthropic.parse(data))
    name, choice = chooser.decide(key, numbers)
    return name, choice, numbers


def arm_settings(a: Arm, events: CacheEventConfig,
                 virtual: VirtualConfig) -> tuple[CacheEventConfig, VirtualConfig]:
    """The gateway settings for this arm's trim level (the CCR store is shared, not copied)."""
    aggressive = a.trim == "aggressive"
    return (dataclasses.replace(events, prune_tools=aggressive),
            dataclasses.replace(virtual, ccr=aggressive, lazy_tools=aggressive))


@dataclass(frozen=True)
class Applied:
    """What ODE did to one request (all of it goes into the numbers-only log)."""

    arm: Arm | None
    requested: str  # model the agent asked for
    sent: str  # model the request went to
    skipped: str = ""  # why the arm was not applied ("" = applied)

    def log(self) -> dict:
        return {"arm": self.arm.name if self.arm else None,
                "trim": self.arm.trim if self.arm and not self.skipped else None,
                "model_requested": self.requested, "model_sent": self.sent,
                "skipped": self.skipped or None}


def apply_arm(provider: str, body: bytes, config: OdeConfig,
              route: RouteConfig, name: str | None = None) -> tuple[bytes | None, Applied | None]:
    """(new body or None if unchanged, what was done or None if ODE does not touch it).
    `name`: the arm to use (default: the forced arm)."""
    if not config.on or provider != "anthropic":
        return None, None
    data = json.loads(body)
    if not isinstance(data, dict) or not _is_agent_request(data):
        return None, None  # side request (title, summary) or not a JSON object
    requested = str(data.get("model", ""))
    chosen = arm(name or config.force_arm)
    if chosen.model.provider == "openai" and not route.gpt_upstream:
        return None, Applied(chosen, requested, requested, skipped="gpt_route_not_set_up")
    if requested == chosen.model.model_id:
        return None, Applied(chosen, requested, requested)
    data["model"] = chosen.model.model_id
    new = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()
    return new, Applied(chosen, requested, chosen.model.model_id)


def starts_turn(data: dict) -> bool:
    """The request ends with a typed user message (no tool results): the agent's next turn.

    System messages the agent appends after it (Claude Code's mid-conversation `role: system`
    context) are skipped: they are not the user's."""
    messages = data.get("messages")
    if not isinstance(messages, list):
        return False
    rest = [m for m in messages if not (isinstance(m, dict) and m.get("role") == "system")]
    if not rest:
        return False
    last = rest[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return False
    content = last.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    return isinstance(content, list) and not any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


# Requests Claude Code makes on its own that end with a "user" message and carry the main
# loop's tools, so they look like a new typed task. Seen live (phase 6 demo, 2026-11-27): after
# every answer Claude Code forks the conversation with its prompt-suggestion instruction (the
# grey "what you might type next" hint); the text is in Claude Code's own program and starts
# with this marker. Not a task: ODE keeps the conversation's arm (its cache is warm there).
AGENT_MACHINE_PREFIXES = {
    "[SUGGESTION MODE:": "prompt_suggestion",
}


def last_user_text(data: dict) -> str:
    """Text of the request's last non-system message, if it is a user message."""
    messages = [m for m in data.get("messages") or []
                if not (isinstance(m, dict) and m.get("role") == "system")]
    if not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "user":
        return ""
    content = messages[-1].get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text")
    return ""


def machine_message(data: dict) -> str | None:
    """Which agent-made message this is (see AGENT_MACHINE_PREFIXES), or None."""
    text = last_user_text(data).lstrip()
    for prefix, name in AGENT_MACHINE_PREFIXES.items():
        if text.startswith(prefix):
            return name
    return None


def task_numbers(body: bytes, router,
                 threshold: float = F.settings.DECISION_THRESHOLD) -> dict | None:
    """The 13 numbers when this request starts a typed task, else None. Numbers only."""
    data = json.loads(body)
    if router is None or not isinstance(data, dict) or not starts_turn(data):
        return None
    machine = machine_message(data)
    if machine is not None:
        return {"task": None, "machine": machine}  # not the user's: the same task goes on
    out = F.request_features(anthropic.parse(data), router)
    if out is None:
        return {"task": None}  # a follow-up ("yes") or a machine message: same task goes on
    x, probs = out
    top = max(probs, key=probs.get)
    return {"task": top, "p": round(probs[top], 4),
            "decision_point": F.is_decision_point(probs, threshold),
            "x": [round(float(v), 4) for v in x]}


class ModelNameRestorer:
    """Writes the requested model name back into the reply's first `"model":"…"`.

    Streams: only the first event (message_start) is held back until it is complete; the rest
    passes through untouched. Plain JSON: the body is held until the end (side requests are
    never swapped, and Claude Code streams its main loop)."""

    MAX_HOLD = 1_000_000

    def __init__(self, sent: str, requested: str, streaming: bool):
        self._pattern = re.compile(rb'("model"\s*:\s*")' + re.escape(sent.encode()) + rb'"')
        self._replacement = rb"\g<1>" + requested.encode().replace(b"\\", b"\\\\") + b'"'
        self._streaming = streaming
        self._buffer = bytearray()
        self._done = False
        self.restored = False

    def _rewrite(self, data: bytes) -> bytes:
        new, n = self._pattern.subn(self._replacement, data, count=1)
        self.restored = n > 0
        return new

    def feed(self, chunk: bytes) -> bytes:
        if self._done:
            return chunk
        self._buffer.extend(chunk)
        if self._streaming:
            end = self._buffer.find(b"\n\n")
            if end < 0 and len(self._buffer) < self.MAX_HOLD:
                return b""
            cut = len(self._buffer) if end < 0 else end + 2
            head, rest = bytes(self._buffer[:cut]), bytes(self._buffer[cut:])
            self._done, self._buffer = True, bytearray()
            return self._rewrite(head) + rest
        if len(self._buffer) >= self.MAX_HOLD:  # too big to hold: pass it on unchanged
            self._done = True
            out, self._buffer = bytes(self._buffer), bytearray()
            return out
        return b""

    def close(self) -> bytes:
        if self._done:
            return b""
        self._done = True
        out, self._buffer = bytes(self._buffer), bytearray()
        return self._rewrite(out)
