"""ODE hooks in the gateway (Phase 3): carry out an arm (model x trim level) for a request.

ODE does not choose yet (Phase 6). Phase 3 adds what a choice needs, driven by a forced arm:

    TOKENCUT_ODE=force                  ODE hooks on; every agent request uses the forced arm
    TOKENCUT_ODE_FORCE_ARM=opus-standard    which arm (names in tokencut/ode/arms.py)
    TOKENCUT_ODE_RESTORE_MODEL=1        reply says the model the agent asked for (see below)

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
from dataclasses import dataclass

from tokencut.gateway.adapters import anthropic
from tokencut.gateway.history import CacheEventConfig
from tokencut.gateway.note import _is_agent_request
from tokencut.gateway.routing import RouteConfig
from tokencut.gateway.virtual import VirtualConfig
from tokencut.ode import features as F
from tokencut.ode.arms import Arm, arm

MODES = ("off", "force")


@dataclass(frozen=True)
class OdeConfig:
    mode: str = "off"
    force_arm: str | None = None
    restore_model: bool = False

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"TOKENCUT_ODE must be one of {MODES}, not {self.mode!r}")
        if self.mode == "force":
            if not self.force_arm:
                raise ValueError("TOKENCUT_ODE=force needs TOKENCUT_ODE_FORCE_ARM")
            arm(self.force_arm)  # raises for an unknown arm name

    @property
    def on(self) -> bool:
        return self.mode != "off"

    @classmethod
    def from_env(cls) -> OdeConfig:
        return cls(mode=os.environ.get("TOKENCUT_ODE", "off").strip() or "off",
                   force_arm=os.environ.get("TOKENCUT_ODE_FORCE_ARM", "").strip() or None,
                   restore_model=os.environ.get("TOKENCUT_ODE_RESTORE_MODEL", "") == "1")


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
              route: RouteConfig) -> tuple[bytes | None, Applied | None]:
    """(new body or None if unchanged, what was done or None if ODE does not touch it)."""
    if not config.on or provider != "anthropic":
        return None, None
    data = json.loads(body)
    if not isinstance(data, dict) or not _is_agent_request(data):
        return None, None  # side request (title, summary) or not a JSON object
    requested = str(data.get("model", ""))
    chosen = arm(config.force_arm)
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


def task_numbers(body: bytes, router,
                 threshold: float = F.settings.DECISION_THRESHOLD) -> dict | None:
    """The 13 numbers when this request starts a typed task, else None. Numbers only."""
    data = json.loads(body)
    if router is None or not isinstance(data, dict) or not starts_turn(data):
        return None
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
