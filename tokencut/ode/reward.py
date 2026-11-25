"""Scoring a finished task: list-price cost from billed usage, then r = s - lambda * c~.

    cost of a request = input x input price + cache reads x read price
                      + cache writes x write price + output x output price   (reasoning included)
    C_task = sum over the task's requests          c~ = min(1, C_task / C_ref)
    r      = s - lambda * c~                        s = 1 solved, 0 not solved

Each request is priced on its own (Haiku's >100k and GPT's >272k rows apply per request). The
cache rebuild after a model switch and the tokens TokenCut adds are already in the billed usage,
so no hand-made penalty is needed. A task with no success signal gets no reward (None), never 0:
"unknown" must not teach ODE that the model failed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import fields

from tokencut.gateway.usage import Usage
from tokencut.ode.arms import MODELS, Model, Price

LAMBDA = 0.5  # Phase 0 value: lambda x C_ref = the most we pay for one extra solved task
# Used only when the usage does not split cache writes by lifetime (usage.py reads the split
# from Anthropic's `cache_creation`). Claude Code marks its cache "ttl": "1h" (CLAUDE.md).
DEFAULT_CACHE_TTL = "1h"


# Models that are not arms but that agents still call for side requests (seen in the pilot:
# Claude Code sent some side requests to Claude Sonnet 5). Same price page and date as arms.py.
SIDE_MODELS = (
    Model("sonnet5", "claude-sonnet-5", "anthropic", 1_000_000, Price(2.00, 2.50, 4.00, 0.20, 10.00)),
)


def model_for(model_id: str) -> Model:
    """The priced model for a request's `model` field (exact id, or a dated variant of it)."""
    for m in (*MODELS, *SIDE_MODELS):
        if model_id == m.model_id or model_id.startswith(m.model_id + "-"):
            return m
    raise ValueError(f"no price for model {model_id!r}")


def prompt_tokens(usage: Usage) -> int:
    return usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens


def usage_from_dict(data: dict) -> Usage:
    """A Usage from a request log line's "usage" (unknown keys ignored, missing ones 0)."""
    known = {f.name for f in fields(Usage)}
    return Usage(**{k: int(v) for k, v in data.items() if k in known})


def request_cost(model: Model, usage: Usage, ttl: str = DEFAULT_CACHE_TTL) -> float:
    """USD at list price for one request's billed usage."""
    for name, value in usage.as_dict().items():
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    p = model.price_for(prompt_tokens(usage))
    split = usage.cache_write_5m_tokens + usage.cache_write_1h_tokens
    if split and split == usage.cache_write_tokens:  # the provider said which cache was written
        writes = usage.cache_write_5m_tokens * p.cache_write_5m \
            + usage.cache_write_1h_tokens * p.cache_write_1h
    else:
        writes = usage.cache_write_tokens * p.cache_write(ttl)
    return (usage.input_tokens * p.input + usage.cache_read_tokens * p.cache_read
            + writes + usage.output_tokens * p.output) / 1e6


def task_cost(requests: Iterable[tuple[str, Usage]], ttl: str = DEFAULT_CACHE_TTL) -> float:
    """USD for a task: (model id, usage) for every request between its start and end."""
    return sum(request_cost(model_for(model_id), usage, ttl) for model_id, usage in requests)


def normalized_cost(cost: float, c_ref: float) -> float:
    """c~ = min(1, cost / C_ref): 0 = free, 1 = as expensive as a worst-case Opus task or more."""
    if not c_ref > 0:
        raise ValueError(f"C_ref must be > 0, got {c_ref}")
    if not (math.isfinite(cost) and cost >= 0):
        raise ValueError(f"cost must be finite and >= 0, got {cost}")
    return min(1.0, cost / c_ref)


def reward(success: bool | None, cost: float, c_ref: float, lam: float = LAMBDA) -> float | None:
    """r = s - lambda * c~, or None when there is no success signal (then ODE does not update)."""
    if lam < 0:
        raise ValueError(f"lambda must be >= 0, got {lam}")
    c = normalized_cost(cost, c_ref)
    if success is None:
        return None
    return (1.0 if success else 0.0) - lam * c
