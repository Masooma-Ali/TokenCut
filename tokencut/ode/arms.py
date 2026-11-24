"""ODE's options ("arms"): 4 models x 2 trim levels, with their prices and context windows.

Prices are list prices in USD per million tokens, checked on 2026-11-24 against
  Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
  OpenAI:    https://developers.openai.com/api/docs/models/gpt-6.1-sol
Two prices depend on the size of the request (each request is priced on its own):
  - Claude Haiku 5.5: a prompt over 100,000 tokens (input + cache reads + cache writes) pays the
    higher row for every token of that request;
  - GPT-6.1 Sol: a prompt over 272,000 input tokens pays 2x input and cache, 1.5x output.
Context windows: the Claude 5.5 models have 1M tokens (Claude API model table); GPT-6.1 Sol takes
at most 922,000 input tokens (1,050,000 context including output).

The team runs on subscriptions, so these prices give a list-price-equivalent cost: billed tokens
from the gateway logs x this table. When a price changes, change it here and log it in
docs/DECISIONS.md: rewards computed with old prices are then not comparable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

PRICES_CHECKED = "2026-11-24"
TRIMS = ("standard", "aggressive")  # aggressive = standard + P1 (lazy tools, tool pruning, CCR)


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float
    output: float

    def cache_write(self, ttl: str) -> float:
        if ttl == "5m":
            return self.cache_write_5m
        if ttl == "1h":
            return self.cache_write_1h
        raise ValueError(f"cache ttl must be '5m' or '1h', got {ttl!r}")


@dataclass(frozen=True)
class Model:
    key: str  # short name used in arm names
    model_id: str  # what the gateway writes into the request's `model` field
    provider: str  # "anthropic" or "openai"
    max_input_tokens: int  # largest prompt the model accepts
    price: Price
    long_price: Price | None = None  # prices for a request over `long_threshold`
    long_threshold: int | None = None  # prompt tokens above which `long_price` applies
    effort: str | None = None  # reasoning effort the arm fixes (None: the agent's own setting)

    def price_for(self, prompt_tokens: int) -> Price:
        """Prices for one request whose prompt (input + cache reads + cache writes) has this size."""
        if self.long_price is not None and self.long_threshold is not None \
                and prompt_tokens > self.long_threshold:
            return self.long_price
        return self.price


HAIKU = Model(
    "haiku", "claude-haiku-5-5", "anthropic", 1_000_000,
    Price(0.10, 0.125, 0.20, 0.01, 0.50),
    long_price=Price(0.50, 0.625, 1.00, 0.05, 2.50), long_threshold=100_000)
SONNET = Model(
    "sonnet", "claude-sonnet-5-5", "anthropic", 1_000_000,
    Price(2.00, 2.50, 4.00, 0.10, 10.00))
OPUS = Model(
    "opus", "claude-opus-5-5", "anthropic", 1_000_000,
    Price(4.00, 5.00, 8.00, 0.20, 20.00))
# OpenAI bills no cache-write premium: tokens written to its cache cost the normal input price.
GPT = Model(
    "gpt", "gpt-6.1-sol", "openai", 922_000,
    Price(2.00, 2.00, 2.00, 0.10, 10.00),
    long_price=Price(4.00, 4.00, 4.00, 0.20, 15.00), long_threshold=272_000,
    effort="medium")
MODELS = (HAIKU, SONNET, OPUS, GPT)


@dataclass(frozen=True)
class Arm:
    model: Model
    trim: str  # "standard" or "aggressive"

    @property
    def name(self) -> str:
        return f"{self.model.key}-{self.trim}"

    @property
    def features(self) -> list[float]:
        """w_a: [aggressive 0/1, log of the base input price]."""
        return [1.0 if self.trim == "aggressive" else 0.0, math.log(self.model.price.input)]


ARMS = tuple(Arm(m, t) for m in MODELS for t in TRIMS)
ARM_NAMES = [a.name for a in ARMS]
ARM_FEATURE_NAMES = ["aggressive", "log_input_price"]
WINDOW_MARGIN = 0.20  # use at most 80% of a model's window (Phase 0 value)


def arm(name: str) -> Arm:
    for a in ARMS:
        if a.name == name:
            return a
    raise ValueError(f"unknown arm {name!r}; arms are {ARM_NAMES}")


def arm_features(arms: Sequence[Arm] = ARMS) -> list[list[float]]:
    """The arm_features table HybridLinUCB takes, in arm order."""
    return [a.features for a in arms]


def fits(a: Arm, context_tokens: int, margin: float = WINDOW_MARGIN) -> bool:
    """True if the conversation leaves the margin free in this arm's context window."""
    return context_tokens <= (1.0 - margin) * a.model.max_input_tokens


def allowed_arms(context_tokens: int, arms: Sequence[Arm] = ARMS,
                 margin: float = WINDOW_MARGIN) -> list[str]:
    """Arms whose window fits the context (the other masks come from the gateway in Phase 6)."""
    return [a.name for a in arms if fits(a, context_tokens, margin)]


def rebuild_cost_usd(a: Arm, context_tokens: int, ttl: str = "1h") -> float:
    """Extra cost of moving the conversation to this arm: its first request writes the whole
    context to the new cache instead of reading it from the old one.

    Any arm change costs this, a trim change too: aggressive trimming changes the tool list,
    which sits at the start of the cached prefix."""
    price = a.model.price_for(context_tokens)
    return context_tokens * (price.cache_write(ttl) - price.cache_read) / 1e6


def switch_costs(current: str | None, context_tokens: int, c_ref: float, lam: float,
                 arms: Sequence[Arm] = ARMS, ttl: str = "1h") -> dict[str, float]:
    """Switch cost of every other arm in reward units (lambda x dollars / C_ref), for choose()."""
    if c_ref <= 0 or lam < 0:
        raise ValueError("need c_ref > 0 and lam >= 0")
    if current is None:
        return {}
    arm(current)
    return {a.name: lam * rebuild_cost_usd(a, context_tokens, ttl) / c_ref
            for a in arms if a.name != current}
