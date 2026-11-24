"""ODE phase 2: the arm table (prices checked 2026-11-24 on the official pricing pages)."""

import math

import pytest

from tokencut.ode import arms as A
from tokencut.ode.bandit import HybridLinUCB


def test_eight_arms_in_a_fixed_order():
    assert A.ARM_NAMES == [
        "haiku-standard", "haiku-aggressive", "sonnet-standard", "sonnet-aggressive",
        "opus-standard", "opus-aggressive", "gpt-standard", "gpt-aggressive"]


def test_prices_match_the_official_pages():
    assert A.HAIKU.price == A.Price(0.10, 0.125, 0.20, 0.01, 0.50)
    assert A.HAIKU.long_price == A.Price(0.50, 0.625, 1.00, 0.05, 2.50)
    assert A.SONNET.price == A.Price(2.00, 2.50, 4.00, 0.10, 10.00)
    assert A.OPUS.price == A.Price(4.00, 5.00, 8.00, 0.20, 20.00)
    assert A.GPT.price.input == 2.00 and A.GPT.price.cache_read == 0.10
    assert A.GPT.price.output == 10.00
    assert [m.model_id for m in A.MODELS] == [
        "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "gpt-6.1-sol"]


def test_long_prompt_tiers():
    assert A.HAIKU.price_for(100_000) is A.HAIKU.price  # "over 100,000" is strict
    assert A.HAIKU.price_for(100_001) is A.HAIKU.long_price
    assert A.GPT.price_for(272_000) is A.GPT.price
    assert A.GPT.price_for(272_001).input == 4.00
    assert A.GPT.price_for(272_001).output == 15.00
    assert A.OPUS.price_for(900_000) is A.OPUS.price  # Opus has no tier


def test_arm_features_are_trim_flag_and_log_price():
    assert A.arm("haiku-standard").features == [0.0, math.log(0.10)]
    assert A.arm("opus-aggressive").features == [1.0, math.log(4.00)]
    assert len(A.arm_features()) == 8


def test_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="haiku-standrad"):
        A.arm("haiku-standrad")


def test_window_mask_keeps_a_20_percent_margin():
    assert A.allowed_arms(500_000) == A.ARM_NAMES
    # 800k = 80% of Claude's 1M: still fits; GPT's 922k x 0.8 = 737.6k does not.
    assert A.allowed_arms(800_000) == A.ARM_NAMES[:6]
    assert A.allowed_arms(850_000) == []


def test_rebuild_cost_uses_write_minus_read():
    # Sonnet, 40k context, 1h cache: 40,000 x (4.00 - 0.10) / 1e6 = $0.156
    assert A.rebuild_cost_usd(A.arm("sonnet-standard"), 40_000) == pytest.approx(0.156)
    # 5-minute cache: 40,000 x (2.50 - 0.10) / 1e6 = $0.096
    assert A.rebuild_cost_usd(A.arm("sonnet-standard"), 40_000, ttl="5m") == pytest.approx(0.096)
    # Haiku over 100k pays the long-prompt row: 120,000 x (1.00 - 0.05) / 1e6 = $0.114
    assert A.rebuild_cost_usd(A.arm("haiku-standard"), 120_000) == pytest.approx(0.114)


def test_switch_costs_in_reward_units():
    costs = A.switch_costs("sonnet-standard", 40_000, c_ref=1.0, lam=0.5)
    assert "sonnet-standard" not in costs  # staying is free
    assert set(costs) == set(A.ARM_NAMES) - {"sonnet-standard"}
    assert costs["opus-standard"] == pytest.approx(0.5 * 40_000 * (8.00 - 0.20) / 1e6)
    assert costs["sonnet-aggressive"] == pytest.approx(0.5 * 0.156)  # trim change rewrites too
    assert A.switch_costs(None, 40_000, c_ref=1.0, lam=0.5) == {}
    with pytest.raises(ValueError):
        A.switch_costs("sonnet-standard", 40_000, c_ref=0.0, lam=0.5)


def test_arm_table_plugs_into_the_bandit():
    bandit = HybridLinUCB(A.ARM_NAMES, A.arm_features(), n_features=3, shared=(0, 1, 2))
    costs = A.switch_costs("sonnet-standard", 40_000, c_ref=1.0, lam=0.5)
    dec = bandit.choose([0.5, 0.5, 1.0], allowed=A.allowed_arms(40_000),
                        current="sonnet-standard", switch_cost=costs)
    assert dec.arm in A.ARM_NAMES
