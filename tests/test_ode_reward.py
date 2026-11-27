"""ODE phase 2: cost from billed usage and the reward, checked against hand calculations."""

import pytest

from tokencut.gateway.usage import Usage
from tokencut.ode import arms as A
from tokencut.ode.reward import (
    model_for,
    normalized_cost,
    request_cost,
    reward,
    task_cost,
)


def test_sonnet_request_cost_by_hand():
    # 2,000 input x $2 + 30,000 cache reads x $0.10 + 1,000 1h writes x $4 + 500 output x $10
    # = 0.004 + 0.003 + 0.004 + 0.005 = $0.016
    usage = Usage(input_tokens=2_000, cache_read_tokens=30_000, cache_write_tokens=1_000,
                  output_tokens=500)
    assert request_cost(A.SONNET, usage) == pytest.approx(0.016)
    # With the 5-minute cache the writes cost $2.50: 0.016 - 0.004 + 0.0025 = $0.0145
    assert request_cost(A.SONNET, usage, ttl="5m") == pytest.approx(0.0145)


def test_haiku_price_jump_counts_cache_reads_in_the_prompt():
    small = Usage(input_tokens=1_000, cache_read_tokens=99_000, output_tokens=1_000)
    big = Usage(input_tokens=1_000, cache_read_tokens=99_001, output_tokens=1_000)
    # 1,000 x 0.10 + 99,000 x 0.01 + 1,000 x 0.50 = 0.0001 + 0.00099 + 0.0005 = $0.00159
    assert request_cost(A.HAIKU, small) == pytest.approx(0.00159)
    # 100,001 prompt tokens: every token at the higher row, 5x
    # 1,000 x 0.50 + 99,001 x 0.05 + 1,000 x 2.50 = 0.0005 + 0.00495005 + 0.0025 = $0.00795005
    assert request_cost(A.HAIKU, big) == pytest.approx(0.00795005)


def test_gpt_long_prompt_tier():
    usage = Usage(input_tokens=300_000, output_tokens=1_000)
    # over 272k: input $4, output $15 -> 1.2 + 0.015 = $1.215
    assert request_cost(A.GPT, usage) == pytest.approx(1.215)


def test_task_cost_sums_requests_and_handles_a_model_switch():
    requests = [
        ("claude-sonnet-5-5", Usage(input_tokens=1_000, output_tokens=100)),   # 0.002 + 0.001
        ("claude-opus-5-5", Usage(cache_write_tokens=10_000, output_tokens=100)),  # 0.08 + 0.002
    ]
    assert task_cost(requests) == pytest.approx(0.003 + 0.082)
    assert task_cost([]) == 0.0


def test_model_ids_and_dated_variants():
    assert model_for("claude-opus-5-5") is A.OPUS
    assert model_for("claude-opus-5-5-20261101") is A.OPUS
    assert model_for("gpt-6.1-sol") is A.GPT
    with pytest.raises(ValueError, match="claude-opus-5"):
        model_for("claude-opus-5")  # a different model, never priced as 5.5
    # Claude Code's side requests to Sonnet 5 (pilot): priced, and never confused with 5.5
    assert model_for("claude-sonnet-5").price.cache_read == 0.20
    assert model_for("claude-sonnet-5-5") is A.SONNET


def test_reward_formula_and_cap():
    assert normalized_cost(0.5, c_ref=1.0) == 0.5
    assert normalized_cost(3.0, c_ref=1.0) == 1.0  # capped
    assert reward(True, 0.5, c_ref=1.0, lam=0.5) == pytest.approx(0.75)
    assert reward(False, 0.5, c_ref=1.0, lam=0.5) == pytest.approx(-0.25)
    assert reward(True, 0.0, c_ref=1.0) == 1.0


def test_no_success_signal_means_no_reward_not_a_failure():
    assert reward(None, 0.5, c_ref=1.0) is None


def test_bad_values_are_refused():
    with pytest.raises(ValueError):
        normalized_cost(1.0, c_ref=0.0)
    with pytest.raises(ValueError):
        normalized_cost(float("nan"), c_ref=1.0)
    with pytest.raises(ValueError):
        reward(True, 1.0, c_ref=1.0, lam=-1.0)
    with pytest.raises(ValueError):
        request_cost(A.SONNET, Usage(input_tokens=-5))
    with pytest.raises(ValueError):
        request_cost(A.SONNET, Usage(cache_write_tokens=10), ttl="10m")
