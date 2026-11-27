"""ODE phase 6: the gateway chooses the arm itself (TOKENCUT_ODE=shadow / on). A small prior
built here, a fake router and a fake provider: no network, no API calls."""

import json
import time

import httpx2
import pytest
from fastapi.testclient import TestClient

from eval.ode_learn import new_bandit
from tests.test_ode_gateway import FakeRouter, Upstream, agent_body, last_record, post
from tokencut.gateway.app import GatewayConfig, create_app
from tokencut.gateway.ode import DEFAULT_ARMS, Chooser, OdeConfig, task_numbers
from tokencut.gateway.routing import RouteConfig
from tokencut.ode import arms as A
from tokencut.ode.features import Standardizer
from tokencut.ode.prior import save_prior

LITELLM = "http://127.0.0.1:4000"
BUGFIX = FakeRouter(bugfix=0.95, no_brief=0.05)


def x_of(text: str, router=BUGFIX) -> list[float]:
    return task_numbers(json.dumps(agent_body(text)).encode(), router)["x"]


def make_prior(tmp_path, rewards: dict[str, float]) -> str:
    """A prior that learned `rewards` for bug-fix messages (unlisted arms: 0.2)."""
    rows = [x_of("fix the bug in shop/cart.py"), x_of("the total is wrong, please fix it now"),
            x_of("crash in a.py and b.py when the cart is empty, fix")]
    std = Standardizer.fit(rows)
    bandit = new_bandit()
    for _ in range(30):
        for x in rows:
            for name in A.ARM_NAMES:
                bandit.update(std.transform(x), name, rewards.get(name, 0.2))
    path = tmp_path / "prior.npz"
    save_prior(path, bandit, std, 0.38, 0.4, 30 * len(rows) * len(A.ARM_NAMES))
    return str(path)


def live(tmp_path, mode="on", rewards=None, gpt=True, prior=None, **ode):
    prior = prior or make_prior(tmp_path, rewards or {"haiku-aggressive": 1.0})
    upstream = Upstream()
    config = GatewayConfig(
        log_dir=tmp_path / "logs", mode="shadow",
        ode=OdeConfig(mode, prior_path=prior, explore=0.0, seed=0, **ode),
        route=RouteConfig(gpt_upstream=LITELLM if gpt else None), router=BUGFIX)
    return create_app(config, transport=httpx2.MockTransport(upstream.handler)), upstream


def test_on_mode_uses_the_chosen_arm_for_the_whole_task(tmp_path):
    app, upstream = live(tmp_path)
    with TestClient(app) as client:
        post(client, agent_body("fix the bug in shop/cart.py"))
        first = last_record(app)["ode"]
        post(client, agent_body("fix the bug in shop/cart.py", ending_with_tool_result=True))
        loop = last_record(app)["ode"]
    assert first["choice"]["decided"] and first["choice"]["arm"] == "haiku-aggressive"
    assert first["model_sent"] == A.HAIKU.model_id and first["trim"] == "aggressive"
    assert upstream.sent(0)["model"] == A.HAIKU.model_id
    # the tool loop of the same task: no new decision, same arm
    assert not loop["choice"]["decided"] and loop["choice"]["arm"] == "haiku-aggressive"
    assert upstream.sent(1)["model"] == A.HAIKU.model_id


def test_shadow_mode_logs_the_choice_and_changes_nothing(tmp_path):
    app, upstream = live(tmp_path, mode="shadow")
    body = agent_body("fix the bug in shop/cart.py")
    with TestClient(app) as client:
        post(client, body)
    record = last_record(app)["ode"]
    assert record["mode"] == "shadow" and record["choice"]["arm"] == "haiku-aggressive"
    assert "model_sent" not in record and len(record["task_numbers"]["x"]) == 13
    assert upstream.requests[0].content == json.dumps(body).encode()  # byte for byte


def test_opus_is_never_chosen_by_default(tmp_path):
    app, upstream = live(tmp_path, rewards={"opus-standard": 1.0, "opus-aggressive": 1.0})
    with TestClient(app) as client:
        post(client, agent_body())
    choice = last_record(app)["ode"]["choice"]
    assert not any(a.startswith("opus") for a in choice["scores"])
    assert not choice["arm"].startswith("opus")
    assert "opus" not in upstream.sent()["model"]
    assert "opus-standard" not in DEFAULT_ARMS


def test_gpt_arms_are_left_out_without_a_litellm_route(tmp_path):
    app, _ = live(tmp_path, rewards={"gpt-aggressive": 1.0}, gpt=False)
    with TestClient(app) as client:
        post(client, agent_body())
        health = client.get("/tokencut/health").json()
    choice = last_record(app)["ode"]["choice"]
    assert not choice["arm"].startswith("gpt")
    assert health["ode_choosing"] and not any(a.startswith("gpt") for a in health["ode_arms"])


def test_a_new_typed_task_decides_again_and_keeps_the_arm_within_the_margin(tmp_path):
    app, _ = live(tmp_path)
    with TestClient(app) as client:
        post(client, agent_body("fix the bug in shop/cart.py"))
        post(client, agent_body("fix the bug in shop/cart.py") | {"messages": [
            *agent_body("fix the bug in shop/cart.py")["messages"],
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
            {"role": "user", "content": [{"type": "text", "text": "now fix the crash too"}]}]})
    choice = last_record(app)["ode"]["choice"]
    assert choice["decided"] and choice["previous"] == "haiku-aggressive"
    assert choice["arm"] == "haiku-aggressive"


def test_side_requests_are_never_touched(tmp_path):
    app, upstream = live(tmp_path)
    body = agent_body("write a title for this chat", tools=False)
    with TestClient(app) as client:
        post(client, body)
    assert "ode" not in last_record(app)
    assert upstream.requests[0].content == json.dumps(body).encode()


def test_a_missing_prior_turns_choosing_off(tmp_path):
    app, upstream = live(tmp_path, prior=str(tmp_path / "missing.npz"))
    body = agent_body()
    with TestClient(app) as client:
        health = client.get("/tokencut/health").json()
        post(client, body)
        stats = client.get("/tokencut/stats").json()
    assert health["ode"] == "on" and not health["ode_choosing"]
    assert upstream.requests[0].content == json.dumps(body).encode()  # fail open
    assert "ode_choices" not in stats


def test_stats_count_the_choices(tmp_path):
    app, _ = live(tmp_path)
    with TestClient(app) as client:
        post(client, agent_body("fix the bug in shop/cart.py"))
        post(client, agent_body("fix the bug in shop/cart.py", ending_with_tool_result=True))
        post(client, agent_body("another task: fix b.py", model="claude-haiku-5-5"))
        stats = client.get("/tokencut/stats").json()["ode_choices"]
    assert stats["decisions"] == 2 and stats["by_arm"] == {"haiku-aggressive": 2}
    assert stats["conversations"] == 2 and stats["random_tries"] == 0
    assert "opus-standard" not in stats["arms"]


def test_random_tries_are_logged_with_their_probability(tmp_path):
    prior = make_prior(tmp_path, {"haiku-aggressive": 1.0})
    chooser = Chooser.load(OdeConfig("on", prior_path=prior, explore=1.0, seed=1),
                           gpt_route=True)
    numbers = task_numbers(json.dumps(agent_body()).encode(), BUGFIX)
    _, log = chooser.decide("conv", numbers)
    assert log["explored"] and log["probability"] == pytest.approx(1 / len(DEFAULT_ARMS), abs=1e-6)


def test_follow_ups_and_unknown_conversations_keep_or_skip(tmp_path):
    chooser = Chooser.load(OdeConfig("on", prior_path=make_prior(tmp_path, {}), explore=0.0),
                           gpt_route=True)
    assert chooser.decide("c", None) == (None, {"decided": False, "arm": None})
    assert chooser.decide("c", {"task": None}) == (None, {"decided": False, "arm": None})
    numbers = task_numbers(json.dumps(agent_body()).encode(), BUGFIX)
    arm, _ = chooser.decide("c", numbers)
    assert chooser.decide("c", {"task": None})[0] == arm  # "yes, go on": same task


def test_a_decision_is_fast(tmp_path):
    chooser = Chooser.load(OdeConfig("on", prior_path=make_prior(tmp_path, {}), explore=0.0),
                           gpt_route=True)
    numbers = task_numbers(json.dumps(agent_body()).encode(), BUGFIX)
    start = time.perf_counter()
    for i in range(200):
        chooser.decide(f"c{i}", numbers)
    per_decision_ms = (time.perf_counter() - start) / 200 * 1000
    assert per_decision_ms < 5, per_decision_ms


def test_the_real_prior_loads_and_leaves_opus_out():
    chooser = Chooser.load(OdeConfig("on"), gpt_route=True)
    assert chooser.rows > 500 and set(chooser.arms) == set(DEFAULT_ARMS)


def test_no_unsure_bonus_without_live_learning(tmp_path, monkeypatch):
    prior = make_prior(tmp_path, {})
    chooser = Chooser.load(OdeConfig("on", prior_path=prior, explore=0.0), gpt_route=True)
    numbers = task_numbers(json.dumps(agent_body()).encode(), BUGFIX)
    _, log = chooser.decide("c", numbers)
    assert chooser.bandit.alpha == 0.0
    assert all(s["bonus"] == 0.0 for s in log["scores"].values())
    monkeypatch.setenv("TOKENCUT_ODE", "on")
    monkeypatch.setenv("TOKENCUT_ODE_ALPHA", "1")
    assert OdeConfig.from_env().alpha == 1.0
    with pytest.raises(ValueError):
        OdeConfig("on", alpha=-1)
