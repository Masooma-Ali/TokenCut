"""The team panel: /tokencut/system (logins, installs, routing, settings, ODE) and the richer
/tokencut/recent feed. Numbers and yes/no only: never keys, tokens, prompts or code."""

import json

import httpx2
import pytest
from fastapi.testclient import TestClient

from tests.test_gateway_app import REQUEST, FakeProvider
from tokencut.gateway import panel
from tokencut.gateway.app import GatewayConfig, create_app
from tokencut.gateway.ode import OdeConfig
from tokencut.gateway.routing import RouteConfig

SECRET = "sk-ant-oat01-SECRET-TOKEN"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENCUT_HOME", str(tmp_path / "tc"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return tmp_path


def _claude_login(home, plan="pro"):
    folder = home / ".claude"
    folder.mkdir()
    (folder / ".credentials.json").write_text(json.dumps(
        {"claudeAiOauth": {"accessToken": SECRET, "refreshToken": SECRET,
                           "subscriptionType": plan}}))


def _client(tmp_path, **config):
    fake = FakeProvider()
    app = create_app(GatewayConfig(log_dir=tmp_path / "logs", **config),
                     transport=httpx2.MockTransport(fake.handler))
    return TestClient(app), app


def test_claude_plan_comes_from_the_login_not_from_a_fixed_text(home):
    assert panel.claude_login()["logged_in"] is False
    _claude_login(home, "max")
    login = panel.claude_login()
    assert login == {"logged_in": True, "plan": "max", "source": "claude_code_login"}
    assert SECRET not in json.dumps(login)


def test_claude_plan_api_key_and_broken_file(home, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    assert panel.claude_login()["plan"] == "api_key"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    (home / ".claude").mkdir()
    (home / ".claude" / ".credentials.json").write_text("not json")
    assert panel.claude_login()["logged_in"] is False  # never raises


def test_system_endpoint_sections_and_no_secrets(home, tmp_path):
    _claude_login(home)
    client, _ = _client(tmp_path, mode="trim", client_key="tc-PRIVATE",
                        route=RouteConfig(gpt_upstream="http://127.0.0.1:4000",
                                          gpt_key="sk-PRIVATE"),
                        ode=OdeConfig(mode="shadow"))
    with client:
        data = client.get("/tokencut/system").json()
    assert {"gateway", "routing", "ode", "logins", "installs", "process"} <= set(data)
    assert data["gateway"]["mode"] == "trim" and data["gateway"]["key_required"] is True
    assert data["routing"]["gpt_upstream"] == "127.0.0.1:4000"  # host only
    assert data["logins"]["claude"]["plan"] == "pro"
    assert data["ode"]["mode"] == "shadow" and "choosing" in data["ode"]
    assert data["process"]["uptime_s"] >= 0
    text = json.dumps(data)
    for secret in (SECRET, "tc-PRIVATE", "sk-PRIVATE"):
        assert secret not in text


def test_system_sections_fail_alone(home, tmp_path, monkeypatch):
    client, _ = _client(tmp_path)
    monkeypatch.setattr(panel, "installs", lambda *a, **k: 1 / 0)
    with client:
        data = client.get("/tokencut/system").json()
    assert data["installs"] == {"error": "ZeroDivisionError"} and "gateway" in data


def test_feed_carries_before_after_ode_cost_and_auth(home, tmp_path):
    client, _ = _client(tmp_path, mode="trim")
    with client:
        client.post("/anthropic/v1/messages", content=json.dumps(REQUEST).encode(),
                    headers={"authorization": f"Bearer {SECRET}",
                             "anthropic-version": "2023-06-01",
                             "content-type": "application/json"})
        feed = client.get("/tokencut/recent?limit=5").json()
    row = feed[0]
    assert {"by_kind", "tokens", "tokens_saved", "via", "auth", "trim_ms", "ode",
            "cost_usd", "error"} <= set(row)
    assert row["auth"] == "bearer" and row["ode"] is None
    assert SECRET not in json.dumps(feed)


def test_feed_ode_compact_form():
    record = {"ts": "2026-11-28T10:00:00.000+00:00", "provider": "anthropic", "route": "claude",
              "analysis": {"model": "claude-haiku-5-5", "tokens_total": 100,
                           "tokens_by_kind": {"tools": 60}, "tool_result_tokens": {}},
              "ode": {"mode": "on", "arm": "haiku-aggressive", "trim": "aggressive",
                      "model_requested": "claude-sonnet-5-5", "model_sent": "claude-haiku-5-5",
                      "skipped": None,
                      "choice": {"decided": True, "arm": "haiku-aggressive", "previous": None,
                                 "explored": False, "greedy": "haiku-aggressive",
                                 "scores": {"haiku-aggressive": {"total": 1.5},
                                            "gpt-standard": {"total": 1.1}}},
                      "task_numbers": {"task": "tests", "p": 0.92}},
              "usage": {"input_tokens": 10, "cache_read_tokens": 0, "cache_write_tokens": 0,
                        "output_tokens": 5}}
    ode = panel.ode_view(record)
    assert ode["arm"] == "haiku-aggressive" and ode["decided"] is True
    assert ode["requested"] == "claude-sonnet-5-5" and ode["sent"] == "claude-haiku-5-5"
    assert ode["task"] == "tests" and ode["p"] == 0.92
    assert ode["scores"]["gpt-standard"] == 1.1
    assert panel.cost_usd(record) > 0
    assert panel.cost_usd({"analysis": {"model": "unknown-model"}, "usage": {}}) is None
