"""ODE phase 3: the gateway carries out a forced arm (model x trim level). Fake provider, no
network, no API calls."""

import json

import httpx2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tokencut.gateway.app import GatewayConfig, create_app
from tokencut.gateway.history import CacheEventConfig
from tokencut.gateway.ode import (
    ModelNameRestorer,
    OdeConfig,
    arm_settings,
    starts_turn,
    task_numbers,
)
from tokencut.gateway.routing import RouteConfig
from tokencut.gateway.task_router import LABELS
from tokencut.gateway.usage import Usage, apply_usage
from tokencut.gateway.virtual import ENABLE_TOOL, VirtualConfig
from tokencut.ode import arms as A
from tokencut.ode.reward import request_cost, usage_from_dict

LITELLM = "http://127.0.0.1:4000"
TOOLS = [{"name": n, "description": f"{n} tool", "input_schema": {"type": "object"}}
         for n in ("Read", "Bash", "WebFetch")]


def agent_body(text="fix the bug in shop/cart.py", model="claude-sonnet-5-5", tools=True,
               ending_with_tool_result=False):
    """A Claude Code main-loop request (3 client tools), or a side request without tools."""
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    if ending_with_tool_result:
        messages += [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "a.py"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "x = 1"}]},
        ]
    body = {"model": model, "max_tokens": 100, "stream": True, "system": "You are Claude Code.",
            "messages": messages}
    if tools:
        body["tools"] = TOOLS
    return body


def sse(model: str) -> bytes:
    return (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"m1","model":"' + model.encode()
        + b'","usage":{"input_tokens":10,"cache_read_input_tokens":100,'
        b'"cache_creation_input_tokens":30,"cache_creation":{"ephemeral_5m_input_tokens":0,'
        b'"ephemeral_1h_input_tokens":30},"output_tokens":1}}}\n\n'
        b"event: message_delta\n"
        b'data: {"type":"message_delta","usage":{"output_tokens":20}}\n\n'
    )


class Chunked(httpx2.AsyncByteStream):
    def __init__(self, data: bytes, size: int = 7):
        self.data, self.size = data, size

    async def __aiter__(self):
        for i in range(0, len(self.data), self.size):
            yield self.data[i:i + self.size]


class Upstream:
    """Fake Anthropic and LiteLLM: records what arrived, answers as the model that was asked."""

    def __init__(self):
        self.requests: list[httpx2.Request] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        try:
            model = json.loads(request.content).get("model", "")
        except ValueError:  # the fail-open test sends a broken body on purpose
            model = ""
        return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                               stream=Chunked(sse(model)))

    def sent(self, i=-1) -> dict:
        return json.loads(self.requests[i].content)


class FakeRouter:
    def __init__(self, **probs):
        p = np.array([probs.get(label, 0.0) for label in LABELS])
        self.p = p / p.sum()
        self.train_prior = self.natural_prior = np.full(len(LABELS), 1 / len(LABELS))

    def probs(self, text):
        return self.p


def gateway(tmp_path, ode: OdeConfig, mode="shadow", gpt=False, lazy=False, router=None):
    upstream = Upstream()
    config = GatewayConfig(
        log_dir=tmp_path / "logs", mode=mode, ode=ode,
        route=RouteConfig(gpt_upstream=LITELLM if gpt else None),
        virtual=VirtualConfig(lazy_tools=lazy),
        router=router or FakeRouter(bugfix=0.95, no_brief=0.05),
    )
    app = create_app(config, transport=httpx2.MockTransport(upstream.handler))
    return app, upstream


def post(client, body, **headers):
    return client.post("/anthropic/v1/messages", content=json.dumps(body).encode(),
                       headers={"x-api-key": "sk-claude", "content-type": "application/json",
                                **headers})


def last_record(app) -> dict:
    return app.state.request_log.recent[-1]


# ---------- settings ----------

def test_ode_is_off_by_default_and_validated():
    assert OdeConfig().mode == "off" and not OdeConfig().on
    with pytest.raises(ValueError, match="FORCE_ARM"):
        OdeConfig(mode="force")
    with pytest.raises(ValueError, match="opus-standrad"):
        OdeConfig(mode="force", force_arm="opus-standrad")
    with pytest.raises(ValueError):
        OdeConfig(mode="on", force_arm="opus-standard")  # Phase 6, not yet


def test_ode_settings_from_environment(monkeypatch):
    monkeypatch.setenv("TOKENCUT_ODE", "force")
    monkeypatch.setenv("TOKENCUT_ODE_FORCE_ARM", "haiku-aggressive")
    monkeypatch.setenv("TOKENCUT_ODE_RESTORE_MODEL", "1")
    assert OdeConfig.from_env() == OdeConfig("force", "haiku-aggressive", True)
    assert GatewayConfig.from_env().ode.force_arm == "haiku-aggressive"


def test_arm_trim_level_sets_the_p1_features():
    events, virtual = CacheEventConfig(), VirtualConfig()
    ev, vi = arm_settings(A.arm("opus-aggressive"), events, virtual)
    assert ev.prune_tools and vi.lazy_tools and vi.ccr
    assert vi.store is virtual.store  # retrieved originals stay in one shared store
    ev, vi = arm_settings(A.arm("opus-standard"),
                          CacheEventConfig(prune_tools=True), VirtualConfig(lazy_tools=True))
    assert not (ev.prune_tools or vi.lazy_tools or vi.ccr)


# ---------- model swap ----------

def test_off_leaves_requests_untouched(tmp_path):
    app, upstream = gateway(tmp_path, OdeConfig())
    with TestClient(app) as client:
        post(client, agent_body())
    assert upstream.sent()["model"] == "claude-sonnet-5-5"
    assert "ode" not in last_record(app)


def test_forced_claude_arm_swaps_the_model_before_trimming(tmp_path):
    app, upstream = gateway(tmp_path, OdeConfig("force", "opus-standard"), mode="trim")
    with TestClient(app) as client:
        response = post(client, agent_body())
    assert response.status_code == 200
    assert str(upstream.requests[-1].url) == "https://api.anthropic.com/v1/messages"
    assert upstream.sent()["model"] == "claude-opus-5-5"
    record = last_record(app)
    assert record["analysis"]["model"] == "claude-opus-5-5"  # trimming saw the real model
    assert record["ode"]["arm"] == "opus-standard"
    assert record["ode"]["model_requested"] == "claude-sonnet-5-5"
    assert record["ode"]["model_sent"] == "claude-opus-5-5"
    assert record["ode"]["skipped"] is None


def test_side_requests_are_left_alone(tmp_path):
    app, upstream = gateway(tmp_path, OdeConfig("force", "opus-standard"))
    with TestClient(app) as client:
        post(client, agent_body(model="claude-haiku-5-5", tools=False))  # e.g. a title request
    assert upstream.sent()["model"] == "claude-haiku-5-5"
    assert "ode" not in last_record(app)


def test_forced_gpt_arm_takes_the_litellm_route(tmp_path):
    app, upstream = gateway(tmp_path, OdeConfig("force", "gpt-standard"), gpt=True)
    with TestClient(app) as client:
        post(client, agent_body())
    request = upstream.requests[-1]
    assert str(request.url) == f"{LITELLM}/v1/messages"
    assert upstream.sent()["model"] == "gpt-6.1-sol"
    assert "x-api-key" not in request.headers  # the Claude login never goes to the GPT side
    assert last_record(app)["route"] == "gpt"


def test_gpt_arm_without_the_litellm_route_is_skipped(tmp_path):
    app, upstream = gateway(tmp_path, OdeConfig("force", "gpt-standard"), gpt=False)
    with TestClient(app) as client:
        post(client, agent_body())
    assert upstream.sent()["model"] == "claude-sonnet-5-5"  # unchanged, still works
    assert last_record(app)["ode"]["skipped"] == "gpt_route_not_set_up"


def test_bad_json_fails_open(tmp_path):
    app, upstream = gateway(tmp_path, OdeConfig("force", "opus-standard"))
    with TestClient(app) as client:
        client.post("/anthropic/v1/messages", content=b"{not json",
                    headers={"content-type": "application/json"})
    assert upstream.requests[-1].content == b"{not json"
    assert last_record(app)["ode"] == {"error": "JSONDecodeError"}


# ---------- trim level per request ----------

def tool_names(body):
    return [t["name"] for t in body.get("tools", [])]


def test_aggressive_arm_turns_on_lazy_tools_for_this_request(tmp_path):
    app, upstream = gateway(tmp_path, OdeConfig("force", "sonnet-aggressive"), mode="trim")
    with TestClient(app) as client:
        post(client, agent_body())
    names = tool_names(upstream.sent())
    assert "WebFetch" not in names and ENABLE_TOOL in names  # unused non-core tool hidden
    assert last_record(app)["ode"]["trim"] == "aggressive"


def test_standard_arm_overrides_a_gateway_wide_lazy_setting(tmp_path):
    app, upstream = gateway(tmp_path, OdeConfig("force", "sonnet-standard"), mode="trim",
                            lazy=True)
    with TestClient(app) as client:
        post(client, agent_body())
    assert tool_names(upstream.sent()) == ["Read", "Bash", "WebFetch"]


# ---------- reply model name ----------

def test_reply_is_passed_on_unchanged_by_default(tmp_path):
    app, _ = gateway(tmp_path, OdeConfig("force", "opus-standard"))
    with TestClient(app) as client:
        response = post(client, agent_body())
    assert response.content == sse("claude-opus-5-5")


def test_reply_model_name_can_be_restored(tmp_path):
    app, _ = gateway(tmp_path, OdeConfig("force", "opus-standard", restore_model=True))
    with TestClient(app) as client:
        response = post(client, agent_body())
    assert response.content == sse("claude-sonnet-5-5")  # what Claude Code asked for
    record = last_record(app)
    assert record["ode"]["model_restored"] is True
    assert record["usage"]["output_tokens"] == 20  # usage still read from the original bytes


def test_restorer_on_split_chunks_and_plain_json():
    stream = sse("gpt-6.1-sol")
    r = ModelNameRestorer("gpt-6.1-sol", "claude-sonnet-5-5", streaming=True)
    out = b"".join(r.feed(stream[i:i + 3]) for i in range(0, len(stream), 3)) + r.close()
    assert out == sse("claude-sonnet-5-5") and r.restored
    r = ModelNameRestorer("gpt-6.1-sol", "claude-sonnet-5-5", streaming=False)
    body = b'{"id": "m", "model": "gpt-6.1-sol", "content": []}'
    assert r.feed(body) == b"" and r.close() == body.replace(b"gpt-6.1-sol", b"claude-sonnet-5-5")
    r = ModelNameRestorer("gpt-6.1-sol", "claude-sonnet-5-5", streaming=True)
    other = b'data: {"model":"something-else"}\n\nrest'
    assert r.feed(other) + r.close() == other and not r.restored


# ---------- task numbers ----------

def test_turn_start_detection():
    assert starts_turn(agent_body())
    assert not starts_turn(agent_body(ending_with_tool_result=True))
    assert not starts_turn({"messages": []})
    # Claude Code may append its own system message after the user's (seen live 2026-11-25).
    with_system = agent_body()
    with_system["messages"].append({"role": "system", "content": "context the agent adds"})
    assert starts_turn(with_system)
    assert not starts_turn({"messages": [{"role": "system", "content": "only system"}]})


def test_task_numbers_only_at_the_start_of_a_typed_task():
    router = FakeRouter(bugfix=0.95, no_brief=0.05)
    numbers = task_numbers(json.dumps(agent_body()).encode(), router, 0.87)
    assert numbers["task"] == "bugfix" and numbers["decision_point"] is True
    assert len(numbers["x"]) == 13 and numbers["x"][-1] == 1.0
    mid_task = json.dumps(agent_body(ending_with_tool_result=True)).encode()
    assert task_numbers(mid_task, router, 0.87) is None
    assert task_numbers(json.dumps(agent_body("yes")).encode(), router, 0.87) == {"task": None}
    assert task_numbers(json.dumps(agent_body()).encode(), None, 0.87) is None


def test_decision_point_uses_odes_own_threshold_not_the_brief_one(tmp_path):
    """Pilot: at the task-brief threshold (0.87) only 'write a test' turns were decision points.
    ODE now decides at any typed task the router does not call no_brief."""
    router = FakeRouter(refactor=0.80, no_brief=0.20)  # how the pilot's bugfix turns were read
    app, _ = gateway(tmp_path, OdeConfig("force", "opus-standard"), router=router)
    with TestClient(app) as client:
        post(client, agent_body("Now make the code change in the repository."))
    numbers = last_record(app)["ode"]["task_numbers"]
    assert numbers["task"] == "refactor" and numbers["decision_point"] is True
    assert task_numbers(json.dumps(agent_body("yes")).encode(), router) == {"task": None}


def test_gateway_logs_task_numbers_without_text(tmp_path):
    app, _ = gateway(tmp_path, OdeConfig("force", "opus-standard"))
    secret = "fix the bug in shop/cart.py PRIVATE-TEXT-123"
    with TestClient(app) as client:
        post(client, agent_body(secret))
    record = last_record(app)
    assert len(record["ode"]["task_numbers"]["x"]) == 13
    assert "PRIVATE-TEXT-123" not in json.dumps(record)  # numbers only


# ---------- cache-write split ----------

def test_usage_reads_the_cache_write_split():
    usage = Usage()
    apply_usage("anthropic", {"type": "message_start", "message": {"usage": {
        "input_tokens": 5, "cache_creation_input_tokens": 300,
        "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200},
    }}}, usage)
    assert (usage.cache_write_5m_tokens, usage.cache_write_1h_tokens) == (100, 200)
    assert usage.as_dict()["cache_write_1h_tokens"] == 200
    assert "cache_write_1h_tokens" not in Usage(cache_write_tokens=5).as_dict()  # old log shape


def test_reward_prices_each_cache_lifetime_when_known():
    split = Usage(cache_write_tokens=300, cache_write_5m_tokens=100, cache_write_1h_tokens=200)
    # Sonnet: 100 x $2.50 + 200 x $4.00 per million = 0.00025 + 0.0008
    assert request_cost(A.SONNET, split) == pytest.approx(0.00105)
    # no split given: all 300 at the default 1h price
    assert request_cost(A.SONNET, Usage(cache_write_tokens=300)) == pytest.approx(0.0012)
    assert usage_from_dict({"input_tokens": 3, "unknown": 9}) == Usage(input_tokens=3)
