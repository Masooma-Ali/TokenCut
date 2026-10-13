"""Cache-event trimming: re-trim history only when the provider cache is lost anyway."""

import copy
import json

import httpx2
from fastapi.testclient import TestClient

from tests.test_gateway_app import reply_json
from tests.test_gateway_trimming import SOURCE, numbered, pytest_log
from tokencut.contracts import Tier
from tokencut.gateway.adapters import anthropic, gemini, openai
from tokencut.gateway.app import GatewayConfig, create_app
from tokencut.gateway.history import CacheEventConfig, detect_event, trim_with_cache_events
from tokencut.gateway.session import ConversationState, SessionStore, conversation_key
from tokencut.gateway.trimmer import TrimConfig

TOOLS = [{"name": n, "description": f"The {n} tool " * 30, "input_schema": {"type": "object"}}
         for n in ("Read", "Bash", "Edit", "WebFetch", "NotebookEdit", "mcp__jira__search")]
# min_tokens high: no on-arrival trims, so these tests isolate the history behaviour
CONFIG = TrimConfig(tier=Tier.MODERATE, min_tokens=10**6)


def turn(i: int, kind: str = "bash", path: str = "/repo/app.py") -> list[dict]:
    if kind == "bash":
        call = {"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": f"pytest -k t{i}"}}
        result = pytest_log(120 + i)
    elif kind == "read":
        call = {"type": "tool_use", "id": f"t{i}", "name": "Read", "input": {"file_path": path}}
        result = numbered(SOURCE)
    else:  # edit
        call = {"type": "tool_use", "id": f"t{i}", "name": "Edit",
                "input": {"file_path": path, "old_string": "a", "new_string": "b"}}
        result = "The file has been updated."
    return [{"role": "assistant", "content": [{"type": "text", "text": f"step {i}"}, call]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}",
                                          "content": result}]}]


def conversation(kinds: list[str], model: str = "claude-sonnet-5-5", path: str = "/repo/app.py") -> dict:
    messages = [{"role": "user", "content": "the tests in app.py fail, please fix"}]
    for i, kind in enumerate(kinds):
        messages += turn(i, kind, path)
    return {"model": model, "max_tokens": 100, "system": "You are a coding agent.",
            "tools": copy.deepcopy(TOOLS), "messages": messages}


def run(body, store, now, events=None, config=CONFIG):
    events = events or CacheEventConfig()
    return trim_with_cache_events("anthropic", body["model"], body, anthropic.parse(body),
                                  config, events, store, now=now)


def results(body):
    return [b["content"] for m in body["messages"] if m["role"] == "user" and isinstance(m["content"], list)
            for b in m["content"] if b.get("type") == "tool_result"]


# ---------- event detection ----------


def test_detect_event():
    cfg = CacheEventConfig(cache_ttl_seconds=300, compact_at_tokens=1000, compact_growth_tokens=100)
    state = ConversationState(model="a", last_seen=100.0)
    assert detect_event(None, "a", 101, 10, {}, cfg) is None  # first sight
    assert detect_event(state, "b", 101, 10, {}, cfg) == "model_switch"
    assert detect_event(state, "a", 100 + 301, 10, {}, cfg) == "cache_expired"
    one_hour = {"system": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}
    assert detect_event(state, "a", 100 + 301, 10, one_hour, cfg) is None  # 1-hour cache still valid
    assert detect_event(state, "a", 101, 1500, {}, cfg) == "context_full"
    state.last_event_tokens = 1450
    assert detect_event(state, "a", 101, 1500, {}, cfg) is None  # no repeat without growth
    assert detect_event(state, "a", 101, 1600, {}, cfg) == "context_full"


def test_switching_back_to_a_recently_used_model_is_not_an_event():
    cfg = CacheEventConfig(cache_ttl_seconds=300)
    state = ConversationState(model="sonnet", last_seen=100.0, model_seen={"haiku": 50.0, "sonnet": 100.0})
    assert detect_event(state, "haiku", 120, 10, {}, cfg) is None  # haiku's cache is still warm
    assert detect_event(state, "haiku", 400, 10, {}, cfg) == "model_switch"  # haiku cold again
    assert detect_event(state, "opus", 120, 10, {}, cfg) == "model_switch"  # never used here

    store = SessionStore()  # end to end: haiku -> sonnet (event) -> haiku within TTL (no event)
    run(conversation(["bash"] * 6, model="haiku"), store, now=0)
    _, _, first = run(conversation(["bash"] * 7, model="sonnet"), store, now=10)
    _, _, back = run(conversation(["bash"] * 8, model="haiku"), store, now=20)
    assert (first, back) == ("model_switch", None)


def test_conversation_key_ignores_model_and_later_turns():
    a = conversation(["bash"], model="m1")
    b = conversation(["bash", "bash", "read"], model="m2")
    assert conversation_key("anthropic", anthropic.parse(a)) == conversation_key("anthropic", anthropic.parse(b))
    other = conversation(["bash"])
    other["messages"][0]["content"] = "a different task"
    assert conversation_key("anthropic", anthropic.parse(other)) != conversation_key("anthropic", anthropic.parse(a))


# ---------- history at events ----------


def test_no_history_trim_without_event():
    store = SessionStore()
    body = conversation(["bash"] * 8)
    new, _, event = run(body, store, now=0)
    assert event is None and new is None  # first sight: nothing changed
    new, _, event = run(conversation(["bash"] * 9), store, now=10)
    assert event is None and new is None  # same model, cache alive: nothing changed


def test_model_switch_masks_old_output_keeps_recent_and_errors():
    store = SessionStore()
    run(conversation(["bash"] * 8), store, now=0)
    body = conversation(["bash"] * 9, model="claude-haiku-4-5")
    new, report, event = run(body, store, now=5)
    assert event == "model_switch"
    out = results(new)
    keep = 4  # moderate tier keeps the last 4 assistant turns
    for old in out[:-keep]:
        assert old.startswith("[TokenCut: earlier output of `pytest -k")
        assert "E       assert 7.0 == 2" in old  # error lines survive
        assert "1 failed" in old  # summary line survives
    assert out[-keep:] == results(body)[-keep:]  # recent results untouched
    assert {d["how"] for d in report.trimmed} == {"history_old_output"}
    assert report.saved > 0
    ids = [b["tool_use_id"] for m in new["messages"] if m["role"] == "user" and isinstance(m["content"], list)
           for b in m["content"]]
    assert ids == [f"t{i}" for i in range(9)]  # every tool call keeps its result


def test_cache_stays_valid_between_events():
    store = SessionStore()
    run(conversation(["bash"] * 8), store, now=0)
    switched, _, _ = run(conversation(["bash"] * 9, model="m2"), store, now=5)
    nxt_body = conversation(["bash"] * 11, model="m2")
    nxt, report, event = run(nxt_body, store, now=20)
    assert event is None
    # everything sent at the event is re-sent byte-for-byte: the new cache prefix is valid
    assert nxt["messages"][: len(switched["messages"])] == switched["messages"]
    assert all(d["how"] == "history_pinned" for d in report.trimmed)


def test_cache_expiry_triggers_event():
    store = SessionStore()
    run(conversation(["bash"] * 8), store, now=0)
    _, _, event = run(conversation(["bash"] * 9), store, now=301)
    assert event == "cache_expired"


def test_duplicate_and_stale_file_reads():
    store = SessionStore()
    kinds = ["read", "bash", "read", "edit", "read"] + ["bash"] * 4  # last read at turn 4
    run(conversation(kinds), store, now=0)
    new, report, _ = run(conversation(kinds, model="m2"), store, now=5)
    out = results(new)
    assert "a newer read of this file appears later" in out[0]  # read 0 superseded by read 2
    assert "a newer read of this file appears later" in out[2]  # read 2 superseded by read 4
    hows = [d["how"] for d in report.trimmed]
    assert hows.count("history_duplicate_read") == 2


def test_file_edited_after_read_gets_stale_note():
    store = SessionStore()
    kinds = ["read", "edit"] + ["bash"] * 5
    run(conversation(kinds), store, now=0)
    new, _, _ = run(conversation(kinds, model="m2"), store, now=5)
    assert "the file was edited afterwards" in results(new)[0]


def test_reread_after_removal_protects_the_file():
    store = SessionStore()
    kinds = ["read"] + ["bash"] * 5
    run(conversation(kinds), store, now=0)
    first, _, _ = run(conversation(kinds, model="m2"), store, now=5)
    assert "read it again if needed" in results(first)[0]
    # the agent reads the file again, then another event happens
    kinds2 = kinds + ["read"] + ["bash"] * 5
    run(conversation(kinds2, model="m2"), store, now=10)
    later, _, event = run(conversation(kinds2, model="m3"), store, now=15)
    assert event == "model_switch"
    out = results(later)
    assert "read it again if needed" in out[0]  # the old removal stays identical
    assert out[6] == numbered(SOURCE)  # the re-read copy is protected, kept whole


def test_tool_pruning_is_opt_in_and_stable():
    store = SessionStore()
    run(conversation(["bash"] * 6), store, now=0)
    new, _, _ = run(conversation(["bash"] * 7, model="m2"), store, now=5)
    assert len(new["tools"]) == len(TOOLS)  # off by default

    store = SessionStore()
    events = CacheEventConfig(prune_tools=True, core_tools=("Read", "Edit"))
    run(conversation(["bash"] * 6), store, now=0, events=events)
    body = conversation(["bash"] * 7, model="m2")
    body["tool_choice"] = {"type": "tool", "name": "WebFetch"}
    new, report, _ = run(body, store, now=5, events=events)
    names = [t["name"] for t in new["tools"]]
    assert names == ["Read", "Bash", "Edit", "WebFetch"]  # core + used + forced; MCP/unused gone
    assert any(d["how"] == "tools_pruned" for d in report.trimmed)
    nxt, _, _ = run(conversation(["bash"] * 8, model="m2"), store, now=9, events=events)
    assert [t["name"] for t in nxt["tools"]] == ["Read", "Bash", "Edit"]  # stable between events


def test_openai_and_gemini_events():
    def oa(n, model):
        msgs = [{"role": "user", "content": "fix the failing tests"}]
        for i in range(n):
            msgs += [{"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": "run_command", "arguments": json.dumps({"cmd": "pytest"})}}]},
                     {"role": "tool", "tool_call_id": f"c{i}", "content": pytest_log(150)}]
        return {"model": model, "messages": msgs}

    store = SessionStore()
    trim_with_cache_events("openai", "a", oa(7, "a"), openai.parse(oa(7, "a")), CONFIG, CacheEventConfig(), store, now=0)
    body = oa(8, "b")
    new, _, event = trim_with_cache_events("openai", "b", body, openai.parse(body), CONFIG, CacheEventConfig(), store, now=1)
    assert event == "model_switch" and new["messages"][2]["content"].startswith("[TokenCut: earlier output")

    def gm(n):
        contents = [{"role": "user", "parts": [{"text": "fix the failing tests"}]}]
        for _ in range(n):
            contents += [{"role": "model", "parts": [{"functionCall": {"name": "run_shell_command", "args": {"command": "pytest"}}}]},
                         {"role": "user", "parts": [{"functionResponse": {"name": "run_shell_command", "response": {"output": pytest_log(150)}}}]}]
        return {"contents": contents}

    store = SessionStore()
    trim_with_cache_events("gemini", "g1", gm(7), gemini.parse(gm(7)), CONFIG, CacheEventConfig(), store, now=0)
    new, _, event = trim_with_cache_events("gemini", "g2", gm(8), gemini.parse(gm(8)), CONFIG, CacheEventConfig(), store, now=1)
    out = new["contents"][2]["parts"][0]["functionResponse"]["response"]["output"]
    assert event == "model_switch" and out.startswith("[TokenCut: earlier output")


# ---------- through the gateway ----------


class Upstream:
    def __init__(self):
        self.bodies: list[dict] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        return reply_json(200, {"usage": {"input_tokens": 1, "output_tokens": 1}})


def test_gateway_trims_history_on_model_switch(tmp_path):
    upstream = Upstream()
    config = GatewayConfig(log_dir=tmp_path, mode="trim")
    app = create_app(config, transport=httpx2.MockTransport(upstream.handler))
    with TestClient(app) as client:
        client.post("/anthropic/v1/messages", json=conversation(["bash"] * 8))
        client.post("/anthropic/v1/messages", json=conversation(["bash"] * 9, model="claude-haiku-4-5"))
        stats = client.get("/tokencut/stats").json()
        health = client.get("/tokencut/health").json()
    assert "[TokenCut: earlier output" in results(upstream.bodies[1])[0]
    assert stats["trimming"]["cache_events"] == {"model_switch": 1}
    assert stats["trimming"]["saved_by"]["history_old_output"] > 0
    assert health["cache_events"] is True and health["prune_tools"] is False


def test_cache_events_can_be_disabled(monkeypatch):
    monkeypatch.setenv("TOKENCUT_CACHE_EVENTS", "0")
    monkeypatch.setenv("TOKENCUT_PRUNE_TOOLS", "1")
    monkeypatch.setenv("TOKENCUT_CORE_TOOLS", "Read, Bash")
    config = GatewayConfig.from_env()
    assert config.cache_events is False
    assert config.events.prune_tools is True and config.events.core_tools == ("Read", "Bash")


# ---------- R10: mask a finished task's large outputs at the next task ----------

TASKS = CacheEventConfig(task_masking=True, task_mask_min_tokens=300)


def two_tasks(n_first: int, n_second: int, model: str = "claude-sonnet-5-5") -> dict:
    """Task 1 (n_first bash turns), then a new user message and n_second more turns."""
    body = conversation(["bash"] * n_first, model=model)
    body["messages"].append({"role": "assistant", "content": [{"type": "text", "text": "done"}]})
    body["messages"].append({"role": "user", "content": "now update the docs"})
    for i in range(n_second):
        body["messages"] += turn(100 + i)
    return body


def test_task_masking_is_off_by_default():
    store = SessionStore()
    run(conversation(["bash"] * 6), store, now=0)
    new, _, _ = run(two_tasks(6, 0), store, now=5)
    assert new is None


def test_new_task_masks_previous_task_outputs():
    store = SessionStore()
    run(conversation(["bash"] * 6), store, now=0, events=TASKS)
    body = two_tasks(6, 0)
    new, report, event = run(body, store, now=5, events=TASKS)
    assert event is None  # same model, warm cache: not a cache event
    out = results(new)
    assert len(out) == 6 and all(o.startswith("[TokenCut: earlier output of `pytest -k") for o in out)
    assert all("1 failed" in o for o in out)  # summary / error lines survive in the note
    assert {d["how"] for d in report.trimmed} == {"task_masked"}
    assert new["messages"][-1] == body["messages"][-1]  # the new user message is untouched
    assert new["messages"][: 1] == body["messages"][: 1]


def test_outputs_of_the_running_task_stay_whole_and_cache_stays_valid():
    store = SessionStore()
    run(conversation(["bash"] * 6), store, now=0, events=TASKS)
    first, _, _ = run(two_tasks(6, 0), store, now=5, events=TASKS)
    body = two_tasks(6, 2)
    nxt, report, _ = run(body, store, now=10, events=TASKS)
    assert nxt["messages"][: len(first["messages"])] == first["messages"]  # identical prefix
    assert results(nxt)[-2:] == results(body)[-2:]  # task 2's own outputs untouched
    assert {d["how"] for d in report.trimmed} == {"history_pinned"}


def test_no_masking_on_first_sight():
    store = SessionStore()
    new, _, _ = run(two_tasks(6, 1), store, now=0, events=TASKS)
    assert new is None  # nothing known about this conversation yet: its cache may be valid


def test_task_masking_skipped_when_it_does_not_pay_off():
    # One modest old output followed by a very long cached tail (a long explanation): the
    # one-time cache rewrite of that tail would cost more than the cache reads saved later.
    long_answer = {"role": "assistant", "content": [{"type": "text", "text": "explanation " * 6000}]}
    task1 = conversation(["bash"])
    task1["messages"].append(long_answer)
    store = SessionStore()
    run(task1, store, now=0, events=TASKS)
    body = copy.deepcopy(task1)
    body["messages"].append({"role": "user", "content": "now update the docs"})
    new, report, _ = run(body, store, now=5, events=TASKS)
    assert new is None and report.trimmed == []


def test_task_masking_settings_from_env(monkeypatch):
    assert GatewayConfig.from_env().events.task_masking is False  # off by default
    monkeypatch.setenv("TOKENCUT_TASK_MASK", "1")
    monkeypatch.setenv("TOKENCUT_TASK_MASK_TOKENS", "1000")
    events = GatewayConfig.from_env().events
    assert events.task_masking is True and events.task_mask_min_tokens == 1000
