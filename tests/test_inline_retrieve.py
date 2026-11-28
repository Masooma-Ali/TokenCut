"""tokencut_retrieve answered inside the reply stream: the agent never sees the unknown tool.

The fake provider streams Anthropic-format events. Every case checks what the client receives
and what the gateway sent upstream."""

import json

import httpx2
from fastapi.testclient import TestClient

from tests.test_gateway_app import REQUEST, reply
from tokencut.gateway.app import GatewayConfig, create_app
from tokencut.gateway.virtual import VirtualConfig

ORIGINAL = "line one\nline two\nline three\n" * 40  # long enough to be a stored original


def sse(event: dict) -> bytes:
    return b"event: " + event["type"].encode() + b"\ndata: " + json.dumps(event).encode() + b"\n\n"


def start(inp=10, out=1) -> bytes:
    return sse({"type": "message_start", "message": {"usage": {
        "input_tokens": inp, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        "output_tokens": out}}})


def text_block(index: int, text: str) -> bytes:
    return (sse({"type": "content_block_start", "index": index,
                 "content_block": {"type": "text", "text": ""}})
            + sse({"type": "content_block_delta", "index": index,
                   "delta": {"type": "text_delta", "text": text}})
            + sse({"type": "content_block_stop", "index": index}))


def tool_block(index: int, call_id: str, name: str, args: dict) -> bytes:
    raw = json.dumps(args)
    half = len(raw) // 2  # arrives in two pieces, like the real thing
    return (sse({"type": "content_block_start", "index": index,
                 "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}}})
            + sse({"type": "content_block_delta", "index": index,
                   "delta": {"type": "input_json_delta", "partial_json": raw[:half]}})
            + sse({"type": "content_block_delta", "index": index,
                   "delta": {"type": "input_json_delta", "partial_json": raw[half:]}})
            + sse({"type": "content_block_stop", "index": index}))


def end(stop_reason: str, out: int = 20) -> bytes:
    return (sse({"type": "message_delta", "delta": {"stop_reason": stop_reason},
                 "usage": {"output_tokens": out}})
            + sse({"type": "message_stop"}))


class Script:
    """Answers upstream calls in order with the given bodies (or (status, body) pairs)."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        item = self.replies[min(len(self.requests), len(self.replies)) - 1]
        status, body = item if isinstance(item, tuple) else (200, item)
        return reply(status, body, "text/event-stream" if status == 200 else "application/json")


def run(script: Script, ccr: bool = True, tmp_path=None):
    virtual = VirtualConfig(ccr=ccr)
    ref = virtual.store.put(ORIGINAL)
    config = GatewayConfig(log_dir=tmp_path, virtual=virtual)
    app = create_app(config, transport=httpx2.MockTransport(script.handler))
    with TestClient(app) as client:
        response = client.post("/anthropic/v1/messages", json=REQUEST,
                               headers={"anthropic-version": "2023-06-01"})
        summary = client.get("/tokencut/stats").json()
    return response, ref, summary


def events(body: bytes) -> list[dict]:
    return [json.loads(line[5:]) for line in body.split(b"\n") if line.startswith(b"data:")]


def retrieve(ref, **extra):
    return {"ref": ref, **extra}


def test_retrieve_is_answered_and_the_agent_never_sees_it(tmp_path):
    # The ref is only known after the store is filled, so build the replies lazily.
    virtual_ref = VirtualConfig().store.put(ORIGINAL)  # refs are content hashes: the same here
    first = (start() + text_block(0, "Let me look.")
             + tool_block(1, "toolu_1", "tokencut_retrieve", retrieve(virtual_ref, start_line=2,
                                                                       end_line=3))
             + end("tool_use"))
    second = start(inp=30, out=2) + text_block(0, "It says line two.") + end("end_turn", out=9)
    script = Script(first, second)
    response, ref, _ = run(script, tmp_path=tmp_path)
    assert ref == virtual_ref
    got = events(response.content)
    assert "tokencut_retrieve" not in response.text
    texts = [e["delta"]["text"] for e in got if e["type"] == "content_block_delta"]
    assert texts == ["Let me look.", "It says line two."]
    starts = [e for e in got if e["type"] == "content_block_start"]
    assert [e["index"] for e in starts] == [0, 1]  # one message, indices continue
    assert [e["type"] for e in got].count("message_start") == 1
    assert [e["type"] for e in got].count("message_stop") == 1
    stop = [e for e in got if e["type"] == "message_delta"]
    assert len(stop) == 1 and stop[0]["delta"]["stop_reason"] == "end_turn"
    # upstream saw the call and the real answer
    assert len(script.requests) == 2
    sent = json.loads(script.requests[1].content)
    assistant, user = sent["messages"][-2:]
    assert assistant["role"] == "assistant"
    assert [b["type"] for b in assistant["content"]] == ["text", "tool_use"]
    assert assistant["content"][1]["input"] == retrieve(ref, start_line=2, end_line=3)
    result = user["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_1"
    assert "line two" in result["content"] and "lines 2-3" in result["content"]


def test_billed_usage_adds_up_over_both_calls(tmp_path):
    ref = VirtualConfig().store.put(ORIGINAL)
    first = start(inp=10, out=1) + tool_block(0, "t1", "tokencut_retrieve", retrieve(ref)) \
        + end("tool_use", out=5)
    second = start(inp=30, out=1) + text_block(0, "ok") + end("end_turn", out=7)
    _, _, summary = run(Script(first, second), tmp_path=tmp_path)
    usage = summary["billed_usage"]
    assert usage["input_tokens"] == 40 and usage["output_tokens"] == 12


def test_mixed_with_a_real_tool_is_passed_on_unchanged(tmp_path):
    ref = VirtualConfig().store.put(ORIGINAL)
    first = (start() + tool_block(0, "t1", "tokencut_retrieve", retrieve(ref))
             + tool_block(1, "t2", "Bash", {"command": "ls"}) + end("tool_use"))
    script = Script(first)
    response, _, _ = run(script, tmp_path=tmp_path)
    assert response.content == first  # byte for byte, the old way (the gateway answers next turn)
    assert len(script.requests) == 1


def test_a_real_tool_before_the_retrieve_call_is_passed_on_unchanged(tmp_path):
    ref = VirtualConfig().store.put(ORIGINAL)
    first = (start() + tool_block(0, "t2", "Bash", {"command": "ls"})
             + tool_block(1, "t1", "tokencut_retrieve", retrieve(ref)) + end("tool_use"))
    script = Script(first)
    response, _, _ = run(script, tmp_path=tmp_path)
    assert response.content == first and len(script.requests) == 1


def test_text_after_the_retrieve_call_is_passed_on_unchanged(tmp_path):
    ref = VirtualConfig().store.put(ORIGINAL)
    first = (start() + tool_block(0, "t1", "tokencut_retrieve", retrieve(ref))
             + text_block(1, "and also this") + end("tool_use"))
    script = Script(first)
    response, _, _ = run(script, tmp_path=tmp_path)
    assert response.content == first and len(script.requests) == 1


def test_failed_continuation_falls_back_to_the_old_way(tmp_path):
    ref = VirtualConfig().store.put(ORIGINAL)
    first = (start() + text_block(0, "Looking.")
             + tool_block(1, "t1", "tokencut_retrieve", retrieve(ref)) + end("tool_use"))
    script = Script(first, (500, b'{"error":"boom"}'))
    response, _, _ = run(script, tmp_path=tmp_path)
    assert response.status_code == 200
    assert response.content == first  # the agent gets the call and answers it with an error
    assert len(script.requests) == 2


def test_no_retrieve_call_is_a_plain_passthrough(tmp_path):
    first = start() + text_block(0, "Hello") + end("end_turn")
    script = Script(first)
    response, _, _ = run(script, tmp_path=tmp_path)
    assert response.content == first and len(script.requests) == 1


def test_ccr_off_leaves_the_stream_alone(tmp_path):
    ref = VirtualConfig().store.put(ORIGINAL)
    first = start() + tool_block(0, "t1", "tokencut_retrieve", retrieve(ref)) + end("tool_use")
    script = Script(first)
    response, _, _ = run(script, ccr=False, tmp_path=tmp_path)
    assert response.content == first and len(script.requests) == 1


def test_a_second_retrieve_in_the_answer_is_also_handled(tmp_path):
    ref = VirtualConfig().store.put(ORIGINAL)
    first = start() + text_block(0, "One.") \
        + tool_block(1, "t1", "tokencut_retrieve", retrieve(ref)) + end("tool_use")
    second = start() + text_block(0, "Two.") \
        + tool_block(1, "t2", "tokencut_retrieve", retrieve(ref, start_line=1, end_line=1)) \
        + end("tool_use")
    third = start() + text_block(0, "Done.") + end("end_turn")
    script = Script(first, second, third)
    response, _, _ = run(script, tmp_path=tmp_path)
    got = events(response.content)
    assert "tokencut_retrieve" not in response.text
    texts = [e["delta"]["text"] for e in got if e["type"] == "content_block_delta"]
    assert texts == ["One.", "Two.", "Done."]
    assert [e["index"] for e in got if e["type"] == "content_block_start"] == [0, 1, 2]
    assert len(script.requests) == 3
    last = json.loads(script.requests[2].content)["messages"]
    assert [m["role"] for m in last[-4:]] == ["assistant", "user", "assistant", "user"]


def test_unknown_reference_gets_the_unavailable_answer(tmp_path):
    first = start() + tool_block(0, "t1", "tokencut_retrieve", {"ref": "tc_deadbeef0000"}) \
        + end("tool_use")
    second = start() + text_block(0, "ok") + end("end_turn")
    script = Script(first, second)
    response, _, _ = run(script, tmp_path=tmp_path)
    assert "tokencut_retrieve" not in response.text
    answer = json.loads(script.requests[1].content)["messages"][-1]["content"][0]["content"]
    assert "no longer available" in answer


def test_thinking_block_is_carried_into_the_continuation(tmp_path):
    ref = VirtualConfig().store.put(ORIGINAL)
    thinking = (sse({"type": "content_block_start", "index": 0,
                     "content_block": {"type": "thinking", "thinking": ""}})
                + sse({"type": "content_block_delta", "index": 0,
                       "delta": {"type": "thinking_delta", "thinking": "hmm"}})
                + sse({"type": "content_block_delta", "index": 0,
                       "delta": {"type": "signature_delta", "signature": "SIG"}})
                + sse({"type": "content_block_stop", "index": 0}))
    first = start() + thinking + tool_block(1, "t1", "tokencut_retrieve", retrieve(ref)) \
        + end("tool_use")
    second = start() + text_block(0, "ok") + end("end_turn")
    script = Script(first, second)
    response, _, _ = run(script, tmp_path=tmp_path)
    assistant = json.loads(script.requests[1].content)["messages"][-2]["content"]
    assert assistant[0] == {"type": "thinking", "thinking": "hmm", "signature": "SIG"}
    assert [e["index"] for e in events(response.content) if e["type"] == "content_block_start"] \
        == [0, 1]
