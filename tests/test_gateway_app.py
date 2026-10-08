"""Gateway server in shadow mode, against a fake provider (no network)."""

import json

import httpx2
import pytest
from fastapi.testclient import TestClient

from tokencut.gateway.app import GatewayConfig, create_app
from tokencut.gateway.usage import UsageSniffer

ANTHROPIC_SSE = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":{"usage":{"input_tokens":12,'
    b'"cache_read_input_tokens":3000,"cache_creation_input_tokens":40,"output_tokens":1}}}\n\n'
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}\n\n'
    b"event: message_delta\n"
    b'data: {"type":"message_delta","usage":{"output_tokens":57}}\n\n'
)
REQUEST = {
    "model": "claude-haiku-4-5",
    "max_tokens": 50,
    "stream": True,
    "system": "Be brief.",
    "messages": [{"role": "user", "content": "hello"}],
}


class ChunkedBody(httpx2.AsyncByteStream):
    """A response body that arrives in small pieces, like a real network stream."""

    def __init__(self, data: bytes, size: int = 16):
        self.data, self.size = data, size

    async def __aiter__(self):
        for i in range(0, len(self.data), self.size):
            yield self.data[i : i + self.size]


def reply(status: int, data: bytes, content_type: str = "application/json") -> httpx2.Response:
    return httpx2.Response(status, headers={"content-type": content_type}, stream=ChunkedBody(data))


def reply_json(status: int, body: dict) -> httpx2.Response:
    return reply(status, json.dumps(body).encode())


class FakeProvider:
    """Records what the gateway sent upstream and answers like the real provider."""

    def __init__(self):
        self.requests: list[httpx2.Request] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/v1/messages":
            return reply(200, ANTHROPIC_SSE, "text/event-stream")
        if path == "/v1/chat/completions":
            body = {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 60},
                },
            }
            return reply_json(200, body)
        if path.startswith("/v1beta/models/"):
            body = {
                "candidates": [],
                "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 5},
            }
            return reply_json(200, body)
        if path == "/v1/models":
            return reply_json(200, {"data": []})
        return reply_json(404, {"error": "not found"})


@pytest.fixture
def setup(tmp_path):
    fake = FakeProvider()
    config = GatewayConfig(log_dir=tmp_path / "logs")
    app = create_app(config, transport=httpx2.MockTransport(fake.handler))
    with TestClient(app) as client:
        yield client, fake, app, tmp_path / "logs"


def test_anthropic_request_forwarded_unchanged_and_measured(setup):
    client, fake, app, log_dir = setup
    raw = json.dumps(REQUEST).encode()
    response = client.post(
        "/anthropic/v1/messages",
        content=raw,
        headers={
            "x-api-key": "sk-test",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    assert response.status_code == 200
    assert response.content == ANTHROPIC_SSE  # stream relayed byte for byte

    sent = fake.requests[0]
    assert sent.content == raw  # shadow mode: body untouched
    assert str(sent.url) == "https://api.anthropic.com/v1/messages"
    assert sent.headers["x-api-key"] == "sk-test"
    assert sent.headers["anthropic-version"] == "2023-06-01"
    assert sent.headers["accept-encoding"] == "identity"  # else usage can't be read

    record = app.state.request_log.recent[-1]
    assert record["status"] == 200 and record["provider"] == "anthropic"
    assert record["analysis"]["model"] == "claude-haiku-4-5"
    assert record["analysis"]["tokens_by_kind"]["system"] > 0
    assert record["usage"] == {
        "input_tokens": 12,
        "cache_read_tokens": 3000,
        "cache_write_tokens": 40,
        "output_tokens": 57,
    }
    logged = (log_dir / f"requests-{record['ts'][:10]}.jsonl").read_text(encoding="utf-8")
    assert "sk-test" not in logged and "hello" not in logged  # numbers only, no secrets or text


def test_openai_and_gemini(setup):
    client, fake, app, _ = setup
    body = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}
    r = client.post(
        "/openai/v1/chat/completions", json=body, headers={"authorization": "Bearer sk-o"}
    )
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "ok"
    assert fake.requests[-1].headers["authorization"] == "Bearer sk-o"
    openai_record = app.state.request_log.recent[-1]
    assert openai_record["usage"]["cache_read_tokens"] == 60
    assert openai_record["usage"]["input_tokens"] == 40

    gbody = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    r = client.post("/gemini/v1beta/models/gemini-3-flash:generateContent?key=g-key", json=gbody)
    assert r.status_code == 200
    assert fake.requests[-1].url.params["key"] == "g-key"  # query string forwarded
    gemini_record = app.state.request_log.recent[-1]
    assert gemini_record["analysis"]["model"] == "gemini-3-flash"
    assert "key" not in gemini_record["path"]  # API key in the query is never logged
    assert gemini_record["usage"]["input_tokens"] == 30


def test_other_paths_are_forwarded_but_not_analysed(setup):
    client, _fake, app, _ = setup
    r = client.get("/anthropic/v1/models")
    assert r.status_code == 200
    assert client.head("/anthropic/v1/models").status_code == 200  # Claude Code's startup check
    record = app.state.request_log.recent[-1]
    assert record["analysis"] is None and "analysis_error" not in record


def test_bad_request_body_still_forwarded(setup):
    client, fake, app, _ = setup
    r = client.post(
        "/anthropic/v1/messages", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert r.status_code == 200  # fail open: the provider decides, not the gateway
    assert fake.requests[-1].content == b"{not json"
    assert "analysis_error" in app.state.request_log.recent[-1]


def test_upstream_error_status_passes_through(setup):
    client, *_ = setup
    r = client.post("/openai/v1/unknown", json={})
    assert r.status_code == 404 and r.json() == {"error": "not found"}


def test_unreachable_provider_gives_502(tmp_path):
    def broken(request):
        raise httpx2.ConnectError("no route")

    app = create_app(GatewayConfig(log_dir=None), transport=httpx2.MockTransport(broken))
    with TestClient(app) as client:
        r = client.post("/anthropic/v1/messages", json=REQUEST)
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "tokencut_upstream_error"
    assert app.state.request_log.recent[-1]["status"] == 502


def test_stats_and_health(setup):
    client, *_ = setup
    client.post("/anthropic/v1/messages", json=REQUEST)
    assert client.get("/tokencut/health").json()["mode"] == "shadow"
    stats = client.get("/tokencut/stats").json()
    assert stats["requests"] == 1 and stats["analysed"] == 1
    assert stats["by_provider"] == {"anthropic": 1}
    assert stats["billed_usage"]["cache_read_tokens"] == 3000
    assert abs(sum(stats["share_by_kind"].values()) - 1.0) < 0.01


def test_capture_saves_bodies_only_when_enabled(tmp_path):
    fake = FakeProvider()
    app = create_app(
        GatewayConfig(log_dir=tmp_path, capture=True), transport=httpx2.MockTransport(fake.handler)
    )
    with TestClient(app) as client:
        client.post("/anthropic/v1/messages", json=REQUEST)
    captures = list((tmp_path / "captures").glob("*_anthropic.json"))
    assert len(captures) == 1 and json.loads(captures[0].read_bytes()) == REQUEST


def test_sniffer_handles_split_lines_and_compression():
    sniffer = UsageSniffer("anthropic", "text/event-stream")
    for i in range(0, len(ANTHROPIC_SSE), 7):  # chunks cut mid-line
        sniffer.feed(ANTHROPIC_SSE[i : i + 7])
    assert sniffer.close().output_tokens == 57
    gzipped = UsageSniffer("openai", "application/json", "gzip")
    gzipped.feed(b"\x1f\x8b...")
    assert gzipped.close() is None
