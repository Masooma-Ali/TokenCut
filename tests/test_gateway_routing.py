"""Dual-plan demo: Claude models go to Anthropic, GPT models go to a local LiteLLM proxy."""

import json

import httpx2
import pytest
from fastapi.testclient import TestClient

from tests.test_gateway_app import ANTHROPIC_SSE, REQUEST, FakeProvider
from tokencut.gateway.app import GatewayConfig, create_app
from tokencut.gateway.routing import (
    RouteConfig,
    gpt_headers,
    model_from_body,
    move_tool_documents,
    pick_route,
)

GPT_UPSTREAM = "http://127.0.0.1:4000"


def _body(model: str) -> bytes:
    return json.dumps({**REQUEST, "model": model}).encode()


@pytest.fixture
def routed(tmp_path):
    fake = FakeProvider()
    config = GatewayConfig(log_dir=tmp_path / "logs", route=RouteConfig(gpt_upstream=GPT_UPSTREAM))
    app = create_app(config, transport=httpx2.MockTransport(fake.handler))
    with TestClient(app) as client:
        yield client, fake, app


def _post(client, model: str):
    return client.post(
        "/anthropic/v1/messages?beta=true",
        content=_body(model),
        headers={
            "authorization": "Bearer claude-plan-token",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )


def test_pick_route():
    cfg = RouteConfig(gpt_upstream=GPT_UPSTREAM)
    assert pick_route("anthropic", "gpt-6.1-sol", cfg) == "gpt"
    assert pick_route("anthropic", "GPT-6-luna", cfg) == "gpt"
    assert pick_route("anthropic", "claude-sonnet-5-5", cfg) == "claude"
    assert pick_route("anthropic", "", cfg) == "claude"
    assert pick_route("openai", "gpt-6.1-sol", cfg) == "openai"  # only Claude Code traffic is routed
    assert pick_route("anthropic", "gpt-6.1-sol", RouteConfig()) == "claude"  # routing off


def test_model_from_body_never_raises():
    assert model_from_body(_body("gpt-6.1-sol")) == "gpt-6.1-sol"
    assert model_from_body(b"not json") == ""
    assert model_from_body(b"[1, 2]") == ""


def test_gpt_headers_drop_the_claude_login():
    sent = gpt_headers({"authorization": "Bearer x", "x-api-key": "k", "anthropic-version": "1"},
                       RouteConfig(gpt_upstream=GPT_UPSTREAM))
    assert "authorization" not in sent and "x-api-key" not in sent
    assert sent["anthropic-version"] == "1"
    keyed = gpt_headers({}, RouteConfig(gpt_upstream=GPT_UPSTREAM, gpt_key="sk-local"))
    assert keyed["authorization"] == "Bearer sk-local"


def test_gpt_model_goes_to_litellm_without_claude_token(routed):
    client, fake, app = routed
    response = _post(client, "gpt-6.1-sol")
    assert response.status_code == 200 and response.content == ANTHROPIC_SSE
    sent = fake.requests[-1]
    assert str(sent.url) == f"{GPT_UPSTREAM}/v1/messages?beta=true"
    assert "authorization" not in sent.headers  # the Claude plan login never leaves for GPT
    record = app.state.request_log.recent[-1]
    assert record["route"] == "gpt" and record["usage"]["output_tokens"] == 57


def test_claude_model_stays_on_anthropic_with_its_login(routed):
    client, fake, app = routed
    _post(client, "claude-sonnet-5-5")
    sent = fake.requests[-1]
    assert str(sent.url) == "https://api.anthropic.com/v1/messages?beta=true"
    assert sent.headers["authorization"] == "Bearer claude-plan-token"
    assert app.state.request_log.recent[-1]["route"] == "claude"


def test_stats_split_by_route_and_recent_feed(routed):
    client, _, _ = routed
    _post(client, "gpt-6.1-sol")
    _post(client, "claude-sonnet-5-5")
    _post(client, "claude-sonnet-5-5")
    stats = client.get("/tokencut/stats").json()
    assert stats["by_route"]["gpt"]["requests"] == 1
    assert stats["by_route"]["claude"]["requests"] == 2
    assert stats["by_route"]["claude"]["billed_usage"]["cache_read_tokens"] == 6000
    recent = client.get("/tokencut/recent").json()
    assert [r["route"] for r in recent] == ["claude", "claude", "gpt"]  # newest first
    assert set(recent[0]) >= {"ts", "route", "model", "tokens", "tokens_saved", "usage"}
    assert "hello" not in json.dumps(recent)  # numbers only


def test_live_page_is_served(routed):
    client, _, _ = routed
    page = client.get("/tokencut/")
    assert page.status_code == 200 and "TokenCut" in page.text


# ---------- Claude through LiteLLM too (TOKENCUT_CLAUDE_VIA_LITELLM=1) ----------


class DownLiteLLM(FakeProvider):
    """LiteLLM not running: every request to it fails to connect."""

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.host == "127.0.0.1":
            raise httpx2.ConnectError("connection refused", request=request)
        return super().handler(request)


def _via(tmp_path, provider_cls=FakeProvider):
    fake = provider_cls()
    route = RouteConfig(gpt_upstream=GPT_UPSTREAM, gpt_key="sk-local", claude_via_litellm=True)
    config = GatewayConfig(log_dir=tmp_path / "logs", route=route)
    app = create_app(config, transport=httpx2.MockTransport(fake.handler))
    return fake, app


def test_claude_via_litellm_keeps_the_claude_login_and_adds_the_proxy_key(tmp_path):
    fake, app = _via(tmp_path)
    with TestClient(app) as client:
        response = _post(client, "claude-haiku-5-5")
    assert response.status_code == 200
    sent = fake.requests[-1]
    assert str(sent.url) == f"{GPT_UPSTREAM}/v1/messages?beta=true"
    assert sent.headers["authorization"] == "Bearer claude-plan-token"  # forwarded by LiteLLM
    assert sent.headers["x-litellm-api-key"] == "Bearer sk-local"
    record = app.state.request_log.recent[-1]
    assert record["route"] == "claude" and record["via"] == "litellm"


def test_gpt_via_litellm_unchanged(tmp_path):
    fake, app = _via(tmp_path)
    with TestClient(app) as client:
        _post(client, "gpt-6.1-sol")
    sent = fake.requests[-1]
    assert str(sent.url) == f"{GPT_UPSTREAM}/v1/messages?beta=true"
    assert sent.headers["authorization"] == "Bearer sk-local"  # Claude login removed
    assert app.state.request_log.recent[-1]["via"] == "litellm"


def test_claude_falls_back_to_anthropic_when_litellm_is_down(tmp_path):
    fake, app = _via(tmp_path, DownLiteLLM)
    with TestClient(app) as client:
        response = _post(client, "claude-haiku-5-5")
    assert response.status_code == 200 and response.content == ANTHROPIC_SSE
    sent = fake.requests[-1]
    assert str(sent.url) == "https://api.anthropic.com/v1/messages?beta=true"
    assert "x-litellm-api-key" not in sent.headers  # the proxy key never goes to Anthropic
    record = app.state.request_log.recent[-1]
    assert record["via"] == "direct" and record["fallback"] == "litellm_unreachable"


def test_non_inference_claude_paths_stay_direct(tmp_path):
    fake, app = _via(tmp_path)
    with TestClient(app) as client:
        client.post("/anthropic/v1/messages/count_tokens", content=_body("claude-haiku-5-5"),
                    headers={"authorization": "Bearer claude-plan-token"})
    assert str(fake.requests[-1].url).startswith("https://api.anthropic.com/")


def test_claude_via_litellm_needs_the_gpt_upstream():
    cfg = RouteConfig(claude_via_litellm=True)  # no LiteLLM address: nothing changes
    assert pick_route("anthropic", "claude-haiku-5-5", cfg) == "claude"


PDF_BLOCK = {"type": "document", "title": "report.pdf",
             "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjcK"}}


def _pdf_read(model: str) -> dict:
    return {"model": model, "max_tokens": 10, "messages": [
        {"role": "user", "content": "What are the payment terms?"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "r1", "name": "Read",
                                           "input": {"file_path": "report.pdf"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "r1", "content": [
            {"type": "text", "text": "PDF file read: report.pdf"}, PDF_BLOCK]}]},
    ]}


def test_move_tool_documents_puts_pdfs_after_the_tool_results():
    # Found by the attachment benchmark: ChatGPT (via LiteLLM) rejects a PDF inside a tool
    # result (400, "expected an image MIME type"), but reads it next to the tool result.
    body = _pdf_read("gpt-6-luna")
    moved, count = move_tool_documents(body)
    assert count == 1
    content = moved["messages"][2]["content"]
    assert [b["type"] for b in content] == ["tool_result", "document"]
    assert content[1] == PDF_BLOCK
    inner = content[0]["content"]
    assert [b["type"] for b in inner] == ["text", "text"] and "report.pdf" in inner[1]["text"]
    assert body["messages"][2]["content"][0]["content"][1] == PDF_BLOCK  # input not modified
    assert move_tool_documents(moved) == (moved, 0)  # nothing left to move: same every turn


def test_move_tool_documents_leaves_other_requests_alone():
    body = {**REQUEST}
    assert move_tool_documents(body) == (body, 0)
    images_only = _pdf_read("gpt-6-luna")
    images_only["messages"][2]["content"][0]["content"][1] = {
        "type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBO"}}
    assert move_tool_documents(images_only)[1] == 0  # LiteLLM already handles images


def _post_body(client, body: dict):
    return client.post("/anthropic/v1/messages", content=json.dumps(body).encode(), headers={
        "authorization": "Bearer claude-plan-token", "anthropic-version": "2023-06-01",
        "content-type": "application/json"})


def test_gpt_route_moves_tool_pdfs_in_shadow_and_trim(tmp_path):
    for mode in ("shadow", "trim"):
        fake = FakeProvider()
        config = GatewayConfig(mode=mode, log_dir=tmp_path / mode,
                               route=RouteConfig(gpt_upstream=GPT_UPSTREAM))
        app = create_app(config, transport=httpx2.MockTransport(fake.handler))
        with TestClient(app) as client:
            assert _post_body(client, _pdf_read("gpt-6-luna")).status_code == 200
            sent = json.loads(fake.requests[-1].content)
            assert [b["type"] for b in sent["messages"][2]["content"]] == ["tool_result", "document"]
            assert app.state.request_log.recent[-1]["gpt_documents_moved"] == 1


def test_claude_route_keeps_tool_pdfs_in_place(routed):
    client, fake, _ = routed
    _post_body(client, _pdf_read("claude-haiku-5-5"))
    sent = json.loads(fake.requests[-1].content)
    assert [b["type"] for b in sent["messages"][2]["content"]] == ["tool_result"]
