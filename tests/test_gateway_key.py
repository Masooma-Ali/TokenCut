"""TokenCut key (TOKENCUT_KEY): only local clients given the key may use the gateway."""

import httpx2
from fastapi.testclient import TestClient

from tests.test_gateway_app import REQUEST, FakeProvider
from tokencut.gateway.app import GatewayConfig, create_app


def client(key: str | None) -> tuple[TestClient, FakeProvider]:
    fake = FakeProvider()
    app = create_app(GatewayConfig(log_dir=None, client_key=key),
                     transport=httpx2.MockTransport(fake.handler))
    return TestClient(app), fake


def test_without_key_setting_everything_passes():
    tc, fake = client(None)
    with tc:
        assert tc.post("/anthropic/v1/messages", json=REQUEST).status_code == 200
    assert len(fake.requests) == 1


def test_missing_or_wrong_key_is_rejected_before_the_provider():
    tc, fake = client("tc-secret")
    with tc:
        missing = tc.post("/anthropic/v1/messages", json=REQUEST)
        wrong = tc.post("/anthropic/v1/messages", json=REQUEST, headers={"x-tokencut-key": "no"})
    assert missing.status_code == wrong.status_code == 401
    assert missing.json()["error"]["type"] == "authentication_error"
    assert fake.requests == []


def test_right_key_passes_and_is_not_forwarded():
    tc, fake = client("tc-secret")
    with tc:
        ok = tc.post("/anthropic/v1/messages", json=REQUEST,
                     headers={"x-tokencut-key": "tc-secret", "authorization": "Bearer user"})
    assert ok.status_code == 200
    sent = fake.requests[0].headers
    assert "x-tokencut-key" not in sent
    assert sent["authorization"] == "Bearer user"  # the user's provider login is untouched


def test_status_pages_stay_open():
    tc, _ = client("tc-secret")
    with tc:
        assert tc.get("/tokencut/health").status_code == 200
        assert tc.get("/tokencut/stats").status_code == 200
