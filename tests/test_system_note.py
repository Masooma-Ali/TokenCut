"""The fixed system note (prompt enhancement): tells the model how to treat TokenCut's markers."""

import copy
import json

import httpx2
from fastapi.testclient import TestClient

from tests.test_gateway_trimming import Upstream, conversation
from tokencut.gateway.app import GatewayConfig, create_app
from tokencut.gateway.note import SYSTEM_NOTE, add_system_note
from tokencut.gateway.records import analyse
from tokencut.gateway.trimmer import TrimConfig

ON = TrimConfig(system_note=True)


def claude_code_body() -> dict:
    body = conversation()
    body["system"] = [
        {"type": "text", "text": "You are Claude Code."},
        {"type": "text", "text": "Be careful.", "cache_control": {"type": "ephemeral"}},
    ]
    return body


def sent(body: dict, config: TrimConfig = ON, provider: str = "anthropic") -> dict:
    path = {"anthropic": "v1/messages", "openai": "v1/chat/completions",
            "gemini": "v1beta/models/gemini-3-pro:generateContent"}[provider]
    _, new = analyse(provider, path, json.dumps(body).encode(), config)
    return json.loads(new) if new is not None else body


def test_note_is_appended_after_the_system_blocks_unchanged():
    body = claude_code_body()
    out = sent(body)
    assert out["system"][:2] == body["system"]  # original blocks, cache_control included
    assert out["system"][2] == {"type": "text", "text": SYSTEM_NOTE}
    assert len(SYSTEM_NOTE.split()) < 120  # small: paid (cached) on every request


def test_note_is_identical_on_every_turn():
    turn1 = claude_code_body()
    turn2 = copy.deepcopy(turn1)
    turn2["messages"] += [
        {"role": "assistant", "content": "Found it."},
        {"role": "user", "content": "now fix it"},
    ]
    a, b = sent(turn1), sent(turn2)
    assert json.dumps(a["system"]) == json.dumps(b["system"])  # same bytes: prompt cache safe


def test_note_never_goes_into_user_messages():
    out = sent(claude_code_body())
    assert SYSTEM_NOTE not in json.dumps(out["messages"])


def test_string_system_and_missing_system():
    body = conversation()
    assert sent(body)["system"] == f"You are a coding agent.\n\n{SYSTEM_NOTE}"
    del body["system"]
    assert sent(body)["system"] == SYSTEM_NOTE


def test_off_by_default():
    body = claude_code_body()
    assert sent(body, TrimConfig())["system"] == body["system"]


def test_side_requests_without_tools_get_no_note():
    body = claude_code_body()
    del body["tools"]  # e.g. Claude Code's title / summary calls
    assert sent(body)["system"] == body["system"]


def test_note_is_not_added_twice():
    body = claude_code_body()
    once = sent(body)
    assert add_system_note("anthropic", once) is None
    assert sum(SYSTEM_NOTE in json.dumps(b) for b in once["system"]) == 1


def test_openai_note_follows_the_leading_system_messages():
    body = {
        "model": "gpt-6",
        "tools": [{"type": "function", "function": {"name": n}} for n in ("read", "bash")],
        "messages": [
            {"role": "developer", "content": "You are Cline."},
            {"role": "user", "content": "fix the bug"},
        ],
    }
    out = sent(body, provider="openai")
    assert out["messages"][0] == body["messages"][0]
    assert out["messages"][1] == {"role": "developer", "content": SYSTEM_NOTE}
    assert out["messages"][2] == body["messages"][1]


def test_gemini_note_is_a_system_instruction_part():
    body = {
        "systemInstruction": {"parts": [{"text": "You are Gemini CLI."}]},
        "tools": [{"functionDeclarations": [{"name": "read_file"}, {"name": "run_shell_command"}]}],
        "contents": [{"role": "user", "parts": [{"text": "fix the bug"}]}],
    }
    out = sent(body, provider="gemini")
    assert out["systemInstruction"]["parts"] == [{"text": "You are Gemini CLI."},
                                                 {"text": SYSTEM_NOTE}]
    assert out["contents"] == body["contents"]


def test_note_is_reported_as_added_tokens():
    analysis, _ = analyse("anthropic", "v1/messages", json.dumps(claude_code_body()).encode(), ON)
    entry = next(t for t in analysis["trim"]["details"] if t["how"] == "system_note")
    assert entry["before"] == 0 and entry["after"] > 0
    real = [t for t in analysis["trim"]["details"] if t["how"] != "system_note"]
    assert analysis["trim"]["segments_trimmed"] == len(real)  # the note is not a trimmed segment


def gateway(mode: str, tmp_path):
    upstream = Upstream()
    config = GatewayConfig(log_dir=tmp_path, mode=mode, trim=ON)
    return create_app(config, transport=httpx2.MockTransport(upstream.handler)), upstream


def test_trim_mode_sends_the_note_and_shadow_mode_does_not(tmp_path):
    for mode, expected in (("trim", True), ("shadow", False)):
        app, upstream = gateway(mode, tmp_path / mode)
        with TestClient(app) as client:
            assert client.post("/anthropic/v1/messages", json=claude_code_body()).status_code == 200
        assert (SYSTEM_NOTE in json.dumps(json.loads(upstream.bodies[0])["system"])) is expected


def test_setting_from_env(monkeypatch):
    monkeypatch.delenv("TOKENCUT_SYSTEM_NOTE", raising=False)
    assert GatewayConfig.from_env().trim.system_note is False
    monkeypatch.setenv("TOKENCUT_SYSTEM_NOTE", "1")
    assert GatewayConfig.from_env().trim.system_note is True
