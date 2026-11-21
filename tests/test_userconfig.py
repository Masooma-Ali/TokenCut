"""~/.tokencut/config.json, private keys, and the environments built from them."""

import os
import stat

import pytest

from tokencut import userconfig


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENCUT_HOME", str(tmp_path))
    return tmp_path


def test_defaults_and_saved_values_merge():
    assert userconfig.load() == userconfig.DEFAULTS
    config = userconfig.load()
    config["gateway"]["tier"] = "aggressive"
    userconfig.save(config)
    loaded = userconfig.load()
    assert loaded["gateway"]["tier"] == "aggressive"
    assert loaded["gateway"]["system_note"] is True  # untouched defaults stay


def test_file_without_new_keys_gets_defaults(home):
    (home / "config.json").write_text('{"gateway": {"mode": "shadow"}}')
    loaded = userconfig.load()
    assert loaded["gateway"]["mode"] == "shadow" and loaded["litellm"]["port"] == 8788


def test_keys_are_private_and_stable(home):
    first = userconfig.key("tokencut")
    assert first.startswith("tc-") and userconfig.key("tokencut") == first
    assert userconfig.key("litellm").startswith("sk-")
    if os.name != "nt":
        assert stat.S_IMODE((home / "tokencut_key").stat().st_mode) == 0o600
    assert userconfig.key("tokencut", create=False) == first


def test_gateway_env_with_connector():
    userconfig.key("tokencut")
    userconfig.key("litellm")
    env = userconfig.gateway_env(userconfig.load(), "http://127.0.0.1:8788", {})
    assert env["TOKENCUT_MODE"] == "trim" and env["TOKENCUT_PROMPT_BRIEF"] == "1"
    assert env["TOKENCUT_GPT_UPSTREAM"] == "http://127.0.0.1:8788"
    assert env["TOKENCUT_CLAUDE_VIA_LITELLM"] == "1"
    assert env["TOKENCUT_KEY"] == userconfig.key("tokencut")
    assert env["TOKENCUT_GPT_KEY"] == userconfig.key("litellm")


def test_gateway_env_user_settings_win_and_no_connector():
    env = userconfig.gateway_env(userconfig.load(), None, {"TOKENCUT_MODE": "shadow"})
    assert env["TOKENCUT_MODE"] == "shadow"
    assert "TOKENCUT_GPT_UPSTREAM" not in env and "TOKENCUT_KEY" not in env


def test_agent_env_sends_key_and_keeps_other_headers():
    secret = userconfig.key("tokencut")
    env = userconfig.agent_env(userconfig.load(), {
        "ANTHROPIC_CUSTOM_HEADERS": "x-team: a\nx-tokencut-key: old"})
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787/anthropic"
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == f"x-team: a\nx-tokencut-key: {secret}"
