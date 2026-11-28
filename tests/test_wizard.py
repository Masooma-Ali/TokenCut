"""`tokencut setup` with every outside step faked (no installs, no logins, no network)."""

import json

import pytest

from tokencut import userconfig
from tokencut.wizard import run_setup


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENCUT_HOME", str(tmp_path))
    return tmp_path


def setup(**overrides):
    lines: list[str] = []
    kwargs = {"ask": lambda q: "", "out": lines.append, "which": lambda name: "/usr/bin/claude",
              "install_connector": lambda: True, "connector_ready": lambda: False,
              "chatgpt_ready": lambda: False, "chatgpt_login": lambda: True,
              "download_router": lambda out: True}
    kwargs.update(overrides)
    return run_setup(**kwargs), "\n".join(lines)


def test_without_claude_code_prints_install_command(home):
    code, text = setup(which=lambda name: None, os_name="posix")
    assert code == 1 and "https://claude.ai/install.sh" in text
    assert not (home / "config.json").exists()


def test_claude_only(home):
    code, text = setup(chatgpt=False)
    config = json.loads((home / "config.json").read_text())
    assert code == 0 and config["litellm"]["enabled"] and not config["chatgpt"]["enabled"]
    assert (home / "litellm_config.yaml").exists()
    assert userconfig.key("tokencut", create=False) and userconfig.key("litellm", create=False)
    assert "Models: Claude\n" in text + "\n"


def test_chatgpt_asked_and_signed_in(home):
    state = {"in": False}

    def login():
        state["in"] = True
        return True

    code, text = setup(ask=lambda q: "y", chatgpt_ready=lambda: state["in"], chatgpt_login=login)
    config = userconfig.load()
    assert code == 0 and config["chatgpt"]["enabled"]
    assert "gpt-6.1-sol" in (home / "litellm_config.yaml").read_text()
    assert "ChatGPT sign-in" in text


def test_failures_leave_a_working_claude_setup(home):
    code, text = setup(chatgpt=True, install_connector=lambda: False,
                       download_router=lambda out: False)
    config = userconfig.load()
    assert code == 0
    assert not config["litellm"]["enabled"] and not config["chatgpt"]["enabled"]
    assert config["gateway"]["prompt_brief"] is False
    assert "Claude will connect directly" in text


def test_prompt_model_is_downloaded_for_ode_even_without_briefs(home):
    config = userconfig.load()
    config["gateway"]["prompt_brief"] = False
    userconfig.save(config)
    asked = []
    code, text = setup(chatgpt=False, download_router=lambda out: asked.append(1) or True)
    assert code == 0 and asked and "ODE (model per task): on - Claude models only" in text


def test_ode_survives_a_failed_prompt_model_download(home):
    code, text = setup(chatgpt=False, download_router=lambda out: False)
    assert code == 0 and userconfig.load()["ode"]["mode"] == "on"
    assert "ODE starts choosing once it downloads" in text
