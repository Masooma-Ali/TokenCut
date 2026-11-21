"""Hidden model connector: pinned install, generated config, no secrets in files."""

import json
import subprocess
from pathlib import Path

import pytest

from tokencut import litellm_proxy, userconfig


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENCUT_HOME", str(tmp_path))
    monkeypatch.delenv("TOKENCUT_LITELLM", raising=False)
    monkeypatch.delenv("TOKENCUT_LITELLM_CONFIG", raising=False)
    return tmp_path


def test_pin_matches_deploy_requirements():
    deploy = Path(__file__).resolve().parents[1] / "deploy" / "litellm" / "requirements.txt"
    assert f"litellm[proxy]=={litellm_proxy.LITELLM_VERSION}" in deploy.read_text()


def test_config_claude_only_and_with_chatgpt(home):
    config = userconfig.load()
    names = [m["model_name"] for m in litellm_proxy.config_dict(config)["model_list"]]
    assert names == ["claude-*", "anthropic/*"]
    config["chatgpt"]["enabled"] = True
    models = litellm_proxy.config_dict(config)["model_list"]
    assert models[0] == {"model_name": "gpt-6.1-sol", "model_info": {"mode": "responses"},
                         "litellm_params": {"model": "chatgpt/gpt-6.1-sol"}}
    secret = userconfig.key("litellm")
    written = litellm_proxy.write_config(config).read_text()
    assert json.loads(written)["general_settings"]["master_key"] == "os.environ/LITELLM_MASTER_KEY"
    assert secret not in written


def test_command_binds_locally(home):
    cmd = litellm_proxy.command(userconfig.load())
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "8788"
    assert Path(cmd[cmd.index("--config") + 1]).exists()


def test_install_writes_stamp_only_on_success(home, monkeypatch):
    monkeypatch.setattr(litellm_proxy.shutil, "which", lambda name: None)
    calls = []

    def ok(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    assert not litellm_proxy.installed()
    assert litellm_proxy.install(run=ok)
    assert any(f"litellm[proxy]=={litellm_proxy.LITELLM_VERSION}" in c for c in calls[-1])
    assert (litellm_proxy.venv_dir() / "tokencut-litellm-version").read_text() == \
        litellm_proxy.LITELLM_VERSION
    (litellm_proxy.venv_dir() / "tokencut-litellm-version").unlink()
    assert not litellm_proxy.install(run=lambda c: subprocess.CompletedProcess(c, 1))
    assert not (litellm_proxy.venv_dir() / "tokencut-litellm-version").exists()


def test_installed_needs_matching_version(home):
    exe = litellm_proxy.executable()
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    stamp = litellm_proxy.venv_dir() / "tokencut-litellm-version"
    stamp.write_text("0.0.1")
    assert not litellm_proxy.installed()  # a TokenCut update with a new pin reinstalls
    stamp.write_text(litellm_proxy.LITELLM_VERSION)
    assert litellm_proxy.installed()


def test_uv_install_commands(home):
    commands = litellm_proxy.install_commands("/usr/bin/uv")
    assert commands[0][:2] == ["/usr/bin/uv", "venv"] and commands[1][:3] == ["/usr/bin/uv", "pip",
                                                                              "install"]
