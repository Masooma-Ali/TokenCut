"""The `tokencut` command: version, default launch, argument hand-off, status text."""

import pytest

from tokencut import cli


@pytest.fixture
def launched(monkeypatch):
    seen = []
    monkeypatch.setattr("tokencut.launch.main", lambda argv: seen.append(argv) or 0)
    return seen


def test_version(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.startswith("tokencut ")


def test_no_command_opens_claude_code(launched):
    assert cli.main([]) == 0
    assert launched == [["code"]]


@pytest.mark.parametrize("argv", [["code", "--model", "haiku", "-p", "hi"],
                                  ["--model", "haiku", "-p", "hi"]])
def test_claude_arguments_pass_through(launched, argv):
    assert cli.main(argv) == 0
    assert launched == [["code", "--model", "haiku", "-p", "hi"]]


def test_setup_flags(monkeypatch):
    seen = []
    monkeypatch.setattr("tokencut.wizard.run_setup", lambda chatgpt: seen.append(chatgpt) or 0)
    for argv in (["setup"], ["setup", "--chatgpt"], ["setup", "--no-chatgpt"]):
        assert cli.main(argv) == 0
    assert seen == [None, True, False]


def test_status_text():
    gateway = "http://127.0.0.1:8787"
    assert "not running" in cli.status_text(None, None, gateway)
    health = {"mode": "trim", "tier": "moderate", "gpt_upstream": "http://127.0.0.1:8788",
              "task_masking": True}
    stats = {"requests": 4, "tokens_total": 1000,
             "trimming": {"tokens_saved_applied": 250, "tokens_saved_possible": 300},
             "billed_usage": {"input_tokens": 10, "cache_read_tokens": 2000,
                              "cache_write_tokens": 30, "output_tokens": 5}}
    text = cli.status_text(health, stats, gateway)
    assert "mode trim" in text and "Claude + GPT" in text and "task_masking" in text
    assert "tokens removed: 250 (25% of 1,000)" in text


def test_logs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TOKENCUT_HOME", str(tmp_path))
    assert cli.main(["logs"]) == 1
    (tmp_path / "gateway.log").write_text("".join(f"line {i}\n" for i in range(100)))
    assert cli.main(["logs", "-n", "2"]) == 0
    assert capsys.readouterr().out.endswith("line 98\nline 99\n")
