"""The `tokencut` command: version, help, and hand-off of `code` arguments to the launcher."""

import pytest

from tokencut import cli


def test_version(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.startswith("tokencut ")


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 2
    assert "code" in capsys.readouterr().out


def test_code_passes_claude_arguments(monkeypatch):
    seen = []
    monkeypatch.setattr("tokencut.launch.main", lambda argv: seen.append(argv) or 0)
    assert cli.main(["code", "--model", "haiku", "-p", "hi"]) == 0
    assert seen == [["code", "--model", "haiku", "-p", "hi"]]
