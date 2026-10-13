"""Phase 3: cache-safe trimming of large tool results inside provider requests."""

import copy
import json

import httpx2
from fastapi.testclient import TestClient

from tests.test_gateway_app import reply_json  # streamed fake-provider response
from tokencut.contracts import SegmentKind, Tier
from tokencut.gateway import trimmer
from tokencut.gateway.adapters import anthropic, gemini, openai
from tokencut.gateway.app import GatewayConfig, create_app
from tokencut.gateway.codeview import parse_listing, trim_file_listing
from tokencut.gateway.textcut import trim_output
from tokencut.gateway.trimmer import TrimConfig, trim_request
from tokencut.tokens import count_tokens


def big_module(n: int = 40) -> str:
    parts = ['"""Billing helpers."""', "import math", ""]
    for i in range(n):
        parts.append(f"def helper_{i}(value):")
        parts.append(f'    """Helper number {i}."""')
        parts.extend(f"    value = value * {j} + math.sqrt({i + j})" for j in range(6))
        parts.append("    return value")
        parts.append("")
    parts.append("def compute_total(items):")
    parts.append("    return sum(helper_3(x) for x in items)")
    return "\n".join(parts) + "\n"


def numbered(source: str, start: int = 1) -> str:
    """Claude Code's Read tool format."""
    return "\n".join(f"{i:>6}→{line}" for i, line in enumerate(source.split("\n"), start))


def pytest_log(n_pass: int = 400) -> str:
    lines = ["===== test session starts ====="]
    lines += [f"tests/test_m.py::test_{i} PASSED" for i in range(n_pass)]
    lines += [
        "tests/test_bill.py::test_total FAILED",
        "===== FAILURES =====",
        ">       assert compute_total([1]) == 2",
        "E       assert 7.0 == 2",
        "tests/test_bill.py:9: AssertionError",
        f"===== 1 failed, {n_pass} passed in 1.0s =====",
    ]
    return "\n".join(lines)


SOURCE = big_module()


def conversation(user_text: str = "compute_total returns the wrong value") -> dict:
    return {
        "model": "claude-haiku-4-5",
        "max_tokens": 100,
        "system": "You are a coding agent.",
        "tools": [{"name": "Read"}, {"name": "Bash"}],
        "messages": [
            {"role": "user", "content": user_text},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "r1",
                        "name": "Read",
                        "input": {"file_path": "/repo/billing/helpers.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "b1",
                        "name": "Bash",
                        "input": {"command": "pytest -q"},
                    },
                    {"type": "tool_use", "id": "b2", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "r1", "content": numbered(SOURCE)},
                    {"type": "tool_result", "tool_use_id": "b1", "content": pytest_log()},
                    {"type": "tool_result", "tool_use_id": "b2", "content": "a.py\nb.py"},
                ],
            },
        ],
    }


def run(body: dict, config: TrimConfig | None = None):
    return trim_request(body, anthropic.parse(body), config or TrimConfig())


# ---------- file listings ----------


def test_parse_listing_numbered_and_plain():
    view = parse_listing(numbered("a = 1\nb = 2", start=41))
    assert view.code == ["a = 1", "b = 2"] and view.first_line == 41
    assert view.prefixes[0].strip() == "41→"
    plain = parse_listing("a = 1\nb = 2\n")
    assert plain.code == ["a = 1", "b = 2"] and plain.prefixes == ["", ""]


def test_file_trim_keeps_relevant_code_verbatim_with_markers():
    listing = numbered(SOURCE)
    out = trim_file_listing(
        listing, "billing/helpers.py", "compute_total helper_3", Tier.AGGRESSIVE
    )
    assert out is not None and count_tokens(out) < count_tokens(listing) / 2
    kept = set(out.split("\n"))
    # the mentioned function and the helper it calls are kept, original lines and numbers
    for line in listing.split("\n"):
        if "def compute_total" in line or "def helper_3(" in line:
            assert line in kept
    assert "[TokenCut: lines " in out and "of billing/helpers.py omitted" in out
    assert "def helper_30(" not in out  # unrelated code dropped


def test_file_trim_skips_non_python_and_tiny_files():
    assert (
        trim_file_listing(numbered("just text\n" * 300), "notes.py", "x", Tier.AGGRESSIVE) is None
    )
    assert trim_file_listing("def f():\n    return 1\n", "f.py", "f", Tier.AGGRESSIVE) is None


def test_offset_read_reports_real_line_numbers():
    listing = numbered(SOURCE, start=101)
    out = trim_file_listing(listing, "h.py", "compute_total", Tier.AGGRESSIVE)
    first_marker = next(line for line in out.split("\n") if "[TokenCut: lines" in line)
    first = int(first_marker.split("lines ")[1].split("-")[0])
    assert first > 100  # numbering follows the listing, not 1


# ---------- command output ----------


def test_output_trim_keeps_failures_and_summary():
    log = pytest_log()
    out = trim_output(log, 300, "test output")
    assert count_tokens(out) < count_tokens(log) / 3
    for needle in ("test_total FAILED", "E       assert 7.0 == 2", "1 failed, 400 passed"):
        assert needle in out
    assert "lines of test output omitted]" in out
    assert trim_output("short", 300) is None


# ---------- whole requests ----------


def test_trim_request_trims_large_results_only():
    body = conversation()
    original = copy.deepcopy(body)
    new_body, report = run(body)
    assert body == original  # input is never modified in place
    results = new_body["messages"][2]["content"]
    assert "[TokenCut: lines" in results[0]["content"]  # file listing -> code trim
    assert "lines of command output omitted]" in results[1]["content"]
    assert results[2]["content"] == "a.py\nb.py"  # small result untouched
    hows = {d["segment"]: d["how"] for d in report.trimmed}
    assert hows == {"m2.b0": "code", "m2.b1": "command_output"}
    assert report.saved > 0
    # everything that is not a tool result is identical, and every tool call keeps its result
    for key in ("system", "tools", "model"):
        assert new_body[key] == original[key]
    assert new_body["messages"][:2] == original["messages"][:2]
    assert [r["tool_use_id"] for r in results] == ["r1", "b1", "b2"]


def test_small_requests_are_not_changed():
    new_body, report = run(conversation(), TrimConfig(min_tokens=10**6))
    assert new_body is None and report.trimmed == []


def test_file_named_by_user_is_protected():
    new_body, report = run(conversation("please fix billing/helpers.py, compute_total is wrong"))
    assert [d["how"] for d in report.trimmed] == ["command_output"]
    assert new_body["messages"][2]["content"][0]["content"] == numbered(SOURCE)


def test_file_already_edited_is_protected():
    body = conversation()
    body["messages"][1]["content"].insert(
        0,
        {
            "type": "tool_use",
            "id": "e0",
            "name": "Edit",
            "input": {
                "file_path": "/repo/billing/helpers.py",
                "old_string": "a",
                "new_string": "b",
            },
        },
    )
    _, report = run(body)
    assert "code" not in [d["how"] for d in report.trimmed]


def test_cat_through_bash_is_treated_as_code():
    body = conversation()
    body["messages"][1]["content"][0] = {
        "type": "tool_use",
        "id": "r1",
        "name": "Bash",
        "input": {"command": "cat billing/helpers.py"},
    }
    body["messages"][2]["content"][0]["content"] = SOURCE  # plain, no line numbers
    _, report = run(body)
    assert report.trimmed[0]["how"] == "code"


def test_cache_safety_earlier_turns_trim_identically():
    """Trimming turn N+1 must leave the first N messages exactly as trimming turn N did,
    even when the new messages mention the file (which would protect it if it were new)."""
    turn1 = conversation()
    turn2 = copy.deepcopy(turn1)
    turn2["messages"] += [
        {"role": "assistant", "content": [{"type": "text", "text": "Found it."}]},
        {"role": "user", "content": "now also look at billing/helpers.py helper_30"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "b3", "name": "Bash", "input": {"command": "pytest -x"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "b3", "content": pytest_log(500)}],
        },
    ]
    trimmed1, _ = run(turn1)
    trimmed2, report2 = run(turn2)
    assert trimmed2["messages"][: len(turn1["messages"])] == trimmed1["messages"]
    assert "m6.b0" in {d["segment"] for d in report2.trimmed}  # the new result is trimmed too


def test_trimming_is_deterministic():
    assert run(conversation())[0] == run(conversation())[0]


def test_openai_and_gemini_results_are_trimmed():
    obody = {
        "model": "gpt-5",
        "messages": [
            {"role": "user", "content": "why do tests fail"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "run_command", "arguments": '{"cmd": "pytest"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": pytest_log()},
        ],
    }
    new_body, report = trim_request(obody, openai.parse(obody), TrimConfig())
    assert report.trimmed and "omitted]" in new_body["messages"][2]["content"]

    gbody = {
        "contents": [
            {"role": "user", "parts": [{"text": "why do tests fail"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "run_shell_command", "args": {"command": "pytest"}}}
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "run_shell_command",
                            "response": {"output": pytest_log()},
                        }
                    }
                ],
            },
        ]
    }
    new_body, report = trim_request(gbody, gemini.parse(gbody), TrimConfig())
    out = new_body["contents"][2]["parts"][0]["functionResponse"]["response"]["output"]
    assert report.trimmed and "omitted]" in out


def test_non_tool_segments_never_trimmed():
    body = conversation()
    body["messages"][0]["content"] = "x " * 5000  # a huge user message
    new_body, report = run(body)
    assert new_body["messages"][0]["content"] == body["messages"][0]["content"]
    kinds = {s.id: s.kind for s in anthropic.parse(body)}
    assert all(kinds[d["segment"]] == SegmentKind.TOOL_RESULT for d in report.trimmed)


# ---------- gateway modes ----------


class Upstream:
    def __init__(self):
        self.bodies: list[bytes] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(request.content)
        return reply_json(200, {"usage": {"input_tokens": 1, "output_tokens": 1}})


def client_for(mode: str, tmp_path):
    upstream = Upstream()
    app = create_app(
        GatewayConfig(log_dir=tmp_path, mode=mode), transport=httpx2.MockTransport(upstream.handler)
    )
    return app, upstream


def test_trim_mode_sends_trimmed_body(tmp_path):
    app, upstream = client_for("trim", tmp_path)
    with TestClient(app) as client:
        assert client.post("/anthropic/v1/messages", json=conversation()).status_code == 200
        assert client.get("/tokencut/health").json()["mode"] == "trim"
        stats = client.get("/tokencut/stats").json()
    sent = json.loads(upstream.bodies[0])
    assert "[TokenCut: lines" in sent["messages"][2]["content"][0]["content"]
    assert app.state.request_log.recent[-1]["trim_applied"] is True
    assert stats["trimming"]["tokens_saved_applied"] > 0


def test_shadow_mode_reports_savings_but_sends_original(tmp_path):
    app, upstream = client_for("shadow", tmp_path)
    raw = json.dumps(conversation()).encode()
    with TestClient(app) as client:
        client.post(
            "/anthropic/v1/messages", content=raw, headers={"content-type": "application/json"}
        )
        stats = client.get("/tokencut/stats").json()
    assert upstream.bodies[0] == raw
    assert stats["trimming"]["tokens_saved_possible"] > 0
    assert stats["trimming"]["tokens_saved_applied"] == 0


def test_trim_failure_forwards_original(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("tokencut.gateway.records.trim_request", broken)
    monkeypatch.setattr("tokencut.gateway.records.trim_with_cache_events", broken)
    app, upstream = client_for("trim", tmp_path)
    raw = json.dumps(conversation()).encode()
    with TestClient(app) as client:
        r = client.post(
            "/anthropic/v1/messages", content=raw, headers={"content-type": "application/json"}
        )
    assert r.status_code == 200 and upstream.bodies[0] == raw
    assert "analysis_error" in app.state.request_log.recent[-1]


def test_mode_from_env(monkeypatch):
    monkeypatch.setenv("TOKENCUT_MODE", "trim")
    monkeypatch.setenv("TOKENCUT_TIER", "aggressive")
    monkeypatch.setenv("TOKENCUT_MIN_TRIM_TOKENS", "500")
    config = GatewayConfig.from_env()
    assert (config.mode, config.trim.tier, config.trim.min_tokens) == ("trim", Tier.AGGRESSIVE, 500)
    assert trimmer.TrimConfig().tier == Tier.MODERATE


# ---------- R8: diffs and output asked for in full stay whole ----------


def git_diff(n_files: int = 12) -> str:
    parts = []
    for i in range(n_files):
        parts += [f"diff --git a/pkg/mod_{i}.py b/pkg/mod_{i}.py",
                  "index 1234abc..5678def 100644",
                  f"--- a/pkg/mod_{i}.py",
                  f"+++ b/pkg/mod_{i}.py",
                  f"@@ -{10 + i},7 +{10 + i},8 @@ def helper_{i}(value):"]
        parts += [f"     value = value * {j} + {i}" for j in range(6)]
        parts += [f"-    return value + {i}", f"+    return value - {i}", "+    # fixed sign"]
    return "\n".join(parts)


def bash_conversation(user_text: str, command: str, output: str) -> dict:
    return {
        "model": "claude-haiku-4-5",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": command}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "b1", "content": output}]},
        ],
    }


def test_git_diff_output_is_never_trimmed():
    diff = git_diff()
    assert count_tokens(diff) > 600
    body = bash_conversation("review my changes", "git diff", diff)
    new_body, report = run(body, TrimConfig(min_tokens=300))
    assert new_body is None and report.trimmed == []


def test_output_the_user_asked_for_in_full_is_not_trimmed():
    body = bash_conversation("show me the full test output", "pytest -q", pytest_log())
    new_body, report = run(body, TrimConfig(min_tokens=300))
    assert new_body is None and report.trimmed == []


def test_whole_test_suite_request_is_still_trimmed():
    # "whole" refers to the suite, not to the output: the log is trimmed as before
    body = bash_conversation("run the whole core test suite and group the failures", "pytest -q", pytest_log())
    _, report = run(body, TrimConfig(min_tokens=300))
    assert [d["how"] for d in report.trimmed] == ["command_output"]
