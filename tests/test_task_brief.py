"""Task briefs in the gateway: placement, rules, repeat rule, cache safety, settings, fail open."""

import copy
import json
from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient

from tests.test_gateway_trimming import Upstream
from tokencut.gateway.adapters import anthropic, gemini, openai
from tokencut.gateway.app import GatewayConfig, create_app
from tokencut.gateway.brief import BRIEFS, add_task_briefs, brief_text
from tokencut.gateway.records import analyse
from tokencut.gateway.trimmer import TrimConfig

ON = TrimConfig(prompt_brief=True)
TOOLS = [{"name": "Read"}, {"name": "Bash"}]


class FakeRouter:
    """Keyword router: records what it was asked, so tests can check the classified text."""

    def __init__(self, fail: bool = False):
        self.seen: list[str] = []
        self.fail = fail

    def classify(self, text: str) -> tuple[str, float]:
        if self.fail:
            raise RuntimeError("model broken")
        self.seen.append(text)
        low = text.lower()
        for word, label, prob in (("maybe", "feature", 0.5), ("fix", "bugfix", 0.95),
                                  ("explain", "explain", 0.93), ("add", "feature", 0.9),
                                  ("commit", "run_action", 0.96), ("chat", "no_brief", 0.97)):
            if word in low:
                return label, prob
        return "no_brief", 0.6


def claude(*user_texts: str) -> dict:
    """A Claude Code style conversation: typed prompts, assistant turns, a tool result."""
    messages = []
    for k, text in enumerate(user_texts):
        messages.append({"role": "user", "content": [
            {"type": "text", "text": "<system-reminder>ctx: fix everything</system-reminder>"},
            {"type": "text", "text": text}]})
        messages.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": f"t{k}", "name": "Bash", "input": {"command": "ls"}}]})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{k}", "content": "a.py b.py"}]})
    return {"model": "claude-haiku-5-5", "max_tokens": 50, "system": "You are a coding agent.",
            "tools": TOOLS, "messages": messages}


def briefed(provider: str, body: dict, router=None, threshold: float = 0.87):
    adapter = {"anthropic": anthropic, "openai": openai, "gemini": gemini}[provider]
    out = copy.deepcopy(body)
    entries, task = add_task_briefs(provider, out, adapter.parse(body), router or FakeRouter(),
                                    threshold)
    return out, entries, task


def test_brief_is_a_new_block_after_the_typed_text_and_tool_results_are_untouched():
    out, entries, task = briefed("anthropic", claude("the login test fails, fix it"))
    blocks = out["messages"][0]["content"]
    assert blocks[:2] == claude("x")["messages"][0]["content"][:1] + [
        {"type": "text", "text": "the login test fails, fix it"}]
    assert blocks[2] == {"type": "text", "text": brief_text("bugfix")}
    assert out["messages"][1:] == claude("the login test fails, fix it")["messages"][1:]
    assert [e["tool"] for e in entries] == ["bugfix"] and entries[0]["how"] == "task_brief"
    assert task == {"label": "bugfix", "prob": 0.95, "briefed": True, "reason": "brief"}


def test_reminders_are_not_classified():
    router = FakeRouter()
    briefed("anthropic", claude("please explain cache events"), router)
    assert router.seen == ["please explain cache events"]


def test_string_content_and_other_providers():
    body = {"model": "m", "tools": TOOLS, "messages": [{"role": "user", "content": "fix the crash"}]}
    out, _, _ = briefed("anthropic", body)
    assert out["messages"][0]["content"] == "fix the crash\n\n" + brief_text("bugfix")

    oa = {"model": "gpt", "tools": [{"type": "function", "function": {"name": n}} for n in ("a", "b")],
          "messages": [{"role": "developer", "content": "You are Cline."},
                       {"role": "user", "content": "explain this function"}]}
    out, _, _ = briefed("openai", oa)
    assert out["messages"][0] == oa["messages"][0]
    assert out["messages"][1]["content"] == "explain this function\n\n" + brief_text("explain")

    ge = {"contents": [{"role": "user", "parts": [{"text": "add a --verbose flag"}]}],
          "tools": [{"functionDeclarations": [{"name": "read_file"}, {"name": "shell"}]}]}
    out, _, _ = briefed("gemini", ge)
    assert out["contents"][0]["parts"] == [{"text": "add a --verbose flag"},
                                           {"text": brief_text("feature")}]


def test_machine_text_follow_ups_no_brief_and_low_confidence_get_nothing():
    for text, reason in (("<command-name>/clear</command-name>", "machine"), ("yes", "followup"),
                         ("[Request interrupted by user]", "machine"), ("let's chat", "no_brief"),
                         ("maybe add caching", "below_threshold")):
        body = claude(text)
        out, entries, task = briefed("anthropic", body)
        assert out == body and entries == [], text
        assert task["reason"] == reason and task["briefed"] is False, text


def test_same_type_is_not_repeated_but_a_new_type_is_briefed():
    out, entries, task = briefed("anthropic", claude("fix the crash", "now fix the warning",
                                                     "explain what you did", "fix the import too"))
    typed = [out["messages"][3 * k]["content"] for k in range(4)]  # the 4 typed messages
    has_brief = [any("tokencut_task_brief" in b["text"] for b in blocks) for blocks in typed]
    assert has_brief == [True, False, True, True]  # bugfix, (bugfix again), explain, bugfix
    assert [e["tool"] for e in entries] == ["bugfix", "explain", "bugfix"]
    assert task == {"label": "bugfix", "prob": 0.95, "briefed": True, "reason": "brief"}
    _, _, repeat = briefed("anthropic", claude("fix the crash", "fix the warning"))
    assert repeat["reason"] == "repeat" and repeat["briefed"] is False


def test_earlier_messages_are_byte_identical_on_every_turn():
    turn1, _, _ = briefed("anthropic", claude("fix the crash"))
    turn2, _, _ = briefed("anthropic", claude("fix the crash", "explain the fix"))
    assert json.dumps(turn2["messages"][:3]) == json.dumps(turn1["messages"][:3])


def test_brief_is_not_added_twice():
    once, _, _ = briefed("anthropic", claude("fix the crash"))
    again, entries, _ = briefed("anthropic", once)
    assert again == once and entries == []


def test_system_role_and_assistant_messages_are_never_briefed():
    body = claude("fix it")
    body["messages"].insert(0, {"role": "system", "content": "fix all bugs you see"})
    body["messages"].append({"role": "assistant", "content": "I will fix it"})
    out, entries, _ = briefed("anthropic", body)
    assert out["messages"][0] == body["messages"][0] and out["messages"][-1] == body["messages"][-1]
    assert len(entries) == 1


def test_side_requests_and_status_summaries_get_no_brief():
    side = claude("fix the crash")
    side["tools"] = []  # a title / summary request, not an agent turn
    out, entries, task = briefed("anthropic", side)
    assert out == side and entries == [] and task == {}
    status = ("Current state: working (for 1m)\nTool calls so far: Bash×9\n"
              "User's most recent ask: \"fix the crash\"\n\nAssistant message tail: ...")
    body = claude(status)
    out, entries, task = briefed("anthropic", body)
    assert out == body and task["reason"] == "machine"


def test_router_error_leaves_the_body_untouched():
    class FailsSecond(FakeRouter):
        def classify(self, text):
            if self.seen:
                raise RuntimeError("model broken")
            return super().classify(text)

    body = claude("fix the crash", "explain the fix")
    out = copy.deepcopy(body)
    with pytest.raises(RuntimeError):
        add_task_briefs("anthropic", out, anthropic.parse(body), FailsSecond(), 0.87)
    assert out == body


def run(body: dict, config: TrimConfig = ON, router=None):
    analysis, new = analyse("anthropic", "v1/messages", json.dumps(body).encode(), config,
                            router=router or FakeRouter())
    return analysis, (json.loads(new) if new else body)


def test_analyse_adds_briefs_only_when_switched_on():
    body = claude("fix the crash")
    analysis, out = run(body)
    assert out["messages"][0]["content"][-1]["text"] == brief_text("bugfix")
    assert analysis["task"]["label"] == "bugfix"
    assert any(d["how"] == "task_brief" for d in analysis["trim"]["details"])
    _, plain = run(body, TrimConfig())
    assert plain["messages"] == body["messages"]


def test_router_failure_forwards_the_request_without_briefs():
    analysis, out = run(claude("fix the crash"), router=FakeRouter(fail=True))
    assert all("tokencut_task_brief" not in json.dumps(m) for m in out["messages"])
    assert "error" in analysis["task"]


def test_trim_mode_sends_briefs_and_shadow_mode_does_not(tmp_path):
    for mode, expected in (("trim", True), ("shadow", False)):
        upstream = Upstream()
        config = GatewayConfig(log_dir=tmp_path / mode, mode=mode, trim=ON, router=FakeRouter())
        app = create_app(config, transport=httpx2.MockTransport(upstream.handler))
        with TestClient(app) as client:
            assert client.post("/anthropic/v1/messages", json=claude("fix the crash")).status_code == 200
            stats = client.get("/tokencut/stats").json()
        sent = json.loads(upstream.bodies[0])
        assert ("tokencut_task_brief" in json.dumps(sent)) is expected
        assert stats["tasks"]["by_label"] == {"bugfix": 1}


def test_settings_from_env(monkeypatch):
    for name in ("TOKENCUT_PROMPT_BRIEF", "TOKENCUT_BRIEF_THRESHOLD"):
        monkeypatch.delenv(name, raising=False)
    config = GatewayConfig.from_env()
    assert config.trim.prompt_brief is False and config.trim.brief_threshold == 0.87
    monkeypatch.setenv("TOKENCUT_PROMPT_BRIEF", "1")
    monkeypatch.setenv("TOKENCUT_BRIEF_THRESHOLD", "0.9")
    monkeypatch.setenv("TOKENCUT_ROUTER_DIR", "/models/router")
    config = GatewayConfig.from_env()
    assert config.trim.prompt_brief is True and config.trim.brief_threshold == 0.9
    assert config.router_dir == "/models/router"


def test_runtime_constants_match_the_training_copy():
    from eval.prompt_router import router_metrics as train
    from tokencut.gateway import task_router as run_time
    assert run_time.LABELS == train.LABELS and run_time.NO_BRIEF == train.NO_BRIEF
    assert (run_time.HEAD_TOKENS, run_time.TAIL_TOKENS) == (train.HEAD_TOKENS, train.TAIL_TOKENS)
    ids = list(range(900))
    assert run_time.cut_token_ids(ids, marker=[-1]) == train.cut_token_ids(ids, marker=[-1])
    assert set(BRIEFS) == set(run_time.LABELS) - {run_time.NO_BRIEF}


MODEL = Path(__file__).resolve().parents[1] / "cache" / "prompt_router" / "models" / "router"


@pytest.mark.skipif(not (MODEL / "router.onnx").exists(), reason="trained router not on this machine")
def test_real_router_recognises_obvious_prompts():
    from tokencut.gateway.task_router import OnnxRouter
    router = OnnxRouter(MODEL)
    for text, label in (("commit and push this to a new branch", "run_action"),
                        ("TypeError: cannot read property 'id' of undefined in cart.js, fix it", "bugfix"),
                        ("what does the parse_listing function do?", "explain")):
        assert router.classify(text)[0] == label, text
