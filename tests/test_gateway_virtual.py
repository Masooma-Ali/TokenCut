"""P1: reversible trimming (CCR) and lazy tool loading, answered by the gateway itself."""

import copy
import json
import re

from tests.test_gateway_trimming import SOURCE, conversation, numbered, pytest_log
from tokencut.gateway.app import GatewayConfig
from tokencut.gateway.ccr import RETRIEVE_TOOL, CCRStore, ref_for, slice_lines, with_ref
from tokencut.gateway.records import analyse
from tokencut.gateway.trimmer import TrimConfig
from tokencut.gateway.virtual import ENABLE_TOOL, VirtualConfig, finish_tools, resolve

NO_TOOL = "<tool_use_error>Error: No such tool available: {}</tool_use_error>"
REF = re.compile(r"\[TokenCut ref (tc_[0-9a-f]{12}):")


def send(body: dict, virtual: VirtualConfig, provider: str = "anthropic",
         config: TrimConfig | None = None) -> tuple[dict, dict]:
    """(analysis, body the gateway would send)."""
    raw = json.dumps(body).encode()
    path = "v1/messages" if provider == "anthropic" else "v1/chat/completions"
    analysis, out = analyse(provider, path, raw, config or TrimConfig(), None, None, virtual)
    return analysis, json.loads(out) if out is not None else body


def tool_names(body: dict) -> list[str]:
    return [t.get("name") or t["function"]["name"] for t in body["tools"]]


def results(body: dict) -> list:
    return [b["content"] for m in body["messages"] if isinstance(m["content"], list)
            for b in m["content"] if b.get("type") == "tool_result"]


def ask_retrieve(body: dict, data: dict, call_id: str = "x1") -> dict:
    """The model called tokencut_retrieve; Claude Code answered 'no such tool'."""
    body = copy.deepcopy(body)
    body["messages"] += [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": call_id, "name": RETRIEVE_TOOL, "input": data}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call_id, "is_error": True,
             "content": NO_TOOL.format(RETRIEVE_TOOL)}]},
    ]
    return body


# ---------- references ----------


def test_reference_depends_only_on_the_original():
    assert ref_for("abc") == ref_for("abc") != ref_for("abd")
    marked = with_ref("short", "the original")
    assert marked.startswith("short\n[TokenCut ref tc_") and RETRIEVE_TOOL in marked


def test_slice_uses_the_tools_own_line_numbers():
    listing = numbered(SOURCE, start=101)  # a Read with an offset
    text, lo, hi = slice_lines(listing, 105, 107)
    assert (lo, hi) == (105, 107)
    assert text.split("\n")[0].strip().startswith("105→")
    plain, lo, hi = slice_lines("a\nb\nc\nd", 2, 3)
    assert plain == "b\nc" and (lo, hi) == (2, 3)
    assert slice_lines("a\nb", None, None)[0] == "a\nb"


def test_store_is_bounded():
    store = CCRStore(max_bytes=10)
    first = store.put("123456")
    store.put("abcdefgh")
    assert store.get(first) is None and len(store) == 1


# ---------- CCR through the gateway ----------


def test_shortened_outputs_carry_a_reference_and_the_tool_is_offered():
    analysis, sent = send(conversation(), VirtualConfig(ccr=True))
    trimmed = [r for r in results(sent) if "[TokenCut ref" in r]
    assert len(trimmed) == 2  # the file read and the long test log
    assert RETRIEVE_TOOL in tool_names(sent)
    hows = {d["how"] for d in analysis["trim"]["details"]}
    assert "ccr_tool" in hows and analysis["trim"]["segments_trimmed"] == 2


def test_without_ccr_nothing_changes():
    _, plain = send(conversation(), VirtualConfig())
    assert not any("[TokenCut ref" in r for r in results(plain))
    assert RETRIEVE_TOOL not in tool_names(plain)


def test_retrieve_returns_the_exact_original_lines():
    virtual = VirtualConfig(ccr=True)
    body = conversation()
    _, sent = send(body, virtual)
    ref = REF.search(results(sent)[0]).group(1)
    assert ref == ref_for(numbered(SOURCE))
    asked = ask_retrieve(body, {"ref": ref, "start_line": 40, "end_line": 45})
    analysis, sent2 = send(asked, virtual)
    answer = sent2["messages"][-1]["content"][0]
    assert "is_error" not in answer
    lines = answer["content"].split("\n")
    assert lines[0] == f"[TokenCut: original of {ref}, lines 40-45]"
    assert lines[1:] == numbered(SOURCE).split("\n")[39:45]  # verbatim, prefixes included
    assert "ccr_retrieved" in {d["how"] for d in analysis["trim"]["details"]}


def test_retrieved_answer_is_stable_and_cache_safe():
    virtual = VirtualConfig(ccr=True)
    body = conversation()
    ref = ref_for(pytest_log())
    asked = ask_retrieve(body, {"ref": ref})
    _, first = send(asked, virtual)
    later = copy.deepcopy(asked)
    later["messages"] += [{"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                          {"role": "user", "content": "thanks, continue"}]
    _, second = send(later, virtual)
    assert second["messages"][: len(first["messages"])] == first["messages"]
    assert second["tools"] == first["tools"]
    retrieved = first["messages"][-1]["content"][0]["content"]
    assert "[TokenCut ref" not in retrieved and "1 failed" in retrieved  # never re-trimmed


def test_retrieve_falls_back_to_the_store():
    virtual = VirtualConfig(ccr=True)
    body = conversation()
    send(body, virtual)  # stores the originals it shortened
    ref = ref_for(numbered(SOURCE))
    compacted = conversation()
    compacted["messages"] = compacted["messages"][:1]  # the agent dropped its old history
    _, sent = send(ask_retrieve(compacted, {"ref": ref, "start_line": 1, "end_line": 2}), virtual)
    assert sent["messages"][-1]["content"][0]["content"].startswith(f"[TokenCut: original of {ref}")
    _, gone = send(ask_retrieve(compacted, {"ref": "tc_000000000000"}), VirtualConfig(ccr=True))
    assert "no longer available" in gone["messages"][-1]["content"][0]["content"]


def test_openai_format():
    body = {
        "model": "gpt-5",
        "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        "messages": [
            {"role": "user", "content": "fix compute_total"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps({"path": "billing/helpers.py"})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": numbered(SOURCE)},
        ],
    }
    virtual = VirtualConfig(ccr=True)
    _, sent = send(body, virtual, "openai")
    assert "[TokenCut ref" in sent["messages"][2]["content"]
    assert sent["tools"][-1]["function"]["name"] == RETRIEVE_TOOL
    body["messages"] += [
        {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function", "function": {
            "name": RETRIEVE_TOOL, "arguments": json.dumps({"ref": ref_for(numbered(SOURCE))})}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "unknown tool"},
    ]
    _, sent = send(body, virtual, "openai")
    assert sent["messages"][-1]["content"].startswith("[TokenCut: original of tc_")


def test_requests_without_tools_are_left_alone():
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert finish_tools("anthropic", body, ccr=True, lazy=True, core_tools=("Read",)) == []
    assert "tools" not in body
    assert resolve("gemini", {"contents": []}, CCRStore()) == []


# ---------- lazy tool loading ----------

MANY_TOOLS = [
    {"name": "Read", "description": "Read a file."},
    {"name": "Bash", "description": "Run a command."},
    {"name": "WebFetch", "description": "Fetch a web page. Long text " + "x " * 400},
    {"name": "NotebookEdit", "description": "Edit a notebook cell. " + "y " * 400,
     "cache_control": {"type": "ephemeral"}},
]


def lazy_body() -> dict:
    body = conversation()
    body["tools"] = copy.deepcopy(MANY_TOOLS)
    return body


LAZY = VirtualConfig(lazy_tools=True, core_tools=("Read", "Bash"))


def test_lazy_tools_keep_core_tools_and_list_the_rest():
    analysis, sent = send(lazy_body(), LAZY)
    assert tool_names(sent) == ["Read", "Bash", ENABLE_TOOL]
    catalog = sent["tools"][-1]["description"]
    assert "- WebFetch: Fetch a web page" in catalog and "- NotebookEdit" in catalog
    assert sent["tools"][1]["cache_control"] == {"type": "ephemeral"}  # breakpoint kept
    entry = next(d for d in analysis["trim"]["details"] if d["how"] == "tools_lazy")
    assert entry["before"] > entry["after"] > 0


def test_enabling_a_tool_brings_it_back_and_answers_the_call():
    body = lazy_body()
    body["messages"] += [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "e1", "name": ENABLE_TOOL, "input": {"names": ["WebFetch"]}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "e1", "is_error": True,
             "content": NO_TOOL.format(ENABLE_TOOL)}]},
    ]
    _, sent = send(body, LAZY)
    assert tool_names(sent) == ["Read", "Bash", "WebFetch", ENABLE_TOOL]
    assert sent["messages"][-1]["content"][0]["content"] == (
        "[TokenCut: enabled WebFetch; call them directly from now on]")


def test_a_tool_called_directly_stays_offered():
    body = lazy_body()
    body["messages"] += [{"role": "assistant", "content": [
        {"type": "tool_use", "id": "w1", "name": "NotebookEdit", "input": {}}]}]
    _, sent = send(body, LAZY)
    assert "NotebookEdit" in tool_names(sent)


def test_lazy_tool_list_is_stable_between_turns():
    _, first = send(lazy_body(), LAZY)
    later = lazy_body()
    later["messages"] += [{"role": "assistant", "content": [{"type": "text", "text": "done"}]},
                          {"role": "user", "content": "next"}]
    _, second = send(later, LAZY)
    assert first["tools"] == second["tools"]


def test_settings_from_env(monkeypatch):
    config = GatewayConfig.from_env()
    assert config.virtual.ccr is False and config.virtual.lazy_tools is False
    monkeypatch.setenv("TOKENCUT_CCR", "1")
    monkeypatch.setenv("TOKENCUT_LAZY_TOOLS", "1")
    monkeypatch.setenv("TOKENCUT_CORE_TOOLS", "Read,Bash")
    config = GatewayConfig.from_env()
    assert config.virtual.ccr and config.virtual.lazy_tools
    assert config.virtual.core_tools == ("Read", "Bash")


def test_history_notes_carry_a_reference_when_ccr_is_on():
    from tests.test_cache_events import CONFIG
    from tests.test_cache_events import conversation as long_conversation
    from tests.test_cache_events import results as res
    from tokencut.gateway.adapters import anthropic
    from tokencut.gateway.history import CacheEventConfig, trim_with_cache_events
    from tokencut.gateway.session import SessionStore

    store = SessionStore()
    config = TrimConfig(tier=CONFIG.tier, min_tokens=CONFIG.min_tokens, ccr=True)
    body = long_conversation(["bash"] * 8)
    trim_with_cache_events("anthropic", body["model"], body, anthropic.parse(body), config,
                           CacheEventConfig(), store, now=0)
    switched = long_conversation(["bash"] * 8, model="claude-haiku-5-5")
    new, _, event = trim_with_cache_events("anthropic", switched["model"], switched,
                                           anthropic.parse(switched), config,
                                           CacheEventConfig(), store, now=5)
    assert event == "model_switch"
    notes = [r for r in res(new) if r.startswith("[TokenCut: earlier output")]
    assert notes and all(REF.search(n) for n in notes)
    assert REF.search(notes[0]).group(1) == ref_for(res(switched)[0])
