"""Gateway adapters: provider request JSON -> segments (Iteration 2, phase 2)."""

from tokencut.contracts import SegmentKind, ToolResultKind
from tokencut.gateway.adapters import anthropic, gemini, openai
from tokencut.gateway.classify import classify_tool_result, looks_like_numbered_file
from tokencut.gateway.records import token_breakdown

FILE_LISTING = "     1→def total(self):\n     2→    return sum(self.items)\n     3→\n"


def lookup(body, path):
    """Follow a segment path into the request body (proves paths are writable locations)."""
    value = body
    for key in path:
        value = value[key]
    return value


def by_id(segments):
    return {s.id: s for s in segments}


# ---------- Anthropic (Claude Code shape) ----------

ANTHROPIC_BODY = {
    "model": "claude-sonnet-5-5",
    "max_tokens": 1024,
    "stream": True,
    "system": [
        {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Follow the repo conventions."},
    ],
    "tools": [{"name": "Read", "description": "Read a file", "input_schema": {"type": "object"}}],
    "messages": [
        {"role": "user", "content": "Cart.total is wrong, fix it"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Look at cart.py first.", "signature": "sig"},
                {"type": "text", "text": "Let me read the file."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "Read",
                    "input": {"file_path": "shop/cart.py"},
                },
                {"type": "tool_use", "id": "tu_2", "name": "Bash", "input": {"command": "pytest"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": FILE_LISTING},
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_2",
                    "content": [
                        {"type": "text", "text": "1 failed, 3 passed"},
                        {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
                    ],
                },
            ],
        },
    ],
}


def test_anthropic_segments():
    segments = anthropic.parse(ANTHROPIC_BODY)
    s = by_id(segments)
    assert [x.kind for x in segments if x.message_index == -1] == [
        SegmentKind.SYSTEM,
        SegmentKind.SYSTEM,
        SegmentKind.TOOLS,
    ]
    assert s["m0"].kind == SegmentKind.USER_TEXT and s["m0"].text == "Cart.total is wrong, fix it"
    assert s["m1.b0"].kind == SegmentKind.THINKING
    assert s["m1.b1"].kind == SegmentKind.ASSISTANT_TEXT
    assert s["m1.b2"].kind == SegmentKind.TOOL_CALL and s["m1.b2"].tool_name == "Read"
    read = s["m2.b0"]
    assert read.kind == SegmentKind.TOOL_RESULT
    assert (read.tool_name, read.tool_call_id) == ("Read", "tu_1")  # name found through the id
    assert read.result_kind == ToolResultKind.FILE_READ
    bash = s["m2.b1.c0"]
    assert bash.result_kind == ToolResultKind.COMMAND_OUTPUT and bash.tool_name == "Bash"
    assert s["m2.b1.c1"].kind == SegmentKind.MEDIA
    assert anthropic.model_of(ANTHROPIC_BODY, "v1/messages") == "claude-sonnet-5-5"


def test_anthropic_paths_point_at_the_text():
    for seg in anthropic.parse(ANTHROPIC_BODY):
        if seg.kind in (
            SegmentKind.USER_TEXT,
            SegmentKind.ASSISTANT_TEXT,
            SegmentKind.TOOL_RESULT,
            SegmentKind.SYSTEM,
            SegmentKind.THINKING,
        ):
            assert lookup(ANTHROPIC_BODY, seg.path) == seg.text


def test_anthropic_tolerates_odd_shapes():
    body = {
        "messages": [
            {"role": "user", "content": None},
            {"role": "user", "content": [42]},
            {"role": "assistant", "content": [{"type": "server_tool_use", "id": "x"}]},
        ]
    }
    segments = anthropic.parse(body)
    assert [s.kind for s in segments] == [SegmentKind.MEDIA]
    assert anthropic.parse({}) == []


# ---------- OpenAI chat completions ----------

OPENAI_BODY = {
    "model": "gpt-5",
    "messages": [
        {"role": "developer", "content": "You are a coding agent."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Why does this fail?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "shop/cart.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": FILE_LISTING},
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
}


def test_anthropic_system_role_messages_are_system_context():
    # Claude Code sends its environment / skills list as role "system" messages inside
    # `messages` (seen live, 2026-11-13). That is agent context, not text the user wrote,
    # like role "system" / "developer" messages in the OpenAI adapter.
    body = {"system": "You are a coding agent.", "messages": [
        {"role": "system", "content": [{"type": "text", "text": "# Environment\n- skills: ..."}]},
        {"role": "user", "content": "fix the bug"},
        {"role": "system", "content": "The following skills are available: ..."},
    ]}
    kinds = [(s.role, s.kind) for s in anthropic.parse(body)]
    assert kinds == [("system", SegmentKind.SYSTEM), ("system", SegmentKind.SYSTEM),
                     ("user", SegmentKind.USER_TEXT), ("system", SegmentKind.SYSTEM)]


def test_openai_segments():
    s = by_id(openai.parse(OPENAI_BODY))
    assert s["tools"].kind == SegmentKind.TOOLS
    assert s["m0"].kind == SegmentKind.SYSTEM  # 'developer' counts as system
    assert s["m1.p0"].text == "Why does this fail?" and s["m1.p1"].kind == SegmentKind.MEDIA
    assert s["m2.t0"].kind == SegmentKind.TOOL_CALL and s["m2.t0"].tool_name == "read_file"
    result = s["m3"]
    assert result.kind == SegmentKind.TOOL_RESULT and result.tool_name == "read_file"
    assert result.result_kind == ToolResultKind.FILE_READ
    for seg in s.values():
        if seg.kind in (SegmentKind.TOOL_RESULT, SegmentKind.USER_TEXT, SegmentKind.SYSTEM):
            assert lookup(OPENAI_BODY, seg.path) == seg.text


# ---------- Gemini ----------

GEMINI_BODY = {
    "systemInstruction": {"parts": [{"text": "You are Gemini CLI."}]},
    "tools": [{"functionDeclarations": [{"name": "run_shell_command"}]}],
    "contents": [
        {"role": "user", "parts": [{"text": "Run the tests"}]},
        {
            "role": "model",
            "parts": [
                {"text": "Planning...", "thought": True},
                {"functionCall": {"name": "run_shell_command", "args": {"command": "pytest"}}},
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "run_shell_command",
                        "response": {"output": "1 failed, 3 passed"},
                    }
                },
                {
                    "functionResponse": {
                        "name": "read_many_files",
                        "response": {"files": ["a", "b"], "count": 2, "x": 1},
                    }
                },
                {"inlineData": {"mimeType": "image/png", "data": "AAAA"}},
            ],
        },
    ],
}


def test_gemini_segments():
    s = by_id(gemini.parse(GEMINI_BODY))
    assert s["system.0"].kind == SegmentKind.SYSTEM
    assert s["m0.p0"].kind == SegmentKind.USER_TEXT
    assert s["m1.p0"].kind == SegmentKind.THINKING
    assert s["m1.p1"].kind == SegmentKind.TOOL_CALL
    shell = s["m2.p0"]
    assert shell.result_kind == ToolResultKind.COMMAND_OUTPUT
    assert shell.path[-1] == "output" and lookup(GEMINI_BODY, shell.path) == "1 failed, 3 passed"
    many = s["m2.p1"]
    assert many.result_kind == ToolResultKind.FILE_READ and many.path[-1] == "response"
    assert s["m2.p2"].kind == SegmentKind.MEDIA
    assert (
        gemini.model_of(GEMINI_BODY, "v1beta/models/gemini-3-pro:generateContent") == "gemini-3-pro"
    )


def test_gemini_snake_case():
    body = {
        "system_instruction": {"parts": [{"text": "sys"}]},
        "contents": [
            {"role": "model", "parts": [{"function_call": {"name": "ls", "args": {}}}]},
            {
                "role": "user",
                "parts": [{"function_response": {"name": "ls", "response": {"output": "a.py"}}}],
            },
        ],
    }
    s = by_id(gemini.parse(body))
    assert s["system.0"].path == ["system_instruction", "parts", 0, "text"]
    assert s["m0.p0"].kind == SegmentKind.TOOL_CALL
    assert lookup(body, s["m1.p0"].path) == "a.py"


# ---------- classification and breakdown ----------


def test_classify_by_name_words_then_content():
    assert classify_tool_result("Read", "") == ToolResultKind.FILE_READ
    assert classify_tool_result("search_file_content", "") == ToolResultKind.SEARCH
    assert classify_tool_result("run_shell_command", "") == ToolResultKind.COMMAND_OUTPUT
    assert classify_tool_result("send_notification", "") == ToolResultKind.OTHER  # not "cat"
    assert classify_tool_result("str_replace_editor", FILE_LISTING) == ToolResultKind.FILE_READ
    assert looks_like_numbered_file("12\tx = 1\n13\ty = 2")
    assert not looks_like_numbered_file("2 passed in 0.1s\nok")


def test_token_breakdown_adds_up():
    segments = anthropic.parse(ANTHROPIC_BODY)
    stats = token_breakdown(segments)
    assert stats["tokens_total"] == sum(s.token_count for s in segments)
    assert stats["tool_result_tokens"]["file_read"] > 0
    assert sum(stats["tool_result_tokens"].values()) == stats["tokens_by_kind"]["tool_result"]
