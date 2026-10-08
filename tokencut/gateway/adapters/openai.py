"""OpenAI Chat Completions API (POST /v1/chat/completions).

Also the format of most "OpenAI-compatible" endpoints: OpenRouter, Cline, Cursor custom
models, local servers.
"""

from tokencut.contracts import Segment, SegmentKind
from tokencut.gateway.adapters.common import SegmentBuilder, as_json

PROVIDER = "openai"
_SYSTEM_ROLES = ("system", "developer")


def model_of(body: dict, path: str) -> str:
    return str(body.get("model", ""))


def parse(body: dict) -> list[Segment]:
    b = SegmentBuilder()
    tools = body.get("tools") or body.get("functions")
    if tools:
        b.add(
            "tools",
            SegmentKind.TOOLS,
            "system",
            -1,
            ["tools" if "tools" in body else "functions"],
            as_json(tools),
        )

    tool_names: dict[str, str] = {}  # tool_call id -> function name
    for i, message in enumerate(body.get("messages", [])):
        role = str(message.get("role", ""))
        if role in _SYSTEM_ROLES:
            kind = SegmentKind.SYSTEM
        elif role == "assistant":
            kind = SegmentKind.ASSISTANT_TEXT
        else:
            kind = SegmentKind.USER_TEXT
        is_result = role in ("tool", "function")
        call_id = str(message.get("tool_call_id", ""))
        name = tool_names.get(call_id, str(message.get("name", "")))

        content = message.get("content")
        parts: list[tuple[str, list[str | int], str]] = []  # (id, path, text)
        if isinstance(content, str):
            parts.append((f"m{i}", ["messages", i, "content"], content))
        elif isinstance(content, list):
            for k, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(
                        (
                            f"m{i}.p{k}",
                            ["messages", i, "content", k, "text"],
                            str(part.get("text", "")),
                        )
                    )
                else:  # image_url, input_audio, file
                    b.add(
                        f"m{i}.p{k}", SegmentKind.MEDIA, role, i, ["messages", i, "content", k], ""
                    )
        for seg_id, path, text in parts:
            if is_result:
                b.add_tool_result(seg_id, role, i, path, text, name, call_id)
            else:
                b.add(seg_id, kind, role, i, path, text)

        for j, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function", {}) if isinstance(call, dict) else {}
            call_name = str(function.get("name", ""))
            tool_names[str(call.get("id", ""))] = call_name
            b.add(
                f"m{i}.t{j}",
                SegmentKind.TOOL_CALL,
                role,
                i,
                ["messages", i, "tool_calls", j, "function", "arguments"],
                str(function.get("arguments", "")),
                call_name,
                str(call.get("id", "")),
            )
    return b.segments
