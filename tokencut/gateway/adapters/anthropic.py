"""Anthropic Messages API (POST /v1/messages): Claude Code and other Claude clients."""

from typing import Any

from tokencut.contracts import Segment, SegmentKind
from tokencut.gateway.adapters.common import SegmentBuilder, as_json

PROVIDER = "anthropic"


def model_of(body: dict, path: str) -> str:
    return str(body.get("model", ""))


def parse(body: dict) -> list[Segment]:
    b = SegmentBuilder()

    system = body.get("system")
    if isinstance(system, str):
        b.add("system", SegmentKind.SYSTEM, "system", -1, ["system"], system)
    elif isinstance(system, list):
        for j, block in enumerate(system):
            if isinstance(block, dict):
                b.add(
                    f"system.{j}",
                    SegmentKind.SYSTEM,
                    "system",
                    -1,
                    ["system", j, "text"],
                    str(block.get("text", "")),
                )

    if body.get("tools"):
        b.add("tools", SegmentKind.TOOLS, "system", -1, ["tools"], as_json(body["tools"]))

    tool_names: dict[str, str] = {}  # tool_use id -> tool name (results only carry the id)
    for i, message in enumerate(body.get("messages", [])):
        role = str(message.get("role", ""))
        # Claude Code sends its environment / skills list as role "system" messages: agent
        # context, not the user's words (as role "system" / "developer" in the OpenAI adapter).
        text_kind = {"assistant": SegmentKind.ASSISTANT_TEXT,
                     "system": SegmentKind.SYSTEM}.get(role, SegmentKind.USER_TEXT)
        content = message.get("content")
        if isinstance(content, str):
            b.add(f"m{i}", text_kind, role, i, ["messages", i, "content"], content)
            continue
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            _parse_block(b, block, role, i, j, text_kind, tool_names)
    return b.segments


def _parse_block(
    b: SegmentBuilder,
    block: Any,
    role: str,
    i: int,
    j: int,
    text_kind: SegmentKind,
    tool_names: dict[str, str],
) -> None:
    if not isinstance(block, dict):
        return
    seg_id = f"m{i}.b{j}"
    path: list[str | int] = ["messages", i, "content", j]
    block_type = block.get("type")
    if block_type == "text":
        b.add(seg_id, text_kind, role, i, path + ["text"], str(block.get("text", "")))
    elif block_type == "thinking":
        b.add(
            seg_id,
            SegmentKind.THINKING,
            role,
            i,
            path + ["thinking"],
            str(block.get("thinking", "")),
        )
    elif block_type == "redacted_thinking":
        b.add(seg_id, SegmentKind.THINKING, role, i, path, "")
    elif block_type == "tool_use":
        call_id = str(block.get("id", ""))
        name = str(block.get("name", ""))
        tool_names[call_id] = name
        b.add(
            seg_id,
            SegmentKind.TOOL_CALL,
            role,
            i,
            path + ["input"],
            as_json(block.get("input", {})),
            name,
            call_id,
        )
    elif block_type == "tool_result":
        call_id = str(block.get("tool_use_id", ""))
        name = tool_names.get(call_id, "")
        inner = block.get("content", "")
        if isinstance(inner, str):
            b.add_tool_result(seg_id, role, i, path + ["content"], inner, name, call_id)
        elif isinstance(inner, list):
            for k, sub in enumerate(inner):
                sub_id = f"{seg_id}.c{k}"
                if isinstance(sub, dict) and sub.get("type") == "text":
                    b.add_tool_result(
                        sub_id,
                        role,
                        i,
                        path + ["content", k, "text"],
                        str(sub.get("text", "")),
                        name,
                        call_id,
                    )
                else:
                    b.add(sub_id, SegmentKind.MEDIA, role, i, path + ["content", k], "")
    else:
        # image, document, server tool blocks, future block types: counted, never trimmed
        text = "" if block_type in ("image", "document") else as_json(block)
        b.add(seg_id, SegmentKind.MEDIA, role, i, path, text)
