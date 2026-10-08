"""Google Gemini API (POST /v1beta/models/{model}:generateContent / :streamGenerateContent).

Gemini CLI and google-genai clients. The REST API accepts camelCase and snake_case keys.
"""

import re

from tokencut.contracts import Segment, SegmentKind
from tokencut.gateway.adapters.common import SegmentBuilder, as_json, get, key_name

PROVIDER = "gemini"
_MODEL_IN_PATH = re.compile(r"models/([^/:]+):")
# Common single-string fields in functionResponse.response (Gemini CLI uses "output").
_RESULT_TEXT_FIELDS = ("output", "content", "result", "text")


def model_of(body: dict, path: str) -> str:
    match = _MODEL_IN_PATH.search(path)
    return match.group(1) if match else ""


def parse(body: dict) -> list[Segment]:
    b = SegmentBuilder()
    system_key = key_name(body, "systemInstruction", "system_instruction")
    system = body.get(system_key)
    if isinstance(system, dict):
        for j, part in enumerate(system.get("parts", [])):
            if isinstance(part, dict) and "text" in part:
                b.add(
                    f"system.{j}",
                    SegmentKind.SYSTEM,
                    "system",
                    -1,
                    [system_key, "parts", j, "text"],
                    str(part["text"]),
                )
    if body.get("tools"):
        b.add("tools", SegmentKind.TOOLS, "system", -1, ["tools"], as_json(body["tools"]))

    for i, content in enumerate(body.get("contents", [])):
        role = str(content.get("role", "user"))
        text_kind = SegmentKind.ASSISTANT_TEXT if role == "model" else SegmentKind.USER_TEXT
        for j, part in enumerate(content.get("parts", [])):
            if not isinstance(part, dict):
                continue
            seg_id = f"m{i}.p{j}"
            path: list[str | int] = ["contents", i, "parts", j]
            call = get(part, "functionCall", "function_call")
            response = get(part, "functionResponse", "function_response")
            if "text" in part:
                kind = SegmentKind.THINKING if part.get("thought") else text_kind
                b.add(seg_id, kind, role, i, path + ["text"], str(part["text"]))
            elif isinstance(call, dict):
                b.add(
                    seg_id,
                    SegmentKind.TOOL_CALL,
                    role,
                    i,
                    path + [key_name(part, "functionCall", "function_call"), "args"],
                    as_json(call.get("args", {})),
                    str(call.get("name", "")),
                    str(call.get("id", "")),
                )
            elif isinstance(response, dict):
                fr_key = key_name(part, "functionResponse", "function_response")
                name = str(response.get("name", ""))
                call_id = str(response.get("id", ""))
                payload = response.get("response", {})
                field = _text_field(payload)
                if field is not None:
                    b.add_tool_result(
                        seg_id,
                        role,
                        i,
                        path + [fr_key, "response", field],
                        payload[field],
                        name,
                        call_id,
                    )
                else:
                    b.add_tool_result(
                        seg_id,
                        role,
                        i,
                        path + [fr_key, "response"],
                        as_json(payload),
                        name,
                        call_id,
                    )
            else:  # inlineData, fileData, executableCode, ...
                b.add(seg_id, SegmentKind.MEDIA, role, i, path, "")
    return b.segments


def _text_field(payload: object) -> str | None:
    """The one string field holding the tool output, if the response has that shape."""
    if not isinstance(payload, dict):
        return None
    for field in _RESULT_TEXT_FIELDS:
        if isinstance(payload.get(field), str) and len(payload) <= 2:
            return field
    return None
