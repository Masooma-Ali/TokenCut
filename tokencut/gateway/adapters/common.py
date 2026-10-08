"""Helpers shared by the provider adapters."""

import json
from typing import Any

from tokencut.contracts import Segment, SegmentKind, ToolResultKind
from tokencut.gateway.classify import classify_tool_result
from tokencut.tokens import count_tokens

JsonPath = list[str | int]


def as_json(value: Any) -> str:
    """Stable text form of a JSON value, for token counting."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def get(obj: dict, camel: str, snake: str, default: Any = None) -> Any:
    """Gemini's REST API accepts both camelCase and snake_case field names."""
    if camel in obj:
        return obj[camel]
    return obj.get(snake, default)


def key_name(obj: dict, camel: str, snake: str) -> str:
    return camel if camel in obj else snake


class SegmentBuilder:
    """Collects segments with unique ids and token counts."""

    def __init__(self) -> None:
        self.segments: list[Segment] = []

    def add(
        self,
        seg_id: str,
        kind: SegmentKind,
        role: str,
        message_index: int,
        path: JsonPath,
        text: str,
        tool_name: str = "",
        tool_call_id: str = "",
        result_kind: ToolResultKind | None = None,
    ) -> None:
        self.segments.append(
            Segment(
                id=seg_id,
                kind=kind,
                role=role,
                message_index=message_index,
                path=list(path),
                text=text,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                result_kind=result_kind,
                token_count=count_tokens(text),
            )
        )

    def add_tool_result(
        self,
        seg_id: str,
        role: str,
        message_index: int,
        path: JsonPath,
        text: str,
        tool_name: str,
        tool_call_id: str,
    ) -> None:
        self.add(
            seg_id,
            SegmentKind.TOOL_RESULT,
            role,
            message_index,
            path,
            text,
            tool_name,
            tool_call_id,
            classify_tool_result(tool_name, text),
        )
