"""Answer `tokencut_retrieve` inside the reply stream, so the agent never sees an unknown tool.

The agent does not know the gateway's own tools (virtual.py). Before, a call to
tokencut_retrieve showed up in Claude Code as "Error: No such tool available" and the gateway
replaced that error in the next request. Now, when a streamed reply ends with only such calls,
the gateway answers them itself, asks the provider to continue (same request + the call and
its answer), and streams the continuation to the agent as part of the same message. The agent
sees one normal reply; the call and its answer never reach its history.

Only when it is clean: the reply must be a Claude-format stream whose tool calls are all
tokencut_retrieve, with nothing after the first such call. Anything else (a real tool next to
it, text after it, an unknown block, a failed continuation, too many rounds) is passed on
exactly as before. Every stream without such a call is untouched, byte for byte.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field, fields

from tokencut.gateway.ccr import RETRIEVE_TOOL, CCRStore
from tokencut.gateway.usage import Usage, UsageSniffer
from tokencut.gateway.virtual import _retrieve_answer

log = logging.getLogger(__name__)

MAX_ROUNDS = 3  # continuations per reply
MAX_HELD_BYTES = 1_000_000
_BREAK = re.compile(rb"\r?\n\r?\n")

Send = Callable[[list[dict]], Awaitable["_Upstream | None"]]


class _Upstream:  # what the stream needs from an httpx2 streamed response
    headers: dict

    def aiter_raw(self) -> AsyncIterator[bytes]: ...

    async def aclose(self) -> None: ...


class CombinedUsage:
    """Billed usage over every upstream call made for one reply (the same close() as a sniffer)."""

    def __init__(self) -> None:
        self.sniffers: list[UsageSniffer] = []

    def new(self, content_type: str) -> UsageSniffer:
        sniffer = UsageSniffer("anthropic", content_type)
        self.sniffers.append(sniffer)
        return sniffer

    def close(self) -> Usage | None:
        parts = [p for p in (s.close() for s in self.sniffers) if p is not None]
        if not parts:
            return None
        total = Usage()
        for part in parts:
            for f in fields(Usage):
                setattr(total, f.name, getattr(total, f.name) + getattr(part, f.name))
        return total


def _data(event: bytes) -> dict | None:
    for line in event.split(b"\n"):
        if line.startswith(b"data:"):
            try:
                obj = json.loads(line[5:].strip())
            except (ValueError, UnicodeDecodeError):
                return None
            return obj if isinstance(obj, dict) else None
    return None


def _shifted(event: bytes, obj: dict, offset: int) -> bytes:
    """The event with its block index moved by `offset` (unchanged when offset is 0)."""
    if not offset or "index" not in obj:
        return event
    obj = {**obj, "index": obj["index"] + offset}
    lines = [b"data: " + json.dumps(obj, separators=(",", ":")).encode()
             if line.startswith(b"data:") else line for line in event.rstrip(b"\r\n").split(b"\n")]
    return b"\n".join(lines) + b"\n\n"


@dataclass
class _Round:
    """What one upstream reply contained, and what was held back."""

    offset: int  # added to this reply's block indexes so the agent sees one message
    blocks: dict[int, dict] = field(default_factory=dict)  # index -> block being rebuilt
    held: list[bytes] = field(default_factory=list)
    held_bytes: int = 0
    first_virtual: int | None = None  # index of the first tokencut_retrieve call
    real_tool: bool = False  # a tool call the agent must answer itself
    clean: bool = True  # nothing prevents answering the held calls here
    stop_reason: str | None = None

    def start_block(self, index: int, block: dict) -> None:
        kind = block.get("type")
        self.blocks[index] = {"type": kind, "name": block.get("name"), "id": block.get("id"),
                              "parts": [], "extra": block}

    def add_delta(self, index: int, delta: dict) -> None:
        block = self.blocks.get(index)
        if block is None:
            self.clean = False
            return
        kind = delta.get("type")
        if kind == "text_delta":
            block["parts"].append(("text", delta.get("text", "")))
        elif kind == "thinking_delta":
            block["parts"].append(("thinking", delta.get("thinking", "")))
        elif kind == "signature_delta":
            block["parts"].append(("signature", delta.get("signature", "")))
        elif kind == "input_json_delta":
            block["parts"].append(("json", delta.get("partial_json", "")))

    def content(self) -> list[dict] | None:
        """The reply as the assistant message to send back, or None if it cannot be rebuilt."""
        out = []
        for index in sorted(self.blocks):
            block = self.blocks[index]
            joined = {key: "".join(v for k, v in block["parts"] if k == key)
                      for key in ("text", "thinking", "signature", "json")}
            kind = block["type"]
            if kind == "text":
                out.append({"type": "text", "text": joined["text"]})
            elif kind == "thinking":
                out.append({"type": "thinking", "thinking": joined["thinking"],
                            "signature": joined["signature"]})
            elif kind == "redacted_thinking":
                out.append({"type": "redacted_thinking", "data": block["extra"].get("data", "")})
            elif kind == "tool_use":
                try:
                    args = json.loads(joined["json"] or "{}")
                except ValueError:
                    return None
                out.append({"type": "tool_use", "id": block["id"], "name": block["name"],
                            "input": args})
            else:
                return None
        return out


class InlineRetrieve:
    """Wraps one streamed reply. `send(messages)` asks the provider to continue the same
    request with these extra messages (a streamed response, or None if it failed)."""

    def __init__(self, store: CCRStore | None, send: Send, usage: CombinedUsage):
        self.store, self.send, self.usage = store, send, usage
        self.answered = 0  # calls answered by the gateway (for the request log)

    async def stream(self, upstream: _Upstream) -> AsyncIterator[bytes]:
        extra: list[dict] = []  # assistant / user message pairs added by earlier rounds
        offset = 0
        for round_no in range(MAX_ROUNDS + 1):
            state = _Round(offset)
            sniffer = self.usage.new(upstream.headers.get("content-type", ""))
            try:
                async for out in self._round(upstream, sniffer, state, round_no):
                    yield out
            finally:
                await upstream.aclose()
            follow = await self._continue(state, extra) if state.held else None
            if follow is None:
                for event in state.held:  # nothing to answer here: pass the held events on
                    yield event
                return
            upstream, offset = follow, offset + (state.first_virtual or 0)

    async def _round(self, upstream: _Upstream, sniffer: UsageSniffer, state: _Round,
                     round_no: int) -> AsyncIterator[bytes]:
        buffer = bytearray()
        async for chunk in upstream.aiter_raw():
            sniffer.feed(chunk)
            buffer.extend(chunk)
            while True:
                match = _BREAK.search(buffer)
                if match is None:
                    break
                event = bytes(buffer[:match.end()])
                del buffer[:match.end()]
                for out in self._event(event, state, round_no):
                    yield out
        if buffer:  # an unfinished last event: pass it on as it came
            for out in self._event(bytes(buffer), state, round_no):
                yield out

    def _event(self, event: bytes, state: _Round, round_no: int) -> list[bytes]:
        """The bytes to send to the agent for this event (none while it is held back)."""
        obj = _data(event)
        if obj is None:
            return [out for out in (self._emit(event, None, state),) if out]
        kind = obj.get("type")
        if kind == "message_start" and round_no > 0:
            return []  # the agent already has this message's start
        if kind == "content_block_start" and isinstance(obj.get("content_block"), dict):
            block = obj["content_block"]
            state.start_block(obj.get("index", -1), block)
            if block.get("type") == "tool_use":
                virtual = block.get("name") == RETRIEVE_TOOL
                if virtual and state.first_virtual is None and not state.real_tool:
                    state.first_virtual = obj["index"]
                elif not virtual:
                    state.real_tool = True
                    state.clean = False
            if state.first_virtual is not None and block.get("type") != "tool_use":
                state.clean = False  # text or thinking after the call: leave it to the agent
        elif kind == "content_block_delta" and isinstance(obj.get("delta"), dict):
            state.add_delta(obj.get("index", -1), obj["delta"])
        elif kind == "message_delta":
            state.stop_reason = (obj.get("delta") or {}).get("stop_reason")
        elif kind == "error":
            state.clean = False
        return [out for out in (self._emit(event, obj, state),) if out]

    @staticmethod
    def _emit(event: bytes, obj: dict | None, state: _Round) -> bytes:
        """Forward the event, or hold it once the first call to answer has started."""
        out = event if obj is None else _shifted(event, obj, state.offset)
        if state.first_virtual is None:
            return out
        state.held.append(out)
        state.held_bytes += len(out)
        if state.held_bytes > MAX_HELD_BYTES:
            state.clean = False
        return b""

    async def _continue(self, state: _Round, extra: list[dict]) -> _Upstream | None:
        """Answer the held calls and open the continuation; None means: pass the reply on."""
        if (not state.clean or state.stop_reason != "tool_use" or state.real_tool
                or len(extra) >= 2 * MAX_ROUNDS):
            return None
        content = state.content()
        if content is None:
            return None
        results = []
        for block in content:
            if block["type"] == "tool_use":
                if block["name"] != RETRIEVE_TOOL:
                    return None
                answer = _retrieve_answer(block["input"], {}, self.store)
                results.append({"type": "tool_result", "tool_use_id": block["id"],
                                "content": answer})
        if not results:
            return None
        follow = await self.send([*extra, {"role": "assistant", "content": content},
                                  {"role": "user", "content": results}])
        if follow is None:
            return None
        extra += [{"role": "assistant", "content": content}, {"role": "user", "content": results}]
        self.answered += len(results)
        state.held = []  # answered here: the agent never gets these events
        return follow
