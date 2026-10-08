"""Read the provider's own token usage from a response while it streams past.

Works on raw bytes as they are forwarded; never changes them. Streaming responses
(text/event-stream) are scanned line by line; plain JSON bodies are parsed at the end.
Gives real billed numbers, including cached input, which the router's cost model needs.
"""

import json
from dataclasses import asdict, dataclass

MAX_BUFFER_BYTES = 10_000_000  # plain JSON bodies larger than this are not parsed


@dataclass
class Usage:
    input_tokens: int = 0  # uncached input
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _int(value: object) -> int:
    return value if isinstance(value, int) else 0


def apply_usage(provider: str, event: object, usage: Usage) -> None:
    """Update `usage` from one JSON object (an SSE event or a whole response body)."""
    if isinstance(event, list):  # Gemini non-SSE streams are a JSON array of chunks
        for item in event:
            apply_usage(provider, item, usage)
        return
    if not isinstance(event, dict):
        return
    if provider == "anthropic":
        # message_start carries input + cache numbers; message_delta the final output count
        data = (
            event.get("message", {}).get("usage") if event.get("type") == "message_start" else None
        )
        data = data or event.get("usage")
        if isinstance(data, dict):
            for field, attr in (
                ("input_tokens", "input_tokens"),
                ("cache_read_input_tokens", "cache_read_tokens"),
                ("cache_creation_input_tokens", "cache_write_tokens"),
                ("output_tokens", "output_tokens"),
            ):
                if field in data:
                    setattr(usage, attr, _int(data[field]))
    elif provider == "openai":
        data = event.get("usage")
        if isinstance(data, dict):
            cached = _int((data.get("prompt_tokens_details") or {}).get("cached_tokens"))
            usage.cache_read_tokens = cached
            usage.input_tokens = max(0, _int(data.get("prompt_tokens")) - cached)
            usage.output_tokens = _int(data.get("completion_tokens"))
    elif provider == "gemini":
        data = event.get("usageMetadata")
        if isinstance(data, dict):  # every chunk repeats the running totals
            cached = _int(data.get("cachedContentTokenCount"))
            usage.cache_read_tokens = cached
            usage.input_tokens = max(0, _int(data.get("promptTokenCount")) - cached)
            usage.output_tokens = _int(data.get("candidatesTokenCount")) + _int(
                data.get("thoughtsTokenCount")
            )


class UsageSniffer:
    def __init__(self, provider: str, content_type: str, content_encoding: str = ""):
        self.provider = provider
        self.usage = Usage()
        self.found = False
        self.streaming = "text/event-stream" in content_type
        self.enabled = content_encoding in ("", "identity")  # can't read compressed bytes
        self._buffer = bytearray()
        self._overflow = False

    def feed(self, chunk: bytes) -> None:
        if not self.enabled:
            return
        if self.streaming:
            self._buffer.extend(chunk)
            *lines, rest = bytes(self._buffer).split(b"\n")
            self._buffer = bytearray(rest)  # an incomplete last line waits for the next chunk
            for line in lines:
                self._line(line)
        elif not self._overflow:
            self._buffer.extend(chunk)
            if len(self._buffer) > MAX_BUFFER_BYTES:
                self._overflow = True
                self._buffer = bytearray()

    def _line(self, line: bytes) -> None:
        line = line.strip()
        if not line.startswith(b"data:"):
            return
        payload = line[len(b"data:") :].strip()
        if not payload or payload == b"[DONE]":
            return
        self._apply(payload)

    def _apply(self, payload: bytes) -> None:
        try:
            event = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return
        before = self.usage.as_dict()
        apply_usage(self.provider, event, self.usage)
        if self.usage.as_dict() != before:
            self.found = True

    def close(self) -> Usage | None:
        if self.enabled and self.streaming and self._buffer:
            self._line(bytes(self._buffer))
        elif self.enabled and not self.streaming and not self._overflow and self._buffer:
            self._apply(bytes(self._buffer))
        self._buffer = bytearray()
        return self.usage if self.found else None
