"""Per-conversation memory for cache-event trimming.

The provider caches the prompt prefix. While it is valid, earlier messages must be sent
byte-for-byte as before. It is lost anyway when the model changes, after the cache's
time-to-live, or when the conversation nears the context limit. At those "cache events"
TokenCut may re-trim the whole history; this module remembers what was decided so every
later turn sends exactly the same trimmed history until the next event.

State lives in memory (local gateway). After a restart the gateway forgets its decisions,
which costs one cache rewrite — never a broken request.
"""

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from tokencut.contracts import Segment, SegmentKind

MAX_CONVERSATIONS = 500  # least recently used conversations are forgotten first


def fingerprint(seg: Segment) -> str:
    """Identity of a segment's ORIGINAL content (stable across turns)."""
    raw = f"{seg.kind.value}\0{seg.tool_call_id}\0{seg.text}"
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()


def conversation_key(provider: str, segments: list[Segment]) -> str | None:
    """Same for every turn of one conversation, whatever model is used: system prompt +
    first user message. None if the request has no user message."""
    system = [s.text for s in segments if s.kind == SegmentKind.SYSTEM]
    first_user = next((s.text for s in segments if s.kind == SegmentKind.USER_TEXT), None)
    if first_user is None:
        return None
    raw = "\0".join([provider, *system, first_user[:4000]])
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()


@dataclass
class ConversationState:
    model: str
    last_seen: float
    pinned: dict[str, str] = field(default_factory=dict)  # fingerprint -> text sent instead
    masked_paths: set[str] = field(default_factory=set)  # files whose old reads were removed
    protected_paths: set[str] = field(default_factory=set)  # re-read after removal: keep
    kept_tools: list[str] | None = None  # tool names offered since the last event
    last_event_tokens: int | None = None  # request size at the last event
    events: int = 0


class SessionStore:
    def __init__(self, max_conversations: int = MAX_CONVERSATIONS):
        self._states: OrderedDict[str, ConversationState] = OrderedDict()
        self._max = max_conversations
        self.lock = threading.Lock()  # trimming runs in worker threads

    def get(self, key: str) -> ConversationState | None:
        state = self._states.get(key)
        if state is not None:
            self._states.move_to_end(key)
        return state

    def put(self, key: str, state: ConversationState) -> None:
        self._states[key] = state
        self._states.move_to_end(key)
        while len(self._states) > self._max:
            self._states.popitem(last=False)

    def __len__(self) -> int:
        return len(self._states)
