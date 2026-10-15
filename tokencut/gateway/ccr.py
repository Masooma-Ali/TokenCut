"""Reversible trimming (CCR): every shortened tool output can be fetched back exactly.

A trimmed or masked output gets one extra line naming a reference, e.g. `tc_1a2b3c4d5e6f`.
The reference is a hash of the ORIGINAL text, so the same output gets the same reference on
every turn and after a gateway restart (the cached prefix stays byte-for-byte the same).

The original is almost always still in the request: agents re-send the full history every
turn and only TokenCut's outbound copy is shortened. The store below is a fallback for
originals that left the request (the agent compacted its own history).
"""

import hashlib
import threading
from collections import OrderedDict

from tokencut.gateway.codeview import parse_listing

REF_PREFIX = "tc_"
RETRIEVE_TOOL = "tokencut_retrieve"
MAX_STORE_BYTES = 64_000_000  # originals kept in RAM (least recently used dropped first)
MAX_RETRIEVE_LINES = 2_000  # one retrieval returns at most this many lines


def ref_for(text: str) -> str:
    return REF_PREFIX + hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def with_ref(shortened: str, original: str) -> str:
    """The shortened text plus the line telling the model how to get the original back."""
    ref = ref_for(original)
    return (
        f"{shortened}\n[TokenCut ref {ref}: this output was shortened; "
        f'{RETRIEVE_TOOL}(ref="{ref}", start_line, end_line) returns omitted lines exactly]'
    )


def slice_lines(original: str, start: int | None, end: int | None) -> tuple[str, int, int]:
    """Lines start..end (inclusive) of the original, numbered as the model saw them: by the
    tool's own line numbers for a numbered file listing, else by position. Returns
    (text, first, last) with the range actually returned."""
    lines = original.split("\n")
    view = parse_listing(original)
    numbered = any(view.prefixes)
    first_number = view.first_line if numbered else 1
    last_number = first_number + len(lines) - 1
    lo = max(first_number, start or first_number)
    hi = min(last_number, end or last_number, lo + MAX_RETRIEVE_LINES - 1)
    if lo > hi:
        return "", lo, hi
    return "\n".join(lines[lo - first_number: hi - first_number + 1]), lo, hi


class CCRStore:
    """Originals by reference, bounded by total size."""

    def __init__(self, max_bytes: int = MAX_STORE_BYTES):
        self._items: OrderedDict[str, str] = OrderedDict()
        self._bytes = 0
        self._max = max_bytes
        self._lock = threading.Lock()  # trimming runs in worker threads

    def put(self, original: str) -> str:
        ref = ref_for(original)
        with self._lock:
            if ref in self._items:
                self._items.move_to_end(ref)
                return ref
            self._items[ref] = original
            self._bytes += len(original)
            while self._bytes > self._max and len(self._items) > 1:
                _, dropped = self._items.popitem(last=False)
                self._bytes -= len(dropped)
        return ref

    def get(self, ref: str) -> str | None:
        with self._lock:
            text = self._items.get(ref)
            if text is not None:
                self._items.move_to_end(ref)
            return text

    def __len__(self) -> int:
        return len(self._items)
