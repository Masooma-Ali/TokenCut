"""Trim long command output (test runs, builds) and search results.

Keeps the lines that matter most for fixing code, in this priority order, until the token
budget is used: the end of the output (test summaries, final error), lines that look like
errors with a little context, then the beginning. Every dropped run becomes one marker.
"""

import re

from tokencut.tokens import count_tokens

_ERROR = re.compile(
    r"error|exception|traceback|failed|failure|assert|fatal|panic|warning|"
    r"\bE\s{2,}|^\s*File \"|^FAIL|^ERROR",
    re.IGNORECASE,
)
TAIL_LINES = 40
HEAD_LINES = 15
ERROR_CONTEXT = 2  # lines kept before and after an error line


def _priority_order(lines: list[str]) -> list[int]:
    """Line indexes in the order we want to keep them (no duplicates)."""
    order: list[int] = []
    seen: set[int] = set()

    def push(i: int) -> None:
        if 0 <= i < len(lines) and i not in seen:
            seen.add(i)
            order.append(i)

    for i in range(len(lines) - 1, max(-1, len(lines) - 1 - TAIL_LINES), -1):
        push(i)
    for i, line in enumerate(lines):
        if _ERROR.search(line):
            for k in range(i - ERROR_CONTEXT, i + ERROR_CONTEXT + 1):
                push(k)
    for i in range(min(HEAD_LINES, len(lines))):
        push(i)
    return order


def trim_output(text: str, budget_tokens: int, what: str = "output") -> str | None:
    """Trimmed text within roughly budget_tokens, or None if no trim is needed/possible."""
    if count_tokens(text) <= budget_tokens:
        return None
    lines = text.split("\n")
    keep: set[int] = set()
    used = 0
    for i in _priority_order(lines):
        cost = count_tokens(lines[i]) + 1
        if used + cost > budget_tokens:
            continue  # a shorter line further down the priority list may still fit
        keep.add(i)
        used += cost

    out: list[str] = []
    i = 0
    while i < len(lines):
        if i in keep:
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and j not in keep:
            j += 1
        out.append(f"[TokenCut: {j - i} lines of {what} omitted]")
        i = j
    result = "\n".join(out)
    return result if count_tokens(result) < count_tokens(text) else None
