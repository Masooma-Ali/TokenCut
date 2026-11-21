"""Task briefs: a short, task-specific note appended to a typed user message (prompt enhancement).

For every typed user message in the request, oldest first: rules and the router
(task_router.py) give a task type; a brief is added only when the type is a real task, the
router is confident (p >= threshold, chosen on dev: RESULTS_REPORT §11) and the type differs from
the last brief given earlier in the same request (no repeats inside a task). The decision depends
only on the request, so the same messages get the same briefs on every turn: earlier messages
stay byte-identical (prompt cache) and nothing is remembered between requests.
"""

from tokencut.contracts import Segment, SegmentKind
from tokencut.gateway.note import _is_agent_request
from tokencut.gateway.task_router import NO_BRIEF, prepare
from tokencut.tokens import count_tokens

TAG = "<tokencut_task_brief"
BRIEFS = {
    "bugfix": ("Reproduce the problem first (a failing test or a command that shows it). Find the "
               "root cause before editing; make the smallest fix; do not change existing tests to "
               "make them pass. Run the failing test and the related tests and report the result."),
    "feature": ("Follow how similar code in this repository is structured. Implement the smallest "
                "version that meets the request, add or update tests for the new behaviour, run "
                "them, and report what changed."),
    "refactor": ("Behaviour must stay the same. Run the relevant tests before and after; change "
                 "structure only, no unrelated edits; keep public interfaces unless asked."),
    "tests": ("Change tests only unless a code bug is confirmed. A new test must fail without the "
              "fix and pass with it; never weaken existing assertions. Run the tests and report."),
    "explain": ("Do not edit files. Read only what is needed (line ranges, not whole files); "
                "answer with file paths and line numbers; keep it short."),
    "run_action": ("Run what was asked and keep output short (e.g. -q); do not change code unless "
                   "asked. For git, check the status first; never force-push or rewrite history "
                   "unless asked."),
}


def brief_text(label: str) -> str:
    return f'{TAG} type="{label}">\n{BRIEFS[label]}\n</tokencut_task_brief>'


def _typed_messages(segments: list[Segment]) -> dict[int, list[Segment]]:
    """User text segments grouped by message, in request order (tool results are not text)."""
    out: dict[int, list[Segment]] = {}
    for seg in segments:
        if seg.kind == SegmentKind.USER_TEXT and seg.message_index >= 0:
            out.setdefault(seg.message_index, []).append(seg)
    return dict(sorted(out.items()))


def _insert(provider: str, body: dict, path: list, text: str) -> None:
    """Append the brief to the message at path[:2] (a new block / part, or after a string)."""
    message = body[path[0]][path[1]]
    if provider == "gemini":
        parts = message["parts"]
        last = max(k for k, p in enumerate(parts) if isinstance(p, dict) and "text" in p)
        parts.insert(last + 1, {"text": text})
        return
    content = message["content"]
    if isinstance(content, str):
        message["content"] = f"{content}\n\n{text}"
        return
    last = max(k for k, b in enumerate(content) if isinstance(b, dict) and b.get("type") == "text")
    content.insert(last + 1, {"type": "text", "text": text})


def add_task_briefs(provider: str, body: dict, segments: list[Segment], router,
                    threshold: float) -> tuple[list[dict], dict]:
    """Add briefs to `body` in place. Returns (report entries, task of the newest typed message).
    Side requests (titles, summaries: fewer than two client tools) get nothing. All messages are
    classified before the body is changed, so a router error leaves it untouched."""
    entries: list[dict] = []
    task: dict = {}
    if not _is_agent_request(body):
        return entries, task
    inserts: list[tuple[list, str]] = []
    last_brief: str | None = None
    for index, group in _typed_messages(segments).items():
        joined = "\n".join(seg.text for seg in group)
        if TAG in joined:  # already briefed (never twice)
            continue
        text, reason = prepare(joined)
        label, prob = (router.classify(text) if text is not None else (None, None))
        if text is None:
            pass
        elif label == NO_BRIEF:
            reason = "no_brief"
        elif prob < threshold:
            reason = "below_threshold"
        elif label == last_brief:
            reason = "repeat"
        else:
            reason = "brief"
            brief = brief_text(label)
            inserts.append((group[-1].path, brief))
            entries.append({"segment": f"m{index}", "how": "task_brief", "tool": label,
                            "before": 0, "after": count_tokens(brief)})
            last_brief = label
        task = {"label": label, "prob": round(prob, 3) if prob is not None else None,
                "briefed": reason == "brief", "reason": reason}
    for path, brief in inserts:
        _insert(provider, body, path, brief)
    return entries, task
