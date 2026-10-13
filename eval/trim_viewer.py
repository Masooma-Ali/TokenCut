"""Before / after view of everything TokenCut changed in a captured session.

Replays the captured requests through the gateway code (same settings as the live run,
same order and timestamps), so the "after" text is exactly what the model received.
The page contains prompts and code, so it is written next to the captures (gitignored).

    python -m eval.trim_viewer logs/gateway/L3 --task-mask
"""

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path

from tokencut.contracts import SegmentKind, Tier
from tokencut.gateway.adapters import anthropic
from tokencut.gateway.adapters.common import lookup
from tokencut.gateway.history import CacheEventConfig
from tokencut.gateway.records import analyse
from tokencut.gateway.session import SessionStore, fingerprint
from tokencut.gateway.trimmer import TrimConfig

HOW_LABELS = {
    "code": "Python file trimmed on arrival",
    "command_output": "command output trimmed on arrival",
    "search": "search output trimmed on arrival",
    "file_head_tail": "file trimmed on arrival",
    "task_masked": "masked at a new task (R10)",
    "history_old_output": "old output replaced at a cache event",
    "history_old_read": "old file read replaced at a cache event",
    "history_duplicate_read": "duplicate read replaced at a cache event",
    "history_stale_read": "stale read replaced at a cache event",
}


def _stamp(path: Path) -> float:
    return datetime.strptime(path.name.split("_")[0], "%Y%m%dT%H%M%S%f").replace(tzinfo=UTC).timestamp()


def collect(run_dir: Path, trim: TrimConfig, events: CacheEventConfig) -> list[dict]:
    """First time each tool output was changed: before / after text and context."""
    store = SessionStore()
    seen: set[str] = set()
    items = []
    for n, path in enumerate(sorted((run_dir / "captures").glob("*.json"))):
        raw = path.read_bytes()
        analysis, new = analyse("anthropic", "v1/messages", raw, trim, (store, events), now=_stamp(path))
        if new is None:
            continue
        body, sent = json.loads(raw), json.loads(new)
        segments = anthropic.parse(body)
        calls = {s.tool_call_id: (s.tool_name, s.text) for s in segments if s.kind == SegmentKind.TOOL_CALL}
        user_turns = 0
        how_by_id = {d["segment"]: d for d in analysis["trim"]["details"]}
        for seg in segments:
            if seg.kind == SegmentKind.USER_TEXT and seg.text.strip():
                user_turns += 1
            if seg.kind != SegmentKind.TOOL_RESULT or seg.id not in how_by_id:
                continue
            fp = fingerprint(seg)
            detail = how_by_id[seg.id]
            if fp in seen or detail["how"] == "history_pinned":
                continue
            seen.add(fp)
            after = lookup(sent, seg.path)
            tool, call_input = calls.get(seg.tool_call_id, (seg.tool_name, ""))
            items.append({"request": n, "how": detail["how"], "tool": tool, "call": call_input,
                          "before_tokens": detail["before"], "after_tokens": detail["after"],
                          "before": seg.text, "after": after if isinstance(after, str) else ""})
    return items


def _lines_html(text: str, kept: set[str], mark_removed: bool) -> str:
    out = []
    for line in text.split("\n"):
        cls = ""
        if mark_removed and line not in kept:
            cls = "rm"
        elif not mark_removed and line.lstrip().startswith("[TokenCut"):
            cls = "note"
        out.append(f'<span class="{cls}">{html.escape(line) or " "}</span>')
    return "\n".join(out)


def render(items: list[dict], title: str) -> str:
    cards = []
    for i, it in enumerate(items, 1):
        kept = set(it["after"].split("\n"))
        saved = it["before_tokens"] - it["after_tokens"]
        cards.append(f"""
<details class="card" {"open" if i == 1 else ""}>
<summary><b>#{i}</b> request {it['request']} · <span class="tag">{html.escape(HOW_LABELS.get(it['how'], it['how']))}</span>
 · {html.escape(it['tool'])} · {it['before_tokens']:,} → {it['after_tokens']:,} tokens (−{saved:,})</summary>
<div class="call">{html.escape(it['call'][:400])}</div>
<div class="cols">
 <div><h4>Before (what the agent produced)</h4><pre>{_lines_html(it['before'], kept, True)}</pre></div>
 <div><h4>After (what the model received)</h4><pre>{_lines_html(it['after'], set(), False)}</pre></div>
</div>
</details>""")
    total_before = sum(it["before_tokens"] for it in items)
    total_after = sum(it["after_tokens"] for it in items)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#fafafa; --fg:#1d1d1f; --card:#fff; --line:#ddd; --rm:#fde2e1; --note:#e3efff; --muted:#666; --tag:#efe7ff; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16171a; --fg:#e8e8ea; --card:#1f2024; --line:#333; --rm:#4a2323; --note:#1f3350; --muted:#aaa; --tag:#3a2f55; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:0; padding:16px; }}
h1 {{ font-size:20px; margin:0 0 4px; }} .sub {{ color:var(--muted); margin-bottom:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; margin:10px 0; padding:8px 12px; }}
summary {{ cursor:pointer; }} .tag {{ background:var(--tag); border-radius:4px; padding:1px 6px; }}
.call {{ color:var(--muted); font:12px ui-monospace, monospace; margin:6px 0; word-break:break-all; }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
@media (max-width: 800px) {{ .cols {{ grid-template-columns:1fr; }} }}
h4 {{ margin:6px 0; font-size:13px; }}
pre {{ font:12px/1.35 ui-monospace, monospace; max-height:520px; overflow:auto; border:1px solid var(--line);
       border-radius:6px; padding:6px; margin:0; white-space:pre; }}
pre span {{ display:block; }} .rm {{ background:var(--rm); }} .note {{ background:var(--note); font-weight:600; }}
.legend span {{ padding:1px 6px; border-radius:4px; margin-right:8px; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="sub">{len(items)} tool outputs changed · {total_before:,} → {total_after:,} tokens the first time each was sent
 (each is re-sent this way on every later request).
 <span class="legend"><span class="rm">removed line</span><span class="note">TokenCut note</span></span></div>
{''.join(cards)}
</body></html>"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--tier", default="moderate")
    parser.add_argument("--task-mask", action="store_true")
    parser.add_argument("--prune-tools", action="store_true")
    args = parser.parse_args(argv)
    trim = TrimConfig(tier=Tier(args.tier))
    events = CacheEventConfig(task_masking=args.task_mask, prune_tools=args.prune_tools)
    items = collect(args.run_dir, trim, events)
    out = args.run_dir / "trim_viewer.html"
    out.write_text(render(items, f"TokenCut before / after — {args.run_dir.name}"), encoding="utf-8")
    print(f"{len(items)} changed outputs -> {out}")


if __name__ == "__main__":
    main()
