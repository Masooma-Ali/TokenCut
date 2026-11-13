"""Gateway measurements: what each request contains, and what the provider billed.

Only numbers are logged (token counts per segment kind, usage), never prompt text or
headers, so the log is safe to keep. Full request bodies are saved only when capture is
switched on, for building parser test fixtures.
"""

import copy
import dataclasses
import json
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from tokencut.contracts import Segment, SegmentKind, ToolResultKind
from tokencut.gateway.adapters import anthropic, gemini, openai
from tokencut.gateway.adapters.common import lookup
from tokencut.gateway.history import CacheEventConfig, trim_with_cache_events
from tokencut.gateway.media import adjust_images, split_pdfs, with_media_tokens
from tokencut.gateway.session import SessionStore
from tokencut.gateway.trimmer import TrimConfig, trim_request
from tokencut.gateway.virtual import VirtualConfig, finish_tools, offers_tools, resolve

ADAPTERS = {"anthropic": anthropic, "openai": openai, "gemini": gemini}
RECENT_RECORDS = 1000


def analyse(
    provider: str,
    path: str,
    body: bytes,
    trim_config: TrimConfig | None = None,
    session: tuple[SessionStore, CacheEventConfig] | None = None,
    now: float | None = None,
    virtual: VirtualConfig | None = None,
) -> tuple[dict, bytes | None]:
    """Split a request into segments and count tokens per kind; with a TrimConfig, also
    work out the cache-safe trim and the gateway's own tools (virtual.py). Returns
    (analysis, body bytes to send or None if unchanged).
    Raises on bad input (the caller then forwards the original body)."""
    start = time.perf_counter()
    data = json.loads(body)
    adapter = ADAPTERS[provider]
    answered = []  # the gateway's answers to its own tools (replace the agent's errors)
    if trim_config is not None and virtual is not None:
        answered = resolve(provider, data, virtual.store)
    model = adapter.model_of(data, path)
    # attachments are forwarded unchanged; their estimated provider tokens are counted
    segments, attachments = with_media_tokens(provider, model, data, adapter.parse(data))
    analysis = {
        "model": model,
        "stream": bool(data.get("stream")) or "streamGenerateContent" in path,
        "segments": len(segments),
        **token_breakdown(segments),
        "media": attachments,
    }
    new_bytes = None
    if trim_config is not None:
        trim_start = time.perf_counter()
        event = None
        ccr = virtual is not None and virtual.ccr and offers_tools(provider, data)
        if ccr:
            trim_config = dataclasses.replace(trim_config, ccr=True)
        if session is not None:  # cache-event trimming (remembers decisions per conversation)
            store, events_config = session
            new_body, report, event = trim_with_cache_events(
                provider, analysis["model"], data, segments, trim_config, events_config, store,
                now=now, total_tokens=analysis["tokens_total"],
            )
        else:
            new_body, report = trim_request(data, segments, trim_config)
        # Attachments (steps 5-7). Applied last and in descending path order: an image may
        # gain a note and a PDF may become two blocks, which shifts later list positions.
        done = {d["segment"] for d in report.trimmed}  # e.g. a duplicate is already a note
        replacements, entries = adjust_images(provider, model, data, segments,
                                              trim_config.media_user, trim_config.resize_images,
                                              trim_config.trim_borders)
        if trim_config.pdf_text_pages:
            pdf_changes, pdf_entries = split_pdfs(provider, data, segments, trim_config.media_user)
            replacements, entries = replacements + pdf_changes, entries + pdf_entries
        pending = [(p, b, e) for (p, b), e in zip(replacements, entries, strict=True)
                   if e["segment"] not in done]
        for block_path, blocks, entry in sorted(pending, key=lambda c: tuple(c[0]), reverse=True):
            if new_body is None:
                new_body = copy.deepcopy(data)
            parent = lookup(new_body, block_path[:-1])
            parent[block_path[-1]:block_path[-1] + 1] = blocks
            report.tokens_before += entry["before"]
            report.tokens_after += entry["after"]
            report.trimmed.append(entry)
        if virtual is not None:
            if ccr:  # keep every original we shortened (fallback if it leaves the request)
                shortened = {d["segment"] for d in report.trimmed}
                for seg in segments:
                    if seg.id in shortened and seg.text:  # attachments have no text to keep
                        virtual.store.put(seg.text)
            out = new_body if new_body is not None else data
            extra = answered + finish_tools(provider, out, ccr, virtual.lazy_tools,
                                            virtual.core_tools)
            for entry in extra:
                report.tokens_before += entry["before"]
                report.tokens_after += entry["after"]
                report.trimmed.append(entry)
            if extra:
                new_body = out
        analysis["trim"] = {
            **report.as_dict(),
            "tier": trim_config.tier.value,
            "cache_event": event,
            "trim_ms": round((time.perf_counter() - trim_start) * 1000, 1),
        }
        if new_body is not None:
            new_bytes = json.dumps(new_body, ensure_ascii=False, separators=(",", ":")).encode()
    analysis["analysis_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return analysis, new_bytes


def token_breakdown(segments: list[Segment]) -> dict:
    by_kind = {kind.value: 0 for kind in SegmentKind}
    by_result = {kind.value: 0 for kind in ToolResultKind}
    for s in segments:
        by_kind[s.kind.value] += s.token_count
        if s.result_kind is not None:
            by_result[s.result_kind.value] += s.token_count
    return {
        "tokens_total": sum(by_kind.values()),
        "tokens_by_kind": by_kind,
        "tool_result_tokens": by_result,
    }


class RequestLog:
    def __init__(self, log_dir: Path | None, capture: bool = False):
        self.log_dir = log_dir
        self.capture = capture
        self.recent: deque[dict] = deque(maxlen=RECENT_RECORDS)

    def add(self, record: dict) -> None:
        self.recent.append(record)
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        day = record["ts"][:10]
        with (self.log_dir / f"requests-{day}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def save_capture(self, provider: str, body: bytes) -> None:
        """Keep a full request body (local only; contains the user's code and prompts)."""
        if not self.capture or self.log_dir is None:
            return
        folder = self.log_dir / "captures"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        (folder / f"{stamp}_{provider}.json").write_bytes(body)

    def summary(self) -> dict:
        """Totals over the recent requests, for GET /tokencut/stats."""
        analysed = [r for r in self.recent if r.get("analysis")]
        by_kind = {kind.value: 0 for kind in SegmentKind}
        by_result = {kind.value: 0 for kind in ToolResultKind}
        usage_total = {
            "input_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "output_tokens": 0,
        }
        providers: dict[str, int] = {}
        for r in self.recent:
            providers[r["provider"]] = providers.get(r["provider"], 0) + 1
            for field, value in (r.get("usage") or {}).items():
                usage_total[field] = usage_total.get(field, 0) + value
        for r in analysed:
            for kind, n in r["analysis"]["tokens_by_kind"].items():
                by_kind[kind] += n
            for kind, n in r["analysis"]["tool_result_tokens"].items():
                by_result[kind] += n
        attachments = {"images": 0, "documents": 0, "unknown": 0, "tokens": 0}
        for r in analysed:
            for field, value in (r["analysis"].get("media") or {}).items():
                attachments[field] = attachments.get(field, 0) + value
        total = sum(by_kind.values())
        trims = [r["analysis"]["trim"] for r in analysed if r["analysis"].get("trim")]
        applied = [r for r in analysed if r.get("trim_applied")]
        return {
            "trimming": {
                "cache_events": _count(
                    [t["cache_event"] for t in trims if t.get("cache_event")]
                ),
                "saved_by": _saved_by(trims),
                "requests_with_trims": sum(1 for t in trims if t["segments_trimmed"]),
                "tokens_saved_possible": sum(t["tokens_saved"] for t in trims),
                "tokens_saved_applied": sum(r["analysis"]["trim"]["tokens_saved"] for r in applied),
            },
            "requests": len(self.recent),
            "analysed": len(analysed),
            "analysis_errors": sum(1 for r in self.recent if r.get("analysis_error")),
            "by_provider": dict(sorted(providers.items())),
            "tokens_total": total,
            "tokens_by_kind": by_kind,
            "share_by_kind": {k: round(v / total, 3) if total else 0.0 for k, v in by_kind.items()},
            "tool_result_tokens": by_result,
            "attachments": attachments,  # estimated provider tokens, forwarded unchanged
            "billed_usage": usage_total,
            "by_route": self._by_route(),
        }

    def _by_route(self) -> dict:
        """Requests, tokens saved and billed usage per route (dual-plan demo)."""
        out: dict[str, dict] = {}
        for r in self.recent:
            route = r.get("route") or r["provider"]
            entry = out.setdefault(route, {
                "requests": 0, "tokens": 0, "tokens_saved": 0,
                "billed_usage": {"input_tokens": 0, "cache_read_tokens": 0,
                                 "cache_write_tokens": 0, "output_tokens": 0},
            })
            entry["requests"] += 1
            entry["tokens"] += _tokens(r)
            entry["tokens_saved"] += _saved(r)
            for field, value in (r.get("usage") or {}).items():
                entry["billed_usage"][field] = entry["billed_usage"].get(field, 0) + value
        return dict(sorted(out.items()))

    def recent_feed(self, limit: int = 40) -> list[dict]:
        """The latest requests, newest first, numbers only (for the live page)."""
        feed = []
        for r in list(self.recent)[-limit:][::-1]:
            trim = (r.get("analysis") or {}).get("trim") or {}
            feed.append({
                "ts": r["ts"],
                "route": r.get("route") or r["provider"],
                "model": (r.get("analysis") or {}).get("model", ""),
                "status": r.get("status"),
                "latency_ms": r.get("latency_ms"),
                "tokens": _tokens(r),
                "tokens_saved": _saved(r),
                "applied": bool(r.get("trim_applied")),
                "cache_event": trim.get("cache_event"),
                "trims": [
                    {"how": d["how"], "before": d["before"], "after": d["after"]}
                    for d in trim.get("details", [])
                ],
                "usage": r.get("usage"),
            })
        return feed


def _tokens(record: dict) -> int:
    return int((record.get("analysis") or {}).get("tokens_total", 0))


def _saved(record: dict) -> int:
    """Tokens removed: what was sent less (trim mode) or what could be (shadow mode)."""
    trim = (record.get("analysis") or {}).get("trim") or {}
    return int(trim.get("tokens_saved", 0))


def _count(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def _saved_by(trims: list[dict]) -> dict[str, int]:
    """Tokens saved per trimming method, summed over requests."""
    out: dict[str, int] = {}
    for t in trims:
        for d in t.get("details", []):
            out[d["how"]] = out.get(d["how"], 0) + d["before"] - d["after"]
    return dict(sorted(out.items()))


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")
