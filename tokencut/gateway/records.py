"""Gateway measurements: what each request contains, and what the provider billed.

Only numbers are logged (token counts per segment kind, usage), never prompt text or
headers, so the log is safe to keep. Full request bodies are saved only when capture is
switched on, for building parser test fixtures.
"""

import json
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from tokencut.contracts import Segment, SegmentKind, ToolResultKind
from tokencut.gateway.adapters import anthropic, gemini, openai
from tokencut.gateway.trimmer import TrimConfig, trim_request

ADAPTERS = {"anthropic": anthropic, "openai": openai, "gemini": gemini}
RECENT_RECORDS = 1000


def analyse(
    provider: str, path: str, body: bytes, trim_config: TrimConfig | None = None
) -> tuple[dict, bytes | None]:
    """Split a request into segments and count tokens per kind; with a TrimConfig, also
    work out the cache-safe trim. Returns (analysis, trimmed body bytes or None).
    Raises on bad input (the caller then forwards the original body)."""
    start = time.perf_counter()
    data = json.loads(body)
    adapter = ADAPTERS[provider]
    segments = adapter.parse(data)
    analysis = {
        "model": adapter.model_of(data, path),
        "stream": bool(data.get("stream")) or "streamGenerateContent" in path,
        "segments": len(segments),
        **token_breakdown(segments),
    }
    new_bytes = None
    if trim_config is not None:
        trim_start = time.perf_counter()
        new_body, report = trim_request(data, segments, trim_config)
        analysis["trim"] = {
            **report.as_dict(),
            "tier": trim_config.tier.value,
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
        total = sum(by_kind.values())
        trims = [r["analysis"]["trim"] for r in analysed if r["analysis"].get("trim")]
        applied = [r for r in analysed if r.get("trim_applied")]
        return {
            "trimming": {
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
            "billed_usage": usage_total,
        }


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")
