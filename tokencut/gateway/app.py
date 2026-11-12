"""TokenCut gateway (Iteration 2).

Sits between a coding agent and its model provider. Each request is split into
segments and measured, forwarded, and the response is streamed straight back while the
provider's billed usage is read from it. Two modes (TOKENCUT_MODE):
    shadow (default)  forward unchanged; report what trimming would save
    trim              send the cache-safe trimmed request (see trimmer.py); fail open

Point an agent at it:
    Claude Code:  ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic
    OpenAI SDKs:  base_url = http://127.0.0.1:8787/openai/v1
    Gemini:       base URL  http://127.0.0.1:8787/gemini
"""

import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx2
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from tokencut.contracts import Tier
from tokencut.gateway.history import CacheEventConfig
from tokencut.gateway.records import RequestLog, analyse, now_iso
from tokencut.gateway.routing import (
    RouteConfig,
    claude_litellm_headers,
    claude_via_litellm,
    gpt_headers,
    model_from_body,
    pick_route,
)
from tokencut.gateway.session import SessionStore
from tokencut.gateway.trimmer import TrimConfig
from tokencut.gateway.usage import UsageSniffer
from tokencut.gateway.virtual import VirtualConfig

log = logging.getLogger("tokencut.gateway")
_LIVE_PAGE = Path(__file__).with_name("live.html")

DEFAULT_UPSTREAMS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
    "gemini": "https://generativelanguage.googleapis.com",
}
# Requests we measure; every other path (model lists, token counting, ...) is only forwarded.
_INFERENCE_PATHS = {
    "anthropic": re.compile(r"(^|/)v1/messages$"),
    "openai": re.compile(r"(^|/)chat/completions$"),
    "gemini": re.compile(r"models/[^/:]+:(generateContent|streamGenerateContent)$"),
}
# Hop-by-hop / transport headers that must not be copied between the two connections.
# accept-encoding is dropped so the provider answers uncompressed and usage can be read.
_DROP_REQUEST_HEADERS = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "te",
    "upgrade",
    "proxy-authorization",
    "proxy-connection",
    "accept-encoding",
}
_DROP_RESPONSE_HEADERS = {"content-length", "connection", "keep-alive", "transfer-encoding"}
_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"]


@dataclass
class GatewayConfig:
    upstreams: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_UPSTREAMS))
    log_dir: Path | None = Path("logs/gateway")
    capture: bool = False  # save full request bodies (local only) for test fixtures
    mode: str = "shadow"  # "shadow": forward unchanged, report possible savings; "trim": apply
    trim: TrimConfig = field(default_factory=TrimConfig)
    cache_events: bool = True  # re-trim history when the provider cache is lost anyway
    events: CacheEventConfig = field(default_factory=CacheEventConfig)
    virtual: VirtualConfig = field(default_factory=VirtualConfig)  # CCR, lazy tool loading
    route: RouteConfig = field(default_factory=RouteConfig)  # dual-plan demo (GPT via LiteLLM)

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        upstreams = {
            name: os.environ.get(f"TOKENCUT_{name.upper()}_UPSTREAM", url).rstrip("/")
            for name, url in DEFAULT_UPSTREAMS.items()
        }
        log_dir = os.environ.get("TOKENCUT_LOG_DIR", "logs/gateway")
        mode = os.environ.get("TOKENCUT_MODE", "shadow")
        if mode not in ("shadow", "trim"):
            raise ValueError(f"TOKENCUT_MODE must be 'shadow' or 'trim', not {mode!r}")
        trim = TrimConfig(
            tier=Tier(os.environ.get("TOKENCUT_TIER", TrimConfig.tier.value)),
            min_tokens=int(os.environ.get("TOKENCUT_MIN_TRIM_TOKENS", TrimConfig.min_tokens)),
        )
        events = CacheEventConfig(
            cache_ttl_seconds=int(os.environ.get("TOKENCUT_CACHE_TTL",
                                                 CacheEventConfig.cache_ttl_seconds)),
            prune_tools=os.environ.get("TOKENCUT_PRUNE_TOOLS", "") == "1",
            task_masking=os.environ.get("TOKENCUT_TASK_MASK", "1") != "0",
            task_mask_min_tokens=int(os.environ.get("TOKENCUT_TASK_MASK_TOKENS",
                                                    CacheEventConfig.task_mask_min_tokens)),
        )
        core = os.environ.get("TOKENCUT_CORE_TOOLS")
        if core:
            events.core_tools = tuple(t.strip() for t in core.split(",") if t.strip())
        virtual = VirtualConfig(
            ccr=os.environ.get("TOKENCUT_CCR", "") == "1",
            lazy_tools=os.environ.get("TOKENCUT_LAZY_TOOLS", "") == "1",
            core_tools=events.core_tools,
        )
        return cls(
            upstreams=upstreams,
            log_dir=Path(log_dir) if log_dir else None,
            capture=os.environ.get("TOKENCUT_CAPTURE", "") == "1",
            mode=mode,
            trim=trim,
            cache_events=os.environ.get("TOKENCUT_CACHE_EVENTS", "1") != "0",
            events=events,
            virtual=virtual,
            route=RouteConfig.from_env(),
        )


def create_app(
    config: GatewayConfig | None = None, transport: httpx2.AsyncBaseTransport | None = None
) -> FastAPI:
    config = config or GatewayConfig.from_env()
    request_log = RequestLog(config.log_dir, config.capture)
    # Conversation memory for cache-event trimming (in shadow mode its decisions are never sent).
    session = (SessionStore(), config.events) if config.cache_events else None
    timeout = httpx2.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx2.AsyncClient(timeout=timeout, transport=transport)
        yield
        await app.state.client.aclose()

    app = FastAPI(title="TokenCut gateway", version="0.2.0", lifespan=lifespan)
    app.state.request_log = request_log

    @app.get("/tokencut/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "mode": config.mode,
            "tier": config.trim.tier.value,
            "cache_events": config.cache_events,
            "prune_tools": config.events.prune_tools,
            "task_masking": config.events.task_masking,
            "ccr": config.virtual.ccr,
            "lazy_tools": config.virtual.lazy_tools,
            "gpt_upstream": config.route.gpt_upstream,
        }

    @app.get("/tokencut/stats")
    async def stats() -> dict:
        return request_log.summary()

    @app.get("/tokencut/recent")
    async def recent() -> list[dict]:
        return request_log.recent_feed()

    @app.get("/tokencut/", response_class=HTMLResponse)
    async def live_page() -> str:
        return _LIVE_PAGE.read_text(encoding="utf-8")

    async def forward(provider: str, path: str, request: Request) -> Response:
        started = time.perf_counter()
        body = await request.body()
        record: dict = {
            "ts": now_iso(),
            "provider": provider,
            "method": request.method,
            "path": path,
            "analysis": None,
        }

        analysis_task = None
        send_body = body
        if request.method == "POST" and _INFERENCE_PATHS[provider].search(path):
            request_log.save_capture(provider, body)
            if config.mode == "trim":
                # Trim before sending. Any failure: send the original request (fail open).
                try:
                    analysis, trimmed = await asyncio.to_thread(
                        analyse, provider, path, body, config.trim, session, None, config.virtual
                    )
                    record["analysis"] = analysis
                    if trimmed is not None:
                        send_body = trimmed
                        record["trim_applied"] = True
                except Exception as exc:
                    log.warning("trimming failed; forwarding original", exc_info=True)
                    record["analysis_error"] = f"{type(exc).__name__}: {exc}"[:300]
            else:
                # Shadow: measure (and work out the possible trim) in a worker thread while the
                # unchanged request is already on its way, so no latency is added.
                analysis_task = asyncio.create_task(
                    asyncio.to_thread(
                        analyse, provider, path, body, config.trim, session, None, config.virtual
                    )
                )

        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in _DROP_REQUEST_HEADERS
        }
        route = pick_route(provider, model_from_body(body), config.route)
        record["route"] = route
        query = f"?{request.url.query}" if request.url.query else ""
        direct_url = f"{config.upstreams[provider]}/{path}{query}"
        direct_headers = headers
        is_inference = _INFERENCE_PATHS[provider].search(path) is not None
        if route == "gpt":
            # Same Anthropic-format body; LiteLLM translates it for the ChatGPT plan.
            url = f"{config.route.gpt_upstream}/{path}{query}"
            headers = gpt_headers(headers, config.route)
            record["via"] = "litellm"
        elif route == "claude" and is_inference and claude_via_litellm(config.route):
            # LiteLLM forwards the Claude plan login to Anthropic; direct send is the fallback.
            url = f"{config.route.gpt_upstream}/{path}{query}"
            headers = claude_litellm_headers(headers, config.route)
            record["via"] = "litellm"
        else:
            url = direct_url
            if route in ("claude", "gpt"):
                record["via"] = "direct"
        # Ask for an uncompressed reply, or usage can't be read (the client's default is gzip).
        headers["accept-encoding"] = "identity"
        client: httpx2.AsyncClient = app.state.client
        try:
            try:
                upstream = await client.send(
                    client.build_request(request.method, url, headers=headers, content=send_body),
                    stream=True,
                )
            except httpx2.RequestError:
                if not (route == "claude" and record.get("via") == "litellm"):
                    raise
                # LiteLLM down: send the Claude request straight to Anthropic (fail open).
                log.warning("LiteLLM unreachable; sending Claude request directly")
                record.update(via="direct", fallback="litellm_unreachable")
                upstream = await client.send(
                    client.build_request(
                        request.method, direct_url, headers=direct_headers, content=send_body
                    ),
                    stream=True,
                )
        except httpx2.RequestError as exc:
            record.update(status=502, error=f"upstream unreachable: {type(exc).__name__}")
            await _finish(record, analysis_task, None, started, request_log)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "type": "tokencut_upstream_error",
                        "message": f"TokenCut could not reach {provider}: {exc}",
                    }
                },
            )

        record["status"] = upstream.status_code
        sniffer = UsageSniffer(
            provider,
            upstream.headers.get("content-type", ""),
            upstream.headers.get("content-encoding", ""),
        )

        async def relay():
            try:
                async for chunk in upstream.aiter_raw():
                    sniffer.feed(chunk)
                    yield chunk
            finally:
                await upstream.aclose()
                await _finish(record, analysis_task, sniffer, started, request_log)

        response_headers = {
            k: v for k, v in upstream.headers.items() if k.lower() not in _DROP_RESPONSE_HEADERS
        }
        return StreamingResponse(
            relay(), status_code=upstream.status_code, headers=response_headers
        )

    for provider in DEFAULT_UPSTREAMS:

        async def route(path: str, request: Request, provider: str = provider) -> Response:
            return await forward(provider, path, request)

        app.add_api_route(
            f"/{provider}/{{path:path}}", route, methods=_METHODS, include_in_schema=False
        )
    return app


async def _finish(
    record: dict,
    analysis_task,
    sniffer: UsageSniffer | None,
    started: float,
    request_log: RequestLog,
) -> None:
    """Complete and store the record. Never raises: measuring must not break forwarding."""
    try:
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        if analysis_task is not None:
            try:
                record["analysis"], _ = await analysis_task
            except Exception as exc:  # bad JSON, unexpected shape: forwarded anyway
                log.warning("could not analyse %s request", record["provider"], exc_info=True)
                record["analysis_error"] = f"{type(exc).__name__}: {exc}"[:300]
        usage = sniffer.close() if sniffer is not None else None
        record["usage"] = usage.as_dict() if usage is not None else None
        request_log.add(record)
    except Exception:
        log.exception("could not record request")
