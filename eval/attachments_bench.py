"""Attachment benchmark: what TokenCut does to images, PDFs and tables, and what it costs.

    python -m eval.attachments_bench sample                      # build the 7 hand-made items
    python -m eval.attachments_datasets --per-benchmark 10       # download real benchmark items
    python -m eval.attachments_bench offline --bench all         # what TokenCut does (free)
    python -m eval.attachments_bench live --bench all --model haiku          # A (shadow) vs B (trim)
    python -m eval.attachments_bench live --bench all --model gpt-6-luna --limit 5
    python -m eval.attachments_bench switch --bench chartqa,docvqa,mmlongbench --limit 1
    python -m eval.attachments_bench report                      # rebuild every report
    python -m eval.attachments_bench serve                       # http://127.0.0.1:8790/

Items (`items.jsonl`, one per line): {"id", "file", "question", "answers" (or "answer"),
"match", "expect"}. `match`: "number" (numbers equal), "relaxed" (ChartQA: numbers within 5 %,
text contained), "any" / "contains" (one answer contained), "all" (every answer contained).

offline: every item is put into a Claude-Code-like request (the agent's Read tool returns the
file, so it is a tool attachment, the case TokenCut changes by default) and passed through the
gateway's own `records.analyse`, without calling a model. Free.

live: each item runs twice through Claude Code (headless): A = gateway in shadow mode (nothing
changed, everything measured), B = trim mode. Every run gets a fresh gateway, and the order
alternates (A first, then B first) so neither arm always profits from a warm prompt cache.
Finished runs are appended at once; a rerun skips them. A usage-limit reply stops the run.

switch: one Claude Code session per item and arm: Claude reads the file and answers, then the
same session continues on a GPT model (LiteLLM), then back on Claude. Checks that attachments
in the history survive a provider switch with TokenCut on.

GPT models go through LiteLLM (docs/TEAMMATE_SETUP.md). If TOKENCUT_GPT_UPSTREAM is not set,
the tool uses http://127.0.0.1:4000 and the local key file %USERPROFILE%/.tokencut/litellm_master_key.

Output: logs/attachments_bench/ (gitignored: it holds the attachments themselves).
"""

import argparse
import base64
import csv
import functools
import html
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import UTC, datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from tokencut.gateway import media
from tokencut.gateway.records import analyse
from tokencut.gateway.textcut import looks_like_table
from tokencut.gateway.trimmer import TrimConfig
from tokencut.tokens import count_tokens

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "logs" / "attachments_bench"
SAMPLE = OUT / "sample"
DATASETS = OUT / "datasets"
REAL = OUT / "real"  # results on the real benchmark items, one folder per benchmark
# Same folder for every live run (identical Claude Code system prompts), outside the repo so
# Claude Code does not load this project's CLAUDE.md into the benchmark requests.
WORK = Path(os.environ.get("TOKENCUT_BENCH_WORK", ROOT.parent / "tokencut_bench_work"))
PORT = 8787
SERVE_PORT = 8790
CLAUDE = shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude.exe")
ARMS = {"A": {"TOKENCUT_MODE": "shadow"}, "B": {"TOKENCUT_MODE": "trim"}}
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".webp": "image/webp", ".gif": "image/gif"}
TIMEOUT_S = 420
CACHE_READ, CACHE_WRITE = 0.1, 1.25  # Anthropic price ratios for the input-equivalent
MIN_TRIM_TOKENS = 2000
LITELLM = "http://127.0.0.1:4000"
LITELLM_KEY_FILE = Path.home() / ".tokencut" / "litellm_master_key"
BENCHMARKS = ["chartqa", "docvqa", "infovqa", "mmlongbench", "wtq"]
BENCH_INFO = {
    "sample": ("Sample set (7 hand-made items)",
               "One item per attachment rule, each with a known answer."),
    "chartqa": ("ChartQA: charts",
                ("Real chart images from ChartQA (test split; Masry et al., ACL Findings 2022). Checks "
                "the border cut and the resize: does the model still read the chart right? Scored "
                "like ChartQA (relaxed accuracy: numbers within 5 %).")),
    "docvqa": ("DocVQA: scanned documents",
               ("Real scanned pages (letters, forms, reports) from DocVQA (validation split; Mathew "
               "et al., WACV 2021). Checks the resize / border cut of document scans: are small "
               "printed words still readable? Correct = one of the official answers appears in the "
               "reply.")),
    "infovqa": ("InfographicVQA: dense infographics",
                ("Real infographics (validation split; Mathew et al., WACV 2022): tall images packed "
                "with small text, the hardest case for shrinking. Correct = one of the official "
                "answers appears in the reply.")),
    "mmlongbench": ("MMLongBench-Doc: long PDFs",
                    ("Real 17-85 page PDFs (reports, slides, papers) from MMLongBench-Doc (Ma et al., "
                    "NeurIPS 2024 D&B). Checks the PDF rule: pages with only text are sent as text, "
                    "pages with pictures, charts or tables stay PDF; the whole PDF is kept when the "
                    "question is about a picture; PDFs over 50 pages are left alone. Scoring is "
                    "strict text matching (the paper uses an LLM judge), so compare A with B "
                    "rather than reading the absolute score.")),
    "wtq": ("WikiTableQuestions: large tables",
            ("Large Wikipedia tables (test split; Pasupat & Liang, ACL 2015) saved as CSV files. "
            "Checks that TokenCut never cuts a table with head / tail (the answer can be in any "
            "row). Expected: TokenCut leaves them alone, answers stay right.")),
}


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


# ---------- sample items (one per attachment step, each with a known answer) ----------


def _screenshot(path: Path, size: tuple[int, int], code: str, build: str) -> None:
    w, h = size
    img = Image.new("RGB", size, (246, 247, 249))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, w, 48), fill=(38, 42, 52))
    draw.text((20, 10), "Inventory Service - Admin Console", fill=(240, 240, 240), font=_font(24))
    for i in range(14):
        draw.text((40, 90 + i * 40), f"12:{10 + i:02d}:03 INFO worker-{i} processed batch {300 + i}",
                  fill=(60, 60, 60), font=_font(20))
    draw.rectangle((40, h // 2 + 120, w - 40, h // 2 + 200), fill=(253, 236, 236), outline=(200, 40, 40))
    draw.text((60, h // 2 + 140), f"Error code: {code} - database connection refused",
              fill=(170, 20, 20), font=_font(34))
    draw.text((w - 160, h - 28), f"build {build}", fill=(120, 120, 120), font=_font(14))
    img.save(path, "PNG")


def _chart(path: Path) -> None:
    img = Image.new("RGB", (1600, 1000), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 200, 150, 1400, 850
    draw.rectangle((left, top, right, bottom), outline=(90, 90, 90), width=2)
    draw.text((left + 20, top + 15), "Units sold by region, Q3", fill=(30, 30, 30), font=_font(30))
    for n, (name, value) in enumerate([("North", 320), ("South", 410), ("East", 560), ("West", 270)]):
        x = left + 120 + n * 260
        draw.rectangle((x, bottom - 60 - value, x + 140, bottom - 60), fill=(60, 110, 190))
        draw.text((x + 30, bottom - 50), name, fill=(30, 30, 30), font=_font(26))
    img.save(path, "PNG")


def _pdf_page(writer: PdfWriter, lines: list[str], extra_ops: list[str] | None = None) -> None:
    page = writer.add_blank_page(612, 792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject(
        {NameObject("/F1"): writer._add_object(font)})})
    ops = ["BT /F1 10 Tf 60 740 Td 14 TL"] + [f"({line}) Tj T*" for line in lines] + ["ET"]
    ops += extra_ops or []
    stream = DecodedStreamObject()
    stream.set_data("\n".join(ops).encode())
    page[NameObject("/Contents")] = writer._add_object(stream)


def _report_pdf(path: Path) -> None:
    filler = [f"Section text line {i}: the supplier and the customer agree to the terms below."
              for i in range(22)]
    writer = PdfWriter()
    _pdf_page(writer, ["Supply agreement, page 1: introduction."] + filler)
    _pdf_page(writer, ["Page 2: commercial terms."] + filler[:10] +
              ["Payment terms: net 45 days from the invoice date."] + filler[10:])
    bars = []
    for n, height in enumerate([180, 260, 140, 220]):  # a vector bar chart with grid lines
        bars.append(f"{120 + n * 100} 200 60 {height} re f")
    grid = [f"100 {200 + i * 20} m 520 {200 + i * 20} l S" for i in range(25)]
    _pdf_page(writer, ["Page 3: deliveries per quarter (chart)."], bars + grid)
    table = []
    for row in range(10):
        for col, value in enumerate([f"Q{row % 4 + 1}", str(100 + row * 7), str(12.5 + row),
                                     str(3 * row)]):
            table.append(f"BT /F1 10 Tf {60 + col * 120} {640 - row * 16} Td ({value}) Tj ET")
    _pdf_page(writer, ["Page 4: delivery figures (table)."], table)
    _pdf_page(writer, ["Page 5: termination and notices."] + filler)
    with path.open("wb") as f:
        writer.write(f)


def _sales_csv(path: Path) -> int:
    rows = ["region,month,product,units"]
    answer = 0
    for i in range(600):
        units = (i * 37) % 900 + 10
        rows.append(f"region_{i % 7},2026-{i % 12 + 1:02d},product_{i},{units}")
        if i == 317:
            answer = units
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return answer


def build_sample(folder: Path = SAMPLE) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    _screenshot(folder / "screenshot_big.png", (1920, 1080), "4711", "20391")
    _screenshot(folder / "screenshot_small.png", (1024, 640), "5120", "18822")
    _chart(folder / "chart_margins.png")
    _report_pdf(folder / "agreement.pdf")
    units = _sales_csv(folder / "sales.csv")
    items = [
        {"id": "big_screenshot", "file": "screenshot_big.png", "match": "number", "answer": "4711",
         "question": "What is the error code shown? Reply with the number only.",
         "expect": "step 5: shrunk to 1568x882 on Claude 4.7+ and OpenAI patch models"},
        {"id": "zoom_question", "file": "screenshot_big.png", "match": "number", "answer": "20391",
         "question": "Zoom in on the bottom-right corner and read the small text: what is the build "
                     "number? Reply with the number only.",
         "expect": "kept at full size (zoom / small-text request)"},
        {"id": "small_screenshot", "file": "screenshot_small.png", "match": "number", "answer": "5120",
         "question": "What is the error code shown? Reply with the number only.",
         "expect": "unchanged (already small)"},
        {"id": "chart_margins", "file": "chart_margins.png", "match": "contains", "answer": "East",
         "question": "Which region has the tallest bar? Reply with the region name only.",
         "expect": "step 6: plain white border cut, note added"},
        {"id": "pdf_text_question", "file": "agreement.pdf", "match": "number", "answer": "45",
         "question": "What are the payment terms, in days? Reply with the number only.",
         "expect": "step 7: pages 1, 2, 5 as text; pages 3 (chart), 4 (table) stay PDF"},
        {"id": "pdf_chart_question", "file": "agreement.pdf", "match": "number", "answer": "4",
         "question": "Look at the chart on page 3: how many bars does it have? Reply with the number "
                     "only.",
         "expect": "whole PDF kept (question is about a chart: intent safety)"},
        {"id": "csv_middle_row", "file": "sales.csv", "match": "number", "answer": str(units),
         "question": "How many units did product_317 sell? Reply with the number only.",
         "expect": "step 2: table never cut with head / tail"},
    ]
    with (folder / "items.jsonl").open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    return folder / "items.jsonl"


def load_items(path: Path) -> list[dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for item in items:
        if "answers" not in item:
            item["answers"] = [item["answer"]]
        item.setdefault("benchmark", "sample")
    return items


def bench_items(bench: str) -> Path:
    return SAMPLE / "items.jsonl" if bench == "sample" else DATASETS / bench / "items.jsonl"


def bench_dir(bench: str) -> Path:
    return REAL / bench


# ---------- scoring ----------


def _numbers(text: str) -> list[float]:
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))]


def _as_number(text: str) -> float | None:
    try:
        return float(text.replace(",", "").strip().rstrip("%").strip())
    except ValueError:
        return None


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w.%]+", " ", text.lower().replace(",", "")).split())


def _hit(answer: str, expected: str, relaxed: bool) -> bool:
    number = _as_number(expected)
    if number is not None:  # numbers: compare as numbers ("45" must not match "1450")
        if relaxed:
            tolerance = 0.05 * abs(number)
        else:
            tolerance = 1e-6 if number.is_integer() else 0.01 * abs(number)
        return any(abs(n - number) <= tolerance + 1e-9 for n in _numbers(answer))
    expected = _norm(expected)
    return bool(expected) and expected in _norm(answer)


def score_item(answer: str, answers: list[str], match: str) -> bool:
    relaxed = match == "relaxed"
    hits = [_hit(answer, str(expected), relaxed) for expected in answers]
    return all(hits) if match == "all" else any(hits)


def score(answer: str, expected: str, match: str) -> bool:
    return score_item(answer, [expected], match)


# ---------- offline: what the gateway does, without a model ----------


def attachment_block(file: Path) -> dict | str:
    """What Claude Code's Read tool returns for the file (block, or numbered text)."""
    suffix = file.suffix.lower()
    data = base64.b64encode(file.read_bytes()).decode()
    if suffix in IMAGE_TYPES:
        return {"type": "image", "source": {"type": "base64", "media_type": IMAGE_TYPES[suffix],
                                            "data": data}}
    if suffix == ".pdf":
        return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                               "data": data}}
    lines = file.read_text(encoding="utf-8", errors="replace").split("\n")
    return "\n".join(f"{n:>6}→{line}" for n, line in enumerate(lines, 1))


def read_request(model: str, question: str, file: Path) -> dict:
    block = attachment_block(file)
    content = block if isinstance(block, str) else [block]
    return {"model": model, "max_tokens": 1024, "messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "read_1", "name": "Read",
                                           "input": {"file_path": str(file.name)}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "read_1",
                                      "content": content}]},
    ]}


def describe(entry: dict) -> str:
    how = entry["how"]
    if how in ("resized", "border_trimmed", "border_trimmed_resized"):
        size = "x".join(map(str, entry.get("from_size") or []))
        to = "x".join(map(str, entry.get("to_size") or []))
        box = entry.get("crop_box")
        cut = f"border cut, kept area {box}; " if box else ""
        return f"image {size} -> {to} ({cut}{how})"
    if how in ("pdf_text_pages",):
        return (f"PDF {entry['pages']} pages: pages {entry.get('text_pages')} as text, pages "
                f"{entry.get('visual_pages') or '-'} kept as PDF")
    if how == "duplicate_attachment":
        return "identical attachment replaced by a note"
    return how


def why_unchanged(item: dict, file: Path, model: str) -> str:
    """Plain-words reason TokenCut sent an attachment as it was."""
    question = item["question"]
    suffix = file.suffix.lower()
    if suffix == ".pdf":
        raw = file.read_bytes()
        pages = media.pdf_pages(raw) or 0
        if pages > media.PDF_MAX_PAGES:
            return f"PDF has {pages} pages (over the {media.PDF_MAX_PAGES}-page limit): left as is"
        if media._VISUAL_QUESTION.search(question):
            return "question is about a picture / chart / table: whole PDF kept (intent check)"
        kinds = media.pdf_page_kinds(raw) or []
        if kinds and all(k == "visual" for k in kinds):
            return f"all {pages} pages have pictures, charts or tables: kept as PDF"
        return "no page could go as text"
    if suffix in IMAGE_TYPES:
        size = media.image_size(file.read_bytes())
        dims = f"{size[0]}x{size[1]}" if size else "?"
        if media._KEEP_DETAIL.search(question):
            return f"image {dims}: question asks for fine detail, full size kept"
        return (f"image {dims}: no plain border, and shrinking would not save 10 % of the tokens "
                f"{model} is billed")
    text = file.read_text(encoding="utf-8", errors="replace")
    if looks_like_table(text) or suffix in (".csv", ".tsv"):
        return f"table ({count_tokens(text):,} tokens): tables are never cut"
    return f"text below the {MIN_TRIM_TOKENS:,}-token trim threshold"


def _sent_attachments(body: dict, out_dir: Path, prefix: str) -> list[str]:
    """Write what the request carries in the tool result to files; return their names."""
    names = []
    content = body["messages"][2]["content"][0]["content"]
    blocks = [content] if isinstance(content, str) else content
    for n, block in enumerate(blocks):
        if isinstance(block, str) or block.get("type") == "text":
            text = block if isinstance(block, str) else block["text"]
            name = f"{prefix}_{n}.txt"
            if not (out_dir / name).exists():
                (out_dir / name).write_text(text, encoding="utf-8")
        else:
            source = block["source"]
            ext = ".pdf" if source["media_type"] == "application/pdf" else "." + source["media_type"].split("/")[1]
            name = f"{prefix}_{n}{ext}"
            if not (out_dir / name).exists():
                (out_dir / name).write_bytes(base64.b64decode(source["data"]))
        names.append(name)
    return names


def offline(items_path: Path, models: list[str], out: Path) -> list[dict]:
    files_dir = out / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        for item in load_items(items_path):
            file = items_path.parent / item["file"]
            body = read_request(model, item["question"], file)
            raw = json.dumps(body).encode()
            started = time.perf_counter()
            analysis, new_bytes = analyse("anthropic", "v1/messages", raw, TrimConfig())
            ms = (time.perf_counter() - started) * 1000
            trim = analysis["trim"]
            details = list(trim["details"])
            short_id = item["id"][:40]
            # Before = the original file (one copy per item); after = only if TokenCut changed it.
            before_files = _sent_attachments(body, files_dir, f"{short_id}_original")
            after_files = (_sent_attachments(json.loads(new_bytes), files_dir, f"{short_id}_{model}")
                           if new_bytes else before_files)
            attachment_before = analysis["media"]["tokens"] + analysis["tokens_by_kind"]["tool_result"]
            did = "; ".join(describe(d) for d in details)
            rows.append({
                "benchmark": item["benchmark"], "model": model, "id": item["id"], "file": item["file"],
                "question": item["question"], "expect": item.get("expect", ""),
                "what_tokencut_did": did or "nothing (sent as is)",
                "why_unchanged": "" if did else why_unchanged(item, file, model),
                "attachment_tokens_before": attachment_before,
                "attachment_tokens_after": attachment_before - trim["tokens_saved"],
                "request_tokens_before": analysis["tokens_total"],
                "request_tokens_after": analysis["tokens_total"] - trim["tokens_saved"],
                "page_images_removed": sum(d.get("page_images_removed", 0) for d in details),
                "gateway_ms": round(ms, 1), "trim_ms": trim["trim_ms"],
                "before_files": before_files, "after_files": after_files,
                "details": details,
            })
    (out / "offline.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


# ---------- live: Claude Code through the gateway, A (shadow) vs B (trim) ----------


class UsageLimitError(RuntimeError):
    """The plan's usage limit (or a rate limit) was hit: stop, rerun later to resume."""


_LIMIT = re.compile(r"usage limit|rate.?limit|limit reached|hit your limit|quota|\b429\b", re.IGNORECASE)
LIVE_FIELDS = [
    "benchmark", "model", "arm", "order", "id", "file", "question", "expected", "answer", "correct",
    "status", "requests", "routes", "models_seen", "input_tokens", "cache_read_tokens",
    "cache_write_tokens", "output_tokens", "input_equivalent", "attachment_tokens_est",
    "tokens_saved_est", "page_images_removed", "what_tokencut_did", "gateway_ms", "request_ms",
    "wall_s", "claude_cost_usd", "finished",
]
SWITCH_FIELDS = [
    "benchmark", "arm", "order", "id", "turn", "model", "answer", "expected", "correct", "status",
    "requests", "routes", "via", "cache_events", "input_tokens", "cache_read_tokens",
    "cache_write_tokens", "output_tokens", "input_equivalent", "tokens_saved_est",
    "what_tokencut_did", "gateway_ms", "wall_s", "finished",
]


def _gpt_env() -> dict:
    """LiteLLM settings for GPT models: the environment's, else the local defaults."""
    if os.environ.get("TOKENCUT_GPT_UPSTREAM"):
        return {}
    if not LITELLM_KEY_FILE.exists():
        return {}
    return {"TOKENCUT_GPT_UPSTREAM": LITELLM,
            "TOKENCUT_GPT_KEY": LITELLM_KEY_FILE.read_text(encoding="utf-8").strip()}


def litellm_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{LITELLM}/health/liveliness", timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


def _gateway(arm: str, log_dir: Path) -> subprocess.Popen:
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **_gpt_env(), **ARMS[arm], "TOKENCUT_LOG_DIR": str(log_dir)}
    env.pop("TOKENCUT_CAPTURE", None)
    out = (log_dir / "gateway.out").open("a", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "tokencut.gateway"], cwd=ROOT, env=env,
                            stdout=out, stderr=subprocess.STDOUT)
    for _ in range(60):
        if proc.poll() is not None:
            raise RuntimeError(f"gateway exited (is another gateway running on port {PORT}?)")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/tokencut/health", timeout=2) as r:
                health = json.loads(r.read())
            if health["mode"] != ARMS[arm]["TOKENCUT_MODE"]:
                proc.terminate()
                raise RuntimeError(f"gateway settings wrong: {health}")
            return proc
        except OSError:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError(f"gateway did not start (is another gateway running on port {PORT}?)")


def _stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def _claude(model: str, prompt: str, transcript: Path, session: str | None = None,
            resume: bool = False, tools: str | None = None) -> dict:
    cmd = [CLAUDE, "-p", prompt, "--model", model, "--output-format", "stream-json", "--verbose",
           "--allowedTools", "Read", "--permission-mode", "auto"]
    if tools:  # e.g. "Read": only these built-in tools exist in the session
        cmd += ["--tools", tools]
    if session:
        cmd += ["--resume", session] if resume else ["--session-id", session]
    env = {**os.environ, "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{PORT}/anthropic"}
    with transcript.open("w", encoding="utf-8") as out:
        try:
            subprocess.run(cmd, cwd=WORK, env=env, stdout=out, stderr=subprocess.STDOUT,
                           timeout=TIMEOUT_S, check=False)
        except subprocess.TimeoutExpired:
            return {"subtype": "timeout"}
    result: dict = {}
    text = transcript.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result = event
    answer = str(result.get("result", "")) if result else text[-500:]
    if (not result or result.get("is_error")) and _LIMIT.search(answer):
        raise UsageLimitError(answer.strip()[:300])
    return result


def _log_rows(log_dir: Path, start: str, end: str) -> list[dict]:
    rows = []
    for path in sorted(log_dir.glob("requests-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("method") == "POST" and record.get("analysis") and start <= record["ts"] <= end:
                rows.append(record)
    return rows


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _measure(records: list[dict]) -> dict:
    usage = [r.get("usage") or {} for r in records]
    billed = {k: sum(u.get(k, 0) for u in usage)
              for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens")}
    details = [d for r in records for d in r["analysis"].get("trim", {}).get("details", [])
               if d.get("kind") in ("image", "pdf") or d["how"] == "duplicate_attachment"]
    return {
        "requests": len(records),
        "routes": "+".join(sorted({str(r.get("route") or "-") for r in records})),
        **billed,
        "input_equivalent": round(billed["input_tokens"] + CACHE_READ * billed["cache_read_tokens"]
                                  + CACHE_WRITE * billed["cache_write_tokens"]),
        "attachment_tokens_est": sum(r["analysis"].get("media", {}).get("tokens", 0) for r in records),
        "tokens_saved_est": sum(r["analysis"].get("trim", {}).get("tokens_saved", 0) for r in records),
        "page_images_removed": sum(d.get("page_images_removed", 0) for d in details),
        "what_tokencut_did": "; ".join(sorted({describe(d) for d in details})) or "nothing",
        "gateway_ms": round(sum(r["analysis"].get("analysis_ms", 0) for r in records), 1),
        "request_ms": round(sum(r.get("latency_ms") or 0 for r in records), 1),
    }


def _append(path: Path, fields: list[str], row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _prepare_work(source: Path) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    for old in WORK.iterdir():
        if old.is_file():
            old.unlink()
    shutil.copy(source, WORK / source.name)


def _prompt(item: dict) -> str:
    return (f"Read the file {item['file']} in the current folder and answer this question: "
            f"{item['question']} Give a short final answer.")


WARM_AFTER_S = 240  # the provider's prompt cache may have cooled down after this long
_last_call = {"t": 0.0}


def _warm(model: str) -> None:
    """One throwaway Claude Code call (shadow gateway, empty folder) if the prompt cache may be
    cold, so neither arm pays alone for writing Claude Code's system prompt into the cache."""
    if time.monotonic() - _last_call["t"] < WARM_AFTER_S:
        return
    log_dir = REAL / "warmup"
    proc = _gateway("A", log_dir)
    try:
        WORK.mkdir(parents=True, exist_ok=True)
        for old in WORK.iterdir():
            if old.is_file():
                old.unlink()
        _claude(model, "Reply with the word OK.", log_dir / "last_warmup.jsonl")
    finally:
        _stop(proc)
    _last_call["t"] = time.monotonic()


def live(items_path: Path, model: str, out: Path, arms: list[str], limit: int | None = None,
         tools: str | None = None) -> list[dict]:
    """Run each item once per arm; append each finished run to live.csv; skip finished runs."""
    items = load_items(items_path)[:limit]
    csv_path = out / "live.csv"
    done = {(r["model"], r["id"], r["arm"]) for r in _read_csv(csv_path)}
    rows = []
    for n, item in enumerate(items):
        order = arms if n % 2 == 0 else arms[::-1]
        for arm in order:
            if (model, item["id"], arm) in done:
                continue
            _warm(model)
            log_dir = out / f"arm_{arm}" / model
            proc = _gateway(arm, log_dir)
            try:
                _prepare_work(items_path.parent / item["file"])
                start, t0 = _now(), time.perf_counter()
                result = _claude(model, _prompt(item), log_dir / f"{item['id'][:60]}.jsonl",
                                 tools=tools)
                wall = time.perf_counter() - t0
                time.sleep(0.5)  # let the gateway write its last log line
                records = _log_rows(log_dir, start, _now())
            finally:
                _stop(proc)
            answer = str(result.get("result", ""))
            row = {
                "benchmark": item["benchmark"], "model": model, "arm": arm,
                "order": f"{order[0]} first", "id": item["id"], "file": item["file"],
                "question": item["question"], "expected": " | ".join(item["answers"]),
                "answer": " ".join(answer.split())[:300],
                "correct": score_item(answer, item["answers"], item.get("match", "any")),
                "status": result.get("subtype", "no result"),
                "models_seen": "+".join(sorted({r["analysis"].get("model", "?") for r in records})),
                **_measure(records), "wall_s": round(wall, 1),
                "claude_cost_usd": result.get("total_cost_usd"), "finished": _now(),
            }
            _last_call["t"] = time.monotonic()
            _append(csv_path, LIVE_FIELDS, row)
            rows.append(row)
            print(f"[{item['benchmark']} {model} {arm}] {item['id']}: correct={row['correct']} "
                  f"input_eq={row['input_equivalent']} did={row['what_tokencut_did'][:60]} "
                  f"wall={wall:.0f}s", flush=True)
        report(out)
    return rows


def switch(items_path: Path, out: Path, arms: list[str], claude_model: str, gpt_model: str,
           limit: int | None = None) -> list[dict]:
    """Claude reads and answers, GPT answers in the same session, Claude answers again."""
    items = load_items(items_path)[:limit]
    csv_path = out / "switch.csv"
    done = {(r["id"], r["arm"]) for r in _read_csv(csv_path) if r["turn"] == "3"}
    rows = []
    for n, item in enumerate(items):
        order = arms if n % 2 == 0 else arms[::-1]
        for arm in order:
            if (item["id"], arm) in done:
                continue
            log_dir = out / f"switch_{arm}"
            session = str(uuid.uuid4())
            again = (f"Without reading the file again, answer this question about {item['file']} "
                     f"once more: {item['question']} Give a short final answer.")
            turns = [(claude_model, _prompt(item), False), (gpt_model, again, True),
                     (claude_model, again, True)]
            _warm(claude_model)
            proc = _gateway(arm, log_dir)
            try:
                _prepare_work(items_path.parent / item["file"])
                turn_rows = []
                for t, (model, prompt, resume) in enumerate(turns, 1):
                    start, t0 = _now(), time.perf_counter()
                    result = _claude(model, prompt, log_dir / f"{item['id'][:60]}_turn{t}.jsonl",
                                     session, resume)
                    wall = time.perf_counter() - t0
                    time.sleep(0.5)
                    records = _log_rows(log_dir, start, _now())
                    answer = str(result.get("result", ""))
                    measured = _measure(records)
                    turn_rows.append({
                        "benchmark": item["benchmark"], "arm": arm, "order": f"{order[0]} first",
                        "id": item["id"], "turn": t, "model": model,
                        "answer": " ".join(answer.split())[:300],
                        "expected": " | ".join(item["answers"]),
                        "correct": score_item(answer, item["answers"], item.get("match", "any")),
                        "status": result.get("subtype", "no result"),
                        "via": "+".join(sorted({str(r.get("via") or "direct") for r in records})),
                        "cache_events": "+".join(
                            str(r["analysis"].get("trim", {}).get("cache_event"))
                            for r in records if r["analysis"].get("trim", {}).get("cache_event"))
                        or "-",
                        **measured, "wall_s": round(wall, 1), "finished": _now(),
                    })
            finally:
                _stop(proc)
            for row in turn_rows:
                _append(csv_path, SWITCH_FIELDS, row)
                print(f"[switch {arm}] {item['id']} turn {row['turn']} {row['model']} via "
                      f"{row['via']}: correct={row['correct']} events={row['cache_events']}", flush=True)
            rows += turn_rows
            report_switch(out)
    return rows


# ---------- reports ----------


STYLE = ("<style>body{font:14px system-ui,sans-serif;margin:24px auto;max-width:1400px;padding:0 16px;"
         "color:#1d1d1f;background:#fff}h1{font-size:24px}h2{font-size:19px;margin-top:32px}"
         "table{border-collapse:collapse;margin:12px 0;width:100%}td,th{border:1px solid #d0d0d5;"
         "padding:6px 8px;vertical-align:top;text-align:left}th{background:#f2f2f5}"
         "img{max-width:220px;border:1px solid #ddd}.ok{color:#0a7a2f}.bad{color:#b00020}"
         ".box{background:#f6f7fb;border:1px solid #dde;padding:10px 14px;border-radius:6px}"
         ".num{text-align:right;white-space:nowrap}.small{color:#555;font-size:12px}</style>")


def _pct(after: float, before: float) -> str:
    return f"{100 * (after - before) / before:+.1f} %" if before else "–"


def _head(title: str, refresh: bool) -> list[str]:
    meta = "<meta http-equiv='refresh' content='20'>" if refresh else ""
    return [(f"<!doctype html><meta charset='utf-8'>{meta}<meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>{STYLE}")]


def _thumbs(out: Path, names: list[str]) -> str:
    parts = []
    for name in names:
        link = html.escape(f"files/{name}")
        path = out / "files" / name
        if not path.exists():
            continue
        if name.endswith((".png", ".jpeg", ".jpg", ".webp", ".gif")):
            with Image.open(path) as img:
                dims = f"{img.width}x{img.height}"
            parts.append(f"<a href='{link}'><img src='{link}' loading='lazy'></a><br>{dims}, "
                         f"{path.stat().st_size / 1024:,.0f} KB")
        elif name.endswith(".pdf"):
            pages = len(PdfReader(path).pages)
            parts.append(f"<a href='{link}'>PDF, {pages} pages, {path.stat().st_size / 1e6:.1f} MB</a>")
        else:
            tokens = count_tokens(path.read_text(encoding="utf-8"))
            parts.append(f"<a href='{link}'>text, {tokens:,} tokens</a>")
    return "<br>".join(parts)


def _ok(value: str) -> str:
    return "✓" if value == "True" else "✗"


def live_summary(rows: list[dict]) -> list[dict]:
    """Per model: items with both arms, correct A / B, token totals, times."""
    out = []
    for model in sorted({r["model"] for r in rows}):
        by = {(r["id"], r["arm"]): r for r in rows if r["model"] == model}
        ids = sorted({i for i, _ in by if (i, "A") in by and (i, "B") in by})
        if not ids:
            continue
        a = [by[(i, "A")] for i in ids]
        b = [by[(i, "B")] for i in ids]
        tok_a = sum(int(r["input_equivalent"]) for r in a)
        tok_b = sum(int(r["input_equivalent"]) for r in b)
        changed = [r for r in b if r["what_tokencut_did"] != "nothing"]
        out.append({
            "model": model, "items": len(ids),
            "correct_a": sum(r["correct"] == "True" for r in a),
            "correct_b": sum(r["correct"] == "True" for r in b),
            "changed": len(changed),
            "tokens_a": tok_a, "tokens_b": tok_b, "change": _pct(tok_b, tok_a),
            "att_a": sum(int(r["attachment_tokens_est"]) for r in a),
            "att_saved": sum(int(r["tokens_saved_est"]) for r in b),
            "gateway_ms": statistics.median(float(r["gateway_ms"]) for r in b),
            "wall_a": statistics.median(float(r["wall_s"]) for r in a),
            "wall_b": statistics.median(float(r["wall_s"]) for r in b),
            "page_images_removed": sum(int(r["page_images_removed"] or 0) for r in b),
        })
    return out


def report(out: Path) -> Path:
    """report.md + report.html for one folder (one benchmark): offline and live results."""
    bench = out.name if out.name in BENCH_INFO else "sample"
    title, about = BENCH_INFO.get(bench, BENCH_INFO["sample"])
    md = [f"# TokenCut attachment benchmark: {title}", "", f"Updated {_now()}.", "",
          "## What this test checks", "", about, ""]
    page = _head(f"TokenCut attachment benchmark: {title}", refresh=True)
    page += [f"<h1>TokenCut attachment benchmark: {html.escape(title)}</h1>",
             (f"<p class='small'>Updated {_now()} · refreshes every 20 s · "
             "<a href='../index.html'>all benchmarks</a></p>"),
             f"<h2>What this test checks</h2><p class='box'>{html.escape(about)}</p>"]
    live_rows = _read_csv(out / "live.csv")
    if live_rows:
        summary = live_summary(live_rows)
        md += ["## Result in one table (live, real model answers)", "",
               ("| Model | Items | Correct without TokenCut (A) | Correct with TokenCut (B) | Items TokenCut changed "
               "| Billed input-equivalent tokens A → B | Attachment tokens saved (est.) | Gateway time (median, B) "
               "| Run time (median) A / B |"), "|---|---|---|---|---|---|---|---|---|"]
        page.append("<h2>Result in one table (live, real model answers)</h2><table><tr><th>Model</th>"
                    "<th>Items</th><th>Correct without TokenCut (A)</th><th>Correct with TokenCut (B)"
                    "</th><th>Items TokenCut changed</th><th>Billed input-equivalent tokens A → B</th>"
                    "<th>Attachment tokens saved (est.)</th><th>Gateway time (median, B)</th>"
                    "<th>Run time (median) A / B</th></tr>")
        for s in summary:
            md.append(f"| {s['model']} | {s['items']} | {s['correct_a']} | {s['correct_b']} | {s['changed']} | "
                      f"{s['tokens_a']:,} → {s['tokens_b']:,} ({s['change']}) | {s['att_saved']:,} of "
                      f"{s['att_a']:,} | {s['gateway_ms']:.0f} ms | {s['wall_a']:.0f} s / {s['wall_b']:.0f} s |")
            page.append(f"<tr><td>{html.escape(s['model'])}</td><td class=num>{s['items']}</td>"
                        f"<td class=num>{s['correct_a']}</td><td class=num>{s['correct_b']}</td>"
                        f"<td class=num>{s['changed']}</td><td class=num>{s['tokens_a']:,} → "
                        f"{s['tokens_b']:,} ({s['change']})</td><td class=num>{s['att_saved']:,} of "
                        f"{s['att_a']:,}</td><td class=num>{s['gateway_ms']:.0f} ms</td>"
                        f"<td class=num>{s['wall_a']:.0f} s / {s['wall_b']:.0f} s</td></tr>")
        page.append("</table>")
        md.append("")
    off_path = out / "offline.json"
    if off_path.exists():
        rows = json.loads(off_path.read_text(encoding="utf-8"))
        md += ["## What TokenCut did to each attachment (offline, no model call)", "",
               ("Each file is sent through the gateway's own code as if the agent's Read tool had "
               "returned it. Token numbers are TokenCut's estimates from each provider's published "
               "formula."), "",
               ("| Model | Item | What TokenCut did | Why nothing changed | Attachment tokens before → after "
                "| PDF page pictures removed | Gateway time |"), "|---|---|---|---|---|---|---|"]
        page.append("<h2>What TokenCut did to each attachment (offline, no model call)</h2><p>Each file "
                    "is sent through the gateway's own code as if the agent's Read tool had returned it. "
                    "Token numbers are TokenCut's estimates from each provider's published formula. "
                    "Click a picture or PDF to open it.</p><table><tr><th>Model</th><th>Item</th>"
                    "<th>Question</th><th>What TokenCut did</th><th>Attachment tokens before → after "
                    "(est.)</th><th>Original</th><th>Sent with TokenCut</th><th>Gateway time</th></tr>")
        for r in rows:
            change = _pct(r["attachment_tokens_after"], r["attachment_tokens_before"])
            reason = r.get("why_unchanged", "")
            md.append(f"| {r['model']} | {r['id']} | {r['what_tokencut_did']} | {reason} | "
                      f"{r['attachment_tokens_before']:,} → {r['attachment_tokens_after']:,} ({change}) | "
                      f"{r['page_images_removed']} | {r['gateway_ms']} ms |")
            did = html.escape(r["what_tokencut_did"])
            if reason:
                did += f"<br><span class=small>{html.escape(reason)}</span>"
            if r["page_images_removed"]:
                did += f"<br><span class=small>{r['page_images_removed']} page pictures not sent</span>"
            after = ("same as original" if r["after_files"] == r["before_files"]
                     else _thumbs(out, r["after_files"]))
            page.append(f"<tr><td>{html.escape(r['model'])}</td><td>{html.escape(r['id'])}</td>"
                        f"<td>{html.escape(r.get('question', ''))}</td><td>{did}</td>"
                        f"<td class=num>{r['attachment_tokens_before']:,} → "
                        f"{r['attachment_tokens_after']:,} ({change})</td>"
                        f"<td>{_thumbs(out, r['before_files'])}</td><td>{after}</td>"
                        f"<td class=num>{r['gateway_ms']} ms</td></tr>")
        page.append("</table>")
        md.append("")
    if live_rows:
        md += ["## Every live run", "", ("| Model | Item | What TokenCut did (B) | Expected | Answer A | Answer B | "
               "Correct A / B | Input-eq. tokens A → B | Cache read / write B | Gateway ms B | Run s A / B | "
               "Order |"), "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        page.append("<h2>Every live run</h2><p>A = TokenCut measures only (request unchanged), B = TokenCut "
                    "on. Each run is a fresh Claude Code session with a freshly started gateway; the arm "
                    "that runs first alternates per item.</p><table><tr><th>Model</th><th>Item</th>"
                    "<th>What TokenCut did (B)</th><th>Expected</th><th>Answer A</th><th>Answer B</th>"
                    "<th>Input-eq. tokens A → B</th><th>Cache read / write B</th><th>Gateway ms B</th>"
                    "<th>Run s A / B</th><th>Route</th></tr>")
        by = {(r["model"], r["id"], r["arm"]): r for r in live_rows}
        for model, item_id in dict.fromkeys((r["model"], r["id"]) for r in live_rows):
            a, b = by.get((model, item_id, "A")), by.get((model, item_id, "B"))
            if not (a and b):
                continue
            change = _pct(float(b["input_equivalent"]), float(a["input_equivalent"]))
            md.append(f"| {model} | {item_id} | {b['what_tokencut_did']} | {b['expected']} | {a['answer'][:80]} | "
                      f"{b['answer'][:80]} | {_ok(a['correct'])} / {_ok(b['correct'])} | "
                      f"{int(a['input_equivalent']):,} → {int(b['input_equivalent']):,} ({change}) | "
                      f"{b['cache_read_tokens']} / {b['cache_write_tokens']} | {b['gateway_ms']} | "
                      f"{a['wall_s']} / {b['wall_s']} | {b['order']} |")
            cls = {k: "ok" if v["correct"] == "True" else "bad" for k, v in (("A", a), ("B", b))}
            page.append(f"<tr><td>{html.escape(model)}</td><td>{html.escape(item_id)}</td>"
                        f"<td>{html.escape(b['what_tokencut_did'])}</td><td>{html.escape(b['expected'])}</td>"
                        f"<td class='{cls['A']}'>{_ok(a['correct'])} {html.escape(a['answer'][:200])}</td>"
                        f"<td class='{cls['B']}'>{_ok(b['correct'])} {html.escape(b['answer'][:200])}</td>"
                        f"<td class=num>{int(a['input_equivalent']):,} → {int(b['input_equivalent']):,} "
                        f"({change})</td><td class=num>{b['cache_read_tokens']} / {b['cache_write_tokens']}"
                        f"</td><td class=num>{b['gateway_ms']}</td><td class=num>{a['wall_s']} / "
                        f"{b['wall_s']}</td><td>{html.escape(b['routes'])}</td></tr>")
        page.append("</table>")
    md += ["", "## How to read the numbers", "",
           "- **Correct A / B**: the model's answer matched the official answer without (A) / with (B) TokenCut.",
           ("- **Billed input-equivalent tokens**: what the provider billed, as uncached input + 0.1 × cache "
           "reads + 1.25 × cache writes (Anthropic price ratios). Claude Code's own system prompt and tool "
           "definitions are in every request of both arms, so the attachment's effect is the A − B difference."),
           ("- **Attachment tokens (est.)**: TokenCut's estimate from the provider's published image / PDF "
           "formula. PDF page pictures are not counted (providers do not publish their cost)."),
           "- **Gateway time**: time TokenCut itself added (reading, counting, trimming, image / PDF work).",
           "- **Run time**: the whole Claude Code run, model included.",
           ("- One run per item and arm: small differences in tokens or time can be noise. Measured: "
           "whichever arm runs first for an item writes ~1,000 more tokens to the prompt cache than "
           "the second (same size as one image's saving), so compare the totals over all items (the "
           "order alternates, so it cancels out) or the estimated attachment saving, not single rows."),
           ("- Claude Code itself shrinks images to at most 2,000 px before sending, so the live "
           "'before' size is often smaller than the original file.")]
    page.append("<h2>How to read the numbers</h2><ul>" + "".join(
        f"<li>{html.escape(line[2:]).replace('**', '')}</li>" for line in md[-7:]) + "</ul>")
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "report.html").write_text("\n".join(page), encoding="utf-8")
    return out / "report.html"


def report_switch(out: Path) -> Path:
    rows = _read_csv(out / "switch.csv")
    md = ["# TokenCut attachment benchmark: model switch Claude → GPT → Claude", "", f"Updated {_now()}.", "",
          "## What this test checks", "",
          ("One Claude Code session per item: turn 1 Claude reads the attachment and answers; turn 2 the "
           "same session continues on a GPT model (ChatGPT plan, through LiteLLM) and answers again "
           "without re-reading; turn 3 goes back to Claude. A = TokenCut measures only, B = TokenCut on. "
           "Checks that the attachment in the history still works after a provider switch and how "
           "the cache events behave."), "",
          ("| Item | Arm | Turn | Model | Route (via) | Cache event | Correct | Answer | Input-eq. tokens | "
          "Gateway ms | Run s |"), "|---|---|---|---|---|---|---|---|---|---|---|"]
    page = _head("TokenCut model switch test", refresh=True)
    page += ["<h1>Model switch: Claude → GPT → Claude</h1>",
             f"<p class='small'>Updated {_now()} · <a href='../index.html'>all benchmarks</a></p>",
             (f"<h2>What this test checks</h2><p class='box'>{html.escape(md[6])}</p><table><tr>"
             "<th>Item</th><th>Arm</th><th>Turn</th><th>Model</th><th>Route (via)</th><th>Cache event</th>"
             "<th>Answer</th><th>Input-eq. tokens</th><th>What TokenCut did</th><th>Gateway ms</th>"
             "<th>Run s</th></tr>")]
    for r in rows:
        md.append(f"| {r['id']} | {r['arm']} | {r['turn']} | {r['model']} | {r['routes']} ({r['via']}) | "
                  f"{r['cache_events']} | {_ok(r['correct'])} | {r['answer'][:80]} | {r['input_equivalent']} | "
                  f"{r['gateway_ms']} | {r['wall_s']} |")
        cls = "ok" if r["correct"] == "True" else "bad"
        page.append(f"<tr><td>{html.escape(r['id'])}</td><td>{r['arm']}</td><td>{r['turn']}</td>"
                    f"<td>{html.escape(r['model'])}</td><td>{html.escape(r['routes'])} "
                    f"({html.escape(r['via'])})</td><td>{html.escape(r['cache_events'])}</td>"
                    f"<td class='{cls}'>{_ok(r['correct'])} {html.escape(r['answer'][:200])}</td>"
                    f"<td class=num>{int(r['input_equivalent']):,}</td>"
                    f"<td>{html.escape(r['what_tokencut_did'])}</td><td class=num>{r['gateway_ms']}</td>"
                    f"<td class=num>{r['wall_s']}</td></tr>")
    page.append("</table>")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "report.html").write_text("\n".join(page), encoding="utf-8")
    return out / "report.html"


def summary(root: Path = REAL) -> Path:
    """index.html + SUMMARY.md: one line per benchmark and model, links to each report."""
    root.mkdir(parents=True, exist_ok=True)
    md = ["# TokenCut attachment benchmark: overall summary", "", f"Updated {_now()}.", "",
          ("Real items from five public benchmarks used in the research papers (10 per benchmark, "
           "fixed random sample). A = TokenCut measures only (nothing changed), B = TokenCut on."), "",
          "## Live results (real model answers)", "",
          ("| Benchmark | Model | Items | Correct A | Correct B | Items changed | Billed input-eq. tokens A → B "
          "| Attachment tokens saved (est.) | Gateway ms (median) | Run s (median) A / B |"),
          "|---|---|---|---|---|---|---|---|---|---|"]
    page = _head("TokenCut attachment benchmark", refresh=True)
    page += ["<h1>TokenCut attachment benchmark: overall summary</h1>",
             f"<p class='small'>Updated {_now()} · refreshes every 20 s</p>",
             ("<p class='box'><b>Start here:</b> <a href='FINDINGS.md'>FINDINGS.md</a> (conclusions, "
              "bugs found, why the numbers are noisy)</p>" if (root / "FINDINGS.md").exists() else ""),
             (f"<p class='box'>{html.escape(md[4])}</p><h2>Live results (real model answers)</h2>"
             "<table><tr><th>Benchmark</th><th>Model</th><th>Items</th><th>Correct A</th><th>Correct B</th>"
             "<th>Items changed</th><th>Billed input-eq. tokens A → B</th><th>Attachment tokens saved "
             "(est.)</th><th>Gateway ms (median)</th><th>Run s (median) A / B</th></tr>")]
    offline_lines = []
    for bench in BENCHMARKS:
        folder = root / bench
        title = BENCH_INFO[bench][0]
        link = f"<a href='{bench}/report.html'>{html.escape(title)}</a>"
        for s in live_summary(_read_csv(folder / "live.csv")):
            md.append(f"| [{title}]({bench}/report.md) | {s['model']} | {s['items']} | {s['correct_a']} | "
                      f"{s['correct_b']} | {s['changed']} | {s['tokens_a']:,} → {s['tokens_b']:,} "
                      f"({s['change']}) | {s['att_saved']:,} of {s['att_a']:,} | {s['gateway_ms']:.0f} | "
                      f"{s['wall_a']:.0f} / {s['wall_b']:.0f} |")
            page.append(f"<tr><td>{link}</td><td>{html.escape(s['model'])}</td><td class=num>{s['items']}</td>"
                        f"<td class=num>{s['correct_a']}</td><td class=num>{s['correct_b']}</td>"
                        f"<td class=num>{s['changed']}</td><td class=num>{s['tokens_a']:,} → {s['tokens_b']:,} "
                        f"({s['change']})</td><td class=num>{s['att_saved']:,} of {s['att_a']:,}</td>"
                        f"<td class=num>{s['gateway_ms']:.0f}</td><td class=num>{s['wall_a']:.0f} / "
                        f"{s['wall_b']:.0f}</td></tr>")
        if (folder / "offline.json").exists():
            rows = json.loads((folder / "offline.json").read_text(encoding="utf-8"))
            for model in dict.fromkeys(r["model"] for r in rows):
                mine = [r for r in rows if r["model"] == model]
                before = sum(r["attachment_tokens_before"] for r in mine)
                after = sum(r["attachment_tokens_after"] for r in mine)
                changed = sum(r["what_tokencut_did"] != "nothing (sent as is)" for r in mine)
                ms = statistics.median(r["gateway_ms"] for r in mine)
                pics = sum(r["page_images_removed"] for r in mine)
                offline_lines.append((bench, title, link, model, len(mine), changed, before, after, ms, pics))
    page.append("</table>")
    md += ["", "## What TokenCut did (offline, every item, no model call)", "",
           ("| Benchmark | Model | Items | Items changed | Attachment tokens before → after (est.) | "
           "PDF page pictures removed | Gateway ms (median) |"), "|---|---|---|---|---|---|---|"]
    page.append("<h2>What TokenCut did (offline, every item, no model call)</h2><table><tr><th>Benchmark</th>"
                "<th>Model</th><th>Items</th><th>Items changed</th><th>Attachment tokens before → after "
                "(est.)</th><th>PDF page pictures removed</th><th>Gateway ms (median)</th></tr>")
    for bench, title, link, model, n, changed, before, after, ms, pics in offline_lines:
        md.append(f"| [{title}]({bench}/report.md) | {model} | {n} | {changed} | {before:,} → {after:,} "
                  f"({_pct(after, before)}) | {pics} | {ms:.0f} |")
        page.append(f"<tr><td>{link}</td><td>{html.escape(model)}</td><td class=num>{n}</td>"
                    f"<td class=num>{changed}</td><td class=num>{before:,} → {after:,} "
                    f"({_pct(after, before)})</td><td class=num>{pics}</td><td class=num>{ms:.0f}</td></tr>")
    page.append("</table>")
    switch_rows = _read_csv(root / "switch" / "switch.csv")
    if switch_rows:
        md += ["", "## Model switch Claude → GPT → Claude", "", "[Details](switch/report.md)", "",
               "| Arm | Turn | Model | Correct | Sessions |", "|---|---|---|---|---|"]
        page.append("<h2>Model switch Claude → GPT → Claude (<a href='switch/report.html'>details</a>)</h2>"
                    "<table><tr><th>Arm</th><th>Turn</th><th>Model</th><th>Correct</th></tr>")
        for arm in ("A", "B"):
            for turn in ("1", "2", "3"):
                mine = [r for r in switch_rows if r["arm"] == arm and r["turn"] == turn]
                if not mine:
                    continue
                ok = sum(r["correct"] == "True" for r in mine)
                md.append(f"| {arm} | {turn} | {mine[0]['model']} | {ok} / {len(mine)} | {len(mine)} |")
                page.append(f"<tr><td>{arm}</td><td>{turn}</td><td>{html.escape(mine[0]['model'])}</td>"
                            f"<td class=num>{ok} / {len(mine)}</td></tr>")
        page.append("</table>")
    page.append("<h2>Benchmarks</h2><ul>" + "".join(
        f"<li><a href='{b}/report.html'>{html.escape(BENCH_INFO[b][0])}</a>: "
        f"{html.escape(BENCH_INFO[b][1])}</li>" for b in BENCHMARKS) + "</ul>")
    (root / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (root / "index.html").write_text("\n".join(page), encoding="utf-8")
    return root / "index.html"


def report_all(root: Path = REAL) -> Path:
    for bench in BENCHMARKS:
        if (root / bench).exists():
            report(root / bench)
    if (root / "switch" / "switch.csv").exists():
        report_switch(root / "switch")
    return summary(root)


def serve(root: Path = REAL, port: int = SERVE_PORT) -> None:
    root.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(root))
    print(f"results: http://127.0.0.1:{port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


def _benches(value: str) -> list[str]:
    return BENCHMARKS if value == "all" else [b.strip() for b in value.split(",") if b.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample")
    p_off = sub.add_parser("offline")
    p_off.add_argument("--bench", default="all", help="all, sample, or e.g. chartqa,wtq")
    p_off.add_argument("--models", default="claude-haiku-5-5,claude-sonnet-4-5,gpt-6-luna")
    p_live = sub.add_parser("live")
    p_live.add_argument("--bench", default="all")
    p_live.add_argument("--model", default="haiku")
    p_live.add_argument("--arms", default="A,B")
    p_live.add_argument("--limit", type=int, default=None, help="first N items per benchmark")
    p_live.add_argument("--tools", default=None,
                        help='only these Claude Code tools, e.g. "Read" (default: all)')
    p_sw = sub.add_parser("switch")
    p_sw.add_argument("--bench", default="chartqa,docvqa,mmlongbench")
    p_sw.add_argument("--claude", default="haiku")
    p_sw.add_argument("--gpt", default="gpt-6-luna")
    p_sw.add_argument("--limit", type=int, default=1)
    sub.add_parser("report")
    p_srv = sub.add_parser("serve")
    p_srv.add_argument("--port", type=int, default=SERVE_PORT)
    args = parser.parse_args(argv)
    if args.cmd == "sample":
        print(f"sample items: {build_sample()}")
    elif args.cmd == "offline":
        for bench in _benches(args.bench):
            out = bench_dir(bench)
            out.mkdir(parents=True, exist_ok=True)
            rows = offline(bench_items(bench), [m.strip() for m in args.models.split(",") if m.strip()], out)
            print(f"{bench}: {len(rows)} offline rows; report: {report(out)}", flush=True)
        print(f"summary: {summary()}")
    elif args.cmd == "live":
        if args.model.startswith(media.DEFAULT_GPT_PREFIXES) and not litellm_ready():
            sys.exit("LiteLLM is not running on port 4000: start deploy\\litellm\\start.ps1 first")
        try:
            for bench in _benches(args.bench):
                out = bench_dir(bench)
                out.mkdir(parents=True, exist_ok=True)
                live(bench_items(bench), args.model, out, [a.strip() for a in args.arms.split(",")],
                     args.limit, args.tools)
                summary()
        except UsageLimitError as exc:
            summary()
            sys.exit(f"stopped: usage limit reached ({exc}). Run the same command later to resume.")
        print(f"summary: {summary()}")
    elif args.cmd == "switch":
        if not litellm_ready():
            sys.exit("LiteLLM is not running on port 4000: start deploy\\litellm\\start.ps1 first")
        out = REAL / "switch"
        out.mkdir(parents=True, exist_ok=True)
        try:
            for bench in _benches(args.bench):
                switch(bench_items(bench), out, ["A", "B"], args.claude, args.gpt, args.limit)
        except UsageLimitError as exc:
            sys.exit(f"stopped: usage limit reached ({exc}). Run the same command later to resume.")
        print(f"summary: {summary()}")
    elif args.cmd == "report":
        print(f"summary: {report_all()}")
    else:
        serve(port=args.port)


if __name__ == "__main__":
    main()
