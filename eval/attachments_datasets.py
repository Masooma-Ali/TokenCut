"""Download a fixed random sample of real benchmark items for the attachment benchmark.

    python -m eval.attachments_datasets --per-benchmark 10

Benchmarks (public, Hugging Face; only the sampled items are downloaded):
  chartqa        ChartQA test (arXiv 2203.10244)            chart images      -> border cut / resize
  docvqa         DocVQA validation (arXiv 2007.00398)       document scans    -> resize / border cut
  infovqa        InfographicVQA validation (arXiv 2104.12756) dense infographics -> resize (worst case)
  mmlongbench    MMLongBench-Doc (arXiv 2407.01523)         long PDFs         -> PDF text pages / intent
  wtq            WikiTableQuestions test (arXiv 1508.00305) largest tables as CSV -> table protection

Output: logs/attachments_bench/datasets/<benchmark>/ with the files and an items.jsonl in the
format eval.attachments_bench reads (id, file, question, answers, match, expect, benchmark).
Same seed -> same sample.
"""

import argparse
import ast
import csv
import io
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

from pypdf import PdfReader

from tokencut.tokens import count_tokens

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "logs" / "attachments_bench" / "datasets"
ROWS = "https://datasets-server.huggingface.co/rows"
SEED = 2026
MMLB_DOCS = "https://huggingface.co/datasets/yubo2333/MMLongBench-Doc/resolve/main/documents/"
MMLB_TREE = "https://huggingface.co/api/datasets/yubo2333/MMLongBench-Doc/tree/main/documents"
MMLB_MAX_MB = 6  # keep downloads small; long PDFs are still common at this size
WTQ_MIN_TOKENS = 1_500  # TokenCut's table rule matters only for large tables


def _get(url: str, binary: bool = False, tries: int = 5) -> bytes | dict | list:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            return data if binary else json.loads(data)
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _rows(dataset: str, config: str, split: str, offset: int, length: int = 1) -> list[dict]:
    query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                    "offset": offset, "length": length})
    return [r["row"] for r in _get(f"{ROWS}?{query}")["rows"]]


def _total(dataset: str, config: str, split: str) -> int:
    query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                    "offset": 0, "length": 1})
    return int(_get(f"{ROWS}?{query}")["num_rows_total"])


def _answers(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        parsed = ast.literal_eval(str(value))
        return [str(v) for v in parsed] if isinstance(parsed, list) else [str(parsed)]
    except (ValueError, SyntaxError):
        return [str(value)]


def _write(folder: Path, items: list[dict]) -> Path:
    path = folder / "items.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def _image_items(name: str, dataset: str, config: str, split: str, n: int, question_key: str,
                 answer_key: str, match: str, expect: str, rng: random.Random) -> Path:
    folder = DATASETS / name
    folder.mkdir(parents=True, exist_ok=True)
    total = _total(dataset, config, split)
    items = []
    for offset in sorted(rng.sample(range(total), n)):
        row = _rows(dataset, config, split, offset)[0]
        raw = _get(row["image"]["src"], binary=True)
        suffix = ".png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
        file = f"{name}_{offset}{suffix}"
        (folder / file).write_bytes(raw)
        items.append({"id": f"{name}_{offset}", "benchmark": name, "file": file,
                      "question": row[question_key], "answers": _answers(row[answer_key]),
                      "match": match, "expect": expect})
        print(f"  {name} {offset}: {len(raw) / 1024:.0f} KB", flush=True)
    return _write(folder, items)


def chartqa(n: int, rng: random.Random) -> Path:
    return _image_items("chartqa", "HuggingFaceM4/ChartQA", "default", "test", n, "query", "label",
                        "relaxed", "border cut / resize where the provider bills by size", rng)


def docvqa(n: int, rng: random.Random) -> Path:
    return _image_items("docvqa", "lmms-lab-encoder/DocVQA", "DocVQA", "validation", n, "question",
                        "answers", "any", "resize / border cut of document scans", rng)


def infovqa(n: int, rng: random.Random) -> Path:
    return _image_items("infovqa", "lmms-lab-encoder/DocVQA", "InfographicVQA", "validation", n,
                        "question", "answers", "any", "resize of dense infographics", rng)


def mmlongbench(n: int, rng: random.Random) -> Path:
    folder = DATASETS / "mmlongbench"
    folder.mkdir(parents=True, exist_ok=True)
    total = _total("yubo2333/MMLongBench-Doc", "default", "train")
    rows = []
    for offset in range(0, total, 100):
        rows += _rows("yubo2333/MMLongBench-Doc", "default", "train", offset, 100)
    sizes = {Path(x["path"]).name: x.get("size", 0) for x in _get(MMLB_TREE)}
    small = {doc for doc, size in sizes.items() if 0 < size <= MMLB_MAX_MB * 1e6}
    answerable = [r for r in rows if r["doc_id"] in small and r["answer"] != "Not answerable"]
    by_doc: dict[str, list[dict]] = {}
    for row in answerable:
        by_doc.setdefault(row["doc_id"], []).append(row)
    items = []
    for doc in rng.sample(sorted(by_doc), min(n, len(by_doc))):
        row = rng.choice(by_doc[doc])
        raw = _get(MMLB_DOCS + urllib.parse.quote(doc), binary=True)
        (folder / doc).write_bytes(raw)
        pages = len(PdfReader(io.BytesIO(raw)).pages)
        match = {"Int": "number", "Float": "number", "List": "all"}.get(row["answer_format"], "any")
        items.append({"id": f"mmlb_{Path(doc).stem}"[:80], "benchmark": "mmlongbench", "file": doc,
                      "question": row["question"], "answers": _answers(row["answer"]),
                      "match": match, "pages": pages,
                      "evidence": f"{row['evidence_sources']} on pages {row['evidence_pages']}",
                      "expect": "PDF: text pages as text, other pages as PDF; whole PDF kept for "
                                "visual questions; unchanged above 50 pages"})
        print(f"  mmlongbench {doc}: {len(raw) / 1e6:.1f} MB, {pages} pages", flush=True)
    return _write(folder, items)


def wtq(n: int, rng: random.Random) -> Path:
    folder = DATASETS / "wtq"
    folder.mkdir(parents=True, exist_ok=True)
    candidates = []
    for offset in range(0, 1500, 100):
        for row in _rows("lighteval/wikitablequestions", "default", "test", offset, 100):
            if count_tokens(row["table_md"]) >= WTQ_MIN_TOKENS:
                candidates.append(row)
    items = []
    for row in rng.sample(candidates, min(n, len(candidates))):
        table = row["table"] if isinstance(row["table"], dict) else ast.literal_eval(row["table"])
        file = f"wtq_{row['id']}.csv"
        with (folder / file).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(table["header"])
            writer.writerows(table["rows"])
        items.append({"id": f"wtq_{row['id']}", "benchmark": "wtq", "file": file,
                      "question": row["question"], "answers": _answers(row["answers"]),
                      "match": "all", "expect": "table kept whole (never cut with head / tail)"})
        print(f"  wtq {row['id']}: {len(table['rows'])} rows", flush=True)
    return _write(folder, items)


BENCHMARKS = {"chartqa": chartqa, "docvqa": docvqa, "infovqa": infovqa,
              "mmlongbench": mmlongbench, "wtq": wtq}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-benchmark", type=int, default=10)
    parser.add_argument("--only", default=",".join(BENCHMARKS))
    args = parser.parse_args(argv)
    for name in [b.strip() for b in args.only.split(",") if b.strip()]:
        print(f"{name}:", flush=True)
        path = BENCHMARKS[name](args.per_benchmark, random.Random(f"{SEED}-{name}"))
        print(f"  -> {path}", flush=True)


if __name__ == "__main__":
    main()
