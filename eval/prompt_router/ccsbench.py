"""Extra TRAINING examples for weak router labels from CCSBench (Conventional-Commit messages).

    python -m eval.prompt_router.ccsbench --per-label 150

CCSBench (huggingface.co/datasets/CSU-OSS/CCSBench, Apache-2.0): 93,807 commits from permissively
licensed repositories, tagged with their Conventional-Commit type. Only the rows we need are
fetched (datasets-server /filter, train split), never the whole 824 MB dataset. The commit type
prefix ("refactor(scope)!: ") is removed, so the model cannot learn the label from the tag.
Training only: dev / test stay the random DevGPT prompts.

Output (local, gitignored): cache/prompt_router/ccsbench.jsonl {id, text, label, source, type}
"""

import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.request

from eval.prompt_router.data import OUT_DIR, clean, prompt_id

API = "https://datasets-server.huggingface.co/filter"
DATASET = "CSU-OSS/CCSBench"
TYPE_TO_LABEL = {"refactor": "refactor", "test": "tests", "docs": "write_text"}
PAGE = 100
_PREFIX = re.compile(r"^\s*[a-z]+(\([^)]*\))?!?:\s*", re.IGNORECASE)
_NOISE = re.compile(r"\b(bump|release|version|merge|changelog|typo|wip|revert)\b|v\d+\.\d+",
                    re.IGNORECASE)
MIN_CHARS, MAX_CHARS = 15, 600


def message_text(message: str) -> str:
    """Commit message without the type prefix, without trailers, cut to MAX_CHARS."""
    body = _PREFIX.sub("", message.strip(), count=1)
    lines = [line for line in body.split("\n")
             if not re.match(r"^\s*(signed-off-by|co-authored-by|breaking change|closes|fixes)\b",
                             line, re.IGNORECASE)]
    return clean("\n".join(lines).strip())[:MAX_CHARS]


def usable(text: str) -> bool:
    """Long enough, Latin script, at least 3 words, not a release / merge / typo commit.

    DevGPT's is_english() wants two common English words; terse commit messages ("split the
    tokenizer into its own module") rarely have them, so only the script is checked here."""
    letters = [c for c in text if c.isalpha()]
    latin = bool(letters) and sum(c.isascii() for c in letters) / len(letters) >= 0.95
    return (MIN_CHARS <= len(text) and latin and len(text.split()) >= 3
            and not _NOISE.search(text))


def fetch_page(commit_type: str, offset: int) -> dict:
    query = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": "train",
                                    "where": f"\"commit_type\"='{commit_type}'",
                                    "offset": offset, "length": PAGE})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{API}?{query}", timeout=60) as response:
                data = json.load(response)
            if "rows" in data:
                return data
        except OSError:
            pass
        time.sleep(2 * (attempt + 1))  # the API rate-limits; back off
    raise RuntimeError(f"CCSBench page {commit_type}@{offset} failed")


def collect(commit_type: str, label: str, n: int, rng: random.Random, seen: set[str]) -> list[dict]:
    first = fetch_page(commit_type, 0)
    total = first["num_rows_total"]
    offsets = list(range(0, total, PAGE))
    rng.shuffle(offsets)  # pages from all over the dataset, not one repository's history
    out: list[dict] = []
    for offset in offsets:
        page = first if offset == 0 else fetch_page(commit_type, offset)
        rows = [r["row"] for r in page["rows"]]
        rng.shuffle(rows)
        for row in rows[:10]:  # at most 10 per page: spread over repositories
            text = message_text(row["message"])
            pid = prompt_id(text)
            if usable(text) and pid not in seen:
                seen.add(pid)
                out.append({"id": f"ccs-{pid}", "text": text, "label": label,
                            "source": "ccsbench", "type": commit_type})
        if len(out) >= n:
            break
    return out[:n]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-label", type=int, default=150)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    seen: set[str] = set()
    rows = []
    for commit_type, label in sorted(TYPE_TO_LABEL.items()):
        got = collect(commit_type, label, args.per_label, rng, seen)
        print(f"{commit_type} -> {label}: {len(got)}")
        rows += got
    with (OUT_DIR / "ccsbench.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
