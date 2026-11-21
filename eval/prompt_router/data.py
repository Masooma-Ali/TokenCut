"""Developer prompts for the prompt router, from DevGPT (Xiao et al., MSR 2024; CC BY 4.0).

    python -m eval.prompt_router.data --zip cache/prompt_router/DevGPT.zip --sample 700

DevGPT: developer <-> ChatGPT conversations shared in GitHub issues, PRs, commits, code,
discussions and Hacker News (zenodo.org/records/16392320). We keep the FIRST prompt of each
conversation (a new task, as the router sees it), English only, 15-4,000 characters,
deduplicated, with key- / token- / e-mail-like strings masked. Same zip + seed -> same
sample, so teammates rebuild identical texts; only ids and labels are committed.

Output (local, gitignored): cache/prompt_router/prompts.jsonl  {id, text, source}

    python -m eval.prompt_router.data --extra

Extra TRAINING prompts for rare labels: keyword search over the prompts NOT in the random sample
(keywords per label in EXTRA_KEYWORDS), a seeded pick per label. Keyword-found prompts are not a
random sample, so they are used for training only; dev / test stay the random 700.
Output: cache/prompt_router/prompts_extra.jsonl  {id, text, source, searched_for}
"""

import argparse
import hashlib
import json
import random
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "cache" / "prompt_router"
MIN_CHARS, MAX_CHARS = 15, 4_000
_SOURCES = ("pr", "issue", "commit", "file", "code", "discussion", "hn")
_SECRET = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}"
    r"|xox[abp]-[A-Za-z0-9-]{10,}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)"
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_ENGLISH = frozenset(["the", "a", "to", "is", "in", "it", "and", "of", "for", "this", "that", "how", "i", "you", "my", "can", "with", "what", "why", "not", "do", "does", "code", "error", "please", "write", "make", "add", "fix"])


def prompt_id(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).lower().encode()).hexdigest()[:12]


def clean(text: str) -> str:
    return _EMAIL.sub("<email>", _SECRET.sub("<secret>", text)).strip()


def is_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) < 0.95:
        return False
    words = re.findall(r"[a-z]+", text.lower())
    return sum(w in _ENGLISH for w in words) >= min(2, len(words))


def _source(name: str) -> str:
    low = name.lower()
    return next((s for s in _SOURCES if f"_{s}_" in low or f"_{s}." in low or f"{s}_sharing" in low),
                "other")


def _first_prompts(node, found: list[str]) -> None:
    """Every conversation's first prompt, wherever it sits in DevGPT's JSON."""
    if isinstance(node, dict):
        conversations = node.get("Conversations")
        if isinstance(conversations, list) and conversations:
            first = conversations[0]
            if isinstance(first, dict) and isinstance(first.get("Prompt"), str):
                found.append(first["Prompt"])
        for value in node.values():
            _first_prompts(value, found)
    elif isinstance(node, list):
        for value in node:
            _first_prompts(value, found)


def extract(zip_path: Path) -> list[dict]:
    rows: dict[str, dict] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(n for n in zf.namelist() if n.endswith(".json")):
            try:
                data = json.loads(zf.read(name))
            except ValueError:
                continue
            prompts: list[str] = []
            _first_prompts(data, prompts)
            for raw in prompts:
                text = clean(raw)
                if not (MIN_CHARS <= len(text) <= MAX_CHARS) or not is_english(text):
                    continue
                pid = prompt_id(text)
                rows.setdefault(pid, {"id": pid, "text": text, "source": _source(name)})
    return [rows[k] for k in sorted(rows)]


def sample(rows: list[dict], n: int, seed: int = 13) -> list[dict]:
    """Seeded sample, spread over the sources in proportion to their size."""
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    picked = []
    for source in sorted(by_source):
        group = by_source[source]
        k = round(n * len(group) / len(rows))
        picked += rng.sample(group, min(k, len(group)))
    rng.shuffle(picked)
    return picked[:n]


EXTRA_KEYWORDS = {
    "tests": r"\b(unit ?tests?|pytest|jest|test ?cases?|testing|mock(ing|s)?|coverage|cypress)\b",
    "review": (r"\b(review|code quality|any (bugs|issues|problems)|is (this|it|my code) "
               r"(correct|right|good|ok|valid)|feedback on|improvements? (to|for|in))\b"),
    "run_action": (r"\b(install|deploy|git (push|pull|rebase|commit|clone|checkout|merge)|"
                   r"docker (run|compose|build)|command to|how (do i|to|can i) run)\b"),
    "refactor": r"\b(refactor|simplif(y|ied)|clean ?up|rewrite|more readable|port (this|it) to)\b",
    "design": (r"\b(architecture|best (way|approach|practice)|which (library|framework|database)|"
               r"pros and cons|trade-?offs?|should i use)\b"),
    "write_text": (r"\b(readme|documentation|docstrings?|commit message|blog post|release notes|"
                   r"changelog)\b"),
}


# Round 2: stricter phrases for the two labels round 1 found least often.
EXTRA_KEYWORDS_2 = {
    "tests": (r"\b(write|add|create|generate|need) (some |a |the )?(unit |integration |e2e )?tests?\b|"
              r"\btests? (for|of) (this|my|the)\b|\b(pytest|jest|unittest|vitest|mocha)\b"),
    "review": (r"\b(review (my|this|the) (code|pr|pull request|function|script)|code review|"
               r"is (this|the following|my) (code |function |script )?(correct|right|ok|safe|valid)|"
               r"what('s| is) wrong with (this|my)|any (bugs|issues|problems) (in|with))\b"),
}


def extra_candidates(rows: list[dict], exclude: set[str], per_label: dict[str, int],
                     seed: int = 17, keywords: dict[str, str] | None = None) -> list[dict]:
    """Seeded keyword candidates per rare label, never a prompt already sampled or picked."""
    keywords = keywords or EXTRA_KEYWORDS
    rng = random.Random(seed)
    taken = set(exclude)
    out = []
    for label in sorted(per_label):
        pattern = re.compile(keywords[label], re.IGNORECASE)
        pool = [r for r in rows if r["id"] not in taken and pattern.search(r["text"])]
        for row in rng.sample(pool, min(per_label[label], len(pool))):
            taken.add(row["id"])
            out.append({**row, "searched_for": label})
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=OUT_DIR / "DevGPT.zip")
    parser.add_argument("--sample", type=int, default=700)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--extra", action="store_true", help="keyword candidates for rare labels")
    parser.add_argument("--extra2", action="store_true", help="round 2: stricter tests / review")
    args = parser.parse_args(argv)
    rows = extract(args.zip)
    if args.extra2:
        taken = {json.loads(line)["id"] for name in ("prompts.jsonl", "prompts_extra.jsonl")
                 for line in (OUT_DIR / name).open(encoding="utf-8")}
        extra = extra_candidates(rows, taken, {"tests": 50, "review": 50}, seed=29,
                                 keywords=EXTRA_KEYWORDS_2)
        with (OUT_DIR / "prompts_extra.jsonl").open("a", encoding="utf-8") as f:  # appended: E171 onwards
            for row in extra:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{len(extra)} round-2 candidates appended")
        return
    if args.extra:
        sampled = {json.loads(line)["id"] for line in (OUT_DIR / "prompts.jsonl").open(encoding="utf-8")}
        per_label = {"tests": 45, "review": 40, "run_action": 30, "refactor": 25, "design": 15,
                     "write_text": 15}
        extra = extra_candidates(rows, sampled, per_label)
        with (OUT_DIR / "prompts_extra.jsonl").open("w", encoding="utf-8") as f:
            for row in extra:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{len(extra)} extra candidates for {sorted(per_label)}")
        return
    picked = sample(rows, args.sample, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for row in picked:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for row in picked:
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    print(f"{len(rows)} usable first prompts; sampled {len(picked)}: {counts}")


if __name__ == "__main__":
    main()
