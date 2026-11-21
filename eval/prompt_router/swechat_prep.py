"""EDA + preprocessing of SWE-chat user prompts into router training data (SWE-chat's own labels).

    python -m eval.prompt_router.swechat_prep        # needs cache/prompt_router/swechat_user_prompts.jsonl

Steps, each counted in the report:
1. drop machine-generated prompts (slash commands, interrupts, harness / CI / sub-agent tags,
   Codex approval requests, image-only messages): in the gateway these get no brief by rule;
2. drop non-English and unlabelled prompts;
3. strip image markup, keep head + tail of long prompts (pasted logs), mask secrets;
4. map SWE-chat's intent + pushback to brief labels (corrections / rejections steer a running
   task: no brief);
5. remove exact duplicates ("yes", "commit and push" ...) with a majority label, cap prompts per
   session and per repository so no developer dominates;
6. split by REPOSITORY (no repository in two splits), balance only the training split.

Outputs (local, gitignored): cache/prompt_router/kaggle_swe/{train,dev,test}.jsonl
Report (numbers only, committed): eval/results/prompt_router/swechat_eda.md
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from eval.prompt_router.data import clean

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "cache" / "prompt_router" / "swechat_user_prompts.jsonl"
OUT_DIR = ROOT / "cache" / "prompt_router" / "kaggle_swe"
REPORT = ROOT / "eval" / "results" / "prompt_router" / "swechat_eda.md"

INTENT_TO_LABEL = {"create new code": "feature", "debug": "bugfix", "refactor": "refactor",
                   "test": "tests", "understand": "explain", "git": "run_action",
                   "connect": "run_action", "other": "no_brief"}
STEERING = {"correction", "rejection"}  # reported in the EDA only
MAX_CHARS = 2_000
PER_SESSION, PER_REPO = 20, 400  # dev / test only: one project cannot decide the score
TRAIN_PER_REPO = 3_000  # training: no single project above ~2-3 % of the data
TRAIN_PER_LABEL = 5_000
EVAL_SIZE = 3_000
SPLIT = {"dev": 0.1, "test": 0.1}

_MACHINE_START = re.compile(
    r"^[\s\"'`]*(<(command-name|command-message|command-args|local-command-stdout|environment_context|"
    r"subagent_notification|turn_aborted|skill|skills|codex_internal_context|recommended_plugins|"
    r"timestamp|task-notification|teammate-message|ci-monitor-event|objective|state_management|"
    r"verification|team_[a-z_]+|system-reminder|user_instructions|permissions)\b"
    r"|\[\d+\] tool [\w.-]+ (call|result):|\[Request interrupted|\[delegate worktree\]|/[A-Za-z][\w:-]*(\s|$)|>>> )",
    re.IGNORECASE)
_MACHINE_ANYWHERE = re.compile(
    r"APPROVAL REQUEST START|The Codex agent has requested|TRANSCRIPT DELTA START|"
    r"^Execution instructions:|^# AGENTS\.md instructions|Another Claude session sent a message|"
    r"<teammate-message|<task-notification|<usage>|<tool_uses>", re.IGNORECASE | re.MULTILINE)
_IMAGE = re.compile(r"<image[^>]*>|</image>|\[Image[^\]]*\]", re.IGNORECASE)


def strip_markup(text: str) -> str:
    return " ".join(_IMAGE.sub(" ", text).split()) if _IMAGE.search(text) else text.strip()


def is_machine(text: str) -> bool:
    """Not typed by the user: harness, slash command, CI / sub-agent notice, approval request."""
    rest = strip_markup(text)
    return (not rest or bool(_MACHINE_START.match(text)) or bool(_MACHINE_ANYWHERE.search(text)))


def cut(text: str, limit: int = MAX_CHARS) -> str:
    """Head + tail: the request is usually at the start, sometimes after a pasted log."""
    if len(text) <= limit:
        return text
    tail = limit // 4
    return text[: limit - tail] + "\n…\n" + text[-tail:]


def target_label(intent: str | None, pushback: str | None) -> str | None:
    """SWE-chat intent -> brief label. Pushback (correction / rejection) does not change it: the
    text says what is asked; repeating a brief is decided in the gateway from the conversation."""
    if intent is None or intent not in INTENT_TO_LABEL:
        return None
    return INTENT_TO_LABEL[intent]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def dedupe(rows: list[dict]) -> list[dict]:
    """One row per normalised text, labelled by majority vote (ties: alphabetical)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_norm(row["text"])].append(row)
    out = []
    for key in sorted(groups):
        members = groups[key]
        votes = Counter(r["label"] for r in members)
        label = min(votes, key=lambda lab: (-votes[lab], lab))
        out.append({**members[0], "label": label, "copies": len(members)})
    return out


def cap(rows: list[dict], key: str, limit: int, rng: random.Random) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    out = []
    for name in sorted(groups):
        members = groups[name]
        out += rng.sample(members, limit) if len(members) > limit else members
    return out


def repo_split(rows: list[dict], seed: int = 7, weight_cap: int | None = None
               ) -> dict[str, list[dict]]:
    """Whole repositories go to train / dev / test, ~10 % each to dev and test.

    weight_cap: a repository counts with at most this many prompts when filling dev / test (it
    is capped there later), so one huge project cannot fill a whole evaluation split."""
    counts = Counter(r["repo_id"] for r in rows)
    if weight_cap is not None:
        counts = Counter({repo: min(c, weight_cap) for repo, c in counts.items()})
    repos = sorted(counts)
    random.Random(seed).shuffle(repos)
    total, assigned, target = sum(counts.values()), {}, dict(SPLIT)
    filled = {name: 0 for name in target}
    for repo in repos:
        name = next((n for n in target if filled[n] < target[n] * total), "train")
        assigned[repo] = name
        if name != "train":
            filled[name] += counts[repo]
    out: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for row in rows:
        out[assigned[row["repo_id"]]].append(row)
    return out


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "-"


def main() -> None:
    rng = random.Random(13)
    raw = [json.loads(line) for line in SOURCE.open(encoding="utf-8")]
    lines = ["# SWE-chat user prompts — EDA and preprocessing for the prompt router", "",
             ("Source: SWE-chat (SALT-NLP, COLM 2026; ODC-BY), all user prompts of "
             "`conversations.parquet`. Labels: SWE-chat's own LLM annotations (`prompt_intent`, "
             "`prompt_pushback`). Numbers only; prompt texts stay local."), ""]

    # ---- EDA on everything ----
    n = len(raw)
    lines += [(f"## Population: {n:,} prompts, {len({r['session_id'] for r in raw}):,} sessions, "
              f"{len({r['repo_id'] for r in raw}):,} repositories"), ""]
    machine = [r for r in raw if is_machine(r["text"])]
    human = [r for r in raw if not is_machine(r["text"])]
    lines += [(f"- machine-generated (slash commands, interrupts, harness / CI / sub-agent / "
              f"approval messages, image-only): {len(machine):,} ({_pct(len(machine), n)})"),
              f"- typed by the user: {len(human):,} ({_pct(len(human), n)})", ""]
    lines += ["| Intent (SWE-chat) | all prompts | user-typed | of which correction / rejection |",
              "|---|---|---|---|"]
    all_int = Counter(r["prompt_intent"] for r in raw)
    hum_int = Counter(r["prompt_intent"] for r in human)
    steer = Counter(r["prompt_intent"] for r in human if r["prompt_pushback"] in STEERING)
    for intent, count in all_int.most_common():
        lines.append(f"| {intent} | {count:,} ({_pct(count, n)}) | {hum_int[intent]:,} | "
                     f"{_pct(steer[intent], hum_int[intent])} |")
    lengths = sorted(len(r["text"]) for r in human)
    first = sum(1 for r in human if r["turn"] == 0)
    lines += ["", (f"User-typed prompts: median length {lengths[len(lengths) // 2]} chars, "
              f"p90 {lengths[int(.9 * len(lengths))]}, p99 {lengths[int(.99 * len(lengths))]}; "
              f"{_pct(sum(x <= 20 for x in lengths), len(lengths))} are ≤ 20 chars; "
              f"first prompt of a session: {_pct(first, len(human))}."),
              "Agents: " + ", ".join(f"{a} {_pct(c, len(human))}" for a, c in
                                     Counter(r["agent"] for r in human).most_common(5)) + ".", ""]

    # ---- preprocessing funnel ----
    funnel = [("all user prompts", n), ("minus machine-generated", len(human))]
    english = [r for r in human if (r["language"] or "").upper() == "ENGLISH"]
    funnel.append(("minus non-English", len(english)))
    rows = []
    for r in english:
        label = target_label(r["prompt_intent"], r["prompt_pushback"])
        if label is None:
            continue
        text = cut(clean(strip_markup(r["text"])))
        if text:
            rows.append({"id": r["id"], "text": text, "label": label, "intent": r["prompt_intent"],
                         "pushback": r["prompt_pushback"], "repo_id": r["repo_id"],
                         "session_id": r["session_id"], "agent": r["agent"], "turn": r["turn"]})
    funnel.append(("minus unlabelled", len(rows)))
    rows = dedupe(rows)
    funnel.append(("minus exact duplicates (majority label kept)", len(rows)))
    lines += ["## Preprocessing funnel", "", "| Step | Prompts left |", "|---|---|"]
    lines += [f"| {step} | {count:,} |" for step, count in funnel]

    splits = repo_split(rows, weight_cap=PER_REPO)
    pool = cap(splits["train"], "repo_id", TRAIN_PER_REPO, rng)  # no per-session cap in training
    for name in ("dev", "test"):
        splits[name] = cap(cap(splits[name], "session_id", PER_SESSION, rng), "repo_id", PER_REPO, rng)
    lines += ["", (f"Training pool: {len(splits['train']):,} prompts in training repositories, "
              f"{len(pool):,} after at most {TRAIN_PER_REPO:,} per repository (no per-session cap). "
              f"Dev / test: at most {PER_SESSION} per session and {PER_REPO} per repository, so one "
              "project cannot decide the score.")]
    rows = pool + splits["dev"] + splits["test"]
    train = []
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_label[r["label"]].append(r)
    for label in sorted(by_label):
        members = by_label[label]
        train += rng.sample(members, min(TRAIN_PER_LABEL, len(members)))
    rng.shuffle(train)
    final = {"train": train}
    for name in ("dev", "test"):  # natural distribution (honest), size-capped
        part = splits[name]
        final[name] = rng.sample(part, min(EVAL_SIZE, len(part)))

    labels = sorted({r["label"] for r in rows})
    lines += ["", "## Splits (whole repositories per split; train balanced, dev / test natural)", "",
              "| Label | available | train | dev | test |", "|---|---|---|---|---|"]
    clean_counts = Counter(r["label"] for r in rows)
    part_counts = {k: Counter(r["label"] for r in v) for k, v in final.items()}
    for label in labels:
        lines.append(f"| {label} | {clean_counts[label]:,} | {part_counts['train'][label]:,} | "
                     f"{part_counts['dev'][label]:,} | {part_counts['test'][label]:,} |")
    lines.append(f"| **total** | {len(rows):,} | {len(final['train']):,} | {len(final['dev']):,} | "
                 f"{len(final['test']):,} |")
    repos = {k: len({r['repo_id'] for r in v}) for k, v in splits.items()}
    lines += ["", f"Repositories: train {repos['train']}, dev {repos['dev']}, test {repos['test']}.",
              "", ("Label mapping: create new code → feature, debug → bugfix, refactor → refactor, "
              "test → tests, understand → explain, git / connect → run_action, other → no_brief. "
              "Corrections / rejections keep their intent (the text says what is asked); the "
              "gateway decides whether to repeat a brief, from the conversation."), ""]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Natural label mix of the training repositories (before balancing): used to correct the
    # model's probabilities for real traffic, so nothing is tuned on dev.
    natural = Counter(r["label"] for r in pool)
    (OUT_DIR / "natural_prior.json").write_text(encoding="utf-8", data=json.dumps(dict(sorted(natural.items())), indent=1))
    for name, part in final.items():
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(encoding="utf-8", data="\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
