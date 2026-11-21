"""Does naming the needed code in the prompt help the gateway keep it? (prompt enhancement)

Two checks, both free (no model call), both through the real gateway trimmer
(`trim_request`, moderate tier). "Gold" = the functions / methods the real fix edits.

    offline   81 held-out SWE-bench Lite tasks: every gold file is read whole (Claude Code
              Read format) under six user prompts (issue text, a vague prompt, either one plus
              the right function names, the vague prompt plus wrong names from the same file,
              the issue plus the file path). Gold kept = all lines of a gold function survive.
    stepd     the recorded step-D Claude Code sessions (arm A, logs/stepd_p1/, local only):
              every Read of a gold file is replayed as it was sent, today and with the model's
              own words before the call or the right names added to the relevance query.

    python -m eval.prompt_hint_eval offline
    python -m eval.prompt_hint_eval stepd

Results (numbers only): eval/results/prompt_hints/{offline,stepd}.json and summary.md.
The right names come from the real fix, so the "+right" rows are an upper bound.
"""

import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

from eval.benchmark import load_chunks
from eval.loader import DEFAULT_REPOS_DIR, fetch_rows, gold_from_patch, load_tasks
from eval.metrics import gold_chunk_ids
from tokencut.gateway import trimmer
from tokencut.gateway.adapters import anthropic
from tokencut.gateway.trimmer import TrimConfig, trim_request
from tokencut.ingestion.python_parser import PythonParser
from tokencut.tokens import count_tokens

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results" / "prompt_hints"
TASKS = ROOT / "eval" / "tasks_iter2.json"
STEPD = ROOT / "logs" / "stepd_p1"
VAGUE = "Fix the bug described in the issue."
MIN_TOKENS = TrimConfig().min_tokens
_NUMBERED = re.compile(r"^\s*(\d+)(?:→|\t)")  # Claude Code Read lines: "12→code" or "12\tcode"


def _listing(text: str) -> str:
    return "\n".join(f"{n:>6}→{line}" for n, line in enumerate(text.split("\n"), 1))


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content or []
                     if isinstance(b, dict) and b.get("type") == "text")


def _kept_lines(text: str) -> set[int]:
    return {int(m.group(1)) for line in text.split("\n") if (m := _NUMBERED.match(line))}


def _whole(chunk, kept: set[int]) -> bool:
    return set(range(chunk.start_line, chunk.end_line + 1)) <= kept


def _git_show(repo: str, commit: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=repo, capture_output=True,
                          check=True, timeout=60).stdout


def _read_request(user: str, path: str, result: str) -> dict:
    return {"model": "claude-haiku-5-5", "max_tokens": 10, "messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "r1", "name": "Read",
                                           "input": {"file_path": f"/repo/{path}"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "r1",
                                      "content": result}]},
    ]}


def offline() -> list[dict]:
    rows = []
    for task in load_tasks(json.loads(TASKS.read_text())["test"]):
        chunks = load_chunks(task)
        gold = set(gold_chunk_ids(chunks, task.gold_lines))
        for path in task.gold_files:
            g = [c for c in chunks if c.file_path == path and c.id in gold
                 and c.kind.value != "module"]
            if not path.endswith(".py") or not g:
                continue
            listing = _listing(_git_show(task.repo_path, task.base_commit, path).decode())
            tokens = count_tokens(listing)
            names = [c.id.split("::", 1)[1] for c in g]
            wrong = sorted(c.id.split("::", 1)[1] for c in chunks if c.file_path == path
                           and c.id not in gold and c.kind.value in ("function", "method"))
            prompts = {
                "issue": task.query,
                "vague": VAGUE,
                "vague+right": f"{VAGUE} The relevant code: {', '.join(names)}.",
                "issue+right": f"{task.query}\n\nThe relevant code: {', '.join(names)}.",
                "vague+wrong": (f"{VAGUE} The relevant code: {', '.join(wrong[:len(names)])}."
                                if wrong else None),
                "issue+file": f"{task.query}\n\nThe bug is in {path}.",
            }
            for variant, user in prompts.items():
                if user is None:
                    continue
                body = _read_request(user, path, listing)
                new, _ = trim_request(body, anthropic.parse(body), TrimConfig())
                sent = new["messages"][2]["content"][0]["content"] if new else listing
                kept = _kept_lines(sent)
                rows.append({"task": task.instance_id, "file": path, "tokens": tokens,
                             "variant": variant, "gold": len(g),
                             "gold_kept": sum(_whole(c, kept) for c in g),
                             "tokens_sent": count_tokens(sent), "trimmed": new is not None})
    return rows


def stepd() -> list[dict]:
    rows_by_id = {r["instance_id"]: r for r in fetch_rows()}
    repo = str(DEFAULT_REPOS_DIR / "sympy__sympy")
    original = trimmer.trim_file_listing
    extra = {"query": ""}

    def with_extra(text, path, query, tier):  # the experiment: more words in the relevance query
        return original(text, path, f"{query}\n{extra['query']}", tier)

    trimmer.trim_file_listing = with_extra
    out = []
    try:
        for task_dir in sorted(STEPD.glob("sympy__sympy-*")):
            row = rows_by_id[task_dir.name]
            gold_files, gold_lines = gold_from_patch(row["patch"])
            bodies = [json.loads(p.read_text()) for p in (task_dir / "A/gateway/captures").glob("*_anthropic.json")]
            body = max((b for b in bodies if "messages" in b), key=lambda b: len(b["messages"]))
            messages, parsed = body["messages"], {}
            for i, message in enumerate(messages[:-1]):
                if message.get("role") != "assistant" or not isinstance(message.get("content"), list):
                    continue
                said = "\n".join(b.get("text", "") for b in message["content"] if b.get("type") == "text")
                for call in message["content"]:
                    if call.get("type") != "tool_use" or call.get("name") != "Read":
                        continue
                    path = next((g for g in gold_files
                                 if call["input"].get("file_path", "").endswith("/" + g)), None)
                    results = messages[i + 1].get("content")
                    k = next((k for k, r in enumerate(results if isinstance(results, list) else [])
                              if isinstance(r, dict) and r.get("tool_use_id") == call["id"]), None)
                    if path is None or k is None:
                        continue
                    result = _text(results[k].get("content"))
                    if path not in parsed:
                        chunks = PythonParser().parse_file(path, _git_show(repo, row["base_commit"], path))
                        parsed[path] = (chunks, set(gold_chunk_ids(chunks, {path: gold_lines[path]})))
                    chunks, gold = parsed[path]
                    shown = _kept_lines(result)
                    g = [c for c in chunks if c.id in gold and c.kind.value != "module" and _whole(c, shown)]
                    if not g:
                        continue
                    sub = copy.deepcopy(body)
                    sub["messages"] = messages[: i + 2]
                    entry = {"task": task_dir.name, "file": path, "gold": len(g),
                             "tokens": count_tokens(result), "ranged": "offset" in call["input"],
                             "model_words": len(said.split())}
                    for variant, words in (("today", ""), ("+model_words", said),
                                           ("+right", ", ".join(c.id.split("::", 1)[1] for c in g))):
                        extra["query"] = words
                        new, _ = trim_request(sub, anthropic.parse(sub), TrimConfig())
                        sent = _text(new["messages"][i + 1]["content"][k]["content"]) if new else result
                        entry[variant] = sum(_whole(c, _kept_lines(sent)) for c in g)
                    out.append(entry)
    finally:
        trimmer.trim_file_listing = original
    return out


def summary(off: list[dict] | None, sd: list[dict] | None) -> str:
    lines = ["# Prompt hints and gateway code trimming", "",
             ("Generated by `python -m eval.prompt_hint_eval`. Gold kept = every line of a function "
              "the real fix edits survives the moderate-tier trim. \"+right\" names come from the "
              "real fix (upper bound)."), ""]
    if off:
        big = [r for r in off if r["tokens"] >= MIN_TOKENS]
        lines += [(f"## Offline: 81 held-out tasks, {len({(r['task'], r['file']) for r in big})} gold "
                   f"files of >= {MIN_TOKENS:,} tokens"), "",
                  "| Prompt | Gold kept | Files with all gold kept | Tokens sent |", "|---|---|---|---|"]
        for v in ("vague", "vague+right", "vague+wrong", "issue", "issue+right", "issue+file"):
            rs = [r for r in big if r["variant"] == v]
            g, k = sum(r["gold"] for r in rs), sum(r["gold_kept"] for r in rs)
            whole = sum(r["gold_kept"] == r["gold"] for r in rs)
            sent = sum(r["tokens_sent"] for r in rs) / sum(r["tokens"] for r in rs)
            lines.append(f"| {v} | {k} / {g} ({k / g:.0%}) | {whole} / {len(rs)} | {sent:.0%} |")
        lines.append("")
    if sd:
        trimmable = [r for r in sd if r["tokens"] >= MIN_TOKENS]
        lines += ["## Step D (real Claude Code sessions, arm A, 29 sympy bugs)", "",
                  (f"- Reads of gold files with gold code shown: {len(sd)}; read as a line range: "
                   f"{sum(r['ranged'] for r in sd)}; >= {MIN_TOKENS:,} tokens (trimmable): {len(trimmable)}"),
                  f"- Reads with no model text before the call: {sum(r['model_words'] == 0 for r in sd)} / {len(sd)}",
                  "", "| Query | Gold kept (trimmable reads) |", "|---|---|"]
        g = sum(r["gold"] for r in trimmable)
        for v in ("today", "+model_words", "+right"):
            lines.append(f"| {v} | {sum(r[v] for r in trimmable)} / {g} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.prompt_hint_eval")
    parser.add_argument("check", choices=["offline", "stepd", "all"])
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.check in ("offline", "all"):
        (OUT / "offline.json").write_text(json.dumps(offline(), indent=1))
    if args.check in ("stepd", "all"):
        (OUT / "stepd.json").write_text(json.dumps(stepd(), indent=1))
    def load(name: str) -> list[dict] | None:
        return json.loads((OUT / name).read_text()) if (OUT / name).exists() else None

    (OUT / "summary.md").write_text(summary(load("offline.json"), load("stepd.json")))
    print((OUT / "summary.md").read_text(), file=sys.stderr)


if __name__ == "__main__":
    main()
