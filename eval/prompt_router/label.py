"""Hand-label prompts for the router (labels: docs/PROMPT_ROUTER_LABELS.md).

    python -m eval.prompt_router.label --who huma
    python -m eval.prompt_router.label --who fatima --second   # 2nd opinion on the agreement set

Shows one unlabeled prompt at a time (no model guess: labels stay blind). Keys: a number picks
a label, `u` = unsure (skip for now, noted), `b` = back (undo last), `q` = quit. Progress is
saved after every answer, so you can stop any time.

Labels (ids + label only, no prompt text) are committed:
  eval/prompt_router/labels.jsonl          first label of every prompt
  eval/prompt_router/labels_second.jsonl   second label of the agreement set (AGREEMENT_N)
"""

import argparse
import json
import random
import textwrap
from pathlib import Path

from eval.prompt_router.data import OUT_DIR
from tokencut.gateway.prompt_router import LABELS

HERE = Path(__file__).parent
LABELS_FILE = HERE / "labels.jsonl"
SECOND_FILE = HERE / "labels_second.jsonl"
AGREEMENT_N = 50
SHOW_CHARS = 1_500


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def agreement_ids(prompts: list[dict]) -> list[str]:
    """Fixed random subset labeled twice (Cohen's kappa)."""
    ids = sorted(p["id"] for p in prompts)
    return sorted(random.Random(7).sample(ids, min(AGREEMENT_N, len(ids))))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--who", required=True, help="your name (stored with each label)")
    parser.add_argument("--second", action="store_true", help="second labeling of the agreement set")
    args = parser.parse_args(argv)

    prompts = load_jsonl(OUT_DIR / "prompts.jsonl")
    out = SECOND_FILE if args.second else LABELS_FILE
    done = load_jsonl(out)
    if args.second:
        firsts = {r["id"]: r["labeler"] for r in load_jsonl(LABELS_FILE)}
        wanted = set(agreement_ids(prompts))
        prompts = [p for p in prompts if p["id"] in wanted and firsts.get(p["id"]) != args.who]
    labeled = {r["id"] for r in done if r["label"] != "unsure"}
    todo = [p for p in prompts if p["id"] not in labeled]
    menu = "  ".join(f"{i}={name}" for i, name in enumerate(LABELS)) + "  u=unsure b=back q=quit"
    i = 0
    while i < len(todo):
        p = todo[i]
        print("\n" + "=" * 78)
        print(f"[{len(labeled) + 1}/{len(labeled) + len(todo) - i}]  id={p['id']}  source={p['source']}")
        body = p["text"] if len(p["text"]) <= SHOW_CHARS else p["text"][:SHOW_CHARS] + " …[cut]"
        print(textwrap.indent(body, "  "))
        print(menu)
        answer = input("> ").strip().lower()
        if answer == "q":
            break
        if answer == "b" and done:
            undone = done.pop()
            labeled.discard(undone["id"])
            todo.insert(i, next(q for q in prompts if q["id"] == undone["id"]))
        elif answer == "u":
            done.append({"id": p["id"], "label": "unsure", "labeler": args.who})
            i += 1
        elif answer.isdigit() and int(answer) < len(LABELS):
            done.append({"id": p["id"], "label": LABELS[int(answer)], "labeler": args.who})
            labeled.add(p["id"])
            i += 1
        else:
            print("?")
            continue
        out.write_text("".join(json.dumps(r) + "\n" for r in done))
    print(f"saved {len(done)} labels to {out}")


if __name__ == "__main__":
    main()
