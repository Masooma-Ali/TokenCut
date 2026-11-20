"""Real coding-agent prompts for the router from SWE-chat (SALT-NLP, COLM 2026; ODC-BY, gated).

Needs pyarrow + huggingface_hub (not TokenCut dependencies; e.g. ~/.tokencut/data-venv) and a
Hugging Face login with access to the gated dataset.

1. Download conversations.parquet (6.4 GB, every turn of every session) once, e.g.
       HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import hf_hub_download; \\
         hf_hub_download('SALT-NLP/SWE-chat', 'conversations.parquet', repo_type='dataset', \\
         local_dir='cache/prompt_router/swechat_raw')"
2. Keep the user prompts only (with text and SWE-chat's own labels), then delete the big file:
       PYTHONPATH=. python -m eval.prompt_router.swechat extract
3. Sample prompts for labelling, at most --per-session per session (no single developer or
   repository dominates):
       PYTHONPATH=. python -m eval.prompt_router.swechat sample --n 1200

Output (local, gitignored):
  cache/prompt_router/swechat_user_prompts.jsonl  every user prompt: {id, text, session_id,
      repo_id, agent, language, turn, chars, prompt_intent, prompt_pushback}
  cache/prompt_router/swechat_prompts.jsonl       the sample for labelling
`prompt_intent` / `prompt_pushback` are SWE-chat's LLM annotations, kept to compare with ours.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "cache" / "prompt_router"
RAW = DIR / "swechat_raw" / "conversations.parquet"
ALL = DIR / "swechat_user_prompts.jsonl"
SAMPLE = DIR / "swechat_prompts.jsonl"
COLUMNS = ["turn_type", "content", "session_id", "repo_id", "agent", "language",
           "conversation_turn_number", "char_count", "prompt_intent", "prompt_pushback"]
MAX_CHARS = 4_000  # longer prompts are mostly pasted files / logs; the same cut as DevGPT


def extract() -> None:
    import pyarrow.parquet as pq  # optional dependency, imported only here

    from eval.prompt_router.data import clean, prompt_id

    pf = pq.ParquetFile(RAW)
    kept = 0
    with ALL.open("w") as out:
        for g in range(pf.metadata.num_row_groups):
            for row in pf.read_row_group(g, columns=COLUMNS).to_pylist():
                if row["turn_type"] != "user_prompt" or not (row["content"] or "").strip():
                    continue
                text = clean(row["content"])
                out.write(json.dumps({
                    "id": f"swe-{prompt_id(text + row['session_id'])}", "text": text,
                    "session_id": row["session_id"], "repo_id": row["repo_id"],
                    "agent": row["agent"], "language": row["language"],
                    "turn": row["conversation_turn_number"], "chars": row["char_count"],
                    "prompt_intent": row["prompt_intent"],
                    "prompt_pushback": row["prompt_pushback"]}, ensure_ascii=False) + "\n")
                kept += 1
            if g % 100 == 0:
                print(f"  {g}/{pf.metadata.num_row_groups} row groups, {kept} prompts", flush=True)
    print(f"{kept} user prompts -> {ALL}")


def sample(n: int, per_session: int, seed: int) -> None:
    rng = random.Random(seed)
    by_session: dict[str, list[dict]] = defaultdict(list)
    with ALL.open() as f:
        for line in f:
            row = json.loads(line)
            if len(row["text"]) <= MAX_CHARS:
                by_session[row["session_id"]].append(row)
    pool = []
    for session in sorted(by_session):
        prompts = by_session[session]
        pool += rng.sample(prompts, min(per_session, len(prompts)))
    picked = rng.sample(pool, min(n, len(pool)))
    with SAMPLE.open("w") as f:
        for row in picked:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(pool)} candidates from {len(by_session)} sessions; sampled {len(picked)} -> {SAMPLE}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["extract", "sample"])
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--per-session", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args(argv)
    if args.step == "extract":
        extract()
    else:
        sample(args.n, args.per_session, args.seed)


if __name__ == "__main__":
    main()
