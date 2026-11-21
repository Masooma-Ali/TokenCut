"""Evaluate the trained prompt router as the gateway would run it: ONNX Runtime on CPU, one
prompt at a time, the training token cut, prior correction, one fixed threshold.

    python -m eval.prompt_router.router_onnx_eval dev     # must reproduce the Kaggle dev numbers
    python -m eval.prompt_router.router_onnx_eval test    # ONCE: the final result

Model folder (local, gitignored): cache/prompt_router/models/router/ (router.onnx,
tokenizer.json, router_config.json, dev_predictions.npz). Data: cache/prompt_router/kaggle_swe/.
Report (numbers only): eval/results/prompt_router/router_<split>.md and .json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eval.prompt_router.router_metrics import bootstrap_ci, brief_curve, evaluate, prior_correct
from tokencut.gateway.task_router import OnnxRouter  # the gateway's runtime router

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "cache" / "prompt_router" / "models" / "router"
DATA = ROOT / "cache" / "prompt_router" / "kaggle_swe"
RESULTS = ROOT / "eval" / "results" / "prompt_router"
THRESHOLD = 0.87  # chosen on dev (middle of the 0.85-0.88 range that meets all three targets)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split", choices=["dev", "test"])
    args = parser.parse_args(argv)
    router = OnnxRouter(MODEL)
    rows = [json.loads(line) for line in (DATA / f"{args.split}.jsonl").open(encoding="utf-8")]
    true = [r["label"] for r in rows]

    probs, times, n_tokens = [], [], []
    for r in rows:
        start = time.perf_counter()
        probs.append(router.probs(r["text"]))
        times.append((time.perf_counter() - start) * 1000)
        n_tokens.append(len(router.tokenizer.encode(r["text"][:8000], add_special_tokens=False).ids))
    probs = np.array(probs)
    corrected = prior_correct(probs, np.array(router.config["train_prior"]),
                              np.array(router.config["natural_prior"]))
    meta = [{**r, "n_tokens": n} for r, n in zip(rows, n_tokens)]

    out = {"split": args.split, "threshold": THRESHOLD, "n": len(rows),
           "cpu_ms_per_prompt_incl_tokenizing": {"median": float(np.median(times)),
                                                 "p95": float(np.percentile(times, 95))}}
    if args.split == "dev":  # the deployable model must reproduce what Kaggle measured
        kaggle = np.load(MODEL / "dev_predictions.npz", allow_pickle=True)
        order = {i: k for k, i in enumerate(kaggle["ids"])}
        ref = kaggle["probs"][[order[r["id"]] for r in rows]]
        out["max_abs_prob_diff_vs_kaggle"] = float(np.abs(ref - probs).max())
        out["same_top_label_as_kaggle"] = float(np.mean(ref.argmax(1) == probs.argmax(1)))

    report = evaluate(corrected, true, meta, ci=args.split == "test")
    point = brief_curve(corrected, true, [THRESHOLD])[0]
    idx = np.arange(len(rows))
    for key in ("all_briefs_right", "coverage", "no_brief_wrongly_briefed"):
        def value(sample, key=key):
            v = brief_curve(corrected[sample], [true[i] for i in sample], [THRESHOLD])[0][key]
            return v if v is not None else 0.0
        point[f"{key}_ci95"] = bootstrap_ci(idx, value, n=1000, seed=7)
    out["at_threshold"] = point
    out["report"] = report
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"router_{args.split}.json").write_text(encoding="utf-8", data=json.dumps(out, indent=1, default=float))

    def pct(v):
        return f"{100 * v:.1f}%"

    def ci(key):
        lo, hi = point[f"{key}_ci95"]
        return f"{pct(point[key])} ({pct(lo)}–{pct(hi)})"

    lines = [f"# Prompt router — {args.split} set, ONNX on CPU (as in the gateway)", "",
             (f"{len(rows):,} SWE-chat prompts from repositories never seen in training; labels: "
             "SWE-chat's own (LLM-annotated). Model: bge-small fine-tuned (Kaggle), exported to "
             f"ONNX; prior-corrected; brief when the top label is a real task and p ≥ {THRESHOLD}."),
             "", "| Value | Result (95% interval) | Target |", "|---|---|---|",
             f"| All briefs right | {ci('all_briefs_right')} | ≥ 85% |",
             f"| Coverage of real tasks | {ci('coverage')} | ≥ 40% |",
             f"| no_brief wrongly briefed | {ci('no_brief_wrongly_briefed')} | ≤ 5% |",
             f"| Briefs given | {point['briefs']:,} of {len(rows):,} prompts | |",
             (f"| CPU time per prompt (incl. tokenizing) | median "
             f"{out['cpu_ms_per_prompt_incl_tokenizing']['median']:.1f} ms, p95 "
             f"{out['cpu_ms_per_prompt_incl_tokenizing']['p95']:.1f} ms | ≤ 50 ms p95 |"), "",
             (f"Without a threshold: accuracy {pct(report['accuracy'])}, macro-F1 "
             f"{report['macro_f1']:.3f}, broad-group accuracy {pct(report['broad_group_accuracy'])}."),
             "", "| Label | Precision | Recall | F1 | n |", "|---|---|---|---|---|"]
    for lab, s in report["per_label"].items():
        lines.append(f"| {lab} | {s['precision']:.2f} | {s['recall']:.2f} | {s['f1']:.2f} | "
                     f"{s['support']} |")
    lines += ["", "| Slice | n | Accuracy |", "|---|---|---|"]
    for name, s in report["slices"].items():
        lines.append(f"| {name} | {s['n']} | {pct(s['accuracy'])} |")
    if "max_abs_prob_diff_vs_kaggle" in out:
        lines += ["", (f"Check against Kaggle's dev predictions: max probability difference "
                  f"{out['max_abs_prob_diff_vs_kaggle']:.2e}, same top label for "
                  f"{pct(out['same_top_label_as_kaggle'])} of prompts.")]
    (RESULTS / f"router_{args.split}.md").write_text(encoding="utf-8", data="\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
