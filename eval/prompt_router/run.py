"""Train and compare prompt-router classifiers on the hand-labeled DevGPT prompts.

    python -m eval.prompt_router.run dev     # choose classifier + thresholds (dev split)
    python -m eval.prompt_router.run test    # report the chosen setup once on the frozen test split

Split: stratified 60 / 20 / 20 by label, fixed seed, written once to splits.json (ids only) and
never changed, so the test split is never tuned on (project rule). Embeddings: fastembed, local.
Reported per classifier: per-class precision / recall / F1, confusion matrix, macro-F1, and
precision vs coverage of the confidence cascade (`ambiguous` when not confident).
Results: eval/results/prompt_router/<split>.md and .json.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from eval.prompt_router.data import OUT_DIR
from eval.prompt_router.label import LABELS_FILE, SECOND_FILE, load_jsonl

EXTRA_FILE = Path(__file__).parent / "labels_extra.jsonl"  # keyword-found, training only
from tokencut.gateway.prompt_router import (
    AMBIGUOUS,
    KNN,
    MAX_CHARS,
    Prototypes,
    SoftmaxHead,
    TwoStage,
    fastembed_embedder,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
SPLITS = HERE / "splits.json"
RESULTS = ROOT / "eval" / "results" / "prompt_router"
EMBED_MODELS = ("BAAI/bge-small-en-v1.5", "jinaai/jina-embeddings-v2-base-code")
CLASSIFIERS = {
    "prototypes": Prototypes,
    "knn5": lambda: KNN(k=5),
    "softmax": SoftmaxHead,
    "softmax_balanced": lambda: SoftmaxHead(class_weight="balanced"),
    "two_stage": lambda: TwoStage(SoftmaxHead),
    "two_stage_balanced": lambda: TwoStage(lambda: SoftmaxHead(class_weight="balanced")),
}
THRESHOLDS = [(p, m) for p in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8) for m in (0.0, 0.1, 0.2, 0.3)]


# ---------- labels and splits ----------

def final_labels(rows: list[dict]) -> dict[str, str]:
    """Last real label per id (an `unsure` answered later is replaced)."""
    out: dict[str, str] = {}
    for r in rows:
        if r["label"] != "unsure":
            out[r["id"]] = r["label"]
    return out


def stratified_split(labels: dict[str, str], seed: int = 21) -> dict[str, list[str]]:
    rng = random.Random(seed)
    splits: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for label in sorted(set(labels.values())):
        ids = sorted(i for i, lab in labels.items() if lab == label)
        rng.shuffle(ids)
        n_dev = n_test = round(0.2 * len(ids))
        splits["test"] += ids[:n_test]
        splits["dev"] += ids[n_test:n_test + n_dev]
        splits["train"] += ids[n_test + n_dev:]
    return {k: sorted(v) for k, v in splits.items()}


# ---------- metrics ----------

def confusion(true: list[str], pred: list[str], labels: list[str]) -> np.ndarray:
    index = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(true, pred):
        m[index[t], index[p]] += 1
    return m


def per_class(true: list[str], pred: list[str], labels: list[str]) -> dict[str, dict]:
    out = {}
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(true, pred))
        n_pred = sum(p == lab for p in pred)
        n_true = sum(t == lab for t in true)
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_true if n_true else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[lab] = {"precision": precision, "recall": recall, "f1": f1, "support": n_true}
    return out


def macro_f1(stats: dict[str, dict]) -> float:
    present = [s["f1"] for s in stats.values() if s["support"]]
    return sum(present) / len(present) if present else 0.0


def cohen_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def cascade(probs: np.ndarray, labels: list[str], min_prob: float, min_margin: float) -> list[str]:
    """Same rule as PromptRouter.classify, for many prompts at once."""
    order = np.argsort(-probs, axis=1, kind="stable")
    out = []
    for row, idx in zip(probs, order):
        top = row[idx[0]]
        second = row[idx[1]] if len(idx) > 1 else 0.0
        out.append(AMBIGUOUS if top < min_prob or top - second < min_margin else labels[idx[0]])
    return out


def routed_scores(true: list[str], pred: list[str]) -> dict:
    """Precision on prompts given a real label (not sent to `ambiguous`), and coverage."""
    real = [(t, p) for t, p in zip(true, pred) if t != AMBIGUOUS]
    routed = [(t, p) for t, p in real if p != AMBIGUOUS]
    return {"coverage": len(routed) / len(real) if real else 0.0,
            "precision": sum(t == p for t, p in routed) / len(routed) if routed else 0.0}


# ---------- run ----------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split", choices=["dev", "test"])
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--embed", choices=EMBED_MODELS, action="append",
                        help="embedding model(s) to compare (default: all)")
    args = parser.parse_args(argv)

    labels = final_labels(load_jsonl(LABELS_FILE))
    if not SPLITS.exists():  # the split covers the random 700 only
        SPLITS.write_text(json.dumps(stratified_split(labels), indent=1))
    splits = json.loads(SPLITS.read_text())
    texts = {p["id"]: p["text"][:MAX_CHARS] for name in ("prompts.jsonl", "prompts_extra.jsonl")
             for p in load_jsonl(OUT_DIR / name)}
    extra = final_labels(load_jsonl(EXTRA_FILE))
    train = [i for i in splits["train"] if i in labels and labels[i] != AMBIGUOUS]
    train += sorted(i for i, lab in extra.items() if lab != AMBIGUOUS and i not in labels)
    labels = {**extra, **labels}  # the random 700 win if an id is in both (it never is)
    evaluate = [i for i in splits[args.split] if i in labels]
    y_train = [labels[i] for i in train]
    y_eval = [labels[i] for i in evaluate]
    all_labels = sorted(set(labels.values()) | {AMBIGUOUS})

    report: dict = {"split": args.split, "n_train": len(train), "n_eval": len(evaluate), "runs": [],
                    "train_counts": {lab: y_train.count(lab) for lab in sorted(set(y_train))}}
    second = final_labels(load_jsonl(SECOND_FILE))
    both = sorted(set(second) & set(labels))
    if both:
        report["kappa"] = cohen_kappa([labels[i] for i in both], [second[i] for i in both])
        report["kappa_n"] = len(both)

    for model_name in args.embed or EMBED_MODELS:
        embed = fastembed_embedder(ROOT / "cache", model_name)
        start = time.perf_counter()
        x_train = embed([texts[i] for i in train])
        x_eval = embed([texts[i] for i in evaluate])
        ms_per_prompt = 1000 * (time.perf_counter() - start) / max(1, len(train) + len(evaluate))
        for clf_name, make in CLASSIFIERS.items():
            model = make().fit(x_train, y_train)
            probs = model.predict_proba(x_eval)
            runs = []
            for min_prob, min_margin in THRESHOLDS:
                pred = cascade(probs, model.labels, min_prob, min_margin)
                stats = per_class(y_eval, pred, all_labels)
                runs.append({"embed": model_name, "classifier": clf_name, "min_prob": min_prob,
                             "min_margin": min_margin, "macro_f1": macro_f1(stats),
                             "accuracy": float(np.mean([t == p for t, p in zip(y_eval, pred)])),
                             **routed_scores(y_eval, pred), "per_class": stats,
                             "confusion": confusion(y_eval, pred, all_labels).tolist(),
                             "embed_ms_per_prompt": ms_per_prompt})
            report["runs"] += runs

    report["labels"] = all_labels
    ok = [r for r in report["runs"] if r["precision"] >= args.target_precision]
    best = max(ok or report["runs"], key=lambda r: (r["coverage"], r["macro_f1"]))
    report["best"] = {k: best[k] for k in ("embed", "classifier", "min_prob", "min_margin",
                                            "macro_f1", "accuracy", "coverage", "precision")}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.split}.json").write_text(json.dumps(report, indent=1))
    (RESULTS / f"{args.split}.md").write_text(render(report, best))
    print(render(report, best))


def render(report: dict, best: dict) -> str:
    labels = report["labels"]
    lines = [f"# Prompt router — {report['split']} split", "",
             f"Train {report['n_train']} prompts, evaluated on {report['n_eval']}."]
    if "kappa" in report:
        lines.append(f"Labeler agreement: Cohen's kappa {report['kappa']:.2f} on {report['kappa_n']} prompts.")
    lines += ["", "## Every classifier, no cascade (always a real label)", "",
              "| Embeddings | Classifier | Accuracy | Macro-F1 |", "|---|---|---|---|"]
    for r in report["runs"]:
        if r["min_prob"] == 0 and r["min_margin"] == 0:
            lines.append(f"| {r['embed'].split('/')[-1]} | {r['classifier']} | "
                         f"{r['accuracy']:.0%} | {r['macro_f1']:.2f} |")
    lines += ["", (f"## Chosen: {best['embed'].split('/')[-1]} + {best['classifier']}, "
                   f"min_prob {best['min_prob']}, min_margin {best['min_margin']}"), "",
              (f"Routed precision {best['precision']:.0%} at coverage {best['coverage']:.0%} "
               f"(the rest goes to `ambiguous`, message unchanged); macro-F1 {best['macro_f1']:.2f}."),
              "", "| Label | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for lab in labels:
        s = best["per_class"][lab]
        lines.append(f"| {lab} | {s['precision']:.2f} | {s['recall']:.2f} | {s['f1']:.2f} | {s['support']} |")
    lines += ["", "Confusion matrix (rows = true, columns = predicted):", "",
              "| | " + " | ".join(lab[:6] for lab in labels) + " |",
              "|---" * (len(labels) + 1) + "|"]
    for lab, row in zip(labels, best["confusion"]):
        lines.append(f"| {lab} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
