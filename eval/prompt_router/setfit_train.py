"""Fine-tune bge-small on the router labels with SetFit (contrastive pairs + a classifier head).

Needs torch + setfit, which are NOT TokenCut dependencies; run it with an environment that has
them, e.g. (memory capped so it cannot freeze the laptop):

    systemd-run --user --scope -p MemoryMax=6G env \
      PYTHONPATH=.:$HOME/.tokencut/setfit-libs <python-with-torch> -m eval.prompt_router.setfit_train dev

Same data as eval.prompt_router.run: train = the random-700 train split + the keyword-found
extras (training only); evaluated on the dev split (or test, once). `ambiguous` is never a
trained class: it is what the router answers when not confident. Reported: accuracy and
macro-F1 on prompts with a real label, broad-group accuracy, and precision vs coverage.
Model saved to cache/prompt_router/setfit-<split>/ (local, gitignored).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eval.prompt_router.data import OUT_DIR
from eval.prompt_router.label import LABELS_FILE, load_jsonl
from eval.prompt_router.run import (
    EXTRA_FILE,
    RESULTS,
    SPLITS,
    final_labels,
    macro_f1,
    per_class,
)
from tokencut.gateway.prompt_router import AMBIGUOUS, MAX_CHARS, STAGE1

BASE = "BAAI/bge-small-en-v1.5"


def main(argv: list[str] | None = None) -> None:
    from datasets import Dataset  # torch-side dependencies, imported only here
    from setfit import SetFitModel, Trainer, TrainingArguments

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split", choices=["dev", "test"])
    parser.add_argument("--iterations", type=int, default=10, help="contrastive pairs per example")
    parser.add_argument("--max-length", type=int, default=256, help="tokens per prompt")
    args = parser.parse_args(argv)

    labels = final_labels(load_jsonl(LABELS_FILE))
    extra = final_labels(load_jsonl(EXTRA_FILE))
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    texts = {p["id"]: p["text"][:MAX_CHARS] for name in ("prompts.jsonl", "prompts_extra.jsonl")
             for p in load_jsonl(OUT_DIR / name)}
    train = [i for i in splits["train"] if labels[i] != AMBIGUOUS]
    train += sorted(i for i, lab in extra.items() if lab != AMBIGUOUS and i not in labels)
    all_labels = {**extra, **labels}
    evaluate = [i for i in splits[args.split] if labels[i] != AMBIGUOUS]
    unclear = [i for i in splits[args.split] if labels[i] == AMBIGUOUS]  # should get NO brief

    names = sorted({all_labels[i] for i in train})
    data = Dataset.from_dict({"text": [texts[i] for i in train],
                              "label": [names.index(all_labels[i]) for i in train]})
    model = SetFitModel.from_pretrained(BASE, labels=names,
                                        head_params={"class_weight": "balanced", "max_iter": 2000})
    model.model_body.max_seq_length = args.max_length
    trainer = Trainer(model=model, train_dataset=data, args=TrainingArguments(
        batch_size=16, num_epochs=1, num_iterations=args.iterations, seed=42, report_to="none"))
    start = time.perf_counter()
    trainer.train()
    train_minutes = (time.perf_counter() - start) / 60

    start = time.perf_counter()
    probs = np.asarray(model.predict_proba([texts[i] for i in evaluate]))
    ms_per_prompt = 1000 * (time.perf_counter() - start) / len(evaluate)
    pred = [names[k] for k in probs.argmax(axis=1)]
    true = [all_labels[i] for i in evaluate]
    stats = per_class(true, pred, sorted(set(true)))
    top = probs.max(axis=1)
    curve = []
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        keep = top >= threshold
        if keep.any():
            curve.append({"min_prob": threshold, "coverage": float(keep.mean()),
                          "precision": float(np.mean([p == t for p, t, k in zip(pred, true, keep) if k])),
                          "broad_precision": float(np.mean([STAGE1[p] == STAGE1[t]
                                                            for p, t, k in zip(pred, true, keep) if k]))})
    # Prompts labelled ambiguous should stay unchanged: how many would wrongly get a brief?
    top_unclear = np.asarray(model.predict_proba([texts[i] for i in unclear])).max(axis=1)
    wrongly_briefed = {str(th): float((top_unclear >= th).mean())
                       for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}
    report = {"split": args.split, "base": BASE, "n_ambiguous": len(unclear),
              "ambiguous_wrongly_briefed": wrongly_briefed, "n_train": len(train), "n_eval": len(evaluate),
              "iterations": args.iterations, "max_length": args.max_length,
              "train_minutes": round(train_minutes, 1), "ms_per_prompt": round(ms_per_prompt, 1),
              "accuracy": float(np.mean([p == t for p, t in zip(pred, true)])),
              "broad_accuracy": float(np.mean([STAGE1[p] == STAGE1[t] for p, t in zip(pred, true)])),
              "macro_f1": macro_f1(stats), "per_class": stats, "coverage_curve": curve}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"setfit_{args.split}.json").write_text(encoding="utf-8", data=json.dumps(report, indent=1))
    model.save_pretrained(str(Path(OUT_DIR) / f"setfit-{args.split}"))
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, indent=1))


if __name__ == "__main__":
    main()
