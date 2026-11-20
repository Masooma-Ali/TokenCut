"""TokenCut prompt router: fine-tune bge-small with SetFit on Kaggle (GPU).

Same setup as eval/prompt_router/setfit_train.py in the repo, standalone.
Usage in a Kaggle notebook cell:
    !pip install -q setfit==1.2.0
    !python /kaggle/input/<your-dataset>/train_setfit_kaggle.py dev
Then, ONCE, after choosing settings on dev:
    !python /kaggle/input/<your-dataset>/train_setfit_kaggle.py test
Outputs in /kaggle/working/: setfit_<split>.json (numbers) and setfit-<split>/ (the model).
Data: DevGPT prompts (Xiao et al., MSR 2024, CC BY 4.0), labels by Claude for TokenCut.
"""

import argparse
import glob
import json
import os
import time

os.environ.setdefault("WANDB_MODE", "disabled")  # no interactive wandb login prompt

import numpy as np
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

BASE = "BAAI/bge-small-en-v1.5"
AMBIGUOUS = "ambiguous"
STAGE1 = {"bugfix": "change_code", "feature": "change_code", "refactor": "change_code",
          "tests": "change_code", "explain": "understand_code", "review": "understand_code",
          "design": "plan", "run_action": "run_action", "write_text": "write_text",
          "followup": "followup", "ambiguous": "ambiguous"}


def load(split):
    path = glob.glob(f"/kaggle/input/**/{split}.jsonl", recursive=True)[0]
    with open(path) as f:
        return [json.loads(line) for line in f.read().split("\n") if line.strip()]


def per_class(true, pred):
    out = {}
    for lab in sorted(set(true)):
        tp = sum(t == lab and p == lab for t, p in zip(true, pred))
        n_pred, n_true = sum(p == lab for p in pred), sum(t == lab for t in true)
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_true if n_true else 0.0
        out[lab] = {"precision": prec, "recall": rec, "support": n_true,
                    "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["dev", "test"])
    parser.add_argument("--iterations", type=int, default=20, help="contrastive pairs per example")
    parser.add_argument("--max-length", type=int, default=512, help="tokens per prompt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--with-ccsbench", action="store_true",
                        help="add ccsbench.jsonl (commit messages) to TRAINING only")
    parser.add_argument("--train-ambiguous", action="store_true",
                        help="train 'ambiguous' as a class (predicted ambiguous = no brief)")
    args = parser.parse_args()
    keep = (lambda r: True) if args.train_ambiguous else (lambda r: r["label"] != AMBIGUOUS)

    train = [r for r in load("train") if keep(r)]
    if args.with_ccsbench:  # training only; dev / test stay DevGPT. Relabelled by Claude.
        train += [r for r in load("ccsbench") if keep(r)]
    rows = load(args.split)
    evaluate = [r for r in rows if r["label"] != AMBIGUOUS]
    unclear = [r for r in rows if r["label"] == AMBIGUOUS]  # should get NO brief
    names = sorted({r["label"] for r in train})
    data = Dataset.from_dict({"text": [r["text"] for r in train],
                              "label": [names.index(r["label"]) for r in train]})
    model = SetFitModel.from_pretrained(BASE, labels=names,
                                        head_params={"class_weight": "balanced", "max_iter": 2000})
    model.model_body.max_seq_length = args.max_length
    trainer = Trainer(model=model, train_dataset=data, args=TrainingArguments(
        batch_size=16, num_epochs=args.epochs, num_iterations=args.iterations, seed=42,
        report_to="none"))
    start = time.perf_counter()
    trainer.train()
    minutes = (time.perf_counter() - start) / 60

    probs = np.asarray(model.predict_proba([r["text"] for r in evaluate]))
    pred = [names[k] for k in probs.argmax(axis=1)]
    true = [r["label"] for r in evaluate]
    stats = per_class(true, pred)
    top = probs.max(axis=1)
    # Prompts labelled ambiguous should stay unchanged: how many would wrongly get a brief?
    probs_u = np.asarray(model.predict_proba([r["text"] for r in unclear]))
    pred_u = [names[k] for k in probs_u.argmax(axis=1)]
    top_u = probs_u.max(axis=1)
    # A brief is added only when the top label is a real task AND it is confident enough.
    curve, wrongly_briefed = [], {}
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        brief = np.array([p != AMBIGUOUS and c >= th for p, c in zip(pred, top)])
        brief_u = np.array([p != AMBIGUOUS and c >= th for p, c in zip(pred_u, top_u)])
        wrongly_briefed[str(th)] = round(float(brief_u.mean()), 3)
        if brief.any():
            pairs = [(p, t) for p, t, b in zip(pred, true, brief) if b]
            right = sum(p == t for p, t in pairs)
            curve.append({"min_prob": th, "coverage": round(float(brief.mean()), 3),
                          "precision": round(right / len(pairs), 3),
                          "broad_precision": round(float(np.mean([STAGE1[p] == STAGE1[t]
                                                                  for p, t in pairs])), 3),
                          "all_briefs_right": round(right / (len(pairs) + int(brief_u.sum())), 3)})
    report = {"split": args.split, "with_ccsbench": args.with_ccsbench,
              "train_ambiguous": args.train_ambiguous,
              "n_ambiguous": len(unclear),
              "ambiguous_wrongly_briefed": wrongly_briefed, "base": BASE, "n_train": len(train), "n_eval": len(evaluate),
              "iterations": args.iterations, "max_length": args.max_length, "epochs": args.epochs,
              "train_minutes": round(minutes, 1),
              "accuracy": round(float(np.mean([p == t for p, t in zip(pred, true)])), 3),
              "broad_accuracy": round(float(np.mean([STAGE1[p] == STAGE1[t]
                                                     for p, t in zip(pred, true)])), 3),
              "macro_f1": round(float(np.mean([s["f1"] for s in stats.values()])), 3),
              "per_class": stats, "coverage_curve": curve}
    os.makedirs("/kaggle/working", exist_ok=True)
    with open(f"/kaggle/working/setfit_{args.split}.json", "w") as f:
        json.dump(report, f, indent=1)
    model.save_pretrained(f"/kaggle/working/setfit-{args.split}")
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, indent=1))
    for lab, s in stats.items():
        print(f"{lab:12} precision {s['precision']:.2f}  recall {s['recall']:.2f}  n={s['support']}")


if __name__ == "__main__":
    main()
