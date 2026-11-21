"""Fine-tune bge-small as the TokenCut prompt router on SWE-chat (Kaggle GPU).

Upload together (one Kaggle dataset): train.jsonl, dev.jsonl, test.jsonl
(cache/prompt_router/kaggle_swe/), this file and router_metrics.py. In a notebook cell:

    !pip install -q "optimum[onnxruntime]"
    !python /kaggle/input/<dataset>/train_router_kaggle.py --limit 400 --epochs 1   # smoke test
    !python /kaggle/input/<dataset>/train_router_kaggle.py              # train, choose on dev
    !python /kaggle/input/<dataset>/train_router_kaggle.py --test       # ONCE, at the very end

Model: BAAI/bge-small-en-v1.5 (33M parameters, MIT) + a 7-label classification head, all
weights fine-tuned (AdamW, linear warm-up / decay, class weights, mixed precision). Each prompt
is cut in TOKENS: first 360 + "…" + last 149 (instruction + final error line), so nothing is
silently lost at the 512-token limit. The best epoch on dev macro-F1 is kept.

Outputs in /kaggle/working/router/: the PyTorch model, router.onnx (+ a check that ONNX and
PyTorch agree), tokenizer, router_config.json (labels, cut sizes) and report_<split>.json with
every value of router_metrics.evaluate (dev; test only with --test), raw and corrected for the
natural label mix of the training repositories (natural_prior.json), and dev_predictions.npz
(probabilities) for analysis without retraining. ONNX via Hugging Face Optimum.
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from router_metrics import (
    HEAD_TOKENS,
    LABELS,
    NO_BRIEF,
    TAIL_TOKENS,
    cut_token_ids,
    evaluate,
    prior_correct,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "BAAI/bge-small-en-v1.5"
OUT = "/kaggle/working/router"


def save_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def load(split: str) -> list[dict]:
    path = glob.glob(f"/kaggle/input/**/{split}.jsonl", recursive=True)[0]
    with open(path) as f:
        return [json.loads(line) for line in f.read().split("\n") if line.strip()]


def priors(train_labels: list[int]) -> tuple[np.ndarray, np.ndarray | None]:
    """(label mix the model was trained on, natural mix of the training repositories)."""
    train = np.bincount(train_labels, minlength=len(LABELS)) / len(train_labels)
    found = glob.glob("/kaggle/input/**/natural_prior.json", recursive=True)
    if not found:
        return train, None
    with open(found[0]) as f:
        counts = json.load(f)
    natural = np.array([counts.get(lab, 0) for lab in LABELS], dtype=float)
    return train, natural / natural.sum()


def encode(tokenizer, texts: list[str]) -> tuple[list[list[int]], list[int]]:
    """Token ids with [CLS] … [SEP], head + tail cut; also the uncut length for the report."""
    marker = tokenizer.convert_tokens_to_ids(["…"]) if "…" in tokenizer.get_vocab() else \
        tokenizer("...", add_special_tokens=False)["input_ids"]
    out, lengths = [], []
    for text in texts:
        ids = tokenizer(text[:6000] + ("\n" + text[-2000:] if len(text) > 8000 else ""),
                        add_special_tokens=False, truncation=False)["input_ids"]
        lengths.append(len(ids))
        ids = cut_token_ids(ids, HEAD_TOKENS, TAIL_TOKENS, marker)
        out.append([tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id])
    return out, lengths


def batches(ids: list[list[int]], labels: list[int] | None, size: int, shuffle: bool, seed: int,
            pad: int):
    order = list(range(len(ids)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), size):
        chunk = order[start:start + size]
        width = max(len(ids[i]) for i in chunk)
        input_ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, i in enumerate(chunk):
            input_ids[row, :len(ids[i])] = torch.tensor(ids[i])
            mask[row, :len(ids[i])] = 1
        yield chunk, input_ids, mask, (torch.tensor([labels[i] for i in chunk]) if labels else None)


@torch.no_grad()
def predict(model, ids, device, pad, size=64) -> np.ndarray:
    model.eval()
    out = np.zeros((len(ids), len(LABELS)), dtype=np.float32)
    for chunk, input_ids, mask, _ in batches(ids, None, size, False, 0, pad):
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
        out[chunk] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return out


def summary(report: dict) -> str:
    lines = [(f"accuracy {report['accuracy']:.3f}  macro-F1 {report['macro_f1']:.3f}  "
             f"broad-group {report['broad_group_accuracy']:.3f}  (n={report['n']})"),
             "threshold  briefs  all_briefs_right  coverage  no_brief_wrongly_briefed"]
    for p in report["brief_curve"]:
        right = "-" if p["all_briefs_right"] is None else f"{p['all_briefs_right']:.3f}"
        ci = p.get("all_briefs_right_ci95")
        ci = f" ({ci[0]:.3f}-{ci[1]:.3f})" if ci else ""
        lines.append(f"  {p['min_prob']:.1f}     {p['briefs']:5d}   {right}{ci:15}  "
                     f"{p['coverage']:.3f}     {p['no_brief_wrongly_briefed']:.3f}")
    lines.append("label        precision recall  f1    n")
    for lab, s in report["per_label"].items():
        lines.append(f"  {lab:11} {s['precision']:.2f}     {s['recall']:.2f}   {s['f1']:.2f}  {s['support']}")
    lines.append("slices (accuracy; at 0.7: all_briefs_right / coverage / no_brief wrongly briefed)")
    for name, s in report["slices"].items():
        c = s["curve_at_0.7"]
        fmt = lambda v: "-" if v is None else f"{v:.2f}"
        lines.append(f"  {name:28} n={s['n']:5d} acc {s['accuracy']:.2f}  "
                     f"{fmt(c['all_briefs_right'])} / {fmt(c['coverage'])} / "
                     f"{fmt(c['no_brief_wrongly_briefed'])}")
    return "\n".join(lines)


def export_onnx(model, tokenizer, device) -> dict:
    """router ONNX for the gateway (ONNX Runtime, CPU) via Hugging Face Optimum, checked against
    PyTorch on single prompts (how the gateway classifies) and on a padded batch.

    The first attempt used torch.onnx.export directly; it froze the attention's padding / position
    logic as constants and differed from PyTorch by up to 0.63 in the logits."""
    from optimum.onnxruntime import ORTModelForSequenceClassification

    onnx_dir = f"{OUT}/onnx"
    ort_model = ORTModelForSequenceClassification.from_pretrained(OUT, export=True)
    ort_model.save_pretrained(onnx_dir)
    tokenizer.save_pretrained(onnx_dir)
    cpu_model = AutoModelForSequenceClassification.from_pretrained(OUT).eval().to("cpu")
    texts = ["fix the failing login test", "commit and push", "what does parse_listing do?",
             "Traceback (most recent call last):\n  File \"app.py\", line 3\nKeyError: 'x'"]
    single = 0.0
    with torch.no_grad():
        for text in texts:
            one = tokenizer([text], return_tensors="pt")
            a = ort_model(**one).logits.detach().numpy()
            b = cpu_model(**one).logits.numpy()
            single = max(single, float(np.abs(a - b).max()))
        batch = tokenizer(texts, padding=True, return_tensors="pt")
        padded = float(np.abs(ort_model(**batch).logits.detach().numpy()
                              - cpu_model(**batch).logits.numpy()).max())
    one = tokenizer(["the login test fails after my change, fix it"], return_tensors="pt")
    start = time.perf_counter()
    for _ in range(50):
        ort_model(**one)
    files = glob.glob(f"{onnx_dir}/*.onnx")
    return {"onnx_dir": onnx_dir,
            "mb": round(sum(os.path.getsize(f) for f in files) / 1e6, 1),
            "max_abs_diff_vs_torch_single_prompts": single,
            "max_abs_diff_vs_torch_padded_batch": padded,
            "cpu_ms_per_prompt": round((time.perf_counter() - start) * 1000 / 50, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="evaluate the saved model on TEST (once)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke test: use only this many train / dev prompts")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)

    if args.test:  # final, once: the model chosen on dev, never retrained here
        tokenizer = AutoTokenizer.from_pretrained(OUT)
        model = AutoModelForSequenceClassification.from_pretrained(OUT).to(device)
        rows = load("test")
        ids, lengths = encode(tokenizer, [r["text"] for r in rows])
        probs = predict(model, ids, device, tokenizer.pad_token_id)
        meta = [{**r, "n_tokens": n} for r, n in zip(rows, lengths)]
        true = [r["label"] for r in rows]
        with open(f"{OUT}/router_config.json") as f:
            config = json.load(f)
        reports = {"raw": evaluate(probs, true, meta, ci=True)}
        if config.get("natural_prior"):
            reports["prior_corrected"] = evaluate(
                prior_correct(probs, np.array(config["train_prior"]), np.array(config["natural_prior"])),
                true, meta, ci=True)
        save_json(reports, f"{OUT}/report_test.json")
        for name, report in reports.items():
            print(f"TEST ({name})\n" + summary(report))
        return

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, id2label=dict(enumerate(LABELS)),
        label2id={lab: i for i, lab in enumerate(LABELS)}).to(device)
    train, dev = load("train"), load("dev")
    if args.limit:  # quick end-to-end check before the real run
        train, dev = train[:args.limit], dev[:max(50, args.limit // 4)]
    train_ids, _ = encode(tokenizer, [r["text"] for r in train])
    dev_ids, dev_len = encode(tokenizer, [r["text"] for r in dev])
    y = [LABELS.index(r["label"]) for r in train]
    train_prior, natural_prior = priors(y)
    counts = np.bincount(y, minlength=len(LABELS))
    weights = torch.tensor(len(y) / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    print(f"train {len(train)}  dev {len(dev)}  device {device}  "
          f"class weights {dict(zip(LABELS, weights.round(decimals=2).tolist()))}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / args.batch)
    warmup = int(0.06 * steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / max(1, warmup), max(0.0, (steps - s) / max(1, steps - warmup))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best, step, start = -1.0, 0, time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for n, (_, input_ids, mask, labels) in enumerate(
                batches(train_ids, y, args.batch, True, args.seed + epoch, tokenizer.pad_token_id), 1):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
            loss = loss_fn(logits.float(), labels.to(device))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            running += loss.item()
            if n % 200 == 0:
                print(f"epoch {epoch} step {n} loss {running / 200:.4f} "
                      f"({(time.perf_counter() - start) / 60:.1f} min)", flush=True)
                running = 0.0
        probs = predict(model, dev_ids, device, tokenizer.pad_token_id)
        meta = [{**r, "n_tokens": n} for r, n in zip(dev, dev_len)]
        true = [r["label"] for r in dev]
        report = evaluate(probs, true, meta)
        print(f"\n=== epoch {epoch} on DEV ===\n" + summary(report), flush=True)
        corrected = None
        if natural_prior is not None:
            corrected = evaluate(prior_correct(probs, train_prior, natural_prior), true, meta)
            print(f"\n--- epoch {epoch} on DEV, corrected for the natural label mix ---\n"
                  + summary(corrected), flush=True)
        if report["macro_f1"] > best:  # keep the best epoch (threshold-free choice)
            best = report["macro_f1"]
            model.save_pretrained(OUT)
            tokenizer.save_pretrained(OUT)
            save_json({"epoch": epoch, "raw": report, "prior_corrected": corrected},
                      f"{OUT}/report_dev.json")
            np.savez_compressed(f"{OUT}/dev_predictions.npz", probs=probs,
                                labels=np.array(true), ids=np.array([r["id"] for r in dev]),
                                train_prior=train_prior,
                                natural_prior=natural_prior if natural_prior is not None else [])

    model = AutoModelForSequenceClassification.from_pretrained(OUT).to(device)
    config = {"base": BASE, "labels": LABELS, "no_brief": NO_BRIEF, "head_tokens": HEAD_TOKENS,
              "tail_tokens": TAIL_TOKENS, "best_dev_macro_f1": best, "epochs": args.epochs,
              "lr": args.lr, "batch": args.batch, "seed": args.seed,
              "train_prior": train_prior.tolist(),
              "natural_prior": natural_prior.tolist() if natural_prior is not None else None,
              "train_minutes": round((time.perf_counter() - start) / 60, 1),
              "export": export_onnx(model, tokenizer, device)}
    save_json(config, f"{OUT}/router_config.json")
    print("\n" + json.dumps(config, indent=1))
    print(f"\nKept epoch with best dev macro-F1 {best:.3f}. Download /kaggle/working/router "
          "(zip it) BEFORE stopping the session.")


if __name__ == "__main__":
    main()
