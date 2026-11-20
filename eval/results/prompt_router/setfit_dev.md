# Prompt router — SetFit fine-tuning, dev split (2026-11-20)

Run on Kaggle (2× T4), `cache/prompt_router/kaggle/train_setfit_kaggle.py dev` (same setup as
`eval/prompt_router/setfit_train.py`): base `BAAI/bge-small-en-v1.5`, 442 training prompts
(random-700 train split + keyword-found extras, `ambiguous` never trained), 20 contrastive
pairs per prompt (17,680 pairs), 1 epoch, max 512 tokens, balanced logistic-regression head,
seed 42; training 11.7 min. Labels by Claude (Opus 5.5). Evaluated on the 97 dev prompts with a
real label (the 42 `ambiguous` dev prompts were not scored in this run).

| | Frozen bge-small + head (best, words + embeddings) | SetFit fine-tuned |
|---|---|---|
| Exact label accuracy | 0.58 | **0.68** |
| Broad-group accuracy | 0.64 | **0.74** |
| Macro-F1 | 0.51 | 0.53 |

Confidence cascade (prompts below the threshold get no brief):

| min_prob | Coverage | Precision (exact) | Precision (broad group) |
|---|---|---|---|
| 0.3 | 95% | 71% | 76% |
| 0.4 | 83% | 74% | 80% |
| 0.5 | 68% | 77% | 85% |
| 0.6 | 53% | 82% | 90% |
| 0.7 | 34% | 94% | 97% |
| 0.8 | 13% | 100% | 100% |

Per label (precision / recall / n): feature 0.90 / 0.84 / 32 · explain 0.84 / 0.62 / 26 ·
bugfix 0.58 / 0.70 / 10 · design 0.50 / 0.86 / 7 · run_action 0.57 / 0.80 / 5 ·
write_text 0.50 / 0.43 / 7 · refactor 0.40 / 0.33 / 6 · tests 0.33 / 0.50 / 2 · review 0.00 / 0.00 / 2.

Limits: 97 prompts (±~9 points); thresholds read on dev; small labels are noise; not yet
measured how many `ambiguous` prompts would wrongly get a brief (added to the script for the
next run); the test split is still untouched.
