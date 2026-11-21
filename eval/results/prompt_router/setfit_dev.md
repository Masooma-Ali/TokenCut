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

## Run 1b — prompts that should get no brief (same model, re-run with the new script)

Identical numbers to the run above (same data, seed 42), plus the 42 dev prompts labelled
`ambiguous` (no clear coding task): share that would still get a brief.

| min_prob | Real-label prompts briefed | … correct | `ambiguous` prompts wrongly briefed |
|---|---|---|---|
| 0.5 | 68% | 77% | 48% (20 / 42) |
| 0.6 | 53% | 82% | 29% (12 / 42) |
| **0.7** | **34%** | **94%** | **10% (4 / 42)** |
| 0.8 | 13% | 100% | 2% (1 / 42) |

All 139 dev prompts at 0.7: ~37 briefs (~31 right, ~2 wrong task type, 4 that should have had
none: ~84% of briefs right), ~102 unchanged. At 0.6: ~42 of ~63 briefs right (~67%).
Working threshold for this model: 0.7.

Limits: 139 prompts (±~9 points on the real-label share); thresholds read on dev; small labels
are noise; the test split is still untouched.

## Runs 2 and 2b — CCSBench commits and a trained "not a task" class (Kaggle, 2× T4)

`eval/prompt_router/setfit_kaggle.py` (copy of the Kaggle script). Run 2: `--with-ccsbench`
(+387 CCSBench commit messages relabelled by Claude: 129 tests, 128 write_text, 102 refactor,
20 feature, 8 bugfix; training only), 829 training prompts, 21.5 min. Run 2b: also
`--train-ambiguous` (`ambiguous` trained as a class; a brief only when the top label is a real
task and confident), 1,064 training prompts, 27.7 min. Same 139 dev prompts (97 real, 42
ambiguous). "All briefs right" = correct briefs / all briefs given, ambiguous prompts included.

| | Run 1b | Run 2 | **Run 2b** |
|---|---|---|---|
| Ambiguous wrongly briefed at 0.6 / 0.7 | 29% / 10% | 43% / 29% | **0% / 0%** |
| Real tasks briefed at 0.6 / 0.7 | 53% / 34% | 55% / 45% | 45% / 37% |
| All briefs right at 0.6 / 0.7 | 67% / 84% | ~65% / 64% | **80% / 83%** |
| Exact / broad accuracy (real tasks, no threshold) | 0.68 / 0.74 | 0.68 / 0.78 | 0.63 / 0.71 |
| Macro-F1 | 0.53 | 0.60 | 0.61 |

Run 2b per label (precision / recall / n): feature 0.90 / 0.59 / 32 · explain 0.78 / 0.54 / 26 ·
bugfix 0.70 / 0.70 / 10 · design 0.55 / 0.86 / 7 · write_text 0.83 / 0.71 / 7 ·
refactor 0.50 / 0.50 / 6 · run_action 0.33 / 0.80 / 5 · tests 0.50 / 1.00 / 2 · review 0.33 / 0.50 / 2.

Reading: CCSBench alone improved weak labels but made the model confident on non-tasks
(26–29% of ambiguous prompts briefed). Training "not a task" as a class removes that (0 of 42
at ≥ 0.6) and makes 0.6 a usable threshold (45% of real tasks briefed, 80% of briefs right).
Lower exact accuracy without threshold mostly = real tasks predicted "not a task" (no brief:
the safe error). Differences in real-task numbers are within noise (97 prompts); the ambiguous
result is the clearest gain. Test split still untouched.
