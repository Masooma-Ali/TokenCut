# Prompt router — dev split

Train 442 prompts, evaluated on 139.

## Every classifier, no cascade (always a real label)

| Embeddings | Classifier | Accuracy | Macro-F1 |
|---|---|---|---|
| bge-small-en-v1.5 | prototypes | 35% | 0.42 |
| bge-small-en-v1.5 | knn5 | 24% | 0.24 |
| bge-small-en-v1.5 | softmax | 36% | 0.28 |
| bge-small-en-v1.5 | softmax_balanced | 37% | 0.39 |
| bge-small-en-v1.5 | two_stage | 37% | 0.30 |
| bge-small-en-v1.5 | two_stage_balanced | 28% | 0.28 |

## Chosen: bge-small-en-v1.5 + two_stage, min_prob 0.6, min_margin 0.0

Routed precision 100% at coverage 8% (the rest goes to `ambiguous`, message unchanged); macro-F1 0.09.

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambiguous | 0.30 | 0.93 | 0.46 | 42 |
| bugfix | 0.00 | 0.00 | 0.00 | 10 |
| design | 0.00 | 0.00 | 0.00 | 7 |
| explain | 0.57 | 0.15 | 0.24 | 26 |
| feature | 1.00 | 0.12 | 0.22 | 32 |
| refactor | 0.00 | 0.00 | 0.00 | 6 |
| review | 0.00 | 0.00 | 0.00 | 2 |
| run_action | 0.00 | 0.00 | 0.00 | 5 |
| tests | 0.00 | 0.00 | 0.00 | 2 |
| write_text | 0.00 | 0.00 | 0.00 | 7 |

Confusion matrix (rows = true, columns = predicted):

| | ambigu | bugfix | design | explai | featur | refact | review | run_ac | tests | write_ |
|---|---|---|---|---|---|---|---|---|---|---|
| ambiguous | 39 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| bugfix | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| design | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| explain | 22 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| feature | 28 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| refactor | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| review | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| run_action | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tests | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| write_text | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
