# Repo index search — test set

## test: 80 tasks

| method | file@1 | file@3 | file@5 | chunk@5 | chunk@10 | chunk@20 |
|---|---|---|---|---|---|---|
| bm25_files | 0.35 | 0.54 | 0.62 | – | – | – |
| bm25 | 0.30 | 0.51 | 0.62 | 0.29 | 0.39 | 0.49 |
| tokencut_v1 | 0.47 | 0.65 | 0.68 | 0.32 | 0.38 | 0.41 |
| index 0.2,0.4,0,0.7 | 0.46 | 0.70 | 0.74 | 0.41 | 0.49 | 0.54 |
| index 0.4,0.4,0.2,1 | 0.41 | 0.65 | 0.72 | 0.34 | 0.44 | 0.55 |

Index weights reported: `index 0.2,0.4,0,0.7` (given with --weights; chosen on dev)

- index 0.2,0.4,0,0.7 − bm25_files on file@5: mean +0.113 (95% CI +0.013 … +0.212), better in 13, worse in 4, sign test p = 0.049
- index 0.2,0.4,0,0.7 vs bm25_files on chunk@10: not comparable
- index 0.2,0.4,0,0.7 − bm25 on file@5: mean +0.113 (95% CI +0.037 … +0.188), better in 10, worse in 1, sign test p = 0.012
- index 0.2,0.4,0,0.7 − bm25 on chunk@10: mean +0.100 (95% CI +0.036 … +0.175), better in 14, worse in 3, sign test p = 0.013
- index 0.2,0.4,0,0.7 − tokencut_v1 on file@5: mean +0.062 (95% CI -0.013 … +0.150), better in 8, worse in 3, sign test p = 0.227
- index 0.2,0.4,0,0.7 − tokencut_v1 on chunk@10: mean +0.110 (95% CI +0.035 … +0.188), better in 12, worse in 2, sign test p = 0.013

Index build + graph + one search per task: median 4.7 s, max 23.0 s (from cached chunks; a running server only searches).
