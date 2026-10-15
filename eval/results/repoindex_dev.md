# Repo index search — dev set

## dev: 39 tasks

| method | file@1 | file@3 | file@5 | chunk@5 | chunk@10 | chunk@20 |
|---|---|---|---|---|---|---|
| bm25_files | 0.23 | 0.64 | 0.67 | – | – | – |
| bm25 | 0.26 | 0.56 | 0.62 | 0.30 | 0.36 | 0.51 |
| tokencut_v1 | 0.31 | 0.49 | 0.67 | 0.27 | 0.34 | 0.45 |
| index 0.2,0.4,0,0.7 | 0.33 | 0.67 | 0.82 | 0.35 | 0.48 | 0.59 |
| index 0.4,0.4,0.2,1 | 0.36 | 0.64 | 0.74 | 0.32 | 0.39 | 0.46 |

Best index weights on this set (mean of file@5 + chunk@10): `index 0.2,0.4,0,0.7`

- index 0.2,0.4,0,0.7 − bm25_files on file@5: mean +0.154 (95% CI +0.026 … +0.282), better in 7, worse in 1, sign test p = 0.070
- index 0.2,0.4,0,0.7 vs bm25_files on chunk@10: not comparable
- index 0.2,0.4,0,0.7 − bm25 on file@5: mean +0.205 (95% CI +0.077 … +0.333), better in 8, worse in 0, sign test p = 0.008
- index 0.2,0.4,0,0.7 − bm25 on chunk@10: mean +0.113 (95% CI +0.032 … +0.212), better in 6, worse in 0, sign test p = 0.031
- index 0.2,0.4,0,0.7 − tokencut_v1 on file@5: mean +0.154 (95% CI +0.026 … +0.282), better in 7, worse in 1, sign test p = 0.070
- index 0.2,0.4,0,0.7 − tokencut_v1 on chunk@10: mean +0.135 (95% CI +0.038 … +0.250), better in 8, worse in 1, sign test p = 0.039

Index build + graph + one search per task: median 1.3 s, max 9.0 s (from cached chunks; a running server only searches).
