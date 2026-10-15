# Mini-SWE-Agent replay through TokenCut (offline)

Settings: `trim`, `moderate`, threshold 500 proxy tokens, R8 on, cache events on (defaults), R10 off (one task per session), no tool pruning. Outputs rewritten as `bash` tool results (see `eval/mini_swe_replay.py`). Proxy tokens (o200k).

| Run | Sessions | Requests | Tokens sent | Tool output share | Saved (all requests) | Saved, as shipped (text protocol) | Median session saving | Sessions with any trim | Cache events |
|---|---|---|---|---|---|---|---|---|---|
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | 75 | 4,374 | 67,655,110 | 57.4 % | **16.3 %** (11,021,077) | 0.0 % | 17.7 % | 74 / 75 | 0 |
| 20251201_mini-v1.17.1_glm-4.6 | 75 | 3,874 | 69,880,243 | 45.3 % | **9.9 %** (6,931,220) | 0.0 % | 12.9 % | 63 / 75 | 2 |

## Where the saving comes from (tokens removed at each output's first trim)

| Run | Rule | Outputs | Tokens removed (first appearance) |
|---|---|---|---|
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | command_output | 413 | 186,176 |
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | code | 55 | 69,054 |
| 20251201_mini-v1.17.1_glm-4.6 | command_output | 207 | 87,156 |
| 20251201_mini-v1.17.1_glm-4.6 | code | 46 | 74,370 |

## Gateway time per request (ms, this laptop)

| Run | Request size (proxy tokens) | Requests | trim median | trim p95 | whole analysis median | p95 |
|---|---|---|---|---|---|---|
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | 0k-10k | 1467 | 17.3 | 119.7 | 26.2 | 131.5 |
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | 10k-25k | 2242 | 71.0 | 198.9 | 91.6 | 220.9 |
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | 25k-50k | 635 | 75.7 | 239.8 | 121.8 | 282.7 |
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | 50k-100k | 30 | 72.8 | 125.5 | 142.3 | 266.4 |
| 20251201_mini-v1.17.1_glm-4.6 | 0k-10k | 1438 | 9.6 | 108.9 | 18.1 | 118.4 |
| 20251201_mini-v1.17.1_glm-4.6 | 10k-25k | 1753 | 50.4 | 213.2 | 73.6 | 236.6 |
| 20251201_mini-v1.17.1_glm-4.6 | 25k-50k | 516 | 87.5 | 214.8 | 124.4 | 246.4 |
| 20251201_mini-v1.17.1_glm-4.6 | 50k-100k | 88 | 18.1 | 39.0 | 90.2 | 143.5 |
| 20251201_mini-v1.17.1_glm-4.6 | >=100k | 79 | 22.2 | 35.8 | 130.2 | 179.5 |
