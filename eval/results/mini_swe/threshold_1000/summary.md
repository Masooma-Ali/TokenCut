# Mini-SWE-Agent replay through TokenCut (offline)

Settings: `trim`, `moderate`, threshold 1,000 proxy tokens, R8 on, cache events on (defaults), R10 off (one task per session), no tool pruning. Outputs rewritten as `bash` tool results (see `eval/mini_swe_replay.py`). Proxy tokens (o200k).

| Run | Sessions | Requests | Tokens sent | Tool output share | Saved (all requests) | Saved, as shipped (text protocol) | Median session saving | Sessions with any trim | Cache events |
|---|---|---|---|---|---|---|---|---|---|
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | 75 | 4,374 | 67,655,110 | 57.4 % | **11.3 %** (7,635,186) | 0.0 % | 10.9 % | 63 / 75 | 0 |
| 20251201_mini-v1.17.1_glm-4.6 | 75 | 3,874 | 69,880,243 | 45.3 % | **7.9 %** (5,545,255) | 0.0 % | 7.0 % | 46 / 75 | 2 |

## Where the saving comes from (tokens removed at each output's first trim)

| Run | Rule | Outputs | Tokens removed (first appearance) |
|---|---|---|---|
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | command_output | 99 | 113,214 |
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | code | 46 | 67,887 |
| 20251201_mini-v1.17.1_glm-4.6 | code | 41 | 74,174 |
| 20251201_mini-v1.17.1_glm-4.6 | command_output | 50 | 55,386 |
| 20251201_mini-v1.17.1_glm-4.6 | history_old_output | 1 | 895 |

## Gateway time per request (ms, this laptop)

| Run | Request size (proxy tokens) | Requests | trim median | trim p95 | whole analysis median | p95 |
|---|---|---|---|---|---|---|
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | 0k-10k | 1467 | 5.4 | 103.5 | 13.5 | 117.5 |
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | 10k-25k | 2242 | 37.0 | 173.0 | 67.7 | 205.8 |
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | 25k-50k | 635 | 39.4 | 253.5 | 89.8 | 313.5 |
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | 50k-100k | 30 | 82.2 | 146.1 | 168.3 | 316.8 |
| 20251201_mini-v1.17.1_glm-4.6 | 0k-10k | 1438 | 1.6 | 105.4 | 11.3 | 114.1 |
| 20251201_mini-v1.17.1_glm-4.6 | 10k-25k | 1753 | 25.7 | 200.9 | 49.1 | 222.8 |
| 20251201_mini-v1.17.1_glm-4.6 | 25k-50k | 516 | 40.9 | 174.0 | 91.0 | 209.9 |
| 20251201_mini-v1.17.1_glm-4.6 | 50k-100k | 88 | 8.0 | 28.1 | 78.9 | 132.9 |
| 20251201_mini-v1.17.1_glm-4.6 | >=100k | 79 | 16.1 | 24.3 | 110.3 | 149.6 |
