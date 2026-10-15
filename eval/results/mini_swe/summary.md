# Mini-SWE-Agent replay through TokenCut (offline)

Settings: `trim`, `moderate`, threshold 2,000 proxy tokens, R8 on, cache events on (defaults), R10 off (one task per session), no tool pruning. Outputs rewritten as `bash` tool results (see `eval/mini_swe_replay.py`). Proxy tokens (o200k).

| Run | Sessions | Requests | Tokens sent | Tool output share | Saved (all requests) | Saved, as shipped (text protocol) | Median session saving | Sessions with any trim | Cache events |
|---|---|---|---|---|---|---|---|---|---|
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | 75 | 4,374 | 67,655,110 | 57.4 % | **6.8 %** (4,608,384) | 0.0 % | 3.3 % | 40 / 75 | 0 |
| 20251201_mini-v1.17.1_glm-4.6 | 75 | 3,874 | 69,880,243 | 45.3 % | **5.7 %** (4,006,683) | 0.0 % | 0.0 % | 37 / 75 | 2 |

## Where the saving comes from (tokens removed at each output's first trim)

| Run | Rule | Outputs | Tokens removed (first appearance) |
|---|---|---|---|
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | code | 28 | 62,370 |
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | command_output | 27 | 53,763 |
| 20251201_mini-v1.17.1_glm-4.6 | code | 33 | 70,261 |
| 20251201_mini-v1.17.1_glm-4.6 | command_output | 14 | 30,675 |
| 20251201_mini-v1.17.1_glm-4.6 | history_old_output | 1 | 895 |

## Gateway time per request (ms, this laptop)

| Run | Request size (proxy tokens) | Requests | trim median | trim p95 | whole analysis median | p95 |
|---|---|---|---|---|---|---|
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | 0k-10k | 1467 | 0.9 | 106.4 | 10.0 | 117.2 |
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | 10k-25k | 2242 | 13.7 | 165.9 | 39.5 | 197.9 |
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | 25k-50k | 635 | 18.2 | 132.8 | 80.4 | 204.7 |
| 20250929_mini-v1.13.3_sonnet-4-5-20250929 | 50k-100k | 30 | 64.7 | 127.0 | 130.4 | 271.8 |
| 20251201_mini-v1.17.1_glm-4.6 | 0k-10k | 1438 | 1.2 | 86.5 | 8.9 | 96.0 |
| 20251201_mini-v1.17.1_glm-4.6 | 10k-25k | 1753 | 13.4 | 139.6 | 31.1 | 160.7 |
| 20251201_mini-v1.17.1_glm-4.6 | 25k-50k | 516 | 19.6 | 103.2 | 59.1 | 149.6 |
| 20251201_mini-v1.17.1_glm-4.6 | 50k-100k | 88 | 7.6 | 11.1 | 74.5 | 99.6 |
| 20251201_mini-v1.17.1_glm-4.6 | >=100k | 79 | 11.1 | 21.8 | 88.4 | 145.0 |
