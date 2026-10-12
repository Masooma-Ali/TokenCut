# Replay of L1: 109 requests, 82 distinct tool outputs

Session proxy tokens: 7,699,187; of which tool outputs: 1,731,043 (22.5 %).

## On-arrival trimming (request-weighted proxy tokens saved)

| threshold | method | outputs trimmed | diffs trimmed | saved | % of session | % of tool output |
|---|---|---|---|---|---|---|
| 2000 | current | 1 | 1 | 21,697 | 0.28 | 1.25 |
| 2000 | r8 | 0 | 0 | 0 | 0.0 | 0.0 |
| 2000 | lamr | 0 | 0 | 0 | 0.0 | 0.0 |
| 1000 | current | 7 | 1 | 506,385 | 6.58 | 29.25 |
| 1000 | r8 | 4 | 0 | 271,904 | 3.53 | 15.71 |
| 1000 | lamr | 4 | 0 | 266,602 | 3.46 | 15.4 |
| 500 | current | 19 | 5 | 640,387 | 8.32 | 36.99 |
| 500 | r8 | 12 | 0 | 397,205 | 5.16 | 22.95 |
| 500 | lamr | 12 | 0 | 359,979 | 4.68 | 20.8 |

## R10: mask a finished task's large outputs (input-equivalent tokens)

| min output size | baseline | with R10 | cache reads saved | extra cache rewrites | change |
|---|---|---|---|---|---|
| 300 | 1,483,560 | 1,317,244 | 198,386 | 32,069 | -11.21 % |
| 1000 | 1,483,560 | 1,383,017 | 122,883 | 22,340 | -6.78 % |
