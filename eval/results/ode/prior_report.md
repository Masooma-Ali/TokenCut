# ODE phase 5b: prior and offline check

Rows: 533 graded turns, 29 tasks, 8 arms (Opus only in the pilot).
C_ref = $0.3813 (Opus p95, 12 turns); cross-check Sonnet p95 x 2 = $0.3375 (174 turns). lambda = 0.4. Turns at or above C_ref (cost score capped at 1): 3.
Rewards: mean 0.741, min -0.400, max 0.994. Turns with context > 100k: 0 of 533.

## Leave-one-task-out (29 tasks, 86 turns where all 6 compared arms were graded; greedy choice, alpha 0)

| Policy | Turns solved | Cost (list price) | Mean reward | Arms chosen |
|---|---|---|---|---|
| ODE per turn (leave-one-task-out) | 71 / 86 | $2.78 | 0.792 | haiku-aggressive 41, haiku-standard 20, gpt-aggressive 16, gpt-standard 5, sonnet-standard 4 |
| ODE per session (leave-one-task-out) | 73 / 86 | $2.68 | 0.817 | haiku-aggressive 38, haiku-standard 36, sonnet-standard 12 |
| always haiku-standard | 66 / 86 | $1.65 | 0.747 | haiku-standard 86 |
| always haiku-aggressive | 71 / 86 | $1.48 | 0.807 | haiku-aggressive 86 |
| always gpt-standard | 69 / 86 | $8.22 | 0.704 | gpt-standard 86 |
| always gpt-aggressive | 70 / 86 | $6.42 | 0.736 | gpt-aggressive 86 |
| always sonnet-standard | 70 / 86 | $8.15 | 0.715 | sonnet-standard 86 |
| always sonnet-aggressive | 70 / 86 | $6.43 | 0.736 | sonnet-aggressive 86 |
| best per turn (oracle) | 79 / 86 | $1.49 | 0.900 | haiku-aggressive 56, haiku-standard 24, sonnet-aggressive 3, gpt-aggressive 2, sonnet-standard 1 |
| best per session (oracle) | 79 / 86 | $2.23 | 0.891 | haiku-aggressive 62, haiku-standard 12, sonnet-aggressive 6, gpt-aggressive 3, sonnet-standard 3 |
| random arm (expected) | 69.3 / 86 | $5.39 | 0.741 | - |

Per turn ignores switch costs (the data has no mid-session switches); per session keeps
the turn-1 choice. Oracles know every arm's real result: an upper bound, not a policy.

## Lambda sweep (per turn, leave-one-task-out; reward scored with that lambda)

| Policy | Turns solved | Cost (list price) | Mean reward | Arms chosen |
|---|---|---|---|---|
| lambda = 0.0 | 71 / 86 | $6.16 | 0.826 | gpt-standard 36, haiku-aggressive 17, gpt-aggressive 13, sonnet-aggressive 10, sonnet-standard 7, haiku-standard 3 |
| lambda = 0.1 | 71 / 86 | $5.07 | 0.810 | gpt-standard 34, haiku-aggressive 23, gpt-aggressive 13, haiku-standard 11, sonnet-standard 5 |
| lambda = 0.2 | 71 / 86 | $3.92 | 0.802 | haiku-aggressive 34, gpt-standard 23, gpt-aggressive 13, haiku-standard 11, sonnet-standard 5 |
| lambda = 0.4 | 71 / 86 | $2.78 | 0.792 | haiku-aggressive 41, haiku-standard 20, gpt-aggressive 16, gpt-standard 5, sonnet-standard 4 |
| lambda = 0.6 | 69 / 86 | $2.39 | 0.758 | haiku-aggressive 46, haiku-standard 23, gpt-aggressive 12, sonnet-standard 3, gpt-standard 2 |
| lambda = 0.8 | 71 / 86 | $2.14 | 0.773 | haiku-aggressive 49, haiku-standard 25, gpt-aggressive 10, sonnet-standard 2 |
| lambda = 1.0 | 71 / 86 | $1.70 | 0.774 | haiku-aggressive 55, haiku-standard 28, sonnet-standard 1, gpt-standard 1, gpt-aggressive 1 |
| lambda = 2.0 | 71 / 86 | $1.68 | 0.723 | haiku-aggressive 56, haiku-standard 28, sonnet-aggressive 1, gpt-standard 1 |

## C_ref sensitivity (per turn, leave-one-task-out)

| Policy | Turns solved | Cost (list price) | Mean reward | Arms chosen |
|---|---|---|---|---|
| C_ref (Opus p95) = $0.381 | 71 / 86 | $2.78 | 0.792 | haiku-aggressive 41, haiku-standard 20, gpt-aggressive 16, gpt-standard 5, sonnet-standard 4 |
| Sonnet p95 x 2 = $0.337 | 70 / 86 | $2.62 | 0.778 | haiku-aggressive 42, haiku-standard 23, gpt-aggressive 14, sonnet-standard 4, gpt-standard 3 |
| C_ref / 2 = $0.191 | 69 / 86 | $2.24 | 0.748 | haiku-aggressive 49, haiku-standard 23, gpt-aggressive 11, sonnet-standard 2, gpt-standard 1 |
| C_ref x 2 = $0.763 | 71 / 86 | $3.86 | 0.802 | haiku-aggressive 35, gpt-standard 22, gpt-aggressive 13, haiku-standard 11, sonnet-standard 5 |

## What the full prior picks for a typical turn of each type (alpha = 1)

| Turn type | Choice | Estimate per arm (bonus) |
|---|---|---|
| bugfix | opus-aggressive | haiku-standard 0.62 (+0.19), haiku-aggressive 0.74 (+0.19), sonnet-standard 0.56 (+0.22), sonnet-aggressive 0.48 (+0.24), opus-standard 0.54 (+0.78), opus-aggressive 0.73 (+0.78), gpt-standard 0.67 (+0.20), gpt-aggressive 0.62 (+0.20) |
| explain | opus-aggressive | haiku-standard 0.94 (+0.18), haiku-aggressive 0.95 (+0.18), sonnet-standard 0.78 (+0.18), sonnet-aggressive 0.87 (+0.18), opus-standard 0.38 (+1.26), opus-aggressive 0.74 (+1.28), gpt-standard 0.82 (+0.18), gpt-aggressive 0.85 (+0.18) |
| tests | opus-standard | haiku-standard 0.77 (+0.19), haiku-aggressive 0.76 (+0.19), sonnet-standard 0.65 (+0.27), sonnet-aggressive 0.59 (+0.33), opus-standard 0.57 (+0.85), opus-aggressive 0.35 (+0.81), gpt-standard 0.80 (+0.23), gpt-aggressive 0.82 (+0.22) |
