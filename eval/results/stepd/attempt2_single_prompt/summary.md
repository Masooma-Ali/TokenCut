# Step D summary (2026-10-13T16:02:57+00:00)

Runs: 6; complete A/B pairs: 3. Arm A = shadow (no trimming), arm B = trim moderate + R8 + R10. Cost = Haiku 5.5 list prices incl. the 100K step and 1-hour cache writes.

| task | resolved A | resolved B | cost A ($) | cost B ($) | requests A | requests B | tokens sent A | tokens sent B | wall A (s) | wall B (s) | re-fetch after trim (B) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sympy__sympy-23117 | True | True | 0.0118 | 0.0119 | 9 | 10 | 368,096 | 379,300 | 76 | 102 | 0 |
| sympy__sympy-24152 | True | True | 0.0115 | 0.0179 | 9 | 16 | 325,334 | 622,372 | 76 | 121 | 0 |
| sympy__sympy-24909 | False | False | 0.0279 | 0.0177 | 23 | 14 | 977,188 | 549,265 | 148 | 121 | 0 |

| measure | A | B | change |
|---|---|---|---|
| resolved | 2/3 | 2/3 | +0 |
| cost ($, raw) | 0.05126 | 0.04752 | -7.3 % |
| cost ($, warm start) | 0.04653 | 0.04279 | -8.0 % |
| tokens sent | 1.671e+06 | 1.551e+06 | -7.2 % |
| requests | 41 | 40 | -2.4 % |
| wall time (s) | 300 | 344 | +14.7 % |
