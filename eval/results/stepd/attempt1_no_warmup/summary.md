# Step D summary (2026-10-13T15:44:07+00:00)

Runs: 6; complete A/B pairs: 3. Arm A = shadow (no trimming), arm B = trim moderate + R8 + R10. Cost = Haiku 5.5 list prices incl. the 100K step and 1-hour cache writes.

| task | resolved A | resolved B | cost A ($) | cost B ($) | requests A | requests B | tokens sent A | tokens sent B | wall A (s) | wall B (s) | re-fetch after trim (B) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sympy__sympy-23117 | True | True | 0.0161 | 0.0149 | 14 | 12 | 538,381 | 457,736 | 107 | 84 | 0 |
| sympy__sympy-24152 | True | True | 0.0172 | 0.0294 | 14 | 9 | 514,635 | 324,591 | 82 | 48 | 0 |
| sympy__sympy-24909 | False | False | 0.0373 | 0.0191 | 14 | 14 | 568,568 | 549,480 | 121 | 110 | 0 |

| measure | A | B | change |
|---|---|---|---|
| resolved | 2/3 | 2/3 | +0 |
| cost ($) | 0.07064 | 0.06337 | -10.3 % |
| tokens sent | 1.622e+06 | 1.332e+06 | -17.9 % |
| requests | 42 | 35 | -16.7 % |
| wall time (s) | 310 | 242 | -21.9 % |
