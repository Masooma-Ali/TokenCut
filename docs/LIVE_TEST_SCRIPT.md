# Live test script — step C

Two repos: **Part 2 (sympy, larger real repo)** is the main one; Part 1 (tiny playground)
was used for the C1 baseline. Each part uses its own fixed task list.

---

# Part 3 — long session on sympy: L1 (moderate) and L2 (shadow)

Goal: a realistic long session (60–100+ requests) with three real SWE-bench bugs and large
test outputs, so history grows, the context-full event can trigger and on-arrival trimming
gets large tool outputs. Same repo copy as Part 2 (`C:\Users\pc\tokencut-sympy`).

Bugs (each verified on this snapshot: the official hidden test fails before the official
fix and passes after it): `sympy__sympy-13480` (coth NameError), `sympy__sympy-13647`
(`Matrix.col_insert` shifts the matrix), `sympy__sympy-13773` (`@` must reject scalars).
Known before any change: `sympy/core/tests` has 105 failures, `test_matrices.py` 5 failures
(old code on Python 3.11 / newer mpmath) — not caused by the bugs.

Threshold: default **2,000 tokens** in both runs (1,000 / 500 are tested offline by replaying
the L2 captures). Run L1 first, then L2 (or the other way round — keep the same tasks).

## Before each run

**Terminal 1 — gateway** (new PowerShell window)
```powershell
cd D:\TokenCut
.venv\Scripts\Activate.ps1
$env:TOKENCUT_LOG_DIR = "logs\gateway\L1"        # L1 = moderate, L2 = shadow
$env:TOKENCUT_MODE = "trim"                      # L2: "shadow"
$env:TOKENCUT_TIER = "moderate"
$env:TOKENCUT_CAPTURE = "1"
python -m tokencut.gateway
```
Tool pruning is **off** in L1/L2 (team member's choice): savings come only from tool-output
trimming and history trimming at cache events. Pruning can still be estimated offline from
the L2 captures.

**Terminal 2 — reset the repo, start Claude Code** (new PowerShell window)
```powershell
cd C:\Users\pc\tokencut-sympy
git reset --hard start; git clean -fd
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8787/anthropic"
claude --model sonnet
```
No model switch in these runs: the long session itself should trigger the context-full event.
**L1 and L2 use Haiku 5.5** (`claude --model haiku`, team member's choice, saves Pro quota).
L1 vs L2 are comparable with each other; not directly with the Sonnet S-runs.

## The tasks (paste one at a time, wait for the full answer)

1. Use C:/Users/pc/tokencut-sympy/.venv/Scripts/python.exe for every Python or pytest command in this session. First, give me a short overview of how this repo is organised: the top-level folders and where the core classes live.
2. Run the whole core test suite with: C:/Users/pc/tokencut-sympy/.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider sympy/core/tests   — then tell me how many tests fail and group the failures by cause. Do not fix anything yet.
3. Fix this bug: `coth(log(tan(x))).subs(x, 2)` raises `NameError: name 'cotm' is not defined` in sympy/functions/elementary/hyperbolic.py (in eval). It fails for many integer values: 2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 18, ...
4. Run sympy/functions/elementary/tests/test_hyperbolic.py and make sure nothing is broken. Add a regression test for the bug if there is none.
5. Explain how the Matrix classes are organised: which classes and mixins exist, in which files, and where operations like col_insert and multiplication are implemented.
6. Fix this bug: Matrix.col_insert() no longer works correctly. With M = eye(6) and V = 2*ones(6, 2), M.col_insert(3, V) puts the two columns of 2s in the right place, but the 3x3 identity block to the right of them is shifted from the bottom three rows to the top three rows.
7. Run sympy/matrices/tests/test_commonmatrix.py and sympy/matrices/tests/test_matrices.py and tell me whether anything is broken by your change. Add a regression test for col_insert.
8. Fix this bug: the @ operator (__matmul__) on matrices just copies __mul__, so `2 @ B` and `B @ 2` work with a scalar. Like NumPy, @ should only work for real matrix multiplication and fail when one argument is not a matrix.
9. Run the two matrix test files from task 7 again and add a test for the @ change.
10. Show me the full git diff of everything we changed so far and review it for mistakes.
11. Run the whole core test suite again (same command as task 2) and compare the result with task 2: did our changes break anything?
12. Summarise every change we made in this session, file by file.

Then type `/exit`, and **do not reset the repo** (the fixes are graded afterwards).

## How it is checked afterwards
| Task | Correct if |
|---|---|
| 3 | hidden test of `sympy__sympy-13480` passes |
| 6 | hidden test of `sympy__sympy-13647` passes |
| 8 | hidden test of `sympy__sympy-13773` passes |
| 2, 11 | failure counts reported correctly; no new failures caused by the changes |
| all | no agent confusion from TokenCut notes; re-reads of omitted content counted |

---

# Part 2 — larger real repo: sympy (main runs)

Repo: copy of sympy at SWE-bench task `sympy__sympy-13471` (commit 3546ac7, 2017), at
`C:\Users\pc\tokencut-sympy`, with its own `.venv` (mpmath + pytest, plus a small
`sitecustomize.py` shim in the venv so 2017 sympy runs on Python 3.11 — the repo itself is
untouched). Big files (e.g. `sympy/core/numbers.py`, 3,871 lines), so trimming triggers at the
**normal 2,000-token threshold** — do NOT set `TOKENCUT_MIN_TRIM_TOKENS` here.
Known before any change: `test_numbers.py` has 2 failures unrelated to the bug
(`test_mpmath_issues`, `test_numpy_to_float`: newer mpmath / no numpy).

## Before every run

**Terminal 1 — gateway** (open a NEW PowerShell window, so old settings are gone)
```powershell
cd D:\TokenCut
.venv\Scripts\Activate.ps1
$env:TOKENCUT_LOG_DIR = "logs\gateway\S1"        # S1, S2, ... per run (table below)
$env:TOKENCUT_MODE = "shadow"
python -m tokencut.gateway
```

**Terminal 2 — reset the repo, start Claude Code**
```powershell
cd C:\Users\pc\tokencut-sympy
git reset --hard start; git clean -fd
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8787/anthropic"
claude --model sonnet
```

| Run | Terminal 1 settings |
|---|---|
| S1 | `TOKENCUT_MODE=shadow` — baseline |
| S2 | `TOKENCUT_MODE=trim`, `TOKENCUT_TIER=moderate`, `TOKENCUT_CACHE_EVENTS=0` — on-arrival trimming only |
| S3 | `TOKENCUT_MODE=trim`, `TOKENCUT_TIER=moderate` — and type `/model haiku` before task 6 (cache event) |
| S4 | S3 + `TOKENCUT_PRUNE_TOOLS=1` |
| S5 | S2 with `TOKENCUT_TIER=aggressive` (and later `conservative`) |

## The tasks (paste one at a time, wait for the full answer)

1. Give me a short overview of how this repo is organised: the top-level folders and where the core classes live.
2. How does sympy turn a Python float or a string like "0.1" into a Float object? Find the relevant code and explain it step by step.
3. How are Float objects pickled and unpickled? Find the code and explain what data is saved.
4. Find the code that decides whether two sympy numbers are equal with == and explain how it works for a Float compared with a Rational.
5. Fix this bug: unpickling a sympy expression containing a float that was pickled in Python 2 fails in Python 3 with `ValueError: invalid literal for int() with base 16: '1L'`, raised at the line `num[1] = long(num[1], 16)`. Python 2 writes the mantissa as a hex string with a trailing 'L'.
6. Add a regression test for this bug in the right test file.
7. Run the tests with: C:/Users/pc/tokencut-sympy/.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider sympy/core/tests/test_numbers.py   and fix any failure caused by your change.
8. Also run sympy/core/tests/test_evalf.py with the same Python and show me the summary line.
9. Summarise every change we made in this session, file by file.

Then type `/exit`.

## How it is checked afterwards
| Task | Correct if |
|---|---|
| 2–4 | explanation matches the code (Float.__new__, `__getnewargs__`/`__getstate__`, `Float.__eq__`) |
| 5 | the official SWE-bench hidden test `test_Float_from_tuple` passes |
| 6–8 | a regression test exists; no NEW failures beyond the 2 known ones |
| 9 | change list matches the diff |
| all | agent did not "fix" the 2 unrelated failures by breaking things |

Tasks 2–4 deliberately do not name files: TokenCut keeps files **you name** whole (edit
safety), so naming `numbers.py` early would stop it being trimmed. Task 5 quotes the
traceback line, as a real bug report would.

---

# Part 1 — tiny playground (C1 baseline)

Same 10 tasks, same order, for every configuration (C1–C5), so runs are comparable.
Agent: Claude Code 2.1.296, model **Sonnet 5.5** (`claude --model sonnet`), Claude Pro login.
Repo: throwaway copy of `fixtures/sample_repo` at `C:\Users\pc\tokencut-playground`.

## Before every run

**Terminal 1 — gateway** (Ctrl+C the old one first: it keeps conversation memory in RAM)
```powershell
cd D:\TokenCut
.venv\Scripts\Activate.ps1
$env:TOKENCUT_LOG_DIR = "logs\gateway\C1"        # change per run: C1, C2, ...
$env:TOKENCUT_MIN_TRIM_TOKENS = "300"            # tiny repo
$env:TOKENCUT_MODE = "shadow"                    # see the table below
python -m tokencut.gateway
```

**Terminal 2 — reset the playground, start Claude Code**
```powershell
cd C:\Users\pc\tokencut-playground
git reset --hard start; git clean -fd            # back to the original files (playground only!)
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8787/anthropic"
claude --model sonnet
```

| Run | Settings in terminal 1 (on top of the ones above) |
|---|---|
| C1 | `TOKENCUT_MODE=shadow` (nothing changed, measures possible savings) |
| C2 | `TOKENCUT_MODE=trim`, `TOKENCUT_TIER=moderate`, `TOKENCUT_CACHE_EVENTS=0` |
| C3 | `TOKENCUT_MODE=trim`, `TOKENCUT_TIER=moderate` (events on); type `/model haiku` before task 6 |
| C4 | C3 + `TOKENCUT_PRUNE_TOOLS=1` |
| C5 | C3 with `TOKENCUT_TIER=aggressive`, then again with `conservative` |

## The tasks (paste one at a time, wait for the full answer)

1. Give me a short overview of this repo: what does each module do?
2. Read shop/cart.py and shop/utils/pricing.py and explain step by step how Cart.total() computes the final price.
3. Search the repo for every call to validate( and tell me which validate function each call uses.
4. Add a method remove_item(self, sku: str) -> bool to Cart that removes the first product with that sku and returns True, or False if no product matched.
5. Create tests/test_cart.py with pytest tests for add_item, remove_item and total (for a normal Customer and a PremiumCustomer).
6. Run the tests with: D:/TokenCut/.venv/Scripts/python.exe -m pytest -q   and fix anything that fails.
7. shop/utils/broken.py has a syntax error. Fix it without changing what the other functions in that file do.
8. Explain what happens in process_payment when the card number is invalid. Which functions are called, in which order?
9. Run the tests again with the same command and show me the summary line.
10. Summarise every change we made in this session, file by file.

Then type `/exit`.

## While running, note down
- anything odd: the agent re-reading a file, mentioning a `[TokenCut: … omitted]` note,
  complaining about a missing tool, errors, a "usage limit" message;
- roughly how long each task took (optional).

## How each task is checked (done afterwards from the files and the answers)
| Task | Correct if |
|---|---|
| 2 | subtotal → apply_discount(subtotal, discount_rate) → add_tax (17 %) |
| 3 | cart.add_item → cart.validate; gateway.process_payment → gateway.validate; refunds.refund → gateway.validate |
| 4 | `Cart.remove_item` exists, removes only the first match, returns bool |
| 5–6, 9 | tests exist and pass |
| 7 | `broken.py` compiles; `fine_function` / `another_fine_function` unchanged |
| 8 | validate fails → refunds.log_failure prints last 4 digits → ValueError("Invalid card") |
| 10 | lists cart.py, tests/test_cart.py, broken.py changes correctly |
