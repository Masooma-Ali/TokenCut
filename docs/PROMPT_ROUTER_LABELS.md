# Prompt router — label guide

Used to hand-label developer prompts and to judge the router. One label per prompt
(single-label). Label what the **user asks the agent to do now**, not what the text is about.

## Labels

Stage 1 is the broad group; stage 2 is used only inside the first three groups.

| Stage 1 | Stage 2 label | Brief the agent gets |
|---|---|---|
| change_code | `bugfix`, `feature`, `refactor`, `tests` | yes |
| understand_code | `explain`, `review` | yes |
| plan | `design` | yes |
| run_action | `run_action` | yes (short) |
| write_text | `write_text` | no |
| followup | `followup` | no |
| ambiguous | `ambiguous` | no (message sent unchanged) |

### `bugfix`
- **In:** something is broken and the user wants it fixed: an error, crash, traceback, wrong
  result, failing test, regression. "Debug this" when a fix is wanted.
- **Out:** only asking *why* it fails, with no fix wanted → `explain`. Adding behaviour that
  never existed → `feature`.
- Examples: "compute_total returns 7 instead of 2 for one item, fix it" · "TypeError in
  Invoice.total when discount is None" · "tests fail after my last commit, can you fix them".

### `feature`
- **In:** new behaviour, option, endpoint, flag, function, UI element, integration; a
  change in behaviour the user wants (not a defect).
- **Out:** same behaviour, better structure → `refactor`. Only tests → `tests`.
- Examples: "add a --verbose flag to the CLI" · "support Gemini requests in the gateway" ·
  "write a function that converts CSV to JSON".

### `refactor`
- **In:** change structure, names, layout, style, performance without changing what the code
  does: rename, split a module, remove duplication, migrate an API, speed up.
- **Out:** the user expects different output → `feature` / `bugfix`.
- Examples: "rename _norm_path to normalize_path everywhere" · "split trimmer.py into smaller
  modules" · "replace the for loops with list comprehensions".

### `tests`
- **In:** write, extend or fix **tests only**; raise coverage; add a regression test.
- **Out:** fixing the code under test → `bugfix`.
- Examples: "add unit tests for parse_listing" · "the test fixture is outdated, update it" ·
  "cover the empty-cart case with a test".

### `explain`
- **In:** understand code or behaviour, no edits: what / how / why, walk through a function,
  meaning of an error, how to use an API.
- **Out:** asks to judge quality or find problems → `review`; asks for a change → change label.
- Examples: "what does parse_listing do?" · "why does this regex not match tabs?" · "explain
  how cache events work".

### `review`
- **In:** judge existing code, no edits yet: review a diff / PR, find bugs or security
  problems, suggest improvements, "is this correct?".
- **Out:** "find and fix" → `bugfix`.
- Examples: "review my PR for bugs" · "is this SQL query safe?" · "what could be improved in
  this class?".

### `design`
- **In:** decide how to build something before building it: compare options, architecture,
  data model, library choice, trade-offs, a plan.
- **Out:** a concrete change is requested → change label.
- Examples: "redis or the database for sessions? compare" · "how should we structure the reward
  function? give 2-3 approaches" · "plan the migration to async".

### `run_action`
- **In:** operate, no code writing: run tests / scripts, install, build, git (commit, push,
  rebase), deploy, start a server, check status.
- **Out:** fix what fails afterwards → `bugfix`.
- Examples: "run the test suite" · "install the requirements" · "commit and push this".

### `write_text`
- **In:** prose for people: README, docs, comments, commit / PR description, report, email.
- **Out:** code docstrings as part of a code change → that change's label.
- Examples: "write a README installation section" · "draft the PR description" · "summarise
  this session".

### `followup`
- **In:** short reply continuing the current task: yes / no / ok / continue / go ahead / do it /
  looks good / option 2.
- Examples: "yes" · "ok go ahead" · "use the second option".

### `ambiguous` (fallback)
- **In:** underspecified ("help", "it doesn't work"), not a coding-agent task (general chat,
  creative writing, maths homework), several unrelated tasks with no main one, or not English.
- Never forced into a real label: a wrong label gives the agent wrong advice.

## Tie-breakers (in order)
1. A request to **change** code beats explain / review / design ("explain and fix" → change).
2. Inside change_code: `bugfix` > `feature` > `refactor` > `tests` (a fix with a test →
   `bugfix`).
3. A pasted error **with** a fix request → `bugfix`; without any request → `explain`.
4. Several requests: label the **main** one (the one most of the text is about); if there is
   no main one → `ambiguous`.

## Labeling rules
- Read only the prompt; do not guess hidden context.
- Test-set prompts are labeled **blind** (no model guess shown). 50 test prompts are labeled by
  two people; agreement is reported (Cohen's κ). Disagreements are settled by the third member.
- Record `unsure` in the notes column rather than guessing.
