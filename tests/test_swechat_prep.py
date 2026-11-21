"""SWE-chat preprocessing for the prompt router: machine prompts, label mapping, dedup, split."""

from eval.prompt_router.swechat_prep import (
    cut,
    dedupe,
    is_machine,
    repo_split,
    strip_markup,
    target_label,
)


def test_machine_generated_prompts_are_recognised():
    for text in (
        "<command-name>/clear</command-name>\n<command-message>clear</command-message>",
        "[Request interrupted by user for tool use]",
        "/review the latest fix",
        "<environment_context>\n<cwd>/repo</cwd>\n</environment_context>",
        "<turn_aborted> The user interrupted the previous turn on purpose.",
        ">>> APPROVAL REQUEST START\nPlanned action JSON:",
        "The Codex agent has requested the following next action:",
        "[delegate worktree]\nRun all file edits inside this worktree",
        "[Image: image/png]",
        "   ",
        '"[delegate worktree]\nRun all file edits inside this worktree',
        "# AGENTS.md instructions for /Users/x/repo\n<INSTRUCTIONS>",
        'Another Claude session sent a message:\n<teammate-message teammate_id="a">hi</teammate-message>',
        "Task done.\n<usage><total_tokens>120</total_tokens><tool_uses>3</tool_uses></usage>",
    ):
        assert is_machine(text), text
    for text in ("yes", "commit and push", "fix the failing login test",
                 "<image name=[Image #1]> the button overlaps the sidebar, fix it"):
        assert not is_machine(text), text


def test_markup_is_removed_but_the_request_kept():
    assert strip_markup("<image name=[Image #1]> the button overlaps, fix it") == \
        "the button overlaps, fix it"
    assert strip_markup("look [Image #2] here") == "look here"


def test_long_prompts_keep_head_and_tail():
    text = "A" * 3000 + "please fix this"
    out = cut(text, 2000)
    assert len(out) <= 2010 and out.startswith("A") and out.endswith("please fix this")
    assert cut("short", 2000) == "short"


def test_target_labels():
    assert target_label("debug", "non_pushback") == "bugfix"
    assert target_label("create new code", "non_pushback") == "feature"
    assert target_label("git", "non_pushback") == "run_action"
    assert target_label("connect", "non_pushback") == "run_action"
    assert target_label("understand", "non_pushback") == "explain"
    assert target_label("other", "non_pushback") == "no_brief"
    assert target_label("refactor", "correction") == "no_brief"  # steering inside a task
    assert target_label("debug", "rejection") == "no_brief"
    assert target_label("debug", "failure_report") == "bugfix"  # "still failing" asks for a fix
    assert target_label(None, None) is None  # unlabelled: dropped


def test_dedupe_keeps_one_copy_with_the_majority_label():
    rows = [{"text": "commit and push", "label": "run_action", "repo_id": "a"},
            {"text": "Commit  and push", "label": "run_action", "repo_id": "b"},
            {"text": "commit and push", "label": "no_brief", "repo_id": "c"},
            {"text": "add a flag", "label": "feature", "repo_id": "a"}]
    out = dedupe(rows)
    assert len(out) == 2
    assert next(r for r in out if r["text"].startswith("commit"))["label"] == "run_action"


def test_repo_split_has_no_repo_in_two_splits_and_is_seeded():
    rows = [{"repo_id": f"r{i % 50}", "text": str(i)} for i in range(1000)]
    a, b = repo_split(rows, seed=3), repo_split(rows, seed=3)
    assert a == b
    repos = {name: {r["repo_id"] for r in part} for name, part in a.items()}
    assert not (repos["train"] & repos["dev"]) and not (repos["train"] & repos["test"])
    assert not (repos["dev"] & repos["test"])
    assert sum(len(p) for p in a.values()) == 1000
