"""Find the Python files of a repository (DECISIONS #12)."""

import logging
import os
import subprocess
from pathlib import Path

from tokencut.contracts import MAX_FILE_BYTES, SKIP_DIR_NAMES

log = logging.getLogger(__name__)


def _git_files(repo: Path) -> list[str] | None:
    """Tracked + untracked-but-not-ignored files, relative to repo. None if git is unusable."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=repo,
            capture_output=True,
            check=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    names = result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    return [name for name in names if name]


def _walk_files(repo: Path) -> list[str]:
    """Fallback when the folder is not a git repo: plain directory walk."""
    found = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIR_NAMES)
        for name in sorted(files):
            found.append((Path(root) / name).relative_to(repo).as_posix())
    return found


def _is_binary(path: Path) -> bool:
    with path.open("rb") as f:
        return b"\0" in f.read(8192)


def list_python_files(repo_path: str | Path) -> list[str]:
    """Return sorted forward-slash paths (relative to repo_path) of the files to parse."""
    repo = Path(repo_path)
    names = _git_files(repo)
    if names is None:
        names = _walk_files(repo)

    selected = set()
    for name in names:
        rel = Path(name)
        if rel.suffix != ".py":
            continue
        if any(part in SKIP_DIR_NAMES for part in rel.parts[:-1]):
            continue
        path = repo / rel
        if not path.is_file():  # e.g. tracked by git but deleted in the working tree
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            log.info("skipping large file %s", name)
            continue
        if _is_binary(path):
            log.info("skipping binary file %s", name)
            continue
        selected.add(rel.as_posix())
    return sorted(selected)
