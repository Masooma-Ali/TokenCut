"""Member A — Repository ingestion and chunking."""

from pathlib import Path

from tokencut.contracts import Chunk


def parse_repo(repo_path: str | Path) -> list[Chunk]:
    """Walk a repository and return one Chunk per module / class / function / method.

    Must:
      - skip ignored, binary and very large files
      - never crash on a file with a syntax error (keep what tree-sitter can parse)
      - return chunks in a deterministic order (file path, then start line)
    """
    raise NotImplementedError("Member A: implement me")
