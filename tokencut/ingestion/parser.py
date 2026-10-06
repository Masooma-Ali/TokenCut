"""Member A — Repository ingestion and chunking."""

import logging
from pathlib import Path

from tokencut.contracts import Chunk
from tokencut.ingestion.python_parser import PythonParser
from tokencut.ingestion.walker import list_python_files

log = logging.getLogger(__name__)


def parse_repo(repo_path: str | Path) -> list[Chunk]:
    """Walk a repository and return one Chunk per module / class / function / method.

    Must:
      - skip ignored, binary and very large files
      - never crash on a file with a syntax error (keep what tree-sitter can parse)
      - return chunks in a deterministic order (file path, then start line)
    """
    repo = Path(repo_path)
    if not repo.is_dir():
        raise FileNotFoundError(f"repository folder not found: {repo}")

    parser = PythonParser()
    chunks: list[Chunk] = []
    for rel_path in list_python_files(repo):
        try:
            source = (repo / rel_path).read_bytes()
        except OSError as exc:
            log.warning("cannot read %s: %s", rel_path, exc)
            continue
        chunks.extend(parser.parse_file(rel_path, source))

    chunks.sort(key=lambda c: (c.file_path, c.start_line, c.id))
    return chunks
