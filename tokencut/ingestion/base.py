"""LanguageParser interface (DECISIONS #6).

Python is the only language in Iteration 1, but parse_repo() talks to parsers
only through this Protocol so another language can be added later without
touching the graph or trimming code.
"""

from typing import Protocol

from tokencut.contracts import Chunk


class LanguageParser(Protocol):
    file_suffixes: tuple[str, ...]

    def parse_file(self, rel_path: str, source: bytes) -> list[Chunk]:
        """Return the chunks of one file. Must never raise on bad source code."""
        ...
