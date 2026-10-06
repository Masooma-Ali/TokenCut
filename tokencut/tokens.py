"""The ONE place where tokens are counted. Swap the tokenizer here only."""

from functools import lru_cache

import tiktoken

ENCODING_NAME = "o200k_base"  # proxy tokenizer for all providers in Iteration 1


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    # First call downloads the encoding file once, then it is cached on disk.
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoding().encode(text, disallowed_special=()))
