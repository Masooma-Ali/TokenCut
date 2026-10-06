"""Member C — Context scoring and 4-tier variant generation.

Note: works on chunks + graph, NOT on a repo path, so the same code can later
trim chunks cut out of an agent's request inside the gateway.
"""

from tokencut.contracts import Chunk, CodeGraph, ContextVariant, ScoredChunk


def score_chunks(chunks: list[Chunk], graph: CodeGraph, query: str) -> list[ScoredChunk]:
    """Score every chunk's relevance to the query (BM25 text score + graph distance)."""
    raise NotImplementedError("Member C: implement me")


def generate_variants(chunks: list[Chunk], graph: CodeGraph, query: str) -> list[ContextVariant]:
    """Return exactly 4 variants (aggressive, moderate, conservative, full).

    Must:
      - respect each tier's token budget
      - have token counts that never decrease from aggressive -> full
      - be deterministic (break score ties by chunk id)
    """
    raise NotImplementedError("Member C: implement me")
