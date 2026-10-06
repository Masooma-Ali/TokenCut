"""Member B — Code-graph construction."""

from tokencut.contracts import Chunk, CodeGraph


def build_graph(chunks: list[Chunk]) -> CodeGraph:
    """Build contains / imports / calls / inherits edges between chunks.

    Must:
      - never point an edge at a chunk id that does not exist
      - mark heuristic call/import guesses with resolved=False
      - handle circular imports without infinite loops
    """
    raise NotImplementedError("Member B: implement me")


def dependency_closure(graph: CodeGraph, chunk_ids: set[str], max_hops: int = 1) -> set[str]:
    """Return chunk_ids plus everything they depend on within max_hops."""
    raise NotImplementedError("Member B: implement me")
