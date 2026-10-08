"""Member C — Context scoring and 4-tier variant generation.

Note: works on chunks + graph, NOT on a repo path, so the same code can later
trim chunks cut out of an agent's request inside the gateway.
"""

from tokencut.contracts import AGGRESSIVE_MIN_TOKENS, Chunk, CodeGraph, ContextVariant, ScoredChunk
from tokencut.graph.queries import GraphIndex
from tokencut.tokens import count_tokens
from tokencut.trimming import scoring
from tokencut.trimming.packing import TierPacker, build_pool, tier_budgets
from tokencut.trimming.query import analyse_query
from tokencut.trimming.render import render_variant


def score_chunks(chunks: list[Chunk], graph: CodeGraph, query: str) -> list[ScoredChunk]:
    """Score every chunk's relevance to the query (BM25 text score + graph distance)."""
    return scoring.score_chunks(chunks, graph, query)


def generate_variants(
    chunks: list[Chunk],
    graph: CodeGraph,
    query: str,
    full_cap_tokens: int | None = None,
    aggressive_min_tokens: int = AGGRESSIVE_MIN_TOKENS,
) -> list[ContextVariant]:
    """Return exactly 4 variants (aggressive, moderate, conservative, full).

    Must:
      - respect each tier's token budget
      - have token counts that never decrease from aggressive -> full
      - be deterministic (break score ties by chunk id)

    full_cap_tokens: None = gateway mode (everything is "full"); a number = evaluation mode
    (top-ranked files up to that many tokens, e.g. EVAL_FULL_CONTEXT_CAP_TOKENS).
    """
    chunks = sorted(chunks, key=lambda c: (c.file_path, c.start_line, c.id))
    index = GraphIndex(graph)
    mentions = analyse_query(query, chunks)
    scores = scoring.score_chunks(chunks, graph, query, mentions=mentions, index=index)

    file_text = scoring.file_text_scores(chunks, query) if full_cap_tokens is not None else None
    pool = build_pool(chunks, scores, full_cap_tokens, file_text)
    full_tokens = count_tokens(render_variant(pool, {c.id for c in pool}, set()))
    budgets = tier_budgets(full_tokens, aggressive_min_tokens)

    order = {c.id: (c.file_path, c.start_line, c.id) for c in pool}
    variants = []
    for packed in TierPacker(pool, scores, index).pack(budgets):
        variants.append(
            ContextVariant(
                tier=packed.tier,
                budget_tokens=packed.budget_tokens,
                kept_chunk_ids=sorted(packed.kept, key=order.__getitem__),
                skeleton_chunk_ids=sorted(packed.skeletons, key=order.__getitem__),
                token_count=packed.token_count,
                rendered_text=packed.rendered_text,
            )
        )
    return variants
