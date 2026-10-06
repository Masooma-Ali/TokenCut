"""Relevance of every chunk to the query (IMPLEMENTATION_PLAN C2).

score = ALPHA * structural_score + (1 - ALPHA) * text_score
  text_score       BM25 over identifier sub-words, normalised to [0, 1]
  structural_score weight / (1 + graph distance) from the chunks the query mentions
"""

import re

from rank_bm25 import BM25Okapi

from tokencut.contracts import Chunk, CodeGraph, ScoredChunk
from tokencut.graph.queries import GraphIndex
from tokencut.trimming.query import SYMBOL_WEIGHT, QueryMentions, analyse_query

ALPHA = 0.5  # (tunable) log changes in the DECISIONS.md change log
PSEUDO_MENTIONS = 3  # fallback when the query names nothing: top-N chunks by text score

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def split_identifier(identifier: str) -> list[str]:
    """'apply_discount' -> ['apply', 'discount']; 'HTTPServerError' -> ['http', 'server', 'error']."""
    words = []
    for part in identifier.split("_"):
        for word in _CAMEL_RE.findall(part):
            if len(word) >= 2:
                words.append(word.lower())
    return words


def tokenize(text: str) -> list[str]:
    words = []
    for identifier in _IDENT_RE.findall(text):
        words.extend(split_identifier(identifier))
    return words


def text_scores(chunks: list[Chunk], query: str) -> list[float]:
    """BM25 score of each chunk, divided by the max (all zeros if nothing matches)."""
    query_words = tokenize(query)
    documents = [tokenize(f"{c.file_path} {c.qualified_name} {c.code}") for c in chunks]
    if not query_words or not any(documents):
        return [0.0] * len(chunks)
    raw = BM25Okapi(documents).get_scores(query_words)
    clipped = [max(0.0, float(s)) for s in raw]
    best = max(clipped, default=0.0)
    if best <= 0:
        return [0.0] * len(chunks)
    return [s / best for s in clipped]


def structural_scores(index: GraphIndex, weights: dict[str, float]) -> dict[str, float]:
    """For each reachable chunk: max over mentioned chunks of weight / (1 + distance)."""
    result: dict[str, float] = {}
    for weight in sorted(set(weights.values()), reverse=True):
        sources = sorted(cid for cid, w in weights.items() if w == weight)
        for cid, distance in index.distances(sources).items():
            result[cid] = max(result.get(cid, 0.0), weight / (1 + distance))
    return result


def score_chunks(
    chunks: list[Chunk],
    graph: CodeGraph,
    query: str,
    mentions: QueryMentions | None = None,
    index: GraphIndex | None = None,
) -> list[ScoredChunk]:
    """Score every chunk's relevance to the query (BM25 text score + graph distance)."""
    if mentions is None:
        mentions = analyse_query(query, chunks)
    if index is None:
        index = GraphIndex(graph)

    texts = text_scores(chunks, query)
    text_by_id = {c.id: t for c, t in zip(chunks, texts)}

    weights = dict(mentions.weights)
    if not weights:
        # Fallback: the query names nothing, so the best text matches act as the mentions.
        ranked = sorted(
            (cid for cid, t in text_by_id.items() if t > 0), key=lambda cid: (-text_by_id[cid], cid)
        )
        weights = {cid: SYMBOL_WEIGHT for cid in ranked[:PSEUDO_MENTIONS]}
    structural = structural_scores(index, weights)

    scored = []
    for c in chunks:
        s = structural.get(c.id, 0.0)
        t = text_by_id[c.id]
        scored.append(
            ScoredChunk(
                chunk_id=c.id, score=ALPHA * s + (1 - ALPHA) * t, structural_score=s, text_score=t
            )
        )
    scored.sort(key=lambda sc: (-sc.score, sc.chunk_id))  # DECISIONS #8: ties by chunk id
    return scored
