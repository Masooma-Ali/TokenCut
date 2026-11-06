"""BM25 over an inverted index that can add and remove documents (incremental updates).

rank_bm25 rebuilds everything for one changed file (1.7 s on django's 32k chunks); here a
changed file only re-indexes its own chunks, and a query touches only the postings of its
words instead of scanning every document.
"""

import math
from collections import Counter


class Bm25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, dict[str, int]] = {}  # word -> {doc key: term frequency}
        self.lengths: dict[str, int] = {}  # doc key -> number of words
        self._terms: dict[str, list[str]] = {}  # doc key -> its distinct words (for removal)
        self._total = 0

    def __len__(self) -> int:
        return len(self.lengths)

    def add(self, key: str, words: list[str]) -> None:
        if key in self.lengths:
            self.remove(key)
        counts = Counter(words)
        for word, tf in counts.items():
            self.postings.setdefault(word, {})[key] = tf
        self.lengths[key] = len(words)
        self._terms[key] = sorted(counts)
        self._total += len(words)

    def remove(self, key: str) -> None:
        if key not in self.lengths:
            return
        for word in self._terms.pop(key):
            docs = self.postings[word]
            docs.pop(key, None)
            if not docs:
                del self.postings[word]
        self._total -= self.lengths.pop(key)

    def scores(self, query_words: list[str]) -> dict[str, float]:
        """Raw BM25 score of every document containing at least one query word."""
        n = len(self.lengths)
        if not n:
            return {}
        avg = self._total / n or 1.0
        result: dict[str, float] = {}
        for word, qtf in sorted(Counter(query_words).items()):
            docs = self.postings.get(word)
            if not docs:
                continue
            idf = math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            for key, tf in docs.items():
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[key] / avg)
                result[key] = result.get(key, 0.0) + qtf * idf * tf * (self.k1 + 1) / norm
        return result


def normalised(scores: dict[str, float]) -> dict[str, float]:
    best = max(scores.values(), default=0.0)
    return {k: v / best for k, v in scores.items()} if best > 0 else {}
