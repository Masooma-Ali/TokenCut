"""Dense-embedding relevance (baseline, and optional signal for TokenCut).

Model: BAAI/bge-small-en-v1.5 through fastembed (ONNX, CPU). Chunk vectors are cached in
SQLite by a hash of (model, text), so a chunk shared by many commits is embedded once.
"""

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from eval.loader import DEFAULT_CACHE_DIR
from tokencut.contracts import Chunk

MODEL = "BAAI/bge-small-en-v1.5"
MAX_CHARS = 1_000  # chunk text cut for speed (~250 tokens); the head of a def says most
BATCH = 32


def chunk_text(chunk: Chunk) -> str:
    return f"{chunk.file_path} {chunk.qualified_name}\n{chunk.code[:MAX_CHARS]}"


class Embedder:
    def __init__(self, model: str = MODEL, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.model_name = model
        self.model = TextEmbedding(model, cache_dir=str(cache_dir / "models"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_dir / "embeddings.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS vec (key TEXT PRIMARY KEY, v BLOB)")

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model_name}\0{text}".encode()).hexdigest()

    def passages(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):
            batch = keys[i:i + 500]
            marks = ",".join("?" * len(batch))
            for key, blob in self.db.execute(f"SELECT key, v FROM vec WHERE key IN ({marks})", batch):
                found[key] = np.frombuffer(blob, dtype=np.float32)
        missing = [(k, t) for k, t in zip(keys, texts) if k not in found]
        if missing:
            vectors = self.model.passage_embed([t for _, t in missing], batch_size=BATCH)
            rows = []
            for (key, _), vec in zip(missing, vectors):
                vec = np.asarray(vec, dtype=np.float32)
                found[key] = vec
                rows.append((key, vec.tobytes()))
            self.db.executemany("INSERT OR REPLACE INTO vec VALUES (?, ?)", rows)
            self.db.commit()
        return np.stack([found[k] for k in keys]) if keys else np.zeros((0, 1), np.float32)

    def similarities(self, query: str, chunks: list[Chunk]) -> dict[str, float]:
        """Cosine similarity of the query to each chunk."""
        if not chunks:
            return {}
        q = np.asarray(next(iter(self.model.query_embed([query]))), dtype=np.float32)
        m = self.passages([chunk_text(c) for c in chunks])
        q = q / (np.linalg.norm(q) or 1.0)
        m = m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)
        return {c.id: float(s) for c, s in zip(chunks, m @ q)}
