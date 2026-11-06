"""Optional dense embeddings for the repo index (opt-in: `--embed`).

Model BAAI/bge-small-en-v1.5 through fastembed (ONNX, CPU), as in the offline benchmark.
Embedding a chunk is slow on a laptop CPU (~13 chunks/s), so vectors are computed ahead of
time (`python -m tokencut.repoindex build --embed`, or in a background thread of the server)
and cached by a hash of the chunk text: a changed chunk is the only one embedded again.
At search time only the short query is embedded; chunks without a vector yet are skipped.
"""

import hashlib
import logging
import sqlite3
import threading
from pathlib import Path

import numpy as np

from tokencut.contracts import Chunk

log = logging.getLogger(__name__)

MODEL = "BAAI/bge-small-en-v1.5"
MAX_CHARS = 1_000  # same cut as the benchmark (~250 tokens; the head of a def says most)
BATCH = 32


def chunk_text(chunk: Chunk) -> str:
    return f"{chunk.file_path} {chunk.qualified_name}\n{chunk.code[:MAX_CHARS]}"


class ChunkEmbedder:
    def __init__(self, cache_dir: Path, model: str = MODEL):
        from fastembed import TextEmbedding  # optional dependency, imported only when used

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self.model = TextEmbedding(model, cache_dir=str(cache_dir / "models"))
        self._db_path = cache_dir / "embeddings.sqlite"
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS vec (key TEXT PRIMARY KEY, v BLOB)")
        self._vectors: dict[str, np.ndarray] = {}  # key -> unit vector (loaded on demand)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _key(self, chunk: Chunk) -> str:
        return hashlib.sha1(f"{self.model_name}\0{chunk_text(chunk)}".encode()).hexdigest()

    def _load(self, keys: list[str]) -> None:
        missing = [k for k in keys if k not in self._vectors]
        with self._connect() as db:
            for i in range(0, len(missing), 500):
                batch = missing[i:i + 500]
                marks = ",".join("?" * len(batch))
                for key, blob in db.execute(f"SELECT key, v FROM vec WHERE key IN ({marks})", batch):
                    vec = np.frombuffer(blob, dtype=np.float32)
                    self._vectors[key] = vec / (np.linalg.norm(vec) or 1.0)

    def embed_missing(self, chunks: list[Chunk], limit: int | None = None) -> int:
        """Compute vectors not cached yet (slow). Returns how many were computed."""
        keys = {self._key(c): c for c in chunks}
        with self._lock:
            self._load(sorted(keys))
        todo = [(k, c) for k, c in sorted(keys.items()) if k not in self._vectors][:limit]
        done = 0
        for i in range(0, len(todo), 256):
            part = todo[i:i + 256]
            vectors = self.model.passage_embed([chunk_text(c) for _, c in part], batch_size=BATCH)
            rows = []
            for (key, _), vec in zip(part, vectors):
                vec = np.asarray(vec, dtype=np.float32)
                rows.append((key, vec.tobytes()))
                with self._lock:
                    self._vectors[key] = vec / (np.linalg.norm(vec) or 1.0)
            with self._connect() as db:
                db.executemany("INSERT OR REPLACE INTO vec VALUES (?, ?)", rows)
            done += len(part)
            log.info("embedded %d / %d chunks", done, len(todo))
        return done

    def similarities(self, query: str, chunks: list[Chunk]) -> dict[str, float]:
        """Cosine similarity of the query to every chunk that already has a vector."""
        keys = [(c.id, self._key(c)) for c in chunks]
        with self._lock:
            self._load([k for _, k in keys])
            known = [(cid, self._vectors[k]) for cid, k in keys if k in self._vectors]
        if not known:
            return {}
        q = np.asarray(next(iter(self.model.query_embed([query]))), dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = np.stack([v for _, v in known]) @ q
        return {cid: float(s) for (cid, _), s in zip(known, sims)}
