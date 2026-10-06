"""Incremental company-local embeddings with exact cosine search and lexical fallback."""

from hashlib import sha256
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from urllib.request import Request, urlopen

import numpy as np

from .search_index import _case_text


class SemanticSearchError(RuntimeError):
    pass


class SemanticIndex:
    def __init__(self, path: Path, model="embeddinggemma", base_url="http://127.0.0.1:11434", embed=None):
        self.path = Path(path)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.provider = sha256(f"v1|{self.base_url}|{model}".encode()).hexdigest()
        self._embed_override = embed

    @staticmethod
    def text(case):
        # Bound text deliberately; store the hash of the exact model input.
        return _case_text(case)[:6000]

    @staticmethod
    def digest(text):
        return sha256(text.encode()).hexdigest()

    def embed(self, texts, timeout=30):
        try:
            if self._embed_override:
                vectors = self._embed_override(texts)
            else:
                request = Request(
                    f"{self.base_url}/api/embed",
                    data=json.dumps({"model": self.model, "input": texts, "truncate": False}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=timeout) as response:
                    vectors = json.load(response)["embeddings"]
            array = np.asarray(vectors, dtype=np.float32)
            if array.ndim != 2 or array.shape[0] != len(texts) or not array.shape[1] or not np.isfinite(array).all():
                raise ValueError("Invalid embedding dimensions or values")
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            if (norms == 0).any():
                raise ValueError("Zero-length embedding")
            return array / norms
        except Exception as error:
            raise SemanticSearchError(f"Semantic model {self.model} unavailable: {error}") from error

    def _existing(self):
        if not self.path.exists():
            return {}
        try:
            with closing(sqlite3.connect(self.path)) as connection:
                rows = connection.execute("SELECT fingerprint, vector FROM embeddings WHERE provider = ?", (self.provider,)).fetchall()
            return {fingerprint: np.frombuffer(vector, dtype=np.float32).copy() for fingerprint, vector in rows}
        except (sqlite3.Error, ValueError):
            return {}

    def build(self, cases, progress=None):
        """Embed only new/changed records; commit each batch for resumable builds."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        texts = list(dict.fromkeys(self.text(case) for case in cases))
        existing = self._existing()
        missing = [text for text in texts if self.digest(text) not in existing]
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS embeddings (provider TEXT, fingerprint TEXT, vector BLOB, PRIMARY KEY(provider, fingerprint))")
            for start in range(0, len(missing), 16):
                batch = missing[start:start + 16]
                vectors = self.embed(batch)
                dimensions = {len(vector) for vector in existing.values()}
                if dimensions and dimensions != {vectors.shape[1]}:
                    raise SemanticSearchError("Embedding dimensions changed. Use a new model name or rebuild the index.")
                connection.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)", [
                    (self.provider, self.digest(text), vector.tobytes()) for text, vector in zip(batch, vectors)
                ])
                connection.commit()
                existing.update({self.digest(text): vector for text, vector in zip(batch, vectors)})
                if progress:
                    progress(min(start + len(batch), len(missing)), len(missing))
        return len(missing)

    def load(self, cases):
        rows = self._existing()
        indices, vectors = [], []
        for index, case in enumerate(cases):
            vector = rows.get(self.digest(self.text(case)))
            if vector is not None:
                indices.append(index)
                vectors.append(vector)
        if not vectors:
            return indices, None
        if len({len(vector) for vector in vectors}) != 1:
            return [], None
        return indices, np.stack(vectors)

    def search(self, query, indices, matrix, allowed):
        if matrix is None:
            return {}
        query_vector = self.embed([query[:6000]], timeout=3)[0]
        if matrix.shape[1] != len(query_vector):
            raise SemanticSearchError("Query and stored embedding dimensions differ.")
        scores = matrix @ query_vector
        return {index: float(score) for index, score in zip(indices, scores) if index in allowed}
