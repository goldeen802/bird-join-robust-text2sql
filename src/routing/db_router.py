from __future__ import annotations
import numpy as np

def _cosine(matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    v = vec / (np.linalg.norm(vec) + 1e-9)
    return m @ v

class DBRouter:
    def __init__(self, db_summaries: dict[str, str], embedder):
        self.db_ids = list(db_summaries)
        self.embedder = embedder
        self.matrix = np.asarray(embedder.encode([db_summaries[d] for d in self.db_ids]), dtype=float)

    def route(self, question: str) -> str:
        q = np.asarray(self.embedder.encode([question]), dtype=float)[0]
        sims = _cosine(self.matrix, q)
        return self.db_ids[int(np.argmax(sims))]
