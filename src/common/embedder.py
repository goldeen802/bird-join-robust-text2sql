from __future__ import annotations

def load_embedder(name: str):
    """Return an object with `.encode(list[str]) -> ndarray`.

    Thin wrapper around sentence-transformers so the same embedder powers the
    DB router, the schema linker, training-data construction, and the demo.
    Kept lazy so importing this module never requires torch/transformers.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name)

    class _Embedder:
        def encode(self, texts):
            return model.encode(list(texts))

    return _Embedder()
