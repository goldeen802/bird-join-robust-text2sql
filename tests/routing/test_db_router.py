import numpy as np
from src.routing.db_router import DBRouter

class BagEmbedder:
    """Deterministic bag-of-words embedder over a fixed vocab (test double)."""
    VOCAB = ["car", "price", "client", "city", "complaint", "sales", "employee"]
    def encode(self, texts):
        out = []
        for t in texts:
            t = t.lower()
            out.append(np.array([t.count(w) for w in self.VOCAB], dtype=float))
        return np.vstack(out)

def test_routes_to_best_matching_db():
    summaries = {
        "car_retails": "car price products",
        "retail_complaints": "client city complaint events",
        "sales": "sales employee",
    }
    r = DBRouter(summaries, BagEmbedder())
    assert r.route("complaints from clients in a city") == "retail_complaints"
    assert r.route("the price of a car") == "car_retails"
