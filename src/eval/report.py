from __future__ import annotations


def _acc(rows):
    return {"n": len(rows),
            "correct": sum(r["correct"] for r in rows),
            "accuracy": (sum(r["correct"] for r in rows) / len(rows)) if rows else 0.0}


def summarize(rows: list[dict]) -> dict:
    single = [r for r in rows if r["n_tables"] <= 1]
    multi = [r for r in rows if r["n_tables"] >= 2]
    return {"overall": _acc(rows), "single": _acc(single), "multi": _acc(multi)}
