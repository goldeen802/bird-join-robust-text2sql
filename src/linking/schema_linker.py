from __future__ import annotations
from dataclasses import dataclass
import re
from src.common.schema import DBSchema, ForeignKey

@dataclass
class LinkedSchema:
    db_id: str
    tables: dict[str, list[str]]          # table -> kept column names
    foreign_keys: list[ForeignKey]

def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))

def _table_text(table) -> str:
    """Short natural-language-ish summary of a table for embedding."""
    return f"{table.name}: " + ", ".join(c.name for c in table.columns)

def _lexical_scores(question, tables) -> list[float]:
    qtoks = _tokens(question)
    out = []
    for t in tables:
        name_toks = _tokens(t.name) | {tok for c in t.columns for tok in _tokens(c.name)}
        out.append(float(len(qtoks & name_toks)))
    return out

def _embed_scores(question, tables, embedder) -> list[float]:
    import numpy as np
    mat = np.asarray(embedder.encode([_table_text(t) for t in tables]), dtype=float)
    q = np.asarray(embedder.encode([question]), dtype=float)[0]
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    q = q / (np.linalg.norm(q) + 1e-9)
    return list(mat @ q)

def link_schema(question, db: DBSchema, linked_values, embedder=None,
                top_k_tables: int = 4) -> LinkedSchema:
    # Always keep tables that have a value link (high recall).
    must_keep = {t for _, t, _ in linked_values}
    tables = list(db.tables.values())
    # Embedding ranking when an embedder is supplied; lexical token overlap
    # otherwise (used by tests and deterministic offline builds).
    scores = (_embed_scores(question, tables, embedder) if embedder is not None
              else _lexical_scores(question, tables))
    order = sorted(range(len(tables)), key=lambda i: scores[i], reverse=True)
    scored = [tables[i] for i in order]
    kept = list(must_keep)
    for t in scored:
        if t.name not in kept and len(kept) < max(top_k_tables, len(must_keep)):
            kept.append(t.name)
    kept_lower = {k.lower() for k in kept}
    tables_out = {name: [c.name for c in db.tables[name].columns]
                  for name in db.tables if name.lower() in kept_lower}
    fks = [fk for fk in db.foreign_keys
           if fk.from_table.lower() in kept_lower and fk.to_table.lower() in kept_lower]
    return LinkedSchema(db.db_id, tables_out, fks)
