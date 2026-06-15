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

def _table_score(question_toks, table, embedder) -> float:
    name_toks = _tokens(table.name) | {t for c in table.columns for t in _tokens(c.name)}
    return len(question_toks & name_toks)

def link_schema(question, db: DBSchema, linked_values, embedder=None,
                top_k_tables: int = 4) -> LinkedSchema:
    qtoks = _tokens(question)
    # Always keep tables that have a value link (high recall).
    must_keep = {t for _, t, _ in linked_values}
    scored = sorted(db.tables.values(),
                    key=lambda t: _table_score(qtoks, t, embedder), reverse=True)
    kept = list(must_keep)
    for t in scored:
        if t.name not in kept and len(kept) < max(top_k_tables, len(must_keep)):
            kept.append(t.name)
    kept_lower = {k.lower() for k in kept}
    tables = {name: [c.name for c in db.tables[name].columns]
              for name in db.tables if name.lower() in kept_lower}
    fks = [fk for fk in db.foreign_keys
           if fk.from_table.lower() in kept_lower and fk.to_table.lower() in kept_lower]
    return LinkedSchema(db.db_id, tables, fks)
