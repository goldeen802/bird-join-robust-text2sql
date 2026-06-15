from __future__ import annotations
import sqlite3
from src.common.schema import Column, ForeignKey, TableSchema, DBSchema

def load_db_schema(sqlite_path: str, db_id: str) -> DBSchema:
    con = sqlite3.connect(sqlite_path)
    con.row_factory = sqlite3.Row
    tables: dict[str, TableSchema] = {}
    fks: list[ForeignKey] = []
    names = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for t in names:
        cols = [Column(t, r["name"], r["type"] or "TEXT")
                for r in con.execute(f'PRAGMA table_info("{t}")')]
        tables[t] = TableSchema(t, cols)
        for r in con.execute(f'PRAGMA foreign_key_list("{t}")'):
            fks.append(ForeignKey(t, r["from"], r["table"], r["to"]))
    con.close()
    return DBSchema(db_id, tables, fks)
