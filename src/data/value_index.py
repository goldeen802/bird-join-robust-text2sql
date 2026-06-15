from __future__ import annotations
import json
import sqlite3
from src.common.schema import DBSchema

_TEXTISH = ("TEXT", "VARCHAR", "CHAR", "CLOB", "STRING")

def build_value_index(sqlite_path: str, db: DBSchema, max_values_per_col: int = 200) -> list[dict]:
    con = sqlite3.connect(sqlite_path)
    con.text_factory = lambda b: b.decode(errors="replace")
    out: list[dict] = []
    for table in db.tables.values():
        for col in table.columns:
            if not any(tok in (col.type or "").upper() for tok in _TEXTISH):
                continue
            try:
                rows = con.execute(
                    f'SELECT DISTINCT "{col.name}" FROM "{table.name}" '
                    f'WHERE "{col.name}" IS NOT NULL LIMIT {max_values_per_col}'
                ).fetchall()
            except sqlite3.Error:
                continue
            for (val,) in rows:
                if isinstance(val, str) and val.strip():
                    out.append({"table": table.name, "column": col.name,
                                "value": val, "value_norm": val.strip().lower()})
    con.close()
    return out

def save_value_index(index: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in index:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def link_values(question: str, index: list[dict]) -> list[tuple[str, str, str]]:
    """Return (value, table, column) for index values appearing in the question."""
    q = question.lower()
    seen: set[tuple[str, str, str]] = set()
    links: list[tuple[str, str, str]] = []
    for row in index:
        if row["value_norm"] and row["value_norm"] in q:
            key = (row["value"], row["table"], row["column"])
            if key not in seen:
                seen.add(key)
                links.append(key)
    return links
