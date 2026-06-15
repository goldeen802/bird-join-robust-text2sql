from __future__ import annotations
import sqlite3
from collections import Counter

def execute_sql(db_path: str, sql: str, timeout: float = 30.0):
    """Return (True, rows) or (False, error_message)."""
    try:
        con = sqlite3.connect(db_path, timeout=timeout)
        con.text_factory = lambda b: b.decode(errors="replace")
        rows = con.execute(sql).fetchall()
        con.close()
        return True, rows
    except sqlite3.Error as e:
        return False, str(e)

def _norm_cell(c):
    return round(c, 6) if isinstance(c, float) else c

def _norm_rows(rows):
    return [tuple(_norm_cell(c) for c in r) for r in rows]

def result_sets_match(a, b, order_sensitive: bool) -> bool:
    na, nb = _norm_rows(a), _norm_rows(b)
    if order_sensitive:
        return na == nb
    return Counter(na) == Counter(nb)

def is_order_sensitive(sql: str) -> bool:
    return "order by" in sql.lower()
