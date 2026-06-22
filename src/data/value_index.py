from __future__ import annotations
import json
import os
import re
import sqlite3
import time
from src.common.schema import DBSchema

_TEXTISH = ("TEXT", "VARCHAR", "CHAR", "CLOB", "STRING")

def build_value_index(sqlite_path: str, db: DBSchema, max_values_per_col: int = 200,
                      per_col_seconds: float = 5.0) -> list[dict]:
    con = sqlite3.connect(sqlite_path)
    con.text_factory = lambda b: b.decode(errors="replace")
    # Bound each DISTINCT scan: a huge text column (e.g. a posts.Body) can
    # otherwise stall indexing for minutes. A timed-out column is just skipped.
    state = {"deadline": 0.0}
    con.set_progress_handler(lambda: 1 if time.monotonic() > state["deadline"] else 0, 10000)
    out: list[dict] = []
    for table in db.tables.values():
        for col in table.columns:
            if not any(tok in (col.type or "").upper() for tok in _TEXTISH):
                continue
            state["deadline"] = time.monotonic() + per_col_seconds
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

def load_or_build_value_index(sqlite_path: str, db: DBSchema, cache_dir: str,
                              max_values_per_col: int = 200) -> list[dict]:
    """Build the value index once and cache it (per db_id) so reruns and the
    eval reuse it instead of rescanning every database each time."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{db.db_id}.jsonl")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    idx = build_value_index(sqlite_path, db, max_values_per_col)
    save_value_index(idx, cache)
    return idx

def link_values(question: str, index: list[dict],
                min_len: int = 3) -> list[tuple[str, str, str]]:
    """Return (value, table, column) for index values appearing in the question.

    A raw substring match floods the prompt: a 1-char status code like 'D'
    matches any question containing the letter d, dragging junk tables into the
    schema. So require `min_len` characters and a word-boundary match ('arena'
    matches as a word, 'd' inside 'and' does not).
    """
    q = question.lower()
    seen: set[tuple[str, str, str]] = set()
    links: list[tuple[str, str, str]] = []
    for row in index:
        v = row["value_norm"]
        if not v or len(v) < min_len:
            continue
        if re.search(rf"(?<!\w){re.escape(v)}(?!\w)", q):
            key = (row["value"], row["table"], row["column"])
            if key not in seen:
                seen.add(key)
                links.append(key)
    return links
