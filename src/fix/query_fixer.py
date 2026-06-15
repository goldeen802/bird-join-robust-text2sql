from __future__ import annotations
from typing import Callable
from src.eval.harness import execute_sql

def fix_query(sql: str, db_path: str,
              regenerate: Callable[[str, str], str], rounds: int = 1) -> str:
    current = sql
    for _ in range(rounds):
        ok, payload = execute_sql(db_path, current)
        if ok:
            return current
        current = regenerate(current, payload)
    return current
