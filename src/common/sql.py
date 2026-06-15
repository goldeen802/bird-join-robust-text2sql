from __future__ import annotations
import sqlglot
from sqlglot import exp

def parse(sql: str):
    return sqlglot.parse_one(sql, read="sqlite")

def tables_in_sql(sql: str) -> set[str]:
    return {t.name.lower() for t in parse(sql).find_all(exp.Table)}

def count_tables(sql: str) -> int:
    return len(tables_in_sql(sql))

def is_single_table(sql: str) -> bool:
    return count_tables(sql) <= 1

def _alias_map(tree) -> dict[str, str]:
    m: dict[str, str] = {}
    for t in tree.find_all(exp.Table):
        real = t.name.lower()
        m[real] = real
        m[t.alias_or_name.lower()] = real
    return m

def join_pairs_in_sql(sql: str) -> list[tuple[str, str, str, str]]:
    """Column==column equalities across two different tables (alias-resolved)."""
    tree = parse(sql)
    amap = _alias_map(tree)
    pairs: list[tuple[str, str, str, str]] = []
    for eq in tree.find_all(exp.EQ):
        l, r = eq.left, eq.right
        if isinstance(l, exp.Column) and isinstance(r, exp.Column):
            lt = amap.get((l.table or "").lower())
            rt = amap.get((r.table or "").lower())
            if lt and rt and lt != rt:
                pairs.append((lt, l.name.lower(), rt, r.name.lower()))
    return pairs
