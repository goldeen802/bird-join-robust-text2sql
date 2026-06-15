from __future__ import annotations
from dataclasses import dataclass, field
import sqlglot
from sqlglot import exp
from src.common.schema import DBSchema
from src.common.sql import parse, join_pairs_in_sql, _alias_map
from src.eval.harness import execute_sql

@dataclass
class ValidationResult:
    sql: str
    parse_ok: bool
    unknown_columns: list[str] = field(default_factory=list)
    invalid_joins: list[tuple] = field(default_factory=list)
    executes: bool = False
    error: str | None = None
    rows: list | None = None

    @property
    def is_valid(self) -> bool:
        return (self.parse_ok and not self.unknown_columns
                and not self.invalid_joins and self.executes)

def _unknown_columns(sql: str, db: DBSchema) -> list[str]:
    tree = parse(sql)
    amap = _alias_map(tree)
    unknown = []
    all_cols = {(t.lower(), c.name.lower())
                 for t in db.tables for c in db.tables[t].columns}
    any_col = {c.name.lower() for t in db.tables for c in db.tables[t].columns}
    for col in tree.find_all(exp.Column):
        name = col.name.lower()
        if col.table:
            real = amap.get(col.table.lower(), col.table.lower())
            if (real, name) not in all_cols:
                unknown.append(f"{col.table}.{col.name}")
        elif name not in any_col:
            unknown.append(col.name)
    return unknown

def _invalid_joins(sql: str, db: DBSchema) -> list[tuple]:
    return [p for p in join_pairs_in_sql(sql)
            if not db.is_fk_pair(p[0], p[1], p[2], p[3])]

def validate(sql: str, db: DBSchema, db_path: str) -> ValidationResult:
    try:
        parse(sql)
    except (sqlglot.errors.ParseError, Exception):
        return ValidationResult(sql=sql, parse_ok=False)
    r = ValidationResult(sql=sql, parse_ok=True)
    r.unknown_columns = _unknown_columns(sql, db)
    r.invalid_joins = _invalid_joins(sql, db)
    ok, payload = execute_sql(db_path, sql)
    r.executes = ok
    if ok:
        r.rows = payload
    else:
        r.error = payload
    return r
