from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Column:
    table: str
    name: str
    type: str

@dataclass(frozen=True)
class ForeignKey:
    from_table: str
    from_col: str
    to_table: str
    to_col: str

@dataclass
class TableSchema:
    name: str
    columns: list[Column]

def _k(t: str, c: str) -> tuple[str, str]:
    return (t.lower(), c.lower())

@dataclass
class DBSchema:
    db_id: str
    tables: dict[str, TableSchema]
    foreign_keys: list[ForeignKey]

    def _tables_lower(self) -> dict[str, TableSchema]:
        return {name.lower(): ts for name, ts in self.tables.items()}

    def has_column(self, table: str, col: str) -> bool:
        ts = self._tables_lower().get(table.lower())
        if ts is None:
            return False
        return any(c.name.lower() == col.lower() for c in ts.columns)

    def fk_pairs(self) -> set[frozenset]:
        return {
            frozenset({_k(fk.from_table, fk.from_col), _k(fk.to_table, fk.to_col)})
            for fk in self.foreign_keys
        }

    def is_fk_pair(self, t1: str, c1: str, t2: str, c2: str) -> bool:
        return frozenset({_k(t1, c1), _k(t2, c2)}) in self.fk_pairs()
