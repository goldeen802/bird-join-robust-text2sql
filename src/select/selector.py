from __future__ import annotations
from src.validate.validator import ValidationResult
from src.common.sql import count_tables

def score_candidate(r: ValidationResult, linked_values) -> float:
    s = 0.0
    if r.executes: s += 10
    if not r.invalid_joins: s += 5
    if not r.unknown_columns: s += 3
    sql_l = r.sql.lower()
    for v, _, _ in linked_values:
        if v.lower() in sql_l:
            s += 1
    try:
        s -= 0.1 * count_tables(r.sql)
    except Exception:
        pass
    return s

def select_best(results: list[ValidationResult], linked_values) -> ValidationResult:
    if not results:
        # All candidates were empty/whitespace -> return a benign empty result
        # instead of raising ValueError on max([]).
        return ValidationResult(sql="", parse_ok=False)
    valid = [r for r in results if r.is_valid] or results
    return max(valid, key=lambda r: score_candidate(r, linked_values))
