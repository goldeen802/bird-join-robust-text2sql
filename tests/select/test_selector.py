from src.validate.validator import ValidationResult
from src.select.selector import select_best, score_candidate

def vr(sql, executes=True, invalid_joins=None, unknown=None):
    return ValidationResult(sql=sql, parse_ok=True, executes=executes,
                            invalid_joins=invalid_joins or [], unknown_columns=unknown or [])

def test_prefers_executable_fk_valid():
    a = vr("SELECT 1", executes=False)
    b = vr("SELECT city FROM client WHERE city='Portland'")
    best = select_best([a, b], linked_values=[("Portland", "client", "city")])
    assert best.sql == b.sql

def test_value_link_bonus_breaks_tie():
    a = vr("SELECT city FROM client")
    b = vr("SELECT city FROM client WHERE city='Portland'")
    assert score_candidate(b, [("Portland", "client", "city")]) > score_candidate(a, [("Portland", "client", "city")])

def test_select_best_handles_no_candidates():
    # All candidates empty -> must return a benign result, not raise on max([]).
    best = select_best([], linked_values=[])
    assert best.sql == "" and not best.is_valid
