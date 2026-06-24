import src.validate.validator as validator_mod
from src.data.schema_loader import load_db_schema
from src.validate.validator import validate

GOOD = ("SELECT COUNT(*) FROM client AS T1 JOIN events AS T2 "
        "ON T1.client_id = T2.Client_ID WHERE T1.city = 'Portland'")
BAD_JOIN = "SELECT COUNT(*) FROM client AS T1 JOIN events AS T2 ON T1.city = T2.Issue"
BAD_COL = "SELECT nope FROM client"

def test_good_query_valid(tiny_db):
    db = load_db_schema(tiny_db, "tiny")
    r = validate(GOOD, db, tiny_db)
    assert r.is_valid and r.executes and not r.invalid_joins

def test_non_fk_join_rejected(tiny_db):
    db = load_db_schema(tiny_db, "tiny")
    r = validate(BAD_JOIN, db, tiny_db)
    assert r.invalid_joins and not r.is_valid

def test_unknown_column_flagged(tiny_db):
    db = load_db_schema(tiny_db, "tiny")
    r = validate(BAD_COL, db, tiny_db)
    assert r.unknown_columns and not r.is_valid

def test_validate_never_raises_when_analysis_crashes(tiny_db, monkeypatch):
    # Simulate the intermittent IndexError some candidates triggered in analysis:
    # validate must return a (non-valid) result, not propagate and kill the example.
    db = load_db_schema(tiny_db, "tiny")
    def boom(*a, **k):
        raise IndexError("list index out of range")
    monkeypatch.setattr(validator_mod, "_unknown_columns", boom)
    r = validate(GOOD, db, tiny_db)
    assert not r.is_valid          # un-analyzable -> can't be chosen as valid
    assert r.executes              # but execution still measured
