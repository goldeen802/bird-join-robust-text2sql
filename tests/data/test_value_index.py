from src.data.schema_loader import load_db_schema
from src.data.value_index import build_value_index, link_values

def test_build_and_link(tiny_db):
    db = load_db_schema(tiny_db, "tiny")
    idx = build_value_index(tiny_db, db, max_values_per_col=50)
    # 'Portland' is a value in client.city
    links = link_values("complaints from Portland about Billing disputes", idx)
    cols = {(t, c) for _, t, c in links}
    assert ("client", "city") in cols
    assert ("events", "Issue") in cols

def test_link_ignores_short_codes_and_substrings():
    # Short codes ('D') and substrings inside words must NOT match, or the
    # prompt fills with junk tables; real word-boundary values still match.
    idx = [
        {"value": "D", "table": "results", "column": "status", "value_norm": "d"},
        {"value": "RE", "table": "constructors", "column": "ref", "value_norm": "re"},
        {"value": "arena", "table": "cards", "column": "avail", "value_norm": "arena"},
    ]
    links = link_values("which cards are reissued and droppable in the arena", idx)
    cols = {(t, c) for _, t, c in links}
    assert ("cards", "avail") in cols          # 'arena' matches as a word
    assert ("results", "status") not in cols   # 'd' must not match
    assert ("constructors", "ref") not in cols  # 're' inside 'reissued' must not match
