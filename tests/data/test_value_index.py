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
