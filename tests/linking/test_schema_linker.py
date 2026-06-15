from src.data.schema_loader import load_db_schema
from src.linking.schema_linker import link_schema, LinkedSchema

def test_keeps_value_linked_tables_and_fk(tiny_db):
    db = load_db_schema(tiny_db, "tiny")
    linked_values = [("Portland", "client", "city"), ("Billing disputes", "events", "Issue")]
    ls = link_schema("how many Billing disputes from Portland",
                     db, linked_values, embedder=None, top_k_tables=4)
    assert isinstance(ls, LinkedSchema)
    # both value-linked tables retained
    assert {"client", "events"} <= set(ls.tables)
    # the FK between them is included
    assert ls.foreign_keys  # non-empty
