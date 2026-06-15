from src.data.schema_loader import load_db_schema


def test_loads_tables_columns_and_fk(tiny_db):
    db = load_db_schema(tiny_db, "tiny")
    assert set(db.tables) == {"client", "events"}
    assert db.has_column("events", "Client_ID")
    assert db.is_fk_pair("events", "Client_ID", "client", "client_id")
