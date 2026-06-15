from src.common.schema import Column, ForeignKey, TableSchema, DBSchema

def make_db():
    client = TableSchema("client", [Column("client", "client_id", "INTEGER"),
                                    Column("client", "city", "TEXT")])
    events = TableSchema("events", [Column("events", "Client_ID", "INTEGER"),
                                    Column("events", "Issue", "TEXT")])
    fk = ForeignKey("events", "Client_ID", "client", "client_id")
    return DBSchema("retail", {"client": client, "events": events}, [fk])

def test_has_column_case_insensitive():
    db = make_db()
    assert db.has_column("CLIENT", "City")
    assert not db.has_column("client", "nope")

def test_is_fk_pair_direction_agnostic():
    db = make_db()
    assert db.is_fk_pair("events", "Client_ID", "client", "client_id")
    assert db.is_fk_pair("client", "client_id", "events", "Client_ID")
    assert not db.is_fk_pair("client", "city", "events", "Issue")
