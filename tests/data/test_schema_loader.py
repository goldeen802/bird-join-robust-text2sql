import sqlite3
from src.data.schema_loader import load_db_schema


def test_loads_tables_columns_and_fk(tiny_db):
    db = load_db_schema(tiny_db, "tiny")
    assert set(db.tables) == {"client", "events"}
    assert db.has_column("events", "Client_ID")
    assert db.is_fk_pair("events", "Client_ID", "client", "client_id")


def test_implicit_pk_foreign_key(tmp_path):
    # FK that references the parent table without naming the column ->
    # SQLite reports `to` as NULL; loader must resolve it to the parent PK
    # instead of producing a None column (which crashes fk_pairs).
    p = tmp_path / "imp.sqlite"
    con = sqlite3.connect(p)
    con.executescript(
        """
        CREATE TABLE parent (pid INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE child (cid INTEGER PRIMARY KEY, parent_id INTEGER,
            FOREIGN KEY (parent_id) REFERENCES parent);
        """
    )
    con.commit()
    con.close()
    db = load_db_schema(str(p), "imp")
    assert db.fk_pairs()  # does not crash on the NULL `to`
    assert db.is_fk_pair("child", "parent_id", "parent", "pid")
