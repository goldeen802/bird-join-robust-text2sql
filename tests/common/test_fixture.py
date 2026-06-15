import sqlite3

def test_tiny_db_has_rows(tiny_db):
    con = sqlite3.connect(tiny_db)
    n = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    con.close()
    assert n == 3
