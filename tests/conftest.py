import sqlite3
import pytest

@pytest.fixture
def tiny_db(tmp_path):
    path = tmp_path / "tiny.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE client (client_id INTEGER PRIMARY KEY, city TEXT);
        CREATE TABLE events (
            event_id INTEGER PRIMARY KEY,
            Client_ID INTEGER,
            Issue TEXT,
            FOREIGN KEY (Client_ID) REFERENCES client(client_id)
        );
        INSERT INTO client VALUES (1,'Portland'),(2,'Chicago');
        INSERT INTO events VALUES (10,1,'Billing disputes'),(11,1,'Late delivery'),(12,2,'Billing disputes');
        """
    )
    con.commit()
    con.close()
    return str(path)
