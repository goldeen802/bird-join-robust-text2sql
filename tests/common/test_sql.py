from src.common.sql import tables_in_sql, join_pairs_in_sql, is_single_table, count_tables

SINGLE = "SELECT COUNT(productCode) FROM products WHERE productLine = 'Classic Cars'"
JOIN = ("SELECT COUNT(*) FROM client AS T1 JOIN events AS T2 "
        "ON T1.client_id = T2.Client_ID WHERE T1.city = 'Portland'")

def test_tables_in_sql():
    assert tables_in_sql(SINGLE) == {"products"}
    assert tables_in_sql(JOIN) == {"client", "events"}

def test_is_single_table():
    assert is_single_table(SINGLE)
    assert not is_single_table(JOIN)

def test_count_tables():
    assert count_tables(SINGLE) == 1
    assert count_tables(JOIN) == 2

def test_join_pairs_resolves_aliases():
    pairs = join_pairs_in_sql(JOIN)
    assert ("client", "client_id", "events", "client_id") in pairs
