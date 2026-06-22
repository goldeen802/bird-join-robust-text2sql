from src.eval.harness import execute_sql, result_sets_match, is_order_sensitive

def test_execute_and_match(tiny_db):
    ok_g, gold = execute_sql(tiny_db, "SELECT city FROM client ORDER BY client_id")
    ok_p, pred = execute_sql(tiny_db, "SELECT city FROM client ORDER BY client_id")
    assert ok_g and ok_p
    assert result_sets_match(gold, pred, order_sensitive=True)

def test_order_insensitive_match(tiny_db):
    _, a = execute_sql(tiny_db, "SELECT city FROM client")
    _, b = execute_sql(tiny_db, "SELECT city FROM client ORDER BY city ASC")
    assert result_sets_match(a, b, order_sensitive=False)
    assert not result_sets_match(a, b, order_sensitive=True)

def test_bad_sql_returns_error(tiny_db):
    ok, payload = execute_sql(tiny_db, "SELECT nope FROM client")
    assert not ok and isinstance(payload, str)

def test_execution_timeout_aborts_runaway_query(tiny_db):
    # Infinite recursive CTE would run forever without an execution budget.
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT x FROM c"
    ok, payload = execute_sql(tiny_db, sql, max_seconds=0.5)
    assert not ok and "interrupt" in payload.lower()

def test_order_sensitivity_detection():
    assert is_order_sensitive("SELECT a FROM t ORDER BY a")
    assert not is_order_sensitive("SELECT a FROM t")
