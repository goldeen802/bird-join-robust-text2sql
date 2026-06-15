from src.generate.generator import clean_sql

def test_clean_sql_strips_fences_and_prefix():
    raw = "```sql\nSELECT 1;\n```"
    assert clean_sql(raw) == "SELECT 1"

def test_clean_sql_takes_first_statement():
    raw = "SELECT a FROM t; SELECT b FROM t;"
    assert clean_sql(raw) == "SELECT a FROM t"
