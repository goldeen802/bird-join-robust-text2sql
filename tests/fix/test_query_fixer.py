from src.fix.query_fixer import fix_query

def test_returns_immediately_when_valid(tiny_db):
    calls = []
    def regen(sql, err): calls.append(err); return "SELECT 1"
    out = fix_query("SELECT city FROM client", tiny_db, regen, rounds=2)
    assert out == "SELECT city FROM client" and calls == []

def test_regenerates_on_error(tiny_db):
    def regen(sql, err): return "SELECT city FROM client"   # fixes it
    out = fix_query("SELECT nope FROM client", tiny_db, regen, rounds=2)
    assert out == "SELECT city FROM client"
