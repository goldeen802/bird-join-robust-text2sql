# tests/test_regression_joins.py
from src.data.schema_loader import load_db_schema
from src.data.value_index import build_value_index
from src.pipeline import answer_question

class FakeGen:
    def __init__(self, sqls): self.sqls = sqls
    def generate(self, prompt, n=8, **kw): return list(self.sqls)

def test_join_question_selects_fk_valid_join(tiny_db):
    db = load_db_schema(tiny_db, "tiny")
    idx = build_value_index(tiny_db, db)
    gen = FakeGen([
        "SELECT COUNT(*) FROM events AS T2 JOIN client AS T1 ON T1.city=T2.Issue",
        "SELECT COUNT(*) FROM client AS T1 JOIN events AS T2 ON T1.client_id=T2.Client_ID WHERE T2.Issue='Billing disputes'",
    ])
    res = answer_question("how many billing disputes", db, idx, tiny_db, gen)
    assert res.is_valid and "client_id=t2.client_id" in res.sql.lower().replace(" ", "")
