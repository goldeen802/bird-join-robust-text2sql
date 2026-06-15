from src.data.schema_loader import load_db_schema
from src.data.value_index import build_value_index
from src.pipeline import answer_question

class FakeGen:
    def __init__(self, sqls): self.sqls = sqls
    def generate(self, prompt, n=8, **kw): return list(self.sqls)

def test_pipeline_picks_executable_fk_join(tiny_db):
    db = load_db_schema(tiny_db, "tiny")
    idx = build_value_index(tiny_db, db)
    gen = FakeGen([
        "SELECT COUNT(*) FROM client AS T1 JOIN events AS T2 ON T1.city = T2.Issue",  # bad FK
        "SELECT COUNT(*) FROM client AS T1 JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.city='Portland'",
    ])
    result = answer_question("how many events from Portland", db, idx, tiny_db, gen)
    assert "client_id = T2.Client_ID".lower() in result.sql.lower()
    assert result.executes
