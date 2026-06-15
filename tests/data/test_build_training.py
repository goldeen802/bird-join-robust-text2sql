from src.data.schema_loader import load_db_schema
from src.data.value_index import build_value_index
from src.data.build_training import make_training_record

def test_make_training_record(tiny_db):
    db = load_db_schema(tiny_db, "tiny")
    idx = build_value_index(tiny_db, db)
    ex = {"db_id": "tiny", "question": "how many Billing disputes from Portland",
          "evidence": "", "SQL": "SELECT COUNT(*) FROM events"}
    rec = make_training_record(ex, db, idx)
    assert rec["target"] == "SELECT COUNT(*) FROM events"
    assert "Schema:" in rec["prompt"]
    assert "events(" in rec["prompt"]
