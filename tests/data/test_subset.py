from src.data.subset import split_by_table_count, build_subset, train_eval_split

EXAMPLES = [
    {"db_id": "a", "question": "q1", "SQL": "SELECT x FROM t1"},
    {"db_id": "a", "question": "q2", "SQL": "SELECT COUNT(*) FROM t1 JOIN t2 ON t1.id=t2.aid"},
    {"db_id": "a", "question": "q3", "SQL": "SELECT y FROM t2 JOIN t3 ON t2.id=t3.bid"},
]

def test_split_by_table_count():
    singles, twos, more = split_by_table_count(EXAMPLES, sql_key="SQL")
    assert len(singles) == 1 and len(twos) == 2 and len(more) == 0

def test_build_subset_is_join_heavy():
    sub = build_subset(EXAMPLES, sql_key="SQL", two_table_ratio=0.7, seed=0)
    counts = [ex["n_tables"] for ex in sub]
    assert all(n <= 2 for n in counts)
    assert counts.count(2) >= counts.count(1)  # join-heavy

def test_train_eval_split_is_disjoint_and_complete():
    items = [{"id": i} for i in range(100)]
    train, evals = train_eval_split(items, eval_frac=0.15, seed=0)
    assert len(evals) == 15 and len(train) == 85
    train_ids = {x["id"] for x in train}
    eval_ids = {x["id"] for x in evals}
    assert train_ids.isdisjoint(eval_ids)          # no leakage
    assert train_ids | eval_ids == set(range(100))  # nothing dropped
    # deterministic for a fixed seed
    assert train_eval_split(items, 0.15, 0)[1] == evals
