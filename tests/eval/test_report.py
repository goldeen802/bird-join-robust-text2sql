from src.eval.report import summarize


def test_summarize_splits_single_vs_multi():
    rows = [
        {"correct": True,  "n_tables": 1},
        {"correct": False, "n_tables": 1},
        {"correct": True,  "n_tables": 2},
        {"correct": False, "n_tables": 2},
        {"correct": True,  "n_tables": 2},
    ]
    rep = summarize(rows)
    assert rep["single"]["accuracy"] == 0.5
    assert round(rep["multi"]["accuracy"], 3) == 0.667
    assert rep["overall"]["n"] == 5
