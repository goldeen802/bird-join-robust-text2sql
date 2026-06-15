from __future__ import annotations
import argparse, json, glob, os, yaml
from src.data.schema_loader import load_db_schema
from src.data.value_index import build_value_index
from src.common.sql import count_tables
from src.eval.harness import execute_sql, result_sets_match, is_order_sensitive
from src.eval.report import summarize
from src.pipeline import answer_question
from src.generate.generator import Generator

def find_db(root, db_id):
    return glob.glob(os.path.join(root, "**", db_id, f"{db_id}.sqlite"), recursive=True)[0]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/pipeline.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    root = cfg["paths"]["bird_root"]
    examples = [json.loads(l) for l in open(cfg["paths"]["subset_eval"], encoding="utf-8")]
    gen = Generator(cfg["model"]["base"], cfg["model"].get("adapter"))
    cache, rows = {}, []
    for ex in examples:
        db_id = ex["db_id"]; path = find_db(root, db_id)
        if db_id not in cache:
            db = load_db_schema(path, db_id); cache[db_id] = (db, build_value_index(path, db))
        db, idx = cache[db_id]
        pred = answer_question(ex["question"], db, idx, path, gen,
                               evidence=ex.get("evidence", ""),
                               n_candidates=cfg["generation"]["n_candidates"])
        ok_g, gold_rows = execute_sql(path, ex["SQL"])
        correct = ok_g and pred.executes and result_sets_match(
            gold_rows, pred.rows, is_order_sensitive(ex["SQL"]))
        rows.append({"db_id": db_id, "question": ex["question"], "pred": pred.sql,
                     "gold": ex["SQL"], "n_tables": count_tables(ex["SQL"]), "correct": bool(correct)})
    rep = summarize(rows)
    os.makedirs("results", exist_ok=True)
    json.dump({"summary": rep, "rows": rows}, open("results/eval.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
