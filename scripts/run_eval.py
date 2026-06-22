from __future__ import annotations
import argparse, json, glob, os, yaml, sys
# Allow `python scripts/run_eval.py` to import the top-level `src` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N examples (0 = all)")
    ap.add_argument("--progress", default="results/eval_progress.jsonl",
                    help="incremental results file; a re-run resumes from it. "
                         "Point this at Google Drive to survive Colab disconnects.")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = cfg["paths"]["bird_root"]
    examples = [json.loads(l) for l in open(cfg["paths"]["subset_eval"], encoding="utf-8")]
    if args.limit:
        examples = examples[:args.limit]
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, **k): return x
    gen = Generator(cfg["model"]["base"], cfg["model"].get("adapter"))

    # Embedding-based schema linker (falls back to lexical if unavailable).
    embedder = None
    if cfg.get("embedder"):
        try:
            from src.common.embedder import load_embedder
            embedder = load_embedder(cfg["embedder"])
            print(f"schema linking: embedding model {cfg['embedder']}")
        except Exception as e:
            print(f"embedder unavailable ({e}); using lexical schema linking")

    # Resume: load any results already computed in a previous (e.g. disconnected) run.
    os.makedirs(os.path.dirname(args.progress) or ".", exist_ok=True)
    done = {}
    if os.path.exists(args.progress):
        for line in open(args.progress, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line); done[r["idx"]] = r
    if done:
        print(f"resuming: {len(done)} examples already done, skipping them")

    cache = {}
    pf = open(args.progress, "a", encoding="utf-8")
    for ex_i, ex in enumerate(tqdm(examples, desc="eval")):
        if ex_i in done:
            continue
        try:
            n_tab = count_tables(ex.get("SQL", ""))
        except Exception:
            n_tab = 1
        try:
            db_id = ex["db_id"]; path = find_db(root, db_id)
            if db_id not in cache:
                db = load_db_schema(path, db_id); cache[db_id] = (db, build_value_index(path, db))
            db, vindex = cache[db_id]
            dbg = {}
            pred = answer_question(ex["question"], db, vindex, path, gen,
                                   evidence=ex.get("evidence", ""),
                                   n_candidates=cfg["generation"]["n_candidates"],
                                   debug=dbg, embedder=embedder)
            ok_g, gold_rows = execute_sql(path, ex["SQL"])
            correct = ok_g and pred.executes and result_sets_match(
                gold_rows, pred.rows, is_order_sensitive(ex["SQL"]))
            row = {"idx": ex_i, "db_id": db_id, "question": ex["question"], "pred": pred.sql,
                   "gold": ex["SQL"], "n_tables": n_tab, "correct": bool(correct),
                   # Triage fields: tells you whether a wrong answer was a linker miss
                   # (gold table absent from linked_tables), a value-link miss, or the
                   # model generating bad SQL from a correct prompt.
                   "linked_tables": dbg.get("linked_tables", []),
                   "linked_values": dbg.get("linked_values", []),
                   "n_valid": dbg.get("n_valid", 0),
                   "prompt": dbg.get("prompt", "")}
        except Exception as e:
            # Never let one bad example abort a multi-hour run; record and move on.
            row = {"idx": ex_i, "db_id": ex.get("db_id"), "question": ex.get("question"),
                   "pred": "", "gold": ex.get("SQL", ""), "n_tables": n_tab,
                   "correct": False, "error": str(e)}
        pf.write(json.dumps(row, ensure_ascii=False) + "\n"); pf.flush()
        done[ex_i] = row
    pf.close()

    rows = [done[i] for i in sorted(done)]
    rep = summarize(rows)
    os.makedirs("results", exist_ok=True)
    json.dump({"summary": rep, "rows": rows}, open("results/eval.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
