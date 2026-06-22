from __future__ import annotations
import random
from src.common.sql import count_tables

def split_by_table_count(examples: list[dict], sql_key: str = "SQL"):
    singles, twos, more = [], [], []
    for ex in examples:
        try:
            n = count_tables(ex[sql_key])
        except Exception:
            continue
        ex = {**ex, "n_tables": n}
        (singles if n <= 1 else twos if n == 2 else more).append(ex)
    return singles, twos, more

def build_subset(examples, sql_key="SQL", two_table_ratio=0.7, seed=0):
    singles, twos, _ = split_by_table_count(examples, sql_key)
    rng = random.Random(seed)
    rng.shuffle(singles); rng.shuffle(twos)
    if not twos:
        return singles
    n_single = int(len(twos) * (1 - two_table_ratio) / two_table_ratio)
    out = twos + singles[:n_single]
    rng.shuffle(out)
    return out

def train_eval_split(examples, eval_frac=0.15, seed=0):
    """Disjoint (train, eval) split so the model is never evaluated on a row it
    was trained on. Deterministic for a given seed."""
    items = list(examples)
    random.Random(seed).shuffle(items)
    n_eval = max(1, round(len(items) * eval_frac))
    return items[n_eval:], items[:n_eval]

def main():
    import json, yaml, glob, os
    cfg = yaml.safe_load(open("configs/pipeline.yaml"))
    root = cfg["paths"]["bird_root"]
    dev = json.load(open(glob.glob(os.path.join(root, "**", "dev.json"), recursive=True)[0]))
    sub = build_subset(dev, sql_key="SQL", two_table_ratio=0.7, seed=0)
    train, evals = train_eval_split(sub, eval_frac=0.15, seed=0)
    for key, rows in (("subset_train_raw", train), ("subset_eval", evals)):
        with open(cfg["paths"][key], "w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {len(train)} train -> {cfg['paths']['subset_train_raw']}, "
          f"{len(evals)} eval -> {cfg['paths']['subset_eval']}")

if __name__ == "__main__":
    main()
