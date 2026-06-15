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

def main():
    import json, yaml, glob, os
    cfg = yaml.safe_load(open("configs/pipeline.yaml"))
    root = cfg["paths"]["bird_root"]
    dev = json.load(open(glob.glob(os.path.join(root, "**", "dev.json"), recursive=True)[0]))
    sub = build_subset(dev, sql_key="SQL", two_table_ratio=0.7, seed=0)
    with open(cfg["paths"]["subset_eval"], "w", encoding="utf-8") as f:
        for ex in sub:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {len(sub)} examples")

if __name__ == "__main__":
    main()
