from __future__ import annotations
import json, os, glob, yaml
from src.data.schema_loader import load_db_schema
from src.data.value_index import build_value_index, link_values
from src.linking.schema_linker import link_schema
from src.prompt.prompt_builder import build_prompt

def make_training_record(ex: dict, db, value_index, embedder=None) -> dict:
    q = ex["question"]
    links = link_values(q, value_index)
    ls = link_schema(q, db, links, embedder=embedder)
    prompt = build_prompt(q, ex.get("evidence", ""), ls, links)
    return {"db_id": ex["db_id"], "prompt": prompt, "target": ex["SQL"]}

def _find_db_path(bird_root: str, db_id: str) -> str:
    hits = glob.glob(os.path.join(bird_root, "**", db_id, f"{db_id}.sqlite"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"no sqlite for {db_id}")
    return hits[0]

def main():
    cfg = yaml.safe_load(open("configs/pipeline.yaml"))
    root = cfg["paths"]["bird_root"]
    examples = [json.loads(l) for l in open(cfg["paths"]["subset_eval"], encoding="utf-8")]
    cache: dict[str, tuple] = {}
    out_path = cfg["paths"]["subset_train"]
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            db_id = ex["db_id"]
            if db_id not in cache:
                path = _find_db_path(root, db_id)
                db = load_db_schema(path, db_id)
                cache[db_id] = (db, build_value_index(path, db))
            db, idx = cache[db_id]
            f.write(json.dumps(make_training_record(ex, db, idx, embedder),
                               ensure_ascii=False) + "\n")
    print(f"wrote training records -> {out_path}")

if __name__ == "__main__":
    main()
