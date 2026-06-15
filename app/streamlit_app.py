# app/streamlit_app.py
from __future__ import annotations
import glob, os, yaml
import streamlit as st
from src.data.schema_loader import load_db_schema
from src.data.value_index import build_value_index
from src.routing.db_router import DBRouter
from src.pipeline import answer_question
from src.generate.generator import Generator

cfg = yaml.safe_load(open("configs/pipeline.yaml"))
ROOT = cfg["paths"]["bird_root"]

@st.cache_resource
def load_everything():
    paths = {os.path.basename(os.path.dirname(p)): p
             for p in glob.glob(os.path.join(ROOT, "**", "*.sqlite"), recursive=True)}
    dbs = {k: load_db_schema(v, k) for k, v in paths.items()}
    idx = {k: build_value_index(paths[k], dbs[k]) for k in paths}
    summaries = {k: " ".join([k] + [c.name for t in d.tables.values() for c in t.columns])
                 for k, d in dbs.items()}
    gen = Generator(cfg["model"]["base"], cfg["model"].get("adapter"))
    return paths, dbs, idx, DBRouter(summaries, _Embedder()), gen

class _Embedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(cfg["embedder"])
    def encode(self, texts): return self.m.encode(list(texts))

st.title("Join-Robust Text-to-SQL (BIRD)")
paths, dbs, idx, router, gen = load_everything()
q = st.text_area("Ask a question", "Among clients from Portland, how many billing disputes?")
override = st.selectbox("Database (auto-detected; override if needed)",
                        ["(auto)"] + list(paths))
if st.button("Generate SQL and run"):
    db_id = router.route(q) if override == "(auto)" else override
    st.caption(f"Routed to database: **{db_id}**")
    res = answer_question(q, dbs[db_id], idx[db_id], paths[db_id], gen,
                          n_candidates=cfg["generation"]["n_candidates"])
    st.code(res.sql, language="sql")
    if res.executes:
        st.dataframe(res.rows)
    else:
        st.error(res.error)
