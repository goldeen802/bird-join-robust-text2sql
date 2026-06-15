# Join-Robust Text-to-SQL on BIRD with a Lightweight SLM — Design Spec

- **Date:** 2026-06-15
- **Status:** Approved (design); implementation plan to follow
- **Author:** Golden Wong (goldenwong000@gmail.com)
- **Suggested repo name:** `bird-join-robust-text2sql`

---

## 1. Goal & success criteria

Build, from scratch, a Text-to-SQL system on a tractable slice of the **real BIRD** benchmark that **does not collapse on multi-table (join) queries**, runs on a **lightweight, fine-tunable SLM under restricted compute** (free Colab / ~16 GB T4), and **removes the manual database/table selection** present in the reference project.

**Primary success metric — the headline result:**
- Report **execution accuracy split by single-table vs multi-table** on a held-out BIRD subset.
- The reference project collapses from **59.6% (single-table) → 1.2% (joins)**. Success = a **small single-vs-multi gap**, with multi-table execution accuracy realistically in the **~25–40%** band. Any double-digit join accuracy already decisively beats the 1.2% baseline.

**Honest comparison caveat (state this everywhere):** the reference project evaluated on its *own easy 8-database augmented set*; we evaluate on *real BIRD*. Absolute numbers are not strictly comparable. The defensible claim is **structural**: *their accuracy collapses the moment a join appears; ours does not, because joins are engineered into every stage.*

## 2. Non-goals (explicitly out of scope for the 1-week build)

- Custom NER training, LambdaMART reranker training (both need their own labeled data).
- Full 95-database BIRD; VES efficiency scoring.
- 3+ table joins (cap at 2-table / single-join for v1).
- Reinforcement learning (Arctic-style execution-RL) — noted as future work, not v1.
- Any hosted-LLM/API path — everything runs locally / on free Colab.

## 3. Background: the reference system and its measured weaknesses

The reference project (Database-Agnostic Text-to-SQL Analytics) pipeline: Streamlit UI with **manual DB + table selection** → spaCy NER intent → value/schema grounding → **CodeT5-Small (60M)** generates ~30 candidates → SQLGlot + DuckDB dry-run validation → **LightGBM LambdaMART** rerank → execute.

Measured weaknesses this design targets:

| Weakness | Evidence | Root cause |
|---|---|---|
| Joins collapse | 59.6% single-table → **1.2% joins** | Training data averaged **1 table/query**; join path never given to the model; invalid-FK joins accepted |
| Filter-change hallucination | "Portland" works, "Chicago" breaks (slide 31) | Stale/empty grounding when a value isn't indexed; suspected cache/prompt reuse |
| Manual schema selection | UI requires picking DB + tables | The human was doing the model's schema-linking |
| Candidate ceiling | Upper bound **48%** | 60M generator caps everything downstream, incl. the reranker |
| Misleading metric | String/SQLGlot exact match | Marks correct-but-different SQL as wrong (project admits this) |

## 4. Architecture

```
user question  (no DB/table picking)
   │
   ▼
[A] DB ROUTER        embed question vs each DB schema summary → select the database
   ▼
[B] SCHEMA LINKER    score tables/columns for relevance (embeddings + value-index) →
   │                 keep the relevant few; build the FK graph for kept tables
   ▼
[C] PROMPT BUILDER   pruned schema + FK relationships + linked values +
   │                 suggested join path + BIRD evidence + SQL rules
   ▼
[D] GENERATOR        Qwen2.5-Coder-1.5B-Instruct (QLoRA) → N diverse candidates
   ▼
[E] QUERY FIXER      execute candidate; on error, feed SQLite error back → regenerate (1–2 rounds)
   ▼
[F] VALIDATOR        SQLGlot parse → reject unknown columns → reject non-FK JOIN…ON → dry-run execute
   ▼
[G] SELECTOR         execution-guided + FK-validity heuristic → final query
   ▼
answer + execution-accuracy eval (single vs multi-table split)
```

**Design throughline:** the system performs the schema-linking the user used to do by hand, and treats **foreign keys as first-class objects at every stage** — grounding [C], generation [D], validation [F], selection [G]. That is the spine of the join win.

### Component detail

- **[A] DB Router** — precompute an embedding per database from its serialized schema (table + column names, descriptions). Embed the question (local `bge-small` / `all-MiniLM`), retrieve top-1 DB. Fallback: keep a manual DB override in the UI in case routing is wrong.
- **[B] Schema Linker** — for the routed DB, rank columns by (embedding similarity to question) + (value-index hits). Keep generous top-k tables/columns (favor **recall** — never prune away a needed table). Build the FK subgraph over kept tables.
- **[C] Prompt Builder** — single canonical template, **byte-identical at train and inference** (prevents the stale-grounding bug). Contains: question, BIRD `evidence`, pruned schema (`table(col:type, …)`), `Foreign keys:` block, `Linked values:` (`Portland → client.city`), `Suggested join: A.x = B.y` when 2 tables, short SQL rules.
- **[D] Generator** — Qwen2.5-Coder-1.5B-Instruct, QLoRA (4-bit). Generate N candidates via temperature sampling + 2 prompt styles (plain + query-plan-style). Fallback: few-shot the instruct base with no fine-tune.
- **[E] Query Fixer** — run each candidate against the real SQLite DB; if it errors, append the error message to the prompt and regenerate (cap 1–2 rounds). Optional/skippable under time pressure.
- **[F] Validator** — SQLGlot parse (`dialect=sqlite`); reject references to non-existent columns; **reject any `JOIN…ON` pair that is not a real foreign key**; execute as a dry-run.
- **[G] Selector** — among valid+executable candidates, score by: valid FK join (+), value-links satisfied (+), uses linked columns (+), fewer tables/clauses (+). Pick top. No trained reranker in v1.

## 5. Dataset plan

- **Source:** real BIRD (train + dev), SQLite-native. Use `sqlite3` directly for execution; SQLGlot for parsing only.
- **Eval:** BIRD **mini-dev (500 ex)**, results reported split by single-table vs multi-table.
- **Train:** filter BIRD train to **~70% two-table (single-join) + 30% single-table** queries; cap to a manageable set of databases with clean FK structure. (Table count per query is derivable by parsing the gold SQL with SQLGlot.)
- **Value index:** precompute per-DB distinct TEXT-column values (capped per column), normalized, for value-linking.
- Include BIRD's **`evidence`** field in every prompt (free external-knowledge signal).
- Materialized artifacts checked into the repo (or scripted to rebuild): `data/subset_train.jsonl`, `data/subset_eval.jsonl`, `data/value_index/`, `data/schema_fk/`.

## 6. Model & training

- **Base:** `Qwen2.5-Coder-1.5B-Instruct` (Apache-2.0). Stretch: 3B if T4 headroom allows.
- **Method:** QLoRA (4-bit bitsandbytes), **fp16** (T4 has no bf16), **standard attention** (no flash-attn on Turing), gradient checkpointing, small batch + grad accumulation.
- **Run:** 2–3 epochs over the filtered subset (~1–3 h); **checkpoint frequently** (free Colab disconnects). Save LoRA adapters to the repo / HF Hub.
- **Reproducibility:** fixed seed, `train.py` + a `configs/qlora.yaml`, and a Colab notebook that runs end-to-end.

## 7. Evaluation

- **Execution accuracy** harness: execute gold SQL and predicted SQL on the SQLite DB; compare result sets (order-insensitive unless the query has `ORDER BY`).
- **Always report the single- vs multi-table split** — this is the headline.
- Secondary diagnostics: candidate upper-bound (oracle over the pool), % parse-valid, % FK-valid, % executable, query-fixer recovery rate.
- A small fixed **regression set** of 2-table questions tracked across iterations.

## 8. GitHub & resume readiness (required)

Repo is built to be browsed by a recruiter and reproduced by a stranger.

**Layout:**
```
bird-join-robust-text2sql/
├── README.md                 # architecture diagram, results table, quickstart, comparison vs baseline
├── LICENSE                   # MIT
├── requirements.txt / pyproject.toml
├── Makefile                  # make data | make train | make eval | make demo
├── configs/                  # qlora.yaml, pipeline.yaml
├── src/
│   ├── routing/              # [A] db router
│   ├── linking/              # [B] schema linker + value index + FK graph
│   ├── prompt/               # [C] prompt builder (single canonical template)
│   ├── generate/             # [D] generator + candidate sampling
│   ├── fix/                  # [E] query fixer
│   ├── validate/             # [F] sqlglot + FK-validity + execution dry-run
│   ├── select/               # [G] selector
│   └── eval/                 # execution-accuracy harness
├── notebooks/                # 01_data_prep, 02_qlora_train, 03_eval (Colab-runnable)
├── tests/                    # FK-validity, schema-linker, eval harness unit tests
├── results/                  # eval reports (json/md), screenshots, comparison table
└── app/                      # streamlit demo (auto DB detection, no manual picker)
```

**README must contain:**
- One-paragraph pitch + the architecture diagram (Section 4).
- **Results table**: single- vs multi-table execution accuracy, with the reference 59.6%→1.2% collapse shown alongside for context (with the comparison caveat stated).
- Quickstart: `pip install -r requirements.txt` → `make data` → `make eval` (and Colab badge for training).
- A short "How it differs from a naive pipeline" section (the Section 9 table).
- Screenshots/GIF of the Streamlit demo answering a join question end-to-end.

**Resume-bullet framing (the artifact to optimize for):**
> *Built a join-robust Text-to-SQL pipeline on the BIRD benchmark using a 1.5B SLM (Qwen2.5-Coder, QLoRA on a single 16 GB GPU). Replaced manual schema selection with automatic DB-routing + schema-linking, and engineered foreign keys into grounding, generation, validation, and selection — closing the single-vs-multi-table accuracy gap that collapses naive pipelines (59% → 1.2%) to a small margin.*

## 9. What makes it better (mapped to measured failures)

| Stage | Reference | Proposed | Why better |
|---|---|---|---|
| Schema scope | Manual DB + table pick | Auto DB-router + schema-linker | Automates the model's hardest job; raises difficulty honestly |
| Intent | spaCy NER (needs labels) | Embedding schema-linking | No annotation cost; better recall on unseen phrasing |
| Grounding | value + schema | + explicit FK graph + join path | Joins fail because the path is never given; we give it |
| Generator | CodeT5-Small 60M | Qwen2.5-Coder-1.5B QLoRA | 25× bigger, code+instruction tuned; lifts the 48% ceiling |
| Self-correction | none | Query fixer (exec feedback) | Repairs broken candidates instead of discarding |
| Validation | SQLGlot + dry-run | + FK-validity filter | Rejects hallucinated non-FK joins (slide-23 failures) |
| Selection | LambdaMART | execution-guided + FK heuristic | No training; targets joins; ceiling-bound reranker avoided |
| Metric | exact match | execution accuracy (split) | Honest; BIRD's real metric |
| Train data | avg 1 table/query | join-heavy (~70% 2-table) | Fixes the root cause of 1.2% joins |

## 10. Timeline (1 week) — spine vs stretch

| Day | Deliverable |
|---|---|
| 1 | Env, BIRD download, **execution-accuracy harness**, schema/FK loader, value index, DB-router + schema-linker (retrieval, no training), materialized train/eval subsets |
| 2 | Prompt builder (canonical template); generate grounded training data; QLoRA scaffold |
| 3 | QLoRA fine-tune Qwen2.5-Coder-1.5B; candidate generation |
| 4 | Query fixer + validator (FK-validity + dry-run) + selector |
| 5 | Full eval on mini-dev, single-vs-multi split; iterate on grounding |
| 6 | Streamlit demo (auto DB detection); README + results artifacts |
| 7 | Buffer / debugging / comparison write-up |

**Guaranteed spine:** BIRD subset + eval harness, auto DB-router + schema-linker, Qwen generator (few-shot or QLoRA), FK-grounded prompt + FK-validity + dry-run + heuristic selection, minimal demo, README with the split-accuracy result.

**Stretch (graceful fallbacks):** QLoRA fine-tune (fallback: few-shot base), query-fixer (fallback: skip), 3B model, learned pairwise selector.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| QLoRA training eats the week | Instruct base is strong few-shot; demo works without fine-tune |
| Auto DB-routing misroutes | Keep manual DB override in UI as backup; routing is a thin retrieval layer |
| Schema-linker prunes a needed table | Favor recall (generous top-k); validate linker on the regression set |
| Free Colab disconnects mid-train | Frequent checkpointing; small subset keeps runs short |
| Over-investing in stretch before spine is solid | Plan gates stretch items behind a working spine |

## 12. Test strategy

- **Day-1 first deliverable = the execution-accuracy harness** (everything is measured against it).
- Unit tests: FK-validity checker, schema-linker recall on known cases, eval result-set comparison.
- Fixed 2-table regression set tracked every iteration.
