# Join-Robust Text-to-SQL on BIRD (lightweight SLM)

Natural-language → SQL on the [BIRD](https://bird-bench.github.io/) benchmark with a
**1.5B** model (Qwen2.5-Coder, QLoRA on a single 16 GB GPU). Unlike naive pipelines,
this system **does not collapse on multi-table queries**: foreign keys are engineered
into grounding, generation, validation, and selection, and the database/tables are
**auto-detected** (no manual picking).

## Architecture
```
question → DB router → schema linker (+FK graph) → FK-grounded prompt
        → Qwen2.5-Coder-1.5B (QLoRA) → query fixer → validator (FK-validity + execution)
        → execution-guided selector → answer
```

## Results (real BIRD, held-out, execution accuracy)
Trained on 1,099 examples, evaluated on a **disjoint, held-out** 194-example set
(no train/test leakage). 1.5B model, QLoRA, fully automated (no manual DB/table pick).

| Query type | This system | Naive baseline* |
|---|---|---|
| Single-table | 27.5% (14/51) | ~59% |
| **Multi-table (joins)** | **16.8% (24/143)** | **1.2%** |
| Overall | 19.6% (38/194) | — |

\*Naive baseline = the reference CodeT5-Small pipeline, measured on its own easier
8-DB set with manual DB/table selection. Absolute numbers are **not** directly
comparable (different, easier data + a human picking tables). The point is the
**join behaviour**: the baseline *collapses* on joins (59.6% → 1.2%, ~50×), while
this system degrades gracefully (27.5% → 16.8%, ~1.6×) — multi-table accuracy is
~14× the baseline's. Absolute accuracy is modest because the generator is a 1.5B
model fine-tuned on ~1k examples under free-tier compute; more training data
(full BIRD `train.json`) is the clear lever to raise it.

## Quickstart
```bash
pip install -r requirements-dev.txt   # local logic + tests
pytest -q                             # all unit tests
# Full pipeline (GPU/Colab):
pip install -r requirements.txt
make data && make eval && make demo
```
Training runs on free Colab: see `notebooks/02_qlora_train.ipynb`.

## How it differs from a naive pipeline
| Stage | Naive | This system |
|---|---|---|
| Schema scope | manual DB+table pick | auto router + schema linker |
| Grounding | values only | + explicit FK graph + join path |
| Generator | CodeT5-Small 60M | Qwen2.5-Coder-1.5B QLoRA |
| Self-correction | none | query fixer (execution feedback) |
| Validation | syntax only | + FK-validity filter + execution |
| Metric | exact match | execution accuracy (single/multi split) |

## License
MIT
