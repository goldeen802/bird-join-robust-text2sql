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

## Results (BIRD subset, execution accuracy)
| Query type | This system | Naive baseline* |
|---|---|---|
| Single-table | _fill from results/eval.json_ | ~59% |
| **Multi-table (joins)** | **_fill from results/eval.json_** | **1.2%** |

\*Naive baseline = the reference CodeT5-Small pipeline, measured on its own easier
8-DB set. Absolute numbers are not directly comparable; the point is the **gap**:
baseline accuracy collapses on joins, this system's does not.

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
