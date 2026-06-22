from __future__ import annotations
from src.data.value_index import link_values
from src.linking.schema_linker import link_schema
from src.prompt.prompt_builder import build_prompt
from src.validate.validator import validate, ValidationResult
from src.select.selector import select_best

def answer_question(question, db, value_index, db_path, generator,
                    evidence: str = "", n_candidates: int = 8,
                    debug: dict | None = None, embedder=None) -> ValidationResult:
    links = link_values(question, value_index)
    ls = link_schema(question, db, links, embedder=embedder)
    prompt = build_prompt(question, evidence, ls, links)
    candidates = generator.generate(prompt, n=n_candidates)
    results = [validate(sql, db, db_path) for sql in candidates if sql.strip()]
    best = select_best(results, links)
    if debug is not None:
        # Filled in-place for failure triage (which stage went wrong); no effect
        # on the return value, so callers/tests that ignore `debug` are unchanged.
        debug["prompt"] = prompt
        debug["linked_tables"] = list(ls.tables.keys())
        debug["linked_values"] = [[v, t, c] for v, t, c in links]
        debug["n_candidates"] = len(candidates)
        debug["n_valid"] = sum(1 for r in results if r.is_valid)
    return best
