from __future__ import annotations
from src.data.value_index import link_values
from src.linking.schema_linker import link_schema
from src.prompt.prompt_builder import build_prompt
from src.validate.validator import validate, ValidationResult
from src.select.selector import select_best

def answer_question(question, db, value_index, db_path, generator,
                    evidence: str = "", n_candidates: int = 8) -> ValidationResult:
    links = link_values(question, value_index)
    ls = link_schema(question, db, links, embedder=None)
    prompt = build_prompt(question, evidence, ls, links)
    candidates = generator.generate(prompt, n=n_candidates)
    results = [validate(sql, db, db_path) for sql in candidates if sql.strip()]
    return select_best(results, links)
