from __future__ import annotations
from src.linking.schema_linker import LinkedSchema

RULES = (
    "Rules: use only the tables and columns above; join only on the listed "
    "foreign keys; use COUNT/AVG/SUM as the question implies; output one SQL query only."
)

def suggest_join_path(ls: LinkedSchema) -> str:
    if len(ls.tables) >= 2 and ls.foreign_keys:
        fk = ls.foreign_keys[0]
        return f"{fk.from_table}.{fk.from_col} = {fk.to_table}.{fk.to_col}"
    return ""

def build_prompt(question: str, evidence: str, ls: LinkedSchema,
                 linked_values: list[tuple[str, str, str]]) -> str:
    lines = ["Translate the question to a single SQLite SQL query.", ""]
    lines.append(f"Question: {question}")
    if evidence:
        lines.append(f"Evidence: {evidence}")
    lines.append("")
    lines.append("Schema:")
    for t, cols in ls.tables.items():
        lines.append(f"  {t}({', '.join(cols)})")
    if ls.foreign_keys:
        lines.append("Foreign keys:")
        for fk in ls.foreign_keys:
            lines.append(f"  {fk.from_table}.{fk.from_col} -> {fk.to_table}.{fk.to_col}")
    if linked_values:
        lines.append("Linked values:")
        for v, t, c in linked_values:
            lines.append(f"  {v} -> {t}.{c}")
    jp = suggest_join_path(ls)
    if jp:
        lines.append(f"Suggested join: {jp}")
    lines.append("")
    lines.append(RULES)
    lines.append("SQL:")
    return "\n".join(lines)
