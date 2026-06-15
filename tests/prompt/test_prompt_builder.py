from src.linking.schema_linker import LinkedSchema
from src.common.schema import ForeignKey
from src.prompt.prompt_builder import build_prompt, suggest_join_path

LS = LinkedSchema(
    db_id="retail",
    tables={"client": ["client_id", "city"], "events": ["event_id", "Client_ID", "Issue"]},
    foreign_keys=[ForeignKey("events", "Client_ID", "client", "client_id")],
)

def test_prompt_contains_schema_fk_values_and_joinpath():
    p = build_prompt("how many Billing disputes from Portland", "", LS,
                     linked_values=[("Portland", "client", "city")])
    assert "client(" in p and "events(" in p
    assert "Foreign keys:" in p
    assert "Portland -> client.city" in p
    assert "Suggested join: events.Client_ID = client.client_id" in p

def test_suggest_join_path_single_fk():
    assert suggest_join_path(LS) == "events.Client_ID = client.client_id"
