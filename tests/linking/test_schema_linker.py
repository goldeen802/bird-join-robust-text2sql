import numpy as np
from src.data.schema_loader import load_db_schema
from src.linking.schema_linker import link_schema, LinkedSchema
from src.common.schema import DBSchema, TableSchema, Column


class FakeEmbedder:
    """Deterministic 2-concept embedder: a 'buying' axis and a 'geo' axis.
    Note 'orders' shares NO tokens with 'purchase', so only the embedding
    path (not lexical overlap) can connect the question to the right table."""
    def encode(self, texts):
        vecs = []
        for t in texts:
            tl = t.lower()
            buy = float(any(w in tl for w in ("purchase", "bought", "order")))
            geo = float(any(w in tl for w in ("city", "country", "location")))
            vecs.append([buy, geo])
        return np.asarray(vecs, dtype=float)


def _shop_db():
    orders = TableSchema("orders", [Column("orders", "order_id", "INT"),
                                    Column("orders", "amount", "REAL")])
    geo = TableSchema("geography", [Column("geography", "city", "TEXT"),
                                    Column("geography", "country", "TEXT")])
    return DBSchema("shop", {"orders": orders, "geography": geo}, [])


def test_embedder_ranks_semantically_relevant_table():
    db = _shop_db()
    # No token overlap between "purchase" and table/column names; lexical
    # scoring ties at 0, so only embedding ranking can pick the right table.
    ls = link_schema("what did the customer purchase", db, [],
                     embedder=FakeEmbedder(), top_k_tables=1)
    assert "orders" in ls.tables
    assert "geography" not in ls.tables


def test_keeps_value_linked_tables_and_fk(tiny_db):
    db = load_db_schema(tiny_db, "tiny")
    linked_values = [("Portland", "client", "city"), ("Billing disputes", "events", "Issue")]
    ls = link_schema("how many Billing disputes from Portland",
                     db, linked_values, embedder=None, top_k_tables=4)
    assert isinstance(ls, LinkedSchema)
    # both value-linked tables retained
    assert {"client", "events"} <= set(ls.tables)
    # the FK between them is included
    assert ls.foreign_keys  # non-empty
