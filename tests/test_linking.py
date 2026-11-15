from querycraft.prompts import fk_hop

TABLES = [
    {"name": "Orders", "fks": [("cust_id", "Customers", "id")]},
    {"name": "Customers", "fks": []},
    {"name": "Items", "fks": [("order_id", "Orders", "id")]},
    {"name": "Unrelated", "fks": []},
]


def test_fk_hop_adds_neighbours_both_directions_only_one_hop():
    assert fk_hop(TABLES, {"orders"}) == {"orders", "customers", "items"}
    assert fk_hop(TABLES, {"customers"}) == {"customers", "orders"}  # items is two hops away
    assert fk_hop(TABLES, set()) == set()
