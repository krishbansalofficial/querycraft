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


def test_set_metrics():
    from querycraft.linker import set_metrics
    m = set_metrics([({"a", "b"}, {"a"}), ({"c"}, {"c", "d"})])
    assert m == {"recall": 0.6667, "precision": 0.6667, "exact": 0.0, "all_gold_kept": 0.5, "mean_predicted": 1.5}
