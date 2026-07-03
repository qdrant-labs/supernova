"""Unit tests for `nova_sweep.grid` — pure functions, no I/O. Covers the
plan's Verification items #5 (`expand_grid`) and #12 (`order_by_rebuild_cost`).
"""

from __future__ import annotations

from nova_sweep.grid import expand_grid, order_by_rebuild_cost


def test_cartesian_product_size():
    combos = expand_grid({"hnsw.m": [8, 16, 32], "hnsw.ef_construct": [10, 100, 1000]})
    assert len(combos) == 9


def test_dotted_path_builds_nested_dict():
    combos = expand_grid({"hnsw.m": [8]})
    assert combos[0]["hnsw"] == {"m": 8}


def test_deterministic_auto_naming():
    combos = expand_grid({"hnsw.m": [8], "hnsw.ef_construct": [100]})
    assert combos[0]["_name"] == "m8_ef_construct100"


def test_empty_grid_is_a_single_default_combination():
    combos = expand_grid({})
    assert combos == [{"_name": "default"}]


def test_null_prunes_the_key_entirely():
    # A real YAML `null` (Python None) means "omit this key", not "set it to None".
    combos = expand_grid({"hnsw_ef": [None, 128]})
    assert combos[0] == {"_name": "default"}
    assert "hnsw_ef" not in combos[0]
    assert combos[1] == {"hnsw_ef": 128, "_name": "hnsw_ef128"}


def test_string_none_is_a_real_value_not_a_pruning_sentinel():
    # nova-load's own quantization vocabulary has a literal `none` VALUE
    # (`quantization.type: none` explicitly clears quantization on reindex) —
    # the string "none" must pass through unchanged, unlike Python None/YAML null.
    combos = expand_grid({"quantization.type": ["none", "int8"]})
    assert combos[0]["quantization"] == {"type": "none"}
    assert combos[0]["_name"] == "typenone"
    assert combos[1]["quantization"] == {"type": "int8"}


def test_order_by_rebuild_cost_groups_expensive_hnsw_combinations():
    combos = expand_grid(
        {
            "quantization.type": ["none", "int8"],
            "hnsw.m": [8, 16, 32],
            "hnsw.ef_construct": [10, 100, 1000],
        }
    )
    ordered = order_by_rebuild_cost(combos)

    assert len(ordered) == len(combos) == 18
    assert set(map(_freeze, ordered)) == set(map(_freeze, combos))  # same set of points

    # Every combination sharing the same (m, ef_construct) must be contiguous.
    seen_hnsw_groups = []
    for combo in ordered:
        key = (combo["hnsw"]["m"], combo["hnsw"]["ef_construct"])
        if not seen_hnsw_groups or seen_hnsw_groups[-1] != key:
            assert key not in seen_hnsw_groups, "an hnsw group was split, not contiguous"
            seen_hnsw_groups.append(key)


def test_order_by_rebuild_cost_is_a_noop_without_expensive_keys():
    combos = expand_grid({"quantization.type": ["none", "int8"], "quantization.always_ram": [True, False]})
    ordered = order_by_rebuild_cost(combos)
    assert ordered == combos  # stable sort, identical (empty) cost key for every combo


def test_order_by_rebuild_cost_is_stable():
    combos = [{"hnsw": {"m": 8}, "_name": "a"}, {"hnsw": {"m": 8}, "_name": "b"}]
    ordered = order_by_rebuild_cost(combos)
    assert [c["_name"] for c in ordered] == ["a", "b"]


def _freeze(d: dict) -> str:
    return repr(sorted(d.items()))
