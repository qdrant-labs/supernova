"""Unit tests for `nova_sweep.grid` — pure functions, no I/O. Covers the
plan's Verification items #5 (`expand_grid`) and #12 (`order_by_rebuild_cost`).
"""

from __future__ import annotations

from types import SimpleNamespace

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


def test_dict_valued_axis_names_are_index_name_safe():
    # Milvus `index_params` / Elastic `index_options` are dict-valued. Their
    # `_name` becomes part of the collection/ES-index name on the data_layouts
    # axis, so it must render as a `[A-Za-z0-9_]` slug of the dict's contents,
    # NOT the Python repr (whose `{`, `'`, and spaces are illegal index names).
    combos = expand_grid({"index_options": [{"type": "int8_hnsw", "m": 16, "ef_construction": 100}]})
    name = combos[0]["_name"]
    assert name == "index_optionstypeint8_hnsw_m16_ef_construction100"
    assert not (set(name) - set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"))
    # the dict value itself is still set intact for config generation
    assert combos[0]["index_options"] == {"type": "int8_hnsw", "m": 16, "ef_construction": 100}


def test_list_valued_axis_name_is_slugged():
    combos = expand_grid({"multi_probe": [[1, 2, 3]]})
    assert combos[0]["_name"] == "multi_probe1_2_3"


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


def test_rbo_p_and_gt_scores_reach_the_query_block_not_the_load_block():
    """`rbo_p` in a `searches` entry belongs to nova-storm's `query:` block.
    Routed to `load:` (where every unrecognised key goes) it would hit
    `deny_unknown_fields` and fail the run, leaving no way to pin `p` across a
    sweep whose `top_k` varies — which silently mixes two different metrics
    into one parquet column."""
    from nova_sweep.backends.base import build_storm_query
    from nova_sweep.config import QueriesConfig, SweepConfig

    cfg = SimpleNamespace(
        queries=QueriesConfig(
            uri="s3://bucket/queries.parquet",
            column="dense_embedding",
            ground_truth_column="hit_ids",
            ground_truth_score_column="hit_scores",
            limit=100,
        )
    )
    query, load = build_storm_query(
        cfg, {"top_k": 100, "rbo_p": 0.95, "hnsw_ef": 64, "concurrency": 8}, {"hnsw_ef"}
    )

    assert query["rbo_p"] == 0.95
    assert query["top_k"] == 100
    assert "rbo_p" not in load, "would be rejected by nova-storm's load block"
    # The score column is what makes every tie-aware bound (recall's and RBO's)
    # available at all; without it the whole sweep reports point estimates.
    assert query["source"]["ground_truth_score_column"] == "hit_scores"
    assert query["search_params"] == {"hnsw_ef": 64}
    assert load == {"concurrency": 8}


def test_rbo_p_is_optional_and_absent_when_not_swept():
    """Unset, nova-storm derives `p` from `top_k` — the config must not pin it
    to some other default on the way through."""
    from nova_sweep.backends.base import build_storm_query
    from nova_sweep.config import QueriesConfig

    cfg = SimpleNamespace(
        queries=QueriesConfig(uri="s3://b/q.parquet", column="dense_embedding", limit=10)
    )
    query, _ = build_storm_query(cfg, {"top_k": 10}, set())
    assert "rbo_p" not in query
    assert "ground_truth_score_column" not in query["source"]
