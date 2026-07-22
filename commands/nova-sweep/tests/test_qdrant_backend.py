"""Unit tests for the Qdrant sweep backend (`nova_sweep.backends.qdrant`):
the generated `nova-load`/`nova-storm` config shapes, and that
`nova_sweep.backends.get_backend` actually dispatches to it. No subprocess,
no live Qdrant.
"""

from __future__ import annotations

import pytest

from nova_sweep.backends import get_backend
from nova_sweep.backends.qdrant import QdrantBackend, QdrantTargetConfig
from nova_sweep.config import CorpusConfig, OutputConfig, QueriesConfig, SweepConfig
from nova_sweep.slices import build_slices


def _cfg(**axes) -> SweepConfig:
    return SweepConfig(
        collection_name="mycollection",
        corpus=CorpusConfig(path="/tmp/corpus", dense_column="dense_embedding"),
        queries=QueriesConfig(
            uri="/tmp/q.parquet", column="dense_embedding", ground_truth_column="hit_ids"
        ),
        target={"type": "qdrant", "url": "http://localhost:6334"},
        output=OutputConfig(path="/tmp/out"),
        **axes,
    )


def _one_slice(cfg: SweepConfig):
    return build_slices(cfg)[0]


# --- backend dispatch --------------------------------------------------------


def test_get_backend_dispatches_qdrant_type_to_qdrant_backend():
    cfg = _cfg()
    assert isinstance(cfg.target, QdrantTargetConfig)
    assert isinstance(get_backend(cfg.target.type), QdrantBackend)


# --- generated nova-load config shapes --------------------------------------


def test_load_config_overlays_data_layout_vectors_and_sets_recreate():
    cfg = _cfg(data_layouts={"vectors.dense.datatype": ["uint8"], "vectors.dense.distance": ["dot"]})
    slc = _one_slice(cfg)
    backend = get_backend(cfg.target.type)
    load_cfg = backend.build_load_config(cfg, slc, recreate=True)

    assert load_cfg["vectors"]["dense"]["column"] == "dense_embedding"
    assert load_cfg["vectors"]["dense"]["datatype"] == "uint8"
    assert load_cfg["vectors"]["dense"]["distance"] == "dot"
    assert load_cfg["vectorstore"]["type"] == "qdrant"
    assert load_cfg["vectorstore"]["collection_name"] == slc.collection_name
    assert load_cfg["vectorstore"]["params"] == {"recreate": True}
    assert load_cfg["datasource"]["id_expression"] == "vf_point_id(filename, file_row_number)"


def test_load_config_carries_structural_data_layout_params_beyond_vectors_dense():
    """`shard_number`/`replication_factor`/etc. are collection-wide create-time
    params that live under `vectorstore.params` (see `QdrantParams` in
    `crates/nova-load/src/stores/qdrant.rs`), not under `vectors.dense` — a
    `data_layouts` axis on one of these must still reach the generated config,
    not just `vectors.dense.*` overrides."""
    cfg = _cfg(data_layouts={"shard_number": [4], "on_disk_payload": [True]})
    slc = _one_slice(cfg)
    backend = get_backend(cfg.target.type)

    load_cfg = backend.build_load_config(cfg, slc, recreate=False)

    assert load_cfg["vectorstore"]["params"]["shard_number"] == 4
    assert load_cfg["vectorstore"]["params"]["on_disk_payload"] is True
    assert load_cfg["vectorstore"]["params"]["recreate"] is False


def test_load_config_raises_a_clear_error_for_a_malformed_vectors_axis():
    """A `data_layouts` axis on the bare reserved `vectors` key (missing the
    `.dense.<field>` suffix) must fail with a clear message, not an opaque
    AttributeError from `dict.get` deep inside config generation."""
    cfg = _cfg(data_layouts={"vectors": ["oops"]})
    slc = _one_slice(cfg)
    backend = get_backend(cfg.target.type)

    with pytest.raises(ValueError, match="`vectors` must expand to a mapping"):
        backend.build_load_config(cfg, slc, recreate=False)


def test_load_config_raises_a_clear_error_for_a_malformed_vectors_dense_axis():
    cfg = _cfg(data_layouts={"vectors.dense": ["oops"]})
    slc = _one_slice(cfg)
    backend = get_backend(cfg.target.type)

    with pytest.raises(ValueError, match="`vectors.dense` must expand to a mapping"):
        backend.build_load_config(cfg, slc, recreate=False)


def test_load_config_recreate_flag_wins_over_a_same_named_data_layout_key():
    """`recreate` is always the caller's explicit decision, never something a
    data_layout axis can override, even if a config oddly declares a
    `data_layouts` key named `recreate`."""
    cfg = _cfg(data_layouts={"recreate": [False]})
    slc = _one_slice(cfg)
    backend = get_backend(cfg.target.type)

    load_cfg = backend.build_load_config(cfg, slc, recreate=True)

    assert load_cfg["vectorstore"]["params"]["recreate"] is True


def test_reindex_config_carries_variant_params_and_excludes_name():
    cfg = _cfg(index_variants={"hnsw.m": [16], "quantization.type": ["int8"]})
    slc = _one_slice(cfg)
    backend = get_backend(cfg.target.type)
    variant = slc.index_variants[0]

    reindex_cfg = backend.build_reindex_config(cfg, slc, variant)

    assert reindex_cfg["vectorstore"]["params"]["hnsw"] == {"m": 16}
    assert reindex_cfg["vectorstore"]["params"]["quantization"] == {"type": "int8"}
    assert "_name" not in reindex_cfg["vectorstore"]["params"]
    # datasource/vectors are required-but-unused by reindex.
    assert "datasource" in reindex_cfg
    assert "vectors" in reindex_cfg
    # recreate/structural params never leak into a reindex request.
    assert "recreate" not in reindex_cfg["vectorstore"]["params"]


def test_delete_config_carries_target_collection_name():
    cfg = _cfg()
    backend = get_backend(cfg.target.type)

    delete_cfg = backend.build_delete_config(cfg, "some_collection")

    assert delete_cfg["vectorstore"]["type"] == "qdrant"
    assert delete_cfg["vectorstore"]["collection_name"] == "some_collection"


def test_storm_config_routes_top_k_search_params_and_load_correctly():
    cfg = _cfg(searches={"top_k": [50], "hnsw_ef": [128], "exact": [False], "batch_size": [8]})
    slc = _one_slice(cfg)
    backend = get_backend(cfg.target.type)
    search = slc.searches[0]

    storm_cfg = backend.build_storm_config(cfg, slc, search)

    assert storm_cfg["query"]["top_k"] == 50
    assert storm_cfg["query"]["search_params"] == {"hnsw_ef": 128, "exact": False}
    assert storm_cfg["load"] == {"batch_size": 8}
    assert storm_cfg["query"]["source"]["ground_truth_column"] == "hit_ids"
    assert storm_cfg["target"]["type"] == "qdrant"
    assert storm_cfg["target"]["collection_name"] == slc.collection_name


def test_storm_config_omits_search_params_block_when_none_declared():
    cfg = _cfg(searches={"batch_size": [4]})
    slc = _one_slice(cfg)
    backend = get_backend(cfg.target.type)
    search = slc.searches[0]

    storm_cfg = backend.build_storm_config(cfg, slc, search)

    assert "search_params" not in storm_cfg["query"]
    assert storm_cfg["query"]["top_k"] == 10  # default when unset
