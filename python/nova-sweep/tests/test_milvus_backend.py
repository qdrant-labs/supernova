"""Unit tests for the Milvus sweep backend (`nova_sweep.backends.milvus`):
the generated `nova-load`/`nova-storm` config shapes (FLAT vectorstore fields,
`{ef, nprobe}` search params), and backend dispatch. No subprocess, no live
Milvus.
"""

from __future__ import annotations

import pytest

from nova_sweep.backends import get_backend
from nova_sweep.backends.milvus import MilvusBackend, MilvusTargetConfig
from nova_sweep.config import CorpusConfig, OutputConfig, QueriesConfig, SweepConfig
from nova_sweep.slices import build_slices


def _cfg(**axes) -> SweepConfig:
    return SweepConfig(
        collection_name="mycollection",
        corpus=CorpusConfig(path="/tmp/corpus", dense_column="dense_embedding"),
        queries=QueriesConfig(
            uri="/tmp/q.parquet", column="dense_embedding", ground_truth_column="hit_ids"
        ),
        target={"type": "milvus", "url": "http://localhost:19530"},
        output=OutputConfig(path="/tmp/out"),
        **axes,
    )


def _one_slice(cfg: SweepConfig):
    return build_slices(cfg)[0]


def test_get_backend_dispatches_milvus_type():
    cfg = _cfg()
    assert isinstance(cfg.target, MilvusTargetConfig)
    assert isinstance(get_backend(cfg.target.type), MilvusBackend)


def test_load_config_flat_vectorstore_with_recreate_and_id_expression():
    cfg = _cfg(data_layouts={"vectors.dense.distance": ["cosine"]})
    slc = _one_slice(cfg)
    load_cfg = get_backend(cfg.target.type).build_load_config(cfg, slc, recreate=True)

    vs = load_cfg["vectorstore"]
    assert vs["type"] == "milvus"
    assert vs["collection_name"] == slc.collection_name
    assert vs["url"] == "http://localhost:19530"
    # FLAT: recreate sits directly on the vectorstore block, NOT under `params`.
    assert vs["recreate"] is True
    assert "params" not in vs
    assert load_cfg["vectors"]["dense"]["distance"] == "cosine"
    assert load_cfg["datasource"]["id_expression"] == "vf_point_id(filename, file_row_number)"


def test_load_config_carries_index_type_and_params_flat():
    cfg = _cfg(data_layouts={"index_type": ["HNSW"]})
    slc = _one_slice(cfg)
    load_cfg = get_backend(cfg.target.type).build_load_config(cfg, slc, recreate=False)

    assert load_cfg["vectorstore"]["index_type"] == "HNSW"
    assert load_cfg["vectorstore"]["recreate"] is False


def test_reindex_config_flat_variant_params_no_recreate():
    cfg = _cfg(index_variants={"index_type": ["IVF_FLAT"], "index_params": [{"nlist": 128}]})
    slc = _one_slice(cfg)
    variant = slc.index_variants[0]
    reindex_cfg = get_backend(cfg.target.type).build_reindex_config(cfg, slc, variant)

    vs = reindex_cfg["vectorstore"]
    assert vs["index_type"] == "IVF_FLAT"
    assert vs["index_params"] == {"nlist": 128}
    assert "_name" not in vs
    assert "recreate" not in vs
    assert "params" not in vs  # flat, not nested


def test_reindex_config_preserves_data_layout_index_params_as_base():
    # An index_type set via data_layouts must survive the per-variant reindex,
    # not be silently reset to AUTOINDEX (Milvus rebuilds from config every time).
    # The implicit "default" variant carries no index_type of its own.
    cfg = _cfg(data_layouts={"index_type": ["IVF_FLAT"], "index_params": [{"nlist": 64}]})
    slc = _one_slice(cfg)
    variant = slc.index_variants[0]
    reindex_cfg = get_backend(cfg.target.type).build_reindex_config(cfg, slc, variant)

    assert reindex_cfg["vectorstore"]["index_type"] == "IVF_FLAT"
    assert reindex_cfg["vectorstore"]["index_params"] == {"nlist": 64}


def test_reindex_variant_overrides_data_layout_index_base():
    # When both a data_layout and a variant set the same knob, the variant wins.
    cfg = _cfg(
        data_layouts={"index_type": ["IVF_FLAT"]},
        index_variants={"index_type": ["HNSW"], "index_params": [{"M": 16}]},
    )
    slc = _one_slice(cfg)
    variant = slc.index_variants[0]
    reindex_cfg = get_backend(cfg.target.type).build_reindex_config(cfg, slc, variant)

    assert reindex_cfg["vectorstore"]["index_type"] == "HNSW"
    assert reindex_cfg["vectorstore"]["index_params"] == {"M": 16}


def test_reindex_config_carries_data_layout_distance():
    # Milvus rebuilds the index with the metric from `vectors.dense.distance`;
    # if the reindex config drops it, nova-load defaults the rebuild to COSINE
    # and silently corrupts a non-cosine collection's metric.
    cfg = _cfg(
        data_layouts={"vectors.dense.distance": ["dot"]},
        index_variants={"index_type": ["HNSW"]},
    )
    slc = _one_slice(cfg)
    variant = slc.index_variants[0]
    reindex_cfg = get_backend(cfg.target.type).build_reindex_config(cfg, slc, variant)

    assert reindex_cfg["vectors"]["dense"]["distance"] == "dot"
    assert reindex_cfg["vectorstore"]["index_type"] == "HNSW"


def test_delete_config():
    cfg = _cfg()
    delete_cfg = get_backend(cfg.target.type).build_delete_config(cfg, "some_coll")
    assert delete_cfg["vectorstore"]["type"] == "milvus"
    assert delete_cfg["vectorstore"]["collection_name"] == "some_coll"


def test_username_without_password_is_rejected():
    with pytest.raises(ValueError, match="must be set together"):
        SweepConfig(
            collection_name="c",
            corpus=CorpusConfig(path="/tmp/c", dense_column="dense_embedding"),
            queries=QueriesConfig(uri="/tmp/q.parquet", column="dense_embedding"),
            target={"type": "milvus", "url": "http://localhost:19530", "username": "root"},
            output=OutputConfig(path="/tmp/o"),
        )


def test_password_without_username_is_rejected():
    with pytest.raises(ValueError, match="must be set together"):
        SweepConfig(
            collection_name="c",
            corpus=CorpusConfig(path="/tmp/c", dense_column="dense_embedding"),
            queries=QueriesConfig(uri="/tmp/q.parquet", column="dense_embedding"),
            target={"type": "milvus", "url": "http://localhost:19530", "password": "pw"},
            output=OutputConfig(path="/tmp/o"),
        )


def test_storm_config_routes_ef_nprobe_to_search_params():
    cfg = _cfg(searches={"top_k": [20], "ef": [64], "batch_size": [8]})
    slc = _one_slice(cfg)
    storm_cfg = get_backend(cfg.target.type).build_storm_config(cfg, slc, slc.searches[0])

    assert storm_cfg["query"]["top_k"] == 20
    assert storm_cfg["query"]["search_params"] == {"ef": 64}
    assert storm_cfg["load"] == {"batch_size": 8}
    assert storm_cfg["query"]["vector_name"] == "dense"
    assert storm_cfg["query"]["source"]["ground_truth_column"] == "hit_ids"
    assert storm_cfg["target"]["type"] == "milvus"
    assert storm_cfg["target"]["collection_name"] == slc.collection_name


def test_storm_config_nprobe_search_param():
    cfg = _cfg(searches={"nprobe": [32]})
    slc = _one_slice(cfg)
    storm_cfg = get_backend(cfg.target.type).build_storm_config(cfg, slc, slc.searches[0])
    assert storm_cfg["query"]["search_params"] == {"nprobe": 32}


def test_auth_flows_into_generated_configs_when_set():
    cfg = _cfg(searches={"top_k": [10]})
    cfg.target.username = "root"
    cfg.target.password = "pw"
    slc = _one_slice(cfg)
    backend = get_backend(cfg.target.type)

    load_cfg = backend.build_load_config(cfg, slc, recreate=False)
    assert load_cfg["vectorstore"]["username"] == "root"
    assert load_cfg["vectorstore"]["password"] == "pw"

    storm_cfg = backend.build_storm_config(cfg, slc, slc.searches[0])
    assert storm_cfg["target"]["username"] == "root"
    assert storm_cfg["target"]["password"] == "pw"
