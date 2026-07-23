"""Unit tests for the Elasticsearch sweep backend
(`nova_sweep.backends.elastic`): the generated `nova-load`/`nova-storm` config
shapes (FLAT vectorstore fields, `index_name` mapping, `{num_candidates}` search
params), and backend dispatch. No subprocess, no live ES.
"""

from __future__ import annotations

import pytest

from nova_sweep.backends import get_backend
from nova_sweep.backends.elastic import ElasticBackend, ElasticTargetConfig
from nova_sweep.config import CorpusConfig, OutputConfig, QueriesConfig, SweepConfig
from nova_sweep.slices import build_slices


def _cfg(target=None, **axes) -> SweepConfig:
    return SweepConfig(
        collection_name="mycollection",
        corpus=CorpusConfig(path="/tmp/corpus", dense_column="dense_embedding"),
        queries=QueriesConfig(
            uri="/tmp/q.parquet", column="dense_embedding", ground_truth_column="hit_ids"
        ),
        target=target or {"type": "elastic", "url": "http://localhost:9200"},
        output=OutputConfig(path="/tmp/out"),
        **axes,
    )


def _one_slice(cfg: SweepConfig):
    return build_slices(cfg)[0]


def test_get_backend_dispatches_elastic_type():
    cfg = _cfg()
    assert isinstance(cfg.target, ElasticTargetConfig)
    assert isinstance(get_backend(cfg.target.type), ElasticBackend)


def test_load_config_maps_collection_to_index_name_flat():
    cfg = _cfg(data_layouts={"vectors.dense.distance": ["cosine"]})
    slc = _one_slice(cfg)
    load_cfg = get_backend(cfg.target.type).build_load_config(cfg, slc, recreate=True)

    vs = load_cfg["vectorstore"]
    assert vs["type"] == "elastic"
    # collection name is mapped to `index_name` (ES has no `collection_name`).
    assert vs["index_name"] == slc.collection_name
    assert "collection_name" not in vs
    assert vs["recreate"] is True
    assert "params" not in vs  # flat
    assert load_cfg["vectors"]["dense"]["distance"] == "cosine"
    assert load_cfg["datasource"]["id_expression"] == "vf_point_id(filename, file_row_number)"


def test_load_config_carries_index_options_flat():
    cfg = _cfg(data_layouts={"index_options": [{"type": "int8_hnsw", "m": 16}]})
    slc = _one_slice(cfg)
    load_cfg = get_backend(cfg.target.type).build_load_config(cfg, slc, recreate=False)
    assert load_cfg["vectorstore"]["index_options"] == {"type": "int8_hnsw", "m": 16}


def test_reindex_config_index_options_flat():
    cfg = _cfg(index_variants={"index_options": [{"type": "hnsw", "m": 32}]})
    slc = _one_slice(cfg)
    variant = slc.index_variants[0]
    reindex_cfg = get_backend(cfg.target.type).build_reindex_config(cfg, slc, variant)

    vs = reindex_cfg["vectorstore"]
    assert vs["index_options"] == {"type": "hnsw", "m": 32}
    assert vs["index_name"] == slc.collection_name
    assert "recreate" not in vs
    assert "_name" not in vs


def test_reindex_config_preserves_data_layout_index_options_as_base():
    # index_options set via data_layouts must be re-asserted as the reindex base
    # (mirrors Milvus), so the two flat backends behave alike.
    cfg = _cfg(data_layouts={"index_options": [{"type": "int8_hnsw", "m": 16}]})
    slc = _one_slice(cfg)
    variant = slc.index_variants[0]
    reindex_cfg = get_backend(cfg.target.type).build_reindex_config(cfg, slc, variant)

    assert reindex_cfg["vectorstore"]["index_options"] == {"type": "int8_hnsw", "m": 16}


def test_username_without_password_is_rejected():
    with pytest.raises(ValueError, match="must be set together"):
        _cfg(target={"type": "elastic", "url": "http://localhost:9200", "username": "u"})


def test_password_without_username_is_rejected():
    with pytest.raises(ValueError, match="must be set together"):
        _cfg(target={"type": "elastic", "url": "http://localhost:9200", "password": "p"})


def test_api_key_alone_is_accepted():
    # api_key without username/password is valid token auth — must NOT be rejected.
    cfg = _cfg(target={"type": "elastic", "url": "http://localhost:9200", "api_key": "k"})
    assert cfg.target.api_key == "k"


def test_reindex_config_carries_data_layout_distance():
    # ES derives `similarity` from `vectors.dense.distance` and rejects an
    # in-place similarity change; if the reindex config drops the distance it
    # defaults to cosine and a non-cosine index fails the immutability check.
    cfg = _cfg(
        data_layouts={"vectors.dense.distance": ["dot"]},
        index_variants={"index_options": [{"type": "int8_hnsw", "m": 32}]},
    )
    slc = _one_slice(cfg)
    variant = slc.index_variants[0]
    reindex_cfg = get_backend(cfg.target.type).build_reindex_config(cfg, slc, variant)

    assert reindex_cfg["vectors"]["dense"]["distance"] == "dot"
    assert reindex_cfg["vectorstore"]["index_options"] == {"type": "int8_hnsw", "m": 32}


def test_delete_config():
    cfg = _cfg()
    delete_cfg = get_backend(cfg.target.type).build_delete_config(cfg, "some_index")
    assert delete_cfg["vectorstore"]["type"] == "elastic"
    assert delete_cfg["vectorstore"]["index_name"] == "some_index"


def test_storm_config_routes_num_candidates_and_index_name():
    cfg = _cfg(searches={"top_k": [15], "num_candidates": [200], "batch_size": [4]})
    slc = _one_slice(cfg)
    storm_cfg = get_backend(cfg.target.type).build_storm_config(cfg, slc, slc.searches[0])

    assert storm_cfg["query"]["top_k"] == 15
    assert storm_cfg["query"]["search_params"] == {"num_candidates": 200}
    assert storm_cfg["load"] == {"batch_size": 4}
    assert storm_cfg["target"]["type"] == "elastic"
    assert storm_cfg["target"]["index_name"] == slc.collection_name
    assert "collection_name" not in storm_cfg["target"]


def test_api_key_and_tls_insecure_flow_through():
    cfg = _cfg(
        target={
            "type": "elastic",
            "url": "https://localhost:9200",
            "api_key": "base64key",
            "tls_insecure": True,
        },
        searches={"top_k": [10]},
    )
    slc = _one_slice(cfg)
    backend = get_backend(cfg.target.type)

    load_cfg = backend.build_load_config(cfg, slc, recreate=False)
    assert load_cfg["vectorstore"]["api_key"] == "base64key"
    assert load_cfg["vectorstore"]["tls_insecure"] is True

    storm_cfg = backend.build_storm_config(cfg, slc, slc.searches[0])
    assert storm_cfg["target"]["api_key"] == "base64key"
    assert storm_cfg["target"]["tls_insecure"] is True
