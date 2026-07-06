"""Unit tests for `nova_sweep.slices.build_slices` — pure, no I/O."""

from __future__ import annotations

from nova_sweep.config import CorpusConfig, OutputConfig, QueriesConfig, SweepConfig
from nova_sweep.slices import build_slices


def _cfg(**axes) -> SweepConfig:
    return SweepConfig(
        corpus=CorpusConfig(path="/tmp/corpus"),
        queries=QueriesConfig(uri="/tmp/q.parquet", column="dense_embedding"),
        target={"type": "qdrant", "url": "http://localhost:6334"},
        output=OutputConfig(path="/tmp/out"),
        **axes,
    )


def test_one_slice_per_data_layout():
    cfg = _cfg(data_layouts={"vectors.dense.datatype": ["float32", "uint8"]})
    slices = build_slices(cfg, "mysweep")
    assert len(slices) == 2
    assert {s.data_layout_name for s in slices} == {"datatypefloat32", "datatypeuint8"}


def test_collection_name_is_sweep_name_plus_layout_name():
    cfg = _cfg(data_layouts={"vectors.dense.datatype": ["uint8"]})
    slices = build_slices(cfg, "mysweep")
    assert slices[0].collection_name == "mysweep_datatypeuint8"


def test_every_slice_shares_the_same_index_variants_and_searches():
    cfg = _cfg(
        data_layouts={"vectors.dense.datatype": ["float32", "uint8"]},
        index_variants={"hnsw.m": [8, 16]},
        searches={"top_k": [10, 100]},
    )
    slices = build_slices(cfg, "mysweep")
    assert len(slices) == 2
    assert slices[0].index_variants == slices[1].index_variants
    assert slices[0].searches == slices[1].searches
    assert len(slices[0].index_variants) == 2
    assert len(slices[0].searches) == 2


def test_no_axes_declared_yields_one_slice_with_one_default_point():
    cfg = _cfg()
    slices = build_slices(cfg, "mysweep")
    assert len(slices) == 1
    assert slices[0].collection_name == "mysweep_default"
    assert slices[0].index_variants == [{"_name": "default"}]
    assert slices[0].searches == [{"_name": "default"}]
