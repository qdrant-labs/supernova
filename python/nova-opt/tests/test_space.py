import numpy as np
import pytest

from nova_opt.space import (
    QUANT_VARIANTS,
    Candidate,
    ConfigSpace,
    Index,
    Layout,
    Quant,
    Search,
    SpaceAxes,
    config_features,
)


def make_candidate(**over) -> Candidate:
    base = dict(
        layout=Layout(segments=8),
        index=Index(m=16, ef_construct=128),
        quant=Quant(variant="none"),
        search=Search(ef_search=64),
    )
    base.update(over)
    return Candidate(**base)


def test_artifact_keys_nest_hierarchically():
    c = make_candidate()
    assert c.index_key[0] == c.layout_key
    assert c.quant_key[0] == c.index_key
    assert c.search_key[0] == c.quant_key


def test_search_change_preserves_quant_key():
    c = make_candidate()
    sibling = c.with_search(Search(ef_search=256, batch_size=32))
    assert sibling.quant_key == c.quant_key
    assert sibling.search_key != c.search_key


def test_layout_change_invalidates_all_keys():
    a = make_candidate(layout=Layout(segments=8))
    b = make_candidate(layout=Layout(segments=16))
    assert a.layout_key != b.layout_key
    assert a.index_key != b.index_key
    assert a.quant_key != b.quant_key


def test_unknown_quant_variant_rejected():
    with pytest.raises(ValueError, match="unknown quantization variant"):
        Quant(variant="bogus")


def test_quant_vocabulary_matches_training_data():
    # the exact quantization_mode strings observed in data.csv
    for variant, mode in [
        ("none", "NONE"),
        ("scalar_default", "SCALAR__scalar_default"),
        ("binary_1bit", "BINARY__DEFAULT"),
        ("binary_1_5bit", "BINARY__ONE_AND_HALF_BITS"),
        ("binary_2bit", "BINARY__TWO_BITS"),
        ("product_x64", "PRODUCT__X64"),
        ("turbo_bits1_5", "TURBO__BITS1_5"),
        ("turbo_bits2", "TURBO__BITS2"),
    ]:
        assert QUANT_VARIANTS[variant].mode == mode


def test_config_features_derivations():
    # bigann-shaped workload: segment_size_kb must reproduce data.csv's value
    c = make_candidate(layout=Layout(segments=128))
    feats = config_features(
        c,
        {
            "corpus_size": 10_000_000,
            "query_count": 10_000,
            "vector_dim": 128,
            "distance_metric": "L2",
        },
    )
    assert feats["data_size_bytes"] == 5_120_000_000
    assert feats["segment_size_kb"] == 39063
    assert feats["quantization"] == "NONE"
    assert feats["hnsw_m"] == 16 and feats["ef_search"] == 64


def test_sample_dedupes_and_respects_exclude():
    axes = SpaceAxes(m=(8, 16), ef_search=(16, 32, 64), ef_construct=(64, 128))
    space = ConfigSpace(axes)
    rng = np.random.default_rng(0)
    first = space.sample(5, rng)
    keys = {c.search_key for c in first}
    assert len(keys) == len(first)
    rest = space.sample(100, rng, exclude=keys)
    assert keys.isdisjoint({c.search_key for c in rest})
    assert len(first) + len(rest) == space.size()


def test_candidate_from_quant_key_roundtrip():
    from nova_opt.space import candidate_from_quant_key

    c = make_candidate(quant=Quant(variant="binary_2bit", always_ram=False))
    rebuilt = candidate_from_quant_key(c.quant_key, Search(ef_search=256))
    assert rebuilt.quant_key == c.quant_key
    assert rebuilt.search.ef_search == 256


def test_biased_sampling_targets_existing_artifacts():
    axes = SpaceAxes(
        segments=(2, 8, 32),
        m=(8, 16, 32, 64),
        ef_construct=(64, 128, 256),
        quant_variant=("none", "scalar_default", "binary_1bit"),
        ef_search=(16, 32, 64, 128, 256),
        batch_size=(1, 8, 32),
    )
    space = ConfigSpace(axes)
    rng = np.random.default_rng(0)
    anchor = make_candidate()
    got = space.sample(
        60, rng, bias_quant_keys=(anchor.quant_key,), bias_fraction=0.7
    )
    hits = sum(1 for c in got if c.quant_key == anchor.quant_key)
    # a solid share of proposals reuse the existing artifact...
    assert hits >= 10
    # ...but uniform draws keep proposing fresh artifacts too
    assert hits < len(got)


def test_children_share_artifact_and_cap():
    axes = SpaceAxes()
    space = ConfigSpace(axes)
    parent = make_candidate()
    kids = space.children(
        parent,
        ef_search=(16, 32, 64, 128, 256),
        batch_size=(1, 8, 32, 128),
        max_children=8,
    )
    assert len(kids) == 8
    assert all(k.quant_key == parent.quant_key for k in kids)
    assert parent.search_key not in {k.search_key for k in kids}
    # ef_search coverage comes before batch-size duplication
    first_batch = {k.search.ef_search for k in kids if k.search.batch_size == 1}
    assert first_batch >= {16, 32, 128, 256}
