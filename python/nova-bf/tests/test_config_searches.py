"""Validation tests for `BruteForceConfig.searches` / `SearchSpec` (config.py) —
no torch dependency, these only exercise pydantic validation. `searches` is
always required: a config always says explicitly which search(es) it runs,
with no flat-params/top-level-filter legacy shape to fall back to.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    Filter,
    FilterCondition,
    OutputConfig,
    ParamsConfig,
    QueriesConfig,
    SearchSpec,
)


def _base(**kw) -> dict:
    return dict(
        corpus=CorpusConfig(path="/tmp/corpus"),
        queries=QueriesConfig(path="/tmp/queries.parquet"),
        output=OutputConfig(path="/tmp/out"),
        **kw,
    )


def test_searches_is_required():
    with pytest.raises(ValidationError, match="searches"):
        BruteForceConfig(**_base())


def test_searches_empty_list_rejected():
    with pytest.raises(ValueError, match="at least one"):
        BruteForceConfig(**_base(searches=[]))


def test_searches_duplicate_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        BruteForceConfig(**_base(searches=[SearchSpec(name="a"), SearchSpec(name="a")]))


def test_search_spec_empty_name_rejected():
    with pytest.raises(ValueError, match="filename"):
        SearchSpec(name="")


def test_search_spec_name_must_be_filename_safe():
    with pytest.raises(ValueError, match="filename"):
        SearchSpec(name="has a space")


def test_search_spec_sparse_euclidean_rejected():
    with pytest.raises(ValueError, match="euclidean"):
        SearchSpec(name="s", vector_type="sparse", metric="euclidean")


def test_params_no_longer_accepts_search_semantics_fields():
    """k/metric/vector_type/corpus_batch_size moved to SearchSpec — ParamsConfig
    only has run-level IO/merge knobs left, so pydantic's own `extra="forbid"`
    now rejects these natively (no custom validator needed)."""
    for bad_kwargs in (
        {"k": 500}, {"metric": "dot"}, {"vector_type": "sparse"}, {"corpus_batch_size": 4096},
    ):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ParamsConfig(**bad_kwargs)


def test_params_run_level_fields_still_work():
    cfg = BruteForceConfig(**_base(
        params=ParamsConfig(io_workers=32, io_thread_count=64, merge_prefetch=True),
        searches=[SearchSpec(name="a")],
    ))
    assert cfg.params.io_workers == 32


def test_filter_lives_on_each_search_not_at_top_level():
    """No top-level `filter:` field exists — each search carries its own."""
    filt = Filter(must=[FilterCondition(field="language", match="eng")])
    cfg = BruteForceConfig(**_base(searches=[SearchSpec(name="a", filter=filt)]))
    assert cfg.searches[0].filter is filt
    assert not hasattr(cfg, "filter")


def test_searches_multiple_specs_preserve_order():
    cfg = BruteForceConfig(**_base(searches=[
        SearchSpec(name="dense_all", vector_type="dense"),
        SearchSpec(name="sparse_all", vector_type="sparse", metric="dot"),
    ]))
    assert [s.name for s in cfg.searches] == ["dense_all", "sparse_all"]
