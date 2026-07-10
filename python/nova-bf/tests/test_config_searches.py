"""Validation tests for `BruteForceConfig.searches` / `SearchSpec` (config.py) —
no torch dependency, these only exercise pydantic validation and the
`effective_specs()`/naming resolution logic.
"""

from __future__ import annotations

import pytest

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
from nova_bf.results import partial_dir, result_name


def _base(**kw) -> dict:
    return dict(
        corpus=CorpusConfig(path="/tmp/corpus"),
        queries=QueriesConfig(path="/tmp/queries.parquet"),
        output=OutputConfig(path="/tmp/out"),
        **kw,
    )


def test_effective_specs_default_matches_legacy_params():
    """No `searches:` → one synthesized spec carrying the flat params/filter
    fields verbatim, name="" (so output filenames are unchanged)."""
    filt = Filter(must=[FilterCondition(field="language", match="eng")])
    cfg = BruteForceConfig(**_base(params=ParamsConfig(k=7, metric="dot", vector_type="sparse"), filter=filt))
    specs = cfg.effective_specs()
    assert len(specs) == 1
    s = specs[0]
    assert s.name == ""
    assert (s.k, s.metric, s.vector_type) == (7, "dot", "sparse")
    assert s.filter is filt


def test_result_name_and_partial_dir_backward_compat_for_default_spec():
    cfg = BruteForceConfig(**_base(params=ParamsConfig(k=42)))
    spec = cfg.effective_specs()[0]
    assert result_name(cfg, spec) == "bf_queries_k42.parquet"
    assert partial_dir(cfg, spec) == "_bf_partial_queries_k42"


def test_searches_empty_list_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        BruteForceConfig(**_base(searches=[]))


def test_searches_empty_name_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        BruteForceConfig(**_base(searches=[SearchSpec(name="")]))


def test_searches_duplicate_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        BruteForceConfig(**_base(searches=[SearchSpec(name="a"), SearchSpec(name="a")]))


@pytest.mark.parametrize("stale_params", [
    ParamsConfig(k=500),
    ParamsConfig(metric="dot"),
    ParamsConfig(vector_type="sparse"),
    ParamsConfig(corpus_batch_size=4096),
])
def test_searches_with_stale_params_field_rejected(stale_params):
    """A non-default params.k/metric/vector_type/corpus_batch_size left over
    from a legacy single-search config would otherwise be silently ignored
    once `searches` takes over — effective_specs() never reads it, so every
    entry that doesn't repeat the value gets SearchSpec's own default with no
    error. Must be caught at config-load time, same as the top-level `filter`
    conflict."""
    with pytest.raises(ValueError, match="ignored"):
        BruteForceConfig(**_base(params=stale_params, searches=[SearchSpec(name="a")]))


def test_searches_with_only_shared_params_fields_set_is_accepted():
    """io_workers/io_thread_count/merge_batch_size/merge_prefetch remain
    run-level knobs that still apply alongside `searches` — only the fields
    that moved onto SearchSpec (k/metric/vector_type/corpus_batch_size) are
    rejected as stale."""
    cfg = BruteForceConfig(**_base(
        params=ParamsConfig(io_workers=32, io_thread_count=64, merge_prefetch=True),
        searches=[SearchSpec(name="a")],
    ))
    assert cfg.params.io_workers == 32


def test_searches_with_top_level_filter_rejected():
    filt = Filter(must=[FilterCondition(field="language", match="eng")])
    with pytest.raises(ValueError, match="ignored when"):
        BruteForceConfig(**_base(searches=[SearchSpec(name="a")], filter=filt))


def test_search_spec_name_must_be_filename_safe():
    with pytest.raises(ValueError, match="filename"):
        SearchSpec(name="has a space")


def test_search_spec_sparse_euclidean_rejected():
    with pytest.raises(ValueError, match="euclidean"):
        SearchSpec(name="s", vector_type="sparse", metric="euclidean")


def test_searches_multiple_specs_resolve_in_order():
    cfg = BruteForceConfig(**_base(searches=[
        SearchSpec(name="dense_all", vector_type="dense"),
        SearchSpec(name="sparse_all", vector_type="sparse", metric="dot"),
    ]))
    assert [s.name for s in cfg.effective_specs()] == ["dense_all", "sparse_all"]
