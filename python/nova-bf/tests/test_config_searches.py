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
    filter_key,
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
    """k/metric/vector_type moved to SearchSpec — ParamsConfig only has
    run-level IO/merge/GPU-batching knobs left, so pydantic's own
    `extra="forbid"` now rejects these natively (no custom validator
    needed)."""
    for bad_kwargs in ({"k": 500}, {"metric": "dot"}, {"vector_type": "sparse"}):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ParamsConfig(**bad_kwargs)


def test_search_spec_no_longer_accepts_corpus_batch_size():
    """`corpus_batch_size` moved OFF `SearchSpec` onto `ParamsConfig`
    (`dense_batch_size`/`sparse_batch_size`) — every search of a vector_type
    ends up sharing one GPU pass over the corpus regardless of grouping (see
    compute.py), so it was never really a per-search setting."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SearchSpec(name="a", corpus_batch_size=4096)


def test_params_dense_and_sparse_batch_size_fields_work():
    cfg = BruteForceConfig(**_base(
        params=ParamsConfig(dense_batch_size=4096, sparse_batch_size=2048),
        searches=[SearchSpec(name="a")],
    ))
    assert cfg.params.dense_batch_size == 4096
    assert cfg.params.sparse_batch_size == 2048


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


def test_filter_key_none_is_distinct_from_any_real_filter():
    filt = Filter(must=[FilterCondition(field="language", match="eng")])
    assert filter_key(None) != filter_key(filt)


def test_filter_key_identical_filters_share_a_key():
    a = Filter(must=[FilterCondition(field="language", match="eng")])
    b = Filter(must=[FilterCondition(field="language", match="eng")])
    assert filter_key(a) == filter_key(b)


def test_filter_key_is_order_sensitive_like_filter_equality():
    """Two filters differing only in `must` list order are NOT the same
    filter (`Filter`'s own by-value equality is order-sensitive too, since
    it's a plain list comparison) — `filter_key` must agree, or two specs
    that `Filter.__eq__` treats as different would incorrectly dedupe."""
    a = Filter(must=[
        FilterCondition(field="language", match="eng"),
        FilterCondition(field="cost", range={"lt": 10}),
    ])
    b = Filter(must=[
        FilterCondition(field="cost", range={"lt": 10}),
        FilterCondition(field="language", match="eng"),
    ])
    assert a != b
    assert filter_key(a) != filter_key(b)


def test_filter_key_treats_numerically_equal_int_and_float_match_as_the_same_filter():
    """Regression test: `Filter`'s own equality is Python's `==`, which treats
    `5 == 5.0` as True — a filter authored with `match: 5` (YAML int) must
    produce the SAME filter_key as one authored with `match: 5.0` (YAML
    float), or two specs meant to share a filter would silently land in
    different dedup buckets (the filter evaluated twice, and — under Path B
    — no shared compaction/transfer) despite `Filter.__eq__` agreeing they're
    identical."""
    int_filt = Filter(must=[FilterCondition(field="cost", match=5)])
    float_filt = Filter(must=[FilterCondition(field="cost", match=5.0)])
    assert int_filt == float_filt
    assert filter_key(int_filt) == filter_key(float_filt)

    # sanity: a genuinely different numeric value still produces a different key
    other = Filter(must=[FilterCondition(field="cost", match=6)])
    assert filter_key(int_filt) != filter_key(other)

    # match lists containing numeric values are canonicalized element-wise too
    list_int = Filter(must=[FilterCondition(field="cost", match=[1, 2, 3])])
    list_float = Filter(must=[FilterCondition(field="cost", match=[1.0, 2.0, 3.0])])
    assert filter_key(list_int) == filter_key(list_float)

    # bool is left alone, not folded into the numeric canonicalization
    bool_filt = Filter(must=[FilterCondition(field="active", match=True)])
    one_filt = Filter(must=[FilterCondition(field="active", match=1)])
    assert filter_key(bool_filt) != filter_key(one_filt)


def test_filter_key_does_not_collide_large_ints_via_float_precision_loss():
    """Regression test: canonicalizing `match` via `float(value)` (an earlier
    version of this fix) silently collides distinct large integers, since
    float64 can't represent every int exactly past 2**53 — e.g.
    `float(2**53) == float(2**53 + 1)`. Python's own `==` (what `Filter`'s
    equality relies on) compares int/float exactly, with no such precision
    loss, so `filter_key` must too — two DIFFERENT filters (differing only in
    a large `match` value, e.g. a nanosecond timestamp or snowflake id in the
    ~1e18 range where this bites) must never produce the same key, or
    `_build_filter_groups`/`distinct_filters` would silently merge them and
    evaluate one spec's results against the WRONG filter."""
    big = 2**53
    a = Filter(must=[FilterCondition(field="id", match=big)])
    b = Filter(must=[FilterCondition(field="id", match=big + 1)])
    assert a != b
    assert float(big) == float(big + 1)  # sanity: this is genuinely a float-precision trap
    assert filter_key(a) != filter_key(b)

    # still unifies a large int with an EQUAL float (no regression on the original fix)
    c = Filter(must=[FilterCondition(field="id", match=float(big))])
    assert a == c
    assert filter_key(a) == filter_key(c)


def test_filter_key_does_not_crash_on_nan_or_inf_match_values():
    """Regression test: `Fraction` (the fix for the 2**53 collision above)
    can't represent NaN or Infinity at all — `Fraction(float("nan"))` raises
    `ValueError`, `Fraction(float("inf"))` raises `OverflowError` — but both
    are legal YAML/pydantic `match` values, so `filter_key` (called
    unconditionally at `run_compute` setup for every spec) must never crash
    on them the way an earlier version of the Fraction fix did."""
    nan_filt = Filter(must=[FilterCondition(field="n", match=float("nan"))])
    inf_filt = Filter(must=[FilterCondition(field="n", match=float("inf"))])
    neg_inf_filt = Filter(must=[FilterCondition(field="n", match=float("-inf"))])

    filter_key(nan_filt)  # must not raise
    filter_key(inf_filt)  # must not raise

    # inf == inf under Python's own `==` (what Filter.__eq__ relies on), so
    # two inf-matching filters must still share a key
    inf_filt2 = Filter(must=[FilterCondition(field="n", match=float("inf"))])
    assert inf_filt == inf_filt2
    assert filter_key(inf_filt) == filter_key(inf_filt2)

    # +inf and -inf are NOT the same filter
    assert inf_filt != neg_inf_filt
    assert filter_key(inf_filt) != filter_key(neg_inf_filt)

    # nan != nan even against itself, so two nan-matching filters (however
    # constructed) must never share a key either
    nan_filt2 = Filter(must=[FilterCondition(field="n", match=float("nan"))])
    assert nan_filt != nan_filt2  # Filter.__eq__ agrees: nan != nan
    assert filter_key(nan_filt) != filter_key(nan_filt2)


def test_params_rejects_non_positive_batch_size():
    """Regression test: `range(0, n_rows, step)` with a non-positive `step`
    is an EMPTY range — a negative or zero `dense_batch_size`/
    `sparse_batch_size` (a plausible config typo) would otherwise silently
    skip every row of every file, producing an empty top-K for every query
    with no error or warning. Reject it at config-load time instead."""
    for bad in (-5, 0):
        with pytest.raises(ValidationError):
            ParamsConfig(dense_batch_size=bad)
        with pytest.raises(ValidationError):
            ParamsConfig(sparse_batch_size=bad)
    # positive values and None (whole-file) still work
    assert ParamsConfig(dense_batch_size=100).dense_batch_size == 100
    assert ParamsConfig(sparse_batch_size=None).sparse_batch_size is None
