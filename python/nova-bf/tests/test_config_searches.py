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
    RangeCondition,
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


def test_search_spec_name_is_optional():
    """A lone SearchSpec can omit `name` — it's left `None` until
    BruteForceConfig assigns a default (a single spec can't disambiguate
    against siblings it can't see)."""
    assert SearchSpec().name is None


def test_search_spec_default_name_derived_from_vector_type_and_metric():
    cfg = BruteForceConfig(**_base(searches=[SearchSpec()]))
    assert cfg.searches[0].name == "dense_cosine"

    cfg = BruteForceConfig(**_base(searches=[SearchSpec(vector_type="sparse", metric="dot")]))
    assert cfg.searches[0].name == "sparse_dot"


def test_search_spec_default_name_gets_filtered_suffix():
    filt = Filter(must=[FilterCondition(field="language", match="eng")])
    cfg = BruteForceConfig(**_base(searches=[SearchSpec(filter=filt)]))
    assert cfg.searches[0].name == "dense_cosine_filtered"

    # an explicit-but-empty filter has no active fields, so it's treated the
    # same as no filter at all — same convention as compute.py's _is_unfiltered.
    cfg = BruteForceConfig(**_base(searches=[SearchSpec(filter=Filter())]))
    assert cfg.searches[0].name == "dense_cosine"


def test_search_spec_default_names_disambiguate_collisions():
    cfg = BruteForceConfig(**_base(searches=[
        SearchSpec(), SearchSpec(), SearchSpec(),
    ]))
    assert [s.name for s in cfg.searches] == ["dense_cosine", "dense_cosine_2", "dense_cosine_3"]


def test_search_spec_default_name_avoids_explicit_name_collision():
    cfg = BruteForceConfig(**_base(searches=[
        SearchSpec(name="dense_cosine"),
        SearchSpec(),
    ]))
    assert [s.name for s in cfg.searches] == ["dense_cosine", "dense_cosine_2"]


def test_search_spec_explicit_name_mixed_with_default_preserves_order():
    cfg = BruteForceConfig(**_base(searches=[
        SearchSpec(),
        SearchSpec(name="my_search", vector_type="sparse", metric="dot"),
        SearchSpec(vector_type="sparse", metric="dot"),
    ]))
    assert [s.name for s in cfg.searches] == ["dense_cosine", "my_search", "sparse_dot"]


def test_filter_is_hashable_and_usable_as_a_dict_key():
    """`Filter`/`FilterCondition`/`RangeCondition` are frozen, so pydantic
    auto-generates `__hash__` from their field values — this is what lets
    compute.py key `keeps`/group directly by the `Filter` object instead of a
    separate string-key scheme."""
    filt = Filter(must=[FilterCondition(field="language", match="eng")])
    d = {filt: "value"}
    assert d[filt] == "value"
    assert hash(FilterCondition(field="cost", range=RangeCondition(lt=10)))
    assert hash(RangeCondition(lt=10))
    # None (the "no filter" case) is hashable too, natively
    assert hash(None) == hash(None)


def test_structurally_identical_filters_hash_and_compare_equal():
    a = Filter(must=[FilterCondition(field="language", match="eng")])
    b = Filter(must=[FilterCondition(field="language", match="eng")])
    assert a == b
    assert hash(a) == hash(b)


def test_filters_differing_in_must_order_are_not_equal():
    """Order-sensitive, same as a plain tuple/list comparison — a `must` list
    reordered in YAML is a genuinely different filter."""
    a = Filter(must=[
        FilterCondition(field="language", match="eng"),
        FilterCondition(field="cost", range=RangeCondition(lt=10)),
    ])
    b = Filter(must=[
        FilterCondition(field="cost", range=RangeCondition(lt=10)),
        FilterCondition(field="language", match="eng"),
    ])
    assert a != b


def test_filter_hash_treats_numerically_equal_int_and_float_match_as_the_same():
    """Native Python `int`/`float` equality (`5 == 5.0`) is exact, with no
    precision loss even for large ints past 2**53 (unlike `float(value)`,
    which would collide `float(2**53) == float(2**53 + 1)`) — a frozen
    `Filter`'s hash/eq comes straight from pydantic's tuple-of-fields hash
    over the native values, so this falls out for free with no
    canonicalization scheme needed."""
    int_filt = Filter(must=[FilterCondition(field="cost", match=5)])
    float_filt = Filter(must=[FilterCondition(field="cost", match=5.0)])
    assert int_filt == float_filt
    assert hash(int_filt) == hash(float_filt)

    other = Filter(must=[FilterCondition(field="cost", match=6)])
    assert int_filt != other

    big = 2**53
    a = Filter(must=[FilterCondition(field="id", match=big)])
    b = Filter(must=[FilterCondition(field="id", match=big + 1)])
    assert a != b
    assert float(big) == float(big + 1)  # sanity: genuinely a float-precision trap
    c = Filter(must=[FilterCondition(field="id", match=float(big))])
    assert a == c
    assert hash(a) == hash(c)

    # Native Python treats `True == 1` (bool is an int subclass), so under
    # this native-equality scheme match=True and match=1 are now the SAME
    # filter — a deliberate behavior change from the old filter_key, which
    # special-cased bool to keep it distinct from 1. That carve-out doesn't
    # survive retiring the custom canonicalization: this is a case where the
    # native scheme is MORE consistent with Filter.__eq__ than the old
    # filter_key was (the old filter_key actually disagreed with
    # Filter.__eq__ here, since `FilterCondition(match=True) ==
    # FilterCondition(match=1)` was already True under Filter's own
    # by-value equality).
    bool_filt = Filter(must=[FilterCondition(field="active", match=True)])
    one_filt = Filter(must=[FilterCondition(field="active", match=1)])
    assert bool_filt == one_filt
    assert hash(bool_filt) == hash(one_filt)


def test_filter_hash_handles_nan_and_inf_match_values():
    """`nan != nan` (even against itself) and `inf`/`-inf` compare equal to
    themselves by sign — native Python float semantics, which a frozen
    `Filter`'s hash/eq inherits directly with no special-casing."""
    nan_filt = Filter(must=[FilterCondition(field="n", match=float("nan"))])
    inf_filt = Filter(must=[FilterCondition(field="n", match=float("inf"))])
    neg_inf_filt = Filter(must=[FilterCondition(field="n", match=float("-inf"))])

    hash(nan_filt)  # must not raise
    hash(inf_filt)  # must not raise

    inf_filt2 = Filter(must=[FilterCondition(field="n", match=float("inf"))])
    assert inf_filt == inf_filt2
    assert hash(inf_filt) == hash(inf_filt2)

    assert inf_filt != neg_inf_filt

    # Two SEPARATE nan-valued Filter instances never compare equal — nan is
    # never equal to anything, itself included — so specs with independently
    # constructed nan filters never wrongly dedupe/merge into one filter
    # group. (Comparing the identical object to itself, `nan_filt ==
    # nan_filt`, is True — pydantic's generated `__eq__` short-circuits on
    # `self is other`, same as CPython dict lookups do internally — but that
    # never happens in compute.py's usage: every `keeps[filter]`/dict-key
    # lookup is either the exact same object stored earlier for that spec, or
    # a distinct spec's distinct nan filter, which is what matters here.)
    nan_filt2 = Filter(must=[FilterCondition(field="n", match=float("nan"))])
    assert nan_filt != nan_filt2


def test_filter_condition_match_list_literal_coerces_to_tuple():
    """YAML/pydantic accepts a list literal for `match` (`match: [1, 2, 3]`)
    but stores it as a tuple — needed for `FilterCondition`/`Filter` to stay
    hashable."""
    cond = FilterCondition(field="cost", match=[1, 2, 3])
    assert cond.match == (1, 2, 3)
    hash(cond)  # must not raise


def test_filter_is_immutable():
    filt = Filter(must=[FilterCondition(field="language", match="eng")])
    with pytest.raises(ValidationError):
        filt.must = ()
    with pytest.raises(ValidationError):
        filt.must[0].match = "other"


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
