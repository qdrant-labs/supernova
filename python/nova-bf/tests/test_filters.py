"""Unit tests for `nova_bf.filters.evaluate` — no torch required."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    Filter,
    FilterCondition,
    OutputConfig,
    QueriesConfig,
    RangeCondition,
    RangeFromQuery,
    SearchSpec,
)
from nova_bf.filters import _literal_then_regex_word_mask, _static_first, evaluate


def _table(**cols):
    return pa.table(cols)


def test_match_scalar_equality():
    t = _table(language=["eng", "fra", "eng", "deu"])
    f = Filter(must=[FilterCondition(field="language", match="eng")])
    assert evaluate(f, t).tolist() == [True, False, True, False]


def test_match_any_of_list():
    t = _table(language=["eng", "fra", "spa", "deu"])
    f = Filter(must=[FilterCondition(field="language", match=["eng", "spa"])])
    assert evaluate(f, t).tolist() == [True, False, True, False]


def test_range_single_bound():
    t = _table(cost=[1, 5, 10, 20])
    f = Filter(must=[FilterCondition(field="cost", range=RangeCondition(lt=10))])
    assert evaluate(f, t).tolist() == [True, True, False, False]


def test_range_combined_bounds_within_one_condition():
    t = _table(cost=[1, 5, 10, 20])
    f = Filter(must=[FilterCondition(field="cost", range=RangeCondition(gte=5, lt=20))])
    assert evaluate(f, t).tolist() == [False, True, True, False]


def test_must_is_and():
    t = _table(language=["eng", "eng", "fra", "fra"], cost=[1, 20, 1, 20])
    f = Filter(must=[
        FilterCondition(field="language", match="eng"),
        FilterCondition(field="cost", range=RangeCondition(lt=10)),
    ])
    assert evaluate(f, t).tolist() == [True, False, False, False]


def test_should_is_any_of():
    t = _table(language=["eng", "fra", "deu"])
    f = Filter(should=[
        FilterCondition(field="language", match="eng"),
        FilterCondition(field="language", match="fra"),
    ])
    assert evaluate(f, t).tolist() == [True, True, False]


def test_must_not_excludes():
    t = _table(language=["eng", "fra", "deu"])
    f = Filter(must_not=[FilterCondition(field="language", match="fra")])
    assert evaluate(f, t).tolist() == [True, False, True]


def test_null_payload_value_never_matches():
    t = _table(cost=[1, None, 10])
    f = Filter(must=[FilterCondition(field="cost", range=RangeCondition(lt=100))])
    assert evaluate(f, t).tolist() == [True, False, True]


def test_match_text_single_word():
    t = _table(text=["a chronic illness", "a brief cold", "chronically ill"])
    f = Filter(must=[FilterCondition(field="text", match_text="chronic")])
    assert evaluate(f, t).tolist() == [True, False, False]


def test_match_text_requires_all_words():
    t = _table(text=[
        "chronic fatigue syndrome",
        "chronic fatigue",
        "fatigue syndrome",
        "unrelated text",
    ])
    f = Filter(must=[FilterCondition(field="text", match_text="chronic fatigue syndrome")])
    assert evaluate(f, t).tolist() == [True, False, False, False]


def test_match_text_case_insensitive():
    t = _table(text=["Chronic Fatigue", "chronic fatigue", "CHRONIC FATIGUE SYNDROME"])
    f = Filter(must=[FilterCondition(field="text", match_text="chronic fatigue")])
    assert evaluate(f, t).tolist() == [True, True, True]


def test_match_text_word_boundary():
    t = _table(text=["a cat sat", "category theory", "cats and dogs"])
    f = Filter(must=[FilterCondition(field="text", match_text="cat")])
    assert evaluate(f, t).tolist() == [True, False, False]


def test_match_text_null_field_never_matches():
    t = _table(text=["chronic fatigue", None, "chronic fatigue syndrome"])
    f = Filter(must=[FilterCondition(field="text", match_text="chronic fatigue")])
    assert evaluate(f, t).tolist() == [True, False, True]


def test_condition_requires_exactly_one_of_match_range_or_match_text():
    with pytest.raises(ValueError):
        FilterCondition(field="cost")
    with pytest.raises(ValueError):
        FilterCondition(field="cost", match=5, range=RangeCondition(lt=10))
    with pytest.raises(ValueError):
        FilterCondition(field="cost", match=5, match_text="a")
    with pytest.raises(ValueError):
        FilterCondition(field="cost", range=RangeCondition(lt=10), match_text="a")
    with pytest.raises(ValueError):
        FilterCondition(field="text", match_text="   ")
    # match_text alone is valid
    FilterCondition(field="text", match_text="chronic fatigue")


def test_range_condition_requires_a_bound():
    with pytest.raises(ValueError):
        RangeCondition()


def test_config_filter_field_is_the_only_declaration_needed():
    # No separate "which columns to read" list — a condition's `field` is itself
    # the declaration, and `.fields()` (what compute.py reads) reflects it directly.
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path="/tmp/corpus"),
        queries=QueriesConfig(path="/tmp/q.parquet"),
        output=OutputConfig(path="/tmp/out"),
        searches=[SearchSpec(
            name="test",
            filter=Filter(must=[FilterCondition(field="language", match="eng")]),
        )],
    )
    assert cfg.searches[0].filter.fields() == {"language"}


# --- per-query conditions (match_from_query / range_from_query / match_text_from_query) ---


def test_match_from_query_scalar():
    t = _table(tenant_id=["A", "B", "A", "C"])
    f = Filter(must=[FilterCondition(field="tenant_id", match_from_query="tenant_id")])
    qv = {"tenant_id": np.array(["A", "C"], dtype=object)}
    mask = evaluate(f, t, qv)
    assert mask.shape == (2, 4)
    assert mask.tolist() == [
        [True, False, True, False],   # query 0 wants tenant A
        [False, False, False, True],  # query 1 wants tenant C
    ]


def test_match_from_query_list_any():
    """A queries column holding a LIST per row is per-query MatchAny."""
    t = _table(category=["books", "electronics", "toys", "books"])
    f = Filter(must=[FilterCondition(field="category", match_from_query="allowed")])
    qv = {"allowed": np.empty(2, dtype=object)}
    qv["allowed"][:] = [["books", "toys"], ["electronics"]]
    mask = evaluate(f, t, qv)
    assert mask.tolist() == [
        [True, False, True, True],
        [False, True, False, False],
    ]


def test_match_from_query_list_any_with_null_corpus_value():
    """Regression test: a null corpus value mixed into an otherwise
    list-valued (per-query MatchAny) match_from_query condition used to
    crash `np.unique` (can't compare `None` to `str` while sorting) before
    the null-exclusion mask ever got a chance to apply. The null row must
    just never match, for any query."""
    t = _table(category=["books", None, "toys"])
    f = Filter(must=[FilterCondition(field="category", match_from_query="allowed")])
    qv = {"allowed": np.empty(1, dtype=object)}
    qv["allowed"][:] = [["books", "toys"]]
    mask = evaluate(f, t, qv)
    assert mask.tolist() == [[True, False, True]]


def test_match_from_query_null_never_matches_on_either_side():
    t = _table(tenant_id=["A", None, "C"])
    f = Filter(must=[FilterCondition(field="tenant_id", match_from_query="tenant_id")])
    qv = {"tenant_id": np.array(["A", None], dtype=object)}
    mask = evaluate(f, t, qv)
    # query 0 (wants "A"): matches row 0 only, never the null corpus row
    assert mask[0].tolist() == [True, False, False]
    # query 1 (itself null): matches nothing, including the null corpus row
    assert mask[1].tolist() == [False, False, False]


def test_match_from_query_list_any_with_nan_query_value():
    """Regression test: a bare NaN mixed into an otherwise list-valued
    (per-query MatchAny) column used to crash `_match_any_membership`
    (`TypeError: 'float' object is not iterable`) — parquet loading turns
    null list rows into `None`, but a hand-built or pandas-sourced column
    can carry NaN instead. Must behave exactly like `None`: that query
    matches nothing."""
    t = _table(category=["books", "electronics", "toys"])
    f = Filter(must=[FilterCondition(field="category", match_from_query="allowed")])
    qv = {"allowed": np.empty(3, dtype=object)}
    qv["allowed"][:] = [["books", "toys"], np.nan, None]
    mask = evaluate(f, t, qv)
    assert mask.tolist() == [
        [True, False, True],
        [False, False, False],  # NaN query: matches nothing, same as None
        [False, False, False],
    ]


def test_range_from_query_single_bound():
    t = _table(cost=[5.0, 15.0, 8.0, 3.0])
    f = Filter(must=[FilterCondition(field="cost", range_from_query=RangeFromQuery(lt="max_budget"))])
    qv = {"max_budget": np.array([10.0, 5.0, 20.0])}
    mask = evaluate(f, t, qv)
    assert mask.tolist() == [
        [True, False, True, True],
        [False, False, False, True],
        [True, True, True, True],
    ]


def test_range_from_query_multi_bound_in_one_condition():
    t = _table(cost=[1.0, 5.0, 10.0, 20.0])
    f = Filter(must=[FilterCondition(
        field="cost", range_from_query=RangeFromQuery(gte="lo", lt="hi"),
    )])
    qv = {"lo": np.array([2.0]), "hi": np.array([15.0])}
    mask = evaluate(f, t, qv)
    assert mask.tolist() == [[False, True, True, False]]


def test_range_from_query_null_query_value_never_matches():
    t = _table(cost=[1.0, 5.0, 10.0])
    f = Filter(must=[FilterCondition(field="cost", range_from_query=RangeFromQuery(lt="max_budget"))])
    qv = {"max_budget": np.array([100.0, np.nan])}
    mask = evaluate(f, t, qv)
    assert mask[0].tolist() == [True, True, True]
    assert mask[1].tolist() == [False, False, False]


def test_range_from_query_null_corpus_value_never_matches():
    t = _table(cost=[1.0, None, 10.0])
    f = Filter(must=[FilterCondition(field="cost", range_from_query=RangeFromQuery(lt="max_budget"))])
    qv = {"max_budget": np.array([100.0])}
    mask = evaluate(f, t, qv)
    assert mask[0].tolist() == [True, False, True]


def test_range_from_query_and_static_range_combine_as_two_conditions():
    """Mixing a static floor with a per-query ceiling is expressed as two
    separate conditions in the same `must` list, not one condition with
    mixed bound sources."""
    t = _table(cost=[-5.0, 1.0, 5.0, 50.0])
    f = Filter(must=[
        FilterCondition(field="cost", range=RangeCondition(gt=0)),
        FilterCondition(field="cost", range_from_query=RangeFromQuery(lt="max_budget")),
    ])
    qv = {"max_budget": np.array([10.0])}
    mask = evaluate(f, t, qv)
    assert mask.tolist() == [[False, True, True, False]]


def test_match_text_from_query_basic():
    t = _table(title=["wireless mouse", "keyboard combo", "wireless keyboard"])
    f = Filter(must=[FilterCondition(field="title", match_text_from_query="phrase")])
    qv = {"phrase": np.array(["wireless", "keyboard"], dtype=object)}
    mask = evaluate(f, t, qv)
    assert mask.tolist() == [
        [True, False, True],
        [False, True, True],
    ]


def test_match_text_from_query_dedup_shares_result_for_identical_phrase():
    t = _table(title=["wireless mouse", "gaming keyboard"])
    f = Filter(must=[FilterCondition(field="title", match_text_from_query="phrase")])
    qv = {"phrase": np.array(["mouse", "keyboard", "mouse"], dtype=object)}
    mask = evaluate(f, t, qv)
    # queries 0 and 2 share an identical phrase -> identical row masks
    assert mask[0].tolist() == mask[2].tolist() == [True, False]
    assert mask[1].tolist() == [False, True]


def test_match_text_from_query_null_phrase_never_matches():
    t = _table(title=["wireless mouse", "gaming keyboard"])
    f = Filter(must=[FilterCondition(field="title", match_text_from_query="phrase")])
    qv = {"phrase": np.array(["mouse", None], dtype=object)}
    mask = evaluate(f, t, qv)
    assert mask[0].tolist() == [True, False]
    assert mask[1].tolist() == [False, False]


def test_match_text_from_query_blank_phrase_never_matches():
    """Regression test: a blank/whitespace-only per-query phrase must never
    reach `_match_text_mask` (zero words -> `text.split()` never sets its
    `mask`, so it returns `None` and the caller's `pc.fill_null(None, ...)`
    crashes) — found live against real query data where every query's
    `title` column happened to be an empty string."""
    t = _table(title=["wireless mouse", "gaming keyboard"])
    f = Filter(must=[FilterCondition(field="title", match_text_from_query="phrase")])
    qv = {"phrase": np.array(["mouse", "", "   "], dtype=object)}
    mask = evaluate(f, t, qv)
    assert mask.shape == (3, 2)
    assert mask[0].tolist() == [True, False]
    assert mask[1].tolist() == [False, False]
    assert mask[2].tolist() == [False, False]


def test_match_text_from_query_shared_word_across_phrases():
    """Two different phrases sharing a word (word-level dedup, not just
    phrase-level dedup) must each still combine correctly with their OTHER
    word(s) — the shared word's cached mask must not leak the other
    phrase's extra condition. Also exercises different casing of the same
    word ("Wireless" vs "wireless") landing on the same cache key without
    changing results, since `ignore_case=True` already makes them equivalent."""
    t = _table(title=["wireless mouse", "wireless keyboard", "wired mouse", "gaming chair"])
    f = Filter(must=[FilterCondition(field="title", match_text_from_query="phrase")])
    qv = {"phrase": np.array(["wireless mouse", "Wireless keyboard"], dtype=object)}
    mask = evaluate(f, t, qv)
    assert mask[0].tolist() == [True, False, False, False]
    assert mask[1].tolist() == [False, True, False, False]


def test_literal_then_regex_word_mask_matches_single_regex_pass():
    """Unit test for `_literal_then_regex_word_mask`'s two-pass shortcut:
    must agree exactly with a plain single regex pass, including the cases
    that make it non-trivial — a literal substring present but NOT at a
    word boundary (e.g. "taxes" contains "tax" but isn't the word "tax"),
    a null corpus value, and a word that appears nowhere at all (exercises
    the all-literal-miss early return, which skips the regex pass
    entirely)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    col = pa.chunked_array([pa.array(["tax season", "taxes are due", None, "no match here", "TAX"])])
    got = _literal_then_regex_word_mask("tax", col)
    want = pc.fill_null(
        pc.match_substring_regex(col, pattern=r"\btax\b", ignore_case=True), False,
    ).to_numpy(zero_copy_only=False)
    assert got.tolist() == want.tolist() == [True, False, False, False, True]

    # a word absent from every row: the literal prefilter alone must already
    # return all-False, without ever reaching the regex pass.
    absent = _literal_then_regex_word_mask("zzz_nowhere", col)
    assert absent.tolist() == [False, False, False, False, False]


def test_static_only_filter_stays_one_dimensional():
    """No per-query condition anywhere -> evaluate() returns (rows,), exactly
    the pre-per-query-filters shape/cost, not (n_queries, rows)."""
    t = _table(language=["eng", "fra"])
    f = Filter(must=[FilterCondition(field="language", match="eng")])
    mask = evaluate(f, t, None)
    assert mask.ndim == 1
    assert mask.tolist() == [True, False]


@pytest.mark.parametrize("group", ["must", "should", "must_not"])
def test_any_per_query_condition_in_any_group_promotes_to_two_dimensional(group):
    t = _table(tenant_id=["A", "B"])
    cond = FilterCondition(field="tenant_id", match_from_query="tenant_id")
    f = Filter(**{group: [cond]})
    qv = {"tenant_id": np.array(["A", "B"])}
    mask = evaluate(f, t, qv)
    assert mask.ndim == 2
    assert mask.shape == (2, 2)


def test_should_group_mixes_static_and_per_query_condition():
    t = _table(is_public=[True, False, False], tenant_id=["A", "B", "C"])
    f = Filter(should=[
        FilterCondition(field="is_public", match=True),
        FilterCondition(field="tenant_id", match_from_query="tenant_id"),
    ])
    qv = {"tenant_id": np.array(["B"])}
    mask = evaluate(f, t, qv)
    # row 0: public -> always eligible. row 1: matches this query's own tenant B.
    # row 2: neither public nor this query's tenant -> excluded.
    assert mask.tolist() == [[True, True, False]]


def test_must_group_mixes_static_and_per_query_condition():
    """Per-query condition listed FIRST in config order: `_static_first`
    reorders it behind the static one during evaluation, which must not
    change the result (AND is commutative) — only when the accumulator
    promotes to 2-D."""
    t = _table(tenant_id=["A", "B", "A", "C"], status=["active", "active", "archived", "active"])
    f = Filter(must=[
        FilterCondition(field="tenant_id", match_from_query="tenant_id"),
        FilterCondition(field="status", match="active"),
    ])
    qv = {"tenant_id": np.array(["A", "B"], dtype=object)}
    mask = evaluate(f, t, qv)
    assert mask.shape == (2, 4)
    # query 0 (tenant A): rows 0/2 are tenant A, but row 2 is archived.
    # query 1 (tenant B): row 1 is tenant B and active.
    assert mask.tolist() == [[True, False, False, False], [False, True, False, False]]


def test_static_first_orders_static_conditions_before_per_query_stably():
    static_a = FilterCondition(field="a", match="x")
    static_b = FilterCondition(field="b", range=RangeCondition(gt=0))
    per_query_c = FilterCondition(field="c", match_from_query="c_col")
    per_query_d = FilterCondition(field="d", range_from_query=RangeFromQuery(lt="d_max"))
    ordered = _static_first([per_query_c, static_a, per_query_d, static_b])
    assert ordered == [static_a, static_b, per_query_c, per_query_d]


def test_range_from_query_requires_a_bound():
    with pytest.raises(ValueError):
        RangeFromQuery()


def test_filter_condition_six_way_exclusivity():
    with pytest.raises(ValueError):
        FilterCondition(field="x", match="a", match_from_query="b")
    with pytest.raises(ValueError):
        FilterCondition(field="x", range_from_query=RangeFromQuery(lt="b"), match_text_from_query="c")
    # each alone is valid
    FilterCondition(field="x", match_from_query="b")
    FilterCondition(field="x", range_from_query=RangeFromQuery(lt="b"))
    FilterCondition(field="x", match_text_from_query="c")
