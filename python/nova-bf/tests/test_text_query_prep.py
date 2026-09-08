"""R1: the query half of `evaluate()`'s text preparation is hoisted out of the
per-file path.

`filters.prepare_text_queries` tokenizes a filter's static phrases and every
per-query phrase once for a whole run; `evaluate(..., prep=...)` consumes it
instead of redoing `tokenize_many` on every corpus file. These tests pin that
the hoist is OUTPUT-NEUTRAL (same mask, bit for bit) and that a prep built for
the wrong filter fails loudly rather than filtering against the wrong phrases.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from nova_bf.config import Filter, FilterCondition
from nova_bf.filters import PackedRowMask, evaluate, prepare_text_queries
from nova_bf.tokenize import tokenize as tokenize_


def _rows(mask):
    return mask.unpack() if isinstance(mask, PackedRowMask) else mask


TEXT = [
    "The quick brown fox",
    "jumps over the LAZY dog",
    "quick, quick! brown-fox",
    None,
    "",
    "   ",
    "Fox",
    "dog dog dog",
]
URL = [
    "https://example.com/quick",
    "http://a.test/dog",
    "https://example.com/brown",
    "https://x.y/z",
    None,
    "https://example.com/quick/dog",
    "ftp://fox.example",
    "",
]


def _table():
    return pa.table({"text": pa.array(TEXT, type=pa.large_string()),
                     "url": pa.array(URL, type=pa.large_string())})


def _qvals(n=6):
    phrases = ["quick brown", "DOG", "fox", None, "!!!", "quick"][:n]
    urls = ["quick", "dog", None, "brown", "z", "example"][:n]
    return {"phrase": np.array(phrases, dtype=object),
            "uq": np.array(urls, dtype=object)}


def _filter():
    return Filter(
        must=(FilterCondition(field="text", match_text_from_query="phrase"),),
        should=(
            FilterCondition(field="url", match_text_from_query="uq"),
            FilterCondition(field="text", match_text="dog"),
        ),
    )


def test_prep_is_output_neutral():
    filt, table, qv = _filter(), _table(), _qvals()
    without = evaluate(filt, table, qv)
    with_prep = evaluate(filt, table, qv, None, prepare_text_queries(filt, qv))
    assert type(without) is type(with_prep)
    assert np.array_equal(_rows(without), _rows(with_prep))
    if isinstance(without, PackedRowMask):
        # the packed BYTES, not just the logical rows
        assert np.array_equal(without.packed, with_prep.packed)


def test_prep_is_reusable_across_files_unchanged():
    """The point of the hoist: one prep, many corpus files."""
    filt, qv = _filter(), _qvals()
    prep = prepare_text_queries(filt, qv)
    for take in ([0, 1, 2, 3], [4, 5, 6, 7], list(range(8))):
        table = _table().take(take)
        a = evaluate(filt, table, qv)
        b = evaluate(filt, table, qv, None, prep)
        assert np.array_equal(_rows(a), _rows(b))


def test_prep_covers_static_only_filter():
    filt = Filter(must=(FilterCondition(field="text", match_text="quick brown"),))
    table = _table()
    a = evaluate(filt, table, None)
    b = evaluate(filt, table, None, None, prepare_text_queries(filt, None))
    assert a.ndim == 1 and np.array_equal(a, b)


def test_prep_for_a_different_filter_raises():
    table, qv = _table(), _qvals()
    other = Filter(must=(FilterCondition(field="url", match_text_from_query="uq"),))
    prep = prepare_text_queries(other, qv)
    with pytest.raises(ValueError, match="does not cover"):
        evaluate(_filter(), table, qv, None, prep)


def test_prep_carries_the_null_phrase_convention():
    """A null/NaN/token-less phrase matches nothing — the prep must preserve
    that, not silently drop the query."""
    filt = Filter(must=(FilterCondition(field="text", match_text_from_query="phrase"),))
    qv = {"phrase": np.array(["quick", None, "!!!", float("nan")], dtype=object)}
    prep = prepare_text_queries(filt, qv)
    assert prep.cond_qsets[filt.must[0]] == [frozenset({"quick"}), None, None, None]
    got = _rows(evaluate(filt, _table(), qv, None, prep))
    assert not got[1].any() and not got[2].any() and not got[3].any()
    assert got[0].any()


def test_a_prep_built_for_a_different_query_COUNT_raises():
    """The other staleness shape, and the one the class docstring promises.

    A prep holds one tokenized phrase per query, so if `query_values` has a
    different height the prep answers for rows nobody asked about — and the
    mask comes back at the PREP's height, filled with the OLD phrases'
    answers. Silently: the condition-coverage check above passes (it is the
    same filter), and nothing downstream re-derives the height except
    `compute.select`, which a direct caller never reaches.
    """
    table = pa.table({"text": pa.array(["alpha beta", "beta gamma", "alpha", "zzz"])})
    filt = Filter(must=(FilterCondition(field="text", match_text_from_query="phrase"),))
    prep = prepare_text_queries(
        filt, {"phrase": np.array(["alpha", "beta", "gamma"], dtype=object)})

    for n in (1, 2, 4, 7):
        with pytest.raises(ValueError, match="built for 3 queries"):
            evaluate(filt, table,
                     {"phrase": np.array(["zzz"] * n, dtype=object)}, None, prep)

    # the matching height still works, and still answers for 3 queries
    got = evaluate(filt, table,
                   {"phrase": np.array(["alpha", "beta", "gamma"], dtype=object)},
                   None, prep)
    assert got.shape[0] == 3 if hasattr(got, "shape") else got.n_queries == 3


def test_the_count_check_covers_every_per_query_condition():
    """Two text conditions on different query columns: a stale height in
    EITHER must be caught, not just the first one checked."""
    table = pa.table({"a": pa.array(["x y", "y z"]), "b": pa.array(["p q", "q r"])})
    filt = Filter(must=(
        FilterCondition(field="a", match_text_from_query="pa_"),
        FilterCondition(field="b", match_text_from_query="pb_"),
    ))
    vals = {"pa_": np.array(["x", "y"], dtype=object),
            "pb_": np.array(["p", "q"], dtype=object)}
    prep = prepare_text_queries(filt, vals)

    for stale in ("pa_", "pb_"):
        bad = dict(vals)
        bad[stale] = np.array(["x", "y", "z"], dtype=object)
        with pytest.raises(ValueError, match="is stale"):
            evaluate(filt, table, bad, None, prep)


# --- evaluate()'s loud-failure guards ---------------------------------------
#
# These three raises exist for DIRECT `evaluate()` callers: `run_compute` draws
# every query column from one table, so it cannot reach them. Nothing tested
# them — all three could be deleted and the suite stayed green — and the source
# comment on the first says silently truncating instead "would be a correctness
# trap", which is exactly the trap an untested guard leaves open.


def test_text_conditions_on_query_columns_of_differing_lengths_raise():
    """Two per-query text conditions whose query columns disagree on height.
    There is no defensible answer: one condition wants n queries, the other m,
    and picking either silently answers for rows nobody asked about."""
    table = pa.table({"a": pa.array(["x y", "y z"]), "b": pa.array(["p q", "q r"])})
    filt = Filter(must=(
        FilterCondition(field="a", match_text_from_query="pa_"),
        FilterCondition(field="b", match_text_from_query="pb_"),
    ))
    qv = {"pa_": np.array(["x", "y"], dtype=object),
          "pb_": np.array(["p", "q", "r"], dtype=object)}
    with pytest.raises(ValueError, match="differing lengths"):
        evaluate(filt, table, qv, None)


def test_a_non_text_per_query_condition_of_another_height_raises():
    """`keep` is promoted to 2-D by a non-text per-query condition. If that
    condition's query column is a different height from the text conditions',
    the two masks cannot be combined row-for-row."""
    table = pa.table({"a": pa.array(["x y", "y z"]),
                      "lang": pa.array(["eng", "fra"])})
    filt = Filter(must=(
        FilterCondition(field="a", match_text_from_query="pa_"),
        FilterCondition(field="lang", match_from_query="ql_"),
    ))
    qv = {"pa_": np.array(["x", "y"], dtype=object),
          "ql_": np.array(["eng", "fra", "deu"], dtype=object)}
    with pytest.raises(ValueError, match="query column length mismatch"):
        evaluate(filt, table, qv, None)


def test_a_should_group_of_another_height_raises():
    """Same mismatch, reached through `rest_or` (the `should` accumulator)
    rather than `keep` — a separate guard, and separately untested."""
    table = pa.table({"a": pa.array(["x y", "y z"]),
                      "lang": pa.array(["eng", "fra"])})
    filt = Filter(
        must=(FilterCondition(field="a", match_text_from_query="pa_"),),
        should=(FilterCondition(field="lang", match_from_query="ql_"),),
    )
    qv = {"pa_": np.array(["x", "y"], dtype=object),
          "ql_": np.array(["eng", "fra", "deu"], dtype=object)}
    with pytest.raises(ValueError, match="query column length mismatch"):
        evaluate(filt, table, qv, None)


def _pq_conds():
    """The per-query conditions every filter below shares verbatim.

    Sharing them is what isolates the STATIC check: `cond_qsets` is keyed by
    the frozen condition objects, so an identical pair keeps the pre-existing
    per-query coverage check satisfied and any raise can only come from the
    static half.
    """
    return (FilterCondition(field="text", match_text_from_query="phrase"),
            FilterCondition(field="url", match_text_from_query="uq"))


def _with_static(phrase, field="text"):
    pq_text, pq_url = _pq_conds()
    return Filter(must=(pq_text,),
                  should=(pq_url, FilterCondition(field=field, match_text=phrase)))


def test_a_prep_missing_the_static_phrases_tokens_raises():
    """`prepare_text_queries` folds STATIC `match_text` into `field_tokens`,
    but the coverage check only walked `match_text_from_query` conditions. So
    a prep from a filter with the same per-query conditions and a DIFFERENT
    static phrase passed, and the static phrase's tokens were then absent from
    the `TokenGrid` — `TokenGrid.__getitem__` does `self._index[token]`, so
    that surfaced as a bare `KeyError('zebra')` from inside the tokenizer
    rather than as a statement about the prep.

    The premises below matter: without them a green result could mean the
    per-query check fired instead, which would leave the static path still
    unguarded.
    """
    table, qv = _table(), _qvals()
    donor, target = _with_static("dog"), _with_static("zebra")
    prep = prepare_text_queries(donor, qv)

    # premise 1: the per-query half IS covered, so the old check cannot fire
    assert all(c in prep.cond_qsets for c in _pq_conds()), (
        "the donor prep must cover the target's per-query conditions, or this "
        "test proves nothing about the static check")
    # premise 2: and the static token really is absent
    assert "zebra" not in prep.field_tokens["text"], (
        f"fixture broken: 'zebra' is already in {sorted(prep.field_tokens['text'])}")

    with pytest.raises(ValueError, match="static match_text"):
        evaluate(target, table, qv, None, prep)


def test_a_prep_that_never_saw_the_static_field_raises():
    """The other half: a static condition on a column the prep never touched.

    This one did NOT raise before — `_match_text_static_mask` finds no entry
    for the field, falls back to `_token_row_masks`, and returns the RIGHT
    answer. Correct by luck: the prep is still the wrong prep, and the run
    silently loses the tokenize-once-per-field hoist it exists to provide. The
    check is strict about it because a prep built from this filter always
    covers every static condition's field.
    """
    table, qv = _table(), _qvals()
    pq_text, _ = _pq_conds()
    # donor touches `text` only; target adds a static condition on `url`
    donor = Filter(must=(pq_text,))
    target = Filter(must=(pq_text,),
                    should=(FilterCondition(field="url", match_text="dog"),))
    prep = prepare_text_queries(donor, qv)

    assert pq_text in prep.cond_qsets, "premise: per-query half is covered"
    assert "url" not in prep.field_tokens, (
        f"fixture broken: prep already knows 'url' ({sorted(prep.field_tokens)})")

    with pytest.raises(ValueError, match="static match_text"):
        evaluate(target, table, qv, None, prep)


def test_a_covered_static_phrase_is_still_accepted():
    """The check must not over-reject: the answer with a matching prep has to
    stay byte-identical to the answer without one.

    A guard that rejected everything would satisfy both tests above.
    """
    table, qv = _table(), _qvals()
    filt = _with_static("dog")
    prep = prepare_text_queries(filt, qv)

    with_prep = _rows(evaluate(filt, table, qv, None, prep))
    without = _rows(evaluate(filt, table, qv))
    assert np.array_equal(with_prep, without), (
        "a filter's own prep changed its answer")

    # a phrase whose tokens the donor prep happens to cover is fine too: the
    # rule is token coverage, not filter identity
    donor = prepare_text_queries(_with_static("dog dog"), qv)
    assert set(tokenize_("dog")) <= donor.field_tokens["text"]
    ok = _rows(evaluate(_with_static("dog"), table, qv, None, donor))
    assert np.array_equal(ok, without), "a covered phrase must give the same rows"
