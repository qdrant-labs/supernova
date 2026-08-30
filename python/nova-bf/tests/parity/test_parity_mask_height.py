"""Three query-axis heights that must not be confused for one another.

A run juggles three different "how many queries" numbers at once:

  1. the queries FILE's row count;
  2. each vector_type's shared score matrix height — the union of the `rows`
     of every spec of that vt (`_resolve_spec_rows`);
  3. each FILTER's per-query mask height — the union of the `rows` of every
     spec sharing THAT filter (`run_compute`'s `filter_rows`), which is what
     the mask-height memory work narrowed from (1) to this.

They coincide on most fixtures, which is exactly what makes confusing them
dangerous: as `_process_shared_batch` puts it, a mask read at the wrong height
masks the wrong queries — it does not raise. The result is a well-formed
top-K for the wrong query, and no error anywhere.

So this file builds runs where all three numbers are deliberately DIFFERENT,
and checks every spec against the oracles per query. A height mix-up shifts
which queries a filter constrains, which the per-query oracles see immediately.

Only a per-query filter with a `match_text`/`match_text_from_query` leaf takes
the CPU-fallback path that materializes the packed `(n_queries, rows)` mask at
all, so those are the filters used here — a GPU-native per-query filter never
builds one and would test nothing about its height.

One asymmetry worth knowing before adding an assertion here. `_unpack_query_axis`
reads the first `n` bits of a packed array, so getting the height WRONG is only
a correctness bug in one direction:

  * too SHORT truncates the mask, and a spec whose queries run past the end
    reads rows that are not there — caught (mutating the height to the
    vector_type's, which is what the pre-narrowing bug did, or to the smallest
    filter's, fails this suite);
  * too TALL only appends all-False padding rows, and `spec_qsel` never indexes
    them, so the answer is unchanged. That is the CONSERVATIVE direction, and
    it is exactly what the code did before the narrowing (the whole file's
    height is the maximum). A test that failed on an over-tall mask would be
    pinning the memory optimization rather than an answer — deliberately not
    asserted, for the same reason `test_parity_combinations` lets a
    `_union_keep` that stops compacting pass.
"""

from __future__ import annotations

import math

import pytest

from . import compare, qdrant_ref
from .cases import K
from .runner import run, spec
from .test_parity_matrix import _filter_from_dict

# Two DISTINCT per-query text filters. Both take the CPU fallback (each has a
# `match_text_from_query` leaf); being distinct `Filter` values, they get
# separate masks and therefore separate heights.
TEXT_A = {"must": [{"field": "title", "match_text_from_query": "q_phrase"}]}
TEXT_B = {
    "must": [{"field": "title", "match_text_from_query": "q_phrase"},
             {"field": "views", "range": {"gte": 500}}],
}
# A per-query text filter combined with a static one and a must_not, so the
# fused query-major combine (`filters.evaluate`'s text path) runs alongside
# ordinary condition-major masks under a narrowed height.
TEXT_C = {
    "must": [{"field": "title", "match_text_from_query": "q_phrase"}],
    "should": [{"field": "language", "match": "eng"},
               {"field": "language", "match": "fra"}],
    "must_not": [{"field": "tier", "match": "gold"}],
}

EVEN = {"column": "query_set", "isin": ["even"]}
ODD = {"column": "query_set", "isin": ["odd"]}
G0 = {"column": "query_third", "isin": ["g0"]}
G01 = {"column": "query_third", "isin": ["g0", "g1"]}


def _rows_of(dataset, selector):
    """Which query indices a `rows` selector names. Takes the dataset
    explicitly rather than closing over a fixture, since this file uses the
    taller `ds_wide` while the helper itself is dataset-agnostic."""
    if selector is None:
        return set(range(len(dataset.queries)))
    col, want = selector["column"], set(selector["isin"])
    return {qi for qi, q in enumerate(dataset.queries) if q["payload"][col] in want}


# (name, vector_type, metric, filter, rows) — the layout is the test.
#
# dense: two specs with DIFFERENT filters and DIFFERENT, complementary `rows`.
#   -> file height 8, dense vt union = even ∪ odd = 8 (full),
#      but TEXT_A's height = |even| = 4 and TEXT_B's = |odd| = 4.
#   All three numbers differ, and the two filters differ from each other.
# sparse: a third filter over a THIRD grouping, so its height (|g0| = 3) is
#   neither of the dense ones and its vt union (|g0 ∪ g0,g1| = 6) is neither
#   the file height nor its own filter height.
LAYOUT = [
    ("d_a_even", "dense", "cosine", TEXT_A, EVEN),
    ("d_b_odd", "dense", "dot", TEXT_B, ODD),
    ("s_c_g0", "sparse", "dot", TEXT_C, G0),
    ("s_a_g01", "sparse", "cosine", TEXT_A, G01),
]


@pytest.fixture(scope="session")
def mixed_heights_run(ds_wide, device):
    return run(
        ds_wide,
        [spec(name, vector_type=vt, metric=m, k=K, filter=f, rows=r)
         for name, vt, m, f, r in LAYOUT],
        out_tag="maskheight", device=device,
        params={"dense_batch_size": 37, "sparse_batch_size": 29},
    )


def test_the_three_heights_really_are_different(ds_wide):
    """A guard on the fixture. If the file height, the vector_type unions and
    the filter unions all came out equal, every assertion below would still
    pass while testing nothing — the exact coincidence the narrowing is
    dangerous under."""
    file_h = len(ds_wide.queries)
    vt_union = {}
    filt_union = {}
    for name, vt, _m, f, r in LAYOUT:
        rows = _rows_of(ds_wide, r)
        vt_union[vt] = vt_union.get(vt, set()) | rows
        key = str(f)
        filt_union[key] = filt_union.get(key, set()) | rows

    assert file_h == 26
    # dense's vt union covers the whole file, so its filters' heights sit
    # below both the file height and their own vector_type's — the case where
    # narrowing actually narrows.
    assert len(vt_union["dense"]) == file_h, vt_union
    # sparse's vt union is a strict subset, and its filters sit on either side
    # of it, so no filter height is recoverable from its vector_type's.
    assert len(vt_union["sparse"]) == 18, vt_union

    heights = sorted(len(v) for v in filt_union.values())
    assert heights == [9, 13, 22], {k: len(v) for k, v in filt_union.items()}
    assert file_h not in heights, (
        "every filter mask here must be strictly shorter than the queries "
        "file, or the narrowing under test is a no-op")
    # The masks are bit-PACKED along the query axis, so a height difference is
    # only observable when it changes the BYTE count — at 8 queries every
    # height is one byte and reading one at the wrong height cannot be seen.
    # Requiring more than one distinct byte count is what keeps this suite
    # able to fail at all; see `corpus.build`'s `n_queries`.
    assert len({math.ceil(h / 8) for h in heights}) > 1, (
        f"heights {heights} all pack into the same number of bytes — the "
        "mask-height tests would be blind")


@pytest.mark.parametrize("entry", LAYOUT, ids=[e[0] for e in LAYOUT])
def test_each_spec_answers_for_its_own_queries(entry, ds_wide, oracle_wide,
                                               mixed_heights_run, device):
    """Every spec must cover exactly the queries its `rows` names, and answer
    each of them exactly as the naive oracle does under its OWN filter.

    A height mix-up does not lose queries — it answers the wrong one — so the
    per-query comparison, not the coverage check, is what catches it. The
    coverage check is here to stop a spec silently answering nothing.
    """
    name, vt, metric, fdict, rsel = entry
    got = mixed_heights_run[name]
    expected_rows = _rows_of(ds_wide, rsel)
    assert set(got) == expected_rows, (
        f"[{device}] {name} covered {sorted(got)}, expected {sorted(expected_rows)}")

    want = oracle_wide.topk(vector_type=vt, metric=metric, k=K,
                       filt=_filter_from_dict(ds_wide, fdict),
                       queries=sorted(expected_rows))
    for qi in sorted(expected_rows):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=metric,
            label=f"[{device}] mixed-heights {name} q{qi}")


@pytest.mark.qdrant
@pytest.mark.parametrize("entry", LAYOUT, ids=[e[0] for e in LAYOUT])
def test_mixed_heights_agree_with_qdrant(entry, ds_wide, client, collection,
                                         mixed_heights_run, device):
    name, vt, metric, fdict, rsel = entry
    got = mixed_heights_run[name]
    picked = sorted(_rows_of(ds_wide, rsel))
    want = qdrant_ref.topk(client, collection, ds_wide, vector_type=vt, metric=metric,
                           k=K, filt=_filter_from_dict(ds_wide, fdict), queries=picked)
    for qi in picked:
        compare.assert_scores_agree(
            got[qi], want[qi], metric=metric,
            label=f"[{device}] mixed-heights {name} q{qi} vs qdrant")


@pytest.mark.parametrize("entry", LAYOUT, ids=[e[0] for e in LAYOUT])
def test_narrowing_a_filters_height_does_not_change_its_answer(
    entry, ds_wide, mixed_heights_run, device
):
    """The A/B for the narrowing itself.

    The same spec is run again as the ONLY user of its filter and with NO
    `rows` selector, which puts that filter's mask back at full file height —
    the pre-narrowing shape. Restricted to the queries the narrowed run owns,
    the two must agree exactly: narrowing a mask's query axis is a memory
    optimization, and an optimization that changed an answer would be a bug
    rather than a saving.

    Membership is asserted exactly. The corpus grid is identical between the
    two runs for this spec (same filter, same batch sizes), so there is no
    reduction-order excuse available here.
    """
    name, vt, metric, fdict, rsel = entry
    full = run(ds_wide, [spec(name, vector_type=vt, metric=metric, k=K, filter=fdict)],
               out_tag=f"fullheight_{name}", device=device,
               params={"dense_batch_size": 37, "sparse_batch_size": 29})[name]
    for qi in sorted(_rows_of(ds_wide, rsel)):
        compare.assert_same_membership(
            mixed_heights_run[name][qi], full[qi],
            label=f"[{device}] {name} q{qi}: narrowed mask vs full-height mask")
        compare.assert_scores_agree(
            mixed_heights_run[name][qi], full[qi], metric=metric,
            label=f"[{device}] {name} q{qi}: narrowed mask vs full-height mask")


def test_a_filter_shared_by_specs_with_different_rows_spans_their_union(
    ds_wide, oracle_wide, device
):
    """When two specs with DIFFERENT `rows` share ONE filter, that filter's
    mask must span the UNION of both — narrowing it to either spec's own rows
    would leave the other reading past the end of its own mask, or reading a
    neighbour's row.

    Both specs are checked, because the failure is asymmetric: the spec whose
    rows came first would look fine.
    """
    specs = [
        spec("shared_even", vector_type="dense", metric="cosine", k=K,
             filter=TEXT_A, rows=EVEN),
        spec("shared_g0", vector_type="dense", metric="cosine", k=K,
             filter=TEXT_A, rows=G0),
    ]
    got = run(ds_wide, specs, out_tag="shared_filter_rows", device=device)
    filt = _filter_from_dict(ds_wide, TEXT_A)
    for s, rsel in zip(specs, (EVEN, G0)):
        picked = sorted(_rows_of(ds_wide, rsel))
        assert set(got[s["name"]]) == set(picked)
        want = oracle_wide.topk(vector_type="dense", metric="cosine", k=K, filt=filt,
                           queries=picked)
        for qi in picked:
            compare.assert_scores_agree(
                got[s["name"]][qi], want[qi], metric="cosine",
                label=f"[{device}] shared-filter-different-rows {s['name']} q{qi}")
