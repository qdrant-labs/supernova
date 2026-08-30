"""Things that must NOT change the answer.

`test_parity_matrix` pins nova-bf against two oracles under one particular
configuration. These pin the axes that configuration happens to fix — tiling,
sharing, query-row subsetting — because every one of them is a knob production
turns, and a knob that silently changes ground truth is worse than one that
fails loudly.

Each is checked against the same two oracles as well, not only against another
nova-bf run: two nova-bf runs agreeing with each other and both being wrong is
exactly the failure a self-comparison cannot see.
"""

from __future__ import annotations

import pytest

from . import cases as cases_mod
from . import compare, qdrant_ref
from .runner import run, spec

# A spread across modalities and filter kinds — static, per-query, text, date —
# small enough to re-run several times per device.
PROBES = [
    cases_mod.CASES_BY_NAME[n] for n in (
        "dedot_nofilter", "decos_match", "deeuc_datetime",
        "spdot_nofilter", "spcos_compound", "spdot_pqrange",
        "mudot_nofilter", "mucos_matchtext", "mudot_pqmatch",
    )
]
PROBE_IDS = [c.id for c in PROBES]

MATRIX_PARAMS = {"dense_batch_size": 37, "sparse_batch_size": 29,
                 "multivector_batch_size": 23}


@pytest.fixture(scope="session")
def whole_file_run(ds, device):
    """The same probes with NO batching at all — one slice per file. The
    coarsest tiling there is, against the matrix run's deliberately awkward
    one."""
    return run(ds, [c.spec() for c in PROBES], out_tag="wholefile", device=device)


@pytest.fixture(scope="session")
def tiny_batch_run(ds, device):
    """The finest tiling: batches far smaller than k, so every file becomes
    many slices and each search's top-K is assembled from a long chain of
    per-slice merges rather than one shot."""
    return run(ds, [c.spec() for c in PROBES], out_tag="tiny", device=device,
               params={"dense_batch_size": 5, "sparse_batch_size": 5,
                       "multivector_batch_size": 3, "multivector_query_block": 2})


@pytest.mark.parametrize("case", PROBES, ids=PROBE_IDS)
@pytest.mark.parametrize("run_name", ["whole_file_run", "tiny_batch_run"])
def test_tiling_does_not_change_the_answer(case, run_name, request, ds, oracle, device):
    """Re-tiling a matmul changes the ORDER its reductions accumulate in, so
    scores may move by a ULP or two — but no document may move in or out of
    the ranking beyond that boundary, and the answer must still be the
    oracle's.

    Compared against the naive oracle rather than against the matrix run, so
    a tiling bug that corrupted both runs identically still fails.
    """
    got = request.getfixturevalue(run_name)[case.name]
    want = oracle.topk(vector_type=case.vector_type, metric=case.metric,
                       k=case.k, filt=_filter_of(case, ds))
    for qi in range(len(ds.queries)):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=case.metric,
            label=f"[{device}] {run_name} {case.id} q{qi}")


@pytest.mark.parametrize("case", PROBES, ids=PROBE_IDS)
def test_a_search_alone_equals_the_same_search_sharing_a_run(
    case, ds, matrix_run, device
):
    """Listing many searches in one run is an IO/compute optimization — every
    search of a vector_type shares one batch grid, and one query matrix, with
    all the others. It must not be observable in the results.

    An UNFILTERED search is held to bit-equality: sharing cannot change its
    matmul at all, because the grid is the whole file either way. A FILTERED
    one is held to tolerance instead, and that is not a hedge — alone, its
    batch compacts to just its surviving rows, which is a genuinely different
    (smaller) matmul whose reductions accumulate in a different order.

    The unfiltered claim is routed through `compare.assert_same_ranking`
    because one path cannot honour it: multivector on CUDA is not
    bit-reproducible between two identical runs, let alone two arrangements of
    one (see `compare.scores_are_reproducible`).
    """
    solo = run(ds, [case.spec()], out_tag=f"solo_{case.name}", device=device,
               params=MATRIX_PARAMS)[case.name]
    shared = matrix_run[case.name]
    for qi in range(len(ds.queries)):
        label = f"[{device}] {case.id} q{qi}: solo vs shared"
        if case.filter_dict is None:
            compare.assert_same_ranking(
                solo[qi], shared[qi], metric=case.metric, device=device,
                vector_type=case.vector_type, label=label)
        else:
            compare.assert_scores_agree(solo[qi], shared[qi],
                                        metric=case.metric, label=label)


# --------------------------------------------------------------- rows subset


ROWS_PROBES = [c for c in PROBES if c.vector_type != "multivector"]


@pytest.mark.parametrize("case", ROWS_PROBES, ids=[c.id for c in ROWS_PROBES])
def test_query_row_subsets_partition_the_same_answer(case, ds, device):
    """`rows` is the QUERY-side selector, orthogonal to `filter`'s corpus-side
    one. Splitting the queries into two complementary subsets and running both
    in one config must reproduce, row for row, what one unsubsetted search
    returns.

    The two halves are `even`/`odd`, so between them they cover every query —
    the case `RowSelector` documents as bit-exact: the shared query matrix
    keeps its full height, so the scoring matmul is unchanged. A subset
    leaving rows unowned would shorten that matrix and is explicitly NOT
    bit-exact; that is a different claim and not what is asserted here.

    The baseline is the same search run ALONE, not the shared matrix run.
    That is not incidental: `rows` selects on the query axis, but the CORPUS
    axis has to be held fixed for a bit-exactness claim to mean anything, and
    it is not fixed between these two configs — a filtered search alone
    compacts its batch to its own surviving rows, while in the matrix run an
    unfiltered search in the same vector_type forces the grid to the whole
    file. Comparing across those two grids compares two different matmuls,
    and they differ in the last float32 ULP for reasons that have nothing to
    do with `rows`.
    """
    baseline = run(ds, [case.spec()], out_tag=f"rowsbase_{case.name}",
                   device=device, params=MATRIX_PARAMS)[case.name]
    even = spec(f"{case.name}_even", vector_type=case.vector_type,
                metric=case.metric, k=case.k, filter=case.filter_dict,
                rows={"column": "query_set", "isin": ["even"]})
    odd = spec(f"{case.name}_odd", vector_type=case.vector_type,
               metric=case.metric, k=case.k, filter=case.filter_dict,
               rows={"column": "query_set", "isin": ["odd"]})
    got = run(ds, [even, odd], out_tag=f"rows_{case.name}", device=device,
              params=MATRIX_PARAMS)

    covered = set()
    for half, name in (("even", even["name"]), ("odd", odd["name"])):
        expected = {qi for qi, q in enumerate(ds.queries)
                    if q["payload"]["query_set"] == half}
        assert set(got[name]) == expected, (
            f"[{device}] {case.id}: the {half!r} selector covered "
            f"{sorted(got[name])}, expected {sorted(expected)}")
        covered |= expected
        for qi in expected:
            compare.assert_identical(
                got[name][qi], baseline[qi],
                label=f"[{device}] {case.id} q{qi}: rows={half} vs unsubsetted run")
    assert covered == set(range(len(ds.queries)))


@pytest.mark.qdrant
@pytest.mark.parametrize("case", ROWS_PROBES, ids=[c.id for c in ROWS_PROBES])
def test_row_subsets_still_agree_with_qdrant(case, ds, client, collection, device):
    """The subset path against the live engine too, so "both halves match the
    full run" cannot pass by both being wrong in the same way."""
    sub = spec(f"{case.name}_sub", vector_type=case.vector_type,
               metric=case.metric, k=case.k, filter=case.filter_dict,
               rows={"column": "query_set", "isin": ["even"]})
    got = run(ds, [sub], out_tag=f"rowsq_{case.name}", device=device,
              params=MATRIX_PARAMS)[sub["name"]]
    picked = sorted(got)
    want = qdrant_ref.topk(client, collection, ds, vector_type=case.vector_type,
                           metric=case.metric, k=case.k,
                           filt=_filter_of(case, ds), queries=picked)
    for qi in picked:
        compare.assert_scores_agree(
            got[qi], want[qi], metric=case.metric,
            label=f"[{device}] {case.id} q{qi}: rows-subset vs qdrant")


def _filter_of(case, ds):
    from .test_parity_matrix import _filter_from_dict

    return _filter_from_dict(ds, case.filter_dict)
