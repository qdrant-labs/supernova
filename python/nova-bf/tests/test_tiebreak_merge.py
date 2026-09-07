"""The cross-worker reduce, against an exact reference.

`merge` takes a fast path — an unstable score-only cut — and repairs only the
rows where a real tie means that cut decided something arbitrarily. Two things
can therefore go wrong independently: the DETECTION can miss an ambiguous row,
and the REPAIR can order one wrongly. Both are checked here by comparing the
whole reduce against a brute-force reference over the same candidates.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from nova_bf.merge import _ambiguous_rows, _id_tie_grid, _topk_merge

NEG = -np.inf


def _lists(rows, typ):
    return pa.array(rows, pa.list_(typ))


def _reference(per_partial, k, use_tie):
    """Exact global top-K per query: score descending, then tiebreak ascending.
    `per_partial` is a list of partials, each a list of per-query candidate
    lists of `(id, score, tie)`."""
    out = []
    n_q = len(per_partial[0])
    for q in range(n_q):
        cands = [c for p in per_partial for c in p[q]]
        key = (
            (lambda c: (-c[1], c[2])) if use_tie
            else (lambda c: (-c[1], c[0]))
        )
        out.append([(i, s) for i, s, _ in sorted(cands, key=key)[:k]])
    return out


def _run(per_partial, k, use_tie):
    score_lists, id_lists, tie_lists = [], [], []
    for p in per_partial:
        score_lists.append(_lists([[s for _, s, _ in row] for row in p], pa.float32()))
        id_lists.append(_lists([[i for i, _, _ in row] for row in p], pa.string()))
        tie_lists.append(_lists([[t for _, _, t in row] for row in p], pa.int64()))
    ids_arr, scores_arr, _ = _topk_merge(
        score_lists, id_lists, tie_lists if use_tie else None, k
    )
    return [
        list(zip(ids_arr[q].as_py(), scores_arr[q].as_py()))
        for q in range(len(per_partial[0]))
    ]


@pytest.mark.parametrize("use_tie", [True, False])
@pytest.mark.parametrize("seed", range(12))
def test_the_reduce_matches_an_exact_reference(seed, use_tie):
    """Scores are drawn from a tiny set so nearly every row ties somewhere,
    which is exactly the regime the fast path has to hand off correctly."""
    rng = np.random.default_rng(seed)
    n_partials = int(rng.integers(1, 5))
    n_q = 4
    k = int(rng.integers(1, 8))
    uid = 0
    per_partial = []
    for _ in range(n_partials):
        rows = []
        for _ in range(n_q):
            # <= k: `_topk_merge`'s contract, and what `compute` always writes
            n = int(rng.integers(0, k + 1))
            row = []
            for _ in range(n):
                # id order and tie order deliberately DISAGREE, so a reduce
                # that used the wrong one cannot pass
                row.append((f"id{uid:04d}", float(rng.integers(0, 4)), -uid))
                uid += 1
            # each partial's own candidates arrive score-descending, as a real
            # partial's do
            row.sort(key=lambda c: -c[1])
            rows.append(row)
        per_partial.append(rows)

    got = _run(per_partial, k, use_tie)
    want = _reference(per_partial, k, use_tie)
    for q in range(n_q):
        assert got[q] == want[q], f"seed={seed} use_tie={use_tie} query={q}"


def test_padding_never_surfaces_as_a_hit():
    """A query with fewer than k candidates keeps only its real ones — the
    `-inf` pad must not be emitted just because its slot sorts well."""
    p = [[[("a", 1.0, 0)]], [[("b", 1.0, 1)]]]
    got = _run(p, 5, True)
    assert got[0] == [("a", 1.0), ("b", 1.0)]


def test_an_infinite_score_is_kept():
    """The gate is `> -inf`, not `isfinite`; `+inf` is a real hit."""
    p = [[[("a", float("inf"), 0), ("b", 1.0, 1)]]]
    got = _run(p, 2, True)
    assert got[0][0] == ("a", float("inf"))


def test_ambiguity_detection_catches_order_and_membership():
    scores = np.array([[5.0, 5.0, 3.0, NEG],
                       [9.0, 8.0, 7.0, NEG],
                       [4.0, 4.0, 4.0, NEG]], dtype=np.float32)
    # kept the top 2 of each row
    top = np.array([[5.0, 5.0], [9.0, 8.0], [4.0, 4.0]], dtype=np.float32)
    amb = _ambiguous_rows(scores, top)
    assert bool(amb[0]), "two SELECTED hits share a score -> order is arbitrary"
    assert not bool(amb[1]), "strictly decreasing, nothing to resolve"
    assert bool(amb[2]), "a third candidate ties the cut -> membership arbitrary"


def test_a_short_row_is_never_membership_ambiguous():
    """Fewer than k real candidates: everything real is kept, so which survived
    cannot have been arbitrary — only the order among equals matters."""
    scores = np.array([[2.0, NEG, NEG, NEG]], dtype=np.float32)
    top = np.array([[2.0, NEG]], dtype=np.float32)
    assert not bool(_ambiguous_rows(scores, top)[0])


def _rank_row(vals):
    """The ranks `_topk_merge` would compute for ONE ambiguous row of `vals`."""
    flat = pa.array(vals, pa.large_string())
    row_idx = np.zeros(len(vals), dtype=np.int64)
    col = np.arange(len(vals))
    return _id_tie_grid([(row_idx, col, flat)], np.array([0]), 1, len(vals))[0]


def test_id_ranks_are_order_preserving():
    """The one property the tie-break rests on: ordering by rank IS ordering by
    id. Ranks are a total order (equal ids take their position, they do not
    share a number) — two partials cannot both hold the same corpus row, so
    within a row equal ids never arise, and where they somehow did the lower
    column still wins either way."""
    vals = ["zebra", "apple", "mango", "apple"]
    r = _rank_row(vals)
    assert [vals[i] for i in np.argsort(r)] == sorted(vals)
    assert len(set(r.tolist())) == len(vals), "ranks must be a total order"


def test_id_ranks_compare_the_whole_id_not_a_prefix():
    """The reduce's exactness for string ids rests on this: two ids agreeing in
    a long head must still separate. The lane path packs ceil(W/8) uint64s and
    covers every byte, so this holds on GPU too."""
    head = "<urn:uuid:" + "0" * 40
    assert _rank_row([head + "b", head + "a"]).tolist() == [1, 0]


def test_non_ascii_ids_still_rank_correctly():
    """Ranking must hold for ids the fixed-width lane path declines."""
    vals = ["zeta", "café", "cafe", "\u03a9mega", "apple"]
    r = _rank_row(vals)
    assert [vals[i] for i in np.argsort(r)] == sorted(vals)


def test_ranking_agrees_with_the_order_compute_sorted_on():
    """`compute` ranks ids with `build_ordinals`; `merge` now calls the SAME
    function, so a tie cannot be broken one way inside a worker and the other
    way across workers. Pinned against pyarrow directly so the guarantee is
    checked, not just the call."""
    vals = ["zebra", "Apple", "apple", "café", "cafe", "10", "9", "", "~x"]
    mine = np.argsort(_rank_row(vals), kind="stable")
    theirs = np.asarray(pc.sort_indices(pa.array(vals), sort_keys=[("", "ascending")]))
    assert mine.tolist() == theirs.tolist()


def test_ids_from_different_partials_rank_against_each_other():
    """The reason every partial is ranked in ONE `build_ordinals` call. Ranking
    each alone would number both partials from zero, and a cross-partial tie
    would then be decided by whichever column happened to be lower."""
    a = pa.array(["m", "z"], pa.large_string())
    b_ = pa.array(["a", "q"], pa.large_string())
    grid = _id_tie_grid(
        [(np.zeros(2, dtype=np.int64), np.array([0, 1]), a),
         (np.zeros(2, dtype=np.int64), np.array([2, 3]), b_)],
        np.array([0]), 1, 4)[0]
    # a, m, q, z  ->  the second partial's "a" must outrank the first's "m"
    assert [["m", "z", "a", "q"][i] for i in np.argsort(grid)] == ["a", "m", "q", "z"]


def test_unranked_slots_sort_past_every_real_hit():
    """Only ambiguous rows are ranked and only real candidates are scattered, so
    every other slot keeps int64 max — which must never outrank a real id."""
    grid = _id_tie_grid(
        [(np.zeros(1, dtype=np.int64), np.array([1]),
          pa.array(["z"], pa.large_string()))],
        np.array([0]), 1, 3)[0]
    assert grid[1] < grid[0] and grid[1] < grid[2]
    assert grid[0] == grid[2] == np.iinfo(np.int64).max


def test_rows_that_are_not_ambiguous_are_never_ranked():
    """Ranking is the expensive half; it must touch only the rows that need it."""
    flat = pa.array(["b", "a", "d", "c"], pa.large_string())
    scatter = [(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), flat)]
    grid = _id_tie_grid(scatter, np.array([1]), 2, 2)
    assert grid.shape == (1, 2)
    # only row 1's ids ("d", "c") were ranked, and against each other
    assert grid[0, 0] > grid[0, 1]


def test_a_partial_longer_than_k_is_refused():
    """Each partial owns k columns of the candidate grid, so a longer row would
    overwrite the next partial's candidates — silent corruption for every
    partial but the last, which raised an IndexError instead."""
    score_lists = [_lists([[1.0, 2.0, 3.0]], pa.float32())]
    id_lists = [_lists([["a", "b", "c"]], pa.string())]
    with pytest.raises(RuntimeError, match="more than k candidates"):
        _topk_merge(score_lists, id_lists, None, 2)


def test_mismatched_hit_tie_lengths_are_refused():
    """A hit_tie column split differently from hit_scores would break ties
    against the wrong hits — silently, and only for some queries."""
    score_lists = [_lists([[1.0, 2.0]], pa.float32())]
    id_lists = [_lists([["a", "b"]], pa.string())]
    tie_lists = [_lists([[0]], pa.int64())]        # one value for two hits
    with pytest.raises(RuntimeError, match="line up row for row"):
        _topk_merge(score_lists, id_lists, tie_lists, 2)
