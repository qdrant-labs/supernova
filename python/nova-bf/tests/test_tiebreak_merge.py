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

from nova_bf.merge import _ambiguous_rows, _string_ranks, _topk_merge

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
    ids_arr, scores_arr = _topk_merge(
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


def test_string_ranks_are_order_preserving_and_dense():
    ids = np.array([["zebra", "apple", "mango", "apple"]], dtype=object)
    r = _string_ranks(ids)
    assert r.tolist() == [[2, 0, 1, 0]]
    flat = ids[0].tolist()
    assert [flat[i] for i in np.argsort(r[0])] == sorted(flat)


def test_string_ranks_compare_the_whole_id_not_a_prefix():
    """The reduce's exactness for string ids rests on this: two ids agreeing in
    a long head must still separate."""
    head = "<urn:uuid:" + "0" * 40
    ids = np.array([[head + "b", head + "a"]], dtype=object)
    assert _string_ranks(ids).tolist() == [[1, 0]]


def test_string_ranks_order_the_same_whichever_width_they_take():
    """Ranking narrows to `astype("S")` (1 byte/char) and falls back to
    `astype("U")` (4) only for non-ASCII, so the two must agree wherever both
    apply — bytewise order over UTF-8 IS code-point order, and it is the order
    `compute` ranked on. A divergence would move the winner with `--num-jobs`.
    """
    ascii_ids = np.array([["zebra", "Apple", "apple", "1", "~", "", "A"]], dtype=object)
    wide = np.unique(ascii_ids.astype("U"), return_inverse=True)[1].reshape(ascii_ids.shape)
    assert _string_ranks(ascii_ids).tolist() == wide.tolist()


def test_non_ascii_ids_still_rank_correctly():
    """The narrow path raises on these; the fallback must still produce ranks
    that sort in code-point order."""
    ids = np.array([["zeta", "café", "cafe", "Ωmega", "apple"]], dtype=object)
    r = _string_ranks(ids)[0]
    flat = ids[0].tolist()
    assert [flat[i] for i in np.argsort(r)] == sorted(flat)


def test_ranking_agrees_with_the_order_compute_sorted_on():
    """`compute` ranks ids with pyarrow (bytewise over UTF-8); `merge` ranks the
    same ids here. The two sides must not disagree, or a tie would be broken one
    way inside a worker and the other way across workers."""
    vals = ["zebra", "Apple", "apple", "café", "cafe", "10", "9", "", "~x"]
    ids = np.array([vals], dtype=object)
    mine = np.argsort(_string_ranks(ids)[0], kind="stable")
    theirs = np.asarray(pc.sort_indices(pa.array(vals), sort_keys=[("", "ascending")]))
    assert mine.tolist() == theirs.tolist()


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
