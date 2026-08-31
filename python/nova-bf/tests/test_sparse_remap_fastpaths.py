"""Two reader-thread fast paths on the sparse remap, and the invariants they
are allowed to assume.

Both replace a general implementation with a cheaper one that is only valid
under a precondition, so both are tested the same way: against the general
implementation they replaced, on inputs that do and do not meet the
precondition.

  * `_coalesce_by_row_col` returns its inputs untouched when they already
    arrive sorted by (row, col) with no pair repeated — which is what a sparse
    embedder emitting each row's token ids ascending and deduped produces. The
    sort and the duplicate SUM still have to happen when they don't, because
    that is what makes a hash-colliding embedder correct.
  * `_vocab_lookup` answers from a direct table indexed by token id instead of
    a binary search, when the vocabulary's largest id makes that table small
    enough to be worth building.

The risk in both is silence: a fast path taken when its precondition does NOT
hold returns a plausible, wrong answer. So there are tests that the fast path
is actually taken (and actually skipped), not just that the answer is right.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova_bf import compute as compute_mod


# --------------------------------------------------------------------------
# _coalesce_by_row_col
# --------------------------------------------------------------------------

def _reference_coalesce(row_ids, col_ids, values):
    """The general implementation, spelled out independently: group (row, col)
    pairs, sum each group's values, return sorted by row then col. Deliberately
    written with dicts rather than as a copy of the numpy version, so it cannot
    share a bug with it."""
    acc: dict[tuple[int, int], float] = {}
    for r, c, v in zip(row_ids.tolist(), col_ids.tolist(), values.tolist()):
        acc[(r, c)] = acc.get((r, c), 0.0) + v
    keys = sorted(acc)
    return (
        np.array([k[0] for k in keys], dtype=row_ids.dtype),
        np.array([k[1] for k in keys], dtype=col_ids.dtype),
        np.array([acc[k] for k in keys], dtype=values.dtype),
    )


def _check(row_ids, col_ids, values):
    got = compute_mod._coalesce_by_row_col(row_ids, col_ids, values)
    want = _reference_coalesce(row_ids, col_ids, values)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=0, atol=0)
    return got


def _csr_like(rng, n_rows, max_per_row, vocab_size, *, sorted_unique):
    """Per-row (row, col, value) triples in CSR order, either strictly
    ascending within each row (the arrival case) or shuffled with repeats."""
    rows, cols, vals = [], [], []
    for r in range(n_rows):
        n = int(rng.integers(0, max_per_row + 1))
        if n == 0:
            continue
        if sorted_unique:
            c = np.sort(rng.choice(vocab_size, size=min(n, vocab_size), replace=False))
        else:
            c = rng.integers(0, vocab_size, size=n)  # unsorted, may repeat
        rows.append(np.full(len(c), r, dtype=np.int64))
        cols.append(np.asarray(c, dtype=np.int64))
        vals.append(rng.random(len(c)).astype(np.float32))
    if not rows:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


@pytest.mark.parametrize("seed", range(12))
def test_already_sorted_and_unique_matches_reference(seed):
    rng = np.random.default_rng(seed)
    _check(*_csr_like(rng, 40, 9, 50, sorted_unique=True))


@pytest.mark.parametrize("seed", range(12))
def test_unsorted_with_duplicates_matches_reference(seed):
    rng = np.random.default_rng(1000 + seed)
    _check(*_csr_like(rng, 40, 9, 12, sorted_unique=False))


def test_duplicates_are_summed_not_overwritten():
    """The property the whole function exists for — a repeated (row, col) must
    contribute the SUM. A fast path that mistook this input for pre-sorted
    would return two rows for one column and a silently wrong score."""
    rows = np.array([0, 0, 0, 1], dtype=np.int64)
    cols = np.array([5, 5, 7, 2], dtype=np.int64)
    vals = np.array([1.5, 2.25, 1.0, 4.0], dtype=np.float32)
    r, c, v = compute_mod._coalesce_by_row_col(rows, cols, vals)
    np.testing.assert_array_equal(r, [0, 0, 1])
    np.testing.assert_array_equal(c, [5, 7, 2])
    np.testing.assert_allclose(v, [3.75, 1.0, 4.0])


def test_fast_path_returns_values_bit_identically():
    """Not just close: the slow path routes values through a float64
    accumulator, and the fast path skips it. With no duplicates those must
    agree to the last bit, or a run's scores would shift depending on which
    path its data happened to take."""
    rng = np.random.default_rng(7)
    n = 4000
    rows = np.repeat(np.arange(n // 4, dtype=np.int64), 4)
    cols = np.tile(np.array([1, 5, 9, 40], dtype=np.int64), n // 4)
    vals = (rng.random(n).astype(np.float32) - 0.5) * 1e7
    fast = compute_mod._coalesce_by_row_col(rows, cols, vals)
    # force the slow path with an input that is one swap away from sorted
    srows, scols, svals = rows.copy(), cols.copy(), vals.copy()
    scols[0], scols[1] = scols[1], scols[0]
    svals[0], svals[1] = svals[1], svals[0]
    slow = compute_mod._coalesce_by_row_col(srows, scols, svals)
    np.testing.assert_array_equal(fast[2].view(np.uint32), slow[2].view(np.uint32))


def test_fast_path_is_actually_taken_when_sorted(monkeypatch):
    rng = np.random.default_rng(3)
    rows, cols, vals = _csr_like(rng, 30, 6, 40, sorted_unique=True)
    monkeypatch.setattr(
        np, "lexsort", lambda *a, **k: pytest.fail("sorted input still sorted"))
    got = compute_mod._coalesce_by_row_col(rows, cols, vals)
    # returned as-is, not copied
    assert got[0] is rows and got[1] is cols and got[2] is vals


def test_slow_path_is_taken_when_not_sorted(monkeypatch):
    """The complement: if the precondition test is too permissive, unsorted
    input silently skips the sort. Prove the sort still runs."""
    called = []
    real = np.lexsort
    monkeypatch.setattr(np, "lexsort", lambda *a, **k: (called.append(1), real(*a, **k))[1])
    for cols in ([7, 5], [5, 5]):          # out of order, then a duplicate
        compute_mod._coalesce_by_row_col(
            np.array([0, 0], np.int64), np.array(cols, np.int64),
            np.array([1.0, 2.0], np.float32),
        )
    assert len(called) == 2


@pytest.mark.parametrize("n", [0, 1, 2])
def test_degenerate_lengths(n):
    rows = np.arange(n, dtype=np.int64)
    cols = np.arange(n, dtype=np.int64)
    vals = np.ones(n, dtype=np.float32)
    r, c, v = compute_mod._coalesce_by_row_col(rows, cols, vals)
    assert len(r) == len(c) == len(v) == n


def test_row_boundary_does_not_hide_a_descending_column():
    """The precondition is lexicographic on (row, col), so a column that DROPS
    across a row boundary is fine, while one that drops within a row is not.
    A check that only looked at `col_ids` would confuse the two."""
    rows = np.array([0, 0, 1, 1], dtype=np.int64)
    cols = np.array([3, 9, 1, 4], dtype=np.int64)   # 9 -> 1 crosses a row
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    got = compute_mod._coalesce_by_row_col(rows, cols, vals)
    assert got[0] is rows                            # fast path: legitimately sorted
    _check(rows, cols, vals)


# --------------------------------------------------------------------------
# _vocab_lookup
# --------------------------------------------------------------------------

def _reference_lookup(vocab, ids):
    """The binary-search implementation the table replaced."""
    if len(vocab) == 0:
        return np.full(len(ids), -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(vocab, ids), len(vocab) - 1)
    return np.where(vocab[pos] == ids, pos, -1).astype(np.int64)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("dtype", [np.uint32, np.int64, np.int32])
def test_lookup_matches_binary_search(seed, dtype):
    rng = np.random.default_rng(seed)
    vocab = np.unique(rng.integers(0, 400, size=rng.integers(1, 60))).astype(dtype)
    # ids deliberately span below, inside, and PAST the vocabulary's largest id
    ids = rng.integers(0, 700, size=200).astype(dtype)
    got = compute_mod._vocab_lookup(vocab, ids)
    np.testing.assert_array_equal(got, _reference_lookup(vocab, ids))
    assert got.dtype == np.int64


def test_lookup_edges():
    vocab = np.array([0, 1, 9, 10], dtype=np.uint32)
    ids = np.array([0, 1, 2, 9, 10, 11, 4_000_000_000], dtype=np.uint32)
    np.testing.assert_array_equal(
        compute_mod._vocab_lookup(vocab, ids), [0, 1, -1, 2, 3, -1, -1])


def test_lookup_empty_vocab_and_empty_ids():
    empty_u32 = np.zeros(0, dtype=np.uint32)
    np.testing.assert_array_equal(
        compute_mod._vocab_lookup(empty_u32, np.array([1, 2], np.uint32)), [-1, -1])
    assert len(compute_mod._vocab_lookup(np.array([3], np.uint32), empty_u32)) == 0


def test_lookup_falls_back_when_table_would_be_too_big(monkeypatch):
    """A hashed embedder using the whole uint32 range would need a 34 GB table.
    Past the budget the search must still be used, and still be right."""
    vocab = np.array([1, 5, 4_000_000_000], dtype=np.uint32)
    ids = np.array([1, 2, 5, 4_000_000_000], dtype=np.uint32)
    want = [0, -1, 1, 2]
    np.testing.assert_array_equal(compute_mod._vocab_lookup(vocab, ids), want)
    # and the small case still agrees after the budget is squeezed to nothing
    monkeypatch.setattr(compute_mod, "_VOCAB_LUT_MAX_BYTES", 0)
    small = np.array([2, 7], dtype=np.uint32)
    ids2 = np.array([1, 2, 7, 9], dtype=np.uint32)
    np.testing.assert_array_equal(
        compute_mod._vocab_lookup(small, ids2), _reference_lookup(small, ids2))


def test_lookup_is_monotone_so_remap_preserves_row_order():
    """`_coalesce_by_row_col`'s fast path relies on this: if a row's raw ids
    ascend, their vocabulary positions ascend too, so dropping the absent ones
    leaves the row still strictly ascending."""
    rng = np.random.default_rng(4)
    vocab = np.unique(rng.integers(0, 1000, size=200)).astype(np.uint32)
    ids = np.sort(rng.choice(1000, size=300, replace=False)).astype(np.uint32)
    pos = compute_mod._vocab_lookup(vocab, ids)
    kept = pos[pos >= 0]
    assert np.all(np.diff(kept) > 0)


# --------------------------------------------------------------------------
# _vocab_lookup is polymorphic — the table only applies to some of its inputs
# --------------------------------------------------------------------------
# This is the gap the first version of these tests had: they covered only
# integer ids, so they passed while `match_from_query`'s STRING vocabularies
# crashed the whole parity suite. Each case below is a way the table would be
# wrong, not merely slower.

def test_lookup_accepts_string_vocab():
    """`match_from_query` on a keyword field calls this with `<U3` language
    codes. A table indexed by id cannot index those at all."""
    vocab = np.array(["deu", "eng", "fra", "spa"], dtype="<U3")
    ids = np.array(["eng", "fra", "deu", "spa", "zzz"], dtype="<U3")
    np.testing.assert_array_equal(
        compute_mod._vocab_lookup(vocab, ids), [1, 2, 0, 3, -1])
    np.testing.assert_array_equal(
        compute_mod._vocab_lookup(vocab, ids), _reference_lookup(vocab, ids))
    assert not compute_mod._lut_applies(vocab, ids)


def test_lookup_accepts_float_vocab():
    vocab = np.array([1.5, 2.5, 9.0])
    ids = np.array([2.5, 1.5, 3.0, 9.0])
    np.testing.assert_array_equal(
        compute_mod._vocab_lookup(vocab, ids), _reference_lookup(vocab, ids))
    assert not compute_mod._lut_applies(vocab, ids)


def test_negative_ids_are_absent_not_wrapped():
    """A negative index would read the table FROM THE END — in range, and
    silently some other id's position."""
    vocab = np.array([0, 5, 6], dtype=np.int64)
    ids = np.array([-3, -1, 0, 5, 6, 9], dtype=np.int64)
    got = compute_mod._vocab_lookup(vocab, ids)
    np.testing.assert_array_equal(got, [-1, -1, 0, 1, 2, -1])
    np.testing.assert_array_equal(got, _reference_lookup(vocab, ids))
    assert not compute_mod._lut_applies(vocab, ids)


def test_negative_vocab_entries_are_handled():
    """Building the table would write `lut[-5]`, corrupting the far end."""
    vocab = np.array([-5, -1, 2], dtype=np.int64)
    ids = np.array([-5, -1, 0, 2, 8], dtype=np.int64)
    got = compute_mod._vocab_lookup(vocab, ids)
    np.testing.assert_array_equal(got, [0, 1, -1, 2, -1])
    np.testing.assert_array_equal(got, _reference_lookup(vocab, ids))
    assert not compute_mod._lut_applies(vocab, ids)


def test_lut_applies_for_the_sparse_remap_shape():
    """The case that has to stay fast: int64 token ids (io.py casts them to
    int64), non-negative, from a subword-sized vocabulary."""
    vocab = np.array([3, 17, 250_000], dtype=np.int64)
    ids = np.array([3, 4, 250_000], dtype=np.int64)
    assert compute_mod._lut_applies(vocab, ids)
    np.testing.assert_array_equal(
        compute_mod._vocab_lookup(vocab, ids), _reference_lookup(vocab, ids))


class _NoMin(np.ndarray):
    """An array that fails if anything reduces it — `np.ndarray.min` is not
    monkeypatchable (immutable type), so the subclass is the way to assert a
    scan did NOT happen."""

    def min(self, *args, **kwargs):  # noqa: A003 - mirroring ndarray's name
        pytest.fail("scanned unsigned ids for negatives")


def test_unsigned_ids_skip_the_negativity_scan():
    """Unsigned ids cannot be negative, so the O(n) check is skipped rather
    than paid over every file's ~1.4e8 nnz."""
    ids = np.array([1, 9], dtype=np.uint32).view(_NoMin)
    assert compute_mod._lut_applies(np.array([1, 4], dtype=np.uint32), ids)


def test_lut_rejected_when_vocab_top_is_huge():
    assert not compute_mod._lut_applies(
        np.array([1, 4_000_000_000], dtype=np.uint32),
        np.array([1], dtype=np.uint32))


def test_lookup_empty_ids_does_not_scan_for_negatives():
    """`ids.min()` on an empty array raises; the gate must short-circuit."""
    assert compute_mod._lut_applies(
        np.array([1, 2], dtype=np.int64), np.zeros(0, dtype=np.int64))
