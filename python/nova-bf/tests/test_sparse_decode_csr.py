"""R2: the per-file sparse decode stays in CSR instead of expanding to COO.

`_remap_sparse_file` and `_sparse_file_norms` used to build an `np.repeat`'d
row-id array — one int64 per nonzero, gigabytes on a production corpus file — so
that a COO helper could decide something the CSR structure already knows.
These tests pin the rewrite against the OLD implementations, reproduced here
verbatim as oracles, over shapes that exercise every branch: empty rows,
trailing empty rows, out-of-vocab drops, no drops at all, duplicate column
ids (the case the COO coalesce exists for) and unsorted rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova_bf import compute as C


# --- the pre-R2 implementations, as oracles --------------------------------


def _old_remap(row_offsets, indices, values, vocab, lut=None):
    n_rows = len(row_offsets) - 1
    row_ids = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(row_offsets))
    idx = C._vocab_lookup(vocab, indices, lut).astype(np.int64)
    keep = idx >= 0
    row_ids, idx, val = row_ids[keep], idx[keep], values[keep]
    row_ids, idx, val = C._coalesce_by_row_col(row_ids, idx, val)
    counts = np.bincount(row_ids, minlength=n_rows)
    return np.concatenate(([0], np.cumsum(counts))).astype(np.int64), idx, val


def _old_norms(row_offsets, indices, values):
    n_rows = len(row_offsets) - 1
    row_ids = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(row_offsets))
    m_rows, _, m_vals = C._coalesce_by_row_col(row_ids, indices.astype(np.int64), values)
    sumsq = np.bincount(m_rows, weights=m_vals.astype(np.float64) ** 2, minlength=n_rows)
    return np.sqrt(sumsq).astype(np.float32)


# --- generators -------------------------------------------------------------


def _csr(rng, n_rows, max_len, vocab_hi, sorted_unique=True, dtype=np.uint32):
    lens = rng.integers(0, max_len + 1, n_rows)
    off = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
    nnz = int(off[-1])
    idx = np.empty(nnz, dtype=dtype)
    for r in range(n_rows):
        lo, hi = int(off[r]), int(off[r + 1])
        n = hi - lo
        if n == 0:
            continue
        if sorted_unique:
            pool = rng.choice(vocab_hi, size=min(n, vocab_hi), replace=False)
            if len(pool) < n:  # not enough distinct ids: shrink the row
                pool = np.concatenate([pool, pool[: n - len(pool)]])
            idx[lo:hi] = np.sort(pool)[:n] if len(np.unique(pool)) == n else pool
        else:
            idx[lo:hi] = rng.integers(0, vocab_hi, n)
    if sorted_unique:
        # guarantee the invariant even where the shrink above could not
        for r in range(n_rows):
            lo, hi = int(off[r]), int(off[r + 1])
            if hi - lo:
                u = np.unique(idx[lo:hi])
                if len(u) < hi - lo:
                    return _csr(rng, n_rows, max_len, vocab_hi, True, dtype)
    vals = (rng.standard_normal(nnz) * 3.0).astype(np.float32)
    return off, idx, vals


# --- tests ------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_remap_matches_the_coo_implementation(seed):
    rng = np.random.default_rng(seed)
    n_rows = int(rng.integers(1, 60))
    vocab_hi = int(rng.integers(4, 200))
    off, idx, vals = _csr(rng, n_rows, 8, vocab_hi)
    # a vocabulary that covers only part of the id space, so entries drop
    vocab = np.unique(rng.choice(vocab_hi, size=max(1, vocab_hi // 2), replace=False)).astype(np.uint32)
    got = C._remap_sparse_file(off, idx, vals, vocab)
    want = _old_remap(off, idx, vals, vocab)
    for g, w, name in zip(got, want, ("offsets", "indices", "values")):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w), err_msg=name)


@pytest.mark.parametrize("seed", range(20))
def test_remap_matches_when_rows_are_unsorted_or_duplicated(seed):
    """The COO path, which is what the coalesce exists for."""
    rng = np.random.default_rng(1000 + seed)
    n_rows = int(rng.integers(1, 40))
    off, idx, vals = _csr(rng, n_rows, 10, 12, sorted_unique=False)
    vocab = np.arange(12, dtype=np.uint32)
    got = C._remap_sparse_file(off, idx, vals, vocab)
    want = _old_remap(off, idx, vals, vocab)
    for g, w, name in zip(got, want, ("offsets", "indices", "values")):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w), err_msg=name)


def test_remap_with_nothing_dropped_keeps_the_structure():
    off, idx, vals = _csr(np.random.default_rng(7), 20, 6, 30)
    vocab = np.arange(30, dtype=np.uint32)
    o, i, v = C._remap_sparse_file(off, idx, vals, vocab)
    np.testing.assert_array_equal(o, off)
    np.testing.assert_array_equal(i, idx.astype(np.int32))
    np.testing.assert_array_equal(v, vals)


def test_remap_with_everything_dropped():
    off, idx, vals = _csr(np.random.default_rng(8), 15, 5, 30)
    vocab = np.array([9999], dtype=np.uint32)
    o, i, v = C._remap_sparse_file(off, idx, vals, vocab)
    assert len(i) == 0 and len(v) == 0
    np.testing.assert_array_equal(o, np.zeros(len(off), dtype=np.int64))


@pytest.mark.parametrize(
    "off,idx",
    [
        (np.array([0, 0, 0], np.int64), np.zeros(0, np.uint32)),          # all empty
        (np.array([0, 2, 2], np.int64), np.array([1, 3], np.uint32)),      # TRAILING empty row
        (np.array([0, 0, 2], np.int64), np.array([1, 3], np.uint32)),      # LEADING empty row
        (np.array([0, 1], np.int64), np.array([5], np.uint32)),            # single entry
    ],
)
def test_remap_degenerate_shapes(off, idx):
    vals = np.arange(len(idx), dtype=np.float32) + 1.0
    vocab = np.arange(10, dtype=np.uint32)
    got = C._remap_sparse_file(off, idx, vals, vocab)
    want = _old_remap(off, idx, vals, vocab)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


@pytest.mark.parametrize("seed", range(40))
def test_norms_are_bit_identical_to_the_coo_implementation(seed):
    """Not "close": the norms divide cosine scores, so a changed last bit is a
    changed ground truth."""
    rng = np.random.default_rng(500 + seed)
    n_rows = int(rng.integers(1, 80))
    sorted_unique = bool(seed % 2)
    off, idx, vals = _csr(rng, n_rows, 300, 400, sorted_unique=sorted_unique)
    got = C._sparse_file_norms(off, idx, vals)
    want = _old_norms(off, idx, vals)
    assert got.dtype == want.dtype == np.float32
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


def test_norms_chunking_does_not_move_a_bit():
    """The chunked accumulation must be bit-identical to the unchunked one for
    a row that spans many chunks' worth of nonzeros."""
    rng = np.random.default_rng(11)
    off = np.array([0, 3_000_000], dtype=np.int64)
    idx = np.arange(3_000_000, dtype=np.uint32)
    vals = (rng.standard_normal(3_000_000) * 1e-3).astype(np.float32)
    big, small = C._CSR_CHUNK_NNZ, 4096
    try:
        C._CSR_CHUNK_NNZ = small
        a = C._sparse_file_norms(off, idx, vals)
    finally:
        C._CSR_CHUNK_NNZ = big
    b = C._sparse_file_norms(off, idx, vals)
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
    np.testing.assert_array_equal(a.view(np.uint32), _old_norms(off, idx, vals).view(np.uint32))


@pytest.mark.parametrize("seed", range(60))
def test_csr_rows_sorted_unique_matches_the_coo_predicate(seed):
    rng = np.random.default_rng(2000 + seed)
    n_rows = int(rng.integers(1, 30))
    off, idx, _ = _csr(rng, n_rows, 6, 8, sorted_unique=bool(seed % 3))
    row_ids = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(off))
    if len(row_ids) == 0:
        want = True
    else:
        col = idx.astype(np.int64)
        want = bool(np.all(
            (row_ids[1:] > row_ids[:-1])
            | ((row_ids[1:] == row_ids[:-1]) & (col[1:] > col[:-1]))
        )) if len(row_ids) > 1 else True
    assert C._csr_rows_sorted_unique(off, idx) == want


@pytest.mark.parametrize("seed", range(30))
def test_csr_row_counts_matches_bincount(seed):
    rng = np.random.default_rng(3000 + seed)
    n_rows = int(rng.integers(1, 50))
    lens = rng.integers(0, 20, n_rows)
    off = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
    keep = rng.random(int(off[-1])) < 0.5
    row_ids = np.repeat(np.arange(n_rows, dtype=np.int64), lens)
    want = np.bincount(row_ids[keep], minlength=n_rows)
    np.testing.assert_array_equal(C._csr_row_counts(off, keep), want)


def test_sparse_to_coo_parts_does_not_widen_or_copy():
    """R2's premise: the stored widths are enough, and the copies were free
    to delete."""
    import pyarrow as pa
    from nova_bf.io import sparse_to_coo_parts

    col = pa.chunked_array([pa.array(
        [{"indices": [1, 5], "values": [1.0, 2.0]}, {"indices": [], "values": []}],
        type=pa.struct([("indices", pa.list_(pa.uint32())),
                        ("values", pa.list_(pa.float32()))]),
    )])
    off, idx, val = sparse_to_coo_parts(col)
    assert idx.dtype == np.uint32, "stored width must survive"
    assert val.dtype == np.float32
    np.testing.assert_array_equal(off, [0, 2, 2])
    np.testing.assert_array_equal(idx, [1, 5])
    np.testing.assert_array_equal(val, [1.0, 2.0])
