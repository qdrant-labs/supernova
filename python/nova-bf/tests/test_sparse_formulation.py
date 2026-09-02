"""`_sparse_scores` picks WHICH OPERAND IS SPARSE. Both choices must agree.

Scoring `Cb @ Q.T` makes the per-slice corpus sparse and streams the densified
query matrix as the dense operand; the swapped form makes the static query
matrix sparse and densifies the corpus slice. They sum the same products in a
different order, so they are not bit-identical — but they must agree on the
one thing that decides which documents come back: the zero-gate set
(`raw == 0` marks a structurally non-overlapping pair as a non-candidate).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from nova_bf.compute import (_SparseQueryCache, _dense_slice_t, _sparse_scores,
                             _sparse_batch_to_csr)
import nova_bf.compute as compute_mod


def _slice(rng, n_rows, vocab, nnz, dtype=np.float32, signed=True):
    """A coalesced CSR slice, built the way `_remap_sparse_file` leaves them:
    per-row sorted, deduped column ids."""
    rows, cols, vals = [], [], []
    for r in range(n_rows):
        c = np.sort(rng.choice(vocab, size=nnz, replace=False))
        rows += [r] * nnz
        cols += c.tolist()
        v = rng.standard_normal(nnz) if signed else rng.random(nnz) + 0.1
        vals += v.astype(dtype).tolist()
    row_offsets = np.arange(0, n_rows * nnz + 1, nnz, dtype=np.int64)
    return (row_offsets, np.asarray(cols, dtype=np.int64),
            np.asarray(vals, dtype=dtype))


def _Q(rng, n_q, vocab, nnz):
    Q = np.zeros((n_q, vocab), dtype=np.float32)
    for q in range(n_q):
        Q[q, rng.choice(vocab, size=nnz, replace=False)] = rng.standard_normal(nnz)
    return torch.from_numpy(Q)


@pytest.fixture
def case():
    rng = np.random.default_rng(7)
    n_rows, vocab, n_q = 64, 128, 40
    ro, idx, val = _slice(rng, n_rows, vocab, nnz=9)
    vocab_arr = np.arange(vocab)
    Cb = _sparse_batch_to_csr(ro, idx, val, 0, n_rows, vocab_arr, "cpu")
    return _Q(rng, n_q, vocab, nnz=5), Cb


def test_both_formulations_agree(case, monkeypatch):
    """The property that matters: same scores to float tolerance, and the same
    zero-gate SET exactly. A disagreement in the gate would return different
    documents, which for ground truth is disqualifying regardless of speed."""
    Q, Cb = case
    swapped = _sparse_scores(Q, Cb, _SparseQueryCache())
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", 0)
    fallback = _sparse_scores(Q, Cb, _SparseQueryCache())

    assert swapped.shape == fallback.shape == (Q.shape[0], Cb.shape[0])
    assert torch.equal(swapped == 0, fallback == 0), "zero-gate sets differ"
    assert torch.allclose(swapped, fallback, rtol=1e-5, atol=1e-6)


def test_swapped_output_is_contiguous(case):
    """The swap returns `(n_q, n_rows)` natively; the old path returned a
    transposed view. Downstream (mask, top-K, merge) reads it row-major, so
    this is a small free win — and a regression here would be silent."""
    Q, Cb = case
    assert _sparse_scores(Q, Cb, _SparseQueryCache()).is_contiguous()


def test_budget_selects_the_fallback(case, monkeypatch):
    """A wide-vocab query set would want a slice-sized dense operand; the
    budget must send it to the transpose form instead of allocating it."""
    Q, Cb = case
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", 0)
    out = _sparse_scores(Q, Cb, _SparseQueryCache())
    assert out.shape == (Q.shape[0], Cb.shape[0])


def test_fallback_retains_no_dense_query_copy(case, monkeypatch):
    """The regression this pins: the fallback used to cache a contiguous `Q.T`
    on the RUN-WIDE cache. That is a second full copy of the largest resident
    tensor (`n_q x vocab` float32), allocated on the branch taken precisely
    when the slice is already too big for the dense operand, and never freed —
    7.2 -> 14.4 GiB at the fineweb shape on a 24 GB card.

    The existing fallback tests could not catch it: each passes a FRESH cache,
    so nothing crosses slices. This one reuses one cache across several slices
    and asserts only sparse representations survive.
    """
    Q, Cb = case
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", 0)
    cache = _SparseQueryCache()
    for _ in range(3):
        _sparse_scores(Q, Cb, cache)

    dense = [name for name, v in vars(cache).items()
             if isinstance(v, torch.Tensor) and v.layout is torch.strided]
    assert not dense, f"run-wide query cache retained dense tensor(s): {dense}"


def test_fallback_scores_are_unchanged_across_slices(case, monkeypatch):
    """Dropping the cache must change allocation lifetime and nothing else:
    a shared cache and a fresh one have to produce identical scores."""
    Q, Cb = case
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", 0)
    shared = _SparseQueryCache()
    first = _sparse_scores(Q, Cb, shared).clone()
    again = _sparse_scores(Q, Cb, shared)
    fresh = _sparse_scores(Q, Cb, _SparseQueryCache())
    assert torch.equal(first, again), "reusing a cache changed the scores"
    assert torch.equal(first, fresh), "a fresh cache changed the scores"


def _reset_branches():
    for k in compute_mod._SPARSE_BRANCHES:
        compute_mod._SPARSE_BRANCHES[k] = 0


def test_batch_size_keeps_a_tail_slice_on_the_same_branch_as_full_slices(monkeypatch):
    """A file's last slice (`n_rows % sparse_batch_size` rows) is narrower
    than every other slice in the batch. Without a run-wide target, it can
    independently decide the dense operand fits when the full-size slices'
    own row count already ruled that out — scoring one file with two
    different accumulation orders. Pinning `_SparseQueryCache.batch_size`
    must make every slice of a run agree."""
    rng = np.random.default_rng(11)
    vocab, nnz = 128, 9
    full_n_rows, tail_n_rows, batch_size = 64, 20, 64

    def make_Cb(n_rows):
        ro, idx, val = _slice(rng, n_rows, vocab, nnz=nnz)
        return _sparse_batch_to_csr(ro, idx, val, 0, n_rows, np.arange(vocab), "cpu")

    Q = _Q(rng, 10, vocab, nnz=5)
    full_Cb, tail_Cb = make_Cb(full_n_rows), make_Cb(tail_n_rows)

    # Budget covers the tail slice's own row count but not the full batch size.
    budget = vocab * 4 * (tail_n_rows + batch_size) // 2
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", budget)
    per = compute_mod._dense_corpus_rows_per_chunk(full_Cb, 4)
    assert tail_n_rows <= per < batch_size, "fixture must sit between the two row counts"

    cache = _SparseQueryCache(batch_size=batch_size)
    _reset_branches()
    _sparse_scores(Q, full_Cb, cache)
    _sparse_scores(Q, tail_Cb, cache)
    assert compute_mod._SPARSE_BRANCHES["scored_fallback"] == 2
    assert compute_mod._SPARSE_BRANCHES["scored_swapped"] == 0

    # Sanity: this is the bug being fixed. With no configured batch size (the
    # only mode a bare cache — every other test in this file, and any direct
    # caller — supports), the tail slice decides for itself and switches
    # branch on its own.
    _reset_branches()
    bare = _SparseQueryCache()
    _sparse_scores(Q, full_Cb, bare)
    _sparse_scores(Q, tail_Cb, bare)
    assert compute_mod._SPARSE_BRANCHES["scored_fallback"] == 1
    assert compute_mod._SPARSE_BRANCHES["scored_swapped"] == 1


def test_batch_size_keeps_the_structural_gate_on_the_same_branch_as_full_slices(monkeypatch):
    """Same fix, for the signed-data overlap gate's whole-vs-chunked switch.

    The whole-slice fast path calls `_dense_slice_t` with `self.Cb` itself;
    the chunked path always rebuilds a fresh CSR sub-tensor, even when only
    one chunk is needed. Recording object identity distinguishes them, so
    this pins that the tail slice takes the SAME (chunked) branch as the
    full-size slice instead of independently deciding it can afford the
    whole-slice path."""
    from nova_bf.compute import SparseBatchSlice

    full_Q, full_Cb = _signed_case(n_q=6, n_rows=64, vocab=24, seed=9)
    _, tail_Cb = _signed_case(n_q=6, n_rows=20, vocab=24, seed=10)
    batch_size = 64

    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", 24 * 4 * 42)
    per = compute_mod._dense_corpus_rows_per_chunk(full_Cb, full_Q.element_size())
    assert 20 <= per < batch_size, "fixture must sit between the two row counts"

    real_dense_slice_t = compute_mod._dense_slice_t
    took_whole_path = []

    def spy(Cb, values):
        took_whole_path.append(Cb is full_Cb or Cb is tail_Cb)
        return real_dense_slice_t(Cb, values)

    monkeypatch.setattr(compute_mod, "_dense_slice_t", spy)

    cache = _SparseQueryCache(batch_size=batch_size)
    full_slice = SparseBatchSlice(Cb=full_Cb, row_norms=None, zero_gate_ok=False,
                                  q_cache=cache)
    tail_slice = SparseBatchSlice(Cb=tail_Cb, row_norms=None, zero_gate_ok=False,
                                  q_cache=cache)
    full_slice._structural_no_overlap(full_Q)
    full_took_whole_path = any(took_whole_path)
    took_whole_path.clear()
    tail_slice._structural_no_overlap(full_Q)
    tail_took_whole_path = any(took_whole_path)

    assert not full_took_whole_path, "full slice unexpectedly took the whole-slice fast path"
    assert tail_took_whole_path == full_took_whole_path, \
        "tail slice took a different branch than the full-size slice"


def _signed_case(n_q=6, n_rows=40, vocab=24, seed=3):
    """SIGNED sparse data, so `zero_gate_ok` is False and the structural
    overlap gate runs instead of the cheap `raw == 0` one."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for r in range(n_rows):
        idx = sorted(rng.choice(vocab, 4, replace=False))
        for c in idx:
            rows.append(r); cols.append(int(c))
            vals.append(float(rng.uniform(-2, 2)))
    crow = np.zeros(n_rows + 1, dtype=np.int64)
    np.add.at(crow, np.asarray(rows) + 1, 1)
    crow = np.cumsum(crow)
    Cb = torch.sparse_csr_tensor(
        torch.tensor(crow), torch.tensor(cols, dtype=torch.int64),
        torch.tensor(vals, dtype=torch.float32),
        size=(n_rows, vocab), check_invariants=False)
    Q = torch.zeros(n_q, vocab, dtype=torch.float32)
    for q in range(n_q):
        idx = rng.choice(vocab, 5, replace=False)
        Q[q, torch.tensor(idx.copy())] = torch.tensor(
            rng.uniform(-2, 2, 5), dtype=torch.float32)
    return Q, Cb


def test_structural_gate_is_chunked_but_identical(monkeypatch):
    """The overlap gate densifies the same `(vocab, n_rows)` operand that
    scoring gates behind `_SPARSE_SWAP_MAX_DENSE_BYTES` — it used to do so with
    NO budget at all, so a slice that scoring refused to densify was densified
    here anyway. It now chunks; chunking must not change the answer.
    """
    from nova_bf.compute import SparseBatchSlice

    Q, Cb = _signed_case()
    whole = SparseBatchSlice(Cb=Cb, row_norms=None,
                             zero_gate_ok=False)._structural_no_overlap(Q)

    # Force several chunks: budget of one row's worth of dense indicator.
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES",
                        Cb.shape[1] * 4)
    per = compute_mod._dense_corpus_rows_per_chunk(Cb, 4)
    assert per < Cb.shape[0], "budget did not actually force chunking"

    chunked = SparseBatchSlice(Cb=Cb, row_norms=None,
                               zero_gate_ok=False)._structural_no_overlap(Q)
    assert chunked.shape == whole.shape == (Q.shape[0], Cb.shape[0])
    assert torch.equal(chunked, whole), "chunking changed the overlap gate"
    assert whole.any() and not whole.all(), "degenerate fixture: gate is constant"


def test_structural_gate_and_scoring_share_one_budget(monkeypatch):
    """The bug was that the two disagreed. Pin that one helper decides both:
    whenever scoring declines to densify, the gate chunks rather than
    allocating the operand scoring just refused."""
    Q, Cb = _signed_case()
    n_rows = Cb.shape[0]

    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES",
                        Cb.shape[1] * 4)
    per = compute_mod._dense_corpus_rows_per_chunk(Cb, Q.element_size())
    scoring_densifies = per >= n_rows
    assert not scoring_densifies, "expected scoring to take the fallback here"

    # Same helper, same answer -> the gate cannot densify the whole slice.
    assert compute_mod._dense_corpus_rows_per_chunk(Cb, Q.element_size()) == per
    assert per >= 1, "budget must always allow at least one row"


def _csr_parts(t):
    return (t.crow_indices().clone(), t.col_indices().clone(),
            t.values().clone())


@pytest.mark.parametrize("n_q,vocab", [(37, 11), (64, 8), (5, 64), (1, 9)])
@pytest.mark.parametrize("kind", ["values", "indicator"])
def test_query_csr_is_identical_however_it_is_blocked(monkeypatch, n_q, vocab,
                                                      kind):
    """`_csr` builds the pattern in row blocks so the transient `(rows, vocab)`
    bool is bounded by block height rather than by `n_q` (1.8 GiB at 100k x
    18k, live next to the 7.2 GiB `Q` it comes from). Blocking must be a pure
    memory optimisation: the crow prefix sum composes across blocks, so every
    block height has to give the same CSR.
    """
    g = torch.Generator().manual_seed(n_q * 100 + vocab)
    Q = torch.randn(n_q, vocab, generator=g)
    Q[Q.abs() < 0.8] = 0.0                      # genuinely sparse
    Q[0] = 0.0                                  # a wholly empty row
    if n_q > 2:
        Q[2] = torch.randn(vocab, generator=g).abs() + 1.0   # a wholly full row

    def build():
        c = compute_mod._SparseQueryCache()
        return _csr_parts(getattr(c, kind)(Q))

    monkeypatch.setattr(compute_mod, "_CSR_BLOCK_BYTES", 1 << 30)   # one block
    whole = build()
    for block_rows in (1, 2, 3, 7, n_q - 1 if n_q > 1 else 1):
        monkeypatch.setattr(compute_mod, "_CSR_BLOCK_BYTES",
                            max(1, block_rows * vocab))
        got = build()
        for a, b, name in zip(got, whole, ("crow", "col", "values")):
            assert torch.equal(a, b), f"{name} differs at block_rows={block_rows}"


def test_query_csr_matches_a_dense_roundtrip():
    """Independent check that the blocked CSR really encodes Q: densifying it
    must reproduce the original matrix exactly."""
    g = torch.Generator().manual_seed(5)
    Q = torch.randn(53, 17, generator=g)
    Q[Q.abs() < 1.0] = 0.0
    csr = compute_mod._SparseQueryCache().values(Q)
    assert torch.equal(csr.to_dense(), Q)
    assert int(csr.values().numel()) == int((Q != 0).sum())


def test_query_csr_handles_an_all_zero_matrix():
    """No nonzeros at all: crow must be all zeros and the arrays empty, not a
    crash on `torch.cat([])`."""
    Q = torch.zeros(9, 6)
    csr = compute_mod._SparseQueryCache().values(Q)
    assert int(csr.values().numel()) == 0
    assert torch.equal(csr.crow_indices(), torch.zeros(10, dtype=torch.int64))
    assert torch.equal(csr.to_dense(), Q)


def test_dense_slice_t_marks_stored_zeros(case):
    """`_dense_slice_t`'s `values` argument exists so the structural gate can
    scatter ONES. A stored 0.0 is still an overlap; deriving the indicator
    from `Cb.values() != 0` would miss it and wrongly call the pair a
    non-candidate."""
    _, Cb = case
    vals = Cb.values().clone()
    vals[0] = 0.0                                    # a stored, explicit zero
    Cb0 = torch.sparse_csr_tensor(Cb.crow_indices(), Cb.col_indices(), vals,
                                  size=Cb.shape, check_invariants=False)
    ones = torch.ones(vals.numel(), dtype=vals.dtype)
    ind = _dense_slice_t(Cb0, ones)
    col0, row0 = int(Cb.col_indices()[0]), 0
    assert ind[col0, row0] == 1.0, "stored zero lost from the structural indicator"
    assert _dense_slice_t(Cb0, vals)[col0, row0] == 0.0


def test_query_cache_builds_once(case):
    """Run-wide by design: the CSR is built on first use and reused by every
    slice thereafter (see `_SparseQueryCache`'s shared-Q invariant)."""
    Q, _ = case
    c = _SparseQueryCache()
    assert c.values(Q) is c.values(Q)
    assert c.indicator(Q) is c.indicator(Q)
    # Only SPARSE representations may be cached here. A dense `(vocab, n_q)`
    # transpose used to be, and doubled peak GPU memory on the branch that is
    # taken precisely when memory is tightest.
    assert not hasattr(c, "transpose"), \
        "a dense representation was re-added to the run-wide query cache"
    assert c.values(Q).values().numel() == int((Q != 0).sum())
    # indicator carries ones; values carries Q's actual numbers
    assert torch.equal(c.indicator(Q).values(),
                       torch.ones_like(c.indicator(Q).values()))
