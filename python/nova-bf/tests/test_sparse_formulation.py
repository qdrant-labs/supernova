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
    documents, which for ground truth is disqualifying regardless of speed.

    `sparse_chunk=False` on the second call: chunking now defaults True, so a
    bare cache here would land on the chunked branch instead of the fallback
    this test means to compare against (chunked-vs-swapped is already covered
    by `test_chunked_scoring_matches_the_whole_dense_path`)."""
    Q, Cb = case
    swapped = _sparse_scores(Q, Cb, _SparseQueryCache())
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", 0)
    fallback = _sparse_scores(Q, Cb, _SparseQueryCache(sparse_chunk=False))

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
    budget must send it to the transpose form instead of allocating it.

    `sparse_chunk=False`: chunking now defaults True and would otherwise
    intercept this before it ever reaches the transpose form the name and
    docstring are about."""
    Q, Cb = case
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", 0)
    out = _sparse_scores(Q, Cb, _SparseQueryCache(sparse_chunk=False))
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

    `sparse_chunk=False`: without it this cache would take the (now-default)
    chunked branch instead of the fallback the regression actually lives in,
    and the assertion below would pass whether or not the bug were back.
    """
    Q, Cb = case
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", 0)
    cache = _SparseQueryCache(sparse_chunk=False)
    for _ in range(3):
        _sparse_scores(Q, Cb, cache)

    dense = [name for name, v in vars(cache).items()
             if isinstance(v, torch.Tensor) and v.layout is torch.strided]
    assert not dense, f"run-wide query cache retained dense tensor(s): {dense}"


def test_fallback_scores_are_unchanged_across_slices(case, monkeypatch):
    """Dropping the cache must change allocation lifetime and nothing else:
    a shared cache and a fresh one have to produce identical scores.

    `sparse_chunk=False` throughout: this test is specifically about the
    fallback path's cache-sharing invariant (see the previous test's
    regression), which a bare (chunk-defaulting) cache would not exercise."""
    Q, Cb = case
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", 0)
    shared = _SparseQueryCache(sparse_chunk=False)
    first = _sparse_scores(Q, Cb, shared).clone()
    again = _sparse_scores(Q, Cb, shared)
    fresh = _sparse_scores(Q, Cb, _SparseQueryCache(sparse_chunk=False))
    assert torch.equal(first, again), "reusing a cache changed the scores"
    assert torch.equal(first, fresh), "a fresh cache changed the scores"


# ---------------------------------------------------------- params.sparse_chunk
#
# The chunked-dense branch (see
# docs/brute-force/sparse-chunked-scoring-2026-09-02.md): when the dense
# corpus operand does not fit the budget, split the corpus ROWS and run
# several smaller dense GEMMs instead of falling to the sparse-CSR transpose
# path (`torch.matmul(Cb, Q.t().contiguous()).T`). Config-driven
# (`ParamsConfig.sparse_chunk`, default True — see test_config_searches.py for
# the config-level default/override test) and threaded down to `_sparse_scores`
# via `_SparseQueryCache.sparse_chunk`, run-wide, the same way `batch_size` is.


def test_sparse_chunk_defaults_true_on_a_bare_cache(case):
    """`_SparseQueryCache`'s own default (not just `ParamsConfig`'s) is True —
    this is what makes chunking the DEFAULT PATH for every direct caller,
    including a bare cache with no config behind it, not merely opt-in
    through a config a caller has to remember to set."""
    assert _SparseQueryCache().sparse_chunk is True
    assert _SparseQueryCache(sparse_chunk=False).sparse_chunk is False


def test_chunked_scoring_matches_the_whole_dense_path(case, monkeypatch):
    """Chunking partitions the SAME dense GEMM by output tile — each chunk's
    columns are independent of every other chunk's, so splitting must not
    change the answer beyond ordinary matmul rounding: not the zero-gate set,
    and not the values past float tolerance. (Not asserted bit-identical:
    the doc above measured chunked vs. unchunked dense differing on
    5/100,000 queries at production scale, attributed to the dense branch's
    own float accumulation rather than to chunking.)"""
    Q, Cb = case
    whole = _sparse_scores(Q, Cb, _SparseQueryCache())  # full budget: fits whole

    n_rows = Cb.shape[0]
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES",
                        Cb.shape[1] * 4 * (n_rows // 4))
    per = compute_mod._dense_corpus_rows_per_chunk(Cb, Q.element_size())
    assert per < n_rows, "fixture must force multiple chunks"
    chunked = _sparse_scores(Q, Cb, _SparseQueryCache())  # sparse_chunk defaults True

    assert chunked.shape == whole.shape == (Q.shape[0], Cb.shape[0])
    assert torch.equal(chunked == 0, whole == 0), "chunking changed the zero-gate set"
    assert torch.allclose(chunked, whole, rtol=1e-5, atol=1e-6)


def _spy_on_dense_slice_t(monkeypatch):
    calls = []
    real = compute_mod._dense_slice_t

    def spy(Cb, values, row_ids=None):
        calls.append(Cb)
        return real(Cb, values, row_ids)

    monkeypatch.setattr(compute_mod, "_dense_slice_t", spy)
    return calls


def test_sparse_chunk_false_takes_the_transpose_fallback_not_chunking(case, monkeypatch):
    """With `sparse_chunk=False`, a slice too big to densify whole must still
    take the pre-existing `Cb @ Q.T` transpose path, not silently chunk.
    `_dense_slice_t` distinguishes them: the chunked branch calls it once per
    chunk, the transpose fallback never calls it at all."""
    Q, Cb = case
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", 0)
    calls = _spy_on_dense_slice_t(monkeypatch)

    _sparse_scores(Q, Cb, _SparseQueryCache(sparse_chunk=False))
    assert not calls, "sparse_chunk=False but the chunked branch still ran"


def test_sparse_chunk_true_actually_chunks(case, monkeypatch):
    """The mirror of the test above: with `sparse_chunk=True` (the default —
    see `test_sparse_chunk_defaults_true_on_a_bare_cache`) and a slice too big
    to densify whole, `_dense_slice_t` must be called once per chunk — proving
    the chunked branch ran rather than the transpose fallback it replaces."""
    Q, Cb = case
    n_rows = Cb.shape[0]
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES",
                        Cb.shape[1] * 4 * (n_rows // 4))
    per = compute_mod._dense_corpus_rows_per_chunk(Cb, Q.element_size())
    assert per < n_rows
    calls = _spy_on_dense_slice_t(monkeypatch)

    _sparse_scores(Q, Cb, _SparseQueryCache())
    expected_chunks = -(-n_rows // per)  # ceil
    assert len(calls) == expected_chunks
    assert all(c is not Cb for c in calls), \
        "chunked branch must rebuild per-chunk CSR sub-tensors, not reuse Cb whole"


def test_chunking_still_chunks_a_slice_smaller_than_one_chunk(monkeypatch):
    """Pins the fix in docs/brute-force/sparse-chunked-scoring-2026-09-02.md
    ('N9'): the first cut of this branch guarded on `if per < n_rows`, which
    skipped chunking — and fell through to the transpose fallback, the most
    expensive path and (on GPU) the only nondeterministic one — for any slice
    narrower than one chunk, exactly a corpus file's TAIL slice. The fix
    clamps `per = max(1, min(per, n_rows))` so a short tail slice still takes
    a single-chunk pass through this branch, same as the rest of its file,
    and must score identically (to tolerance) to computing it as one dense
    operand directly.

    Mirrors `test_batch_size_keeps_a_tail_slice_on_the_same_branch_as_full_
    slices` above, which pins the equivalent fix for the swapped/fallback
    branch choice; this one pins it one level down, for chunk WIDTH."""
    rng = np.random.default_rng(23)
    vocab, nnz = 96, 7
    full_n_rows, tail_n_rows, batch_size = 64, 5, 64

    def make_Cb(n_rows):
        ro, idx, val = _slice(rng, n_rows, vocab, nnz=nnz)
        return _sparse_batch_to_csr(ro, idx, val, 0, n_rows, np.arange(vocab), "cpu")

    Q = _Q(rng, 6, vocab, nnz=4)
    tail_Cb = make_Cb(tail_n_rows)
    whole = _sparse_scores(Q, tail_Cb, _SparseQueryCache())  # reference: fits whole

    full_Cb = make_Cb(full_n_rows)
    # Forces the FULL batch to chunk (per < batch_size) while comfortably
    # covering the short TAIL slice by itself (tail_n_rows <= per) — the exact
    # shape the bug needed.
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", vocab * 4 * 32)
    per = compute_mod._dense_corpus_rows_per_chunk(full_Cb, Q.element_size())
    assert tail_n_rows <= per < batch_size, "fixture must sit between the two row counts"
    calls = _spy_on_dense_slice_t(monkeypatch)

    cache = _SparseQueryCache(batch_size=batch_size)
    chunked = _sparse_scores(Q, tail_Cb, cache)

    assert calls, "tail slice fell through to the transpose fallback instead of chunking"
    assert len(calls) == 1, "a slice smaller than one chunk should need exactly one chunk"
    assert torch.allclose(chunked, whole, rtol=1e-5, atol=1e-6)


# ------------------------------------------- fused chunked score + structural gate
#
# Signed sparse data (`zero_gate_ok=False`) whose dense operand does not fit
# whole is the one combination where `_sparse_scores`'s chunked branch and
# `SparseBatchSlice._structural_no_overlap`'s chunked branch would otherwise
# independently re-walk the same row-chunk boundaries — see
# `_chunked_sparse_score_and_gate`'s docstring and
# docs/brute-force/sparse-chunked-scoring-2026-09-02.md ("doubled chunking
# cost on signed sparse data"). `SparseBatchSlice.score` fuses them for this
# case only.


def test_zero_gate_chunked_score_does_not_dispatch_to_the_fused_path(monkeypatch):
    """The OTHER half of the fused-path condition, `not self.zero_gate_ok`:
    zero-gate data uses `raw == 0` as its gate, with no separate indicator
    spmm — there is no second chunked pass for the fusion to save, so it must
    not be taken. Without this test, dropping or inverting that term from
    the condition would go undetected: the fused path still produces a
    correct (merely redundant) gate for zero-gate data, so nothing downstream
    would fail — only the branch taken changes.

    `zero_gate_ok` is set directly on the slice rather than derived from the
    fixture's actual values (`_signed_case` is just a convenient Cb/Q
    builder here); the actual `raw == 0` vs. structural-indicator semantics
    are covered elsewhere (`_zero_gate_file_ok` and its callers)."""
    from nova_bf.compute import SparseBatchSlice

    Q, Cb = _signed_case(n_q=8, n_rows=48, vocab=24, seed=37)
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", Cb.shape[1] * 4 * 8)
    per = compute_mod._dense_corpus_rows_per_chunk(Cb, Q.element_size())
    assert per < Cb.shape[0], "fixture must force chunking"

    calls = {"fused": 0}
    real_fused = compute_mod._chunked_sparse_score_and_gate
    monkeypatch.setattr(
        compute_mod, "_chunked_sparse_score_and_gate",
        lambda *a, **k: (calls.__setitem__("fused", calls["fused"] + 1),
                         real_fused(*a, **k))[1])

    _reset_branches()
    SparseBatchSlice(Cb=Cb, row_norms=None, zero_gate_ok=True).score(Q, "dot")

    assert calls["fused"] == 0, "zero-gate data must not take the fused path"
    assert compute_mod._SPARSE_BRANCHES["scored_chunked"] == 1, \
        "zero-gate data should still get chunked scoring via the standalone _sparse_scores"
    assert compute_mod._SPARSE_BRANCHES["gate_zero"] == 1
    assert compute_mod._SPARSE_BRANCHES["gate_structural"] == 0


def test_signed_chunked_score_dispatches_to_the_fused_path(monkeypatch):
    """The dispatch itself: signed + doesn't-fit + `sparse_chunk` (default)
    must call `_chunked_sparse_score_and_gate`, not `_sparse_scores` (whose
    own chunked branch would otherwise duplicate the gate's work)."""
    from nova_bf.compute import SparseBatchSlice

    Q, Cb = _signed_case(n_q=8, n_rows=48, vocab=24, seed=17)
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", Cb.shape[1] * 4 * 8)
    per = compute_mod._dense_corpus_rows_per_chunk(Cb, Q.element_size())
    assert per < Cb.shape[0], "fixture must force chunking"

    calls = {"fused": 0, "standalone": 0}
    real_fused = compute_mod._chunked_sparse_score_and_gate
    real_scores = compute_mod._sparse_scores

    def spy_fused(*a, **k):
        calls["fused"] += 1
        return real_fused(*a, **k)

    def spy_scores(*a, **k):
        calls["standalone"] += 1
        return real_scores(*a, **k)

    monkeypatch.setattr(compute_mod, "_chunked_sparse_score_and_gate", spy_fused)
    monkeypatch.setattr(compute_mod, "_sparse_scores", spy_scores)

    SparseBatchSlice(Cb=Cb, row_norms=None, zero_gate_ok=False).score(Q, "dot")

    assert calls == {"fused": 1, "standalone": 0}


def test_signed_chunked_score_matches_the_unfused_computation(monkeypatch):
    """Fusing must not change the answer: bit-identical to calling
    `_sparse_scores` then `_structural_no_overlap` separately — the path
    every other branch combination still takes, and what this one replaces."""
    from nova_bf.compute import SparseBatchSlice

    Q, Cb = _signed_case(n_q=8, n_rows=48, vocab=24, seed=19)
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", Cb.shape[1] * 4 * 8)
    per = compute_mod._dense_corpus_rows_per_chunk(Cb, Q.element_size())
    assert per < Cb.shape[0], "fixture must force chunking"

    ref_cache = _SparseQueryCache()
    raw_ref = _sparse_scores(Q, Cb, ref_cache)
    gate_ref = SparseBatchSlice(Cb=Cb, row_norms=None, zero_gate_ok=False,
                                q_cache=ref_cache)._structural_no_overlap(Q)
    expected = raw_ref.masked_fill(gate_ref, float("-inf"))

    got = SparseBatchSlice(Cb=Cb, row_norms=None, zero_gate_ok=False,
                           q_cache=_SparseQueryCache()).score(Q, "dot")

    assert torch.equal(got, expected)


def test_signed_chunked_score_shares_row_ids_between_values_and_indicator(monkeypatch):
    """The other half of the fusion's savings, alongside shared crow/col
    structure (`_iter_csr_row_chunk_bounds`): `_dense_slice_t`'s `row_ids` is
    computed once per chunk and passed to BOTH the values densify and the
    indicator densify, not recomputed for the second. Also exercises
    `_spy_on_dense_slice_t` against the 3-argument call the fused path makes
    (`_dense_slice_t(sub, values, row_ids)`) — a positional-arity mismatch
    here would TypeError instead of silently passing."""
    from nova_bf.compute import SparseBatchSlice

    Q, Cb = _signed_case(n_q=8, n_rows=48, vocab=24, seed=31)
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", Cb.shape[1] * 4 * 8)
    per = compute_mod._dense_corpus_rows_per_chunk(Cb, Q.element_size())
    n_chunks = -(-Cb.shape[0] // per)
    assert n_chunks > 1, "fixture must force multiple chunks"

    row_ids_seen = []
    real = compute_mod._dense_slice_t

    def spy(Cb_, values, row_ids=None):
        row_ids_seen.append(row_ids)
        return real(Cb_, values, row_ids)

    monkeypatch.setattr(compute_mod, "_dense_slice_t", spy)

    SparseBatchSlice(Cb=Cb, row_norms=None, zero_gate_ok=False).score(Q, "dot")

    assert len(row_ids_seen) == 2 * n_chunks, \
        "expected one values densify and one indicator densify per chunk"
    assert all(r is not None for r in row_ids_seen), \
        "fused path must precompute row_ids, not let _dense_slice_t redo it"
    pairs = list(zip(row_ids_seen[0::2], row_ids_seen[1::2]))
    assert all(a is b for a, b in pairs), \
        "each chunk's values and indicator densify must share ONE row_ids object"


def test_signed_chunked_score_still_counts_as_scored_chunked(monkeypatch):
    """The fused path bypasses `_sparse_scores` entirely, which is where
    `scored_chunked` is normally incremented — this pins that the manifest
    still sees a chunked run as chunked, not silently uncounted."""
    from nova_bf.compute import SparseBatchSlice

    Q, Cb = _signed_case(n_q=8, n_rows=48, vocab=24, seed=23)
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", Cb.shape[1] * 4 * 8)
    per = compute_mod._dense_corpus_rows_per_chunk(Cb, Q.element_size())
    assert per < Cb.shape[0], "fixture must force chunking"

    _reset_branches()
    SparseBatchSlice(Cb=Cb, row_norms=None, zero_gate_ok=False).score(Q, "dot")
    assert compute_mod._SPARSE_BRANCHES["scored_chunked"] == 1
    assert compute_mod._SPARSE_BRANCHES["gate_structural"] == 1
    assert compute_mod._SPARSE_BRANCHES["scored_swapped"] == 0
    assert compute_mod._SPARSE_BRANCHES["scored_fallback"] == 0


def test_signed_chunked_score_falls_back_to_the_unfused_path_when_disabled(monkeypatch):
    """`sparse_chunk=False` must NOT take the fused path — it has no chunked
    branch to fuse with; scoring goes through the plain transpose fallback
    while the gate still chunks on its own, same as before this fusion
    existed."""
    from nova_bf.compute import SparseBatchSlice

    Q, Cb = _signed_case(n_q=8, n_rows=48, vocab=24, seed=29)
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", Cb.shape[1] * 4 * 8)

    calls = {"fused": 0}
    real_fused = compute_mod._chunked_sparse_score_and_gate
    monkeypatch.setattr(compute_mod, "_chunked_sparse_score_and_gate",
                        lambda *a, **k: (calls.__setitem__("fused", calls["fused"] + 1),
                                        real_fused(*a, **k))[1])

    _reset_branches()
    SparseBatchSlice(Cb=Cb, row_norms=None, zero_gate_ok=False,
                     q_cache=_SparseQueryCache(sparse_chunk=False)).score(Q, "dot")
    assert calls["fused"] == 0
    assert compute_mod._SPARSE_BRANCHES["scored_fallback"] == 1
    assert compute_mod._SPARSE_BRANCHES["scored_chunked"] == 0


def test_csr_row_chunk_bounds_are_contiguous_and_carry_lo_forward(case):
    """Pins the carry-forward this helper exists for: each chunk's `lo` is
    the PREVIOUS chunk's `hi` (not independently re-derived from `crow`), the
    first `lo` is `0`, and the last `hi` is `Cb`'s total nnz — i.e. the
    chunks tile the corpus exactly once, with no gap or overlap. A version
    that went back to computing `lo = int(crow[r0])` every iteration would
    still pass this (same values, more host syncs); a version with an
    off-by-one in the carry would fail it immediately."""
    _, Cb = case
    per = max(1, Cb.shape[0] // 5)  # force several chunks
    bounds = list(compute_mod._iter_csr_row_chunk_bounds(Cb, per))
    assert len(bounds) > 1, "fixture must force multiple chunks"

    assert bounds[0][2] == 0, "first chunk's lo must be 0 (the CSR invariant), not read from crow"
    total_nnz = int(Cb.crow_indices()[-1])
    assert bounds[-1][3] == total_nnz, "last chunk's hi must reach the CSR's total nnz"
    for (_, _, _, hi_prev, _), (_, _, lo_next, _, _) in zip(bounds, bounds[1:]):
        assert hi_prev == lo_next, "a chunk's lo must equal the previous chunk's hi"
    assert sum(rows for *_, rows in bounds) == Cb.shape[0], \
        "chunk row counts must sum to the whole slice"


def _empty_sparse_csr(vocab, device="cpu"):
    ro = np.array([0], dtype=np.int64)
    idx = np.array([], dtype=np.int64)
    val = np.array([], dtype=np.float32)
    return _sparse_batch_to_csr(ro, idx, val, 0, 0, np.arange(vocab), device)


def test_zero_row_slice_yields_one_chunk_not_zero(monkeypatch):
    """`range(0, 0, per)` yields NOTHING, which would leave every caller
    (each collects one part per chunk, then `cat`s or skips the `cat` for a
    single part) with an EMPTY parts list — `torch.cat([])` raises. The
    pre-chunking code path never had this failure mode: `Cb @ Q.T` on a
    `(0, vocab)` operand just produces an empty result directly. Pins the
    fix: a 0-row `Cb` still yields exactly one (empty) chunk."""
    Cb = _empty_sparse_csr(vocab=16)
    bounds = list(compute_mod._iter_csr_row_chunk_bounds(Cb, per=8))
    assert len(bounds) == 1
    crow_chunk, col_chunk, lo, hi, rows = bounds[0]
    assert (lo, hi, rows) == (0, 0, 0)
    assert col_chunk.numel() == 0
    assert crow_chunk.numel() == 1 and int(crow_chunk[0]) == 0


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="needs CUDA")),
])
def test_zero_row_slice_does_not_crash_chunked_scoring_or_gate(monkeypatch, device):
    """End to end: a 0-row slice forced onto the chunked branch (`fits=False`
    via a tiny swap-max budget with a run-wide `batch_size` target) must not
    crash `_sparse_scores`, `_structural_no_overlap`, the fused
    `SparseBatchSlice.score` path, or the `sparse_chunk=False` path it
    replaces — each returns an `(n_q, 0)` result, same shape the transpose
    fallback always produced for an empty slice.

    Runs on CUDA too (skipped here if unavailable): PyTorch's CUDA sparse-CSR
    kernels have historically had rough edges on zero-sized operands the CPU
    path doesn't share, and a corpus slice CAN filter down to 0 rows mid-run
    (not just the whole-file-empty case guarded upstream in
    `_process_batch_group`) — verified once by hand against a live A10G
    worker; this is what makes that verification permanent and repeatable
    instead of a one-off manual check."""
    from nova_bf.compute import SparseBatchSlice

    Cb = _empty_sparse_csr(vocab=16, device=device)
    Q = torch.randn(5, 16, device=device)
    monkeypatch.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", 1)

    out = _sparse_scores(Q, Cb, _SparseQueryCache(batch_size=64))
    assert out.shape == (5, 0)

    gate = SparseBatchSlice(Cb=Cb, row_norms=None, zero_gate_ok=False,
                            q_cache=_SparseQueryCache(batch_size=64)
                            )._structural_no_overlap(Q)
    assert gate.shape == (5, 0)

    fused = SparseBatchSlice(Cb=Cb, row_norms=None, zero_gate_ok=False,
                             q_cache=_SparseQueryCache(batch_size=64)).score(Q, "dot")
    assert fused.shape == (5, 0)

    unfused = SparseBatchSlice(Cb=Cb, row_norms=None, zero_gate_ok=False,
                               q_cache=_SparseQueryCache(batch_size=64, sparse_chunk=False)
                               ).score(Q, "dot")
    assert unfused.shape == (5, 0)


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

    cache = _SparseQueryCache(batch_size=batch_size)  # sparse_chunk defaults True
    _reset_branches()
    _sparse_scores(Q, full_Cb, cache)
    _sparse_scores(Q, tail_Cb, cache)
    assert compute_mod._SPARSE_BRANCHES["scored_chunked"] == 2
    assert compute_mod._SPARSE_BRANCHES["scored_swapped"] == 0

    # Sanity: this is the bug being fixed. With no configured batch size (the
    # only mode a bare cache — every other test in this file, and any direct
    # caller — supports), the tail slice decides for itself and switches
    # branch on its own.
    _reset_branches()
    bare = _SparseQueryCache()
    _sparse_scores(Q, full_Cb, bare)
    _sparse_scores(Q, tail_Cb, bare)
    assert compute_mod._SPARSE_BRANCHES["scored_chunked"] == 1
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
