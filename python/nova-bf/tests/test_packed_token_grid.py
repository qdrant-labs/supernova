"""R7: the corpus token grid, and the whole fused combine, are bit-packed.

`_token_row_masks` used to return one `(n_rows,)` bool per query token — an
`(n_tokens, n_rows)` array, 3.8 GB per corpus file per reader thread on the
production filter — and `evaluate`'s combine moved a megabyte per token per
query combo before packing the finished row on the way out. Both are packed
now.

Three things can go wrong and none of them is loud, so each gets a test:
the padding bits of the last byte leaking as 1; two concurrent batches
sharing a byte of the grid; and the packed combine disagreeing with the
bool one.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from concurrent.futures import ThreadPoolExecutor

from nova_bf import filters as F
from nova_bf.config import Filter, FilterCondition
from nova_bf.filters import (
    PackedRowMask, TokenGrid, _packed_not, _packed_ones, _phrase_mask,
    _token_row_masks, _tail_mask, evaluate,
)


# --- padding bits -----------------------------------------------------------


@pytest.mark.parametrize("n_rows", list(range(0, 33)) + [1000, 1001, 4097])
def test_tail_mask_marks_exactly_the_valid_bits(n_rows):
    """`np.packbits` is big-endian: row `i` is bit `7 - (i % 8)` of byte
    `i // 8`. The mask must select exactly the rows that exist."""
    ones = np.ones(n_rows, dtype=bool)
    packed = np.packbits(ones)
    if n_rows == 0:
        # Zero rows pack to zero bytes: there is no tail byte to mask, and the
        # parametrization used to `return` here and report as passed while
        # asserting nothing. Assert the shape instead.
        assert packed.size == 0
        assert _tail_mask(0) == 0xFF, "a full-byte mask is the no-tail convention"
        return
    assert packed[-1] == _tail_mask(n_rows)


@pytest.mark.parametrize("n_rows", list(range(1, 33)) + [1000, 1001])
def test_packed_ones_and_not_keep_padding_at_zero(n_rows):
    ones = _packed_ones(n_rows)
    np.testing.assert_array_equal(
        np.unpackbits(ones, count=n_rows).astype(bool),
        np.ones(n_rows, dtype=bool))
    if n_rows & 7:
        assert ones[-1] & (0xFF & ~_tail_mask(n_rows)) == 0
    zeros = np.zeros((n_rows + 7) // 8, dtype=np.uint8)
    got = _packed_not(zeros, n_rows)
    np.testing.assert_array_equal(got, ones)
    if n_rows & 7:
        assert got[-1] & (0xFF & ~_tail_mask(n_rows)) == 0
    # double negation is the identity on the valid bits AND the padding
    np.testing.assert_array_equal(_packed_not(got, n_rows), zeros)


def test_a_must_not_row_does_not_leak_padding_rows():
    """The sharp case: a `must_not` is the only `~` in the combine, and a
    leaked padding bit would show up as phantom rows in `unpack`."""
    n = 13  # not a multiple of 8
    rows = ["alpha"] * n
    table = pa.table({"text": pa.array(rows, type=pa.large_string())})
    filt = Filter(must_not=(FilterCondition(field="text",
                                            match_text_from_query="kw"),))
    qv = {"kw": np.array(["beta", "alpha"], dtype=object)}
    got = evaluate(filt, table, qv)
    assert isinstance(got, PackedRowMask)
    assert got.packed.shape == (2, 2)
    # every padding bit is 0
    assert (got.packed[:, -1] & (0xFF & ~_tail_mask(n)) == 0).all()
    un = got.unpack()
    assert un.shape == (2, n)
    assert un[0].all() and not un[1].any()


# --- the grid itself --------------------------------------------------------


def test_token_grid_shape_and_accessors():
    col = pa.chunked_array([pa.array(
        ["a b", "b c", None, "", "c a"], type=pa.large_string())])
    g = _token_row_masks(col, {"a", "b", "c", "zz"}, 5)
    assert isinstance(g, TokenGrid)
    assert g.packed.shape == (4, 1)
    assert g.n_rows == 5
    assert set(g.keys()) == {"a", "b", "c", "zz"}
    assert "a" in g and "nope" not in g
    assert g.mask("a").tolist() == [True, False, False, False, True]
    assert g.mask("zz").tolist() == [False] * 5
    # `__getitem__` is the PACKED row, and `_phrase_mask` stays packed
    assert g["a"].dtype == np.uint8
    assert _phrase_mask(g, ["a", "c"]).dtype == np.uint8
    np.testing.assert_array_equal(
        np.unpackbits(_phrase_mask(g, ["a", "c"]), count=5).astype(bool),
        np.array([False, False, False, False, True]))


@pytest.mark.parametrize("width", [2, 3, 8, 32])
@pytest.mark.parametrize("n_rows", [4099, 8193, 12_345])
@pytest.mark.parametrize("batch", [4096, 4093, 4097, 1001])
def test_concurrent_batches_never_share_a_byte(width, n_rows, batch, monkeypatch):
    """The race a batch boundary inside a byte would introduce.

    Forced with a small batch size and a wide pool, over a row count that is
    NOT a multiple of the batch size, repeated so a race would show up.

    The batch size is parametrized over UNALIGNED values on purpose. With only
    the aligned 4096 this test could not fail: `_token_row_masks`'s
    `batch_rows = max(8, batch_rows & ~7)` re-imposition is a no-op on a
    multiple of 8, so deleting that line left this test green — the safety net
    the test is named for was never the reason it passed. With 4093/4097/1001
    the re-imposition is what keeps the batches byte-disjoint, and removing it
    makes this fail (usually loudly, via a broadcast error, but for some
    `(off % 8, n_here)` pairs the widths coincide and a batch writes at a
    shifted bit offset instead — silently wrong, which is why the aligned-only
    version was not good enough).
    """
    rng = np.random.default_rng(width * 100 + n_rows % 97)
    vocab = ["alpha", "beta", "gamma", "delta"]
    rows = [" ".join(rng.choice(vocab, size=int(rng.integers(1, 4))))
            for _ in range(n_rows)]
    col = pa.chunked_array([pa.array(rows, type=pa.large_string())])
    ref = {t: np.array([t in r.split() for r in rows]) for t in vocab}
    monkeypatch.setattr(F, "_scan_batch_rows", lambda *a, **k: batch)
    for _ in range(3):
        with ThreadPoolExecutor(max_workers=width) as pool:
            got = _token_row_masks(col, set(vocab), n_rows, pool)
        for t in vocab:
            np.testing.assert_array_equal(got.mask(t), ref[t], err_msg=t)


def test_batch_size_is_always_byte_aligned():
    for nbytes, n_rows, width, n_tok in [
        (50 * 1_000_000, 1_000_000, 2, 1),
        (50 * 1_000_000, 1_000_000, 32, 1),
        (50 * 1_000_000, 1_000_000, 7, 3_498),
        (10 * 1_000, 1_000, 1, 100_000),
        (1 << 40, 10_000, 3, 1),
        # A case where the TEXT bound is the binding one. In every case above
        # the parallelism term is smaller, so the text cap never decides the
        # answer and deleting it from the sizer changed nothing measurable.
        # 4 KiB rows put the text cap at 8192 while the parallelism term is
        # 25000, so the text bound wins and is actually under test.
        (4096 * 100_000, 100_000, 2, 1),
    ]:
        got = F._scan_batch_rows(nbytes, n_rows, width, n_tok)
        # Byte alignment is the invariant that matters: concurrent batches
        # write into one packed grid, so every batch must own whole bytes.
        assert got % 8 == 0, (nbytes, n_rows, width, n_tok, got)
        assert got >= 8, (nbytes, n_rows, width, n_tok, got)
        # The 4096 floor applies only when no MEMORY bound binds. It used to be
        # applied last and unconditionally, which silently defeated
        # `_SUBGRID_BYTES`: the 100k-token case below allocated 819 MB per
        # batch per thread against a documented 16 MB cap.
        cap = min(F._BATCH_TEXT_BYTES // max(1, nbytes // max(1, n_rows)),
                  F._SUBGRID_BYTES // max(1, n_tok))
        if cap >= 4096:
            assert got >= 4096, (nbytes, n_rows, width, n_tok, got)
        # BOTH memory bounds, not just the subgrid one. The text bound had no
        # property assertion anywhere -- only transcriptions of the formula --
        # so deleting it from the sizer produced a single failure, via a mirror.
        #
        # Guarded on satisfiability: when the implied cap is under 8 the floor
        # of 8 wins and the bound cannot be met (8 rows is the smallest
        # byte-aligned batch). The `1 << 40` case below is exactly that corner
        # -- 110 MB per row makes the text cap 0 -- and it overshoots 26x
        # unavoidably. Asserting unconditionally would pin that as correct.
        bpr = max(1, nbytes // max(1, n_rows))
        if F._BATCH_TEXT_BYTES // bpr >= 8:
            assert got * bpr <= F._BATCH_TEXT_BYTES, (
                f"{got * bpr} bytes of text per batch exceeds the "
                f"{F._BATCH_TEXT_BYTES} cap (batch={got}, {bpr} B/row)")
        if F._SUBGRID_BYTES // max(1, n_tok) >= 8:
            assert got * n_tok <= F._SUBGRID_BYTES, (
                f"sub_grid of {got * n_tok} bytes exceeds the "
                f"{F._SUBGRID_BYTES} cap")


# --- the combine ------------------------------------------------------------


def _bool_reference(filt, table, qv):
    """`evaluate`'s answer, materialised as a plain bool array."""
    got = evaluate(filt, table, qv)
    return got.unpack() if isinstance(got, PackedRowMask) else got


@pytest.mark.parametrize("seed", range(40))
def test_packed_combine_matches_the_per_condition_reference(seed):
    """Against `_match_text_from_query_mask`, the independent condition-major
    builder — the same A/B the pre-R7 fused combine was pinned by, now with
    row counts that are deliberately not multiples of 8."""
    rng = np.random.default_rng(seed)
    words = ["alpha", "beta", "gamma", "delta", "eps", "zeta"]
    n = int(rng.integers(1, 40))
    rows, urls = [], []
    for _ in range(n):
        rows.append(" ".join(rng.choice(words, size=int(rng.integers(0, 4))))
                    or None)
        urls.append(" ".join(rng.choice(words, size=int(rng.integers(0, 3)))))
    table = pa.table({"text": pa.array(rows, type=pa.large_string()),
                      "url": pa.array(urls, type=pa.large_string())})
    n_q = int(rng.integers(1, 12))

    def phrases():
        out = []
        for _ in range(n_q):
            k = int(rng.integers(0, 3))
            out.append(" ".join(rng.choice(words, size=k)) if k else None)
        return np.array(out, dtype=object)

    must = FilterCondition(field="text", match_text_from_query="m")
    sh1 = FilterCondition(field="url", match_text_from_query="s1")
    sh2 = FilterCondition(field="url", match_text_from_query="s2")
    mnot = FilterCondition(field="text", match_text_from_query="nn")
    qv = {"m": phrases(), "s1": phrases(), "s2": phrases(), "nn": phrases()}
    filt = Filter(must=(must,), should=(sh1, sh2), must_not=(mnot,))

    got = _bool_reference(filt, table, qv)
    ref = (F._match_text_from_query_mask(must, table, qv)
           & (F._match_text_from_query_mask(sh1, table, qv)
              | F._match_text_from_query_mask(sh2, table, qv))
           & ~F._match_text_from_query_mask(mnot, table, qv))
    np.testing.assert_array_equal(got, ref)


def test_a_two_dimensional_keep_still_narrows_per_query():
    """The `keep_2d` branch: a non-text per-query condition promotes the
    accumulator, so the combine cannot broadcast one packed row."""
    n = 11
    table = pa.table({
        "text": pa.array(["alpha"] * n, type=pa.large_string()),
        "n": pa.array(list(range(n))),
    })
    filt = Filter(must=(
        FilterCondition(field="text", match_text_from_query="kw"),
        FilterCondition(field="n", range_from_query={"gte": "lo"}),
    ))
    qv = {"kw": np.array(["alpha", "alpha", "beta"], dtype=object),
          "lo": np.array([0, 5, 0])}
    got = _bool_reference(filt, table, qv)
    assert got[0].tolist() == [True] * n
    assert got[1].tolist() == [i >= 5 for i in range(n)]
    assert not got[2].any()


# --- the three checks that were correct but undefended ----------------------
#
# None of these could produce a wrong answer as the code stands. Each was
# reverted by an adversarial reviewer with ZERO test failures, which is the
# failure mode that produced most of this change's findings: a guard defended
# only by a comment.


def test_a_mask_wider_than_its_row_count_is_rejected():
    """`PackedRowMask` requires EXACTLY `(n_rows + 7) // 8` bytes, not "at
    least". The laxer check let a mask carry stale trailing bytes, and an
    all-false mask with garbage in a spare byte then answers "yes, something
    matched" — silently, on a ground-truth path.

    `any()` now masks the last byte's padding, so a dirty bit INSIDE the final
    valid byte can no longer lie. That is not this guard's job and does not
    replace it: the tail mask covers one byte, while a mask three bytes too
    wide carries whole stale bytes that `any()` still reads in full. Both are
    needed, and this pins the width in BOTH directions; reverting `!=` to `<=`
    previously changed no test result.
    """
    from nova_bf.filters import PackedRowMask

    # exact is accepted
    PackedRowMask(np.zeros((2, 2), dtype=np.uint8), 9)     # 9 rows -> 2 bytes
    PackedRowMask(np.zeros((1, 1), dtype=np.uint8), 1)
    PackedRowMask(np.zeros((3, 0), dtype=np.uint8), 0)     # 0 rows -> 0 bytes

    # too WIDE: the case the old `<=` allowed
    for n_rows, width in ((1, 5), (9, 3), (8, 2), (0, 1)):
        with pytest.raises(ValueError, match="needs exactly"):
            PackedRowMask(np.zeros((2, width), dtype=np.uint8), n_rows)

    # too NARROW, for completeness
    with pytest.raises(ValueError, match="needs exactly"):
        PackedRowMask(np.zeros((2, 1), dtype=np.uint8), 17)

    # and the concrete harm the guard prevents: stale bytes make `.any()` lie
    stale = np.zeros((1, 5), dtype=np.uint8)
    stale[0, 4] = 0xFF                       # garbage past the 1 real row
    assert stale.any(), "fixture must actually carry stale bytes"
    with pytest.raises(ValueError):
        PackedRowMask(stale, 1)


def test_a_failing_scan_batch_waits_for_its_siblings():
    """`_token_row_masks` submits every batch, then drains ALL of them before
    re-raising. Batches write into one shared `grid` through a closure, so
    propagating the first error while siblings are still running leaves threads
    scribbling into an array the caller has abandoned.

    Made observable by failing one batch and counting how many others got to
    their final store. With the drain that count is deterministic — every
    sibling is awaited — so this cannot flake. Reverting the drain to
    `for f in futures: f.result()` previously changed no test result.
    """
    import time

    rows = [f"alpha beta gamma {i}" for i in range(96)]
    col = pa.chunked_array([pa.array(rows, type=pa.large_string())])
    tokens = {"alpha", "beta", "gamma"}

    completed: list[int] = []
    calls: list[int] = []
    real_packbits = F.np.packbits
    real_split = F.pc.split_pattern_regex

    def counting_packbits(*a, **kw):
        out = real_packbits(*a, **kw)
        time.sleep(0.05)          # widen the window a broken drain would skip
        completed.append(1)
        return out

    def failing_split(*a, **kw):
        calls.append(1)
        if len(calls) == 2:      # fail one batch, not the first
            raise RuntimeError("injected scan failure")
        return real_split(*a, **kw)

    # The pool must OUTLIVE the call and must not be wrapped in a `with`:
    # `ThreadPoolExecutor.__exit__` calls `shutdown(wait=True)`, which waits
    # for every submitted batch on the way out — so a `with` block makes the
    # siblings finish whether the drain exists or not, and the test passes for
    # the wrong reason. (That is also exactly why the OLD per-call pool was
    # safe and the SHARED pool is not: the hazard is a pool that survives the
    # failure.) Snapshot the completion count AT the raise, not after.
    at_raise = None
    pool = ThreadPoolExecutor(max_workers=4)
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(F, "_scan_batch_rows", lambda *a, **k: 8)
        monkeypatch.setattr(F.np, "packbits", counting_packbits)
        monkeypatch.setattr(F.pc, "split_pattern_regex", failing_split)
        try:
            F._token_row_masks(col, tokens, len(rows), pool)
        except RuntimeError as exc:
            assert "injected scan failure" in str(exc)
            at_raise = len(completed)
    finally:
        monkeypatch.undo()
        pool.shutdown(wait=True)

    assert at_raise is not None, "the injected failure never propagated"
    # 96 rows / 8 = 12 batches; one failed before its store, so 11 must finish.
    assert len(calls) == 12, f"expected 12 batches, saw {len(calls)}"
    assert at_raise == 11, (
        f"only {at_raise} of 11 sibling batches had finished when the error "
        f"surfaced — the drain is not waiting, so threads are still writing "
        f"into a grid the caller has released")


def test_pool_width_uses_the_usable_cpu_count_not_the_machines():
    """`_pool_width` sizes the batching, and it must ask how many CPUs this
    PROCESS may use — not how many the machine has. The two differ inside a
    container with a CPU quota, or when the process is pinned to a subset of
    cores, and the machine count is the larger one, which under-sizes batches.

    Tested by making the two answers DIFFER, which no real box here does — on
    this machine `os.cpu_count()` and `_usable_cpu_count()` are both 24, so a
    mutation swapping them is invisible without this.
    """
    from nova_bf import compute

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(F.os, "cpu_count", lambda: 99)
        monkeypatch.setattr(compute, "_usable_cpu_count", lambda: 3)

        assert F._pool_width(None) == 3, (
            "sized against the machine's cores instead of the process's quota")

        # a real pool's own width still wins — it IS the thread count
        class _Pool:
            _max_workers = 7

        assert F._pool_width(_Pool()) == 7
        # ...and a pool that does not advertise one falls back to the quota
        assert F._pool_width(object()) == 3
    finally:
        monkeypatch.undo()


def test_any_ignores_the_last_bytes_padding_bits():
    """`any()` must agree with `unpack().any()` for EVERY mask, not only for
    the ones this module happens to build.

    The padding bits of the final byte are outside `n_rows`. `unpack()` trims
    them (`count=n_rows`); a raw `self.packed.any()` does not. So a mask whose
    only set bits are padding used to report True while `unpack()` reported
    False — one object, two methods, contradicting each other on a
    ground-truth path.

    Nothing in `src` can build such a mask today (the combine's closing `&`
    against `_packed_ones` re-zeroes the tail whatever the parts held, which
    is why reverting the tail mask breaks no end-to-end test). This pins the
    property at the class instead, so the guarantee does not depend on every
    future caller knowing the convention.
    """
    from nova_bf.filters import PackedRowMask, _tail_mask

    for n_rows in range(1, 33):
        width = (n_rows + 7) // 8
        pad = (~_tail_mask(n_rows)) & 0xFF
        if not pad:
            continue                      # a byte-multiple has no padding
        packed = np.zeros((2, width), dtype=np.uint8)
        packed[:, -1] = pad               # ONLY padding bits set
        m = PackedRowMask(packed, n_rows)
        assert not m.unpack().any(), f"n_rows={n_rows}: fixture must be all-false"
        assert not m.any(), (
            f"n_rows={n_rows}: any() reported a match from padding bits alone "
            f"(last byte {pad:#04x}, valid mask {_tail_mask(n_rows):#04x}) — "
            f"it disagrees with unpack()")

    # and a real bit in the same byte must still register, so the mask is not
    # simply zeroing the last byte
    for n_rows in (1, 5, 9, 17):
        width = (n_rows + 7) // 8
        packed = np.zeros((2, width), dtype=np.uint8)
        packed[0, (n_rows - 1) // 8] = 1 << (7 - ((n_rows - 1) & 7))
        m = PackedRowMask(packed, n_rows)
        assert m.any() and m.unpack().any(), (
            f"n_rows={n_rows}: the LAST real row must still count")


def test_indexing_a_mask_with_an_integer_says_why():
    """`pm[q]` is the natural thing to write and cannot work: `packed[q]` is
    1-D, so the constructor rejects it — but its message talks about a 1-D
    array, which points at the packing rather than at the index that caused
    it. The error has to name the index and the fix.
    """
    import numpy as np_
    import pytest as pt

    from nova_bf.filters import PackedRowMask

    m = PackedRowMask(np.packbits(
        np.array([[1, 0, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=bool), axis=1), 5)

    for idx in (0, 1, np_.int64(1), np_.intp(0)):
        with pt.raises(TypeError, match="must stay 2-D") as ei:
            m[idx]
        msg = str(ei.value)
        assert f"{int(idx)}:{int(idx) + 1}" in msg, (
            f"the error must show the slice that works, got: {msg}")
        assert ".unpack()" in msg, f"and the bool-row alternative, got: {msg}"

    # the forms that DO keep the axis still work, and mean the same thing
    for keep in (slice(0, 1), [0], np.array([0])):
        assert m[keep].n_queries == 1
        assert m[keep].unpack().tolist() == [m.unpack()[0].tolist()]

    # `bool` is not an integer index here: it ADDS an axis, so it must fall to
    # the constructor's ndim check rather than claim the axis was dropped.
    with pt.raises(ValueError, match="2-D uint8"):
        m[True]
