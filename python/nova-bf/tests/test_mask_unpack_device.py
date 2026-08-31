"""`_unpack_query_axis_device` must be `_unpack_query_axis`'s exact image.

`select` expands the CPU-fallback per-query filter mask on the compute device
now, instead of on the consumer thread. That expansion decides WHICH QUERY each
mask row constrains, and as `_process_shared_batch` puts it, a mask read at the
wrong height masks the wrong queries and does not raise — the run just produces
a well-formed top-K for the wrong query. So the device version is pinned
against the numpy one it replaced, cell for cell, rather than against a
hand-written expectation.

The numpy version is therefore the ORACLE here, not a second implementation
under test: these tests fail if the device path drifts from it, in either
direction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nova_bf import compute as compute_mod

# Heights that exercise every remainder mod 8, both sides of one byte, and the
# production height (5000 filtered_text queries -> 625 bytes).
HEIGHTS = [1, 2, 3, 7, 8, 9, 13, 15, 16, 17, 22, 26, 64, 65, 100, 5000]


def _both(mask_or_packed, n_queries, device="cpu"):
    packed = mask_or_packed
    ref = compute_mod._unpack_query_axis(packed, n_queries)
    got = compute_mod._unpack_query_axis_device(packed, n_queries, device)
    return ref, got


@pytest.mark.parametrize("n_queries", HEIGHTS)
@pytest.mark.parametrize("n_rows", [1, 2, 7, 64, 257])
def test_matches_numpy_expansion(n_queries, n_rows):
    rng = np.random.default_rng(n_queries * 1000 + n_rows)
    mask = rng.random((n_queries, n_rows)) < 0.35
    packed = compute_mod._pack_query_axis(mask)
    ref, got = _both(packed, n_queries)
    assert got.dtype is torch.bool
    assert tuple(got.shape) == (n_queries, n_rows)
    np.testing.assert_array_equal(got.numpy(), ref)
    # and both really are the mask that went in
    np.testing.assert_array_equal(got.numpy(), mask)


@pytest.mark.parametrize("n_queries", [1, 7, 8, 9, 13, 26])
def test_matches_numpy_on_strided_column_slice(n_queries):
    """The shape `select` actually passes: a column slice of the WHOLE file's
    packed mask, which is a strided (non-contiguous) view, not a fresh array.
    A stride mishandled here would read the wrong rows with no error."""
    rng = np.random.default_rng(n_queries)
    mask = rng.random((n_queries, 500)) < 0.4
    packed = compute_mod._pack_query_axis(mask)
    saw_strided = False
    for lo, hi in [(0, 1), (3, 4), (0, 250), (17, 400), (499, 500)]:
        view = packed[:, lo:hi]
        saw_strided |= not view.flags["C_CONTIGUOUS"]
        ref, got = _both(view, n_queries)
        np.testing.assert_array_equal(got.numpy(), ref)
        np.testing.assert_array_equal(got.numpy(), mask[:, lo:hi])
    # A one-byte-tall mask slices to `(1, w)`, which numpy still calls
    # contiguous — so only the taller heights actually exercise the strides.
    # Assert that at least one case did, rather than assuming.
    assert saw_strided == (math.ceil(n_queries / 8) > 1)


@pytest.mark.parametrize("n_queries", [1, 5, 8, 13])
def test_matches_numpy_on_fancy_indexed_rows(n_queries):
    """The other `true_rows` form: `orig_rows[r0:r1]`, i.e. a fancy index,
    which lands non-monotonic row order on a compacted batch."""
    rng = np.random.default_rng(n_queries + 77)
    mask = rng.random((n_queries, 300)) < 0.5
    packed = compute_mod._pack_query_axis(mask)
    cols = rng.permutation(300)[:64]
    ref, got = _both(packed[:, cols], n_queries)
    np.testing.assert_array_equal(got.numpy(), ref)
    np.testing.assert_array_equal(got.numpy(), mask[:, cols])


def test_all_true_and_all_false_masks():
    for value in (True, False):
        for n_queries in (1, 8, 13):
            mask = np.full((n_queries, 33), value)
            ref, got = _both(compute_mod._pack_query_axis(mask), n_queries)
            np.testing.assert_array_equal(got.numpy(), ref)
            assert bool(got.all()) is value


@pytest.mark.parametrize("n_queries,packed_height", [(13, 3), (1, 2), (9, 4)])
def test_over_tall_height_pads_with_false_like_numpy(n_queries, packed_height):
    """Asking for MORE queries than there are packed bits pads with all-`False`
    rows — numpy's `count=` behaviour, and the direction
    `tests/parity/test_parity_mask_height.py` documents as the conservative
    one. Truncating instead would silently shorten a mask."""
    rng = np.random.default_rng(packed_height)
    packed = rng.integers(0, 256, size=(packed_height, 11), dtype=np.uint8)
    want = packed_height * 8 + n_queries          # deliberately past the end
    ref = compute_mod._unpack_query_axis(packed, want)
    got = compute_mod._unpack_query_axis_device(packed, want, "cpu")
    assert tuple(got.shape) == (want, 11)
    np.testing.assert_array_equal(got.numpy(), ref)
    assert not got[packed_height * 8:].any()      # the padding is False


def test_short_height_truncates_like_numpy():
    """The other direction: fewer queries than packed bits reads a PREFIX of
    the query axis. Padding here instead would invent queries."""
    rng = np.random.default_rng(5)
    mask = rng.random((26, 40)) < 0.5
    packed = compute_mod._pack_query_axis(mask)
    for n in (1, 7, 8, 9, 25):
        ref = compute_mod._unpack_query_axis(packed, n)
        got = compute_mod._unpack_query_axis_device(packed, n, "cpu")
        np.testing.assert_array_equal(got.numpy(), ref)
        np.testing.assert_array_equal(got.numpy(), mask[:n])


def test_bit_order_is_query_major_not_reversed():
    """The failure this catches: a reversed shift order (`0..7` instead of
    `7..0`) still round-trips through pack/unpack pairs built the same way, so
    it passes a symmetric round-trip test while permuting queries within each
    byte. Pinned against a hand-built packed byte instead."""
    # one row, 8 queries, only query 0 true -> big-endian bit 7 -> byte 0x80
    packed = np.array([[0x80]], dtype=np.uint8)
    got = compute_mod._unpack_query_axis_device(packed, 8, "cpu").numpy()
    assert got[0, 0] and not got[1:, 0].any()
    # only query 7 true -> bit 0 -> 0x01
    got = compute_mod._unpack_query_axis_device(
        np.array([[0x01]], dtype=np.uint8), 8, "cpu").numpy()
    assert got[7, 0] and not got[:7, 0].any()


def test_packed_any_is_exact_empty_slice_test():
    """`select` early-outs on `packed_np.any()` instead of `.any()` on the
    expanded device tensor. That is only sound because a packed byte is 0 iff
    every query bit it holds is 0 — so the two agree on EVERY mask, including
    ones whose only true cells sit in the padding bits of the last byte."""
    rng = np.random.default_rng(11)
    for n_queries in (1, 5, 8, 9, 13, 26):
        for _ in range(40):
            mask = rng.random((n_queries, 9)) < 0.06
            packed = compute_mod._pack_query_axis(mask)
            assert bool(packed.any()) == bool(mask.any())
            for lo, hi in [(0, 1), (2, 5), (0, 9)]:
                assert bool(packed[:, lo:hi].any()) == bool(mask[:, lo:hi].any())


def test_shifts_tensor_is_cached_per_device():
    """Rebuilt per call it would be a host-to-device copy per batch slice."""
    compute_mod._unpack_query_axis_device(
        np.zeros((2, 3), dtype=np.uint8), 9, "cpu")
    dev = torch.device("cpu")
    first = compute_mod._bit_shifts(dev)
    assert compute_mod._bit_shifts(dev) is first
    np.testing.assert_array_equal(first.numpy(), [128, 64, 32, 16, 8, 4, 2, 1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("n_queries", [1, 9, 13, 5000])
def test_matches_numpy_on_cuda(n_queries):
    rng = np.random.default_rng(n_queries)
    mask = rng.random((n_queries, 129)) < 0.3
    packed = compute_mod._pack_query_axis(mask)
    ref = compute_mod._unpack_query_axis(packed, n_queries)
    got = compute_mod._unpack_query_axis_device(packed[:, 3:100], n_queries, "cuda")
    assert got.device.type == "cuda"
    np.testing.assert_array_equal(got.cpu().numpy(), ref[:, 3:100])
