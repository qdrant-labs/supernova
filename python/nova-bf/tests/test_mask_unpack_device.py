"""`_unpack_row_axis_device` must be `_unpack_row_axis`'s exact image.

`select` expands the CPU-fallback per-query filter mask on the compute device
now, instead of on the consumer thread. That expansion decides WHICH ROW each
mask bit constrains, and as `_process_shared_batch` puts it, a mask read at the
wrong offset masks the wrong rows and does not raise — the run just produces a
well-formed top-K over the wrong corpus rows. So the device version is pinned
against the numpy one, cell for cell, rather than against a hand-written
expectation.

The numpy version is therefore the ORACLE here, not a second implementation
under test: these tests fail if the device path drifts from it, in either
direction.

The packing runs along the ROW axis (see `filters.PackedRowMask`), so the two
things that can go wrong are the bit ORDER within a byte and the `bit_offset`
trim that handles a slice whose first row is not byte-aligned. Both get their
own pinned test below, because both survive a symmetric round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nova_bf import compute as compute_mod
from nova_bf.filters import PackedRowMask, pack_rows

# Row counts that exercise every remainder mod 8, both sides of one byte, and
# a production-ish width.
ROWS = [1, 2, 3, 7, 8, 9, 13, 15, 16, 17, 22, 26, 64, 65, 100, 4096]


def _both(packed, n_rows, bit_offset=0, device="cpu"):
    ref = compute_mod._unpack_row_axis(packed, n_rows, bit_offset)
    got = compute_mod._unpack_row_axis_device(packed, n_rows, device, bit_offset)
    return ref, got


@pytest.mark.parametrize("n_rows", ROWS)
@pytest.mark.parametrize("n_queries", [1, 2, 7, 64, 257])
def test_matches_numpy_expansion(n_queries, n_rows):
    rng = np.random.default_rng(n_queries * 1000 + n_rows)
    mask = rng.random((n_queries, n_rows)) < 0.35
    pm = pack_rows(mask)
    ref, got = _both(pm.packed, n_rows)
    assert got.dtype is torch.bool
    assert tuple(got.shape) == (n_queries, n_rows)
    np.testing.assert_array_equal(got.numpy(), ref)
    # and both really are the mask that went in
    np.testing.assert_array_equal(got.numpy(), mask)
    # the pure-numpy reference round-trips too
    np.testing.assert_array_equal(pm.unpack(), mask)


@pytest.mark.parametrize("n_queries", [1, 7, 8, 26])
def test_matches_numpy_on_byte_range_slice(n_queries):
    """The shape `select` actually passes on the production path: a BYTE-RANGE
    column slice of the whole file's packed mask, which is a strided
    (non-contiguous) view, plus the `bit_offset` that trims the rows riding
    along in the first byte. A stride or an offset mishandled here reads the
    wrong rows with no error."""
    rng = np.random.default_rng(n_queries)
    n = 500
    mask = rng.random((n_queries, n)) < 0.4
    packed = pack_rows(mask).packed
    saw_strided = False
    # Deliberately includes ranges that are NOT byte-aligned on either end.
    for r0, r1 in [(0, 8), (0, 1), (3, 9), (8, 64), (17, 400), (496, 500),
                   (0, 500), (255, 256), (1, 499)]:
        view = packed[:, r0 >> 3 : (r1 + 7) >> 3]
        saw_strided |= not view.flags["C_CONTIGUOUS"]
        ref, got = _both(view, r1 - r0, r0 & 7)
        np.testing.assert_array_equal(got.numpy(), ref)
        np.testing.assert_array_equal(got.numpy(), mask[:, r0:r1])
    # A one-query mask slices to `(1, w)`, which numpy still calls contiguous
    # — so only the taller cases actually exercise the strides. Assert which
    # ones did, rather than assuming.
    assert saw_strided == (n_queries > 1)


def test_all_true_and_all_false_masks():
    for value in (True, False):
        for n_queries in (1, 8, 13):
            mask = np.full((n_queries, 33), value)
            ref, got = _both(pack_rows(mask).packed, 33)
            np.testing.assert_array_equal(got.numpy(), ref)
            assert bool(got.all()) is value


def test_bit_order_is_big_endian_not_reversed():
    """The failure this catches: a reversed shift order (`0..7` instead of
    `7..0`) still round-trips through pack/unpack pairs built the same way, so
    it passes a symmetric round-trip test while permuting rows within each
    byte. Pinned against a hand-built packed byte instead.

    This is also the bit order `np.packbits` uses by default, which is what
    makes `PackedRowMask.unpack` and `np.unpackbits` interchangeable."""
    # one query, 8 rows, only row 0 true -> big-endian bit 7 -> byte 0x80
    packed = np.array([[0x80]], dtype=np.uint8)
    got = compute_mod._unpack_row_axis_device(packed, 8, "cpu").numpy()
    assert got[0, 0] and not got[0, 1:].any()
    # only row 7 true -> bit 0 -> 0x01
    got = compute_mod._unpack_row_axis_device(
        np.array([[0x01]], dtype=np.uint8), 8, "cpu").numpy()
    assert got[0, 7] and not got[0, :7].any()
    # and packbits agrees with both
    m = np.zeros((1, 8), dtype=bool)
    m[0, 0] = True
    assert pack_rows(m).packed.tolist() == [[0x80]]


def test_bit_offset_reads_the_rows_it_says():
    """`bit_offset` is the one piece of arithmetic `select` does that the
    packing itself does not: it must skip exactly that many leading rows of
    the byte window, not round to a byte."""
    rng = np.random.default_rng(3)
    mask = rng.random((5, 64)) < 0.5
    packed = pack_rows(mask).packed
    for off in range(8):
        got = compute_mod._unpack_row_axis_device(packed, 16, "cpu", off).numpy()
        np.testing.assert_array_equal(got, mask[:, off : off + 16])


class TestPackedSliceAny:
    """`select` skips a whole slice when no query keeps any of its rows.

    That check MUST be exact, and the boundary bytes are why: they also carry
    rows outside the slice. Skipping a slice contributes nothing to the top-K;
    keeping it contributes `-inf` candidates carrying REAL ordinals, which
    outrank the `-inf` sentinel and can enter an under-filled row. So a
    conservative "maybe" is not a free choice here, it changes results.
    """

    def test_agrees_with_the_expanded_mask_everywhere(self):
        rng = np.random.default_rng(11)
        for n_queries in (1, 5, 8, 13):
            for _ in range(60):
                n = 40
                mask = rng.random((n_queries, n)) < 0.05
                packed = pack_rows(mask).packed
                for r0, r1 in [(0, 1), (0, 8), (3, 9), (8, 16), (7, 33),
                               (39, 40), (0, 40), (17, 18)]:
                    assert compute_mod._packed_slice_any(packed, r0, r1 - r0) \
                        == bool(mask[:, r0:r1].any())

    def test_a_bit_set_only_outside_the_slice_does_not_count(self):
        """The exact case byte-truthiness would get wrong."""
        mask = np.zeros((1, 16), dtype=bool)
        mask[0, 7] = True                       # last row of byte 0
        packed = pack_rows(mask).packed
        assert packed[0, 0] != 0                # the byte IS non-zero
        assert compute_mod._packed_slice_any(packed, 0, 7) is False
        assert compute_mod._packed_slice_any(packed, 7, 1) is True
        # and from the other side: a bit in the tail padding of the last byte
        mask2 = np.zeros((1, 16), dtype=bool)
        mask2[0, 12] = True
        packed2 = pack_rows(mask2).packed
        assert compute_mod._packed_slice_any(packed2, 8, 4) is False
        assert compute_mod._packed_slice_any(packed2, 8, 5) is True

    def test_single_byte_window_masks_both_ends(self):
        """A window inside ONE byte has to apply the head AND tail mask at
        once — the easy bug is applying only one and reading the other side's
        neighbours."""
        mask = np.zeros((1, 8), dtype=bool)
        mask[0, 1] = mask[0, 6] = True
        packed = pack_rows(mask).packed
        assert compute_mod._packed_slice_any(packed, 2, 4) is False
        assert compute_mod._packed_slice_any(packed, 1, 1) is True
        assert compute_mod._packed_slice_any(packed, 6, 1) is True

    def test_empty_and_degenerate_windows(self):
        packed = pack_rows(np.ones((3, 16), dtype=bool)).packed
        assert compute_mod._packed_slice_any(packed, 0, 0) is False
        assert compute_mod._packed_slice_any(
            np.zeros((0, 2), dtype=np.uint8), 0, 16) is False


def test_shifts_tensor_is_cached_per_device():
    """Rebuilt per call it would be a host-to-device copy per batch slice."""
    compute_mod._unpack_row_axis_device(
        np.zeros((2, 3), dtype=np.uint8), 9, "cpu")
    dev = torch.device("cpu")
    first = compute_mod._bit_shifts(dev)
    assert compute_mod._bit_shifts(dev) is first
    np.testing.assert_array_equal(first.numpy(), [128, 64, 32, 16, 8, 4, 2, 1])


class TestPackedRowMask:
    def test_carries_its_row_count_and_reports_the_logical_shape(self):
        mask = np.zeros((3, 13), dtype=bool)
        pm = pack_rows(mask)
        assert pm.packed.shape == (3, 2)      # 13 rows -> 2 bytes
        assert pm.shape == (3, 13)            # ...but the mask is 13 wide
        assert pm.n_rows == 13 and pm.n_queries == 3 and pm.ndim == 2

    def test_query_axis_narrowing_is_a_plain_index(self):
        rng = np.random.default_rng(2)
        mask = rng.random((10, 37)) < 0.5
        pm = pack_rows(mask)
        rows = np.array([0, 3, 9])
        np.testing.assert_array_equal(pm[rows].unpack(), mask[rows])
        assert pm[rows].n_rows == 37

    def test_any_is_exact(self):
        rng = np.random.default_rng(4)
        for _ in range(50):
            mask = rng.random((4, 21)) < 0.03
            assert pack_rows(mask).any() == bool(mask.any())

    def test_rejects_a_mask_it_cannot_describe(self):
        with pytest.raises(ValueError):
            PackedRowMask(np.zeros(4, dtype=np.uint8), 4)          # 1-D
        with pytest.raises(ValueError):
            PackedRowMask(np.zeros((2, 2), dtype=bool), 4)         # not uint8
        with pytest.raises(ValueError):
            PackedRowMask(np.zeros((2, 2), dtype=np.uint8), 17)    # 17 > 2*8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("n_queries", [1, 9, 5000])
def test_matches_numpy_on_cuda(n_queries):
    rng = np.random.default_rng(n_queries)
    mask = rng.random((n_queries, 4096)) < 0.3
    packed = pack_rows(mask).packed
    for r0, r1 in [(0, 4096), (0, 4096 - 3), (8, 2048), (13, 1000)]:
        view = packed[:, r0 >> 3 : (r1 + 7) >> 3]
        got = compute_mod._unpack_row_axis_device(
            view, r1 - r0, "cuda", r0 & 7)
        assert got.device.type == "cuda"
        np.testing.assert_array_equal(got.cpu().numpy(), mask[:, r0:r1])
