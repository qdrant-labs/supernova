"""`build_ordinals` has two paths; they must be indistinguishable.

Ranking a rank's ids is the whole cost of `tiebreak='id'` -- 429s of a 500s
startup on a real shard, single-threaded, while the GPU is idle. The fast path
packs FIXED-WIDTH ids into uint64 lanes and sorts them on the GPU, which is
sound because for byte strings of equal length lexicographic order IS the
numeric order of the bytes read big-endian.

Everything else -- variable-width ids, no CUDA, a device too small, the kill
switch -- falls back to Arrow's string sort. These tests pin BOTH halves: that
the paths agree exactly, and that the fallback is really reachable rather than
dead code that rots (this repo has been bitten by unexercised feature-gated
paths before).
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from nova_bf.tiebreak import (
    _NO_GPU_ORDINALS,
    _fixed_width,
    _pack_lanes,
    build_ordinals,
)

try:
    import torch
    HAVE_CUDA = torch.cuda.is_available()
except ImportError:
    HAVE_CUDA = False


def _ids(n, seed=0, width=32, dup=0):
    rng = np.random.default_rng(seed)
    hexc = list("0123456789abcdef")
    out = ["<urn:uuid:" + "".join(rng.choice(hexc, width)) + ">" for _ in range(n)]
    for i in range(dup):                     # force exact-duplicate ids
        out[n // 2 + i] = out[i]
    return out


def _arrow_reference(arr) -> np.ndarray:
    """The canonical answer: Arrow's (id, pos) sort, i.e. the fallback."""
    n = len(arr)
    tb = pa.table({"id": arr, "pos": pa.array(np.arange(n, dtype=np.int64))})
    perm = np.asarray(
        pc.sort_indices(tb, sort_keys=[("id", "ascending"), ("pos", "ascending")])
    )
    ordinals = np.empty(n, dtype=np.uint32)
    ordinals[perm] = np.arange(n, dtype=np.uint32)
    return ordinals


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------
@pytest.mark.parametrize("arr,want", [
    (pa.array(["aaaa", "bbbb"], pa.string()), 4),
    (pa.array(["aaaa", "bbbb"], pa.large_string()), 4),
    (pa.array(["a", "bb"], pa.string()), None),            # variable width
    (pa.array(["x" * 80], pa.string()), None),             # wider than the cap
    (pa.array([], pa.string()), None),                     # nothing to measure
    (pa.array([1, 2], pa.int64()), None),                  # not a string column
])
def test_fixed_width_guard(arr, want):
    assert _fixed_width([arr]) == want


def test_fixed_width_rejects_mixed_widths_across_chunks():
    """Each chunk is internally uniform, but they disagree — still no fast path."""
    a = pa.array(["aaaa", "bbbb"], pa.string())
    b = pa.array(["ccccc"], pa.string())
    assert _fixed_width([a]) == 4 and _fixed_width([b]) == 5
    assert _fixed_width([a, b]) is None


def test_fixed_width_handles_sliced_arrays():
    """A slice keeps the parent's buffers; the offset must be honoured."""
    a = pa.array(["aaaa", "bbbb", "cccc", "dddd"], pa.string())
    assert _fixed_width([a.slice(1, 2)]) == 4


# --------------------------------------------------------------------------
# the property the fast path rests on
# --------------------------------------------------------------------------
def test_packed_lanes_order_matches_string_order():
    """Fixed-width byte order == big-endian numeric order of the lanes.

    Duplicates are included so the trailing position tie-break is exercised:
    a stable LSD over the lanes must agree with Arrow's explicit (id, pos).
    """
    ids = _ids(20_000, seed=3, dup=25)
    arr = pa.array(ids, pa.string())
    W = _fixed_width([arr])
    lanes = _pack_lanes([arr], W, len(ids), 4)
    lex = np.lexsort(
        (np.arange(len(ids)),) + tuple(lanes[:, j] for j in range(lanes.shape[1] - 1, -1, -1))
    )
    tb = pa.table({"id": arr, "pos": pa.array(np.arange(len(ids), dtype=np.int64))})
    ref = np.asarray(
        pc.sort_indices(tb, sort_keys=[("id", "ascending"), ("pos", "ascending")])
    )
    assert np.array_equal(lex, ref)


def test_pack_lanes_is_thread_count_invariant():
    ids = _ids(5_000, seed=4)
    arr = pa.array(ids, pa.string())
    W = _fixed_width([arr])
    one = _pack_lanes([arr], W, len(ids), 1)
    many = _pack_lanes([arr], W, len(ids), 8)
    assert np.array_equal(one, many)


# --------------------------------------------------------------------------
# the two paths agree
# --------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_CUDA, reason="needs CUDA for the fast path")
@pytest.mark.parametrize("dup", [0, 50])
def test_gpu_and_cpu_paths_give_identical_ordinals(monkeypatch, dup):
    ids = _ids(50_000, seed=5, dup=dup)
    arr = pa.array(ids, pa.string())

    gpu = build_ordinals([arr])                       # fast path
    monkeypatch.setenv(_NO_GPU_ORDINALS, "1")
    cpu = build_ordinals([arr])                       # forced fallback

    assert np.array_equal(gpu[0], cpu[0])
    assert np.array_equal(gpu[0], _arrow_reference(arr))


@pytest.mark.skipif(not HAVE_CUDA, reason="needs CUDA for the fast path")
def test_gpu_path_matches_across_file_splits(monkeypatch):
    """Ranking is global: how rows are divided into files must not matter."""
    ids = _ids(30_000, seed=6, dup=10)
    whole = pa.array(ids, pa.string())
    split = [pa.array(ids[j:j + 3000], pa.string()) for j in range(0, 30_000, 3000)]
    a = np.concatenate(build_ordinals([whole]))
    b = np.concatenate(build_ordinals(split))
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# the fallback is REACHABLE, not dead code
# --------------------------------------------------------------------------
def test_variable_width_ids_take_the_fallback_and_are_correct():
    """No fast path exists for these; the answer must still be right."""
    ids = ["a", "bb", "ccc", "b", "bbb", "a"]
    arr = pa.array(ids, pa.string())
    assert _fixed_width([arr]) is None
    assert np.array_equal(build_ordinals([arr])[0], _arrow_reference(arr))


def test_kill_switch_forces_the_fallback(monkeypatch):
    ids = _ids(2_000, seed=8, dup=5)
    arr = pa.array(ids, pa.string())
    monkeypatch.setenv(_NO_GPU_ORDINALS, "1")
    assert np.array_equal(build_ordinals([arr])[0], _arrow_reference(arr))


def test_fallback_used_when_torch_is_missing(monkeypatch):
    """No torch at all -- `_gpu_perm` must decline, not explode."""
    import builtins

    real = builtins.__import__

    def no_torch(name, *a, **kw):
        if name == "torch":
            raise ImportError("torch disabled for this test")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    ids = _ids(1_000, seed=9)
    arr = pa.array(ids, pa.string())
    assert np.array_equal(build_ordinals([arr])[0], _arrow_reference(arr))


def test_integer_ids_take_the_fallback():
    """Integer columns already sort numerically; no packing involved."""
    arr = pa.array([5, 3, 9, 3, 1], pa.int64())
    assert _fixed_width([arr]) is None
    assert np.array_equal(build_ordinals([arr])[0], _arrow_reference(arr))


# --------------------------------------------------------------------------
# degenerate input
# --------------------------------------------------------------------------
def test_identical_ids_still_rank_by_position():
    """Every id byte-identical: the ranking degenerates to corpus order."""
    arr = pa.array(["same"] * 5, pa.string())
    assert np.array_equal(build_ordinals([arr])[0], np.arange(5, dtype=np.uint32))


# --------------------------------------------------------------------------
# the narrow-key GPU mode (device too small for int64 keys)
# --------------------------------------------------------------------------
def _force_free_vram(monkeypatch, nbytes):
    """Make the guard see exactly `nbytes` free, to pick a specific mode.

    `_gpu_mode` adds back `memory_reserved - memory_allocated` (torch reports
    its own cache as used), so those have to be pinned too or a warm CUDA
    context shifts the budget out from under the test.
    """
    import torch

    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda *a, **k: (nbytes, nbytes), raising=False)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0, raising=False)


@pytest.mark.skipif(not HAVE_CUDA, reason="needs CUDA")
@pytest.mark.parametrize("dup", [0, 40])
def test_narrow_key_mode_matches_wide_key_mode(monkeypatch, dup):
    """A device that fits int32 halves but not int64 keys must still be exact.

    This is the tier that exists so a small GPU degrades to 24.7s rather than
    surrendering to the 503s CPU path -- it has to be bit-identical to be worth
    having.
    """
    ids = _ids(40_000, seed=21, dup=dup)
    arr = pa.array(ids, pa.string())
    n = len(ids)

    wide = build_ordinals([arr])                       # free VRAM -> mode 64
    _force_free_vram(monkeypatch, n * 8 * 6)           # fits 32 only
    narrow = build_ordinals([arr])

    assert np.array_equal(wide[0], narrow[0])
    assert np.array_equal(narrow[0], _arrow_reference(arr))


@pytest.mark.skipif(not HAVE_CUDA, reason="needs CUDA")
def test_too_little_vram_falls_back_to_cpu(monkeypatch):
    """Below even the narrow mode's budget the GPU is declined -- and the
    decline happens in `_gpu_mode`, BEFORE `_pack_lanes` runs, so a machine
    that will not use the GPU never pays the lane allocation (14.1 GiB at
    315M rows)."""
    from nova_bf.tiebreak import _gpu_mode, _pack_lanes
    import nova_bf.tiebreak as tb

    ids = _ids(5_000, seed=22)
    arr = pa.array(ids, pa.string())
    n = len(ids)
    _force_free_vram(monkeypatch, n * 8 * 2)           # fits neither mode
    assert _gpu_mode(n) is None

    packed = {"n": 0}
    real = _pack_lanes
    monkeypatch.setattr(tb, "_pack_lanes",
                        lambda *a, **k: (packed.__setitem__("n", packed["n"] + 1),
                                         real(*a, **k))[1])
    assert np.array_equal(build_ordinals([arr])[0], _arrow_reference(arr))
    assert packed["n"] == 0, "lanes were packed for a GPU that was declined"


# --------------------------------------------------------------------------
# the GPU ranking logic, driven on CPU tensors
#
# Without this, `_gpu_perm` has NO executable coverage off a GPU -- and a
# mutation sweep showed 8 of 12 injected bugs surviving the suite on a
# CUDA-free box, including reversing the LSD lane order and disabling sort
# stability, i.e. exactly the two failures that corrupt ground truth silently.
# Patching torch.cuda lets the real function run on CPU tensors: everything
# except actual CUDA kernel behaviour is then exercised in ordinary CI.
# --------------------------------------------------------------------------
@pytest.fixture
def gpu_on_cpu(monkeypatch):
    """Run the real GPU ranking logic on CPU tensors.

    `mem_get_info` MUST be stubbed too: without a driver it raises, `_gpu_mode`
    swallows that and returns None, and `build_ordinals` never enters the fast
    path at all -- so a test that looks like it covers the fast path covers the
    fallback instead, and fail-open makes it return the right answer anyway.
    """
    torch = pytest.importorskip("torch")
    # A developer box with either switch set would otherwise fail these: the
    # documented way to pin a GPU box for the device-parity comparison is
    # NOVA_BF_DEVICE=cpu, which makes `_gpu_ready` decline and the fast path
    # never run.
    monkeypatch.delenv("NOVA_BF_DEVICE", raising=False)
    monkeypatch.delenv(_NO_GPU_ORDINALS, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda *a, **k: (1 << 62, 1 << 62), raising=False)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0, raising=False)
    real_device = torch.device
    monkeypatch.setattr(torch, "device", lambda spec="cpu": real_device("cpu"))
    return torch


@pytest.fixture
def gpu_perm_spy(monkeypatch):
    """Record every `_gpu_perm` call so a test can PROVE the fast path ran.

    Necessary because the fast path fails OPEN: when it declines or errors,
    `build_ordinals` returns the Arrow answer, which is the same answer. So
    "the result is correct" is no evidence the fast path executed.
    """
    import nova_bf.tiebreak as tb

    seen = []
    real = tb._gpu_perm
    monkeypatch.setattr(tb, "_gpu_perm",
                        lambda lanes, mode: (seen.append(mode), real(lanes, mode))[1])
    return seen


def _lane_perm_reference(arr) -> np.ndarray:
    n = len(arr)
    tb = pa.table({"id": arr, "pos": pa.array(np.arange(n, dtype=np.int64))})
    return np.asarray(
        pc.sort_indices(tb, sort_keys=[("id", "ascending"), ("pos", "ascending")])
    )


@pytest.mark.parametrize("mode", [64, 32])
@pytest.mark.parametrize("dup", [0, 30])
def test_gpu_perm_logic_matches_arrow_on_cpu_tensors(gpu_on_cpu, mode, dup):
    """Both key widths, with and without duplicate ids."""
    from nova_bf.tiebreak import _gpu_perm, _pack_lanes

    ids = _ids(4_000, seed=31, dup=dup)
    arr = pa.array(ids, pa.string())
    W = _fixed_width([arr])
    got = _gpu_perm(_pack_lanes([arr], W, len(ids), 2), mode)
    assert got is not None
    assert np.array_equal(got, _lane_perm_reference(arr))


def test_gpu_perm_handles_non_ascii_ids(gpu_on_cpu):
    """The uint->int sign flips are no-ops on ASCII, so every other test in the
    repo would pass with them deleted. A byte >= 0x80 in the top position of a
    lane is what makes them load-bearing."""
    from nova_bf.tiebreak import _gpu_perm, _pack_lanes

    # equal BYTE length (8), differing character length, bytes above 0x7f
    ids = ["éaaaaaa", "zzzzzzzz", "aaaaaaaa", "ÿbbbbbb",
           "0000000A", "aaaaéaa", "€aaaaa"]
    arr = pa.array(ids, pa.string())
    assert _fixed_width([arr]) is not None, "fixture must be fixed-width in BYTES"
    for mode in (64, 32):
        got = _gpu_perm(_pack_lanes([arr], _fixed_width([arr]), len(ids), 2), mode)
        assert np.array_equal(got, _lane_perm_reference(arr)), f"mode {mode}"


def test_gpu_perm_declines_when_it_would_overflow_int32(gpu_on_cpu):
    """`_gpu_mode` must refuse row counts the int32 permutation cannot hold.

    MAX_ROWS_PER_WORKER is 2**32-1 but int32 stops at 2**31-1, and
    `torch.arange` WRAPS silently rather than raising, after which numpy reads
    the negatives as valid reverse indices.

    The boundary is asserted as LITERALS, not against `_MAX_INT32_ROWS`:
    written against the constant the test holds for any value of it, including
    2**32-1, which removes the guard's entire purpose.
    """
    from nova_bf.tiebreak import _gpu_mode

    assert _gpu_mode(2**31) is None, "2**31 does not fit a signed int32"
    assert _gpu_mode(2**31 - 1) == 64, "2**31-1 is the largest that does"


# --------------------------------------------------------------------------
# offset handling: the half whose failure is SILENT
# --------------------------------------------------------------------------
def test_fixed_width_slice_detects_an_ignored_offset():
    """`_fixed_width([uniform_parent.slice(...)])` passes even if `c.offset` is
    ignored, because every row is the same width anyway. Only a parent with
    UNEQUAL widths outside the slice can catch it."""
    parent = pa.array(["aa", "bb", "cccc", "dddd", "e"], pa.string())
    assert _fixed_width([parent.slice(2, 2)]) == 4      # the slice IS uniform
    assert _fixed_width([parent]) is None               # the parent is not


def test_byte_rows_reads_the_sliced_rows_not_the_parents():
    """`_byte_rows` shares `_fixed_width`'s offset arithmetic, but a bug here
    packs the WRONG ROWS and corrupts the ranking silently instead of just
    losing the fast path. Untested until now."""
    from nova_bf.tiebreak import _byte_rows

    parent = pa.array(["aaaa", "bbbb", "zzzz", "dddd", "cccc", "ffff"], pa.string())
    sl = parent.slice(2, 3)                             # zzzz, dddd, cccc
    rows = _byte_rows(sl, 4)
    got = [bytes(r).decode() for r in rows]
    assert got == ["zzzz", "dddd", "cccc"], got
    # and end to end: ordinals must rank the SLICE's ids
    assert np.array_equal(build_ordinals([sl])[0],
                          np.array([2, 1, 0], dtype=np.uint32))


# --------------------------------------------------------------------------
# chunks are not files
# --------------------------------------------------------------------------
def test_multi_chunk_files_keep_the_per_file_contract():
    """Production reads a multi-row-group parquet column as a MULTI-CHUNK
    ChunkedArray, but every other test passes one chunk per file -- so the
    chunk/file confusion this code already had once has no coverage."""
    a = pa.array([f"id{v:08d}" for v in [5, 1, 9]], pa.string())
    b = pa.array([f"id{v:08d}" for v in [7, 3]], pa.string())
    files = [pa.chunked_array([a[:1], a[1:2], a[2:]]),   # 3 chunks
             pa.chunked_array([b[:1], b[1:]])]           # 2 chunks
    got = build_ordinals(files)
    assert [len(o) for o in got] == [len(f) for f in files] == [3, 2]
    flat = np.concatenate(got)
    ref = _arrow_reference(pa.chunked_array([a, b]).combine_chunks())
    assert np.array_equal(flat, ref)


def test_fast_path_multi_chunk_files_keep_the_per_file_contract(
        gpu_on_cpu, gpu_perm_spy):
    """Same contract as the CPU test above, but through the FAST PATH.

    The two paths have SEPARATE split loops, so a chunks-vs-files mix-up in the
    fast path is invisible to a test that only reaches the fallback -- which is
    what happens on any CUDA-free machine. Driving `_gpu_perm` on CPU tensors
    is what makes this reachable in ordinary CI.
    """
    a = pa.array([f"id{v:08d}" for v in [5, 1, 9]], pa.string())
    b = pa.array([f"id{v:08d}" for v in [7, 3]], pa.string())
    files = [pa.chunked_array([a[:1], a[1:2], a[2:]]),   # 3 chunks, 1 file
             pa.chunked_array([b[:1], b[1:]])]           # 2 chunks, 1 file
    got = build_ordinals(files)
    assert gpu_perm_spy, "the FAST PATH did not run — this test proves nothing"
    assert [len(o) for o in got] == [len(f) for f in files] == [3, 2], \
        "fast path split the ordinals by CHUNK instead of by FILE"
    ref = _arrow_reference(pa.chunked_array([a, b]).combine_chunks())
    assert np.array_equal(np.concatenate(got), ref)


# --------------------------------------------------------------------------
# the mode decision itself (previously unpinned: a mutant that picks 64 when
# only 32 fits survived the whole suite, and would OOM a small device)
# --------------------------------------------------------------------------
def _mode_with_free(monkeypatch, free, reserved=0, allocated=0):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("NOVA_BF_DEVICE", raising=False)
    monkeypatch.delenv(_NO_GPU_ORDINALS, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda *a, **k: (free, free), raising=False)
    monkeypatch.setattr(torch.cuda, "memory_reserved",
                        lambda *a, **k: reserved, raising=False)
    monkeypatch.setattr(torch.cuda, "memory_allocated",
                        lambda *a, **k: allocated, raising=False)
    from nova_bf.tiebreak import _gpu_mode
    return _gpu_mode


def test_gpu_mode_thresholds_are_exact(monkeypatch):
    """Pin both boundaries. Nothing did, so picking the WIDE key when only the
    narrow one fits passed the suite — and would OOM a device that had been
    correctly measured as too small."""
    n = 1_000_000
    need64, need32 = n * 8 * 7, n * 8 * 6
    gm = _mode_with_free(monkeypatch, need64)
    assert gm(n) == 64, "exactly need64 must take the wide key"
    gm = _mode_with_free(monkeypatch, need64 - 1)
    assert gm(n) == 32, "one byte under need64 must narrow, not stay wide"
    gm = _mode_with_free(monkeypatch, need32)
    assert gm(n) == 32, "exactly need32 must take the narrow key"
    gm = _mode_with_free(monkeypatch, need32 - 1)
    assert gm(n) is None, "one byte under need32 must decline to the CPU"


def test_gpu_mode_adds_back_torchs_reclaimable_cache(monkeypatch):
    """`mem_get_info` counts torch's cached-but-unallocated pool as USED, so a
    warm process under-reports free memory and would drop a tier for no reason.
    The correction is `+ (reserved - allocated)`; the SIGN matters — flipping it
    survived the suite until this test."""
    n = 1_000_000
    need64 = n * 8 * 7
    # Raw free is one byte short of the wide key...
    gm = _mode_with_free(monkeypatch, need64 - 1)
    assert gm(n) == 32
    # ...but torch is holding a big reclaimable cache, so it really does fit.
    gm = _mode_with_free(monkeypatch, need64 - 1, reserved=10 << 20, allocated=0)
    assert gm(n) == 64, "reclaimable cache was not added back (or was subtracted)"


# --------------------------------------------------------------------------
# _gpu_ready's own decisions
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value,ready", [
    ("cpu", False), ("CPU", False), ("cuda", True), ("CUDA", True),
    ("gpu", False),           # unrecognised -> decline (compute._select_device RAISES)
    ("cuda:0", False),        # unrecognised -> decline
    ("", True),               # unset -> auto
])
def test_gpu_ready_honours_nova_bf_device(monkeypatch, value, ready):
    """A run pinned to `cpu` must not touch the GPU, or the parity harness's
    both-devices comparison compares CUDA ordinals against themselves."""
    torch = pytest.importorskip("torch")
    monkeypatch.delenv(_NO_GPU_ORDINALS, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("NOVA_BF_DEVICE", value)
    from nova_bf.tiebreak import _gpu_ready
    assert _gpu_ready() is ready


def test_gpu_ready_requires_an_actual_cuda_device(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("NOVA_BF_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from nova_bf.tiebreak import _gpu_ready
    assert _gpu_ready() is False


def test_gpu_ready_survives_a_torch_that_fails_to_import(monkeypatch):
    """A broken CUDA runtime raises OSError, not ImportError — and this
    function's whole job is to answer the question without raising."""
    import builtins

    real = builtins.__import__
    for exc in (ImportError("no torch"),
                OSError("libcudart.so.12: cannot open shared object file")):
        def boom(name, *a, _e=exc, **kw):
            if name == "torch":
                raise _e
            return real(name, *a, **kw)
        monkeypatch.setattr(builtins, "__import__", boom)
        monkeypatch.delenv("NOVA_BF_DEVICE", raising=False)
        import nova_bf.tiebreak as tb
        assert tb._gpu_ready() is False, f"{type(exc).__name__} was not absorbed"
        monkeypatch.undo()
        monkeypatch.setattr(builtins, "__import__", real)


# --------------------------------------------------------------------------
# OOM degrades, bugs do not
# --------------------------------------------------------------------------
def test_is_oom_matches_untyped_cuda_oom_but_not_real_bugs():
    """A cub/argsort workspace exhaustion often arrives as a plain RuntimeError
    rather than the allocator's typed OutOfMemoryError, so matching on type
    alone would turn a graceful degrade into a crash. The converse matters more:
    a device-side assert or a shape bug must NOT be mistaken for an OOM."""
    torch = pytest.importorskip("torch")
    from nova_bf.tiebreak import _is_oom

    assert _is_oom(RuntimeError("CUDA error: out of memory"))
    assert _is_oom(torch.cuda.OutOfMemoryError("tried to allocate"))
    assert not _is_oom(RuntimeError("device-side assert triggered"))
    assert not _is_oom(ValueError("shape mismatch"))
    assert not _is_oom(TypeError("_gpu_perm() missing 1 required argument"))


def test_gpu_perm_reraises_a_bug_and_degrades_on_oom(gpu_on_cpu, monkeypatch):
    """Fail-open returns the RIGHT answer by another route, which is exactly why
    a swallowed bug is invisible. Bugs must surface; OOM must not."""
    import torch
    from nova_bf.tiebreak import _gpu_perm, _pack_lanes

    arr = pa.array(_ids(500, seed=41), pa.string())
    lanes = _pack_lanes([arr], _fixed_width([arr]), 500, 2)

    monkeypatch.setattr(torch, "argsort",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("CUDA error: out of memory")))
    assert _gpu_perm(lanes, 64) is None, "an OOM should degrade to the CPU path"

    monkeypatch.setattr(torch, "argsort",
                        lambda *a, **k: (_ for _ in ()).throw(
                            ValueError("this is a bug, not an OOM")))
    with pytest.raises(ValueError, match="this is a bug"):
        _gpu_perm(lanes, 64)


def test_all_paths_agree_on_a_hard_case(gpu_on_cpu, gpu_perm_spy, monkeypatch):
    """The case the parity module can only run on real CUDA, run here on CPU
    tensors so it is covered in ordinary CI.

    Deliberately hostile: ids NOT monotonic in corpus position (so the ranking
    is not the identity and a wrong order is visible), exact DUPLICATE ids (so
    the position tie-break actually decides something), 11 bytes (so
    `nlanes == 2` and lane order / the 32-bit half split are observable), and
    several chunks per file (so the per-file split is exercised).
    """
    import nova_bf.tiebreak as tb

    rng = np.random.default_rng(17)
    vals = [f"{(g * 2654435761) % 10**11:011d}" for g in range(900)]
    vals[100:110] = vals[:10]                       # exact duplicates
    rng.shuffle(vals)
    files = [
        pa.chunked_array([pa.array(vals[0:120], pa.string()),
                          pa.array(vals[120:300], pa.string())]),
        pa.chunked_array([pa.array(vals[300:640], pa.string())]),
        pa.chunked_array([pa.array(vals[640:700], pa.string()),
                          pa.array(vals[700:900], pa.string())]),
    ]
    flat = [v for f in files for v in f.to_pylist()]
    assert flat != sorted(flat), "fixture must not be monotonic"
    assert len(set(flat)) < len(flat), "fixture must contain duplicate ids"
    assert _fixed_width([c for f in files for c in f.chunks]) == 11

    wide = build_ordinals(files)
    assert gpu_perm_spy == [64], f"wide path did not run: {gpu_perm_spy}"

    monkeypatch.setattr(tb, "_gpu_mode", lambda total: 32)
    narrow = build_ordinals(files)
    assert gpu_perm_spy == [64, 32], f"narrow path did not run: {gpu_perm_spy}"

    monkeypatch.setenv(_NO_GPU_ORDINALS, "1")
    cpu = build_ordinals(files)
    assert gpu_perm_spy == [64, 32], "the CPU path should not have used the GPU"

    ref = _arrow_reference(pa.chunked_array(
        [c for f in files for c in f.chunks]).combine_chunks())
    assert [len(a) for a in wide] == [len(f) for f in files]
    for i, (a, b, c) in enumerate(zip(wide, narrow, cpu)):
        assert np.array_equal(a, b), f"file {i}: int64 vs int32 key modes differ"
        assert np.array_equal(a, c), f"file {i}: GPU vs CPU ordinals differ"
    assert np.array_equal(np.concatenate(wide), ref), "disagrees with Arrow"
    assert not np.array_equal(np.concatenate(wide), np.arange(len(flat))), \
        "ordinals are the identity — the fixture is not scrambling"


# --------------------------------------------------------------------------
# the TOCTOU re-check's OTHER two branches
#
# `_gpu_mode` is called once to choose a mode and AGAIN after `_pack_lanes`,
# because packing takes minutes and another process can take the memory in
# between. Every other test patches `_gpu_mode` to a CONSTANT, so both calls
# agree and only the `again == mode` branch ever ran — the decline and
# downgrade branches had no coverage on any device, and mutants that threw the
# re-check away or ran the GPU with the rejected mode both survived.
# --------------------------------------------------------------------------
def _mode_sequence(monkeypatch, *values):
    """Make successive `_gpu_mode` calls return `values` in order."""
    import nova_bf.tiebreak as tb

    seq = list(values)
    calls = []

    def fake(total):
        v = seq[min(len(calls), len(seq) - 1)]
        calls.append(v)
        return v

    monkeypatch.setattr(tb, "_gpu_mode", fake)
    return calls


def test_toctou_recheck_declining_falls_back_to_cpu(gpu_on_cpu, monkeypatch):
    """Memory taken away DURING packing: the re-check returns None and the run
    must fall back, not sort with a mode that no longer fits."""
    import nova_bf.tiebreak as tb

    arr = pa.array(_ids(2_000, seed=51, dup=8), pa.string())
    calls = _mode_sequence(monkeypatch, 64, None)
    seen = []
    real = tb._gpu_perm
    monkeypatch.setattr(tb, "_gpu_perm",
                        lambda lanes, mode: (seen.append(mode), real(lanes, mode))[1])

    got = build_ordinals([arr])
    assert calls == [64, None], f"the re-check did not happen: {calls}"
    assert seen == [], f"the GPU ran with a mode the re-check rejected: {seen}"
    assert np.array_equal(got[0], _arrow_reference(arr)), "fallback answer wrong"


def test_toctou_recheck_downgrading_uses_the_narrower_mode(gpu_on_cpu, monkeypatch):
    """Memory shrank but not to nothing: the run must use the mode the SECOND
    call returned, not the first."""
    import nova_bf.tiebreak as tb

    arr = pa.array(_ids(2_000, seed=52, dup=8), pa.string())
    calls = _mode_sequence(monkeypatch, 64, 32)
    seen = []
    real = tb._gpu_perm
    monkeypatch.setattr(tb, "_gpu_perm",
                        lambda lanes, mode: (seen.append(mode), real(lanes, mode))[1])

    got = build_ordinals([arr])
    assert calls == [64, 32], f"the re-check did not happen: {calls}"
    assert seen == [32], f"ran with the stale mode instead of the re-checked one: {seen}"
    assert np.array_equal(got[0], _arrow_reference(arr))


def _meminfo(monkeypatch, avail_bytes, *, line=True):
    """Stub `/proc/meminfo` so the host guard is tested against a KNOWN
    MemAvailable rather than whatever this box happens to have free.

    The previous version of these assertions read real `/proc/meminfo`, so
    whether they held depended on the machine — on a 256 GiB box the "179 GiB
    must be declined" case would have passed the guard.
    """
    import builtins
    import io

    # `/proc/meminfo` is in kB and `_host_can_pack` multiplies back by 1024, so
    # a byte-granular stub silently rounds DOWN and the value under test is not
    # the value asserted. Refuse unaligned input rather than mislead.
    assert avail_bytes % 1024 == 0, \
        f"{avail_bytes} is not kB-aligned; /proc/meminfo cannot express it"
    body = (f"MemTotal:       999999999 kB\n"
            + (f"MemAvailable:   {avail_bytes // 1024} kB\n" if line else "")
            + "Buffers:            12345 kB\n")
    real_open = builtins.open

    def fake_open(path, *a, **kw):
        if str(path) == "/proc/meminfo":
            return io.StringIO(body)
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_host_memory_guard_declines_an_impossible_allocation(monkeypatch):
    """`_gpu_mode` sizes DEVICE memory; the packed lanes live on the HOST and
    were sized by nothing. numpy's MemoryError wording is not matched by
    `_is_oom`, and the likelier outcome is a lazy calloc that succeeds and then
    gets the process killed while the pages are touched."""
    from nova_bf.tiebreak import _host_can_pack

    _meminfo(monkeypatch, 8 << 30)
    assert _host_can_pack(1_000, 6) is True
    assert _host_can_pack(3_000_000_000, 8) is False, "179 GiB must be declined"


def test_host_memory_guard_scales_with_the_lane_COUNT(monkeypatch):
    """The whole point of `nlanes` living here.

    `_gpu_mode` is a function of `total` alone (device residency is
    lane-independent — `_gpu_perm` stages one column at a time), so this is the
    ONLY place the `total * nlanes * 8` host array is sized. A guard that
    ignored `nlanes` would be sizing a 1-lane array for a 64-byte id and let
    the 8x allocation through.
    """
    from nova_bf.tiebreak import _host_can_pack

    total = 200_000_000                       # 1.6 GB per lane
    _meminfo(monkeypatch, 6 << 30)            # fits 1 lane at 1.5x, not 8
    assert _host_can_pack(total, 1) is True, "1 lane (1.5 GiB) must fit in 6 GiB"
    assert _host_can_pack(total, 8) is False, \
        "8 lanes (11.9 GiB) must NOT fit in 6 GiB — the guard ignored nlanes"


def test_host_memory_guard_headroom_boundary_is_exact(monkeypatch):
    """1.5x, pinned on both sides: packing also reads the id buffers, so a
    guard that required only `need` would approve an allocation that then
    SIGKILLs the rank while the pages are touched."""
    from nova_bf.tiebreak import _host_can_pack

    # 2**20 rows x 6 lanes makes `need * 3 // 2` exactly 73728 kB, so both
    # sides of the boundary are representable in `/proc/meminfo`'s units.
    total, nlanes = 1 << 20, 6
    need = total * nlanes * 8
    assert (need * 3 // 2) % 1024 == 0
    _meminfo(monkeypatch, need * 3 // 2)
    assert _host_can_pack(total, nlanes) is True, "exactly 1.5x must be allowed"
    _meminfo(monkeypatch, need * 3 // 2 - 1024)
    assert _host_can_pack(total, nlanes) is False, "one kB under 1.5x must decline"
    _meminfo(monkeypatch, need)
    assert _host_can_pack(total, nlanes) is False, \
        "`need` with no headroom must decline"


def test_host_memory_guard_is_permissive_when_it_cannot_measure(monkeypatch):
    """Best-effort by design: a container with no `MemAvailable` line, or an
    unreadable `/proc`, must not turn the fast path off. Fail OPEN here — the
    guard is an optimisation-preserving safety net, not a correctness gate, and
    the CPU fallback it would force is 5.5x slower."""
    from nova_bf.tiebreak import _host_can_pack

    _meminfo(monkeypatch, 0, line=False)
    assert _host_can_pack(3_000_000_000, 8) is True, \
        "no MemAvailable line must proceed, not decline"

    import builtins
    real_open = builtins.open

    def boom(path, *a, **kw):
        if str(path) == "/proc/meminfo":
            raise OSError("simulated: /proc not mounted")
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", boom)
    assert _host_can_pack(3_000_000_000, 8) is True, \
        "an unreadable /proc must proceed, not decline"


@pytest.mark.parametrize("width", [8, 5, 47, 16, 9])
def test_packed_lanes_are_big_endian_bytes_zero_padded_on_the_right(width):
    """The exact layout, not just the order it induces.

    A lane is the id's bytes read big-endian, and a width that is not a
    multiple of 8 pads the LAST lane on the RIGHT with zeros -- pad on the left
    and every short id would sort as though it began with NULs. `_pack_lanes`
    reinterprets the arrow buffer instead of assembling this arithmetically, so
    the layout it assumes is worth stating outright.
    """
    rng = np.random.default_rng(width)
    vals = ["".join(chr(c) for c in rng.integers(33, 127, size=width))
            for _ in range(64)]
    arr = pa.array(vals, pa.string())
    assert _fixed_width([arr]) == width
    lanes = _pack_lanes([arr], width, len(vals), 4)
    nl = (width + 7) // 8
    assert lanes.shape == (len(vals), nl) and lanes.dtype == np.uint64

    want = np.zeros((len(vals), nl), dtype=np.uint64)
    for i, v in enumerate(vals):
        b = v.encode() + b"\x00" * (nl * 8 - width)
        for j in range(nl):
            want[i, j] = int.from_bytes(b[j * 8:(j + 1) * 8], "big")
    assert np.array_equal(lanes, want)
