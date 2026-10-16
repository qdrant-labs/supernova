"""Read-path performance changes that must not move a single output byte.

- `dense_to_2d` returns a NUMPY-owned array filled chunk by chunk (no
  `combine_chunks`), so freeing a file's vectors no longer frees a multi-GB
  Arrow buffer with the GIL held on the GPU-feeding thread.
- `Store._ranged_download` copies each range with the GIL released and runs on
  one process-wide pool, still bounded per file.
- The Arrow-native read (`io_ranged_get: false`) pins pre_buffer/use_threads.
- A dense/sparse batch wider than the Triton top-K kernel's MAX_BLOCK warns
  (INFO when unset = whole-file slices), only where the kernel can run.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nova_bf import io as io_mod
from nova_bf.io import Store, dense_to_2d


def _old_dense_to_2d(col: pa.ChunkedArray) -> np.ndarray:
    """The pre-change implementation, verbatim: the equality reference."""
    col = col.combine_chunks()
    n = len(col)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if pa.types.is_fixed_size_list(col.type):
        dim = col.type.list_size
        flat = col.flatten().to_numpy(zero_copy_only=False)
    else:
        flat = col.values.to_numpy(zero_copy_only=False)
        dim = len(flat) // n
    if flat.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        flat = flat.astype(np.float32, copy=False)
    return np.ascontiguousarray(flat.reshape(n, dim))


DIM = 7
LIST_KINDS = {
    "list": lambda t: pa.list_(t),
    "large_list": lambda t: pa.large_list(t),
    "fixed_size_list": lambda t: pa.list_(t, DIM),
}
FLOATS = {"f16": (np.float16, pa.float16()), "f32": (np.float32, pa.float32()),
          "f64": (np.float64, pa.float64())}


def _column(kind: str, dt: str, layout: str, n: int = 50) -> pa.ChunkedArray:
    np_t, pa_t = FLOATS[dt]
    rng = np.random.default_rng(zlib.crc32(f"{kind}/{dt}/{layout}".encode()))
    # f64 values that do NOT round-trip through f32 exercise the narrowing.
    vals = rng.standard_normal((n, DIM)).astype(np_t)
    typ = LIST_KINDS[kind](pa_t)
    flat = pa.array(vals.ravel(), type=pa_t)
    if kind == "fixed_size_list":
        whole = pa.FixedSizeListArray.from_arrays(flat, DIM)
    else:
        offs = np.arange(0, n * DIM + 1, DIM)
        cls = pa.LargeListArray if kind == "large_list" else pa.ListArray
        whole = cls.from_arrays(pa.array(offs, pa.int64() if kind == "large_list" else pa.int32()), flat)
    assert whole.type == typ
    if layout == "single":
        return pa.chunked_array([whole])
    if layout == "chunked":
        return pa.chunked_array([whole.slice(0, 13), whole.slice(13, 0),
                                 whole.slice(13, 20), whole.slice(33)], type=typ)
    if layout == "sliced":  # chunks with nonzero logical offsets
        return pa.chunked_array([whole.slice(5, 11), whole.slice(29, 17)], type=typ)
    raise AssertionError(layout)


@pytest.mark.parametrize("layout", ["single", "chunked", "sliced"])
@pytest.mark.parametrize("dt", list(FLOATS))
@pytest.mark.parametrize("kind", list(LIST_KINDS))
def test_dense_to_2d_matches_old_path(kind, dt, layout):
    col = _column(kind, dt, layout)
    got = dense_to_2d(col)
    want = _old_dense_to_2d(col)
    assert got.dtype == want.dtype
    assert got.shape == want.shape
    assert got.flags.c_contiguous
    assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("kind", list(LIST_KINDS))
def test_dense_to_2d_result_is_numpy_owned(kind):
    """Not a view of an Arrow buffer: the array owns its memory, so dropping it
    never frees an Arrow allocation. The OLD path returned a view."""
    col = _column(kind, "f32", "single")
    got = dense_to_2d(col)
    assert got.flags.owndata and got.base is None
    old = _old_dense_to_2d(col)
    assert not old.flags.owndata  # the property this change exists to remove


def test_dense_to_2d_int_list_widens_like_old_path():
    col = pa.chunked_array([pa.array([[1, 2], [3, 4]], pa.list_(pa.int32()))])
    got = dense_to_2d(col)
    assert got.dtype == np.float32
    assert got.tobytes() == _old_dense_to_2d(col).tobytes()


def test_dense_to_2d_empty():
    col = pa.chunked_array([], type=pa.list_(pa.float32()))
    got = dense_to_2d(col)
    assert got.shape == (0, 0) and got.dtype == np.float32


def test_dense_to_2d_rejects_null_rows():
    col = pa.chunked_array([pa.array([[1.0, 2.0], None], pa.list_(pa.float32(), 2))])
    with pytest.raises(ValueError, match="null"):
        dense_to_2d(col)


def test_dense_to_2d_rejects_ragged_lists_the_old_path_misread():
    """[1] + [2,3,4] has 4 values over 2 rows: the old path silently reshaped
    it into two width-2 rows that straddle the documents."""
    col = pa.chunked_array([pa.array([[1.0], [2.0, 3.0, 4.0]], pa.list_(pa.float32()))])
    assert _old_dense_to_2d(col).shape == (2, 2)  # the silent misread
    with pytest.raises(ValueError, match="ragged"):
        dense_to_2d(col)


def test_dense_to_2d_rejects_width_change_across_chunks():
    col = pa.chunked_array([pa.array([[1.0, 2.0]], pa.list_(pa.float32())),
                            pa.array([[1.0, 2.0, 3.0]], pa.list_(pa.float32()))])
    with pytest.raises(ValueError, match="ragged"):
        dense_to_2d(col)


# --- ranged path --------------------------------------------------------------

@pytest.fixture
def big_file(tmp_path):
    """A file with incompressible bytes so many ranges each carry real data."""
    rng = np.random.default_rng(1)
    table = pa.table({
        "id": pa.array(range(4096)),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(rng.standard_normal(4096 * 32).astype(np.float32)), 32),
    })
    store = Store(str(tmp_path))
    store.write("c.parquet", table)
    return store.list_parquets()[0].read_path


@pytest.mark.parametrize("width", [1, 3, 64])
def test_ranged_download_is_byte_exact(big_file, monkeypatch, width):
    monkeypatch.setattr(io_mod, "_RANGED_GET_BYTES", 4099)  # ragged last range
    store = Store(big_file, ranged_get=True, ranged_get_concurrency=width)
    size = store.fs.get_file_info(big_file).size
    assert size > 10 * 4099
    got = store._ranged_download(big_file, size)
    with open(big_file, "rb") as fh:
        assert got.to_pybytes() == fh.read()


def test_ranged_download_honours_per_file_concurrency(big_file, monkeypatch):
    """The shared pool must not lift a file's own bound (merge sizes it)."""
    monkeypatch.setattr(io_mod, "_RANGED_GET_BYTES", 2048)
    store = Store(big_file, ranged_get=True, ranged_get_concurrency=2)
    size = store.fs.get_file_info(big_file).size
    live, peak, lock = [0], [0], threading.Lock()
    real = io_mod._copy_range

    def slow_copy(dst, buf):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.002)
        real(dst, buf)
        with lock:
            live[0] -= 1

    monkeypatch.setattr(io_mod, "_copy_range", slow_copy)
    store._ranged_download(big_file, size)
    assert 1 <= peak[0] <= 2


def test_ranged_download_reuses_one_pool(big_file, monkeypatch):
    monkeypatch.setattr(io_mod, "_RANGED_GET_BYTES", 8192)
    store = Store(big_file, ranged_get=True)
    size = store.fs.get_file_info(big_file).size
    store._ranged_download(big_file, size)
    pool = io_mod._fetch_pool()
    store._ranged_download(big_file, size)
    assert io_mod._fetch_pool() is pool


def test_ranged_download_failure_raises_after_inflight_ranges(big_file, monkeypatch):
    monkeypatch.setattr(io_mod, "_RANGED_GET_BYTES", 2048)
    store = Store(big_file, ranged_get=True, ranged_get_concurrency=4)
    size = store.fs.get_file_info(big_file).size
    calls = [0]
    real = io_mod._copy_range

    def flaky(dst, buf):
        calls[0] += 1
        if calls[0] == 3:
            raise OSError("range failed")
        real(dst, buf)

    monkeypatch.setattr(io_mod, "_copy_range", flaky)
    with pytest.raises(OSError, match="range failed"):
        store._ranged_download(big_file, size)


def test_ranged_download_failure_in_the_last_window_raises(big_file, monkeypatch):
    """With the window wider than the file, every range is drained by the
    FINAL result loop, not the in-loop one; a failure there must still raise."""
    monkeypatch.setattr(io_mod, "_RANGED_GET_BYTES", 2048)
    store = Store(big_file, ranged_get=True, ranged_get_concurrency=10**6)
    size = store.fs.get_file_info(big_file).size
    calls, lock = [0], threading.Lock()
    real = io_mod._copy_range

    def flaky(dst, buf):
        with lock:
            calls[0] += 1
            n = calls[0]
        if n == 3:
            raise OSError("range failed")
        real(dst, buf)

    monkeypatch.setattr(io_mod, "_copy_range", flaky)
    with pytest.raises(OSError, match="range failed"):
        store._ranged_download(big_file, size)


def test_ranged_download_drains_inflight_ranges_before_closing(big_file, monkeypatch):
    """A failed range must not let the file handle close under a sibling
    range that is still running."""
    monkeypatch.setattr(io_mod, "_RANGED_GET_BYTES", 2048)
    store = Store(big_file, ranged_get=True, ranged_get_concurrency=4)
    size = store.fs.get_file_info(big_file).size
    live, live_at_close, calls, lock = [0], [], [0], threading.Lock()
    real = io_mod._copy_range

    class Handle:
        def __init__(self, f):
            self.f = f

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            live_at_close.append(live[0])
            self.f.close()

        def read_at(self, n, off):
            return self.f.read_at(n, off)

    class FS:
        def __init__(self, fs):
            self.fs = fs

        def open_input_file(self, path):
            return Handle(self.fs.open_input_file(path))

    def flaky(dst, buf):
        with lock:
            calls[0] += 1
            n = calls[0]
            live[0] += 1
        try:
            if n == 1:
                time.sleep(0.3)  # the slow sibling
            elif n == 2:
                raise OSError("range failed")
            real(dst, buf)
        finally:
            with lock:
                live[0] -= 1

    store.fs = FS(store.fs)
    monkeypatch.setattr(io_mod, "_copy_range", flaky)
    with pytest.raises(OSError, match="range failed"):
        store._ranged_download(big_file, size)
    assert live_at_close == [0]


@pytest.mark.parametrize("n", [0, 1, 4097])
def test_copy_range_rejects_wrong_length(n):
    """A 1-byte buffer would broadcast across the range without the check."""
    dst = np.zeros(4096, dtype=np.uint8)
    with pytest.raises(OSError, match="expected 4096"):
        io_mod._copy_range(dst, b"\x07" * n)
    assert not dst.any()


@pytest.mark.parametrize("raw,want", [("", 512), ("64", 64), ("abc", 512), ("0", 512)])
def test_fetch_pool_size_env_is_tolerant(monkeypatch, raw, want):
    monkeypatch.setenv("NOVA_BF_FETCH_POOL_SIZE", raw)
    assert io_mod._fetch_pool_size() == want


def _max_tick_gap(copy) -> tuple[float, float]:
    """Run `copy()` on this thread while a pure-Python ticker runs on another;
    return (copy seconds, longest gap between ticks during the copy). A copy
    that holds the GIL starves the ticker for its whole duration."""
    stamps: list[float] = []
    stop = threading.Event()

    def tick():
        while not stop.is_set():
            stamps.append(time.perf_counter())

    old = sys.getswitchinterval()
    sys.setswitchinterval(0.0005)
    th = threading.Thread(target=tick)
    th.start()
    try:
        time.sleep(0.02)
        t0 = time.perf_counter()
        copy()
        t1 = time.perf_counter()
        time.sleep(0.02)
    finally:
        stop.set()
        th.join()
        sys.setswitchinterval(old)
    inside = [s for s in stamps if t0 <= s <= t1]
    pts = [t0, *inside, t1]
    return t1 - t0, max(b - a for a, b in zip(pts, pts[1:]))


def test_copy_range_releases_the_gil():
    """GIL probe. The control (the old memoryview slice-assign) must starve the
    ticker, or this machine cannot tell the two apart and the test skips."""
    n = 256 << 20
    src = np.ones(n, dtype=np.uint8).tobytes()  # `read_at` returns bytes
    dst = np.empty(n, dtype=np.uint8)

    def old_copy():
        memoryview(dst)[:] = src

    dur_old, gap_old = _max_tick_gap(old_copy)
    if dur_old < 0.02 or gap_old < 0.8 * dur_old:
        pytest.skip(f"GIL probe cannot resolve here (copy {dur_old:.3f}s, gap {gap_old:.3f}s)")
    dst[:] = 0
    dur_new, gap_new = _max_tick_gap(lambda: io_mod._copy_range(dst, src))
    assert dst.all()
    assert gap_new < 0.5 * dur_new, (dur_new, gap_new)


# --- Arrow-native read ---------------------------------------------------------

def test_arrow_native_read_pins_prebuffer_and_threads(big_file, monkeypatch):
    seen = {}
    real = pq.read_table

    def spy(source, **kw):
        seen.update(kw)
        return real(source, **kw)

    monkeypatch.setattr(io_mod.pq, "read_table", spy)
    store = Store(big_file)
    got = store.read_columns(big_file, ["vector"])
    assert seen["pre_buffer"] is True and seen["use_threads"] is True
    assert seen["filesystem"] is store.fs and seen["columns"] == ["vector"]
    assert got.column_names == ["vector"]


# --- MAX_BLOCK warning -----------------------------------------------------------

@pytest.fixture
def kernel_applies(monkeypatch):
    from nova_bf import topk_triton

    monkeypatch.setattr(topk_triton, "kernel_may_apply", lambda: True)


@pytest.mark.parametrize("vt", ["dense", "sparse"])
def test_batch_above_max_block_warns(caplog, kernel_applies, vt):
    from nova_bf import compute, topk_triton

    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        assert compute._resolve_vt_batch_size(topk_triton.MAX_BLOCK + 1, 10, vt) \
            == topk_triton.MAX_BLOCK + 1
    assert any("MAX_BLOCK" in r.getMessage() and r.levelno == logging.WARNING
               and f"{vt}_batch_size" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("vt", ["dense", "sparse"])
def test_unset_batch_is_info_not_warning(caplog, kernel_applies, vt):
    from nova_bf import compute

    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        compute._resolve_vt_batch_size(None, 10, vt)
    hits = [r for r in caplog.records if "MAX_BLOCK" in r.getMessage()]
    assert hits and all(r.levelno == logging.INFO for r in hits)


@pytest.mark.parametrize("bs,vt", [("max", "dense"), (10**6, "multivector")])
def test_no_max_block_message_otherwise(caplog, kernel_applies, bs, vt):
    from nova_bf import compute, topk_triton

    bs = topk_triton.MAX_BLOCK if bs == "max" else bs
    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        compute._resolve_vt_batch_size(bs, 10, vt)
    assert not any("MAX_BLOCK" in r.getMessage() for r in caplog.records)


def test_no_max_block_message_where_the_kernel_cannot_run(caplog, monkeypatch):
    from nova_bf import compute, topk_triton

    monkeypatch.setattr(topk_triton, "kernel_may_apply", lambda: False)
    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        compute._resolve_vt_batch_size(topk_triton.MAX_BLOCK + 1, 10, "dense")
        compute._resolve_vt_batch_size(None, 10, "dense")
    assert not any("MAX_BLOCK" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("device,k", [("cpu", 10), ("cuda", 8193)])
def test_no_max_block_message_on_cpu_device_or_huge_k(caplog, kernel_applies, device, k):
    """A CPU run device can't use the kernel; nor can k > MAX_BLOCK at any batch."""
    import torch

    from nova_bf import compute, topk_triton

    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        compute._resolve_vt_batch_size(topk_triton.MAX_BLOCK + 1, k, "dense",
                                       torch.device(device))
    assert not any("MAX_BLOCK" in r.getMessage() for r in caplog.records)


def test_kernel_may_apply_gates(monkeypatch):
    import torch

    from nova_bf import topk_triton

    if topk_triton._cutfill is None:
        pytest.skip("triton kernel not importable here")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("NOVA_BF_NO_TOPK_KERNEL", raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert topk_triton.kernel_may_apply()
    monkeypatch.setenv("NOVA_BF_NO_TOPK_KERNEL", "1")
    assert not topk_triton.kernel_may_apply()
    monkeypatch.delenv("NOVA_BF_NO_TOPK_KERNEL")
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    assert not topk_triton.kernel_may_apply()
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not topk_triton.kernel_may_apply()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_fetch_pool_survives_fork_while_its_lock_is_held(big_file, monkeypatch):
    """A fork while another thread holds the pool lock must not deadlock the
    child's first ranged read (the at-fork hook gives it a fresh lock/pool)."""
    monkeypatch.setattr(io_mod, "_RANGED_GET_BYTES", 4099)
    store = Store(big_file, ranged_get=True, ranged_get_concurrency=4)
    size = store.fs.get_file_info(big_file).size
    with open(big_file, "rb") as fh:
        want = fh.read()
    store._ranged_download(big_file, size)  # parent pool exists
    held, release = threading.Event(), threading.Event()

    def hold():
        with io_mod._FETCH_POOL_LOCK:
            held.set()
            release.wait(10)

    th = threading.Thread(target=hold)
    th.start()
    held.wait(5)
    try:
        pid = os.fork()
        if pid == 0:  # child: exit code says whether the read worked
            code = 1
            try:
                signal.alarm(10)
                code = 0 if store._ranged_download(big_file, size).to_pybytes() == want else 2
            finally:
                os._exit(code)
    finally:
        release.set()
        th.join()
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
