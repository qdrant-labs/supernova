"""CUDA-only stress tests of the pooled, windowed device copy of `parts`
(`compute._resident_*`). Uploads are blocking copies on the current stream,
so a buffer or window may be refilled while earlier kernels reading it are
still queued; each test queues those reads behind a long `torch.cuda._sleep`
and checks every mask bit-exact against the PackedRowMask reference.
Skipped without CUDA.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from nova_bf import compute as C, filters as F  # noqa: E402

import test_combo_row_mask as T  # noqa: E402  sibling test module

DEV = "cuda"
SLEEP = 50_000_000      # cycles: ~25-50 ms on current GPUs


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setattr(F, "_COMBO_PARTS_GUARD", False)
    C._close_resident_pools()
    F.reset_combo_usage()
    yield
    torch.cuda.synchronize()
    C._close_resident_pools()


def _check(pair, trs, ctx):
    old, cm = pair
    return [(C._combo_mask_device(cm, tr, DEV), T._old_build(old, tr), f"{ctx} {tr}")
            for tr in trs]


def _verify(pending):
    torch.cuda.synchronize()
    for got, want, ctx in pending:
        assert (got is None) == (want is None), ctx
        if got is not None:
            assert torch.equal(got.cpu(), want), ctx


def _trs(n):
    return [slice(0, n), slice(n // 3, n), np.arange(0, n, 3, dtype=np.int64)]


def test_buffer_reuse_waits_for_previous_file_reads():
    """File N's masks are queued behind a long kernel; its buffer is released
    and refilled with file N+1's (different) parts right away. The refill must
    not land before file N's queued reads ran."""
    a = T._sized_mask(4000, seed=1)
    torch.cuda._sleep(SLEEP)
    pend = _check(a, _trs(4000), "file N")
    buf = a[1]._parts_dev[DEV].buf
    C._resident_release(a[1])
    b = T._sized_mask(4000, seed=2)
    assert not np.array_equal(a[1].parts, b[1].parts)
    pend += _check(b, _trs(4000), "file N+1")
    assert b[1]._parts_dev[DEV].buf is buf          # really the same buffer
    _verify(pend)
    assert F.combo_usage()["resident_reuses"] == 1


def test_window_reload_waits_for_queued_reads(monkeypatch):
    """One-column windows: every slice reloads the window buffer while the
    previous slices' combines are still queued behind a long kernel."""
    monkeypatch.setattr(C, "_RESIDENT_WINDOW_BYTES", 1)
    pair = T._sized_mask(3000, seed=10)
    torch.cuda._sleep(SLEEP)
    trs = [slice(r0, r0 + 40) for r0 in range(0, 2960, 40)]           # crossing windows
    trs += [np.arange(r0, r0 + 200, 7, dtype=np.int64) for r0 in range(2600, 0, -400)]
    pend = _check(pair, trs, "windowed")
    assert pair[1]._parts_dev[DEV].width == 1
    _verify(pend)
    assert F.combo_usage()["resident_uploads"] > len(trs)


def test_host_parts_changed_after_upload_is_not_seen():
    """The upload is complete (host side) when the build returns: later writes
    to the host `parts` must not reach the device copy."""
    old, cm = T._sized_mask(2000, seed=20)
    pend = _check((old, cm), [slice(0, 2000)], "before")
    keep = cm.parts.copy()
    cm.parts[...] = 0xA5
    pend += _check((old, cm), [slice(0, 2000)], "after scribble")   # same window
    _verify(pend)
    cm.parts[...] = keep


def test_close_with_live_masks():
    """`_close_resident_pools` while masks hold buffers: they stay valid and
    exact; releasing them afterwards pools them into a fresh pool."""
    pairs = [T._sized_mask(2500, seed=30 + i) for i in range(3)]
    torch.cuda._sleep(SLEEP)
    pend = [x for i, p in enumerate(pairs) for x in _check(p, _trs(2500), f"m{i}")]
    C._close_resident_pools()
    pend += [x for i, p in enumerate(pairs) for x in _check(p, _trs(2500), f"after close m{i}")]
    for _, cm in pairs:
        C._resident_release(cm)
    pend += _check(T._sized_mask(2500, seed=40), _trs(2500), "fresh pool")
    _verify(pend)
