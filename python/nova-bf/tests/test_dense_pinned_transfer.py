"""Dense pinned H2D overlap and its observable transfer contract."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nova_bf import compute
from nova_bf.compute import DenseCorpusBatch


def test_dense_transfer_mode_defaults_to_overlap_and_validates(monkeypatch):
    monkeypatch.delenv("NOVA_BF_PINNED", raising=False)
    assert compute._dense_transfer_mode() == 2
    for value, expected in (("0", 0), ("2", 2)):
        monkeypatch.setenv("NOVA_BF_PINNED", value)
        assert compute._dense_transfer_mode() == expected
    for value in ("on", "-1", "1", "3"):
        monkeypatch.setenv("NOVA_BF_PINNED", value)
        with pytest.raises(ValueError, match="must be 0 .* or 2"):
            compute._dense_transfer_mode()


def test_prefetch_inline_switch_parses_and_validates(monkeypatch):
    monkeypatch.delenv("NOVA_BF_PREFETCH_INLINE", raising=False)
    assert compute._dense_prefetch_inline() is False
    for value in ("", "0", "false", "No", "off"):
        monkeypatch.setenv("NOVA_BF_PREFETCH_INLINE", value)
        assert compute._dense_prefetch_inline() is False
    for value in ("1", "true", "YES", " on "):
        monkeypatch.setenv("NOVA_BF_PREFETCH_INLINE", value)
        assert compute._dense_prefetch_inline() is True
    monkeypatch.setenv("NOVA_BF_PREFETCH_INLINE", "2")
    with pytest.raises(ValueError, match="NOVA_BF_PREFETCH_INLINE"):
        compute._dense_prefetch_inline()


def test_dense_prefetch_keeps_transfer_seam_and_issues_next_first(monkeypatch):
    """Every slice crosses `transfer`, and i+1 is issued before i is scored.

    The kill switch must restore exactly the previous inline call order.
    """
    monkeypatch.setenv("NOVA_BF_PREFETCH_INLINE", "1")
    main = threading.get_ident()
    calls = []
    stream = object()

    class Event:
        def record(self, got_stream):
            assert got_stream is stream

        def synchronize(self):
            pass

    class Slice:
        def __init__(self, index):
            self.index = index

        def wait_on(self, got_stream):
            assert got_stream is stream
            calls.append(("wait", self.index))

    class Batch:
        def prefetch(self, r0, r1, device):
            assert device == "cuda:7"
            assert threading.get_ident() == main, "inline staging left the thread"
            calls.append(("prefetch", r0 // 4, r0, r1))

        def transfer(self, r0, r1, device):
            assert device == "cuda:7"
            index = r0 // 4
            calls.append(("transfer", index, r0, r1))
            return Slice(index)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    ranges = [(0, 4), (4, 8), (8, 10)]
    compute._run_dense_prefetch(
        Batch(), ranges, "cuda:7",
        lambda r0, r1, sl: calls.append(("score", sl.index, r0, r1)),
    )
    assert calls == [
        ("prefetch", 0, 0, 4),
        ("prefetch", 1, 4, 8),
        ("transfer", 0, 0, 4),
        ("wait", 0),
        ("score", 0, 0, 4),
        ("prefetch", 2, 8, 10),
        ("transfer", 1, 4, 8),
        ("wait", 1),
        ("score", 1, 4, 8),
        ("transfer", 2, 8, 10),
        ("wait", 2),
        ("score", 2, 8, 10),
    ]


@pytest.fixture(autouse=True)
def _join_staging_helper():
    """The staging helper is run-scoped; direct callers here must join it."""
    yield
    compute._close_dense_staging()


@pytest.fixture(autouse=True)
def _fresh_device_streams(monkeypatch):
    """The copy stream is cached per device for the process (see
    `compute._device_streams`); tests that fake `torch.cuda.Stream` must not
    leave their fakes in that cache."""
    monkeypatch.setattr(compute, "_DEVICE_STREAMS", {})


class _SplitBatch:
    """Fake batch exposing the split prefetch surface the scheduler uses."""

    def prefetch(self, r0, r1, device):
        self.stage_prefetch(self.prepare_prefetch(r0, r1, device))

    def prepare_prefetch(self, r0, r1, device):
        return (r0, r1)

    def stage_prefetch(self, plan):
        pass


@pytest.mark.parametrize("inline", ["1", "0"])
def test_dense_prefetch_bounds_delayed_compute_across_batches(monkeypatch, inline):
    """Fast DMA cannot leave an entire batch's device tensors awaiting compute.

    Model a GPU that makes no compute progress until the host waits on a
    recorded event. DMA is instant, so waiting on DMA alone cannot pass.
    Device allocation happens in `prepare_prefetch` on the feeding thread in
    both modes, so the lookahead peak is deterministic and must be reached in
    helper mode too.
    """
    monkeypatch.setenv("NOVA_BF_PREFETCH_INLINE", inline)
    submitted = completed = allocated = peak = 0
    staged = 0
    stream = object()
    main = threading.get_ident()

    class Event:
        def record(self, got_stream):
            assert got_stream is stream
            self.through = submitted

        def synchronize(self):
            nonlocal completed
            completed = max(completed, self.through)

    class Slice:
        def wait_on(self, got_stream):
            assert got_stream is stream  # device-side wait makes no host progress

    class Batch(_SplitBatch):
        def prepare_prefetch(self, r0, r1, device):
            nonlocal allocated, peak
            assert threading.get_ident() == main, "device allocation left the thread"
            allocated += 1
            peak = max(peak, allocated - completed)
            assert allocated - completed <= 3, "unbounded device allocation backlog"
            return (r0, r1)

        def stage_prefetch(self, plan):
            nonlocal staged
            staged += 1

        def transfer(self, r0, r1, device):
            return Slice()

    def score(*args):
        nonlocal submitted
        submitted += 1
        assert submitted - completed <= 2, "compute submission is not throttled"

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    # Repeated batches also catch a final event left undrained on each return.
    for count in (128, 1, 37):
        compute._run_dense_prefetch(
            Batch(), [(i, i + 1) for i in range(count)], "cuda", score,
        )
        assert submitted == completed == allocated == staged
    assert peak == 3, "the test must exercise concurrent compute and lookahead"


def _stage_threads():
    return [t for t in threading.enumerate() if t.name.startswith("nova-bf-stage")]


def _fake_cuda(monkeypatch):
    stream = object()

    class Event:
        def record(self, got_stream):
            assert got_stream is stream

        def synchronize(self):
            pass

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    return stream


class _Slice:
    def __init__(self, index):
        self.index = index

    def wait_on(self, stream):
        pass


def test_dense_prefetch_stages_next_slice_on_helper_overlapping_score(monkeypatch):
    """Default mode: allocation for i+1 happens on the feeding thread BEFORE
    slice i is scored; staging of i+1 runs off-thread WHILE slice i is scored
    and finishes before slice i+1 is transferred."""
    monkeypatch.delenv("NOVA_BF_PREFETCH_INLINE", raising=False)
    _fake_cuda(monkeypatch)
    main = threading.get_ident()
    n = 6
    started = [threading.Event() for _ in range(n)]
    finished = [threading.Event() for _ in range(n)]
    log = []
    lock = threading.Lock()

    def note(*item):
        with lock:
            log.append(item)

    class Batch(_SplitBatch):
        def prepare_prefetch(self, r0, r1, device):
            note("prepare", r0, threading.get_ident() == main)
            return (r0, r1)

        def stage_prefetch(self, plan):
            r0 = plan[0]
            note("stage_start", r0, threading.get_ident() == main)
            started[r0].set()
            time.sleep(0.01)
            note("stage_end", r0)
            finished[r0].set()

        def transfer(self, r0, r1, device):
            assert finished[r0].is_set(), f"slice {r0} transferred before staged"
            note("transfer", r0)
            return _Slice(r0)

    def score(r0, r1, sl):
        note("score_start", r0)
        if r0 + 1 < n:
            # Deterministic overlap: the next staging must start while this
            # slice is still being scored. Inline staging would never get here.
            assert started[r0 + 1].wait(5), "next slice was not staged concurrently"
        note("score_end", r0)

    compute._run_dense_prefetch(
        Batch(), [(i, i + 1) for i in range(n)], "cuda", score
    )
    thread_of = {(kind, r0): rest[0] for (kind, r0, *rest) in log if rest}
    assert all(thread_of[("prepare", i)] for i in range(n)), (
        "device allocation must stay on the feeding thread")
    assert thread_of[("stage_start", 0)] is True, (
        "slice 0 has nothing to overlap; stage it inline")
    assert all(not thread_of[("stage_start", i)] for i in range(1, n)), (
        "staging stayed on the feeding thread")
    pos = {(kind, r0): i for i, (kind, r0, *_) in enumerate(log)}
    for i in range(n):
        assert pos[("prepare", i)] < pos[("stage_start", i)]
        assert pos[("stage_end", i)] < pos[("transfer", i)]
        if i + 1 < n:
            # Allocation order matches inline: before slice i's temporaries.
            assert pos[("prepare", i + 1)] < pos[("score_start", i)]
            # Serial staging: one staging in flight at a time.
            assert pos[("stage_end", i)] < pos[("stage_start", i + 1)]
            assert pos[("stage_start", i + 1)] < pos[("score_end", i)]


def test_staging_helper_is_reused_across_calls_and_joined_at_close(monkeypatch):
    """One helper per run, not per file: its thread (and OpenMP team) persists
    across batches until `_close_dense_staging` joins it."""
    monkeypatch.delenv("NOVA_BF_PREFETCH_INLINE", raising=False)
    _fake_cuda(monkeypatch)
    idents = []

    class Batch(_SplitBatch):
        def stage_prefetch(self, plan):
            if plan[0] > 0:
                idents.append(threading.get_ident())

        def transfer(self, r0, r1, device):
            return _Slice(r0)

    for _ in range(3):
        compute._run_dense_prefetch(
            Batch(), [(i, i + 1) for i in range(4)], "cuda", lambda *a: None
        )
    assert len(idents) == 9 and len(set(idents)) == 1
    assert len(_stage_threads()) == 1
    compute._close_dense_staging()
    assert not _stage_threads()
    compute._close_dense_staging()  # idempotent


@pytest.mark.parametrize("fail_release", [False, True])
def test_run_compute_joins_the_staging_helper(tmp_path, monkeypatch, fail_release):
    """`run_compute`'s finally joins the run-scoped helper, on failure too."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from nova_bf import twopass
    from nova_bf.compute import run_compute
    from nova_bf.config import load_config

    rng = np.random.default_rng(0)
    for d in ("corpus", "queries", "out"):
        (tmp_path / d).mkdir()
    for name, n in (("corpus/c0.parquet", 16), ("queries/q.parquet", 3)):
        v = rng.standard_normal((n, 4)).astype(np.float32)
        pq.write_table(pa.table({
            "id": pa.array([f"r{i}" for i in range(n)]),
            "emb": pa.array(list(v.tolist()), type=pa.list_(pa.float32(), 4)),
        }), tmp_path / name)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"""
corpus: {{path: {tmp_path / 'corpus'}, dense_column: emb, id_column: id}}
queries: {{path: {tmp_path / 'queries'}, dense_column: emb, id_column: id}}
output: {{path: {tmp_path / 'out'}}}
params: {{io_workers: 1}}
searches:
  - {{name: d, vector_type: dense, metric: cosine, k: 2}}
""")
    real_release = twopass.release

    def release():
        # Stand in for a scan that used the helper: it exists at teardown.
        compute._dense_staging_pool().submit(lambda: None).result()
        assert _stage_threads()
        real_release()
        if fail_release:
            raise RuntimeError("injected release failure")

    monkeypatch.setattr(twopass, "release", release)
    if fail_release:
        with pytest.raises(RuntimeError, match="injected release failure"):
            run_compute(load_config(str(cfg_path)))
    else:
        run_compute(load_config(str(cfg_path)))
    assert compute._DENSE_STAGING["pool"] is None
    assert not _stage_threads()


@pytest.mark.parametrize("exc_type", [RuntimeError, KeyboardInterrupt])
def test_dense_prefetch_staging_error_surfaces_on_feeding_thread(
    monkeypatch, exc_type
):
    monkeypatch.delenv("NOVA_BF_PREFETCH_INLINE", raising=False)
    _fake_cuda(monkeypatch)
    scored = []

    class Batch(_SplitBatch):
        def stage_prefetch(self, plan):
            if plan[0] == 2:
                raise exc_type("injected staging failure")

        def transfer(self, r0, r1, device):
            return _Slice(r0)

    with pytest.raises(exc_type, match="injected staging failure"):
        compute._run_dense_prefetch(
            Batch(), [(i, i + 1) for i in range(5)], "cuda",
            lambda r0, r1, sl: scored.append(r0),
        )
    # Slice 2 was never staged, so it must never be scored.
    assert scored == [0, 1]


@pytest.mark.parametrize("exc_type", [RuntimeError, KeyboardInterrupt])
def test_dense_prefetch_score_error_joins_inflight_staging(monkeypatch, exc_type):
    """An error while the helper is mid-copy waits for that copy before
    returning and keeps the ORIGINAL exception."""
    monkeypatch.delenv("NOVA_BF_PREFETCH_INLINE", raising=False)
    _fake_cuda(monkeypatch)
    in_copy = threading.Event()
    done = []

    class Batch(_SplitBatch):
        def stage_prefetch(self, plan):
            if plan[0] == 2:
                in_copy.set()
                time.sleep(0.05)
                done.append(plan[0])
                raise ValueError("staging error must not mask the score error")

        def transfer(self, r0, r1, device):
            return _Slice(r0)

    def score(r0, r1, sl):
        if r0 == 1:
            assert in_copy.wait(5)
            raise exc_type("injected score failure")

    with pytest.raises(exc_type, match="injected score failure"):
        compute._run_dense_prefetch(
            Batch(), [(i, i + 1) for i in range(5)], "cuda", score
        )
    # The helper is run-scoped and still alive, so only the in-call wait can
    # make this hold at return.
    assert done == [2], "returned before the in-flight staging finished"


def test_unwind_does_not_swallow_a_second_keyboard_interrupt(monkeypatch):
    """`finally` logs a staging Exception but must let a BaseException out."""
    monkeypatch.delenv("NOVA_BF_PREFETCH_INLINE", raising=False)
    _fake_cuda(monkeypatch)

    class Future:
        def result(self):
            raise KeyboardInterrupt("second interrupt")

    class Pool:
        def submit(self, fn, plan):
            return Future()

    class Batch(_SplitBatch):
        def transfer(self, r0, r1, device):
            return _Slice(r0)

    def score(r0, r1, sl):
        raise RuntimeError("first failure")

    monkeypatch.setattr(compute, "_dense_staging_pool", lambda: Pool())
    with pytest.raises(KeyboardInterrupt, match="second interrupt"):
        compute._run_dense_prefetch(
            Batch(), [(0, 1), (1, 2)], "cuda", score
        )


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_helper_staging_gates_ring_reuse_and_allocates_on_feeding_thread(
    monkeypatch, dtype
):
    """Real `DenseCorpusBatch` staging with CUDA objects faked on CPU.

    Every device allocation (H2D destination and fp32 widen target) happens
    on the feeding thread before the previous slice is scored. A slot is only
    refilled after its previous DMA event is synchronized. Each scored slice
    holds its own rows, widened to float32.
    """
    monkeypatch.delenv("NOVA_BF_PREFETCH_INLINE", raising=False)
    import contextlib

    compute_stream = object()
    main = threading.get_ident()

    class Event:
        def __init__(self):
            self.synced = False

        def record(self, got_stream):
            pass

        def synchronize(self):
            self.synced = True

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: compute_stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: "copy-stream")
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    devices = []
    monkeypatch.setattr(
        torch.cuda, "device",
        lambda d: devices.append(d) or contextlib.nullcontext(),
    )
    # CPU stand-in for `slice.wait_on`: real record_stream needs a CUDA stream.
    monkeypatch.setattr(compute.DenseBatchSlice, "wait_on", lambda self, s: None)

    log = []
    real_empty = torch.empty

    def empty(*args, **kwargs):
        if "device" in kwargs:
            log.append(("alloc", threading.get_ident() == main))
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)

    arr = (np.arange(11 * 3).reshape(11, 3) / 4).astype(dtype)
    violations = []

    class Batch(DenseCorpusBatch):
        def prepare_prefetch(self, r0, r1, device):
            log.append(("prepare", r0))
            return super().prepare_prefetch(r0, r1, device)

        def stage_prefetch(self, plan):
            prev = self._ring_ready[self._ring_i % self._RING_SIZE]
            super().stage_prefetch(plan)
            if prev is not None and not prev.synced:
                violations.append((plan.r0, plan.r1))

    batch = Batch(arr)
    batch.configure_transfer(2, 2)
    # pin_memory needs CUDA.
    batch._ring = [real_empty((2, 3), dtype=torch.from_numpy(arr).dtype)
                   for _ in range(2)]
    batch._ring_ready = [None, None]
    ranges = [(i, min(i + 2, 11)) for i in range(0, 11, 2)]
    seen = []

    def score(r0, r1, sl):
        log.append(("score", r0))
        time.sleep(0.005)
        assert sl.Cb.dtype is torch.float32
        np.testing.assert_array_equal(sl.Cb.numpy(), arr[r0:r1].astype(np.float32))
        seen.append((r0, r1))

    compute._run_dense_prefetch(batch, ranges, "cpu", score)
    assert seen == ranges
    assert violations == [], "ring slot refilled before its previous DMA finished"
    assert batch._prefetched == {}
    allocs = [on_main for kind, on_main in
              (e for e in log if e[0] == "alloc")]
    per_slice = 1 if dtype == np.float32 else 2
    assert len(allocs) == per_slice * len(ranges)
    assert all(allocs), "device memory was allocated off the feeding thread"
    order = [e for e in log if e[0] in ("prepare", "score")]
    for i in range(len(ranges) - 1):
        assert order.index(("prepare", ranges[i + 1][0])) < order.index(
            ("score", ranges[i][0]))
    assert all(isinstance(d, torch.device) for d in devices)


def test_prepare_resolves_an_unindexed_cuda_device_on_the_calling_thread(
    monkeypatch,
):
    """The helper must get an explicit index: it does not share the feeding
    thread's current-device context."""
    import contextlib

    seen = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: seen.append(device))
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(
        torch.cuda, "device", lambda d: seen.append(d) or contextlib.nullcontext()
    )
    real_empty = torch.empty
    monkeypatch.setattr(
        torch, "empty",
        lambda *a, **k: real_empty(*a, **{x: y for x, y in k.items()
                                          if x not in ("device", "pin_memory")}),
    )
    batch = DenseCorpusBatch(np.zeros((4, 2), dtype=np.float32))
    batch.configure_transfer(4, 2)
    plan = batch.prepare_prefetch(0, 4, "cuda")
    assert plan.device == torch.device("cuda", 3)
    assert seen and all(d == torch.device("cuda", 3) for d in seen)


def test_every_batch_stages_on_the_one_per_device_copy_stream(monkeypatch):
    """Not a fresh pooled stream per batch (per file): round-robin pooling
    would sooner or later hand one out that aliases the pass-one side stream
    (see `compute._device_streams`)."""
    import contextlib

    class Stream:
        n = 0

        def __init__(self, device=None):
            Stream.n += 1
            self.cuda_stream = Stream.n

    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "device", lambda d: contextlib.nullcontext())
    real_empty = torch.empty
    monkeypatch.setattr(
        torch, "empty",
        lambda *a, **k: real_empty(*a, **{x: y for x, y in k.items()
                                          if x not in ("device", "pin_memory")}),
    )
    got = []
    for _ in range(5):
        batch = DenseCorpusBatch(np.zeros((4, 2), dtype=np.float32))
        batch.configure_transfer(4, 2)
        batch.prepare_prefetch(0, 4, "cuda:0")
        got.append(batch._copy_stream)
    assert all(s is compute._dense_copy_stream("cuda:0") for s in got)
    assert got[0] is not compute._pass_one_stream("cuda:0")


def test_transfer_configuration_uses_resolved_step_not_survivor_count():
    batch = DenseCorpusBatch(np.zeros((3, 5), dtype=np.float16))
    batch.configure_transfer(4096, 2)
    assert batch._pinned_rows == 4096
    assert batch._transfer_mode == 2


def test_copy_stream_slice_waits_before_recording_compute_ownership():
    calls = []

    class Ready:
        pass

    class Tensor:
        shape = (4, 3)

        def record_stream(self, stream):
            calls.append(("record", stream))

    class Stream:
        def wait_event(self, ready):
            calls.append(("wait", ready))

    ready, stream = Ready(), Stream()
    sl = compute.DenseBatchSlice(Tensor(), ready=ready)
    sl.wait_on(stream)
    assert calls == [("wait", ready), ("record", stream)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("inline", ["0", "1"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_pinned_overlap_is_bitwise_identical_across_ring_wraps(
    monkeypatch, dtype, inline
):
    """Exercise a ragged tail and repeated ring reuse with real CUDA streams,
    with helper-thread staging (default) and inline staging (kill switch)."""
    monkeypatch.setenv("NOVA_BF_PREFETCH_INLINE", inline)
    from nova_bf.config import SearchSpec
    from nova_bf.tiebreak import sentinel_key

    rng = np.random.default_rng(7)
    corpus = np.ascontiguousarray(rng.standard_normal((35, 12)).astype(dtype))
    queries = np.ascontiguousarray(rng.standard_normal((7, 12)).astype(np.float32))
    spec = SearchSpec(name="dense", vector_type="dense", metric="cosine", k=5)

    # Keep this test about transfer correctness. Kernel and two-pass parity have
    # their own suites, and allowing either here would make a failure ambiguous.
    monkeypatch.setenv("NOVA_BF_NO_TOPK_KERNEL", "1")
    monkeypatch.setenv("NOVA_BF_NO_FOLD_KERNEL", "1")

    def run(mode):
        monkeypatch.setenv("NOVA_BF_PINNED", str(mode))
        q = torch.tensor(queries, dtype=torch.float32, device="cuda")
        qn = q.norm(dim=1).clamp_min(1e-12)
        top_key = [sentinel_key((len(queries), spec.k), "cuda")]
        top_enc = [torch.zeros(
            (len(queries), spec.k), dtype=torch.int64, device="cuda"
        )]
        threshold = [sentinel_key((len(queries),), "cuda")]
        batch = DenseCorpusBatch(corpus)
        compute._process_batch_group(
            batch, [0], [spec], [q], [qn], top_key, top_enc, threshold,
            4, 0, "cuda", None,
            lambda _m, rows, _true_rows, _cache: (rows, None, None),
            [None], [None], [qn.contiguous()], two_pass=False,
        )
        torch.cuda.synchronize()
        return (
            top_key[0].cpu().numpy().copy(),
            top_enc[0].cpu().numpy().copy(),
            batch,
        )

    ref_key, ref_enc, _ = run(0)
    got_key, got_enc, batch = run(2)
    np.testing.assert_array_equal(got_key, ref_key)
    np.testing.assert_array_equal(got_enc, ref_enc)
    assert len(batch._ring) == 2
    assert {slot.shape[0] for slot in batch._ring} == {4}
