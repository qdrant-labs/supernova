"""Cross-worker threshold exchange for safe top-k pruning.

Workers exchange their top-m scores per query to compute a tighter lower
bound on the global k-th score, where m = min(k, ceil(2k/N)).
Scores are published as float16 rounded toward -inf to preserve safety.
Publishing and polling run asynchronously at file boundaries.

Peer files use raw PyArrow I/O and zero-copy NumPy views to minimize GIL
contention. Sequence numbers track updates, while GPU kernels sanitize
scores. Background threads fetch changed files concurrently.

Safety guarantees:
- Run-specific IDs prevent thresholds from incompatible runs being reused.
- Missing or invalid peer data only reduces pruning effectiveness.
- The merge verifies applied thresholds against final top-k results.
- Ties remain eligible, and non-finite scores cannot raise thresholds.
- External thresholds never modify local top-k state.
- Network failures fall back to local pruning without blocking the scan.

Cross-worker pruning may leave partial results shorter than k, but the
merged global top-k remains exact when threshold verification passes.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import math
import os
import queue
import re
import struct
import threading
import time
import weakref

import numpy as np
import pyarrow.fs as pafs

from nova_bf.io import Store

logger = logging.getLogger(__name__)

SUBDIR = "_thresholds"
_FILE_RE = re.compile(r"thr_rank(\d+)\.thr$")
# How many empty polls before warning that no peer has been seen.
_NO_PEER_WARN_POLLS = 3


def file_name(rank: int) -> str:
    return f"thr_rank{rank:05d}.thr"

# File format: one file per rank, atomically replaced on each publish.
# Little-endian header stores version, search count, sequence number,
# data offset, and total length. Each search entry specifies its name,
# shape, and offset to an aligned float16 array.
# Readers validate metadata, shapes, and file length to reject invalid
# or incomplete files. Code hashes isolate incompatible formats.

MAGIC = b"NOVABFTX"
FORMAT_VERSION = 1
ALIGN = 64
_FIXED = struct.Struct("<8sIIqQQ")
_ENTRY = struct.Struct("<QQQ")
_NAME_LEN = struct.Struct("<H")
_MAX_SEARCHES = 1 << 16


def _align(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def encode_header(seq: int, arrays: dict[str, np.ndarray]) -> tuple[bytes, list[tuple[int, np.ndarray]]]:
    """Build an aligned file header and return payload offsets.

    Requires C-contiguous 2-D float16 arrays; payloads remain zero-copy.
    """
    names = list(arrays)
    encoded = [n.encode() for n in names]
    hdr_len = _FIXED.size + sum(_NAME_LEN.size + len(e) + _ENTRY.size for e in encoded)
    data_start = _align(hdr_len)
    off, parts, entries = data_start, [], []
    total = data_start
    for n, e in zip(names, encoded):
        a = arrays[n]
        if a.dtype != np.float16 or a.ndim != 2 or not a.flags.c_contiguous:
            raise ValueError(f"{n!r}: expected a C-contiguous 2-D float16 array")
        entries.append(_NAME_LEN.pack(len(e)) + e + _ENTRY.pack(a.shape[0], a.shape[1], off))
        parts.append((off, a))
        total = off + a.nbytes
        off = _align(total)
    hdr = _FIXED.pack(MAGIC, FORMAT_VERSION, len(names), int(seq), data_start, total) + b"".join(entries)
    return hdr + bytes(data_start - len(hdr)), parts


def write_stream(sink, header: bytes, parts: list[tuple[int, np.ndarray]]) -> None:
    """Write the header and aligned arrays directly without payload copies."""
    sink.write(header)
    pos = len(header)
    for off, a in parts:
        if off > pos:
            sink.write(bytes(off - pos))
        if a.nbytes:
            sink.write(a)
        pos = off + a.nbytes


def encode(seq: int, arrays: dict[str, np.ndarray]) -> bytes:
    """Serialize the complete file for testing; publishing uses streaming.

    Like `publish`: float16 arrays are written as given, anything else goes
    through `fp16_floor`, so a score is never rounded up into an unsafe bound.
    """
    header, parts = encode_header(seq, {
        n: (np.ascontiguousarray(a) if np.asarray(a).dtype == np.float16
            else fp16_floor(np.asarray(a)))
        for n, a in arrays.items()})
    out = bytearray(header)
    for off, a in parts:
        out += bytes(off - len(out))
        out += a.tobytes()
    return bytes(out)


def decode(buf, sizes: dict[str, tuple[int, int]]) -> tuple[int, dict[str, np.ndarray]]:
    """Decode a threshold file into its sequence number and zero-copy arrays.

    Validates the header, search names, shapes, and payload bounds.
    Raises ValueError for invalid or incompatible files.
    """
    mv = memoryview(buf).cast("B")
    size = len(mv)
    if size < _FIXED.size:
        raise ValueError(f"file too short ({size} bytes)")
    magic, version, n, seq, data_start, total = _FIXED.unpack_from(mv, 0)
    if magic != MAGIC:
        raise ValueError("bad magic (not a threshold file)")
    if version != FORMAT_VERSION:
        raise ValueError(f"format version {version}, expected {FORMAT_VERSION}")
    if total != size:
        raise ValueError(f"file is {size} bytes, header says {total} (truncated?)")
    if not _FIXED.size <= data_start <= size:
        raise ValueError(f"data_start {data_start} out of range")
    if n > _MAX_SEARCHES or n != len(sizes):
        raise ValueError(f"{n} searches, expected {len(sizes)}")
    pos = _FIXED.size
    end = data_start          # arrays must be in order and must not overlap
    out: dict[str, np.ndarray] = {}
    for _ in range(n):
        if pos + _NAME_LEN.size > data_start:
            raise ValueError("header entries overrun data_start")
        (ln,) = _NAME_LEN.unpack_from(mv, pos)
        pos += _NAME_LEN.size
        if pos + ln + _ENTRY.size > data_start:
            raise ValueError("header entries overrun data_start")
        name = bytes(mv[pos: pos + ln]).decode("utf-8")
        pos += ln
        h, m, off = _ENTRY.unpack_from(mv, pos)
        pos += _ENTRY.size
        if name not in sizes or name in out:
            raise ValueError(f"unexpected or duplicate search {name!r}")
        if (h, m) != sizes[name]:
            raise ValueError(f"{name}: shape {(h, m)}, expected {sizes[name]}")
        nbytes = h * m * 2
        if off % 2 or off < end or off + nbytes > size:
            raise ValueError(f"{name}: payload [{off}, {off + nbytes}) out of range "
                             "or overlapping the previous array")
        end = off + nbytes
        out[name] = (np.frombuffer(buf, dtype=np.float16, count=h * m, offset=off).reshape(h, m)
                     if nbytes else np.empty((h, m), np.float16))
    return int(seq), out



def hash_array(h, arr: np.ndarray) -> None:
    """Feed an array's dtype, shape and bytes to a hashlib object without
    copying contiguous data. Zero-size arrays (a multivector query set with no
    tokens, a sparse query set with an empty vocabulary) contribute only their
    dtype and shape: `memoryview.cast` refuses a zero in the shape."""
    a = np.ascontiguousarray(arr)
    h.update(f"{a.dtype.str}{a.shape}".encode())
    if a.size:
        h.update(memoryview(a).cast("B"))


_CODE_HASH: str | None = None


def code_hash() -> str:
    """sha256 over this package's Python sources, part of the exchange run
    id: thresholds published by other code (an earlier kernel, a different
    bound) are never folded. Computed once per process."""
    global _CODE_HASH
    if _CODE_HASH is None:
        import hashlib
        h = hashlib.sha256()
        pkg = os.path.dirname(os.path.abspath(__file__))
        for name in sorted(os.listdir(pkg)):
            if name.endswith(".py"):
                with open(os.path.join(pkg, name), "rb") as f:
                    h.update(name.encode()); h.update(f.read())
        _CODE_HASH = h.hexdigest()
    return _CODE_HASH


def subdir_for(run_id: str) -> str:
    return f"{SUBDIR}/{run_id[:16]}"


PUBLISH_FACTOR = 2


def publish_width(k: int, num_jobs: int) -> int:
    """Scores per query each worker publishes: enough that N lists almost
    surely contain the whole global top-k (expected share k/N, published 2k/N),
    never more than k. The total exchanged per query is ~2k whatever N is."""
    return int(max(1, min(k, -(-PUBLISH_FACTOR * k // num_jobs))))


def fp16_floor(a: np.ndarray) -> np.ndarray:
    """float32 -> float16 rounded TOWARD -inf (so a published score never
    exceeds the real one: the bound stays a bound). Non-finite -> -inf.
    Magnitudes beyond float16 saturate: +big -> 65504, -big -> -inf."""
    a = np.where(np.isfinite(a), a, -np.inf).astype(np.float32, copy=False)
    x = a.astype(np.float16)
    up = x.astype(np.float32) > a
    if up.any():
        x[up] = np.nextafter(x[up], np.float16(-np.inf))
    return x


# Track exchange I/O to prevent PyArrow shutdown crashes from daemon threads.
# At exit, stop new I/O and wait up to _EXIT_DRAIN_S for active operations.
# Register after PyArrow so cleanup runs before its S3 shutdown handler.
_IO = {"n": 0, "closing": False}
_IO_CV = threading.Condition()
_LIVE: "weakref.WeakSet[ThresholdExchange]" = weakref.WeakSet()
_EXIT_DRAIN_S = 5.0


def _io_enter() -> bool:
    with _IO_CV:
        if _IO["closing"]:
            return False
        _IO["n"] += 1
        return True


def _io_exit() -> None:
    with _IO_CV:
        _IO["n"] -= 1
        _IO_CV.notify_all()


def _drain_at_exit(timeout_s: float = _EXIT_DRAIN_S) -> None:
    with _IO_CV:
        _IO["closing"] = True
    for ex in list(_LIVE):
        ex._stop.set()
        # Never block exit on the lock (it is only ever held briefly).
        if ex._cv.acquire(timeout=0.5):
            try:
                ex._cv.notify_all()
            finally:
                ex._cv.release()
    with _IO_CV:
        _IO_CV.wait_for(lambda: _IO["n"] == 0, timeout=timeout_s)


def _reset_io_in_child() -> None:
    global _IO_CV
    _IO_CV = threading.Condition()
    _IO["n"] = 0


os.register_at_fork(after_in_child=_reset_io_in_child)
atexit.register(_drain_at_exit)


def _host_tensor(a: np.ndarray):
    """Create a zero-copy Torch view, copying only read-only arrays."""
    import torch
    if not a.flags.writeable:
        a = np.array(a)
    return torch.from_numpy(a)


def union_kth(own, peers: list[np.ndarray], k: int, device, chunk: int = 8192):
    """Compute the per-query k-th largest score across local and peer results.

    Treats non-finite scores as -inf and processes rows in chunks
    to limit GPU memory usage.
    """
    import torch

    h = own.shape[0]
    out = torch.full((h,), float("-inf"), dtype=torch.float32, device=own.device)
    width = own.shape[1] + sum(int(p.shape[1]) for p in peers)
    if width < k:
        return out
    neg_inf = torch.tensor(float("-inf"), dtype=torch.float32, device=own.device)
    ninf = float("-inf")
    for r0 in range(0, h, chunk):
        r1 = min(h, r0 + chunk)
        parts = [torch.where(torch.isfinite(own[r0:r1]), own[r0:r1], neg_inf)]
        for p in peers:
            # Out of place: `.float()` is a no-op for a float32 peer on CPU,
            # and an in-place sanitise would then rewrite the caller's array.
            d = _host_tensor(p[r0:r1]).to(own.device).float()
            parts.append(torch.nan_to_num(d, nan=ninf, posinf=ninf, neginf=ninf))
        cat = torch.cat(parts, dim=1)
        out[r0:r1] = torch.topk(cat, k, dim=1, sorted=True).values[:, -1]
    return out
class _Task:
    """Queued fetch task that can be cancelled before execution."""
    __slots__ = ("arg", "out", "state")

    QUEUED, STARTED, CANCELLED = 0, 1, 2

    def __init__(self, arg, out: queue.SimpleQueue):
        self.arg = arg
        self.out = out
        self.state = _Task.QUEUED


class _FetchPool:
    """Daemon thread pool for cancellable fetch tasks.

    Workers return results or exceptions through per-caller queues.
    Pending tasks may be cancelled; active tasks are not interrupted.
    """

    def __init__(self, fn, size: int, name: str):
        self._fn = fn
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()  # Synchronize task state transitions.
        self._open = True
        self._threads: list[threading.Thread] = []
        try:
            for i in range(size):
                t = threading.Thread(target=self._work, name=f"{name}-{i}", daemon=True)
                t.start()
                self._threads.append(t)
        except BaseException:
            self.close()
            raise

    @property
    def is_open(self) -> bool:
        return self._open

    def _work(self) -> None:
        while True:
            task = self._q.get()
            if task is None:
                return
            with self._state_lock:
                if task.state != _Task.QUEUED:     # cancelled while queued
                    continue
                task.state = _Task.STARTED
            try:
                res = self._fn(task.arg)
            except BaseException as exc:  # noqa: BLE001 — never strand the caller
                res = ("get", exc)
            task.out.put((task.arg, res))

    def submit(self, arg, out: queue.SimpleQueue) -> "_Task | None":
        """Queue a task, or return None if the pool is closed."""
        with self._lock:
            if not self._open:
                return None
            task = _Task(arg, out)
            self._q.put(task)
            return task

    def cancel(self, tasks) -> list:
        """Cancel pending tasks, remove them from the queue, and return their args."""
        with self._state_lock:
            done = []
            for t in tasks:
                if t.state == _Task.QUEUED:
                    t.state = _Task.CANCELLED
                    done.append(t.arg)
        if done:
            # Remove cancelled tasks while preserving live tasks and sentinels.
            with self._lock:
                keep = []
                while True:
                    try:
                        keep.append(self._q.get_nowait())
                    except queue.Empty:
                        break
                for item in keep:
                    if item is None or item.state != _Task.CANCELLED:
                        self._q.put(item)
        return done

    def close(self, timeout: float = 0.0) -> None:
        """Stop workers after queued tasks finish, with a bounded join timeout."""
        with self._lock:
            if not self._open:
                return
            self._open = False
            for _ in self._threads:
                self._q.put(None)
        deadline = time.monotonic() + timeout
        for t in self._threads:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            t.join(left)


class ThresholdExchange:
    """Exchange top-m score lists between workers.

    Each worker publishes its latest thresholds and polls peers for updates.
    Peer shapes must match `sizes`; `fetch_concurrency` limits parallel GETs.
    """

    def __init__(self, store: Store, run_id: str, rank: int, num_jobs: int,
                 sizes: dict[str, tuple[int, int]], refresh_s: float = 5.0,
                 publish_min_interval_s: float = 1.0, fetch_concurrency: int = 8):
        if num_jobs < 2:
            raise ValueError("threshold exchange needs at least two workers")
        if not 0 <= rank < num_jobs:
            raise ValueError(f"rank {rank} out of range for {num_jobs} workers")
        if not run_id:
            raise ValueError("threshold exchange needs a run id")

        # Validate inputs independently of the configuration layer.
        if not (isinstance(refresh_s, (int, float)) and math.isfinite(refresh_s)
                and 0 < refresh_s <= threading.TIMEOUT_MAX):
            raise ValueError("refresh_s must be a finite number > 0 and <= "
                             f"threading.TIMEOUT_MAX, got {refresh_s!r}")
        if not (isinstance(publish_min_interval_s, (int, float))
                and math.isfinite(publish_min_interval_s) and publish_min_interval_s >= 0):
            raise ValueError("publish_min_interval_s must be a finite number >= 0, "
                             f"got {publish_min_interval_s!r}")
        if (isinstance(fetch_concurrency, bool) or not isinstance(fetch_concurrency, int)
                or fetch_concurrency < 1):
            raise ValueError(f"fetch_concurrency must be an int >= 1, got {fetch_concurrency!r}")
        for n, hm in sizes.items():
            if (len(hm) != 2
                    or any(isinstance(x, bool) or not isinstance(x, int) for x in hm)
                    or hm[0] < 0 or hm[1] < 1):
                raise ValueError(f"sizes[{n!r}] must be (rows >= 0, width >= 1), got {hm!r}")

        self.store = store
        self.run_id = run_id
        self.rank = rank
        self.num_jobs = num_jobs
        self.sizes = {n: (int(hm[0]), int(hm[1])) for n, hm in sizes.items()}
        self.refresh_s = float(refresh_s)
        self.publish_min_interval_s = float(publish_min_interval_s)
        self.subdir = subdir_for(run_id)
        self._root = f"{store.root.rstrip('/')}/{self.subdir}"
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        # Latest unpublished state.
        self._pending: dict[str, np.ndarray] | None = None

        # Initialize sequence numbers from wall-clock time to distinguish restarts.
        self._seq = time.time_ns() // 1_000_000
        self._last_write_t = 0.0

        # Latest peer results as zero-copy float16 views; sanitized on the GPU.
        self._peers_lists: dict[int, dict[str, np.ndarray]] = {}
        self._version = 0

        # Track peer versions and avoid redundant reads.
        self._peer_sig: dict[int, tuple] = {}
        self._peer_seq: dict[int, int] = {}
        self._warned: set[str] = set()
        self._empty_polls = 0
        self._stop = threading.Event()
        self._closed = False
        self.fetch_concurrency = int(fetch_concurrency)

        # Prevent duplicate fetches, including across timed-out poll rounds.
        self._inflight: set[int] = set()
        self._inflight_lock = threading.Lock()

        # Serialize poll rounds to prevent concurrent state updates.
        self._round_lock = threading.Lock()

        # Initialize all thread-owned statistics before starting workers.
        self.stats = {
            "enabled": True, "rank": rank, "num_jobs": num_jobs, "run_id": run_id[:16],
            "publishes": 0, "publish_errors": 0,
            "polls": 0, "poll_errors": 0, "peer_files_read": 0, "peer_files_skipped": 0,
            "peer_files_stale": 0,
            "peer_ranks_seen": 0, "versions": 0,
            "peer_gets": 0, "poll_seconds_max": 0.0,
            "poll_rounds_timed_out": 0, "peer_fetches_still_running": 0,
            "fetch_concurrency": self.fetch_concurrency,
            "publish_width": {n: m for n, (_, m) in self.sizes.items()},
            "versions_applied": 0, "applies": 0, "first_applied_file": None,
        }

        if not store.is_s3:
            os.makedirs(self._root, exist_ok=True)

        # Use at most one fetch worker per peer.
        _workers = min(self.fetch_concurrency, num_jobs - 1)
        self._pool = (_FetchPool(self._fetch_cand, _workers, "thr-exchange-get")
                      if _workers > 1 else None)
        _LIVE.add(self)

        try:
            # Load existing peer thresholds before starting background threads.
            self.poll_once()
            self._pub_thread = threading.Thread(
                target=self._publisher, name="thr-exchange-pub", daemon=True
            )
            self._poll_thread = threading.Thread(
                target=self._poller, name="thr-exchange-poll", daemon=True
            )
            self._pub_thread.start()
            self._poll_thread.start()
        except BaseException:
            # Clean up partially initialized workers if startup fails.
            self._stop.set()
            with self._cv:
                self._cv.notify_all()
            if self._pool is not None:
                self._pool.close()
            _LIVE.discard(self)
            raise

    # ---- producer side ------------------------------------------------------
    def publish(self, arrays: dict[str, np.ndarray]) -> None:
        """Queue the latest top-m lists without blocking.

        Converts scores to float16 rounded toward -inf. Existing float16
        arrays may be reused without copying and must not be modified.
        """
        clean = {}
        for name, (h, m) in self.sizes.items():
            a = np.asarray(arrays[name])
            if a.shape != (h, m):
                raise ValueError(f"threshold array for {name!r} has shape {a.shape}, expected {(h, m)}")
            clean[name] = np.ascontiguousarray(a) if a.dtype == np.float16 else fp16_floor(a)
        with self._cv:
            self._pending = clean
            self._cv.notify()

    def _warn_once(self, key: str, msg: str, *args) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg, *args)

    def _publisher(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop.is_set():
                    self._cv.wait(timeout=1.0)
                if self._pending is None and self._stop.is_set():
                    return
                arrays, self._pending = self._pending, None
            self._write_spaced(arrays)

    def _write_spaced(self, arrays: dict[str, np.ndarray]) -> None:
        """Rate-limit writes while publishing the latest queued thresholds."""
        wait = self.publish_min_interval_s - (time.monotonic() - self._last_write_t)
        if wait > 0:
            time.sleep(wait)
        try:
            self._write(arrays)
            self.stats["publishes"] += 1
        except Exception as exc:  # noqa: BLE001 — best effort
            self.stats["publish_errors"] += 1
            self._warn_once("publish", "threshold exchange: publish failed (%s); peers prune "
                            "less. Further failures are counted, not logged.", exc)
        self._last_write_t = time.monotonic()

    def _write(self, arrays: dict[str, np.ndarray]) -> None:
        self._seq += 1
        header, parts = encode_header(self._seq, arrays)
        self._put(f"{self.subdir}/{file_name(self.rank)}", header, parts)

    def _put(self, name: str, header: bytes, parts: list[tuple[int, np.ndarray]]) -> None:
        """Stream the header and arrays without copying the payload."""
        if not _io_enter():
            raise RuntimeError("interpreter exiting")
        try:
            self._put_io(name, header, parts)
        finally:
            _io_exit()

    def _put_io(self, name: str, header: bytes, parts: list[tuple[int, np.ndarray]]) -> None:
        if self.store.is_s3:
            # S3 publishes complete objects; readers reject truncated payloads.
            path = self.store._prepare_write(name)
            with self.store.fs.open_output_stream(path) as sink:
                write_stream(sink, header, parts)
        else:
            # Write to a unique temporary file and atomically replace the target.
            tmp = f"{name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
            path = self.store._prepare_write(tmp)
            try:
                with self.store.fs.open_output_stream(path) as sink:
                    write_stream(sink, header, parts)
                os.replace(path, path[: -(len(tmp) - len(name))])
            except BaseException:
                try:
                    os.unlink(path)
                except OSError:
                    pass
                raise

    # ---- consumer side ------------------------------------------------------
    def _poller(self) -> None:
        while not self._stop.wait(self.refresh_s):
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001 — a dead poller = no exchange
                self.stats["poll_errors"] += 1
                self._warn_once("poll_exc", "threshold exchange: poll raised (%s); "
                                "continuing with the last known peer thresholds", exc)

    # How long a poll round waits for a round still running on another
    # thread (only ever the poller outliving close()'s join) before giving up.
    _ROUND_LOCK_TIMEOUT_S = 60.0
    # Longest a round waits for its GETs, pooled or one at a time (see
    # `_fetched`).
    _ROUND_DEADLINE_S = 60.0

    def _fetch(self, path: str):
        """Fetch and decode a peer file, returning a tagged result."""
        if not _io_enter():
            return ("get", RuntimeError("interpreter exiting"))
        try:
            with self.store.fs.open_input_file(path) as f:
                buf = f.read_buffer()        # C++ read into a pyarrow Buffer
        except Exception as exc:  # noqa: BLE001
            return ("get", exc)
        finally:
            _io_exit()
        try:
            seq, got = decode(buf, self.sizes)
        except Exception as exc:  # noqa: BLE001
            return ("bad", exc)
        return ("ok", seq, got)

    def _fetch_cand(self, cand: tuple):
        try:
            return self._fetch(cand[1])
        finally:
            self._release((cand[0],))

    def _claim(self, cands: list) -> list:
        """Claim peers without outstanding GETs to prevent duplicate fetches."""
        with self._inflight_lock:
            mine = [c for c in cands if c[0] not in self._inflight]
            self._inflight.update(c[0] for c in mine)
        if len(mine) < len(cands):
            self.stats["peer_fetches_still_running"] += len(cands) - len(mine)
        return mine

    def _release(self, peers) -> None:
        with self._inflight_lock:
            self._inflight.difference_update(peers)

    def _spawn(self, cand: tuple, out: queue.SimpleQueue) -> None:
        """Run a fetch on a daemon thread and report its result."""
        def run():
            try:
                res = self._fetch_cand(cand)
            except BaseException as exc:  # noqa: BLE001 — never strand the round
                res = ("get", exc)
            out.put((cand, res))
        try:
            threading.Thread(target=run, name="thr-exchange-get-single", daemon=True).start()
        except BaseException as exc:  # noqa: BLE001
            self._release((cand[0],))
            out.put((cand, ("get", exc)))

    def _fetched(self, cands: list):
        """Fetch peers concurrently, yielding results in completion order.

        Bounds concurrency and round duration. Cancels queued tasks on exit
        while allowing active GETs to finish independently.
        """
        deadline = time.monotonic() + self._ROUND_DEADLINE_S
        cands = self._claim(cands)
        if not cands:
            return
        sequential = self.fetch_concurrency <= 1 or len(cands) <= 1
        pool, transient = self._pool, False
        if not sequential and (pool is None or not pool.is_open):
            try:
                pool = _FetchPool(self._fetch_cand, min(self.fetch_concurrency, len(cands)),
                                  "thr-exchange-get-final")
                transient = True
            except Exception:  # noqa: BLE001 — no threads to spare: one GET at a time
                pool, sequential = None, True
        limit = 1 if sequential else self.fetch_concurrency
        out: queue.SimpleQueue = queue.SimpleQueue()
        in_flight, nxt = 0, 0
        queued: list[_Task] = []     # this round's pool tasks
        try:
            while nxt < len(cands) or in_flight:
                while nxt < len(cands) and in_flight < limit:
                    c = cands[nxt]
                    nxt += 1
                    task = None if sequential else pool.submit(c, out)
                    if task is None:
                        self._spawn(c, out)
                    else:
                        queued.append(task)
                    in_flight += 1
                try:
                    c, res = out.get(timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    self.stats["poll_errors"] += 1
                    self.stats["poll_rounds_timed_out"] += 1
                    self._warn_once(
                        "round_deadline", "threshold exchange: %d peer GET(s) still "
                        "running after %.0fs; abandoning them this round (counted in "
                        "poll_rounds_timed_out)", in_flight, self._ROUND_DEADLINE_S)
                    return
                in_flight -= 1
                yield c, res
        finally:
            # Release unsubmitted peers; only active GETs retain their claims.
            self._release(c[0] for c in cands[nxt:])
            if queued:
                self._release(c[0] for c in pool.cancel(queued))
            if transient:
                pool.close()

    def poll_once(self) -> int:
        """Fetch peer updates and return the number of newer files applied."""
        if not self._round_lock.acquire(timeout=self._ROUND_LOCK_TIMEOUT_S):
            self.stats["poll_errors"] += 1
            self._warn_once("round_lock", "threshold exchange: a previous poll round is "
                            "still running after %.0fs (a stuck GET?); skipping this one",
                            self._ROUND_LOCK_TIMEOUT_S)
            return 0
        try:
            t0 = time.perf_counter()
            folded = self._poll_round()
            self.stats["poll_seconds_max"] = max(self.stats["poll_seconds_max"],
                                                 round(time.perf_counter() - t0, 4))
            return folded
        finally:
            self._round_lock.release()

    def _poll_round(self) -> int:
        # Discover changed peer files and fetch them concurrently.
        folded = 0
        self.stats["polls"] += 1
        try:
            # Treat missing prefixes as empty until peers publish.
            infos = self.store.fs.get_file_info(
                pafs.FileSelector(self._root, recursive=False, allow_not_found=True))
        except Exception as exc:  # noqa: BLE001
            self.stats["poll_errors"] += 1
            self._warn_once("list", "threshold exchange: cannot list %s (%s); this worker "
                            "will not see its peers' thresholds (check s3:ListBucket on the "
                            "output prefix)", self._root, exc)
            return 0
        peers = set()
        cands = []
        for info in infos:
            if info.type != pafs.FileType.File:
                continue
            m = _FILE_RE.search(info.path)
            if not m:
                continue
            peer = int(m.group(1))
            if (peer == self.rank or peer >= self.num_jobs
                    or info.path.rsplit("/", 1)[-1] != file_name(peer)):
                # Only the canonical name: one candidate per peer per round
                # (fetched concurrently, two would race on its bookkeeping).
                continue
            peers.add(peer)
            sig = (info.size, info.mtime_ns if info.mtime_ns is not None else info.mtime)
            if self._peer_sig.get(peer) == sig:
                continue
            cands.append((peer, info.path, sig))
        updates: dict[int, dict[str, np.ndarray]] = {}
        
        # Ensure fetch cleanup runs under _round_lock, even on errors.
        with contextlib.closing(self._fetched(cands)) as fetched:
            for (peer, path, sig), res in fetched:
                self.stats["peer_gets"] += 1
                if res[0] != "ok":
                    if res[0] == "bad":
                        # Cache invalid files; retry transport failures.
                        self._peer_sig[peer] = sig
                    self.stats["peer_files_skipped"] += 1
                    self._warn_once("skip", "threshold exchange: skipping unreadable peer file %s "
                                    "(%s); further skips are counted in peer_files_skipped "
                                    "(check s3:GetObject on the output prefix if this persists)",
                                    path, res[1])
                    continue

                _, seq, got = res
                self._peer_sig[peer] = sig
                if self._peer_seq.get(peer, -1) >= seq:
                    # Ignore stale or previously processed versions.
                    self.stats["peer_files_stale"] += 1
                    continue

                self._peer_seq[peer] = seq
                self.stats["peer_files_read"] += 1
                folded += 1
                updates[peer] = got
        self.stats["peer_ranks_seen"] = max(self.stats["peer_ranks_seen"], len(peers))
        if not peers:
            self._empty_polls += 1
            if self._empty_polls == _NO_PEER_WARN_POLLS:
                self._warn_once("nopeers", "threshold exchange: no peer thresholds under %s "
                                "after %d polls; is output.path shared by every worker?",
                                self._root, _NO_PEER_WARN_POLLS)
        if updates:
            # Apply peer updates as a single version without comparing payloads.
            with self._lock:
                self._peers_lists.update(updates)
                self._version += 1
                self.stats["versions"] = self._version
        return folded

    def latest(self, seen_version: int) -> tuple[int, dict[str, list[np.ndarray]] | None, bool]:
        """Return the latest zero-copy peer lists and whether their version changed."""
        with self._lock:
            if not self._peers_lists:
                return self._version, None, False
            lists = {n: [pl[n] for pl in self._peers_lists.values()] for n in self.sizes}
            return self._version, lists, self._version != seen_version

    def close(self, final: bool = True) -> None:
        """Stop the threads. With ``final``, flush the last pending publish
        first so late-starting peers inherit this worker's converged state."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        self._pub_thread.join(timeout=30.0)
        self._poll_thread.join(timeout=5.0)
        if self._pool is not None:
            # No join beyond a moment: a fetch worker inside a GET finishes it
            # alone (daemon thread); a round still running on the poller gives
            # its remaining fetches their own threads. A later `poll_once` (the
            # end-of-scan refresh) uses a transient pool, see `_fetched`.
            self._pool.close(timeout=1.0)
        with self._cv:
            pending, self._pending = self._pending, None
        # Only write if the publisher really stopped: a writer still in flight
        # after the join timeout must not race the final object.
        if final and pending is not None and not self._pub_thread.is_alive():
            self._write_spaced(pending)


# Slack accounts for FP32 rounding differences when a failed worker restarts
# with a different GPU or GEMM configuration. Exact FP32 cosine scoring is
# bounded by ~2*d*u absolute error; TF32 and other metrics are not covered.
# Merge-time verification catches any resulting threshold violations.
SLACK_REL = 2.0 ** -13
SLACK_ABS_MIN = 2.0 ** -20
FP32_UNIT = 2.0 ** -24

def lower_for_slack(scores: np.ndarray, dim: int | None) -> np.ndarray:
    s = np.ascontiguousarray(scores, dtype=np.float32)
    finite = np.isfinite(s)
    slack_abs = max(SLACK_ABS_MIN, 2.0 * (dim or 0) * FP32_UNIT)
    lowered = s.copy()
    lowered[finite] = (s[finite] - (np.abs(s[finite]) * SLACK_REL + slack_abs)).astype(np.float32)
    return lowered


def keys_from_scores(scores: np.ndarray, device, dim: int | None = None) -> "torch.Tensor":
    """Packed int64 keys for external threshold SCORES, lowered by the re-run
    allowance above (``dim`` = scored dimension, for the absolute term) and
    carrying the worst tie-break ordinal, so a local candidate with an equal
    score still outranks them."""
    import torch

    from nova_bf.tiebreak import TIE_WORST, pack

    t = torch.from_numpy(lower_for_slack(scores, dim)).to(device)
    return pack(t, torch.tensor(TIE_WORST, dtype=torch.int64, device=device))
