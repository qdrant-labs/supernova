"""Cross-worker threshold exchange for safe top-k pruning.

Workers exchange their top-m scores per query to compute a tighter lower
bound on the global k-th score, where m = min(k, ceil(2k/N)).
Scores are published as float16 rounded toward -inf to preserve safety.
Publishing and polling run asynchronously at file boundaries.

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

import io
import logging
import os
import re
import threading
import time

import numpy as np
import pyarrow.fs as pafs

from nova_bf.io import Store

logger = logging.getLogger(__name__)

SUBDIR = "_thresholds"
_FILE_RE = re.compile(r"thr_rank(\d+)\.npz$")
_KEY_PREFIX = "s_"          # np.savez reserves names such as `file`/`allow_pickle`
# How many empty polls before warning that no peer has been seen.
_NO_PEER_WARN_POLLS = 3


def file_name(rank: int) -> str:
    return f"thr_rank{rank:05d}.npz"



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


def union_kth(own, peers: list[np.ndarray], k: int, device, chunk: int = 8192):
    """Per query, the k-th largest over `own` ((h, m) float32 tensor on
    `device`, this worker's current top-m, non-finite allowed) and every peer
    list ((h, m') float16 host arrays). -inf where fewer than k entries exist.
    Done in row chunks so the (rows, N*m) concatenation stays small."""
    import torch

    h = own.shape[0]
    out = torch.full((h,), float("-inf"), dtype=torch.float32, device=own.device)
    width = own.shape[1] + sum(int(p.shape[1]) for p in peers)
    if width < k:
        return out
    neg_inf = torch.tensor(float("-inf"), dtype=torch.float32, device=own.device)
    for r0 in range(0, h, chunk):
        r1 = min(h, r0 + chunk)
        parts = [torch.where(torch.isfinite(own[r0:r1]), own[r0:r1], neg_inf)]
        for p in peers:
            parts.append(torch.from_numpy(np.ascontiguousarray(p[r0:r1])).to(own.device).float())
        cat = torch.cat(parts, dim=1)
        out[r0:r1] = torch.topk(cat, k, dim=1, sorted=True).values[:, -1]
    return out


class ThresholdExchange:
    """Publish this worker's top-m score lists and keep every peer's latest.

    ``run_id`` scopes the exchange directory to one run (see module doc).
    ``sizes`` maps search name -> (rows, m): the height of that search's
    running state and the published width (`publish_width`); peer arrays with
    any other shape are ignored.
    """

    def __init__(self, store: Store, run_id: str, rank: int, num_jobs: int,
                 sizes: dict[str, int], refresh_s: float = 5.0,
                 publish_min_interval_s: float = 1.0):
        if num_jobs < 2:
            raise ValueError("threshold exchange needs at least two workers")
        if not 0 <= rank < num_jobs:
            raise ValueError(f"rank {rank} out of range for {num_jobs} workers")
        if not run_id:
            raise ValueError("threshold exchange needs a run id")
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
        self._pending: dict[str, np.ndarray] | None = None   # latest unpublished
        self._last_published: dict[str, np.ndarray] | None = None
        # Monotone across re-runs of the same rank (a re-run starting at 1
        # would be ignored by peers that saw the first attempt's higher seq).
        self._seq = time.time_ns() // 1_000_000
        self._last_write_t = 0.0
        # rank -> {search: (h, m) float16}, each peer's latest list.
        self._peers_lists: dict[int, dict[str, np.ndarray]] = {}
        self._version = 0
        self._peer_sig: dict[int, tuple] = {}     # rank -> (size, mtime) last folded
        self._peer_seq: dict[int, int] = {}
        self._warned: set[str] = set()
        self._empty_polls = 0
        self._stop = threading.Event()
        self._closed = False
        self.stats = {
            "enabled": True, "rank": rank, "num_jobs": num_jobs, "run_id": run_id[:16],
            "publishes": 0, "publishes_skipped_unchanged": 0, "publish_errors": 0,
            "polls": 0, "poll_errors": 0, "peer_files_read": 0, "peer_files_skipped": 0,
            "peer_ranks_seen": 0, "versions": 0,
            "publish_width": {n: m for n, (_, m) in self.sizes.items()},
            # Filled in by the compute loop.
            "versions_applied": 0, "applies": 0, "first_applied_file": None,
        }
        if not store.is_s3:
            os.makedirs(self._root, exist_ok=True)
        # A worker that starts after its peers (fleet scale-up, a re-run rank)
        # should inherit their thresholds before scanning its first file.
        self.poll_once()
        self._pub_thread = threading.Thread(target=self._publisher, name="thr-exchange-pub", daemon=True)
        self._poll_thread = threading.Thread(target=self._poller, name="thr-exchange-poll", daemon=True)
        self._pub_thread.start()
        self._poll_thread.start()

    # ---- producer side ------------------------------------------------------

    def publish(self, arrays: dict[str, np.ndarray]) -> None:
        """Hand over this worker's current top-m lists, (h, m) per search,
        float32 or float16 (latest wins; never blocks). Stored as float16
        rounded toward -inf."""
        clean = {}
        for name, (h, m) in self.sizes.items():
            a = np.asarray(arrays[name])
            if a.shape != (h, m):
                raise ValueError(f"threshold array for {name!r} has shape {a.shape}, expected {(h, m)}")
            clean[name] = a.astype(np.float16, copy=True) if a.dtype == np.float16 else fp16_floor(a)
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
        """Write unless nothing changed; keep writes >= the minimum interval
        apart so a change is always visible in a 1-second mtime."""
        if self._last_published is not None and all(
            np.array_equal(arrays[n], self._last_published[n]) for n in arrays
        ):
            self.stats["publishes_skipped_unchanged"] += 1
            return
        wait = self.publish_min_interval_s - (time.monotonic() - self._last_write_t)
        if wait > 0:
            time.sleep(wait)
        try:
            self._write(arrays)
            self.stats["publishes"] += 1
            self._last_published = arrays
        except Exception as exc:  # noqa: BLE001 — best effort by design
            self.stats["publish_errors"] += 1
            self._warn_once("publish", "threshold exchange: publish failed (%s); peers prune "
                            "less. Further failures are counted, not logged.", exc)
        self._last_write_t = time.monotonic()

    def _write(self, arrays: dict[str, np.ndarray]) -> None:
        self._seq += 1
        buf = io.BytesIO()
        np.savez(buf, seq=np.int64(self._seq),
                 **{_KEY_PREFIX + n: a for n, a in arrays.items()})
        self._put(f"{self.subdir}/{file_name(self.rank)}", buf.getvalue())

    def _put(self, name: str, data: bytes) -> None:
        if self.store.is_s3:
            # An S3 PUT is atomic: readers see the old object or the new one.
            self.store.write_bytes(name, data)
        else:
            # Local: write beside under a per-write name, then rename, so a
            # reader never sees a torn file and two writers never share a tmp.
            tmp = f"{name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
            final = self.store.write_bytes(tmp, data)
            os.replace(final, final[: -(len(tmp) - len(name))])

    # ---- consumer side ------------------------------------------------------

    def _poller(self) -> None:
        while not self._stop.wait(self.refresh_s):
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001 — a dead poller = no exchange
                self.stats["poll_errors"] += 1
                self._warn_once("poll_exc", "threshold exchange: poll raised (%s); "
                                "continuing with the last known peer thresholds", exc)

    def poll_once(self) -> int:
        """Fold every peer file that changed since last time. Returns the
        number of peer files folded."""
        folded = 0
        self.stats["polls"] += 1
        try:
            # allow_not_found: before any peer has published, the prefix does not
            # exist yet (S3 has no directories); that is "no peers", not an error.
            infos = self.store.fs.get_file_info(
                pafs.FileSelector(self._root, recursive=False, allow_not_found=True))
        except Exception as exc:  # noqa: BLE001
            self.stats["poll_errors"] += 1
            self._warn_once("list", "threshold exchange: cannot list %s (%s); this worker "
                            "will not see its peers' thresholds (check s3:ListBucket on the "
                            "output prefix)", self._root, exc)
            return 0
        peers = set()
        updates: dict[int, dict[str, np.ndarray]] = {}
        for info in infos:
            if info.type != pafs.FileType.File:
                continue
            m = _FILE_RE.search(info.path)
            if not m:
                continue
            peer = int(m.group(1))
            if peer == self.rank or peer >= self.num_jobs:
                continue
            peers.add(peer)
            sig = (info.size, info.mtime_ns if info.mtime_ns is not None else info.mtime)
            if self._peer_sig.get(peer) == sig:
                continue
            try:
                with self.store.fs.open_input_file(info.path) as f:
                    raw = f.read()
                arrays = np.load(io.BytesIO(raw))
                seq = int(arrays["seq"])
                got = {}
                for name, (h, m) in self.sizes.items():
                    a = np.asarray(arrays[_KEY_PREFIX + name])
                    if a.shape != (h, m):
                        raise ValueError(f"{name}: shape {a.shape}, expected {(h, m)}")
                    # Only finite scores may ever raise the bound: a +inf entry
                    # would count as a candidate above everything and NaN
                    # would poison the top-k; both become -inf.
                    a = a.astype(np.float16, copy=False)
                    got[name] = np.where(np.isfinite(a), a, np.float16(-np.inf)).astype(np.float16)
            except Exception as exc:  # noqa: BLE001 — torn/foreign file: skip, prune less
                self.stats["peer_files_skipped"] += 1
                # Not DEBUG: with Get denied on the prefix (List allowed) every
                # peer file is skipped and the exchange silently does nothing.
                self._warn_once("skip", "threshold exchange: skipping unreadable peer file %s "
                                "(%s); further skips are counted in peer_files_skipped "
                                "(check s3:GetObject on the output prefix if this persists)",
                                info.path, exc)
                continue
            self._peer_sig[peer] = sig
            if self._peer_seq.get(peer, -1) >= seq:
                continue          # an older object won a racing write; ignore it
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
            with self._lock:
                changed = False
                for peer, got in updates.items():
                    old = self._peers_lists.get(peer)
                    if old is None or any(not np.array_equal(got[n], old[n]) for n in got):
                        self._peers_lists[peer] = got
                        changed = True
                if changed:
                    self._version += 1
                    self.stats["versions"] = self._version
        return folded

    def latest(self, seen_version: int) -> tuple[int, dict[str, list[np.ndarray]] | None, bool]:
        """``(version, lists, changed)``: `lists` maps search -> every peer's
        latest (h, m) float16 list (None when no peer has published);
        `changed` says whether a peer list changed since ``seen_version``.
        The arrays are the poller's own (it never mutates one in place), so
        no copy is made."""
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
