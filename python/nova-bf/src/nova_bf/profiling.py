"""Optional compute instrumentation for read timing and profiler windows.

Kept separate from `compute.py` because it observes runs rather than executing
them. Hot-path `perf_counter` calls remain beside the statements they measure.

All instrumentation is opt-in. Read timing changes the I/O path, so its results
are for stage comparison, not production throughput.
"""
from __future__ import annotations

import logging
import os
import time
from threading import Lock

logger = logging.getLogger("nova_bf.profiling")


# --- optional read-phase timing ------------------------------------------- 
# Split read time into fetch/decode stages for profiling. Disabled by default 
# because instrumentation changes the read path and adds extra I/O/copies.
_READ_TIMING = "NOVA_BF_READ_TIMING"

# Timed read stages, in execution order. Keep dense, multivector, and sparse
# sub-stages separate so profiling identifies the actual decode cost.
READ_SPLIT_FIELDS = (
    "fetch", "decode_parquet", "dates",
    "dense_cast", "multivector_cast",
    "sparse_decode", "sparse_norms", "sparse_gate", "sparse_remap",
    "ids", "select",
)

# Non-time read metrics are accumulated separately from stage timings.
_COUNTER_FIELDS = {"bytes": "read_bytes", "fetch_mode": "ranged_files"}
_READ_SPLIT: dict[str, float] = {}
_READ_COUNTERS: dict[str, float] = {}
_READ_SPLIT_LOCK = Lock()


def read_timing_on() -> bool:
    return bool(os.environ.get(_READ_TIMING))


def read_split_add(parts: dict) -> None:
    """Fold one file's split into the run totals. Reader threads race here.

    Seconds go to the split; the two non-seconds counters are routed out of it
    under names that say what they are.
    """
    with _READ_SPLIT_LOCK:
        for k, v in parts.items():
            if k in _COUNTER_FIELDS:
                name = _COUNTER_FIELDS[k]
                _READ_COUNTERS[name] = _READ_COUNTERS.get(name, 0.0) + v
            else:
                _READ_SPLIT[k] = _READ_SPLIT.get(k, 0.0) + v


def read_split_totals() -> dict:
    """Per-stage SECONDS, summed over reader threads."""
    with _READ_SPLIT_LOCK:
        return dict(_READ_SPLIT)


def read_counter_totals() -> dict:
    """`read_bytes` (total fetched) and `ranged_files` (how many took the
    ranged path). Counts and bytes, deliberately not in the seconds dict."""
    with _READ_SPLIT_LOCK:
        return dict(_READ_COUNTERS)


def reset_read_split() -> None:
    with _READ_SPLIT_LOCK:
        _READ_SPLIT.clear()
        _READ_COUNTERS.clear()


def fetch_and_decode(store, read_path: str, columns, column_groups=None):
    """Read one Parquet file while timing fetch and decode separately.

    Profiling buffers the whole file before decoding, unlike production reads
    that fetch only requested column chunks. Timings and bytes are therefore
    diagnostic only; per-group decode timings are independent and non-additive.

    That buffering costs real S3 egress: measured up to ~400x the bytes a
    narrow column selection actually needs. Do not leave this on.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from nova_bf.io import _RANGED_GET_MIN_BYTES

    t = {}
    t0 = time.perf_counter()
    size = None
    ranged = getattr(store, "_ranged_download", None)
    if getattr(store, "ranged_get", False) and ranged is not None:
        size = store.fs.get_file_info(read_path).size
    if size is not None and size >= _RANGED_GET_MIN_BYTES:
        buf = ranged(read_path, size)
        t["fetch_mode"] = 1.0
    else:
        with store.fs.open_input_file(read_path) as fh:
            buf = fh.read()
        t["fetch_mode"] = 0.0
    t["fetch"] = time.perf_counter() - t0
    t["bytes"] = buf.size if hasattr(buf, "size") else len(buf)

    t0 = time.perf_counter()
    table = pq.read_table(pa.BufferReader(buf), columns=columns)
    t["decode_parquet"] = time.perf_counter() - t0

    for name, cols in (column_groups or {}).items():
        cols = [c for c in cols if c in table.column_names]
        if not cols:
            continue
        t0 = time.perf_counter()
        pq.read_table(pa.BufferReader(buf), columns=cols)
        t[f"decode_parquet_{name}"] = time.perf_counter() - t0
    return table, t


# --- optional steady-state profiling window --------------------------------
# Profile only selected files so traces capture warmed-up/pruned steady state
# instead of startup. `START:END` is 1-based and inclusive.
PROFILE_FILES = "NOVA_BF_PROFILE_FILES"
PROFILE_OUT = "NOVA_BF_PROFILE_OUT"
_PROF: dict = {"window": None, "prof": None, "out": ".", "active": False}


def parse_window():
    """`(start, end)` 1-based inclusive, or `None`."""
    raw = os.environ.get(PROFILE_FILES, "").strip()
    if not raw:
        return None
    try:
        a, b = raw.split(":")
        lo, hi = int(a), int(b)
    except ValueError:
        raise ValueError(
            f"{PROFILE_FILES}={raw!r} must be START:END, 1-based file "
            f"positions, inclusive (e.g. 60:62)"
        ) from None
    if lo < 1 or hi < lo:
        raise ValueError(f"{PROFILE_FILES}={raw!r}: need 1 <= START <= END")
    return lo, hi


def start(n: int) -> None:
    """Open the window if file `n` (1-based) is its first file."""
    w = _PROF["window"]
    if w is None or _PROF["active"] or n != w[0]:
        return
    import torch
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    _PROF["prof"] = profile(activities=acts, record_shapes=False, with_stack=False)
    _PROF["prof"].__enter__()
    _PROF["active"] = True
    # `n` as well as the window: the window is what was REQUESTED, and a test
    # or a reader comparing a trace against the log needs the file it actually
    # opened on.
    logger.info("torch.profiler ON at file %d for files %d-%d", n, w[0], w[1])


def stop(n: int) -> None:
    """Close and export if file `n` (1-based) is the window's last file."""
    w = _PROF["window"]
    if w is None or not _PROF["active"] or n != w[1]:
        return
    import gzip
    import shutil

    prof = _PROF["prof"]
    prof.__exit__(None, None, None)
    _PROF["active"] = False
    out = _PROF["out"]
    os.makedirs(out, exist_ok=True)
    tbl = prof.key_averages().table(
        sort_by="self_device_time_total", row_limit=60)
    with open(os.path.join(out, "C_kernels.txt"), "w") as fh:
        fh.write(tbl)
    raw = os.path.join(out, "C_trace.json")
    prof.export_chrome_trace(raw)
    with open(raw, "rb") as f, gzip.open(raw + ".gz", "wb") as g:
        shutil.copyfileobj(f, g)
    os.remove(raw)
    _PROF["prof"] = None
    logger.info("torch.profiler OFF after file %d; wrote %s", n, out)


class _NoMark:
    """The shut-window case: enter, exit, do nothing, allocate nothing."""

    __slots__ = ()
    _rf = None                      # the contract callers and tests check

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# Reuse one no-op marker to avoid allocations when profiling is disabled.
_NO_MARK = _NoMark()


class _RecordMark:
    """The open-window case: a real `torch.profiler.record_function`."""

    __slots__ = ("_rf",)

    def __init__(self, name: str):
        import torch

        self._rf = torch.profiler.record_function(name)

    def __enter__(self):
        self._rf.__enter__()
        return self

    def __exit__(self, *exc):
        self._rf.__exit__(*exc)
        return False


def slice_mark(name: str):
    """Mark one corpus slice, but only while the profiling window is open.

    A function rather than a class so the shut case can hand back a shared
    singleton instead of building anything. The window is checked here, at
    entry, because it opens and closes DURING the scan -- a mark decided once
    at import would annotate the wrong files.
    """
    return _RecordMark(name) if _PROF["active"] else _NO_MARK


def configure() -> tuple[int, int] | None:
    """Load profiling configuration and reset any stale PROFILER state.

    Not the read-split accumulator — see the note at its reset in `compute`.
    """
    if _PROF["active"]:
        # Properly close a stale profiler before resetting its state.
        logger.warning(
            "a previous profiling window was never closed (its END was past the "
            "last file); closing it now. Its trace was not written."
        )
        stale = _PROF["prof"]
        if stale is not None:
            try:
                stale.__exit__(None, None, None)
            except BaseException as exc:            # noqa: BLE001
                logger.warning("could not close the stale profiler: %r", exc)
    _PROF["active"] = False
    _PROF["prof"] = None
    _PROF["window"] = parse_window()
    _PROF["out"] = os.environ.get(PROFILE_OUT, ".")
    return _PROF["window"]
