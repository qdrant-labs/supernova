"""Uniform parquet IO over local paths and ``s3://`` URIs, via pyarrow.fs.

One abstraction, no per-backend code in the compute/merge logic: a `Store` wraps
a pyarrow FileSystem + root, and lists/reads/writes parquet the same way whether
the root is a local directory or an S3 prefix. S3 uses the standard AWS
credential chain (env / profile / instance role); region from `AWS_REGION` or
pyarrow's per-bucket resolution.
"""

from __future__ import annotations

import logging
import os
import threading

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq


@dataclass
class ParquetFile:
    read_path: str  # path handed to pyarrow (includes the bucket for s3)
    key: str        # loader-consistent filename used for id derivation
    size: int | None = None   # bytes, from the listing (content-ish identity)


# Parallel ranged-read tuning for LARGE parquet files (see Store.read_columns).
# A reader can only parallelize along a file's internal structure (row groups /
# column chunks). Files written with ~ONE row group cause the download to read
# one giant column chunk — so the read degenerates into a single sequential stream
# that no IO thread pool can speed up. Fetching the whole file as many
# concurrent byte ranges FIRST (the `aws s3 cp` strategy), then parsing from
# memory, sidesteps that regardless of parquet internals, on s3 and POSIX roots
# alike. OPT-IN via `Store(uri, ranged_get=True)` — wired to
# `params.io_ranged_get` in the compute config — and even then only for files
# of at least _RANGED_GET_MIN_BYTES (below that the normal reader is already
# fine). Costs one file's raw bytes of extra RAM while that file is parsed.
_RANGED_GET_BYTES = 64 * 1024 * 1024
_RANGED_GET_CONCURRENCY = 24
_RANGED_GET_MIN_BYTES = 256 * 1024 * 1024
# Process-wide pool for concurrent ranged GETs, capped at 512 threads.
# Per-file concurrency is limited separately. Threads are created on demand;
# NOVA_BF_FETCH_POOL_SIZE overrides the default at pool initialization.
_FETCH_POOL_SIZE_DEFAULT = 512
_FETCH_POOL: ThreadPoolExecutor | None = None
_FETCH_POOL_PID: int | None = None
_FETCH_POOL_LOCK = threading.Lock()

logger = logging.getLogger(__name__)


def _reset_fetch_pool_in_child() -> None:
    # A fork while another thread held the lock would otherwise deadlock the
    # child's first `_fetch_pool()`; the parent's pool threads don't exist there.
    global _FETCH_POOL_LOCK, _FETCH_POOL
    _FETCH_POOL_LOCK = threading.Lock()
    _FETCH_POOL = None


os.register_at_fork(after_in_child=_reset_fetch_pool_in_child)


def _fetch_pool_size() -> int:
    raw = os.environ.get("NOVA_BF_FETCH_POOL_SIZE", "").strip()
    try:
        size = int(raw) if raw else _FETCH_POOL_SIZE_DEFAULT
        if size < 1:
            raise ValueError(size)
    except ValueError:
        logger.warning("NOVA_BF_FETCH_POOL_SIZE=%r is not a positive integer; "
                       "using %d", raw, _FETCH_POOL_SIZE_DEFAULT)
        size = _FETCH_POOL_SIZE_DEFAULT
    return size


def _fetch_pool() -> ThreadPoolExecutor:
    """The process-wide ranged-GET pool, created on first use.
    
    Re-created after a fork, since a child inherits the executor
    object but not its threads."""
    global _FETCH_POOL, _FETCH_POOL_PID
    with _FETCH_POOL_LOCK:
        if _FETCH_POOL is None or _FETCH_POOL_PID != os.getpid():
            _FETCH_POOL = ThreadPoolExecutor(
                max_workers=_fetch_pool_size(), thread_name_prefix="nova-bf-fetch",
            )
            _FETCH_POOL_PID = os.getpid()
        return _FETCH_POOL


def _copy_range(dst: np.ndarray, buf) -> None:
    """Copy a fetched range without holding the GIL, avoiding GPU stalls.
    Validate length to prevent unintended NumPy broadcasting."""
    if len(buf) != dst.size:
        raise OSError(f"ranged read returned {len(buf)} bytes, expected {dst.size}")
    np.copyto(dst, np.frombuffer(buf, dtype=np.uint8))


# rust_fetch I/O, on by default (`params.rust_fetch_dense` / `_columns`).
# Dense: Rust fetches dense chunks via concurrent ranged GETs and decodes
# directly into (n, dim), avoiding intermediate Arrow buffers and copies.
# Preserves bitwise equivalence with dense_to_2d; unsupported layouts
# fall back to PyArrow. Columns: see Store.read_columns_fetched.
# Settings come from `configure_rust_fetch` (called by compute with the run's
# `params`); NOVA_BF_RUST_FETCH_DENSE / _COLUMNS / _CONC / _PART_MIB / _CHUNKS
# override them per run.
_NATIVE_DEFAULTS = {"dense": True, "columns": True, "concurrency": 128,
                    "part_mib": 16, "chunks": 8}
_NATIVE_CFG = dict(_NATIVE_DEFAULTS)


def configure_rust_fetch(dense: bool | None = None, columns: bool | None = None,
                         concurrency: int | None = None, part_mib: int | None = None,
                         chunks: int | None = None) -> None:
    """Set the run's rust_fetch settings (None = built-in default).
    NOVA_BF_RUST_FETCH_* environment variables still take precedence."""
    given = {"dense": dense, "columns": columns, "concurrency": concurrency,
             "part_mib": part_mib, "chunks": chunks}
    for k, v in given.items():
        _NATIVE_CFG[k] = _NATIVE_DEFAULTS[k] if v is None else v


_NATIVE_ENV = {"dense": "NOVA_BF_RUST_FETCH_DENSE", "columns": "NOVA_BF_RUST_FETCH_COLUMNS",
               "concurrency": "NOVA_BF_RUST_FETCH_CONC", "part_mib": "NOVA_BF_RUST_FETCH_PART_MIB",
               "chunks": "NOVA_BF_RUST_FETCH_CHUNKS"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean (1/0, true/false), got {raw!r}")


def rust_fetch_settings() -> dict:
    """The RESOLVED rust_fetch settings (environment over configuration)."""
    out = {}
    for k, v in _NATIVE_CFG.items():
        env = _NATIVE_ENV[k]
        out[k] = _env_bool(env, v) if isinstance(v, bool) else _env_int(env, v)
    return out


_NATIVE: dict = {"pid": None, "clients": {}, "warned": set(), "orphans": []}
_NATIVE_LOCK = threading.Lock()


def _reset_native_lock_in_child() -> None:
    # Reset synchronization after fork to prevent inherited lock deadlocks
    # and clear in-flight state from threads that no longer exist.
    global _NATIVE_LOCK, _INFLIGHT_CV
    _NATIVE_LOCK = threading.Lock()
    _INFLIGHT_CV = threading.Condition()
    _INFLIGHT["n"] = 0


os.register_at_fork(after_in_child=_reset_native_lock_in_child)

# Track native calls to prevent Python 3.13 shutdown crashes.
# At exit, block new calls and wait for active Rust reads to finish.
_INFLIGHT = {"n": 0, "closing": False}
_INFLIGHT_CV = threading.Condition()


def _native_enter() -> bool:
    with _INFLIGHT_CV:
        if _INFLIGHT["closing"]:
            return False
        _INFLIGHT["n"] += 1
        return True


def _native_exit() -> None:
    with _INFLIGHT_CV:
        _INFLIGHT["n"] -= 1
        _INFLIGHT_CV.notify_all()


def _drain_native_at_exit(timeout_s: float = 60.0) -> None:
    with _INFLIGHT_CV:
        _INFLIGHT["closing"] = True
    # Cancel in-flight calls (S3 parts stop at once; a local pread finishes
    # its current part), then wait for them to come back out of Rust.
    if _NATIVE["pid"] == os.getpid():
        for client in list(_NATIVE["clients"].values()):
            try:
                client.cancel()
            except Exception:
                pass
    with _INFLIGHT_CV:
        _INFLIGHT_CV.wait_for(lambda: _INFLIGHT["n"] == 0, timeout=timeout_s)


import atexit  # noqa: E402

atexit.register(_drain_native_at_exit)

# Use PyArrow for custom S3 endpoints unsupported by the native client.
_ENDPOINT_ENV = ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT")


def rust_fetch_columns_enabled() -> bool:
    """Download (not decode) the pyarrow-read columns through the native
    transport -- see Store.read_columns_fetched. `params.rust_fetch_columns`,
    overridden by NOVA_BF_RUST_FETCH_COLUMNS."""
    return _env_bool("NOVA_BF_RUST_FETCH_COLUMNS", _NATIVE_CFG["columns"])


# Fetch padding, range coalescing, and tail-read limits.
_FETCH_PAD = 4096
_FETCH_COALESCE = 8 << 10  # Match PyArrow's pre_buffer hole limit.
_FETCH_TAIL = 1 << 20

# Bounded cache of recently fetched column data.
import collections as _collections  # noqa: E402

_FETCH_KEEPALIVE: _collections.deque = _collections.deque(maxlen=4)


def _fetch_image(client, url: str, size: int, pin):
    """Fetch the Parquet footer into a sparse file-sized buffer.

    Returns (image, have_from), or None if footer validation fails.
    """
    if size < 12:
        return None
    image = _zeroed_image(size)
    if os.environ.get("NOVA_BF_RUST_FETCH_CANARY"):
        image.fill(0xA5)  # tests: make any read of an unfetched byte loud
    tail = min(size, _FETCH_TAIL)
    client.read_ranges_into(url, image, [(size - tail, tail, size - tail)], if_match=pin)
    if bytes(image[size - 4:]) != b"PAR1":
        return None
    flen = int.from_bytes(bytes(image[size - 8:size - 4]), "little")
    if flen + 8 > size:
        return None
    have_from = size - tail
    if flen + 8 > tail:
        start = size - 8 - flen
        client.read_ranges_into(url, image, [(start, size - tail - start, start)], if_match=pin)
        have_from = start
    return image, have_from


def _zeroed_image(size: int) -> np.ndarray:
    """Create a zero-initialized, writable uint8 array with lazy allocation.

    Uses anonymous mmap for large arrays to avoid allocating untouched pages
    and disables huge pages to minimize memory residency.
    """
    import mmap

    if size < (1 << 20):
        return np.zeros(size, dtype=np.uint8)
    m = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    if hasattr(mmap, "MADV_NOHUGEPAGE"):
        m.madvise(mmap.MADV_NOHUGEPAGE)
    return np.frombuffer(m, dtype=np.uint8)


def _release_image(image: np.ndarray, table: pa.Table) -> None:
    """Release decoded image pages unless Arrow buffers still reference them."""
    import mmap

    base = image.ctypes.data
    end = base + image.nbytes
    for col in table.columns:
        for chunk in col.chunks:
            stack = [chunk]
            while stack:
                arr = stack.pop()
                for b in arr.buffers():
                    if b is not None and b.size and base <= b.address < end:
                        return
                stack.extend(getattr(arr, "children", None) or [])
                dct = getattr(arr, "dictionary", None)
                if dct is not None:
                    stack.append(dct)
    m = image.base
    m = getattr(m, "obj", m)  # np.frombuffer(mmap) -> memoryview -> mmap
    if isinstance(m, mmap.mmap) and hasattr(mmap, "MADV_DONTNEED"):
        m.madvise(mmap.MADV_DONTNEED)


def _leaf_count(t: pa.DataType) -> int:
    if isinstance(t, pa.BaseExtensionType):  # incl. C++ canonical (uuid, json, opaque)
        return _leaf_count(t.storage_type)
    if pa.types.is_struct(t):
        return sum(_leaf_count(t.field(i).type) for i in range(t.num_fields))
    if pa.types.is_map(t):
        return _leaf_count(t.key_type) + _leaf_count(t.item_type)
    if (pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t)
            or pa.types.is_list_view(t) or pa.types.is_large_list_view(t)):
        return _leaf_count(t.value_type)
    return 1


def _column_ranges(image: np.ndarray, size: int, columns: list[str], have_from: int | None = None):
    """Compute merged Parquet column fetch ranges, excluding the fetched footer.

    Returns None for unsupported or ambiguous metadata to trigger PyArrow fallback.
    """
    end = size if have_from is None else have_from
    try:
        pf = pq.ParquetFile(pa.BufferReader(pa.py_buffer(image)))
        md, arrow = pf.metadata, pf.schema_arrow
        names = arrow.names
        counts = [_leaf_count(f.type) for f in arrow]
        if sum(counts) != md.num_columns or len(set(names)) != len(names):
            return None
        want = set(columns)
        if not want <= set(names):
            return None
        leaves, j = [], 0
        for name, c in zip(names, counts):
            if name in want:
                leaves.extend(range(j, j + c))
            j += c
        # Chunk offsets come from rust_fetch, NOT pyarrow's RowGroupMetaData
        # .column(): on some malformed size statistics that accessor lets a
        # C++ exception reach std::terminate and aborts the whole process
        # (the plain pyarrow read raises a clean error on the same file).
        # chunk_spans refuses those footers, so we decline instead.
        import rust_fetch

        flen = int.from_bytes(bytes(image[size - 8:size - 4]), "little")
        if flen <= 0 or flen > size - 12:
            return None
        footer = bytes(image[size - 8 - flen:size - 8])
        spans = []
        for start, dpo, length in rust_fetch.chunk_spans(footer, leaves):
            if dpo is not None and 0 < dpo < start:
                start = dpo
            if start < 0 or length < 0 or start + length > size:
                return None
            s0, e0 = start, min(end, start + length + _FETCH_PAD)
            if s0 < e0:
                spans.append((s0, e0))
    except Exception:
        return None
    if not spans:
        return []
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + _FETCH_COALESCE:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e - s, s) for s, e in merged]


def rust_fetch_dense_enabled() -> bool:
    """`params.rust_fetch_dense`, overridden by NOVA_BF_RUST_FETCH_DENSE."""
    return _env_bool("NOVA_BF_RUST_FETCH_DENSE", _NATIVE_CFG["dense"])


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        v = int(raw) if raw else default
        if v < 1:
            raise ValueError(v)
        return v
    except ValueError:
        _warn_once(f"env:{name}:{raw}", "%s=%r is not a positive integer; using %d",
                   name, raw, default)
        return default


def _warn_once(key: str, msg: str, *args) -> None:
    with _NATIVE_LOCK:
        if key in _NATIVE["warned"]:
            return
        _NATIVE["warned"].add(key)
    logger.warning(msg, *args)


def _native_client(region: str | None):
    """Return a process-local Rust fetch client for the given region.

    Creates clients lazily and reinitializes them after fork.
    Returns None if rust_fetch is unavailable.
    """
    try:
        import rust_fetch
    except ImportError as e:
        _warn_once("import", "native I/O is on but rust_fetch is not importable "
                   "(%s); reading with pyarrow. Install python/nova-bf/rust_io "
                   "(`make bf`) or set params.rust_fetch_dense/rust_fetch_columns: false", e)
        return None
    with _NATIVE_LOCK:
        if _NATIVE["pid"] != os.getpid():
            # A client inherited across fork() cannot be used in the child
            # (its runtime threads don't exist there; rust_fetch refuses and
            # leaks it on drop). Build fresh ones for this process.
            _NATIVE["orphans"].append(_NATIVE["clients"])
            _NATIVE["pid"], _NATIVE["clients"] = os.getpid(), {}
        conc = _env_int("NOVA_BF_RUST_FETCH_CONC", _NATIVE_CFG["concurrency"])
        part = _env_int("NOVA_BF_RUST_FETCH_PART_MIB", _NATIVE_CFG["part_mib"]) << 20
        # Keyed by the tuning too: a later run in this process with other
        # settings must not reuse a client built with the old ones.
        key = (region, conc, part)
        client = _NATIVE["clients"].get(key)
        if client is None:
            client = rust_fetch.Client(region=region, concurrency=conc, part_size=part)
            _NATIVE["clients"][key] = client
        return client


def _is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def _fs_and_path(uri: str) -> tuple[pafs.FileSystem, str]:
    if "://" in uri:
        return pafs.FileSystem.from_uri(uri)
    # bare local path (relative or absolute)
    return pafs.LocalFileSystem(), os.path.abspath(uri)


class Store:
    """A Parquet root backed by local storage or S3."""

    def __init__(self, uri: str, ranged_get: bool = False,
                 ranged_get_concurrency: int | None = None):
        self.uri = uri
        self.is_s3 = _is_s3(uri)
        self.ranged_get = ranged_get

        # Per-file GET concurrency; account for parallel file reads.
        self.ranged_get_concurrency = ranged_get_concurrency or _RANGED_GET_CONCURRENCY
        self.fs, self.root = _fs_and_path(uri)

        # Enable native reads only for supported local/S3 configurations.
        # Match PyArrow's resolved S3 region.
        self._native_ok, self._native_region = False, None
        self._native_failed = False
        if self.is_s3:
            from urllib.parse import parse_qs, urlsplit

            q = parse_qs(urlsplit(uri).query)
            self._native_ok = set(q) <= {"region"} and not any(
                os.environ.get(k) for k in _ENDPOINT_ENV)
            self._native_region = getattr(self.fs, "region", None) or (
                q["region"][-1] if "region" in q else None)
        elif "://" not in uri or uri.startswith("file://"):
            self._native_ok = True

    def _loader_key(self, read_path: str) -> str:
        # The loader's id `filename` is the object key (s3, no bucket) or the
        # absolute path (local). pyarrow's s3 path includes the bucket as the
        # first segment, so strip it to match.
        return read_path.split("/", 1)[1] if self.is_s3 else read_path

    def list_parquets(self, subpath: str | None = None) -> list[ParquetFile]:
        """
        Every `*.parquet` under the root (or root/subpath), sorted.
        """
        base = f"{self.root.rstrip('/')}/{subpath}" if subpath else self.root
        info = self.fs.get_file_info(base)
        if info.type == pafs.FileType.File:
            entries = [info]
        elif info.type == pafs.FileType.NotFound:
            return []
        else:
            entries = self.fs.get_file_info(pafs.FileSelector(base, recursive=True))
        out = [
            ParquetFile(read_path=e.path, key=self._loader_key(e.path), size=e.size)
            for e in entries
            if e.type == pafs.FileType.File and e.path.endswith(".parquet")
        ]
        out.sort(key=lambda f: f.read_path)
        return out

    def read_schema(self, read_path: str) -> pa.Schema:
        """The file's schema, footer only — no column data.

        Used to record what dtype the vectors were STORED as (see
        `results.provenance`), which the loaders can't report because they
        upcast to float32 on the way in.
        """
        return pq.read_schema(read_path, filesystem=self.fs)

    def read_columns(self, read_path: str, columns: list[str] | None,
                     ranged: bool | None = None) -> pa.Table:
        """`columns` of one parquet file.

        `ranged` overrides the store's `ranged_get` for this one call; `None`
        (the default) uses the store setting.

        WHY THE OVERRIDE EXISTS. Both paths below project `columns`, so they
        return the same table -- they differ only in how the bytes arrive.
        Ranged-GET buffers the WHOLE file (`_ranged_download` allocates the
        file's full size) and then parses the wanted columns out of it, which
        is the right trade when the caller wants most of the file's bytes, as
        the vector scan does. It is the wrong trade for a narrow column: the
        `tiebreak='id'` startup pass wants one id column, ~180 KB of a ~4.3 GB
        pubmed shard, and buying all 4.3 GB to keep 0.004% of it cost 1055 s
        per rank and (at the default 32-way fan-out) ~137 GB of resident
        buffers, which OOM-killed all 16 ranks of a pubmed GT run on
        2026-09-22 before a single corpus vector was scored.

        Note what the ranged branch gates on: FILE size. That answers "is this
        file big?" when the question that decides the trade is "are we reading
        most of it?" -- so a caller reading one narrow column has to opt out
        here rather than rely on the size gate to do the right thing.
        """
        use_ranged = self.ranged_get if ranged is None else ranged
        if use_ranged:
            size = self.fs.get_file_info(read_path).size
            if size is not None and size >= _RANGED_GET_MIN_BYTES:
                return pq.read_table(
                    pa.BufferReader(self._ranged_download(read_path, size)),
                    columns=columns,
                )
        # Arrow-native read with column projection, parallel I/O, and decoding.
        # Faster for files with many row groups; use io_ranged_get for few row groups.
        return pq.read_table(read_path, filesystem=self.fs, columns=columns,
                             pre_buffer=True, use_threads=True)

    def read_dense(self, read_path: str, column: str,
                   size: int | None = None) -> np.ndarray | None:
        r = self.read_dense_versioned(read_path, column)
        return None if r is None else r[0]

    def same_version(self, read_path: str, etag: str | None) -> bool:
        """Check whether the object still matches the given EntityTag to prevent stale reads.

        Returns True if no ETag is provided, or False if verification fails.
        """
        if etag is None:
            return True
        if not _native_enter():
            return False  # shutting down: let pyarrow re-read, never call in
        try:
            client = _native_client(self._native_region)
            if client is None:
                return False
            url = f"s3://{read_path}" if self.is_s3 else read_path
            try:
                return client.head(url)[1] == etag
            except Exception:
                return False
        finally:
            _native_exit()

    def read_dense_versioned(self, read_path: str, column: str):
        """Read a dense column as a contiguous array using the native reader.

        Returns (array, ETag), or None to fall back to PyArrow.
        Preserves bitwise equivalence; transport errors propagate.
        """
        if self._native_failed:
            return None
        if not (self._native_ok and rust_fetch_dense_enabled()):
            if rust_fetch_dense_enabled() and not self._native_ok:
                _warn_once(f"store:{self.uri}", "native dense read: %s is not a "
                        "plain s3:// or local root, or an AWS_ENDPOINT_URL* override "
                        "is set; reading with pyarrow", self.uri)
            return None
        client = _native_client(self._native_region)
        if client is None:
            return None
        import rust_fetch

        url = f"s3://{read_path}" if self.is_s3 else read_path

        # Defer special fields and Hive-partitioned paths to PyArrow
        # to preserve dataset API semantics.
        if column.startswith("__") or any(
                "=" in seg for seg in read_path.split("/")[:-1]):
            return None
        if not _native_enter():
            return None  # Interpreter shutting down.
        try:
            return self._read_dense_native(client, rust_fetch, read_path, url, column)
        finally:
            _native_exit()

    def _read_dense_native(self, client, rust_fetch, read_path, url, column):
        try:
            # Recheck object size and validate PyArrow schema compatibility.
            plan = client.dense_plan(url, column, None)
            try:
                schema = pq.read_schema(read_path, filesystem=self.fs)
                typ = schema.field(column).type
            except Exception as e:
                raise rust_fetch.UnsupportedLayout(f"pyarrow cannot read the schema: {e}") from e
            # pyarrow's dataset read (the fallback) refuses ANY duplicated
            # top-level name, even one that is not read -- refuse too.
            if len(set(schema.names)) != len(schema.names):
                raise rust_fetch.UnsupportedLayout("duplicate top-level field names")

            plain = (
                (pa.types.is_list(typ) or pa.types.is_large_list(typ)
                or (pa.types.is_fixed_size_list(typ)
                    # A 0-row file has no dim to compare, but pyarrow still
                    # refuses a negative ARROW:schema list size on read.
                    and typ.list_size >= 0
                    and (plan.n_rows == 0 or typ.list_size == plan.dim)))
                and not isinstance(typ, pa.ExtensionType)
                and typ.value_type == pa.from_numpy_dtype(np.dtype(plan.dtype))
            )
            if not plain:
                raise rust_fetch.UnsupportedLayout(f"pyarrow reads {column!r} as {typ}")
            if plan.n_rows == 0:
                return np.zeros((0, 0), dtype=np.float32), plan.etag

            out = np.empty((plan.n_rows, plan.dim), dtype=plan.dtype)
            client.read_dense_into(plan, out.reshape(-1).view(np.uint8),
                                _env_int("NOVA_BF_RUST_FETCH_CHUNKS", _NATIVE_CFG["chunks"]))
            return out, plan.etag

        except (ValueError, MemoryError, rust_fetch.CorruptParquet) as e:
            # Fall back for unsupported layouts, invalid sizes, or corrupt data.
            kind = type(e).__name__
            _warn_once(f"fallback:{kind}:{str(e).replace(url, '')}", "native dense read "
                    "of %s fell back to pyarrow: %s: %s (further files with this "
                    "reason are not logged)", read_path, kind, e)
            return None

        except FileNotFoundError:
            return None  # Let PyArrow report missing files.

        except OSError as e:
            if _INFLIGHT["closing"]:
                return None

            # Disable native reads after transport failure; use PyArrow thereafter.
            self._native_failed = True
            logger.warning("native dense transport failed on %s (%s); using pyarrow "
                        "for the rest of this run on %s", read_path, e, self.uri)
            return None

    def read_columns_fetched(self, read_path: str, columns: list[str],
                         etag: str | None = None):
        """Fetch selected Parquet columns natively and decode with PyArrow.

        Returns (table, ETag), or None to fall back to standard reads.
        Uses version-pinned ranged GETs and lazily allocated buffers.
        """
        if self._native_failed or not (self._native_ok and rust_fetch_columns_enabled()):
            return None

        # Preserve PyArrow dataset semantics for special fields and partitions.
        if any(c.startswith("__") for c in columns) or any(
                "=" in seg for seg in read_path.split("/")[:-1]):
            return None
        client = _native_client(self._native_region)
        if client is None:
            return None
        import rust_fetch

        url = f"s3://{read_path}" if self.is_s3 else read_path
        if not _native_enter():
            return None
        try:
            try:
                size, now = client.head(url)
                if etag is not None and now != etag:
                    return None
                pin = now
                got = _fetch_image(client, url, size, pin)
                if got is None:
                    return None
                image, have_from = got

                # Fetch requested column chunks after reading the footer.
                ranges = _column_ranges(image, size, columns, have_from)
                if ranges is None:
                    return None
                if ranges:
                    client.read_ranges_into(url, image, ranges, if_match=pin)

            except (ValueError, MemoryError, rust_fetch.CorruptParquet) as e:
                kind = type(e).__name__
                _warn_once(f"fetch:{kind}:{str(e).replace(url, '')}", "native fetch of %s "
                        "fell back to pyarrow: %s: %s", read_path, kind, e)
                return None
            except FileNotFoundError:
                return None
            except OSError as e:
                if _INFLIGHT["closing"]:
                    return None
                if "object changed" in str(e):
                    return None
                self._native_failed = True
                logger.warning("native fetch transport failed on %s (%s); using pyarrow "
                            "for the rest of this run on %s", read_path, e, self.uri)
                return None

            buf = pa.py_buffer(image)
            try:
                table = pq.read_table(pa.BufferReader(buf), columns=columns, use_threads=True)
            except Exception:
                return None
            finally:
                # Retain buffers to avoid destruction on Arrow threads during shutdown.
                _FETCH_KEEPALIVE.append(buf)

            # Reclaim image pages when decoded data no longer references them.
            _release_image(image, table)
            return table, pin
        finally:
            _native_exit()

    def _ranged_download(self, read_path: str, size: int):
        """Download a file using concurrent byte-range reads on S3 or POSIX.

        Limits per-file concurrency using the shared fetch pool. Copies release
        the GIL; failed reads propagate after in-flight requests finish.
        Temporarily requires one file's size in host RAM.
        """
        data = np.empty(size, dtype=np.uint8)
        width = max(1, self.ranged_get_concurrency)
        pool = _fetch_pool()
        with self.fs.open_input_file(read_path) as f:
            def fetch(lo: int) -> None:
                hi = min(lo + _RANGED_GET_BYTES, size)
                _copy_range(data[lo:hi], f.read_at(hi - lo, lo))

            pending: set = set()
            try:
                for lo in range(0, size, _RANGED_GET_BYTES):
                    if len(pending) >= width:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in done:
                            fut.result()
                    pending.add(pool.submit(fetch, lo))
                done, pending = wait(pending)
                for fut in done:
                    fut.result()
            finally:
                wait(pending)
        return pa.py_buffer(data)

    def write(self, filename: str, table: pa.Table) -> str:
        """
        Write a table to root/filename (creating local parent dirs).
        """
        path = self._prepare_write(filename)
        with self.fs.open_output_stream(path) as sink:
            pq.write_table(table, sink, compression="snappy")
        return path

    def write_bytes(self, filename: str, data: bytes) -> str:
        """Write raw bytes to root/filename — the run manifest (see manifest.py).

        Same path/parent-dir handling as `write`, so a manifest lands beside the
        parquets it describes whether the root is local or s3://.
        """
        path = self._prepare_write(filename)
        with self.fs.open_output_stream(path) as sink:
            sink.write(data)
        return path

    def _prepare_write(self, filename: str) -> str:
        path = f"{self.root.rstrip('/')}/{filename}"
        if not self.is_s3:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


def dense_to_2d(col) -> np.ndarray:
    """Convert an Arrow list column to a contiguous, NumPy-owned (n, dim) array.

    Validates uniform dimensions and non-null rows. Preserves float16/float32
    to avoid unnecessary CPU conversion and PCIe traffic; other types use
    float32. Copies release the GIL, avoiding GPU stalls from Arrow buffer
    cleanup. Float16 conversion to float32 occurs on the GPU.
    """
    chunks = list(col.chunks) if isinstance(col, pa.ChunkedArray) else [col]
    n = sum(len(c) for c in chunks)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    typ = chunks[0].type
    fixed = pa.types.is_fixed_size_list(typ)
    if not (fixed or pa.types.is_list(typ) or pa.types.is_large_list(typ)):
        raise TypeError(f"dense column must be a list of floats, got {typ}")
    parts: list[tuple[int, np.ndarray]] = []
    dim = typ.list_size if fixed else None
    for ch in chunks:
        if len(ch) == 0:
            continue
        if ch.null_count:
            raise ValueError(
                f"dense column has {ch.null_count} null row(s); every corpus "
                "and query row needs a vector"
            )
        if not fixed:
            # Uniform width, checked: O(rows), not O(values).
            lens = np.diff(ch.offsets.to_numpy(zero_copy_only=False))
            if dim is None:
                dim = int(lens[0])
            if not (lens == dim).all():
                raise ValueError(
                    f"dense column is ragged: list lengths {int(lens.min())}.."
                    f"{int(lens.max())}, a dense vector needs one width"
                )
        # `flatten` honours a sliced chunk's logical offset for both list kinds.
        parts.append((len(ch), ch.flatten().to_numpy(zero_copy_only=False)))
    src = parts[0][1].dtype
    dtype = src if src in (np.dtype(np.float16), np.dtype(np.float32)) else np.dtype(np.float32)
    out = np.empty((n, dim), dtype=dtype)
    r = 0
    for rows, flat in parts:
        if flat.size != rows * dim:
            raise ValueError(
                f"dense column chunk holds {flat.size} values for {rows} rows "
                f"of width {dim}"
            )
        np.copyto(out[r:r + rows], flat.reshape(rows, dim), casting="unsafe")
        r += rows
    return out


def _stitch_ragged_chunks(chunks: list) -> tuple[np.ndarray, np.ndarray]:
    """Decode each chunk on its own and concatenate — see `multivector_to_ragged`.

    Every chunk goes through the single-array path, so all of that path's
    validity/slice/shape handling applies per chunk. Only two things are joined
    here: the per-document token counts, re-prefix-summed into int64 offsets,
    and the float32 token matrix.

    A width disagreement between chunks is left to `np.concatenate` to reject.
    The decoder upstream guarantees a consistent token dimension, and this
    function does not know which FILE it is decoding — so it could only say
    "chunk 3 disagrees with chunk 0", which is no more useful than numpy
    naming the two shapes.
    """
    counts_parts: list[np.ndarray] = []
    mats: list[np.ndarray] = []
    dim = 0
    for chunk in chunks:
        off, flat = multivector_to_ragged(chunk)
        counts_parts.append(np.diff(off))
        if flat.shape[1] > 0:
            dim = flat.shape[1]
        mats.append(flat)
    counts = (np.concatenate(counts_parts) if counts_parts
              else np.zeros(0, dtype=np.int64))
    doc_offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    # A chunk of only null/empty docs decodes to a (0, 0) matrix whose zero
    # width would poison the concat, so drop the empties. `dim` is then used
    # only if EVERY chunk was empty, to keep the returned width meaningful.
    nonempty = [m for m in mats if m.shape[0] > 0]
    flat_tokens = (np.concatenate(nonempty, axis=0) if nonempty
                   else np.zeros((0, dim), dtype=np.float32))
    return doc_offsets, flat_tokens


def multivector_to_ragged(col: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
    """A `list<list<float32>>` column (one doc = outer entry, one D-dim token
    vector = inner entry) → `(doc_offsets, flat_tokens)`.

    Same flatten-the-Arrow-buffer-once approach as `dense_to_2d`/
    `sparse_to_coo_parts` — no per-row/per-token Python. `doc_offsets` is
    length n+1 (token-index prefix sums; doc `i`'s tokens are rows
    `doc_offsets[i]:doc_offsets[i+1]` of `flat_tokens`), `flat_tokens` is the
    `(total_tokens, D)` float32 matrix of every token across every doc,
    concatenated in doc order. This is the exact ragged analog of CSR's
    `(row_offsets, values)`.

    A null outer entry (nova-embed's `on_empty_input="null"`) or a non-null
    but empty inner list both decode to a zero-width span (`doc_offsets[i] ==
    doc_offsets[i+1]`) — i.e. a zero-token doc, which the compute path treats
    as a non-candidate (`-inf`), the same way the sparse path treats a
    zero-overlap doc.

    Robustness (all O(1) / O(n_docs), so the hot corpus path is unaffected):
    - The validity bitmap, not the offsets, decides a null doc's token count.
      Arrow does NOT guarantee a null list slot has equal adjacent offsets, so
      trusting the offsets alone could count a null doc's stray physical span
      as real tokens (nova-embed writes zero-span nulls, so this only guards
      against arrays from other producers / some slice+concat paths).
    - Buffer offsets (outer/inner) are honored so a sliced Arrow array (logical
      offset != 0) decodes correctly rather than misaligning tokens to docs.
    - Wrong-shape input fails loudly: a dense `list<float32>` column, or token
      floats containing nulls (which would silently become NaN and poison
      scoring), raise a clear error instead of an obscure one downstream.

    The token width D is taken from the first token; per-token width variance
    is left to `reshape` to catch (an O(total_tokens) uniformity scan would tax
    the corpus path, and nova-embed emits a uniform width by construction).

    Chunks are decoded INDIVIDUALLY and stitched in numpy."""
    chunks = list(col.chunks) if isinstance(col, pa.ChunkedArray) else [col]
    if len(chunks) > 1:
        return _stitch_ragged_chunks(chunks)
    col = chunks[0] if chunks else col.combine_chunks()
    n = len(col)
    if n == 0:
        return np.zeros(1, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    if not (pa.types.is_list(col.type) or pa.types.is_large_list(col.type)):
        raise TypeError(f"multivector column must be a list of token vectors, got {col.type}")
    inner = col.values  # child ListArray: one entry per token across all docs
    if not (pa.types.is_list(inner.type) or pa.types.is_large_list(inner.type)):
        raise TypeError(
            f"multivector column must nest list<float32> token vectors; inner type is "
            f"{inner.type} (a flat list<float32> is a DENSE vector — use vector_type=dense)"
        )
    outer_off = col.offsets.to_numpy(zero_copy_only=False).astype(np.int64)  # (n+1,), token indices
    lengths = np.diff(outer_off)  # physical span (token count) per doc
    interspersed = False
    if col.null_count:
        valid = np.asarray(col.is_valid())  # (n,) bool
        interspersed = not bool((lengths[~valid] == 0).all())  # a null slot carrying real tokens?
        lengths = np.where(valid, lengths, 0)  # a null doc is always zero-token
    doc_offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    if int(doc_offsets[-1]) == 0:
        # every doc null/empty — no tokens, so D is unknowable from the data
        return doc_offsets, np.zeros((0, 0), dtype=np.float32)
    if inner.values.null_count:  # O(1) when there's no validity bitmap (the norm)
        raise ValueError(
            "multivector token values contain nulls — a token vector must be fully "
            "populated float32 (a null would decode to NaN and poison scoring)"
        )
    inner_off = inner.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    tok_lo, tok_hi = int(outer_off[0]), int(outer_off[-1])
    dim = int(inner_off[tok_lo + 1] - inner_off[tok_lo])  # token width, from the first token
    val_lo, val_hi = int(inner_off[tok_lo]), int(inner_off[tok_hi])
    phys = inner.values.to_numpy(zero_copy_only=False)[val_lo:val_hi].reshape(tok_hi - tok_lo, dim)
    if interspersed:  # rare: drop the stray tokens a null slot physically carried
        phys = phys[np.repeat(valid, np.diff(outer_off))]
    return doc_offsets, np.ascontiguousarray(phys, dtype=np.float32)


def sparse_to_coo_parts(col: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A `struct<indices: list<uint32>, values: list<float32>>` column → CSR parts.

    Same flatten-the-Arrow-buffer-once approach as `dense_to_2d`: no per-row
    Python conversion. Returns `(row_offsets, indices, values)` — `row_offsets`
    is length n+1 (CSR `crow_indices`), `indices`/`values` are the flat,
    concatenated-across-rows nonzero entries (CSR `col_indices`/`values`).

    `indices` keeps the STORED integer width — never widened, never copied.
    `values` is float32: `copy=False` passes a float32-stored column straight
    through, and a float64-stored one is NARROWED here (scoring is float32
    throughout, so carrying float64 further would only cost bandwidth).

    It used to widen `indices` to int64 and re-`astype` `values`
    unconditionally, which on a fineweb corpus file (~220 M nonzeros)
    allocated 1.76 GB for the widening and copied 0.88 GB of float32 onto
    itself, in the reader thread, once per file. The int64 width was never
    needed: `_vocab_lookup` indexes its table with whatever integer dtype it
    is handed, and torch's CSR builder is fed the REMAPPED column ids
    (`compute._remap_sparse_file`), not these.

    Consequently callers must not assume a dtype. Everything downstream is
    dtype-agnostic on purpose: `_build_query_vocab` (`np.unique`),
    `_vocab_lookup` (LUT gather or `searchsorted`), `_zero_gate_file_ok`
    (comparisons), `_sparse_file_norms` (accumulates in float64 explicitly).
    """
    col = col.combine_chunks()
    n = len(col)
    if n == 0:
        return np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32)
    idx_list = col.field("indices")
    val_list = col.field("values")
    row_offsets = idx_list.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    indices = idx_list.values.to_numpy(zero_copy_only=False)
    values = val_list.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return row_offsets, indices, values
