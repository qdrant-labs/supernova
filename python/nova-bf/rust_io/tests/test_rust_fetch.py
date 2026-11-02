"""rust_fetch tests. Local (file://) always; S3 against a local S3-compatible
server (minio) when RUST_FETCH_TEST_S3_ENDPOINT is set, e.g.

    docker run -d --name rust-fetch-minio -p 19000:9000 \
        -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
        quay.io/minio/minio server /data
    RUST_FETCH_TEST_S3_ENDPOINT=http://localhost:19000 pytest -q tests/
"""

import mmap
import os
import threading
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import rust_fetch

S3_ENDPOINT = os.environ.get("RUST_FETCH_TEST_S3_ENDPOINT")
S3_BUCKET = "rust-fetch-test"


def rand_bytes(n: int, seed: int = 0) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("rf")
    out = {}
    for name, n in [("empty", 0), ("one", 1), ("small", 1000),
                    ("odd", 3 * (1 << 20) + 12345), ("big", 40 * (1 << 20) + 7)]:
        p = d / f"{name}.bin"
        b = rand_bytes(n, seed=n)
        p.write_bytes(b)
        out[name] = (str(p), b)
    return out


@pytest.fixture(scope="module")
def s3(files):
    """{name: (s3 url, bytes)} uploaded to minio, or skip."""
    if not S3_ENDPOINT:
        pytest.skip("RUST_FETCH_TEST_S3_ENDPOINT not set")
    from pyarrow import fs as pafs
    host = S3_ENDPOINT.split("://", 1)[1]
    s3fs = pafs.S3FileSystem(access_key="minioadmin", secret_key="minioadmin",
                             endpoint_override=host, scheme="http", region="us-east-1", allow_bucket_creation=True)
    s3fs.create_dir(S3_BUCKET)
    out = {}
    for name, (_, b) in files.items():
        with s3fs.open_output_stream(f"{S3_BUCKET}/data/{name}.bin") as f:
            f.write(b)
        out[name] = (f"s3://{S3_BUCKET}/data/{name}.bin", b)
    return out


def s3_client(**kw):
    os.environ.update(AWS_ACCESS_KEY_ID="minioadmin", AWS_SECRET_ACCESS_KEY="minioadmin")
    return rust_fetch.Client(region="us-east-1", endpoint=S3_ENDPOINT, allow_http=True, **kw)


# ---------------------------------------------------------------- core reads

PART_CONC = [(1 << 10, 1), (4096, 7), (65536, 64), (1 << 20, 3), (16 << 20, 64), (1 << 30, 2)]


def check_full_and_partial(client, objs):
    for name, (url, b) in objs.items():
        assert client.size(url) == len(b)
        dest = np.empty(len(b) + 5, dtype=np.uint8)
        dest[-5:] = 0xEE
        assert client.read_into(url, dest) == len(b)
        assert dest[: len(b)].tobytes() == b, name
        assert (dest[-5:] == 0xEE).all(), "wrote past the requested length"
        if len(b) > 10:
            for off, ln in [(0, 1), (1, len(b) - 1), (len(b) // 3, len(b) // 3), (len(b) - 1, 1)]:
                d = bytearray(ln)
                assert client.read_into(url, d, offset=off, length=ln) == ln
                assert bytes(d) == b[off:off + ln], (name, off, ln)
            # length=None from an offset reads to EOF
            d = np.zeros(len(b), np.uint8)
            assert client.read_into(url, d, offset=7) == len(b) - 7
            assert d[: len(b) - 7].tobytes() == b[7:]


@pytest.mark.parametrize("part,conc", PART_CONC)
def test_local_full_and_partial(files, part, conc):
    if part <= 4096 and conc == 1:
        files = {k: v for k, v in files.items() if k != "big"}  # 40k sequential parts: slow, not useful
    c = rust_fetch.Client(part_size=part, concurrency=conc)
    check_full_and_partial(c, {k: ("file://" + p, b) for k, (p, b) in files.items()})


def test_plain_absolute_path(files):
    p, b = files["small"]
    d = bytearray(len(b))
    rust_fetch.Client().read_into(p, d)
    assert bytes(d) == b


def test_buffer_kinds(files):
    p, b = files["odd"]
    c = rust_fetch.Client(part_size=1 << 20)
    # numpy, bytearray, memoryview slice, anonymous mmap, pyarrow mutable buffer
    np_dest = np.empty(len(b), np.uint8)
    ba = bytearray(len(b) + 100)
    mv = memoryview(ba)[50:50 + len(b)]
    mm = mmap.mmap(-1, len(b))
    pa_buf = pa.allocate_buffer(len(b))  # mutable
    for dest, view in [(np_dest, lambda: np_dest.tobytes()), (mv, lambda: bytes(mv)),
                       (mm, lambda: mm[:]), (np.frombuffer(pa_buf, np.uint8), lambda: pa_buf.to_pybytes())]:
        assert c.read_into("file://" + p, dest) == len(b)
        assert view() == b
    assert ba[:50] == bytes(50) and ba[50 + len(b):] == bytes(50)
    mm.close()


def test_read_ranges_into(files):
    p, b = files["big"]
    c = rust_fetch.Client(part_size=1 << 20, concurrency=16)
    ranges = [(len(b) - 100, 100, 0), (5, 3 << 20, 1000), (12345, 1, 200), (0, 0, 999)]
    dest = np.full((3 << 20) + 1000, 0xAA, np.uint8)
    assert c.read_ranges_into("file://" + p, dest, ranges) == 100 + (3 << 20) + 1
    assert dest[:100].tobytes() == b[-100:]
    assert dest[200] == b[12345]
    assert (dest[101:200] == 0xAA).all() and (dest[201:1000] == 0xAA).all()
    assert dest[1000:].tobytes() == b[5:5 + (3 << 20)]


# ---------------------------------------------------------------- validation

def test_dest_validation(files):
    p, b = files["small"]
    url = "file://" + p
    c = rust_fetch.Client()
    with pytest.raises(ValueError, match="read-only"):
        c.read_into(url, b"\0" * len(b))
    ro = np.zeros(len(b), np.uint8)
    ro.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        c.read_into(url, ro)
    with pytest.raises(ValueError, match="too small"):
        c.read_into(url, bytearray(len(b) - 1))
    with pytest.raises(ValueError, match="too small"):
        c.read_into(url, bytearray(10), offset=0, length=11)
    with pytest.raises(ValueError, match="contiguous"):
        c.read_into(url, np.zeros(2 * len(b), np.uint8)[::2])
    with pytest.raises(TypeError, match="byte buffer"):
        c.read_into(url, np.zeros(len(b), np.float32))
    with pytest.raises(TypeError):
        c.read_into(url, "not a buffer")
    with pytest.raises(ValueError, match="overlap"):
        c.read_ranges_into(url, bytearray(100), [(0, 10, 0), (20, 10, 5)])
    with pytest.raises(ValueError, match="too small"):
        c.read_ranges_into(url, bytearray(100), [(0, 10, 95)])
    with pytest.raises(ValueError):
        rust_fetch.Client(concurrency=0)
    with pytest.raises(ValueError):
        rust_fetch.Client(part_size=0)


def test_errors(files, tmp_path):
    p, b = files["small"]
    c = rust_fetch.Client(retries=1)
    with pytest.raises(FileNotFoundError):
        c.read_into(f"file://{tmp_path}/missing.bin", bytearray(10), length=10)
    with pytest.raises(FileNotFoundError):
        c.size(f"file://{tmp_path}/missing.bin")
    with pytest.raises(OSError, match="shorter than requested"):
        c.read_into("file://" + p, bytearray(2000), length=1500)
    with pytest.raises(ValueError, match="past the end"):
        c.read_into("file://" + p, bytearray(10), offset=len(b) + 1)
    with pytest.raises(ValueError, match="unsupported URL"):
        c.read_into("gs://x/y", bytearray(1), length=1)
    assert c.read_into("file://" + files["empty"][0], bytearray(0)) == 0


# ---------------------------------------------------------------- GIL release

def gil_probe(fn, tick=0.001):
    """Run fn() while a Python thread ticks; return (fn result, max tick gap s)."""
    stop = threading.Event()
    gaps = []

    def ticker():
        last = time.perf_counter()
        while not stop.is_set():
            time.sleep(tick)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    t = threading.Thread(target=ticker)
    t.start()
    time.sleep(0.05)
    try:
        r = fn()
    finally:
        stop.set()
        t.join()
    return r, max(gaps), len(gaps)


@pytest.fixture(scope="module")
def huge(tmp_path_factory):
    n = int(os.environ.get("RUST_FETCH_TEST_GIL_BYTES", 1 << 30))
    p = tmp_path_factory.mktemp("gil") / "huge.bin"
    blk = rand_bytes(1 << 24, seed=1)
    with open(p, "wb") as f:
        for i in range(0, n, len(blk)):
            f.write(blk[: min(len(blk), n - i)])
    return str(p), n


def test_gil_released_during_large_read(huge):
    p, n = huge
    c = rust_fetch.Client(part_size=8 << 20, concurrency=32)
    dest = np.empty(n, np.uint8)
    c.read_into("file://" + p, dest)  # warm page cache
    t0 = time.perf_counter()
    (got, max_gap, ticks) = gil_probe(lambda: c.read_into("file://" + p, dest))
    dt = time.perf_counter() - t0
    assert got == n
    print(f"\nrust_fetch {n / 2**30:.2f} GiB local: {n / dt / 1e9:.2f} GB/s, "
          f"max GIL-probe gap {max_gap * 1e3:.2f} ms over {ticks} ticks")
    assert max_gap < 0.010, f"Python thread stalled {max_gap * 1e3:.1f} ms"
    with open(p, "rb") as f:
        head = f.read(1 << 24)
    assert dest[: 1 << 24].tobytes() == head


def test_gil_probe_detects_holding_copy(huge):
    """Control: the probe DOES see the stalls of the old pattern (a
    `view[lo:hi] = f.read_at(...)` copy under the GIL), so the test above is
    meaningful. Informational: asserts only that the probe ran."""
    from concurrent.futures import ThreadPoolExecutor
    from pyarrow import fs as pafs
    p, n = huge
    lfs = pafs.LocalFileSystem()
    data = np.empty(n, np.uint8)
    view = memoryview(data)
    step = 64 << 20

    def old():
        with lfs.open_input_file(p) as f:
            def fetch(lo):
                hi = min(lo + step, n)
                view[lo:hi] = f.read_at(hi - lo, lo)
            with ThreadPoolExecutor(32) as pool:
                for fu in [pool.submit(fetch, lo) for lo in range(0, n, step)]:
                    fu.result()
    _, max_gap, ticks = gil_probe(old)
    print(f"\nold pyarrow read_at pattern: max GIL-probe gap {max_gap * 1e3:.2f} ms over {ticks} ticks")
    assert ticks > 0


# ---------------------------------------------------------------- parquet / buffer reuse

def test_reused_buffer_parquet_tables_do_not_alias(tmp_path):
    """The README's pattern reuses ONE buffer across files. That is only safe if
    pq.read_table copies out of the input buffer -- check, for uncompressed
    PLAIN data (the case most likely to be zero-copy), that a table survives
    the buffer being overwritten."""
    c = rust_fetch.Client(part_size=1 << 16)
    paths = []
    for i, comp in enumerate(["none", "snappy"]):
        t = pa.table({"id": pa.array(np.arange(100_000) + i * 10**6),
                      "v": pa.FixedSizeListArray.from_arrays(
                          pa.array(np.random.default_rng(i).random(100_000 * 8, dtype=np.float32)), 8)})
        path = str(tmp_path / f"{i}.parquet")
        pq.write_table(t, path, compression=comp, use_dictionary=False)
        paths.append((path, t))
    buf = np.empty(max(os.path.getsize(p) for p, _ in paths), np.uint8)
    tables = []
    for path, _ in paths:
        n = c.read_into("file://" + path, buf)
        tables.append(pq.read_table(pa.BufferReader(pa.py_buffer(buf[:n]))))
        if len(tables) == 1:
            buf[:] = 0xFF  # clobber before reading the next file
    for (path, expect), got in zip(paths, tables):
        assert got.equals(expect), path


# ---------------------------------------------------------------- S3 (minio)

@pytest.mark.parametrize("part,conc", [(1 << 16, 8), (1 << 20, 64), (5 << 20, 3), (64 << 20, 4)])
def test_s3_full_and_partial(s3, part, conc):
    check_full_and_partial(s3_client(part_size=part, concurrency=conc), s3)


def test_s3_ranges_and_errors(s3):
    c = s3_client(part_size=1 << 20, retries=1)
    url, b = s3["big"]
    dest = np.zeros(3000 + (2 << 20), np.uint8)
    c.read_ranges_into(url, dest, [(len(b) - 3000, 3000, 0), (10, 2 << 20, 3000)])
    assert dest[:3000].tobytes() == b[-3000:]
    assert dest[3000:].tobytes() == b[10:10 + (2 << 20)]
    with pytest.raises(FileNotFoundError):
        c.read_into(f"s3://{S3_BUCKET}/nope.bin", bytearray(10), length=10)
    with pytest.raises(FileNotFoundError):
        c.size(f"s3://{S3_BUCKET}/nope.bin")
    small_url, small = s3["small"]
    with pytest.raises(OSError):
        c.read_into(small_url, bytearray(5000), length=5000)


def test_s3_threads_share_client(s3):
    """Several Python threads on one client (nova-bf's io_workers pattern)."""
    from concurrent.futures import ThreadPoolExecutor
    c = s3_client(part_size=1 << 20, concurrency=16)
    url, b = s3["big"]

    def one(_):
        d = np.empty(len(b), np.uint8)
        c.read_into(url, d)
        return d.tobytes() == b
    with ThreadPoolExecutor(4) as pool:
        assert all(pool.map(one, range(8)))
