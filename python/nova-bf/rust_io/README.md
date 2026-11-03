# rust_fetch (nova-bf native I/O)

A small Rust extension (pyo3 + object_store + tokio) that moves nova-bf's
corpus reads off pyarrow's I/O path. Python import name: `rust_fetch`.

## Why

On an A10G node, nova-bf's Pes2o run is limited by reading, not by the GPU.
pyarrow's S3 transport tops out around 1–1.8 GB/s (with GIL stalls) and its
parse + `dense_to_2d` copy around 2 GB/s, while the 25 Gbps NIC delivers
~3 GB/s. This crate reaches NIC line rate on S3 and decodes the dense vector
column at 6–8 GB/s, GIL-free.

## What it does

- **Transport.** Concurrent ranged GETs (S3 or local files), written straight
  into a caller-provided buffer with the GIL released, with retries, timeouts,
  cancellation (Ctrl-C / `Client.cancel()`), and every multi-GET read pinned to
  one object version (ETag / If-Match).
- **Dense decode.** For a `list<float32|float16>` vector column: fetch only
  that column's chunks and decode the pages straight into a row-major
  `(n, dim)` array — output bytes identical to pyarrow's. Any layout it cannot
  reproduce exactly (nulls, ragged rows, other dtypes, Brotli, malformed or
  ambiguous metadata, ...) raises `UnsupportedLayout` and the caller falls back
  to pyarrow.

## How nova-bf uses it

Both paths are on by default and fall back to pyarrow on any doubt. They are
set in the config's `params:` block; the env var overrides it for one run
(`=0` turns a path off, `=1` on):

| `params` key (default) | Env override | Effect |
|---|---|---|
| `rust_fetch_dense` (true) | `NOVA_BF_RUST_FETCH_DENSE` | The dense corpus column is downloaded and decoded in Rust. |
| `rust_fetch_columns` (true) | `NOVA_BF_RUST_FETCH_COLUMNS` | The other corpus columns (sparse, ids, filter/date fields) and the `tiebreak='id'` id pass are *downloaded* in Rust (footer + needed column chunks only); pyarrow still decodes them. |
| `rust_fetch_concurrency` (128) | `NOVA_BF_RUST_FETCH_CONC` | Ranged GETs in flight per process, shared by all files. |
| `rust_fetch_part_mib` (16) | `NOVA_BF_RUST_FETCH_PART_MIB` | Bytes per ranged GET. |
| `rust_fetch_chunks` (8) | `NOVA_BF_RUST_FETCH_CHUNKS` | Row groups fetched/decoded at once per file. |

If `rust_fetch` is not installed, nova-bf logs one warning and reads with
pyarrow. The resolved settings are recorded in the run record under
`rust_fetch`.

What still goes through pyarrow: listing the corpus, loading the query file,
writing partials and manifests, `nova-bf merge`, the per-file schema check, and
every fallback.

Outputs with these on are byte-identical to runs with them off; see
`python/nova-bf/tests/test_native_dense.py`.

## API sketch

```python
import numpy as np, rust_fetch

c = rust_fetch.Client(region=None, concurrency=64, part_size=16 << 20)
size, etag = c.head("s3://bucket/key")
c.read_ranges_into("s3://bucket/key", buf, [(src_off, length, dst_off), ...], if_match=etag)

out = rust_fetch.read_dense(c, "s3://bucket/part-0.parquet", "embedding")  # (n, dim) array
```

Errors: `UnsupportedLayout` (a ValueError: use pyarrow instead),
`CorruptParquet` (an OSError), `FileNotFoundError`, `OSError` for transport
failures. One `Client` per process; it is thread-safe but not usable across
`fork()`.

## Caveats

- The raw crate does not parse the footer's `ARROW:schema` metadata. Callers
  that need pyarrow equivalence must cross-check `pq.read_schema` (nova-bf does).
- If the interpreter exits while a daemon thread is inside a call, CPython
  aborts the process; join threads, or cancel and drain in an atexit hook
  (nova-bf does).
- A hung local filesystem is not bounded by `part_timeout_s` (a local read is
  never abandoned mid-write).
- Linux only.

## Build and test

`make bf` installs it alongside nova-bf (editable; needs cargo), and nova-dist's
bf workers install it with `uv tool install --with 'rust_fetch @ git+...#subdirectory=python/nova-bf/rust_io'`
(see `configs/skypilot/bf.yaml`). `make test` runs both test suites below.

```bash
uv pip install -e python/nova-bf/rust_io                      # from the repo root
cargo test --manifest-path python/nova-bf/rust_io/Cargo.toml --no-default-features
uv run --directory python/nova-bf --extra dev --with ./rust_io pytest -q rust_io/tests
# S3 cases skip unless RUST_FETCH_TEST_S3_ENDPOINT points at a minio
```

Layout: `src/core.rs` transport, `src/dense.rs` parquet dense decoder,
`src/python.rs` bindings, `tests/` (incl. hand-crafted adversarial parquet files).
