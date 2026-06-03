# Corpus Layout Convention

The same layout applies whether you write a corpus to S3, HuggingFace Storage Buckets, or local disk — only the URI scheme changes. This page uses S3 paths in examples; substitute `hf://buckets/ns/name/` (buckets are flat — no `data/` subdir) or `file:///abs/path/` for the other backends.

Every supernova corpus lives under a single **prefix**:

```
s3://<bucket>/<dataset>/<embedder>/<slice>/
```

Example:

```
s3://qdrant--vectorforge/fineweb/embedder-bge-large-en-v1.5/cc-main-2025-26/
```

## Corpus files

The embedder pipeline shards output by worker rank:

```
<prefix>/rank00/batch_00000000.parquet
<prefix>/rank01/batch_00000000.parquet
...
```

All pipeline tools (`nova load`, `nova load-dist`, `nova brute-force`, `nova generate-queries`) discover corpus files by listing `<prefix>/**/*.parquet` via `supernova.destinations.discover_corpus_parquets`. That function takes a `Destination` (`S3Destination`, `HfDestination`, or `LocalDestination`) and returns absolute URIs, with the same `eval/`-exclusion rule applied for every scheme.

## The `eval/` subdirectory

Evaluation artifacts are written to an `eval/` subdirectory directly under the prefix:

```
<prefix>/eval/queries_1000.parquet               # sampled query set
<prefix>/eval/brute_force_queries_1000_k1000.parquet  # ground-truth nearest neighbours
<prefix>/eval/_bf_partial_queries_1000_k1000/    # intermediate worker outputs (safe to delete after merge)
```

`discover_corpus_parquets` always excludes `<prefix>/eval/`. This means:

- `nova load` / `nova load-dist` — will not try to load eval files into Qdrant
- `nova brute-force` — will not scan eval files as corpus

## Globbing across multiple slices

If you want to load several corpus slices into a single Qdrant collection, pass the **embedder-level prefix** instead of a slice-level prefix:

```
s3://qdrant--vectorforge/fineweb/embedder-bge-large-en-v1.5/
  eval/                          ← eval artifacts live here
    queries_1000.parquet
    brute_force_queries_1000_k1000.parquet
  cc-main-2025-26/
    rank00/*.parquet
    rank01/*.parquet
  cc-main-2024-80/
    rank00/*.parquet
    ...
```

The rule is: **any key containing `/eval/` as a path component is excluded**, regardless of where `eval/` sits relative to the prefix you pass in. You can glob at the slice level or the embedder level and the exclusion always works correctly.

## Point ID scheme

Every corpus row gets a deterministic UUID:

```
md5("{bare_key}:{row_offset}") formatted as UUID
```

Where `bare_key` is the URI minus the scheme + container portion (`supernova.destinations.bare_key_for_uri`), and `row_offset` is the 0-based physical row index within the parquet file.

| Scheme | Anchor (stripped) | Bare key example |
|--------|-------------------|------------------|
| `s3://bucket/...` | `s3://{bucket}/` | `prefix/rank00/batch_00000000.parquet` |
| `hf://buckets/ns/name/...` | `hf://buckets/{ns}/{name}/` | `rank00/batch_0.parquet` (flat — no `data/` prefix) |
| `hf://datasets/ns/repo/...` (legacy reads) | `hf://datasets/{ns}/{repo}/` | `data/rank00/batch_0.parquet` |
| `file:///abs/path/...` | `file://` | `/abs/path/rank00/batch_0.parquet` |

The hash recipe is implemented once in `supernova.utils.make_point_id` and the bare-key derivation in `supernova.destinations.bare_key_for_uri`. Three places must agree on both:

- `nova load` — Qdrant point IDs via the `vf_point_id(filename, file_row_number)` DuckDB macro registered by `DataReader._register_macros`.
- `nova brute-force` — IDs for nearest-neighbour hits.
- `nova generate-queries` — `__source_file__` and `__source_row__` provenance columns let the eval side reconstruct each query's own point ID.

`file_row_number` is critical here: it's a DuckDB virtual column that always reflects the parquet's physical row index, regardless of parallel scan order. **Do not** use `ROW_NUMBER() OVER (PARTITION BY filename)` — that reflects DuckDB's scan ordering and produces different IDs under concurrency. There's a regression test in `tests/test_loader_id_expression.py` documenting both behaviours.

The IDs are stable across runs as long as the parquet files are not rewritten or relocated to a different container (different bucket / repo / mount point — see [Loader Architecture](loader-architecture.md#id-space-anchoring) for the trade-off).
