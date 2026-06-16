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

The corpus consumers (`nova load`, `nova load-dist`) discover corpus files by listing `<prefix>/**/*.parquet` via `supernova.destinations.discover_corpus_parquets`. That function takes a `Destination` (`S3Destination`, `HfDestination`, or `LocalDestination`) and returns absolute URIs, with the same `eval/`-exclusion rule applied for every scheme.

## The `eval/` subdirectory

`eval/` is a reserved namespace directly under the prefix for recall-evaluation artifacts (sampled query sets and ground-truth nearest neighbours). The recall-eval commands that populate it aren't shipped in the current CLI, but the namespace and its exclusion rule are still honoured so corpus consumers never mistake eval artifacts for corpus data:

```
<prefix>/eval/queries_1000.parquet               # sampled query set
<prefix>/eval/ground_truth_1000_k1000.parquet   # ground-truth nearest neighbours
```

`discover_corpus_parquets` always excludes `<prefix>/eval/`, so `nova load` / `nova load-dist` never try to load eval artifacts into Qdrant.

## Globbing across multiple slices

If you want to load several corpus slices into a single Qdrant collection, pass the **embedder-level prefix** instead of a slice-level prefix:

```
s3://qdrant--vectorforge/fineweb/embedder-bge-large-en-v1.5/
  eval/                          ← eval artifacts live here
    queries_1000.parquet
    ground_truth_1000_k1000.parquet
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

The hash recipe is implemented once in `supernova.utils.make_point_id` and the bare-key derivation in `supernova.destinations.bare_key_for_uri`. `nova load` computes Qdrant point IDs from it via the `vf_point_id(filename, file_row_number)` DuckDB macro registered by `DataReader._register_macros`. Any recall-eval tooling that maps query rows back to corpus point IDs must use the same recipe and bare-key form, or recall@k breaks silently.

`file_row_number` is critical here: it's a DuckDB virtual column that always reflects the parquet's physical row index, regardless of parallel scan order. **Do not** use `ROW_NUMBER() OVER (PARTITION BY filename)` — that reflects DuckDB's scan ordering and produces different IDs under concurrency. There's a regression test in `tests/test_loader_id_expression.py` documenting both behaviours.

The IDs are stable across runs as long as the parquet files are not rewritten or relocated to a different container (different bucket / repo / mount point — see [Loader Architecture](loader-architecture.md#id-space-anchoring) for the trade-off).
