# S3 Layout Convention

Every vectorforge corpus lives under a single **prefix** in S3:

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

All pipeline tools (`vf load`, `vf load-dist`, `vf push-hf`, `vf brute-force`) discover corpus files by listing `<prefix>/**/*.parquet` via `vectorforge.utils.discover_corpus_parquets`.

## The `eval/` subdirectory

Evaluation artifacts are written to an `eval/` subdirectory directly under the prefix:

```
<prefix>/eval/queries_1000.parquet               # sampled query set
<prefix>/eval/brute_force_queries_1000_k1000.parquet  # ground-truth nearest neighbours
<prefix>/eval/_bf_partial_queries_1000_k1000/    # intermediate worker outputs (safe to delete after merge)
```

`discover_corpus_parquets` always excludes `<prefix>/eval/`. This means:

- `vf load` / `vf load-dist` — will not try to load eval files into Qdrant
- `vf push-hf` — will not upload eval files to HuggingFace (see below)
- `vf brute-force` — will not scan eval files as corpus

**To push eval files to HuggingFace**, point `vf push-hf` directly at the eval subfolder:

```bash
vf push-hf s3://<bucket>/<prefix>/eval username/dataset-eval
```

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
md5("{relative_key}:{row_offset}") formatted as UUID
```

Where `relative_key` is the parquet file path relative to the prefix (e.g. `rank00/batch_00000000.parquet`) and `row_offset` is the 0-based row index within that file.

This scheme is implemented once in `vectorforge.utils.make_point_id` and used by:

- `vf brute-force` — assigns IDs to nearest-neighbour hits
- `vf load` — assigns Qdrant point IDs via `vf_point_id(filename, ROW_NUMBER() OVER (PARTITION BY filename) - 1)`
- `vf generate-queries` — records the query's own point ID for recall comparison

The IDs are stable across runs as long as the parquet files are not rewritten.
