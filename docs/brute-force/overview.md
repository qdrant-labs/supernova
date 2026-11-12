# Brute Force

`nova bf` computes exact top-*k* nearest neighbors as ground truth for evaluating approximate vector search (e.g., recall@k). In distributed runs, workers scan disjoint corpus shards and merge their partial top-*k* results.

![nova bf pipeline: read corpus batches, score on GPU, prune, and merge](../fig/nova_bf_general.png){ .center width="600" }

## Configuration

```yaml
corpus:
  path: s3://my-bucket/dataset/model
  dense_column: dense_embedding

queries:
  path: s3://my-bucket/dataset/model/queries.parquet
  dense_column: dense_embedding
  payload_fields: [text]

output:
  path: s3://my-bucket/dataset/model/eval

params:
  io_workers: 16

searches:
  - name: dense_all
    vector_type: dense
    metric: cosine
    k: 1000
```

Each `searches` entry defines an independent search. Compatible searches share corpus reads. Environment variables support `${VAR}` and `${VAR:-default}` expansion.

## Vector types

- `dense`: `list<float32>` vectors with cosine, dot-product, or Euclidean scoring.
- `sparse`: index/value vectors with cosine or dot-product scoring.
- `multivector`: token vectors with ColBERT-style MaxSim scoring.

## Filters

Filters restrict eligible corpus rows using Qdrant-style `must` (AND), `should` (OR), and `must_not` conditions:

```yaml
filter:
  must:
    - field: language
      match: eng
  should:
    - field: source
      match: [wiki, news]
  must_not:
    - field: cost
      range: {gte: 10}
```

Conditions are not nested, each uses one filter type, and null values never match.

### Exact matches

`match` accepts a scalar or a list of alternatives:

```yaml
- field: language
  match: eng
- field: category_id
  match: [3, 7, 12]
```

### Ranges

`range` accepts any combination of `gt`, `gte`, `lt`, and `lte`:

```yaml
- field: cost
  range: {gte: 1, lt: 10}
```

Declare date fields under `corpus.date_fields`. Date bounds use RFC 3339 unless configured otherwise:

```yaml
corpus:
  date_fields: {published_at: rfc3339, crawl_day: "%Y%m%d", ingested_at: epoch_s}

filter:
  must:
    - field: published_at
      range: {gte: "2013-01-01T00:00:00Z"}
```

### Text matches

`match_text` requires every word, in any order. It uses lowercase word tokenization without stemming or stopword removal.

```yaml
- field: text
  match_text: chronic fatigue syndrome
```

### Per-query filters

`match_from_query`, `range_from_query`, and `match_text_from_query` read values from query columns:

```yaml
filter:
  must:
    - field: tenant_id
      match_from_query: tenant_id
    - field: cost
      range_from_query: {lt: max_budget}
    - field: title
      match_text_from_query: search_phrase
```

Static and per-query conditions can be combined:

```yaml
should:
  - field: is_public
    match: true
  - field: tenant_id
    match_from_query: tenant_id
```

Use separate conditions for literal and query-derived range bounds. Declare per-query dates in `queries.date_fields`. GPU execution supports `match_from_query` and `range_from_query`; `match_text_from_query` runs on CPU and may be expensive because it constructs a query-by-row mask for each file.

## Running

Single GPU:

```bash
nova bf compute my_eval.yaml
```

Distributed:

```bash
nova bf compute my_eval.yaml --num-jobs 8 --job-rank $RANK
nova bf merge my_eval.yaml
```

`nova dist bf` can provision and run distributed workers. Use `--max-files N` for testing only; its output is not full-corpus ground truth.

## Output

Each search writes:

```text
{output.path}/bf_<queries-stem>_<name>_k<K>.parquet
```

| Column | Description |
| --- | --- |
| `query_id` | Query identifier |
| `hit_ids` | Ranked corpus IDs |
| `hit_scores` | Corresponding scores |

A manifest records configuration, hardware, code version, and timings.

## IDs and ties

`hit_ids` must match the IDs used when loading the corpus into the vector database. By default, IDs are derived deterministically from file paths and row numbers; alternatively, set `corpus.id_column`.

Exact-score ties use corpus order (`ordinal`, default) or `corpus.id_column`.

## Performance and tuning

Execution overlaps corpus I/O, CPU decoding, and GPU scoring. Reading and decoding often dominate runtime.

### I/O and batching

- `io_thread_count`: S3 fetch concurrency. For files with many row groups, consider a higher value (e.g., 128) with `io_ranged_get` disabled.
- `io_ranged_get`: parallel range downloads, useful for files with one large row group.
- `rust_fetch_dense` / `rust_fetch_columns` (default `true`): Rust downloads and directly decodes dense vectors; other selected columns are downloaded in Rust and decoded by PyArrow. Unsupported layouts fall back to PyArrow with equivalent output.
- `rust_fetch_concurrency` (default 128), `rust_fetch_part_mib` (16), and `rust_fetch_chunks` (8): tune the native reader. The full Pes2o run used concurrency 512. `NOVA_BF_RUST_FETCH_DENSE`, `_COLUMNS`, `_CONC`, `_PART_MIB`, and `_CHUNKS` override these settings for a run (`0` disables the corresponding read path).
- `cpu_thread_count`: CPU decoding and filtering threads.
- `io_workers`: files processed concurrently.
- `dense_batch_size` / `sparse_batch_size`: bound GPU memory. Dense defaults to 8192; sparse defaults to whole-file slices. Values above 8192 bypass the optimized Triton top-*k* selection kernel and may be roughly 4× slower.

### Two-pass pruning and rescoring

`two_pass` safely excludes candidates unable to enter the top-*k*. Searches with the same vector type and metric share a pass. It requires contiguous query-row selections; interleaved selections disable two-pass for that group.

`two_pass_rescore` controls the second pass:

- `slice`: rescores qualifying queries across a full corpus slice using FP32 GEMM. Output is bit-identical to one-pass scoring.
- `tiles` (default): rescores only potentially competitive query–corpus tiles (typically 128 rows), reducing work as thresholds tighten. It remains pruning-safe, but different summation order can alter scores by floating-point rounding and reorder near ties.

Tiles mode can fall back to slice rescoring when its expected savings are low. It engages when the previous slice's live query–tile fraction is at most 20%; slice mode uses a 50% live-query threshold, and a slice with more than roughly one-third live work can fall back to slice rescoring. The `kernels.twopass` manifest records decisions and tile work.

For tiles, per-score differences are bounded approximately by `2 γ_d Σ|q_i c_i|`, where `γ_d = d × 2⁻²⁴`; near-tied candidates can exchange ranks. Results carry `nova_bf.two_pass_rescore=tiles` metadata and can merge with slice-mode partials. Set `NOVA_BF_TWOPASS_RESCORE` or `NOVA_BF_TWOPASS_TILE_THRESHOLD` to override the mode or cutoff. A10G validation (October 16–17, 2026) found exact recall within the derived score tolerance. Choose `slice` when bitwise agreement with one-pass scoring is required.

`pipeline_pass_one` (default `true`) overlaps the next dense slice's first pass with host-side processing of the previous slice. Decisions still use current thresholds; speculative work may be discarded or rerun. It uses extra GPU memory for one slice and its intermediate results.

In `slice` mode, pipelining preserves bit-identical results. In `tiles` mode, additional memory pressure may trigger a different fallback path, changing near-tie ordering. Configure with `NOVA_BF_PIPELINE_PASS_ONE`; the manifest records activity under `kernels.twopass.pipeline_*`. On A10G, measured late-file GPU time decreased from 1.39 to 1.11 seconds per file without output differences (October 16–17, 2026).

### Distributed threshold exchange

`threshold_exchange` (default `true`, disabled automatically for a single worker) shares running score thresholds across workers to improve pruning. Each worker publishes up to `min(k, 2k/N)` top scores per query as downward-rounded FP16. Workers prune against the *k*-th score in the union of their own and peers' lists. Missing or invalid peer updates only reduce pruning effectiveness.

All workers need a shared `output.path` with listing and read permissions. Every `threshold_exchange_refresh_s` (default 5 s), workers poll changed peer files using up to `threshold_exchange_fetch_concurrency` (default 8) concurrent GETs. Files live under `_thresholds/<run-id>/`; the run ID fingerprints the configuration, input files, query vectors, worker count, and code. This scratch directory may be removed after completion.

Each partial records its applied threshold. During `nova bf merge`, the final per-query *k*-th score must meet the largest recorded threshold; a violation fails verification even with force enabled. This protects against inconsistent or stale peer data. Partial files are self-contained, but must be merged with a compatible version aware of `applied_threshold` and `nan_hits`.

If validation fails, rerun the named ranks with `threshold_exchange: false`. Queries affected by invalid (`NaN`/infinite) scores use the recorded `nan_hits` count to reduce the final result length, matching single-node behavior. When exchanging thresholds, missing ranks or mixed runs can invalidate the merge.

For rank retries, prefer the original scoring configuration (GPU, batching, and precision). The rerun tolerance covers exact FP32 cosine scoring, not necessarily dot product, Euclidean distance, TF32, or other scoring-order differences.

### Partial-result trimming

`trim_partials` (default `true`, active only with threshold exchange) refreshes peer thresholds after the final corpus file, then omits local results strictly below the applied external threshold. Ties and queries with `NaN` or `+inf` scores are retained. The threshold is a global top-*k* lower bound, so omitted entries cannot be winners; merge-time threshold verification also covers trimming.

Trimmed partials may contain roughly `k/N` rather than `k` entries per query, reducing merge I/O. Trimmed and untrimmed partials can be combined; the manifest records `params.trim_partials_effective` and per-search trim statistics. Re-running a rank with different scoring hardware or settings can cause small score differences and fail verification; rerun that rank with `threshold_exchange: false` if necessary.

### Precision

`allow_tf32` accelerates matrix multiplication at the cost of exact FP32 scoring.