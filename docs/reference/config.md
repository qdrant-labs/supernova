# Embedder Config Reference

Every `vf embed` / `vf embed-dist` run is driven by a YAML config. This page documents every key, its default, and when you'd change it.

Example config (`configs/embedder/example.yaml`):

```yaml
source:
  type: huggingface
  dataset_name: corag/kilt-corpus
  split: train
  text_template: "Title: {title}\n\nContents: {contents}"

dense_embedder:
  type: sentence_transformer
  model: intfloat/multilingual-e5-large
  trust_remote_code: true
  batch_size: 64
  dtype: bfloat16
  max_tokens: 256
  truncate: true

pipeline:
  chunk_size: 10000
  num_workers: 1
  flush_threshold: 50000

storage:
  type: s3
  bucket: qdrant--vectorforge
  prefix: corag--kilt-corpus/embed-multilingual-e5-large
  output_dir: /tmp/vectorforge
```

---

## `source:`

Where to read rows from.

| Key | Default | Notes |
|-----|---------|-------|
| `type` | *required* | `huggingface` (file-level sharded parquet reader). `huggingface_parquet` is accepted as a legacy alias for the same source. |
| `dataset_name` | *required* | HF hub ID, e.g. `corag/kilt-corpus`. |
| `config` | `None` | HF dataset config name (for multi-config datasets). |
| `split` | `train` | HF split. |
| `text_template` | `None` | `str.format(**row)` template, e.g. `"Title: {title}\n\n{contents}"`. Use this OR `text_field`, not both. |
| `text_field` | `None` | Single column name to use as text. Use when no template formatting is needed. |
| `exclude_columns` | `[]` | Columns to drop from the output parquet. |
| `total_rows_override` | `None` | Manual total-row count to use instead of the auto-detected one. Useful when the dataset's metadata reports a partial row count (e.g., HF's parquet-conversion sampling). Get the real total from the dataset's HF page ("Estimated number of rows") and paste it here. |
| `path_filter` | auto-match by `split` | Glob (or `regex:<expr>`, or list of either) over parquet paths. Useful for repos that mix splits or configs at different paths. |
| `metadata_workers` | `4` | Parallelism for the per-file footer fetches at init. Kept low by default to avoid bursting HF resolver rate limits when many ranks start simultaneously. |
| `prefetch` | `false` | Download each parquet to local disk before streaming. Eliminates per-batch HTTP range requests at the cost of a one-time sequential download — strongly recommended for multi-rank runs. |
| `prefetch_dir` | `/tmp/vectorforge_parquet` | Local directory for prefetched files. |

### How `--num-jobs` slices the dataset

When `vf embed` is invoked with `--num-jobs N` (which is what `vf embed-dist` does for each worker), the `N` ranks divide the full dataset evenly:

```
rank_offset = rank * ceil(dataset_total / num_jobs)
rank_limit  = min(ceil(dataset_total / num_jobs), dataset_total - rank_offset)
```

Worker logs will print a line like:

```
Job 5/10: offset=500000 limit=100000 (dataset_total=1000000)
```

which makes the slicing auditable at a glance.

### How it works

The source lists the dataset's parquet files on the Hub, reads each file's parquet footer for row counts, and maps `(offset, limit)` to a contiguous file range — yielding rows by reading those files directly. Init cost: one HF list-repo call + parallel footer reads (~3-10s for ~1000 files). Per-rank streaming cost: O(1) jump to the right file, then sequential read.

For multi-rank runs, set `prefetch: true` so each worker downloads its assigned files to local disk once instead of issuing per-batch HTTP range requests.

---

## `dense_embedder:`

How to produce dense embeddings.

### Common keys

| Key | Default | Notes |
|-----|---------|-------|
| `type` | *required* | `sentence_transformer` or `openai`. |
| `model` | *required* | HF model ID (ST) or model name (OpenAI). |
| `batch_size` | `32` (ST) | Records per forward pass. Bigger = better GPU utilization up to a memory ceiling. |
| `dtype` | `float32` | `float32`, `float16`, or `bfloat16`. Use `bfloat16` on modern GPUs (A10G+) for 2x memory savings with no quality loss. |

### sentence_transformer-specific

| Key | Default | Notes |
|-----|---------|-------|
| `trust_remote_code` | `false` | Required for some HF models (e.g. GTE, Alibaba-NLP). |
| `device` | auto-detect | `cuda`, `mps`, `cpu`. Leave unset unless debugging. |
| `max_tokens` | model default | Override the model's `max_seq_length`. Lower = faster, less semantic coverage. |
| `truncate` | `false` | If `true`, long docs get truncated at `max_tokens` (one embedding per doc). If `false`, long docs get split into pieces and each piece is embedded separately (multiple records per source row). |

### Truncate vs. split behavior

- **`truncate: false` (default)**: a 2000-token doc with `max_tokens=512` becomes 4 records, each with its own 512-token embedding. Preserves full semantic coverage. Longer runtime.
- **`truncate: true`**: same doc becomes 1 record with an embedding of just the first 512 tokens. Faster, loses content past the cutoff. Good for retrieval on short-context models.

---

## `multivector_embedder:`

Multi-vector embeddings (ColBERT-style): each input text produces N vectors of D floats,
where N varies with input length. Stored in parquet as `list<list<float32>>`.

| Key | Default | Notes |
|-----|---------|-------|
| `type` | *required* | `bge_m3` (currently the only option). |
| `model` | `BAAI/bge-m3` | HF model ID. Only models with a ColBERT-style multi-vector head are supported. |
| `batch_size` | `32` | Records per forward pass. |
| `dtype` | `float32` | `float32` or `float16`. `bfloat16` is not supported by the FlagEmbedding path. |
| `max_tokens` | model native (8192 for bge-m3) | Lower = faster. |
| `truncate` | `false` | Same contract as the sentence_transformer embedder. `true` = chop long docs at `max_tokens`, emit one record. `false` = split long docs into `max_tokens`-sized pieces, emit one multivector record per piece. |
| `device` | auto-detect | `cuda` / `mps` / `cpu`. |

Output column name is controlled by `pipeline.multivector_embedding_column` (default `multivector_embedding`).

Multi-vector embedders run as an **additional** forward pass on top of any dense/sparse
embedders. If you set all three (`dense_embedder`, `sparse_embedder`, `multivector_embedder`),
the engine runs dense+sparse via the hybrid path (one pass) and multi-vector separately (a
second pass). Multi-vector does not fuse with the hybrid path today.

### Truncate vs. split behavior (multivector)

Matches the dense sentence_transformer semantics:

- **`truncate: false` (default)**: a 2000-token doc with `max_tokens=512` becomes 4 multivector records, each covering a distinct 512-token slice of the original. Useful when you want passage-level multivector coverage of long docs.
- **`truncate: true`**: same doc becomes 1 multivector record covering only the first 512 tokens. Rest of the document is dropped. Faster, simpler indexing downstream.

---

### `multivector_embedder.pooling:` — derive a dense vector too

You can opt in to pooling the multi-vector output down to a single dense vector — no extra
forward pass. Useful when you want a standard dense column *and* multi-vector from the same
model without paying for two separate embedders.

Nested under `multivector_embedder` because it only applies in that context. Cannot be
combined with a separate top-level `dense_embedder` (both would produce a dense column).

```yaml
multivector_embedder:
  type: bge_m3
  model: BAAI/bge-m3
  batch_size: 64
  dtype: bfloat16
  max_tokens: 768
  truncate: true
  pooling:
    type: mean                      # mean | max | cls | last
    pooled_column_name: dense_embedding
    normalize: true                 # L2-normalize after pooling. default: true.
```

| Key | Default | Notes |
|-----|---------|-------|
| `type` | *required* | `mean`, `max`, `cls` (first token), or `last` (last token). `mean` is the common retrieval choice. |
| `pooled_column_name` | `dense_embedding` | Name of the dense column written to parquet. Overrides `pipeline.dense_embedding_column` when pooling is configured. |
| `normalize` | `true` | L2-normalize each pooled vector so cosine similarity on the output is meaningful. |

### Why pool?

bge-m3 (and other models with a colbert-style head) produces N vectors per text. For some
downstream systems you want a single vector per doc (standard dense retrieval, smaller index,
simpler reranking). Rather than running a separate dense model, you can pool the multi-vector
output and get a single vector "for free." Mean-pooling of L2-normalized token vectors followed
by L2-normalization is the standard recipe and matches how most dense models are actually trained.

### Resulting parquet

When pooling is enabled, your parquet gets **both** columns:

```
multivector_embedding: list<list<float32>>     # N vectors per row
dense_embedding: list<float32>                 # pooled, N → 1 vector
```

Downstream code reading only `dense_embedding` works unchanged.

---

## `pipeline:`

How the in-process pipeline is shaped. These are the main memory/throughput tuning knobs.

| Key | Default | Notes |
|-----|---------|-------|
| `chunk_size` | `10000` | Rows per chunker batch. Each chunk is a unit of work handed to a GPU worker. |
| `num_workers` | `8` | In-process async GPU workers per job. For single-GPU instances (g5.xlarge), set to `1` — additional workers compete for the same GPU and provide ~0% throughput gain. |
| `flush_threshold` | `100000` | Records accumulated in the `ResultBuffer` before flushing to a parquet file. Also determines parquet size. |
| `max_text_length` | `None` | Character-level truncation applied before tokenization. Useful for bounding pathological outliers (docs with 100K+ chars). |
| `dense_embedding_column` | `dense_embedding` | Output column name for dense vectors. |
| `sparse_embedding_column` | `sparse_embedding` | Output column name for sparse vectors. |
| `multivector_embedding_column` | `multivector_embedding` | Output column name for multi-vector embeddings. |
| `rendered_text_column` | `text` | Column name for the template-rendered string that gets sent to the embedder. Default shadows the source's raw `text` field (if it has one). Set to something else (e.g. `rendered_text`) to keep both — the raw field passes through under its original name. |
| `shard_by_rank` | `false` | Put each rank's output in its own subdir instead of a flat layout. Default: `rank42_batch_00000000.parquet`. With `true`: `rank42/batch_00000000.parquet`. Useful when producing thousands of parquets — avoids one enormous flat S3 directory. |

### Memory math

Rough per-job memory footprint during steady state:

```
work_queue     = num_workers × 2 chunks × chunk_size × avg_row_bytes
result_queue   = unbounded (!) — can accumulate during slow flushes
buffer         = flush_threshold × (row_bytes + dim × 4)
model          = num_workers × model_params × bytes_per_param
pyarrow spike  = ~2× buffer size during write_batch
```

On a 16GB g5.xlarge, for **short text (e.g. Amazon reviews, ~100 tokens)**:
- `chunk_size: 100000, flush_threshold: 100000, num_workers: 2` is fine (~4-6 GB peak).

For **long text (KILT articles, FineWiki, arxiv, etc.)**:
- Use `chunk_size: 10000, flush_threshold: 50000, num_workers: 1` to stay under 16GB.
- Or step up to `g5.2xlarge` (32GB RAM) and keep larger values.

### Output file shape

With `num_jobs=N` and `flush_threshold=F`:

- Rows per job = `total_rows / N`
- Parquets per job = `ceil(rows_per_job / F)`
- Total parquets = `N × parquets_per_job`

For 1B rows, `num_jobs=50`, `flush_threshold=1_000_000`: 50 workers × 20 parquets each = 1,000 files of 1M rows. Each job's slice must be **≥ flush_threshold** or you get one undersized parquet per job (the drain at end-of-pipeline flushes whatever remained).

Parquet filenames are prefixed by rank in distributed mode: `rank042_batch_00000000.parquet`. Manifest is `rank042__manifest.json`. Single-job runs drop the prefix.

With `shard_by_rank: true`, each rank writes into its own subdir: `rank042/batch_00000000.parquet` + `rank042/_manifest.json`. Recommended when you'd otherwise have thousands of parquets in a single S3 prefix — 50 ranks × 130 files each reads much nicer than 6500 siblings.

### Rendered text vs. raw source text

The `text` column in output parquets holds the **rendered template** (what was actually sent to the embedder), not the raw source field. If your source has a column named `text` (as HuggingFaceFW/finewiki does), it's shadowed by the rendered output at write time.

To keep both in the parquet, rename the rendered column:

```yaml
pipeline:
  rendered_text_column: rendered_text   # rendered template lands here
  # `text` column now passes through from the source unshadowed
```

Template rendering itself is unaffected — `text_template: "Title: {title}\nText: {text}"` still uses the original HF field names. The rename only applies at parquet write time.

---

## `storage:`

Where to write parquets + manifests.

| Key | Default | Notes |
|-----|---------|-------|
| `type` | `s3` | `s3`, `hf`, or `local`. |
| `bucket` | *required (s3)* | S3 bucket name. |
| `prefix` | *required (s3)*, `""` (hf) | S3 key prefix, or sub-path inside an HF bucket. |
| `bucket_id` | *required (hf)* | HF Storage Bucket id, `namespace/bucket-name`. Writes land at `hf://buckets/{bucket_id}/...`. |
| `token` | `None` | HF access token; falls back to `$HF_TOKEN` if unset. |
| `private` | `true` | For `type: hf`. |
| `output_dir` | `/tmp/vectorforge` | Local scratch dir where parquets are written before upload. Auto-cleaned after each flush. |

The `output_dir` is also the final destination for `type: local`.

---

## Distributed resources (`cli/run_embed_distributed.py:DEFAULT_RESOURCES`)

These aren't in the YAML — they're set in code and shared across all `vf embed-dist` runs. Override via a config's `resources:` block if you need something different for a specific job.

| Key | Default | Notes |
|-----|---------|-------|
| `accelerators` | `A10G:1` | GPU spec: `<TYPE>:<COUNT>`. |
| `cloud` | `aws` | |
| `use_spot` | `True` | Override per-run with `--on-demand`. |
| `disk_size` | `150` | GB. DLAMI images need ≥100GB. |
| `image_id` | AWS DLAMI per region | See below. |
| `any_of` | `[us-east-1, us-west-2, us-east-2]` | Regions SkyPilot can provision in. Picks whichever has capacity first. |

### AMI choice

We pin the AWS Deep Learning AMI (Amazon Linux 2023) explicitly because SkyPilot's default AMI ships with NVIDIA driver 535 (CUDA 12.2), which doesn't support the latest torch wheels (built for CUDA 13.0). The DLAMI has driver 580+ and supports modern torch directly.

Refresh the DLAMI IDs when they age out:

```bash
for r in us-east-1 us-west-2 us-east-2; do
  aws ec2 describe-images --region $r --owners amazon \
    --filters 'Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Amazon Linux 2023) *' \
              'Name=architecture,Values=x86_64' \
    --query 'sort_by(Images, &CreationDate) | [-1].ImageId' --output text
done
```

---

## Pool shape (`pool.yaml` generated by `vf embed-dist`)

| Key | Default | Notes |
|-----|---------|-------|
| `pool.min_workers` | `max_workers` (burst, default; `0` with `--ramp`) | Floor. Workers won't scale below this. |
| `pool.max_workers` | `num_jobs` | Ceiling. SkyPilot autoscales up to here as jobs queue. |

Burst is the default because SkyPilot's autoscaler ramps **one replica per ~3-minute cycle** — far too slow for known-sized batch runs. Pass `--ramp` if you actually want gradual provisioning.

---

## Environment variables forwarded to workers

These get forwarded from your local shell when `vf embed-dist` creates the pool:

- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
- `AWS_REGION`, `AWS_DEFAULT_REGION`
- `HF_TOKEN`
- `OPENAI_API_KEY`

Also set automatically by SkyPilot on each pool job:

- `SKYPILOT_JOB_RANK` — 0-indexed rank among `--num-jobs` (read by `vf embed` for slicing)
- `SKYPILOT_NUM_JOBS` — total job count

---

## Tuning cheatsheet

| Symptom | First thing to try |
|---------|-------------------|
| Worker OOM on long-text dataset | `chunk_size: 10000`, `num_workers: 1`, `flush_threshold: 50000` |
| Worker OOM on short text | `num_workers: 1` (the second worker doubles model memory for ~0% speedup) |
| Pool scales up slowly | Burst is the default; check the plan output. Only `--ramp` opts out. |
| Only a few workers come up | Check quota (spot vs on-demand), drop `use_spot`, request quota bump |
| Pipeline slower than predicted | Check `dtype` (should be `bfloat16` on GPU), check you're actually on GPU (look for `Loading ... on cuda` in logs) |
| High-rank jobs take forever to start | If using `prefetch: true`, each worker downloads its slice up front — that's expected. Without prefetch, check that footer-fetching at init isn't being rate-limited (lower `metadata_workers`). |
| Many small parquets | Increase `flush_threshold`. Each job must still have ≥`flush_threshold` rows (else one small parquet per job). |