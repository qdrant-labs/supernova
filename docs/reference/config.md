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
  s3_bucket: qdrant--vectorforge
  s3_prefix: corag--kilt-corpus/embed-multilingual-e5-large
  output_dir: /tmp/vectorforge
```

---

## `source:`

Where to read rows from.

| Key | Default | Notes |
|-----|---------|-------|
| `type` | *required* | Currently only `huggingface` is supported. |
| `dataset_name` | *required* | HF hub ID, e.g. `corag/kilt-corpus`. |
| `config` | `None` | HF dataset config name (for multi-config datasets). |
| `split` | `train` | HF split. |
| `text_template` | `None` | `str.format(**row)` template, e.g. `"Title: {title}\n\n{contents}"`. Use this OR `text_field`, not both. |
| `text_field` | `None` | Single column name to use as text. Use when no template formatting is needed. |
| `exclude_columns` | `[]` | Columns to drop from the output parquet. |
| `offset` | `None` | Skip N rows. Set automatically by `--num-jobs` / `--job-rank`. |
| `limit` | `None` | Take at most N rows. Set automatically by `--num-jobs` / `--job-rank`. |

Streaming caveat: for large datasets, `ds.skip(offset)` iterates one row at a time. Rank N pays O(N × rows_per_job) startup cost. Large ranks (e.g. rank 99 on a 45M-row dataset) can take 30+ min before producing their first chunk.

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

---

## `storage:`

Where to write parquets + manifests.

| Key | Default | Notes |
|-----|---------|-------|
| `type` | `s3` | `s3`, `hf`, or `local`. |
| `s3_bucket` | *required (s3)* | |
| `s3_prefix` | *required (s3)* | |
| `repo_id` | *required (hf)* | For `type: hf`. |
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
| `pool.min_workers` | `0` (or `max_workers` with `--burst`) | Floor. Workers won't scale below this. |
| `pool.max_workers` | `num_jobs` | Ceiling. SkyPilot autoscales up to here as jobs queue. |

SkyPilot's autoscaler ramps **one replica per ~3-minute cycle** by default — way too slow for batch runs. Always use `--burst` for known-sized batch work.

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
| Pool scales up slowly | `--burst` |
| Only a few workers come up | Check quota (spot vs on-demand), drop `use_spot`, request quota bump |
| Pipeline slower than predicted | Check `dtype` (should be `bfloat16` on GPU), check you're actually on GPU (look for `Loading ... on cuda` in logs) |
| High-rank jobs take forever to start | HF streaming skip cost. Jobs with offset ≥ 10M can take 20+ min to begin producing chunks. |
| Many small parquets | Increase `flush_threshold`. Each job must still have ≥`flush_threshold` rows (else one small parquet per job). |