# Embedding Overview

supernova's embedding pipeline streams data from a source, runs any number of configured embedders over it (dense, sparse, multivector — text or image inputs), and writes the results as parquet files to S3 or HuggingFace Hub.

![Embedding Pipeline](../fig/embedding_pipeline.svg)

## Configuration

Embedding configs live in `configs/embedder/` and have four sections:

```yaml
source:
  type: huggingface
  dataset_name: nick007x/arxiv-papers
  split: train

embedders:
  - name: gte
    kind: dense                    # dense | sparse | multivector (output shape)
    type: sentence_transformer     # backend implementation
    model: Alibaba-NLP/gte-multilingual-base
    input_column: abstract         # which source column this entry embeds
    modality: text                 # how to decode it: text | image (required)
    trust_remote_code: true
    batch_size: 64
    dtype: bfloat16

pipeline:
  chunk_size: 100000               # records per batch
  num_workers: 2                   # async workers (local mode)
  flush_threshold: 100000          # records before writing parquet

storage:
  type: object_store               # object_store (s3/gcs/azure), hf, or local
  path: s3://my-bucket/arxiv-papers/gte-multilingual-base
  output_dir: /tmp/supernova       # local staging dir before upload
```

`embedders:` is a list — add more entries to embed the same column with several models (comparison runs), or different columns with different models (e.g. CLIP on an image column next to MiniLM on the caption column). Each entry declares three independent axes:

- **`kind`** — the output shape (dense / sparse / multivector). Drives the parquet column type and the manifest.
- **`type`** — the backend implementation. The same backend name may exist for several kinds (`sentence_transformer` is both a dense and a sparse backend).
- **`modality`** — how the input column's values are decoded (`text` | `image`). **Required, no default**: the pipeline refuses to guess, so a wrong-modality run dies at launch instead of after hours of embedding file paths as prose. Transport is handled for you — an image column may hold file paths, raw bytes, or HuggingFace `{bytes, path}` structs.

Unknown entry keys (`batch_size`, `dtype`, `device`, …) pass through to the backend constructor. See [Dense Embedders](dense-embedders.md) and [Sparse Embedders](sparse-embedders.md) for the available backends.

Two automatic optimizations: a dense + sparse entry pair pointing at the same `sentence_transformer` model and input column is fused into a single forward pass, and entries sharing an identical backend config share one loaded model instance.

### Column naming

Each entry writes to its own column, defaulting to `{name}_embedding`:

```yaml
embedders:
  - name: minilm                   # → column "minilm_embedding"
    ...
  - name: ada
    output_column: dense_ada       # explicit override
    ...
```

Column names must be unique and must not collide with source columns.

### Empty inputs

`pipeline.on_empty_input` controls what happens when a row's input column is empty (`skip` drops the row, `null` writes a null embedding, `error` aborts). Default is `skip`; skipped rows are counted in the manifest (`rows_skipped_empty_input`) and a skip rate above 1% logs a warning — a high rate usually means a wrong `input_column`.

## Sources

### HuggingFace

Streams from any HuggingFace dataset:

```yaml
source:
  type: huggingface
  dataset_name: mteb/tweet_sentiment_extraction
  split: train
```

Sources are pure row producers — *what* gets embedded is declared per embedder entry (`input_column`), not on the source.

**Derived columns** — compose a new column from several fields with a format template; embedder entries can then use it as `input_column`, and it lands in the output parquet like any other column:

```yaml
source:
  type: huggingface
  dataset_name: nick007x/arxiv-papers
  render_columns:
    combined: "{title}: {abstract}"
```

**Text splitting** — opt in via the `chunking:` block (`strategy: fixed_char`, …). The chunker splits the input column's text into pieces before embedding; every entry reading that column receives the same pieces. A splitting strategy requires a single `input_column` across all entries — splitting one column while another entry reads a different one would produce inconsistent row counts, and errors at launch.

## Storage backends

Where the embedded parquet lands — chosen by `type`:

- **`object_store`** (alias `s3`) — any cloud object store via a `path` URI: S3 (`s3://`), GCS (`gs://`), Azure (`az://`), and S3-compatible stores (R2, B2, MinIO, Spaces) with an `endpoint`.
- **`hf`** — a HuggingFace Storage Bucket.
- **`local`** — the local filesystem, no upload.

```yaml
storage:
  type: object_store
  path: s3://my-bucket/arxiv/gte-base    # or gs://… , az://container/…
```

See [Storage backends](storage.md) for every provider, examples, credentials, and options.

## Output format

Every parquet file has a flat schema. Two groups of columns:

**Pass-through source columns** — every column from the source row (after `source.exclude_columns` filtering, plus any `render_columns`) lands verbatim under its original name; types are inferred by pyarrow. When a chunker is active, the input column holds the *chunk* that was embedded (other columns are replicated across a row's chunks).

**Embedding columns** — one per embedder entry, named `{name}_embedding` unless overridden with `output_column`. The arrow type follows the entry's `kind`:

| Kind | Type |
|------|------|
| `dense` | `list<float32>` |
| `sparse` | `struct{indices: list<uint32>, values: list<float32>}` |
| `multivector` | `list<list<float32>>` (N vectors per row) |

Embedding columns are always written with their declared type — under `on_empty_input: null` an empty input becomes a null value, never a schema change.

So if your HuggingFace source has columns `title`, `abstract`, `author`, those land alongside the embedding columns:

```sql
SELECT title, length(dense_embedding) AS dim
FROM read_parquet('s3://my-bucket/dataset/model/**/*.parquet')
LIMIT 10;
```

Unique row IDs are derived deterministically at **load time** from `(parquet_file_path, file_row_number)` using `vf_point_id`. The embed-side parquets do not carry an explicit `row_id` column; identity is anchored to physical row position so a re-read of the same parquet always produces the same IDs.

## Running locally

```bash
nova embed configs/embedder/my_dataset.yaml
```

The local runner uses async workers with a priority queue buffer to ensure ordered output. Good for development and small datasets.
