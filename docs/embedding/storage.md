# Storage backends

The `storage` block decides where embedded parquet lands. The backend is chosen
by `type`; the rest of the block configures it. Three backends:

| `type` | Writes to | Config key |
|--------|-----------|------------|
| `object_store` (alias `s3`) | Any cloud object store (S3, GCS, Azure) + S3-compatible | `path` (a URI) |
| `hf` | A HuggingFace Storage Bucket | `bucket_id` |
| `local` | The local filesystem (no upload) | `output_dir` |

Each chunk of the run produces one parquet file — `batch_00000000.parquet`,
`batch_00000001.parquet`, … — written under the destination.

## Object store

`object_store` writes to any cloud object store through
[`obstore`](https://developmentseed.org/obstore/) (Python bindings to Apache's
Rust `object_store` crate — async, multipart uploads). The destination is a
single `path` URI whose **scheme selects the provider**:

```yaml
storage:
  type: object_store            # `s3` is an alias for the same backend
  path: s3://my-bucket/arxiv/gte-base
```

`obstore` parses the URI, so everything after the bucket becomes the store's
prefix automatically — you don't split bucket and prefix yourself.

### Supported providers

| Provider | Scheme | Example `path` |
|----------|--------|----------------|
| Amazon S3 | `s3://` | `s3://my-bucket/arxiv/gte-base` |
| Google Cloud Storage | `gs://` | `gs://my-bucket/arxiv/gte-base` |
| Azure Blob Storage | `az://` | `az://my-container/arxiv/gte-base` |
| Cloudflare R2 | `s3://` + `endpoint` | see below |
| Backblaze B2 | `s3://` + `endpoint` | see below |
| MinIO / DigitalOcean Spaces / other S3-compatible | `s3://` + `endpoint` | see below |

R2, B2, MinIO, and Spaces all speak the S3 API — they aren't separate schemes.
Keep `s3://` and point at their endpoint.

### Examples

**Amazon S3**

```yaml
storage:
  type: object_store
  path: s3://my-bucket/arxiv/gte-base
```

**Google Cloud Storage**

```yaml
storage:
  type: object_store
  path: gs://my-bucket/arxiv/gte-base
```

**Azure Blob Storage**

```yaml
storage:
  type: object_store
  path: az://my-container/arxiv/gte-base
  config:
    account_name: mystorageaccount     # or set AZURE_STORAGE_ACCOUNT_NAME
```

**Cloudflare R2** (and any S3-compatible store — B2, MinIO, DigitalOcean Spaces)

```yaml
storage:
  type: object_store
  path: s3://my-bucket/arxiv/gte-base
  endpoint: https://<account-id>.r2.cloudflarestorage.com
  region: auto                          # R2 ignores region but the client wants one
```

### Credentials

Credentials come from each provider's **standard chain** — nothing is written to
disk by supernova:

- **S3 / S3-compatible** — boto3's chain: `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
  (+ `AWS_SESSION_TOKEN`), a `~/.aws` profile, or the instance role. This matches
  the rest of supernova (loading, brute-force). For R2/B2, set the S3 keys to that
  provider's access key + secret.
- **GCS** — `GOOGLE_SERVICE_ACCOUNT` (path to a key file) or
  `GOOGLE_SERVICE_ACCOUNT_KEY` (inline JSON).
- **Azure** — `AZURE_STORAGE_ACCOUNT_NAME` + `AZURE_STORAGE_ACCOUNT_KEY` (or a SAS
  token).

Anything the chain doesn't cover can be passed explicitly under a `config:` block,
which is forwarded to `obstore` verbatim — see
[`obstore`'s store config](https://developmentseed.org/obstore/latest/api/store/)
for the full key list per provider.

### Options

| Key | Applies to | Meaning |
|-----|------------|---------|
| `path` | all | Destination URI (`s3://` / `gs://` / `az://`). Required. |
| `endpoint` | S3-compatible | Custom S3 endpoint (R2/B2/MinIO/Spaces). |
| `region` | S3 | Region override (e.g. `auto` for R2). |
| `config` | all | Extra provider options forwarded to `obstore`. |
| `output_dir` | all | Local **staging** dir batches are written to before upload (default `/tmp/nova_embed`). |

For real **S3**, the bucket is auto-created if it doesn't exist. Other providers
require the bucket/container to already exist.

## HuggingFace Hub

Uploads to a HuggingFace Storage Bucket (`hf://buckets/owner/name/...`):

```yaml
storage:
  type: hf
  bucket_id: your-org/dataset-name--model-name
  prefix: ""                # optional subpath inside the bucket
  private: true
```

## Local

No upload — parquet stays on the local filesystem. Handy for a quick run or a
shared volume:

```yaml
storage:
  type: local
  output_dir: /tmp/supernova
```

## Reading it back

The `path` you write to is the same URI the downstream tools read from: point
[`nova load`](../loading/overview.md)'s `datasource.path` or
[`nova bf`](../brute-force/overview.md)'s `corpus.path` at it. (S3 read support is
first-class today; GCS/Azure reads depend on the reader — confirm a downstream
read before committing a run to a non-S3 store.)
