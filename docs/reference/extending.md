# Extending supernova

Three things you can plug in. Each lives in a different layer with a different ABC and registry.

| What you want | Where it lives | ABC | Used by |
|---|---|---|---|
| Read raw text/data into the embed pipeline (e.g. Common Crawl, a custom JSONL feed) | `supernova/sources/` | `DatasetSource` | `nova embed` |
| Add a new corpus backend (write embedded parquets, then read them back later — e.g. GCS, Backblaze) | `supernova/destinations.py` + `supernova/storage/` + `supernova/loader/datasource/` | `Destination` (dataclass) + `StorageBackend` + `DataReader` | `nova embed`, `nova load` |
| Add a new vector DB to load *into* (e.g. Weaviate, Pinecone) | `supernova/loader/vectorstore/` | `VectorStore` | `nova load` |

The three are deliberately separate. A "source" is read-only raw input — Common Crawl is a source you'd never write back to. A "destination" is the corpus-shaped storage on both sides of the bridge — embedder writes parquets to it, loader reads them back. A "vector store" is the loading sink — Qdrant today, possibly more later.

---

## 1. Adding a raw input source

A source produces dicts, one per row, plus a way to derive the text to embed.

### What to implement

```python
# supernova/sources/common_crawl.py
from typing import Iterator
from supernova.models import Record
from supernova.sources.base import DatasetSource


class CommonCrawlSource(DatasetSource):
    def __init__(self, snapshot: str, path_filter: str | None = None):
        self.snapshot = snapshot
        self.path_filter = path_filter

    @property
    def source_name(self) -> str:
        return f"common-crawl/{self.snapshot}"

    def stream(self) -> Iterator[dict]:
        # Yield raw rows. Pull from WARC files, an HTTP feed, whatever.
        for record in self._iter_warc_records():
            yield {"url": record.url, "text": record.payload}

    def format_record(self, row: dict) -> Record:
        # Decide what becomes `text` and which fields tag along as columns.
        return Record(text=row["text"], columns={"url": row["url"]})

    def get_total_rows(self) -> int:
        # Used for distributed slicing. If unknown, return a best estimate or
        # implement total_rows_override via the config (see HuggingFaceSource).
        return self._count_rows()
```

### Where to register

`cli/run_embedder.py` — add to `SOURCE_REGISTRY`:

```python
SOURCE_REGISTRY = {
    "huggingface": HuggingFaceSource,
    "huggingface_parquet": HuggingFaceSource,   # alias of huggingface
    "common_crawl": CommonCrawlSource,          # new
}
```

### How configs reach it

```yaml
source:
  type: common_crawl
  snapshot: CC-MAIN-2025-26
  path_filter: "warc/*.gz"
```

`build_source(cfg)` strips the `type` key and passes the rest as kwargs, so any `__init__` parameter is YAML-addressable.

### What you get for free

- Default chunking via `DatasetSource.get_chunks()` — splits long texts using the embedder's tokenizer, batches by `chunk_size`, drops empties.
- Distributed slicing via `--num-jobs` / `--job-rank` — auto-computes per-job offset/limit from `get_total_rows()`.

If your source has unusual chunking needs (e.g. group records by a key before chunking), override `get_chunks()` directly — see `HuggingFaceSource` for a non-default example.

---

## 2. Adding a new corpus backend

This is the meatier one because a corpus backend is *three coordinated pieces*: identification (URI parsing, path math), reading (DuckDB → upserts), and writing (the embed pipeline's parquet sink). Walking through GCS as an example.

### Step 1: register the URI scheme

`supernova/destinations.py` is the single source of truth for "where does this corpus live?" Add a dataclass and wire it through four functions.

```python
# supernova/destinations.py

@dataclass(frozen=True)
class GsDestination:
    bucket: str
    prefix: str

    @property
    def scheme(self) -> str:
        return "gs"

    @property
    def root_uri(self) -> str:
        return f"gs://{self.bucket}/{self.prefix}".rstrip("/")

    def child_uri(self, sub: str) -> str:
        sub = sub.lstrip("/")
        if not self.prefix:
            return f"gs://{self.bucket}/{sub}"
        return f"gs://{self.bucket}/{self.prefix}/{sub}"

    def eval_uri(self, filename: str) -> str:
        return self.child_uri(f"{EVAL_SUBDIR}/{filename}")
```

Add a branch in each of the scheme-keyed helpers in `supernova/destinations.py`. As of today there are nine, and a complete backend touches them all (the eval-side commands won't work otherwise):

| Helper | What you add |
|--------|---------------|
| `parse_destination(uri)` | recognise the new scheme and return your dataclass |
| `discover_corpus_parquets(dest)` | list every `.parquet` under the destination, excluding `eval/` |
| `filesystem_for_uri(uri)` | return a pyarrow- (or fsspec-) compatible filesystem object for `pq.read_table(..., filesystem=fs)` |
| `fs_path_for_uri(uri)` | strip the scheme to whatever path your filesystem expects |
| `bare_key_for_uri(uri)` | the per-file identifier used by `make_point_id`; must agree on both sides of the loader/eval split |
| `list_parquets_under(prefix_uri)` | recursive `.parquet` list for arbitrary prefixes (eval artifacts) — does not exclude `eval/` |
| `upload_file_to_uri(local, dest_uri)` | one-shot eval-artifact uploads (queries, brute-force outputs) |
| `upload_bytes_to_uri(data, dest_uri)` | same, for in-memory bytes |
| `datasource_to_destination(ds_cfg)` | build a `Destination` from a loader-config `datasource:` block |

Sketch:

```python
def parse_destination(uri: str) -> Destination:
    if uri.startswith("s3://"):
        ...
    if uri.startswith("hf://buckets/"):
        ...
    if uri.startswith("gs://"):
        rest = uri[len("gs://"):]
        bucket, _, prefix = rest.partition("/")
        if not bucket:
            raise ValueError(f"gs:// URI is missing bucket: {uri!r}")
        return GsDestination(bucket=bucket, prefix=prefix.rstrip("/"))
    raise ValueError(...)


def _discover_gs(dest: GsDestination) -> list[str]:
    from google.cloud import storage
    client = storage.Client()
    eval_segment = f"/{EVAL_SUBDIR}/"
    uris = []
    for blob in client.list_blobs(dest.bucket, prefix=dest.prefix):
        if blob.name.endswith(".parquet") and eval_segment not in blob.name:
            uris.append(f"gs://{dest.bucket}/{blob.name}")
    return sorted(uris)


def filesystem_for_uri(uri: str):
    ...
    if uri.startswith("gs://"):
        from gcsfs import GCSFileSystem
        return GCSFileSystem()


def bare_key_for_uri(uri: str) -> str:
    ...
    if uri.startswith("gs://"):
        rest = uri[len("gs://"):]
        _, _, key = rest.partition("/")
        return key
```

Also extend `datasource_to_destination` so a loader config with `type: gcs` works.

Add tests in `tests/test_destinations.py` mirroring the S3 ones — eval URI placement, bare-key derivation, parse round-trip.

### Step 2: write path — `StorageBackend`

`supernova/storage/gcs.py`:

```python
from supernova.storage.base import StorageBackend


class GcsBackend(StorageBackend):
    def __init__(self, bucket: str, prefix: str):
        self.bucket = bucket
        self.prefix = prefix
        self._client = None
        self._ready = False

    @property
    def destination(self) -> str:
        return f"gs://{self.bucket}/{self.prefix}"

    async def ensure_ready(self) -> None:
        # Create the bucket if it doesn't exist, etc. async-safe via asyncio.to_thread.

    async def upload_file(self, local_path: str, remote_subpath: str | None = None) -> None:
        # Upload to f"{self.prefix}/{remote_subpath or basename}".
        # Honor remote_subpath verbatim (keeps shard_by_rank paths like rank00/batch_*.parquet).

    async def upload_bytes(self, data: bytes, filename: str) -> None:
        # Used for manifest JSON. Per-backend convention: where does it live?
        # S3Backend and HuggingFaceBackend both put it under the prefix alongside
        # the parquets — bucket URIs are flat, no auto-detection subdir needed.
```

Most backends just dump everything under the prefix. If your backend has a "dataset auto-detection" subdir convention, write parquets to the auto-detected subdir while keeping manifests/READMEs wherever the backend expects them.

Register in `cli/run_embedder.py:build_storage`:

```python
def build_storage(cfg: dict):
    storage_type = cfg.pop("type", "s3")
    if storage_type == "s3":
        return S3Backend(...)
    elif storage_type == "hf":
        return HuggingFaceBackend(...)
    elif storage_type == "gcs":
        return GcsBackend(bucket=cfg["bucket"], prefix=cfg["prefix"])
    elif storage_type == "local":
        return LocalBackend(...)
    raise ValueError(...)
```

### Step 3: read path — `DataReader`

`supernova/loader/datasource/gcs.py`:

```python
from typing import Iterable
from .base import DataReader


class GcsDataReader(DataReader):
    def __init__(self, gcs_bucket: str, gcs_prefix: str, file_list: list[str] | None = None,
                 id_expression: str = "row_id", vectors=None, payload_fields=None,
                 duckdb_memory_limit="2GB", duckdb_threads=2):
        super().__init__(
            id_expression=id_expression, vectors=vectors,
            payload_fields=payload_fields,
            duckdb_memory_limit=duckdb_memory_limit, duckdb_threads=duckdb_threads,
        )
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix.rstrip("/")
        self.file_list = file_list

    @property
    def glob_path(self) -> str:
        return f"gs://{self.gcs_bucket}/{self.gcs_prefix}/**/*.parquet"

    def _root_uri_prefix(self) -> str:
        # The base class uses this to register vf_point_id. The bare key fed
        # to make_point_id is everything after this prefix — must agree with
        # bare_key_for_uri("gs://...") on the brute-force / query-gen side.
        return f"gs://{self.gcs_bucket}/"

    def _iter_sources(self) -> Iterable[str]:
        suffix = self._parquet_kwargs
        if self.file_list:
            for f in self.file_list:
                yield f"read_parquet('{f}'{suffix})"
        elif self._parquet_kwargs:
            yield f"read_parquet('{self.glob_path}'{suffix})"
        else:
            yield f"'{self.glob_path}'"

    def _configure_connection(self) -> None:
        conn = self._conn
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        # GCS auth — service account JSON, application default credentials, etc.
        # See S3DataReader / HuggingFaceDataReader for the auth-injection pattern.
```

The base class handles macro registration (`vf_point_id`, `make_point_id`, `vf_uuid_from_hex`) automatically — you just declare what to strip via `_root_uri_prefix()`.

Register in `cli/run_loader.py:DATASOURCE_REGISTRY`:

```python
DATASOURCE_REGISTRY = {
    "s3": S3DataReader,
    "huggingface": HuggingFaceDataReader,
    "gcs": GcsDataReader,
}
```

### Why three pieces, not one

You might wonder why `Destination`, `StorageBackend`, and `DataReader` aren't unified. They could be — but each is at a different layer:

- `Destination` is pure data: parsing, path math, listing files. No IO machinery.
- `StorageBackend` is the embed-pipeline-side write contract (async, parquet upload, manifest sidecar).
- `DataReader` is the load-pipeline-side read contract (DuckDB connection, SQL macros, batched fetch).

A new backend ends up touching all three because each does a real, distinct thing. The split keeps each ABC small and lets one piece evolve without breaking the others — e.g. you can add a new vector store without touching any backend.

### Configs

```yaml
# Embed run that writes to GCS
storage:
  type: gcs
  bucket: my-gcs-bucket
  prefix: my-dataset/embed-run-1

# Loader run that reads it back
datasource:
  type: gcs
  gcs_bucket: my-gcs-bucket
  gcs_prefix: my-dataset/embed-run-1
  id_expression: "vf_point_id(filename, file_row_number)"
```

---

## 3. Adding a new vector store

Lives in `supernova/loader/vectorstore/`. The contract is async because Qdrant's client is async; if your store has only a sync client, wrap calls in `asyncio.to_thread`.

```python
# supernova/loader/vectorstore/weaviate.py
from supernova.loader.vectorstore.base import VectorStore


class WeaviateVectorStore(VectorStore):
    def __init__(self, url: str, api_key: str | None = None,
                 collection_name: str = "default", vectors=None, params=None):
        self.url = url
        self.collection_name = collection_name
        self.vectors = vectors
        self.params = params or {}
        # construct client...

    @property
    def name(self) -> str:
        return f"weaviate({self.collection_name})"

    async def ensure_collection(self, dimensions: dict[str, int]) -> None:
        # Create collection if it doesn't exist. dimensions is {vector_name: dim}
        # for dense + multivector; sparse vectors are absent.

    async def upsert_batch(self, points: list[dict]) -> None:
        # Each point: {id: str, vectors: {name: value}, payload: dict}.
        # For multivector: value is list[list[float]]. For sparse: {"indices", "values"}.

    async def close(self) -> None:
        # Clean up the client.

    # Optional — implement these if your store has a fast-bulk-load mode.
    # No-op default in the base class is fine if it doesn't.
    # async def defer_indexing(self) -> None: ...
    # async def enable_indexing(self) -> None: ...
    # async def wait_for_indexing(self) -> None: ...
```

Register in `cli/run_loader.py:VECTORSTORE_REGISTRY`:

```python
VECTORSTORE_REGISTRY = {
    "qdrant": QdrantVectorStore,
    "weaviate": WeaviateVectorStore,
}
```

### Deferred indexing

`nova load` calls `defer_indexing()` before bulk writes and `enable_indexing()` + `wait_for_indexing()` after. If your store doesn't support disabling indexing during loads, leave the methods as no-ops — the runner just skips them and writes will be slower but correct. See [loader-architecture.md](loader-architecture.md#deferred-indexing) for the full lifecycle.

### Configs

```yaml
vectorstore:
  type: weaviate
  url: ${WEAVIATE_URL}
  api_key: ${WEAVIATE_API_KEY}
  collection_name: my-collection
  params:
    # arbitrary backend-specific knobs
    quantization:
      type: scalar
```

`build_vectorstore()` in `cli/run_loader.py` extracts `url`, `api_key`, `collection_name` as known fields and passes everything else as `params=...`.

---

## Checklist when adding a backend (any kind)

- [ ] ABC implemented; abstract methods covered.
- [ ] Registry updated in `cli/run_*.py` (which one depends on layer — see table at top).
- [ ] Tests added: at minimum, URI parse round-trip + eval-URI placement for destinations; backend-specific behavior for storage / readers / vector stores.
- [ ] If the backend has any "dataset auto-detection" subdir convention, make sure eval artifacts live *outside* it.
- [ ] If the new reader uses an unusual auth pattern, mirror the env-var injection in `_configure_connection()`.
- [ ] Add a config example under `configs/` so future-you remembers the YAML shape.
