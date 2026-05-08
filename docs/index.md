# vectorforge

vectorforge is a toolkit for building large-scale vector search benchmarks. It handles two tasks:

1. **Embedding generation** -- take a dataset, embed it with any model (dense, sparse, or both), produce parquet files
2. **Vector store loading** -- take pre-embedded parquet files, load them into a database

Both tasks are designed for massive parallelization. Embedding generation fans out across SkyPilot GPU instances. Vector store loading fans out across SkyPilot spot instances. The datasets we work with are often hundreds of millions of rows and hundreds of gigabytes.

## Mental model

The two pipelines are independent. You can run them separately, on different machines, at different times. The parquet files at the destination are the bridge between them -- each embedding run produces many parquet files (one per chunk/slice), stored under a common URI prefix.

```
Source (HuggingFace) --> [Embedding Pipeline] --> Destination --> [Loading Pipeline] --> Qdrant
```

Destinations and corpora are addressed by URI. Three schemes are supported today:

| Scheme | Used for |
|--------|----------|
| `s3://bucket/prefix` | S3 buckets (the typical production destination) |
| `hf://buckets/ns/name[/subdir]` | HuggingFace Storage Buckets (mutable object storage on the Hub, good for community sharing) |
| `file:///abs/path` | Local filesystem (handy for tests and single-machine flows) |

The same URI is recognised by every command that operates on a corpus -- `vf load`, `vf brute-force`, `vf generate-queries`, etc.

### Embedding pipeline

![Embedding Pipeline](fig/embedding_pipeline.svg)

A HuggingFace dataset is split into N chunks. Each chunk is assigned to a SkyPilot compute node (CPU for API-based embedders like OpenAI, GPU for local models like sentence-transformers). Each node embeds its chunk and uploads the result as a parquet file to S3. The result is many parquet files under a shared S3 prefix -- each containing a batch of rows with the original text plus embedding columns (dense, sparse, or both).

### Loading pipeline

![Loading Pipeline](fig/ingestion_pipelione.svg)

The many parquet files on S3 are divided into N groups. Each group is assigned to an EC2 spot instance (provisioned by SkyPilot). Each instance streams its assigned parquet files via DuckDB and upserts the vectors into a shared Qdrant cluster. HNSW indexing is deferred until all data is loaded, then built in one efficient pass.

## Key design principles

- **Streaming** -- neither pipeline loads the full dataset into memory. Data is processed in chunks/batches throughout.
- **Core library + thin orchestrators** -- all business logic lives in `vectorforge/`. CLI scripts are thin wrappers that import and use the library.
- **YAML-driven** -- every pipeline run is defined by a YAML config. No hardcoded datasets, models, or destinations.
- **Flat parquet output** -- embedding output is flat columnar data (no nested JSON). Payload composition happens at load time, not at embed time.

## CLI

All subcommands are dispatched through a single `vf` entrypoint (a click group). `vf --help` lists everything and stays fast — heavy ML imports only load when the relevant subcommand actually runs.

### Embed

| Command | Purpose |
|---------|---------|
| `vf embed` | Embed a dataset locally |
| `vf embed-dist` | Distribute embedding across a SkyPilot GPU pool |
| `vf partition` | Run the embed pipeline with a no-op embedder (validates sharding without GPU spend) |
| `vf partition-dist` | Distribute the no-op pipeline across a CPU pool |

### Load

| Command | Purpose |
|---------|---------|
| `vf load` | Load pre-embedded data into a vector store |
| `vf load-dist` | Distribute loading across a SkyPilot pool. `--finalize` enables Qdrant indexing afterwards |

### Eval (queries + ground truth)

| Command | Purpose |
|---------|---------|
| `vf generate-queries` | Sample N rows from a corpus as eval queries (writes `{corpus}/eval/queries_<N>.parquet`) |
| `vf brute-force` | Single-GPU exact-NN search for recall ground truth |
| `vf brute-force-dist` | Distribute brute-force across a SkyPilot GPU pool, writing partial results |
| `vf brute-force-merge` | Merge per-rank partial results into the final top-K parquet |

### Introspect

| Command | Purpose |
|---------|---------|
| `vf analysis` | Analyze a completed embedding run (schema, throughput, cost) |
| `vf throughput-predict` | Predict embedding throughput + cost from a config |

See [CLI reference](reference/cli.md) for all flags, [config reference](reference/config.md) for every YAML knob and tuning advice, and [S3 layout](reference/s3-layout.md) for how corpora and eval artifacts are organised.
