# vectorforge

vectorforge is a toolkit for building large-scale vector search benchmarks. It handles two tasks:

1. **Embedding generation** -- take a dataset, embed it with any model (dense, sparse, or both), produce parquet files
2. **Vector store loading** -- take pre-embedded parquet files, load them into a database

Both tasks are designed for massive parallelization. Embedding generation fans out across SkyPilot GPU instances. Vector store loading fans out across SkyPilot spot instances. The datasets we work with are often hundreds of millions of rows and hundreds of gigabytes.

## Mental model

The two pipelines are independent. You can run them separately, on different machines, at different times. The parquet files on S3 are the bridge between them -- each embedding run produces many parquet files (one per chunk/slice), stored under a common S3 prefix.

```
Source (HuggingFace) --> [Embedding Pipeline] --> S3 (many parquets) --> [Loading Pipeline] --> Qdrant
```

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

| Command | Purpose |
|---------|---------|
| `vf embed` | Embed a dataset locally |
| `vf embed-dist` | Distribute embedding across SkyPilot GPU pool |
| `vf load` | Load pre-embedded data into a vector store |
| `vf load-dist` | Distribute loading across SkyPilot spot instances |
| `vf analysis` | Analyze a completed embedding run (schema, throughput, cost) |

See [config reference](reference/config.md) for every YAML knob and tuning advice, and [CLI reference](reference/cli.md) for all flags. For very large datasets (≥100M rows), see [incremental / windowed runs](reference/config.md#incremental--windowed-runs).
