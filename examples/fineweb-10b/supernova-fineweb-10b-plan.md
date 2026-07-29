# Supernova FineWeb 10B Migration Plan

## Goal
Implement a production-ready Supernova loading path that upserts 10B points from `s3://fineweb-gte-with-payloads/resharded/` into Qdrant using `nova dist load` (SkyPilot), preserving dense-only vector ingest and the payload schema validated from `data/train-part0__0047.parquet`.

## Source-of-Truth Schema
- Sample file: `data/train-part0__0047.parquet`
- Key columns:
  - `id` (string)
  - payload fields: `text`, `url`, `dump`, `date`, `language`, `language_score`, `token_count`
  - vector: `dense_embedding` (`list<halffloat>`, dim 768)

## Locked Decisions
- Orchestration: `nova dist load` with SkyPilot
- ID strategy: `id_expression: "vf_point_id(filename, file_row_number)"`
- Vector shape for first rollout: dense-only
- Data source: `s3://fineweb-gte-with-payloads/resharded/`

## Migration Mapping (Old Workflow -> Supernova)
- `PARQUET_ROOT` -> `datasource.path`
- `QDRANT_URL`, `QDRANT_API_KEY` -> `vectorstore.url`, `vectorstore.api_key`
- Upsert chunk/concurrency/retries -> `loader.batch_size`, `loader.concurrency`, `loader.upsert_retries`
- Deferred indexing behavior -> `nova load` lifecycle (`prepare` -> `load` -> `finalize`)
- Segment and optimizer tuning -> `vectorstore.params.optimizers.*`

## Two-Stage Rollout

### Stage 1: Smoke on 3-node cluster
- Use a smoke collection name (for example, `fineweb_gte_10b_smoke`).
- Restrict input to a small subset using `datasource.file_list` (3-20 parquet files).
- Validate end-to-end:
  - `nova load inspect load-smoke.yaml --num-jobs 1 --job-rank 0`
  - `nova dist load load-smoke.yaml --num-jobs 3`
  - `nova dist load load-smoke.yaml --finalize`
- Pass criteria:
  - config/schema inspection succeeds
  - no persistent upsert/file retry storms
  - finalize completes and collection is healthy

### Stage 2: Scale on 19-node cluster
- Use production collection name (for example, `fineweb_gte_10b_dense_v1`).
- Remove smoke `file_list` restriction (or point to a full list).
- Run:
  - `nova load inspect load-full.yaml --num-jobs 1 --job-rank 0`
  - `nova dist load load-full.yaml --num-jobs 19`
  - `nova dist load load-full.yaml --finalize`
- Increase `--num-jobs` only after stable behavior at 19.

## Continuation Strategy
- With `vf_point_id(filename, file_row_number)`, IDs are deterministic.
- Re-running overlapping files is idempotent (safe, but less efficient).
- Prefer maintaining a remaining-files `file_list` when continuing after interruptions.

## Command Correction
- Incorrect: `nova dist load load load.yaml`
- Correct: `nova dist load load.yaml --num-jobs <N>`

## Risks and Mitigations
- Credential/session expiry: refresh AWS credentials before long runs.
- Existing collection mismatch: use explicit collection naming and recreate policy.
- File-level failures at scale: set `loader.max_failed_files` as a fail-fast threshold.
- Throughput instability: tune `batch_size` and `concurrency` via staged rollout.
