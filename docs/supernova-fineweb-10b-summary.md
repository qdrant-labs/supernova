# Supernova FineWeb 10B Session Summary

## Primary Goal

Migrate the 10B-point Qdrant ingest workflow from prior project patterns into `supernova` using:

- `nova-load` for collection lifecycle and ingest behavior.
- `nova-dist` + SkyPilot for distributed worker orchestration on GCP.
- Deterministic point IDs using `vf_point_id(filename, file_row_number)`.
- S3-backed parquet sources with session AWS credentials.

## What Was Implemented

### 1) Distributed ingest and config path

- Established loader configs for smoke/probe/full workflows in `configs/loader/`.
- Ensured worker jobs use distributed partitioning (`--num-jobs`, `--job-rank`) correctly.
- Added/updated GCP SkyPilot resource configs for efficient worker pools.

### 2) Qdrant collection tuning and indexing controls

- Added support for payload index policy before bulk ingest:
  - `none`
  - `all`
  - `specific` fields
- Updated configs to include Turbo quantization settings targeting 4-bit mode.
- Kept HNSW in memory where requested by setting `hnsw.on_disk: false`.

### 3) Resume/checkpoint capability for interrupted loads

Implemented robust checkpointing in `nova-load`:

- New module: `crates/nova-load/src/checkpoint.rs`
- Metadata guards for safe resume (rank, num_jobs, datasource identity, config fingerprint).
- Persisted completed file keys and periodic flushes.
- Runtime CLI support:
  - `--resume`
  - `--checkpoint-path`
- Integrated checkpoint filtering and save flow into `load_files`.

Related code touched:

- `crates/nova-load/src/lib.rs`
- `crates/nova-load/src/main.rs`
- `crates/nova-load/src/config.rs`
- `crates/nova-load/src/checkpoint.rs` (new)
- `configs/loader/example.yaml`
- `README.md` (resume + recovery drill docs)

### 4) Finalize behavior fix for indexing re-enable

Issue addressed: if `optimizers.indexing_threshold: 0`, finalize could leave indexing effectively disabled.

Fix in `crates/nova-load/src/stores/qdrant.rs`:

- Added fallback constant `DEFAULT_ENABLED_INDEXING_THRESHOLD = 20_000`.
- `enable_indexing()` now forces non-zero threshold during finalize if configured value is `0`.
- Added unit tests for defaulting behavior and configured non-zero behavior.

## Configs Created/Updated for Incremental Targets

### 100M probe path

- `configs/loader/fineweb_100m_probe.yaml`
- `configs/loader/fineweb_100m_probe_workers.yaml`

Notes:

- Uses subset file list (about 98.9M rows from selected files).
- `recreate: true` on controller config, `recreate: false` on worker config.

### Incremental append files

- `configs/loader/fineweb_100m_probe_nextfile_workers.yaml`
  - Single non-overlapping parquet file append test.
- `configs/loader/fineweb_200m_increment_workers.yaml`
  - Multi-file non-overlapping append set to grow total toward ~200M.

## Operational Findings

- Early "0 points" states were often startup windows; workers later progressed.
- Some launches failed due to SkyPilot state/capacity constraints:
  - Controller/pool interruptions
  - GCP zone capacity issues in `us-central1-a`
  - Service/pool slot exhaustion (`Max number of services reached: 1/1`)
- Relative config paths can fail when launched from nested directories; absolute paths were used to resolve this.

## Current State (Latest Known)

- Jobs controller is up.
- Existing pool `nova-load-fineweb_100m_probe_nextfile` is being torn down (`SHUTTING_DOWN`) to free capacity.
- Once fully removed, relaunch of `fineweb_200m_increment_workers.yaml` should proceed.

## Suggested Relaunch Sequence

1. Confirm stale pool is fully gone:
   - `sky jobs pool status`
2. Relaunch incremental 200M append workers with current credentials/resources.
3. Monitor:
   - `sky jobs queue`
   - `sky jobs logs <job-id>`
4. Validate collection count via Qdrant API endpoint.
5. Run finalize to ensure indexing is re-enabled after bulk ingest.

## Key Takeaways

- The codebase now supports resumable distributed ingest with checkpoint safety.
- Payload indexing policy is configurable and can be applied pre-ingest.
- Finalize indexing recovery is protected against `indexing_threshold=0`.
- Remaining blocker for the next 100M append is infrastructure lifecycle (pool slot availability), not loader logic.

## New Throughput Tuning Notes

- Added a dedicated tuning runbook: `docs/fineweb-upsert-tuning-runbook.md`.
- Captured observed burst/backpressure behavior from the fresh 100M run (queue growth + transient RPS dips).
- Updated steady-state loader defaults for future runs:
  - `configs/loader/fineweb_100m_probe_workers.yaml`
  - `configs/loader/fineweb_10b_full.yaml`

