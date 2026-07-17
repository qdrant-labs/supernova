# FineWeb Qdrant Upsert Tuning Runbook (Toward 10B)

## Purpose

Capture practical tuning guidance from the 100M distributed ingest so we can run a steadier and more predictable 10B upsert.

## Observed Behavior (100M fresh run)

- Workers reached stable high throughput most of the time (~14k-17k pts/s per worker in sampled logs).
- Short, sharp dips occurred (examples seen: ~558 pts/s, ~1063 pts/s) before recovering.
- Qdrant `update_queue.length` grew high during peak ingest (hundreds of thousands), indicating write-side pressure/backlog.
- We also observed intermittent source-side retries:
  - `Generic S3 error: error decoding response body`
- Net result: throughput is high but bursty; occasional stalls are expected under this pressure profile.

## Interpretation

RPS dips were not primarily from worker crashes (workers stayed `RUNNING`), but from a mix of:

- temporary Qdrant write backpressure under burst load, and
- intermittent S3 read/retry delays.

This is consistent with "too much instantaneous upsert pressure" rather than a hard failure condition.

## Config Changes Applied (Steadier Defaults)

### `configs/loader/fineweb_100m_probe_workers.yaml`

Updated loader profile:

- `batch_size`: `256 -> 192`
- `concurrency`: `6 -> 4`
- `file_look_ahead`: `2 -> 1`
- `file_retries`: `5 -> 6`
- `upsert_retries`: `5 -> 8`
- `max_failed_files`: `3 -> 5`

### `configs/loader/fineweb_10b_full.yaml`

Updated loader profile:

- `batch_size`: `1024 -> 384`
- `concurrency`: `32 -> 12`
- `file_look_ahead`: `4 -> 2`
- `file_retries`: `5 -> 6`
- `upsert_retries`: `5 -> 8`
- `max_failed_files`: unchanged (`20`)

These values are intended to reduce burst pressure while preserving good sustained throughput.

## Operating Guidance for 10B

1. Start with the current steady profile (`batch_size=384`, `concurrency=12`).
2. Watch these indicators every few minutes:
   - SkyPilot worker state (`RUNNING` vs retry/fail).
   - Qdrant `points_count` slope (steady growth).
   - Qdrant `update_queue.length` trend.
3. If queue length grows continuously and RPS becomes bursty:
   - reduce `concurrency` first (e.g., `12 -> 10 -> 8`).
4. If queue is stable but CPU/network headroom exists:
   - increase `batch_size` modestly (e.g., `384 -> 512`) before increasing `concurrency`.
5. Keep retries elevated for long runs due to transient S3/network blips.

## Suggested Decision Rules

- **Stable zone**: queue oscillates but does not trend up for long windows; point count slope remains smooth.
- **Overpressure**: queue trends upward for multiple intervals, RPS has frequent deep dips, or retry warnings increase.
- **Underutilized**: queue remains very low and worker logs show low sustained RPS without upstream errors.

## Command Snippets

Check workers/jobs:

`sky jobs queue --limit 20`

Check pool:

`sky jobs pool status <pool-name>`

Check collection health and queue:

`curl -sS -H "api-key: $QDRANT_API_KEY" "https://qdrant-fineweb-10b.mavcode.io/collections/<collection_name>"`

After distributed load completion, re-enable indexing:

`nova dist load <worker-config> --finalize`

