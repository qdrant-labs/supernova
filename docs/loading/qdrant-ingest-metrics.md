# Qdrant Ingest Metrics: Points/sec vs RPS

## Why this note exists

During the hybrid cloud smoke run, we saw a useful pattern:

- Higher request pressure is not automatically better ingest performance.
- Smaller batches can increase RPS while reducing total ingest efficiency.
- Oversized batches can fail hard (`413 Payload Too Large`) even when "RPS looks low".

For `nova-load`, the main KPI is **how fast durable points land in the target collection**.
That is a throughput problem first, not a requests-per-second contest.

## TL;DR

- Use **points/sec** as the primary performance metric.
- Use **RPS/QPS** as a supporting pressure/shape metric.
- Always pair both with queue depth, error rate, and latency.

## Why RPS alone is misleading

RPS is the count of requests, not the amount of useful work.

- If you halve `batch_size`, RPS may go up while points/sec stays flat or drops.
- If you increase `batch_size`, RPS may go down while points/sec goes up.
- If requests are too large, you can hit gateway/proxy limits (for example `413`) and lose effective throughput.

So "more RPS" can mean:

- better utilization,
- or just more overhead,
- or active overload.

## The metrics hierarchy for ingest tuning

1. **Primary:** points/sec (effective ingest throughput)
2. **Guardrails:** upsert success rate, error rate, and p95/p99 latency
3. **Pressure indicators:** update queue length, deferred points, running optimizations
4. **Cost/context:** RPS, CPU, IO, memory

## Prometheus/Grafana dashboard design

The queries below use metrics exposed by Qdrant 1.18.x (confirmed on `qdrant-hybrid-cloud`).
Assume a dashboard variable:

- `$collection` = target collection id/name (for example `fineweb_gte_10b_hybrid_cloud_smoke`)

### Panel 1: Effective points/sec (primary KPI)

Use collection growth slope:

```promql
clamp_min(deriv(collection_points{id="$collection"}[5m]), 0)
```

Notes:

- This is the closest store-side truth for ingest throughput.
- `deriv` on a gauge is intentionally used to estimate slope.
- `clamp_min(..., 0)` avoids negative spikes from scrape jitter.

### Panel 2: Upsert RPS (successful)

```promql
sum(rate(grpc_responses_total{endpoint="/qdrant.Points/Upsert",status="0"}[1m]))
```

### Panel 3: Upsert error RPS (non-zero statuses)

```promql
sum(rate(grpc_responses_total{endpoint="/qdrant.Points/Upsert",status!="0"}[1m]))
```

### Panel 4: Upsert success ratio

```promql
sum(rate(grpc_responses_total{endpoint="/qdrant.Points/Upsert",status="0"}[5m]))
/
sum(rate(grpc_responses_total{endpoint="/qdrant.Points/Upsert"}[5m]))
```

### Panel 5: Approx points/request (derived)

Use this to interpret whether throughput changes came from batch shape or request count:

```promql
clamp_min(deriv(collection_points{id="$collection"}[5m]), 0)
/
clamp_min(sum(rate(grpc_responses_total{endpoint="/qdrant.Points/Upsert",status="0"}[5m])), 0.0001)
```

### Panel 6: Update queue pressure

```promql
collection_update_queue_length{id="$collection"}
```

### Panel 7: Deferred points

```promql
collection_update_queue_deferred_points{id="$collection"}
```

### Panel 8: Optimization activity (index/build background work)

```promql
collection_running_optimizations{id="$collection"}
```

### Panel 9: Upsert p95 latency

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(grpc_responses_duration_seconds_bucket{endpoint="/qdrant.Points/Upsert",status="0"}[5m])
  )
)
```

### Panel 10: Upsert p99 latency

```promql
histogram_quantile(
  0.99,
  sum by (le) (
    rate(grpc_responses_duration_seconds_bucket{endpoint="/qdrant.Points/Upsert",status="0"}[5m])
  )
)
```

## Reading the dashboard (decision rules)

### Healthy ingest

- points/sec is stable or rising
- error RPS near zero
- queue length oscillates but does not trend up for long windows

### Overpressure

- points/sec flattens/drops
- queue length trends upward
- latency and/or error RPS rises

Action: lower `concurrency` first, then reduce `batch_size` if needed.

### Underutilized

- low queue pressure
- low latency
- low error rate
- points/sec below expected

Action: increase `batch_size` first (within payload limits), then `concurrency`.

## Converting points/sec to RPS (and back)

Approximate relation:

```text
RPS ~= points_per_sec / avg_points_per_request
```

In one smoke profile (`batch_size=8`) with ~650-750 points/sec:

- RPS ~= 81-94

This is why RPS should be treated as a shape/control metric: it changed mainly because batch size changed.

## Suggested alerts

- **Ingest stalled**
  - `points/sec < X` for 10m while job is running
- **Error burst**
  - `upsert error RPS > 0` for 5m (or above environment-specific threshold)
- **Queue runaway**
  - `collection_update_queue_length` increasing for N consecutive evaluation windows
- **Latency regression**
  - p99 upsert latency above baseline for 10m

Tune thresholds per cluster size and dataset profile.

