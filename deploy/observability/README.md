# Local observability stack

TimescaleDB (the metrics sink) + Grafana (pre-wired to it), for developing and
eyeballing the `supernova.metrics` Postgres backend locally — no cloud needed.

## Run

```bash
docker compose -f deploy/observability/docker-compose.yml up -d
pip install 'psycopg[binary]'
export SN_METRICS_DB_URL='postgresql://postgres:supernova@localhost:5432/supernova?sslmode=disable'
python deploy/observability/demo_emit.py        # generates data, no Qdrant needed
```

Then open Grafana at http://localhost:3000 (anonymous admin, no login). The
`supernova-timescale` datasource is already configured; the `init()` call in the
backend creates the `runs` / `samples` / `events` tables on first connect.

To drive it from a real load test instead, add a metrics block to a storm config:

```yaml
metrics:
  type: postgres
  dsn: ${SN_METRICS_DB_URL}
```

## Panel queries (Grafana → Explore, or a new panel)

Fleet QPS — one row per query, so `count` *is* throughput, no separate metric:

```sql
SELECT time_bucket('1 second', ts) AS time, count(*) AS qps
FROM samples
WHERE metric = 'latency_ms' AND $__timeFilter(ts)
GROUP BY 1 ORDER BY 1;
```

Latency percentiles over time, computed across ALL nodes (the correct way —
never average per-node percentiles):

```sql
SELECT time_bucket('5 seconds', ts) AS time,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY value) AS p50,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY value) AS p95,
       percentile_cont(0.99) WITHIN GROUP (ORDER BY value) AS p99
FROM samples
WHERE metric = 'latency_ms' AND $__timeFilter(ts)
GROUP BY 1 ORDER BY 1;
```

Runs:

```sql
SELECT run_id, command, node_id, started_at, finished_at, status, summary
FROM runs ORDER BY started_at DESC;
```

## SkyPilot's metrics (separate, optional)

SkyPilot runs its API server **locally** by default (it's the orchestrator;
`sky api info` shows the endpoint — you can also host it in the cloud), and it
exposes Prometheus `/metrics`. So it's scrapeable locally even while the fleet
runs on EC2. But those metrics describe SkyPilot's *orchestration* (request
rates, managed-job and cluster state, fleet GPU), not your workload's
latency/QPS — so it's a **separate datasource** from this Timescale sink, and it
only has anything interesting once you're actually running a fleet
(`storm-dist`); a single-machine `nova storm` doesn't drive it.

To chart it: run a Prometheus that scrapes the API server's `/metrics`, then add
that Prometheus as a second Grafana datasource. Worth wiring in the fleet phase,
not for local single-machine runs.

## On EC2

Workers can't reach this local db. Point `SN_METRICS_DB_URL` at Neon instead
(same backend, different DSN); `nova storm-dist` forwards it to workers
automatically because the config references `${SN_METRICS_DB_URL}`. Grafana can
stay local and read from Neon over the internet.

Note: this local Timescale is the full community edition (compression,
continuous aggregates), unlike Neon's Apache-2 build — fine, we don't rely on
those, but it means local can do anything Neon can and more.