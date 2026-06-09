"""PostgreSQL / TimescaleDB metrics backend — the Neon sink.

Emitting is sync and never touches the network: observe/log/event just enqueue.
A single background thread owns a connection and flushes the queue in batches
every ``flush_interval_s`` (or sooner once ``batch_size`` rows pile up), so DB
latency never perturbs the workload being measured and data shows up in Grafana
within ~a second — not at the end of the run.

Fail-open at runtime: a flush error is logged and dropped, never raised into the
caller. ``init()`` is the exception — it connects and builds schema up front, so
a bad DSN fails fast *before* the workload spins up.
"""

import logging
import queue
import threading
import time
from datetime import datetime, timezone

import psycopg
from psycopg.types.json import Jsonb

from .base import MetricsBackend

logger = logging.getLogger("supernova.metrics")

_SCHEMA = """
create table if not exists runs (
    run_id        text primary key,
    command       text,
    node_id       text,
    experiment_id text,
    started_at    timestamptz not null default now(),
    finished_at   timestamptz,
    status        text,
    config        jsonb,
    summary       jsonb
);
create table if not exists samples (
    ts      timestamptz not null,
    run_id  text not null,
    node_id text,
    metric  text not null,
    value   double precision not null,
    tags    jsonb
);
create index if not exists samples_run_metric_ts on samples (run_id, metric, ts);
create table if not exists events (
    ts      timestamptz not null,
    run_id  text not null,
    node_id text,
    message text not null,
    tags    jsonb
);
"""

_SECRET_HINT = ("key", "token", "secret", "password", "dsn", "api")


def _redact(obj):
    """Mask secret-looking values before storing the config (it's resolved by now,
    so it holds real API keys / the DSN). Keeps the run's params without leaking."""
    if isinstance(obj, dict):
        return {
            k: ("***" if any(h in k.lower() for h in _SECRET_HINT) else _redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


class PostgresBackend(MetricsBackend):
    def __init__(self, dsn, *, flush_interval_s=1.0, batch_size=5000, queue_size=200_000):
        self._dsn = dsn
        self._flush_interval_s = flush_interval_s
        self._batch_size = batch_size
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._conn = None  # main-thread connection: schema + the runs row only
        self._run_id = None
        self._node_id = None
        self._dropped = 0

    def init(self):
        # Connect + build/validate schema before any workload runs; a bad DSN
        # raises here (fail-fast), not mid-run. Then start the flush thread.
        self._conn = psycopg.connect(self._dsn, autocommit=True)
        with self._conn.cursor() as cur:
            for stmt in filter(str.strip, _SCHEMA.split(";")):
                cur.execute(stmt)
            # Backfill the column on tables created before experiments existed.
            cur.execute("alter table runs add column if not exists experiment_id text")
            try:
                cur.execute("create extension if not exists timescaledb")
                cur.execute("select create_hypertable('samples', 'ts', if_not_exists => true)")
            except Exception as e:
                # No TimescaleDB (or no privilege) -> a plain table works fine.
                logger.debug("samples stays a plain table (no timescaledb): %s", e)
        logger.info("metrics: connected to postgres, schema ready")
        self._thread = threading.Thread(target=self._drain, name="metrics-flush", daemon=True)
        self._thread.start()

    def start(self, run_id, context):
        self._run_id = run_id
        self._node_id = context.get("node_id")
        # ON CONFLICT DO NOTHING so replicated fleet workers (sharing one run_id
        # via NOVA_RUN_ID) don't fight over the row; the first one writes it.
        self._conn.execute(
            "insert into runs (run_id, command, node_id, experiment_id, status, config) "
            "values (%s, %s, %s, %s, 'running', %s) on conflict (run_id) do nothing",
            (
                run_id,
                context.get("command"),
                self._node_id,
                context.get("experiment_id"),
                Jsonb(_redact(context.get("config", {}))),
            ),
        )

    def log(self, name, value, **tags):
        self._enqueue(("s", time.time(), name, float(value), tags))

    def observe(self, name, value, **tags):
        self._enqueue(("s", time.time(), name, float(value), tags))

    def event(self, message, **tags):
        self._enqueue(("e", time.time(), message, tags))

    def summary(self, values):
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "update runs set summary = %s where run_id = %s",
                (Jsonb(values), self._run_id),
            )
        except Exception as e:
            logger.warning("metrics summary write failed: %s", e)

    def finish(self, status="ok"):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        if self._conn is not None:
            try:
                self._conn.execute(
                    "update runs set finished_at = now(), status = %s where run_id = %s",
                    (status, self._run_id),
                )
            finally:
                self._conn.close()
        if self._dropped:
            logger.warning("metrics: dropped %d emissions under backpressure", self._dropped)

    def _enqueue(self, item):
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self._dropped += 1  # DB can't keep up -> drop, never block the hot path

    def _drain(self):
        try:
            conn = psycopg.connect(self._dsn, autocommit=True)
        except Exception:
            # Without this the thread dies silently and samples vanish with no
            # clue why. Loud + return: the run still finishes (fail-open), but the
            # failure is now in the logs instead of a mystery.
            logger.exception("metrics flush thread could not connect; no samples will be written")
            return
        logger.info("metrics flush thread connected")
        try:
            while not self._stop.is_set():
                self._write_batch(conn, self._collect())
            self._write_batch(conn, self._collect(final=True))  # leftovers
        except Exception:
            logger.exception("metrics flush thread crashed; remaining samples lost")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _collect(self, final=False):
        batch = []
        if not final:
            # Block up to flush_interval for the first row -> wakes at least once
            # per interval so low-QPS runs still stream live, not just at the end.
            try:
                batch.append(self._q.get(timeout=self._flush_interval_s))
            except queue.Empty:
                return batch
        while len(batch) < self._batch_size:
            try:
                batch.append(self._q.get_nowait())
            except queue.Empty:
                break
        return batch

    def _write_batch(self, conn, batch):
        if not batch:
            return
        try:
            samples = [
                (datetime.fromtimestamp(it[1], timezone.utc), self._run_id, self._node_id, it[2], it[3], Jsonb(it[4]))
                for it in batch if it[0] == "s"
            ]
            events = [
                (datetime.fromtimestamp(it[1], timezone.utc), self._run_id, self._node_id, it[2], Jsonb(it[3]))
                for it in batch if it[0] == "e"
            ]
            with conn.cursor() as cur:
                if samples:
                    cur.executemany(
                        "insert into samples (ts, run_id, node_id, metric, value, tags) "
                        "values (%s, %s, %s, %s, %s, %s)",
                        samples,
                    )
                if events:
                    cur.executemany(
                        "insert into events (ts, run_id, node_id, message, tags) "
                        "values (%s, %s, %s, %s, %s)",
                        events,
                    )
            logger.debug("metrics flushed %d rows", len(batch))
        except Exception as e:
            logger.warning("metrics flush dropped %d rows: %s", len(batch), e)