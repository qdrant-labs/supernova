-- Canonical metrics schema — the single source of truth every language's
-- metrics client writes to. Keep this byte-compatible with any other client
-- (the Python `supernova.metrics` Postgres backend); a per-language conformance
-- test writes a run + samples + summary and reads them back to guard drift.
--
-- `create table if not exists` means whichever tool's init() runs first creates
-- the tables and the rest no-op — so the contract is enforced at the DB, not by
-- sharing code across the polyglot fleet.
--
-- TimescaleDB (hypertable on `samples`) and the `experiment_id` backfill are
-- applied best-effort by the client *after* this script, since they need an
-- extension / may run against a pre-existing table.

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
