//! PostgreSQL / TimescaleDB sink — the Neon backend.
//!
//! The sync `postgres` crate runs its own internal runtime, so it **cannot** be
//! called from inside our async (`#[tokio::main]`) runtime — doing so panics
//! with "Cannot start a runtime from within a runtime". Therefore *every* DB
//! call (connect, the `runs` row, samples, summary, finish) lives on one
//! dedicated OS thread; the sink's methods just hand it messages over a channel.
//!
//! `observe`/`log`/`event` enqueue without blocking and the worker flushes in
//! batches (every `FLUSH_INTERVAL`, or sooner once `BATCH_SIZE` piles up), so DB
//! latency never perturbs the workload and data lands in Grafana within ~a
//! second. Lifecycle messages (`start`/`summary`/`finish`) are rare and use a
//! blocking send so they're never dropped under backpressure.
//!
//! Fail-open at runtime: a flush error is logged and dropped, never raised into
//! the caller. [`PostgresSink::connect`] is the exception — the worker connects
//! and builds schema up front and reports the result, so a bad DSN fails fast
//! before the workload spins up.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{
    Receiver, RecvTimeoutError, SyncSender, TrySendError, channel, sync_channel,
};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};

use native_tls::TlsConnector;
use postgres::{Client, Statement};
use postgres_native_tls::MakeTlsConnector;
use serde_json::json;

use crate::{MetricsError, MetricsSink, RunContext, redact};

/// The canonical schema, embedded at build time. Single source of truth shared
/// with every other language's client (see `schema.sql`).
const SCHEMA: &str = include_str!("../schema.sql");

const FLUSH_INTERVAL: Duration = Duration::from_secs(1);
const BATCH_SIZE: usize = 5_000;
const QUEUE_SIZE: usize = 200_000;

/// A message to the DB worker. Lifecycle variants (`Start`/`Summary`/`Finish`)
/// are rare and sent blocking; `Sample`/`Event` are the high-rate stream.
/// Strings are owned so the worker doesn't borrow from the caller.
enum Cmd {
    Start {
        run_id: String,
        command: String,
        node_id: Option<String>,
        experiment_id: Option<String>,
        config: serde_json::Value,
    },
    Sample {
        ts: SystemTime,
        metric: String,
        value: f64,
        ok: Option<bool>,
    },
    Event {
        ts: SystemTime,
        message: String,
    },
    Summary(serde_json::Value),
    Finish(String),
}

pub struct PostgresSink {
    tx: SyncSender<Cmd>,
    stop: Arc<AtomicBool>,
    worker: Mutex<Option<JoinHandle<()>>>,
    dropped: Arc<AtomicU64>,
}

impl PostgresSink {
    /// Spawn the DB worker, wait for it to connect + build schema, and return
    /// once it's ready. A bad DSN (or unreachable DB) errors here.
    pub fn connect(dsn: &str) -> Result<Self, MetricsError> {
        let (tx, rx) = sync_channel::<Cmd>(QUEUE_SIZE);
        let (ready_tx, ready_rx) = channel::<Result<(), MetricsError>>();
        let stop = Arc::new(AtomicBool::new(false));

        let worker = {
            let dsn = dsn.to_string();
            let stop = stop.clone();
            thread::Builder::new()
                .name("metrics-pg".into())
                .spawn(move || worker_loop(dsn, rx, stop, ready_tx))
                .map_err(|e| MetricsError::Unsupported(format!("metrics thread: {e}")))?
        };

        // Block until the worker has connected + validated schema (fail-fast).
        // This briefly parks the calling thread at startup — fine, and crucially
        // not a `postgres` call on the async runtime (that's what panics).
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                tx,
                stop,
                worker: Mutex::new(Some(worker)),
                dropped: Arc::new(AtomicU64::new(0)),
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(MetricsError::Unsupported(
                "metrics worker died during init".into(),
            )),
        }
    }

    /// Hot-path enqueue: drop on a full queue, never block the workload.
    fn enqueue(&self, cmd: Cmd) {
        if let Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) =
            self.tx.try_send(cmd)
        {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Lifecycle send: rare and must not be dropped, so block until queued.
    fn send_control(&self, cmd: Cmd) {
        let _ = self.tx.send(cmd);
    }
}

impl MetricsSink for PostgresSink {
    fn start(&self, run_id: &str, ctx: &RunContext<'_>) {
        self.send_control(Cmd::Start {
            run_id: run_id.to_string(),
            command: ctx.command.to_string(),
            node_id: ctx.node_id.map(str::to_string),
            experiment_id: ctx.experiment_id.map(str::to_string),
            // Redact here so secrets never sit in the channel or reach the DB.
            config: redact(ctx.config),
        });
    }

    fn log(&self, name: &str, value: f64) {
        self.enqueue(Cmd::Sample {
            ts: SystemTime::now(),
            metric: name.to_string(),
            value,
            ok: None,
        });
    }

    fn observe(&self, name: &str, value: f64, ok: bool) {
        self.enqueue(Cmd::Sample {
            ts: SystemTime::now(),
            metric: name.to_string(),
            value,
            ok: Some(ok),
        });
    }

    fn event(&self, message: &str) {
        self.enqueue(Cmd::Event {
            ts: SystemTime::now(),
            message: message.to_string(),
        });
    }

    fn summary(&self, values: &serde_json::Value) {
        self.send_control(Cmd::Summary(values.clone()));
    }

    fn finish(&self, status: &str) {
        self.send_control(Cmd::Finish(status.to_string()));
        // Signal the worker to drain + exit, then wait for it.
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.worker.lock().unwrap().take() {
            let _ = handle.join();
        }
        let dropped = self.dropped.load(Ordering::Relaxed);
        if dropped > 0 {
            tracing::warn!("metrics: dropped {dropped} emissions under backpressure");
        }
    }
}

fn make_tls() -> Result<MakeTlsConnector, MetricsError> {
    let connector = TlsConnector::new().map_err(|e| MetricsError::Tls(e.to_string()))?;
    Ok(MakeTlsConnector::new(connector))
}

/// The DB worker: connects, reports readiness, then batches the command stream
/// into Postgres until `stop` is set and the channel has drained. Owns the only
/// `Client` so no `postgres` call ever touches the async runtime.
fn worker_loop(
    dsn: String,
    rx: Receiver<Cmd>,
    stop: Arc<AtomicBool>,
    ready: std::sync::mpsc::Sender<Result<(), MetricsError>>,
) {
    let mut client = match init_client(&dsn) {
        Ok(c) => c,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let stmts = match prepare_statements(&mut client) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready.send(Err(MetricsError::Postgres(Box::new(e))));
            return;
        }
    };
    let _ = ready.send(Ok(()));
    tracing::info!("metrics: connected to postgres, schema ready");

    let mut run = RunIdent::default();
    loop {
        let mut batch: Vec<Cmd> = Vec::new();
        match rx.recv_timeout(FLUSH_INTERVAL) {
            Ok(c) => batch.push(c),
            Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
        while batch.len() < BATCH_SIZE {
            match rx.try_recv() {
                Ok(c) => batch.push(c),
                Err(_) => break,
            }
        }
        process_batch(&mut client, &stmts, &mut run, &batch);

        if stop.load(Ordering::Relaxed) {
            let mut rest: Vec<Cmd> = Vec::new();
            while let Ok(c) = rx.try_recv() {
                rest.push(c);
            }
            process_batch(&mut client, &stmts, &mut run, &rest);
            break;
        }
    }
    let _ = client.close();
}

fn init_client(dsn: &str) -> Result<Client, MetricsError> {
    let mut client = Client::connect(dsn, make_tls()?).map_err(|e| MetricsError::Postgres(Box::new(e)))?;
    client
        .batch_execute(SCHEMA)
        .map_err(|e| MetricsError::Postgres(Box::new(e)))?;
    // Backfill the column on tables created before experiments existed.
    let _ = client.batch_execute("alter table runs add column if not exists experiment_id text");
    // TimescaleDB is best-effort: a plain table works fine without it.
    if let Err(e) = client.batch_execute(
        "create extension if not exists timescaledb; \
         select create_hypertable('samples', 'ts', if_not_exists => true);",
    ) {
        tracing::debug!("samples stays a plain table (no timescaledb): {e}");
    }
    Ok(client)
}

/// Prepared inserts for the high-rate paths, reused across flush transactions.
struct Stmts {
    sample: Statement,
    event: Statement,
}

fn prepare_statements(client: &mut Client) -> Result<Stmts, postgres::Error> {
    Ok(Stmts {
        sample: client.prepare(
            "insert into samples (ts, run_id, node_id, metric, value, tags) \
             values ($1, $2, $3, $4, $5, $6)",
        )?,
        event: client.prepare(
            "insert into events (ts, run_id, node_id, message, tags) \
             values ($1, $2, $3, $4, $5)",
        )?,
    })
}

#[derive(Default)]
struct RunIdent {
    run_id: Option<String>,
    node_id: Option<String>,
}

/// Apply a batch in one transaction, in order. Lifecycle commands run as single
/// statements; samples/events use the prepared inserts.
fn process_batch(client: &mut Client, stmts: &Stmts, run: &mut RunIdent, batch: &[Cmd]) {
    if batch.is_empty() {
        return;
    }
    let res: Result<(), postgres::Error> = (|| {
        let mut tx = client.transaction()?;
        for cmd in batch {
            match cmd {
                Cmd::Start {
                    run_id,
                    command,
                    node_id,
                    experiment_id,
                    config,
                } => {
                    run.run_id = Some(run_id.clone());
                    run.node_id = node_id.clone();
                    // ON CONFLICT DO NOTHING so replicated workers sharing one
                    // run_id don't fight over the row.
                    tx.execute(
                        "insert into runs (run_id, command, node_id, experiment_id, status, config) \
                         values ($1, $2, $3, $4, 'running', $5) on conflict (run_id) do nothing",
                        &[run_id, command, node_id, experiment_id, config],
                    )?;
                }
                Cmd::Sample {
                    ts,
                    metric,
                    value,
                    ok,
                } => {
                    if let Some(rid) = &run.run_id {
                        let tags = ok.map(|b| json!({ "ok": b }));
                        tx.execute(&stmts.sample, &[ts, rid, &run.node_id, metric, value, &tags])?;
                    }
                }
                Cmd::Event { ts, message } => {
                    if let Some(rid) = &run.run_id {
                        let tags: Option<serde_json::Value> = None;
                        tx.execute(&stmts.event, &[ts, rid, &run.node_id, message, &tags])?;
                    }
                }
                Cmd::Summary(values) => {
                    if let Some(rid) = &run.run_id {
                        tx.execute("update runs set summary = $1 where run_id = $2", &[values, rid])?;
                    }
                }
                Cmd::Finish(status) => {
                    if let Some(rid) = &run.run_id {
                        tx.execute(
                            "update runs set finished_at = now(), status = $1 where run_id = $2",
                            &[status, rid],
                        )?;
                    }
                }
            }
        }
        tx.commit()?;
        Ok(())
    })();

    if let Err(e) = res {
        tracing::warn!("metrics flush dropped {} rows: {e}", batch.len());
    }
}
