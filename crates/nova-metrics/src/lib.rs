//! Where storm/load measurements go — the Rust client for the shared metrics
//! contract.
//!
//! This is the Rust half of a polyglot story: the tools never share a metrics
//! *object* at runtime (they're separate processes, often on separate
//! machines), they share a **schema**. Every language writes the same
//! `runs`/`samples`/`events` tables (see `schema.sql`) with the same run/node
//! id conventions; this crate is one thin client of that contract, mirroring the
//! Python `supernova.metrics` backend.
//!
//! A [`MetricsSink`] is *where measurements go*. The base trait is all no-ops
//! ([`NullSink`] is just that), so a backend overrides only the verbs it cares
//! about. Backends are **fail-open**: a metrics hiccup must never crash the
//! workload it observes — buffer, drop-on-overflow, swallow-and-warn, never
//! panic into the caller. The one exception is connect/setup, which fails fast
//! so a bad DSN errors before the workload spins up.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

mod stdout;
pub use stdout::StdoutSink;

#[cfg(feature = "postgres")]
mod pg;
#[cfg(feature = "postgres")]
pub use pg::PostgresSink;

/// The ambient identity a run is opened with; every emission inherits it.
pub struct RunContext<'a> {
    pub command: &'a str,
    pub node_id: Option<&'a str>,
    pub experiment_id: Option<&'a str>,
    /// The resolved run config, stored on the `runs` row. Secret-looking keys
    /// are redacted by the sink before it's persisted.
    pub config: &'a serde_json::Value,
}

/// A metrics backend. All methods default to no-ops, so [`NullSink`] needs no
/// body and a real backend overrides only what it uses. `Send + Sync` because
/// one sink is shared (via `Arc`) across every concurrent request.
pub trait MetricsSink: Send + Sync {
    /// Open the run: a DB backend writes the `runs` row here.
    fn start(&self, _run_id: &str, _ctx: &RunContext<'_>) {}

    /// A scalar time-series point (rolling QPS, writes/sec, batch size).
    fn log(&self, _name: &str, _value: f64) {}

    /// One sample of a distribution — a single query's latency. `ok` flags
    /// whether the underlying operation succeeded (stored in the sample's tags).
    fn observe(&self, _name: &str, _value: f64, _ok: bool) {}

    /// A timestamped annotation ("indexing enabled", an error). Low-volume.
    fn event(&self, _message: &str) {}

    /// The final result record for this run/node (p50/p95/p99, totals).
    fn summary(&self, _values: &serde_json::Value) {}

    /// Flush and close. Always called, including on the error path.
    fn finish(&self, _status: &str) {}
}

/// The no-op sink: drops everything. Used when metrics are disabled.
pub struct NullSink;
impl MetricsSink for NullSink {}

/// Metrics sink config, dispatched on `type:`. Defaults (an absent block) map
/// to [`StdoutSink`] — local-first, no setup. `Serialize` so it round-trips
/// into the `runs.config` blob (the `dsn` is redacted there).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MetricsConfig {
    Stdout,
    Null,
    Postgres { dsn: String },
}

#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    #[cfg(feature = "postgres")]
    #[error(transparent)]
    Postgres(Box<::postgres::Error>),
    #[cfg(feature = "postgres")]
    #[error("tls setup failed: {0}")]
    Tls(String),
    #[error("{0}")]
    Unsupported(String),
}

#[cfg(feature = "postgres")]
impl From<::postgres::Error> for MetricsError {
    fn from(e: ::postgres::Error) -> Self {
        MetricsError::Postgres(Box::new(e))
    }
}

/// Build the shared sink for this config. `None` → [`StdoutSink`]. A Postgres
/// sink connects + validates schema here (fail-fast on a bad DSN).
pub fn build_sink(cfg: Option<&MetricsConfig>) -> Result<Arc<dyn MetricsSink>, MetricsError> {
    match cfg {
        None | Some(MetricsConfig::Stdout) => Ok(Arc::new(StdoutSink::new())),
        Some(MetricsConfig::Null) => Ok(Arc::new(NullSink)),
        #[cfg(feature = "postgres")]
        Some(MetricsConfig::Postgres { dsn }) => Ok(Arc::new(PostgresSink::connect(dsn)?)),
        #[cfg(not(feature = "postgres"))]
        Some(MetricsConfig::Postgres { .. }) => Err(MetricsError::Unsupported(
            "metrics.type=postgres but this binary was built without the `postgres` feature".into(),
        )),
    }
}

/// The unique id for ONE execution. A distributed run shares one id: the
/// controller mints it and forwards `NOVA_RUN_ID`, so every worker reports under
/// the same run (distinguished by `node_id`). A local run with no env var falls
/// back to `{base}-{unix_secs}` — unique enough to avoid colliding on the
/// `runs` primary key across reruns.
pub fn resolve_run_id(base: &str) -> String {
    if let Ok(id) = std::env::var("NOVA_RUN_ID")
        && !id.is_empty()
    {
        return id;
    }
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{base}-{secs}")
}

/// Keys whose values are masked before the config is stored — it's resolved by
/// now, so it holds real API keys / the DSN.
const SECRET_HINTS: &[&str] = &["key", "token", "secret", "password", "dsn", "api"];

/// Mask secret-looking values in a config blob, recursively. Shared by sinks
/// that persist the config.
pub fn redact(value: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| {
                    let lk = k.to_ascii_lowercase();
                    let masked = SECRET_HINTS.iter().any(|h| lk.contains(h));
                    let out = if masked {
                        Value::String("***".into())
                    } else {
                        redact(v)
                    };
                    (k.clone(), out)
                })
                .collect(),
        ),
        Value::Array(arr) => Value::Array(arr.iter().map(redact).collect()),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn redacts_secret_keys_recursively() {
        let cfg = json!({
            "url": "http://host",
            "api_key": "sk-123",
            "target": { "qdrant_token": "t", "collection": "c" },
            "list": [{ "password": "p" }],
        });
        let r = redact(&cfg);
        assert_eq!(r["url"], json!("http://host"));
        assert_eq!(r["api_key"], json!("***"));
        assert_eq!(r["target"]["qdrant_token"], json!("***"));
        assert_eq!(r["target"]["collection"], json!("c"));
        assert_eq!(r["list"][0]["password"], json!("***"));
    }

    #[test]
    fn run_id_prefers_env() {
        // SAFETY: single-threaded test.
        unsafe {
            std::env::set_var("NOVA_RUN_ID", "frosty-mango-123");
        }
        assert_eq!(resolve_run_id("storm"), "frosty-mango-123");
        unsafe {
            std::env::remove_var("NOVA_RUN_ID");
        }
        assert!(resolve_run_id("storm").starts_with("storm-"));
    }

    #[test]
    fn build_sink_defaults_to_stdout() {
        // None and an explicit stdout both yield a working sink; null too.
        assert!(build_sink(None).is_ok());
        assert!(build_sink(Some(&MetricsConfig::Stdout)).is_ok());
        assert!(build_sink(Some(&MetricsConfig::Null)).is_ok());
    }
}
