//! Query targets — the backends a storm fires at.
//!
//! A [`QueryTarget`] is a thin adapter: "fire one nearest-neighbour query and
//! report its latency." The load *shape* (concurrency, duration, rate) lives in
//! the [runner](crate::runner), so a backend stays minimal and the same runner
//! drives any store. Targets are built once and shared across every concurrent
//! request via `Arc`, so the trait is `Send + Sync` (the gRPC client multiplexes
//! concurrent calls over one connection).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

use crate::config::QueryConfig;
use crate::errors::TargetError;

pub mod qdrant;

/// Outcome of a single query. A failure is recorded here (`ok = false`) rather
/// than aborting the run — a storm measures how a cluster behaves under load,
/// and errors at the limit are a finding, not a crash.
#[derive(Debug, Clone)]
pub struct QueryOutcome {
    pub latency: Duration,
    pub ok: bool,
    /// Number of points the query returned (sanity check). Independent of
    /// `ids` — a point lacking a resolvable id is dropped from `ids` but still
    /// counted here, so don't assume `matched == ids.len()`.
    pub matched: usize,
    /// The point ids actually returned, best-first — `None` when there's
    /// nothing meaningful to report: recall tracking wasn't on for this run
    /// (`QdrantTarget::collect_ids` is `false`) or the query failed (`!ok`).
    /// `Some(vec![])` is a real, different thing — recall tracking was on,
    /// the query succeeded, and it just matched nothing.
    pub ids: Option<Vec<String>>,
    pub error: Option<String>,
}

/// A backend a storm sends queries to. `Display` is the name used in logs
/// (e.g. `qdrant(products)`).
#[async_trait]
pub trait QueryTarget: Send + Sync + std::fmt::Display {
    /// Fire a single nearest-neighbour query for `vector` and return its latency
    /// + outcome. The top-k / vector-name knobs are baked into the target at
    /// construction, so the hot path is just the vector.
    async fn query(&self, vector: &[f32]) -> QueryOutcome;

    /// Tear down connections. Default: nothing (clients close on drop).
    async fn close(&self) -> Result<(), TargetError> {
        Ok(())
    }
}

/// Target backend config, dispatched on `type:`. Each backend owns its config
/// struct in its own module.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TargetConfig {
    Qdrant(qdrant::QdrantConfig),
}

impl TargetConfig {
    /// Connect and build the shared target. `query` carries the vector name and
    /// top-k the backend bakes in.
    pub fn into_target(self, query: &QueryConfig) -> Result<Arc<dyn QueryTarget>, TargetError> {
        match self {
            TargetConfig::Qdrant(c) => Ok(Arc::new(c.into_target(query)?)),
        }
    }
}
