//! Query targets — the backends a storm fires at.
//!
//! A [`QueryTarget`] is a thin adapter: "fire one batch dispatch of nearest-
//! neighbour queries and report its latency." A batch of 1 is not a special
//! case — it's just the default. The load *shape* (concurrency, duration,
//! rate, batch size) lives in the [runner](crate::runner), so a backend stays
//! minimal and the same runner drives any store. Targets are built once and
//! shared across every concurrent request via `Arc`, so the trait is
//! `Send + Sync` (the gRPC client multiplexes concurrent calls over one
//! connection).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

use crate::config::QueryConfig;
use crate::errors::TargetError;

pub mod qdrant;

/// Outcome of a single batch dispatch (one `query_batch` round-trip, covering
/// `vectors.len()` queries). A failure is recorded here (`ok = false`) rather
/// than aborting the run — a storm measures how a cluster behaves under load,
/// and errors at the limit are a finding, not a crash. `latency`/`ok`/`error`
/// describe the one round-trip, not any individual query inside it — a single
/// gRPC call's timing can't be honestly disaggregated into per-query numbers.
#[derive(Debug, Clone)]
pub struct BatchOutcome {
    pub latency: Duration,
    pub ok: bool,
    /// One entry per submitted query, in the same order as the input
    /// `vectors` — the point ids that query actually returned, best-first.
    /// `None` at a position means there's nothing meaningful to report for
    /// that query: recall tracking wasn't on for this run
    /// (`QdrantTarget::collect_ids` is `false`) or the whole dispatch failed
    /// (`!ok`). `Some(vec![])` is a real, different thing — recall tracking
    /// was on, the dispatch succeeded, and that query just matched nothing.
    pub ids: Vec<Option<Vec<String>>>,
    pub error: Option<String>,
}

/// A backend a storm sends queries to. `Display` is the name used in logs
/// (e.g. `qdrant(products)`).
#[async_trait]
pub trait QueryTarget: Send + Sync + std::fmt::Display {
    /// Fire one batch dispatch covering all of `vectors` in a single
    /// round-trip and return its latency + outcome. The top-k / vector-name
    /// knobs are baked into the target at construction, so the hot path is
    /// just the vectors. A single-element slice is not a special case — it's
    /// the default (`LoadProfile::batch_size == 1`).
    async fn query_batch(&self, vectors: &[&[f32]]) -> BatchOutcome;

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
