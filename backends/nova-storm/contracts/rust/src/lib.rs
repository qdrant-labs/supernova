//! Shared Rust interface for `nova storm` backends.
//!
//! This crate is the Rust embodiment of the language-neutral contract in
//! `contracts/nova-storm/v1.yaml`. A Rust target backend depends on it and
//! implements [`QueryTarget`]; the compiler then enforces the method set at
//! build time. Any backend is additionally checked at runtime by
//! `nova contract check`, which compares the backend's `capabilities --json`
//! against the same YAML contract. Keep the three in lockstep:
//!
//! - the [`QueryTarget`] method names here,
//! - the `methods:` list in `contracts/nova-storm/v1.yaml`,
//! - the `methods` array a backend advertises from `capabilities --json`.
//!
//! Only the backend-agnostic surface lives here. The `target.type` dispatch
//! enum and each backend's connection/query config stay in the backend crate.

use std::time::Duration;

use async_trait::async_trait;

/// Errors from a [`QueryTarget`] backend.
///
/// Backend-neutral by construction: a backend renders its own client error to
/// string form at the trait boundary via [`TargetError::backend`], so this
/// crate never depends on any particular vector-DB client. Note this covers
/// setup/teardown only — a *query* failure during the load run is recorded as a
/// non-fatal error sample (see [`BatchOutcome`]), not surfaced here.
#[derive(Debug, thiserror::Error)]
pub enum TargetError {
    /// A backend client error (the vector DB's own error), captured as its
    /// string form so this type stays backend-neutral.
    #[error("{0}")]
    Backend(String),
    /// Backend-agnostic failure, e.g. a config the backend can't honour.
    #[error("{0}")]
    Other(String),
}

impl TargetError {
    /// Wrap any backend error as a neutral [`TargetError::Backend`]. Use at the
    /// trait boundary, e.g. `builder.build().map_err(TargetError::backend)?`.
    pub fn backend<E: std::fmt::Display>(err: E) -> Self {
        TargetError::Backend(err.to_string())
    }
}

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
///
/// The method set here is the canonical Rust contract for a `nova storm`
/// backend and must match `methods:` in `contracts/nova-storm/v1.yaml`.
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
