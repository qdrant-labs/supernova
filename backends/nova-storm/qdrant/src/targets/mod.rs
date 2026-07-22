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

use serde::Deserialize;

use crate::config::QueryConfig;
use crate::errors::TargetError;

// The backend-agnostic contract (trait + `BatchOutcome`) lives in the shared
// Rust interface crate `nova-storm-contract-rust`. Re-export it so the rest of
// this backend keeps referring to `crate::targets::{QueryTarget, BatchOutcome}`
// unchanged, and so the qdrant module implements exactly that trait.
pub use nova_storm_contract_rust::{BatchOutcome, QueryTarget};

pub mod qdrant;

/// Target backend config, dispatched on `type:`. Each backend owns its config
/// struct in its own module. This dispatch enum names concrete backends, so it
/// stays here in the backend crate rather than in the neutral contract.
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
