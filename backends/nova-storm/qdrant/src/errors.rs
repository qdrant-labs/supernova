//! Typed errors for storm.
//!
//! Like `nova-load`, each layer has its own enum; [`StormError`] aggregates them
//! for the binary.

// `TargetError` is part of the backend-agnostic contract and lives in the
// shared `nova-storm-contract-rust` crate. Re-export it so `crate::errors::
// TargetError` still resolves and `StormError`'s `#[from]` below keeps working.
pub use nova_storm_contract_rust::TargetError;

/// Errors loading the query-vector set from parquet.
#[derive(Debug, thiserror::Error)]
pub enum QueryLoadError {
    #[error(transparent)]
    Duckdb(Box<duckdb::Error>),

    /// A query-vector row wasn't the shape we expected (e.g. not a float list).
    #[error("{0}")]
    Other(String),
}

impl From<duckdb::Error> for QueryLoadError {
    fn from(e: duckdb::Error) -> Self {
        QueryLoadError::Duckdb(Box::new(e))
    }
}

/// Top-level storm error returned by the binary.
#[derive(Debug, thiserror::Error)]
pub enum StormError {
    #[error(transparent)]
    Config(#[from] crate::config::ConfigError),
    #[error(transparent)]
    Target(#[from] TargetError),
    #[error(transparent)]
    QueryLoad(#[from] QueryLoadError),
    #[error("{0}")]
    Other(String),
}
