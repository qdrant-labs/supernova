//! Typed errors for storm.
//!
//! Like `nova-load`, each layer has its own enum; [`StormError`] aggregates them
//! for the binary.

/// Errors from a [`QueryTarget`](crate::targets::QueryTarget) backend.
///
/// Backend client errors are boxed: `QdrantError` is large and an unboxed
/// variant would bloat every `Result`. Note this covers setup/teardown only —
/// a *query* failure during the load run is recorded as a non-fatal error
/// sample (see [`BatchOutcome`](crate::targets::BatchOutcome)), not surfaced here.
#[derive(Debug, thiserror::Error)]
pub enum TargetError {
    #[error(transparent)]
    Qdrant(Box<qdrant_client::QdrantError>),

    /// Backend-agnostic failure, e.g. a config the backend can't honour.
    #[error("{0}")]
    Other(String),
    #[error("operation `{operation}` is not supported by target `{target}`")]
    UnsupportedOperation { operation: String, target: String },
}

impl From<qdrant_client::QdrantError> for TargetError {
    fn from(e: qdrant_client::QdrantError) -> Self {
        TargetError::Qdrant(Box::new(e))
    }
}

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
