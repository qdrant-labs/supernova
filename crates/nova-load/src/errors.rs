//! Typed errors for the loader.
//!
//! Each backend layer has its own error enum; backend-specific variants are
//! gated behind the backend's feature so the core never references a client
//! crate that isn't compiled in. [`LoadError`] aggregates them for the runner
//! and the binary.

/// Errors from a [`VectorStore`](crate::stores::VectorStore) backend.
///
/// Backend client errors are boxed: they're large, and an unboxed variant
/// would bloat every `Result` (including the `Ok` path on hot calls like
/// `upsert_batch`). The manual `From` keeps `?` ergonomic.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[cfg(feature = "qdrant")]
    #[error(transparent)]
    Qdrant(Box<qdrant_client::QdrantError>),

    /// Backend-agnostic failure, e.g. an existing collection whose config
    /// conflicts with the requested one.
    #[error("{0}")]
    Other(String),
}

#[cfg(feature = "qdrant")]
impl From<qdrant_client::QdrantError> for StoreError {
    fn from(e: qdrant_client::QdrantError) -> Self {
        StoreError::Qdrant(Box::new(e))
    }
}

/// Errors from a [`DataReader`](crate::sources::DataReader) backend.
#[derive(Debug, thiserror::Error)]
pub enum ReaderError {
    #[cfg(any(feature = "s3", feature = "local", feature = "huggingface"))]
    #[error(transparent)]
    Duckdb(Box<duckdb::Error>),

    /// A value in a row didn't have the shape its vector spec promised.
    #[error("{0}")]
    Other(String),
}

#[cfg(any(feature = "s3", feature = "local", feature = "huggingface"))]
impl From<duckdb::Error> for ReaderError {
    fn from(e: duckdb::Error) -> Self {
        ReaderError::Duckdb(Box::new(e))
    }
}

/// Top-level loader error returned by the runner and the binary.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error(transparent)]
    Config(#[from] crate::config::ConfigError),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Reader(#[from] ReaderError),
    /// Metrics-sink setup failed (e.g. a bad DSN) — fail fast before the load.
    #[error(transparent)]
    Metrics(#[from] nova_metrics::MetricsError),
}
