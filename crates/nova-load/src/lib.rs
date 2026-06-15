#[cfg(not(any(feature = "qdrant")))]
compile_error!("enable at least one store backend feature, e.g. --features qdrant");

#[cfg(not(any(feature = "s3", feature = "local", feature = "huggingface")))]
compile_error!("enable at least one source backend feature, e.g. --features local");

pub mod config;
pub mod errors;
pub mod runner;
pub mod sources;
pub mod stores;

/// Per-vector size (by vector name) for dense and multivector vectors. Produced
/// by a [`DataReader`](sources::DataReader) and consumed by a
/// [`VectorStore`](stores::VectorStore) at collection creation, so it lives here
/// rather than in either module.
pub type DimensionsMap = std::collections::HashMap<String, usize>;