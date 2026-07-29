mod qdrant;
#[cfg(feature = "elastic")]
mod elastic;
#[cfg(feature = "milvus")]
mod milvus_store;

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::VectorSpec;

/// Errors from a [`VectorStore`](crate::stores::VectorStore) backend.
///
/// Backend client errors are boxed: they're large, and an unboxed variant
/// would bloat every `Result` (including the `Ok` path on hot calls like
/// `upsert_batch`). The manual `From` keeps `?` ergonomic.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error(transparent)]
    Qdrant(Box<qdrant_client::QdrantError>),
    /// Backend-agnostic failure, e.g. an existing collection whose config
    /// conflicts with the requested one.
    #[error("{0}")]
    Other(String),
}

impl From<qdrant_client::QdrantError> for StoreError {
    fn from(err: qdrant_client::QdrantError) -> Self {
        StoreError::Qdrant(Box::new(err))
    }
}

/// Vectorstore backend config, dispatched on `type:`. Each backend owns its
/// config struct in its own module; the variant is gated on the same feature.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VectorStoreConfig {
    // Boxed: QdrantConfig (with its HNSW/quantization/optimizer sub-configs) is far
    // larger than the other variants, so an unboxed variant sizes the whole enum to
    // it (clippy::large_enum_variant). This config is built once per load, not hot.
    Qdrant(Box<qdrant::QdrantConfig>),
    #[cfg(feature = "elastic")]
    Elastic(elastic::ElasticConfig),
    #[cfg(feature = "milvus")]
    Milvus(milvus_store::MilvusConfig),
}

impl VectorStoreConfig {
    /// Connect to the backend, building the live client once. Consumes the
    /// config (it's parsed once, then handed straight here) and returns the
    /// runtime store as a trait object — connection errors surface here, at
    /// startup, rather than mid-load.
    pub async fn connect(self) -> Result<Box<dyn VectorStore>, StoreError> {
        match self {
            VectorStoreConfig::Qdrant(c) => Ok(Box::new(c.connect().await?)),
            #[cfg(feature = "elastic")]
            VectorStoreConfig::Elastic(c) => Ok(Box::new(c.connect().await?)),
            #[cfg(feature = "milvus")]
            VectorStoreConfig::Milvus(c) => Ok(Box::new(c.connect().await?)),
        }
    }
}

/// What any backend needs to create or verify a collection: the named vector
/// specs (backend-agnostic) plus dimensions resolved from the parquet schema.
/// Collection-wide tuning (shards, HNSW, quantization) is backend-specific and
/// lives in each store's own config, not here.
pub struct CollectionSchema {
    pub vectors: HashMap<String, VectorSpec>,
    pub dims: HashMap<String, u64>,
}

/// A point id. Backends differ on what they accept (Qdrant: `u64` or a UUID
/// string; others allow arbitrary strings), so the core models the two shapes
/// the reader can produce and lets each backend enforce its own rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PointId {
    Integer(u64),
    String(String),
}

/// One named vector's value, as read from the source. Covers the three shapes a
/// backend like Qdrant accepts; the reader emits the variant matching the
/// vector's configured [`kind`](crate::config::VectorKind).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorValue {
    Dense(Vec<f32>),
    Multi(Vec<Vec<f32>>),
    Sparse { indices: Vec<u32>, values: Vec<f32> },
}

/// A single point to upsert: an id, its named vectors, and a payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    /// Point id (integer or string — see [`PointId`]).
    pub id: PointId,
    /// Vectors keyed by vector name.
    pub vectors: HashMap<String, VectorValue>,
    /// Arbitrary metadata stored alongside the vectors.
    #[serde(default)]
    pub payload: serde_json::Map<String, serde_json::Value>,
}

/// A vector store backend. `Display` is the human-readable name used in logs
/// (e.g. `qdrant(my_collection)`).
#[async_trait]
pub trait VectorStore: Send + Sync + std::fmt::Display {
    /// Create the target collection if absent (or verify it exists), from the
    /// backend-agnostic [`CollectionSchema`].
    async fn ensure_collection(&self, schema: &CollectionSchema) -> Result<(), StoreError>;

    /// Ensure payload indexes according to backend-specific config, using the
    /// resolved payload field names from the datasource config.
    ///
    /// Default is a no-op so backends opt in explicitly.
    async fn ensure_payload_indexes(
        &self,
        _payload_fields: &HashMap<String, String>,
    ) -> Result<(), StoreError> {
        Ok(())
    }

    /// Upsert a batch of points.
    async fn upsert_batch(&self, points: Vec<Point>) -> Result<(), StoreError>;

    /// Clean up connections.
    async fn close(&self) -> Result<(), StoreError>;

    /// Disable indexing for fast bulk loading. Called before upserts begin.
    /// Backends that have nothing to disable must still implement this
    /// explicitly (e.g. with an `Ok(())` body) — there is no default, so
    /// every backend states its behavior rather than silently inheriting one.
    async fn defer_indexing(&self) -> Result<(), StoreError>;

    /// Re-enable indexing after bulk load. Called after all upserts complete.
    /// Receives the same [`CollectionSchema`] passed to `ensure_collection` so
    /// backends that defer index *creation* until the data is in (e.g. Milvus,
    /// which loads into an unindexed collection and builds the index here) have
    /// the vector specs + dims on hand — the caller re-derives it in the
    /// distributed `finalize` step, which never called `ensure_collection`.
    async fn enable_indexing(&self, schema: &CollectionSchema) -> Result<(), StoreError>;

    /// Block until indexing is complete. Called after enable_indexing().
    ///
    /// Returns the instant when the backend first entered the green state that
    /// ultimately held long enough to be accepted as converged.
    async fn wait_for_indexing(&self) -> Result<std::time::Instant, StoreError>;

    /// Log a one-line indexing-time report after a load's `wait_for_indexing`.
    /// `effective` is the converged instant minus when indexing was kicked off
    /// (build time, excluding the stability hold). Default: report it as
    /// `index_seconds` — correct for backends that build the index *after*
    /// the upload (Qdrant, Milvus). Backends whose real cost is NOT in this
    /// window — Elasticsearch builds the HNSW graph inline during ingest, so the
    /// post-load window is ~0 — override this to report a meaningful figure.
    ///
    /// CAVEAT — these timings are NOT directly comparable across backends. Each
    /// vector store accounts for "indexing time" differently, so treat the
    /// numbers as within-backend signals, not an apples-to-apples benchmark:
    ///   - Qdrant / Milvus: `index_seconds` = a distinct post-upload index build
    ///     we time directly (Qdrant defers HNSW; Milvus builds after insert).
    ///   - Elasticsearch: builds the graph inline *during* ingest, so
    ///     `index_seconds` is ES's own `index_time` stat (fused ingest+build),
    ///     not the post-upload window — and ingest cost is really in the loader's
    ///     throughput (`pts/s`), not here.
    ///   - Milvus also reports a separate `load_seconds` (pulling the index into
    ///     memory) that the others have no equivalent of.
    async fn report_index_time(&self, effective: std::time::Duration) {
        tracing::info!(
            "{self} indexing finished: index_seconds={:.3}",
            effective.as_secs_f64()
        );
    }

    /// Re-apply index settings from config to an *already-existing* collection —
    /// does not touch data. Backends patch in place where they can (Qdrant:
    /// HNSW/quantization/optimizer overrides); where they can't (Milvus), this
    /// drops and rebuilds the index with the configured type/params/metric.
    /// Receives the [`CollectionSchema`] so backends can read per-vector settings
    /// (e.g. Milvus's metric from the distance). Callers that need to block until
    /// the change reconverges should call [`wait_for_indexing`]
    /// (`VectorStore::wait_for_indexing`) afterward, same as the
    /// `enable_indexing`/`wait_for_indexing` split. Backends that can't patch in
    /// place and have nothing to rebuild must still implement this (e.g. no-op).
    async fn reindex(&self, schema: &CollectionSchema) -> Result<(), StoreError>;

    /// Delete the collection if it exists. A no-op if it doesn't.
    async fn delete_collection(&self) -> Result<(), StoreError>;
}
