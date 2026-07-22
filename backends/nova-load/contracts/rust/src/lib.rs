//! Shared Rust interface for `nova load` backends.
//!
//! This crate is the Rust embodiment of the language-neutral contract in
//! `contracts/nova-load/v1.yaml`. A Rust backend depends on it and implements
//! [`VectorStore`]; the compiler then enforces the method set at build time. A
//! non-Rust backend (or any backend at all) is instead checked at runtime by
//! `nova contract check`, which compares the backend's `capabilities --json`
//! against the same YAML contract. Keep the three in lockstep:
//!
//! - the [`VectorStore`] trait method names here,
//! - the `methods:` list in `contracts/nova-load/v1.yaml`,
//! - the `methods` array a backend advertises from `capabilities --json`.
//!
//! Only the genuinely backend-agnostic surface lives here. Backend-specific
//! config (collection tuning, connection details, the `vectorstore.type`
//! dispatch enum) stays in each backend crate.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Errors from a [`VectorStore`] backend.
///
/// Backend-neutral by construction: a backend renders its own client error to
/// string form at the trait boundary via [`StoreError::backend`], so this crate
/// never depends on any particular vector-DB client.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// A backend client error (the vector DB's own error), captured as its
    /// string form so this type stays backend-neutral.
    #[error("{0}")]
    Backend(String),
    /// Backend-agnostic failure, e.g. an existing collection whose config
    /// conflicts with the requested one.
    #[error("{0}")]
    Other(String),
}

impl StoreError {
    /// Wrap any backend error as a neutral [`StoreError::Backend`]. Use at the
    /// trait boundary, e.g. `client.foo().await.map_err(StoreError::backend)?`.
    pub fn backend<E: std::fmt::Display>(err: E) -> Self {
        StoreError::Backend(err.to_string())
    }
}

/// One named vector's spec. The scalar knobs (distance, datatype, comparator,
/// modifier) are strings interpreted by the store. HNSW/quantization tuning is
/// collection-wide (see each backend's store params), not per-vector.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorSpec {
    #[serde(rename = "type")]
    pub kind: VectorKind,
    /// Parquet column the reader pulls this vector from.
    pub column: String,
    /// Dense vector dimensionality. Optional: when omitted the loader infers it
    /// from the parquet schema (the column is a fixed-size list). Ignored for
    /// sparse vectors, which have no fixed size. Read by the store at
    /// collection-creation time; ignored by the reader.
    #[serde(default)]
    pub size: Option<u64>,
    /// Read by the store at collection-creation time; ignored by the reader.
    #[serde(default)]
    pub distance: Option<String>,
    /// Multivector comparator (e.g. `max_sim`); only meaningful for `multivector`.
    #[serde(default)]
    pub comparator: Option<String>,
    #[serde(default)]
    pub datatype: Option<String>,
    #[serde(default)]
    pub on_disk: Option<bool>,
    /// Sparse re-weighting modifier (e.g. `idf`); only meaningful for `sparse`.
    #[serde(default)]
    pub modifier: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorKind {
    Dense,
    Sparse,
    Multivector,
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
/// vector's configured [`kind`](VectorKind).
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
///
/// The method set here is the canonical Rust contract for a `nova load`
/// backend and must match `methods:` in `contracts/nova-load/v1.yaml`.
#[async_trait]
pub trait VectorStore: Send + Sync + std::fmt::Display {
    /// Create the target collection if absent (or verify it exists), from the
    /// backend-agnostic [`CollectionSchema`].
    async fn ensure_collection(&self, schema: &CollectionSchema) -> Result<(), StoreError>;

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
    async fn enable_indexing(&self) -> Result<(), StoreError>;

    /// Block until indexing is complete. Called after enable_indexing().
    ///
    /// Returns the instant when the backend first entered the green state that
    /// ultimately held long enough to be accepted as converged.
    async fn wait_for_indexing(&self) -> Result<std::time::Instant, StoreError>;

    /// Patch index-affecting collection settings (HNSW/quantization/optimizer
    /// overrides, from this store's own config) on an *already-existing*
    /// collection, in place — does not touch data. Callers that need to block
    /// until the change has reconverged should call [`wait_for_indexing`]
    /// (`VectorStore::wait_for_indexing`) afterward, same as the existing
    /// `enable_indexing`/`wait_for_indexing` split. Backends that can't patch
    /// in place must still implement this explicitly (e.g. as a no-op).
    async fn reindex(&self) -> Result<(), StoreError>;

    /// Delete the collection if it exists. A no-op if it doesn't.
    async fn delete_collection(&self) -> Result<(), StoreError>;
}
