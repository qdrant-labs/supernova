use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::CollectionSchema;
use crate::errors::StoreError;

#[cfg(feature = "qdrant")]
pub mod qdrant;

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
    /// Create or verify the target collection/index exists, given the
    /// fully-resolved set of vectors (static specs + sizes discovered at read
    /// time). See [`CollectionSchema`](crate::config::CollectionSchema).
    async fn ensure_collection(&self, schema: &CollectionSchema) -> Result<(), StoreError>;

    /// Upsert a batch of points.
    async fn upsert_batch(&self, points: Vec<Point>) -> Result<(), StoreError>;

    /// Clean up connections.
    async fn close(&self) -> Result<(), StoreError>;

    /// Disable indexing for fast bulk loading. Called before upserts begin.
    async fn defer_indexing(&self) -> Result<(), StoreError> {
        Ok(())
    }

    /// Re-enable indexing after bulk load. Called after all upserts complete.
    async fn enable_indexing(&self) -> Result<(), StoreError> {
        Ok(())
    }

    /// Block until indexing is complete. Called after enable_indexing().
    async fn wait_for_indexing(&self) -> Result<(), StoreError> {
        Ok(())
    }
}

/// Vectorstore backend config, dispatched on `type:`. Each backend owns its
/// config struct in its own module; the variant is gated on the same feature.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VectorStoreConfig {
    #[cfg(feature = "qdrant")]
    Qdrant(qdrant::QdrantConfig),
}

impl VectorStoreConfig {
    /// Connect and build the shared store for this config.
    pub fn into_store(self) -> Result<Arc<dyn VectorStore>, StoreError> {
        match self {
            #[cfg(feature = "qdrant")]
            VectorStoreConfig::Qdrant(c) => {
                let store: Arc<dyn VectorStore> = Arc::new(c.into_store()?);
                Ok(store)
            }
        }
    }
}