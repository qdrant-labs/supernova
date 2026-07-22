mod qdrant;

use serde::Deserialize;

// The backend-agnostic contract (trait + shared types) lives in the shared
// Rust interface crate `nova-load-contract-rust`. Re-export it so the rest of
// this backend keeps referring to `crate::stores::{VectorStore, Point, ...}`
// unchanged, and so the qdrant module implements exactly that trait.
pub use nova_load_contract_rust::{
    CollectionSchema, Point, PointId, StoreError, VectorStore, VectorValue,
};

/// Vectorstore backend config, dispatched on `type:`. Each backend owns its
/// config struct in its own module. This dispatch enum names concrete backends,
/// so it stays here in the backend crate rather than in the neutral contract.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VectorStoreConfig {
    Qdrant(qdrant::QdrantConfig),
}

impl VectorStoreConfig {
    /// Connect to the backend, building the live client once. Consumes the
    /// config (it's parsed once, then handed straight here) and returns the
    /// runtime store as a trait object — connection errors surface here, at
    /// startup, rather than mid-load.
    pub async fn connect(self) -> Result<Box<dyn VectorStore>, StoreError> {
        match self {
            VectorStoreConfig::Qdrant(c) => Ok(Box::new(c.connect().await?)),
        }
    }
}
