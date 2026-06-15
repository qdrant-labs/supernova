use std::collections::HashMap;
use std::fmt;

use async_trait::async_trait;
use qdrant_client::qdrant::{
    CollectionStatus, CreateCollectionBuilder, Distance, HnswConfigDiffBuilder,
    MultiVectorComparator, MultiVectorConfigBuilder, OptimizersConfigDiffBuilder, PointStruct,
    SparseVectorParams, UpdateCollectionBuilder, UpsertPointsBuilder, Vector, VectorParams,
    VectorParamsBuilder, VectorParamsMap, vectors_config,
};
use qdrant_client::{Payload, Qdrant};
use serde::Deserialize;
use tokio::time::{Duration, sleep};

use super::{Point, PointId, VectorStore, VectorValue};
use crate::config::{CollectionSchema, VectorKind};
use crate::errors::StoreError;

const MAX_UPSERT_RETRIES: u32 = 5;
const DEFAULT_INDEXING_THRESHOLD: u64 = 20_000;
const STATUS_POLL_SECS: u64 = 5;

pub struct QdrantVectorStore {
    client: Qdrant,
    collection_name: String,
    params: QdrantParams,
    upsert_wait: bool,
}

impl QdrantConfig {
    /// Connect and build the store for this config.
    pub fn into_store(self) -> Result<QdrantVectorStore, StoreError> {
        let mut builder = Qdrant::from_url(&self.url);
        if let Some(key) = self.api_key {
            builder = builder.api_key(key);
        }
        let client = builder.build()?;
        let upsert_wait = resolve_upsert_wait(self.params.upsert_wait);
        Ok(QdrantVectorStore {
            client,
            collection_name: self.collection_name,
            params: self.params,
            upsert_wait,
        })
    }
}

impl QdrantVectorStore {
    fn build_create_collection(
        &self,
        schema: &CollectionSchema,
    ) -> Result<CreateCollectionBuilder, StoreError> {
        let mut dense: HashMap<String, VectorParams> = HashMap::new();
        let mut sparse: HashMap<String, SparseVectorParams> = HashMap::new();

        for (name, v) in schema {
            match v.kind {
                VectorKind::Dense | VectorKind::Multivector => {
                    let size = v.size.ok_or_else(|| {
                        StoreError::Other(format!("vector {name:?} has no dimension"))
                    })? as u64;
                    let mut params = VectorParamsBuilder::new(size, distance(v.distance.as_deref()));
                    if let Some(on_disk) = v.on_disk {
                        params = params.on_disk(on_disk);
                    }
                    if v.kind == VectorKind::Multivector {
                        params = params.multivector_config(MultiVectorConfigBuilder::new(
                            MultiVectorComparator::MaxSim,
                        ));
                    }
                    dense.insert(name.clone(), params.build());
                }
                VectorKind::Sparse => {
                    sparse.insert(name.clone(), SparseVectorParams::default());
                }
            }
        }

        let mut builder = CreateCollectionBuilder::new(&self.collection_name)
            .vectors_config(vectors_config::Config::ParamsMap(VectorParamsMap { map: dense }));
        if !sparse.is_empty() {
            builder = builder.sparse_vectors_config(sparse);
        }

        let p = &self.params;
        if let Some(n) = p.shard_number {
            builder = builder.shard_number(n);
        }
        if let Some(n) = p.replication_factor {
            builder = builder.replication_factor(n);
        }
        if let Some(n) = p.write_consistency_factor {
            builder = builder.write_consistency_factor(n);
        }
        if let Some(b) = p.on_disk_payload {
            builder = builder.on_disk_payload(b);
        }
        if let Some(h) = &p.hnsw_config {
            builder = builder.hnsw_config(build_hnsw(h));
        }
        if let Some(o) = &p.optimizers_config {
            builder = builder.optimizers_config(build_optimizers(o));
        }
        // TODO: quantization, per-vector datatype, and the immutable-config
        // mismatch check on an existing collection are parsed but not yet applied.
        Ok(builder)
    }

    async fn set_indexing_threshold(&self, threshold: u64) -> Result<(), StoreError> {
        let update = UpdateCollectionBuilder::new(&self.collection_name)
            .optimizers_config(OptimizersConfigDiffBuilder::default().indexing_threshold(threshold));
        self.client.update_collection(update).await?;
        Ok(())
    }
}

impl fmt::Display for QdrantVectorStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "qdrant({})", self.collection_name)
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn ensure_collection(&self, schema: &CollectionSchema) -> Result<(), StoreError> {
        if self.client.collection_exists(&self.collection_name).await? {
            if !self.params.recreate {
                return Ok(());
            }
            self.client.delete_collection(&self.collection_name).await?;
        }
        let create = self.build_create_collection(schema)?;
        self.client.create_collection(create).await?;
        Ok(())
    }

    async fn upsert_batch(&self, points: Vec<Point>) -> Result<(), StoreError> {
        let qdrant_points: Vec<PointStruct> = points.into_iter().map(to_point_struct).collect();
        let request = UpsertPointsBuilder::new(&self.collection_name, qdrant_points)
            .wait(self.upsert_wait)
            .build();

        let mut attempt = 0;
        loop {
            match self.client.upsert_points(request.clone()).await {
                Ok(_) => return Ok(()),
                Err(e) => {
                    attempt += 1;
                    if attempt >= MAX_UPSERT_RETRIES {
                        return Err(e.into());
                    }
                    sleep(Duration::from_secs(1 << (attempt - 1))).await;
                }
            }
        }
    }

    async fn close(&self) -> Result<(), StoreError> {
        // The gRPC channel is closed when the client is dropped.
        Ok(())
    }

    async fn defer_indexing(&self) -> Result<(), StoreError> {
        self.set_indexing_threshold(0).await
    }

    async fn enable_indexing(&self) -> Result<(), StoreError> {
        self.set_indexing_threshold(DEFAULT_INDEXING_THRESHOLD).await
    }

    async fn wait_for_indexing(&self) -> Result<(), StoreError> {
        loop {
            let info = self.client.collection_info(&self.collection_name).await?;
            let status = info.result.map(|r| r.status).unwrap_or_default();
            if status == CollectionStatus::Green as i32 {
                return Ok(());
            }
            sleep(Duration::from_secs(STATUS_POLL_SECS)).await;
        }
    }
}

fn distance(name: Option<&str>) -> Distance {
    match name.unwrap_or("cosine").to_ascii_lowercase().as_str() {
        "dot" => Distance::Dot,
        "euclid" => Distance::Euclid,
        "manhattan" => Distance::Manhattan,
        _ => Distance::Cosine,
    }
}

fn build_hnsw(c: &HnswConfig) -> HnswConfigDiffBuilder {
    let mut b = HnswConfigDiffBuilder::default();
    if let Some(m) = c.m {
        b = b.m(m);
    }
    if let Some(ef) = c.ef_construct {
        b = b.ef_construct(ef);
    }
    if let Some(t) = c.full_scan_threshold {
        b = b.full_scan_threshold(t);
    }
    if let Some(t) = c.max_indexing_threads {
        b = b.max_indexing_threads(t);
    }
    if let Some(d) = c.on_disk {
        b = b.on_disk(d);
    }
    if let Some(m) = c.payload_m {
        b = b.payload_m(m);
    }
    b
}

fn build_optimizers(c: &OptimizersConfig) -> OptimizersConfigDiffBuilder {
    let mut b = OptimizersConfigDiffBuilder::default();
    if let Some(t) = c.deleted_threshold {
        b = b.deleted_threshold(t);
    }
    if let Some(n) = c.vacuum_min_vector_number {
        b = b.vacuum_min_vector_number(n);
    }
    if let Some(n) = c.default_segment_number {
        b = b.default_segment_number(n);
    }
    if let Some(n) = c.max_segment_size {
        b = b.max_segment_size(n);
    }
    if let Some(n) = c.memmap_threshold {
        b = b.memmap_threshold(n);
    }
    if let Some(n) = c.indexing_threshold {
        b = b.indexing_threshold(n);
    }
    if let Some(n) = c.flush_interval_sec {
        b = b.flush_interval_sec(n);
    }
    // TODO: max_optimization_threads uses a wrapper type; wire it up later.
    b
}

fn to_point_struct(point: Point) -> PointStruct {
    let id = match point.id {
        PointId::Integer(n) => qdrant_client::qdrant::PointId::from(n),
        PointId::String(s) => qdrant_client::qdrant::PointId::from(s),
    };
    let vectors: HashMap<String, Vector> = point
        .vectors
        .into_iter()
        .map(|(name, v)| (name, to_vector(v)))
        .collect();
    PointStruct::new(id, vectors, Payload::from(point.payload))
}

fn to_vector(v: VectorValue) -> Vector {
    match v {
        VectorValue::Dense(data) => Vector::from(data),
        VectorValue::Multi(data) => Vector::new_multi(data),
        VectorValue::Sparse { indices, values } => Vector::new_sparse(indices, values),
    }
}

fn resolve_upsert_wait(from_params: bool) -> bool {
    // NOVA_UPSERT_WAIT overrides the YAML, matching the Python loader.
    match std::env::var("NOVA_UPSERT_WAIT") {
        Ok(v) => matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => from_params,
    }
}

#[derive(Debug, Deserialize)]
pub struct QdrantConfig {
    pub url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_collection")]
    pub collection_name: String,
    #[serde(default)]
    pub params: QdrantParams,
}

/// Collection-creation + behavior tuning. `deny_unknown_fields` turns a typo
/// (`shard_numbr`) into a hard parse error instead of a silently dropped param.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QdrantParams {
    /// Wait for each upsert to be applied before returning (default: false).
    #[serde(default)]
    pub upsert_wait: bool,
    /// Drop + recreate the collection if it exists with a conflicting config.
    #[serde(default)]
    pub recreate: bool,

    pub shard_number: Option<u32>,
    pub sharding_method: Option<String>,
    pub replication_factor: Option<u32>,
    pub write_consistency_factor: Option<u32>,
    pub on_disk_payload: Option<bool>,

    pub hnsw_config: Option<HnswConfig>,
    pub optimizers_config: Option<OptimizersConfig>,
    pub quantization: Option<QuantizationConfig>,
    // TODO: wal_config / strict_mode_config when a benchmark needs them.
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HnswConfig {
    pub m: Option<u64>,
    pub ef_construct: Option<u64>,
    pub full_scan_threshold: Option<u64>,
    pub max_indexing_threads: Option<u64>,
    pub on_disk: Option<bool>,
    pub payload_m: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizersConfig {
    pub deleted_threshold: Option<f64>,
    pub vacuum_min_vector_number: Option<u64>,
    pub default_segment_number: Option<u64>,
    pub max_segment_size: Option<u64>,
    pub memmap_threshold: Option<u64>,
    pub indexing_threshold: Option<u64>,
    pub flush_interval_sec: Option<u64>,
    pub max_optimization_threads: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationConfig {
    #[serde(rename = "type", default = "default_quant_kind")]
    pub kind: QuantizationKind,
    #[serde(default = "default_true")]
    pub always_ram: bool,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationKind {
    Scalar,
    Binary,
}

fn default_collection() -> String {
    "default".to_string()
}
fn default_true() -> bool {
    true
}
fn default_quant_kind() -> QuantizationKind {
    QuantizationKind::Scalar
}

// Needs a source variant too, since the parsed config requires a `datasource`.
// Run with `--features full` (or `qdrant,local`).
#[cfg(all(test, feature = "local"))]
mod tests {
    use crate::config::load_config_str;
    use crate::stores::VectorStoreConfig;

    const YAML: &str = r#"
vectors:
  dense:
    type: dense
    column: dense_embedding
    distance: cosine
vectorstore:
  type: qdrant
  collection_name: mteb_tweets
  url: http://localhost:6333
  params:
    shard_number: 12
    replication_factor: 2
    recreate: false
datasource:
  type: local
  path: /data/*.parquet
loader:
  batch_size: 256
  concurrency: 4
"#;

    #[test]
    fn parses_qdrant_config() {
        let cfg = load_config_str(YAML).expect("should parse");
        assert_eq!(cfg.loader.batch_size, 256);
        match &cfg.vectorstore {
            VectorStoreConfig::Qdrant(q) => {
                assert_eq!(q.collection_name, "mteb_tweets");
                assert_eq!(q.params.shard_number, Some(12));
                assert_eq!(q.params.replication_factor, Some(2));
                assert!(!q.params.recreate);
            }
        }
    }

    #[test]
    fn rejects_unknown_param_key() {
        let yaml = YAML.replace("shard_number:", "shard_numbr:");
        let err = load_config_str(&yaml).expect_err("typo should be rejected");
        assert!(matches!(err, crate::config::ConfigError::Yaml(_)));
    }
}