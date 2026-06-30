use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

use qdrant_client::qdrant::{
    CollectionStatus, CreateCollectionBuilder, Datatype, Distance, HnswConfigDiff, Modifier,
    MultiVectorComparator, MultiVectorConfigBuilder, OptimizersConfigDiff,
    OptimizersConfigDiffBuilder, PointStruct, QuantizationType, ScalarQuantization,
    SparseIndexConfigBuilder, SparseVectorConfig, SparseVectorParams, SparseVectorParamsBuilder,
    UpdateCollectionBuilder, UpsertPointsBuilder, Vector, VectorParams, VectorParamsBuilder,
    VectorParamsMap, VectorsConfig, vectors_config,
};
use qdrant_client::{Payload, Qdrant};

use crate::config::{HnswConfig, OptimizersConfig, QuantizationConfig, VectorKind, VectorSpec};
use crate::stores::{CollectionSchema, Point, PointId, StoreError, VectorStore, VectorValue};

/// Connection + store settings for a Qdrant backend, as written under
/// `vectorstore:` in the YAML.
///
/// Note what is *not* here: the per-vector schema (distance, size, datatype…)
/// lives in the top-level `vectors:` section, because the vector name is the key
/// shared between extraction (which parquet column) and storage (the collection
/// schema). The `CreateCollection` request is assembled in
/// [`build_create_collection`] from those vector specs plus the collection-wide
/// [`QdrantParams`] here.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QdrantConfig {
    pub url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_collection")]
    pub collection_name: String,
    /// Whether upserts block until the write is applied (slower, stronger).
    #[serde(default)]
    pub upsert_wait: bool,
    /// Collection-wide creation params. All optional; Qdrant defaults apply.
    #[serde(default)]
    pub params: Option<QdrantParams>,
}

/// Manual `Debug` so the API key never lands in logs, errors, or `--dry-run`
/// output — only whether one is set.
impl fmt::Debug for QdrantConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QdrantConfig")
            .field("url", &self.url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("collection_name", &self.collection_name)
            .field("upsert_wait", &self.upsert_wait)
            .field("params", &self.params)
            .finish()
    }
}

/// Collection-wide knobs — attributes of the whole collection, not of any single
/// vector. (Per-vector knobs belong on [`VectorSpec`] in the `vectors:` section.)
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QdrantParams {
    #[serde(default)]
    pub shard_number: Option<u32>,
    #[serde(default)]
    pub replication_factor: Option<u32>,
    #[serde(default)]
    pub write_consistency_factor: Option<u32>,
    #[serde(default)]
    pub on_disk_payload: Option<bool>,
    /// Collection-wide HNSW index overrides.
    #[serde(default)]
    pub hnsw: Option<HnswConfig>,
    /// Collection-wide scalar quantization.
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    /// Optimizer overrides.
    #[serde(default)]
    pub optimizers: Option<OptimizersConfig>,
    /// If the collection already exists with incompatible (immutable) params,
    /// drop + recreate instead of erroring. Consumed by the loader, not part of
    /// the `CreateCollection` request itself.
    #[serde(default)]
    pub recreate: bool,
}

/// Errors that surface only when translating the backend-agnostic config into
/// concrete Qdrant request types.
#[derive(Debug, thiserror::Error)]
pub enum QdrantConfigError {
    #[error(
        "vector `{0}` is dense but has no size: set `size:` in the vectors config or ensure the loader can infer it from the parquet schema"
    )]
    MissingSize(String),
    #[error(
        "vector `{name}`: unknown distance `{value}` (expected one of: cosine, dot, euclid, manhattan)"
    )]
    UnknownDistance { name: String, value: String },
    #[error(
        "vector `{name}`: unknown datatype `{value}` (expected one of: float32, uint8, float16)"
    )]
    UnknownDatatype { name: String, value: String },
    #[error("vector `{name}`: unknown comparator `{value}` (expected: max_sim)")]
    UnknownComparator { name: String, value: String },
    #[error("vector `{name}`: unknown sparse modifier `{value}` (expected one of: none, idf)")]
    UnknownModifier { name: String, value: String },
    #[error("unknown quantization type `{0}` (expected: int8)")]
    UnknownQuantizationType(String),
}

/// Build a `CreateCollection` request by gathering fields from three sources:
/// - `collection_name` from the store config
/// - per-vector schema from `vectors` (the top-level specs), partitioned into
///   dense (`vectors_config`) and sparse (`sparse_vectors_config`)
/// - collection-wide knobs from `params`
///
/// `dims` supplies resolved dimensionality for dense vectors by name. The loader
/// fills it from the parquet schema; an explicit `size:` on a [`VectorSpec`]
/// takes precedence. Sparse vectors ignore `dims`.
pub fn build_create_collection(
    collection_name: &str,
    vectors: &HashMap<String, VectorSpec>,
    params: &QdrantParams,
    dims: &HashMap<String, u64>,
) -> Result<CreateCollectionBuilder, QdrantConfigError> {
    let mut dense: HashMap<String, VectorParams> = HashMap::new();
    let mut sparse: HashMap<String, SparseVectorParams> = HashMap::new();

    for (name, spec) in vectors {
        match spec.kind {
            VectorKind::Dense => {
                dense.insert(name.clone(), dense_params(name, spec, dims, false)?);
            }
            VectorKind::Multivector => {
                dense.insert(name.clone(), dense_params(name, spec, dims, true)?);
            }
            VectorKind::Sparse => {
                sparse.insert(name.clone(), sparse_params(name, spec)?);
            }
        }
    }

    // Named vectors always go in a map, even when there's only one.
    let mut builder = CreateCollectionBuilder::new(collection_name);
    if !dense.is_empty() {
        builder = builder.vectors_config(VectorsConfig {
            config: Some(vectors_config::Config::ParamsMap(VectorParamsMap::from(
                dense,
            ))),
        });
    }
    if !sparse.is_empty() {
        builder = builder.sparse_vectors_config(SparseVectorConfig::from(sparse));
    }

    if let Some(v) = params.shard_number {
        builder = builder.shard_number(v);
    }
    if let Some(v) = params.replication_factor {
        builder = builder.replication_factor(v);
    }
    if let Some(v) = params.write_consistency_factor {
        builder = builder.write_consistency_factor(v);
    }
    if let Some(v) = params.on_disk_payload {
        builder = builder.on_disk_payload(v);
    }
    if let Some(h) = &params.hnsw {
        builder = builder.hnsw_config(hnsw_diff(h));
    }
    if let Some(q) = &params.quantization {
        builder = builder.quantization_config(scalar_quant(q)?);
    }
    if let Some(o) = &params.optimizers {
        builder = builder.optimizers_config(optimizers_diff(o));
    }

    Ok(builder)
}

/// Build dense (or multivector) `VectorParams` from one spec.
fn dense_params(
    name: &str,
    spec: &VectorSpec,
    dims: &HashMap<String, u64>,
    multivector: bool,
) -> Result<VectorParams, QdrantConfigError> {
    let size = spec
        .size
        .or_else(|| dims.get(name).copied())
        .ok_or_else(|| QdrantConfigError::MissingSize(name.to_string()))?;

    let mut b = VectorParamsBuilder::new(size, parse_distance(name, spec.distance.as_deref())?);
    if let Some(on_disk) = spec.on_disk {
        b = b.on_disk(on_disk);
    }
    if let Some(dt) = parse_datatype(name, spec.datatype.as_deref())? {
        b = b.datatype(dt);
    }
    if multivector {
        let comparator = parse_comparator(name, spec.comparator.as_deref())?;
        b = b.multivector_config(MultiVectorConfigBuilder::new(comparator));
    }
    Ok(b.build())
}

fn sparse_params(name: &str, spec: &VectorSpec) -> Result<SparseVectorParams, QdrantConfigError> {
    let mut idx = SparseIndexConfigBuilder::default();
    if let Some(on_disk) = spec.on_disk {
        idx = idx.on_disk(on_disk);
    }
    if let Some(dt) = parse_datatype(name, spec.datatype.as_deref())? {
        idx = idx.datatype(dt);
    }
    let mut b = SparseVectorParamsBuilder::default().index(idx);
    if let Some(m) = parse_modifier(name, spec.modifier.as_deref())? {
        b = b.modifier(m);
    }
    Ok(b.build())
}

/// Map the config HNSW knobs onto a qdrant `HnswConfigDiff` (unset fields stay
/// `None`, so the server keeps its defaults).
fn hnsw_diff(h: &HnswConfig) -> HnswConfigDiff {
    HnswConfigDiff {
        m: h.m,
        ef_construct: h.ef_construct,
        full_scan_threshold: h.full_scan_threshold,
        max_indexing_threads: h.max_indexing_threads,
        on_disk: h.on_disk,
        payload_m: h.payload_m,
        ..Default::default()
    }
}

fn optimizers_diff(o: &OptimizersConfig) -> OptimizersConfigDiff {
    OptimizersConfigDiff {
        deleted_threshold: o.deleted_threshold,
        vacuum_min_vector_number: o.vacuum_min_vector_number,
        default_segment_number: o.default_segment_number,
        max_segment_size: o.max_segment_size,
        memmap_threshold: o.memmap_threshold,
        indexing_threshold: o.indexing_threshold,
        flush_interval_sec: o.flush_interval_sec,
        ..Default::default()
    }
}

fn scalar_quant(q: &QuantizationConfig) -> Result<ScalarQuantization, QdrantConfigError> {
    // Only scalar int8 is supported today; default to it when unspecified.
    let kind = match q.kind.as_deref().map(str::to_ascii_lowercase).as_deref() {
        None | Some("int8") => QuantizationType::Int8,
        Some(other) => {
            return Err(QdrantConfigError::UnknownQuantizationType(
                other.to_string(),
            ));
        }
    };
    Ok(ScalarQuantization {
        r#type: kind as i32,
        quantile: q.quantile,
        always_ram: q.always_ram,
    })
}

fn parse_modifier(name: &str, value: Option<&str>) -> Result<Option<Modifier>, QdrantConfigError> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        None => Ok(None),
        Some("none") => Ok(Some(Modifier::None)),
        Some("idf") => Ok(Some(Modifier::Idf)),
        Some(other) => Err(QdrantConfigError::UnknownModifier {
            name: name.to_string(),
            value: other.to_string(),
        }),
    }
}

fn parse_distance(name: &str, value: Option<&str>) -> Result<Distance, QdrantConfigError> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        // Cosine is the sensible default for embeddings when unspecified.
        None | Some("cosine") => Ok(Distance::Cosine),
        Some("dot") => Ok(Distance::Dot),
        Some("euclid" | "euclidean" | "l2") => Ok(Distance::Euclid),
        Some("manhattan" | "l1") => Ok(Distance::Manhattan),
        Some(other) => Err(QdrantConfigError::UnknownDistance {
            name: name.to_string(),
            value: other.to_string(),
        }),
    }
}

fn parse_datatype(name: &str, value: Option<&str>) -> Result<Option<Datatype>, QdrantConfigError> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        None => Ok(None),
        Some("float32" | "f32") => Ok(Some(Datatype::Float32)),
        Some("uint8" | "u8") => Ok(Some(Datatype::Uint8)),
        Some("float16" | "f16") => Ok(Some(Datatype::Float16)),
        Some(other) => Err(QdrantConfigError::UnknownDatatype {
            name: name.to_string(),
            value: other.to_string(),
        }),
    }
}

fn parse_comparator(
    name: &str,
    value: Option<&str>,
) -> Result<MultiVectorComparator, QdrantConfigError> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        None | Some("max_sim" | "maxsim") => Ok(MultiVectorComparator::MaxSim),
        Some(other) => Err(QdrantConfigError::UnknownComparator {
            name: name.to_string(),
            value: other.to_string(),
        }),
    }
}
fn default_collection() -> String {
    "default".to_string()
}

/// A connected Qdrant backend. Unlike the data sources (where the config itself
/// implements the trait), the store holds an initialized client reused across
/// every `upsert_batch`, so it's a distinct runtime object built once.
pub struct QdrantStore {
    client: Qdrant,
    collection_name: String,
    params: QdrantParams,
    upsert_wait: bool,
}

impl QdrantConfig {
    /// Build the client once. Consumes the config to avoid cloning its fields.
    pub async fn connect(self) -> Result<QdrantStore, StoreError> {
        let client = Qdrant::from_url(&self.url)
            .api_key(self.api_key)
            // .check_compatibility(false) // skip since the log is annoying
            .build()?;
        Ok(QdrantStore {
            client,
            collection_name: self.collection_name,
            params: self.params.unwrap_or_default(),
            upsert_wait: self.upsert_wait,
        })
    }
}

impl fmt::Display for QdrantStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "qdrant({})", self.collection_name)
    }
}

/// One named vector value → the qdrant wire `Vector` (leans on qdrant-client's
/// own `From` impls for each shape).
impl From<VectorValue> for Vector {
    fn from(value: VectorValue) -> Self {
        match value {
            VectorValue::Dense(d) => d.into(),
            VectorValue::Multi(m) => m.into(),
            VectorValue::Sparse { indices, values } => {
                indices.into_iter().zip(values).collect::<Vec<(u32, f32)>>().into()
            }
        }
    }
}

/// A read point → a qdrant `PointStruct`. Allowed by the orphan rule because
/// `Point` is local; gives `Into<PointStruct>` for free.
impl From<Point> for PointStruct {
    fn from(point: Point) -> Self {
        let vectors: HashMap<String, Vector> = point
            .vectors
            .into_iter()
            .map(|(name, value)| (name, value.into()))
            .collect();
        let payload = Payload::from(point.payload);
        match point.id {
            PointId::Integer(n) => PointStruct::new(n, vectors, payload),
            PointId::String(s) => PointStruct::new(s, vectors, payload),
        }
    }
}

#[async_trait]
impl VectorStore for QdrantStore {
    async fn ensure_collection(&self, schema: &CollectionSchema) -> Result<(), StoreError> {
        let exists = self.client.collection_exists(self.collection_name.as_str()).await?;
        if exists {
            if !self.params.recreate {
                // We dont confirm that the collection schema matches the config
                // in the future we could... but for now, just assume the user knows what they're doing if they set recreate=false.
                return Ok(());
            }
            self.client.delete_collection(self.collection_name.as_str()).await?;
        }

        let request = build_create_collection(
            &self.collection_name,
            &schema.vectors,
            &self.params,
            &schema.dims,
        )
        .map_err(|e| StoreError::Other(e.to_string()))?;
        self.client.create_collection(request).await?;
        Ok(())
    }

    async fn upsert_batch(&self, points: Vec<Point>) -> Result<(), StoreError> {
        let points: Vec<PointStruct> = points.into_iter().map(PointStruct::from).collect();
        self.client
            .upsert_points(
                UpsertPointsBuilder::new(self.collection_name.as_str(), points)
                    .wait(self.upsert_wait),
            )
            .await?;
        Ok(())
    }

    async fn close(&self) -> Result<(), StoreError> {
        // The gRPC channel cleans up on drop; nothing to do.
        Ok(())
    }

    async fn defer_indexing(&self) -> Result<(), StoreError> {
        // Stop the indexing optimizer so bulk upserts don't pay HNSW build cost.
        self.client
            .update_collection(
                UpdateCollectionBuilder::new(self.collection_name.as_str())
                    .optimizers_config(OptimizersConfigDiffBuilder::default().indexing_threshold(0)),
            )
            .await?;
        Ok(())
    }

    async fn enable_indexing(&self) -> Result<(), StoreError> {
        let threshold = self
            .params
            .optimizers
            .as_ref()
            .and_then(|o| o.indexing_threshold)
            .unwrap_or(20_000);
        self.client
            .update_collection(
                UpdateCollectionBuilder::new(self.collection_name.as_str()).optimizers_config(
                    OptimizersConfigDiffBuilder::default().indexing_threshold(threshold),
                ),
            )
            .await?;
        Ok(())
    }

    async fn wait_for_indexing(&self) -> Result<(), StoreError> {
        // Poll until the collection reports green (optimizers idle).
        // TODO: add a timeout/backoff so a stuck optimizer can't loop forever.
        loop {
            let resp = self.client.collection_info(self.collection_name.as_str()).await?;
            let status = resp.result.map(|r| r.status).unwrap_or_default();
            if status == CollectionStatus::Green as i32 {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rstest::rstest;

    use qdrant_client::qdrant::quantization_config;

    use crate::config::LoadConfig;
    use crate::stores::VectorStoreConfig;

    /// Deserialize the "every param" fixture at the repo root into a `LoadConfig`.
    fn load_fixture() -> LoadConfig {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/configs/qdrant_all_params.yaml"
        );
        let yaml =
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read fixture {path}: {e}"));
        serde_yaml::from_str(&yaml).expect("fixture should deserialize into LoadConfig")
    }

    fn qdrant_store(cfg: &LoadConfig) -> &QdrantConfig {
        let VectorStoreConfig::Qdrant(store) = &cfg.vectorstore;
        store
    }

    /// A minimal dense spec with everything unset, for the size-resolution tests.
    fn bare_dense_spec() -> VectorSpec {
        VectorSpec {
            kind: VectorKind::Dense,
            column: "c".into(),
            size: None,
            distance: None,
            comparator: None,
            datatype: None,
            on_disk: None,
            modifier: None,
        }
    }

    /// Every collection param we support must round-trip from YAML into the
    /// `CreateCollection` request. When you add a knob to `VectorSpec` or
    /// `QdrantParams`, add it to `tests/configs/qdrant_all_params.yaml` and assert
    /// it here — that pairing is what keeps "all params available" honest.
    #[test]
    fn all_collection_params_flow_through() {
        let cfg = load_fixture();
        let store = qdrant_store(&cfg);

        // Store-level settings (not part of the CreateCollection request).
        assert_eq!(store.url, "http://localhost:6334");
        assert_eq!(store.api_key.as_deref(), Some("secret"));
        assert_eq!(store.collection_name, "everything");
        assert!(store.upsert_wait);
        assert!(store.params.as_ref().is_some_and(|p| p.recreate));

        // Sizes are explicit in the fixture, so no inference is needed.
        let cc = build_create_collection(
            &store.collection_name,
            &cfg.vectors,
            store.params.as_ref().unwrap_or(&QdrantParams::default()),
            &HashMap::new(),
        )
        .expect("build should succeed")
        .build();

        // Collection-wide params.
        assert_eq!(cc.collection_name, "everything");
        assert_eq!(cc.shard_number, Some(12));
        assert_eq!(cc.replication_factor, Some(2));
        assert_eq!(cc.write_consistency_factor, Some(3));
        assert_eq!(cc.on_disk_payload, Some(true));

        // Collection-wide HNSW.
        let chnsw = cc.hnsw_config.as_ref().expect("collection hnsw");
        assert_eq!(chnsw.m, Some(16));
        assert_eq!(chnsw.ef_construct, Some(100));
        assert_eq!(chnsw.full_scan_threshold, Some(10000));
        assert_eq!(chnsw.max_indexing_threads, Some(4));
        assert_eq!(chnsw.on_disk, Some(true));
        assert_eq!(chnsw.payload_m, Some(8));

        // Collection-wide scalar quantization.
        let quant = match cc
            .quantization_config
            .as_ref()
            .and_then(|q| q.quantization.as_ref())
            .expect("quantization present")
        {
            quantization_config::Quantization::Scalar(s) => s,
            other => panic!("expected scalar quantization, got {other:?}"),
        };
        assert_eq!(quant.r#type, QuantizationType::Int8 as i32);
        assert_eq!(quant.quantile, Some(0.99));
        assert_eq!(quant.always_ram, Some(true));

        // Collection-wide optimizers.
        let opt = cc.optimizers_config.as_ref().expect("optimizers");
        assert_eq!(opt.default_segment_number, Some(4));
        assert_eq!(opt.max_segment_size, Some(200000)); // set via the `max_segment_size_kb` alias
        assert_eq!(opt.memmap_threshold, Some(50000));
        assert_eq!(opt.indexing_threshold, Some(20000));

        // Dense + multivector params live in the named-vector map.
        let dense = match cc
            .vectors_config
            .as_ref()
            .and_then(|v| v.config.as_ref())
            .expect("vectors_config present")
        {
            vectors_config::Config::ParamsMap(m) => &m.map,
            other => panic!("expected named ParamsMap, got {other:?}"),
        };

        let d = &dense["dense"];
        assert_eq!(d.size, 384);
        assert_eq!(d.distance, Distance::Cosine as i32);
        assert_eq!(d.datatype, Some(Datatype::Float32 as i32));
        assert_eq!(d.on_disk, Some(true));
        assert!(d.multivector_config.is_none());
        // HNSW/quantization are collection-wide, not per-vector.
        assert!(d.hnsw_config.is_none());
        assert!(d.quantization_config.is_none());

        let c = &dense["colbert"];
        assert_eq!(c.size, 128);
        assert_eq!(c.distance, Distance::Dot as i32);
        assert_eq!(c.datatype, Some(Datatype::Float16 as i32));
        assert_eq!(c.on_disk, Some(false));
        assert_eq!(
            c.multivector_config
                .as_ref()
                .expect("multivector_config")
                .comparator,
            MultiVectorComparator::MaxSim as i32,
        );

        // Sparse params live in their own map.
        let sparse = &cc
            .sparse_vectors_config
            .as_ref()
            .expect("sparse_vectors_config present")
            .map;
        let s = &sparse["sparse"];
        assert_eq!(
            s.index.as_ref().expect("sparse index config").on_disk,
            Some(true),
        );
        assert_eq!(s.modifier, Some(Modifier::Idf as i32));
    }

    /// Dense size is required: absent from both the spec and the inferred dims,
    /// the build must fail loudly rather than silently produce a 0-dim vector.
    #[test]
    fn dense_without_size_errors() {
        let mut vectors = HashMap::new();
        vectors.insert("d".to_string(), bare_dense_spec());
        let err = build_create_collection("c", &vectors, &QdrantParams::default(), &HashMap::new())
            .unwrap_err();
        assert!(matches!(err, QdrantConfigError::MissingSize(_)));
    }

    /// An explicit `size:` is absent but the loader supplies it via `dims`.
    #[test]
    fn dense_size_inferred_from_dims() {
        let mut vectors = HashMap::new();
        vectors.insert("d".to_string(), bare_dense_spec());
        let dims = HashMap::from([("d".to_string(), 768u64)]);
        let cc = build_create_collection("c", &vectors, &QdrantParams::default(), &dims)
            .expect("build")
            .build();
        let dense = match cc.vectors_config.unwrap().config.unwrap() {
            vectors_config::Config::ParamsMap(m) => m.map,
            other => panic!("expected ParamsMap, got {other:?}"),
        };
        assert_eq!(dense["d"].size, 768);
    }

    #[rstest]
    #[case(None, Distance::Cosine)]
    #[case(Some("cosine"), Distance::Cosine)]
    #[case(Some("Dot"), Distance::Dot)]
    #[case(Some("euclid"), Distance::Euclid)]
    #[case(Some("euclidean"), Distance::Euclid)]
    #[case(Some("l2"), Distance::Euclid)]
    #[case(Some("manhattan"), Distance::Manhattan)]
    #[case(Some("L1"), Distance::Manhattan)]
    fn distance_parses(#[case] input: Option<&str>, #[case] expected: Distance) {
        assert_eq!(parse_distance("v", input).unwrap() as i32, expected as i32);
    }

    #[rstest]
    #[case(Some("float32"), Some(Datatype::Float32))]
    #[case(Some("f16"), Some(Datatype::Float16))]
    #[case(Some("uint8"), Some(Datatype::Uint8))]
    #[case(None, None)]
    fn datatype_parses(#[case] input: Option<&str>, #[case] expected: Option<Datatype>) {
        let got = parse_datatype("v", input).unwrap();
        assert_eq!(got.map(|d| d as i32), expected.map(|d| d as i32));
    }

    #[rstest]
    #[case(None, None)]
    #[case(Some("none"), Some(Modifier::None))]
    #[case(Some("IDF"), Some(Modifier::Idf))]
    fn modifier_parses(#[case] input: Option<&str>, #[case] expected: Option<Modifier>) {
        let got = parse_modifier("v", input).unwrap();
        assert_eq!(got.map(|m| m as i32), expected.map(|m| m as i32));
    }

    #[test]
    fn quantization_defaults_to_int8() {
        let q = QuantizationConfig::default();
        assert_eq!(
            scalar_quant(&q).unwrap().r#type,
            QuantizationType::Int8 as i32
        );
    }

    #[test]
    fn unknown_values_error() {
        assert!(matches!(
            parse_distance("v", Some("hamming")),
            Err(QdrantConfigError::UnknownDistance { .. })
        ));
        assert!(matches!(
            parse_datatype("v", Some("bf16")),
            Err(QdrantConfigError::UnknownDatatype { .. })
        ));
        assert!(matches!(
            parse_comparator("v", Some("min_sim")),
            Err(QdrantConfigError::UnknownComparator { .. })
        ));
        assert!(matches!(
            parse_modifier("v", Some("relu")),
            Err(QdrantConfigError::UnknownModifier { .. })
        ));
        assert!(matches!(
            scalar_quant(&QuantizationConfig {
                kind: Some("fp4".into()),
                ..Default::default()
            }),
            Err(QdrantConfigError::UnknownQuantizationType(_))
        ));
    }
}
