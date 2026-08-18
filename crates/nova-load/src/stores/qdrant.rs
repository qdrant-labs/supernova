use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Deserialize;

use qdrant_client::qdrant::{
    BinaryQuantization, BinaryQuantizationEncoding, CollectionStatus, CompressionRatio,
    CreateCollectionBuilder, CreateShardKeyBuilder, CreateShardKeyRequestBuilder, Datatype,
    Disabled, Distance, HnswConfigDiff, Modifier, MultiVectorComparator, MultiVectorConfigBuilder,
    OptimizersConfigDiff, OptimizersConfigDiffBuilder, PointStruct, ProductQuantization,
    QuantizationType, ScalarQuantization, ShardKey, ShardKeySelector, ShardingMethod,
    SparseIndexConfigBuilder, SparseVectorConfig, SparseVectorParams, SparseVectorParamsBuilder,
    TurboQuantBitSize, TurboQuantization, UpdateCollectionBuilder, UpsertPointsBuilder, Vector,
    VectorParams, VectorParamsBuilder, VectorParamsMap, VectorsConfig, quantization_config,
    quantization_config_diff, vectors_config,
};
use qdrant_client::{Payload, Qdrant};

use crate::config::{HnswConfig, OptimizersConfig, QuantizationConfig, VectorKind, VectorSpec};
use crate::stores::{
    CollectionSchema, CustomSharding, Point, PointId, ShardKeyValue, StoreError, VectorStore,
    VectorValue,
};

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
    /// Custom (user-defined) sharding: the collection is created with
    /// `sharding_method: custom` and every point routes to the shard key its
    /// `shard_key` expression evaluates to. See [`CustomSharding`].
    #[serde(default)]
    pub custom_sharding: Option<CustomSharding>,
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
            .field("custom_sharding", &self.custom_sharding)
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
    #[error("unknown quantization type `{0}` (expected one of: scalar, product, binary, turbo, none)")]
    UnknownQuantizationType(String),
    #[error("unknown quantization compression `{0}` (expected one of: x4, x8, x16, x32, x64)")]
    UnknownCompressionRatio(String),
    #[error(
        "unknown quantization encoding `{0}` (expected one of: one_bit, two_bits, one_and_half_bits)"
    )]
    UnknownQuantizationEncoding(String),
    #[error("unknown quantization `bits` `{0}` (expected one of: 1, 1.5, 2, 4)")]
    UnknownTurboBits(f32),
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
///
/// `custom_sharding` switches the collection to `sharding_method: custom`; note
/// that flips `shard_number`'s meaning to *shards per shard key*.
pub fn build_create_collection(
    collection_name: &str,
    vectors: &HashMap<String, VectorSpec>,
    params: &QdrantParams,
    dims: &HashMap<String, u64>,
    custom_sharding: Option<&CustomSharding>,
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

    if custom_sharding.is_some() {
        builder = builder.sharding_method(ShardingMethod::Custom as i32);
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
    if let Some(q) = &params.quantization
        && let Some(quant) = quantization_for_create(q)?
    {
        builder = builder.quantization_config(quant);
    }
    if let Some(o) = &params.optimizers {
        builder = builder.optimizers_config(optimizers_diff(o));
    }

    Ok(builder)
}

/// Build an `UpdateCollection` request that patches index-affecting settings
/// (HNSW/quantization/optimizers) on an *already-existing* collection, from
/// the same [`QdrantParams`] shape [`build_create_collection`] reads at
/// creation time — reusing the identical `hnsw_diff`/`optimizers_diff`/
/// `parse_quantization` helpers, just fed into `UpdateCollectionBuilder`
/// instead of `CreateCollectionBuilder` (and, for quantization, through
/// [`quantization_for_update`] instead of [`quantization_for_create`], since
/// `type: none` clears quantization here rather than being a no-op). Only
/// patches knobs that are actually `Some`; anything unset in `params` is left
/// alone server-side, not reset to a default. Structural params
/// (`shard_number`, `replication_factor`, per-vector `distance`/`datatype`/
/// `size`) aren't patchable on an existing collection and are deliberately
/// not read here — see `nova sweep`'s `data_layouts`/`index_variants` split
/// for why.
pub fn build_update_collection(
    collection_name: &str,
    params: &QdrantParams,
) -> Result<UpdateCollectionBuilder, QdrantConfigError> {
    let mut builder = UpdateCollectionBuilder::new(collection_name);
    if let Some(h) = &params.hnsw {
        builder = builder.hnsw_config(hnsw_diff(h));
    }
    if let Some(q) = &params.quantization {
        builder = builder.quantization_config(quantization_for_update(q)?);
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

/// One fully-parsed quantization method, before it's converted into whichever
/// protobuf shape the caller needs ([`quantization_config::Quantization`] for
/// `CreateCollection`, [`quantization_config_diff::Quantization`] for
/// `UpdateCollection`). `None` (no quantization) only has a diff
/// representation — see [`quantization_for_create`].
enum ParsedQuantization {
    None,
    Scalar(ScalarQuantization),
    Product(ProductQuantization),
    Binary(BinaryQuantization),
    Turbo(TurboQuantization),
}

fn parse_quantization(q: &QuantizationConfig) -> Result<ParsedQuantization, QdrantConfigError> {
    match q.kind.as_deref().map(str::to_ascii_lowercase).as_deref() {
        // A bare `quantization: {}` block means scalar (int8) quantization.
        None | Some("scalar") => Ok(ParsedQuantization::Scalar(ScalarQuantization {
            r#type: QuantizationType::Int8 as i32,
            quantile: q.quantile,
            always_ram: q.always_ram,
        })),
        Some("product") => Ok(ParsedQuantization::Product(ProductQuantization {
            compression: parse_compression(q.compression.as_deref())? as i32,
            always_ram: q.always_ram,
        })),
        Some("binary") => Ok(ParsedQuantization::Binary(BinaryQuantization {
            always_ram: q.always_ram,
            encoding: parse_quantization_encoding(q.encoding.as_deref())?.map(|e| e as i32),
            query_encoding: None,
        })),
        Some("turbo") => Ok(ParsedQuantization::Turbo(TurboQuantization {
            always_ram: q.always_ram,
            bits: parse_turbo_bits(q.bits)?.map(|b| b as i32),
        })),
        Some("none") => Ok(ParsedQuantization::None),
        Some(other) => Err(QdrantConfigError::UnknownQuantizationType(other.to_string())),
    }
}

fn parse_compression(value: Option<&str>) -> Result<CompressionRatio, QdrantConfigError> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        // x16 is Qdrant's own server-side default.
        None | Some("x16") => Ok(CompressionRatio::X16),
        Some("x4") => Ok(CompressionRatio::X4),
        Some("x8") => Ok(CompressionRatio::X8),
        Some("x32") => Ok(CompressionRatio::X32),
        Some("x64") => Ok(CompressionRatio::X64),
        Some(other) => Err(QdrantConfigError::UnknownCompressionRatio(other.to_string())),
    }
}

fn parse_quantization_encoding(
    value: Option<&str>,
) -> Result<Option<BinaryQuantizationEncoding>, QdrantConfigError> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        // one_bit is the server default; leave unset rather than round-trip it.
        None | Some("one_bit") => Ok(None),
        Some("two_bits") => Ok(Some(BinaryQuantizationEncoding::TwoBits)),
        Some("one_and_half_bits") => Ok(Some(BinaryQuantizationEncoding::OneAndHalfBits)),
        Some(other) => Err(QdrantConfigError::UnknownQuantizationEncoding(
            other.to_string(),
        )),
    }
}

fn parse_turbo_bits(value: Option<f32>) -> Result<Option<TurboQuantBitSize>, QdrantConfigError> {
    match value {
        // Leave unset rather than round-trip the server's own default.
        None => Ok(None),
        Some(1.0) => Ok(Some(TurboQuantBitSize::Bits1)),
        Some(1.5) => Ok(Some(TurboQuantBitSize::Bits15)),
        Some(2.0) => Ok(Some(TurboQuantBitSize::Bits2)),
        Some(4.0) => Ok(Some(TurboQuantBitSize::Bits4)),
        Some(other) => Err(QdrantConfigError::UnknownTurboBits(other)),
    }
}

/// Quantization config for `CreateCollection` — `None` (no quantization) is
/// a no-op here (`quantization_config::Quantization` has no "disabled"
/// variant; there's nothing to turn off on a collection that doesn't exist
/// yet), so the caller just skips setting `.quantization_config(...)`.
fn quantization_for_create(
    q: &QuantizationConfig,
) -> Result<Option<quantization_config::Quantization>, QdrantConfigError> {
    Ok(match parse_quantization(q)? {
        ParsedQuantization::None => None,
        ParsedQuantization::Scalar(s) => Some(s.into()),
        ParsedQuantization::Product(p) => Some(p.into()),
        ParsedQuantization::Binary(b) => Some(b.into()),
        ParsedQuantization::Turbo(t) => Some(t.into()),
    })
}

/// Quantization diff for `UpdateCollection` — unlike `create`, `None` is
/// meaningful here: it's how `nova load reindex` explicitly clears
/// quantization off a collection that already has it (omitting
/// `quantization:` from the config instead would leave the server's current
/// setting untouched), so it maps to Qdrant's `Disabled` diff variant.
fn quantization_for_update(
    q: &QuantizationConfig,
) -> Result<quantization_config_diff::Quantization, QdrantConfigError> {
    Ok(match parse_quantization(q)? {
        ParsedQuantization::None => Disabled {}.into(),
        ParsedQuantization::Scalar(s) => s.into(),
        ParsedQuantization::Product(p) => p.into(),
        ParsedQuantization::Binary(b) => b.into(),
        ParsedQuantization::Turbo(t) => t.into(),
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
    custom_sharding: Option<CustomSharding>,
    /// Shard keys known to exist on the server — created by this process, or
    /// observed after losing a create race. Async mutex because concurrent
    /// batch upserts can discover new keys simultaneously.
    ensured_keys: tokio::sync::Mutex<HashSet<ShardKeyValue>>,
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
            custom_sharding: self.custom_sharding,
            ensured_keys: tokio::sync::Mutex::new(HashSet::new()),
        })
    }
}

/// A loader shard-key value → the qdrant wire `ShardKey`.
fn qdrant_shard_key(value: &ShardKeyValue) -> ShardKey {
    match value {
        ShardKeyValue::Keyword(s) => s.clone().into(),
        ShardKeyValue::Number(n) => (*n).into(),
    }
}

/// A loader shard-key value → a single-key `ShardKeySelector` for upserts.
fn qdrant_shard_selector(value: &ShardKeyValue) -> ShardKeySelector {
    match value {
        ShardKeyValue::Keyword(s) => s.clone().into(),
        ShardKeyValue::Number(n) => (*n).into(),
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

impl QdrantStore {
    /// Make sure `key` exists as a shard key on the collection, creating it on
    /// first sight. Keys are created lazily as the load discovers them (the
    /// distinct key set isn't known up front and is never computed from the
    /// data), with an in-process cache so the steady-state cost is one lock +
    /// set lookup per batch. Racing creators — concurrent batches here, or
    /// other fleet workers — are fine: whoever loses re-checks
    /// `list_shard_keys` and accepts the key if it exists now.
    async fn ensure_shard_key(&self, key: &ShardKeyValue) -> Result<(), StoreError> {
        let Some(sharding) = &self.custom_sharding else { return Ok(()) };
        let mut seen = self.ensured_keys.lock().await;
        if seen.contains(key) {
            return Ok(());
        }

        let mut create = CreateShardKeyBuilder::default().shard_key(qdrant_shard_key(key));
        if let Some(n) = sharding.shards_number {
            create = create.shards_number(n);
        }
        if let Some(r) = sharding.replication_factor {
            create = create.replication_factor(r);
        }
        let request =
            CreateShardKeyRequestBuilder::new(self.collection_name.as_str()).request(create);
        match self.client.create_shard_key(request).await {
            Ok(resp) if resp.result => {
                tracing::info!("{self} created shard key `{key}`");
            }
            outcome => {
                // Failed or refused — usually a racing worker won. Accept if
                // the key exists now; otherwise surface the original failure.
                let live = self.client.list_shard_keys(self.collection_name.as_str()).await?;
                let want = qdrant_shard_key(key);
                if !live.shard_keys.iter().any(|d| d.key.as_ref() == Some(&want)) {
                    return Err(match outcome {
                        Err(err) => err.into(),
                        Ok(_) => StoreError::Other(format!(
                            "qdrant refused to create shard key `{key}` on {self}"
                        )),
                    });
                }
            }
        }
        seen.insert(key.clone());
        Ok(())
    }

    /// Create the configured `pre_create` shard keys — an explicit list from
    /// the config, never computed from the data. A no-op when empty. Runs from
    /// `ensure_collection`, so a fleet's master creates them during `prepare`.
    async fn precreate_shard_keys(&self) -> Result<(), StoreError> {
        let Some(sharding) = &self.custom_sharding else { return Ok(()) };
        for key in &sharding.pre_create {
            self.ensure_shard_key(key).await?;
        }
        Ok(())
    }

    /// When custom sharding is configured against an *existing* collection
    /// (recreate: false), confirm it was actually created with
    /// `sharding_method: custom` — otherwise every upsert would fail with an
    /// opaque server error, long after the downloads started.
    async fn verify_custom_sharding(&self) -> Result<(), StoreError> {
        if self.custom_sharding.is_none() {
            return Ok(());
        }
        let method = self
            .client
            .collection_info(self.collection_name.as_str())
            .await?
            .result
            .and_then(|r| r.config)
            .and_then(|c| c.params)
            .and_then(|p| p.sharding_method);
        if method != Some(ShardingMethod::Custom as i32) {
            return Err(StoreError::Other(format!(
                "custom_sharding is configured, but the existing collection `{}` was not \
                 created with custom sharding; recreate it (params.recreate: true) or drop \
                 the custom_sharding block",
                self.collection_name
            )));
        }
        Ok(())
    }

    /// Sanity check: confirm the collection's live config reflects the HNSW and
    /// optimizer index params we requested. Every field set in config must match
    /// what `collection_info` reports; unset fields are skipped (Qdrant keeps its
    /// defaults). Runs after indexing settles in `wait_for_indexing`.
    async fn verify_params(&self) -> Result<(), StoreError> {
        if self.params.hnsw.is_none() && self.params.optimizers.is_none() {
            return Ok(());
        }
        let config = self
            .client
            .collection_info(self.collection_name.as_str())
            .await?
            .result
            .and_then(|r| r.config);

        if let Some(want) = &self.params.hnsw {
            let live = config.as_ref().and_then(|c| c.hnsw_config.clone()).unwrap_or_default();
            self.check_u64("hnsw.m", want.m, live.m)?;
            self.check_u64("hnsw.ef_construct", want.ef_construct, live.ef_construct)?;
            self.check_u64(
                "hnsw.full_scan_threshold",
                want.full_scan_threshold,
                live.full_scan_threshold,
            )?;
            self.check_u64(
                "hnsw.max_indexing_threads",
                want.max_indexing_threads,
                live.max_indexing_threads,
            )?;
            self.check_u64("hnsw.payload_m", want.payload_m, live.payload_m)?;
            if let Some(w) = want.on_disk
                && live.on_disk != Some(w)
            {
                return Err(StoreError::Other(format!(
                    "sanity check FAILED on {self} hnsw.on_disk: requested {w} but live \
                     config has {:?}",
                    live.on_disk
                )));
            }
        }

        if let Some(want) = &self.params.optimizers {
            let live = config.as_ref().and_then(|c| c.optimizer_config.clone()).unwrap_or_default();
            self.check_u64(
                "optimizers.indexing_threshold",
                want.indexing_threshold,
                live.indexing_threshold,
            )?;
            self.check_u64(
                "optimizers.default_segment_number",
                want.default_segment_number,
                live.default_segment_number,
            )?;
            self.check_u64(
                "optimizers.max_segment_size",
                want.max_segment_size,
                live.max_segment_size,
            )?;
            self.check_u64("optimizers.memmap_threshold", want.memmap_threshold, live.memmap_threshold)?;
        }

        tracing::info!("{self} index params verified against live collection config");
        Ok(())
    }

    fn check_u64(&self, field: &str, want: Option<u64>, got: Option<u64>) -> Result<(), StoreError> {
        if let Some(w) = want
            && got != Some(w)
        {
            return Err(StoreError::Other(format!(
                "sanity check FAILED on {self} {field}: requested {w} but live config has {got:?}"
            )));
        }
        Ok(())
    }
}

#[async_trait]
impl VectorStore for QdrantStore {
    fn custom_sharding(&self) -> Option<&CustomSharding> {
        self.custom_sharding.as_ref()
    }

    async fn ensure_collection(&self, schema: &CollectionSchema) -> Result<(), StoreError> {
        let exists = self.client.collection_exists(self.collection_name.as_str()).await?;
        if exists {
            if !self.params.recreate {
                // We dont confirm that the collection schema matches the config
                // in the future we could... but for now, just assume the user knows what they're doing if they set recreate=false.
                // Custom sharding IS verified though: a mismatch there fails
                // every single upsert, so catch it before the load starts.
                self.verify_custom_sharding().await?;
                self.precreate_shard_keys().await?;
                return Ok(());
            }
            self.client.delete_collection(self.collection_name.as_str()).await?;
        }

        let request = build_create_collection(
            &self.collection_name,
            &schema.vectors,
            &self.params,
            &schema.dims,
            self.custom_sharding.as_ref(),
        )
        .map_err(|e| StoreError::Other(e.to_string()))?;
        self.client.create_collection(request).await?;
        self.precreate_shard_keys().await?;
        Ok(())
    }

    async fn upsert_batch(&self, points: Vec<Point>) -> Result<(), StoreError> {
        if points.is_empty() {
            return Ok(());
        }

        // Under custom sharding the selector scopes the whole request, so the
        // batch must be key-homogeneous — the loader guarantees it, but a mixed
        // batch would silently misroute points, so refuse rather than assume.
        let shard_key = match &self.custom_sharding {
            Some(_) => {
                let key = points.first().and_then(|p| p.shard_key.clone()).ok_or_else(|| {
                    StoreError::Other(
                        "custom_sharding is configured but a point arrived without a shard \
                         key (loader bug: the reader sets one per point)"
                            .into(),
                    )
                })?;
                if points.iter().any(|p| p.shard_key.as_ref() != Some(&key)) {
                    return Err(StoreError::Other(
                        "upsert batch mixes shard keys (loader bug: batches must be \
                         key-homogeneous)"
                            .into(),
                    ));
                }
                self.ensure_shard_key(&key).await?;
                Some(key)
            }
            None => None,
        };

        let points: Vec<PointStruct> = points.into_iter().map(PointStruct::from).collect();
        let mut request = UpsertPointsBuilder::new(self.collection_name.as_str(), points)
            .wait(self.upsert_wait);
        if let Some(key) = &shard_key {
            request = request.shard_key_selector(qdrant_shard_selector(key));
        }
        self.client.upsert_points(request).await?;
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

    async fn enable_indexing(&self, _schema: &CollectionSchema) -> Result<(), StoreError> {
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

    async fn wait_for_indexing(&self) -> Result<Instant, StoreError> {
        // Poll until the collection reports green (optimizers idle) *and
        // stays there* for GREEN_HOLD straight — a single Green sample isn't
        // enough of a guarantee: right after a config patch or bulk upsert,
        // the optimizer is scheduled asynchronously, so an immediate poll
        // can still observe stale Green left over from before the change
        // even started. Also fail fast (instead of looping forever) the
        // moment the optimizer itself reports broken, surfacing its error.
        const POLL_INTERVAL: Duration = Duration::from_secs(1);
        const GREEN_HOLD: Duration = Duration::from_secs(5);

        let mut green_since: Option<Instant> = None;
        // TODO: add an overall timeout so a stuck (non-erroring) optimizer can't loop forever.
        loop {
            let resp = self.client.collection_info(self.collection_name.as_str()).await?;
            let result = resp.result;

            if let Some(status) = result.as_ref().and_then(|r| r.optimizer_status.as_ref())
                && !status.ok
            {
                return Err(StoreError::Other(format!(
                    "qdrant optimizer error on {self}: {}",
                    status.error
                )));
            }

            let is_green = result.is_some_and(|r| r.status == CollectionStatus::Green as i32);
            if is_green {
                let since = *green_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= GREEN_HOLD {
                    // Sanity-check the live config against the requested index
                    // params. Runs after `since` (the converged instant we
                    // return), so it's not counted in the caller's build timing.
                    self.verify_params().await?;
                    return Ok(since);
                }
            } else {
                green_since = None;
            }

            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    async fn reindex(&self, _schema: &CollectionSchema) -> Result<(), StoreError> {
        let builder = build_update_collection(&self.collection_name, &self.params)
            .map_err(|e| StoreError::Other(e.to_string()))?;
        self.client.update_collection(builder).await?;
        Ok(())
    }

    async fn delete_collection(&self) -> Result<(), StoreError> {
        if self.client.collection_exists(self.collection_name.as_str()).await? {
            self.client.delete_collection(self.collection_name.as_str()).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rstest::rstest;

    use qdrant_client::qdrant::{quantization_config, quantization_config_diff};

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
        // Was irrefutable when Qdrant was the only variant. The elastic/milvus
        // features add variants; the wildcard arm is gated on them so this stays
        // an exhaustive, warning-free match whether or not those features are on.
        // (These tests are Qdrant-only.)
        match &cfg.vectorstore {
            VectorStoreConfig::Qdrant(store) => store.as_ref(),
            #[cfg(any(feature = "elastic", feature = "milvus"))]
            _ => panic!("test fixture must be a qdrant vectorstore"),
        }
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

        // Custom sharding: the expression + per-key knobs parse, and pre_create
        // accepts both keyword and number keys (untagged).
        let sharding = store.custom_sharding.as_ref().expect("custom_sharding present");
        assert_eq!(sharding.shard_key, "label");
        assert_eq!(sharding.shards_number, Some(2));
        assert_eq!(sharding.replication_factor, Some(2));
        assert_eq!(
            sharding.pre_create,
            vec![
                ShardKeyValue::Keyword("tenant_a".into()),
                ShardKeyValue::Number(42),
            ]
        );

        // Sizes are explicit in the fixture, so no inference is needed.
        let cc = build_create_collection(
            &store.collection_name,
            &cfg.vectors,
            store.params.as_ref().unwrap_or(&QdrantParams::default()),
            &HashMap::new(),
            store.custom_sharding.as_ref(),
        )
        .expect("build should succeed")
        .build();

        // Collection-wide params.
        assert_eq!(cc.collection_name, "everything");
        assert_eq!(cc.sharding_method, Some(ShardingMethod::Custom as i32));
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

    /// `build_update_collection` must patch exactly the knobs present in
    /// `params` — proves `reindex` would send the right diff without needing a
    /// live collection to reindex against.
    #[test]
    fn update_collection_carries_hnsw_quantization_and_optimizers() {
        let params = QdrantParams {
            hnsw: Some(HnswConfig { m: Some(32), ef_construct: Some(200), ..Default::default() }),
            quantization: Some(QuantizationConfig { kind: Some("scalar".into()), ..Default::default() }),
            optimizers: Some(OptimizersConfig { default_segment_number: Some(4), ..Default::default() }),
            ..Default::default()
        };
        let uc = build_update_collection("c", &params).expect("builds").build();

        assert_eq!(uc.collection_name, "c");
        let hnsw = uc.hnsw_config.expect("hnsw_config present");
        assert_eq!(hnsw.m, Some(32));
        assert_eq!(hnsw.ef_construct, Some(200));

        let quant = match uc.quantization_config.expect("quantization_config present").quantization {
            Some(quantization_config_diff::Quantization::Scalar(s)) => s,
            other => panic!("expected scalar quantization, got {other:?}"),
        };
        assert_eq!(quant.r#type, QuantizationType::Int8 as i32);

        let opt = uc.optimizers_config.expect("optimizers_config present");
        assert_eq!(opt.default_segment_number, Some(4));
    }

    /// Structural fields (shard_number, recreate, etc.) aren't part of an
    /// update — `build_update_collection` only ever reads hnsw/quantization/
    /// optimizers, so an all-unset `QdrantParams` produces an update request
    /// with nothing set (a legal, if pointless, no-op patch).
    #[test]
    fn update_collection_omits_unset_knobs() {
        let uc = build_update_collection("c", &QdrantParams::default()).expect("builds").build();
        assert!(uc.hnsw_config.is_none());
        assert!(uc.quantization_config.is_none());
        assert!(uc.optimizers_config.is_none());
    }

    /// End-to-end version of `none_quantization_is_a_create_noop_but_clears_on_update`
    /// through the actual `reindex` request path: a `nova load reindex` with
    /// `type: none` must patch quantization off.
    #[test]
    fn update_collection_can_disable_quantization() {
        let params = QdrantParams {
            quantization: Some(QuantizationConfig { kind: Some("none".into()), ..Default::default() }),
            ..Default::default()
        };
        let uc = build_update_collection("c", &params).expect("builds").build();
        assert!(matches!(
            uc.quantization_config.expect("quantization_config present").quantization,
            Some(quantization_config_diff::Quantization::Disabled(_))
        ));
    }

    /// Dense size is required: absent from both the spec and the inferred dims,
    /// the build must fail loudly rather than silently produce a 0-dim vector.
    #[test]
    fn dense_without_size_errors() {
        let mut vectors = HashMap::new();
        vectors.insert("d".to_string(), bare_dense_spec());
        let err =
            build_create_collection("c", &vectors, &QdrantParams::default(), &HashMap::new(), None)
                .unwrap_err();
        assert!(matches!(err, QdrantConfigError::MissingSize(_)));
    }

    /// An explicit `size:` is absent but the loader supplies it via `dims`.
    #[test]
    fn dense_size_inferred_from_dims() {
        let mut vectors = HashMap::new();
        vectors.insert("d".to_string(), bare_dense_spec());
        let dims = HashMap::from([("d".to_string(), 768u64)]);
        let cc = build_create_collection("c", &vectors, &QdrantParams::default(), &dims, None)
            .expect("build")
            .build();
        let dense = match cc.vectors_config.unwrap().config.unwrap() {
            vectors_config::Config::ParamsMap(m) => m.map,
            other => panic!("expected ParamsMap, got {other:?}"),
        };
        assert_eq!(dense["d"].size, 768);
    }

    /// Without `custom_sharding`, the create request must not set a sharding
    /// method — the server default (auto) applies.
    #[test]
    fn no_custom_sharding_leaves_sharding_method_unset() {
        let mut vectors = HashMap::new();
        vectors.insert("d".to_string(), bare_dense_spec());
        let dims = HashMap::from([("d".to_string(), 8u64)]);
        let cc = build_create_collection("c", &vectors, &QdrantParams::default(), &dims, None)
            .expect("build")
            .build();
        assert_eq!(cc.sharding_method, None);
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
    fn quantization_defaults_to_scalar_int8() {
        let q = QuantizationConfig::default();
        let quant = quantization_for_create(&q).unwrap();
        match quant {
            Some(quantization_config::Quantization::Scalar(s)) => {
                assert_eq!(s.r#type, QuantizationType::Int8 as i32);
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[rstest]
    #[case(None, CompressionRatio::X16)]
    #[case(Some("x4"), CompressionRatio::X4)]
    #[case(Some("x8"), CompressionRatio::X8)]
    #[case(Some("X16"), CompressionRatio::X16)]
    #[case(Some("x32"), CompressionRatio::X32)]
    #[case(Some("x64"), CompressionRatio::X64)]
    fn product_quantization_parses(
        #[case] compression: Option<&str>,
        #[case] expected: CompressionRatio,
    ) {
        let q = QuantizationConfig {
            kind: Some("product".into()),
            compression: compression.map(String::from),
            ..Default::default()
        };
        match quantization_for_create(&q).unwrap() {
            Some(quantization_config::Quantization::Product(p)) => {
                assert_eq!(p.compression, expected as i32);
            }
            other => panic!("expected product, got {other:?}"),
        }
    }

    #[rstest]
    #[case(None, None)]
    #[case(Some("one_bit"), None)]
    #[case(Some("two_bits"), Some(BinaryQuantizationEncoding::TwoBits))]
    #[case(
        Some("one_and_half_bits"),
        Some(BinaryQuantizationEncoding::OneAndHalfBits)
    )]
    fn binary_quantization_parses(
        #[case] encoding: Option<&str>,
        #[case] expected: Option<BinaryQuantizationEncoding>,
    ) {
        let q = QuantizationConfig {
            kind: Some("binary".into()),
            encoding: encoding.map(String::from),
            always_ram: Some(true),
            ..Default::default()
        };
        match quantization_for_create(&q).unwrap() {
            Some(quantization_config::Quantization::Binary(b)) => {
                assert_eq!(b.encoding, expected.map(|e| e as i32));
                assert_eq!(b.always_ram, Some(true));
            }
            other => panic!("expected binary, got {other:?}"),
        }
    }

    #[rstest]
    #[case(None, None)]
    #[case(Some(1.0), Some(TurboQuantBitSize::Bits1))]
    #[case(Some(1.5), Some(TurboQuantBitSize::Bits15))]
    #[case(Some(2.0), Some(TurboQuantBitSize::Bits2))]
    #[case(Some(4.0), Some(TurboQuantBitSize::Bits4))]
    fn turbo_quantization_parses(
        #[case] bits: Option<f32>,
        #[case] expected: Option<TurboQuantBitSize>,
    ) {
        let q = QuantizationConfig { kind: Some("turbo".into()), bits, ..Default::default() };
        match quantization_for_create(&q).unwrap() {
            Some(quantization_config::Quantization::Turboquant(t)) => {
                assert_eq!(t.bits, expected.map(|b| b as i32));
            }
            other => panic!("expected turbo, got {other:?}"),
        }
    }

    #[test]
    fn turbo_quantization_rejects_unknown_bits() {
        assert!(matches!(
            quantization_for_create(&QuantizationConfig {
                kind: Some("turbo".into()),
                bits: Some(3.0),
                ..Default::default()
            }),
            Err(QdrantConfigError::UnknownTurboBits(v)) if v == 3.0
        ));
    }

    /// `none` is a no-op at creation (nothing to turn off yet — same effect
    /// as omitting `quantization:`), but on an update it's the only way to
    /// explicitly clear quantization off a collection that already has it.
    #[test]
    fn none_quantization_is_a_create_noop_but_clears_on_update() {
        let q = QuantizationConfig { kind: Some("none".into()), ..Default::default() };
        assert_eq!(quantization_for_create(&q).unwrap(), None);
        assert!(matches!(
            quantization_for_update(&q),
            Ok(quantization_config_diff::Quantization::Disabled(_))
        ));
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
            quantization_for_create(&QuantizationConfig {
                kind: Some("fp4".into()),
                ..Default::default()
            }),
            Err(QdrantConfigError::UnknownQuantizationType(_))
        ));
        assert!(matches!(
            quantization_for_create(&QuantizationConfig {
                kind: Some("product".into()),
                compression: Some("x2".into()),
                ..Default::default()
            }),
            Err(QdrantConfigError::UnknownCompressionRatio(_))
        ));
        assert!(matches!(
            quantization_for_create(&QuantizationConfig {
                kind: Some("binary".into()),
                encoding: Some("three_bits".into()),
                ..Default::default()
            }),
            Err(QdrantConfigError::UnknownQuantizationEncoding(_))
        ));
    }
}
