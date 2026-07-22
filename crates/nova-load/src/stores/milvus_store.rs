//! Milvus load backend (feature `milvus`, crate `milvus-sdk-rust`).
//!
//! Milvus inserts are **columnar** (`FieldColumn` per field), so `upsert_batch`
//! transposes the row-shaped [`Point`]s into one id column + one column per dense
//! vector. The primary key is a **varchar** (every id is stringified, so UUID
//! point ids work), and vectors are float-vector fields.
//!
//! Scope for now: **dense vectors, id only**. This SDK exposes no JSON/dynamic
//! field, so **payload is not persisted** — it's dropped with a warning. Sparse
//! and multivector values error out.
//!
//! Index lifecycle (bulk-load shaped): `ensure_collection` creates the collection
//! with **no index** so the upsert pays no per-insert index cost; `enable_indexing`
//! then flushes and builds the configured index; `wait_for_indexing` polls until
//! the build is steady, then explicitly `load`s the collection into memory so it's
//! queryable. The load is timed and reported SEPARATELY from the index build (it's
//! not part of `index_seconds`).
//!
//! Two ops go over Milvus's **REST API** (`/v2/vectordb/...`, same host:port as
//! gRPC in 2.4+) rather than the gRPC SDK, because the vendored `milvus-sdk-rust`
//! 0.1.0 is too old: (1) index create/describe/drop — the SDK's `MetricType` enum
//! has no `COSINE` (and its `From<IndexDescription>` would *panic* on one), and
//! can't express arbitrary index types; (2) build progress — the SDK's proto drops
//! `pending_index_rows`, the true "indexing finished" signal. Everything else
//! (collection create/insert/flush) uses the gRPC SDK.
//!
//! Milvus inserts are **columnar** (`FieldColumn` per field), so `upsert_batch`
//! transposes the row-shaped [`Point`]s into one id column + one column per dense
//! vector. The primary key is a **varchar** (every id is stringified, so UUID
//! point ids work), and vectors are float-vector fields.
//!
//! Worker note: distributed `load` workers call `upsert_batch` without ever
//! calling `ensure_collection` (the controller does that in `prepare`). So the
//! insert path builds its `FieldSchema`s from the data itself (dim inferred from
//! the vectors) rather than a schema stashed at create time.

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use milvus::client::{Client, ClientBuilder};
use milvus::data::FieldColumn;
use milvus::schema::{CollectionSchemaBuilder, FieldSchema};
use serde::Deserialize;
use serde_json::json;

use crate::config::VectorKind;
use crate::stores::{CollectionSchema, Point, PointId, StoreError, VectorStore, VectorValue};

const PK_FIELD: &str = "id";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MilvusConfig {
    /// e.g. `http://localhost:19530`.
    pub url: String,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub password: Option<String>,
    #[serde(default = "default_collection")]
    pub collection_name: String,
    /// Max length for the varchar primary key (UUIDs need 36).
    #[serde(default = "default_id_len")]
    pub id_max_length: i32,
    /// Vector index type to build in `enable_indexing`, passed through to Milvus
    /// verbatim — any type this Milvus build supports works (e.g. `IVF_FLAT`,
    /// `IVF_SQ8`, `HNSW`, `DISKANN`, `FLAT`, `AUTOINDEX`). Defaults to
    /// `AUTOINDEX` (Milvus picks a sensible index).
    #[serde(default = "default_index_type")]
    pub index_type: String,
    /// Extra index-build params, passed through verbatim (e.g. `{nlist: 128}` for
    /// IVF, `{M: 16, efConstruction: 200}` for HNSW). Ignored for `AUTOINDEX`
    /// (Milvus rejects extra params there). The metric is NOT set here — it comes
    /// from the top-level `vectors:` distance, like every other backend.
    #[serde(default)]
    pub index_params: HashMap<String, serde_json::Value>,
    /// Drop + recreate the collection if it already exists.
    #[serde(default)]
    pub recreate: bool,
}

fn default_collection() -> String {
    "default".to_string()
}
fn default_id_len() -> i32 {
    128
}
fn default_index_type() -> String {
    "AUTOINDEX".to_string()
}

pub struct MilvusStore {
    client: Client,
    collection_name: String,
    id_max_length: i32,
    recreate: bool,
    /// Base URL for Milvus's REST API (same host:port as gRPC in 2.4+). Used
    /// only by `wait_for_indexing` to read `pending_index_rows`, which the gRPC
    /// SDK's bundled proto is too old to expose. See `wait_for_indexing`.
    rest_base: String,
    /// REST bearer token (`username:password`), if credentials were configured.
    auth_token: Option<String>,
    http: reqwest::Client,
    /// Vector index type + extra params to build (from config), passed through to
    /// Milvus's REST index-create so any supported index works.
    index_type: String,
    index_params: HashMap<String, serde_json::Value>,
    /// What `enable_indexing` asked Milvus to build, recorded so the post-build
    /// sanity check in `wait_for_indexing` can confirm it stuck. `enable_indexing`
    /// and `wait_for_indexing` always run on the same store instance (both via
    /// `finish_indexing`), so this hands off between them without touching the
    /// trait signature.
    expected_indexes: Mutex<Vec<ExpectedIndex>>,
}

/// The index config `enable_indexing` requested for one vector field — checked
/// against the built index in `wait_for_indexing`.
#[derive(Clone)]
struct ExpectedIndex {
    field_name: String,
    index_name: String,
    index_type: String,
    metric_type: String,
    /// Build params we asked for (e.g. `{M: 16, efConstruction: 200}` / `{nlist: 128}`).
    /// Verified against the gRPC-reported params (REST describe omits them).
    params: HashMap<String, serde_json::Value>,
}

/// A JSON scalar as a plain string, so a requested `16` (number) and a
/// Milvus-reported `"16"` (string) compare equal.
fn scalar_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

impl MilvusConfig {
    pub async fn connect(self) -> Result<MilvusStore, StoreError> {
        // The gRPC and REST APIs share the same endpoint; keep the URL for REST
        // before it's moved into the gRPC client builder below.
        let rest_base = self.url.trim_end_matches('/').to_string();
        let auth_token = match (&self.username, &self.password) {
            (Some(u), Some(p)) => Some(format!("{u}:{p}")),
            _ => None,
        };

        // Owned String: tonic's Endpoint is TryFrom<String>, not From<&String>.
        let mut builder = ClientBuilder::new(self.url);
        if let Some(u) = &self.username {
            builder = builder.username(u);
        }
        if let Some(p) = &self.password {
            builder = builder.password(p);
        }
        let client = builder.build().await.map_err(to_other)?;
        // `Client::builder().build()` surfaces a TLS-stack init failure as an
        // error rather than panicking like `Client::new()`.
        let http = reqwest::Client::builder().build().map_err(to_other)?;
        Ok(MilvusStore {
            client,
            collection_name: self.collection_name,
            id_max_length: self.id_max_length,
            recreate: self.recreate,
            rest_base,
            auth_token,
            http,
            index_type: self.index_type,
            index_params: self.index_params,
            expected_indexes: Mutex::new(Vec::new()),
        })
    }
}

/// One index's build progress, as returned by the REST
/// `/v2/vectordb/indexes/describe` endpoint. Field names match the JSON.
#[derive(Debug, Deserialize)]
struct IndexProgress {
    #[serde(rename = "indexName", default)]
    index_name: String,
    #[serde(rename = "fieldName", default)]
    field_name: String,
    #[serde(rename = "indexType", default)]
    index_type: String,
    #[serde(rename = "metricType", default)]
    metric_type: String,
    #[serde(rename = "indexState", default)]
    index_state: String,
    #[serde(rename = "indexedRows", default)]
    indexed_rows: i64,
    #[serde(rename = "totalRows", default)]
    total_rows: i64,
    /// Rows queued for indexing but not yet built. This can be > 0 even when
    /// `indexed_rows == total_rows` (Milvus reports "done", then keeps going),
    /// so it — not the indexed/total pair — is the true "indexing finished"
    /// signal. Absent from the gRPC SDK's old proto; the reason we poll REST.
    #[serde(rename = "pendingRows", default)]
    pending_rows: i64,
    #[serde(rename = "failReason", default)]
    fail_reason: String,
}

fn to_other<E: std::fmt::Display>(e: E) -> StoreError {
    StoreError::Other(e.to_string())
}

/// Qdrant-style distance name → Milvus REST metric type. Unlike the gRPC SDK's
/// closed `MetricType` enum (which has no COSINE), the REST index-create accepts
/// the real metric names Milvus supports — so cosine is genuine cosine, not an
/// inner-product stand-in.
fn rest_metric(distance: Option<&str>) -> &'static str {
    match distance {
        None | Some("cosine") => "COSINE",
        Some("euclid") => "L2",
        Some("dot") => "IP",
        Some(other) => {
            tracing::warn!("milvus: unknown distance `{other}`, defaulting to COSINE metric");
            "COSINE"
        }
    }
}

impl fmt::Display for MilvusStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "milvus({})", self.collection_name)
    }
}

impl MilvusStore {
    /// POST a Milvus REST v2 request and return its `data` payload, erroring on a
    /// non-zero response code. Only used by `wait_for_indexing` (see the note
    /// there on why index-build progress goes over REST rather than gRPC).
    async fn rest_post(&self, path: &str, body: serde_json::Value) -> Result<serde_json::Value, StoreError> {
        let url = format!("{}/v2/vectordb/{path}", self.rest_base);
        let mut req = self.http.post(&url).json(&body);
        if let Some(token) = &self.auth_token {
            req = req.bearer_auth(token);
        }
        let resp = req.send().await.map_err(to_other)?;
        // Check the HTTP status before parsing JSON: a proxy 502, an auth 401
        // returning HTML, or hitting the gRPC port by mistake would otherwise
        // surface as an opaque serde parse error instead of the real status.
        let status = resp.status();
        if !status.is_success() {
            let detail = resp.text().await.unwrap_or_default();
            return Err(StoreError::Other(format!("milvus REST {path} HTTP {status}: {detail}")));
        }
        let body: serde_json::Value = resp.json().await.map_err(to_other)?;
        if body.get("code").and_then(serde_json::Value::as_i64) != Some(0) {
            return Err(StoreError::Other(format!("milvus REST {path} failed: {body}")));
        }
        Ok(body.get("data").cloned().unwrap_or(serde_json::Value::Null))
    }

    /// Names of the indexes on this collection (usually one per vector field).
    async fn rest_index_names(&self) -> Result<Vec<String>, StoreError> {
        let data = self
            .rest_post("indexes/list", json!({ "collectionName": self.collection_name }))
            .await?;
        serde_json::from_value(data).map_err(to_other)
    }

    /// Describe one index (returns a row per segment/field the endpoint reports).
    async fn rest_describe(&self, index_name: &str) -> Result<Vec<IndexProgress>, StoreError> {
        let data = self
            .rest_post(
                "indexes/describe",
                json!({ "collectionName": self.collection_name, "indexName": index_name }),
            )
            .await?;
        serde_json::from_value(data).map_err(to_other)
    }

    /// Create one vector index over `field` via the REST API, applying the
    /// configured index type + params and the given metric. Uses REST (not the
    /// gRPC SDK) so we can request any index type and — critically — real
    /// `COSINE`, which the SDK's closed enum can't express.
    async fn rest_create_index(
        &self,
        field: &str,
        index_name: &str,
        metric: &str,
    ) -> Result<(), StoreError> {
        let mut index_param = serde_json::Map::new();
        index_param.insert("fieldName".into(), json!(field));
        index_param.insert("indexName".into(), json!(index_name));
        index_param.insert("metricType".into(), json!(metric));
        // Milvus rejects extra params for AUTOINDEX ("only metric type can be
        // passed"); for every other type the index type + params go INSIDE the
        // `params` object (a top-level `indexType` is silently treated as
        // AUTOINDEX).
        if self.index_type.eq_ignore_ascii_case("AUTOINDEX") {
            if !self.index_params.is_empty() {
                tracing::warn!("{self}: index_params are ignored for AUTOINDEX");
            }
        } else {
            let mut params = serde_json::Map::new();
            params.insert("index_type".into(), json!(self.index_type));
            for (k, v) in &self.index_params {
                params.insert(k.clone(), v.clone());
            }
            index_param.insert("params".into(), serde_json::Value::Object(params));
        }

        self.rest_post(
            "indexes/create",
            json!({
                "collectionName": self.collection_name,
                "indexParams": [serde_json::Value::Object(index_param)],
            }),
        )
        .await?;
        Ok(())
    }

    /// Sum index-build progress across the named indexes, returning
    /// `(indexed_rows, total_rows, pending_index_rows, all_finished)`. Errors if
    /// any index is in the `Failed` state (surfacing its fail reason).
    async fn rest_index_progress(
        &self,
        names: &[String],
    ) -> Result<(i64, i64, i64, bool), StoreError> {
        let (mut indexed, mut total, mut pending) = (0i64, 0i64, 0i64);
        let mut all_finished = true;
        for name in names {
            for p in &self.rest_describe(name).await? {
                if p.index_state == "Failed" {
                    return Err(StoreError::Other(format!(
                        "milvus index `{}` build failed on {self}: {}",
                        p.index_name, p.fail_reason
                    )));
                }
                if p.index_state != "Finished" {
                    all_finished = false;
                }
                indexed += p.indexed_rows;
                total += p.total_rows;
                pending += p.pending_rows;
            }
        }
        Ok((indexed, total, pending, all_finished))
    }

    /// The index build params to verify: those we actually sent. AUTOINDEX takes
    /// no params (Milvus rejects them), so there's nothing to check there.
    fn params_to_verify(&self) -> HashMap<String, serde_json::Value> {
        if self.index_type.eq_ignore_ascii_case("AUTOINDEX") {
            HashMap::new()
        } else {
            self.index_params.clone()
        }
    }

    /// Read an index's build params over raw gRPC `DescribeIndex`. Milvus's REST
    /// describe omits build params (M/efConstruction/nlist); the gRPC response
    /// carries them in its `params` key-value list. We read those raw — no
    /// conversion to the SDK's `IndexInfo`, whose `MetricType` parse would panic
    /// on COSINE. Returns a flat string map of {index_type, metric_type, + the
    /// nested build params}. Uses a fresh unauthenticated connection, so callers
    /// treat an error as "couldn't verify" rather than fatal.
    async fn grpc_index_params(
        &self,
        field_name: &str,
        index_name: &str,
    ) -> Result<HashMap<String, String>, StoreError> {
        use milvus::proto::common::{MsgBase, MsgType};
        use milvus::proto::milvus::DescribeIndexRequest;
        use milvus::proto::milvus::milvus_service_client::MilvusServiceClient;

        let mut client =
            MilvusServiceClient::connect(self.rest_base.clone()).await.map_err(to_other)?;
        let req = DescribeIndexRequest {
            base: Some(MsgBase {
                msg_type: MsgType::DescribeIndex as i32,
                msg_id: 0,
                timestamp: 0,
                source_id: 0,
                target_id: 0,
            }),
            db_name: String::new(),
            collection_name: self.collection_name.clone(),
            field_name: field_name.to_string(),
            index_name: index_name.to_string(),
        };
        let resp = client.describe_index(req).await.map_err(to_other)?.into_inner();
        let desc = resp
            .index_descriptions
            .into_iter()
            .find(|d| d.index_name == index_name || d.field_name == field_name)
            .ok_or_else(|| {
                StoreError::Other(format!("gRPC describe returned no index `{index_name}`"))
            })?;

        let mut out = HashMap::new();
        for kv in desc.params {
            if kv.key == "params" {
                // Nested JSON, e.g. {"M":"16","efConstruction":"200"} or {"nlist":128}.
                if let Ok(serde_json::Value::Object(m)) =
                    serde_json::from_str::<serde_json::Value>(&kv.value)
                {
                    for (k, v) in m {
                        out.insert(k, scalar_string(&v));
                    }
                }
            } else {
                out.insert(kv.key, kv.value);
            }
        }
        Ok(out)
    }

    /// Load the collection into memory so it's queryable, timing the load and
    /// reporting it on its own line (kept out of the index-build timing). Milvus's
    /// SDK `load` blocks until the load reaches 100%.
    async fn load_collection(&self) -> Result<(), StoreError> {
        let started = Instant::now();
        let collection =
            self.client.get_collection(self.collection_name.as_str()).await.map_err(to_other)?;
        collection.load(1).await.map_err(to_other)?;
        tracing::info!(
            "{self} load finished: load_seconds={:.3}",
            started.elapsed().as_secs_f64()
        );
        Ok(())
    }

    /// Sanity check: confirm each index `enable_indexing` asked Milvus to build
    /// actually came back with the index type + metric we requested. Runs after
    /// convergence in `wait_for_indexing`; NOT part of the timed build (the caller
    /// measures up to the converged instant, which predates this). Comparisons are
    /// case-insensitive (Milvus normalizes the names it echoes back).
    async fn verify_indexes(&self) -> Result<(), StoreError> {
        let expected = self.expected_indexes.lock().expect("index lock").clone();
        for e in &expected {
            let described = self.rest_describe(&e.index_name).await?;
            let got = described.iter().find(|p| p.index_name == e.index_name).ok_or_else(|| {
                StoreError::Other(format!(
                    "sanity check: index `{}` not found after build on {self}",
                    e.index_name
                ))
            })?;
            if !got.index_type.eq_ignore_ascii_case(&e.index_type)
                || !got.metric_type.eq_ignore_ascii_case(&e.metric_type)
            {
                return Err(StoreError::Other(format!(
                    "sanity check FAILED on {self} index `{}`: requested \
                     index_type={}/metric_type={} but Milvus built \
                     index_type={}/metric_type={}",
                    e.index_name, e.index_type, e.metric_type, got.index_type, got.metric_type
                )));
            }
            tracing::info!(
                "{self} index `{}` verified: index_type={} metric_type={}",
                e.index_name,
                got.index_type,
                got.metric_type
            );

            // Build params (M/efConstruction/nlist) aren't in the REST describe —
            // read them over gRPC and confirm each one we requested is reflected.
            // Best-effort: if gRPC is unreachable (e.g. auth), warn rather than
            // fail, since type+metric are already verified above.
            if !e.params.is_empty() {
                match self.grpc_index_params(&e.field_name, &e.index_name).await {
                    Ok(built) => {
                        for (k, v) in &e.params {
                            let want = scalar_string(v);
                            match built.get(k) {
                                Some(got) if *got == want => {}
                                other => {
                                    return Err(StoreError::Other(format!(
                                        "sanity check FAILED on {self} index `{}`: requested \
                                         {k}={want} but built index reports {k}={other:?}",
                                        e.index_name
                                    )));
                                }
                            }
                        }
                        tracing::info!(
                            "{self} index `{}` params verified: {:?}",
                            e.index_name,
                            e.params
                        );
                    }
                    Err(err) => tracing::warn!(
                        "{self} index `{}`: could not read build params over gRPC to verify \
                         ({err}); verified index_type + metric_type only",
                        e.index_name
                    ),
                }
            }
        }
        Ok(())
    }
}

/// Validate that every vector is dense (Milvus backend scope), erroring on
/// sparse/multivector. Returns the dense vector names.
fn dense_field_names(schema: &CollectionSchema) -> Result<Vec<&String>, StoreError> {
    schema
        .vectors
        .iter()
        .map(|(name, spec)| match spec.kind {
            VectorKind::Dense => Ok(name),
            VectorKind::Sparse | VectorKind::Multivector => Err(StoreError::Other(format!(
                "milvus backend supports dense vectors only for now; vector `{name}` is {:?}",
                spec.kind
            ))),
        })
        .collect()
}

/// Dense vector fields with their dims — for collection creation, which needs
/// the dim. Errors on sparse/multivector or unresolved dims.
fn dense_fields(schema: &CollectionSchema) -> Result<Vec<(&String, i64)>, StoreError> {
    let mut out = Vec::new();
    for name in dense_field_names(schema)? {
        let dim = *schema.dims.get(name).ok_or_else(|| {
            StoreError::Other(format!("vector `{name}`: dims not resolved"))
        })?;
        out.push((name, dim as i64));
    }
    Ok(out)
}

#[async_trait]
impl VectorStore for MilvusStore {
    async fn ensure_collection(&self, schema: &CollectionSchema) -> Result<(), StoreError> {
        let name = self.collection_name.as_str();
        let dense = dense_fields(schema)?;

        if self.client.has_collection(name).await.map_err(to_other)? {
            if !self.recreate {
                return Ok(());
            }
            self.client.drop_collection(name).await.map_err(to_other)?;
        }

        tracing::warn!(
            "milvus backend does not persist payload (the SDK has no JSON field); \
             only the id + dense vectors are stored"
        );

        // Create the collection WITHOUT any index: leaving the vector fields
        // unindexed lets the bulk upsert run without paying index-build cost per
        // insert. The real index is created later in `enable_indexing`, once the
        // data is in (see there). `defer_indexing` is a no-op for the same reason
        // — there's no index to pause during the load.
        let mut builder = CollectionSchemaBuilder::new(name, "nova-load");
        builder.add_field(FieldSchema::new_primary_varchar(
            PK_FIELD,
            "point id",
            false,
            self.id_max_length,
        ));
        for (vname, dim) in &dense {
            builder.add_field(FieldSchema::new_float_vector(vname.as_str(), "", *dim));
        }
        let milvus_schema = builder.build().map_err(to_other)?;
        self.client.create_collection(milvus_schema, None).await.map_err(to_other)?;
        Ok(())
    }

    async fn upsert_batch(&self, points: Vec<Point>) -> Result<(), StoreError> {
        if points.is_empty() {
            return Ok(());
        }
        // Transpose rows → columns. Infer each vector's dim from the first point
        // (workers don't see the CollectionSchema — see the module note).
        let mut ids: Vec<String> = Vec::with_capacity(points.len());
        let mut vecs: HashMap<String, (i64, Vec<f32>)> = HashMap::new();

        for point in points {
            ids.push(match point.id {
                PointId::Integer(n) => n.to_string(),
                PointId::String(s) => s,
            });
            for (vname, value) in point.vectors {
                match value {
                    VectorValue::Dense(d) => {
                        let entry =
                            vecs.entry(vname).or_insert_with(|| (d.len() as i64, Vec::new()));
                        entry.1.extend(d);
                    }
                    VectorValue::Sparse { .. } | VectorValue::Multi(_) => {
                        return Err(StoreError::Other(format!(
                            "milvus backend supports dense vectors only; vector `{vname}` is not dense"
                        )));
                    }
                }
            }
        }

        let pk = FieldSchema::new_primary_varchar(PK_FIELD, "", false, self.id_max_length);
        let mut columns = vec![FieldColumn::new(&pk, ids)];
        for (vname, (dim, flat)) in vecs {
            let fs = FieldSchema::new_float_vector(&vname, "", dim);
            columns.push(FieldColumn::new(&fs, flat));
        }

        let collection =
            self.client.get_collection(self.collection_name.as_str()).await.map_err(to_other)?;
        collection.insert(columns, None).await.map_err(to_other)?;
        Ok(())
    }

    async fn defer_indexing(&self) -> Result<(), StoreError> {
        // No-op: `ensure_collection` creates the collection with NO index, so
        // there's nothing to pause during the bulk upsert. The index is built
        // afterward in `enable_indexing`.
        Ok(())
    }

    async fn enable_indexing(&self, schema: &CollectionSchema) -> Result<(), StoreError> {
        // The collection was created unindexed (see `ensure_collection`). Now that
        // the bulk upsert is done, persist the buffered inserts and build the
        // real index (via REST — see `rest_create_index`). We don't load here —
        // the load happens once the build converges, in `wait_for_indexing`, timed
        // separately. Only field names are needed (not dims — the index create is
        // by field), so this doesn't force the caller to resolve dims in the
        // `finalize` path.
        let dense = dense_field_names(schema)?;

        let collection =
            self.client.get_collection(self.collection_name.as_str()).await.map_err(to_other)?;
        collection.flush().await.map_err(to_other)?;

        // Skip fields that already carry an index (a re-run over an existing
        // collection); we still record every field as "expected" so the sanity
        // check verifies pre-existing indexes match the requested config too.
        let existing = self.rest_index_names().await?;
        let mut expected = Vec::with_capacity(dense.len());
        for &vname in &dense {
            let index_name = format!("{vname}_idx");
            let metric = rest_metric(schema.vectors[vname.as_str()].distance.as_deref());
            if !existing.iter().any(|n| n == &index_name) {
                self.rest_create_index(vname, &index_name, metric).await?;
            }
            expected.push(ExpectedIndex {
                field_name: vname.to_string(),
                index_name,
                index_type: self.index_type.clone(),
                metric_type: metric.to_string(),
                params: self.params_to_verify(),
            });
        }
        *self.expected_indexes.lock().expect("index lock") = expected;
        Ok(())
    }

    async fn wait_for_indexing(&self) -> Result<Instant, StoreError> {
        // Adapted from VECHINI's Milvus loader (milvus/scripts/index.py). Milvus
        // builds segment indexes asynchronously, and — the important part — it
        // will report an index "Finished" with `indexed_rows == total_rows`, then
        // decide to index more (you can observe indexed=total AND pending > 0 at
        // the same time). So the honest "indexing is really done" signal is
        // `pending_index_rows == 0`, and it has to *hold*: VECHINI waits for it to
        // stay 0 over a window. We do the same, with a 30s hold.
        //
        // `pending_index_rows` isn't in the gRPC SDK's bundled proto (its
        // `IndexDescription` stops at field 8), so — exactly the field pymilvus
        // reads via DescribeIndex — we read it from Milvus's REST API instead
        // (same endpoint, `/v2/vectordb/indexes/describe`). We log the counts each
        // poll so a jump back up is visible.
        //
        // Completion = every index `Finished` AND `pending_index_rows == 0`,
        // HELD for the window. We deliberately do NOT gate on row counts:
        //  - `get_collection_stats` rowCount lags (it read 0 for a fully-loaded
        //    collection all through the build), so it's not a usable target;
        //  - right after `create_index` Milvus briefly reports pending=0/Finished
        //    (stale) — but the hold + regression-reset below absorbs that: when
        //    the build actually starts, pending jumps > 0 and restarts the window.
        // This also makes an *empty* collection converge (pending stays 0) instead
        // of spinning to the timeout.
        const POLL_INTERVAL: Duration = Duration::from_secs(2);
        const PENDING_ZERO_HOLD: Duration = Duration::from_secs(30);
        const TIMEOUT: Duration = Duration::from_secs(3600);

        // Which indexes to watch (usually one per vector field). Works in the
        // distributed `finalize` process too, which never saw the schema.
        let names = self.rest_index_names().await?;
        if names.is_empty() {
            tracing::warn!("no indexes found on {self}; nothing to wait for");
            return Ok(Instant::now());
        }

        let start = Instant::now();
        let mut pending_zero_since: Option<Instant> = None;
        loop {
            if start.elapsed() >= TIMEOUT {
                return Err(StoreError::Other(format!(
                    "timed out after {}s waiting for milvus index build on {self}",
                    TIMEOUT.as_secs()
                )));
            }

            let (indexed, total, pending, all_finished) = self.rest_index_progress(&names).await?;

            // pending_index_rows == 0, every index Finished, held over the window.
            let converged = all_finished && pending == 0;
            tracing::info!(
                "{self} index build: indexed_rows={indexed} total_rows={total} \
                 pending_index_rows={pending} finished={all_finished}"
            );

            if converged {
                let since = *pending_zero_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= PENDING_ZERO_HOLD {
                    tracing::info!(
                        "{self} index build converged (indexed_rows={indexed}, \
                         pending_index_rows held 0 for {}s)",
                        PENDING_ZERO_HOLD.as_secs()
                    );
                    // Sanity-check the built index against what we requested. This
                    // runs AFTER `since` (the converged instant we return), so the
                    // caller's build-time measurement excludes it.
                    self.verify_indexes().await?;
                    // Explicitly load the collection into memory so it's queryable.
                    // Timed and reported SEPARATELY from the index build — the
                    // returned `since` predates this, so `index_seconds`
                    // excludes the load. Runs for both the enable-index and the
                    // reindex flows (both end here).
                    self.load_collection().await?;
                    return Ok(since);
                }
            } else if pending_zero_since.take().is_some() {
                // We thought it was done; Milvus went back to work. Restart the
                // hold window — this is the "jumps back up" case.
                tracing::warn!(
                    "{self} index build regressed (indexed_rows={indexed}, \
                     pending_index_rows={pending}, finished={all_finished}); \
                     restarting {}s stability window",
                    PENDING_ZERO_HOLD.as_secs()
                );
            }

            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    async fn reindex(&self, schema: &CollectionSchema) -> Result<(), StoreError> {
        // Milvus has no in-place index patch, so "reindex" = drop each vector
        // index and rebuild it with the configured type/params/metric. The metric
        // comes from config (the field's `vectors:` distance) when that field is
        // configured; otherwise the field's existing metric is preserved. The
        // subsequent `wait_for_indexing` waits for the rebuild + sanity-checks it.
        let names = self.rest_index_names().await?;
        if names.is_empty() {
            tracing::warn!("{self}: no indexes to reindex");
            return Ok(());
        }

        // Known Milvus issue: dropping and rebuilding an index can occasionally
        // regress query performance. Observed up to Milvus 2.6.6; may be fixed in
        // later builds — not something we can control, but worth flagging.
        tracing::warn!(
            "{self}: reindex drops and rebuilds each vector index. NOTE — Milvus has \
             a known issue where drop+rebuild can occasionally hurt query \
             performance (observed up to 2.6.6, possibly fixed since). See \
             https://github.com/milvus-io/milvus/discussions/47149"
        );

        // An index can't be dropped while the collection is loaded (a prior load —
        // e.g. from `enable_indexing`/`wait_for_indexing` — leaves it in memory),
        // so release it first. `wait_for_indexing` reloads it after the rebuild.
        // Best-effort: releasing an unloaded collection is a harmless no-op.
        if let Err(e) =
            self.rest_post("collections/release", json!({ "collectionName": self.collection_name }))
                .await
        {
            tracing::debug!("{self}: release before reindex was a no-op or failed: {e}");
        }

        let mut expected = Vec::with_capacity(names.len());
        for index_name in &names {
            let info = self
                .rest_describe(index_name)
                .await?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    StoreError::Other(format!("reindex: index `{index_name}` vanished on {self}"))
                })?;
            let field = info.field_name;
            let old_type = info.index_type;
            // New metric from config if the field is a configured vector; else keep
            // the existing one (reindex is usually about the index type, not the
            // distance, but honor a changed `vectors:` distance when present).
            let metric = match schema.vectors.get(&field) {
                Some(spec) => rest_metric(spec.distance.as_deref()).to_string(),
                None => info.metric_type,
            };

            tracing::info!(
                "{self}: reindexing `{index_name}` on field `{field}`: \
                 index_type {old_type} → {}, metric_type → {metric}",
                self.index_type
            );
            self.rest_post(
                "indexes/drop",
                json!({ "collectionName": self.collection_name, "indexName": index_name }),
            )
            .await?;
            self.rest_create_index(&field, index_name, &metric).await?;

            expected.push(ExpectedIndex {
                field_name: field,
                index_name: index_name.clone(),
                index_type: self.index_type.clone(),
                metric_type: metric,
                params: self.params_to_verify(),
            });
        }
        *self.expected_indexes.lock().expect("index lock") = expected;
        Ok(())
    }

    async fn close(&self) -> Result<(), StoreError> {
        Ok(())
    }

    async fn delete_collection(&self) -> Result<(), StoreError> {
        let name = self.collection_name.as_str();
        if self.client.has_collection(name).await.map_err(to_other)? {
            self.client.drop_collection(name).await.map_err(to_other)?;
        }
        Ok(())
    }
}
