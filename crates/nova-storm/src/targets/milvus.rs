//! Milvus implementation of [`QueryTarget`] (feature `milvus`).
//!
//! Dual search transport, chosen from the collection's metric at setup:
//!   - **L2 / IP** → the gRPC SDK's `Collection::search` (native, typed).
//!   - **COSINE** → Milvus's **REST** `/v2/vectordb/entities/search`, because the
//!     Rust SDK's `MetricType` enum has no `COSINE` variant (it can't even
//!     construct the argument, and parsing "COSINE" panics). We log a warning
//!     when this path is taken — REST may be slower than the native SDK call.
//!
//! A batch of N query vectors is one round-trip on both paths. Milvus requires
//! the collection loaded before search, so we load it at setup. Payload is not
//! persisted by the milvus load backend, so `with_payload` is a no-op here.
//! Filters are not supported yet (rejected at construction). Recall uses the
//! returned varchar/int ids.

use std::borrow::Cow;
use std::fmt;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use milvus::client::ClientBuilder;
use milvus::collection::{Collection, SearchOption};
use milvus::index::MetricType;
use milvus::value::Value as MValue;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{BatchOutcome, QueryTarget};
use crate::config::QueryConfig;
use crate::errors::TargetError;
use crate::queries::QueryVector;

/// Connection + target settings for a Milvus backend (`type: milvus`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MilvusConfig {
    /// e.g. `http://localhost:19530` (gRPC + REST share the endpoint in 2.4+).
    pub url: String,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub password: Option<String>,
    #[serde(default = "default_collection")]
    pub collection_name: String,
}

fn default_collection() -> String {
    "default".to_string()
}

/// Milvus search-time tuning (`query.search_params` for a `milvus` target).
/// `deny_unknown_fields` rejects a non-Milvus key (e.g. a Qdrant/ES param).
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct MilvusSearchParams {
    /// HNSW search breadth (Qdrant `hnsw_ef` analog).
    #[serde(default)]
    pub ef: Option<u64>,
    /// IVF probe count.
    #[serde(default)]
    pub nprobe: Option<u64>,
}

/// Which search transport this target uses (decided from the collection metric).
enum Transport {
    /// L2 / IP via the gRPC SDK, carrying a live loaded collection handle.
    /// Boxed: `Collection` is large relative to the `Rest` variant.
    Sdk { collection: Box<Collection>, metric: MetricType },
    /// COSINE via REST (the SDK enum can't express cosine).
    Rest,
}

pub struct MilvusTarget {
    transport: Transport,
    // REST client bits — used by the REST search path and by setup
    // (metric detect + collection load).
    http: reqwest::Client,
    rest_base: String,
    auth_token: Option<String>,
    collection_name: String,
    vector_field: String,
    top_k: i32,
    ef: Option<u64>,
    nprobe: Option<u64>,
    collect_ids: bool,
}

fn to_other<E: std::fmt::Display>(e: E) -> TargetError {
    TargetError::Other(e.to_string())
}

/// POST a Milvus REST v2 request and return its `data` payload, checking the
/// HTTP status before parsing JSON and then the response `code`.
async fn rest_post(
    http: &reqwest::Client,
    base: &str,
    token: &Option<String>,
    path: &str,
    body: Value,
) -> Result<Value, TargetError> {
    let url = format!("{base}/v2/vectordb/{path}");
    let mut req = http.post(&url).json(&body);
    if let Some(token) = token {
        req = req.bearer_auth(token);
    }
    let resp = req.send().await.map_err(to_other)?;
    let status = resp.status();
    if !status.is_success() {
        let detail = resp.text().await.unwrap_or_default();
        return Err(TargetError::Other(format!("milvus REST {path} HTTP {status}: {detail}")));
    }
    let body: Value = resp.json().await.map_err(to_other)?;
    if body.get("code").and_then(Value::as_i64) != Some(0) {
        return Err(TargetError::Other(format!("milvus REST {path} failed: {body}")));
    }
    Ok(body.get("data").cloned().unwrap_or(Value::Null))
}

/// The metric (`COSINE`/`L2`/`IP`) of the index on `field`, read over REST (the
/// SDK's typed describe can't decode COSINE). A collection can carry several
/// indexes (multiple vector fields, or scalar indexes with no metric), so we
/// find the one whose `fieldName` matches the field we'll query rather than
/// blindly taking the first.
async fn detect_metric(
    http: &reqwest::Client,
    base: &str,
    token: &Option<String>,
    collection: &str,
    field: &str,
) -> Result<String, TargetError> {
    let names: Vec<String> = serde_json::from_value(
        rest_post(http, base, token, "indexes/list", json!({ "collectionName": collection })).await?,
    )
    .map_err(to_other)?;
    for index_name in names {
        let described = rest_post(
            http,
            base,
            token,
            "indexes/describe",
            json!({ "collectionName": collection, "indexName": index_name }),
        )
        .await?;
        let descs = described.as_array().ok_or_else(|| {
            TargetError::Other(format!("milvus index describe is not an array: {described}"))
        })?;
        for d in descs {
            if d.get("fieldName").and_then(Value::as_str) == Some(field) {
                return d.get("metricType").and_then(Value::as_str).map(str::to_string).ok_or_else(
                    || TargetError::Other(format!("milvus index for field `{field}` has no metricType")),
                );
            }
        }
    }
    Err(TargetError::Other(format!(
        "milvus collection `{collection}` has no vector index on field `{field}`"
    )))
}

/// Ensure the collection is loaded into memory (search errors otherwise). Issues
/// a load and polls until it reports loaded — idempotent if already loaded.
async fn ensure_loaded(
    http: &reqwest::Client,
    base: &str,
    token: &Option<String>,
    collection: &str,
) -> Result<(), TargetError> {
    const POLL: Duration = Duration::from_secs(1);
    const TIMEOUT: Duration = Duration::from_secs(300);
    rest_post(http, base, token, "collections/load", json!({ "collectionName": collection })).await?;
    let start = Instant::now();
    loop {
        let d = rest_post(
            http,
            base,
            token,
            "collections/describe",
            json!({ "collectionName": collection }),
        )
        .await?;
        // Milvus 2.4 REST reports `LoadStateLoaded`; accept `loaded` too (and
        // case-insensitively) so an API-version change doesn't spin to timeout.
        if d.get("load")
            .and_then(Value::as_str)
            .is_some_and(|s| s.eq_ignore_ascii_case("LoadStateLoaded") || s.eq_ignore_ascii_case("loaded"))
        {
            return Ok(());
        }
        if start.elapsed() >= TIMEOUT {
            return Err(TargetError::Other(format!(
                "milvus collection `{collection}` did not report loaded within {}s",
                TIMEOUT.as_secs()
            )));
        }
        tokio::time::sleep(POLL).await;
    }
}

impl MilvusConfig {
    pub async fn into_target(self, query: &QueryConfig) -> Result<MilvusTarget, TargetError> {
        if query.filter.is_some() {
            return Err(TargetError::Other(
                "filters are not yet supported for the milvus target".to_string(),
            ));
        }
        // Milvus always searches a named vector field (`annsField`); no unnamed
        // default like Qdrant, so `vector_name` (the field) is required.
        let vector_field = query.vector_name.clone().ok_or_else(|| {
            TargetError::Other(
                "milvus target requires `query.vector_name` (the annsField to search)".to_string(),
            )
        })?;
        if query.with_payload {
            tracing::warn!(
                "milvus: `with_payload` has no effect — the milvus load backend doesn't persist \
                 payload, so there's nothing to fetch"
            );
        }

        let sp: MilvusSearchParams = query
            .search_params
            .as_ref()
            .map(|v| serde_yaml::from_value(v.clone()))
            .transpose()
            .map_err(|e| TargetError::Other(format!("milvus search_params: {e}")))?
            .unwrap_or_default();
        // `ef` (HNSW) and `nprobe` (IVF) apply to different index types; both at
        // once is an ambiguous config.
        if sp.ef.is_some() && sp.nprobe.is_some() {
            return Err(TargetError::Other(
                "milvus search_params cannot set both `ef` (HNSW) and `nprobe` (IVF)".to_string(),
            ));
        }
        // Milvus caps top_k at 16384; guard the u64→i32 conversion regardless.
        let top_k = i32::try_from(query.top_k)
            .map_err(|_| TargetError::Other(format!("milvus top_k {} exceeds i32::MAX", query.top_k)))?;

        let rest_base = self.url.trim_end_matches('/').to_string();
        // Basic auth needs both parts; one without the other is a config error
        // (it would otherwise silently disable auth and surface as a 401 later).
        let auth_token = match (&self.username, &self.password) {
            (Some(u), Some(p)) => Some(format!("{u}:{p}")),
            (None, None) => None,
            (Some(_), None) => {
                return Err(TargetError::Other("milvus username set without password".to_string()));
            }
            (None, Some(_)) => {
                return Err(TargetError::Other("milvus password set without username".to_string()));
            }
        };
        // A per-request timeout so a single hung HTTP call can't block past the
        // load-wait bound (which only checks between requests).
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(to_other)?;
        let collection_name = self.collection_name;

        // Load the collection (search requires it), then decide the transport
        // from the metric of the index on the field we'll actually query.
        ensure_loaded(&http, &rest_base, &auth_token, &collection_name).await?;
        let metric =
            detect_metric(&http, &rest_base, &auth_token, &collection_name, &vector_field).await?;

        let transport = match metric.to_uppercase().as_str() {
            "COSINE" => {
                tracing::warn!(
                    "milvus({collection_name}): cosine collection → querying over REST. The Rust \
                     SDK has no native cosine metric, so search may be slower than the SDK (L2/IP) \
                     path."
                );
                Transport::Rest
            }
            "L2" | "IP" => {
                // gRPC client only needed for the SDK path.
                let mut builder = ClientBuilder::new(self.url);
                if let Some(u) = &self.username {
                    builder = builder.username(u);
                }
                if let Some(p) = &self.password {
                    builder = builder.password(p);
                }
                let client = builder.build().await.map_err(to_other)?;
                let collection = client.get_collection(&collection_name).await.map_err(to_other)?;
                let metric = if metric.eq_ignore_ascii_case("L2") {
                    MetricType::L2
                } else {
                    MetricType::IP
                };
                Transport::Sdk { collection: Box::new(collection), metric }
            }
            other => {
                return Err(TargetError::Other(format!(
                    "milvus target: unsupported collection metric `{other}` (expected COSINE, L2, or IP)"
                )));
            }
        };

        Ok(MilvusTarget {
            transport,
            http,
            rest_base,
            auth_token,
            collection_name,
            vector_field,
            top_k,
            ef: sp.ef,
            nprobe: sp.nprobe,
            collect_ids: query.source.ground_truth_column.is_some(),
        })
    }
}

impl fmt::Display for MilvusTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "milvus({})", self.collection_name)
    }
}

/// A whole-batch failure outcome (no per-query ids).
fn fail(started: Instant, n: usize, error: String) -> BatchOutcome {
    BatchOutcome { latency: started.elapsed(), ok: false, ids: vec![None; n], error: Some(error) }
}

/// One SDK-returned id as a plain string (varchar → itself, int → decimal).
fn sdk_id_string(v: &MValue) -> Option<String> {
    match v {
        MValue::String(s) => Some(s.to_string()),
        MValue::Long(i) => Some(i.to_string()),
        _ => None,
    }
}

/// One REST-returned id as a plain string (string or number).
fn rest_id_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

impl MilvusTarget {
    /// Search-param key/values common to both transports (`ef`/`nprobe`).
    fn search_param_pairs(&self) -> Vec<(&'static str, u64)> {
        let mut v = Vec::new();
        if let Some(ef) = self.ef {
            v.push(("ef", ef));
        }
        if let Some(nprobe) = self.nprobe {
            v.push(("nprobe", nprobe));
        }
        v
    }

    async fn search_sdk(
        &self,
        collection: &Collection,
        metric: MetricType,
        queries: &[&QueryVector],
        started: Instant,
    ) -> BatchOutcome {
        // Dense-only transport: the guard lives HERE, next to the use, so a
        // future call site (warm-up probe, retry path) cannot bypass it and
        // panic — a sparse query is a per-dispatch data error, never a crash.
        let Some(data) = queries
            .iter()
            .map(|q| q.vector.as_dense().map(|v| MValue::FloatArray(Cow::Borrowed(v))))
            .collect::<Option<Vec<MValue>>>()
        else {
            return fail(started, queries.len(), "the milvus target does not support sparse queries".to_string());
        };
        let mut option = SearchOption::new();
        for (k, v) in self.search_param_pairs() {
            option.add_param(k, json!(v));
        }
        let out_fields: Vec<String> = Vec::new();
        match collection
            .search(data, self.vector_field.as_str(), self.top_k, metric, out_fields, &option)
            .await
        {
            Ok(results) if results.len() != queries.len() => fail(
                started,
                queries.len(),
                format!("search returned {} results for {} queries", results.len(), queries.len()),
            ),
            Ok(results) => {
                if !self.collect_ids {
                    return BatchOutcome {
                        latency: started.elapsed(),
                        ok: true,
                        ids: vec![None; results.len()],
                        error: None,
                    };
                }
                let mut ids = Vec::with_capacity(results.len());
                for r in &results {
                    // A returned id we can't stringify is an unexpected response
                    // — fail rather than silently drop it and understate recall.
                    let mut q = Vec::with_capacity(r.id.len());
                    for v in &r.id {
                        let Some(s) = sdk_id_string(v) else {
                            return fail(
                                started,
                                queries.len(),
                                "milvus search returned an id that isn't a string or int".to_string(),
                            );
                        };
                        q.push(s);
                    }
                    ids.push(Some(q));
                }
                BatchOutcome { latency: started.elapsed(), ok: true, ids, error: None }
            }
            Err(e) => fail(started, queries.len(), e.to_string()),
        }
    }

    async fn search_rest(&self, queries: &[&QueryVector], started: Instant) -> BatchOutcome {
        // Same in-place dense-only guard as `search_sdk` — see the note there.
        let Some(vectors) = queries
            .iter()
            .map(|q| q.vector.as_dense())
            .collect::<Option<Vec<&[f32]>>>()
        else {
            return fail(started, queries.len(), "the milvus target does not support sparse queries".to_string());
        };
        let mut body = json!({
            "collectionName": self.collection_name,
            "data": vectors,
            "annsField": self.vector_field,
            "limit": self.top_k,
            "outputFields": [],
        });
        let params: serde_json::Map<String, Value> =
            self.search_param_pairs().into_iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
        if !params.is_empty() {
            body["searchParams"] = json!({ "params": Value::Object(params) });
        }

        let data = match rest_post(
            &self.http,
            &self.rest_base,
            &self.auth_token,
            "entities/search",
            body,
        )
        .await
        {
            Ok(d) => d,
            Err(e) => return fail(started, queries.len(), e.to_string()),
        };
        let Some(arr) = data.as_array() else {
            return fail(started, queries.len(), format!("milvus REST search: `data` not an array: {data}"));
        };

        // The REST response is a FLAT list of hits across all query vectors, in
        // submission order. Regroup by `top_k` — which requires every query to
        // have returned exactly `top_k` hits. If not (a query matched fewer than
        // top_k), the flat list can't be split back per-query without
        // misaligning recall, so fail the batch (as the Qdrant target does on a
        // count mismatch).
        let k = self.top_k as usize;
        if arr.len() != queries.len() * k {
            return fail(
                started,
                queries.len(),
                format!(
                    "milvus REST search returned {} hits for {} queries × top_k {k}; can't regroup \
                     (a query returned fewer than top_k)",
                    arr.len(),
                    queries.len()
                ),
            );
        }
        if !self.collect_ids {
            return BatchOutcome {
                latency: started.elapsed(),
                ok: true,
                ids: vec![None; queries.len()],
                error: None,
            };
        }
        let mut ids = Vec::with_capacity(queries.len());
        for i in 0..queries.len() {
            let mut q = Vec::with_capacity(k);
            for h in &arr[i * k..(i + 1) * k] {
                // Milvus REST keys the PK as `id` regardless of the PK field name
                // (verified). A missing/odd id is an unexpected response → fail.
                let Some(s) = rest_id_string(&h["id"]) else {
                    return fail(
                        started,
                        queries.len(),
                        format!("milvus REST hit has a missing/invalid `id`: {h}"),
                    );
                };
                q.push(s);
            }
            ids.push(Some(q));
        }
        BatchOutcome { latency: started.elapsed(), ok: true, ids, error: None }
    }
}

#[async_trait]
impl QueryTarget for MilvusTarget {
    async fn query_batch(&self, queries: &[&QueryVector]) -> BatchOutcome {
        let started = Instant::now();
        if queries.is_empty() {
            return BatchOutcome { latency: started.elapsed(), ok: true, ids: Vec::new(), error: None };
        }
        match &self.transport {
            Transport::Sdk { collection, metric } => {
                self.search_sdk(collection, *metric, queries, started).await
            }
            Transport::Rest => self.search_rest(queries, started).await,
        }
    }
}
