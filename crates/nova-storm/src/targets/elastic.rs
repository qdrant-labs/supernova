//! Elasticsearch implementation of [`QueryTarget`] (feature `elastic`).
//!
//! kNN search via `_search` with a top-level `knn` clause; a batch of N queries
//! is one `_msearch` round-trip (the client has no batched-kNN endpoint). The
//! metric is fixed at index time by the field mapping's `similarity`, so nothing
//! metric-related is set per query — only the search-breadth knob
//! `num_candidates` (ES's analog of Qdrant's `hnsw_ef`). Filters are not
//! supported yet (rejected at construction). Recall uses the returned `_id`s.

use std::fmt;
use std::time::Instant;

use async_trait::async_trait;
use elasticsearch::auth::Credentials;
use elasticsearch::cert::CertificateValidation;
use elasticsearch::http::request::JsonBody;
use elasticsearch::http::transport::{SingleNodeConnectionPool, TransportBuilder};
use elasticsearch::{Elasticsearch, MsearchParts};
use serde::Deserialize;
use serde_json::{Value, json};
use url::Url;

use super::{BatchOutcome, QueryTarget};
use crate::config::QueryConfig;
use crate::errors::TargetError;
use crate::queries::QueryVector;

/// Connection + target settings for an Elasticsearch backend (`type: elastic`).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElasticConfig {
    /// Node URL, e.g. `https://localhost:9200`.
    pub url: String,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub password: Option<String>,
    /// Base64 `id:api_key` (alternative to username/password).
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_index")]
    pub index_name: String,
    /// Skip TLS cert validation (a default ES 8 dev node serves a self-signed
    /// cert). DEV ONLY.
    #[serde(default)]
    pub tls_insecure: bool,
    /// Per-request timeout in seconds (the same generous-default reasoning as
    /// the qdrant target's `timeout_s`: a load test must not count its own
    /// honest slow requests as errors). Unset = 300s.
    #[serde(default = "default_elastic_timeout_s")]
    pub timeout_s: u64,
}

fn default_elastic_timeout_s() -> u64 {
    300
}

fn default_index() -> String {
    "default".to_string()
}

/// Manual `Debug` so secrets never reach logs.
impl fmt::Debug for ElasticConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ElasticConfig")
            .field("url", &self.url)
            .field("username", &self.username)
            .field("password", &self.password.as_ref().map(|_| "<redacted>"))
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("index_name", &self.index_name)
            .field("tls_insecure", &self.tls_insecure)
            .finish()
    }
}

/// ES search-time tuning (`query.search_params` for an `elastic` target).
/// `deny_unknown_fields` rejects a non-ES key (e.g. a Qdrant/Milvus param).
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ElasticSearchParams {
    /// Per-shard HNSW candidate pool — ES's analog of Qdrant's `hnsw_ef`.
    /// Validated `>= k` when set; **omitted when unset** so ES applies its own
    /// default (which is `> k`), rather than pinning it to the lowest-recall `k`.
    #[serde(default)]
    pub num_candidates: Option<u64>,
}

pub struct ElasticTarget {
    client: Elasticsearch,
    index_name: String,
    vector_field: String,
    top_k: u64,
    /// `None` → omit `num_candidates` so ES applies its own default (which is
    /// `> k`); pinning it to `k` would be the lowest-recall setting. When set, we
    /// validate `>= k` at construction.
    num_candidates: Option<u64>,
    with_payload: crate::config::WithPayload,
    collect_ids: bool,
}

/// Flatten an error + its `source()` chain into one message.
fn to_other<E: std::error::Error>(e: E) -> TargetError {
    let mut msg = e.to_string();
    let mut src = e.source();
    while let Some(s) = src {
        msg.push_str(&format!(": {s}"));
        src = s.source();
    }
    TargetError::Other(msg)
}

impl ElasticConfig {
    pub async fn into_target(self, query: &QueryConfig) -> Result<ElasticTarget, TargetError> {
        if query.filter.is_some() {
            return Err(TargetError::Other(
                "filters are not yet supported for the elastic target".to_string(),
            ));
        }
        // ES always searches a named `dense_vector` field; there's no unnamed
        // default like Qdrant, so `vector_name` (the field) is required.
        let vector_field = query.vector_name.clone().ok_or_else(|| {
            TargetError::Other(
                "elastic target requires `query.vector_name` (the dense_vector field to search)"
                    .to_string(),
            )
        })?;

        let sp: ElasticSearchParams = query
            .search_params
            .as_ref()
            .map(|v| serde_yaml::from_value(v.clone()))
            .transpose()
            .map_err(|e| TargetError::Other(format!("elastic search_params: {e}")))?
            .unwrap_or_default();
        // ES requires num_candidates >= k — validate at startup for a clear error
        // rather than failing every dispatch. Unset → omitted (ES default).
        if let Some(nc) = sp.num_candidates
            && nc < query.top_k
        {
            return Err(TargetError::Other(format!(
                "elastic num_candidates ({nc}) must be >= top_k ({})",
                query.top_k
            )));
        }

        let creds = if let Some(key) = self.api_key {
            Some(Credentials::EncodedApiKey(key))
        } else if let (Some(u), Some(p)) = (self.username, self.password) {
            Some(Credentials::Basic(u, p))
        } else {
            None
        };
        let pool = SingleNodeConnectionPool::new(Url::parse(&self.url).map_err(to_other)?);
        let mut builder =
            TransportBuilder::new(pool).timeout(std::time::Duration::from_secs(self.timeout_s));
        if let Some(creds) = creds {
            builder = builder.auth(creds);
        }
        if self.tls_insecure {
            builder = builder.cert_validation(CertificateValidation::None);
        }
        let transport = builder.build().map_err(to_other)?;

        Ok(ElasticTarget {
            client: Elasticsearch::new(transport),
            index_name: self.index_name,
            vector_field,
            top_k: query.top_k,
            num_candidates: sp.num_candidates,
            with_payload: query.with_payload.clone(),
            collect_ids: query.source.ground_truth_column.is_some(),
        })
    }
}

impl fmt::Display for ElasticTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "elastic({})", self.index_name)
    }
}

/// A whole-batch failure outcome (no per-query ids).
fn fail(started: Instant, n: usize, error: String) -> BatchOutcome {
    BatchOutcome {
        latency: started.elapsed(),
        ok: false,
        ids: vec![None; n],
        scores: vec![None; n],
        error: Some(error),
        timed_out: false,
    }
}

#[async_trait]
impl QueryTarget for ElasticTarget {
    async fn query_batch(&self, queries: &[&QueryVector]) -> BatchOutcome {
        let started = Instant::now();
        if queries.is_empty() {
            return BatchOutcome {
                latency: started.elapsed(),
                ok: true,
                ids: Vec::new(),
                scores: Vec::new(),
                error: None,
                timed_out: false,
            };
        }

        // What to return in `_source`. To match Qdrant's `with_payload` (payload
        // only, NOT the vector), exclude the vector field — ES stores the vector
        // in `_source`, so a plain `_source: true` would refetch+serialize it and
        // inflate the measured cost.
        let source = match &self.with_payload {
            crate::config::WithPayload::Enable(true) => {
                json!({ "excludes": [self.vector_field.as_str()] })
            }
            crate::config::WithPayload::Enable(false) => json!(false),
            // Field include-list (RAG shape): return exactly these _source
            // fields; the vector field is excluded implicitly by not being
            // listed.
            crate::config::WithPayload::Fields(fields) => json!({ "includes": fields }),
        };

        // `_msearch` body: one header line + one kNN search-body line per query.
        // `num_candidates` is omitted when unset so ES applies its own default.
        let mut body: Vec<JsonBody<Value>> = Vec::with_capacity(queries.len() * 2);
        for q in queries {
            // Dense-only target: guard at the point of use, so no separate
            // check can drift out of sync — a sparse query is a per-dispatch
            // data error, never a panic.
            let Some(dense) = q.vector.as_dense() else {
                return fail(
                    started,
                    queries.len(),
                    "the elastic target does not support sparse queries".to_string(),
                );
            };
            body.push(json!({ "index": self.index_name }).into());
            let mut knn = json!({
                "field": self.vector_field,
                "query_vector": dense,
                "k": self.top_k,
            });
            if let Some(nc) = self.num_candidates {
                knn["num_candidates"] = json!(nc);
            }
            body.push(json!({ "knn": knn, "_source": source, "size": self.top_k }).into());
        }

        let resp = match self
            .client
            .msearch(MsearchParts::None)
            .body(body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return fail(started, queries.len(), e.to_string()),
        };
        // Check the HTTP status before parsing JSON — a proxy 502 / auth 401 may
        // return non-JSON (HTML) that would otherwise surface as an opaque parse
        // error instead of the real status.
        let status = resp.status_code();
        if !status.is_success() {
            let detail = resp.text().await.unwrap_or_default();
            return fail(
                started,
                queries.len(),
                format!("msearch HTTP {status}: {detail}"),
            );
        }
        let val: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return fail(started, queries.len(), e.to_string()),
        };

        let Some(responses) = val["responses"].as_array() else {
            return fail(
                started,
                queries.len(),
                format!("msearch: no `responses` array: {val}"),
            );
        };
        // A count mismatch means responses can't be zipped positionally against
        // the submitted queries — treat as failure (as the Qdrant target does).
        if responses.len() != queries.len() {
            return fail(
                started,
                queries.len(),
                format!(
                    "msearch returned {} responses for {} queries",
                    responses.len(),
                    queries.len()
                ),
            );
        }

        let mut ids = Vec::with_capacity(queries.len());
        for (i, r) in responses.iter().enumerate() {
            // Any of these means this query's hits are incomplete or wrong —
            // fail the whole batch rather than score recall against a partial or
            // malformed result (a silent understated/zero recall would look like
            // an engine problem, not the infra failure it is).
            if let Some(err) = r.get("error").filter(|e| !e.is_null()) {
                return fail(
                    started,
                    queries.len(),
                    format!("msearch item {i} error: {err}"),
                );
            }
            if r["timed_out"].as_bool().unwrap_or(false) {
                return fail(
                    started,
                    queries.len(),
                    format!("msearch item {i} timed out: {r}"),
                );
            }
            let failed_shards = r["_shards"]["failed"].as_u64().unwrap_or(0);
            if failed_shards > 0 {
                return fail(
                    started,
                    queries.len(),
                    format!(
                        "msearch item {i}: {failed_shards} failed shard(s): {}",
                        r["_shards"]
                    ),
                );
            }
            let Some(hits) = r["hits"]["hits"].as_array() else {
                return fail(
                    started,
                    queries.len(),
                    format!("msearch item {i}: no `hits.hits` array: {r}"),
                );
            };

            if self.collect_ids {
                let mut query_ids = Vec::with_capacity(hits.len());
                for h in hits {
                    let Some(id) = h["_id"].as_str() else {
                        return fail(
                            started,
                            queries.len(),
                            format!("msearch item {i}: a hit has no string `_id`: {h}"),
                        );
                    };
                    query_ids.push(id.to_string());
                }
                ids.push(Some(query_ids));
            } else {
                ids.push(None);
            }
        }

        BatchOutcome {
            latency: started.elapsed(),
            ok: true,
            // This backend's response parsing doesn't extract scores, so the
            // tie-aware recall bounds collapse to exact recall here (see
            // docs/storm/recall.md). Left unparsed deliberately rather than
            // guessed: elastic reports `_score` (higher-is-better, with
            // l2_norm already inverted), a different convention again —
            // parsing it needs its own verification, not milvus's.
            scores: vec![None; ids.len()],
            ids,
            error: None,
            timed_out: false,
        }
    }
}
