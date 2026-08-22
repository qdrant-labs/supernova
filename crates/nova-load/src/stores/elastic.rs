//! Elasticsearch load backend (feature `elastic`).
//!
//! Maps each [`Point`] to an ES document: dense vectors become `dense_vector`
//! fields, the payload lands as ordinary (dynamically-mapped) fields, and the
//! point id becomes the document `_id`. Upserts go through the bulk API.
//!
//! Scope for now: **dense vectors + payload**. Sparse and multivector values
//! error out — ES models those differently (`sparse_vector`, and no real
//! multivector), and this backend exists to get dense corpora into an ES cluster.

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use elasticsearch::auth::Credentials;
use elasticsearch::cert::CertificateValidation;
use elasticsearch::http::transport::{SingleNodeConnectionPool, TransportBuilder};
use elasticsearch::indices::{
    IndicesCreateParts, IndicesDeleteParts, IndicesExistsParts, IndicesForcemergeParts,
    IndicesGetMappingParts, IndicesPutMappingParts, IndicesPutSettingsParts, IndicesRefreshParts,
    IndicesStatsParts,
};
use elasticsearch::{BulkOperation, BulkParts, Elasticsearch};
use serde::Deserialize;
use serde_json::{Value, json};
use url::Url;

use crate::config::VectorKind;
use crate::stores::{CollectionSchema, Point, PointId, StoreError, VectorStore, VectorValue};

/// Connection + store settings for an Elasticsearch backend (`vectorstore:`).
/// Per-vector schema (dims, distance) comes from the top-level `vectors:` block,
/// like every other backend.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElasticConfig {
    /// Node URL, e.g. `https://localhost:9200`.
    pub url: String,
    /// Basic-auth username (paired with `password`).
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub password: Option<String>,
    /// Base64 `id:api_key` (an alternative to username/password).
    #[serde(default)]
    pub api_key: Option<String>,
    /// Target index. Defaults to `default`.
    #[serde(default = "default_index")]
    pub index_name: String,
    /// Skip TLS certificate validation. Needed for a default ES 8 dev node, which
    /// serves HTTPS with a self-signed cert. DEV ONLY — don't use in production.
    #[serde(default)]
    pub tls_insecure: bool,
    /// `dense_vector` `index_options` passed through verbatim into the field
    /// mapping (e.g. `{type: int8_hnsw, m: 16, ef_construction: 200}`). When
    /// unset, ES picks its default. The `similarity` is NOT set here — it comes
    /// from the per-vector `distance`, like every other backend.
    #[serde(default)]
    pub index_options: Option<HashMap<String, Value>>,
    /// Drop + recreate the index if it already exists (instead of reusing it).
    #[serde(default)]
    pub recreate: bool,
}

fn default_index() -> String {
    "default".to_string()
}

/// Manual `Debug` so secrets never reach logs / `--dry-run`.
impl fmt::Debug for ElasticConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ElasticConfig")
            .field("url", &self.url)
            .field("username", &self.username)
            .field("password", &self.password.as_ref().map(|_| "<redacted>"))
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("index_name", &self.index_name)
            .field("tls_insecure", &self.tls_insecure)
            .field("index_options", &self.index_options)
            .field("recreate", &self.recreate)
            .finish()
    }
}

pub struct ElasticStore {
    client: Elasticsearch,
    index_name: String,
    recreate: bool,
    /// `dense_vector` index_options from config, applied at mapping creation and
    /// (re)applied by `reindex`.
    index_options: Option<HashMap<String, Value>>,
    /// Per-vector distance→similarity + index_options that the last create/reindex
    /// asked for, checked against the live mapping by the sanity check in
    /// `wait_for_indexing`. Populated by `enable_indexing`/`reindex` on the same
    /// store instance (both run via `finish_indexing`/`reindex`).
    expected: Mutex<Vec<ExpectedField>>,
}

/// The mapping config expected for one dense_vector field, checked against the
/// live mapping after (re)indexing.
#[derive(Clone)]
struct ExpectedField {
    field: String,
    similarity: String,
    index_options: Option<HashMap<String, Value>>,
}

impl ElasticConfig {
    pub async fn connect(self) -> Result<ElasticStore, StoreError> {
        let creds = if let Some(key) = self.api_key {
            Some(Credentials::EncodedApiKey(key))
        } else if let (Some(u), Some(p)) = (self.username, self.password) {
            Some(Credentials::Basic(u, p))
        } else {
            None
        };

        // Always go through TransportBuilder so auth + TLS validation are
        // configurable (Transport::single_node can do neither).
        let pool = SingleNodeConnectionPool::new(Url::parse(&self.url).map_err(to_other)?);
        let mut builder = TransportBuilder::new(pool);
        if let Some(creds) = creds {
            builder = builder.auth(creds);
        }
        if self.tls_insecure {
            builder = builder.cert_validation(CertificateValidation::None);
        }
        let transport = builder.build().map_err(to_other)?;

        Ok(ElasticStore {
            client: Elasticsearch::new(transport),
            index_name: self.index_name,
            recreate: self.recreate,
            index_options: self.index_options,
            expected: Mutex::new(Vec::new()),
        })
    }
}

/// Flatten an error and its `source()` chain into one message — the top-level
/// elasticsearch/reqwest message is often just "error sending request for url
/// (...)", with the real cause (connection refused, invalid certificate, TLS
/// handshake) one or two links down.
fn to_other<E: std::error::Error>(e: E) -> StoreError {
    let mut msg = e.to_string();
    let mut src = e.source();
    while let Some(s) = src {
        msg.push_str(&format!(": {s}"));
        src = s.source();
    }
    StoreError::Other(msg)
}

/// Qdrant-style distance name → ES `dense_vector` similarity.
fn similarity(distance: Option<&str>) -> &'static str {
    match distance.unwrap_or("cosine") {
        "dot" => "dot_product",
        "euclid" => "l2_norm",
        _ => "cosine",
    }
}

impl fmt::Display for ElasticStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "elastic({})", self.index_name)
    }
}

impl ElasticStore {
    /// Build the `mappings.properties` for the index: one `dense_vector` per
    /// dense vector spec. Payload fields are left to ES dynamic mapping.
    fn mappings(&self, schema: &CollectionSchema) -> Result<Value, StoreError> {
        let mut props = serde_json::Map::new();
        for (name, spec) in &schema.vectors {
            match spec.kind {
                VectorKind::Dense => {
                    let dims = schema.dims.get(name).ok_or_else(|| {
                        StoreError::Other(format!("vector `{name}`: dims not resolved"))
                    })?;
                    let mut field = serde_json::Map::from_iter([
                        ("type".to_string(), json!("dense_vector")),
                        ("dims".to_string(), json!(dims)),
                        ("index".to_string(), json!(true)),
                        ("similarity".to_string(), json!(similarity(spec.distance.as_deref()))),
                    ]);
                    if let Some(opts) = &self.index_options {
                        field.insert("index_options".to_string(), json!(opts));
                    }
                    props.insert(name.clone(), Value::Object(field));
                }
                VectorKind::Sparse | VectorKind::Multivector => {
                    return Err(StoreError::Other(format!(
                        "elastic backend supports dense vectors only for now; \
                         vector `{name}` is {:?}",
                        spec.kind
                    )));
                }
            }
        }
        Ok(json!({ "mappings": { "properties": props } }))
    }

    /// PUT the index's `refresh_interval` setting. `Value::Null` clears the
    /// override (ES falls back to its 1s default); a string like `"-1"` disables
    /// periodic refresh for bulk loading.
    async fn put_refresh_interval(&self, value: Value) -> Result<(), StoreError> {
        let resp = self
            .client
            .indices()
            .put_settings(IndicesPutSettingsParts::Index(&[self.index_name.as_str()]))
            .body(json!({ "index": { "refresh_interval": value } }))
            .send()
            .await
            .map_err(to_other)?;
        if !resp.status_code().is_success() {
            let detail = resp.text().await.unwrap_or_default();
            return Err(StoreError::Other(format!("put refresh_interval failed: {detail}")));
        }
        Ok(())
    }

    /// Force a refresh so newly-indexed documents become searchable now.
    async fn refresh(&self) -> Result<(), StoreError> {
        let resp = self
            .client
            .indices()
            .refresh(IndicesRefreshParts::Index(&[self.index_name.as_str()]))
            .send()
            .await
            .map_err(to_other)?;
        if !resp.status_code().is_success() {
            let detail = resp.text().await.unwrap_or_default();
            return Err(StoreError::Other(format!("refresh failed: {detail}")));
        }
        Ok(())
    }

    /// Number of merges currently running on the index (`_stats/merge` →
    /// `indices.<name>.total.merges.current`). ES builds each segment's HNSW
    /// graph when the segment is written and rebuilds it as segments merge, so a
    /// settled merge count is the "no more index building happening" signal —
    /// the ES analog of Qdrant's green status and Milvus's `pending_index_rows`.
    async fn merges_current(&self) -> Result<i64, StoreError> {
        let index = self.index_name.as_str();
        let resp = self
            .client
            .indices()
            .stats(IndicesStatsParts::IndexMetric(&[index], &["merge"]))
            .send()
            .await
            .map_err(to_other)?;
        if !resp.status_code().is_success() {
            let detail = resp.text().await.unwrap_or_default();
            return Err(StoreError::Other(format!("merge stats failed: {detail}")));
        }
        let body: Value = resp.json().await.map_err(to_other)?;
        // Prefer the per-index total; fall back to the cluster-wide `_all` total.
        let current = body["indices"][index]["total"]["merges"]["current"]
            .as_i64()
            .or_else(|| body["_all"]["total"]["merges"]["current"].as_i64())
            .unwrap_or(0);
        Ok(current)
    }

    /// ES's own cumulative index time (ms). ES builds each `dense_vector` HNSW
    /// graph inline at segment flush *during ingestion* — there's no separate
    /// post-upload build phase like Milvus/Qdrant — so this stat (not our
    /// post-load `index_seconds`) is the meaningful "indexing cost" figure.
    async fn index_time_millis(&self) -> Result<i64, StoreError> {
        let index = self.index_name.as_str();
        let resp = self
            .client
            .indices()
            .stats(IndicesStatsParts::IndexMetric(&[index], &["indexing"]))
            .send()
            .await
            .map_err(to_other)?;
        let body: Value = resp.json().await.map_err(to_other)?;
        Ok(body["indices"][index]["total"]["indexing"]["index_time_in_millis"]
            .as_i64()
            .or_else(|| body["_all"]["total"]["indexing"]["index_time_in_millis"].as_i64())
            .unwrap_or(0))
    }

    /// The live `mappings.properties` object for this index.
    async fn get_properties(&self) -> Result<Value, StoreError> {
        let index = self.index_name.as_str();
        let resp = self
            .client
            .indices()
            .get_mapping(IndicesGetMappingParts::Index(&[index]))
            .send()
            .await
            .map_err(to_other)?;
        if !resp.status_code().is_success() {
            let detail = resp.text().await.unwrap_or_default();
            return Err(StoreError::Other(format!("get mapping failed: {detail}")));
        }
        let body: Value = resp.json().await.map_err(to_other)?;
        Ok(body[index]["mappings"]["properties"].clone())
    }

    /// PUT a single field's mapping definition (used to update `index_options`).
    /// ES enforces which changes are legal (e.g. HNSW `m` may only increase) and
    /// rejects the rest — we surface its error verbatim.
    async fn put_field_mapping(&self, field: &str, def: Value) -> Result<(), StoreError> {
        let index = self.index_name.as_str();
        let resp = self
            .client
            .indices()
            .put_mapping(IndicesPutMappingParts::Index(&[index]))
            .body(json!({ "properties": { field: def } }))
            .send()
            .await
            .map_err(to_other)?;
        if !resp.status_code().is_success() {
            let detail = resp.text().await.unwrap_or_default();
            return Err(StoreError::Other(format!("put mapping for `{field}` failed: {detail}")));
        }
        Ok(())
    }

    /// Trigger an async force-merge to a single segment. Rewriting the segments
    /// re-encodes existing vectors into the current `index_options` format (an
    /// Update Mapping alone leaves already-indexed vectors in their old format),
    /// so this is what makes a `reindex` actually take effect. Async so it
    /// doesn't block on a long merge; `wait_for_indexing` waits for it to settle.
    async fn force_merge(&self) -> Result<(), StoreError> {
        let index = self.index_name.as_str();
        let resp = self
            .client
            .indices()
            .forcemerge(IndicesForcemergeParts::Index(&[index]))
            .max_num_segments(1)
            .wait_for_completion(false)
            .send()
            .await
            .map_err(to_other)?;
        if !resp.status_code().is_success() {
            let detail = resp.text().await.unwrap_or_default();
            return Err(StoreError::Other(format!("force merge failed: {detail}")));
        }
        Ok(())
    }

    /// Per-vector expected mapping (similarity from distance + configured
    /// index_options), erroring on sparse/multivector.
    fn expected_fields(&self, schema: &CollectionSchema) -> Result<Vec<ExpectedField>, StoreError> {
        schema
            .vectors
            .iter()
            .map(|(name, spec)| match spec.kind {
                VectorKind::Dense => Ok(ExpectedField {
                    field: name.clone(),
                    similarity: similarity(spec.distance.as_deref()).to_string(),
                    index_options: self.index_options.clone(),
                }),
                VectorKind::Sparse | VectorKind::Multivector => Err(StoreError::Other(format!(
                    "elastic backend supports dense vectors only for now; \
                     vector `{name}` is {:?}",
                    spec.kind
                ))),
            })
            .collect()
    }

    /// Sanity check: confirm the live mapping reflects what we asked for — the
    /// `similarity` matches and every configured `index_options` key is present
    /// with the requested value (ES fills in defaults for the rest, so this is a
    /// subset check, not full equality). Runs after (re)indexing settles.
    async fn verify_expected(&self) -> Result<(), StoreError> {
        let expected = self.expected.lock().expect("mapping lock").clone();
        if expected.is_empty() {
            return Ok(());
        }
        let props = self.get_properties().await?;
        for e in &expected {
            let got = &props[&e.field];
            if got.is_null() {
                return Err(StoreError::Other(format!(
                    "sanity check: field `{}` missing from mapping on {self}",
                    e.field
                )));
            }
            let got_sim = got["similarity"].as_str().unwrap_or_default();
            if got_sim != e.similarity {
                return Err(StoreError::Other(format!(
                    "sanity check FAILED on {self} field `{}`: requested similarity={} \
                     but mapping has {got_sim}",
                    e.field, e.similarity
                )));
            }
            if let Some(opts) = &e.index_options {
                let got_opts = &got["index_options"];
                for (k, v) in opts {
                    if got_opts.get(k) != Some(v) {
                        return Err(StoreError::Other(format!(
                            "sanity check FAILED on {self} field `{}`: requested \
                             index_options.{k}={v} but mapping has {:?}",
                            e.field,
                            got_opts.get(k)
                        )));
                    }
                }
            }
            tracing::info!(
                "{self} field `{}` verified: similarity={got_sim} index_options={}",
                e.field,
                got["index_options"]
            );
        }
        Ok(())
    }
}

#[async_trait]
impl VectorStore for ElasticStore {
    async fn ensure_collection(&self, schema: &CollectionSchema) -> Result<(), StoreError> {
        let index = self.index_name.as_str();
        let exists = self
            .client
            .indices()
            .exists(IndicesExistsParts::Index(&[index]))
            .send()
            .await
            .map_err(to_other)?
            .status_code()
            .is_success();

        if exists {
            if !self.recreate {
                return Ok(());
            }
            self.client
                .indices()
                .delete(IndicesDeleteParts::Index(&[index]))
                .send()
                .await
                .map_err(to_other)?;
        }

        let body = self.mappings(schema)?;
        let resp = self
            .client
            .indices()
            .create(IndicesCreateParts::Index(index))
            .body(body)
            .send()
            .await
            .map_err(to_other)?;
        if !resp.status_code().is_success() {
            let detail = resp.text().await.unwrap_or_default();
            return Err(StoreError::Other(format!("create index `{index}` failed: {detail}")));
        }
        Ok(())
    }

    async fn upsert_batch(&self, points: Vec<Point>) -> Result<(), StoreError> {
        let mut ops: Vec<BulkOperation<Value>> = Vec::with_capacity(points.len());
        for point in points {
            let id = match &point.id {
                PointId::Integer(n) => n.to_string(),
                PointId::String(s) => s.clone(),
            };
            let mut doc = Value::Object(point.payload);
            let obj = doc.as_object_mut().expect("payload is an object");
            for (name, value) in point.vectors {
                match value {
                    VectorValue::Dense(d) => {
                        obj.insert(name, json!(d));
                    }
                    VectorValue::Sparse { .. } | VectorValue::Multi(_) => {
                        return Err(StoreError::Other(format!(
                            "elastic backend supports dense vectors only; \
                             vector `{name}` is not dense"
                        )));
                    }
                }
            }
            ops.push(BulkOperation::index(doc).id(id).into());
        }

        let resp = self
            .client
            .bulk(BulkParts::Index(self.index_name.as_str()))
            .body(ops)
            .send()
            .await
            .map_err(to_other)?;
        if !resp.status_code().is_success() {
            let detail = resp.text().await.unwrap_or_default();
            return Err(StoreError::Other(format!("bulk upsert failed: {detail}")));
        }
        // A 200 can still carry per-item failures; surface the first one.
        let body: Value = resp.json().await.map_err(to_other)?;
        if body["errors"].as_bool() == Some(true) {
            let first = body["items"]
                .as_array()
                .and_then(|items| items.iter().find_map(|it| it["index"]["error"].as_object()))
                .map(|e| Value::Object(e.clone()).to_string())
                .unwrap_or_else(|| "unknown bulk item error".to_string());
            return Err(StoreError::Other(format!("bulk upsert had item errors: {first}")));
        }
        Ok(())
    }

    async fn point_exists(&self, _id: &PointId) -> Result<bool, StoreError> {
        // The `--continue` resume probe is Qdrant-only for now; implementing
        // it here needs a cheap get-by-id (doable, just untested).
        Err(StoreError::Other(
            "`--continue` resume probing is not implemented for the elastic backend".into(),
        ))
    }

    async fn close(&self) -> Result<(), StoreError> {
        Ok(())
    }

    async fn defer_indexing(&self) -> Result<(), StoreError> {
        // Disable periodic refresh so bulk indexing doesn't pay to build a new
        // searchable segment on every interval. The index is created by
        // `ensure_collection` before this runs, so the settings PUT targets an
        // existing index. Restored in `enable_indexing`.
        self.put_refresh_interval(json!("-1")).await
    }

    async fn enable_indexing(&self, schema: &CollectionSchema) -> Result<(), StoreError> {
        // Restore the default refresh behavior (null = fall back to the 1s
        // default), then force one refresh so everything bulk-loaded becomes
        // searchable immediately rather than on the next interval.
        self.put_refresh_interval(Value::Null).await?;
        self.refresh().await?;
        // Record what the mapping should be so `wait_for_indexing` can verify it.
        *self.expected.lock().expect("mapping lock") = self.expected_fields(schema)?;
        Ok(())
    }

    async fn wait_for_indexing(&self) -> Result<Instant, StoreError> {
        // Reach a genuine steady state, like the Qdrant and Milvus backends —
        // not just "docs are searchable". ES writes each segment's HNSW graph when
        // the segment is flushed and then rebuilds graphs as segments merge in the
        // background, so "indexing is really done" means: the latest writes are
        // flushed (force a refresh) AND no merges are running — HELD for a window,
        // so a brief lull between merge waves doesn't look like completion. Return
        // the instant the merges first settled (start of the hold), so the caller
        // measures real build time excluding the hold.
        const POLL_INTERVAL: Duration = Duration::from_secs(1);
        const SETTLED_HOLD: Duration = Duration::from_secs(5);
        const TIMEOUT: Duration = Duration::from_secs(1800);

        // Flush the last segments (with their graphs) so they're searchable and
        // counted; also correct when called standalone (e.g. after `reindex`).
        self.refresh().await?;

        let start = Instant::now();
        let mut settled_since: Option<Instant> = None;
        loop {
            if start.elapsed() >= TIMEOUT {
                return Err(StoreError::Other(format!(
                    "timed out after {}s waiting for merges to settle on {self}",
                    TIMEOUT.as_secs()
                )));
            }

            let current = self.merges_current().await?;
            tracing::info!("{self} indexing: merges_current={current}");

            if current == 0 {
                let since = *settled_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= SETTLED_HOLD {
                    tracing::info!(
                        "{self} indexing settled (no merges running for {}s)",
                        SETTLED_HOLD.as_secs()
                    );
                    // Sanity-check the mapping (runs after `since`, so it's not
                    // counted in the caller's timing). The indexing-time figure is
                    // reported separately by `report_index_time`.
                    self.verify_expected().await?;
                    return Ok(since);
                }
            } else if settled_since.take().is_some() {
                // Merges resumed after a lull — restart the hold window.
                tracing::warn!(
                    "{self} merges resumed (merges_current={current}); restarting {}s window",
                    SETTLED_HOLD.as_secs()
                );
            }

            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    async fn report_index_time(&self, effective: std::time::Duration) {
        // ES builds the HNSW graph inline during ingestion, so `effective` (the
        // post-load merge-settle window) is NOT the indexing cost — it's ~0.
        // Report ES's own cumulative index_time as the headline figure instead.
        match self.index_time_millis().await {
            Ok(ms) => tracing::info!(
                "{self} indexing finished: index_seconds={:.3} \
                 (ES builds inline during ingest; post-load merge-settle={:.3}s)",
                ms as f64 / 1000.0,
                effective.as_secs_f64()
            ),
            Err(e) => tracing::info!(
                "{self} indexing finished: index_seconds={:.3} \
                 (ES index_time unavailable: {e})",
                effective.as_secs_f64()
            ),
        }
    }

    async fn reindex(&self, schema: &CollectionSchema) -> Result<(), StoreError> {
        // ES reindex, in-place: update each dense_vector field's `index_options`
        // via the Update Mapping API, then force-merge so existing vectors are
        // re-encoded into the new format (an Update Mapping alone leaves already-
        // indexed vectors in their old format). The subsequent `wait_for_indexing`
        // waits for the merge to settle and sanity-checks the live mapping.
        //
        // `similarity`/`dims` are immutable in ES — if the requested distance
        // differs from the existing mapping, we error out rather than silently
        // no-op'ing (changing it requires a NEW index + the `_reindex` API).
        let expected = self.expected_fields(schema)?;
        let props = self.get_properties().await?;

        for e in &expected {
            let existing = &props[&e.field];
            if existing.is_null() {
                return Err(StoreError::Other(format!(
                    "reindex: field `{}` not found in mapping on {self}",
                    e.field
                )));
            }
            let existing_sim = existing["similarity"].as_str().unwrap_or_default();
            if existing_sim != e.similarity {
                return Err(StoreError::Other(format!(
                    "reindex: elastic can't change `similarity` in place on `{}` \
                     ({existing_sim} → {}); that needs a new index + the _reindex API",
                    e.field, e.similarity
                )));
            }
            match &self.index_options {
                Some(opts) => {
                    let mut def = existing.as_object().cloned().unwrap_or_default();
                    def.insert("index_options".to_string(), json!(opts));
                    tracing::info!(
                        "{self}: reindexing `{}` → index_options={}",
                        e.field,
                        json!(opts)
                    );
                    // ES validates legality (e.g. HNSW `m` may only increase).
                    self.put_field_mapping(&e.field, Value::Object(def)).await?;
                }
                None => tracing::warn!(
                    "{self}: no index_options configured; reindex only force-merges to \
                     consolidate segments (no mapping change) on `{}`",
                    e.field
                ),
            }
        }

        *self.expected.lock().expect("mapping lock") = expected;
        // Rewrite existing vectors into the (possibly new) format.
        self.force_merge().await?;
        Ok(())
    }

    async fn delete_collection(&self) -> Result<(), StoreError> {
        // Best-effort: a 404 (index absent) is fine.
        self.client
            .indices()
            .delete(IndicesDeleteParts::Index(&[self.index_name.as_str()]))
            .send()
            .await
            .map_err(to_other)?;
        Ok(())
    }
}
