//! Elasticsearch load backend (feature `elastic`).
//!
//! Maps each [`Point`] to an ES document: dense vectors become `dense_vector`
//! fields, the payload lands as ordinary (dynamically-mapped) fields, and the
//! point id becomes the document `_id`. Upserts go through the bulk API.
//!
//! Scope for now: **dense vectors + payload**. Sparse and multivector values
//! error out — ES models those differently (`sparse_vector`, and no real
//! multivector), and this backend exists to get dense corpora into an ES cluster.

use std::fmt;

use async_trait::async_trait;
use elasticsearch::auth::Credentials;
use elasticsearch::http::transport::{SingleNodeConnectionPool, Transport, TransportBuilder};
use elasticsearch::indices::{IndicesCreateParts, IndicesDeleteParts, IndicesExistsParts};
use elasticsearch::{BulkOperation, BulkParts, Elasticsearch};
use serde::Deserialize;
use serde_json::{Value, json};
use url::Url;

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
            .field("recreate", &self.recreate)
            .finish()
    }
}

pub struct ElasticStore {
    client: Elasticsearch,
    index_name: String,
    recreate: bool,
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

        let client = match creds {
            None => Elasticsearch::new(Transport::single_node(&self.url).map_err(to_other)?),
            Some(creds) => {
                let pool = SingleNodeConnectionPool::new(Url::parse(&self.url).map_err(to_other)?);
                let transport =
                    TransportBuilder::new(pool).auth(creds).build().map_err(to_other)?;
                Elasticsearch::new(transport)
            }
        };

        Ok(ElasticStore { client, index_name: self.index_name, recreate: self.recreate })
    }
}

fn to_other<E: std::fmt::Display>(e: E) -> StoreError {
    StoreError::Other(e.to_string())
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
        use crate::config::VectorKind;

        let mut props = serde_json::Map::new();
        for (name, spec) in &schema.vectors {
            match spec.kind {
                VectorKind::Dense => {
                    let dims = schema.dims.get(name).ok_or_else(|| {
                        StoreError::Other(format!("vector `{name}`: dims not resolved"))
                    })?;
                    props.insert(
                        name.clone(),
                        json!({
                            "type": "dense_vector",
                            "dims": dims,
                            "index": true,
                            "similarity": similarity(spec.distance.as_deref()),
                        }),
                    );
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

    async fn close(&self) -> Result<(), StoreError> {
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
