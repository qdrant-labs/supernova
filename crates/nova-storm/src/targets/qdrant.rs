//! Qdrant implementation of [`QueryTarget`].

use std::fmt;
use std::time::Instant;

use async_trait::async_trait;
use qdrant_client::qdrant::QueryPointsBuilder;
use qdrant_client::{Qdrant, QdrantError};
use serde::{Deserialize, Serialize};

use super::{QueryOutcome, QueryTarget};
use crate::config::QueryConfig;
use crate::errors::TargetError;

/// Fires nearest-neighbour queries at a Qdrant collection over gRPC.
pub struct QdrantTarget {
    client: Qdrant,
    collection_name: String,
    vector_name: Option<String>,
    top_k: u64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QdrantConfig {
    pub url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_collection")]
    pub collection_name: String,
}

fn default_collection() -> String {
    "default".to_string()
}

impl QdrantConfig {
    /// Connect and build the target, baking in the query knobs from `query`.
    pub fn into_target(self, query: &QueryConfig) -> Result<QdrantTarget, TargetError> {
        // The filter *shape* is Qdrant-native (`must`/`should`/...), but those
        // protobuf types don't round-trip through serde, so compiling one from
        // YAML is a real chunk of work. Until that lands, refuse a configured
        // filter loudly rather than measure unfiltered load as if it were
        // filtered (mirrors the Python base class raising on unsupported).
        if query.filter.is_some() {
            return Err(TargetError::Other(
                "query.filter is not yet supported by the Rust storm; remove it or run \
                 unfiltered for now"
                    .into(),
            ));
        }

        let mut builder = Qdrant::from_url(&self.url);
        if let Some(key) = self.api_key {
            builder = builder.api_key(key);
        }
        let client = builder.build()?;

        Ok(QdrantTarget {
            client,
            collection_name: self.collection_name,
            vector_name: query.vector_name.clone(),
            top_k: query.top_k,
        })
    }
}

impl fmt::Display for QdrantTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "qdrant({})", self.collection_name)
    }
}

#[async_trait]
impl QueryTarget for QdrantTarget {
    async fn query(&self, vector: &[f32]) -> QueryOutcome {
        let mut builder = QueryPointsBuilder::new(&self.collection_name)
            .query(vector.to_vec())
            .limit(self.top_k)
            .with_payload(false);
        if let Some(name) = &self.vector_name {
            builder = builder.using(name.clone());
        }

        let started = Instant::now();
        match self.client.query(builder).await {
            Ok(resp) => QueryOutcome {
                latency: started.elapsed(),
                ok: true,
                matched: resp.result.len(),
                error: None,
            },
            Err(e) => QueryOutcome {
                latency: started.elapsed(),
                ok: false,
                matched: 0,
                error: Some(qdrant_error_string(&e)),
            },
        }
    }
}

fn qdrant_error_string(e: &QdrantError) -> String {
    e.to_string()
}

#[cfg(test)]
mod tests {
    use crate::config::StormConfig;
    use crate::errors::TargetError;

    fn cfg_with_filter(filter: bool) -> StormConfig {
        let filter_block = if filter {
            "  filter:\n    must:\n      - key: k\n        match:\n          value: v\n"
        } else {
            ""
        };
        let yaml = format!(
            "target:\n  type: qdrant\n  url: http://localhost:6334\n  collection_name: c\n\
             query:\n  vector_name: dense\n  top_k: 5\n  source:\n    uri: /tmp/q.parquet\n    column: e\n{filter_block}"
        );
        crate::config::load_config_str(&yaml).expect("parses")
    }

    #[test]
    fn rejects_unsupported_filter() {
        // `Arc<dyn QueryTarget>` isn't Debug, so match rather than expect_err.
        let cfg = cfg_with_filter(true);
        match cfg.target.into_target(&cfg.query) {
            Err(TargetError::Other(_)) => {}
            Err(e) => panic!("expected Other, got {e:?}"),
            Ok(_) => panic!("expected a filter to be rejected"),
        }
    }

    #[test]
    fn builds_target_without_filter() {
        // `from_url(...).build()` is lazy (no connection yet), so this succeeds
        // offline and exercises the construction path.
        let cfg = cfg_with_filter(false);
        let target = cfg.target.into_target(&cfg.query).expect("builds");
        assert_eq!(target.to_string(), "qdrant(c)");
    }
}
