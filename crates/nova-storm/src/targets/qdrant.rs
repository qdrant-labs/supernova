//! Qdrant implementation of [`QueryTarget`].

use std::fmt;
use std::time::Instant;

use async_trait::async_trait;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::{QueryBatchPointsBuilder, QueryPointsBuilder, ScoredPoint};
use serde::Deserialize;

use super::{BatchOutcome, QueryTarget};
use crate::config::QueryConfig;
use crate::errors::TargetError;

/// Fires nearest-neighbour queries at a Qdrant collection over gRPC.
pub struct QdrantTarget {
    client: Qdrant,
    collection_name: String,
    vector_name: Option<String>,
    top_k: u64,
    /// Whether to materialize `BatchOutcome.ids`. Skipped (leaving each
    /// position `None`) when the run has no `ground_truth_column` configured
    /// — recall is the only consumer of `ids`, so collecting it otherwise is
    /// a wasted allocation (a `String` clone per returned point) on every
    /// query.
    collect_ids: bool,
}

#[derive(Debug, Deserialize)]
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
            collect_ids: query.source.ground_truth_column.is_some(),
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
    async fn query_batch(&self, vectors: &[&[f32]]) -> BatchOutcome {
        let query_points: Vec<_> = vectors
            .iter()
            .map(|vector| {
                let mut builder = QueryPointsBuilder::new(&self.collection_name)
                    .query(vector.to_vec())
                    .limit(self.top_k)
                    .with_payload(false);
                if let Some(name) = &self.vector_name {
                    builder = builder.using(name.clone());
                }
                builder.build()
            })
            .collect();
        let request = QueryBatchPointsBuilder::new(&self.collection_name, query_points);

        let started = Instant::now();
        match self.client.query_batch(request).await {
            Ok(resp) => {
                debug_assert_eq!(
                    resp.result.len(),
                    vectors.len(),
                    "query_batch response length must match the submitted batch size"
                );
                BatchOutcome {
                    latency: started.elapsed(),
                    ok: true,
                    ids: resp
                        .result
                        .into_iter()
                        .map(|batch_result| {
                            self.collect_ids
                                .then(|| batch_result.result.iter().filter_map(point_id_string).collect())
                        })
                        .collect(),
                    error: None,
                }
            }
            Err(e) => BatchOutcome {
                latency: started.elapsed(),
                ok: false,
                ids: vec![None; vectors.len()],
                error: Some(e.to_string()),
            },
        }
    }
}

/// A scored point's id as a plain string — `Uuid` verbatim (what a
/// `nova-load`-populated collection always uses), `Num` decimal-formatted.
/// Canonicalizing both to strings here is what lets recall be computed as a
/// plain string-set intersection against `hit_ids`, which `nova bf` also
/// always stores as strings (see its own `str(...)` coercion) regardless of
/// whether the underlying collection uses UUID or integer ids.
fn point_id_string(point: &ScoredPoint) -> Option<String> {
    match point.id.as_ref()?.point_id_options.as_ref()? {
        PointIdOptions::Uuid(s) => Some(s.clone()),
        PointIdOptions::Num(n) => Some(n.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use qdrant_client::qdrant::PointId;

    use super::*;
    use crate::config::StormConfig;

    fn cfg() -> StormConfig {
        let yaml = "target:\n  type: qdrant\n  url: http://localhost:6334\n  collection_name: c\n\
                    query:\n  vector_name: dense\n  top_k: 5\n  source:\n    uri: /tmp/q.parquet\n    column: e\n";
        StormConfig::from_yaml(yaml).expect("parses")
    }

    fn scored(id: Option<PointId>) -> ScoredPoint {
        ScoredPoint {
            id,
            payload: Default::default(),
            score: 0.0,
            version: 0,
            vectors: None,
            shard_key: None,
            order_value: None,
        }
    }

    #[test]
    fn point_id_string_formats_uuid_and_num() {
        let uuid = PointId { point_id_options: Some(PointIdOptions::Uuid("abc-123".into())) };
        assert_eq!(point_id_string(&scored(Some(uuid))), Some("abc-123".to_string()));

        let num = PointId { point_id_options: Some(PointIdOptions::Num(42)) };
        assert_eq!(point_id_string(&scored(Some(num))), Some("42".to_string()));

        assert_eq!(point_id_string(&scored(None)), None);
    }

    #[test]
    fn builds_target() {
        // `from_url(...).build()` is lazy (no connection yet), so this succeeds
        // offline and exercises the construction path.
        let cfg = cfg();
        let target = cfg.target.into_target(&cfg.query).expect("builds");
        assert_eq!(target.to_string(), "qdrant(c)");
    }

    fn qdrant_config(target: crate::targets::TargetConfig) -> QdrantConfig {
        match target {
            crate::targets::TargetConfig::Qdrant(c) => c,
        }
    }

    #[test]
    fn collect_ids_follows_ground_truth_column_config() {
        let cfg = cfg(); // no ground_truth_column
        let target = qdrant_config(cfg.target).into_target(&cfg.query).expect("builds");
        assert!(!target.collect_ids);

        let yaml = "target:\n  type: qdrant\n  url: http://localhost:6334\n  collection_name: c\n\
                    query:\n  top_k: 5\n  source:\n    uri: /tmp/q.parquet\n    column: e\n    ground_truth_column: hit_ids\n";
        let cfg = StormConfig::from_yaml(yaml).expect("parses");
        let target = qdrant_config(cfg.target).into_target(&cfg.query).expect("builds");
        assert!(target.collect_ids);
    }
}
