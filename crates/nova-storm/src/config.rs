//! Parsing for the storm `.yaml`.
//!
//! Top level has three keys the single-machine binary cares about: `target`
//! (the cluster under test, dispatched on its `type`), `query` (what to search
//! with + where the query vectors live), and `load` (the per-worker profile).
//!
//! `metrics`, `dispatch`, and `resources` belong to the distributed dispatcher
//! and SkyPilot; the binary ignores them but parses them so the shipped config
//! validates under `deny_unknown_fields`.

use nova_metrics::MetricsConfig;
use serde::{Deserialize, Serialize};

use crate::targets::TargetConfig;

/// The full parsed storm config. `Serialize` so the resolved config can be
/// stored on the `runs` row (secrets are redacted by the metrics sink).
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StormConfig {
    pub target: TargetConfig,
    pub query: QueryConfig,
    #[serde(default)]
    pub load: LoadProfile,

    /// Where measurements go. Absent → stdout (local-first, no setup).
    #[serde(default)]
    pub metrics: Option<MetricsConfig>,
    // Consumed by `nova storm-dist` / SkyPilot; ignored here. Parsed as opaque
    // values so the keys are allowed but never silently mistyped into `target`.
    #[serde(default)]
    pub dispatch: Option<serde_yaml::Value>,
    #[serde(default)]
    pub resources: Option<serde_yaml::Value>,
}

/// What to query with: the named vector, how many neighbours, and where the
/// query vectors are read from.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QueryConfig {
    /// Named vector to search (`None` for a single-vector collection).
    #[serde(default)]
    pub vector_name: Option<String>,
    #[serde(default = "default_top_k")]
    pub top_k: u64,
    pub source: QuerySource,
    /// Optional vendor-native payload filter applied to every query. Parsing it
    /// into a backend filter object is the backend's job (see
    /// [`QdrantTarget`](crate::targets::qdrant)); a populated filter that a
    /// backend can't compile is an error, never silently dropped.
    #[serde(default)]
    pub filter: Option<serde_yaml::Value>,
}

/// Where the query vectors come from — a parquet at a local path or `s3://`
/// URI, read via DuckDB.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QuerySource {
    pub uri: String,
    pub column: String,
    /// How many query vectors to load and cycle through.
    #[serde(default = "default_limit")]
    pub limit: usize,
}

/// Per-worker load shape, replicated (NOT sharded) across the fleet.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoadProfile {
    /// Closed-loop: requests held in flight. Paced: in-flight ceiling.
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    /// How long to keep firing, in seconds.
    #[serde(default = "default_duration")]
    pub duration_s: f64,
    /// Reserved: stagger task starts. Not yet honoured (see runner TODO).
    #[serde(default)]
    pub ramp_s: f64,
    /// `0` (default) = closed-loop, measuring max throughput at `concurrency`.
    /// `>0` = open-loop paced at this many queries/sec per worker, with
    /// `concurrency` as the in-flight cap. The YAML key is `qps`.
    #[serde(rename = "qps", default)]
    pub target_qps: f64,
}

impl Default for LoadProfile {
    fn default() -> Self {
        Self {
            concurrency: default_concurrency(),
            duration_s: default_duration(),
            ramp_s: 0.0,
            target_qps: 0.0,
        }
    }
}

fn default_top_k() -> u64 {
    10
}
fn default_limit() -> usize {
    5000
}
fn default_concurrency() -> usize {
    32
}
fn default_duration() -> f64 {
    60.0
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse config: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("environment variable '{0}' is not set")]
    MissingEnv(String),
}

/// Read and parse a storm config from a file path.
pub fn load_config_file(path: &str) -> Result<StormConfig, ConfigError> {
    let yaml = std::fs::read_to_string(path)?;
    load_config_str(&yaml)
}

/// Parse a storm config from a YAML string, resolving `${VAR}` references
/// against the environment first (an unset var is an error).
pub fn load_config_str(yaml: &str) -> Result<StormConfig, ConfigError> {
    let mut value: serde_yaml::Value = serde_yaml::from_str(yaml)?;
    resolve_env_vars(&mut value)?;
    Ok(serde_yaml::from_value(value)?)
}

/// Recursively expand `${VAR}` in every string in the YAML tree.
fn resolve_env_vars(value: &mut serde_yaml::Value) -> Result<(), ConfigError> {
    match value {
        serde_yaml::Value::String(s) => {
            if let Some(expanded) = expand_env(s)? {
                *s = expanded;
            }
        }
        serde_yaml::Value::Sequence(seq) => {
            for v in seq.iter_mut() {
                resolve_env_vars(v)?;
            }
        }
        serde_yaml::Value::Mapping(map) => {
            for (_k, v) in map.iter_mut() {
                resolve_env_vars(v)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Expand `${VAR}` references. Returns `None` if the string has none.
fn expand_env(s: &str) -> Result<Option<String>, ConfigError> {
    if !s.contains("${") {
        return Ok(None);
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        match after.find('}') {
            Some(end) => {
                let name = &after[..end];
                let val = std::env::var(name)
                    .map_err(|_| ConfigError::MissingEnv(name.to_string()))?;
                out.push_str(&val);
                rest = &after[end + 1..];
            }
            None => {
                // unterminated `${` — leave it verbatim
                out.push_str("${");
                rest = after;
            }
        }
    }
    out.push_str(rest);
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_env_references() {
        // SAFETY: single-threaded test; no other thread reads the environment.
        unsafe {
            std::env::set_var("NOVA_STORM_TEST_VAR", "resolved");
        }
        assert_eq!(
            expand_env("a/${NOVA_STORM_TEST_VAR}/b").unwrap().as_deref(),
            Some("a/resolved/b")
        );
        assert!(expand_env("no refs here").unwrap().is_none());
    }

    #[test]
    fn missing_env_reference_errors() {
        let err = expand_env("${NOVA_STORM_DEFINITELY_UNSET}").unwrap_err();
        assert!(matches!(err, ConfigError::MissingEnv(_)));
    }

    #[cfg(feature = "qdrant")]
    #[test]
    fn parses_storm_config_and_qps_alias() {
        let yaml = r#"
target:
  type: qdrant
  url: http://localhost:6334
  collection_name: products
query:
  vector_name: dense
  top_k: 10
  source:
    uri: /tmp/queries.parquet
    column: embedding
    limit: 1000
load:
  concurrency: 8
  duration_s: 300
  qps: 75
"#;
        let cfg = load_config_str(yaml).expect("should parse");
        assert_eq!(cfg.query.top_k, 10);
        assert_eq!(cfg.query.source.limit, 1000);
        assert_eq!(cfg.load.concurrency, 8);
        assert_eq!(cfg.load.target_qps, 75.0); // `qps` -> target_qps
    }
}

/// Guards that the shipped example config stays valid against the schema.
#[cfg(all(test, feature = "qdrant"))]
mod shipped_configs {
    use super::load_config_file;

    #[test]
    fn storm_test_yaml_parses() {
        // SAFETY: single-threaded test; the YAML references these via ${...}.
        unsafe {
            std::env::set_var("QDRANT_URL", "http://localhost:6334");
            std::env::set_var("QDRANT_API_KEY", "test");
            std::env::set_var("SN_METRICS_DB_URL", "postgres://localhost/test");
        }
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../configs/storm/test.yaml");
        load_config_file(path).expect("configs/storm/test.yaml should parse");
    }
}
