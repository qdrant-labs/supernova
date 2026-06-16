//! Parsing for the load `.yaml`.
//!
//! Top level has four sibling keys: `vectors` (per-vector specs shared by the
//! reader and the store), `datasource`, `vectorstore`, and `loader`. The
//! `datasource` and `vectorstore` blocks each dispatch on a `type` field to a
//! backend-specific variant; the store backends own their config structs in
//! their own modules (see [`crate::stores`]).

use std::collections::HashMap;

use serde::Deserialize;

use crate::stores::VectorStoreConfig;
use crate::sources::DataSourceConfig;

/// The full parsed load config. This is the top-level struct deserialized from the YAML; it references the backend-specific configs in the `stores` and `sources` modules.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadConfig {
    /// Per-vector specs keyed by vector name. Shared by the reader (which
    /// column to read) and the store (how to create the collection).
    pub vectors: HashMap<String, VectorSpec>,
    pub datasource: DataSourceConfig,
    pub vectorstore: VectorStoreConfig,
    #[serde(default)]
    pub loader: LoaderConfig,
    /// Where load metrics go. Absent → stdout (local-first, no setup).
    #[serde(default)]
    pub metrics: Option<nova_metrics::MetricsConfig>,
}

/// One named vector's spec.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorSpec {
    #[serde(rename = "type")]
    pub kind: VectorKind,
    /// Parquet column the reader pulls this vector from.
    pub column: String,
    /// Read by the store at collection-creation time; ignored by the reader.
    #[serde(default)]
    pub distance: Option<String>,
    #[serde(default)]
    pub comparator: Option<String>,
    #[serde(default)]
    pub datatype: Option<String>,
    #[serde(default)]
    pub on_disk: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorKind {
    Dense,
    Sparse,
    Multivector,
}

/// A vector's full creation spec: its static config ([`VectorSpec`]) plus the
/// `size` discovered from the data at read time. The runner zips config specs
/// with reader-reported dimensions into one of these per vector, so a store's
/// [`ensure_collection`](crate::stores::VectorStore::ensure_collection) gets a
/// complete description and never has to merge two maps itself.
#[derive(Debug, Clone)]
pub struct ResolvedVector {
    pub kind: VectorKind,
    /// Dimensionality; `None` for sparse vectors (no fixed size).
    pub size: Option<usize>,
    pub distance: Option<String>,
    pub comparator: Option<String>,
    pub datatype: Option<String>,
    pub on_disk: Option<bool>,
}

/// The fully-resolved set of vectors to create, keyed by vector name.
pub type CollectionSchema = HashMap<String, ResolvedVector>;

/// Zip static [`VectorSpec`]s with reader-discovered dimensions into a
/// [`CollectionSchema`]. Sparse vectors carry `size: None`.
pub fn resolve_schema(
    vectors: &HashMap<String, VectorSpec>,
    dimensions: &crate::DimensionsMap,
) -> CollectionSchema {
    vectors
        .iter()
        .map(|(name, spec)| {
            (
                name.clone(),
                ResolvedVector {
                    kind: spec.kind,
                    size: dimensions.get(name).copied(),
                    distance: spec.distance.clone(),
                    comparator: spec.comparator.clone(),
                    datatype: spec.datatype.clone(),
                    on_disk: spec.on_disk,
                },
            )
        })
        .collect()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoaderConfig {
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    #[serde(default)]
    pub prefetch_size: Option<usize>,
    /// Target writes (points) per second per worker; `None` = unbounded.
    #[serde(default)]
    pub wps: Option<f64>,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: default_batch_size(),
            concurrency: default_concurrency(),
            prefetch_size: None,
            wps: None,
        }
    }
}

fn default_batch_size() -> usize {
    1000
}
fn default_concurrency() -> usize {
    8
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

/// Read and parse a load config from a file path.
pub fn load_config_file(path: &str) -> Result<LoadConfig, ConfigError> {
    let yaml = std::fs::read_to_string(path)?;
    load_config_str(&yaml)
}

/// Like [`load_config_file`], but also returns the resolved config as JSON for
/// the metrics `runs.config` blob. Built from the env-expanded YAML value (not
/// a re-serialization of the typed struct), so it captures the config verbatim
/// without needing `Serialize` on every backend struct. Secrets are redacted by
/// the sink before this is persisted.
pub fn load_config_file_with_json(
    path: &str,
) -> Result<(LoadConfig, serde_json::Value), ConfigError> {
    let yaml = std::fs::read_to_string(path)?;
    let mut value: serde_yaml::Value = serde_yaml::from_str(&yaml)?;
    resolve_env_vars(&mut value)?;
    let json = serde_json::to_value(&value).unwrap_or(serde_json::Value::Null);
    let cfg = serde_yaml::from_value(value)?;
    Ok((cfg, json))
}

/// Parse a load config from a YAML string, resolving `${VAR}` references
/// against the environment first (an unset var is an error).
pub fn load_config_str(yaml: &str) -> Result<LoadConfig, ConfigError> {
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

/// Expand `${VAR}` references. Returns `None` if the string has none (so the
/// caller can skip the allocation).
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
            std::env::set_var("NOVA_TEST_VAR", "resolved");
        }
        assert_eq!(
            expand_env("a/${NOVA_TEST_VAR}/b").unwrap().as_deref(),
            Some("a/resolved/b")
        );
        assert!(expand_env("no refs here").unwrap().is_none());
    }

    #[test]
    fn missing_env_reference_errors() {
        let err = expand_env("${NOVA_DEFINITELY_UNSET_VAR}").unwrap_err();
        assert!(matches!(err, ConfigError::MissingEnv(_)));
    }
}

/// Guards that the shipped example config stays valid against the schema.
#[cfg(all(test, feature = "s3", feature = "qdrant"))]
mod shipped_configs {
    use super::load_config_file;

    #[test]
    fn loader_test_yaml_parses() {
        // SAFETY: single-threaded test; the YAML references these via ${...}.
        unsafe {
            std::env::set_var("QDRANT_URL", "http://localhost:6334");
            std::env::set_var("QDRANT_API_KEY", "test");
        }
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../configs/loader/test.yaml");
        load_config_file(path).expect("configs/loader/test.yaml should parse");
    }
}
