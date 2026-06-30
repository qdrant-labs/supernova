use std::collections::HashMap;
use std::env;

use serde::Deserialize;

use crate::sources::DataSourceConfig;
use crate::stores::VectorStoreConfig;

/// The full parsed load config. This is the top-level struct deserialized from the YAML; it references the backend-specific configs in the `stores` and `sources` modules.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadConfig { 
    pub datasource: DataSourceConfig,
    pub vectorstore: VectorStoreConfig,
    pub vectors: HashMap<String, VectorSpec>,
    #[serde(default)]
    pub loader: LoaderConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoaderConfig {
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    /// How many files to download + read *ahead* while the current file's
    /// batches are still being upserted, so the store connection stays busy
    /// during S3/DuckDB time. 1 = no prefetch.
    #[serde(default = "default_file_look_ahead")]
    pub file_look_ahead: usize,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: default_batch_size(),
            concurrency: default_concurrency(),
            file_look_ahead: default_file_look_ahead(),
        }
    }
}

fn default_batch_size() -> usize {
    256
}

fn default_file_look_ahead() -> usize {
    2
}

impl LoadConfig {
    /// Parse a config from YAML text, expanding `${VAR}` environment-variable
    /// references first (see [`expand_env`]).
    pub fn from_yaml(yaml: &str) -> Result<Self, ConfigError> {
        let expanded = expand_env(yaml)?;
        Ok(serde_yaml::from_str(&expanded)?)
    }

    /// Read and parse a config file, expanding `${VAR}` references.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let yaml = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_yaml(&yaml)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file `{path}`: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("config YAML is invalid: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error(
        "environment variable `{0}` referenced in config is not set; set it or supply a default with `${{{0}:-...}}`"
    )]
    MissingEnvVar(String),
    #[error("unterminated `${{` placeholder in config (missing closing `}}`)")]
    UnterminatedPlaceholder,
}

/// Expand `${VAR}` references in `input` from the process environment.
///
/// Supported syntax:
///   - `${VAR}`           — replaced with `$VAR`; errors if unset or empty.
///   - `${VAR:-default}`  — replaced with `$VAR`, or `default` if unset/empty.
///   - `$$`               — an escaped literal `$` (not treated as a reference).
///
/// A bare `$` not followed by `{` or `$` is left untouched.
pub fn expand_env(input: &str) -> Result<String, ConfigError> {
    expand_env_with(input, |key| env::var(key).ok())
}

/// [`expand_env`] with an injectable variable lookup, so tests don't have to
/// mutate the real process environment.
fn expand_env_with(
    input: &str,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<String, ConfigError> {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;

    while let Some(pos) = rest.find('$') {
        out.push_str(&rest[..pos]);
        let after = &rest[pos + 1..];

        if let Some(tail) = after.strip_prefix('$') {
            // `$$` -> literal `$`.
            out.push('$');
            rest = tail;
        } else if let Some(tail) = after.strip_prefix('{') {
            let end = tail.find('}').ok_or(ConfigError::UnterminatedPlaceholder)?;
            let (var, default) = match tail[..end].split_once(":-") {
                Some((var, default)) => (var, Some(default)),
                None => (&tail[..end], None),
            };
            let value = lookup(var)
                .filter(|s| !s.is_empty())
                .or_else(|| default.map(str::to_string))
                .ok_or_else(|| ConfigError::MissingEnvVar(var.to_string()))?;
            out.push_str(&value);
            rest = &tail[end + 1..];
        } else {
            // Bare `$` — leave it as-is.
            out.push('$');
            rest = after;
        }
    }
    out.push_str(rest);
    Ok(out)
}

/// One named vector's spec. The scalar knobs (distance, datatype, comparator,
/// modifier) are strings interpreted by the store. HNSW/quantization tuning is
/// collection-wide (see the store params), not per-vector.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorSpec {
    #[serde(rename = "type")]
    pub kind: VectorKind,
    /// Parquet column the reader pulls this vector from.
    pub column: String,
    /// Dense vector dimensionality. Optional: when omitted the loader infers it
    /// from the parquet schema (the column is a fixed-size list). Ignored for
    /// sparse vectors, which have no fixed size. Read by the store at
    /// collection-creation time; ignored by the reader.
    #[serde(default)]
    pub size: Option<u64>,
    /// Read by the store at collection-creation time; ignored by the reader.
    #[serde(default)]
    pub distance: Option<String>,
    /// Multivector comparator (e.g. `max_sim`); only meaningful for `multivector`.
    #[serde(default)]
    pub comparator: Option<String>,
    #[serde(default)]
    pub datatype: Option<String>,
    #[serde(default)]
    pub on_disk: Option<bool>,
    /// Sparse re-weighting modifier (e.g. `idf`); only meaningful for `sparse`.
    #[serde(default)]
    pub modifier: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorKind {
    Dense,
    Sparse,
    Multivector,
}

/// Collection-wide HNSW index parameters. All fields optional; the store
/// applies its own defaults for any left unset.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HnswConfig {
    #[serde(default)]
    pub m: Option<u64>,
    #[serde(default)]
    pub ef_construct: Option<u64>,
    #[serde(default)]
    pub full_scan_threshold: Option<u64>,
    #[serde(default)]
    pub max_indexing_threads: Option<u64>,
    #[serde(default)]
    pub on_disk: Option<bool>,
    #[serde(default)]
    pub payload_m: Option<u64>,
}

/// Scalar quantization parameters.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationConfig {
    /// Quantization type; currently only `int8` is supported. Defaults to `int8`.
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub quantile: Option<f32>,
    #[serde(default)]
    pub always_ram: Option<bool>,
}

/// Optimizer parameters (collection-wide only). Passed straight through to
/// Qdrant's `OptimizersConfigDiff`. The size-based knobs (`max_segment_size`,
/// `memmap_threshold`, `indexing_threshold`) are in KILOBYTES, matching the
/// collection API — each also accepts the `*_kb` spelling Qdrant's server
/// `config.yaml` uses, as an alias, so either name works.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizersConfig {
    #[serde(default)]
    pub deleted_threshold: Option<f64>,
    #[serde(default)]
    pub vacuum_min_vector_number: Option<u64>,
    #[serde(default)]
    pub default_segment_number: Option<u64>,
    #[serde(default, alias = "max_segment_size_kb")]
    pub max_segment_size: Option<u64>,
    #[serde(default, alias = "memmap_threshold_kb")]
    pub memmap_threshold: Option<u64>,
    #[serde(default, alias = "indexing_threshold_kb")]
    pub indexing_threshold: Option<u64>,
    #[serde(default)]
    pub flush_interval_sec: Option<u64>,
}

// Should utilize as many CPUs - 1 without going over ~4
fn default_concurrency() -> usize {
    std::thread::available_parallelism().map(|n| n.get().saturating_sub(1)).unwrap_or(1)    
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand(input: &str, vars: &[(&str, &str)]) -> Result<String, ConfigError> {
        let map: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        expand_env_with(input, |k| map.get(k).cloned())
    }

    #[test]
    fn substitutes_set_vars() {
        let out = expand(
            "api_key: ${QDRANT_API_KEY}",
            &[("QDRANT_API_KEY", "secret")],
        )
        .unwrap();
        assert_eq!(out, "api_key: secret");
    }

    #[test]
    fn multiple_and_adjacent() {
        let out = expand(
            "${A}://${B}:${C}",
            &[("A", "http"), ("B", "host"), ("C", "6334")],
        )
        .unwrap();
        assert_eq!(out, "http://host:6334");
    }

    #[test]
    fn missing_var_without_default_errors() {
        let err = expand("x: ${NOPE}", &[]).unwrap_err();
        assert!(matches!(err, ConfigError::MissingEnvVar(v) if v == "NOPE"));
    }

    #[test]
    fn default_used_when_unset_or_empty() {
        assert_eq!(expand("x: ${NOPE:-fallback}", &[]).unwrap(), "x: fallback");
        assert_eq!(
            expand("x: ${EMPTY:-fallback}", &[("EMPTY", "")]).unwrap(),
            "x: fallback"
        );
        assert_eq!(
            expand("x: ${SET:-fallback}", &[("SET", "real")]).unwrap(),
            "x: real"
        );
    }

    #[test]
    fn escaped_dollar_and_bare_dollar() {
        assert_eq!(expand("price: $$5", &[]).unwrap(), "price: $5");
        assert_eq!(expand("cost: 5$ each", &[]).unwrap(), "cost: 5$ each");
    }

    #[test]
    fn unterminated_placeholder_errors() {
        assert!(matches!(
            expand("x: ${UNCLOSED", &[]).unwrap_err(),
            ConfigError::UnterminatedPlaceholder
        ));
    }
}
