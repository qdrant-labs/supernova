//! Parsing for the storm `.yaml`.
//!
//! Three top-level keys: `target` (the cluster under test, dispatched on its
//! `type`), `query` (what to search with + where the query vectors live), and
//! `load` (the per-worker profile).
//!
//! Mirrors `nova-load`'s config paradigm: `${VAR}` references are expanded from
//! the environment before deserializing (see [`expand_env`]).

use std::env;

use serde::Deserialize;

use crate::targets::TargetConfig;

/// The full parsed storm config.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StormConfig {
    pub target: TargetConfig,
    pub query: QueryConfig,
    #[serde(default)]
    pub load: LoadProfile,
}

impl StormConfig {
    /// Parse from YAML text, expanding `${VAR}` references first.
    pub fn from_yaml(yaml: &str) -> Result<Self, ConfigError> {
        let expanded = expand_env(yaml)?;
        let cfg: Self = serde_yaml::from_str(&expanded)?;
        // A `.limit(0)` query returns nothing and, if `ground_truth_column` is
        // set, divides recall by 0 (NaN) — reject at config time rather than
        // silently corrupting the summary.
        if cfg.query.top_k == 0 {
            return Err(ConfigError::ZeroTopK);
        }
        if cfg.load.batch_size == 0 {
            return Err(ConfigError::ZeroBatchSize);
        }
        Ok(cfg)
    }

    /// Read and parse a config file, expanding `${VAR}` references.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let yaml = std::fs::read_to_string(path)
            .map_err(|source| ConfigError::Read { path: path.display().to_string(), source })?;
        Self::from_yaml(&yaml)
    }
}

/// What to query with: the named vector, how many neighbours, and where the
/// query vectors are read from.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryConfig {
    /// Named vector to search (`None` for a single-vector collection).
    #[serde(default)]
    pub vector_name: Option<String>,
    #[serde(default = "default_top_k")]
    pub top_k: u64,
    pub source: QuerySource,
    /// Server-side search-time tuning (Qdrant's `SearchParams`) — distinct from
    /// `load`'s client-side knobs (concurrency/batch_size/rps). `None` (default)
    /// leaves every one of these at the collection's own defaults.
    #[serde(default)]
    pub search_params: Option<SearchParamsConfig>,
}

/// Server-side search-time tuning, passed straight through to Qdrant's
/// `SearchParams` on every query. All fields optional; the server applies its
/// own defaults for any left unset — same convention as `HnswConfig`/
/// `QuantizationConfig` in `nova-load`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchParamsConfig {
    /// HNSW beam-search width at query time. Higher = more accurate, slower.
    #[serde(default)]
    pub hnsw_ef: Option<u64>,
    /// Search without approximation (brute-force exact search for this query).
    #[serde(default)]
    pub exact: Option<bool>,
    #[serde(default)]
    pub quantization: Option<QuantizationSearchParamsConfig>,
}

/// Quantization behavior at query time (distinct from `nova-load`'s
/// `QuantizationConfig`, which controls how vectors are quantized at index
/// time — this controls how a query uses that quantized index).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationSearchParamsConfig {
    /// Skip the quantized index entirely for this query.
    #[serde(default)]
    pub ignore: Option<bool>,
    /// Re-score quantized top-k candidates against the original vectors.
    #[serde(default)]
    pub rescore: Option<bool>,
    /// How many extra candidates to preselect via the quantized index before
    /// rescoring (e.g. `2.4` with `top_k: 100` preselects 240, then rescores
    /// down to 100).
    #[serde(default)]
    pub oversampling: Option<f64>,
}

/// Where the query vectors come from — a parquet at a local path or `s3://`
/// URI, read via DuckDB.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuerySource {
    pub uri: String,
    pub column: String,
    /// How many query vectors to load and cycle through.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// A `list<string>` column in the same file holding each query's known-correct
    /// top-k point ids (e.g. `nova bf`'s own `hit_ids` output, reused directly —
    /// no separate ground-truth file or id-matching needed since it's read
    /// alongside the vector in the same row). `None` (default) → no recall
    /// tracking; a null value for a given row → that row just has no ground
    /// truth (not an error), so it still contributes latency but no recall.
    #[serde(default)]
    pub ground_truth_column: Option<String>,
}

/// Per-worker load shape, replicated (NOT sharded) across the fleet.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadProfile {
    /// Closed-loop: requests held in flight. Paced: in-flight ceiling.
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    /// How long to keep firing, in seconds.
    #[serde(default = "default_duration")]
    pub duration_s: f64,
    /// `0` (default) = closed-loop, measuring max throughput at `concurrency`.
    /// `>0` = open-loop paced at this many *batch dispatches*/sec per worker,
    /// with `concurrency` as the in-flight cap. The YAML key is `rps`.
    #[serde(rename = "rps", default)]
    pub target_rps: f64,
    /// How many query vectors go in a single dispatch (`query_batch` RPC).
    /// `1` (default) is not a special case — every dispatch is a batch, just
    /// of size 1 by default, so existing configs behave identically.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

impl Default for LoadProfile {
    fn default() -> Self {
        Self {
            concurrency: default_concurrency(),
            duration_s: default_duration(),
            target_rps: 0.0,
            batch_size: default_batch_size(),
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
fn default_batch_size() -> usize {
    1
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file `{path}`: {source}")]
    Read { path: String, source: std::io::Error },
    #[error("config YAML is invalid: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("environment variable `{0}` referenced in config is not set; set it or supply a default with `${{{0}:-...}}`")]
    MissingEnvVar(String),
    #[error("unterminated `${{` placeholder in config (missing closing `}}`)")]
    UnterminatedPlaceholder,
    #[error("query.top_k must be greater than 0")]
    ZeroTopK,
    #[error("load.batch_size must be greater than 0")]
    ZeroBatchSize,
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
            out.push('$'); // `$$` -> literal `$`.
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
            out.push('$'); // bare `$` — leave as-is.
            rest = after;
        }
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_storm_config_and_rps() {
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
  rps: 75
  batch_size: 16
"#;
        let cfg = StormConfig::from_yaml(yaml).expect("should parse");
        assert_eq!(cfg.query.top_k, 10);
        assert_eq!(cfg.query.source.limit, 1000);
        assert_eq!(cfg.load.concurrency, 8);
        assert_eq!(cfg.load.target_rps, 75.0); // `rps` -> target_rps
        assert_eq!(cfg.load.batch_size, 16);
    }

    #[test]
    fn load_profile_defaults_when_absent() {
        let yaml = r#"
target:
  type: qdrant
  url: http://localhost:6334
  collection_name: c
query:
  source:
    uri: /tmp/q.parquet
    column: embedding
"#;
        let cfg = StormConfig::from_yaml(yaml).expect("parses");
        assert_eq!(cfg.load.concurrency, 32);
        assert_eq!(cfg.load.target_rps, 0.0);
        assert_eq!(cfg.load.batch_size, 1);
        assert_eq!(cfg.query.top_k, 10);
        assert_eq!(cfg.query.source.ground_truth_column, None);
    }

    #[test]
    fn rejects_zero_top_k() {
        let yaml = r#"
target:
  type: qdrant
  url: http://localhost:6334
  collection_name: c
query:
  top_k: 0
  source:
    uri: /tmp/q.parquet
    column: embedding
"#;
        assert!(matches!(StormConfig::from_yaml(yaml).unwrap_err(), ConfigError::ZeroTopK));
    }

    #[test]
    fn rejects_zero_batch_size() {
        let yaml = r#"
target:
  type: qdrant
  url: http://localhost:6334
  collection_name: c
query:
  source:
    uri: /tmp/q.parquet
    column: embedding
load:
  batch_size: 0
"#;
        assert!(matches!(StormConfig::from_yaml(yaml).unwrap_err(), ConfigError::ZeroBatchSize));
    }

    #[test]
    fn qps_key_is_no_longer_accepted() {
        let yaml = r#"
target:
  type: qdrant
  url: http://localhost:6334
  collection_name: c
query:
  source:
    uri: /tmp/q.parquet
    column: embedding
load:
  qps: 75
"#;
        // `qps` was replaced by `rps` with no back-compat alias -- an old config
        // using it now hits `deny_unknown_fields` like any other typo'd key.
        assert!(matches!(StormConfig::from_yaml(yaml).unwrap_err(), ConfigError::Yaml(_)));
    }

    #[test]
    fn parses_ground_truth_column() {
        let yaml = r#"
target:
  type: qdrant
  url: http://localhost:6334
  collection_name: c
query:
  source:
    uri: /tmp/q.parquet
    column: embedding
    ground_truth_column: hit_ids
"#;
        let cfg = StormConfig::from_yaml(yaml).expect("parses");
        assert_eq!(cfg.query.source.ground_truth_column.as_deref(), Some("hit_ids"));
    }

    fn expand(input: &str, vars: &[(&str, &str)]) -> Result<String, ConfigError> {
        let map: std::collections::HashMap<String, String> =
            vars.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        expand_env_with(input, |k| map.get(k).cloned())
    }

    #[test]
    fn expands_and_defaults() {
        assert_eq!(expand("url: ${U}", &[("U", "x")]).unwrap(), "url: x");
        assert_eq!(expand("url: ${U:-fallback}", &[]).unwrap(), "url: fallback");
        assert!(matches!(expand("${NOPE}", &[]).unwrap_err(), ConfigError::MissingEnvVar(_)));
    }
}
