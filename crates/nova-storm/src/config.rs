//! Parsing for the storm `.yaml`.
//!
//! Four top-level keys: `target` (the cluster under test, dispatched on its
//! `type`), `query` (what to search with + where the query vectors live),
//! `load` (the per-worker profile), and optional `report` (per-dispatch
//! time-series output — see [`crate::report::ReportConfig`]).
//!
//! Mirrors `nova-load`'s config paradigm: `${VAR}` references are expanded from
//! the environment before deserializing (see [`expand_env`]).

use std::env;

use serde::Deserialize;

use crate::filter::Filter;
use crate::report::ReportConfig;
use crate::targets::TargetConfig;

/// The full parsed storm config.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StormConfig {
    pub target: TargetConfig,
    pub query: QueryConfig,
    #[serde(default)]
    pub load: LoadProfile,
    /// Per-dispatch time-series output. Absent (default) = summary only.
    #[serde(default)]
    pub report: Option<ReportConfig>,
}

impl StormConfig {
    /// Parse from YAML text, expanding `${VAR}` references first.
    pub fn from_yaml(yaml: &str) -> Result<Self, ConfigError> {
        let expanded = expand_env(yaml)?;
        let mut cfg: Self = serde_yaml::from_str(&expanded)?;
        // Normalize the vector name once: trim padding (` sparse ` would pass
        // validation but fail at dispatch with the untrimmed name) and treat a
        // blank as absent for BOTH modalities.
        cfg.query.vector_name = cfg
            .query
            .vector_name
            .take()
            .map(|n| n.trim().to_string())
            .filter(|n| !n.is_empty());
        // A `.limit(0)` query returns nothing and, if `ground_truth_column` is
        // set, divides recall by 0 (NaN) — reject at config time rather than
        // silently corrupting the summary.
        if cfg.query.top_k == 0 {
            return Err(ConfigError::ZeroTopK);
        }
        if cfg.load.batch_size == 0 {
            return Err(ConfigError::ZeroBatchSize);
        }
        // Sparse vectors are named in every backend that has them, so a sparse
        // query with no `vector_name` (blank normalized to None above) can
        // only ever fail at dispatch time -- reject it here where the fix is
        // obvious.
        if cfg.query.vector_type == VectorType::Sparse && cfg.query.vector_name.is_none() {
            return Err(ConfigError::SparseRequiresVectorName);
        }
        // Only the qdrant target speaks sparse. Rejected here rather than
        // per-dispatch: the per-dispatch guards fail in MICROSECONDS (no
        // network round-trip), so a sparse config against a dense-only target
        // would otherwise spin every worker flat-out for the whole duration,
        // accumulate millions of ~0ms latency samples, and still exit 0.
        if cfg.query.vector_type == VectorType::Sparse
            && !matches!(cfg.target, crate::targets::TargetConfig::Qdrant(_))
        {
            return Err(ConfigError::SparseTargetUnsupported);
        }
        if let Some(filter) = &cfg.query.filter {
            filter.validate()?;
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

/// Whether the query column holds dense vectors (a `list<float>`) or sparse
/// ones (a `struct{indices: list<int>, values: list<float>}` — the shape
/// `nova embed`'s sparse output uses.  Mirrors `nova bf`'s per-set `vector_type` vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorType {
    #[default]
    Dense,
    Sparse,
}

/// What to query with: the named vector, how many neighbours, and where the
/// query vectors are read from.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryConfig {
    /// Named vector to search (`None` for a single-vector collection).
    #[serde(default)]
    pub vector_name: Option<String>,
    /// Dense (default) or sparse queries — see [`VectorType`]. Sparse always
    /// requires `vector_name` (sparse vectors are named in every backend that
    /// has them), and only the qdrant target supports it today.
    #[serde(default)]
    pub vector_type: VectorType,
    #[serde(default = "default_top_k")]
    pub top_k: u64,
    /// Ask the server to return each hit's full payload. Default `false`
    /// (ids/scores only). Turn it on when the production traffic being
    /// modeled fetches payloads — payload retrieval is real server-side work
    /// (reads payload storage, possibly from disk, for every hit) and real
    /// response bytes, so benchmarking without it understates latency for
    /// such workloads.
    #[serde(default)]
    pub with_payload: bool,
    pub source: QuerySource,
    /// Server-side search-time tuning, **interpreted per backend**. It's a raw
    /// value here so each target validates it against its own schema and rejects
    /// unsupported keys (e.g. Qdrant `{hnsw_ef, exact, quantization}`, Milvus
    /// `{ef, nprobe}`, Elastic `{num_candidates}`). `None` (default) leaves the
    /// backend's own search defaults in place. Distinct from `load`'s
    /// client-side knobs (concurrency/batch_size/rps).
    #[serde(default)]
    pub search_params: Option<serde_yaml::Value>,
    /// Payload/metadata filter applied to every query in the run — see
    /// [`crate::filter::Filter`]. `None` (default) is an unfiltered search.
    /// (Only the Qdrant target supports filters today; milvus/elastic reject one.)
    #[serde(default)]
    pub filter: Option<Filter>,
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
    /// `0` (default) = timed run: fire for `duration_s`, queries cycling
    /// round-robin. `>0` = FIXED-WORK run: fire every loaded query exactly
    /// this many times, then stop — `duration_s` is ignored. Fixed work makes
    /// run length data-dependent but the measurement composition exact: each
    /// query contributes equally to recall and latency, so the mean recall is
    /// the true mean over the query set (directly comparable to a brute-force
    /// ground-truth sweep), not a mean over whichever firings a timer allowed.
    #[serde(default)]
    pub passes: usize,
}

impl Default for LoadProfile {
    fn default() -> Self {
        Self {
            concurrency: default_concurrency(),
            duration_s: default_duration(),
            target_rps: 0.0,
            batch_size: default_batch_size(),
            passes: 0,
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
    #[error(
        "query.vector_type is `sparse` but query.vector_name is not set — sparse vectors are \
         always named; set vector_name to the collection's sparse vector (e.g. `sparse`)"
    )]
    SparseRequiresVectorName,
    #[error(
        "query.vector_type is `sparse` but the target does not support sparse queries — only \
         the qdrant target speaks sparse today"
    )]
    SparseTargetUnsupported,
    #[error(
        "filter condition on `{field}` must set exactly one of `match`, `range`, `match_text`, \
         `match_from_query`, `range_from_query`, or `match_text_from_query`"
    )]
    FilterConditionNotExactlyOne { field: String },
    #[error("filter condition on `{field}` has a blank `match_text` — it needs at least one word")]
    FilterConditionBlankMatchText { field: String },
    #[error("filter condition on `{field}` range needs at least one of gt/gte/lt/lte")]
    FilterConditionEmptyRange { field: String },
    #[error("filter condition on `{field}` has an empty `match: []` list — it would never match anything")]
    FilterConditionEmptyMatchAny { field: String },
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
        assert!(!cfg.query.with_payload); // default: ids/scores only
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
    fn parses_with_payload() {
        let yaml = r#"
target:
  type: qdrant
  url: http://localhost:6334
  collection_name: c
query:
  with_payload: true
  source:
    uri: /tmp/q.parquet
    column: embedding
"#;
        let cfg = StormConfig::from_yaml(yaml).expect("parses");
        assert!(cfg.query.with_payload);
    }

    #[test]
    fn parses_report_section_and_defaults_to_none() {
        let base = r#"
target:
  type: qdrant
  url: http://localhost:6334
  collection_name: c
query:
  source:
    uri: /tmp/q.parquet
    column: embedding
"#;
        // absent -> None: summary-only, exactly the pre-report behavior
        let cfg = StormConfig::from_yaml(base).expect("parses");
        assert!(cfg.report.is_none());

        let with_report = format!("{base}report:\n  format: csv\n  path: /tmp/ts.csv\n");
        let cfg = StormConfig::from_yaml(&with_report).expect("parses");
        let report = cfg.report.expect("present");
        assert_eq!(report.format, crate::report::ReportFormat::Csv);
        assert_eq!(report.path, "/tmp/ts.csv");

        // unknown format dies at parse time like any other config typo
        let bad = format!("{base}report:\n  format: sqlite\n  path: /tmp/ts.db\n");
        assert!(matches!(StormConfig::from_yaml(&bad).unwrap_err(), ConfigError::Yaml(_)));
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

    /// A minimal valid config with `extra` spliced into the `query` section.
    fn yaml_with_query_extras(extra: &str) -> String {
        format!(
            "target:\n  type: qdrant\n  url: http://localhost:6334\n  collection_name: c\n\
             query:\n{extra}  source:\n    uri: /tmp/q.parquet\n    column: embedding\n"
        )
    }

    #[test]
    fn vector_type_defaults_to_dense() {
        let cfg = StormConfig::from_yaml(&yaml_with_query_extras("")).expect("parses");
        assert_eq!(cfg.query.vector_type, VectorType::Dense);
    }

    #[test]
    fn parses_sparse_vector_type_with_name() {
        let cfg = StormConfig::from_yaml(&yaml_with_query_extras(
            "  vector_name: sparse\n  vector_type: sparse\n",
        ))
        .expect("parses");
        assert_eq!(cfg.query.vector_type, VectorType::Sparse);
        assert_eq!(cfg.query.vector_name.as_deref(), Some("sparse"));
    }

    #[test]
    fn sparse_without_vector_name_is_rejected() {
        let err = StormConfig::from_yaml(&yaml_with_query_extras("  vector_type: sparse\n"))
            .unwrap_err();
        assert!(matches!(err, ConfigError::SparseRequiresVectorName));
    }

    #[test]
    fn blank_vector_name_is_rejected_for_sparse() {
        let err = StormConfig::from_yaml(&yaml_with_query_extras(
            "  vector_name: \"  \"\n  vector_type: sparse\n",
        ))
        .unwrap_err();
        assert!(matches!(err, ConfigError::SparseRequiresVectorName));
    }

    // Sparse against a dense-only target is rejected at CONFIG time (a
    // per-dispatch guard fails in microseconds, so it would otherwise spin a
    // full-duration ~0ms error loop and still exit 0). The elastic/milvus
    // variants only exist under their features, so the negative case is
    // feature-gated; the qdrant-passes case runs always.
    #[cfg(feature = "elastic")]
    #[test]
    fn sparse_against_a_dense_only_target_is_rejected() {
        let yaml = "target:\n  type: elastic\n  url: http://localhost:9200\n  index_name: c\n\
             query:\n  vector_name: sparse\n  vector_type: sparse\n  source:\n    uri: /tmp/q.parquet\n    column: e\n";
        let err = StormConfig::from_yaml(yaml).unwrap_err();
        assert!(matches!(err, ConfigError::SparseTargetUnsupported));
    }

    #[test]
    fn vector_name_is_trimmed_and_blank_means_absent() {
        let cfg = StormConfig::from_yaml(&yaml_with_query_extras(
            "  vector_name: \" sparse \"\n  vector_type: sparse\n",
        ))
        .expect("parses");
        assert_eq!(cfg.query.vector_name.as_deref(), Some("sparse")); // padding gone
        let cfg = StormConfig::from_yaml(&yaml_with_query_extras("  vector_name: \"  \"\n"))
            .expect("parses");
        assert_eq!(cfg.query.vector_name, None); // blank dense name -> absent
    }

    #[test]
    fn unknown_vector_type_is_rejected() {
        let err = StormConfig::from_yaml(&yaml_with_query_extras("  vector_type: hybrid\n"))
            .unwrap_err();
        assert!(matches!(err, ConfigError::Yaml(_)));
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
