use std::collections::HashMap;
use std::env;

use serde::Deserialize;

use crate::sources::DataSourceConfig;
use crate::stores::VectorStoreConfig;

// Per-vector spec types are part of the backend-agnostic contract; they live in
// the shared `nova-load-contract-rust` crate. Re-export here so existing
// `crate::config::{VectorSpec, VectorKind}` paths across this backend are
// unchanged.
pub use nova_load_contract_rust::{VectorKind, VectorSpec};

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
    /// Retry a file's download + read this many times (with exponential backoff)
    /// before giving up on it. 0 = try once, no retries.
    #[serde(default = "default_file_retries")]
    pub file_retries: usize,
    /// Retry a failed batch upsert this many times (exponential backoff) before
    /// aborting. Unlike a bad file (which is skipped), a persistent upsert failure
    /// usually means the store is down or misconfigured, so it aborts the run.
    #[serde(default = "default_upsert_retries")]
    pub upsert_retries: usize,
    /// Abort the whole run if more than this many files are skipped after
    /// exhausting their retries. `None` (default) skips every failing file and
    /// keeps going; set a ceiling to fail fast on a systemic problem instead of
    /// silently skipping the whole corpus.
    #[serde(default)]
    pub max_failed_files: Option<usize>,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: default_batch_size(),
            concurrency: default_concurrency(),
            file_look_ahead: default_file_look_ahead(),
            file_retries: default_file_retries(),
            upsert_retries: default_upsert_retries(),
            max_failed_files: None,
        }
    }
}

fn default_batch_size() -> usize {
    256
}

fn default_file_look_ahead() -> usize {
    2
}

fn default_file_retries() -> usize {
    5
}

fn default_upsert_retries() -> usize {
    5
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

// `VectorSpec` and `VectorKind` moved to the shared `nova-load-contract-rust`
// crate (re-exported at the top of this module). They are part of the
// backend-agnostic contract, not qdrant-specific config.

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

/// Quantization parameters — one collection-wide method, mirroring Qdrant's
/// own `scalar`/`product`/`binary`/`turbo` split, plus `none`. Fields not
/// relevant to the chosen `type` are simply ignored (not rejected), so
/// switching `type` doesn't require deleting the other methods' fields first.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationConfig {
    /// Quantization method: `scalar` (the default when unset), `product`,
    /// `binary`, `turbo`, or `none`. `none` means "no quantization" — a no-op
    /// at collection creation (same as omitting `quantization:` entirely),
    /// but on `nova load reindex` it's the way to explicitly *clear*
    /// quantization off a collection that already has it (omitting
    /// `quantization:` on a reindex leaves the server's current setting
    /// untouched, since reindex only patches knobs it's told about).
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    /// Scalar only: fraction of the range to keep, e.g. `0.99` clips the
    /// top/bottom 1% of outliers before mapping to int8.
    #[serde(default)]
    pub quantile: Option<f32>,
    /// Product only: compression ratio (`x4`, `x8`, `x16`, `x32`, `x64`) —
    /// higher shrinks the index further at the cost of recall. Defaults to
    /// `x16` (Qdrant's own default) when unset.
    #[serde(default)]
    pub compression: Option<String>,
    /// Binary only: bit encoding (`one_bit`, the default; `two_bits` and
    /// `one_and_half_bits` trade back some of binary's compression for
    /// accuracy).
    #[serde(default)]
    pub encoding: Option<String>,
    /// Turbo only: bits per dimension — one of `1`, `1.5`, `2`, `4`. Leave
    /// unset for the server's own default.
    #[serde(default)]
    pub bits: Option<f32>,
    /// All methods: keep quantized vectors in RAM regardless of the main
    /// storage config.
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
