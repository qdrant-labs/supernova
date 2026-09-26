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
        // A negative or non-finite tolerance makes `scores_tied` false for
        // EVERY pair, including exactly-equal scores: ties would silently
        // vanish and every above-cutoff result would land in `missing_from_gt`.
        if let Some(eps) = cfg.query.tie_epsilon {
            if !eps.is_finite() || eps < 0.0 {
                return Err(ConfigError::BadTieEpsilon(eps));
            }
        }
        // p outside (0, 1) breaks the geometric weighting RBO is defined by:
        // at 0 only depth 1 counts, at 1 the weights never decay (and the
        // series does not converge), and outside the range they go negative or
        // diverge — in every case the reported value is not RBO. Only a
        // CONFIGURED value can be wrong; the derived default is in range by
        // construction.
        if let Some(p) = cfg
            .query
            .rbo_p
            .filter(|p| !p.is_finite() || *p <= 0.0 || *p >= 1.0)
        {
            return Err(ConfigError::BadRboP(p));
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
        let yaml = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_yaml(&yaml)
    }
}

impl QueryConfig {
    /// The `p` this run actually uses: configured, or derived from `top_k`.
    /// Named apart from the field so a caller cannot read the raw `Option` by
    /// accident and silently fall back to some other default of its own.
    pub fn effective_rbo_p(&self) -> f64 {
        self.rbo_p.unwrap_or_else(|| default_rbo_p_for(self.top_k))
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
    /// Two scores within this RELATIVE tolerance (`|a-b| / (1+|b|)`) count as
    /// the same score, so an engine that returned a DIFFERENT member of a tie
    /// still counts as correct.
    ///
    /// `None` (default) → derived from the collection's stored `datatype`,
    /// probed once at startup; see [`tie_epsilon_for_datatype`] for the
    /// measured table. Set it explicitly to override, e.g. when the backend
    /// can't report its datatype or a quantization mode widens the gap.
    #[serde(default)]
    pub tie_epsilon: Option<f64>,
    /// RBO's persistence parameter: the probability the notional reader
    /// continues past each rank, which sets how hard the metric weights the
    /// top of the ranking. Expected reading depth is `1/(1-p)`, and `p^top_k`
    /// of the metric's weight sits BELOW `top_k`, where the run cannot observe
    /// it (the summary reports that residual).
    ///
    /// `None` (default) derives it from `top_k` so exactly
    /// [`RBO_DEFAULT_RESIDUAL`] is left unobserved — see
    /// [`default_rbo_p_for`]. A FIXED default cannot serve both ends of the
    /// `top_k` range: 0.95 leaves 0.6% unobserved at `top_k=100` but 60% at
    /// `top_k=10`, where a perfect ranking would then score 0.40.
    ///
    /// Set it explicitly to weight the very top harder (larger `p`, bigger
    /// residual) or to fix it across a sweep whose `top_k` varies. **State it
    /// in any published result** — an RBO without its `p` is not reproducible,
    /// and values at different `p` are not comparable.
    #[serde(default)]
    pub rbo_p: Option<f64>,
    /// Most ground-truth ids to keep per query that tie with its k-th place
    /// but fall outside `top_k` — the equally-correct answers rank agreement's
    /// upper bound is allowed to credit (see `queries::CutoffTies`).
    ///
    /// Ids are interned once per query set, so they cost ~12 bytes each when
    /// tails overlap across queries and up to ~150 when they do not (measured
    /// — the interning shares the strings, not the indices). A ground truth whose scores are FLAT keeps its
    /// whole tail for every query, which is where a cap earns its keep: a
    /// 1000-deep all-tied list at `top_k=10` retains 990 per query.
    ///
    /// Exceeding it drops that query's allowance ENTIRELY, never a subset —
    /// keeping the first few would decide the metric by the ground truth's
    /// tail order, which is precisely the ordering it treats as meaningless.
    /// A dropped query's upper bound narrows toward its exact value, and the
    /// count is warned about.
    #[serde(default = "default_max_cutoff_ties")]
    pub max_cutoff_ties: usize,
    /// What payload the server returns with each hit. Default `false`
    /// (ids/scores only). `true` = every payload field; a LIST of field names
    /// (e.g. `[text]`) = only those fields — the shape a RAG workload has,
    /// where each hit's document body comes back but nothing else. Either
    /// non-false form makes the server do the payload-storage reads (possibly
    /// from disk) for every hit, which is real cost a benchmark without it
    /// understates.
    #[serde(default)]
    pub with_payload: WithPayload,
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

/// `query.with_payload`: a plain bool, or a list of payload field names to
/// return (server-side include selector — the fields are trimmed at the
/// source, not client-side).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum WithPayload {
    Enable(bool),
    Fields(Vec<String>),
}

impl Default for WithPayload {
    fn default() -> Self {
        WithPayload::Enable(false)
    }
}

impl WithPayload {
    /// Does this setting make the server read payload storage at all?
    pub fn is_enabled(&self) -> bool {
        match self {
            WithPayload::Enable(enabled) => *enabled,
            WithPayload::Fields(fields) => !fields.is_empty(),
        }
    }
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
    /// The matching `list<float>` score column (e.g. nova-bf's `hit_scores`),
    /// read alongside `ground_truth_column`. Enables the tie-aware numbers:
    /// the ground truth's score at the top-k cutoff, and how many of its
    /// entries share that score.
    #[serde(default)]
    pub ground_truth_score_column: Option<String>,
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

/// Default for [`QueryConfig::max_cutoff_ties`]. Generous because the ids are
/// interned, so they cost ~12 bytes each where a near-duplicate corpus makes
/// the same ids recur across queries, and up to ~150 where every query's tail
/// is distinct (measured). 100k per query is generous for any real ground
/// truth; it exists to stop a degenerate one taking the process with it.
fn default_max_cutoff_ties() -> usize {
    100_000
}

/// Share of RBO's weight the derived default leaves below `top_k` — the part
/// the run cannot observe, reported as `rbo_residual`. 1% is small enough that
/// a perfect ranking scores 0.99 (reading as "perfect" at the 4 decimals the
/// summary prints) without pushing `p` so low that only the first two or three
/// ranks carry any weight.
pub const RBO_DEFAULT_RESIDUAL: f64 = 0.01;

/// The `p` that leaves exactly [`RBO_DEFAULT_RESIDUAL`] of RBO's weight below
/// `top_k`, i.e. `p^top_k == RBO_DEFAULT_RESIDUAL`.
///
/// Derived rather than fixed because the residual `p^top_k` depends entirely
/// on the depth being measured: this yields ~0.63 at `top_k=10` and ~0.955 at
/// `top_k=100`, where any single constant leaves one end badly truncated.
///
/// Be clear about what this costs. `p` is the METRIC — it sets how the
/// weight is distributed over ranks — while depth is only a measurement limit;
/// Webber et al.'s whole point is that at a FIXED `p`, rankings evaluated to
/// different depths remain comparable up to the residual width. So deriving
/// `p` from `top_k` does sacrifice comparability across depths: `rbo@10` at
/// p=0.63 (expected reading depth 2.7) and `rbo@100` at p=0.955 (depth 22) are
/// different metrics sharing a name. Set `rbo_p` explicitly for anything that
/// compares across `top_k`, and state it.
///
/// It also goes degenerate at a shallow `top_k`, with no warning: `top_k=1`
/// gives `p=0.01`, making `rbo@1` essentially `precision@1`, and `top_k=5`
/// gives `p=0.398`, which puts 60% of the weight on rank 1 alone. A metric
/// added because recall cannot see order collapses back toward a single-rank
/// measure exactly where the ranking is shortest. Set `rbo_p` explicitly below
/// a `top_k` of about 10.
pub fn default_rbo_p_for(top_k: u64) -> f64 {
    // `top_k == 0` is rejected before this is reachable; the `max` keeps a
    // stray caller from dividing by zero into a p of 0. The upper clamp keeps
    // the result strictly below 1 for an absurdly deep `top_k`, preserving the
    // invariant the validation above enforces for configured values.
    let k = top_k.max(1) as f64;
    RBO_DEFAULT_RESIDUAL.powf(1.0 / k).min(0.999_999)
}

/// Score tolerance to use for a given stored `datatype`, when the config
/// leaves `query.tie_epsilon` unset.
///
/// MEASURED, not guessed: worst-case relative gap (`|a-b| / (1+|b|)`) between
/// nova-bf's published `hit_scores` and live Qdrant exact search, over 4,000
/// dim-128 vectors x 100 queries x top-20:
///
/// | datatype | dot     | cosine  | default |
/// |----------|---------|---------|---------|
/// | float32  | 6.5e-07 | 1.5e-07 | 5e-06   |
/// | float16  | 4.2e-04 | 7.3e-05 | 2e-03   |
///
/// Each default carries roughly 5-8x headroom over the measured worst case.
/// A too-LARGE tolerance only widens the tie-tolerant upper bound, which is
/// already an upper bound; a too-small one silently under-reports ties. So
/// anything unrecognized — including `uint8`, whose quantization loss the
/// measurement above did not exercise — gets the conservative value.
pub fn tie_epsilon_for_datatype(datatype: Option<&str>) -> f64 {
    match datatype {
        Some("float32") => 5e-6,
        Some("float16") => 2e-3,
        _ => 2e-3,
    }
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
    #[error(
        "query.tie_epsilon must be a finite, non-negative relative tolerance (got {0}) — a \
         negative or NaN value makes every score comparison fail, hiding all ties"
    )]
    BadTieEpsilon(f64),
    #[error(
        "query.rbo_p must be strictly between 0 and 1 (got {0}) — it is a geometric decay \
         factor over ranking depth, so 0, 1 and anything outside that range do not define an RBO"
    )]
    BadRboP(f64),
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
    #[error(
        "filter condition on `{field}` has an empty `match: []` list — it would never match anything"
    )]
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
        assert!(!cfg.query.with_payload.is_enabled()); // default: ids/scores only
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
        assert!(matches!(
            StormConfig::from_yaml(yaml).unwrap_err(),
            ConfigError::ZeroTopK
        ));
    }

    #[test]
    fn rbo_p_defaults_and_rejects_values_outside_the_open_unit_interval() {
        let cfg = |line: &str| {
            format!(
                r#"
target:
  type: qdrant
  url: http://localhost:6334
  collection_name: c
query:
  top_k: 10
{line}
  source:
    uri: /tmp/q.parquet
    column: embedding
"#
            )
        };
        // Unset -> derived from top_k, so an existing config keeps working and
        // still gets an RBO whose residual is 1% whatever depth it measures at.
        let derived = StormConfig::from_yaml(&cfg("")).unwrap();
        assert_eq!(derived.query.rbo_p, None, "nothing was configured");
        assert!(
            (derived.query.effective_rbo_p().powi(10) - RBO_DEFAULT_RESIDUAL).abs() < 1e-12,
            "the derived p must leave exactly the target residual at top_k=10, got {}",
            derived.query.effective_rbo_p()
        );
        // A configured value wins and is used verbatim.
        let set = StormConfig::from_yaml(&cfg("  rbo_p: 0.98")).unwrap();
        assert_eq!(set.query.rbo_p, Some(0.98));
        assert_eq!(set.query.effective_rbo_p(), 0.98);
        // The endpoints are excluded, not just the values beyond them: at 0
        // only depth 1 would count, and at 1 the weights never decay.
        for bad in ["0.0", "1.0", "-0.5", "1.5", ".nan"] {
            assert!(
                matches!(
                    StormConfig::from_yaml(&cfg(&format!("  rbo_p: {bad}"))).unwrap_err(),
                    ConfigError::BadRboP(_)
                ),
                "rbo_p: {bad} must be rejected"
            );
        }
    }

    #[test]
    fn the_derived_rbo_p_tracks_top_k_instead_of_being_one_constant() {
        // The whole reason it is derived: a single constant is badly wrong at
        // one end of the range. 0.95 would leave 60% of the metric unobserved
        // at top_k=10 while being about right at 100.
        let shallow = default_rbo_p_for(10);
        let deep = default_rbo_p_for(100);
        assert!(shallow < deep, "{shallow} < {deep}");
        for k in [1u64, 5, 10, 100, 1000] {
            let p = default_rbo_p_for(k);
            assert!(p > 0.0 && p < 1.0, "top_k={k} gave p={p}, outside (0,1)");
            assert!(
                (p.powi(k as i32) - RBO_DEFAULT_RESIDUAL).abs() < 1e-9,
                "top_k={k}: residual {} is not the target",
                p.powi(k as i32)
            );
        }
        // Absurd depths must still satisfy the `p < 1` invariant the config
        // validation enforces for configured values, rather than rounding to 1.
        assert!(default_rbo_p_for(u64::MAX) < 1.0);
        // `top_k == 0` is rejected upstream; this must not divide by zero into
        // a p of 0, which would make the weighting degenerate.
        assert!(default_rbo_p_for(0) > 0.0);
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
        assert!(matches!(
            StormConfig::from_yaml(yaml).unwrap_err(),
            ConfigError::ZeroBatchSize
        ));
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
        assert!(matches!(
            StormConfig::from_yaml(yaml).unwrap_err(),
            ConfigError::Yaml(_)
        ));
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
        assert!(cfg.query.with_payload.is_enabled());
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
        assert!(matches!(
            StormConfig::from_yaml(&bad).unwrap_err(),
            ConfigError::Yaml(_)
        ));
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
        assert_eq!(
            cfg.query.source.ground_truth_column.as_deref(),
            Some("hit_ids")
        );
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
        let err =
            StormConfig::from_yaml(&yaml_with_query_extras("  vector_type: sparse\n")).unwrap_err();
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
    fn qdrant_timeout_defaults_generous_and_is_settable() {
        use crate::targets::TargetConfig;
        // default: 300s, NOT the qdrant client's 5s (a load tester must not
        // count its own honest slow dispatches as errors)
        let cfg = StormConfig::from_yaml(&yaml_with_query_extras("")).expect("parses");
        let TargetConfig::Qdrant(q) = &cfg.target else {
            panic!("qdrant target")
        };
        assert_eq!((q.timeout_s, q.connect_timeout_s), (300, 10));

        let yaml = "target:\n  type: qdrant\n  url: http://localhost:6334\n  collection_name: c\n  timeout_s: 600\n  connect_timeout_s: 30\n\
             query:\n  source:\n    uri: /tmp/q.parquet\n    column: e\n";
        let cfg = StormConfig::from_yaml(yaml).expect("parses");
        let TargetConfig::Qdrant(q) = &cfg.target else {
            panic!("qdrant target")
        };
        assert_eq!((q.timeout_s, q.connect_timeout_s), (600, 30));
    }

    #[test]
    fn with_payload_accepts_bool_and_field_list() {
        let cfg = StormConfig::from_yaml(&yaml_with_query_extras("  with_payload: [text]\n"))
            .expect("parses");
        assert_eq!(
            cfg.query.with_payload,
            WithPayload::Fields(vec!["text".into()])
        );
        assert!(cfg.query.with_payload.is_enabled());

        let cfg = StormConfig::from_yaml(&yaml_with_query_extras("  with_payload: true\n"))
            .expect("parses");
        assert_eq!(cfg.query.with_payload, WithPayload::Enable(true));

        // an empty include list selects nothing -> payload reads stay off
        let cfg = StormConfig::from_yaml(&yaml_with_query_extras("  with_payload: []\n"))
            .expect("parses");
        assert!(!cfg.query.with_payload.is_enabled());
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
        let err =
            StormConfig::from_yaml(&yaml_with_query_extras("  vector_type: hybrid\n")).unwrap_err();
        assert!(matches!(err, ConfigError::Yaml(_)));
    }

    fn expand(input: &str, vars: &[(&str, &str)]) -> Result<String, ConfigError> {
        let map: std::collections::HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        expand_env_with(input, |k| map.get(k).cloned())
    }

    #[test]
    fn expands_and_defaults() {
        assert_eq!(expand("url: ${U}", &[("U", "x")]).unwrap(), "url: x");
        assert_eq!(expand("url: ${U:-fallback}", &[]).unwrap(), "url: fallback");
        assert!(matches!(
            expand("${NOPE}", &[]).unwrap_err(),
            ConfigError::MissingEnvVar(_)
        ));
    }
}
