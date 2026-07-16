//! Qdrant implementation of [`QueryTarget`].

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::{
    Condition, QuantizationSearchParams, QueryBatchPointsBuilder, QueryPointsBuilder, Range as QdrantRange,
    ScoredPoint, SearchParams,
};
use qdrant_client::qdrant::Filter as QdrantFilter;
use serde::Deserialize;

use super::{BatchOutcome, QueryTarget};
use crate::config::{QuantizationSearchParamsConfig, QueryConfig, SearchParamsConfig};
use crate::errors::TargetError;
use crate::filter::{Filter, FilterCondition, FilterFieldValue, MatchSpec, MatchValue, RangeCondition, RangeFromQuery};
use crate::queries::QueryVector;

/// Fires nearest-neighbour queries at a Qdrant collection over gRPC.
pub struct QdrantTarget {
    client: Qdrant,
    collection_name: String,
    vector_name: Option<String>,
    top_k: u64,
    /// Server-side search-time tuning, converted once at construction and
    /// applied unchanged to every query — same "bake in the knobs once"
    /// pattern as `top_k`/`vector_name`. `None` when unconfigured, so the
    /// query builder skips `.params(...)` entirely and the server's own
    /// defaults apply.
    search_params: Option<SearchParams>,
    /// Whether to materialize `BatchOutcome.ids`. Skipped (leaving each
    /// position `None`) when the run has no `ground_truth_column` configured
    /// — recall is the only consumer of `ids`, so collecting it otherwise is
    /// a wasted allocation (a `String` clone per returned point) on every
    /// query.
    collect_ids: bool,
    /// Translated once at construction and applied unchanged to every
    /// query — set only when `query.filter` has no `_from_query` condition
    /// (a uniform filter has nothing to resolve per query). `None` when
    /// unfiltered OR when the filter is per-query (see `per_query_filter`).
    static_filter: Option<QdrantFilter>,
    /// The config-level filter, kept around only when it has at least one
    /// `_from_query` condition — re-translated fresh for each query inside
    /// `query_batch`, using that query's own `QueryVector::filter_values`.
    /// Mutually exclusive with `static_filter` being `Some`.
    per_query_filter: Option<Filter>,
}

impl From<&QuantizationSearchParamsConfig> for QuantizationSearchParams {
    fn from(q: &QuantizationSearchParamsConfig) -> Self {
        QuantizationSearchParams { ignore: q.ignore, rescore: q.rescore, oversampling: q.oversampling }
    }
}

impl From<&SearchParamsConfig> for SearchParams {
    fn from(p: &SearchParamsConfig) -> Self {
        SearchParams {
            hnsw_ef: p.hnsw_ef,
            exact: p.exact,
            quantization: p.quantization.as_ref().map(QuantizationSearchParams::from),
            indexed_only: None,
            acorn: None,
        }
    }
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

        let (static_filter, per_query_filter) = match &query.filter {
            None => (None, None),
            Some(f) if f.is_per_query() => (None, Some(f.clone())),
            Some(f) => (Some(to_qdrant_filter(f, None)?), None),
        };

        Ok(QdrantTarget {
            client,
            collection_name: self.collection_name,
            vector_name: query.vector_name.clone(),
            top_k: query.top_k,
            search_params: query.search_params.as_ref().map(SearchParams::from),
            collect_ids: query.source.ground_truth_column.is_some(),
            static_filter,
            per_query_filter,
        })
    }
}

impl fmt::Display for QdrantTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "qdrant({})", self.collection_name)
    }
}

impl QdrantTarget {
    /// The filter to attach to `query`'s request, if any: the pre-translated
    /// `static_filter` (cheap clone of a small protobuf struct), or a fresh
    /// translation of `per_query_filter` against `query.filter_values` —
    /// exactly one of the two is ever set (see their docstrings on
    /// `QdrantTarget`).
    fn effective_filter(&self, query: &QueryVector) -> Result<Option<QdrantFilter>, TargetError> {
        if let Some(filter) = &self.static_filter {
            return Ok(Some(filter.clone()));
        }
        match &self.per_query_filter {
            Some(filter) => to_qdrant_filter(filter, Some(&query.filter_values)).map(Some),
            None => Ok(None),
        }
    }
}

/// Translate a backend-agnostic [`Filter`] into a Qdrant [`QdrantFilter`].
/// `query_values` must be `Some` iff `filter.is_per_query()` — the only
/// caller with a per-query filter (`QdrantTarget::effective_filter`) always
/// supplies it.
fn to_qdrant_filter(
    filter: &Filter,
    query_values: Option<&HashMap<String, FilterFieldValue>>,
) -> Result<QdrantFilter, TargetError> {
    let group = |conds: &[FilterCondition]| -> Result<Vec<Condition>, TargetError> {
        conds.iter().map(|c| to_qdrant_condition(c, query_values)).collect()
    };
    Ok(QdrantFilter {
        must: group(&filter.must)?,
        should: group(&filter.should)?,
        must_not: group(&filter.must_not)?,
        min_should: None,
    })
}

/// Translate one [`FilterCondition`] into a Qdrant [`Condition`]. Exactly one
/// of the condition's six fields is set (`Filter::validate` enforces this at
/// config-load time), so this reads as one branch per kind.
fn to_qdrant_condition(
    cond: &FilterCondition,
    query_values: Option<&HashMap<String, FilterFieldValue>>,
) -> Result<Condition, TargetError> {
    if let Some(spec) = &cond.r#match {
        return to_qdrant_match_condition(&cond.field, spec);
    }
    if let Some(range) = &cond.range {
        return Ok(Condition::range(&cond.field, to_qdrant_range(range)));
    }
    if let Some(text) = &cond.match_text {
        return Ok(Condition::matches_text(&cond.field, text.clone()));
    }

    // Everything below is a `_from_query` variant — `query_values` is
    // guaranteed `Some` here by construction (`QdrantTarget::effective_filter`
    // only calls this with `None` for a filter that has no per-query
    // condition at all), and every column a `_from_query` leaf names is
    // guaranteed present (`queries::load_query_vectors` projects exactly the
    // columns `Filter::query_fields` names, erroring at load time on a NULL).
    // A missing key here would mean those two guarantees drifted apart — a
    // logic bug, not a reachable runtime state.
    let query_values = query_values.expect("per-query filter translation requires resolved query_values");
    let lookup = |col: &str| {
        query_values
            .get(col)
            .unwrap_or_else(|| panic!("filter column `{col}` missing from query's resolved filter_values"))
    };

    if let Some(col) = &cond.match_from_query {
        return to_qdrant_match_from_value(&cond.field, lookup(col));
    }
    if let Some(range) = &cond.range_from_query {
        return to_qdrant_range_from_query(&cond.field, range, query_values);
    }
    if let Some(col) = &cond.match_text_from_query {
        return match lookup(col) {
            FilterFieldValue::Text(s) => Ok(Condition::matches_text(&cond.field, s.clone())),
            _ => Err(TargetError::Other(format!(
                "filter condition on `{}`: `match_text_from_query` column `{col}` must be text",
                cond.field
            ))),
        };
    }
    unreachable!("Filter::validate ensures exactly one of the six condition kinds is set")
}

fn to_qdrant_range(range: &RangeCondition) -> QdrantRange {
    QdrantRange { gt: range.gt, gte: range.gte, lt: range.lt, lte: range.lte }
}

/// Translate a `match`/`match_from_query` value. Qdrant's own match condition
/// has no float variant (only keyword/integer/bool, plus `MatchAny` lists of
/// integers or keywords) — a `MatchValue::Float` is rejected here, not at
/// config-parse time, since the config type isn't inherently bound to Qdrant.
fn to_qdrant_match_condition(field: &str, spec: &MatchSpec) -> Result<Condition, TargetError> {
    match spec {
        MatchSpec::One(MatchValue::Bool(b)) => Ok(Condition::matches(field, *b)),
        MatchSpec::One(MatchValue::Int(i)) => Ok(Condition::matches(field, *i)),
        MatchSpec::One(MatchValue::Str(s)) => Ok(Condition::matches(field, s.clone())),
        MatchSpec::One(MatchValue::Float(v)) => Err(float_match_error(field, *v)),
        MatchSpec::Any(values) => to_qdrant_match_any(field, values),
    }
}

/// Qdrant's `MatchAny` only has `Vec<i64>`/`Vec<String>` constructors — no
/// bool, no float, no mixed-type list.
fn to_qdrant_match_any(field: &str, values: &[MatchValue]) -> Result<Condition, TargetError> {
    if values.iter().all(|v| matches!(v, MatchValue::Int(_))) {
        let ints =
            values.iter().map(|v| if let MatchValue::Int(i) = v { *i } else { unreachable!() }).collect::<Vec<_>>();
        return Ok(Condition::matches(field, ints));
    }
    if values.iter().all(|v| matches!(v, MatchValue::Str(_))) {
        let strs = values
            .iter()
            .map(|v| if let MatchValue::Str(s) = v { s.clone() } else { unreachable!() })
            .collect::<Vec<_>>();
        return Ok(Condition::matches(field, strs));
    }
    Err(TargetError::Other(format!(
        "filter condition on `{field}`: qdrant's MatchAny only supports an all-integer or all-string \
         list (got a mix, a bool, or a float)"
    )))
}

fn float_match_error(field: &str, value: f64) -> TargetError {
    TargetError::Other(format!(
        "filter condition on `{field}`: qdrant match does not support the float value `{value}` — use \
         `range` with equal `gte`/`lte` for a numeric equality check"
    ))
}

/// Translate one query's resolved `_from_query` value for a `match`-shaped
/// condition. A numeric value must be a whole number — Qdrant's match takes
/// `i64`, not `f64` — same constraint [`to_qdrant_match_condition`] enforces
/// for a literal, just discovered from data instead of config.
fn to_qdrant_match_from_value(field: &str, value: &FilterFieldValue) -> Result<Condition, TargetError> {
    match value {
        FilterFieldValue::Text(s) => Ok(Condition::matches(field, s.clone())),
        FilterFieldValue::Num(n) => integer_value(*n)
            .map(|i| Condition::matches(field, i))
            .ok_or_else(|| non_integer_from_query_error(field, *n)),
        FilterFieldValue::TextList(xs) => Ok(Condition::matches(field, xs.clone())),
        FilterFieldValue::NumList(xs) => xs
            .iter()
            .map(|&n| integer_value(n))
            .collect::<Option<Vec<_>>>()
            .map(|ints| Condition::matches(field, ints))
            .ok_or_else(|| {
                TargetError::Other(format!(
                    "filter condition on `{field}`: a per-query MatchAny list has a non-integer value \
                     — qdrant match only supports whole numbers"
                ))
            }),
    }
}

fn to_qdrant_range_from_query(
    field: &str,
    range: &RangeFromQuery,
    query_values: &HashMap<String, FilterFieldValue>,
) -> Result<Condition, TargetError> {
    let bound = |col: &Option<String>| -> Result<Option<f64>, TargetError> {
        let Some(name) = col else { return Ok(None) };
        let value = query_values
            .get(name)
            .unwrap_or_else(|| panic!("filter column `{name}` missing from query's resolved filter_values"));
        match value {
            FilterFieldValue::Num(n) => Ok(Some(*n)),
            _ => Err(TargetError::Other(format!(
                "filter condition on `{field}`: `range_from_query` column `{name}` must be numeric"
            ))),
        }
    };
    Ok(Condition::range(
        field,
        QdrantRange { gt: bound(&range.gt)?, gte: bound(&range.gte)?, lt: bound(&range.lt)?, lte: bound(&range.lte)? },
    ))
}

fn integer_value(n: f64) -> Option<i64> {
    (n.fract() == 0.0).then_some(n as i64)
}

fn non_integer_from_query_error(field: &str, value: f64) -> TargetError {
    TargetError::Other(format!(
        "filter condition on `{field}`: qdrant match does not support the non-integer per-query value \
         `{value}` — use `range_from_query` instead"
    ))
}

#[async_trait]
impl QueryTarget for QdrantTarget {
    async fn query_batch(&self, queries: &[&QueryVector]) -> BatchOutcome {
        let query_points: Vec<_> = match queries
            .iter()
            .map(|q| {
                let mut builder = QueryPointsBuilder::new(&self.collection_name)
                    .query(q.vector.to_vec())
                    .limit(self.top_k)
                    .with_payload(false);
                if let Some(name) = &self.vector_name {
                    builder = builder.using(name.clone());
                }
                if let Some(params) = self.search_params {
                    builder = builder.params(params);
                }
                if let Some(filter) = self.effective_filter(q)? {
                    builder = builder.filter(filter);
                }
                Ok(builder.build())
            })
            .collect::<Result<Vec<_>, TargetError>>()
        {
            Ok(points) => points,
            // A per-query filter that can't be translated (e.g. a non-integer
            // value under a keyword `match_from_query`) is a data problem
            // with this dispatch, not a crash — same "errors at the limit are
            // a finding" treatment as a Qdrant RPC failure below.
            Err(e) => {
                return BatchOutcome {
                    latency: Duration::from_secs(0),
                    ok: false,
                    ids: vec![None; queries.len()],
                    error: Some(e.to_string()),
                };
            }
        };
        let request = QueryBatchPointsBuilder::new(&self.collection_name, query_points);

        let started = Instant::now();
        match self.client.query_batch(request).await {
            // A length mismatch here means the response can no longer be
            // zipped positionally against the submitted queries without
            // risking a query's recall being scored against another query's
            // results — treat it the same as a hard failure (no ids) rather
            // than silently misaligning them. `debug_assert_eq!` alone isn't
            // enough: this runs in release builds, the mode real storms use.
            Ok(resp) if resp.result.len() != queries.len() => BatchOutcome {
                latency: started.elapsed(),
                ok: false,
                ids: vec![None; queries.len()],
                error: Some(format!(
                    "query_batch returned {} results for {} submitted queries",
                    resp.result.len(),
                    queries.len()
                )),
            },
            Ok(resp) => BatchOutcome {
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
            },
            Err(e) => BatchOutcome {
                latency: started.elapsed(),
                ok: false,
                ids: vec![None; queries.len()],
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

    #[test]
    fn search_params_absent_by_default() {
        let cfg = cfg(); // no search_params block
        let target = qdrant_config(cfg.target).into_target(&cfg.query).expect("builds");
        assert!(target.search_params.is_none());
    }

    #[test]
    fn search_params_parse_and_convert() {
        let yaml = "target:\n  type: qdrant\n  url: http://localhost:6334\n  collection_name: c\n\
                    query:\n  vector_name: dense\n  top_k: 5\n  source:\n    uri: /tmp/q.parquet\n    column: e\n\
                    \x20 search_params:\n    hnsw_ef: 128\n    exact: false\n    quantization:\n      ignore: false\n      rescore: true\n      oversampling: 2.5\n";
        let cfg = StormConfig::from_yaml(yaml).expect("parses");
        let target = qdrant_config(cfg.target).into_target(&cfg.query).expect("builds");
        let params = target.search_params.expect("search_params should be Some");
        assert_eq!(params.hnsw_ef, Some(128));
        assert_eq!(params.exact, Some(false));
        let quant = params.quantization.expect("quantization should be Some");
        assert_eq!(quant.ignore, Some(false));
        assert_eq!(quant.rescore, Some(true));
        assert_eq!(quant.oversampling, Some(2.5));
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

    fn cond(yaml: &str) -> FilterCondition {
        serde_yaml::from_str(yaml).expect("parses")
    }

    #[test]
    fn translates_scalar_match_range_and_text() {
        let c = to_qdrant_condition(&cond("field: category\nmatch: shoes\n"), None).unwrap();
        assert!(matches!(
            c.condition_one_of,
            Some(qdrant_client::qdrant::condition::ConditionOneOf::Field(_))
        ));

        to_qdrant_condition(&cond("field: price\nrange:\n  gte: 10.0\n  lt: 100.0\n"), None).unwrap();
        to_qdrant_condition(&cond("field: description\nmatch_text: waterproof hiking\n"), None).unwrap();
    }

    #[test]
    fn translates_homogeneous_match_any_lists() {
        to_qdrant_condition(&cond("field: tag\nmatch: [a, b, c]\n"), None).unwrap();
        to_qdrant_condition(&cond("field: code\nmatch: [1, 2, 3]\n"), None).unwrap();
    }

    #[test]
    fn rejects_float_match_value() {
        let err = to_qdrant_condition(&cond("field: price\nmatch: 9.99\n"), None).unwrap_err();
        assert!(err.to_string().contains("float"));
    }

    #[test]
    fn rejects_mixed_type_match_any() {
        // untagged enum coerces `1` to an int and `\"a\"` to a string -- a
        // deliberately mixed list, which qdrant's MatchAny can't express.
        let err = to_qdrant_condition(&cond("field: tag\nmatch: [1, \"a\"]\n"), None).unwrap_err();
        assert!(err.to_string().contains("MatchAny"));
    }

    #[test]
    fn translates_per_query_variants_from_resolved_values() {
        let values = HashMap::from([
            ("tenant_column".to_string(), FilterFieldValue::Text("acme".to_string())),
            ("max_budget".to_string(), FilterFieldValue::Num(42.0)),
            ("phrase_column".to_string(), FilterFieldValue::Text("waterproof hiking".to_string())),
        ]);

        to_qdrant_condition(
            &cond("field: tenant_id\nmatch_from_query: tenant_column\n"),
            Some(&values),
        )
        .unwrap();
        to_qdrant_condition(
            &cond("field: budget\nrange_from_query:\n  lt: max_budget\n"),
            Some(&values),
        )
        .unwrap();
        to_qdrant_condition(
            &cond("field: description\nmatch_text_from_query: phrase_column\n"),
            Some(&values),
        )
        .unwrap();
    }

    #[test]
    fn rejects_non_integer_per_query_match_value() {
        let values = HashMap::from([("budget_column".to_string(), FilterFieldValue::Num(9.5))]);
        let err = to_qdrant_condition(&cond("field: budget\nmatch_from_query: budget_column\n"), Some(&values))
            .unwrap_err();
        assert!(err.to_string().contains("non-integer"));
    }

    #[test]
    fn to_qdrant_filter_groups_must_should_must_not() {
        let filter: Filter = serde_yaml::from_str(
            "must:\n  - field: category\n    match: shoes\nshould:\n  - field: color\n    match: red\nmust_not:\n  - field: tag\n    match: discontinued\n",
        )
        .unwrap();
        let translated = to_qdrant_filter(&filter, None).unwrap();
        assert_eq!(translated.must.len(), 1);
        assert_eq!(translated.should.len(), 1);
        assert_eq!(translated.must_not.len(), 1);
    }

    #[test]
    fn into_target_bakes_static_filter_when_not_per_query() {
        let yaml = "target:\n  type: qdrant\n  url: http://localhost:6334\n  collection_name: c\n\
                    query:\n  top_k: 5\n  source:\n    uri: /tmp/q.parquet\n    column: e\n  filter:\n    must:\n      - field: category\n        match: shoes\n";
        let cfg = StormConfig::from_yaml(yaml).expect("parses");
        let target = qdrant_config(cfg.target).into_target(&cfg.query).expect("builds");
        assert!(target.static_filter.is_some());
        assert!(target.per_query_filter.is_none());
    }

    #[test]
    fn into_target_keeps_per_query_filter_unbaked() {
        let yaml = "target:\n  type: qdrant\n  url: http://localhost:6334\n  collection_name: c\n\
                    query:\n  top_k: 5\n  source:\n    uri: /tmp/q.parquet\n    column: e\n  filter:\n    must:\n      - field: tenant_id\n        match_from_query: tenant_column\n";
        let cfg = StormConfig::from_yaml(yaml).expect("parses");
        let target = qdrant_config(cfg.target).into_target(&cfg.query).expect("builds");
        assert!(target.static_filter.is_none());
        assert!(target.per_query_filter.is_some());
    }

    #[test]
    fn into_target_rejects_a_float_match_in_a_static_filter_at_construction() {
        let yaml = "target:\n  type: qdrant\n  url: http://localhost:6334\n  collection_name: c\n\
                    query:\n  top_k: 5\n  source:\n    uri: /tmp/q.parquet\n    column: e\n  filter:\n    must:\n      - field: price\n        match: 9.99\n";
        let cfg = StormConfig::from_yaml(yaml).expect("parses");
        assert!(qdrant_config(cfg.target).into_target(&cfg.query).is_err());
    }
}
