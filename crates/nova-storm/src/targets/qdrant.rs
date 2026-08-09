//! Qdrant implementation of [`QueryTarget`].

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use async_trait::async_trait;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::{
    Condition, DatetimeRange, QuantizationSearchParams, Query, QueryBatchPointsBuilder,
    QueryPointsBuilder, Range as QdrantRange, ScoredPoint, SearchParams, Timestamp, VectorInput,
};
use qdrant_client::qdrant::Filter as QdrantFilter;
use serde::Deserialize;

use super::{BatchOutcome, QueryTarget};
use crate::config::{QueryConfig, WithPayload};
use crate::errors::TargetError;
use crate::filter::{Filter, FilterCondition, FilterFieldValue, MatchSpec, MatchValue, RangeCondition, RangeFromQuery};
use crate::queries::{QueryVector, VectorData};

/// Qdrant's server-side search-time tuning (`query.search_params` for a
/// `qdrant` target). All fields optional; the server applies its own defaults
/// for any left unset. `deny_unknown_fields` rejects a key that isn't a Qdrant
/// search param — so a Milvus/Elastic knob under a `qdrant` target is caught.
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

/// Quantization behavior at query time.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationSearchParamsConfig {
    /// Skip the quantized index entirely for this query.
    #[serde(default)]
    pub ignore: Option<bool>,
    /// Re-score quantized top-k candidates against the original vectors.
    #[serde(default)]
    pub rescore: Option<bool>,
    /// Extra candidates to preselect via the quantized index before rescoring.
    #[serde(default)]
    pub oversampling: Option<f64>,
}

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
    /// The payload selector built once from `query.with_payload` (bool or an
    /// include-list of fields, e.g. just `text` for a RAG-shaped workload).
    /// The payloads are dropped on arrival — storm measures, it doesn't
    /// consume — but requesting them makes the server do the payload reads
    /// and ship the bytes, which is the cost being modeled.
    with_payload: qdrant_client::qdrant::with_payload_selector::SelectorOptions,
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
    /// Per-operation (query) timeout in seconds. The qdrant client's own
    /// default is FIVE seconds — far too short for a load tester, whose whole
    /// job is to push a cluster into multi-second latencies; hitting it turns
    /// honest slow dispatches into "operation was cancelled / Timeout
    /// expired" errors and corrupts the error counts. Unset = 300s.
    #[serde(default = "default_timeout_s")]
    pub timeout_s: u64,
    /// Connection-establishment timeout in seconds. Unset = 10s.
    #[serde(default = "default_connect_timeout_s")]
    pub connect_timeout_s: u64,
}

fn default_timeout_s() -> u64 {
    300
}

fn default_connect_timeout_s() -> u64 {
    10
}

fn default_collection() -> String {
    "default".to_string()
}

impl QdrantConfig {
    /// Connect and build the target, baking in the query knobs from `query`.
    pub fn into_target(self, query: &QueryConfig) -> Result<QdrantTarget, TargetError> {
        // skip_compatibility_check: without it, `build()` fires a synchronous
        // version-probe RPC before the run starts — an unreachable server then
        // stalls construction for the full `timeout_s` instead of failing on
        // the first dispatch, where the error is actually counted.
        let mut builder = Qdrant::from_url(&self.url)
            .timeout(std::time::Duration::from_secs(self.timeout_s))
            .connect_timeout(std::time::Duration::from_secs(self.connect_timeout_s))
            .skip_compatibility_check();
        if let Some(key) = self.api_key {
            builder = builder.api_key(key);
        }
        let client = builder.build()?;

        // Parse the raw, backend-specific `search_params` into Qdrant's schema;
        // `deny_unknown_fields` rejects a param that isn't Qdrant's here.
        let search_params: Option<SearchParamsConfig> = query
            .search_params
            .as_ref()
            .map(|v| serde_yaml::from_value(v.clone()))
            .transpose()
            .map_err(|e| TargetError::Other(format!("qdrant search_params: {e}")))?;

        let (static_filter, per_query_filter) = match &query.filter {
            None => (None, None),
            Some(f) => {
                // Defense in depth: `StormConfig::from_yaml` already calls
                // this, but `StormConfig`'s fields are all `pub` and
                // `nova_storm::run` is a public library entry point, so a
                // caller that hand-builds one (skipping `from_yaml`) would
                // otherwise reach `to_qdrant_condition`'s per-query lookups
                // with a shape `Filter::validate` was supposed to rule out.
                f.validate().map_err(|e| TargetError::Other(e.to_string()))?;
                if f.is_per_query() {
                    // A per-query filter defers ITS OWN `_from_query` leaves
                    // to request time (they need real data to translate) —
                    // but any fully-static sibling condition in the same
                    // filter can, and should, still be validated now: without
                    // this, a bad static leaf (e.g. a float `match`) would
                    // only surface once `query_batch` starts running, failing
                    // every single dispatch for the run's whole duration
                    // instead of at startup.
                    validate_static_conditions(f)?;
                    (None, Some(f.clone()))
                } else {
                    (Some(to_qdrant_filter(f, None)?), None)
                }
            }
        };

        Ok(QdrantTarget {
            client,
            collection_name: self.collection_name,
            vector_name: query.vector_name.clone(),
            top_k: query.top_k,
            search_params: search_params.as_ref().map(SearchParams::from),
            collect_ids: query.source.ground_truth_column.is_some(),
            with_payload: {
                use qdrant_client::qdrant::with_payload_selector::SelectorOptions;
                match &query.with_payload {
                    WithPayload::Enable(enabled) => SelectorOptions::Enable(*enabled),
                    // Server-side include selector: only these fields come
                    // back (the RAG shape -- e.g. just `text`).
                    WithPayload::Fields(fields) => SelectorOptions::Include(
                        qdrant_client::qdrant::PayloadIncludeSelector { fields: fields.clone() },
                    ),
                }
            },
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

/// Validate every fully-static condition in `filter` — even one that also has
/// per-query siblings elsewhere in the same `must`/`should`/`must_not` group —
/// by actually translating it (and discarding the result). Called eagerly at
/// `QdrantConfig::into_target` time so a bad static leaf (e.g. a float
/// `match`) fails at startup like the purely-static case does, instead of
/// only surfacing once `query_batch` starts running (which would fail every
/// single dispatch for the run's whole duration on what was a catchable
/// config mistake). Per-query conditions are skipped — they need real
/// `QueryVector` data to translate, which doesn't exist yet at this point.
fn validate_static_conditions(filter: &Filter) -> Result<(), TargetError> {
    filter
        .must
        .iter()
        .chain(filter.should.iter())
        .chain(filter.must_not.iter())
        .filter(|c| !c.is_per_query())
        .try_for_each(|c| to_qdrant_condition(c, None).map(|_| ()))
}

/// Look up a `_from_query` column's resolved value for the current query.
/// Present by construction: `queries::load_query_vectors` projects exactly the
/// columns `Filter::query_fields` names (erroring at load time on NULL). A
/// missing key would mean that invariant broke (e.g. a hand-built config via the
/// public API) — return an `Err` (a failed batch) rather than panic a worker
/// mid-storm ("errors at the limit are a finding, not a crash").
fn resolve_query_value<'a>(
    query_values: &'a HashMap<String, FilterFieldValue>,
    col: &str,
) -> Result<&'a FilterFieldValue, TargetError> {
    query_values.get(col).ok_or_else(|| {
        TargetError::Other(format!("filter column `{col}` missing from the query's filter_values"))
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

    // Everything below is a `_from_query` variant — `query_values` is `Some`
    // here by construction (`QdrantTarget::effective_filter` only passes `None`
    // for a filter with no per-query condition). Return an `Err` rather than
    // panic if that invariant is ever violated (e.g. a hand-built config).
    let query_values = query_values.ok_or_else(|| {
        TargetError::Other("per-query filter translation requires resolved query_values".to_string())
    })?;

    if let Some(col) = &cond.match_from_query {
        return to_qdrant_match_from_value(&cond.field, resolve_query_value(query_values, col)?);
    }
    if let Some(range) = &cond.range_from_query {
        return to_qdrant_range_from_query(&cond.field, range, query_values);
    }
    if let Some(col) = &cond.match_text_from_query {
        return match resolve_query_value(query_values, col)? {
            FilterFieldValue::Text(s) if !s.trim().is_empty() => {
                Ok(Condition::matches_text(&cond.field, s.clone()))
            }
            FilterFieldValue::Text(_) => Err(TargetError::Other(format!(
                "filter condition on `{}`: `match_text_from_query` column `{col}` resolved to a blank \
                 value for this query — use a non-matching placeholder instead of an empty string",
                cond.field
            ))),
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

/// A query's nearest-neighbour input in either modality. Dense stays the
/// plain float vector it always was; sparse becomes qdrant's paired
/// indices/values input (dot-scored server-side — qdrant sparse has no other
/// metric). `query.vector_name` ("using") rides separately on the builder.
fn query_input(vector: &VectorData) -> Query {
    match vector {
        VectorData::Dense(v) => Query::new_nearest(v.clone()),
        VectorData::Sparse { indices, values } => {
            Query::new_nearest(VectorInput::new_sparse(indices.clone(), values.clone()))
        }
    }
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
/// condition. `Int`/`IntList` (an exact integer-typed queries column, e.g. a
/// `BIGINT` tenant id) map straight across with no precision loss. A `Num`
/// value must be a whole number — Qdrant's match takes `i64`, not `f64` —
/// same constraint [`to_qdrant_match_condition`] enforces for a literal, just
/// discovered from data instead of config; this path is only reached for a
/// genuinely float/decimal-typed queries column.
fn to_qdrant_match_from_value(field: &str, value: &FilterFieldValue) -> Result<Condition, TargetError> {
    match value {
        FilterFieldValue::Text(s) => Ok(Condition::matches(field, s.clone())),
        FilterFieldValue::Int(i) => Ok(Condition::matches(field, *i)),
        FilterFieldValue::Num(n) => integer_value(*n)
            .map(|i| Condition::matches(field, i))
            .ok_or_else(|| non_integer_from_query_error(field, *n)),
        FilterFieldValue::TextList(xs) => Ok(Condition::matches(field, xs.clone())),
        FilterFieldValue::IntList(xs) => Ok(Condition::matches(field, xs.clone())),
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

/// A `range_from_query` condition translates to one of TWO qdrant conditions,
/// decided by the DATA: numeric column values build the f64 `Range` they
/// always did; text values are parsed as RFC3339 datetimes and build a
/// `DatetimeRange` (qdrant's datetime-indexed fields — e.g. fineweb's `date`
/// — take timestamps on the gRPC wire, not strings). Mixing the two kinds
/// across one condition's bounds is rejected: a range over one field is over
/// one type.
fn to_qdrant_range_from_query(
    field: &str,
    range: &RangeFromQuery,
    query_values: &HashMap<String, FilterFieldValue>,
) -> Result<Condition, TargetError> {
    enum Bound {
        Num(f64),
        Datetime(Timestamp),
    }
    let bound = |col: &Option<String>| -> Result<Option<Bound>, TargetError> {
        let Some(name) = col else { return Ok(None) };
        match resolve_query_value(query_values, name)? {
            FilterFieldValue::Num(n) => Ok(Some(Bound::Num(*n))),
            // Qdrant range bounds are f64; reject an integer that can't round-trip
            // exactly (|i| > 2^53) rather than silently shifting the bound.
            FilterFieldValue::Int(i) => {
                let f = *i as f64;
                // `f as i64` saturates, so guard the top-of-range rounding case
                // (i64::MAX → 2^63) explicitly before the round-trip check.
                if f >= TWO_POW_63 || f as i64 != *i {
                    return Err(TargetError::Other(format!(
                        "filter condition on `{field}`: `range_from_query` column `{name}` value `{i}` \
                         can't be represented exactly as an f64 range bound"
                    )));
                }
                Ok(Some(Bound::Num(f)))
            }
            FilterFieldValue::Text(s) => Ok(Some(Bound::Datetime(parse_datetime_bound(field, name, s)?))),
            _ => Err(TargetError::Other(format!(
                "filter condition on `{field}`: `range_from_query` column `{name}` must be numeric \
                 or an RFC3339 datetime string"
            ))),
        }
    };

    let (gt, gte, lt, lte) = (bound(&range.gt)?, bound(&range.gte)?, bound(&range.lt)?, bound(&range.lte)?);
    let any_datetime = [&gt, &gte, &lt, &lte].iter().any(|b| matches!(b, Some(Bound::Datetime(_))));
    if !any_datetime {
        let num = |b: Option<Bound>| b.map(|b| if let Bound::Num(n) = b { n } else { unreachable!() });
        return Ok(Condition::range(
            field,
            QdrantRange { gt: num(gt), gte: num(gte), lt: num(lt), lte: num(lte), ..Default::default() },
        ));
    }
    let ts = |b: Option<Bound>| -> Result<Option<Timestamp>, TargetError> {
        match b {
            None => Ok(None),
            Some(Bound::Datetime(t)) => Ok(Some(t)),
            Some(Bound::Num(_)) => Err(TargetError::Other(format!(
                "filter condition on `{field}`: `range_from_query` mixes datetime and numeric \
                 bounds — one range condition is over one field of one type"
            ))),
        }
    };
    Ok(Condition::datetime_range(
        field,
        DatetimeRange { gt: ts(gt)?, gte: ts(gte)?, lt: ts(lt)?, lte: ts(lte)?, ..Default::default() },
    ))
}

/// One datetime string as the `Timestamp` qdrant's gRPC `DatetimeRange`
/// carries (re-exported by qdrant-client, so it always matches qdrant's own
/// prost version). Parsing lives in [`crate::datetime`] — shared with the
/// loader, which VALIDATES and normalizes every range-bound value at load
/// time, so by the time a value reaches this per-dispatch path it is
/// canonical RFC3339 and cannot fail. The error context here covers direct
/// library callers that skip the loader.
fn parse_datetime_bound(field: &str, column: &str, raw: &str) -> Result<Timestamp, TargetError> {
    let parsed = crate::datetime::parse_datetime_utc(raw).map_err(|e| {
        TargetError::Other(format!(
            "filter condition on `{field}`: `range_from_query` column `{column}` value `{raw}` is \
             not a recognized datetime — accepted forms are RFC3339 (offsets and pre-1970 dates \
             included), `YYYY-MM-DD HH:MM:SS[+HH]`, and date-only `YYYY-MM-DD`, all read as UTC \
             when offsetless: {e}"
        ))
    })?;
    let total_nanos = parsed.unix_timestamp_nanos();
    // Timestamp is (seconds, nanos 0..1e9); euclidean split keeps pre-1970
    // values well-formed (e.g. 1969-12-31T23:59:59.5Z -> seconds=-1, nanos=5e8).
    Ok(Timestamp {
        seconds: total_nanos.div_euclid(1_000_000_000) as i64,
        nanos: total_nanos.rem_euclid(1_000_000_000) as i32,
        ..Default::default()
    })
}

/// 2^63 = i64::MAX+1 — the first f64 above the i64 range (i64::MAX itself isn't
/// exactly representable as f64; i64::MIN = -2^63 IS). Used to reject f64 values
/// that would *saturate* on an `as i64` cast rather than convert faithfully.
const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;

/// A whole-valued, in-range `f64` as `i64`. Rejects non-finite, non-whole, and
/// out-of-range values — a plain `n as i64` SATURATES (e.g. `1e30 → i64::MAX`),
/// which would silently submit the wrong id to a `match`.
fn integer_value(n: f64) -> Option<i64> {
    if !n.is_finite() || n.fract() != 0.0 || n < i64::MIN as f64 || n >= TWO_POW_63 {
        return None;
    }
    Some(n as i64)
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
        // Started before request-building (not just before the RPC) so a
        // per-query filter translation failure below reports a real elapsed
        // duration — same treatment `started.elapsed()` already gets for an
        // RPC-level `Err` further down — instead of a fake `0` that would
        // skew this dispatch's contribution to the run's latency percentiles.
        let started = Instant::now();
        if queries.is_empty() {
            return BatchOutcome { latency: started.elapsed(), ok: true, ids: Vec::new(), error: None, timed_out: false, };
        }
        let query_points: Vec<_> = match queries
            .iter()
            .map(|q| {
                let mut builder = QueryPointsBuilder::new(&self.collection_name)
                    .query(query_input(&q.vector))
                    .limit(self.top_k)
                    .with_payload(self.with_payload.clone());
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
                    latency: started.elapsed(),
                    ok: false,
                    ids: vec![None; queries.len()],
                    error: Some(e.to_string()), timed_out: false,
                };
            }
        };
        let request = QueryBatchPointsBuilder::new(&self.collection_name, query_points);

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
                timed_out: false,
            },
            Ok(resp) => {
                if !self.collect_ids {
                    return BatchOutcome {
                        latency: started.elapsed(),
                        ok: true,
                        ids: vec![None; resp.result.len()],
                        error: None, timed_out: false,
                    };
                }
                let mut ids = Vec::with_capacity(resp.result.len());
                for batch_result in &resp.result {
                    // A scored point without a usable id is an unexpected
                    // response — fail rather than silently drop it and understate
                    // recall (consistent with the milvus/elastic targets).
                    let mut query_ids = Vec::with_capacity(batch_result.result.len());
                    for point in &batch_result.result {
                        let Some(id) = point_id_string(point) else {
                            return BatchOutcome {
                                latency: started.elapsed(),
                                ok: false,
                                ids: vec![None; queries.len()],
                                error: Some(
                                    "qdrant returned a scored point without a valid id".to_string(),
                                ), timed_out: false,
                            };
                        };
                        query_ids.push(id);
                    }
                    ids.push(Some(query_ids));
                }
                BatchOutcome { latency: started.elapsed(), ok: true, ids, error: None, timed_out: false, }
            }
            Err(e) => BatchOutcome {
                latency: started.elapsed(),
                ok: false,
                ids: vec![None; queries.len()],
                timed_out: is_client_timeout(&e),
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
/// Was this failure the CLIENT's own deadline rather than the server or the
/// network? gRPC status codes are protocol constants (CANCELLED = 1,
/// DEADLINE_EXCEEDED = 4 — fixed by the gRPC spec, not by tonic), so matching
/// on the numeric code needs no tonic dependency. The client's tonic timeout
/// layer surfaces `timeout_s` expiry as CANCELLED ("Timeout expired"); note
/// the client also silently RETRIES a cancelled read once on a rebuilt
/// channel, so an observed timeout costs ~2x `timeout_s` of wall time and
/// double the server work — one more reason `timeout_s` must sit far above
/// honest latency.
fn is_client_timeout(err: &qdrant_client::QdrantError) -> bool {
    match err {
        qdrant_client::QdrantError::ResponseError { status } => {
            matches!(status.code() as i32, 1 | 4) // CANCELLED | DEADLINE_EXCEEDED
        }
        _ => false,
    }
}

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

    #[tokio::test]
    async fn builds_target() {
        // `from_url(...).build()` is lazy (no connection yet), so this succeeds
        // offline and exercises the construction path.
        let cfg = cfg();
        let target = cfg.target.into_target(&cfg.query).await.expect("builds");
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
        // `let ... else` so this stays exhaustive whether or not the optional
        // `elastic`/`milvus` variants are compiled in.
        #[allow(irrefutable_let_patterns)]
        let crate::targets::TargetConfig::Qdrant(c) = target
        else {
            panic!("expected a qdrant target config");
        };
        c
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

    #[test]
    fn into_target_rejects_an_invalid_static_sibling_of_a_per_query_condition() {
        // The filter overall IS per-query (one match_from_query leaf), but a
        // fully-static sibling condition (a float match) is invalid -- this
        // must still fail at construction, not silently defer to query_batch.
        let yaml = "target:\n  type: qdrant\n  url: http://localhost:6334\n  collection_name: c\n\
                    query:\n  top_k: 5\n  source:\n    uri: /tmp/q.parquet\n    column: e\n  filter:\n    must:\n      - field: tenant_id\n        match_from_query: tenant_column\n      - field: price\n        match: 9.99\n";
        let cfg = StormConfig::from_yaml(yaml).expect("parses");
        let err = qdrant_config(cfg.target).into_target(&cfg.query).map(|_| ()).unwrap_err();
        assert!(err.to_string().contains("float"));
    }

    #[test]
    fn rejects_blank_match_text_from_query_value() {
        let values =
            HashMap::from([("phrase_column".to_string(), FilterFieldValue::Text("   ".to_string()))]);
        let err = to_qdrant_condition(
            &cond("field: description\nmatch_text_from_query: phrase_column\n"),
            Some(&values),
        )
        .unwrap_err();
        assert!(err.to_string().contains("blank"));
    }

    #[test]
    fn translates_exact_integer_and_int_list_from_query_values() {
        let values = HashMap::from([
            ("tenant_column".to_string(), FilterFieldValue::Int(9_007_199_254_740_993)),
            ("codes_column".to_string(), FilterFieldValue::IntList(vec![1, 2, 3])),
        ]);
        to_qdrant_condition(&cond("field: tenant_id\nmatch_from_query: tenant_column\n"), Some(&values))
            .unwrap();
        to_qdrant_condition(&cond("field: code\nmatch_from_query: codes_column\n"), Some(&values)).unwrap();
    }

    // ---------------------------------------------------------------- sparse

    #[test]
    fn query_input_builds_dense_and_sparse() {
        use crate::queries::VectorData;

        let dense = query_input(&VectorData::Dense(vec![0.1, 0.2]));
        assert_eq!(dense, Query::new_nearest(vec![0.1, 0.2]));

        let sparse =
            query_input(&VectorData::Sparse { indices: vec![7, 42], values: vec![0.5, 0.25] });
        assert_eq!(
            sparse,
            Query::new_nearest(VectorInput::new_sparse(vec![7u32, 42], vec![0.5f32, 0.25]))
        );
        // The two modalities must not collapse into the same wire shape.
        assert_ne!(sparse, Query::new_nearest(vec![0.5f32, 0.25]));
    }

    // ------------------------------------------------------------- datetime

    /// `2017-09-19T11:23:19Z` / `2018-05-28T09:22:20Z` — real fineweb `date`
    /// values, epoch seconds computed independently.
    const T1: (&str, i64) = ("2017-09-19T11:23:19Z", 1_505_820_199);
    const T2: (&str, i64) = ("2018-05-28T09:22:20Z", 1_527_499_340);

    fn datetime_values() -> HashMap<String, FilterFieldValue> {
        HashMap::from([
            ("date_gte".to_string(), FilterFieldValue::Text(T1.0.to_string())),
            ("date_lt".to_string(), FilterFieldValue::Text(T2.0.to_string())),
            ("ls_gte".to_string(), FilterFieldValue::Num(0.5)),
        ])
    }

    #[test]
    fn text_range_bounds_become_a_datetime_range() {
        let condition = to_qdrant_condition(
            &cond("field: date\nrange_from_query:\n  gte: date_gte\n  lt: date_lt\n"),
            Some(&datetime_values()),
        )
        .unwrap();
        let expected = Condition::datetime_range(
            "date",
            DatetimeRange {
                gt: None,
                gte: Some(Timestamp { seconds: T1.1, nanos: 0 }),
                lt: Some(Timestamp { seconds: T2.1, nanos: 0 }),
                lte: None,
            },
        );
        assert_eq!(condition, expected);
    }

    #[test]
    fn numeric_range_bounds_still_become_a_numeric_range() {
        let condition = to_qdrant_condition(
            &cond("field: language_score\nrange_from_query:\n  gte: ls_gte\n"),
            Some(&datetime_values()),
        )
        .unwrap();
        let expected = Condition::range(
            "language_score",
            QdrantRange { gt: None, gte: Some(0.5), lt: None, lte: None },
        );
        assert_eq!(condition, expected);
    }

    #[test]
    fn mixed_datetime_and_numeric_bounds_are_rejected() {
        let err = to_qdrant_condition(
            &cond("field: date\nrange_from_query:\n  gte: date_gte\n  lt: ls_gte\n"),
            Some(&datetime_values()),
        )
        .unwrap_err();
        assert!(err.to_string().contains("mixes datetime and numeric"), "{err}");
    }

    /// gRPC CANCELLED (what tonic's timeout layer emits as "Timeout expired")
    /// and DEADLINE_EXCEEDED are client-side timeouts; everything else is not.
    #[test]
    fn client_timeouts_are_classified_by_grpc_code() {
        use qdrant_client::QdrantError;
        let timeout_codes =
            [tonic::Status::cancelled("Timeout expired"), tonic::Status::deadline_exceeded("x")];
        for status in timeout_codes {
            assert!(is_client_timeout(&QdrantError::ResponseError { status }));
        }
        let not_timeouts = [
            tonic::Status::invalid_argument("Wrong input: Not existing vector name"),
            tonic::Status::unavailable("connect refused"),
            tonic::Status::not_found("no collection"),
        ];
        for status in not_timeouts {
            assert!(!is_client_timeout(&QdrantError::ResponseError { status }));
        }
    }

    #[test]
    fn a_non_datetime_string_bound_is_a_clear_error() {
        let values = HashMap::from([(
            "date_gte".to_string(),
            FilterFieldValue::Text("last tuesday".to_string()),
        )]);
        let err = to_qdrant_condition(
            &cond("field: date\nrange_from_query:\n  gte: date_gte\n"),
            Some(&values),
        )
        .unwrap_err();
        // Must be the datetime PARSER's message, not the generic "must be
        // numeric or..." fallback — a mutation deleting the Text arm entirely
        // was shown to survive a looser "RFC3339" substring.
        assert!(err.to_string().contains("not a recognized datetime"), "{err}");
    }

    /// One helper: the condition `field: date / range_from_query: gte: col`
    /// translated against a single text value.
    fn datetime_gte(value: &str) -> Result<Condition, TargetError> {
        let values =
            HashMap::from([("col".to_string(), FilterFieldValue::Text(value.to_string()))]);
        to_qdrant_condition(&cond("field: date\nrange_from_query:\n  gte: col\n"), Some(&values))
    }

    fn gte_timestamp(condition: Condition) -> Timestamp {
        use qdrant_client::qdrant::condition::ConditionOneOf;
        let ConditionOneOf::Field(f) = condition.condition_one_of.expect("field condition") else {
            panic!("not a field condition")
        };
        f.datetime_range.expect("datetime range").gte.expect("gte bound set")
    }

    #[test]
    fn pre_1970_datetimes_are_accepted_with_negative_seconds() {
        let ts = gte_timestamp(datetime_gte("1969-12-31T23:59:59Z").unwrap());
        assert_eq!((ts.seconds, ts.nanos), (-1, 0));
        // fractional pre-1970: euclidean split keeps nanos in 0..1e9
        let ts = gte_timestamp(datetime_gte("1969-12-31T23:59:59.5Z").unwrap());
        assert_eq!((ts.seconds, ts.nanos), (-1, 500_000_000));
    }

    #[test]
    fn nonzero_utc_offsets_are_accepted_and_normalized() {
        // 12:00 at +02:00 == 10:00Z
        let plus2 = gte_timestamp(datetime_gte("2017-09-19T12:00:00+02:00").unwrap());
        let utc = gte_timestamp(datetime_gte("2017-09-19T10:00:00Z").unwrap());
        assert_eq!(plus2, utc);
    }

    #[test]
    fn fractional_seconds_carry_into_nanos() {
        let ts = gte_timestamp(datetime_gte("2017-09-19T11:23:19.123Z").unwrap());
        assert_eq!((ts.seconds, ts.nanos), (T1.1, 123_000_000));
    }

    #[test]
    fn duckdb_space_separated_form_is_read_as_utc() {
        let space = gte_timestamp(datetime_gte("2017-09-19 11:23:19").unwrap());
        assert_eq!((space.seconds, space.nanos), (T1.1, 0));
        // and with an explicit offset after the space form
        let off = gte_timestamp(datetime_gte("2017-09-19 13:23:19+02:00").unwrap());
        assert_eq!(off.seconds, T1.1);
    }

    #[test]
    fn whitespace_padding_is_trimmed() {
        let ts = gte_timestamp(datetime_gte("  2017-09-19T11:23:19Z  ").unwrap());
        assert_eq!(ts.seconds, T1.1);
    }

    #[test]
    fn all_four_datetime_bounds_translate() {
        let values = HashMap::from([
            ("a".to_string(), FilterFieldValue::Text(T1.0.to_string())),
            ("b".to_string(), FilterFieldValue::Text(T2.0.to_string())),
        ]);
        let condition = to_qdrant_condition(
            &cond("field: date\nrange_from_query:\n  gt: a\n  gte: a\n  lt: b\n  lte: b\n"),
            Some(&values),
        )
        .unwrap();
        let expected = Condition::datetime_range(
            "date",
            DatetimeRange {
                gt: Some(Timestamp { seconds: T1.1, nanos: 0 }),
                gte: Some(Timestamp { seconds: T1.1, nanos: 0 }),
                lt: Some(Timestamp { seconds: T2.1, nanos: 0 }),
                lte: Some(Timestamp { seconds: T2.1, nanos: 0 }),
            },
        );
        assert_eq!(condition, expected);
    }
}
