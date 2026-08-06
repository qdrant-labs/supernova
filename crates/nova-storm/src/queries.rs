//! Loading the query-vector set.
//!
//! Storm cycles through a fixed set of query vectors pulled from a parquet file
//! (a local path or an `s3://` URI), read once at startup via DuckDB. The set is
//! held in memory and reused round-robin across the whole run, so this is a
//! one-shot synchronous read — there's nothing to stream or overlap. (Contrast
//! `nova-load`, which downloads huge corpus files; a small query set reads fine
//! straight from `s3://` via httpfs.)
//!
//! If `source.ground_truth_column` is set, each row's known-correct top-k point
//! ids are read alongside its vector in the *same* query — e.g. pointing
//! straight at `nova bf`'s own output parquet (`hit_ids`), which already carries
//! the query vector forward if `nova bf`'s `queries.payload_fields` included it.
//! Reading both columns from the same row sidesteps ever needing to join two
//! files by an independently-derived query id: the vector and its ground truth
//! arrive already paired, by construction.

use std::collections::{HashMap, HashSet};

use duckdb::Connection;
use duckdb::types::Value;

use crate::config::{QuerySource, VectorType};
use crate::errors::QueryLoadError;
use crate::filter::{Filter, FilterFieldValue};

/// A query vector in either modality. Sparse is the `{indices, values}` pair
/// qdrant's sparse search takes; [`sparse_vector`] guarantees by construction
/// that the lengths match, indices are unique (qdrant rejects duplicates),
/// and each index fits u32. Order is not normalized — qdrant sorts
/// server-side.
#[derive(Debug, Clone, PartialEq)]
pub enum VectorData {
    Dense(Vec<f32>),
    Sparse { indices: Vec<u32>, values: Vec<f32> },
}

impl VectorData {
    /// Dense dimensionality, or sparse non-zero count — "how big is this
    /// query," for logs and sanity checks.
    pub fn len(&self) -> usize {
        match self {
            VectorData::Dense(v) => v.len(),
            VectorData::Sparse { indices, .. } => indices.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The dense components, for targets that only speak dense.
    pub fn as_dense(&self) -> Option<&[f32]> {
        match self {
            VectorData::Dense(v) => Some(v),
            VectorData::Sparse { .. } => None,
        }
    }
}

/// One query vector plus its ground truth, if configured. Bundled into one
/// struct (rather than two parallel `Vec`s) so the two can never drift out of
/// index alignment. `ground_truth` is a `HashSet`, not the raw `Vec` the
/// column decodes to, so `recall_at_k` (called once per query *firing*, and
/// queries cycle round-robin through a fixed, reused set of these) doesn't
/// rebuild a set from the same ids over and over — it's built once, here, at
/// load time.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryVector {
    pub vector: VectorData,
    /// `None` when `ground_truth_column` isn't configured, or when this row's
    /// value is SQL NULL — either way, just "no known-correct answer for this
    /// query," not an error: the query still runs and contributes latency.
    pub ground_truth: Option<HashSet<String>>,
    /// This query's own resolved value for every queries column referenced by
    /// a `_from_query` condition in `query.filter` (see
    /// [`Filter::query_fields`]) — empty when no per-query filter is
    /// configured. Keyed by queries-column name, so a backend's translation
    /// (e.g. `targets::qdrant::to_qdrant_condition`) can look up exactly the
    /// column a given `FilterCondition` names.
    pub filter_values: HashMap<String, FilterFieldValue>,
}

/// Read up to `source.limit` query vectors (and, if configured, each one's
/// ground truth and/or per-query filter values) from `source.uri`, decoding
/// the vector column per `vector_type`. Rows whose vector column is NULL *or
/// empty* (zero-length list, or a sparse struct with NULL/empty `indices`)
/// are excluded in SQL, before `LIMIT`, so they never consume the query
/// budget; a NULL ground-truth value on an otherwise-valid row just means
/// that query has no known-correct answer. Values in `range_from_query`
/// columns are validated (and datetimes normalized to RFC3339) here at load
/// time — a malformed bound fails the run up front, never per-dispatch.
///
/// `filter` is `query.filter`, if configured — when it has any `_from_query`
/// condition, this also projects every queries column those conditions name
/// (see [`Filter::query_fields`]) into each [`QueryVector::filter_values`].
/// Unlike a NULL ground-truth value, a NULL in one of *these* columns is a
/// hard error: nova-bf's own convention for "no filter value for this query"
/// is a non-matching placeholder (e.g. its MS MARCO config's unused
/// `domain_slot_N` columns hold `"zzznomatchzzz000"`), not NULL, so a NULL
/// here means the queries file wasn't built for this filter.
/// Single-quote-escape a path/URI for inclusion in a DuckDB string literal.
/// Config is operator-authored (trusted), but a path like `/data/o'brien/`
/// must not be a parse error.
fn sql_str(raw: &str) -> String {
    raw.replace('\'', "''")
}

pub fn load_query_vectors(
    source: &QuerySource,
    vector_type: VectorType,
    filter: Option<&Filter>,
) -> Result<Vec<QueryVector>, QueryLoadError> {
    let conn = Connection::open_in_memory()?;
    // httpfs lets DuckDB read `s3://` (and `http(s)://`); harmless for local paths.
    conn.execute_batch("INSTALL httpfs; LOAD httpfs;")?;
    configure_s3(&conn)?;

    let filter_columns: Vec<&str> =
        filter.map(|f| f.query_fields().into_iter().collect()).unwrap_or_default();
    let range_columns = filter.map(|f| f.range_columns()).unwrap_or_default();

    // Config is operator-authored (trusted). Columns are quoted so names with
    // odd characters survive. `cols` is the single source of truth for both the
    // SQL projection and "how many/which columns to read per row" below.
    let mut cols: Vec<&str> = vec![source.column.as_str()];
    cols.extend(source.ground_truth_column.as_deref());
    let has_gt = source.ground_truth_column.is_some();
    cols.extend(filter_columns.iter().copied());

    let projection = cols.iter().map(|c| format!("\"{c}\"")).collect::<Vec<_>>().join(", ");
    // Preflight the vector column's TYPE so a dense/sparse config mismatch
    // fails with a pointable message here — without this, the shape-dependent
    // SQL predicate below dies first, as an opaque DuckDB binder error.
    let column_type: String = conn
        .query_row(
            &format!(
                "SELECT column_type FROM (DESCRIBE SELECT \"{col}\" FROM read_parquet('{uri}'))",
                col = source.column,
                uri = sql_str(&source.uri),
            ),
            [],
            |row| row.get(0),
        )?;
    // A `STRUCT(...)[]` (LIST of structs) is neither modality — don't let the
    // prefix check misread it as sparse.
    let type_trimmed = column_type.trim();
    let is_struct = type_trimmed.starts_with("STRUCT") && !type_trimmed.ends_with("[]");
    match vector_type {
        VectorType::Sparse if !is_struct => {
            return Err(QueryLoadError::Other(format!(
                "sparse query column `{}` is `{column_type}`, not a struct{{indices, values}} — \
                 is this column really sparse, or should query.vector_type be `dense`?",
                source.column
            )));
        }
        // Right shape, wrong field names ("idx"/"vals"): caught here with the
        // full type string, instead of as an opaque binder error from the
        // struct_extract in the main query below.
        VectorType::Sparse
            if !(column_type.contains("indices") && column_type.contains("values")) =>
        {
            return Err(QueryLoadError::Other(format!(
                "sparse query column `{}` is `{column_type}` — a sparse struct needs `indices` \
                 and `values` fields",
                source.column
            )));
        }
        VectorType::Dense if is_struct => {
            return Err(QueryLoadError::Other(format!(
                "query column `{}` is `{column_type}`, not a list of floats — should \
                 query.vector_type be `sparse`?",
                source.column
            )));
        }
        _ => {}
    }

    // Rows with nothing to query are excluded in SQL, BEFORE `LIMIT`, so they
    // never consume the query budget: a file of [valid, empty, valid] with
    // limit 2 must yield 2 queries, not 1. For sparse that also covers the
    // struct-of-NULL-fields shape (`{indices: NULL, values: NULL}`), which is
    // NOT SQL NULL itself — `coalesce(len(...), 0)` treats it as empty. The
    // Rust-side skip below stays as a backstop for shapes SQL can't see.
    let non_empty = match vector_type {
        VectorType::Dense => format!("coalesce(len(\"{col}\"), 0) > 0", col = source.column),
        VectorType::Sparse => format!(
            "coalesce(len(struct_extract(\"{col}\", 'indices')), 0) > 0",
            col = source.column
        ),
    };
    let sql = format!(
        "SELECT {projection} FROM read_parquet('{uri}') WHERE \"{col}\" IS NOT NULL AND {non_empty} LIMIT {limit}",
        uri = sql_str(&source.uri),
        col = source.column,
        limit = source.limit,
    );

    let n_cols = cols.len();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], move |row| {
        (0..n_cols).map(|i| row.get::<_, Value>(i)).collect::<duckdb::Result<Vec<_>>>()
    })?;

    let mut out = Vec::new();
    let mut empty_skipped = 0usize;
    for row in rows {
        let mut values = row?.into_iter();
        let vector = values.next().expect("vector column always present");
        let vector = match vector_type {
            VectorType::Dense => {
                let dense = float_list(vector)?;
                if dense.is_empty() {
                    empty_skipped += 1;
                    continue;
                }
                VectorData::Dense(dense)
            }
            VectorType::Sparse => {
                let (indices, values) = sparse_vector(vector)?;
                // An all-zero query (e.g. a blank document whose every term
                // was pruned) has nothing to search with. The SQL predicate
                // above already excludes these pre-LIMIT; this is the backstop
                // for anything it can't see, and the count keeps a file full
                // of them visible.
                if indices.is_empty() {
                    empty_skipped += 1;
                    continue;
                }
                VectorData::Sparse { indices, values }
            }
        };
        let ground_truth = if has_gt { values.next() } else { None };
        let ground_truth = match ground_truth {
            None | Some(Value::Null) => None,
            Some(v) => Some(string_list(v)?.into_iter().collect::<HashSet<_>>()),
        };

        // Everything remaining, in the same order as `filter_columns`.
        let mut filter_values = HashMap::with_capacity(filter_columns.len());
        for (name, value) in filter_columns.iter().zip(values) {
            if matches!(value, Value::Null) {
                return Err(QueryLoadError::Other(format!(
                    "filter column `{name}` is NULL for a query row — `_from_query` columns must \
                     never be NULL; use a non-matching placeholder value instead (e.g. nova-bf's \
                     `domain_slot_N` convention)"
                )));
            }
            let is_range_bound = range_columns.contains(*name);
            filter_values.insert(
                (*name).to_string(),
                filter_field_value(value, is_range_bound).map_err(|e| {
                    QueryLoadError::Other(format!("filter column `{name}`: {e}"))
                })?,
            );
        }

        out.push(QueryVector { vector, ground_truth, filter_values });
    }
    if empty_skipped > 0 {
        tracing::warn!(
            "skipped {empty_skipped} query row(s) with an empty vector (nothing to search with)"
        );
    }
    Ok(out)
}

/// Decode one sparse query value: a DuckDB `STRUCT(indices <int list>,
/// values <float list>)` — the shape the fineweb corpus's `sparse_embedding`
/// and `nova bf`'s sparse query files both carry. Field lookup is by NAME,
/// not position, so `{values, indices}` order in the source schema is fine.
/// Indices must be non-negative and fit `u32` (qdrant's sparse index domain);
/// the two lists must be the same length.
fn sparse_vector(value: Value) -> Result<(Vec<u32>, Vec<f32>), QueryLoadError> {
    let Value::Struct(fields) = value else {
        return Err(QueryLoadError::Other(
            "sparse query column is not a struct{indices, values} — is this column really \
             sparse, or should query.vector_type be `dense`?"
                .into(),
        ));
    };
    let field = |name: &str| {
        fields.iter().find(|(k, _)| k == name).map(|(_, v)| v).ok_or_else(|| {
            let seen: Vec<&str> = fields.iter().map(|(k, _)| k.as_str()).collect();
            QueryLoadError::Other(format!(
                "sparse struct has no `{name}` field (fields present: {seen:?})"
            ))
        })
    };

    // `{indices: NULL, values: NULL}` is a common writer spelling of "no
    // sparse embedding" — the struct itself is NOT SQL NULL, so it survives
    // the loader's NULL filter. Treat it as empty (the caller skips empties)
    // rather than aborting a whole run on one such row.
    if matches!(field("indices")?, Value::Null) && matches!(field("values")?, Value::Null) {
        return Ok((Vec::new(), Vec::new()));
    }

    let indices = match field("indices")? {
        Value::List(xs) | Value::Array(xs) => xs
            .iter()
            .map(|v| {
                int_value(v)
                    .and_then(|i| u32::try_from(i).ok())
                    .ok_or_else(|| {
                        QueryLoadError::Other(format!(
                            "sparse index `{v:?}` is not a non-negative integer fitting u32"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?,
        other => {
            return Err(QueryLoadError::Other(format!(
                "sparse struct `indices` is not a list (got {other:?})"
            )));
        }
    };

    let values = match field("values")? {
        Value::Null => {
            return Err(QueryLoadError::Other(format!(
                "sparse row has {} indices but a NULL `values` field — corrupt row? (a row with \
                 BOTH fields NULL is treated as \"no embedding\" and skipped)",
                indices.len()
            )));
        }
        Value::List(xs) | Value::Array(xs) => xs
            .iter()
            .map(|v| {
                // floats and ints both make sense as sparse weights (BM25-style
                // term frequencies are integer-typed); NULL elements do not.
                float(v).or_else(|| int_value(v).map(|i| i as f32)).ok_or_else(|| {
                    QueryLoadError::Other(format!(
                        "sparse struct `values` element `{v:?}` is not a number \
                         (a NULL inside the list?)"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        other => {
            return Err(QueryLoadError::Other(format!(
                "sparse struct `values` is not a list (got {other:?})"
            )));
        }
    };

    if indices.len() != values.len() {
        return Err(QueryLoadError::Other(format!(
            "sparse query has {} indices but {} values — corrupt row?",
            indices.len(),
            values.len()
        )));
    }
    // Qdrant requires unique sparse indices (validate_sparse_vector_impl) and
    // rejects duplicates per-request — catching it here fails the load with a
    // pointable error instead of a full-duration 100%-error run. (Order does
    // NOT matter: qdrant sorts queries server-side.)
    let mut unique = std::collections::HashSet::with_capacity(indices.len());
    if let Some(dup) = indices.iter().find(|i| !unique.insert(**i)) {
        return Err(QueryLoadError::Other(format!(
            "sparse query has duplicate index {dup}"
        )));
    }
    Ok((indices, values))
}

/// Point DuckDB's httpfs at S3 using the standard AWS environment variables.
fn configure_s3(conn: &Connection) -> Result<(), QueryLoadError> {
    let region = std::env::var("AWS_REGION")
        .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
        .unwrap_or_else(|_| "us-east-1".to_string());
    conn.execute_batch(&format!("SET s3_region='{region}';"))?;

    let key = std::env::var("AWS_ACCESS_KEY_ID").unwrap_or_default();
    let secret = std::env::var("AWS_SECRET_ACCESS_KEY").unwrap_or_default();
    if !key.is_empty() && !secret.is_empty() {
        conn.execute_batch(&format!(
            "SET s3_access_key_id='{key}'; SET s3_secret_access_key='{secret}';"
        ))?;
    }
    if let Ok(token) = std::env::var("AWS_SESSION_TOKEN")
        && !token.is_empty()
    {
        conn.execute_batch(&format!("SET s3_session_token='{token}';"))?;
    }
    Ok(())
}

/// Coerce a DuckDB `LIST`/`ARRAY` of floats into a `Vec<f32>`.
fn float_list(value: Value) -> Result<Vec<f32>, QueryLoadError> {
    match value {
        Value::List(xs) | Value::Array(xs) => xs
            .iter()
            .map(float)
            .collect::<Option<_>>()
            .ok_or_else(|| QueryLoadError::Other("query column is not a list of floats".into())),
        _ => Err(QueryLoadError::Other("query column is not a list of floats".into())),
    }
}

/// Coerce a DuckDB float/double value to `f32`.
fn float(v: &Value) -> Option<f32> {
    match *v {
        Value::Float(f) => Some(f),
        Value::Double(d) => Some(d as f32),
        _ => None,
    }
}

/// Coerce a DuckDB numeric value (any int width, signed or unsigned, or
/// float/double/decimal) to `f64`. `None` for anything else. Used for range
/// bounds, which are `f64` regardless of the source column's exact type
/// (`RangeCondition`/`RangeFromQuery` are `f64`-based like `nova-bf`'s own —
/// see `crate::filter`), so widening an integer column here is fine even
/// above `f64`'s 2^53 exact-integer limit.
fn numeric(v: &Value) -> Option<f64> {
    match v {
        Value::Float(f) => Some(*f as f64),
        Value::Double(d) => Some(*d),
        // `rust_decimal::Decimal` isn't a direct dependency of this crate —
        // going through its `Display` impl avoids needing one just for this.
        Value::Decimal(d) => d.to_string().parse().ok(),
        _ => int_value(v).map(|i| i as f64),
    }
}

/// Every DuckDB integer-width variant, signed or unsigned.
fn is_integer_value(v: &Value) -> bool {
    matches!(
        v,
        Value::TinyInt(_)
            | Value::SmallInt(_)
            | Value::Int(_)
            | Value::BigInt(_)
            | Value::HugeInt(_)
            | Value::UTinyInt(_)
            | Value::USmallInt(_)
            | Value::UInt(_)
            | Value::UBigInt(_)
    )
}

/// Coerce a DuckDB integer value to `i64` *exactly* — every int width,
/// signed or unsigned, with no precision loss (unlike widening straight to
/// `f64`, which silently loses precision above 2^53 — the bug this exists to
/// avoid for an exact-equality `match_from_query` id). `None` for a value
/// whose magnitude doesn't fit `i64` (an out-of-range `HugeInt`/`UBigInt`) or
/// a non-integer `Value`.
fn int_value(v: &Value) -> Option<i64> {
    match *v {
        Value::TinyInt(i) => Some(i as i64),
        Value::SmallInt(i) => Some(i as i64),
        Value::Int(i) => Some(i as i64),
        Value::BigInt(i) => Some(i),
        Value::HugeInt(i) => i64::try_from(i).ok(),
        Value::UTinyInt(i) => Some(i as i64),
        Value::USmallInt(i) => Some(i as i64),
        Value::UInt(i) => Some(i as i64),
        Value::UBigInt(i) => i64::try_from(i).ok(),
        _ => None,
    }
}

/// A DuckDB TIMESTAMP value as an RFC3339 UTC string (DuckDB timestamps are
/// epoch-anchored and offsetless). Kept as text so `FilterFieldValue` stays
/// backend-agnostic
fn timestamp_rfc3339(unit: duckdb::types::TimeUnit, raw: i64) -> Result<String, QueryLoadError> {
    use duckdb::types::TimeUnit;
    let nanos: i128 = match unit {
        TimeUnit::Second => i128::from(raw) * 1_000_000_000,
        TimeUnit::Millisecond => i128::from(raw) * 1_000_000,
        TimeUnit::Microsecond => i128::from(raw) * 1_000,
        TimeUnit::Nanosecond => i128::from(raw),
    };
    let t = time::OffsetDateTime::from_unix_timestamp_nanos(nanos).map_err(|e| {
        QueryLoadError::Other(format!("timestamp filter column value out of range: {e}"))
    })?;
    crate::datetime::to_rfc3339(t).map_err(QueryLoadError::Other)
}

/// A DuckDB DATE value (days since epoch) as RFC3339 midnight UTC.
fn date32_rfc3339(days: i32) -> Result<String, QueryLoadError> {
    let t = time::OffsetDateTime::from_unix_timestamp(i64::from(days) * 86_400)
        .map_err(|e| QueryLoadError::Other(format!("date filter column value out of range: {e}")))?;
    crate::datetime::to_rfc3339(t).map_err(QueryLoadError::Other)
}

/// Coerce one `_from_query`-referenced column's value into a
/// [`FilterFieldValue`] — text, an exact integer, a float/decimal, or a
/// homogeneous list of one of those. Callers already reject `Value::Null`
/// before this is called (see the NULL check in [`load_query_vectors`]).
fn filter_field_value(value: Value, is_range_bound: bool) -> Result<FilterFieldValue, QueryLoadError> {
    // RANGE-bound columns get datetime handling AT LOAD TIME: text is parsed
    // now (a malformed bound fails the run before any load is offered, not as
    // a full-duration per-dispatch error loop) and stored as canonical
    // RFC3339; TIMESTAMP/DATE columns are rendered to the same. MATCH columns
    // keep the strict pre-existing rules — a datetime value there is an
    // error, not a silent keyword-match-on-a-timestamp-string that can never
    // hit.
    if is_range_bound {
        match &value {
            // numeric-looking text stays a numeric bound; anything else must
            // be a datetime.
            Value::Text(s) => {
                if let Ok(n) = s.trim().parse::<f64>() {
                    return Ok(FilterFieldValue::Num(n));
                }
                let parsed = crate::datetime::parse_datetime_utc(s).map_err(|e| {
                    QueryLoadError::Other(format!(
                        "range bound value `{s}` is neither numeric nor a recognized datetime \
                         (RFC3339, `YYYY-MM-DD HH:MM:SS`, or date-only): {e}"
                    ))
                })?;
                let normalized = crate::datetime::to_rfc3339(parsed)
                    .map_err(QueryLoadError::Other)?;
                return Ok(FilterFieldValue::Text(normalized));
            }
            Value::Timestamp(unit, raw) => {
                return timestamp_rfc3339(*unit, *raw).map(FilterFieldValue::Text);
            }
            Value::Date32(days) => {
                return date32_rfc3339(*days).map(FilterFieldValue::Text);
            }
            _ => {}
        }
    } else if matches!(&value, Value::Timestamp(..) | Value::Date32(_)) {
        return Err(QueryLoadError::Other(
            "a TIMESTAMP/DATE column can only feed `range_from_query` (qdrant match has no \
             datetime); for an exact-moment condition use a range with equal gte/lte"
                .into(),
        ));
    }
    if let Value::Text(s) = &value {
        return Ok(FilterFieldValue::Text(s.clone()));
    }

    if let Value::List(xs) | Value::Array(xs) = &value {
        return filter_field_list_value(xs);
    }
    if is_integer_value(&value) {
        return int_value(&value).map(FilterFieldValue::Int).ok_or_else(|| {
            QueryLoadError::Other(format!(
                "filter column value `{value:?}` is out of range for an exact integer match \
                 (Qdrant match only supports i64)"
            ))
        });
    }
    numeric(&value).map(FilterFieldValue::Num).ok_or_else(|| {
        QueryLoadError::Other("filter column is not text, numeric, or a list of either".into())
    })
}

/// [`filter_field_value`]'s list case: a `LIST`/`ARRAY` of all-text, all-
/// integer (kept exact, same reasoning as [`int_value`]), or otherwise
/// all-numeric (widened to `f64`, e.g. a mix of int and double columns —
/// unusual but not actively wrong for `range_from_query`, which is `f64`
/// anyway).
fn filter_field_list_value(xs: &[Value]) -> Result<FilterFieldValue, QueryLoadError> {
    if xs.iter().all(|v| matches!(v, Value::Text(_))) {
        return Ok(FilterFieldValue::TextList(
            xs.iter()
                .filter_map(|v| match v {
                    Value::Text(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
        ));
    }
    if xs.iter().all(is_integer_value) {
        return xs.iter().map(int_value).collect::<Option<Vec<_>>>().map(FilterFieldValue::IntList).ok_or_else(
            || QueryLoadError::Other("filter column list has an integer value out of range for i64".into()),
        );
    }
    xs.iter().map(numeric).collect::<Option<Vec<_>>>().map(FilterFieldValue::NumList).ok_or_else(|| {
        QueryLoadError::Other("filter column list is neither all-text nor all-numeric".into())
    })
}

/// Coerce a DuckDB `LIST`/`ARRAY` of ids into a `Vec<String>` — the same
/// shape as [`float_list`], for a ground-truth `hit_ids` column instead of a
/// vector column. Ids are commonly integers (a corpus row number) rather than
/// strings, so any integer width is stringified the same way point ids
/// eventually get compared as strings elsewhere in storm.
fn string_list(value: Value) -> Result<Vec<String>, QueryLoadError> {
    match value {
        Value::List(xs) | Value::Array(xs) => xs
            .iter()
            .map(id_string)
            .collect::<Option<_>>()
            .ok_or_else(|| {
                QueryLoadError::Other("ground_truth column is not a list of strings or integers".into())
            }),
        _ => Err(QueryLoadError::Other("ground_truth column is not a list of strings or integers".into())),
    }
}

/// Stringify a single ground-truth id: text passes through, any integer width
/// (signed or unsigned) is formatted as its decimal string. Anything else
/// (float, bool, blob, ...) isn't a sensible id and is rejected.
fn id_string(v: &Value) -> Option<String> {
    match v {
        Value::Text(s) => Some(s.clone()),
        Value::TinyInt(i) => Some(i.to_string()),
        Value::SmallInt(i) => Some(i.to_string()),
        Value::Int(i) => Some(i.to_string()),
        Value::BigInt(i) => Some(i.to_string()),
        Value::HugeInt(i) => Some(i.to_string()),
        Value::UTinyInt(i) => Some(i.to_string()),
        Value::USmallInt(i) => Some(i.to_string()),
        Value::UInt(i) => Some(i.to_string()),
        Value::UBigInt(i) => Some(i.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_parquet(path: &std::path::Path, rows: usize) {
        let conn = Connection::open_in_memory().unwrap();
        // A fixed 3-element float list per row: [i, i+1, i+2]. The path is
        // quote-escaped so tests can use adversarial directory names.
        conn.execute_batch(&format!(
            "COPY (SELECT [i::FLOAT, (i + 1)::FLOAT, (i + 2)::FLOAT] AS embedding \
             FROM range({rows}) r(i)) TO '{}' (FORMAT PARQUET)",
            sql_str(&path.display().to_string())
        ))
        .unwrap();
    }

    /// Like [`write_parquet`], plus a `hit_ids` column: row `i` gets
    /// `["gt-{i}-a", "gt-{i}-b"]`, except row 0 which gets a NULL ground truth
    /// (to exercise the "no known-correct answer for this one" path).
    fn write_parquet_with_ground_truth(path: &std::path::Path, rows: usize) {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "COPY (SELECT [i::FLOAT, (i + 1)::FLOAT, (i + 2)::FLOAT] AS embedding, \
             CASE WHEN i = 0 THEN NULL ELSE ['gt-' || i || '-a', 'gt-' || i || '-b'] END AS hit_ids \
             FROM range({rows}) r(i)) TO '{}' (FORMAT PARQUET)",
            path.display()
        ))
        .unwrap();
    }

    fn source(uri: String, ground_truth_column: Option<&str>) -> QuerySource {
        QuerySource {
            uri,
            column: "embedding".into(),
            limit: 10,
            ground_truth_column: ground_truth_column.map(str::to_string),
        }
    }

    #[test]
    fn loads_and_limits_query_vectors() {
        let dir = std::env::temp_dir().join(format!("nova_storm_q_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("queries.parquet");
        write_parquet(&file, 100);

        let vectors = load_query_vectors(&source(file.display().to_string(), None), VectorType::Dense, None).unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(vectors.len(), 10); // LIMIT honoured
        assert!(vectors.iter().all(|v| v.vector.len() == 3)); // full dim per row
        assert!(vectors.iter().all(|v| v.ground_truth.is_none())); // no column configured
    }

    #[test]
    fn loads_ground_truth_column_when_configured() {
        let dir = std::env::temp_dir().join(format!("nova_storm_qgt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("queries.parquet");
        write_parquet_with_ground_truth(&file, 5);

        let vectors =
            load_query_vectors(&source(file.display().to_string(), Some("hit_ids")), VectorType::Dense, None).unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(vectors.len(), 5);
        // row 0's ground truth was NULL -> None, not an error, not skipped.
        assert_eq!(vectors[0].ground_truth, None);
        assert_eq!(
            vectors[1].ground_truth,
            Some(HashSet::from(["gt-1-a".to_string(), "gt-1-b".to_string()]))
        );
        assert_eq!(
            vectors[4].ground_truth,
            Some(HashSet::from(["gt-4-a".to_string(), "gt-4-b".to_string()]))
        );
    }

    #[test]
    fn ground_truth_column_accepts_integer_ids() {
        let dir = std::env::temp_dir().join(format!("nova_storm_qgtint_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("queries.parquet");
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "COPY (SELECT [i::FLOAT, (i + 1)::FLOAT, (i + 2)::FLOAT] AS embedding, \
             [i, i + 1]::BIGINT[] AS hit_ids FROM range(3) r(i)) TO '{}' (FORMAT PARQUET)",
            file.display()
        ))
        .unwrap();

        let vectors =
            load_query_vectors(&source(file.display().to_string(), Some("hit_ids")), VectorType::Dense, None).unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(vectors.len(), 3);
        assert_eq!(
            vectors[1].ground_truth,
            Some(HashSet::from(["1".to_string(), "2".to_string()]))
        );
    }

    #[test]
    fn ground_truth_column_wrong_type_errors() {
        let dir = std::env::temp_dir().join(format!("nova_storm_qgtbad_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("queries.parquet");
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "COPY (SELECT [i::FLOAT, (i + 1)::FLOAT, (i + 2)::FLOAT] AS embedding, \
             i AS hit_ids FROM range(3) r(i)) TO '{}' (FORMAT PARQUET)",
            file.display()
        ))
        .unwrap();

        let result = load_query_vectors(&source(file.display().to_string(), Some("hit_ids")), VectorType::Dense, None);

        std::fs::remove_dir_all(&dir).ok();

        assert!(result.is_err());
    }

    fn parse_filter(yaml: &str) -> Filter {
        serde_yaml::from_str(yaml).expect("parses")
    }

    #[test]
    fn loads_per_query_filter_columns_into_filter_values() {
        let dir = std::env::temp_dir().join(format!("nova_storm_qf_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("queries.parquet");
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "COPY (SELECT [i::FLOAT, (i + 1)::FLOAT, (i + 2)::FLOAT] AS embedding, \
             'tenant-' || i AS tenant_column, (i * 10)::DOUBLE AS max_budget \
             FROM range(3) r(i)) TO '{}' (FORMAT PARQUET)",
            file.display()
        ))
        .unwrap();

        let filter = parse_filter(
            "must:\n  - field: tenant_id\n    match_from_query: tenant_column\n  - field: budget\n    range_from_query:\n      lt: max_budget\n",
        );
        let vectors =
            load_query_vectors(&source(file.display().to_string(), None), VectorType::Dense, Some(&filter)).unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(vectors.len(), 3);
        assert_eq!(
            vectors[1].filter_values.get("tenant_column"),
            Some(&FilterFieldValue::Text("tenant-1".to_string()))
        );
        assert_eq!(
            vectors[2].filter_values.get("max_budget"),
            Some(&FilterFieldValue::Num(20.0))
        );
    }

    #[test]
    fn null_in_a_from_query_column_is_a_hard_error() {
        let dir = std::env::temp_dir().join(format!("nova_storm_qfnull_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("queries.parquet");
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "COPY (SELECT [i::FLOAT, (i + 1)::FLOAT, (i + 2)::FLOAT] AS embedding, \
             CASE WHEN i = 0 THEN NULL ELSE 'tenant-' || i END AS tenant_column \
             FROM range(3) r(i)) TO '{}' (FORMAT PARQUET)",
            file.display()
        ))
        .unwrap();

        let filter =
            parse_filter("must:\n  - field: tenant_id\n    match_from_query: tenant_column\n");
        let result =
            load_query_vectors(&source(file.display().to_string(), None), VectorType::Dense, Some(&filter));

        std::fs::remove_dir_all(&dir).ok();

        assert!(result.is_err());
    }

    #[test]
    fn large_bigint_filter_column_round_trips_exactly() {
        // Above f64's 2^53 exact-integer limit (9007199254740992) -- widening
        // straight to f64 before converting back to i64 would silently round
        // this to 9007199254740992, submitting the WRONG tenant id to Qdrant
        // with no error. `filter_field_value` must keep it exact via `Int`.
        const BIG: i64 = 9_007_199_254_740_993;
        let dir = std::env::temp_dir().join(format!("nova_storm_qfbig_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("queries.parquet");
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "COPY (SELECT [1.0::FLOAT] AS embedding, {BIG}::BIGINT AS tenant_column) \
             TO '{}' (FORMAT PARQUET)",
            file.display()
        ))
        .unwrap();

        let filter =
            parse_filter("must:\n  - field: tenant_id\n    match_from_query: tenant_column\n");
        let vectors =
            load_query_vectors(&source(file.display().to_string(), None), VectorType::Dense, Some(&filter)).unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(vectors[0].filter_values.get("tenant_column"), Some(&FilterFieldValue::Int(BIG)));
    }

    #[test]
    fn decimal_filter_column_loads_as_numeric() {
        let dir = std::env::temp_dir().join(format!("nova_storm_qfdec_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("queries.parquet");
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "COPY (SELECT [1.0::FLOAT] AS embedding, 42.5::DECIMAL(10,2) AS max_budget) \
             TO '{}' (FORMAT PARQUET)",
            file.display()
        ))
        .unwrap();

        let filter =
            parse_filter("must:\n  - field: budget\n    range_from_query:\n      lt: max_budget\n");
        let vectors =
            load_query_vectors(&source(file.display().to_string(), None), VectorType::Dense, Some(&filter)).unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(vectors[0].filter_values.get("max_budget"), Some(&FilterFieldValue::Num(42.5)));
    }

    // ---------------------------------------------------------------- sparse

    /// A scratch parquet whose `embedding` column is the sparse
    /// `struct{indices, values}` shape, built from a SQL body so each test
    /// controls the exact rows.
    fn write_sparse_parquet(path: &std::path::Path, select_body: &str) {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "COPY ({select_body}) TO '{}' (FORMAT PARQUET)",
            path.display()
        ))
        .unwrap();
    }

    fn in_dir(name: &str, f: impl FnOnce(&std::path::Path)) {
        let dir = std::env::temp_dir().join(format!("nova_storm_{name}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        f(&dir);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn loads_sparse_query_vectors() {
        in_dir("sparse_ok", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT {'indices': [i::BIGINT, (i + 100)::BIGINT], 'values': [0.5::DOUBLE, 0.25::DOUBLE]} AS embedding \
                 FROM range(3) r(i)",
            );
            let vectors =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Sparse, None)
                    .unwrap();
            assert_eq!(vectors.len(), 3);
            assert_eq!(
                vectors[1].vector,
                VectorData::Sparse { indices: vec![1, 101], values: vec![0.5, 0.25] }
            );
        });
    }

    #[test]
    fn skips_empty_sparse_rows() {
        in_dir("sparse_empty", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT CASE WHEN i = 1 \
                    THEN {'indices': []::BIGINT[], 'values': []::DOUBLE[]} \
                    ELSE {'indices': [i::BIGINT], 'values': [1.0::DOUBLE]} END AS embedding \
                 FROM range(3) r(i)",
            );
            let vectors =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Sparse, None)
                    .unwrap();
            // Row 1 (empty) is skipped; rows 0 and 2 survive.
            assert_eq!(vectors.len(), 2);
            assert!(vectors.iter().all(|v| !v.vector.is_empty()));
        });
    }

    #[test]
    fn sparse_type_on_a_dense_column_is_a_clear_error() {
        in_dir("sparse_on_dense", |dir| {
            let file = dir.join("q.parquet");
            write_parquet(&file, 2); // plain list<float> column
            let err =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Sparse, None)
                    .unwrap_err();
            assert!(err.to_string().contains("not a struct"), "{err}");
        });
    }

    #[test]
    fn dense_type_on_a_sparse_column_is_a_clear_error() {
        in_dir("dense_on_sparse", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT {'indices': [1::BIGINT], 'values': [1.0::DOUBLE]} AS embedding",
            );
            let err =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Dense, None)
                    .unwrap_err();
            assert!(err.to_string().contains("not a list of floats"), "{err}");
        });
    }

    #[test]
    fn negative_sparse_index_is_rejected() {
        in_dir("sparse_neg", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT {'indices': [-1::BIGINT], 'values': [1.0::DOUBLE]} AS embedding",
            );
            let err =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Sparse, None)
                    .unwrap_err();
            assert!(err.to_string().contains("non-negative"), "{err}");
        });
    }

    #[test]
    fn sparse_index_above_u32_is_rejected() {
        in_dir("sparse_big", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT {'indices': [4294967296::BIGINT], 'values': [1.0::DOUBLE]} AS embedding",
            );
            let err =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Sparse, None)
                    .unwrap_err();
            assert!(err.to_string().contains("u32"), "{err}");
        });
    }

    #[test]
    fn mismatched_sparse_lengths_are_rejected() {
        in_dir("sparse_mismatch", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT {'indices': [1::BIGINT, 2::BIGINT], 'values': [1.0::DOUBLE]} AS embedding",
            );
            let err =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Sparse, None)
                    .unwrap_err();
            assert!(err.to_string().contains("2 indices but 1 values"), "{err}");
        });
    }

    #[test]
    fn sparse_rows_keep_ground_truth_alignment() {
        in_dir("sparse_gt", |dir| {
            let file = dir.join("q.parquet");
            // Row 1 is empty-sparse (skipped); its ground truth must vanish with
            // it rather than shifting onto row 2.
            write_sparse_parquet(
                &file,
                "SELECT CASE WHEN i = 1 \
                    THEN {'indices': []::BIGINT[], 'values': []::DOUBLE[]} \
                    ELSE {'indices': [i::BIGINT], 'values': [1.0::DOUBLE]} END AS embedding, \
                 ['gt-' || i] AS hit_ids FROM range(3) r(i)",
            );
            let vectors = load_query_vectors(
                &source(file.display().to_string(), Some("hit_ids")),
                VectorType::Sparse,
                None,
            )
            .unwrap();
            assert_eq!(vectors.len(), 2);
            assert!(vectors[0].ground_truth.as_ref().unwrap().contains("gt-0"));
            assert!(vectors[1].ground_truth.as_ref().unwrap().contains("gt-2"));
        });
    }

    /// Real sparse vocabularies live far above 255, and the exact top of the
    /// u32 domain must round-trip — a narrowing regression (e.g. through u8)
    /// must fail here.
    #[test]
    fn u32_max_sparse_index_is_accepted() {
        in_dir("sparse_u32max", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT {'indices': [30000::BIGINT, 4294967295::BIGINT], 'values': [1.0::DOUBLE, 2.0::DOUBLE]} AS embedding",
            );
            let vectors =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Sparse, None)
                    .unwrap();
            assert_eq!(
                vectors[0].vector,
                VectorData::Sparse { indices: vec![30_000, u32::MAX], values: vec![1.0, 2.0] }
            );
        });
    }

    /// The struct's field ORDER must not matter, only the names — a
    /// positional-lookup regression must fail here.
    #[test]
    fn reversed_struct_field_order_still_decodes() {
        in_dir("sparse_reversed", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT {'values': [7.0::DOUBLE], 'indices': [3::BIGINT]} AS embedding",
            );
            let vectors =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Sparse, None)
                    .unwrap();
            assert_eq!(
                vectors[0].vector,
                VectorData::Sparse { indices: vec![3], values: vec![7.0] }
            );
        });
    }

    #[test]
    fn duplicate_sparse_indices_are_rejected_at_load() {
        in_dir("sparse_dup", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT {'indices': [5::BIGINT, 5::BIGINT], 'values': [1.0::DOUBLE, 2.0::DOUBLE]} AS embedding",
            );
            let err =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Sparse, None)
                    .unwrap_err();
            assert!(err.to_string().contains("duplicate index 5"), "{err}");
        });
    }

    /// A struct whose FIELDS are NULL is not SQL NULL — it must be skipped
    /// like an empty row, not abort the load.
    #[test]
    fn null_field_struct_rows_are_skipped_not_fatal() {
        in_dir("sparse_nullfields", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT CASE WHEN i = 1 \
                    THEN {'indices': NULL::BIGINT[], 'values': NULL::DOUBLE[]} \
                    ELSE {'indices': [i::BIGINT], 'values': [1.0::DOUBLE]} END AS embedding \
                 FROM range(3) r(i)",
            );
            let vectors =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Sparse, None)
                    .unwrap();
            assert_eq!(vectors.len(), 2);
        });
    }

    /// Empty rows must not consume the LIMIT budget: [valid, empty, valid,
    /// empty, valid] with limit 3 yields 3 queries, not 2.
    #[test]
    fn empty_rows_do_not_consume_the_limit_budget() {
        in_dir("sparse_limit", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT CASE WHEN i % 2 = 1 \
                    THEN {'indices': []::BIGINT[], 'values': []::DOUBLE[]} \
                    ELSE {'indices': [i::BIGINT], 'values': [1.0::DOUBLE]} END AS embedding \
                 FROM range(5) r(i)",
            );
            let mut src = source(file.display().to_string(), None);
            src.limit = 3;
            let vectors = load_query_vectors(&src, VectorType::Sparse, None).unwrap();
            assert_eq!(vectors.len(), 3);
        });
    }

    #[test]
    fn empty_dense_rows_are_skipped_like_sparse_ones() {
        in_dir("dense_empty", |dir| {
            let file = dir.join("q.parquet");
            let conn = Connection::open_in_memory().unwrap();
            conn.execute_batch(&format!(
                "COPY (SELECT CASE WHEN i = 0 THEN []::FLOAT[] ELSE [i::FLOAT] END AS embedding \
                 FROM range(3) r(i)) TO '{}' (FORMAT PARQUET)",
                file.display()
            ))
            .unwrap();
            let vectors =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Dense, None)
                    .unwrap();
            assert_eq!(vectors.len(), 2);
        });
    }

    /// BM25-style integer term weights are losslessly usable as sparse values.
    #[test]
    fn integer_typed_sparse_values_are_accepted() {
        in_dir("sparse_intvals", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT {'indices': [1::BIGINT, 2::BIGINT], 'values': [3::BIGINT, 4::BIGINT]} AS embedding",
            );
            let vectors =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Sparse, None)
                    .unwrap();
            assert_eq!(
                vectors[0].vector,
                VectorData::Sparse { indices: vec![1, 2], values: vec![3.0, 4.0] }
            );
        });
    }

    /// Per-query filter values must stay glued to their own query across
    /// empty-row skips, exactly like ground truth does.
    #[test]
    fn filter_values_keep_alignment_across_empty_skips() {
        in_dir("sparse_filter_align", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT CASE WHEN i = 1 \
                    THEN {'indices': []::BIGINT[], 'values': []::DOUBLE[]} \
                    ELSE {'indices': [i::BIGINT], 'values': [1.0::DOUBLE]} END AS embedding, \
                 'tenant-' || i AS tenant FROM range(3) r(i)",
            );
            let filter = parse_filter("must:\n  - field: t\n    match_from_query: tenant\n");
            let vectors = load_query_vectors(
                &source(file.display().to_string(), None),
                VectorType::Sparse,
                Some(&filter),
            )
            .unwrap();
            assert_eq!(vectors.len(), 2);
            assert_eq!(
                vectors[0].filter_values.get("tenant"),
                Some(&FilterFieldValue::Text("tenant-0".into()))
            );
            assert_eq!(
                vectors[1].filter_values.get("tenant"),
                Some(&FilterFieldValue::Text("tenant-2".into()))
            );
        });
    }

    /// A genuine TIMESTAMP column becomes RFC3339 text, ready for the qdrant
    /// datetime path — previously only VARCHAR worked, with no hint why.
    #[test]
    fn timestamp_typed_filter_column_loads_as_rfc3339_text() {
        in_dir("ts_col", |dir| {
            let file = dir.join("q.parquet");
            let conn = Connection::open_in_memory().unwrap();
            conn.execute_batch(&format!(
                "COPY (SELECT [1.0::FLOAT] AS embedding, \
                 TIMESTAMP '2017-09-19 11:23:19' AS date_gte) TO '{}' (FORMAT PARQUET)",
                file.display()
            ))
            .unwrap();
            let filter =
                parse_filter("must:\n  - field: date\n    range_from_query:\n      gte: date_gte\n");
            let vectors = load_query_vectors(
                &source(file.display().to_string(), None),
                VectorType::Dense,
                Some(&filter),
            )
            .unwrap();
            let Some(FilterFieldValue::Text(rfc)) = vectors[0].filter_values.get("date_gte") else {
                panic!("expected text: {:?}", vectors[0].filter_values)
            };
            assert_eq!(rfc, "2017-09-19T11:23:19Z");
        });
    }

    /// The SQL predicate (not just the Rust backstop) must keep empty DENSE
    /// rows from eating the LIMIT budget, same as the sparse variant.
    #[test]
    fn empty_dense_rows_do_not_consume_the_limit_budget() {
        in_dir("dense_limit", |dir| {
            let file = dir.join("q.parquet");
            let conn = Connection::open_in_memory().unwrap();
            conn.execute_batch(&format!(
                "COPY (SELECT CASE WHEN i % 2 = 1 THEN []::FLOAT[] ELSE [i::FLOAT] END AS embedding \
                 FROM range(5) r(i)) TO '{}' (FORMAT PARQUET)",
                file.display()
            ))
            .unwrap();
            let mut src = source(file.display().to_string(), None);
            src.limit = 3;
            let vectors = load_query_vectors(&src, VectorType::Dense, None).unwrap();
            assert_eq!(vectors.len(), 3);
        });
    }

    /// NULL-fields structs must be excluded pre-LIMIT too (the coalesce
    /// default in the SQL predicate is what does it).
    #[test]
    fn null_field_structs_do_not_consume_the_limit_budget() {
        in_dir("nullstruct_limit", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT CASE WHEN i % 2 = 1 \
                    THEN {'indices': NULL::BIGINT[], 'values': NULL::DOUBLE[]} \
                    ELSE {'indices': [i::BIGINT], 'values': [1.0::DOUBLE]} END AS embedding \
                 FROM range(5) r(i)",
            );
            let mut src = source(file.display().to_string(), None);
            src.limit = 3;
            let vectors = load_query_vectors(&src, VectorType::Sparse, None).unwrap();
            assert_eq!(vectors.len(), 3);
        });
    }

    /// Indices present + values NULL is CORRUPTION (weights are gone), not
    /// "no embedding" — it must abort loudly, with a message that says which
    /// shape it saw. (Both-NULL is the tolerated skip; that asymmetry is
    /// deliberate and this test pins it.)
    #[test]
    fn values_null_with_real_indices_is_a_clear_error() {
        in_dir("values_null", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT {'indices': [1::BIGINT, 2::BIGINT], 'values': NULL::DOUBLE[]} AS embedding",
            );
            let err =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Sparse, None)
                    .unwrap_err();
            assert!(err.to_string().contains("2 indices but a NULL `values`"), "{err}");
        });
    }

    /// Every TimeUnit arm produces the same instant — a per-arm multiplier
    /// regression must fail here.
    #[test]
    fn timestamp_rfc3339_covers_every_time_unit() {
        use duckdb::types::TimeUnit;
        let secs = 1_505_820_199i64; // 2017-09-19T11:23:19Z
        for (unit, raw) in [
            (TimeUnit::Second, secs),
            (TimeUnit::Millisecond, secs * 1_000),
            (TimeUnit::Microsecond, secs * 1_000_000),
            (TimeUnit::Nanosecond, secs * 1_000_000_000),
        ] {
            assert_eq!(
                timestamp_rfc3339(unit, raw).unwrap(),
                "2017-09-19T11:23:19Z",
                "{unit:?}"
            );
        }
        // pre-1970 negative raw
        assert_eq!(timestamp_rfc3339(TimeUnit::Second, -1).unwrap(), "1969-12-31T23:59:59Z");
    }

    #[test]
    fn date32_column_becomes_midnight_utc() {
        // 2017-09-19 is 17428 days after epoch
        assert_eq!(date32_rfc3339(17_428).unwrap(), "2017-09-19T00:00:00Z");
        in_dir("date_col", |dir| {
            let file = dir.join("q.parquet");
            let conn = Connection::open_in_memory().unwrap();
            conn.execute_batch(&format!(
                "COPY (SELECT [1.0::FLOAT] AS embedding, DATE '2017-09-19' AS date_gte) \
                 TO '{}' (FORMAT PARQUET)",
                file.display()
            ))
            .unwrap();
            let filter =
                parse_filter("must:\n  - field: date\n    range_from_query:\n      gte: date_gte\n");
            let vectors = load_query_vectors(
                &source(file.display().to_string(), None),
                VectorType::Dense,
                Some(&filter),
            )
            .unwrap();
            assert_eq!(
                vectors[0].filter_values.get("date_gte"),
                Some(&FilterFieldValue::Text("2017-09-19T00:00:00Z".into()))
            );
        });
    }

    /// A malformed datetime in a range-bound column fails the LOAD — never a
    /// full-duration per-dispatch error loop.
    #[test]
    fn malformed_range_bound_datetime_fails_at_load() {
        in_dir("bad_bound", |dir| {
            let file = dir.join("q.parquet");
            let conn = Connection::open_in_memory().unwrap();
            conn.execute_batch(&format!(
                "COPY (SELECT [1.0::FLOAT] AS embedding, '2013/05/18' AS date_gte) \
                 TO '{}' (FORMAT PARQUET)",
                file.display()
            ))
            .unwrap();
            let filter =
                parse_filter("must:\n  - field: date\n    range_from_query:\n      gte: date_gte\n");
            let err = load_query_vectors(
                &source(file.display().to_string(), None),
                VectorType::Dense,
                Some(&filter),
            )
            .unwrap_err();
            assert!(err.to_string().contains("date_gte"), "{err}");
            assert!(err.to_string().contains("not"), "{err}");
        });
    }

    /// TIMESTAMPTZ text ("+00" hour-only offset, DuckDB's own rendering) and
    /// numeric-looking text both load as valid range bounds.
    #[test]
    fn range_bound_text_forms_normalize_at_load() {
        in_dir("bound_forms", |dir| {
            let file = dir.join("q.parquet");
            let conn = Connection::open_in_memory().unwrap();
            conn.execute_batch(&format!(
                "COPY (SELECT [1.0::FLOAT] AS embedding, \
                 '2017-09-19 09:23:19+00' AS date_gte, '42.5' AS ls_gte) \
                 TO '{}' (FORMAT PARQUET)",
                file.display()
            ))
            .unwrap();
            let filter = parse_filter(
                "must:\n  - field: date\n    range_from_query:\n      gte: date_gte\n\
                 \x20 - field: ls\n    range_from_query:\n      gte: ls_gte\n",
            );
            let vectors = load_query_vectors(
                &source(file.display().to_string(), None),
                VectorType::Dense,
                Some(&filter),
            )
            .unwrap();
            assert_eq!(
                vectors[0].filter_values.get("date_gte"),
                Some(&FilterFieldValue::Text("2017-09-19T09:23:19Z".into()))
            );
            assert_eq!(vectors[0].filter_values.get("ls_gte"), Some(&FilterFieldValue::Num(42.5)));
        });
    }

    /// A TIMESTAMP column under a MATCH condition is a load error — silently
    /// keyword-matching an RFC3339 string would select nothing forever.
    #[test]
    fn timestamp_under_match_from_query_is_rejected() {
        in_dir("ts_match", |dir| {
            let file = dir.join("q.parquet");
            let conn = Connection::open_in_memory().unwrap();
            conn.execute_batch(&format!(
                "COPY (SELECT [1.0::FLOAT] AS embedding, \
                 TIMESTAMP '2017-09-19 11:23:19' AS exact_date) TO '{}' (FORMAT PARQUET)",
                file.display()
            ))
            .unwrap();
            let filter =
                parse_filter("must:\n  - field: date\n    match_from_query: exact_date\n");
            let err = load_query_vectors(
                &source(file.display().to_string(), None),
                VectorType::Dense,
                Some(&filter),
            )
            .unwrap_err();
            assert!(err.to_string().contains("range_from_query"), "{err}");
        });
    }

    #[test]
    fn uri_with_a_single_quote_loads() {
        in_dir("o'brien", |dir| {
            let file = dir.join("q.parquet");
            write_parquet(&file, 2);
            let vectors =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Dense, None)
                    .unwrap();
            assert_eq!(vectors.len(), 2);
        });
    }

    /// ONE field misnamed must still be a preflight error — the check is
    /// "indices AND values present", and this pins the AND (a mutation to OR
    /// survived when only the both-wrong case was tested).
    #[test]
    fn one_misnamed_sparse_field_is_a_preflight_error() {
        in_dir("one_wrong_field", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT {'indices': [1::BIGINT], 'vals': [1.0::DOUBLE]} AS embedding",
            );
            let err =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Sparse, None)
                    .unwrap_err();
            assert!(err.to_string().contains("needs `indices`"), "{err}");
        });
    }

    /// Wrong struct FIELD NAMES are caught by the preflight with the real
    /// type string, not by an opaque binder error from the main query.
    #[test]
    fn wrong_sparse_field_names_are_a_preflight_error() {
        in_dir("wrong_fields", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT {'idx': [1::BIGINT], 'vals': [1.0::DOUBLE]} AS embedding",
            );
            let err =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Sparse, None)
                    .unwrap_err();
            assert!(err.to_string().contains("needs `indices`"), "{err}");
            assert!(err.to_string().contains("idx"), "{err}");
        });
    }

    /// A LIST of structs is neither modality; the preflight must not misread
    /// its STRUCT(...)[] type as sparse.
    #[test]
    fn list_of_structs_is_not_treated_as_sparse() {
        in_dir("structlist", |dir| {
            let file = dir.join("q.parquet");
            write_sparse_parquet(
                &file,
                "SELECT [{'indices': [1::BIGINT], 'values': [1.0::DOUBLE]}] AS embedding",
            );
            let err =
                load_query_vectors(&source(file.display().to_string(), None), VectorType::Sparse, None)
                    .unwrap_err();
            assert!(err.to_string().contains("not a struct"), "{err}");
        });
    }
}
