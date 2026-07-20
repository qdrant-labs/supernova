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

use crate::config::QuerySource;
use crate::errors::QueryLoadError;
use crate::filter::{Filter, FilterFieldValue};

/// One query vector plus its ground truth, if configured. Bundled into one
/// struct (rather than two parallel `Vec`s) so the two can never drift out of
/// index alignment. `ground_truth` is a `HashSet`, not the raw `Vec` the
/// column decodes to, so `recall_at_k` (called once per query *firing*, and
/// queries cycle round-robin through a fixed, reused set of these) doesn't
/// rebuild a set from the same ids over and over — it's built once, here, at
/// load time.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryVector {
    pub vector: Vec<f32>,
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
/// ground truth and/or per-query filter values) from `source.uri`. Rows whose
/// vector column is NULL are skipped entirely (there's nothing to query
/// with); a NULL ground-truth value on an otherwise-valid row just means that
/// query has no known-correct answer.
///
/// `filter` is `query.filter`, if configured — when it has any `_from_query`
/// condition, this also projects every queries column those conditions name
/// (see [`Filter::query_fields`]) into each [`QueryVector::filter_values`].
/// Unlike a NULL ground-truth value, a NULL in one of *these* columns is a
/// hard error: nova-bf's own convention for "no filter value for this query"
/// is a non-matching placeholder (e.g. its MS MARCO config's unused
/// `domain_slot_N` columns hold `"zzznomatchzzz000"`), not NULL, so a NULL
/// here means the queries file wasn't built for this filter.
pub fn load_query_vectors(
    source: &QuerySource,
    filter: Option<&Filter>,
) -> Result<Vec<QueryVector>, QueryLoadError> {
    let conn = Connection::open_in_memory()?;
    // httpfs lets DuckDB read `s3://` (and `http(s)://`); harmless for local paths.
    conn.execute_batch("INSTALL httpfs; LOAD httpfs;")?;
    configure_s3(&conn)?;

    let filter_columns: Vec<&str> =
        filter.map(|f| f.query_fields().into_iter().collect()).unwrap_or_default();

    // Config is operator-authored (trusted). Columns are quoted so names with
    // odd characters survive. `cols` is the single source of truth for both the
    // SQL projection and "how many/which columns to read per row" below.
    let mut cols: Vec<&str> = vec![source.column.as_str()];
    cols.extend(source.ground_truth_column.as_deref());
    let has_gt = source.ground_truth_column.is_some();
    cols.extend(filter_columns.iter().copied());

    let projection = cols.iter().map(|c| format!("\"{c}\"")).collect::<Vec<_>>().join(", ");
    let sql = format!(
        "SELECT {projection} FROM read_parquet('{uri}') WHERE \"{col}\" IS NOT NULL LIMIT {limit}",
        uri = source.uri,
        col = source.column,
        limit = source.limit,
    );

    let n_cols = cols.len();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], move |row| {
        (0..n_cols).map(|i| row.get::<_, Value>(i)).collect::<duckdb::Result<Vec<_>>>()
    })?;

    let mut out = Vec::new();
    for row in rows {
        let mut values = row?.into_iter();
        let vector = values.next().expect("vector column always present");
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
            filter_values.insert((*name).to_string(), filter_field_value(value)?);
        }

        out.push(QueryVector { vector: float_list(vector)?, ground_truth, filter_values });
    }
    Ok(out)
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

/// Coerce one `_from_query`-referenced column's value into a
/// [`FilterFieldValue`] — text, an exact integer, a float/decimal, or a
/// homogeneous list of one of those. Callers already reject `Value::Null`
/// before this is called (see the NULL check in [`load_query_vectors`]).
fn filter_field_value(value: Value) -> Result<FilterFieldValue, QueryLoadError> {
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
        // A fixed 3-element float list per row: [i, i+1, i+2].
        conn.execute_batch(&format!(
            "COPY (SELECT [i::FLOAT, (i + 1)::FLOAT, (i + 2)::FLOAT] AS embedding \
             FROM range({rows}) r(i)) TO '{}' (FORMAT PARQUET)",
            path.display()
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

        let vectors = load_query_vectors(&source(file.display().to_string(), None), None).unwrap();

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
            load_query_vectors(&source(file.display().to_string(), Some("hit_ids")), None).unwrap();

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
            load_query_vectors(&source(file.display().to_string(), Some("hit_ids")), None).unwrap();

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

        let result = load_query_vectors(&source(file.display().to_string(), Some("hit_ids")), None);

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
            load_query_vectors(&source(file.display().to_string(), None), Some(&filter)).unwrap();

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
            load_query_vectors(&source(file.display().to_string(), None), Some(&filter));

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
            load_query_vectors(&source(file.display().to_string(), None), Some(&filter)).unwrap();

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
            load_query_vectors(&source(file.display().to_string(), None), Some(&filter)).unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(vectors[0].filter_values.get("max_budget"), Some(&FilterFieldValue::Num(42.5)));
    }
}
