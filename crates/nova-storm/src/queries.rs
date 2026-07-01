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

use std::collections::HashSet;

use duckdb::Connection;
use duckdb::types::Value;

use crate::config::QuerySource;
use crate::errors::QueryLoadError;

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
}

/// Read up to `source.limit` query vectors (and, if configured, each one's
/// ground truth) from `source.uri`. Rows whose vector column is NULL are
/// skipped entirely (there's nothing to query with); a NULL ground-truth value
/// on an otherwise-valid row just means that query has no known-correct answer.
pub fn load_query_vectors(source: &QuerySource) -> Result<Vec<QueryVector>, QueryLoadError> {
    let conn = Connection::open_in_memory()?;
    // httpfs lets DuckDB read `s3://` (and `http(s)://`); harmless for local paths.
    conn.execute_batch("INSTALL httpfs; LOAD httpfs;")?;
    configure_s3(&conn)?;

    // Config is operator-authored (trusted). Columns are quoted so names with
    // odd characters survive. `cols` is the single source of truth for both the
    // SQL projection and "how many columns to read per row" below — there's
    // exactly one place that decides whether a ground-truth column is in play.
    let cols: Vec<&str> =
        std::iter::once(source.column.as_str()).chain(source.ground_truth_column.as_deref()).collect();
    let projection = cols.iter().map(|c| format!("\"{c}\"")).collect::<Vec<_>>().join(", ");
    let sql = format!(
        "SELECT {projection} FROM read_parquet('{uri}') WHERE \"{col}\" IS NOT NULL LIMIT {limit}",
        uri = source.uri,
        col = source.column,
        limit = source.limit,
    );

    let has_gt = cols.len() > 1;
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], move |row| {
        let vector = row.get::<_, Value>(0)?;
        let ground_truth = if has_gt { Some(row.get::<_, Value>(1)?) } else { None };
        Ok((vector, ground_truth))
    })?;

    let mut out = Vec::new();
    for row in rows {
        let (vector, ground_truth) = row?;
        let ground_truth = match ground_truth {
            None | Some(Value::Null) => None,
            Some(v) => Some(string_list(v)?.into_iter().collect::<HashSet<_>>()),
        };
        out.push(QueryVector { vector: float_list(vector)?, ground_truth });
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

/// Coerce a DuckDB `LIST`/`ARRAY` of strings into a `Vec<String>` — the same
/// shape as [`float_list`], for a ground-truth `hit_ids` column instead of a
/// vector column.
fn string_list(value: Value) -> Result<Vec<String>, QueryLoadError> {
    match value {
        Value::List(xs) | Value::Array(xs) => xs
            .iter()
            .map(|v| match v {
                Value::Text(s) => Some(s.clone()),
                _ => None,
            })
            .collect::<Option<_>>()
            .ok_or_else(|| QueryLoadError::Other("ground_truth column is not a list of strings".into())),
        _ => Err(QueryLoadError::Other("ground_truth column is not a list of strings".into())),
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

        let vectors = load_query_vectors(&source(file.display().to_string(), None)).unwrap();

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
            load_query_vectors(&source(file.display().to_string(), Some("hit_ids"))).unwrap();

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

        let result = load_query_vectors(&source(file.display().to_string(), Some("hit_ids")));

        std::fs::remove_dir_all(&dir).ok();

        assert!(result.is_err());
    }
}
