//! Loading the query-vector set.
//!
//! Storm cycles through a fixed set of query vectors pulled from a parquet file
//! (a local path or an `s3://` URI), read once at startup via DuckDB. The set is
//! held in memory and reused round-robin across the whole run, so this is a
//! one-shot synchronous read — there's nothing to stream or overlap. (Contrast
//! `nova-load`, which downloads huge corpus files; a small query set reads fine
//! straight from `s3://` via httpfs.)

use duckdb::Connection;
use duckdb::types::Value;

use crate::config::QuerySource;
use crate::errors::QueryLoadError;

/// Read up to `source.limit` query vectors from `source.column` of the parquet
/// at `source.uri`. Rows whose vector column is NULL are skipped.
pub fn load_query_vectors(source: &QuerySource) -> Result<Vec<Vec<f32>>, QueryLoadError> {
    let conn = Connection::open_in_memory()?;
    // httpfs lets DuckDB read `s3://` (and `http(s)://`); harmless for local paths.
    conn.execute_batch("INSTALL httpfs; LOAD httpfs;")?;
    configure_s3(&conn)?;

    // Config is operator-authored (trusted). The column is quoted so names with
    // odd characters survive.
    let sql = format!(
        "SELECT \"{col}\" FROM read_parquet('{uri}') WHERE \"{col}\" IS NOT NULL LIMIT {limit}",
        col = source.column,
        uri = source.uri,
        limit = source.limit,
    );

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| row.get::<_, Value>(0))?;

    let mut vectors = Vec::new();
    for row in rows {
        vectors.push(float_list(row?)?);
    }
    Ok(vectors)
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

    #[test]
    fn loads_and_limits_query_vectors() {
        let dir = std::env::temp_dir().join(format!("nova_storm_q_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("queries.parquet");
        write_parquet(&file, 100);

        let source = QuerySource {
            uri: file.display().to_string(),
            column: "embedding".into(),
            limit: 10,
        };
        let vectors = load_query_vectors(&source).unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(vectors.len(), 10); // LIMIT honoured
        assert!(vectors.iter().all(|v| v.len() == 3)); // full dim per row
    }
}
