//! The shared DuckDB reading engine.
//!
//! [`DuckDbReader`] holds all the machinery — connection setup, macro
//! registration, SELECT building, streaming, vector extraction — and delegates
//! the per-source bits (where the files are, how to authenticate) to a
//! [`SourceBackend`]. Each `sources/<type>.rs` provides a thin backend; this is
//! the engine they share.

use std::collections::HashMap;

use duckdb::Connection;
use duckdb::types::Value;

use super::ReaderOptions;
use crate::DimensionsMap;
use crate::config::{VectorKind, VectorSpec};
use crate::errors::ReaderError;
use crate::stores::{Point, PointId, VectorValue};

/// The per-source hooks the [`DuckDbReader`] engine delegates to. A backend
/// decides *where* and *how* to read; the engine does the reading.
pub trait SourceBackend: Send {
    /// DuckDB-readable glob matching every parquet at this source.
    fn glob_path(&self) -> String;

    /// Combined FROM-clause expression scanning every file (for metadata
    /// queries — count, dimensions — where one parallel scan is cheap).
    fn source_sql(&self, parquet_kwargs: &str) -> String;

    /// One FROM-clause expression per file, so DuckDB releases decode buffers
    /// between files instead of buffering the whole corpus.
    fn iter_sources(&self, parquet_kwargs: &str) -> Vec<String>;

    /// URI prefix stripped from the `filename` column to recover the bare key
    /// fed to the `make_point_id` macro.
    fn root_uri_prefix(&self) -> String;

    /// Configure the connection (credentials, extensions) before reading.
    /// Default: nothing (local needs none).
    fn configure_connection(&self, _conn: &Connection) -> Result<(), ReaderError> {
        Ok(())
    }

    /// List every parquet file at this source via DuckDB's `glob`, excluding
    /// `eval/` artifacts, sorted for determinism. Used to shard files across
    /// distributed jobs.
    fn discover(&self) -> Result<Vec<String>, ReaderError> {
        let conn = Connection::open_in_memory()?;
        self.configure_connection(&conn)?;
        let mut stmt = conn.prepare(&format!(
            "SELECT file FROM glob('{}') ORDER BY file",
            self.glob_path()
        ))?;
        let files = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<String>, _>>()?
            .into_iter()
            .filter(|f| !f.contains("/eval/"))
            .collect();
        Ok(files)
    }
}

/// One vector to read, in stable SELECT order.
struct VectorEntry {
    name: String,
    column: String,
    kind: VectorKind,
}

pub struct DuckDbReader<B: SourceBackend> {
    backend: B,
    vectors: Vec<VectorEntry>,
    payload_fields: Vec<(String, String)>,
    id_expression: String,
    memory_limit: String,
    threads: u32,
    chunk_size: usize,
    parquet_kwargs: String,
    conn: Option<Connection>,
}

impl<B: SourceBackend> DuckDbReader<B> {
    pub fn new(
        backend: B,
        vectors: &HashMap<String, VectorSpec>,
        options: ReaderOptions,
        chunk_size: usize,
    ) -> Self {
        let vectors = vectors
            .iter()
            .map(|(name, spec)| VectorEntry {
                name: name.clone(),
                column: spec.column.clone(),
                kind: spec.kind,
            })
            .collect();
        let payload_fields = options.payload_fields.into_iter().collect();
        let parquet_kwargs = parquet_kwargs(&options.id_expression);
        Self {
            backend,
            vectors,
            payload_fields,
            id_expression: options.id_expression,
            memory_limit: options.duckdb_memory_limit,
            threads: options.duckdb_threads,
            chunk_size,
            parquet_kwargs,
            conn: None,
        }
    }

    /// Open + configure the connection on first use.
    fn ensure_connection(&mut self) -> Result<(), ReaderError> {
        if self.conn.is_none() {
            let conn = Connection::open_in_memory()?;
            conn.execute_batch(&format!(
                "SET memory_limit='{}'; SET threads={};",
                self.memory_limit, self.threads
            ))?;
            self.backend.configure_connection(&conn)?;
            self.register_macros(&conn)?;
            self.conn = Some(conn);
        }
        Ok(())
    }

    /// Register the id macros (mirrors the Python `_register_macros`): a content
    /// id derived from `md5(file:row)` formatted as a UUID, plus `vf_point_id`
    /// which strips the backend's URI prefix from the `filename` column first.
    fn register_macros(&self, conn: &Connection) -> Result<(), ReaderError> {
        let prefix_len = self.backend.root_uri_prefix().len();
        conn.execute_batch(&format!(
            "CREATE OR REPLACE MACRO vf_uuid_from_hex(h) AS (
                substr(h,1,8)||'-'||substr(h,9,4)||'-'||substr(h,13,4)||'-'||
                substr(h,17,4)||'-'||substr(h,21,12));
             CREATE OR REPLACE MACRO make_point_id(f,r) AS (
                vf_uuid_from_hex(md5(f||':'||CAST(r AS VARCHAR))));
             CREATE OR REPLACE MACRO vf_point_id(fname,rnum) AS (
                make_point_id(substr(fname,{}),rnum));",
            prefix_len + 1
        ))?;
        Ok(())
    }

    /// `id_expression, vcol1, vcol2, ..., pcol1, pcol2, ...`
    fn build_select(&self) -> String {
        let mut cols = vec![self.id_expression.clone()];
        cols.extend(self.vectors.iter().map(|v| v.column.clone()));
        cols.extend(self.payload_fields.iter().map(|(_, col)| col.clone()));
        cols.join(", ")
    }

    fn row_to_point(&self, row: &duckdb::Row<'_>) -> Result<Point, ReaderError> {
        let id = point_id(row.get::<_, Value>(0)?)?;

        let mut vectors = HashMap::with_capacity(self.vectors.len());
        for (i, v) in self.vectors.iter().enumerate() {
            let value = row.get::<_, Value>(i + 1)?;
            vectors.insert(v.name.clone(), vector_value(v.kind, value, &v.name)?);
        }

        let base = 1 + self.vectors.len();
        let mut payload = serde_json::Map::new();
        for (offset, (key, _)) in self.payload_fields.iter().enumerate() {
            let value = row.get::<_, Value>(base + offset)?;
            insert_payload(&mut payload, key, value);
        }

        Ok(Point {
            id,
            vectors,
            payload,
        })
    }
}

impl<B: SourceBackend> super::DataReader for DuckDbReader<B> {
    fn dimensions(&mut self) -> Result<DimensionsMap, ReaderError> {
        self.ensure_connection()?;
        let source = self.backend.source_sql(&self.parquet_kwargs);
        let conn = self.conn.as_ref().unwrap();

        let mut dims = DimensionsMap::new();
        for v in &self.vectors {
            let sql = match v.kind {
                VectorKind::Dense => format!(
                    "SELECT length({c}) FROM {source} WHERE {c} IS NOT NULL LIMIT 1",
                    c = v.column
                ),
                VectorKind::Multivector => format!(
                    "SELECT length({c}[1]) FROM {source} \
                     WHERE {c} IS NOT NULL AND length({c}) > 0 LIMIT 1",
                    c = v.column
                ),
                VectorKind::Sparse => continue,
            };
            let size: i64 = conn.prepare(&sql)?.query_row([], |r| r.get(0))?;
            dims.insert(v.name.clone(), size as usize);
        }
        Ok(dims)
    }

    fn total_count(&mut self) -> Result<u64, ReaderError> {
        self.ensure_connection()?;
        let source = self.backend.source_sql(&self.parquet_kwargs);
        let conn = self.conn.as_ref().unwrap();
        let sql = format!("SELECT count(*) FROM {source}");
        let n: i64 = conn.prepare(&sql)?.query_row([], |r| r.get(0))?;
        Ok(n as u64)
    }

    ///
    /// Drive the read to completion, handing each chunk of points to `sink`. Consumes
    /// the reader (the connection closes on drop). Runs on a blocking thread; the runner's `sink` forwards chunks into a channel. Chunk size is configured on the
    /// reader; the runner re-slices chunks into upsert batches.
    fn read(
        mut self: Box<Self>,
        sink: &mut dyn FnMut(Vec<Point>) -> Result<(), ReaderError>,
    ) -> Result<(), ReaderError> {
        self.ensure_connection()?;
        let select = self.build_select();
        let sources = self.backend.iter_sources(&self.parquet_kwargs);
        // Move the connection into a local so per-source statements can borrow
        // it without making `self` self-referential.
        let conn = self.conn.take().unwrap();

        for source in sources {
            let sql = format!("SELECT {select} FROM {source}");
            let mut stmt = conn.prepare(&sql)?;
            let mut rows = stmt.query([])?;
            let mut chunk: Vec<Point> = Vec::with_capacity(self.chunk_size);
            while let Some(row) = rows.next()? {
                chunk.push(self.row_to_point(row)?);
                if chunk.len() >= self.chunk_size {
                    sink(std::mem::take(&mut chunk))?;
                }
            }
            if !chunk.is_empty() {
                sink(chunk)?;
            }
        }
        Ok(())
    }
}

/// FROM-clause for the combined scan of every file (metadata queries). Shared
/// by the file-glob backends; they differ only in the `glob` they pass.
pub(crate) fn combined_source(
    glob: &str,
    file_list: Option<&[String]>,
    parquet_kwargs: &str,
) -> String {
    match file_list {
        Some(files) => {
            let lit = files
                .iter()
                .map(|f| format!("'{f}'"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("read_parquet([{lit}]{parquet_kwargs})")
        }
        None if !parquet_kwargs.is_empty() => format!("read_parquet('{glob}'{parquet_kwargs})"),
        None => format!("'{glob}'"),
    }
}

/// One FROM-clause per file, so DuckDB releases decode buffers between files.
pub(crate) fn per_file_sources(
    glob: &str,
    file_list: Option<&[String]>,
    parquet_kwargs: &str,
) -> Vec<String> {
    match file_list {
        Some(files) => files
            .iter()
            .map(|f| format!("read_parquet('{f}'{parquet_kwargs})"))
            .collect(),
        None if !parquet_kwargs.is_empty() => {
            vec![format!("read_parquet('{glob}'{parquet_kwargs})")]
        }
        None => vec![combined_source(glob, None, parquet_kwargs)],
    }
}

/// Comma-prefixed `read_parquet` kwargs based on which virtual columns the
/// id expression references (mirrors the Python `_parquet_kwargs`).
fn parquet_kwargs(id_expression: &str) -> String {
    let mut parts = Vec::new();
    if references_word(id_expression, "filename") {
        parts.push("filename=true");
    }
    if references_word(id_expression, "file_row_number") {
        parts.push("file_row_number=true");
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!(", {}", parts.join(", "))
    }
}

/// Word-boundary match so `filename` matches but `myfilename` does not.
fn references_word(haystack: &str, word: &str) -> bool {
    haystack.match_indices(word).any(|(i, _)| {
        let before = haystack[..i].chars().next_back();
        let after = haystack[i + word.len()..].chars().next();
        let boundary = |c: Option<char>| c.is_none_or(|c| !c.is_alphanumeric() && c != '_');
        boundary(before) && boundary(after)
    })
}

fn point_id(value: Value) -> Result<PointId, ReaderError> {
    match value {
        Value::Text(s) | Value::Enum(s) => Ok(PointId::String(s)),
        other => integer(&other)
            .map(PointId::Integer)
            .ok_or_else(|| ReaderError::Other(format!("point id is not an integer or string: {other:?}"))),
    }
}

fn vector_value(kind: VectorKind, value: Value, name: &str) -> Result<VectorValue, ReaderError> {
    match kind {
        VectorKind::Dense => Ok(VectorValue::Dense(float_list(value, name)?)),
        VectorKind::Multivector => {
            let (Value::List(rows) | Value::Array(rows)) = value else {
                return Err(shape_err(name, "list of lists"));
            };
            let multi = rows
                .into_iter()
                .map(|row| float_list(row, name))
                .collect::<Result<_, _>>()?;
            Ok(VectorValue::Multi(multi))
        }
        VectorKind::Sparse => {
            let Value::Struct(fields) = value else {
                return Err(shape_err(name, "struct{indices, values}"));
            };
            let indices = fields
                .get(&"indices".to_string())
                .ok_or_else(|| shape_err(name, "struct with `indices`"))?;
            let values = fields
                .get(&"values".to_string())
                .ok_or_else(|| shape_err(name, "struct with `values`"))?;
            let indices = match indices {
                Value::List(xs) | Value::Array(xs) => xs
                    .iter()
                    .map(|v| integer(v).map(|n| n as u32))
                    .collect::<Option<_>>()
                    .ok_or_else(|| shape_err(name, "integer `indices`"))?,
                _ => return Err(shape_err(name, "list `indices`")),
            };
            let values = match values {
                Value::List(xs) | Value::Array(xs) => xs
                    .iter()
                    .map(float)
                    .collect::<Option<_>>()
                    .ok_or_else(|| shape_err(name, "float `values`"))?,
                _ => return Err(shape_err(name, "list `values`")),
            };
            Ok(VectorValue::Sparse { indices, values })
        }
    }
}

fn float_list(value: Value, name: &str) -> Result<Vec<f32>, ReaderError> {
    match value {
        Value::List(xs) | Value::Array(xs) => xs
            .iter()
            .map(float)
            .collect::<Option<_>>()
            .ok_or_else(|| shape_err(name, "list of floats")),
        _ => Err(shape_err(name, "list of floats")),
    }
}

fn shape_err(name: &str, want: &str) -> ReaderError {
    ReaderError::Other(format!("vector {name:?} is not a {want}"))
}

/// Coerce any DuckDB integer value to `u64`.
fn integer(v: &Value) -> Option<u64> {
    match *v {
        Value::TinyInt(n) => u64::try_from(n).ok(),
        Value::SmallInt(n) => u64::try_from(n).ok(),
        Value::Int(n) => u64::try_from(n).ok(),
        Value::BigInt(n) => u64::try_from(n).ok(),
        Value::HugeInt(n) => u64::try_from(n).ok(),
        Value::UTinyInt(n) => Some(n as u64),
        Value::USmallInt(n) => Some(n as u64),
        Value::UInt(n) => Some(n as u64),
        Value::UBigInt(n) => Some(n),
        _ => None,
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

/// Insert a payload column. JSON-string columns that parse to an object are
/// unpacked into the payload (mirrors the Python legacy-blob handling).
fn insert_payload(payload: &mut serde_json::Map<String, serde_json::Value>, key: &str, value: Value) {
    if let Value::Text(s) = &value
        && let Ok(serde_json::Value::Object(map)) = serde_json::from_str(s)
    {
        payload.extend(map);
        return;
    }
    payload.insert(key.to_string(), value_to_json(value));
}

fn value_to_json(value: Value) -> serde_json::Value {
    use serde_json::Value as J;
    match value {
        Value::Null => J::Null,
        Value::Boolean(b) => J::Bool(b),
        Value::Float(f) => serde_json::json!(f),
        Value::Double(d) => serde_json::json!(d),
        Value::Text(s) | Value::Enum(s) => J::String(s),
        Value::List(xs) | Value::Array(xs) => J::Array(xs.into_iter().map(value_to_json).collect()),
        Value::Struct(fields) => J::Object(
            fields
                .iter()
                .map(|(k, v)| (k.clone(), value_to_json(v.clone())))
                .collect(),
        ),
        ref other => integer(other)
            .map(|n| serde_json::json!(n))
            .unwrap_or_else(|| J::String(format!("{other:?}"))),
    }
}