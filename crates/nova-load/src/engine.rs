//! Read path: open one local parquet file with DuckDB and yield [`Point`]s.
//!
//! DuckDB only ever sees a **local** file (the source downloads remote files to
//! a temp path first), so its scanning stays on the rock-solid local path. The
//! stable logical filename is injected into the query as a `filename` column —
//! never DuckDB's `filename` pseudo-column, which would be the temp path and
//! break deterministic ids.

use std::collections::HashMap;
use std::path::PathBuf;

use duckdb::arrow::array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array,
    ListArray, StringArray, StructArray, UInt32Array, UInt64Array,
};
use duckdb::arrow::datatypes::DataType;
use duckdb::{Connection, params};

use crate::config::{VectorKind, VectorSpec};
use crate::stores::{Point, PointId, VectorValue};

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Duck(#[from] duckdb::Error),
    #[error("schema mismatch reading `{file}`: {detail}")]
    Schema { file: String, detail: String },
}

/// A self-contained read of one local parquet file. Owns its inputs so it can be
/// moved into `spawn_blocking` (DuckDB is synchronous).
pub struct ReadJob {
    /// Local path DuckDB reads (a temp download, or a borrowed local file).
    pub path: PathBuf,
    /// Stable logical filename injected for id expressions (the `FileRef` key).
    pub filename: String,
    /// Named vector specs (which column feeds which vector, and its kind).
    pub vectors: HashMap<String, VectorSpec>,
    /// Payload field name → source column.
    pub payload: HashMap<String, String>,
    /// SQL id expression, e.g. `vf_point_id(filename, file_row_number)`.
    pub id_expression: String,
    /// Cap the number of rows read. `None` reads the whole file; `Some(n)` is
    /// used to cheaply sample a file (e.g. inferring dimensions from one row).
    pub limit: Option<usize>,
}

impl ReadJob {
    /// Run the read, returning every point in the file. Blocking — call from
    /// `spawn_blocking`.
    pub fn run(&self) -> Result<Vec<Point>, EngineError> {
        let conn = Connection::open_in_memory()?;
        register_macros(&conn)?;

        let sql = self.build_sql();
        let mut stmt = conn.prepare(&sql)?;

        let mut points = Vec::new();
        for batch in stmt.query_arrow(params![])? {
            let id_col = self.column(&batch, "__id")?;
            for row in 0..batch.num_rows() {
                points.push(self.build_point(&batch, id_col, row)?);
            }
        }
        Ok(points)
    }

    /// `SELECT {id} AS __id, {col AS vec_name}…, {col AS payload_field}…`
    /// over the parquet, with the logical filename injected and `file_row_number`
    /// exposed.
    fn build_sql(&self) -> String {
        let mut projections = vec![format!("{} AS __id", self.id_expression)];
        for (name, spec) in &self.vectors {
            projections.push(format!("{} AS \"{}\"", spec.column, esc_ident(name)));
        }
        for (field, column) in &self.payload {
            projections.push(format!("{} AS \"{}\"", column, esc_ident(field)));
        }
        let limit = match self.limit {
            Some(n) => format!(" LIMIT {n}"),
            None => String::new(),
        };
        format!(
            "SELECT {} FROM (SELECT *, '{}' AS filename \
             FROM read_parquet('{}', file_row_number = true)){}",
            projections.join(", "),
            esc_str(&self.filename),
            esc_str(&self.path.to_string_lossy()),
            limit,
        )
    }

    fn build_point(
        &self,
        batch: &duckdb::arrow::array::RecordBatch,
        id_col: &dyn Array,
        row: usize,
    ) -> Result<Point, EngineError> {
        let id = self.read_id(id_col, row)?;

        let mut vectors = HashMap::with_capacity(self.vectors.len());
        for (name, spec) in &self.vectors {
            let col = self.column(batch, name)?;
            let value = match spec.kind {
                VectorKind::Dense => VectorValue::Dense(self.dense_at(col, row)?),
                VectorKind::Multivector => VectorValue::Multi(self.multi_at(col, row)?),
                VectorKind::Sparse => {
                    let (indices, values) = self.sparse_at(col, row)?;
                    VectorValue::Sparse { indices, values }
                }
            };
            vectors.insert(name.clone(), value);
        }

        let mut payload = serde_json::Map::with_capacity(self.payload.len());
        for field in self.payload.keys() {
            let col = self.column(batch, field)?;
            payload.insert(field.clone(), cell_to_json(col, row));
        }

        Ok(Point { id, vectors, payload })
    }

    fn column<'b>(
        &self,
        batch: &'b duckdb::arrow::array::RecordBatch,
        name: &str,
    ) -> Result<&'b dyn Array, EngineError> {
        batch
            .column_by_name(name)
            .map(|c| c.as_ref())
            .ok_or_else(|| self.schema_err(format!("missing column `{name}` in result")))
    }

    fn read_id(&self, col: &dyn Array, row: usize) -> Result<PointId, EngineError> {
        if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
            return Ok(PointId::String(a.value(row).to_string()));
        }
        if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
            return Ok(PointId::Integer(a.value(row) as u64));
        }
        if let Some(a) = col.as_any().downcast_ref::<UInt64Array>() {
            return Ok(PointId::Integer(a.value(row)));
        }
        Err(self.schema_err(format!("id column must be string or integer, got {:?}", col.data_type())))
    }

    fn dense_at(&self, col: &dyn Array, row: usize) -> Result<Vec<f32>, EngineError> {
        if let Some(a) = col.as_any().downcast_ref::<FixedSizeListArray>() {
            return self.list_to_f32(a.value(row).as_ref());
        }
        if let Some(a) = col.as_any().downcast_ref::<ListArray>() {
            return self.list_to_f32(a.value(row).as_ref());
        }
        Err(self.schema_err(format!("dense vector column is not a list: {:?}", col.data_type())))
    }

    fn multi_at(&self, col: &dyn Array, row: usize) -> Result<Vec<Vec<f32>>, EngineError> {
        let list = col
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| self.schema_err("multivector column is not a list of vectors".into()))?;
        let inner = list.value(row);
        (0..inner.len()).map(|i| self.dense_at(inner.as_ref(), i)).collect()
    }

    fn sparse_at(&self, col: &dyn Array, row: usize) -> Result<(Vec<u32>, Vec<f32>), EngineError> {
        let s = col
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| self.schema_err("sparse vector column is not a struct".into()))?;
        let indices = s
            .column_by_name("indices")
            .ok_or_else(|| self.schema_err("sparse struct missing `indices`".into()))?;
        let values = s
            .column_by_name("values")
            .ok_or_else(|| self.schema_err("sparse struct missing `values`".into()))?;

        let idx = indices
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| self.schema_err("sparse `indices` is not a list".into()))?
            .value(row);
        let val = values
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| self.schema_err("sparse `values` is not a list".into()))?
            .value(row);

        Ok((self.list_to_u32(idx.as_ref())?, self.list_to_f32(val.as_ref())?))
    }

    fn list_to_f32(&self, values: &dyn Array) -> Result<Vec<f32>, EngineError> {
        if let Some(a) = values.as_any().downcast_ref::<Float32Array>() {
            return Ok(a.values().to_vec());
        }
        if let Some(a) = values.as_any().downcast_ref::<Float64Array>() {
            return Ok(a.values().iter().map(|&v| v as f32).collect());
        }
        Err(self.schema_err(format!("expected float list, got {:?}", values.data_type())))
    }

    fn list_to_u32(&self, values: &dyn Array) -> Result<Vec<u32>, EngineError> {
        if let Some(a) = values.as_any().downcast_ref::<Int32Array>() {
            return Ok(a.values().iter().map(|&v| v as u32).collect());
        }
        if let Some(a) = values.as_any().downcast_ref::<Int64Array>() {
            return Ok(a.values().iter().map(|&v| v as u32).collect());
        }
        if let Some(a) = values.as_any().downcast_ref::<UInt32Array>() {
            return Ok(a.values().to_vec());
        }
        Err(self.schema_err(format!("expected integer list, got {:?}", values.data_type())))
    }

    fn schema_err(&self, detail: String) -> EngineError {
        EngineError::Schema { file: self.filename.clone(), detail }
    }
}

/// Infer dense/multivector dimensions from the first point of a file. An
/// explicit `size:` on the spec wins; sparse vectors have no fixed size.
pub fn infer_dims(points: &[Point], vectors: &HashMap<String, VectorSpec>) -> HashMap<String, u64> {
    let mut dims = HashMap::new();
    let Some(first) = points.first() else { return dims };
    for (name, spec) in vectors {
        if let Some(size) = spec.size {
            dims.insert(name.clone(), size);
            continue;
        }
        match first.vectors.get(name) {
            Some(VectorValue::Dense(v)) => {
                dims.insert(name.clone(), v.len() as u64);
            }
            Some(VectorValue::Multi(m)) => {
                if let Some(v) = m.first() {
                    dims.insert(name.clone(), v.len() as u64);
                }
            }
            _ => {}
        }
    }
    dims
}

/// Register the id macros: a content id derived from `md5(file:row)` formatted
/// as a UUID. The logical filename is already canonical (the source's key), so
/// `vf_point_id` is just `make_point_id` — no prefix stripping needed here,
/// unlike the Python pipeline that fed DuckDB raw URIs.
fn register_macros(conn: &Connection) -> Result<(), EngineError> {
    conn.execute_batch(
        "CREATE OR REPLACE MACRO vf_uuid_from_hex(h) AS (
            substr(h,1,8)||'-'||substr(h,9,4)||'-'||substr(h,13,4)||'-'||
            substr(h,17,4)||'-'||substr(h,21,12));
         CREATE OR REPLACE MACRO make_point_id(f,r) AS (
            vf_uuid_from_hex(md5(f||':'||CAST(r AS VARCHAR))));
         CREATE OR REPLACE MACRO vf_point_id(fname,rnum) AS (make_point_id(fname,rnum));",
    )?;
    Ok(())
}

/// Convert one Arrow cell to JSON for the payload. Covers the common scalar
/// types; anything else becomes null (extend as needed).
fn cell_to_json(col: &dyn Array, row: usize) -> serde_json::Value {
    use serde_json::Value;
    if col.is_null(row) {
        return Value::Null;
    }
    match col.data_type() {
        DataType::Utf8 => col
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|a| Value::String(a.value(row).to_string()))
            .unwrap_or(Value::Null),
        DataType::Int64 => col
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|a| Value::from(a.value(row)))
            .unwrap_or(Value::Null),
        DataType::Int32 => col
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(|a| Value::from(a.value(row) as i64))
            .unwrap_or(Value::Null),
        DataType::Float64 => col
            .as_any()
            .downcast_ref::<Float64Array>()
            .and_then(|a| serde_json::Number::from_f64(a.value(row)))
            .map(Value::Number)
            .unwrap_or(Value::Null),
        DataType::Float32 => col
            .as_any()
            .downcast_ref::<Float32Array>()
            .and_then(|a| serde_json::Number::from_f64(a.value(row) as f64))
            .map(Value::Number)
            .unwrap_or(Value::Null),
        DataType::Boolean => col
            .as_any()
            .downcast_ref::<BooleanArray>()
            .map(|a| Value::Bool(a.value(row)))
            .unwrap_or(Value::Null),
        _ => Value::Null,
    }
}

/// Escape a string for a single-quoted SQL literal.
fn esc_str(s: &str) -> String {
    s.replace('\'', "''")
}

/// Escape a double-quoted SQL identifier.
fn esc_ident(s: &str) -> String {
    s.replace('"', "\"\"")
}
