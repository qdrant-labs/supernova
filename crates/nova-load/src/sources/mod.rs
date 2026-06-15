//! Reading pre-embedded data to load.
//!
//! Mirrors the Python `DataReader` ABC: a backend (S3, HuggingFace, local)
//! streams chunks of points and answers the metadata the runner needs up front
//! (per-vector dimensions, total count).
//!
//! Reading is synchronous: a parquet scan is a single sequential pull, so there
//! is nothing to overlap and `async` would buy nothing. The runner bridges this
//! blocking reader to the async store side once, by driving it on a blocking
//! thread that feeds a channel (mirroring the Python sync-generator producer).
//!
//! [`DataReader::read`] uses a push model (`read(sink)`) rather than a pull
//! `next_chunk`: DuckDB's `Statement`/`Rows` borrow the `Connection`, so a
//! persisted cursor in `&mut self` would be self-referential. Owning the read
//! loop inside `read` keeps those borrows as locals and sidesteps that.
//!
//! Unlike [`VectorStore`](crate::stores::VectorStore), which is shared across
//! worker tasks behind an `Arc`, a reader is owned by the single producer task,
//! so it only needs `Send` (not `Sync`).

use std::collections::HashMap;

use serde::Deserialize;

use crate::DimensionsMap;
use crate::config::VectorSpec;
use crate::errors::ReaderError;
use crate::stores::Point;

#[cfg(any(feature = "s3", feature = "local", feature = "huggingface"))]
mod engine;
#[cfg(any(feature = "s3", feature = "local", feature = "huggingface"))]
pub use engine::{DuckDbReader, SourceBackend};

#[cfg(feature = "s3")]
pub mod s3;
#[cfg(feature = "local")]
pub mod local;
#[cfg(feature = "huggingface")]
pub mod huggingface;

pub trait DataReader: Send {
    /// Per-vector dimensions for dense and multivector vectors.
    fn dimensions(&mut self) -> Result<DimensionsMap, ReaderError>;

    /// Total number of records to be read (for progress reporting).
    fn total_count(&mut self) -> Result<u64, ReaderError>;

    /// Drive the read to completion, handing each chunk of points to `sink`.
    /// Consumes the reader (the connection closes on drop). Runs on a blocking
    /// thread; the runner's `sink` forwards chunks into a channel. Chunk size is
    /// configured on the reader; the runner re-slices chunks into upsert batches.
    fn read(
        self: Box<Self>,
        sink: &mut dyn FnMut(Vec<Point>) -> Result<(), ReaderError>,
    ) -> Result<(), ReaderError>;
}

/// Datasource backend, dispatched on `type:`. Each variant flattens
/// [`ReaderOptions`] so the common read keys sit as flat siblings of the
/// type-specific ones in YAML.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DataSourceConfig {
    #[cfg(feature = "s3")]
    S3(s3::S3Config),
    #[cfg(feature = "huggingface")]
    Huggingface(huggingface::HuggingfaceConfig),
    #[cfg(feature = "local")]
    Local(local::LocalConfig),
}

impl DataSourceConfig {
    /// Build the boxed reader for this source. `chunk_size` is how many points
    /// the reader buffers before handing a chunk to the runner.
    pub fn into_reader(
        self,
        vectors: &HashMap<String, VectorSpec>,
        chunk_size: usize,
    ) -> Result<Box<dyn DataReader>, ReaderError> {
        match self {
            #[cfg(feature = "s3")]
            DataSourceConfig::S3(c) => Ok(Box::new(c.into_reader(vectors, chunk_size))),
            #[cfg(feature = "local")]
            DataSourceConfig::Local(c) => Ok(Box::new(c.into_reader(vectors, chunk_size))),
            #[cfg(feature = "huggingface")]
            DataSourceConfig::Huggingface(_) => Err(ReaderError::Other(
                "huggingface datasource is not yet implemented".into(),
            )),
        }
    }

    /// Discover the corpus files and keep only this job's round-robin shard,
    /// recording it as the `file_list`. Returns how many files were assigned.
    pub fn shard(&mut self, num_jobs: usize, job_rank: usize) -> Result<usize, ReaderError> {
        let num_jobs = num_jobs.max(1);
        let files = self.discover_files()?;
        let shard: Vec<String> = files
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % num_jobs == job_rank % num_jobs)
            .map(|(_, f)| f)
            .collect();
        let n = shard.len();
        self.set_file_list(Some(shard));
        Ok(n)
    }

    fn discover_files(&self) -> Result<Vec<String>, ReaderError> {
        match self {
            #[cfg(feature = "local")]
            DataSourceConfig::Local(c) => local::LocalBackend::new(c.path.clone(), None).discover(),
            #[cfg(feature = "s3")]
            DataSourceConfig::S3(c) => {
                s3::S3Backend::new(c.bucket.clone(), c.prefix.clone(), None).discover()
            }
            #[cfg(feature = "huggingface")]
            DataSourceConfig::Huggingface(_) => Err(ReaderError::Other(
                "huggingface discovery is not yet implemented".into(),
            )),
        }
    }

    fn set_file_list(&mut self, files: Option<Vec<String>>) {
        match self {
            #[cfg(feature = "local")]
            DataSourceConfig::Local(c) => c.file_list = files,
            #[cfg(feature = "s3")]
            DataSourceConfig::S3(c) => c.file_list = files,
            #[cfg(feature = "huggingface")]
            DataSourceConfig::Huggingface(c) => c.file_list = files,
        }
    }
}

/// Read options common to every datasource: how to derive the point id, which
/// columns become payload, and DuckDB engine tuning. Defined once and flattened
/// into each source config.
#[derive(Debug, Deserialize)]
pub struct ReaderOptions {
    #[serde(default = "default_id_expression")]
    pub id_expression: String,
    #[serde(default)]
    pub payload_fields: HashMap<String, String>,
    #[serde(default = "default_memory_limit")]
    pub duckdb_memory_limit: String,
    #[serde(default = "default_threads")]
    pub duckdb_threads: u32,
}

fn default_id_expression() -> String {
    "row_id".to_string()
}
fn default_memory_limit() -> String {
    "2GB".to_string()
}
fn default_threads() -> u32 {
    2
}

#[cfg(all(test, feature = "local"))]
mod tests {
    use super::*;

    #[test]
    fn flatten_pulls_sibling_keys_and_keeps_defaults() {
        let yaml = "type: local\npath: /d/*.parquet\nid_expression: hash(text)\nduckdb_threads: 8\n";
        let cfg: DataSourceConfig = serde_yaml::from_str(yaml).unwrap();
        match cfg {
            DataSourceConfig::Local(l) => {
                assert_eq!(l.path, "/d/*.parquet");
                assert_eq!(l.reader.id_expression, "hash(text)"); // flattened sibling
                assert_eq!(l.reader.duckdb_threads, 8); // flattened sibling
                assert_eq!(l.reader.duckdb_memory_limit, "2GB"); // default still fills
            }
            #[allow(unreachable_patterns)]
            _ => panic!("expected local datasource"),
        }
    }
}