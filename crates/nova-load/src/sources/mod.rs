mod local;
mod s3;

use std::collections::HashMap;
use std::path::PathBuf;
use tempfile;

use async_trait::async_trait;
use serde::Deserialize;

type Result<T> = std::result::Result<T, SourceError>;

#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("failed to list files in source: {0}")]
    List(String),
    #[error("failed to fetch file `{0}` from source")]
    Fetch(String),
}

/// Datasource backend, dispatched on `type:`. Each variant flattens
/// [`ReaderOptions`] so the common read keys sit as flat siblings of the
/// type-specific ones in YAML.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DataSourceConfig {
    Local(local::LocalConfig),
    S3(s3::S3Config),
    // Huggingface(huggingface::HuggingfaceConfig),
    // Local(local::LocalConfig),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRef {
    /// Source-defined identifier, used both to `fetch` the file and to order
    /// work deterministically across workers.
    pub key: String,
    /// Size in bytes if the listing knew it.
    pub size: Option<u64>,
}

pub struct LocalFile {
    /// The file these bytes came from. `source.key` is the stable logical
    /// filename for id expressions — deliberately separate from `path()`.
    pub source: FileRef,
    location: Location,
}

impl LocalFile {
    /// Where the bytes actually live on disk right now — feed this to duckdb's
    /// `read_parquet`. (Use `source.key` for the stable logical filename.)
    pub fn path(&self) -> &std::path::Path {
        match &self.location {
            Location::Owned(p) => p,
            Location::Borrowed(p) => p,
        }
    }
}

enum Location {
    /// Downloaded to a temp path we own; deleted when this `LocalFile` drops.
    Owned(tempfile::TempPath),
    /// A pre-existing local file we borrow and must NOT delete.
    Borrowed(PathBuf),
}

#[async_trait]
pub trait DataSource {
    /// List the files to read from this source, as `FileRef`s, in a
    /// deterministic order so distributed workers can partition them disjointly.
    async fn list_files(&self) -> Result<Vec<FileRef>>;
    /// Materialize one file on local disk and return a handle to it: a path
    /// duckdb can `read_parquet`, plus the originating `FileRef` for its stable
    /// name. Remote sources download to a temp removed when the handle drops;
    /// the local source borrows the input file in place.
    async fn fetch(&self, file: &FileRef) -> Result<LocalFile>;
}

impl DataSourceConfig {
    /// Shared read options, regardless of backend.
    pub fn reader(&self) -> &ReaderOptions {
        match self {
            DataSourceConfig::Local(c) => &c.reader,
            DataSourceConfig::S3(c) => &c.reader,
        }
    }
}

/// Dispatch to the concrete backend so callers can hold a `DataSourceConfig`
/// and treat it as a `DataSource` directly.
#[async_trait]
impl DataSource for DataSourceConfig {
    async fn list_files(&self) -> Result<Vec<FileRef>> {
        match self {
            DataSourceConfig::Local(c) => c.list_files().await,
            DataSourceConfig::S3(c) => c.list_files().await,
        }
    }

    async fn fetch(&self, file: &FileRef) -> Result<LocalFile> {
        match self {
            DataSourceConfig::Local(c) => c.fetch(file).await,
            DataSourceConfig::S3(c) => c.fetch(file).await,
        }
    }
}

fn default_id_expression() -> String {
    "uuid()".to_string()
}

fn default_memory_limit() -> String {
    "2GB".to_string()
}

fn default_threads() -> u32 {
    2
}
