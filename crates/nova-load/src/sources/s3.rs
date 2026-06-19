use async_trait::async_trait;
use futures::StreamExt;
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjPath;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

use crate::sources::{DataSource, FileRef, LocalFile, Location, ReaderOptions, Result, SourceError};

#[derive(Debug, Deserialize)]
pub struct S3Config {
    pub bucket: String,
    /// Restrict listing to keys under this prefix.
    #[serde(default)]
    pub prefix: Option<String>,
    /// Explicit keys (relative to `prefix`) instead of listing the prefix.
    #[serde(default)]
    pub file_list: Option<Vec<String>>,
    #[serde(flatten)]
    pub reader: ReaderOptions,
}

impl S3Config {
    /// Build a client using the standard AWS credential chain (env vars, shared
    /// config/profile, or instance role). A fresh client per call keeps this
    /// simple; listing/fetching are infrequent enough that it doesn't matter.
    fn store(&self) -> Result<impl ObjectStore> {
        AmazonS3Builder::from_env()
            .with_bucket_name(&self.bucket)
            .build()
            .map_err(|e| SourceError::List(format!("build S3 client for `{}`: {e}", self.bucket)))
    }
}

#[async_trait]
impl DataSource for S3Config {
    async fn list_files(&self) -> Result<Vec<FileRef>> {
        let store = self.store()?;

        let mut files = match &self.file_list {
            // Explicit keys: HEAD each so we still record sizes.
            Some(list) => {
                let mut out = Vec::with_capacity(list.len());
                for name in list {
                    let key = join(self.prefix.as_deref(), name);
                    let meta = store
                        .head(&ObjPath::from(key.clone()))
                        .await
                        .map_err(|e| SourceError::List(format!("head `{key}`: {e}")))?;
                    out.push(FileRef { key, size: Some(meta.size as u64) });
                }
                out
            }
            None => {
                let prefix = self.prefix.as_deref().map(ObjPath::from);
                let mut stream = store.list(prefix.as_ref());
                let mut out = Vec::new();
                while let Some(meta) = stream.next().await {
                    let meta = meta.map_err(|e| SourceError::List(e.to_string()))?;
                    let key = meta.location.to_string();
                    if key.ends_with(".parquet") {
                        out.push(FileRef { key, size: Some(meta.size as u64) });
                    }
                }
                out
            }
        };

        // Deterministic order so distributed workers partition disjointly.
        files.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(files)
    }

    async fn fetch(&self, file: &FileRef) -> Result<LocalFile> {
        let store = self.store()?;
        let path = ObjPath::from(file.key.clone());

        let result = store
            .get(&path)
            .await
            .map_err(|e| SourceError::Fetch(format!("{}: {e}", file.key)))?;

        // Stream straight to a temp file so we never hold the whole object in RAM.
        let tmp = tempfile::NamedTempFile::new()
            .map_err(|e| SourceError::Fetch(format!("create temp for `{}`: {e}", file.key)))?;
        let handle = tmp
            .reopen()
            .map_err(|e| SourceError::Fetch(format!("reopen temp for `{}`: {e}", file.key)))?;
        let mut writer = tokio::fs::File::from_std(handle);

        let mut stream = result.into_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| SourceError::Fetch(format!("{}: {e}", file.key)))?;
            writer
                .write_all(&chunk)
                .await
                .map_err(|e| SourceError::Fetch(format!("write temp for `{}`: {e}", file.key)))?;
        }
        writer
            .flush()
            .await
            .map_err(|e| SourceError::Fetch(format!("flush temp for `{}`: {e}", file.key)))?;

        Ok(LocalFile {
            source: file.clone(),
            location: Location::Owned(tmp.into_temp_path()),
        })
    }
}

/// Join an optional prefix with a key, avoiding a double slash.
fn join(prefix: Option<&str>, name: &str) -> String {
    match prefix {
        Some(p) if !p.is_empty() => format!("{}/{}", p.trim_end_matches('/'), name),
        _ => name.to_string(),
    }
}
