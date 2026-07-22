use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjPath;
use object_store::{ClientOptions, ObjectStore, RetryConfig};
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

use crate::sources::{DataSource, FileRef, LocalFile, Location, ReaderOptions, Result, SourceError};

#[derive(Debug, Deserialize)]
pub struct S3Config {
    /// Full `s3://bucket/prefix` URI. The bucket names the store; the rest
    /// restricts listing to keys under that prefix (prefix optional).
    pub path: String,
    /// Explicit keys (relative to the prefix) instead of listing the prefix.
    #[serde(default)]
    pub file_list: Option<Vec<String>>,
    /// Per-file S3 download timeout, in seconds. object_store defaults this to
    /// 30s covering the WHOLE request (headers + body), which is far too short
    /// for multi-GB parquet under load contention — the body stream aborts as
    /// "error decoding response body" and the file is retried into the same wall.
    /// Default 600s; `0` disables the timeout entirely (rely on retries only).
    #[serde(default = "default_download_timeout_secs")]
    pub download_timeout_secs: u64,
    #[serde(flatten)]
    pub reader: ReaderOptions,
}

fn default_download_timeout_secs() -> u64 {
    600
}

impl S3Config {
    /// Split `path` (`s3://bucket/prefix`) into `(bucket, Option<prefix>)`.
    fn bucket_prefix(&self) -> Result<(&str, Option<&str>)> {
        let rest = self.path.strip_prefix("s3://").ok_or_else(|| {
            SourceError::List(format!("datasource path `{}` must be an s3:// URI", self.path))
        })?;
        let (bucket, prefix) = match rest.split_once('/') {
            Some((b, p)) => (b, Some(p.trim_end_matches('/')).filter(|p| !p.is_empty())),
            None => (rest, None),
        };
        if bucket.is_empty() {
            return Err(SourceError::List(format!(
                "datasource path `{}` has no bucket (expected s3://bucket/prefix)",
                self.path
            )));
        }
        Ok((bucket, prefix))
    }

    /// Build a client using the standard AWS credential chain (env vars, shared
    /// config/profile, or instance role). A fresh client per call keeps this
    /// simple; listing/fetching are infrequent enough that it doesn't matter.
    fn store(&self) -> Result<impl ObjectStore> {
        let (bucket, _) = self.bucket_prefix()?;

        // Raise object_store's 30s default request timeout — it covers the full
        // body stream, so multi-GB downloads time out. Also lift RetryConfig's
        // retry_timeout (default 3min) to match, so it can't cap a long download.
        let mut opts = ClientOptions::new();
        let mut retry = RetryConfig::default();
        if self.download_timeout_secs == 0 {
            opts = opts.with_timeout_disabled();
            retry.retry_timeout = Duration::from_secs(3600);
        } else {
            let d = Duration::from_secs(self.download_timeout_secs);
            opts = opts.with_timeout(d);
            retry.retry_timeout = d.max(retry.retry_timeout);
        }

        AmazonS3Builder::from_env()
            .with_bucket_name(bucket)
            .with_client_options(opts)
            .with_retry(retry)
            .build()
            .map_err(|e| SourceError::List(format!("build S3 client for `{bucket}`: {e}")))
    }
}

#[async_trait]
impl DataSource for S3Config {
    async fn list_files(&self) -> Result<Vec<FileRef>> {
        let store = self.store()?;
        let (_, prefix) = self.bucket_prefix()?;

        let mut files = match &self.file_list {
            // Explicit keys: HEAD each so we still record sizes.
            Some(list) => {
                let mut out = Vec::with_capacity(list.len());
                for name in list {
                    let key = join(prefix, name);
                    let meta = store
                        .head(&ObjPath::from(key.clone()))
                        .await
                        .map_err(|e| SourceError::List(format!("head `{key}`: {e}")))?;
                    out.push(FileRef { key, size: Some(meta.size as u64) });
                }
                out
            }
            None => {
                let prefix = prefix.map(ObjPath::from);
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

    /// Stop at the first `.parquet` instead of listing the whole prefix — the
    /// `list` stream is lazy/paginated, so this returns after the first page.
    async fn first_file(&self) -> Result<Option<FileRef>> {
        let store = self.store()?;
        let (_, prefix) = self.bucket_prefix()?;
        if let Some(list) = &self.file_list {
            return match list.first() {
                Some(name) => {
                    let key = join(prefix, name);
                    let meta = store
                        .head(&ObjPath::from(key.clone()))
                        .await
                        .map_err(|e| SourceError::List(format!("head `{key}`: {e}")))?;
                    Ok(Some(FileRef { key, size: Some(meta.size as u64) }))
                }
                None => Ok(None),
            };
        }
        let prefix = prefix.map(ObjPath::from);
        let mut stream = store.list(prefix.as_ref());
        while let Some(meta) = stream.next().await {
            let meta = meta.map_err(|e| SourceError::List(e.to_string()))?;
            let key = meta.location.to_string();
            if key.ends_with(".parquet") {
                return Ok(Some(FileRef { key, size: Some(meta.size as u64) }));
            }
        }
        Ok(None)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(path: &str) -> S3Config {
        S3Config {
            path: path.to_string(),
            file_list: None,
            download_timeout_secs: default_download_timeout_secs(),
            reader: serde_yaml::from_str("id_expression: row_id").unwrap(),
        }
    }

    #[test]
    fn parses_bucket_and_prefix() {
        assert_eq!(cfg("s3://b/p/q").bucket_prefix().unwrap(), ("b", Some("p/q")));
        assert_eq!(cfg("s3://b/p/q/").bucket_prefix().unwrap(), ("b", Some("p/q")));
        assert_eq!(cfg("s3://b").bucket_prefix().unwrap(), ("b", None));
        assert_eq!(cfg("s3://b/").bucket_prefix().unwrap(), ("b", None));
    }

    #[test]
    fn rejects_non_s3_or_empty_bucket() {
        assert!(cfg("/local/path").bucket_prefix().is_err());
        assert!(cfg("s3:///p").bucket_prefix().is_err());
    }

    #[test]
    fn deserializes_s3_datasource_with_path() {
        let cfg: crate::sources::DataSourceConfig =
            serde_yaml::from_str("type: s3\npath: s3://b/p/q\nid_expression: row_id\n").unwrap();
        match cfg {
            crate::sources::DataSourceConfig::S3(s) => {
                assert_eq!(s.bucket_prefix().unwrap(), ("b", Some("p/q")));
            }
            _ => panic!("expected s3 datasource"),
        }
    }
}
