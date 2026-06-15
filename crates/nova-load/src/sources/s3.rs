use std::collections::HashMap;

use duckdb::Connection;
use serde::Deserialize;

use super::engine::{combined_source, per_file_sources};
use super::{DuckDbReader, ReaderOptions, SourceBackend};
use crate::config::VectorSpec;
use crate::errors::ReaderError;

#[derive(Debug, Deserialize)]
pub struct S3Config {
    pub bucket: String,
    #[serde(default)]
    pub prefix: Option<String>,
    #[serde(default)]
    pub file_list: Option<Vec<String>>,

    #[serde(flatten)]
    pub reader: ReaderOptions,
}

impl S3Config {
    pub fn into_reader(
        self,
        vectors: &HashMap<String, VectorSpec>,
        chunk_size: usize,
    ) -> DuckDbReader<S3Backend> {
        // Files are read directly over DuckDB's httpfs.
        let backend = S3Backend::new(self.bucket, self.prefix, self.file_list);
        DuckDbReader::new(backend, vectors, self.reader, chunk_size)
    }
}

/// Reads parquet from S3 via DuckDB's `httpfs` extension. Credentials come from
/// the standard AWS environment variables.
pub struct S3Backend {
    bucket: String,
    prefix: String,
    file_list: Option<Vec<String>>,
}

impl S3Backend {
    pub fn new(bucket: String, prefix: Option<String>, file_list: Option<Vec<String>>) -> Self {
        let prefix = prefix.unwrap_or_default().trim_end_matches('/').to_string();
        Self {
            bucket,
            prefix,
            file_list,
        }
    }

}

impl SourceBackend for S3Backend {
    fn glob_path(&self) -> String {
        format!("s3://{}/{}/**/*.parquet", self.bucket, self.prefix)
    }

    fn source_sql(&self, parquet_kwargs: &str) -> String {
        combined_source(&self.glob_path(), self.file_list.as_deref(), parquet_kwargs)
    }

    fn iter_sources(&self, parquet_kwargs: &str) -> Vec<String> {
        per_file_sources(&self.glob_path(), self.file_list.as_deref(), parquet_kwargs)
    }

    fn root_uri_prefix(&self) -> String {
        // vf_point_id strips this, leaving the full S3 key as the id input,
        // independent of the prefix used to scope this load.
        format!("s3://{}/", self.bucket)
    }

    fn configure_connection(&self, conn: &Connection) -> Result<(), ReaderError> {
        conn.execute_batch("INSTALL httpfs; LOAD httpfs;")?;

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_s3_source_expressions() {
        let b = S3Backend::new("my-bucket".into(), Some("corpus/".into()), None);
        assert_eq!(b.root_uri_prefix(), "s3://my-bucket/");
        // bare glob when no virtual columns are needed
        assert_eq!(b.source_sql(""), "'s3://my-bucket/corpus/**/*.parquet'");
        // wraps in read_parquet when filename-based ids need the virtual column
        assert_eq!(
            b.source_sql(", filename=true"),
            "read_parquet('s3://my-bucket/corpus/**/*.parquet', filename=true)"
        );
    }

    #[test]
    fn file_list_overrides_glob() {
        let b = S3Backend::new("b".into(), None, Some(vec!["s3://b/a.parquet".into()]));
        assert_eq!(
            b.iter_sources(""),
            vec!["read_parquet('s3://b/a.parquet')".to_string()]
        );
    }
}
