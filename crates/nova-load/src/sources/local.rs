use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use super::engine::{combined_source, per_file_sources};
use super::{DuckDbReader, ReaderOptions, SourceBackend};
use crate::config::VectorSpec;

#[derive(Debug, Deserialize)]
pub struct LocalConfig {
    pub path: String,
    #[serde(default)]
    pub file_list: Option<Vec<String>>,

    #[serde(flatten)]
    pub reader: ReaderOptions,
}

impl LocalConfig {
    /// Build the DuckDB-backed reader for this local source.
    pub fn into_reader(
        self,
        vectors: &HashMap<String, VectorSpec>,
        chunk_size: usize,
    ) -> DuckDbReader<LocalBackend> {
        let backend = LocalBackend::new(self.path, self.file_list);
        DuckDbReader::new(backend, vectors, self.reader, chunk_size)
    }
}

/// Reads parquet from the local filesystem. DuckDB reads local files natively,
/// so there are no credentials and no connection setup.
pub struct LocalBackend {
    path: String,
    root_dir: String,
    file_list: Option<Vec<String>>,
}

impl LocalBackend {
    pub fn new(path: String, file_list: Option<Vec<String>>) -> Self {
        // Resolve `~` and make absolute so the `filename` column DuckDB reports
        // (an absolute path) lines up with `root_dir` for filename-derived ids.
        let path = resolve_path(&path);
        let file_list = file_list
            .map(|files| files.iter().map(|f| resolve_path(strip_file_scheme(f))).collect());
        let root_dir = root_dir_of(&path);
        Self {
            path,
            root_dir,
            file_list,
        }
    }
}

impl SourceBackend for LocalBackend {
    /// A directory is read recursively; an explicit glob or single `.parquet`
    /// file is used as-is.
    fn glob_path(&self) -> String {
        if self.path.contains('*') || self.path.ends_with(".parquet") {
            self.path.clone()
        } else {
            format!("{}/**/*.parquet", self.path.trim_end_matches('/'))
        }
    }

    fn source_sql(&self, parquet_kwargs: &str) -> String {
        combined_source(&self.glob_path(), self.file_list.as_deref(), parquet_kwargs)
    }

    fn iter_sources(&self, parquet_kwargs: &str) -> Vec<String> {
        per_file_sources(&self.glob_path(), self.file_list.as_deref(), parquet_kwargs)
    }

    fn root_uri_prefix(&self) -> String {
        format!("{}/", self.root_dir)
    }
}

/// Expand a leading `~` and make the path absolute, without touching the
/// filesystem (so it works on globs and not-yet-existing paths).
fn resolve_path(path: &str) -> String {
    let expanded = expanduser(path);
    std::path::absolute(&expanded)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or(expanded)
}

fn expanduser(path: &str) -> String {
    if path == "~" {
        std::env::var("HOME").unwrap_or_else(|_| path.to_string())
    } else if let Some(rest) = path.strip_prefix("~/") {
        match std::env::var("HOME") {
            Ok(home) => format!("{}/{}", home.trim_end_matches('/'), rest),
            Err(_) => path.to_string(),
        }
    } else {
        path.to_string()
    }
}

fn strip_file_scheme(uri: &str) -> &str {
    uri.strip_prefix("file://").unwrap_or(uri)
}

/// Base dir that per-row `filename`s are made relative to, so filename-based
/// ids are stable regardless of where the dir sits on disk.
fn root_dir_of(path: &str) -> String {
    if let Some((head, _)) = path.split_once('*') {
        head.trim_end_matches('/').to_string()
    } else if path.ends_with(".parquet") {
        Path::new(path)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default()
    } else {
        path.trim_end_matches('/').to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::DataReader;
    use crate::stores::{PointId, VectorValue};
    use duckdb::Connection;

    fn write_parquet(path: &Path) {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "COPY (SELECT * FROM (VALUES \
               (0, [0.1, 0.2, 0.3]::FLOAT[]), \
               (1, [0.4, 0.5, 0.6]::FLOAT[])) \
             t(row_id, dense_embedding)) TO '{}' (FORMAT PARQUET)",
            path.display()
        ))
        .unwrap();
    }

    fn dense_vectors() -> HashMap<String, VectorSpec> {
        serde_yaml::from_str("dense:\n  type: dense\n  column: dense_embedding\n").unwrap()
    }

    #[test]
    fn discover_lists_parquets_and_excludes_eval() {
        let dir = std::env::temp_dir().join(format!("nova_discover_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("eval")).unwrap();
        write_parquet(&dir.join("a.parquet"));
        write_parquet(&dir.join("b.parquet"));
        write_parquet(&dir.join("eval/c.parquet")); // must be excluded

        let backend = LocalBackend::new(dir.display().to_string(), None);
        let files = backend.discover().unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|f| !f.contains("/eval/")));
        assert!(files.iter().any(|f| f.ends_with("a.parquet")));
        assert!(files == { let mut s = files.clone(); s.sort(); s }); // sorted
    }

    #[test]
    fn reads_local_dense_parquet() {
        let dir = std::env::temp_dir().join(format!("nova_local_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("data.parquet");
        write_parquet(&file);

        let cfg: LocalConfig =
            serde_yaml::from_str(&format!("path: {}\n", file.display())).unwrap();
        let mut reader = cfg.into_reader(&dense_vectors(), 1000);

        assert_eq!(reader.dimensions().unwrap()["dense"], 3);
        assert_eq!(reader.total_count().unwrap(), 2);

        let mut points = Vec::new();
        Box::new(reader)
            .read(&mut |chunk| {
                points.extend(chunk);
                Ok(())
            })
            .unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(points.len(), 2);
        points.sort_by_key(|p| match &p.id {
            PointId::Integer(n) => *n,
            _ => unreachable!(),
        });
        assert!(matches!(points[0].id, PointId::Integer(0)));
        match &points[0].vectors["dense"] {
            VectorValue::Dense(v) => assert_eq!(v.len(), 3),
            other => panic!("expected dense, got {other:?}"),
        }
    }

    #[test]
    fn filename_derived_ids_are_stable_uuids() {
        let dir = std::env::temp_dir().join(format!("nova_local_fid_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_parquet(&dir.join("data.parquet"));

        // id_expression references `filename`/`file_row_number`, which the engine
        // detects to enable the read_parquet virtual columns + the id macros.
        let yaml = format!(
            "path: {}\nid_expression: vf_point_id(filename, file_row_number)\n",
            dir.display()
        );

        let read_ids = || {
            let cfg: LocalConfig = serde_yaml::from_str(&yaml).unwrap();
            let mut ids = Vec::new();
            Box::new(cfg.into_reader(&dense_vectors(), 1000))
                .read(&mut |chunk| {
                    ids.extend(chunk.into_iter().map(|p| match p.id {
                        PointId::String(s) => s,
                        other => panic!("expected uuid string, got {other:?}"),
                    }));
                    Ok(())
                })
                .unwrap();
            ids.sort();
            ids
        };

        let first = read_ids();
        let second = read_ids(); // same file+rows must yield the same ids
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(first, second); // deterministic
        assert_eq!(first.len(), 2);
        assert!(first[0] != first[1]); // distinct per row
        assert!(first.iter().all(|id| id.len() == 36 && id.matches('-').count() == 4));
    }
}