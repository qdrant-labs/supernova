use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::Deserialize;

use crate::sources::{
    DataSource, FileRef, LocalFile, Location, ReaderOptions, Result, SourceError,
};

#[derive(Debug, Deserialize)]
pub struct LocalConfig {
    pub path: String,
    /// Explicit files (relative to `path`) instead of scanning the directory.
    #[serde(default)]
    pub file_list: Option<Vec<String>>,
    #[serde(flatten)]
    pub reader: ReaderOptions,
}

#[async_trait]
impl DataSource for LocalConfig {
    async fn list_files(&self) -> Result<Vec<FileRef>> {
        let base = Path::new(&self.path);

        let mut files = match &self.file_list {
            Some(list) => list
                .iter()
                .map(|name| file_ref(&base.join(name)))
                .collect::<Result<Vec<_>>>()?,
            None => {
                let mut found = Vec::new();
                collect_parquet(base, &mut found)?;
                found
            }
        };

        // Deterministic order so distributed workers partition disjointly.
        files.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(files)
    }

    async fn fetch(&self, file: &FileRef) -> Result<LocalFile> {
        // Already on disk — borrow the input in place. Never copy or delete it.
        Ok(LocalFile {
            source: file.clone(),
            location: Location::Borrowed(PathBuf::from(&file.key)),
        })
    }

    /// First `.parquet` found, without walking the whole tree (no sort).
    async fn first_file(&self) -> Result<Option<FileRef>> {
        let base = Path::new(&self.path);
        if let Some(list) = &self.file_list {
            return match list.first() {
                Some(name) => Ok(Some(file_ref(&base.join(name))?)),
                None => Ok(None),
            };
        }
        find_first_parquet(base)
    }
}

/// Depth-first search for the first `*.parquet` under `dir`; stops as soon as it
/// finds one. Entry order is filesystem-defined (fine for schema sampling).
fn find_first_parquet(dir: &Path) -> Result<Option<FileRef>> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| SourceError::List(format!("read dir `{}`: {e}", dir.display())))?;
    for entry in entries {
        let path = entry
            .map_err(|e| SourceError::List(format!("read dir `{}`: {e}", dir.display())))?
            .path();
        if path.is_dir() {
            if let Some(found) = find_first_parquet(&path)? {
                return Ok(Some(found));
            }
        } else if path.extension().is_some_and(|ext| ext == "parquet") {
            return Ok(Some(file_ref(&path)?));
        }
    }
    Ok(None)
}

/// Build a `FileRef` for a local path, reading its size via `stat`.
fn file_ref(path: &Path) -> Result<FileRef> {
    let size = std::fs::metadata(path)
        .map_err(|e| SourceError::List(format!("stat `{}`: {e}", path.display())))?
        .len();
    Ok(FileRef {
        key: path.to_string_lossy().into_owned(),
        size: Some(size),
    })
}

/// Recursively collect `*.parquet` files under `dir` (mirrors S3's flat-prefix
/// semantics, which match every key beneath the prefix).
fn collect_parquet(dir: &Path, out: &mut Vec<FileRef>) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| SourceError::List(format!("read dir `{}`: {e}", dir.display())))?;
    for entry in entries {
        let path = entry
            .map_err(|e| SourceError::List(format!("read dir `{}`: {e}", dir.display())))?
            .path();
        if path.is_dir() {
            collect_parquet(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "parquet") {
            out.push(file_ref(&path)?);
        }
    }
    Ok(())
}