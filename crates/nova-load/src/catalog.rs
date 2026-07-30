use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use duckdb::{Connection, params};

#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error("catalog input path does not exist: {0}")]
    MissingInput(String),
    #[error("catalog build input must be a directory: {0}")]
    InputNotDirectory(String),
    #[error("no parquet files found under: {0}")]
    NoParquetFiles(String),
    #[error("duplicate relative_path `{0}` while merging catalogs")]
    DuplicatePath(String),
    #[error("catalog `{0}` missing supported path column (expected one of: relative_path, path, filename)")]
    MissingPathColumn(String),
    #[error("duckdb error: {0}")]
    Duck(#[from] duckdb::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogEntry {
    pub relative_path: String,
    pub file_size: Option<u64>,
}

pub fn build_catalog(input_dir: &Path, output: &Path, resume: bool) -> Result<usize, CatalogError> {
    if !input_dir.exists() {
        return Err(CatalogError::MissingInput(input_dir.display().to_string()));
    }
    let root = input_dir.canonicalize()?;
    if !root.is_dir() {
        return Err(CatalogError::InputNotDirectory(root.display().to_string()));
    }

    let mut parquet_files = Vec::new();
    collect_parquet(&root, &mut parquet_files)?;
    if parquet_files.is_empty() {
        return Err(CatalogError::NoParquetFiles(root.display().to_string()));
    }
    parquet_files.sort();

    let mut entries: BTreeMap<String, Option<u64>> = BTreeMap::new();
    if resume && output.exists() {
        for entry in load_catalog_entries(output)? {
            entries.insert(entry.relative_path, entry.file_size);
        }
    }

    for file in parquet_files {
        let rel = to_relative_slash(&root, &file)?;
        entries.entry(rel).or_insert_with(|| fs::metadata(&file).ok().map(|m| m.len()));
    }

    write_catalog(output, entries)?;
    Ok(load_catalog_entries(output)?.len())
}

pub fn merge_catalogs(inputs: &[PathBuf], output: &Path) -> Result<usize, CatalogError> {
    let files = expand_catalog_inputs(inputs)?;
    if files.is_empty() {
        return Err(CatalogError::MissingInput(
            "no catalog parquet files were provided".to_string(),
        ));
    }

    let mut merged: BTreeMap<String, Option<u64>> = BTreeMap::new();
    for catalog in files {
        for entry in load_catalog_entries(&catalog)? {
            if merged.contains_key(&entry.relative_path) {
                return Err(CatalogError::DuplicatePath(entry.relative_path));
            }
            merged.insert(entry.relative_path, entry.file_size);
        }
    }

    write_catalog(output, merged)?;
    Ok(load_catalog_entries(output)?.len())
}

fn write_catalog(output: &Path, entries: BTreeMap<String, Option<u64>>) -> Result<(), CatalogError> {
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let conn = Connection::open_in_memory()?;
    conn.execute_batch(
        "CREATE TABLE catalog(relative_path VARCHAR NOT NULL, file_size BIGINT);",
    )?;
    let mut stmt = conn.prepare("INSERT INTO catalog(relative_path, file_size) VALUES (?, ?)")?;
    for (relative_path, file_size) in entries {
        let size = file_size.map(|v| v as i64);
        stmt.execute(params![relative_path, size])?;
    }
    let sql = format!(
        "COPY (SELECT relative_path, file_size FROM catalog ORDER BY relative_path) TO '{}' \
         (FORMAT PARQUET, COMPRESSION ZSTD);",
        esc_str(&output.to_string_lossy()),
    );
    conn.execute_batch(&sql)?;
    Ok(())
}

fn load_catalog_entries(path: &Path) -> Result<Vec<CatalogEntry>, CatalogError> {
    let conn = Connection::open_in_memory()?;
    let path = path.to_string_lossy();
    let path_cols = ["relative_path", "path", "filename"];
    let size_cols = ["file_size", "size"];

    for path_col in path_cols {
        for size_col in size_cols {
            if let Ok(entries) = read_catalog_rows(&conn, &path, path_col, Some(size_col)) {
                return Ok(entries);
            }
        }
        if let Ok(entries) = read_catalog_rows(&conn, &path, path_col, None) {
            return Ok(entries);
        }
    }

    Err(CatalogError::MissingPathColumn(path.into_owned()))
}

fn read_catalog_rows(
    conn: &Connection,
    catalog_path: &str,
    path_col: &str,
    size_col: Option<&str>,
) -> Result<Vec<CatalogEntry>, CatalogError> {
    let sql = match size_col {
        Some(size_col) => format!(
            "SELECT CAST({} AS VARCHAR) AS p, CAST({} AS BIGINT) AS s \
             FROM read_parquet('{}')",
            quote_ident(path_col),
            quote_ident(size_col),
            esc_str(catalog_path)
        ),
        None => format!(
            "SELECT CAST({} AS VARCHAR) AS p FROM read_parquet('{}')",
            quote_ident(path_col),
            esc_str(catalog_path)
        ),
    };

    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let path: String = row.get(0)?;
        let size = if size_col.is_some() {
            row.get::<usize, Option<i64>>(1)?.map(|v| v as u64)
        } else {
            None
        };
        out.push(CatalogEntry {
            relative_path: path.trim_start_matches('/').to_string(),
            file_size: size,
        });
    }
    Ok(out)
}

fn expand_catalog_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, CatalogError> {
    let mut out = Vec::new();
    for input in inputs {
        if !input.exists() {
            return Err(CatalogError::MissingInput(input.display().to_string()));
        }
        if input.is_dir() {
            for entry in fs::read_dir(input)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file()
                    && path
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("parquet"))
                {
                    out.push(path);
                }
            }
        } else {
            out.push(input.clone());
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

fn collect_parquet(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), CatalogError> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_parquet(&path, out)?;
        } else if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("parquet"))
        {
            out.push(path);
        }
    }
    Ok(())
}

fn to_relative_slash(root: &Path, path: &Path) -> Result<String, CatalogError> {
    let rel = path
        .strip_prefix(root)
        .map_err(|_| CatalogError::MissingInput(path.display().to_string()))?;
    Ok(rel.to_string_lossy().replace('\\', "/"))
}

fn quote_ident(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

fn esc_str(s: &str) -> String {
    s.replace('\'', "''")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_catalog_writes_sorted_relative_paths() {
        let dir = tempfile::tempdir().expect("tempdir");
        let a = dir.path().join("a");
        let b = dir.path().join("b");
        fs::create_dir_all(&a).expect("mkdir a");
        fs::create_dir_all(&b).expect("mkdir b");
        fs::write(a.join("2.parquet"), b"xx").expect("write a/2");
        fs::write(b.join("1.parquet"), b"xxx").expect("write b/1");

        let out = dir.path().join("catalog.parquet");
        let n = build_catalog(dir.path(), &out, false).expect("build");
        assert_eq!(n, 2);

        let entries = load_catalog_entries(&out).expect("load");
        assert_eq!(entries[0].relative_path, "a/2.parquet");
        assert_eq!(entries[1].relative_path, "b/1.parquet");
    }

    #[test]
    fn merge_catalogs_combines_inputs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let c1 = dir.path().join("c1.parquet");
        let c2 = dir.path().join("c2.parquet");
        write_catalog(
            &c1,
            BTreeMap::from([(String::from("x/1.parquet"), Some(10))]),
        )
        .expect("write c1");
        write_catalog(
            &c2,
            BTreeMap::from([(String::from("y/2.parquet"), Some(20))]),
        )
        .expect("write c2");

        let out = dir.path().join("merged.parquet");
        let n = merge_catalogs(&[c1, c2], &out).expect("merge");
        assert_eq!(n, 2);
        let entries = load_catalog_entries(&out).expect("load");
        assert_eq!(entries.len(), 2);
    }
}
