//! `nova inspect <path>` — count vectors and show the parquet schema at a path,
//! local or S3, with no abstraction over the two.
//!
//! The trick: DuckDB's glob (`*`, `**`, `?`, `[…]`) lives in its virtual
//! filesystem layer, and both the local FS and the S3 filesystem (httpfs) plug
//! into it. So `s3://bucket/p/**/*.parquet` and `/data/**/*.parquet` go through
//! the exact same glob → list → read path. We just hand DuckDB the glob string;
//! the only S3-specific step is loading httpfs + a credential_chain secret.

use std::process::ExitCode;

use clap::Parser;
use duckdb::Connection;

/// Count vectors + show the parquet schema at a path (local or `s3://`).
#[derive(Debug, Parser)]
#[command(name = "nova-inspect", version, about)]
struct Cli {
    /// A parquet file, a directory/prefix, or a glob. Local (`/data/embeds`) or
    /// S3 (`s3://bucket/prefix`). A bare dir/prefix gets `/**/*.parquet`
    /// appended; anything with `*` or ending in `.parquet` is used as-is.
    path: String,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match inspect(&cli.path) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn inspect(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let glob = normalize_glob(path);
    let conn = Connection::open_in_memory()?;
    if is_s3(&glob) {
        setup_s3(&conn)?;
    }

    // File count first — cheap, and lets us give a clean message instead of
    // DuckDB's "No files found" error when a glob matches nothing.
    let files: i64 = conn.query_row(
        &format!("SELECT count(*) FROM glob('{}')", esc(&glob)),
        [],
        |r| r.get(0),
    )?;
    if files == 0 {
        println!("no parquet files match: {glob}");
        return Ok(());
    }

    // count(*) over parquet reads row counts from the footers — no full scan.
    let rows: i64 = conn.query_row(
        &format!("SELECT count(*) FROM read_parquet('{}')", esc(&glob)),
        [],
        |r| r.get(0),
    )?;

    // DESCRIBE gives the unified logical schema (column name + type, e.g.
    // FLOAT[768] for a fixed-size vector column → dimension is visible).
    let mut stmt = conn.prepare(&format!(
        "DESCRIBE SELECT * FROM read_parquet('{}')",
        esc(&glob)
    ))?;
    let schema: Vec<(String, String)> = stmt
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
        .collect::<Result<_, _>>()?;

    // Vector columns are written as variable-length lists (`FLOAT[]`), so DESCRIBE
    // omits the dimension. Probe it from the first row — the one thing you most
    // want when inspecting embeddings. Fixed-size arrays (`FLOAT[768]`) already
    // carry the size, so we only probe the empty-bracket lists.
    let probe: Vec<&str> = schema
        .iter()
        .filter(|(_, ty)| ty == "FLOAT[]" || ty == "DOUBLE[]")
        .map(|(name, _)| name.as_str())
        .collect();
    let dims = if rows > 0 && !probe.is_empty() {
        probe_dims(&conn, &glob, &probe)?
    } else {
        Vec::new()
    };

    println!("path:   {glob}");
    println!("files:  {}", fmt_int(files));
    println!("rows:   {}", fmt_int(rows));
    println!("\nschema:");
    let width = schema.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
    for (name, ty) in &schema {
        match dims.iter().find(|(n, _)| n == name).and_then(|(_, d)| *d) {
            Some(dim) => println!("  {name:<width$}  {ty}  (dim {dim})"),
            None => println!("  {name:<width$}  {ty}"),
        }
    }
    Ok(())
}

/// Read each listed column's element count from the first row (one query, one
/// row). Returns `(column, Some(dim))`, or `None` for a column null in that row.
fn probe_dims(
    conn: &Connection,
    glob: &str,
    cols: &[&str],
) -> Result<Vec<(String, Option<i64>)>, duckdb::Error> {
    let projection = cols
        .iter()
        .map(|c| format!("len(\"{}\")", c.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!("SELECT {projection} FROM read_parquet('{}') LIMIT 1", esc(glob));
    conn.query_row(&sql, [], |row| {
        Ok(cols
            .iter()
            .enumerate()
            .map(|(i, c)| (c.to_string(), row.get::<_, Option<i64>>(i).unwrap_or(None)))
            .collect())
    })
}

/// A bare directory/prefix becomes a recursive parquet glob; an explicit glob or
/// single `.parquet` file is left untouched.
fn normalize_glob(path: &str) -> String {
    if path.contains('*') || path.ends_with(".parquet") {
        path.to_string()
    } else {
        format!("{}/**/*.parquet", path.trim_end_matches('/'))
    }
}

fn is_s3(path: &str) -> bool {
    path.starts_with("s3://")
}

/// Load httpfs and create an S3 secret that reuses the standard AWS credential
/// chain (env vars, shared config/profile, instance role) — same resolution the
/// rest of nova uses. Region comes from the environment when set.
fn setup_s3(conn: &Connection) -> Result<(), duckdb::Error> {
    conn.execute_batch("INSTALL httpfs; LOAD httpfs;")?;
    let region = std::env::var("AWS_REGION")
        .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
        .ok();
    let secret = match region {
        Some(r) => format!(
            "CREATE OR REPLACE SECRET nova_s3 (TYPE s3, PROVIDER credential_chain, REGION '{}');",
            esc(&r)
        ),
        None => "CREATE OR REPLACE SECRET nova_s3 (TYPE s3, PROVIDER credential_chain);".into(),
    };
    conn.execute_batch(&secret)
}

/// Escape a string for a single-quoted SQL literal.
fn esc(s: &str) -> String {
    s.replace('\'', "''")
}

/// Group an integer with thousands separators for readable counts.
fn fmt_int(n: i64) -> String {
    let digits = n.unsigned_abs().to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 { format!("-{out}") } else { out }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_appends_recursive_glob_to_bare_paths() {
        assert_eq!(normalize_glob("/data/embeds"), "/data/embeds/**/*.parquet");
        assert_eq!(
            normalize_glob("s3://bucket/fiqa/"),
            "s3://bucket/fiqa/**/*.parquet"
        );
    }

    #[test]
    fn normalize_leaves_globs_and_files_untouched() {
        assert_eq!(normalize_glob("/data/*.parquet"), "/data/*.parquet");
        assert_eq!(normalize_glob("s3://b/p/**/*.parquet"), "s3://b/p/**/*.parquet");
        assert_eq!(normalize_glob("/data/one.parquet"), "/data/one.parquet");
    }

    #[test]
    fn fmt_int_groups_thousands() {
        assert_eq!(fmt_int(0), "0");
        assert_eq!(fmt_int(999), "999");
        assert_eq!(fmt_int(64286), "64,286");
        assert_eq!(fmt_int(1234567), "1,234,567");
    }
}
