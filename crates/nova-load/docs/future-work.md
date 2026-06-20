# nova-load — future work & design notes

Deliberately-deferred extensions, with enough detail to pick up cheaply later.
These are *not* bugs or gaps in correctness — they're scope decisions.

## Non-parquet inputs (JSONL, CSV, …)

Today the loader assumes parquet. Supporting other formats does **not** require a
new data source backend — the abstraction boundary is already in the right place:

- `DataSource` (`list_files` / `fetch`) is **format-blind**: it moves bytes.
  Local / S3 / HF differ only in *where* bytes live, not *how* they're parsed. A
  `.jsonl` file lists and downloads exactly like a `.parquet` one.
- The **only** parquet assumption lives in one function: `engine::ReadJob::build_sql`,
  which hardcodes `read_parquet(...)`.

DuckDB reads JSONL/CSV natively and exposes the **same Arrow batches**, so the
entire extraction path (dense/sparse/payload, the id macros) reuses unchanged:

- JSONL → `read_json_auto('file', format='newline_delimited')`
- CSV   → `read_csv_auto('file')`

### How to add it (≈10 lines)

1. A `Format { Parquet, Jsonl, Csv }` enum, inferred from the file extension
   (or an explicit `format:` override in the datasource config).
2. One `match` in `build_sql` choosing the scan function.

```rust
let scan = match format {
    Format::Parquet => format!("read_parquet('{p}', file_row_number = true)"),
    Format::Jsonl   => format!("read_json_auto('{p}', format='newline_delimited')"),
    Format::Csv     => format!("read_csv_auto('{p}')"),
};
```

**Caveat:** only parquet has the `file_row_number` pseudo-column that
`vf_point_id(filename, file_row_number)` relies on. For JSONL/CSV, the id
expression needs `row_number() OVER ()` instead, or an id field from the data.

### Why NOT a `JsonlBackend`

It conflates two orthogonal axes — *where bytes live* (source) and *how bytes
are parsed* (reader) — and forces a combinatorial explosion (S3-parquet,
S3-jsonl, local-parquet, …). Keep them separate: source = location, reader =
format.

## Distributed fleet lifecycle

See `--no-manage-indexing` and the master/fleet/finalize flow. (Document the
chosen design here once implemented.)
