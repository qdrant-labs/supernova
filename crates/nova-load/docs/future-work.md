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

The load is split into phases, exposed as CLI subcommands so an orchestrator can
drive a fleet with no inter-worker coordination:

```
nova-load run <config>                              # single node: all phases
nova-load prepare <config>                          # master, once
nova-load load <config> --num-jobs N --job-rank R   # each worker
nova-load finalize <config>                         # master, once
nova-load inspect <config> [--num-jobs --job-rank]  # dry inspection
```

Phase → `VectorStore` trait methods:

| Phase    | Does                                            | Who         |
|----------|-------------------------------------------------|-------------|
| prepare  | `ensure_collection` + `defer_indexing`          | master      |
| load     | partition files → read → `upsert_batch`         | every worker|
| finalize | `enable_indexing` + `wait_for_indexing`         | master      |

Orchestration (e.g. SkyPilot):

1. master: `nova-load prepare config.yaml`
2. fleet:  `nova-load load config.yaml --num-jobs $N --job-rank $SKYPILOT_JOB_RANK`
3. master (after all workers exit): `nova-load finalize config.yaml`

### Design rationale

- **Master-controller, not leader-election.** Setup fully completes before any
  worker starts, so workers never poll/wait for the collection or race on the
  indexing toggle. Preserves the zero-coordination property of stride
  partitioning (`plan::partition`).
- **Subcommands, not just flags.** Each phase is independently retryable and
  idempotent — re-run a dead worker's rank, or re-run `finalize` if it timed out.
  Deterministic point ids mean a re-run rank just re-upserts the same points.
- **`recreate` footgun is closed structurally.** Dropping a collection only
  happens in `ensure_collection`, which only `prepare`/`run` call. A fleet
  `load` physically cannot drop data, regardless of config.

### Dimension inference in `prepare`

`prepare` needs vector dims before the fleet starts. If every vector has an
explicit `size:`, no read happens. Otherwise it samples **one row** of the first
file (`ReadJob.limit = Some(1)`) to measure them — cheap, master-only, once.
