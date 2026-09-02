# Parameter Sweeps

`nova sweep` orchestrates `nova-load` and `nova-storm` across a matrix of
collection/index/search configurations, producing one combined report
(recall, latency, throughput per point). It answers "how does this index or
search setting affect recall/latency" across many settings at once, instead
of hand-running `nova load` + `nova storm` once per combination.

**Three targets are implemented: `qdrant`, `milvus`, and `elastic`.**
`target.type` (see below) is **required** and dispatches to a backend adapter,
mirroring `nova-load`'s `VectorStore` and `nova-storm`'s `QueryTarget`
extension points. A future backend needs its own adapter module registered the
same way, without touching the runner.

**Ground truth is out of scope.** `nova sweep` assumes a `nova bf`-shaped
parquet (a dense vector column plus a `hit_ids` column of known-correct
top-K point ids) has already been computed separately, and just points at
it — exactly the shape `nova-storm`'s own `query.source` expects. It never
invokes `nova bf` itself.

```
                          ┌─▶ nova load ──▶ store ──▶ nova storm
nova bf (ground truth) ───┤        ▲                       │
                          └────────┴─── nova sweep orchestrates both ───┘
```

## Configuration

Sweep configs live in `configs/sweep/`.

```yaml
collection_name: my_model_sweep

corpus:
  path: s3://my-bucket/dataset/model      # nova-load's datasource.path
  dense_column: dense_embedding

# Exactly nova-storm's own query.source shape — computed separately (e.g. a
# `nova bf compute` run) and passed through into every generated storm config.
queries:
  uri: s3://my-bucket/dataset/model/queries.parquet
  column: dense_embedding
  ground_truth_column: hit_ids
  limit: 1000

target:
  type: qdrant             # REQUIRED — qdrant | milvus | elastic
  url: ${QDRANT_URL}
  api_key: ${QDRANT_API_KEY}
  recreate: never          # never (default) | always — see "Collections" below

data_layouts:
  vectors.dense.distance: [cosine]
  vectors.dense.datatype: [float32, uint8]

index_variants:
  quantization.type: [none, scalar]
  hnsw.m: [8, 16, 32]
  hnsw.ef_construct: [64, 128]

searches:
  top_k: [10]
  hnsw_ef: [64, 128]
  batch_size: [1, 8]

output:
  path: s3://my-bucket/sweeps/model/      # or a local dir
```

`${VAR}` / `${VAR:-default}` references are expanded from the environment,
same as every other tool. See `configs/sweep/example.yaml` for a
fully-annotated version of this file.

## Target backends

`target.type` selects which backend `nova sweep` drives — it's dispatched the
same way `nova-load`'s `vectorstore.type` and `nova-storm`'s `target.type`
are, so a config's `target:` block is backend-specific past `type` and
`recreate`. `type` is **required**: an omitted, null, or unknown `type` is a
hard config error at parse time (there is no default). The three implemented
backends and their extra `target:` fields:

- **`qdrant`** — `url`, `api_key`. Structural params (`hnsw.*`,
  `quantization.*`, `vectors.dense.*`) nest under `vectorstore.params` in the
  generated `nova-load` config; search params are `{hnsw_ef, exact,
  quantization}`.
- **`milvus`** — `url`, `username`, `password` (`username`/`password` must be
  set together, or neither — a lone one is a config error). `vectorstore`
  fields are FLAT: `index_type` (`HNSW`/`IVF_FLAT`/`AUTOINDEX`/…) and
  `index_params` (`{M, efConstruction}`, `{nlist}`, …). Search params are
  `{ef, nprobe}`. The metric comes from `vectors.dense.distance` (not a
  separate field) on both the fresh load *and* every reindex. **Caveat:** an
  `index_type` on the `data_layouts` axis must carry its *complete*
  `index_params` (a bare `HNSW` with no `M`/`efConstruction` fails at index
  creation). A data_layout's index spec is the base of every reindex, so
  `index_variants` can sweep `index_params` under a fixed `index_type`; don't
  switch to a different index family in a variant without also overriding
  `index_params`, or the mismatched params error out per-slice.
- **`elastic`** — `url`, `username`/`password` or `api_key`, `tls_insecure`
  (as with milvus, `username`/`password` must be set together — use `api_key`
  for token auth). The swept collection name maps to the ES **index name**.
  `vectorstore` fields are FLAT: `index_options` (`{type, m, ef_construction}`
  — ES 8.x defaults to `int8_hnsw`). Search param is `{num_candidates}` (must
  be `>= top_k`). **Caveat:** `similarity`/`distance` and the `index_options`
  `type` are fixed at field creation and CANNOT be changed by `reindex`, and
  HNSW `m` may only *increase* — so put a distance/type change on the
  `data_layouts` axis (fresh load), never `index_variants` (an in-place
  reindex to a conflicting mapping errors out per-slice).

Adding a new backend means writing a new adapter module under
`nova_sweep.backends` and registering it in `_REGISTRY`; it does not require
changes to the sweep runner, which only calls backend-neutral methods
(`collection_exists`, `build_load_config`, `build_reindex_config`,
`build_delete_config`, `build_storm_config`).

## Running

```bash
nova sweep configs/sweep/my_sweep.yaml
nova sweep configs/sweep/my_sweep.yaml --dry-run       # preview only, no execution
nova sweep configs/sweep/my_sweep.yaml --skip-insert   # reuse existing collections
nova sweep configs/sweep/my_sweep.yaml --cleanup       # delete inserted collections when done
```

## Axes: cartesian grids

`data_layouts`, `index_variants`, and `searches` share one shape: a flat
dict of **dotted-path key → list of values**. `nova sweep` expands the full
cross-product of each axis (e.g. `hnsw.m: [8, 16, 32]` × `hnsw.ef_construct:
[64, 128]` → 6 combinations) and auto-names every combination by joining
`{last_dotted_segment}{value}` for each key — e.g. `m8_ef_construct64`.

A YAML `null` at a leaf omits that key entirely from the generated config
(rather than setting it to `null`) — useful for an axis where one option is
"leave this unset" (e.g. `search_params.exact: [null, true, false]`). This is
`null`-only: the *string* `none` is a normal value, not a pruning sentinel —
`quantization.type: [none, scalar]`'s `none` option produces
`quantization: {type: none}`, which `nova-load` itself already understands
(a no-op on create, an explicit "clear quantization" on `reindex`).

## How it works

For each expanded `data_layouts` entry, `nova sweep` creates **one collection
for its entire lifetime** (a Qdrant/Milvus collection or an ES index):

1. **Load once.** `nova-load run` creates the collection and inserts the
   corpus — this is the only step that reads/writes data.
2. **Patch in place, per `index_variants` entry.** `nova-load reindex`
   patches index settings on that *same* collection — never a new collection,
   never a delete/recreate between variants.
3. **Search, per `searches` entry.** `nova-storm --json` runs against the
   now-patched collection; its structured summary (recall, latency,
   throughput) becomes one report row.

Only `data_layouts` changes force a reload — anything fixed at collection
creation belongs here: `vectors.dense.distance`/`datatype`/`size` and
`shard_number` for Qdrant, and for Elastic the vector `similarity` and the
`index_options` `type` (both immutable in ES). Everything under
`index_variants` is patchable on an already-loaded collection — Qdrant HNSW/
quantization/optimizers via `update_collection`, Milvus by drop+rebuild of the
index, Elastic via the Update Mapping API + force-merge (only *widening*
changes like an HNSW `m` increase; a conflicting change is recorded as a
per-slice error, not a crash). `searches` never touches the store at all.

Collection names are now explicit in the config, not inferred from the file
name: the config's `collection_name` is used directly when there is only the
implicit default layout, otherwise each expanded `data_layouts` entry is named
`<collection_name>_<data_layout_name>`.

### Rebuild-cost ordering

Qdrant `index_variants` are walked in an order chosen to minimize rebuild
cost, not declaration order. HNSW changes are treated as **expensive** (Qdrant
fully rebuilds the graph on any `hnsw_config` change); quantization/optimizer
changes are treated as **cheap** — small-scale empirical testing showed a
quantization-only change reindexes meaningfully faster than an HNSW change
on the same collection. Combinations are sorted so expensive fields change
as infrequently as possible — e.g. for `quantization.type: [none, scalar]` ×
`hnsw.m: [8, 16]` (4 combinations), both `hnsw.m: 8` variants run before
either `hnsw.m: 16` variant, regardless of how the axes were declared. The
*set* of combinations tested is unaffected — only the order.

This cost sort keys only off Qdrant's `hnsw.*` fields, so for Milvus
(`index_type`/`index_params`) and Elastic (`index_options`) it is a stable
no-op and variants run in **declaration order**. This matters for Elastic,
where HNSW `m` may only increase across in-place reindexes: list such
variants in increasing order yourself.

## Collections

If a `data_layouts` entry's target collection (`<collection_name>_<data_layout
name>`, or just `collection_name` for the implicit default layout) **already exists** when
its turn comes up, `nova sweep` does not guess, warn-and-continue, or
silently delete it — **it errors and exits immediately**, before touching
anything:

```
error: collection 'my_sweep_distancecosine' (data_layout 'distancecosine')
already exists. Pass --skip-insert to reuse it as-is, or set
`target.recreate: always` in the config to force a fresh reload.
```

- **`--skip-insert`** reuses the existing collection as-is: the insert phase
  is skipped for that data_layout, and its `index_variants`/`searches` run
  normally against the data already there. This is also how you re-run a
  sweep that only changed `index_variants`/`searches` without re-reading and
  re-upserting the whole corpus.
- **`target.recreate: always`** unconditionally deletes and reloads every
  run, regardless of `--skip-insert` — an explicit opt-in for when you know
  the existing collection is stale (the corpus or a data_layout changed).
- **`--cleanup`** deletes every collection *this run* inserted into, once
  it's done — never one only reused via `--skip-insert`. Collections are
  kept by default otherwise (no flag needed to preserve them).

## Output

One flattened row per (`data_layout` × `index_variant` × `search`) point,
written to `<output.path>/sweep_results.parquet`:

| Column(s) | Meaning |
|-----------|---------|
| `collection_name`, `data_layout_name`, `index_variant_name`, `search_name` | Which point this row is |
| `data_layout.*`, `index_variant.*`, `search.*` | That point's flattened parameters (e.g. `index_variant.hnsw.m`) |
| `reindex_seconds`, `search_seconds` | Wall-clock timing for that point's `reindex`/`storm` calls |
| `ok`, `error` | Whether the point succeeded; the failure reason if not |
| `requests`, `errors`, `qps`, `p50_ms`, `p95_ms`, `p99_ms`, `max_ms` | `nova-storm`'s own summary fields, straight from its `--json` output |
| `full_recall.mean`/`.n`, `short_recall.mean`/`.n`, `total_recall.mean`/`.n`, `empty_ground_truth`, `filter_overreturn` | `nova-storm`'s recall buckets, flattened from its `--json` output: `full` = queries with ≥`top_k` ground-truth ids (scored against `top_k`), `short` = fewer (scored against their own length), `total` = both; `empty_ground_truth` counts firings whose ground truth was present but empty (excluded from every bucket); `filter_overreturn` counts suspected filter leaks — firings where, with a filter configured, the vdb returned more ids than the (exhaustive) ground truth holds (visibility only — recall unchanged) |
| `full_recall_tolerant`, `ties.mean`/`.max`/`.fraction_of_queries`, `tie_epsilon`, `tie_epsilon_source`, `tie_disabled_reason`, `missing_from_gt`, `short_returns`, `full_recall_queries`, `top_k`, `schema_version` | the tie-aware half of the same summary (see [Storm → Recall measurement](../storm/recall.md)). `full_recall` is the exact lower bound and `full_recall_tolerant` the tie-tolerant upper bound; `ties.*` describes how tied the ground truth is at the cutoff; `missing_from_gt` is non-zero only when results beat the ground truth's cutoff yet are absent from it (a stale ground truth or the wrong corpus). All of the tie fields are `null` when tie reporting was disabled or never configured. **`schema_version` is what distinguishes a run whose ground truth was truncated to `top_k` from an older one where it was not — `full_recall` means something stricter from version 2 on.** |

A failed `reindex`/`storm` call records an error row for the remaining
points under it and moves on — it doesn't abort the whole sweep. A
pre-existing-collection collision (above) is the one thing that *does* abort
the whole run, since it's a config/intent problem, not a runtime failure.

## Distribution

`nova sweep` is single-machine and sequential today — no `--num-jobs`/
`--job-rank`. This isn't an oversight: a single target's slices run
one after another regardless (contending for the same instance buys nothing),
so sharding only has a coherent meaning once there's more than one
independent target to spread slices across. That (bin-packing slices across
N SkyPilot-launched Qdrant instances, plus `nova dist sweep`'s real fan-out)
is designed for but not built — `nova dist sweep <config>` exists today only
as a stub that tells you so and exits.
