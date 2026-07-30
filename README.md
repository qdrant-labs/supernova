<p align="center">
  <img src="docs/fig/supernova_logo.svg" alt="supernova" width="380">
</p>

# supernova

Generate massive pre-embedded datasets, load them into vector databases,
load-test the result, and expose core workflows via web APIs.

## Install

Requirements: [uv](https://docs.astral.sh/uv/), and [Rust](https://rustup.rs/)
(`cargo`) for the `load` / `storm` tools.

```bash
make all        # nova dispatcher + embed + load + storm + web + dist
```

Or install just what you need:

```bash
make cli        # the `nova` dispatcher only (zero deps, instant)
make embed      # nova embed   (heavy: torch, sentence-transformers)
make load       # nova load    (Rust binary)
make storm      # nova storm    (Rust binary)
make web        # nova web     (Rust web service: Axum + dashboard assets)
make dist       # nova dist    (SkyPilot orchestration; controller-side)
```

Make sure your tool dirs are on `PATH` so `nova` can find the sub-tools:

```bash
export PATH="$HOME/.cargo/bin:$PATH"          # Rust binaries (nova-load, nova-storm, nova-web)
# and your uv/pip user-scripts dir for nova / nova-embed, e.g.
export PATH="$HOME/.local/bin:$PATH"
```

Verify:

```bash
nova --help     # lists every nova-* tool found on PATH
```

## Quickstart

```bash
# 1. Embed a dataset → parquet
nova embed configs/embedder/test.yaml

# 2. Load the parquet into Qdrant
nova load run configs/loader/test.yaml

# 3. Load-test the collection
nova storm configs/storm/test.yaml
```

Every run is driven by a YAML config; `${VAR}` / `${VAR:-default}` references are
expanded from the environment. See the [docs](#docs) for each tool's config.

### Distributed

Each tool partitions its own work, so a fleet is N copies with a rank — the rank
is the only thing that differs between workers:

```bash
nova embed configs/embedder/test.yaml --num-jobs 50 --job-rank $RANK
# load splits into prepare (once) / load (per worker) / finalize (once):
nova load prepare configs/loader/test.yaml
nova load load    configs/loader/test.yaml --num-jobs 50 --job-rank $RANK
nova load finalize configs/loader/test.yaml
```

Resume a distributed worker from its last completed file:

```bash
nova load load configs/loader/test.yaml --num-jobs 50 --job-rank $RANK --resume
```

By default, checkpoints are persisted under `.nova-load-checkpoints/` on each
worker. You can override the location per run:

```bash
nova load load configs/loader/test.yaml --num-jobs 50 --job-rank $RANK \
  --resume --checkpoint-path /mnt/checkpoints/load.json
```

Recovery drill (interrupted distributed load):

```bash
# 1) Create/defer once (safe to rerun with recreate:false)
nova load prepare configs/loader/test.yaml

# 2) Launch workers
for RANK in $(seq 0 49); do
  nova load load configs/loader/test.yaml --num-jobs 50 --job-rank $RANK --resume &
done

# 3) Simulate interruption and restart workers (same num-jobs/job-rank)
pkill -f "nova-load load"
for RANK in $(seq 0 49); do
  nova load load configs/loader/test.yaml --num-jobs 50 --job-rank $RANK --resume &
done

# 4) Finalize once after all workers complete
nova load finalize configs/loader/test.yaml
```

### Faster S3 startup with a catalog

For very large S3 prefixes, startup can spend significant time listing objects.
`nova-load` supports an optional local parquet catalog to skip S3 listing and
start partitioning immediately.

Add this under `datasource` in a loader config:

```yaml
datasource:
  type: s3
  path: s3://your-bucket/your-prefix/
  catalog: /path/to/catalog_all_with_payload.parquet
```

Catalog columns:

- Path column (first match): `relative_path` (preferred), `path`, or `filename`
- Optional size column: `file_size` or `size`

Notes:

- This speeds up **file discovery/planning** (inspect/prepare/load startup), not
  the per-worker S3 download + upsert throughput.
- Catalog paths are normalized against `datasource.path` prefix.
- On SkyPilot, `datasource.catalog` must point to a file present on each worker.
- Concrete examples:
  - `configs/loader/fineweb_10b_full_with_catalog.yaml` (repo-local path)
  - `configs/loader/fineweb_10b_full_with_catalog_env.yaml` (worker env path)

You can run that yourself on any fleet, or let **`nova dist`** drive SkyPilot for
you (`make dist` to install it). It provisions a pool and submits the ranked jobs.
Compute (resources + how a worker installs the tool) has sensible built-in
defaults, so it works with no extra files:

```bash
nova dist embed configs/embedder/test.yaml --num-jobs 50
nova dist load  configs/loader/test.yaml  --num-jobs 50
nova dist load  configs/loader/test.yaml  --finalize        # after workers finish
nova dist storm configs/storm/test.yaml   --num-jobs 10
```

## Web Service

`nova-web` exposes `nova load` and `nova storm` workflows as async jobs over HTTP
and serves the rebranded Angular dashboard (`supernova-dashboard`) static assets.
It also supports SkyPilot orchestration via `nova dist` passthrough job endpoints.

```bash
# build dashboard assets
cd web/supernova-dashboard
npm install
npm run build

# run web service from repo root
cd ../..
cargo run -p nova-web
```

Environment:
- `PORT` (default `8080`)
- `DIST_DIR` (default `web/supernova-dashboard/dist/supernova-dashboard/browser`)
- `QDRANT_URL` / `QDRANT_API_KEY` for Qdrant helper endpoints
- `NOVA_DIST_BIN` (default `nova`) if `nova dist` is installed under another name

Core endpoints:
- `POST /api/v1/load/run`
- `POST /api/v1/storm/run`
- `POST /api/v1/storm/report`
- `POST /api/v1/dist/load`
- `POST /api/v1/dist/storm`
- `GET /api/v1/jobs`

To override, drop a `~/.nova/skypilot/<tool>.yaml` or pass `--resources my.yaml`
— merged by key over the defaults, so you can change just `setup:` (e.g. a dev
build) and keep the default resources. Add `--dry-run` to inspect the generated
pool/job YAMLs without launching. Templates live in `configs/skypilot/`.

## Project structure

```
supernova/
├── pyproject.toml          # the `nova` dispatcher (src/cli/)
├── src/cli/                # git-style dispatch: nova <cmd> -> nova-<cmd>
├── crates/                 # Rust tools
│   ├── nova-load/          #   nova load
│   ├── nova-storm/         #   nova storm
│   └── nova-web/           #   nova web
├── python/
│   ├── nova-embed/         # nova embed (ML pipeline; [embed] extra)
│   └── nova-dist/          # nova dist  (SkyPilot orchestration)
├── configs/                # example YAML configs (+ skypilot/ resource templates)
├── web/supernova-dashboard/# Angular frontend served by nova-web
├── docs/                   # zensical docs site
└── Makefile
```

## Docs

```bash
make docs       # serve at http://localhost:8000
```

Start with **Getting Started → Installation / Quickstart**, then the per-tool
sections and the **Reference**.
