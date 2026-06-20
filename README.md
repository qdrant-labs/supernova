<p align="center">
  <img src="docs/fig/supernova_logo.svg" alt="supernova" width="380">
</p>

# supernova

Generate massive pre-embedded datasets, load them into vector databases, and
load-test the result — at scale.

## Install

Requirements: [uv](https://docs.astral.sh/uv/), and [Rust](https://rustup.rs/)
(`cargo`) for the `load` / `storm` tools.

```bash
make all        # nova dispatcher + embed + load + storm
```

Or install just what you need:

```bash
make cli        # the `nova` dispatcher only (zero deps, instant)
make embed      # nova embed   (heavy: torch, sentence-transformers)
make load       # nova load    (Rust binary)
make storm      # nova storm    (Rust binary)
```

Make sure your tool dirs are on `PATH` so `nova` can find the sub-tools:

```bash
export PATH="$HOME/.cargo/bin:$PATH"          # Rust binaries (nova-load, nova-storm)
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

Each tool partitions its own work, so a fleet is N copies with a rank:

```bash
# one master prepares the collection, then every worker loads its slice
nova load prepare configs/loader/test.yaml
nova load load    configs/loader/test.yaml --num-jobs 50 --job-rank $RANK
nova load finalize configs/loader/test.yaml

nova embed configs/embedder/test.yaml --num-jobs 50 --job-rank $RANK
```

Orchestration (SkyPilot, etc.) is external: it just invokes `nova <tool>` on each
node with that node's rank.

## Project structure

```
supernova/
├── pyproject.toml          # the `nova` dispatcher (src/cli/)
├── src/cli/                # git-style dispatch: nova <cmd> -> nova-<cmd>
├── crates/                 # Rust tools
│   ├── nova-load/          #   nova load
│   └── nova-storm/         #   nova storm
├── python/
│   └── nova-embed/         # nova embed (ML pipeline; [embed] extra)
├── configs/                # example YAML configs
├── docs/                   # zensical docs site
└── Makefile
```

## Docs

```bash
make docs       # serve at http://localhost:8000
```

Start with **Getting Started → Installation / Quickstart**, then the per-tool
sections and the **Reference**.
