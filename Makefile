# supernova — install the polyglot CLI and its sub-tools.
#
#   make all      install the `nova` dispatcher + every sub-tool (embed, load, storm, inspect, bf, dist, sweep)
#   make cli      just the `nova` dispatcher (zero deps, instant)
#   make embed    the `nova embed` Python tool (heavy: torch, sentence-transformers)
#   make load     the `nova load` Rust binary
#   make storm    the `nova storm` Rust binary
#   make inspect  the `nova inspect` Rust binary (count vectors + parquet schema)
#   make bf       the `nova bf` Python tool (brute-force ground truth; torch)
#   make dist     the `nova dist` orchestrator (SkyPilot; controller-side only)
#   make sweep    the `nova sweep` parameter-sweep orchestrator (controller-side only)
#   make docs     serve the docs locally (zensical)
#   make test     run Rust + Python tests
#   make parity   nova-bf ground-truth parity vs naive + live Qdrant (see tests/parity)
#
# Sub-tools follow the git model: each installs a `nova-<cmd>` on PATH, and the
# `nova` dispatcher execs it. Install only the ones you need.

.PHONY: all cli embed load storm inspect bf dist sweep docs docs-build test parity clean

all: cli embed load storm inspect bf dist sweep
	@echo
	@echo "✓ installed nova + embed/load/storm/inspect/bf/dist/sweep. Check with: nova --help"

# The `nova` dispatcher (root pyproject). Zero deps — installs anywhere instantly.
cli:
	uv pip install -e .

# `nova embed` — Python, with the ML stack (torch, sentence-transformers, …).
embed:
	uv pip install -e 'python/nova-embed[embed]'

# `nova load` — Rust binary, into ~/.cargo/bin.
# Extra backends (elastic, milvus) are OFF by default so the common qdrant-only
# install stays fast (they pull the elasticsearch + milvus/gRPC dep trees). Opt in:
#   make load LOAD_FEATURES=elastic,milvus     (needs `protoc` for milvus)
# The released fleet binary (rust-binaries.yml) always ships them.
LOAD_FEATURES ?=
load:
	cargo install --path crates/nova-load $(if $(LOAD_FEATURES),--features $(LOAD_FEATURES))

# `nova storm` — Rust binary, into ~/.cargo/bin.
storm:
	cargo install --path crates/nova-storm

# `nova inspect` — Rust binary, into ~/.cargo/bin.
inspect:
	cargo install --path crates/nova-inspect

# `nova bf` — Python brute-force ground truth. `[compute]` pulls torch (GPU);
# drop the extra for a controller that only runs `nova bf merge`.
bf:
	uv pip install -e 'python/nova-bf[compute]'

# `nova dist` — SkyPilot orchestrator. Controller-side only (your laptop / a
# dispatch box); workers never need it. Still part of `make all` (pulls in
# skypilot[aws], the heaviest dep in the whole install).
dist:
	uv pip install -e python/nova-dist

# `nova sweep` — parameter sweep orchestrator (drives nova-load/nova-storm
# subprocesses). Controller-side only, same precedent as `dist` — but, like
# `dist`, still part of `make all`.
sweep:
	uv pip install -e python/nova-sweep

# Live docs at http://localhost:8000 (no install needed; uvx fetches zensical).
docs:
	uvx zensical serve

docs-build:
	uvx zensical build

test:
	cargo test
	uv run --directory python/nova-embed --extra dev pytest -q || true
	uv run --directory python/nova-bf --extra dev pytest -q || true
	uv run --directory python/nova-sweep --extra dev pytest -q || true

# nova-bf three-way parity: nova-bf vs a plain-Python reference vs a live
# Qdrant, over dense/sparse/multivector, the filter language, sharded
# compute+merge and the Triton kernels. Starts a throwaway Qdrant if none is
# reachable, and picks up a GPU on its own.
# See python/nova-bf/tests/parity/README.md
parity:
	uv run --directory python/nova-bf --extra dev --with qdrant-client \
		bash ../../scripts/parity.sh

clean:
	cargo clean
	rm -rf site dist python/*/dist
