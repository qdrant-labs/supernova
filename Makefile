# supernova — install the polyglot CLI and its sub-tools.
#
#   make all      install the `nova` dispatcher + every sub-tool (embed, load, storm, inspect, bf, dist, sweep)
#   make cli      just the `nova` dispatcher (zero deps, instant)
#   make embed    the `nova embed` Python tool (heavy: torch, sentence-transformers)
#   make load     the `nova load` Rust binary
#   make storm    the `nova storm` Rust binary
#   make inspect  the `nova inspect` Rust binary (count vectors + parquet schema)
#   make bf       the `nova bf` Python tool (brute-force ground truth; torch + native readers)
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
# Extra backends (elastic, opensearch, milvus) are OFF by default so the common
# qdrant-only install stays fast (they pull the elasticsearch/opensearch +
# milvus/gRPC dep trees). Opt in:
#   make load LOAD_FEATURES=elastic,opensearch,milvus   (needs `protoc` for milvus)
# The released fleet binary (rust-binaries.yml) always ships them.
LOAD_FEATURES ?=
load:
	cargo install --path crates/nova-load $(if $(LOAD_FEATURES),--features $(LOAD_FEATURES))

# `nova storm` — Rust binary, into ~/.cargo/bin.
# Same knob as `load`: the extra QUERY targets are off by default and opted into
# the same way. (It used to be missing entirely, so a `make storm` install could
# never reach a non-qdrant target even though the code was there.)
#   make storm STORM_FEATURES=elastic,opensearch,milvus
# Unlike `load`, the released fleet binary ships only `opensearch` — storm's
# elastic and milvus targets have never been built in CI, so they stay opt-in
# until a change can verify them.
STORM_FEATURES ?=
storm:
	cargo install --path crates/nova-storm $(if $(STORM_FEATURES),--features $(STORM_FEATURES))

# `nova inspect` — Rust binary, into ~/.cargo/bin.
inspect:
	cargo install --path crates/nova-inspect

# `nova bf` — Python brute-force ground truth. `[compute]` pulls torch (GPU);
# drop the extra for a controller that only runs `nova bf merge`.
# Two native companions are built from source (needs cargo): nova-textscan
# (match_text scan) and rust_fetch (python/nova-bf/rust_io: the native S3/dense
# reader behind NOVA_BF_NATIVE_DENSE / NOVA_BF_NATIVE_FETCH). Both are optional
# at runtime -- nova-bf falls back to pyarrow / Python when they are missing.
bf:
	uv pip install -e 'python/nova-bf[compute]' ./crates/nova-textscan ./python/nova-bf/rust_io

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

# rust_io is a standalone crate (own [workspace]), so the root `cargo test`
# does not reach it; --no-default-features skips the pyo3 bindings, which need
# libpython at link time. Its Python tests run in nova-bf's env (S3 cases skip
# unless RUST_FETCH_TEST_S3_ENDPOINT points at a minio).
NOVA_BF_NATIVE := --with $(CURDIR)/crates/nova-textscan --with $(CURDIR)/python/nova-bf/rust_io
test:
	cargo test
	cargo test --manifest-path python/nova-bf/rust_io/Cargo.toml --no-default-features
	uv run --directory python/nova-embed --extra dev pytest -q || true
	uv run --directory python/nova-bf --extra dev $(NOVA_BF_NATIVE) pytest -q
	uv run --directory python/nova-bf --extra dev $(NOVA_BF_NATIVE) pytest -q rust_io/tests
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
	cargo clean --manifest-path python/nova-bf/rust_io/Cargo.toml
	rm -rf site dist python/*/dist
