# supernova — install the polyglot CLI and its sub-tools.
#
#   make all      install the `nova` dispatcher + every sub-tool (embed, load, storm, inspect)
#   make cli      just the `nova` dispatcher (zero deps, instant)
#   make embed    the `nova embed` Python tool (heavy: torch, sentence-transformers)
#   make load     the `nova load` Rust binary
#   make storm    the `nova storm` Rust binary
#   make inspect  the `nova inspect` Rust binary (count vectors + parquet schema)
#   make bf       the `nova bf` Python tool (brute-force ground truth; torch)
#   make dist     the `nova dist` orchestrator (SkyPilot; controller-side only)
#   make docs     serve the docs locally (zensical)
#   make test     run Rust + Python tests
#
# Sub-tools follow the git model: each installs a `nova-<cmd>` on PATH, and the
# `nova` dispatcher execs it. Install only the ones you need.

.PHONY: all cli embed load storm inspect bf dist docs docs-build test clean

all: cli embed load storm inspect bf dist
	@echo
	@echo "✓ installed nova + embed/load/storm/inspect/bf/dist. Check with: nova --help"

# The `nova` dispatcher (root pyproject). Zero deps — installs anywhere instantly.
cli:
	uv pip install -e .

# `nova embed` — Python, with the ML stack (torch, sentence-transformers, …).
embed:
	uv pip install -e 'python/nova-embed[embed]'

# `nova load` — Rust binary, into ~/.cargo/bin.
load:
	cargo install --path crates/nova-load

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
# dispatch box); workers never need it. Not part of `make all`.
dist:
	uv pip install -e python/nova-dist

# Live docs at http://localhost:8000 (no install needed; uvx fetches zensical).
docs:
	uvx zensical serve

docs-build:
	uvx zensical build

test:
	cargo test
	uv run --extra dev pytest python/nova-embed -q || true

clean:
	cargo clean
	rm -rf site dist python/*/dist
