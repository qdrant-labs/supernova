# supernova — install the polyglot CLI and its sub-tools.
#
#   make all      install the `nova` dispatcher + every sub-tool (embed, load, storm, inspect, bf, dist, sweep, contract)
#   make cli      just the `nova` dispatcher (zero deps, instant)
#   make embed    the `nova embed` Python tool (heavy: torch, sentence-transformers)
#   make load     the `nova load` shim + `nova-load-qdrant` backend (Rust)
#   make storm    the `nova storm` shim + `nova-storm-qdrant` backend (Rust)
#   make contract the `nova contract` conformance checker (Rust)
#   make inspect  the `nova inspect` Rust binary (count vectors + parquet schema)
#   make bf       the `nova bf` Python tool (brute-force ground truth; torch)
#   make dist     the `nova dist` orchestrator (SkyPilot; controller-side only)
#   make sweep    the `nova sweep` parameter-sweep orchestrator (controller-side only)
#   make docs     serve the docs locally (zensical)
#   make test     run Rust + Python tests
#
# Sub-tools follow the git model: each installs a `nova-<cmd>` on PATH, and the
# `nova` dispatcher execs it. `load`/`storm` are now a thin shim command that
# dispatches to a backend executable (`nova-<cmd>-<backend>`); installing them
# installs both. Install only the ones you need.

.PHONY: all cli embed load storm contract inspect bf dist sweep docs docs-build test clean

all: cli embed load storm contract inspect bf dist sweep
	@echo
	@echo "✓ installed nova + embed/load/storm/contract/inspect/bf/dist/sweep. Check with: nova --help"

# The `nova` dispatcher (root pyproject). Zero deps — installs anywhere instantly.
cli:
	uv pip install -e .

# `nova embed` — Python, with the ML stack (torch, sentence-transformers, …).
embed:
	uv pip install -e 'commands/nova-embed[embed]'

# `nova load` — the shim (`nova-load`) + the Qdrant backend (`nova-load-qdrant`),
# both into ~/.cargo/bin. The shim reads `vectorstore.type` and execs the backend.
load:
	cargo install --path commands/nova-load
	cargo install --path backends/nova-load/qdrant

# `nova storm` — the shim (`nova-storm`) + the Qdrant backend (`nova-storm-qdrant`).
storm:
	cargo install --path commands/nova-storm
	cargo install --path backends/nova-storm/qdrant

# `nova contract` — the language-neutral backend conformance checker.
contract:
	cargo install --path commands/nova-contract

# `nova inspect` — Rust binary, into ~/.cargo/bin.
inspect:
	cargo install --path commands/nova-inspect

# `nova bf` — Python brute-force ground truth. `[compute]` pulls torch (GPU);
# drop the extra for a controller that only runs `nova bf merge`.
bf:
	uv pip install -e 'commands/nova-bf[compute]'

# `nova dist` — SkyPilot orchestrator. Controller-side only (your laptop / a
# dispatch box); workers never need it. Still part of `make all` (pulls in
# skypilot[aws], the heaviest dep in the whole install).
dist:
	uv pip install -e commands/nova-dist

# `nova sweep` — parameter sweep orchestrator (drives nova-load/nova-storm
# subprocesses). Controller-side only, same precedent as `dist` — but, like
# `dist`, still part of `make all`.
sweep:
	uv pip install -e commands/nova-sweep

# Live docs at http://localhost:8000 (no install needed; uvx fetches zensical).
docs:
	uvx zensical serve

docs-build:
	uvx zensical build

test:
	cargo test
	# Build the backends + checker, then run contract conformance at shape/dry-run
	# level (no live backend needed). These fail the build if a backend drifts
	# from its contract.
	cargo build -q -p nova-contract -p nova-load-qdrant -p nova-storm-qdrant
	./target/debug/nova-contract check ./target/debug/nova-load-qdrant --contract contracts/nova-load/v1.yaml --level dry-run --fixtures tests/contracts/nova-load
	./target/debug/nova-contract check ./target/debug/nova-storm-qdrant --contract contracts/nova-storm/v1.yaml --level dry-run --fixtures tests/contracts/nova-storm
	uv run --directory commands/nova-embed --extra dev pytest -q || true
	uv run --directory commands/nova-bf --extra dev pytest -q || true
	uv run --directory commands/nova-sweep --extra dev pytest -q || true

clean:
	cargo clean
	rm -rf site dist python/*/dist
