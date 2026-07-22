# `nova-storm` backend contract

This directory holds the **language-neutral** contract for `nova storm`
backends: [`v1.yaml`](v1.yaml). It is the canonical description of what any
`nova-storm-<backend>` executable must do, regardless of language.

See [`../nova-load/README.md`](../nova-load/README.md) for the full explanation
of the command/backend/contract model and the two-layer (compile-time +
runtime) enforcement scheme — it applies identically here. This file covers only
what's specific to `nova storm`.

## Layout

```
commands/
  nova-storm/                # user-facing shim: `nova storm` → picks a backend
backends/
  nova-storm/
    contracts/rust/          # shared Rust interface (compile-time enforcement)
    qdrant/                  # the Qdrant backend  → builds `nova-storm-qdrant`
contracts/
  nova-storm/
    v1.yaml                  # canonical language-neutral contract (this dir)
    README.md
```

## The two layers, for storm

1. **Native Rust interface.** `backends/nova-storm/contracts/rust` defines the
   `QueryTarget` trait plus `BatchOutcome` and `TargetError`. Every Rust storm
   backend depends on this crate and `impl QueryTarget for …`; a wrong or
   missing method is a compile error.

2. **Language-neutral contract.** [`v1.yaml`](v1.yaml) declares the required
   commands, method names, load-generation modes, and features. Backends
   advertise theirs via `capabilities --json`; `nova contract check` compares.

## Shim / backend-executable model

- `commands/nova-storm` builds the shim **`nova-storm`**. It reads only
  `target.type` from the config, maps it (`qdrant` → `nova-storm-qdrant`), and
  `exec`s the backend with the original args. Crucially, because it execs, the
  backend's stdout — including the single `--json` summary line a caller like
  `nova sweep` parses — is emitted directly, with no wrapper on it.
- `backends/nova-storm/qdrant` builds the backend **`nova-storm-qdrant`**.

Note storm's CLI is `nova-storm-qdrant <config> [--json]` (a positional config,
not subcommands) plus the `capabilities` descriptor command. The contract's
`run` command names that default positional invocation.

## Running contract checks

```bash
nova contract check "$(command -v nova-storm-qdrant)" --contract contracts/nova-storm/v1.yaml
# dev build:
nova-contract check ./target/debug/nova-storm-qdrant --contract contracts/nova-storm/v1.yaml
```

`--level live` runs the storm itself (the contract's `live_check` is just
`{config}`), so point `--config` at a throwaway collection with a short
`load.duration_s`.

## Storm-specific invariants a backend must honor

These are in `v1.yaml`'s `behavior_notes` and are load-bearing for correctness
across a fleet — do not "simplify" them away in a new backend:

- **Closed-loop vs open-loop** are distinct load shapes; open-loop paces on a
  virtual schedule to avoid coordinated omission.
- **A failed dispatch is a recorded sample** (`ok:false`), never a hard abort.
- **Percentiles come from raw samples**, nearest-rank, computed once — never
  averaged per-worker (so fleet-wide merges stay correct).
- **Work is replicated** across fleet workers (every worker runs the same query
  mix), in contrast to nova-load's stride partitioning.
- **Recall** is `hits/top_k` per query when ground truth is provided, aggregated
  across queries; absent (not `0.0`) when there's no ground truth.

## Adding a new backend

Same recipe as nova-load: create `backends/nova-storm/<name>/` building
`nova-storm-<name>`, implement `QueryTarget` (Rust) or the equivalent, advertise
`capabilities --json` with `contract: nova-storm-backend/v1`, and make
`nova contract check` pass. A config with `target.type: <name>` then dispatches
automatically — no shim change.
