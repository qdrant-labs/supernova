# `nova-load` backend contract

This directory holds the **language-neutral** contract for `nova load` backends:
[`v1.yaml`](v1.yaml). It is the canonical description of what any
`nova-load-<backend>` executable must do, regardless of the language it's
written in.

## The command / backend / contract model

`supernova` is organized by **command and backend, not by language**:

```
commands/
  nova-load/                 # user-facing shim: `nova load` → picks a backend
backends/
  nova-load/
    contracts/rust/          # shared Rust interface (compile-time enforcement)
    qdrant/                  # the Qdrant backend  → builds `nova-load-qdrant`
contracts/
  nova-load/
    v1.yaml                  # canonical language-neutral contract (this dir)
    README.md
```

A backend lives at `backends/nova-load/<backend-name>/` and is named for the
**store it targets** (`qdrant`, and one day `milvus`, `vespa`, …), not the
language it happens to be written in. Backends that share a language share a
per-language interface package under `backends/nova-load/contracts/<language>/`.

## The two-layer contract model

There are two complementary layers of enforcement:

1. **Native per-language interface — compile-time, within one language.**
   `backends/nova-load/contracts/rust` is a Rust crate defining the
   `VectorStore` trait plus the neutral data types (`Point`, `VectorValue`,
   `PointId`, `CollectionSchema`, `VectorSpec`, `StoreError`). Every Rust
   backend depends on this crate and `impl VectorStore for …`. If a Rust
   backend is missing a method or has the wrong signature, **it doesn't
   compile**. A future Go backend would get an analogous `contracts/go/`
   package.

2. **Language-neutral contract + `nova contract check` — runtime, across
   languages.** [`v1.yaml`](v1.yaml) declares the required commands, flags,
   method names, vector kinds, and point-id types. Each backend advertises what
   it actually supports via `capabilities --json`. `nova contract check`
   compares the two. This works for a backend in *any* language, because it only
   ever talks to the executable — never to source code.

Keep the three in lockstep when you change the contract: the trait in
`contracts/rust`, the `methods:`/`commands:` in `v1.yaml`, and the
`capabilities --json` each backend prints.

## The shim / backend-executable model

`nova load …` still works exactly as before. Under the hood:

- `commands/nova-load` builds the executable **`nova-load`** — a thin *shim*.
  It reads only `vectorstore.type` from the config, maps it to a backend
  executable (`qdrant` → `nova-load-qdrant`), and **`exec`s** that backend with
  the original args unchanged. Because it execs (replaces the process) rather
  than spawns, stdin/stdout/stderr and the exit code pass straight through — the
  shim adds no runtime layer.
- `backends/nova-load/qdrant` builds **`nova-load-qdrant`** — the real backend.

The `nova` dispatcher finds `nova-load` on `PATH` exactly as before; the shim
then finds `nova-load-qdrant` (preferring a sibling next to itself, else `PATH`).

Contract checks target the **backend** executable, never the shim:

```bash
nova contract check "$(command -v nova-load-qdrant)" --contract contracts/nova-load/v1.yaml
# or against a dev build:
nova-contract check ./target/debug/nova-load-qdrant --contract contracts/nova-load/v1.yaml
```

Levels: `--level shape` (capabilities vs contract only), `dry-run` (default;
adds cheap behavioral checks like capabilities-determinism), `live` (runs the
contract's `live_check` — for load, `inspect <config>` — needs `--config`).

## Adding a new backend (e.g. Milvus)

1. Create `backends/nova-load/milvus/` as a new crate building the executable
   **`nova-load-milvus`**.
2. If it's Rust, depend on `nova-load-contract-rust` and `impl VectorStore` —
   the compiler enforces the method set. If it's another language, implement the
   equivalent interface (add `backends/nova-load/contracts/<language>/` if it's
   the first backend in that language).
3. Implement `capabilities --json` advertising `contract: nova-load-backend/v1`
   and the commands/methods/kinds it supports.
4. `nova contract check nova-load-milvus --contract contracts/nova-load/v1.yaml`
   must pass.
5. No change to `commands/nova-load` is needed: a config with
   `vectorstore.type: milvus` dispatches to `nova-load-milvus` automatically.

Do **not** teach the shim to pretend to be a backend, and do not give
orchestrators (`nova-sweep`) their own backend layer — they call `nova load` /
`nova storm`, which do the dispatch.
