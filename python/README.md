# supernova — Python side

The Python half of the polyglot `supernova` toolset. The Rust half lives in
`../crates`. The two are **not** code-coupled: they meet only at runtime through
the git-style `nova` dispatcher and shared data contracts (the YAML config,
parquet/point formats).

## The dispatcher model

`nova <cmd> [args...]` finds an executable named `nova-<cmd>` on `PATH` and
replaces itself with it (`os.execv`). A command can be implemented in any
language:

- **Rust** — a binary like `nova-load`, installed by `cargo install`.
- **Python** — a console script like `nova-embed`, installed by `pip`.

The dispatcher doesn't know or care which. To add a command, just put a
`nova-<name>` executable on `PATH`.

## Packages

| Package     | Location         | Provides     | Deps                | Install where        |
|-------------|------------------|--------------|---------------------|----------------------|
| `nova-cli`  | repo root        | `nova`       | none                | everywhere (instant) |
| `nova-embed`| `python/nova-embed` | `nova-embed` | torch, sentence-transformers | embedding machines |

The **dispatcher lives at the repo root** (`pyproject.toml` + `src/nova_cli/`),
so `uv pip install -e .` from the root installs `nova`. It's the project's front
door and the spine of the polyglot tool, so it sits at the top rather than buried
as just-another-package. It's deliberately dependency-free.

Heavy commands are separate packages under `python/`, installed only where
needed — like `git-*` subcommands.

## Dev setup

```sh
uv pip install -e .                     # the `nova` dispatcher (from repo root)
uv pip install -e python/nova-embed     # embedding command (heavy)
cargo install --path crates/nova-load   # the `nova-load` Rust binary

nova --help        # lists discovered nova-* commands
nova load inspect configs/loader/test.yaml
nova embed ...
```

> Ensure your Python user-scripts dir (e.g. `~/.local/bin` or
> `~/Library/Python/X.Y/bin`) and `~/.cargo/bin` are on `PATH`.
