//! `nova-storm` — the user-facing front controller for `nova storm`.
//!
//! A thin shim: it reads only `target.type` from the config and `exec`s the
//! matching backend (`nova-storm-<type>`, e.g. `nova-storm-qdrant`) with argv
//! unchanged. All the dispatch logic lives in the shared [`nova_shim`] crate so
//! it stays identical to the `nova-load` shim; only the [`nova_shim::Spec`]
//! below differs. (Because it execs, storm's single `--json` summary line is
//! emitted straight from the backend, unwrapped.)

use std::process::ExitCode;

fn main() -> ExitCode {
    nova_shim::dispatch(&nova_shim::Spec {
        program: "nova-storm",
        dispatch_key: &["target", "type"],
        backend_prefix: "nova-storm-",
        default_type: "qdrant",
        install_hint: "`make storm` or `cargo install --path backends/nova-storm/qdrant`",
    })
}
