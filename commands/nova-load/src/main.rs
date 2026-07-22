//! `nova-load` — the user-facing front controller for `nova load`.
//!
//! A thin shim: it reads only `vectorstore.type` from the config and `exec`s the
//! matching backend (`nova-load-<type>`, e.g. `nova-load-qdrant`) with argv
//! unchanged. All the dispatch logic lives in the shared [`nova_shim`] crate so
//! it stays identical to the `nova-storm` shim; only the [`nova_shim::Spec`]
//! below differs.

use std::process::ExitCode;

fn main() -> ExitCode {
    nova_shim::dispatch(&nova_shim::Spec {
        program: "nova-load",
        dispatch_key: &["vectorstore", "type"],
        backend_prefix: "nova-load-",
        default_type: "qdrant",
        install_hint: "`make load` or `cargo install --path backends/nova-load/qdrant`",
    })
}
