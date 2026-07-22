//! Shared front-controller logic for the `nova-*` command shims.
//!
//! A shim (`commands/nova-load`, `commands/nova-storm`) reads only enough of the
//! config to learn which backend to use, maps that to a backend executable, and
//! **replaces this process** with it via `execv`, passing the original args
//! through untouched. Because it execs rather than spawns, the backend inherits
//! stdin/stdout/stderr and its exit code / signals surface directly — the shim
//! adds no layer at runtime. (This is why storm's single `--json` summary line
//! is emitted straight from the backend, unwrapped.)
//!
//! All of that is identical across shims; only the config key, the backend-name
//! prefix, and a few labels differ. Those live in [`Spec`], so the behavior is
//! defined here exactly once and the load/storm shims cannot drift apart.
//!
//! Backend resolution prefers an executable sitting next to the shim (so a
//! `target/debug` dev build finds its sibling backend without install), then
//! falls back to a normal `PATH` lookup (so `cargo install` layouts work).

use std::ffi::OsString;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, ExitCode};

/// What distinguishes one shim from another. Everything else in [`dispatch`] is
/// shared.
pub struct Spec {
    /// This shim's program name, used in error messages, e.g. `"nova-load"`.
    pub program: &'static str,
    /// Nested config key that selects the backend, e.g.
    /// `&["vectorstore", "type"]` (load) or `&["target", "type"]` (storm).
    pub dispatch_key: &'static [&'static str],
    /// Backend executables are named `<prefix><type>`, e.g. `"nova-load-"`.
    pub backend_prefix: &'static str,
    /// Backend type assumed when no config file is present in the args (so
    /// `--help`, `--version`, and `capabilities` still reach a real backend).
    pub default_type: &'static str,
    /// Human hint appended to the "backend not found" error, e.g.
    /// ``"`make load` or `cargo install --path backends/nova-load/qdrant`"``.
    pub install_hint: &'static str,
}

/// Resolve the backend from the config type in the process args and `exec` it,
/// preserving argv. On success this never returns (the process is replaced);
/// the returned [`ExitCode`] is only reached on a resolution/exec failure.
pub fn dispatch(spec: &Spec) -> ExitCode {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();

    let backend_type = match resolve_backend_type(spec, &args) {
        Ok(t) => t,
        Err(msg) => {
            eprintln!("{}: {msg}", spec.program);
            return ExitCode::from(2);
        }
    };

    let backend = format!("{}{backend_type}", spec.backend_prefix);
    let program = resolve_backend_program(&backend);

    // Replace this process with the backend. On success this never returns.
    let err = Command::new(&program).args(&args).exec();

    // Only reached if exec itself failed (e.g. backend not installed).
    eprintln!(
        "{}: failed to exec backend `{}` (for {}=`{backend_type}`): {err}\n\
         hint: install it, e.g. {}",
        spec.program,
        program.to_string_lossy(),
        spec.dispatch_key.join("."),
        spec.install_hint,
    );
    ExitCode::from(127)
}

/// Inspect the args for a config file and return its dispatch type. Falls back
/// to `spec.default_type` when no config file is present (help/version/capabilities).
fn resolve_backend_type(spec: &Spec, args: &[OsString]) -> Result<String, String> {
    for arg in args {
        // Skip flags; the config is always a bare positional path.
        if arg.to_string_lossy().starts_with('-') {
            continue;
        }
        let path = Path::new(arg);
        if !path.is_file() {
            continue; // subcommand tokens (`load`, `prepare`, `capabilities`, …) are not files
        }
        // First existing file among the args is the config.
        return read_dispatch_type(spec, path);
    }
    Ok(spec.default_type.to_string())
}

/// Parse `path` as YAML and read the nested `spec.dispatch_key` string. `${VAR}`
/// references elsewhere in the config parse fine as plain scalars — the shim does
/// not need to expand them, only to read the (literal) backend type.
fn read_dispatch_type(spec: &Spec, path: &Path) -> Result<String, String> {
    let key = spec.dispatch_key.join(".");
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read config `{}`: {e}", path.display()))?;
    let value: serde_yaml::Value = serde_yaml::from_str(&text)
        .map_err(|e| format!("failed to parse config `{}` as YAML: {e}", path.display()))?;

    let mut node = &value;
    for k in spec.dispatch_key {
        node = node
            .get(k)
            .ok_or_else(|| format!("config `{}` is missing `{key}`", path.display()))?;
    }
    node.as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("`{key}` in `{}` must be a string", path.display()))
}

/// Resolve the backend executable: prefer a sibling next to this shim (dev
/// builds, and installs where both land in the same dir), else let `exec`
/// resolve `backend` on `PATH`.
fn resolve_backend_program(backend: &str) -> OsString {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(backend);
            if candidate.is_file() {
                return candidate.into_os_string();
            }
        }
    }
    backend.into()
}
