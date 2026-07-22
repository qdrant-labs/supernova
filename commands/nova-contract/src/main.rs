//! `nova-contract` — a generic, language-neutral backend conformance checker.
//!
//! Usage:
//!   nova contract check <backend-exe> --contract <path> [--level ...] [--json]
//!   nova-contract check <backend-exe> --contract <path> ...
//!
//! It runs the backend's `capabilities --json`, then validates that descriptor
//! against a shared contract spec (`contracts/<command>/vN.yaml`). This is the
//! cross-language half of the two-layer model: the per-language interface crate
//! (`backends/<command>/contracts/<language>/`) enforces the contract at compile
//! time within one language; this checker enforces it at runtime for *any*
//! backend, whatever language it's written in, by only ever talking to the
//! backend's executable.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::Deserialize;

#[derive(Debug, Parser)]
#[command(name = "nova-contract", version, about = "Language-neutral backend contract checker")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Check a backend executable against a contract spec.
    Check(CheckArgs),
}

#[derive(Debug, Args)]
struct CheckArgs {
    /// The backend executable to check: a path (e.g. `target/debug/nova-load-qdrant`)
    /// or a name resolved on `PATH` (e.g. `nova-load-qdrant`). NOTE: point this at
    /// a *backend*, not a user-facing shim like `nova-load`.
    backend: String,
    /// Path to the language-neutral contract YAML (e.g. contracts/nova-load/v1.yaml).
    #[arg(long)]
    contract: PathBuf,
    /// How hard to check. `shape`: capabilities vs contract only. `dry-run`
    /// (default): + behavioral checks needing no live backend. `live`: + run a
    /// declared live check (requires `--config`).
    #[arg(long, value_enum, default_value_t = Level::DryRun)]
    level: Level,
    /// Directory of conformance fixtures (reserved for fixture-driven checks).
    #[arg(long)]
    fixtures: Option<PathBuf>,
    /// Config file for `--level live` (substituted into the contract's live_check).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Emit the report as a single JSON object instead of a human-readable table.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Level {
    Shape,
    #[value(name = "dry-run")]
    DryRun,
    Live,
}

/// The language-neutral contract spec. Unknown keys (description, behavior
/// notes, etc.) are ignored on purpose — they're human docs, not machine rules.
#[derive(Debug, Deserialize)]
struct Contract {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    version: u32,
    /// The full `id/version` string a conforming backend must advertise.
    contract: String,
    #[serde(default)]
    required_commands: Vec<String>,
    #[serde(default)]
    required_methods: Vec<String>,
    #[serde(default)]
    required_vector_kinds: Vec<String>,
    #[serde(default)]
    required_point_id_types: Vec<String>,
    #[serde(default)]
    required_search_modes: Vec<String>,
    #[serde(default)]
    required_features: Vec<String>,
    #[serde(default)]
    required_flags: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    live_check: Option<LiveCheck>,
}

/// A backend command the `live` level runs against a real target. `{config}` in
/// `args` is replaced by the `--config` path.
#[derive(Debug, Deserialize)]
struct LiveCheck {
    args: Vec<String>,
}

/// One validation result.
struct Check {
    name: &'static str,
    ok: bool,
    detail: String,
}

impl Check {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Check { name, ok: true, detail: detail.into() }
    }
    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Check { name, ok: false, detail: detail.into() }
    }
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Cmd::Check(args) => check(args),
    }
}

fn check(args: CheckArgs) -> ExitCode {
    let contract = match load_contract(&args.contract) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("nova-contract: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut checks: Vec<Check> = Vec::new();

    // --- Run `capabilities --json` (the one thing every backend must do). ---
    let caps_json = match run_capabilities(&args.backend) {
        Ok((true, stdout, _)) => {
            checks.push(Check::pass("capabilities-runs", "`capabilities --json` exited 0"));
            stdout
        }
        Ok((false, stdout, stderr)) => {
            checks.push(Check::fail(
                "capabilities-runs",
                format!("`capabilities --json` exited non-zero; stderr: {}", stderr.trim()),
            ));
            stdout
        }
        Err(e) => {
            // Can't even spawn the backend — report and stop.
            checks.push(Check::fail("capabilities-runs", e));
            return report(&args, &contract, &checks);
        }
    };

    // --- Parse the descriptor. ---
    let caps: serde_json::Value = match serde_json::from_str(&caps_json) {
        Ok(v) => {
            checks.push(Check::pass("capabilities-json-parses", "valid JSON"));
            v
        }
        Err(e) => {
            checks.push(Check::fail("capabilities-json-parses", format!("invalid JSON: {e}")));
            return report(&args, &contract, &checks);
        }
    };

    // --- SHAPE checks: descriptor vs contract. ---
    shape_checks(&contract, &caps, &mut checks);

    // --- DRY-RUN checks: cheap behavioral checks, no live backend. ---
    if args.level >= Level::DryRun {
        dry_run_checks(&args.backend, &caps_json, &mut checks);
    }

    // --- LIVE checks: run a declared command against a real target. ---
    if args.level == Level::Live {
        live_checks(&args, &contract, &mut checks);
    }

    report(&args, &contract, &checks)
}

fn shape_checks(contract: &Contract, caps: &serde_json::Value, checks: &mut Vec<Check>) {
    // Contract id/version must match exactly.
    let advertised = caps.get("contract").and_then(|v| v.as_str()).unwrap_or("");
    if advertised == contract.contract {
        checks.push(Check::pass("contract-id-matches", format!("`{advertised}`")));
    } else {
        checks.push(Check::fail(
            "contract-id-matches",
            format!("backend advertises `{advertised}`, contract expects `{}`", contract.contract),
        ));
    }

    subset_check("commands", &contract.required_commands, caps, "commands", checks);
    subset_check("methods", &contract.required_methods, caps, "methods", checks);
    subset_check("vector-kinds", &contract.required_vector_kinds, caps, "vector_kinds", checks);
    subset_check("point-id-types", &contract.required_point_id_types, caps, "point_id_types", checks);
    subset_check("search-modes", &contract.required_search_modes, caps, "search_modes", checks);
    subset_check("features", &contract.required_features, caps, "features", checks);

    // Required flags: `flags` is a map of command -> advertised flags.
    if !contract.required_flags.is_empty() {
        let advertised_flags = caps.get("flags");
        for (cmd, required) in &contract.required_flags {
            let have: Vec<String> = advertised_flags
                .and_then(|m| m.get(cmd))
                .map(json_str_array)
                .unwrap_or_default();
            let missing: Vec<&String> = required.iter().filter(|f| !have.contains(f)).collect();
            let name: &'static str = Box::leak(format!("flags[{cmd}]").into_boxed_str());
            if missing.is_empty() {
                checks.push(Check::pass(name, format!("advertises {required:?}")));
            } else {
                checks.push(Check::fail(name, format!("missing {missing:?} (has {have:?})")));
            }
        }
    }
}

/// Assert every entry in `required` appears in `caps[caps_key]`.
fn subset_check(
    name: &'static str,
    required: &[String],
    caps: &serde_json::Value,
    caps_key: &str,
    checks: &mut Vec<Check>,
) {
    if required.is_empty() {
        return; // contract declares nothing here — skip silently
    }
    let have = caps.get(caps_key).map(json_str_array);
    match have {
        None => checks.push(Check::fail(name, format!("backend advertises no `{caps_key}`"))),
        Some(have) => {
            let missing: Vec<&String> = required.iter().filter(|r| !have.contains(r)).collect();
            if missing.is_empty() {
                checks.push(Check::pass(name, format!("all {} present", required.len())));
            } else {
                checks.push(Check::fail(name, format!("missing {missing:?}")));
            }
        }
    }
}

fn dry_run_checks(backend: &str, first: &str, checks: &mut Vec<Check>) {
    // capabilities must be deterministic — a moving descriptor can't be a
    // contract. Run it again and compare byte-for-byte.
    match run_capabilities(backend) {
        Ok((_, second, _)) if second == first => {
            checks.push(Check::pass("capabilities-deterministic", "identical across two runs"));
        }
        Ok((_, _, _)) => {
            checks.push(Check::fail("capabilities-deterministic", "descriptor changed between runs"));
        }
        Err(e) => checks.push(Check::fail("capabilities-deterministic", e)),
    }
}

fn live_checks(args: &CheckArgs, contract: &Contract, checks: &mut Vec<Check>) {
    let Some(live) = &contract.live_check else {
        checks.push(Check::fail(
            "live-check",
            "contract declares no `live_check`; nothing to run at --level live",
        ));
        return;
    };
    let Some(config) = &args.config else {
        checks.push(Check::fail("live-check", "--level live requires --config <path>"));
        return;
    };
    // Substitute {config} in the declared args.
    let subbed: Vec<String> = live
        .args
        .iter()
        .map(|a| a.replace("{config}", &config.display().to_string()))
        .collect();
    match Command::new(&args.backend).args(&subbed).output() {
        Ok(out) if out.status.success() => {
            checks.push(Check::pass("live-check", format!("`{}` exited 0", subbed.join(" "))));
        }
        Ok(out) => checks.push(Check::fail(
            "live-check",
            format!(
                "`{}` exited non-zero; stderr: {}",
                subbed.join(" "),
                String::from_utf8_lossy(&out.stderr).trim()
            ),
        )),
        Err(e) => checks.push(Check::fail("live-check", format!("failed to spawn: {e}"))),
    }
}

/// Read `<backend> capabilities --json`. Returns (exit-ok, stdout, stderr).
fn run_capabilities(backend: &str) -> Result<(bool, String, String), String> {
    let out = Command::new(backend)
        .arg("capabilities")
        .arg("--json")
        .output()
        .map_err(|e| format!("failed to spawn `{backend} capabilities --json`: {e}"))?;
    Ok((
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    ))
}

fn json_str_array(v: &serde_json::Value) -> Vec<String> {
    v.as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
        .unwrap_or_default()
}

fn load_contract(path: &PathBuf) -> Result<Contract, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read contract `{}`: {e}", path.display()))?;
    serde_yaml::from_str(&text)
        .map_err(|e| format!("failed to parse contract `{}`: {e}", path.display()))
}

/// Print the report (human or JSON) and return the process exit code.
fn report(args: &CheckArgs, contract: &Contract, checks: &[Check]) -> ExitCode {
    let ok = checks.iter().all(|c| c.ok);

    if args.json {
        let arr: Vec<serde_json::Value> = checks
            .iter()
            .map(|c| serde_json::json!({ "name": c.name, "ok": c.ok, "detail": c.detail }))
            .collect();
        let obj = serde_json::json!({
            "backend": args.backend,
            "contract": contract.contract,
            "level": format!("{:?}", args.level).to_lowercase(),
            "ok": ok,
            "checks": arr,
        });
        println!("{}", serde_json::to_string_pretty(&obj).expect("report serializes"));
    } else {
        println!("contract check: {} vs {}", args.backend, contract.contract);
        for c in checks {
            let mark = if c.ok { "✓" } else { "✗" };
            println!("  {mark} {:<26} {}", c.name, c.detail);
        }
        println!("{}", if ok { "PASS" } else { "FAIL" });
    }

    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
