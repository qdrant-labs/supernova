use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

use nova_load::config::LoadConfig;
use nova_load::plan::Partition;

/// Load vectors from a datasource into a vector store, per a YAML config.
#[derive(Debug, Parser)]
#[command(name = "nova-load", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Single-node load: prepare, load every file, then finalize indexing.
    Run(RunArgs),
    /// Master step: create the collection and defer indexing for a fleet load.
    Prepare(RunArgs),
    /// Worker step: load this worker's slice (run `prepare` first; assumes the
    /// collection exists and does not manage indexing).
    Load(LoadArgs),
    /// Master step: re-enable indexing and wait for it to settle. Run once,
    /// after every worker's `load` has finished.
    Finalize(RunArgs),
    /// Patch HNSW/quantization/optimizer settings on an already-existing
    /// collection in place and wait for it to complete optimization. Does not touch data.
    Reindex(RunArgs),
    /// Delete the collection if it exists.
    Delete(RunArgs),
    /// Inspect the config and the file list without connecting or loading.
    Inspect(LoadArgs),
    /// Print this backend's capabilities as stable JSON (the machine-readable
    /// contract descriptor `nova-contract` validates against
    /// `contracts/nova-load/v1.yaml`). Does not connect or load.
    Capabilities(CapabilitiesArgs),
}

/// Args for `capabilities`. `--json` is accepted for forward-compat and CLI
/// symmetry; output is always JSON regardless.
#[derive(Debug, Args)]
struct CapabilitiesArgs {
    /// Emit JSON (the default and only format today).
    #[arg(long)]
    json: bool,
}

/// The stable capabilities descriptor for this backend. Kept in lockstep with
/// `contracts/nova-load/v1.yaml`: the `commands`, `methods`, `vector_kinds`,
/// and `point_id_types` here are exactly what that contract declares required,
/// and `nova-contract check` fails if this drifts from the contract file.
///
/// - `commands` mirror the clap subcommands below.
/// - `methods` mirror the `VectorStore` trait in `stores/mod.rs`.
/// - `vector_kinds` mirror `VectorValue` (`stores/mod.rs`).
/// - `point_id_types` mirror `PointId` (`stores/mod.rs`).
fn capabilities_json() -> serde_json::Value {
    serde_json::json!({
        "contract": "nova-load-backend/v1",
        "backend": "qdrant",
        "commands": [
            "capabilities", "run", "prepare", "load",
            "finalize", "inspect", "reindex", "delete"
        ],
        "methods": [
            "ensure_collection", "upsert_batch", "close", "defer_indexing",
            "enable_indexing", "wait_for_indexing", "reindex", "delete_collection"
        ],
        "vector_kinds": ["dense", "sparse", "multivector"],
        "point_id_types": ["integer", "string"],
        "flags": {
            "load": ["--num-jobs", "--job-rank"],
            "inspect": ["--num-jobs", "--job-rank"]
        }
    })
}

/// Args for phases that act on the whole dataset (no partitioning).
#[derive(Debug, Args)]
struct RunArgs {
    /// Path to the loader config YAML.
    config: PathBuf,
}

/// Args for phases that operate on a single worker's slice.
#[derive(Debug, Args)]
struct LoadArgs {
    /// Path to the loader config YAML.
    config: PathBuf,
    /// Total number of parallel loader jobs (for distributed runs).
    #[arg(long, default_value_t = 1)]
    num_jobs: usize,
    /// This job's index, in `[0, num_jobs)`. Each job loads its own slice.
    #[arg(long, default_value_t = 0)]
    job_rank: usize,
}

impl LoadArgs {
    fn partition(&self) -> Result<Partition, String> {
        Partition { rank: self.job_rank, num_jobs: self.num_jobs }.validate()
    }
}

fn load_config(path: &PathBuf) -> Result<LoadConfig, ExitCode> {
    LoadConfig::from_path(path).map_err(|err| {
        eprintln!("error: failed to load config `{}`: {err}", path.display());
        ExitCode::FAILURE
    })
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nova_load=info".into()),
        )
        .init();

    match run(Cli::parse().command).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(code) => code,
    }
}

/// Dispatch a subcommand. Returns `Err(ExitCode)` for any failure so `main`
/// stays a thin shell.
async fn run(command: Command) -> Result<(), ExitCode> {
    // `capabilities` is a pure descriptor: no config, no connection, always JSON.
    if let Command::Capabilities(_) = command {
        println!(
            "{}",
            serde_json::to_string_pretty(&capabilities_json())
                .expect("capabilities descriptor is always serializable")
        );
        return Ok(());
    }

    let result = match command {
        Command::Run(a) => nova_load::run(load_config(&a.config)?).await,
        Command::Prepare(a) => nova_load::prepare(load_config(&a.config)?).await,
        Command::Finalize(a) => nova_load::finalize(load_config(&a.config)?).await,
        Command::Reindex(a) => nova_load::reindex(load_config(&a.config)?).await,
        Command::Delete(a) => nova_load::delete(load_config(&a.config)?).await,
        Command::Load(a) => {
            let partition = a.partition().map_err(|e| {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            })?;
            nova_load::load(load_config(&a.config)?, partition).await
        }
        Command::Inspect(a) => {
            let partition = a.partition().map_err(|e| {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            })?;
            nova_load::inspect(load_config(&a.config)?, partition).await
        }
        // Handled above with an early return before this match.
        Command::Capabilities(_) => unreachable!("capabilities handled before dispatch"),
    };

    result.map_err(|err| {
        eprintln!("error: {err}");
        ExitCode::FAILURE
    })
}
