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
    /// Resume from a previously persisted checkpoint (if found).
    #[arg(long, default_value_t = false)]
    resume: bool,
    /// Optional checkpoint file path override.
    #[arg(long)]
    checkpoint_path: Option<PathBuf>,
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
            nova_load::load(
                load_config(&a.config)?,
                partition,
                nova_load::LoadRuntimeOptions {
                    resume: a.resume,
                    checkpoint_path: a.checkpoint_path.clone(),
                },
            )
            .await
        }
        Command::Inspect(a) => {
            let partition = a.partition().map_err(|e| {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            })?;
            nova_load::inspect(load_config(&a.config)?, partition).await
        }
    };

    result.map_err(|err| {
        eprintln!("error: {err}");
        ExitCode::FAILURE
    })
}
