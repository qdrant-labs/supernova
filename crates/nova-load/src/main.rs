use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use nova_load::config::LoadConfig;
use nova_load::plan::Partition;

/// Load vectors from a datasource into a vector store, per a YAML config.
#[derive(Debug, Parser)]
#[command(name = "nova-load", version, about)]
struct Cli {
    /// Path to the loader config YAML.
    config: PathBuf,
    /// Total number of parallel loader jobs (for distributed runs).
    #[arg(long, default_value_t = 1)]
    num_jobs: usize,
    /// This job's index, in `[0, num_jobs)`. Each job loads its own slice.
    #[arg(long, default_value_t = 0)]
    job_rank: usize,
    /// Inspect the config and file list without connecting or loading anything.
    #[arg(long)]
    dry_run: bool,
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nova_load=info".into()),
        )
        .init();

    let cli = Cli::parse();

    let partition = match (Partition { rank: cli.job_rank, num_jobs: cli.num_jobs }).validate() {
        Ok(p) => p,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    tracing::info!(job_rank = partition.rank, num_jobs = partition.num_jobs, "starting nova-load");

    let config = match LoadConfig::from_path(&cli.config) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("error: failed to load config `{}`: {err}", cli.config.display());
            return ExitCode::FAILURE;
        }
    };

    let result = if cli.dry_run {
        nova_load::dry_run(config, partition).await
    } else {
        nova_load::run_loader(config, partition).await
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}