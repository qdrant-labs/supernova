use clap::Parser;

use nova_load::config::{LoadConfig, load_config_file};
use nova_load::errors::LoadError;
use nova_load::runner::run_loader;

/// Load pre-embedded data into a vector store.
#[derive(Parser)]
#[command(name = "nova-load", about, version)]
struct Cli {
    /// Path to the load config YAML.
    config: String,

    /// Skip collection creation + indexing lifecycle (for distributed workers
    /// where the master handles it).
    #[arg(long)]
    no_manage_indexing: bool,

    /// Total number of parallel jobs; auto-shards the corpus files by rank.
    #[arg(long)]
    num_jobs: Option<usize>,

    /// This job's rank (0-indexed). Defaults to $SKYPILOT_JOB_RANK, else 0.
    #[arg(long)]
    job_rank: Option<usize>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    if let Err(e) = run().await {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), LoadError> {
    let cli = Cli::parse();

    let LoadConfig {
        vectors,
        mut datasource,
        vectorstore,
        loader,
    } = load_config_file(&cli.config)?;

    if let Some(num_jobs) = cli.num_jobs {
        let job_rank = cli.job_rank.unwrap_or_else(default_job_rank);
        let assigned = datasource.shard(num_jobs, job_rank)?;
        tracing::info!(num_jobs, job_rank, files = assigned, "sharded corpus");
        if assigned == 0 {
            tracing::info!("no files assigned to this shard; nothing to do");
            return Ok(());
        }
    }

    // The reader buffers `prefetch_size` points per chunk; the runner re-slices
    // those into `batch_size` upserts. Default prefetch to 10x the batch.
    let chunk_size = loader
        .prefetch_size
        .unwrap_or(loader.batch_size.saturating_mul(10));

    let reader = datasource.into_reader(&vectors, chunk_size)?;
    let store = vectorstore.into_store()?;

    let stats = run_loader(reader, store, &vectors, &loader, !cli.no_manage_indexing).await?;

    tracing::info!(
        loaded = stats.loaded,
        total = stats.total,
        errors = stats.errors,
        elapsed_secs = stats.elapsed_secs,
        "load complete"
    );
    Ok(())
}

/// Distributed workers get their rank from SkyPilot's env var by default.
fn default_job_rank() -> usize {
    std::env::var("SKYPILOT_JOB_RANK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}
