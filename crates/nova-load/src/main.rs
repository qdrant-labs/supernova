use clap::Parser;

use nova_metrics::{RunContext, build_sink, resolve_run_id};
use nova_load::config::{LoadConfig, load_config_file_with_json};
use nova_load::errors::LoadError;
use nova_load::runner::{finalize, run_loader, setup_collection};

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

    /// Control plane only: create/verify the collection + defer indexing, then
    /// exit. The distributed controller runs this once before launching workers.
    #[arg(long, conflicts_with_all = ["finalize", "num_jobs"])]
    setup_only: bool,

    /// Control plane only: re-enable indexing + wait for the build, then exit.
    /// The distributed controller runs this after all workers complete.
    #[arg(long, conflicts_with_all = ["setup_only", "num_jobs"])]
    finalize: bool,
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

    let (cfg, config_json) = load_config_file_with_json(&cli.config)?;
    let LoadConfig {
        vectors,
        mut datasource,
        vectorstore,
        loader,
        metrics,
    } = cfg;

    // Control-plane-only modes for the distributed controller. They bracket the
    // workers' data load (which runs --no-manage-indexing), so neither opens a
    // metrics run — they're brief, single-shot Qdrant calls.
    if cli.finalize {
        finalize(vectorstore.into_store()?).await?;
        tracing::info!("indexing finalized");
        return Ok(());
    }
    if cli.setup_only {
        let reader = datasource.into_reader(&vectors, loader.batch_size.max(1))?;
        setup_collection(reader, vectorstore.into_store()?, &vectors).await?;
        tracing::info!("collection ready (indexing deferred)");
        return Ok(());
    }

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

    // Metrics: build the sink (a bad DSN fails fast here), then open the run.
    // A distributed fleet shares one run via $NOVA_RUN_ID; node_id is the rank.
    let sink = build_sink(metrics.as_ref())?;
    let run_id = resolve_run_id("load");
    let node_id = cli
        .job_rank
        .map(|r| r.to_string())
        .or_else(|| std::env::var("SKYPILOT_JOB_RANK").ok())
        .unwrap_or_else(|| "local".into());
    let experiment_id = std::env::var("NOVA_EXPERIMENT_ID").ok();
    sink.start(
        &run_id,
        &RunContext {
            command: "load",
            node_id: Some(&node_id),
            experiment_id: experiment_id.as_deref(),
            config: &config_json,
        },
    );

    let result = run_loader(
        reader,
        store,
        &vectors,
        &loader,
        !cli.no_manage_indexing,
        sink.clone(),
    )
    .await;

    match &result {
        Ok(stats) => {
            tracing::info!(
                loaded = stats.loaded,
                total = stats.total,
                errors = stats.errors,
                elapsed_secs = stats.elapsed_secs,
                "load complete"
            );
            let wps_avg = if stats.elapsed_secs > 0.0 {
                stats.loaded as f64 / stats.elapsed_secs
            } else {
                0.0
            };
            sink.summary(&serde_json::json!({
                "total": stats.total,
                "loaded": stats.loaded,
                "errors": stats.errors,
                "elapsed_secs": stats.elapsed_secs,
                "wps_avg": wps_avg,
            }));
            sink.finish("ok");
        }
        Err(_) => sink.finish("error"),
    }

    result.map(|_| ())
}

/// Distributed workers get their rank from SkyPilot's env var by default.
fn default_job_rank() -> usize {
    std::env::var("SKYPILOT_JOB_RANK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}
