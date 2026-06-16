use std::io::IsTerminal;
use std::time::Duration;

use clap::Parser;
use indicatif::ProgressBar;

use nova_storm::config::{LoadProfile, StormConfig, load_config_file};
use nova_storm::errors::StormError;
use nova_storm::queries::load_query_vectors;
use nova_storm::runner::run_storm;

/// Load-test a vector store from this machine (the `nova storm` workload).
#[derive(Parser)]
#[command(name = "nova-storm", about, version)]
struct Cli {
    /// Path to the storm config YAML.
    config: String,
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

async fn run() -> Result<(), StormError> {
    let cli = Cli::parse();

    let StormConfig { target, query, load, .. } = load_config_file(&cli.config)?;

    // CLI flags win over the YAML's load block.
    let profile = LoadProfile {
        concurrency: load.concurrency,
        duration_s: load.duration_s,
        ramp_s: load.ramp_s,
        target_qps: load.target_qps,
    };

    let vectors = load_query_vectors(&query.source)?;
    if vectors.is_empty() {
        return Err(StormError::Other(format!(
            "no query vectors loaded from {:?} (column {:?})",
            query.source.uri, query.source.column
        )));
    }

    let target = target.into_target(&query)?;

    let mode = if profile.target_qps > 0.0 {
        format!(
            "paced {:.0} qps/worker (cap {} in-flight)",
            profile.target_qps, profile.concurrency
        )
    } else {
        format!("closed-loop concurrency={}", profile.concurrency)
    };
    tracing::info!(
        target = %target,
        query_vectors = vectors.len(),
        duration_s = profile.duration_s,
        "storm: {mode}"
    );

    // A spinner so a long run doesn't look frozen; hidden when not a TTY.
    let spinner = if std::io::stderr().is_terminal() {
        let pb = ProgressBar::new_spinner();
        pb.enable_steady_tick(Duration::from_millis(120));
        pb.set_message(format!("storm running for {:.0}s…", profile.duration_s));
        pb
    } else {
        ProgressBar::hidden()
    };

    let results = run_storm(target, vectors, &profile).await;
    spinner.finish_and_clear();

    let summary = results.summary();
    println!("{}", "=".repeat(50));
    println!("{summary}");
    println!("{}", "=".repeat(50));
    Ok(())
}
