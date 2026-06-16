use std::io::IsTerminal;
use std::time::Duration;

use clap::Parser;
use indicatif::ProgressBar;

use nova_metrics::{RunContext, build_sink, resolve_run_id};
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

    let cfg = load_config_file(&cli.config)?;
    // Snapshot the resolved config for the `runs` row before consuming `cfg`.
    // The sink redacts secrets (api_key, dsn, ...) before persisting it.
    let config_json = serde_json::to_value(&cfg).unwrap_or(serde_json::Value::Null);
    let StormConfig {
        target,
        query,
        load,
        metrics,
        ..
    } = cfg;

    let profile: LoadProfile = load;

    let vectors = load_query_vectors(&query.source)?;
    if vectors.is_empty() {
        return Err(StormError::Other(format!(
            "no query vectors loaded from {:?} (column {:?})",
            query.source.uri, query.source.column
        )));
    }

    let target = target.into_target(&query)?;

    // Metrics: build the sink (a bad DSN fails fast here), then open the run.
    // run_id comes from $NOVA_RUN_ID when a controller minted one (so a fleet
    // reports under one run); node_id is the worker's rank.
    let sink = build_sink(metrics.as_ref())?;
    let run_id = resolve_run_id("storm");
    let node_id = std::env::var("SKYPILOT_JOB_RANK").unwrap_or_else(|_| "local".into());
    let experiment_id = std::env::var("NOVA_EXPERIMENT_ID").ok();
    sink.start(
        &run_id,
        &RunContext {
            command: "storm",
            node_id: Some(&node_id),
            experiment_id: experiment_id.as_deref(),
            config: &config_json,
        },
    );

    let mode = if profile.target_qps > 0.0 {
        format!(
            "paced {:.0} qps/worker (cap {} in-flight)",
            profile.target_qps, profile.concurrency
        )
    } else {
        format!("closed-loop concurrency={}", profile.concurrency)
    };
    tracing::info!(
        run_id = %run_id,
        node_id = %node_id,
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

    let results = run_storm(target, vectors, &profile, sink.clone()).await;
    spinner.finish_and_clear();

    let summary = results.summary();
    sink.summary(&serde_json::to_value(&summary).unwrap_or(serde_json::Value::Null));
    sink.finish("ok");

    println!("{}", "=".repeat(50));
    println!("{summary}");
    println!("{}", "=".repeat(50));
    Ok(())
}
