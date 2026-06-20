//! `nova storm` — sustained query load testing for vector stores.
//!
//! A *storm* fires nearest-neighbour queries at a target cluster and records
//! the latency distribution. Work is **replicated, not partitioned**: every
//! worker runs the same [`LoadProfile`](config::LoadProfile), so total offered
//! load is roughly `num_workers × {concurrency or qps}`.
//!
//! The split mirrors `nova-load`: a backend ([`QueryTarget`](targets::QueryTarget))
//! is a thin "fire one query, report its latency" adapter, and the runner owns
//! the load *shape* (how many in flight, for how long, at what rate). The same
//! [`run_storm`](runner::run_storm) drives any backend.

pub mod config;
pub mod errors;
pub mod queries;
pub mod runner;
pub mod targets;

use std::io::IsTerminal;
use std::time::Duration;

use indicatif::ProgressBar;

use config::StormConfig;
use errors::StormError;
use queries::load_query_vectors;
use runner::Summary;

/// Run a storm end to end: load the query set, connect the target, drive the
/// load profile, and return this worker's latency summary.
pub async fn run(config: StormConfig) -> Result<Summary, StormError> {
    let StormConfig { target, query, load } = config;

    let vectors = load_query_vectors(&query.source)?;
    if vectors.is_empty() {
        return Err(StormError::Other(format!(
            "no query vectors loaded from {:?} (column {:?})",
            query.source.uri, query.source.column
        )));
    }

    let target = target.into_target(&query)?;

    let mode = if load.target_qps > 0.0 {
        format!("paced {:.0} qps/worker (cap {} in-flight)", load.target_qps, load.concurrency)
    } else {
        format!("closed-loop concurrency={}", load.concurrency)
    };
    tracing::info!(
        target = %target,
        query_vectors = vectors.len(),
        duration_s = load.duration_s,
        "storm: {mode}"
    );

    // A spinner so a long run doesn't look frozen; hidden when not a TTY.
    let spinner = if std::io::stderr().is_terminal() {
        let pb = ProgressBar::new_spinner();
        pb.enable_steady_tick(Duration::from_millis(120));
        pb.set_message(format!("storm running for {:.0}s…", load.duration_s));
        pb
    } else {
        ProgressBar::hidden()
    };

    let results = runner::run_storm(target, vectors, &load).await;
    spinner.finish_and_clear();

    Ok(results.summary())
}
