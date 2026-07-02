//! `nova storm` — sustained query load testing for vector stores.
//!
//! A *storm* fires nearest-neighbour queries at a target cluster and records
//! the latency distribution. Work is **replicated, not partitioned**: every
//! worker runs the same [`LoadProfile`](config::LoadProfile), so total offered
//! load is roughly `num_workers × {concurrency or rps} × batch_size`.
//!
//! The split mirrors `nova-load`: a backend ([`QueryTarget`](targets::QueryTarget))
//! is a thin "fire one batch dispatch, report its latency" adapter, and the
//! runner owns the load *shape* (how many in flight, for how long, at what
//! rate, how many queries per dispatch). The same [`run_storm`](runner::run_storm)
//! drives any backend.

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
    let with_ground_truth = vectors.iter().filter(|v| v.ground_truth.is_some()).count();

    // `batch_size` is only checked for `> 0` at config-load time (it has no
    // visibility into how many rows `query.source` will actually yield) — a
    // `batch_size` bigger than the loaded set is harmless (each repeated
    // position still scores independently and correctly, see `batch_indices`)
    // but silently sends duplicate vectors within a single dispatch and does
    // more work than the operator likely intended. Warn up front, the same
    // way a too-short ground-truth list is warned about below.
    if load.batch_size > vectors.len() {
        tracing::warn!(
            "load.batch_size={} exceeds the {} loaded query vectors — each batch dispatch will \
             contain duplicate vectors (harmless, but likely not intended; raise query.source.limit \
             or lower batch_size)",
            load.batch_size,
            vectors.len(),
        );
    }

    let target = target.into_target(&query)?;

    let mode = if load.target_rps > 0.0 {
        format!(
            "paced {:.0} rps/worker (batch_size={}, cap {} in-flight)",
            load.target_rps, load.batch_size, load.concurrency
        )
    } else {
        format!("closed-loop concurrency={} batch_size={}", load.concurrency, load.batch_size)
    };
    tracing::info!(
        target = %target,
        query_vectors = vectors.len(),
        duration_s = load.duration_s,
        batch_size = load.batch_size,
        "storm: {mode}"
    );
    if query.source.ground_truth_column.is_some() {
        // Surfaces a wrong column name / all-null column immediately, rather
        // than silently as a missing `mean_recall` at the end of the run.
        tracing::info!(
            "recall tracking: {with_ground_truth}/{} queries have ground truth (column {:?})",
            vectors.len(),
            query.source.ground_truth_column,
        );
        // recall_at_k divides by top_k, so a ground-truth list shorter than
        // top_k silently caps recall (e.g. bf's k=10 vs storm's top_k=100 caps
        // every recall sample at 0.10) with no other signal — warn up front
        // rather than let it read as a real search-quality regression.
        let short = vectors
            .iter()
            .filter_map(|v| v.ground_truth.as_ref())
            .filter(|gt| (gt.len() as u64) < query.top_k)
            .count();
        if short > 0 {
            tracing::warn!(
                "{short}/{with_ground_truth} ground-truth lists hold fewer than top_k={} ids — \
                 recall will read artificially low for those queries (the ground truth's own k \
                 must be at least storm's top_k)",
                query.top_k,
            );
        }
    }

    // A spinner so a long run doesn't look frozen; hidden when not a TTY.
    let spinner = if std::io::stderr().is_terminal() {
        let pb = ProgressBar::new_spinner();
        pb.enable_steady_tick(Duration::from_millis(120));
        pb.set_message(format!("storm running for {:.0}s…", load.duration_s));
        pb
    } else {
        ProgressBar::hidden()
    };

    let results = runner::run_storm(target, vectors, &load, query.top_k).await;
    spinner.finish_and_clear();

    Ok(results.summary())
}
