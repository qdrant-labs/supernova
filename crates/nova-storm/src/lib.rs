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
pub mod datetime;
pub mod errors;
pub mod filter;
pub mod queries;
pub mod report;
pub mod runner;
pub mod targets;

use std::io::IsTerminal;
use std::time::Duration;

use indicatif::ProgressBar;

use config::StormConfig;
use errors::StormError;
use queries::load_query_vectors;
use runner::Summary;

/// Whether the run explicitly disables quantization rescoring. Navigated
/// defensively through the free-form `search_params` value: anything missing
/// or the wrong shape means "not disabled", the same as the server default.
fn rescore_disabled(search_params: &Option<serde_yaml::Value>) -> bool {
    let Some(sp) = search_params.as_ref() else {
        return false;
    };
    // Two sibling params make quantization irrelevant to the returned score,
    // so `rescore: false` alongside either is not a reason to distrust it:
    //   * `exact: true` scans the original vectors, skipping the index;
    //   * `quantization.ignore: true` skips quantized storage outright.
    let bypassed = |path: &[&str]| {
        path.iter()
            .try_fold(sp, |node, key| node.get(*key))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    };
    if bypassed(&["exact"]) || bypassed(&["quantization", "ignore"]) {
        return false;
    }
    sp.get("quantization")
        .and_then(|q| q.get("rescore"))
        .and_then(|r| r.as_bool())
        .is_some_and(|rescore| !rescore)
}

/// Run a storm end to end: load the query set, connect the target, drive the
/// load profile, and return this worker's latency summary.
pub async fn run(config: StormConfig) -> Result<Summary, StormError> {
    let StormConfig {
        target,
        query,
        load,
        report,
    } = config;

    // Open the time-series sink FIRST: begin() creating the file (or a db
    // sink its tables) is exactly the step that fails on a bad path, and it
    // must die here — before query loading and before any load is offered.
    let recorder = match &report {
        Some(cfg) => {
            let mut r = cfg.build();
            r.begin().map_err(|e| {
                StormError::Other(format!("report sink `{}` failed to open: {e}", cfg.path))
            })?;
            tracing::info!("time-series report: {:?} -> {}", cfg.format, cfg.path);
            Some(r)
        }
        None => None,
    };

    tracing::info!(
        "queries: {:?} via vector `{}`",
        query.vector_type,
        query.vector_name.as_deref().unwrap_or("(unnamed)"),
    );
   
    let target = target.into_target(&query).await?;
    let scoring = target.scoring_profile().await;
    // A failed probe downgrades the tolerance to the conservative default and
    // disables tie reporting; an operator seeing those needs to know it was
    // a probe failure rather than a property of their collection.
    if scoring.distance.is_none() && query.source.ground_truth_score_column.is_some() {
        tracing::warn!(
            "could not read the collection's scoring config (distance/datatype/quantization) \
             — score-based tie reporting is disabled and the conservative tolerance applies"
        );
    }
    let probed_datatype = scoring.datatype.clone();
    let tie_epsilon = query
        .tie_epsilon
        .unwrap_or_else(|| config::tie_epsilon_for_datatype(probed_datatype.as_deref()));
    let tie_epsilon_source = match (&query.tie_epsilon, &probed_datatype) {
        (Some(_), _) => "configured".to_string(),
        (None, Some(dt)) => format!("auto, {dt}"),
        (None, None) => "auto, datatype unknown".to_string(),
    };
    if query.source.ground_truth_score_column.is_some() {
        match (&query.tie_epsilon, &probed_datatype) {
            (Some(e), _) => tracing::info!("score tie tolerance: {e:.1e} (configured)"),
            (None, Some(dt)) => {
                tracing::info!("score tie tolerance: {tie_epsilon:.1e} (auto, datatype={dt})")
            }
            (None, None) => tracing::info!(
                "score tie tolerance: {tie_epsilon:.1e} (auto, datatype unknown — conservative)"
            ),
        }
    }
    // If a query and corpus set live in different precision spaces and
    // rescoring is not used, disable tie-aware reporting.
    let tie_disabled_reason: Option<String> = if query.source.ground_truth_score_column.is_none()
        || query.source.ground_truth_column.is_none()
    {
        None
    } else if rescore_disabled(&query.search_params) && scoring.quantized {
        Some(
            "the collection is quantized and this run sets quantization.rescore=false, so \
             returned scores are in quantized space"
                .to_string(),
        )
    } else if scoring.distance.is_none() {
        // Orientation decides the SIGN of every comparison. Guessing it wrong
        // makes `missing_from_gt` fire on essentially every miss, so an
        // unknown distance disables score-based reporting instead of assuming
        // larger-is-better — the failure mode that assumption produces is a
        // confident false alarm, not a missing number.
        Some(
            "the collection's distance function could not be determined, so the score \
             orientation is unknown"
                .to_string(),
        )
    } else {
        None
    };
    // Determine the orientation of the engine's scores based on the distance function.
    let engine_higher_is_better =
        !matches!(scoring.distance.as_deref(), Some("euclid" | "manhattan"));
    if tie_disabled_reason.is_some() {
        // Collecting per-point scores costs an allocation per query inside the
        // measured latency window; nothing will read them now.
        target.disable_score_collection();
    }
    if query.source.ground_truth_score_column.is_some() {
        if let Some(reason) = &tie_disabled_reason {
            tracing::warn!(
                "tie reporting disabled for this run: {reason}. Exact recall is unaffected."
            );
        }
        if probed_datatype.as_deref() == Some("uint8") {
            tracing::warn!(
                "collection datatype is uint8: vector components are stored as integers, so \
                 scores only match a ground truth built from the SAME quantized values — \
                 otherwise the tie-tolerant bound and `missing_from_gt` are unreliable"
            );
        }
    }
    // A score column with no id column loads nothing and reports nothing —
    // silently, which reads as "ties just didn't happen" rather than a config
    // mistake.
    if query.source.ground_truth_score_column.is_some()
        && query.source.ground_truth_column.is_none()
    {
        tracing::warn!(
            "query.source.ground_truth_score_column is set without ground_truth_column — \
             scores are only meaningful alongside the ids they belong to, so no recall or \
             tie reporting will be produced"
        );
    }
    // The target is already connected by this point, so close it on the way
    // out of the failure paths below rather than dropping a live connection.
    let vectors = match load_query_vectors(
        &query.source,
        query.vector_type,
        query.filter.as_ref(),
        query.top_k,
    ) {
        Ok(v) => v,
        Err(e) => {
            let _ = target.close().await;
            return Err(e.into());
        }
    };
    if vectors.is_empty() {
        let _ = target.close().await;
        return Err(StormError::Other(format!(
            "no query vectors loaded from {:?} (column {:?}) — NULL and empty vectors are \
             excluded; a file of only those loads nothing",
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

    let mode = if load.target_rps > 0.0 {
        format!(
            "paced {:.0} rps/worker (batch_size={}, cap {} in-flight)",
            load.target_rps, load.batch_size, load.concurrency
        )
    } else {
        format!(
            "closed-loop concurrency={} batch_size={}",
            load.concurrency, load.batch_size
        )
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
        // Ground-truth lists shorter than top_k are scored against their own
        // length (not top_k) and land in the summary's separate `recall_short`
        // bucket — so they no longer read as an artificial regression, but the
        // split is worth flagging up front so the operator expects two buckets.
        // Positional depth, matching how `recall_at_k` buckets: a duplicate id
        // shrinks the deduplicated set without making the ground truth
        // shallower, and this preflight must not promise a bucket the run will
        // not produce.
        let short = vectors
            .iter()
            .filter(|v| {
                // Non-empty only: an empty ground truth produces no recall
                // sample at all and is tallied as `recall_empty_gt`, so
                // promising it in the short bucket would be a lie.
                v.ground_truth.as_ref().is_some_and(|g| !g.is_empty())
                    && (v.gt_depth as u64) < query.top_k
            })
            .count();
        if short > 0 {
            tracing::info!(
                "{short}/{with_ground_truth} ground-truth lists hold fewer than top_k={k} ids \
                 — those queries are scored against their own ground-truth length and reported \
                 separately as `recall@{k}_short`",
                k = query.top_k,
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

    let results = runner::run_storm(
        target,
        vectors,
        &load,
        query.top_k,
        runner::ScoreComparison {
            epsilon: tie_epsilon,
            epsilon_source: tie_epsilon_source,
            disabled_reason: tie_disabled_reason,
            // A score column with no id column compares nothing.
            configured: query.source.ground_truth_score_column.is_some()
                && query.source.ground_truth_column.is_some(),
            engine_higher_is_better,
        },
        query.filter.is_some(),
        recorder,
    )
    .await;
    spinner.finish_and_clear();

    Ok(results.summary())
}

#[cfg(test)]
mod tests {
    use super::rescore_disabled;

    fn yaml(src: &str) -> Option<serde_yaml::Value> {
        Some(serde_yaml::from_str(src).unwrap())
    }

    #[test]
    fn quantization_bypasses_mean_rescore_false_is_not_a_concern() {
        // `exact: true` and `quantization.ignore: true` both skip quantized
        // storage, so the returned score is exact and `rescore: false` says
        // nothing about it. Treating those as "scores are in quantized space"
        // would withhold tie reporting from a run that never used it.
        assert!(!rescore_disabled(&yaml(
            "exact: true\nquantization:\n  rescore: false"
        )));
        assert!(!rescore_disabled(&yaml(
            "quantization:\n  ignore: true\n  rescore: false"
        )));
        // …but the plain combination still is.
        assert!(rescore_disabled(&yaml(
            "exact: false\nquantization:\n  rescore: false"
        )));
    }

    #[test]
    fn rescore_disabled_only_when_explicitly_false() {
        assert!(rescore_disabled(&yaml("quantization:\n  rescore: false")));
        assert!(!rescore_disabled(&yaml("quantization:\n  rescore: true")));
        // Absent, wrong shape, or no search_params at all -> server default.
        assert!(!rescore_disabled(&yaml(
            "quantization:\n  oversampling: 2.0"
        )));
        assert!(!rescore_disabled(&yaml("hnsw_ef: 128")));
        assert!(!rescore_disabled(&yaml("quantization: notamap")));
        assert!(!rescore_disabled(&None));
    }
}
