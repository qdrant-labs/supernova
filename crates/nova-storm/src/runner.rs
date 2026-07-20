//! The load generator + single-worker result summary.
//!
//! Two modes, chosen by [`LoadProfile::target_rps`](crate::config::LoadProfile):
//!
//! * `0` — **closed-loop**: hold `concurrency` requests in flight for the whole
//!   window, each task firing the next batch dispatch the instant its previous
//!   one returns. Measures the max throughput the cluster gives at that depth.
//! * `>0` — **open-loop paced**: launch one batch dispatch every `1/target_rps`
//!   seconds on a fixed virtual schedule, with `concurrency` as an in-flight
//!   ceiling. The fixed schedule is what avoids coordinated omission — a slow
//!   response can't delay the next launch and hide latency — and keeps the
//!   offered rate tracking the target without overshooting it.
//!
//! A batch dispatch (one `query_batch` round-trip) is the atomic unit of load
//! regardless of [`LoadProfile::batch_size`](crate::config::LoadProfile) — a
//! batch of 1 is not a special case, it's just the default. Latency is one
//! sample per dispatch; recall stays per-query within it (see [`BatchOutcome`](
//! crate::targets::BatchOutcome)).
//!
//! Aggregating ACROSS workers is a separate step and must merge latency
//! *distributions*, never average per-worker percentiles.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::{Semaphore, mpsc};
use tokio::task::JoinSet;
use tokio::time::{Duration, Instant, sleep_until};

use crate::config::LoadProfile;
use crate::queries::QueryVector;
use crate::targets::QueryTarget;

/// One worker's raw measurements. Latencies (and recalls, if ground truth is
/// configured) are kept as full samples — not pre-aggregated — so a fleet merge
/// can recompute true percentiles/means instead of averaging per-worker stats.
#[derive(Debug, Clone)]
pub struct StormResults {
    /// One entry per batch dispatch (one `query_batch` round-trip), not per
    /// query — a single gRPC round-trip's timing can't be honestly
    /// disaggregated into per-query numbers.
    pub latencies_ms: Vec<f64>,
    /// One entry per query that had ground truth (see [`QueryVector::ground_truth`]).
    /// Recall stays per-query even though latency doesn't: `QueryBatchResponse`
    /// gives one distinct result per submitted query, so each query's recall is
    /// still individually real, not approximated from the batch.
    pub recalls: Vec<f64>,
    /// Count of batch dispatches, not individual queries.
    pub n_ok: u64,
    pub n_err: u64,
    pub wall_s: f64,
    /// How many query vectors went in each dispatch — carried alongside the
    /// raw samples so `summary()` can self-describe regardless of what
    /// `LoadProfile` is in scope.
    pub batch_size: usize,
}

/// Aggregated stats for THIS worker. Fleet-wide stats must merge raw samples
/// from every worker, not average these.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Summary {
    /// Batch dispatches (round-trips), not individual queries.
    pub requests: u64,
    pub errors: u64,
    pub batch_size: usize,
    /// Batch dispatch rate — round-trips/sec, not query throughput.
    pub requests_per_sec: f64,
    /// Actual query throughput: `requests_per_sec * batch_size`.
    pub qps: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    /// `None` when no query in this run had ground truth (feature unused, or
    /// misconfigured — e.g. wrong column name — which looks the same from here).
    /// `mean_recall` alone can hide a bimodal distribution (some queries near-
    /// perfect, some near-zero) the same way it would for latency — hence
    /// `median_recall`/`min_recall` alongside it, from the same raw samples.
    pub mean_recall: Option<f64>,
    pub median_recall: Option<f64>,
    /// The single worst query's recall — the "tail" that matters for recall,
    /// which is the LOW end (unlike latency's p95/p99, which watch the high
    /// end) — so this is a min, not a high percentile.
    pub min_recall: Option<f64>,
}

impl StormResults {
    pub fn summary(&self) -> Summary {
        let mut ms = self.latencies_ms.clone();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut rc = self.recalls.clone();
        rc.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let total = self.n_ok + self.n_err;
        let requests_per_sec = if self.wall_s > 0.0 { total as f64 / self.wall_s } else { 0.0 };
        Summary {
            requests: total,
            errors: self.n_err,
            batch_size: self.batch_size,
            requests_per_sec,
            qps: requests_per_sec * self.batch_size as f64,
            p50_ms: percentile(&ms, 50.0),
            p95_ms: percentile(&ms, 95.0),
            p99_ms: percentile(&ms, 99.0),
            max_ms: ms.last().copied().unwrap_or(0.0),
            mean_recall: (!rc.is_empty()).then(|| rc.iter().sum::<f64>() / rc.len() as f64),
            median_recall: (!rc.is_empty()).then(|| percentile(&rc, 50.0)),
            min_recall: rc.first().copied(),
        }
    }
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut lines = vec![
            format!("{:>16}: {}", "requests", self.requests),
            format!("{:>16}: {}", "errors", self.errors),
            format!("{:>16}: {}", "batch_size", self.batch_size),
            format!("{:>16}: {:.1}", "requests_per_sec", self.requests_per_sec),
            format!("{:>16}: {:.1}", "qps", self.qps),
            format!("{:>16}: {:.2}", "p50_ms", self.p50_ms),
            format!("{:>16}: {:.2}", "p95_ms", self.p95_ms),
            format!("{:>16}: {:.2}", "p99_ms", self.p99_ms),
            format!("{:>16}: {:.2}", "max_ms", self.max_ms),
        ];
        if let Some(r) = self.mean_recall {
            lines.push(format!("{:>16}: {:.4}", "mean_recall", r));
        }
        if let Some(r) = self.median_recall {
            lines.push(format!("{:>16}: {:.4}", "median_recall", r));
        }
        if let Some(r) = self.min_recall {
            lines.push(format!("{:>16}: {:.4}", "min_recall", r));
        }
        write!(f, "{}", lines.join("\n"))
    }
}

/// Nearest-rank percentile (NIST) over a pre-sorted slice (ms). Empty → 0.
fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let n = sorted_ms.len();
    // 1-based rank = ceil(p/100 * n), clamped into the slice.
    let rank = ((p / 100.0) * n as f64).ceil().max(1.0) as usize;
    sorted_ms[rank.min(n) - 1]
}

/// Recall@k for one query: the fraction of the known-correct top-k ids
/// (`ground_truth`) that appear among the ids the target actually `returned`.
/// Divides by `k` (not `ground_truth.len()` or `returned.len()`) — the
/// conventional recall@k definition — so `ground_truth` should hold at least
/// `k` ids (nova-bf's own `k` at or above storm's `top_k`) or recall reads
/// artificially low; `lib.rs` warns at startup if any loaded row is short.
/// `ground_truth` is already a `HashSet` (built once at load time in
/// `queries.rs`, not per call) since this runs on every query firing.
/// `returned` is deduped before counting hits — a target that ever repeated an
/// id within one query's results must not let that repeat count twice, which
/// would push recall above the `1.0` ceiling a fraction is supposed to have.
fn recall_at_k(returned: &[String], ground_truth: &HashSet<String>, k: u64) -> f64 {
    let hits = returned
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>()
        .into_iter()
        .filter(|id| ground_truth.contains(*id))
        .count();
    hits as f64 / k as f64
}

/// One batch dispatch's observation forwarded from a worker to the collector
/// (and, verbatim, to a configured [`Recorder`](crate::report::Recorder) —
/// this IS the time-series row). `latency_ms`/`ok` describe the one
/// round-trip; `recalls` holds 0..N values, one per query in the batch that
/// had both ground truth and returned ids.
#[derive(Debug, Clone)]
pub struct DispatchSample {
    /// Seconds since the run started, stamped at dispatch COMPLETION (the
    /// same moment the latency sample exists) in the worker — not at collector
    /// receive time, which could lag behind under load.
    pub t_s: f64,
    pub latency_ms: f64,
    pub ok: bool,
    /// A query contributes no entry here (not a `0.0` entry) when it had no
    /// ground truth to compare against, OR the whole dispatch failed (`!ok`)
    /// — a failed request has no "returned ids" to score, so it must not
    /// count as recall=0. Conflating the two would make `mean_recall` crash
    /// under load-induced errors even when every *successful* query has
    /// perfect recall — a different, already-visible finding via
    /// `errors`/`requests_per_sec`, not one recall should also report.
    pub recalls: Vec<f64>,
}

/// Build a [`DispatchSample`] from a completed batch dispatch, applying the
/// "only score recall on success" rule above per-query within the batch.
/// `idxs[i]` is the vectors-index the i-th slot in `out.ids` corresponds to.
/// `out.ids[i]` being `None` already covers both "no ground truth was
/// tracked" and "the dispatch failed" — see `BatchOutcome::ids` — so `zip`
/// alone is the whole rule; no separate `out.ok` check is needed here.
/// `started` anchors the sample's `t_s` on the run's time axis.
fn dispatch_sample(
    out: &crate::targets::BatchOutcome,
    idxs: &[usize],
    vectors: &[QueryVector],
    top_k: u64,
    started: Instant,
) -> DispatchSample {
    let recalls = idxs
        .iter()
        .zip(out.ids.iter())
        .filter_map(|(&i, ids)| {
            ids.as_ref().zip(vectors[i].ground_truth.as_ref()).map(|(ids, gt)| recall_at_k(ids, gt, top_k))
        })
        .collect();
    DispatchSample {
        t_s: started.elapsed().as_secs_f64(),
        latency_ms: out.latency.as_secs_f64() * 1000.0,
        ok: out.ok,
        recalls,
    }
}

/// The batch-of-`batch_size` indices into a round-robin `vectors` set of
/// length `n`, starting at `start` and wrapping around. Pulled out on its own
/// so the wraparound math is unit-testable without a mock run.
///
/// Precondition: `n > 0` — callers must not invoke this against an empty
/// vector set (`% 0` panics). Every current caller is already guarded by
/// `lib.rs::run()` rejecting an empty query-vector set before `run_storm` is
/// reachable; the assertion exists so a future caller added inside this
/// module fails loudly instead of hitting a raw divide-by-zero.
fn batch_indices(start: usize, batch_size: usize, n: usize) -> Vec<usize> {
    debug_assert!(n > 0, "batch_indices requires a non-empty vector set");
    (0..batch_size).map(|offset| (start + offset) % n).collect()
}

/// Drive one worker's load profile against `target` and collect latencies
/// (and recall, for queries carrying ground truth).
///
/// `vectors` is the query set to cycle through (round-robin). A dispatch failure
/// is recorded as an error sample, not a hard error — see [`BatchOutcome`](
/// crate::targets::BatchOutcome). `top_k` is the denominator for recall — see
/// [`recall_at_k`].
pub async fn run_storm(
    target: Arc<dyn QueryTarget>,
    vectors: Vec<QueryVector>,
    profile: &LoadProfile,
    top_k: u64,
    recorder: Option<Box<dyn crate::report::Recorder>>,
) -> StormResults {
    let vectors = Arc::new(vectors);
    let (tx, mut rx) = mpsc::unbounded_channel::<DispatchSample>();

    // Collector: drain samples into the raw distributions + counts, forwarding
    // each to the recorder (time-series sink) as it lands. Owning both in one
    // task keeps the workers lock-free on the hot path — a slow sink can lag
    // the collector, never the load loop (the channel is unbounded).
    let collector = tokio::spawn(async move {
        let mut recorder = recorder;
        let mut latencies = Vec::new();
        let mut recalls = Vec::new();
        let mut n_ok = 0u64;
        let mut n_err = 0u64;
        while let Some(s) = rx.recv().await {
            if let Some(r) = recorder.as_mut()
                && let Err(e) = r.record(&s)
            {
                // Time-series is auxiliary: losing it mid-run (disk full,
                // closed pipe) must not kill the load test. Warn once,
                // stop recording, keep the summary intact.
                tracing::warn!("report sink failed, disabling time-series output: {e}");
                recorder = None;
            }
            latencies.push(s.latency_ms);
            recalls.extend(s.recalls);
            if s.ok {
                n_ok += 1;
            } else {
                n_err += 1;
            }
        }
        if let Some(mut r) = recorder
            && let Err(e) = r.finish()
        {
            tracing::warn!("report sink failed on finish: {e}");
        }
        (latencies, recalls, n_ok, n_err)
    });

    let started = Instant::now();
    let stop_at = started + Duration::from_secs_f64(profile.duration_s);
    let batch_size = profile.batch_size.max(1);

    if profile.target_rps > 0.0 {
        run_paced(&target, &vectors, profile, started, stop_at, top_k, &tx).await;
    } else {
        run_closed_loop(&target, &vectors, profile, started, stop_at, top_k, &tx).await;
    }

    // Drop the last sender so the collector's `recv` loop ends.
    drop(tx);
    let wall_s = started.elapsed().as_secs_f64();

    let (latencies_ms, recalls, n_ok, n_err) = collector.await.unwrap_or_default();
    let _ = target.close().await;

    StormResults { latencies_ms, recalls, n_ok, n_err, wall_s, batch_size }
}

/// Hold `concurrency` requests in flight until the window closes; each task
/// fires the next query the instant its previous one returns.
async fn run_closed_loop(
    target: &Arc<dyn QueryTarget>,
    vectors: &Arc<Vec<QueryVector>>,
    profile: &LoadProfile,
    started: Instant,
    stop_at: Instant,
    top_k: u64,
    tx: &mpsc::UnboundedSender<DispatchSample>,
) {
    let n = vectors.len();
    let batch_size = profile.batch_size.max(1);
    let cursor = Arc::new(AtomicUsize::new(0));
    let mut workers = JoinSet::new();

    for _ in 0..profile.concurrency.max(1) {
        let target = target.clone();
        let vectors = vectors.clone();
        let tx = tx.clone();
        let cursor = cursor.clone();
        workers.spawn(async move {
            while Instant::now() < stop_at {
                // fetch_add wraps far below usize::MAX over any real run.
                let start = cursor.fetch_add(batch_size, Ordering::Relaxed) % n;
                let idxs = batch_indices(start, batch_size, n);
                let queries: Vec<&QueryVector> = idxs.iter().map(|&i| &vectors[i]).collect();
                let out = target.query_batch(&queries).await;
                let _ = tx.send(dispatch_sample(&out, &idxs, &vectors, top_k, started));
            }
        });
    }

    while workers.join_next().await.is_some() {}
}

/// Open-loop: launch a batch dispatch on a fixed `1/target_rps` schedule
/// regardless of whether prior ones have returned. `concurrency` caps in-flight
/// requests as a safety valve — when the cluster can't keep up the cap fills,
/// `acquire` stalls the dispatcher, and the achieved rate sags below target
/// (which is the finding, not an error).
async fn run_paced(
    target: &Arc<dyn QueryTarget>,
    vectors: &Arc<Vec<QueryVector>>,
    profile: &LoadProfile,
    started: Instant,
    stop_at: Instant,
    top_k: u64,
    tx: &mpsc::UnboundedSender<DispatchSample>,
) {
    let n = vectors.len();
    let batch_size = profile.batch_size.max(1);
    let interval = Duration::from_secs_f64(1.0 / profile.target_rps);
    let sem = Arc::new(Semaphore::new(profile.concurrency.max(1)));
    let mut inflight = JoinSet::new();
    let mut idx = 0usize;
    // Fixed virtual schedule: each launch is pinned to `next`, which only ever
    // advances by `interval`. Falling behind admits the next launch immediately
    // (sleep_until is already in the past), so the average tracks target.
    let mut next = Instant::now();

    while Instant::now() < stop_at {
        let permit = sem.clone().acquire_owned().await.expect("semaphore not closed");
        // acquire may have blocked; re-check the deadline before launching.
        if Instant::now() >= stop_at {
            break;
        }
        let start = idx % n;
        idx += batch_size;
        let idxs = batch_indices(start, batch_size, n);
        let target = target.clone();
        let vectors = vectors.clone();
        let tx = tx.clone();
        inflight.spawn(async move {
            let queries: Vec<&QueryVector> = idxs.iter().map(|&i| &vectors[i]).collect();
            let out = target.query_batch(&queries).await;
            let _ = tx.send(dispatch_sample(&out, &idxs, &vectors, top_k, started));
            drop(permit); // release the in-flight slot
        });

        next += interval;
        sleep_until(next).await;
    }

    while inflight.join_next().await.is_some() {}
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::targets::BatchOutcome;
    use async_trait::async_trait;

    /// A target that "answers" instantly with a fixed set of ids (or a hard
    /// error, if `fail` is set), for exercising the generator (and recall)
    /// without a real cluster.
    struct MockTarget {
        ids: Vec<String>,
        fail: bool,
    }

    impl MockTarget {
        fn ok(ids: Vec<String>) -> Self {
            Self { ids, fail: false }
        }
    }

    impl std::fmt::Display for MockTarget {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "mock")
        }
    }

    #[async_trait]
    impl QueryTarget for MockTarget {
        async fn query_batch(&self, queries: &[&QueryVector]) -> BatchOutcome {
            if self.fail {
                return BatchOutcome {
                    latency: Duration::from_micros(100),
                    ok: false,
                    ids: vec![None; queries.len()],
                    error: Some("mock failure".into()),
                };
            }
            BatchOutcome {
                latency: Duration::from_micros(100),
                ok: true,
                ids: vec![Some(self.ids.clone()); queries.len()],
                error: None,
            }
        }
    }

    fn vectors() -> Vec<QueryVector> {
        (0..16)
            .map(|i| QueryVector {
                vector: vec![i as f32; 4],
                ground_truth: None,
                filter_values: HashMap::new(),
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn closed_loop_fires_many_and_records_each() {
        let profile = LoadProfile { concurrency: 4, duration_s: 0.2, target_rps: 0.0, batch_size: 1 };
        let target = Arc::new(MockTarget::ok(vec![]));
        let results = run_storm(target, vectors(), &profile, 10, None).await;
        let summary = results.summary();

        assert!(summary.requests > 0);
        assert_eq!(summary.errors, 0);
        // every request contributes exactly one latency sample
        assert_eq!(results.latencies_ms.len() as u64, summary.requests);
        assert!(summary.requests_per_sec > 0.0);
        // batch_size 1 -> requests_per_sec and qps (actual query throughput) coincide
        assert!((summary.qps - summary.requests_per_sec).abs() < 1e-9);
        // no query in `vectors()` carries ground truth -> recall untouched, not zero
        assert!(results.recalls.is_empty());
        assert_eq!(summary.mean_recall, None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn paced_does_not_overshoot_target_rps() {
        let target_rps = 200.0;
        let duration_s = 0.5;
        let profile = LoadProfile { concurrency: 16, duration_s, target_rps, batch_size: 1 };
        let target = Arc::new(MockTarget::ok(vec![]));
        let results = run_storm(target, vectors(), &profile, 10, None).await;
        let summary = results.summary();

        // The whole point: pacing holds the offered rate at/under target. Allow a
        // small ceiling slack for scheduling, but it must not run open-throttle.
        let ceiling = (target_rps * duration_s) as u64 + profile.concurrency as u64;
        assert!(summary.requests > 0);
        assert!(
            summary.requests <= ceiling,
            "paced run overshot: {} > {}",
            summary.requests,
            ceiling
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recall_is_computed_only_for_queries_with_ground_truth() {
        // MockTarget always "returns" exactly these 2 ids, regardless of query.
        let target = Arc::new(MockTarget::ok(vec!["a".into(), "b".into()]));
        // Half the queries have ground truth overlapping 1-of-2 returned ids
        // (recall@4 = 1/4 = 0.25 each); the other half have none.
        let vectors: Vec<QueryVector> = (0..10)
            .map(|i| QueryVector {
                vector: vec![i as f32; 4],
                ground_truth: if i % 2 == 0 {
                    Some(HashSet::from(["a".to_string(), "z".into(), "y".into(), "x".into()]))
                } else {
                    None
                },
                filter_values: HashMap::new(),
            })
            .collect();

        let profile = LoadProfile { concurrency: 2, duration_s: 0.15, target_rps: 0.0, batch_size: 1 };
        let results = run_storm(target, vectors, &profile, 4, None).await;
        let summary = results.summary();

        // every recorded recall sample must be exactly 0.25 -- never 0, never
        // computed against a query that had no ground truth.
        assert!(!results.recalls.is_empty());
        assert!(results.recalls.iter().all(|&r| (r - 0.25).abs() < 1e-9));
        assert_eq!(summary.mean_recall, Some(0.25));
        // fewer recall samples than total requests -- only the ground-truthed half
        assert!((results.recalls.len() as u64) < summary.requests);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failed_queries_never_contribute_a_recall_sample() {
        // A live-Qdrant smoke test caught this: a target that fails every query
        // (e.g. wrong vector_name, transient overload) must not report
        // mean_recall=0.0 -- that would read as "search is bad" when the real
        // finding is "every request errored," which `errors`/`requests_per_sec`
        // already surface distinctly.
        let target = Arc::new(MockTarget { ids: vec![], fail: true });
        let vectors: Vec<QueryVector> = (0..8)
            .map(|i| QueryVector {
                vector: vec![i as f32; 4],
                ground_truth: Some(HashSet::from(["a".to_string()])),
                filter_values: HashMap::new(),
            })
            .collect();

        let profile = LoadProfile { concurrency: 2, duration_s: 0.15, target_rps: 0.0, batch_size: 1 };
        let results = run_storm(target, vectors, &profile, 1, None).await;
        let summary = results.summary();

        assert_eq!(summary.errors, summary.requests); // every query failed
        assert!(results.recalls.is_empty()); // -> zero recall SAMPLES, not samples of 0.0
        assert_eq!(summary.mean_recall, None); // -> "unknown", not "search returned nothing"
    }

    #[test]
    fn recall_at_k_divides_by_k_not_returned_or_ground_truth_len() {
        let returned = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let ground_truth =
            HashSet::from(["a".to_string(), "b".to_string(), "z".to_string(), "y".to_string()]);
        // 2 of the 3 returned ids are in ground_truth, k=4 -> 2/4, not 2/3 or 2/4-of-gt-len coincidence
        assert_eq!(recall_at_k(&returned, &ground_truth, 4), 0.5);
        assert_eq!(recall_at_k(&returned, &ground_truth, 2), 1.0); // capped by k, not clamped to <=1 elsewhere
        assert_eq!(recall_at_k(&[], &ground_truth, 4), 0.0);
        assert_eq!(recall_at_k(&returned, &HashSet::new(), 4), 0.0);
    }

    #[test]
    fn recall_at_k_dedupes_returned_so_a_repeated_id_cannot_exceed_1_0() {
        // "a" appears 3 times in `returned` -- must still count as a single
        // hit, not 3, or recall would read 1.5 for k=2 (impossible for a
        // fraction that's supposed to be capped at 1.0).
        let returned = vec!["a".to_string(), "a".to_string(), "a".to_string()];
        let ground_truth = HashSet::from(["a".to_string(), "b".to_string()]);
        assert_eq!(recall_at_k(&returned, &ground_truth, 2), 0.5);
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let sorted: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert_eq!(percentile(&sorted, 50.0), 50.0);
        assert_eq!(percentile(&sorted, 99.0), 99.0);
        assert_eq!(percentile(&[], 99.0), 0.0);
    }

    #[test]
    fn summary_display_appends_recall_lines_only_when_present() {
        let base = Summary {
            requests: 10,
            errors: 0,
            batch_size: 1,
            requests_per_sec: 5.0,
            qps: 5.0,
            p50_ms: 1.0,
            p95_ms: 2.0,
            p99_ms: 3.0,
            max_ms: 4.0,
            mean_recall: None,
            median_recall: None,
            min_recall: None,
        };
        assert!(!base.to_string().contains("recall"));

        let with_recall = Summary { mean_recall: Some(0.87), ..base };
        assert!(with_recall.to_string().contains("mean_recall"));
        assert!(!with_recall.to_string().contains("median_recall")); // still None -> still absent
        assert!(with_recall.to_string().ends_with("0.8700"));
    }

    #[test]
    fn summary_serializes_to_json_for_a_calling_tool_to_parse() {
        let summary = Summary {
            requests: 10,
            errors: 1,
            batch_size: 4,
            requests_per_sec: 5.0,
            qps: 20.0,
            p50_ms: 1.0,
            p95_ms: 2.0,
            p99_ms: 3.0,
            max_ms: 4.0,
            mean_recall: Some(0.87),
            median_recall: Some(0.9),
            min_recall: Some(0.5),
        };
        let json = serde_json::to_string(&summary).expect("serializes");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        assert_eq!(parsed["requests"], 10);
        assert_eq!(parsed["batch_size"], 4);
        assert_eq!(parsed["qps"], 20.0);
        assert_eq!(parsed["mean_recall"], 0.87);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn median_and_min_recall_reflect_the_distribution_not_just_the_mean() {
        // 3 distinct queries, each with a different, deterministic recall by
        // construction: 0.0 (no overlap), 0.5 (half), 1.0 (full) -- proves
        // median_recall/min_recall are computed independently from the raw
        // `recalls` samples, not just re-derived from mean_recall.
        struct PerQueryTarget;
        impl std::fmt::Display for PerQueryTarget {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "per-query-mock")
            }
        }
        #[async_trait]
        impl QueryTarget for PerQueryTarget {
            async fn query_batch(&self, queries: &[&QueryVector]) -> BatchOutcome {
                // vector[0] selects which of the 3 fixed ids come back.
                let ids = queries
                    .iter()
                    .map(|q| {
                        Some(match q.vector[0] as i64 {
                            0 => vec![], // 0/2 in ground truth -> recall 0.0
                            1 => vec!["a".to_string()], // 1/2 -> recall 0.5
                            _ => vec!["a".to_string(), "b".to_string()], // 2/2 -> recall 1.0
                        })
                    })
                    .collect();
                BatchOutcome { latency: Duration::from_micros(100), ok: true, ids, error: None }
            }
        }
        let gt = Some(HashSet::from(["a".to_string(), "b".to_string()]));
        let vectors = vec![
            QueryVector { vector: vec![0.0], ground_truth: gt.clone(), filter_values: HashMap::new() },
            QueryVector { vector: vec![1.0], ground_truth: gt.clone(), filter_values: HashMap::new() },
            QueryVector { vector: vec![2.0], ground_truth: gt, filter_values: HashMap::new() },
        ];
        // duration long enough to cycle through all 3 at concurrency=1 several times
        let profile = LoadProfile { concurrency: 1, duration_s: 0.1, target_rps: 0.0, batch_size: 1 };
        let results = run_storm(Arc::new(PerQueryTarget), vectors, &profile, 2, None).await;

        // Only asserts the pipeline actually produced all 3 distinct values --
        // NOT their exact proportions, which depend on how many times each of
        // the 3 round-robin slots happened to be hit inside a fixed wall-clock
        // window (not guaranteed 1:1:1). The exact mean/median/min math itself
        // is covered deterministically, with no timing dependency, by
        // `summary_aggregates_recall_distribution_correctly` below.
        assert!(results.recalls.iter().any(|&r| (r - 0.0).abs() < 1e-9));
        assert!(results.recalls.iter().any(|&r| (r - 0.5).abs() < 1e-9));
        assert!(results.recalls.iter().any(|&r| (r - 1.0).abs() < 1e-9));
    }

    #[test]
    fn summary_aggregates_recall_distribution_correctly() {
        // Deterministic, no async/timing involved -- exercises StormResults::summary()
        // directly, so it can assert exact mean/median/min without depending on
        // how many times a round-robin cycle happened to repeat within a window.
        let results = StormResults {
            latencies_ms: vec![1.0, 2.0, 3.0],
            recalls: vec![0.0, 0.5, 1.0],
            n_ok: 3,
            n_err: 0,
            wall_s: 1.0,
            batch_size: 1,
        };
        let summary = results.summary();

        assert_eq!(summary.min_recall, Some(0.0)); // the worst query, not hidden by the mean
        assert!((summary.mean_recall.unwrap() - 0.5).abs() < 1e-9); // (0+0.5+1)/3 = 0.5
        assert!((summary.median_recall.unwrap() - 0.5).abs() < 1e-9);
        // mean and median coincide here (symmetric distribution) -- min is the
        // one that actually differs from both, proving it's not just an alias.
        assert_ne!(summary.min_recall, summary.mean_recall);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recorder_receives_one_timestamped_row_per_dispatch() {
        use crate::report::{Recorder, ReportConfig, ReportFormat};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ts.csv").to_string_lossy().into_owned();
        let cfg = ReportConfig { format: ReportFormat::Csv, path: path.clone() };
        let mut recorder = cfg.build();
        recorder.begin().expect("begin");

        let profile = LoadProfile { concurrency: 4, duration_s: 0.2, target_rps: 0.0, batch_size: 1 };
        let target = Arc::new(MockTarget::ok(vec![]));
        let results = run_storm(target, vectors(), &profile, 10, Some(recorder)).await;
        let summary = results.summary();

        let text = std::fs::read_to_string(&path).expect("csv written");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "t_s,latency_ms,ok,recalls");
        // exactly one row per dispatch — the time series IS the raw run
        assert_eq!(lines.len() as u64, 1 + summary.requests);
        // timestamps are on the run's time axis: non-negative, within the
        // window (plus scheduling slack), and present on every row
        for line in &lines[1..] {
            let t: f64 = line.split(',').next().unwrap().parse().expect("t_s parses");
            assert!(t >= 0.0 && t < 5.0, "t_s out of range: {t}");
        }
    }

    #[test]
    fn batch_indices_wrap_around() {
        assert_eq!(batch_indices(6, 5, 8), vec![6, 7, 0, 1, 2]);
        assert_eq!(batch_indices(0, 3, 8), vec![0, 1, 2]);
        assert_eq!(batch_indices(0, 1, 8), vec![0]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn batches_group_multiple_queries_per_dispatch_and_score_each() {
        // Records the length of every query_batch call, and returns ids chosen
        // by position-within-the-batch so each slot scores a distinct, known
        // recall -- proving query_batch is actually called with `batch_size`
        // vectors together (not looped internally), and each query in the
        // batch gets its own correct recall score.
        struct BatchCapturingTarget {
            call_lens: std::sync::Mutex<Vec<usize>>,
        }
        impl std::fmt::Display for BatchCapturingTarget {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "batch-capturing-mock")
            }
        }
        #[async_trait]
        impl QueryTarget for BatchCapturingTarget {
            async fn query_batch(&self, queries: &[&QueryVector]) -> BatchOutcome {
                self.call_lens.lock().unwrap().push(queries.len());
                // position 0 -> 0/2 in ground truth -> recall 0.0
                // position 1 -> 1/2 -> recall 0.5
                // position 2 -> 2/2 -> recall 1.0
                let ids = (0..queries.len())
                    .map(|pos| {
                        Some(match pos % 3 {
                            0 => vec![],
                            1 => vec!["a".to_string()],
                            _ => vec!["a".to_string(), "b".to_string()],
                        })
                    })
                    .collect();
                BatchOutcome { latency: Duration::from_micros(100), ok: true, ids, error: None }
            }
        }

        let gt = Some(HashSet::from(["a".to_string(), "b".to_string()]));
        let vectors: Vec<QueryVector> = (0..9)
            .map(|i| QueryVector {
                vector: vec![i as f32],
                ground_truth: gt.clone(),
                filter_values: HashMap::new(),
            })
            .collect();
        let batch_size = 3;
        let profile = LoadProfile { concurrency: 1, duration_s: 0.15, target_rps: 0.0, batch_size };
        let target = Arc::new(BatchCapturingTarget { call_lens: std::sync::Mutex::new(Vec::new()) });
        let results = run_storm(target.clone(), vectors, &profile, 2, None).await;

        let call_lens = target.call_lens.lock().unwrap();
        assert!(!call_lens.is_empty());
        assert!(call_lens.iter().all(|&len| len == batch_size), "{call_lens:?}");

        assert!(results.recalls.iter().any(|&r| (r - 0.0).abs() < 1e-9));
        assert!(results.recalls.iter().any(|&r| (r - 0.5).abs() < 1e-9));
        assert!(results.recalls.iter().any(|&r| (r - 1.0).abs() < 1e-9));
        assert_eq!(results.summary().batch_size, batch_size);
    }
}
