//! The load generator + single-worker result summary.
//!
//! Two modes, chosen by [`LoadProfile::target_qps`](crate::config::LoadProfile):
//!
//! * `0` — **closed-loop**: hold `concurrency` requests in flight for the whole
//!   window, each task firing the next query the instant its previous one
//!   returns. Measures the max throughput the cluster gives at that depth.
//! * `>0` — **open-loop paced**: launch one query every `1/target_qps` seconds
//!   on a fixed virtual schedule, with `concurrency` as an in-flight ceiling.
//!   The fixed schedule is what avoids coordinated omission — a slow response
//!   can't delay the next launch and hide latency — and keeps the offered rate
//!   tracking the target without overshooting it.
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
    pub latencies_ms: Vec<f64>,
    /// One entry per query that had ground truth (see [`QueryVector::ground_truth`]).
    /// Shorter than `latencies_ms` whenever some/all queries have none — that's
    /// expected, not an error.
    pub recalls: Vec<f64>,
    pub n_ok: u64,
    pub n_err: u64,
    pub wall_s: f64,
}

/// Aggregated stats for THIS worker. Fleet-wide stats must merge raw samples
/// from every worker, not average these.
#[derive(Debug, Clone)]
pub struct Summary {
    pub requests: u64,
    pub errors: u64,
    pub throughput_qps: f64,
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
        Summary {
            requests: total,
            errors: self.n_err,
            throughput_qps: if self.wall_s > 0.0 { total as f64 / self.wall_s } else { 0.0 },
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
            format!("{:>16}: {:.1}", "throughput_qps", self.throughput_qps),
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
fn recall_at_k(returned: &[String], ground_truth: &HashSet<String>, k: u64) -> f64 {
    let hits = returned.iter().filter(|id| ground_truth.contains(id.as_str())).count();
    hits as f64 / k as f64
}

/// One query observation forwarded from a worker to the collector.
struct Sample {
    latency_ms: f64,
    ok: bool,
    /// `None` when this query had no ground truth to compare against, OR the
    /// query itself failed (`!ok`) — a failed request has no "returned ids" to
    /// score, so it must not count as recall=0. Conflating the two would make
    /// `mean_recall` crash under load-induced errors even when every
    /// *successful* query has perfect recall — a different, already-visible
    /// finding via `errors`/`throughput_qps`, not one recall should also report.
    recall: Option<f64>,
}

/// Build a [`Sample`] from a completed query, applying the "only score recall
/// on success" rule above in one place (both load-shape functions call this).
/// `out.ids` being `None` already covers both "no ground truth was tracked"
/// and "the query failed" — see `QueryOutcome::ids` — so `zip` alone is the
/// whole rule; no separate `out.ok` check is needed here anymore.
fn sample_from(out: &crate::targets::QueryOutcome, ground_truth: Option<&HashSet<String>>, top_k: u64) -> Sample {
    let recall = out.ids.as_ref().zip(ground_truth).map(|(ids, gt)| recall_at_k(ids, gt, top_k));
    Sample { latency_ms: out.latency.as_secs_f64() * 1000.0, ok: out.ok, recall }
}

/// Drive one worker's load profile against `target` and collect latencies
/// (and recall, for queries carrying ground truth).
///
/// `vectors` is the query set to cycle through (round-robin). A query failure is
/// recorded as an error sample, not a hard error — see [`QueryOutcome`]. `top_k`
/// is the denominator for recall — see [`recall_at_k`].
pub async fn run_storm(
    target: Arc<dyn QueryTarget>,
    vectors: Vec<QueryVector>,
    profile: &LoadProfile,
    top_k: u64,
) -> StormResults {
    let vectors = Arc::new(vectors);
    let (tx, mut rx) = mpsc::unbounded_channel::<Sample>();

    // Collector: drain samples into the raw distributions + counts. Owning the
    // accumulation in one task keeps the workers lock-free on the hot path.
    let collector = tokio::spawn(async move {
        let mut latencies = Vec::new();
        let mut recalls = Vec::new();
        let mut n_ok = 0u64;
        let mut n_err = 0u64;
        while let Some(s) = rx.recv().await {
            latencies.push(s.latency_ms);
            if let Some(r) = s.recall {
                recalls.push(r);
            }
            if s.ok {
                n_ok += 1;
            } else {
                n_err += 1;
            }
        }
        (latencies, recalls, n_ok, n_err)
    });

    let started = Instant::now();
    let stop_at = started + Duration::from_secs_f64(profile.duration_s);

    if profile.target_qps > 0.0 {
        run_paced(&target, &vectors, profile, stop_at, top_k, &tx).await;
    } else {
        run_closed_loop(&target, &vectors, profile, stop_at, top_k, &tx).await;
    }

    // Drop the last sender so the collector's `recv` loop ends.
    drop(tx);
    let wall_s = started.elapsed().as_secs_f64();

    let (latencies_ms, recalls, n_ok, n_err) = collector.await.unwrap_or_default();
    let _ = target.close().await;

    StormResults { latencies_ms, recalls, n_ok, n_err, wall_s }
}

/// Hold `concurrency` requests in flight until the window closes; each task
/// fires the next query the instant its previous one returns.
async fn run_closed_loop(
    target: &Arc<dyn QueryTarget>,
    vectors: &Arc<Vec<QueryVector>>,
    profile: &LoadProfile,
    stop_at: Instant,
    top_k: u64,
    tx: &mpsc::UnboundedSender<Sample>,
) {
    let n = vectors.len();
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
                let i = cursor.fetch_add(1, Ordering::Relaxed) % n;
                let qv = &vectors[i];
                let out = target.query(&qv.vector).await;
                let _ = tx.send(sample_from(&out, qv.ground_truth.as_ref(), top_k));
            }
        });
    }

    while workers.join_next().await.is_some() {}
}

/// Open-loop: launch a query on a fixed `1/target_qps` schedule regardless of
/// whether prior ones have returned. `concurrency` caps in-flight requests as a
/// safety valve — when the cluster can't keep up the cap fills, `acquire` stalls
/// the dispatcher, and the achieved QPS sags below target (which is the finding,
/// not an error).
async fn run_paced(
    target: &Arc<dyn QueryTarget>,
    vectors: &Arc<Vec<QueryVector>>,
    profile: &LoadProfile,
    stop_at: Instant,
    top_k: u64,
    tx: &mpsc::UnboundedSender<Sample>,
) {
    let n = vectors.len();
    let interval = Duration::from_secs_f64(1.0 / profile.target_qps);
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
        let i = idx % n;
        idx += 1;
        let target = target.clone();
        let vectors = vectors.clone();
        let tx = tx.clone();
        inflight.spawn(async move {
            let qv = &vectors[i];
            let out = target.query(&qv.vector).await;
            let _ = tx.send(sample_from(&out, qv.ground_truth.as_ref(), top_k));
            drop(permit); // release the in-flight slot
        });

        next += interval;
        sleep_until(next).await;
    }

    while inflight.join_next().await.is_some() {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::targets::QueryOutcome;
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
        async fn query(&self, _vector: &[f32]) -> QueryOutcome {
            if self.fail {
                return QueryOutcome {
                    latency: Duration::from_micros(100),
                    ok: false,
                    ids: None,
                    error: Some("mock failure".into()),
                };
            }
            QueryOutcome {
                latency: Duration::from_micros(100),
                ok: true,
                ids: Some(self.ids.clone()),
                error: None,
            }
        }
    }

    fn vectors() -> Vec<QueryVector> {
        (0..16).map(|i| QueryVector { vector: vec![i as f32; 4], ground_truth: None }).collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn closed_loop_fires_many_and_records_each() {
        let profile = LoadProfile { concurrency: 4, duration_s: 0.2, target_qps: 0.0 };
        let target = Arc::new(MockTarget::ok(vec![]));
        let results = run_storm(target, vectors(), &profile, 10).await;
        let summary = results.summary();

        assert!(summary.requests > 0);
        assert_eq!(summary.errors, 0);
        // every request contributes exactly one latency sample
        assert_eq!(results.latencies_ms.len() as u64, summary.requests);
        assert!(summary.throughput_qps > 0.0);
        // no query in `vectors()` carries ground truth -> recall untouched, not zero
        assert!(results.recalls.is_empty());
        assert_eq!(summary.mean_recall, None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn paced_does_not_overshoot_target_qps() {
        let target_qps = 200.0;
        let duration_s = 0.5;
        let profile = LoadProfile { concurrency: 16, duration_s, target_qps };
        let target = Arc::new(MockTarget::ok(vec![]));
        let results = run_storm(target, vectors(), &profile, 10).await;
        let summary = results.summary();

        // The whole point: pacing holds the offered rate at/under target. Allow a
        // small ceiling slack for scheduling, but it must not run open-throttle.
        let ceiling = (target_qps * duration_s) as u64 + profile.concurrency as u64;
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
            })
            .collect();

        let profile = LoadProfile { concurrency: 2, duration_s: 0.15, target_qps: 0.0 };
        let results = run_storm(target, vectors, &profile, 4).await;
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
        // finding is "every request errored," which `errors`/`throughput_qps`
        // already surface distinctly.
        let target = Arc::new(MockTarget { ids: vec![], fail: true });
        let vectors: Vec<QueryVector> = (0..8)
            .map(|i| QueryVector {
                vector: vec![i as f32; 4],
                ground_truth: Some(HashSet::from(["a".to_string()])),
            })
            .collect();

        let profile = LoadProfile { concurrency: 2, duration_s: 0.15, target_qps: 0.0 };
        let results = run_storm(target, vectors, &profile, 1).await;
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
            throughput_qps: 5.0,
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
            async fn query(&self, vector: &[f32]) -> QueryOutcome {
                // vector[0] selects which of the 3 fixed ids come back.
                let ids = match vector[0] as i64 {
                    0 => vec![], // 0/2 in ground truth -> recall 0.0
                    1 => vec!["a".to_string()], // 1/2 -> recall 0.5
                    _ => vec!["a".to_string(), "b".to_string()], // 2/2 -> recall 1.0
                };
                QueryOutcome { latency: Duration::from_micros(100), ok: true, ids: Some(ids), error: None }
            }
        }
        let gt = Some(HashSet::from(["a".to_string(), "b".to_string()]));
        let vectors = vec![
            QueryVector { vector: vec![0.0], ground_truth: gt.clone() },
            QueryVector { vector: vec![1.0], ground_truth: gt.clone() },
            QueryVector { vector: vec![2.0], ground_truth: gt },
        ];
        // duration long enough to cycle through all 3 at concurrency=1 several times
        let profile = LoadProfile { concurrency: 1, duration_s: 0.1, target_qps: 0.0 };
        let results = run_storm(Arc::new(PerQueryTarget), vectors, &profile, 2).await;

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
        };
        let summary = results.summary();

        assert_eq!(summary.min_recall, Some(0.0)); // the worst query, not hidden by the mean
        assert!((summary.mean_recall.unwrap() - 0.5).abs() < 1e-9); // (0+0.5+1)/3 = 0.5
        assert!((summary.median_recall.unwrap() - 0.5).abs() < 1e-9);
        // mean and median coincide here (symmetric distribution) -- min is the
        // one that actually differs from both, proving it's not just an alias.
        assert_ne!(summary.min_recall, summary.mean_recall);
    }
}
