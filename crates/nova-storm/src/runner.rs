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
//!   can't delay the next launch and hide latency — and guarantees the offered
//!   rate tracks the target without overshooting it.
//!
//! Aggregating ACROSS workers is a separate step (see `nova storm-dist`) and
//! must merge latency *distributions*, never average per-worker percentiles.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use nova_metrics::MetricsSink;
use serde::Serialize;
use tokio::sync::{Semaphore, mpsc};
use tokio::task::JoinSet;
use tokio::time::{Duration, Instant, sleep_until};

use crate::config::LoadProfile;
use crate::targets::QueryTarget;

/// One worker's raw measurements. Latencies are kept as a full sample (not
/// pre-aggregated) so a fleet merge can recompute true percentiles.
#[derive(Debug, Clone)]
pub struct StormResults {
    pub latencies_ms: Vec<f64>,
    pub n_ok: u64,
    pub n_err: u64,
    pub wall_s: f64,
}

/// Aggregated stats for THIS worker. Fleet-wide stats must merge raw samples
/// from every worker, not average these.
#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub requests: u64,
    pub errors: u64,
    pub throughput_qps: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

impl StormResults {
    pub fn summary(&self) -> Summary {
        let mut ms = self.latencies_ms.clone();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let total = self.n_ok + self.n_err;
        Summary {
            requests: total,
            errors: self.n_err,
            throughput_qps: if self.wall_s > 0.0 {
                total as f64 / self.wall_s
            } else {
                0.0
            },
            p50_ms: percentile(&ms, 50.0),
            p95_ms: percentile(&ms, 95.0),
            p99_ms: percentile(&ms, 99.0),
            max_ms: ms.last().copied().unwrap_or(0.0),
        }
    }
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{:>16}: {}", "requests", self.requests)?;
        writeln!(f, "{:>16}: {}", "errors", self.errors)?;
        writeln!(f, "{:>16}: {:.1}", "throughput_qps", self.throughput_qps)?;
        writeln!(f, "{:>16}: {:.2}", "p50_ms", self.p50_ms)?;
        writeln!(f, "{:>16}: {:.2}", "p95_ms", self.p95_ms)?;
        writeln!(f, "{:>16}: {:.2}", "p99_ms", self.p99_ms)?;
        write!(f, "{:>16}: {:.2}", "max_ms", self.max_ms)
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

/// One latency observation forwarded from a worker to the collector.
struct Sample {
    latency_ms: f64,
    ok: bool,
}

/// Drive one worker's load profile against `target` and collect latencies.
///
/// `vectors` is the query set to cycle through (round-robin). A query failure
/// is recorded as an error sample, not a hard error — see [`QueryOutcome`].
/// Every sample is also streamed to `sink` (live latency in Grafana when the
/// Postgres backend is configured; a no-op for the null sink).
pub async fn run_storm(
    target: Arc<dyn QueryTarget>,
    vectors: Vec<Vec<f32>>,
    profile: &LoadProfile,
    sink: Arc<dyn MetricsSink>,
) -> StormResults {
    let vectors = Arc::new(vectors);
    let (tx, mut rx) = mpsc::unbounded_channel::<Sample>();

    // Collector: drain samples into the raw distribution + counts, and forward
    // each as a live observation. Owning the accumulation in one task keeps the
    // workers lock-free on the hot path; `observe` only enqueues, never blocks.
    let collector = tokio::spawn(async move {
        let mut latencies = Vec::new();
        let mut n_ok = 0u64;
        let mut n_err = 0u64;
        while let Some(s) = rx.recv().await {
            sink.observe("latency_ms", s.latency_ms, s.ok);
            latencies.push(s.latency_ms);
            if s.ok {
                n_ok += 1;
            } else {
                n_err += 1;
            }
        }
        (latencies, n_ok, n_err)
    });

    // TODO: honour profile.ramp_s (stagger task starts) and add a coordinated
    // fleet-wide start so all workers hammer simultaneously.
    let started = Instant::now();
    let stop_at = started + Duration::from_secs_f64(profile.duration_s);

    if profile.target_qps > 0.0 {
        run_paced(&target, &vectors, profile, stop_at, &tx).await;
    } else {
        run_closed_loop(&target, &vectors, profile, stop_at, &tx).await;
    }

    // Drop the last sender so the collector's `recv` loop ends.
    drop(tx);
    let wall_s = started.elapsed().as_secs_f64();

    let (latencies_ms, n_ok, n_err) = collector.await.unwrap_or_default();
    let _ = target.close().await;

    StormResults {
        latencies_ms,
        n_ok,
        n_err,
        wall_s,
    }
}

/// Hold `concurrency` requests in flight until the window closes; each task
/// fires the next query the instant its previous one returns.
async fn run_closed_loop(
    target: &Arc<dyn QueryTarget>,
    vectors: &Arc<Vec<Vec<f32>>>,
    profile: &LoadProfile,
    stop_at: Instant,
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
                let out = target.query(&vectors[i]).await;
                let _ = tx.send(Sample {
                    latency_ms: out.latency.as_secs_f64() * 1000.0,
                    ok: out.ok,
                });
            }
        });
    }

    while workers.join_next().await.is_some() {}
}

/// Open-loop: launch a query on a fixed `1/target_qps` schedule regardless of
/// whether prior ones have returned. `concurrency` caps in-flight requests as a
/// safety valve — when the cluster can't keep up the cap fills, `acquire`
/// stalls the dispatcher, and the achieved QPS sags below target (which is the
/// finding, not an error).
async fn run_paced(
    target: &Arc<dyn QueryTarget>,
    vectors: &Arc<Vec<Vec<f32>>>,
    profile: &LoadProfile,
    stop_at: Instant,
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
        let permit = sem
            .clone()
            .acquire_owned()
            .await
            .expect("semaphore not closed");
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
            let out = target.query(&vectors[i]).await;
            let _ = tx.send(Sample {
                latency_ms: out.latency.as_secs_f64() * 1000.0,
                ok: out.ok,
            });
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
    use nova_metrics::NullSink;

    /// A target that "answers" instantly, for exercising the generator without
    /// a real cluster.
    struct MockTarget;

    impl std::fmt::Display for MockTarget {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "mock")
        }
    }

    #[async_trait]
    impl QueryTarget for MockTarget {
        async fn query(&self, _vector: &[f32]) -> QueryOutcome {
            QueryOutcome {
                latency: Duration::from_micros(100),
                ok: true,
                matched: 1,
                error: None,
            }
        }
    }

    fn vectors() -> Vec<Vec<f32>> {
        (0..16).map(|i| vec![i as f32; 4]).collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn closed_loop_fires_many_and_records_each() {
        let profile = LoadProfile {
            concurrency: 4,
            duration_s: 0.2,
            ramp_s: 0.0,
            target_qps: 0.0,
        };
        let results =
            run_storm(Arc::new(MockTarget), vectors(), &profile, Arc::new(NullSink)).await;
        let summary = results.summary();

        assert!(summary.requests > 0);
        assert_eq!(summary.errors, 0);
        // every request contributes exactly one latency sample
        assert_eq!(results.latencies_ms.len() as u64, summary.requests);
        assert!(summary.throughput_qps > 0.0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn paced_does_not_overshoot_target_qps() {
        let target_qps = 200.0;
        let duration_s = 0.5;
        let profile = LoadProfile {
            concurrency: 16,
            duration_s,
            ramp_s: 0.0,
            target_qps,
        };
        let results =
            run_storm(Arc::new(MockTarget), vectors(), &profile, Arc::new(NullSink)).await;
        let summary = results.summary();

        // The whole point: pacing holds the offered rate at/under target. Allow
        // a small ceiling slack for scheduling, but it must not run open-throttle
        // (closed-loop with a ~0-latency mock would be orders of magnitude more).
        let ceiling = (target_qps * duration_s) as u64 + profile.concurrency as u64;
        assert!(summary.requests > 0);
        assert!(
            summary.requests <= ceiling,
            "paced run overshot: {} > {}",
            summary.requests,
            ceiling
        );
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let sorted: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert_eq!(percentile(&sorted, 50.0), 50.0);
        assert_eq!(percentile(&sorted, 99.0), 99.0);
        assert_eq!(percentile(&[], 99.0), 0.0);
    }
}
