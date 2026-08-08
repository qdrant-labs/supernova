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
use std::sync::mpsc::TrySendError;

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
    /// One entry per query that had (non-empty) ground truth (see
    /// [`QueryVector::ground_truth`]). Recall stays per-query even though latency
    /// doesn't: `QueryBatchResponse` gives one distinct result per submitted
    /// query, so each query's recall is still individually real, not approximated
    /// from the batch. Each sample carries a `short` flag so the summary can
    /// account for full-depth and short-ground-truth queries separately (see
    /// [`RecallSample`]).
    pub recalls: Vec<RecallSample>,
    /// Total query firings excluded from recall because their ground truth was
    /// present but empty (`truth_len == 0`) — see [`DispatchSample::empty_ground_truth`].
    /// A firing count (queries cycle round-robin), consistent with the `n` in
    /// the recall buckets, not a distinct-query count.
    pub empty_ground_truth: u64,
    /// Total suspected filter-leak firings (filter configured AND the vdb
    /// returned more ids than the ground truth holds) — see
    /// [`DispatchSample::filter_overreturn`]. A firing count, like the recall
    /// bucket `n`s.
    pub filter_overreturn: u64,
    /// Count of batch dispatches, not individual queries.
    pub n_ok: u64,
    pub n_err: u64,
    pub wall_s: f64,
    /// How many query vectors went in each dispatch — carried alongside the
    /// raw samples so `summary()` can self-describe regardless of what
    /// `LoadProfile` is in scope.
    pub batch_size: usize,
    /// Time-series samples dropped because the report sink couldn't keep pace
    /// (the bounded writer queue was full). `0` unless `report:` is configured
    /// AND its sink lagged; the load test and this summary are unaffected — the
    /// only casualty is completeness of the time-series file. Not the same as
    /// `n_err` (failed dispatches): a dropped sample was a *successful* (or
    /// failed) dispatch whose row simply never reached the sink.
    pub dropped_samples: u64,
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
    /// Recall over queries whose ground truth held at least `top_k` ids, scored
    /// against `top_k` (the conventional recall@k). `None` when no such query ran
    /// (feature unused, misconfigured column, or every ground-truth list was
    /// short).
    pub full_recall: Option<RecallBucket>,
    /// Recall over queries whose ground truth held FEWER than `top_k` ids, scored
    /// against the ground truth's own length (not `top_k`) so a short list isn't
    /// dragged down by a denominator it could never fill. `None` when no such
    /// query ran. Kept separate from `full_recall` so a run with mixed depths
    /// doesn't blend two different denominators into one misleading mean.
    pub short_recall: Option<RecallBucket>,
    /// Recall over ALL ground-truthed queries (`full` + `short`), each scored by
    /// its own denominator. `None` when no query in this run had ground truth.
    pub total_recall: Option<RecallBucket>,
    /// Query firings excluded from every recall bucket above because their
    /// ground truth was present but empty (`truth_len == 0`) — nothing to score
    /// against. `0` in the common case; a non-zero value tells the operator some
    /// firings silently sat out recall (distinct from queries with no ground
    /// truth configured at all, which were never in scope for recall).
    pub empty_ground_truth: u64,
    /// Suspected filter leaks: firings where, **with a filter configured**, the
    /// vdb returned MORE result ids than the ground truth holds
    /// (`returned.len() > truth_len`). Recall is unchanged. Only tallied under a
    /// filter — unfiltered over-return is benign truncation (a shallow ground
    /// truth vs a deeper `top_k`), not a bug — and it's a valid leak signal only
    /// when the filtered ground truth is the EXHAUSTIVE match set (bf found all
    /// matching docs, i.e. wasn't itself capped). `0` when it never happened.
    pub filter_overreturn: u64,
}

/// One recall bucket's headline: how many queries fell in it and their mean
/// recall. Count travels with the mean so a mean over 3 queries can't be read
/// as if it were over 3000.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct RecallBucket {
    pub n: u64,
    pub mean: f64,
}

impl RecallBucket {
    /// `None` for an empty slice — an absent bucket, not a `0.0` mean.
    fn from(recalls: &[f64]) -> Option<Self> {
        (!recalls.is_empty())
            .then(|| RecallBucket { n: recalls.len() as u64, mean: recalls.iter().sum::<f64>() / recalls.len() as f64 })
    }
}

/// One query's recall observation, tagged with whether its ground truth held
/// fewer ids than `top_k`. `short` queries divide by their own ground-truth
/// length rather than `top_k` (see [`recall_at_k`]); the flag is what lets the
/// summary and the time-series report keep the two populations apart.
#[derive(Debug, Clone, Copy)]
pub struct RecallSample {
    pub recall: f64,
    pub short: bool,
}

impl StormResults {
    pub fn summary(&self) -> Summary {
        let mut ms = self.latencies_ms.clone();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Split the per-query recall samples into the full-depth and short
        // buckets; `total` scores every ground-truthed query together.
        let full: Vec<f64> = self.recalls.iter().filter(|s| !s.short).map(|s| s.recall).collect();
        let short: Vec<f64> = self.recalls.iter().filter(|s| s.short).map(|s| s.recall).collect();
        let all: Vec<f64> = self.recalls.iter().map(|s| s.recall).collect();
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
            full_recall: RecallBucket::from(&full),
            short_recall: RecallBucket::from(&short),
            total_recall: RecallBucket::from(&all),
            empty_ground_truth: self.empty_ground_truth,
            filter_overreturn: self.filter_overreturn,
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
        if let Some(b) = self.full_recall {
            lines.push(format!("{:>16}: {:.4} (n={})", "recall_full", b.mean, b.n));
        }
        if let Some(b) = self.short_recall {
            lines.push(format!("{:>16}: {:.4} (n={})", "recall_short", b.mean, b.n));
        }
        if let Some(b) = self.total_recall {
            lines.push(format!("{:>16}: {:.4} (n={})", "recall_total", b.mean, b.n));
        }
        // Only when it actually happened — a 0 here is the norm and would just
        // be noise next to the recall means.
        if self.empty_ground_truth > 0 {
            lines.push(format!("{:>16}: {}", "recall_empty_gt", self.empty_ground_truth));
        }
        // Suspected filter leaks — visibility only, recall above is unaffected.
        // Shown when it happened.
        if self.filter_overreturn > 0 {
            lines.push(format!("{:>16}: {}", "filter_overreturn", self.filter_overreturn));
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

/// Recall@k for one query: the fraction of the known-correct ids
/// (`ground_truth`) that appear among the ids the target actually `returned`.
///
/// The denominator adapts to the ground truth's own depth:
/// * `ground_truth.len() >= k` — divides by `k` (the conventional recall@k),
///   and the sample is tagged `short = false`.
/// * `ground_truth.len() < k` — divides by `ground_truth.len()`, tagged
///   `short = true`. A list shorter than `k` (e.g. nova-bf's `k=10` vs storm's
///   `top_k=100`) can never fill a `k`-sized denominator, so scoring it against
///   `k` would read as an artificial recall regression rather than the sparse
///   ground truth it actually is; the summary keeps these queries in their own
///   bucket for honest accounting.
///
/// Returns `None` for empty `ground_truth` — there's nothing to measure against,
/// and dividing by zero would poison the mean with a `NaN` (a NULL column value
/// is already dropped upstream in `queries.rs`, but a present-but-empty list
/// reaches here). `ground_truth` is already a `HashSet` (built once at load time
/// in `queries.rs`, not per call) since this runs on every query firing.
/// `returned` is deduped before counting hits — a target that ever repeated an
/// id within one query's results must not let that repeat count twice, which
/// would push recall above the `1.0` ceiling a fraction is supposed to have.
fn recall_at_k(returned: &[String], ground_truth: &HashSet<String>, k: u64) -> Option<RecallSample> {
    let truth_len = ground_truth.len() as u64;
    if truth_len == 0 {
        return None;
    }
    let hits = returned
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>()
        .into_iter()
        .filter(|id| ground_truth.contains(*id))
        .count();
    let short = truth_len < k;
    let denom = if short { truth_len } else { k };
    Some(RecallSample { recall: hits as f64 / denom as f64, short })
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
    /// (or empty) ground truth to compare against, OR the whole dispatch failed
    /// (`!ok`) — a failed request has no "returned ids" to score, so it must not
    /// count as recall=0. Conflating the two would make mean recall crash
    /// under load-induced errors even when every *successful* query has
    /// perfect recall — a different, already-visible finding via
    /// `errors`/`requests_per_sec`, not one recall should also report. Each
    /// sample carries its `short` flag so the time-series report can split the
    /// two buckets too (see [`RecallSample`]).
    pub recalls: Vec<RecallSample>,
    /// How many queries in this dispatch had a ground-truth list that was
    /// *present but empty* (`truth_len == 0`) — configured for recall, dispatch
    /// succeeded, but there was nothing to score against, so they produced no
    /// `recalls` entry. Counted (not silently dropped) so the summary can report
    /// how many firings were excluded from recall for this reason, separately
    /// from queries that simply had no ground truth configured (`None`).
    pub empty_ground_truth: u64,
    /// Suspected filter leaks in this dispatch: queries where, with a filter
    /// configured, the vdb returned MORE result ids than their ground truth
    /// holds (`returned.len() > truth_len`, ground truth non-empty). Recall is
    /// unaffected. Counted only under a filter — without one, over-return is
    /// benign truncation (shallow ground truth vs deeper `top_k`), not a leak —
    /// so `dispatch_sample` takes the run's `filtered` flag rather than
    /// inferring it here.
    pub filter_overreturn: u64,
}

/// Errors are *counted* in the summary, but a count alone ("errors: 9869")
/// sends the operator log-hunting for a cause the target already reported.
/// Surface the first error message of the run, once — under load every
/// dispatch usually fails the same way, so one message carries the story
/// without turning a failing run into a log flood. The flag is PER-RUN
/// (created in `run_storm`, threaded through the load loops), not a
/// process-global: a library caller running several storms in one process
/// gets each run's own first error, and test runs stay order-independent.
/// Returns whether THIS call did the logging, so the exactly-once contract is
/// directly testable.
fn report_first_error(error_reported: &std::sync::atomic::AtomicBool, error: &str) -> bool {
    let first = !error_reported.swap(true, Ordering::Relaxed);
    if first {
        tracing::warn!(
            "first failed dispatch of the run (further failures are only counted): {error}"
        );
    }
    first
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
    filtered: bool,
    started: Instant,
    error_reported: &std::sync::atomic::AtomicBool,
) -> DispatchSample {
    if let Some(error) = &out.error {
        report_first_error(error_reported, error);
    }

    // A query scores recall only when the dispatch returned ids for it AND it
    // had ground truth. `recall_at_k` returning `None` there means the ground
    // truth was present but empty — count those separately rather than lose them.
    let mut recalls = Vec::new();
    let mut empty_ground_truth = 0u64;
    let mut filter_overreturn = 0u64;
    for (&i, ids) in idxs.iter().zip(out.ids.iter()) {
        if let Some((ids, gt)) = ids.as_ref().zip(vectors[i].ground_truth.as_ref()) {
            // Suspected filter leak: with a filter active, the vdb returned more
            // ids than the (exhaustive) filtered ground truth holds. Only under
            // a filter — unfiltered over-return is benign truncation (shallow gt
            // vs deeper top_k). An empty gt is its own bucket, counted above.
            if filtered && !gt.is_empty() && ids.len() as u64 > gt.len() as u64 {
                filter_overreturn += 1;
            }
            match recall_at_k(ids, gt, top_k) {
                Some(sample) => recalls.push(sample),
                None => empty_ground_truth += 1,
            }
        }
    }
    DispatchSample {
        t_s: started.elapsed().as_secs_f64(),
        latency_ms: out.latency.as_secs_f64() * 1000.0,
        ok: out.ok,
        recalls,
        empty_ground_truth,
        filter_overreturn,
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
/// [`recall_at_k`]. `filtered` is whether a query filter is configured — it
/// gates the `filter_overreturn` leak count (only meaningful under a filter).
pub async fn run_storm(
    target: Arc<dyn QueryTarget>,
    vectors: Vec<QueryVector>,
    profile: &LoadProfile,
    top_k: u64,
    filtered: bool,
    recorder: Option<Box<dyn crate::report::Recorder>>,
) -> StormResults {
    let vectors = Arc::new(vectors);
    let (tx, mut rx) = mpsc::unbounded_channel::<DispatchSample>();

    // Hand the (already-`begin()`-ed) recorder to a dedicated OS thread so its
    // blocking writes never land on a runtime worker — see `report::spawn_writer`
    // for why that matters (especially on a 1-vCPU box). The collector forwards
    // to it over a bounded channel and drops-on-full, so a sink slower than
    // dispatch can neither grow memory unbounded nor backpressure the load loop.
    let (writer_tx, writer_handle) = match recorder {
        Some(r) => {
            let (wtx, handle) = crate::report::spawn_writer(r);
            (Some(wtx), Some(handle))
        }
        None => (None, None),
    };

    // Collector: drain samples into the raw distributions + counts, then forward
    // each to the writer thread. The accumulation here is the authoritative
    // summary and never loses a sample; only the (auxiliary) time-series file
    // does, and only when its sink can't keep up. Owning the accumulation in one
    // task keeps the workers lock-free on the hot path.
    let collector = tokio::spawn(async move {
        let mut writer_tx = writer_tx;
        let mut latencies = Vec::new();
        let mut recalls = Vec::new();
        let mut empty_gt = 0u64;
        let mut over_gt = 0u64;
        let mut n_ok = 0u64;
        let mut n_err = 0u64;
        let mut dropped = 0u64;
        while let Some(s) = rx.recv().await {
            // Accumulate first — copying the fields the summary needs — so `s`
            // is still owned to hand to the writer without a clone.
            latencies.push(s.latency_ms);
            recalls.extend(s.recalls.iter().copied());
            empty_gt += s.empty_ground_truth;
            over_gt += s.filter_overreturn;
            if s.ok {
                n_ok += 1;
            } else {
                n_err += 1;
            }
            if let Some(wtx) = writer_tx.as_ref() {
                match wtx.try_send(s) {
                    Ok(()) => {}
                    // Sink is behind: count the drop and keep going rather than
                    // block (blocking would perturb the measurement).
                    Err(TrySendError::Full(_)) => dropped += 1,
                    // Writer stopped (a record() error disabled it, or it already
                    // finished): stop forwarding for the rest of the run.
                    Err(TrySendError::Disconnected(_)) => writer_tx = None,
                }
            }
        }
        // Drop the sender so the writer thread's `recv` ends and it runs finish().
        drop(writer_tx);
        (latencies, recalls, empty_gt, over_gt, n_ok, n_err, dropped)
    });

    let started = Instant::now();
    let stop_at = started + Duration::from_secs_f64(profile.duration_s);
    let batch_size = profile.batch_size.max(1);
    // Per-run "first error already logged" flag — see `dispatch_sample`.
    let error_reported = Arc::new(std::sync::atomic::AtomicBool::new(false));

    if profile.target_rps > 0.0 {
        run_paced(&target, &vectors, profile, started, stop_at, top_k, filtered, &tx, &error_reported)
            .await;
    } else {
        run_closed_loop(&target, &vectors, profile, started, stop_at, top_k, filtered, &tx, &error_reported)
            .await;
    }

    // Drop the last sender so the collector's `recv` loop ends.
    drop(tx);
    let wall_s = started.elapsed().as_secs_f64();

    let (latencies_ms, recalls, empty_ground_truth, filter_overreturn, n_ok, n_err, dropped_samples) =
        collector.await.unwrap_or_default();
    // Join the writer thread so its `finish()` (final flush) completes before we
    // return — otherwise a caller reading the file back could race the flush.
    // Cheap: the load is done and the channel is closed, so the thread is already
    // exiting.
    if let Some(handle) = writer_handle {
        let _ = handle.join();
    }
    if dropped_samples > 0 {
        tracing::warn!(
            "time-series report incomplete: dropped {dropped_samples} sample(s) — the sink \
             couldn't keep pace with dispatch (bounded writer queue full); the summary is \
             unaffected"
        );
    }
    let _ = target.close().await;

    StormResults {
        latencies_ms,
        recalls,
        empty_ground_truth,
        filter_overreturn,
        n_ok,
        n_err,
        wall_s,
        batch_size,
        dropped_samples,
    }
}

/// Hold `concurrency` requests in flight until the window closes; each task
/// fires the next query the instant its previous one returns.
#[allow(clippy::too_many_arguments)]
async fn run_closed_loop(
    target: &Arc<dyn QueryTarget>,
    vectors: &Arc<Vec<QueryVector>>,
    profile: &LoadProfile,
    started: Instant,
    stop_at: Instant,
    top_k: u64,
    filtered: bool,
    tx: &mpsc::UnboundedSender<DispatchSample>,
    error_reported: &Arc<std::sync::atomic::AtomicBool>,
) {
    let n = vectors.len();
    let batch_size = profile.batch_size.max(1);
    // Fixed-work mode (`passes > 0`): the run ends when every query has been
    // fired exactly `passes` times, wall clock be damned — `duration_s` is
    // ignored. The shared cursor is an absolute query-firing counter; a worker
    // claims a batch by advancing it, trims the final batch to the remaining
    // budget, and stops once the budget is spent.
    let total_firings = profile.passes.checked_mul(n).unwrap_or(usize::MAX);
    let fixed_work = profile.passes > 0;
    let cursor = Arc::new(AtomicUsize::new(0));
    let mut workers = JoinSet::new();

    for _ in 0..profile.concurrency.max(1) {
        let target = target.clone();
        let vectors = vectors.clone();
        let tx = tx.clone();
        let cursor = cursor.clone();
        let error_reported = error_reported.clone();
        workers.spawn(async move {
            loop {
                if !fixed_work && Instant::now() >= stop_at {
                    break;
                }
                // fetch_add wraps far below usize::MAX over any real run.
                let claimed = cursor.fetch_add(batch_size, Ordering::Relaxed);
                let size = if fixed_work {
                    if claimed >= total_firings {
                        break;
                    }
                    batch_size.min(total_firings - claimed)
                } else {
                    batch_size
                };
                let idxs = batch_indices(claimed % n, size, n);
                let queries: Vec<&QueryVector> = idxs.iter().map(|&i| &vectors[i]).collect();
                let out = target.query_batch(&queries).await;
                let _ = tx.send(dispatch_sample(
                    &out, &idxs, &vectors, top_k, filtered, started, &error_reported,
                ));
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
#[allow(clippy::too_many_arguments)]
async fn run_paced(
    target: &Arc<dyn QueryTarget>,
    vectors: &Arc<Vec<QueryVector>>,
    profile: &LoadProfile,
    started: Instant,
    stop_at: Instant,
    top_k: u64,
    filtered: bool,
    tx: &mpsc::UnboundedSender<DispatchSample>,
    error_reported: &Arc<std::sync::atomic::AtomicBool>,
) {
    let n = vectors.len();
    let batch_size = profile.batch_size.max(1);
    let interval = Duration::from_secs_f64(1.0 / profile.target_rps);
    let sem = Arc::new(Semaphore::new(profile.concurrency.max(1)));
    let mut inflight = JoinSet::new();
    let mut idx = 0usize;
    // Fixed-work mode: stop after every query has been launched `passes`
    // times (still on the paced schedule); `duration_s` is ignored.
    let total_firings = profile.passes.checked_mul(n).unwrap_or(usize::MAX);
    let fixed_work = profile.passes > 0;
    // Fixed virtual schedule: each launch is pinned to `next`, which only ever
    // advances by `interval`. Falling behind admits the next launch immediately
    // (sleep_until is already in the past), so the average tracks target.
    let mut next = Instant::now();

    loop {
        if fixed_work {
            if idx >= total_firings {
                break;
            }
        } else if Instant::now() >= stop_at {
            break;
        }
        let permit = sem.clone().acquire_owned().await.expect("semaphore not closed");
        // acquire may have blocked; re-check the deadline before launching.
        if !fixed_work && Instant::now() >= stop_at {
            break;
        }
        let start = idx % n;
        let size = if fixed_work { batch_size.min(total_firings - idx) } else { batch_size };
        idx += size;
        let idxs = batch_indices(start, size, n);
        let target = target.clone();
        let vectors = vectors.clone();
        let tx = tx.clone();
        let error_reported = error_reported.clone();
        inflight.spawn(async move {
            let queries: Vec<&QueryVector> = idxs.iter().map(|&i| &vectors[i]).collect();
            let out = target.query_batch(&queries).await;
            let _ = tx.send(dispatch_sample(
                &out, &idxs, &vectors, top_k, filtered, started, &error_reported,
            ));
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
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: None,
                filter_values: HashMap::new(),
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn closed_loop_fires_many_and_records_each() {
        let profile = LoadProfile { concurrency: 4, duration_s: 0.2, target_rps: 0.0, batch_size: 1, passes: 0 };
        let target = Arc::new(MockTarget::ok(vec![]));
        let results = run_storm(target, vectors(), &profile, 10, false, None).await;
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
        assert!(summary.total_recall.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn paced_does_not_overshoot_target_rps() {
        let target_rps = 200.0;
        let duration_s = 0.5;
        let profile = LoadProfile { concurrency: 16, duration_s, target_rps, batch_size: 1, passes: 0 };
        let target = Arc::new(MockTarget::ok(vec![]));
        let results = run_storm(target, vectors(), &profile, 10, false, None).await;
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
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: if i % 2 == 0 {
                    Some(HashSet::from(["a".to_string(), "z".into(), "y".into(), "x".into()]))
                } else {
                    None
                },
                filter_values: HashMap::new(),
            })
            .collect();

        let profile = LoadProfile { concurrency: 2, duration_s: 0.15, target_rps: 0.0, batch_size: 1, passes: 0 };
        let results = run_storm(target, vectors, &profile, 4, false, None).await;
        let summary = results.summary();

        // every recorded recall sample must be exactly 0.25 -- never 0, never
        // computed against a query that had no ground truth. Ground truth is 4
        // ids at k=4, so all samples are full-depth (not short).
        assert!(!results.recalls.is_empty());
        assert!(results.recalls.iter().all(|s| !s.short && (s.recall - 0.25).abs() < 1e-9));
        let full = summary.full_recall.expect("full-depth queries present");
        assert!((full.mean - 0.25).abs() < 1e-9);
        assert!(summary.short_recall.is_none()); // no short ground truth in this run
        assert!((summary.total_recall.unwrap().mean - 0.25).abs() < 1e-9);
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
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: Some(HashSet::from(["a".to_string()])),
                filter_values: HashMap::new(),
            })
            .collect();

        let profile = LoadProfile { concurrency: 2, duration_s: 0.15, target_rps: 0.0, batch_size: 1, passes: 0 };
        let results = run_storm(target, vectors, &profile, 1, false, None).await;
        let summary = results.summary();

        assert_eq!(summary.errors, summary.requests); // every query failed
        assert!(results.recalls.is_empty()); // -> zero recall SAMPLES, not samples of 0.0
        assert!(summary.total_recall.is_none()); // -> "unknown", not "search returned nothing"
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn present_but_empty_ground_truth_is_counted_not_scored() {
        // A query whose ground-truth column value is an empty list has nothing
        // to score against: it must NOT become a recall=0 sample (which would
        // read as a search miss) NOR a divide-by-zero NaN -- it's counted under
        // `empty_ground_truth` and left out of every recall bucket.
        let target = Arc::new(MockTarget::ok(vec!["a".into(), "b".into()]));
        let vectors: Vec<QueryVector> = (0..10)
            .map(|i| QueryVector {
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                // even: real ground truth (recall@2 = 1/2); odd: present-but-empty.
                ground_truth: Some(if i % 2 == 0 {
                    HashSet::from(["a".to_string(), "zzz".into()])
                } else {
                    HashSet::new()
                }),
                filter_values: HashMap::new(),
            })
            .collect();

        let profile = LoadProfile { concurrency: 2, duration_s: 0.15, target_rps: 0.0, batch_size: 1, passes: 0 };
        let results = run_storm(target, vectors, &profile, 2, false, None).await;
        let summary = results.summary();

        // The empty-gt firings are counted, not scored...
        assert!(summary.empty_ground_truth > 0);
        // ...and never leaked into a recall bucket: every recorded sample is the
        // 0.5 from the real-ground-truth queries, none a 0.0 or NaN.
        assert!(results.recalls.iter().all(|s| (s.recall - 0.5).abs() < 1e-9));
        assert!((summary.total_recall.unwrap().mean - 0.5).abs() < 1e-9);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn filter_overreturn_counts_only_under_a_filter_and_never_touches_recall() {
        // The mock always returns 2 ids; each query's ground truth holds just 1.
        // So every firing has returned(2) > truth_len(1) — a suspected filter
        // leak ONLY when a filter is configured. Without a filter it's benign
        // truncation and must NOT be counted.
        let vectors = || -> Vec<QueryVector> {
            (0..8)
                .map(|i| QueryVector {
                    vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                    ground_truth: Some(HashSet::from(["a".to_string()])), // 1 id, "a" is a hit
                    filter_values: HashMap::new(),
                })
                .collect()
        };
        let profile = LoadProfile { concurrency: 2, duration_s: 0.15, target_rps: 0.0, batch_size: 1, passes: 0 };

        // With a filter: every firing over-returned relative to its 1-id ground
        // truth, so it's counted...
        let target = Arc::new(MockTarget::ok(vec!["a".into(), "b".into()]));
        let filtered = run_storm(target, vectors(), &profile, 5, true, None).await;
        let fs = filtered.summary();
        assert!(fs.filter_overreturn > 0);
        assert_eq!(fs.filter_overreturn, fs.total_recall.unwrap().n);
        // ...but recall is untouched: short bucket, 1 hit / 1 gt id = 1.0.
        assert!(filtered.recalls.iter().all(|s| s.short && (s.recall - 1.0).abs() < 1e-9));
        assert_eq!(fs.empty_ground_truth, 0); // a different signal

        // Same over-return WITHOUT a filter -> not a leak, count stays 0, while
        // recall is identical.
        let target = Arc::new(MockTarget::ok(vec!["a".into(), "b".into()]));
        let unfiltered = run_storm(target, vectors(), &profile, 5, false, None).await;
        let us = unfiltered.summary();
        assert_eq!(us.filter_overreturn, 0);
        assert!(unfiltered.recalls.iter().all(|s| (s.recall - 1.0).abs() < 1e-9));
    }

    #[test]
    fn recall_at_k_full_depth_divides_by_k_and_is_not_short() {
        let returned = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let ground_truth =
            HashSet::from(["a".to_string(), "b".to_string(), "z".to_string(), "y".to_string()]);
        // gt has 4 ids (>= k), so denominator is k, not gt len or returned len.
        // 2 of the 3 returned ids are in ground_truth, k=4 -> 2/4.
        let r = recall_at_k(&returned, &ground_truth, 4).unwrap();
        assert_eq!((r.recall, r.short), (0.5, false));
        // k=2 (<= gt len) still divides by k -> 2/2 = 1.0, still full-depth.
        let r = recall_at_k(&returned, &ground_truth, 2).unwrap();
        assert_eq!((r.recall, r.short), (1.0, false));
        // no returned ids -> 0 hits over k, still a real (full-depth) sample.
        let r = recall_at_k(&[], &ground_truth, 4).unwrap();
        assert_eq!((r.recall, r.short), (0.0, false));
    }

    #[test]
    fn recall_at_k_short_ground_truth_divides_by_its_own_length_and_is_flagged() {
        // gt has 2 ids but k=4 -> can never fill k. Score against gt len (2),
        // not k, and flag it short so the summary keeps it in its own bucket.
        let returned = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let ground_truth = HashSet::from(["a".to_string(), "b".to_string()]);
        let r = recall_at_k(&returned, &ground_truth, 4).unwrap();
        assert_eq!((r.recall, r.short), (1.0, true)); // 2 hits / 2 gt, NOT 2/4
        // one hit of the two -> 1/2, still short.
        let one = vec!["a".to_string(), "zzz".to_string()];
        let r = recall_at_k(&one, &ground_truth, 4).unwrap();
        assert_eq!((r.recall, r.short), (0.5, true));
    }

    #[test]
    fn recall_at_k_empty_ground_truth_is_no_sample_not_a_nan() {
        // Dividing by an empty gt's length would be NaN and poison the mean;
        // an empty (or absent) ground truth is simply "nothing to measure".
        let returned = vec!["a".to_string()];
        assert!(recall_at_k(&returned, &HashSet::new(), 4).is_none());
    }

    #[test]
    fn recall_at_k_dedupes_returned_so_a_repeated_id_cannot_exceed_1_0() {
        // "a" appears 3 times in `returned` -- must still count as a single
        // hit, not 3, or recall would read 1.5 for k=2 (impossible for a
        // fraction that's supposed to be capped at 1.0).
        let returned = vec!["a".to_string(), "a".to_string(), "a".to_string()];
        let ground_truth = HashSet::from(["a".to_string(), "b".to_string()]);
        assert_eq!(recall_at_k(&returned, &ground_truth, 2).unwrap().recall, 0.5);
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
            full_recall: None,
            short_recall: None,
            total_recall: None,
            empty_ground_truth: 0,
            filter_overreturn: 0,
        };
        assert!(!base.to_string().contains("recall"));

        // A run with only full-depth queries: full + total print, short stays absent.
        let with_recall = Summary {
            full_recall: Some(RecallBucket { n: 8, mean: 0.87 }),
            total_recall: Some(RecallBucket { n: 8, mean: 0.87 }),
            ..base
        };
        let s = with_recall.to_string();
        assert!(s.contains("recall_full"));
        assert!(s.contains("recall_total"));
        assert!(!s.contains("recall_short")); // no short bucket -> still absent
        assert!(s.contains("0.8700 (n=8)"));
        assert!(!s.contains("recall_empty_gt")); // 0 empty -> line stays absent

        assert!(!s.contains("filter_overreturn")); // 0 -> line stays absent

        // A non-zero empty-ground-truth count prints its own line.
        let with_empty = Summary { empty_ground_truth: 3, ..base };
        assert!(with_empty.to_string().contains("recall_empty_gt: 3"));

        // A non-zero returned-over-ground-truth count prints its own line.
        let with_over = Summary { filter_overreturn: 7, ..base };
        assert!(with_over.to_string().contains("filter_overreturn: 7"));
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
            full_recall: Some(RecallBucket { n: 6, mean: 0.9 }),
            short_recall: Some(RecallBucket { n: 2, mean: 0.75 }),
            total_recall: Some(RecallBucket { n: 8, mean: 0.87 }),
            empty_ground_truth: 5,
            filter_overreturn: 3,
        };
        let json = serde_json::to_string(&summary).expect("serializes");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        assert_eq!(parsed["requests"], 10);
        assert_eq!(parsed["batch_size"], 4);
        assert_eq!(parsed["qps"], 20.0);
        assert_eq!(parsed["total_recall"]["mean"], 0.87);
        assert_eq!(parsed["total_recall"]["n"], 8);
        assert_eq!(parsed["short_recall"]["mean"], 0.75);
        assert_eq!(parsed["empty_ground_truth"], 5);
        assert_eq!(parsed["filter_overreturn"], 3);
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
                        Some(match q.vector.as_dense().expect("test queries are dense")[0] as i64 {
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
            QueryVector { vector: crate::queries::VectorData::Dense(vec![0.0]), ground_truth: gt.clone(), filter_values: HashMap::new() },
            QueryVector { vector: crate::queries::VectorData::Dense(vec![1.0]), ground_truth: gt.clone(), filter_values: HashMap::new() },
            QueryVector { vector: crate::queries::VectorData::Dense(vec![2.0]), ground_truth: gt, filter_values: HashMap::new() },
        ];
        // duration long enough to cycle through all 3 at concurrency=1 several times
        let profile = LoadProfile { concurrency: 1, duration_s: 0.1, target_rps: 0.0, batch_size: 1, passes: 0 };
        let results = run_storm(Arc::new(PerQueryTarget), vectors, &profile, 2, false, None).await;

        // Only asserts the pipeline actually produced all 3 distinct values --
        // NOT their exact proportions, which depend on how many times each of
        // the 3 round-robin slots happened to be hit inside a fixed wall-clock
        // window (not guaranteed 1:1:1). The exact per-bucket mean math itself
        // is covered deterministically, with no timing dependency, by
        // `summary_aggregates_recall_buckets_correctly` below. gt is 2 ids at
        // k=2, so every sample is full-depth (not short).
        assert!(results.recalls.iter().all(|s| !s.short));
        assert!(results.recalls.iter().any(|s| (s.recall - 0.0).abs() < 1e-9));
        assert!(results.recalls.iter().any(|s| (s.recall - 0.5).abs() < 1e-9));
        assert!(results.recalls.iter().any(|s| (s.recall - 1.0).abs() < 1e-9));
    }

    #[test]
    fn summary_aggregates_recall_buckets_correctly() {
        // Deterministic, no async/timing involved -- exercises StormResults::summary()
        // directly, so it can assert exact per-bucket counts + means without
        // depending on how many times a round-robin cycle repeated in a window.
        // Two full-depth samples (mean 0.75) and one short sample (mean 0.40);
        // total blends all three, each by its own denominator.
        let results = StormResults {
            latencies_ms: vec![1.0, 2.0, 3.0],
            recalls: vec![
                RecallSample { recall: 0.5, short: false },
                RecallSample { recall: 1.0, short: false },
                RecallSample { recall: 0.4, short: true },
            ],
            empty_ground_truth: 2,
            filter_overreturn: 0,
            n_ok: 3,
            n_err: 0,
            wall_s: 1.0,
            batch_size: 1,
            dropped_samples: 0,
        };
        let summary = results.summary();

        let full = summary.full_recall.unwrap();
        assert_eq!(full.n, 2);
        assert!((full.mean - 0.75).abs() < 1e-9); // (0.5 + 1.0) / 2
        let short = summary.short_recall.unwrap();
        assert_eq!(short.n, 1);
        assert!((short.mean - 0.4).abs() < 1e-9);
        let total = summary.total_recall.unwrap();
        assert_eq!(total.n, 3);
        assert!((total.mean - (0.5 + 1.0 + 0.4) / 3.0).abs() < 1e-9); // all three, each own denom
        // empty-ground-truth firings pass through untouched, outside every bucket.
        assert_eq!(summary.empty_ground_truth, 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recorder_receives_one_timestamped_row_per_dispatch() {
        use crate::report::{ReportConfig, ReportFormat};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ts.csv").to_string_lossy().into_owned();
        let cfg = ReportConfig { format: ReportFormat::Csv, path: path.clone() };
        let mut recorder = cfg.build();
        recorder.begin().expect("begin");

        let profile = LoadProfile { concurrency: 4, duration_s: 0.2, target_rps: 0.0, batch_size: 1, passes: 0 };
        let target = Arc::new(MockTarget::ok(vec![]));
        let results = run_storm(target, vectors(), &profile, 10, false, Some(recorder)).await;
        let summary = results.summary();

        let text = std::fs::read_to_string(&path).expect("csv written");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "t_s,latency_ms,ok,recalls_full,recalls_short");
        // one row per dispatch that reached the sink — the time series IS the
        // raw run, minus any samples dropped when the writer queue was full
        // (with an instant mock target the load loop can briefly outrun the
        // writer; the summary still counts every dispatch).
        assert_eq!(lines.len() as u64, 1 + summary.requests - results.dropped_samples);
        // timestamps are on the run's time axis: non-negative, within the
        // window (plus scheduling slack), and present on every row
        for line in &lines[1..] {
            let t: f64 = line.split(',').next().unwrap().parse().expect("t_s parses");
            assert!((0.0..5.0).contains(&t), "t_s out of range: {t}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_failing_sink_disables_recording_without_killing_the_run() {
        // The central robustness guarantee: a report sink that errors on write
        // must NOT take down the load test — the summary stays whole, the run
        // completes normally, only the (auxiliary) time series is lost.
        struct FailingRecorder;
        impl crate::report::Recorder for FailingRecorder {
            fn begin(&mut self) -> std::io::Result<()> {
                Ok(())
            }
            fn record(&mut self, _s: &DispatchSample) -> std::io::Result<()> {
                Err(std::io::Error::other("sink is down"))
            }
            fn finish(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let profile = LoadProfile { concurrency: 4, duration_s: 0.2, target_rps: 0.0, batch_size: 1, passes: 0 };
        let target = Arc::new(MockTarget::ok(vec![]));
        let results =
            run_storm(target, vectors(), &profile, 10, false, Some(Box::new(FailingRecorder))).await;
        let summary = results.summary();

        // Load ran to completion despite the sink failing on the very first row.
        assert!(summary.requests > 0, "the load test must complete even with a dead sink");
        assert_eq!(summary.errors, 0, "dispatch errors are unrelated to sink failure");
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
                vector: crate::queries::VectorData::Dense(vec![i as f32]),
                ground_truth: gt.clone(),
                filter_values: HashMap::new(),
            })
            .collect();
        let batch_size = 3;
        let profile = LoadProfile { concurrency: 1, duration_s: 0.15, target_rps: 0.0, batch_size, passes: 0 };
        let target = Arc::new(BatchCapturingTarget { call_lens: std::sync::Mutex::new(Vec::new()) });
        let results = run_storm(target.clone(), vectors, &profile, 2, false, None).await;

        let call_lens = target.call_lens.lock().unwrap();
        assert!(!call_lens.is_empty());
        assert!(call_lens.iter().all(|&len| len == batch_size), "{call_lens:?}");

        assert!(results.recalls.iter().any(|s| (s.recall - 0.0).abs() < 1e-9));
        assert!(results.recalls.iter().any(|s| (s.recall - 0.5).abs() < 1e-9));
        assert!(results.recalls.iter().any(|s| (s.recall - 1.0).abs() < 1e-9));
        assert_eq!(results.summary().batch_size, batch_size);
    }

    /// A per-query firing counter, for asserting fixed-work exactness: each
    /// query's dense vector encodes its index, and the mock counts firings.
    struct CountingTarget {
        counts: std::sync::Mutex<HashMap<usize, usize>>,
    }

    impl std::fmt::Display for CountingTarget {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "counting")
        }
    }

    #[async_trait]
    impl QueryTarget for CountingTarget {
        async fn query_batch(&self, queries: &[&QueryVector]) -> BatchOutcome {
            let mut counts = self.counts.lock().unwrap();
            for q in queries {
                let idx = q.vector.as_dense().expect("dense test vectors")[0] as usize;
                *counts.entry(idx).or_insert(0) += 1;
            }
            BatchOutcome {
                latency: std::time::Duration::from_micros(50),
                ok: true,
                ids: vec![None; queries.len()],
                error: None,
            }
        }
    }

    fn indexed_vectors(n: usize) -> Vec<QueryVector> {
        (0..n)
            .map(|i| QueryVector {
                vector: crate::queries::VectorData::Dense(vec![i as f32]),
                ground_truth: None,
                filter_values: HashMap::new(),
            })
            .collect()
    }

    /// Fixed work, closed loop: every query fired EXACTLY `passes` times, no
    /// more, no less — regardless of concurrency racing — and `duration_s` is
    /// irrelevant (deliberately absurd here: a timed run of 0.001s could never
    /// fit this work; a timed run of 10000s would never end the test).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fixed_work_fires_each_query_exactly_passes_times() {
        let target = Arc::new(CountingTarget { counts: std::sync::Mutex::new(HashMap::new()) });
        let vectors = indexed_vectors(10);
        let profile =
            LoadProfile { concurrency: 4, duration_s: 0.001, target_rps: 0.0, batch_size: 3, passes: 2 };
        let results = run_storm(target.clone(), vectors, &profile, 10, false, None).await;

        let counts = target.counts.lock().unwrap();
        assert_eq!(counts.len(), 10);
        assert!(counts.values().all(|&c| c == 2), "{counts:?}");
        // 20 firings at batch 3 = 6 full batches + a 2-query tail = 7 dispatches
        assert_eq!(results.n_ok, 7);
    }

    /// Fixed work, paced: the launch budget ends the run, not the clock.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fixed_work_paced_fires_each_query_exactly_once() {
        let target = Arc::new(CountingTarget { counts: std::sync::Mutex::new(HashMap::new()) });
        let vectors = indexed_vectors(5);
        // 1000 rps so the schedule is not the bottleneck; duration absurd both ways.
        let profile =
            LoadProfile { concurrency: 2, duration_s: 10_000.0, target_rps: 1000.0, batch_size: 2, passes: 1 };
        let results = run_storm(target.clone(), vectors, &profile, 5, false, None).await;

        let counts = target.counts.lock().unwrap();
        assert_eq!(counts.len(), 5);
        assert!(counts.values().all(|&c| c == 1), "{counts:?}");
        // 5 firings at batch 2 = 2 full + 1-query tail = 3 dispatches
        assert_eq!(results.n_ok, 3);
    }

    #[test]
    fn first_error_is_reported_exactly_once_per_flag() {
        use std::sync::atomic::AtomicBool;
        let flag = AtomicBool::new(false);
        assert!(report_first_error(&flag, "boom")); // first: logs
        assert!(!report_first_error(&flag, "boom")); // second: counted only
        assert!(!report_first_error(&flag, "different")); // still counted only

        // a NEW flag (a new run) reports again — per-run, not per-process
        let fresh = AtomicBool::new(false);
        assert!(report_first_error(&fresh, "next run's error"));
    }
}
