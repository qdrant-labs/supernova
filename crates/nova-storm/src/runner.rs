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
use crate::queries::{GtCutoff, QueryVector, scores_tied};
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
    /// Total firings where the VDB returned fewer than `top_k` ids.
    pub short_returns: u64,
    /// Total results scoring better than the ground truth's k-th place yet
    /// absent from it — a ground-truth/collection mismatch, not a tie.
    pub missing_from_gt: u64,
    /// Distinct queries behind the `recall@k` line — non-empty ground truth,
    /// at least `top_k` deep. Not the loaded query count.
    pub full_recall_queries: u64,
    /// Ground-truth tie stats at the top-k cutoff, when scores were available.
    pub ties: Option<TieStats>,
    pub top_k: u64,
    pub tie_epsilon: f64,
    pub tie_epsilon_source: String,
    /// `Some(reason)` when tie reporting was withheld — see `run_storm`.
    pub tie_disabled_reason: Option<String>,
    /// Whether a ground-truth score column was configured at all. Distinct
    /// from `tie_disabled_reason`: a run that never asked for tie reporting
    /// isn't a run where it was refused, so it gets no banner — but it must
    /// still not emit a tolerance that was never applied.
    pub scores_configured: bool,
    /// Count of batch dispatches, not individual queries.
    pub n_ok: u64,
    pub n_err: u64,
    /// Of `n_err`, how many were CLIENT-side deadline expiries (`timeout_s`).
    pub n_timeout: u64,
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
    /// Of `errors`, how many were timeouts — the client's `timeout_s` expiry
    /// (gRPC CANCELLED/DEADLINE_EXCEEDED) or the server's own search timeout
    /// (qdrant: INTERNAL wrapping "timed out after"). "Too slow" — a
    /// saturation signal — as opposed to "broken". Any cell with
    /// `timeouts > 0` also has censored tail latency: the timed-out
    /// dispatches contribute samples at (or, for server cuts, near) the
    /// deadline instead of their honest duration.
    pub timeouts: u64,
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
    /// Tie-tolerant mean recall over the same firings as `full_recall` — the
    /// UPPER bound (see [`RecallSample::tolerant`]). Equals `full_recall` when
    /// no score column is configured or nothing ties.
    pub full_recall_tolerant: Option<f64>,
    /// Tie-tolerant mean over the SHORT bucket's firings — the same upper
    /// bound as `full_recall_tolerant`, for queries whose ground truth is
    /// shallower than `top_k`. Ties are not a full-bucket phenomenon: under a
    /// selective filter a shallow ground truth is the norm, and those are
    /// exactly the queries whose cutoff is most likely to be tied.
    pub short_recall_tolerant: Option<f64>,
    /// Tie-tolerant mean over both buckets, matching `total_recall`.
    pub total_recall_tolerant: Option<f64>,
    /// Firings where the engine returned fewer than `top_k` ids, excluding
    /// queries whose ground truth is present but empty. Deflates recall for
    /// full-depth queries, whose denominator stays `top_k`.
    pub short_returns: u64,
    /// Results scoring better than the ground truth's k-th place yet absent
    /// from it. Non-zero means the ground truth and the collection disagree.
    pub missing_from_gt: u64,
    /// Queries ELIGIBLE for the `recall@k` line — loaded with a non-empty
    /// ground truth at least `top_k` deep. Not the number of vectors loaded
    /// (empty and shallow ground truths are excluded; a latency-only run
    /// reports 0), and not necessarily the number that fired: a short or paced
    /// run can stop before cycling through them all, in which case the mean
    /// covers a subset of this. The buckets' `n` counts FIRINGS.
    pub full_recall_queries: u64,
    pub ties: Option<TieStats>,
    pub top_k: u64,
    /// `None` when tie reporting was disabled — the tolerance was never
    /// applied, so reporting it beside `full_recall_tolerant: null` would
    /// suggest a comparison that did not happen.
    pub tie_epsilon: Option<f64>,
    pub tie_epsilon_source: Option<String>,
    /// `Some(reason)` when every tie-derived field was withheld because
    /// returned scores are not comparable to the ground truth's — a quantized
    /// collection queried with `rescore: false` (measured 3.6e-02 to 26.4
    /// relative error, vs 2.4e-07 with rescoring on), or a distance function
    /// nova-bf stores with the opposite sign. `full_recall_tolerant` and
    /// `ties` are then `None` and `missing_from_gt` is 0. Exact recall is
    /// unaffected.
    pub tie_disabled_reason: Option<String>,
    /// Bumped when a field's MEANING changes without its type changing —
    /// `nova sweep` cannot otherwise tell a pre-truncation run's recall from a
    /// post-truncation one. 2 = ground truth truncated to `top_k`.
    pub schema_version: u32,
}

/// How tied the ground truth is AT the top-k cutoff, across the loaded query
/// set. Ties there are why recall is a range: several documents are equally
/// correct, so an engine returning a different one is not wrong.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct TieStats {
    pub mean: f64,
    pub max: u32,
    /// Share of queries whose k-th place is tied with at least one other doc.
    pub fraction_of_queries: f64,
    /// How many queries this describes — every one with a non-empty ground
    /// truth and a derived cutoff, in EITHER recall bucket. Carried explicitly
    /// so the line can state its own denominator instead of borrowing the one
    /// on the recall line above it, which counts a different population.
    pub queries: u64,
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
        (!recalls.is_empty()).then(|| RecallBucket {
            n: recalls.len() as u64,
            mean: recalls.iter().sum::<f64>() / recalls.len() as f64,
        })
    }
}

/// How a run compares engine scores against its ground truth's. Bundled
/// rather than passed as loose `bool`s: three adjacent booleans in an
/// eleven-argument call transpose silently, and swapping `higher_is_better`
/// with `filtered` would invert every score comparison — the exact failure the
/// orientation handling exists to prevent — while still compiling.
#[derive(Debug, Clone)]
pub struct ScoreComparison {
    /// Relative tolerance within which two scores count as the same.
    pub epsilon: f64,
    /// How that tolerance was chosen, for the summary.
    pub epsilon_source: String,
    /// `Some(reason)` when scores are not comparable at all and every
    /// tie-derived field must be withheld.
    pub disabled_reason: Option<String>,
    /// Whether a ground-truth score column AND id column were both configured.
    pub configured: bool,
    /// False for euclid/manhattan, where the engine returns a raw distance.
    pub engine_higher_is_better: bool,
}

impl ScoreComparison {
    /// Whether tie-derived numbers should be reported at all: they must have
    /// been asked for, and not refused.
    pub fn reported(&self) -> bool {
        self.configured && self.disabled_reason.is_none()
    }
}

/// One query's recall observation, tagged with whether its ground truth held
/// fewer ids than `top_k`. `short` queries divide by their own ground-truth
/// length rather than `top_k` (see [`recall_at_k`]); the flag is what lets the
/// summary and the time-series report keep the two populations apart.
#[derive(Debug, Clone, Copy)]
pub struct RecallSample {
    /// Exact id-set recall — the LOWER bound. Every id counted here is
    /// unambiguously correct.
    pub recall: f64,
    /// Tie-tolerant recall — the UPPER bound. Adds results that aren't in the
    /// ground truth but scored the same as its k-th place: the ground truth
    /// picked one member of a tie, the engine picked another, and both are
    /// equally right. Equals `recall` when no score column is configured or
    /// nothing ties.
    pub tolerant: f64,
    pub short: bool,
    /// Results that scored BETTER than the ground truth's k-th place yet are
    /// absent from it. Not a tie — the ground truth and the collection
    /// disagree (stale GT, wrong corpus version). Counted, never folded into
    /// `tolerant`, which would hide it as a recall gain.
    pub missing_from_gt: u32,
}

impl StormResults {
    pub fn summary(&self) -> Summary {
        let mut ms = self.latencies_ms.clone();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Split the per-query recall samples into the full-depth and short
        // buckets; `total` scores every ground-truthed query together.
        let full: Vec<f64> = self
            .recalls
            .iter()
            .filter(|s| !s.short)
            .map(|s| s.recall)
            .collect();
        let short: Vec<f64> = self
            .recalls
            .iter()
            .filter(|s| s.short)
            .map(|s| s.recall)
            .collect();
        let all: Vec<f64> = self.recalls.iter().map(|s| s.recall).collect();
        // The tie-tolerant mean covers the SAME firings as `full`, so the two
        // are directly comparable as a lower/upper pair.
        // Reporting requires that ties were both ASKED for and possible.
        let ties_reported = self.scores_configured && self.tie_disabled_reason.is_none();
        let full_tol: Vec<f64> = self
            .recalls
            .iter()
            .filter(|s| !s.short)
            .map(|s| s.tolerant)
            .collect();
        let short_tol: Vec<f64> =
            self.recalls.iter().filter(|s| s.short).map(|s| s.tolerant).collect();
        let all_tol: Vec<f64> = self.recalls.iter().map(|s| s.tolerant).collect();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let total = self.n_ok + self.n_err;
        let requests_per_sec = if self.wall_s > 0.0 {
            total as f64 / self.wall_s
        } else {
            0.0
        };
        Summary {
            requests: total,
            errors: self.n_err,
            timeouts: self.n_timeout,
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
            full_recall_tolerant: (ties_reported && !full_tol.is_empty())
                .then(|| mean(&full_tol)),
            short_recall_tolerant: (ties_reported && !short_tol.is_empty())
                .then(|| mean(&short_tol)),
            total_recall_tolerant: (ties_reported && !all_tol.is_empty())
                .then(|| mean(&all_tol)),
            short_returns: self.short_returns,
            // Withheld with the rest when scores are incomparable: it is the
            // loud "stale ground truth" alarm, and a number derived from
            // incomparable scores is exactly what must not fire it.
            missing_from_gt: if ties_reported {
                self.missing_from_gt
            } else {
                0
            },
            full_recall_queries: self.full_recall_queries,
            ties: ties_reported.then_some(self.ties).flatten(),
            top_k: self.top_k,
            tie_epsilon: ties_reported.then_some(self.tie_epsilon),
            tie_epsilon_source: ties_reported.then(|| self.tie_epsilon_source.clone()),
            tie_disabled_reason: self.tie_disabled_reason.clone(),
            schema_version: 2,
        }
    }
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut lines = vec![
            format!("{:>16}: {}", "requests", self.requests),
            format!("{:>16}: {}", "errors", self.errors),
            format!("{:>16}: {}", "batch_size", self.batch_size),
        ];
        if self.timeouts > 0 {
            // Inserted right after the counts: a timing-out cell is "too slow
            // for timeout_s", not "broken", and its tail latency below is
            // censored at the timeout value.
            lines.insert(
                2,
                format!(
                    "{:>16}: {} (client timeout_s or server search timeout; tail latency censored)",
                    "timeouts", self.timeouts
                ),
            );
        }
        lines.extend([
            format!("{:>16}: {:.1}", "requests_per_sec", self.requests_per_sec),
            format!("{:>16}: {:.1}", "qps", self.qps),
            format!("{:>16}: {:.2}", "p50_ms", self.p50_ms),
            format!("{:>16}: {:.2}", "p95_ms", self.p95_ms),
            format!("{:>16}: {:.2}", "p99_ms", self.p99_ms),
            format!("{:>16}: {:.2}", "max_ms", self.max_ms),
        ]);
        // Recall is reported at the depth it was measured at, and as a RANGE
        // when ties make the exact value genuinely ambiguous: the ground truth
        // recorded one member of a tied k-th place, the engine may return
        // another, and both are correct. Lower = exact id match, upper = tied
        // scores count. They collapse to one number when nothing ties, which
        // is also what a run without a score column reports.
        let label = format!("recall@{}", self.top_k);
        // A bucket prints as a RANGE when its tie-tolerant bound is meaningfully
        // above the exact one. 1e-4 is the resolution both endpoints print at,
        // so a smaller gap would render as `0.8631 – 0.8631`.
        let value = |exact: f64, tolerant: Option<f64>| match tolerant {
            Some(t) if t >= exact + 1e-4 => format!("{exact:.4} – {t:.4}"),
            _ => format!("{exact:.4}"),
        };
        if let Some(b) = self.full_recall {
            // "eligible", not "queries": a paced or short run may not have
            // cycled through all of them, and claiming the mean covers every
            // one would overstate it.
            lines.push(format!(
                "{:>16}: {}  ({} eligible queries, {} firings)",
                label,
                value(b.mean, self.full_recall_tolerant),
                self.full_recall_queries,
                b.n
            ));
        }
        if let Some(b) = self.short_recall {
            lines.push(format!(
                "{:>16}: {} ({} firings whose ground truth held <{} ids)",
                format!("{label}_short"),
                value(b.mean, self.short_recall_tolerant),
                b.n,
                self.top_k
            ));
        }
        // Only when both buckets exist: otherwise it just repeats the one above.
        // Both buckets, or it just repeats whichever single line printed above.
        if let (Some(t), true, true) = (
            self.total_recall,
            self.short_recall.is_some(),
            self.full_recall.is_some(),
        ) {
            lines.push(format!(
                "{:>16}: {} (n={})",
                "recall_total",
                value(t.mean, self.total_recall_tolerant),
                t.n
            ));
        }
        // The ties that make the range a range, and the tolerance that decided
        // what counts as tied — both only when scores were actually available.
        if let Some(reason) = &self.tie_disabled_reason {
            lines.push(format!(
                "{:>16}: disabled — {reason}. Exact recall above is unaffected.",
                "tie_reporting"
            ));
        }
        if let Some(t) = self.ties {
            lines.push(format!(
                "{:>16}: {:.1} avg, {} max — {:.1}% of {} queries with a cutoff",
                "ties_at_cutoff",
                t.mean,
                t.max,
                t.fraction_of_queries * 100.0,
                t.queries
            ));
        }
        // Outside the `ties` block: the tolerance was applied whenever tie
        // reporting ran, even on a run that happened to find none, and `--json`
        // reports it on exactly that condition. The two must not disagree.
        if let (Some(eps), Some(src)) = (self.tie_epsilon, self.tie_epsilon_source.as_ref()) {
            lines.push(format!("{:>16}: {:.1e} ({})", "tie_epsilon", eps, src));
        }
        // Alarms — shown only when they fire (a 0 is the norm and reads as noise).
        if self.missing_from_gt > 0 {
            lines.push(format!(
                "{:>16}: {}  (scored above their query's ground-truth cutoff yet absent \
                 from it — stale GT or wrong corpus)",
                "missing_from_gt", self.missing_from_gt
            ));
        }
        if self.short_returns > 0 {
            lines.push(format!(
                "{:>16}: {}  (returned fewer than the ground truth holds, or than \
                 top_k={} where it is deeper)",
                "short_returns", self.short_returns, self.top_k
            ));
        }
        // Only when it actually happened — a 0 here is the norm and would just
        // be noise next to the recall means.
        if self.empty_ground_truth > 0 {
            lines.push(format!(
                "{:>16}: {}",
                "recall_empty_gt", self.empty_ground_truth
            ));
        }
        // Suspected filter leaks — visibility only, recall above is unaffected.
        // Shown when it happened.
        if self.filter_overreturn > 0 {
            lines.push(format!(
                "{:>16}: {}",
                "filter_overreturn", self.filter_overreturn
            ));
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
fn recall_at_k(
    returned: &[String],
    scores: Option<&[f32]>,
    ground_truth: &HashSet<String>,
    gt_depth: usize,
    cutoff: Option<GtCutoff>,
    k: u64,
    tie_epsilon: f64,
    engine_higher_is_better: bool,
) -> Option<RecallSample> {
    let truth_len = ground_truth.len() as u64;
    if truth_len == 0 {
        return None;
    }
    // Depth is POSITIONAL: a repeated id shrinks the deduped set without making
    // the ground truth shallower, so classifying on the set length would move a
    // full-depth query into the forgiving `short` bucket. Bare `gt_depth`,
    // matching `run_storm`'s eligibility filter and the lib.rs preflight — the
    // loader guarantees `gt_depth >= truth_len`.
    //
    // The SHORT bucket then divides by the deduped count, since `hits` can
    // never exceed it and dividing by the positional length would put 1.0 out
    // of reach. The full bucket divides by `k` by design, so a full-depth
    // ground truth containing a repeat caps below 1.0 — that ground truth is
    // malformed, and the loader warns about it.
    let depth = gt_depth as u64;
    let short = depth < k;
    let denom = if short { truth_len } else { k };

    // Orientation is a property of the QUERY, not of any one result, so it is
    // resolved ONCE here rather than per returned id. `None` means "do not
    // compare scores at all": either none were collected, or the ground truth
    // is distance-valued against a larger-is-better engine, which no sign flip
    // reconciles (see below).
    let compare = match (scores, cutoff) {
        (Some(scores), Some(cutoff)) => {
            // Put both sides in larger-is-nearer orientation. The engine's raw
            // distance is negated at the comparison; the ground truth's cutoff
            // is negated when its own list ascends. With no ordering signal
            // (all-equal scores, or one hit under a selective filter) fall back
            // to the sign: a distance engine's ground truth holding a
            // non-negative score must be raw distances, since nova-bf stores
            // them negated.
            let gt_ascending = cutoff
                .ascending
                .unwrap_or(!engine_higher_is_better && cutoff.score > 0.0);
            // Negation recovers a NEGATED similarity or distance, which is
            // always signed opposite to the engine's convention. A ground truth
            // ascending through NON-NEGATIVE values against a larger-is-better
            // engine is neither — it is distance-valued, such as `1 - cos`, and
            // flipping it leaves the two sides a constant apart, firing
            // `missing_from_gt` on every result. `>= 0.0` rather than `> 0.0`
            // because such a list bottoms out AT zero over a near-duplicate
            // corpus; skipping an exactly-orthogonal negated similarity is the
            // harmless side of that ambiguity.
            if gt_ascending && engine_higher_is_better && cutoff.score >= 0.0 {
                None
            } else {
                let cutoff_score = if gt_ascending { -cutoff.score } else { cutoff.score };
                // Whether the orientation was READ from the data or guessed.
                // Treated asymmetrically below: a tie only widens a bound that
                // is already an upper bound, but `missing_from_gt` is a loud
                // "your ground truth is stale" claim and must not rest on a
                // guess.
                Some((scores, cutoff_score, cutoff.ascending.is_some()))
            }
        }
        _ => None,
    };

    // ONE pass over the response and ONE set. `seen` both deduplicates (a
    // target that repeats an id must not have it counted twice) and gates the
    // tie check, so this runs per query per firing at half the allocations and
    // half the walks it used to.
    let mut seen: HashSet<&str> = HashSet::with_capacity(returned.len());
    let mut hits = 0usize;
    let (mut near_ties, mut missing_from_gt) = (0usize, 0u32);
    for (i, id) in returned.iter().enumerate() {
        if !seen.insert(id.as_str()) {
            continue;
        }
        if ground_truth.contains(id.as_str()) {
            hits += 1;
            continue; // already correct: no tie question to ask
        }
        let Some((scores, cutoff_score, orientation_known)) = compare else {
            continue;
        };
        // Positional pairing; a response shorter on scores than on ids simply
        // contributes no tie information for the tail.
        let Some(raw) = scores.get(i) else { continue };
        let score = if engine_higher_is_better { *raw } else { -*raw };
        if scores_tied(score, cutoff_score, tie_epsilon) {
            near_ties += 1; // equally correct at the cutoff
        } else if orientation_known && score > cutoff_score {
            missing_from_gt += 1; // better than the k-th yet unknown
        }
    }

    let recall = hits as f64 / denom as f64;
    // Capped: the upper bound is still a recall, and a pathological response
    // full of boundary-scoring ids must not push it past 1.0.
    let tolerant = ((hits + near_ties) as f64 / denom as f64).min(1.0);
    Some(RecallSample {
        recall,
        tolerant,
        short,
        missing_from_gt,
    })
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
    /// This dispatch failed on the CLIENT's own deadline (see
    /// [`BatchOutcome::timed_out`]) — counted apart from other errors, because
    /// a timing-out cell is a saturation finding while a transport error is a
    /// broken run, and its latency sample sits AT the timeout value, so any
    /// cell with `timeouts > 0` has artificially censored tail percentiles.
    pub timed_out: bool,
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
    /// Queries in this dispatch where the engine returned FEWER than `top_k`
    /// ids — a thin index or a tight filter rather than a ranking failure.
    /// Counted whether or not the query has ground truth; queries with a
    /// present-but-empty ground truth are excluded, since returning nothing is
    /// correct for those (they are counted as `empty_ground_truth`).
    pub short_returns: u64,
    /// Results in this dispatch that scored better than their ground truth's
    /// k-th place yet were absent from it — see [`RecallSample::missing_from_gt`].
    pub missing_from_gt: u64,
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
fn report_first_error(
    error_reported: &std::sync::atomic::AtomicBool,
    error: &str,
    timed_out: bool,
) -> bool {
    let first = !error_reported.swap(true, Ordering::Relaxed);
    if first {
        let hint = if timed_out {
            " [timeout: the query outlived a deadline. Client-side (\"Timeout expired\"): raise \
             the target's timeout_s if the cluster is healthy-but-slow; the client also retries \
             a cancelled read once, so each such timeout costs ~2x timeout_s and doubles server \
             work. Server-side (\"timed out after ...\"): raise the server's search timeout \
             (qdrant: storage.performance.search_timeout_sec, 60s when unset)]"
        } else {
            ""
        };
        tracing::warn!(
            "first failed dispatch of the run (further failures are only counted): {error}{hint}"
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
    tie_epsilon: f64,
    scores_comparable: bool,
    engine_higher_is_better: bool,
    filtered: bool,
    started: Instant,
    error_reported: &std::sync::atomic::AtomicBool,
) -> DispatchSample {
    if let Some(error) = &out.error {
        report_first_error(error_reported, error, out.timed_out);
    }

    // A query scores recall only when the dispatch returned ids for it AND it
    // had ground truth. `recall_at_k` returning `None` there means the ground
    // truth was present but empty — count those separately rather than lose them.
    let mut recalls = Vec::new();
    let mut empty_ground_truth = 0u64;
    let mut filter_overreturn = 0u64;
    let mut short_returns = 0u64;
    for (pos, (&i, ids)) in idxs.iter().zip(out.ids.iter()).enumerate() {
        // Incomparable scores are dropped here, at the source, so no
        // tie-derived value is ever computed from them.
        let scores = scores_comparable
            .then(|| out.scores.get(pos).and_then(|s| s.as_deref()))
            .flatten();
        // Counted for every query that returned SOMETHING, with or without
        // ground truth: a short response is a property of the engine, not of
        // whether we can score it. Queries whose ground truth is present but
        // empty are excluded — returning nothing is the expected outcome
        // there, and they are already tallied as `recall_empty_gt`.
        if let Some(returned) = ids.as_ref() {
            // Not counted when returning fewer is the EXPECTED outcome:
            //   * a filter is active — a selective one legitimately matches
            //     fewer than top_k docs for most queries, which would otherwise
            //     make this alarm fire on every firing of the run;
            //   * the ground truth itself is shallower than top_k — the corpus
            //     does not hold that many matches to return;
            //   * the ground truth is present but empty (counted separately as
            //     `empty_ground_truth`).
            // The bar is what the corpus can actually supply for THIS query:
            // `top_k`, or the ground truth's own depth when it is shallower.
            //   * empty ground truth -> nothing expected (already counted as
            //     `empty_ground_truth`);
            //   * NO ground truth under a filter -> unknowable. A selective
            //     filter legitimately matches only a handful of docs, and with
            //     nothing to compare against, counting every firing would make
            //     this alarm fire on the whole run.
            let gt = vectors[i].ground_truth.as_ref();
            let expected = match gt {
                Some(g) if g.is_empty() => 0,
                // DEDUPED count, matching the short bucket's denominator: a
                // perfect engine returns distinct ids, so a ground truth
                // holding a repeat cannot be answered with more than it has,
                // and counting the positional depth would flag a query the
                // recall math simultaneously scores 1.0.
                Some(g) => top_k.min(g.len() as u64),
                None if filtered => 0,
                None => top_k,
            };
            if (returned.len() as u64) < expected {
                short_returns += 1;
            }
        }
        if let Some((ids, gt)) = ids.as_ref().zip(vectors[i].ground_truth.as_ref()) {
            // Suspected filter leak: with a filter active, the vdb returned more
            // ids than the (exhaustive) filtered ground truth holds. Only under
            // a filter — unfiltered over-return is benign truncation (shallow gt
            // vs deeper top_k). An empty gt is its own bucket, counted above.
            if filtered && !gt.is_empty() && ids.len() as u64 > gt.len() as u64 {
                filter_overreturn += 1;
            }
            match recall_at_k(
                ids,
                scores,
                gt,
                vectors[i].gt_depth,
                vectors[i].gt_cutoff,
                top_k,
                tie_epsilon,
                engine_higher_is_better,
            ) {
                Some(sample) => recalls.push(sample),
                None => empty_ground_truth += 1,
            }
        }
    }
    let missing_from_gt = recalls.iter().map(|r| r.missing_from_gt as u64).sum();
    DispatchSample {
        t_s: started.elapsed().as_secs_f64(),
        latency_ms: out.latency.as_secs_f64() * 1000.0,
        ok: out.ok,
        timed_out: out.timed_out,
        recalls,
        empty_ground_truth,
        filter_overreturn,
        short_returns,
        missing_from_gt,
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
    // Relative score tolerance for calling two scores tied — resolved from
    // the collection's datatype (or the config) before the run starts.
    cmp: ScoreComparison,
    filtered: bool,
    recorder: Option<Box<dyn crate::report::Recorder>>,
) -> StormResults {
    // Static facts about the query set — computed once, before any load is
    // offered, from the cutoffs the loader already derived.
    let scores_comparable = cmp.disabled_reason.is_none();
    let engine_higher_is_better = cmp.engine_higher_is_better;
    // Only queries that actually carry ground truth contribute to recall, so
    // reporting the full loaded count next to a recall mean overstates how
    // many queries that mean is over.
    let queries = vectors
        .iter()
        .filter(|v| {
            // The count printed beside `recall@k` must describe THAT bucket:
            // empty ground truths never score, and shallow ones land in
            // `recall@k_short`.
            v.ground_truth.as_ref().is_some_and(|g| !g.is_empty()) && v.gt_depth as u64 >= top_k
        })
        .count() as u64;
    // Same eligibility filter as `full_recall_queries` below, plus one more:
    // rows whose score column was SQL NULL have no cutoff and drop out here.
    // Those rows also make `full_recall_tolerant` conservative — with no
    // cutoff their `tolerant` equals their exact recall — so the upper bound
    // errs downward rather than inventing a tie.
    // Every query that can contribute a tie to EITHER bucket: a non-empty
    // ground truth with a derived cutoff. Deliberately NOT restricted to the
    // full bucket — under a selective filter a shallow ground truth is the
    // norm, and those cutoffs are the most likely to be tied.
    let cutoffs: Vec<u32> = vectors
        .iter()
        .filter(|v| v.ground_truth.as_ref().is_some_and(|g| !g.is_empty()))
        .filter_map(|v| v.gt_cutoff.map(|c| c.ties))
        .collect();
    let ties = (!cutoffs.is_empty()).then(|| TieStats {
        mean: cutoffs.iter().map(|t| *t as f64).sum::<f64>() / cutoffs.len() as f64,
        max: cutoffs.iter().copied().max().unwrap_or(0),
        fraction_of_queries: cutoffs.iter().filter(|t| **t > 1).count() as f64
            / cutoffs.len() as f64,
        queries: cutoffs.len() as u64,
    });
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
        let mut short_ret = 0u64;
        let mut missing_gt = 0u64;
        let mut n_ok = 0u64;
        let mut n_err = 0u64;
        let mut n_timeout = 0u64;
        let mut dropped = 0u64;
        while let Some(s) = rx.recv().await {
            // Accumulate first — copying the fields the summary needs — so `s`
            // is still owned to hand to the writer without a clone.
            latencies.push(s.latency_ms);
            recalls.extend(s.recalls.iter().copied());
            empty_gt += s.empty_ground_truth;
            over_gt += s.filter_overreturn;
            short_ret += s.short_returns;
            missing_gt += s.missing_from_gt;
            if s.ok {
                n_ok += 1;
            } else {
                n_err += 1;
                if s.timed_out {
                    n_timeout += 1;
                }
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
        (
            latencies, recalls, empty_gt, over_gt, short_ret, missing_gt, n_ok, n_err, n_timeout,
            dropped,
        )
    });

    let started = Instant::now();
    let stop_at = started + Duration::from_secs_f64(profile.duration_s);
    let batch_size = profile.batch_size.max(1);
    // Per-run "first error already logged" flag — see `dispatch_sample`.
    let error_reported = Arc::new(std::sync::atomic::AtomicBool::new(false));

    if profile.target_rps > 0.0 {
        run_paced(
            &target,
            &vectors,
            profile,
            started,
            stop_at,
            top_k,
            cmp.epsilon,
            scores_comparable,
            engine_higher_is_better,
            filtered,
            &tx,
            &error_reported,
        )
        .await;
    } else {
        run_closed_loop(
            &target,
            &vectors,
            profile,
            started,
            stop_at,
            top_k,
            cmp.epsilon,
            scores_comparable,
            engine_higher_is_better,
            filtered,
            &tx,
            &error_reported,
        )
        .await;
    }

    // Drop the last sender so the collector's `recv` loop ends.
    drop(tx);
    let wall_s = started.elapsed().as_secs_f64();

    let (
        latencies_ms,
        recalls,
        empty_ground_truth,
        filter_overreturn,
        short_returns,
        missing_from_gt,
        n_ok,
        n_err,
        n_timeout,
        dropped_samples,
    ) = collector.await.unwrap_or_default();
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
        short_returns,
        missing_from_gt,
        full_recall_queries: queries,
        ties,
        top_k,
        tie_epsilon: cmp.epsilon,
        tie_epsilon_source: cmp.epsilon_source.clone(),
        tie_disabled_reason: cmp.disabled_reason.clone(),
        scores_configured: cmp.configured,
        filter_overreturn,
        n_ok,
        n_err,
        n_timeout,
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
    tie_epsilon: f64,
    scores_comparable: bool,
    engine_higher_is_better: bool,
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
                    &out,
                    &idxs,
                    &vectors,
                    top_k,
                    tie_epsilon,
                    scores_comparable,
                    engine_higher_is_better,
                    filtered,
                    started,
                    &error_reported,
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
    tie_epsilon: f64,
    scores_comparable: bool,
    engine_higher_is_better: bool,
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
        let permit = sem
            .clone()
            .acquire_owned()
            .await
            .expect("semaphore not closed");
        // acquire may have blocked; re-check the deadline before launching.
        if !fixed_work && Instant::now() >= stop_at {
            break;
        }
        let start = idx % n;
        let size = if fixed_work {
            batch_size.min(total_firings - idx)
        } else {
            batch_size
        };
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
                &out,
                &idxs,
                &vectors,
                top_k,
                tie_epsilon,
                scores_comparable,
                engine_higher_is_better,
                filtered,
                started,
                &error_reported,
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
                    scores: vec![None; queries.len()],
                    error: Some("mock failure".into()),
                    timed_out: false,
                };
            }
            BatchOutcome {
                latency: Duration::from_micros(100),
                ok: true,
                ids: vec![Some(self.ids.clone()); queries.len()],
                scores: vec![None; queries.len()],
                error: None,
                timed_out: false,
            }
        }
    }

    fn vectors() -> Vec<QueryVector> {
        (0..16)
            .map(|i| QueryVector {
                gt_cutoff: None,
                gt_depth: 2,
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: None,
                filter_values: HashMap::new(),
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn closed_loop_fires_many_and_records_each() {
        let profile = LoadProfile {
            concurrency: 4,
            duration_s: 0.2,
            target_rps: 0.0,
            batch_size: 1,
            passes: 0,
        };
        let target = Arc::new(MockTarget::ok(vec![]));
        let results = run_storm(target, vectors(), &profile, 10, cmp(None), false, None).await;
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
        let profile = LoadProfile {
            concurrency: 16,
            duration_s,
            target_rps,
            batch_size: 1,
            passes: 0,
        };
        let target = Arc::new(MockTarget::ok(vec![]));
        let results = run_storm(target, vectors(), &profile, 10, cmp(None), false, None).await;
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
                gt_cutoff: None,
                gt_depth: 4, // matches the 4-id ground truth below
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: if i % 2 == 0 {
                    Some(HashSet::from([
                        "a".to_string(),
                        "z".into(),
                        "y".into(),
                        "x".into(),
                    ]))
                } else {
                    None
                },
                filter_values: HashMap::new(),
            })
            .collect();

        let profile = LoadProfile {
            concurrency: 2,
            duration_s: 0.15,
            target_rps: 0.0,
            batch_size: 1,
            passes: 0,
        };
        let results = run_storm(target, vectors, &profile, 4, cmp(None), false, None).await;
        let summary = results.summary();

        // every recorded recall sample must be exactly 0.25 -- never 0, never
        // computed against a query that had no ground truth. Ground truth is 4
        // ids at k=4, so all samples are full-depth (not short).
        assert!(!results.recalls.is_empty());
        assert!(
            results
                .recalls
                .iter()
                .all(|s| !s.short && (s.recall - 0.25).abs() < 1e-9)
        );
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
        let target = Arc::new(MockTarget {
            ids: vec![],
            fail: true,
        });
        let vectors: Vec<QueryVector> = (0..8)
            .map(|i| QueryVector {
                gt_cutoff: None,
                gt_depth: 1, // matches the 1-id ground truth on this fixture
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: Some(HashSet::from(["a".to_string()])),
                filter_values: HashMap::new(),
            })
            .collect();

        let profile = LoadProfile {
            concurrency: 2,
            duration_s: 0.15,
            target_rps: 0.0,
            batch_size: 1,
            passes: 0,
        };
        let results = run_storm(target, vectors, &profile, 1, cmp(None), false, None).await;
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
                gt_cutoff: None,
                gt_depth: 2,
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

        let profile = LoadProfile {
            concurrency: 2,
            duration_s: 0.15,
            target_rps: 0.0,
            batch_size: 1,
            passes: 0,
        };
        let results = run_storm(target, vectors, &profile, 2, cmp(None), false, None).await;
        let summary = results.summary();

        // The empty-gt firings are counted, not scored...
        assert!(summary.empty_ground_truth > 0);
        // ...and never leaked into a recall bucket: every recorded sample is the
        // 0.5 from the real-ground-truth queries, none a 0.0 or NaN.
        assert!(
            results
                .recalls
                .iter()
                .all(|s| (s.recall - 0.5).abs() < 1e-9)
        );
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
                    gt_cutoff: None,
                    gt_depth: 1, // matches the 1-id ground truth on this fixture
                    vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                    ground_truth: Some(HashSet::from(["a".to_string()])), // 1 id, "a" is a hit
                    filter_values: HashMap::new(),
                })
                .collect()
        };
        let profile = LoadProfile {
            concurrency: 2,
            duration_s: 0.15,
            target_rps: 0.0,
            batch_size: 1,
            passes: 0,
        };

        // With a filter: every firing over-returned relative to its 1-id ground
        // truth, so it's counted...
        let target = Arc::new(MockTarget::ok(vec!["a".into(), "b".into()]));
        let filtered = run_storm(target, vectors(), &profile, 5, cmp(None), true, None).await;
        let fs = filtered.summary();
        assert!(fs.filter_overreturn > 0);
        assert_eq!(fs.filter_overreturn, fs.total_recall.unwrap().n);
        // ...but recall is untouched: short bucket, 1 hit / 1 gt id = 1.0.
        assert!(
            filtered
                .recalls
                .iter()
                .all(|s| s.short && (s.recall - 1.0).abs() < 1e-9)
        );
        assert_eq!(fs.empty_ground_truth, 0); // a different signal

        // Same over-return WITHOUT a filter -> not a leak, count stays 0, while
        // recall is identical.
        let target = Arc::new(MockTarget::ok(vec!["a".into(), "b".into()]));
        let unfiltered = run_storm(target, vectors(), &profile, 5, cmp(None), false, None).await;
        let us = unfiltered.summary();
        assert_eq!(us.filter_overreturn, 0);
        assert!(
            unfiltered
                .recalls
                .iter()
                .all(|s| (s.recall - 1.0).abs() < 1e-9)
        );
    }

    #[test]
    fn recall_at_k_full_depth_divides_by_k_and_is_not_short() {
        let returned = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let ground_truth = HashSet::from([
            "a".to_string(),
            "b".to_string(),
            "z".to_string(),
            "y".to_string(),
        ]);
        // gt has 4 ids (>= k), so denominator is k, not gt len or returned len.
        // 2 of the 3 returned ids are in ground_truth, k=4 -> 2/4.
        let r = recall_at_k(&returned, None, &ground_truth, 4, None, 4, 2e-4, true).unwrap();
        assert_eq!((r.recall, r.short), (0.5, false));
        // k=2 (<= gt len) still divides by k -> 2/2 = 1.0, still full-depth.
        let r = recall_at_k(&returned, None, &ground_truth, 4, None, 2, 2e-4, true).unwrap();
        assert_eq!((r.recall, r.short), (1.0, false));
        // no returned ids -> 0 hits over k, still a real (full-depth) sample.
        let r = recall_at_k(&[], None, &ground_truth, 4, None, 4, 2e-4, true).unwrap();
        assert_eq!((r.recall, r.short), (0.0, false));
    }

    #[test]
    fn recall_at_k_short_ground_truth_divides_by_its_own_length_and_is_flagged() {
        // gt has 2 ids but k=4 -> can never fill k. Score against gt len (2),
        // not k, and flag it short so the summary keeps it in its own bucket.
        let returned = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let ground_truth = HashSet::from(["a".to_string(), "b".to_string()]);
        let r = recall_at_k(&returned, None, &ground_truth, 2, None, 4, 2e-4, true).unwrap();
        assert_eq!((r.recall, r.short), (1.0, true)); // 2 hits / 2 gt, NOT 2/4
        // one hit of the two -> 1/2, still short.
        let one = vec!["a".to_string(), "zzz".to_string()];
        let r = recall_at_k(&one, None, &ground_truth, 2, None, 4, 2e-4, true).unwrap();
        assert_eq!((r.recall, r.short), (0.5, true));
    }

    #[test]
    fn recall_at_k_empty_ground_truth_is_no_sample_not_a_nan() {
        // Dividing by an empty gt's length would be NaN and poison the mean;
        // an empty (or absent) ground truth is simply "nothing to measure".
        let returned = vec!["a".to_string()];
        assert!(recall_at_k(&returned, None, &HashSet::new(), 0, None, 4, 2e-4, true).is_none());
    }

    #[test]
    fn recall_at_k_dedupes_returned_so_a_repeated_id_cannot_exceed_1_0() {
        // "a" appears 3 times in `returned` -- must still count as a single
        // hit, not 3, or recall would read 1.5 for k=2 (impossible for a
        // fraction that's supposed to be capped at 1.0).
        let returned = vec!["a".to_string(), "a".to_string(), "a".to_string()];
        let ground_truth = HashSet::from(["a".to_string(), "b".to_string()]);
        assert_eq!(
            recall_at_k(&returned, None, &ground_truth, 2, None, 2, 2e-4, true)
                .unwrap()
                .recall,
            0.5
        );
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let sorted: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert_eq!(percentile(&sorted, 50.0), 50.0);
        assert_eq!(percentile(&sorted, 99.0), 99.0);
        assert_eq!(percentile(&[], 99.0), 0.0);
    }

    // ---- tie-aware recall (see `recall_at_k`) --------------------------

    /// The comparison config these tests run under: a normal larger-is-better
    /// engine with the tolerance configured explicitly.
    fn cmp(disabled_reason: Option<String>) -> ScoreComparison {
        ScoreComparison {
            epsilon: 2e-4,
            epsilon_source: "configured".to_string(),
            disabled_reason,
            configured: true,
            engine_higher_is_better: true,
        }
    }

    fn gt(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn without_scores_the_bounds_collapse_to_exact_recall() {
        let returned = vec!["a".to_string(), "z".to_string()];
        let r = recall_at_k(&returned, None, &gt(&["a", "b"]), 2, None, 2, 2e-4, true).unwrap();
        assert_eq!(r.recall, 0.5);
        assert_eq!(r.tolerant, r.recall, "no scores -> nothing to call a tie");
        assert_eq!(r.missing_from_gt, 0);
    }

    #[test]
    fn a_result_tied_with_the_cutoff_counts_toward_the_upper_bound_only() {
        // "z" isn't in the ground truth but scores exactly what its 2nd place
        // does — the ground truth picked one member of a tie, the engine the
        // other. Exact recall must not credit it; tolerant must.
        let returned = vec!["a".to_string(), "z".to_string()];
        let scores = [0.9f32, 0.5f32];
        let cutoff = GtCutoff {
            score: 0.5,
            ties: 3,
            ascending: Some(false),
        };
        let r = recall_at_k(
            &returned,
            Some(&scores),
            &gt(&["a", "b"]),
            2,
            Some(cutoff),
            2,
            2e-4,
            true,
        )
        .unwrap();
        assert_eq!(r.recall, 0.5, "exact recall is the LOWER bound");
        assert_eq!(r.tolerant, 1.0, "tied result is equally correct");
        assert_eq!(r.missing_from_gt, 0, "a tie is not a mismatch");
    }

    #[test]
    fn a_result_scoring_above_the_cutoff_is_a_mismatch_not_a_tie() {
        // Scoring BETTER than the ground truth's k-th place while absent from
        // it means the ground truth and the collection disagree. It must be
        // counted, and must not inflate the tolerant bound.
        let returned = vec!["a".to_string(), "z".to_string()];
        let scores = [0.9f32, 0.8f32];
        let cutoff = GtCutoff {
            score: 0.5,
            ties: 1,
            ascending: Some(false),
        };
        let r = recall_at_k(
            &returned,
            Some(&scores),
            &gt(&["a", "b"]),
            2,
            Some(cutoff),
            2,
            2e-4,
            true,
        )
        .unwrap();
        assert_eq!(r.recall, 0.5);
        assert_eq!(r.tolerant, 0.5, "must NOT be credited as a tie");
        assert_eq!(r.missing_from_gt, 1);
    }

    #[test]
    fn tolerant_recall_is_capped_at_one() {
        // Every result sits on the cutoff score: without a cap the ratio would
        // exceed 1.0 and stop being a fraction.
        let returned: Vec<String> = ["w", "x", "y", "z"].iter().map(|s| s.to_string()).collect();
        let scores = [0.5f32; 4];
        let cutoff = GtCutoff {
            score: 0.5,
            ties: 9,
            ascending: Some(false),
        };
        let r = recall_at_k(
            &returned,
            Some(&scores),
            &gt(&["a", "b"]),
            2,
            Some(cutoff),
            2,
            2e-4,
            true,
        )
        .unwrap();
        assert_eq!(r.recall, 0.0);
        assert_eq!(r.tolerant, 1.0);
    }

    #[test]
    fn the_tolerance_is_relative_and_respected() {
        let returned = vec!["z".to_string()];
        let cutoff = GtCutoff {
            score: 1.0,
            ties: 2,
            ascending: Some(false),
        };
        // 1e-5 off the cutoff: inside a 2e-4 tolerance, outside a 1e-9 one.
        let near = [1.000_01f32];
        let loose = recall_at_k(
            &returned,
            Some(&near),
            &gt(&["a"]),
            1,
            Some(cutoff),
            1,
            2e-4,
            true,
        )
        .unwrap();
        assert_eq!(loose.tolerant, 1.0, "within tolerance -> tied");
        let tight = recall_at_k(
            &returned,
            Some(&near),
            &gt(&["a"]),
            1,
            Some(cutoff),
            1,
            1e-9,
            true,
        )
        .unwrap();
        assert_eq!(tight.tolerant, 0.0, "outside tolerance -> not tied");
        assert_eq!(tight.missing_from_gt, 1, "and it scored above the cutoff");
    }

    #[test]
    fn summary_display_appends_recall_lines_only_when_present() {
        let base = Summary {
            requests: 10,
            errors: 0,
            timeouts: 0,
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
            full_recall_tolerant: None,
            short_recall_tolerant: None,
            total_recall_tolerant: None,
            short_returns: 0,
            missing_from_gt: 0,
            full_recall_queries: 10,
            ties: None,
            top_k: 10,
            tie_epsilon: Some(2e-4),
            tie_epsilon_source: Some("configured".to_string()),
            tie_disabled_reason: None,
            schema_version: 2,
        };
        assert!(!base.to_string().contains("recall"));

        // Recall prints at the depth it was measured at, with the query and
        // firing counts kept apart (queries cycle round-robin, so `n` firings
        // is not a distinct-query count).
        let with_recall = Summary {
            full_recall: Some(RecallBucket { n: 8, mean: 0.87 }),
            total_recall: Some(RecallBucket { n: 8, mean: 0.87 }),
            ..base.clone()
        };
        let s = with_recall.to_string();
        assert!(s.contains("recall@10: 0.8700"), "{s}");
        assert!(s.contains("(10 eligible queries, 8 firings)"), "{s}");
        // No short bucket -> `recall_total` would just repeat the line above.
        assert!(!s.contains("recall_total"), "{s}");
        assert!(!s.contains("recall@10_short"), "{s}");
        assert!(
            !s.contains("ties_at_cutoff"),
            "no scores configured -> no tie line: {s}"
        );
        assert!(!s.contains("recall_empty_gt"));
        assert!(!s.contains("filter_overreturn"));

        // With BOTH buckets, the blended total earns its line back.
        let both = Summary {
            full_recall: Some(RecallBucket { n: 8, mean: 0.87 }),
            short_recall: Some(RecallBucket { n: 2, mean: 0.40 }),
            total_recall: Some(RecallBucket { n: 10, mean: 0.78 }),
            ..base.clone()
        };
        let s = both.to_string();
        assert!(s.contains("recall@10_short"), "{s}");
        assert!(s.contains("recall_total"), "{s}");

        // Ties make recall a RANGE: exact id match .. tied scores also count.
        let tied = Summary {
            full_recall: Some(RecallBucket { n: 8, mean: 0.8814 }),
            full_recall_tolerant: Some(0.9691),
            short_recall_tolerant: None,
            total_recall_tolerant: None,
            ties: Some(TieStats {
                mean: 6.8,
                max: 41,
                fraction_of_queries: 0.775,
                queries: 10,
            }),
            ..base.clone()
        };
        let s = tied.to_string();
        assert!(s.contains("0.8814 – 0.9691"), "{s}");
        assert!(s.contains("ties_at_cutoff: 6.8 avg, 41 max"), "{s}");
        assert!(s.contains("77.5% of 10 queries with a cutoff"), "{s}");
        assert!(s.contains("tie_epsilon"), "{s}");

        // No ties -> one number, not a degenerate range.
        let untied = Summary {
            full_recall: Some(RecallBucket { n: 8, mean: 0.8814 }),
            full_recall_tolerant: Some(0.8814),
            short_recall_tolerant: None,
            total_recall_tolerant: None,
            ..base.clone()
        };
        assert!(!untied.to_string().contains("–"), "{}", untied.to_string());

        // Alarms appear only when they fire.
        let with_empty = Summary {
            empty_ground_truth: 3,
            ..base.clone()
        };
        assert!(with_empty.to_string().contains("recall_empty_gt: 3"));
        let with_over = Summary {
            filter_overreturn: 7,
            ..base.clone()
        };
        assert!(with_over.to_string().contains("filter_overreturn: 7"));
        let with_missing = Summary {
            missing_from_gt: 4471,
            ..base.clone()
        };
        assert!(with_missing.to_string().contains("missing_from_gt: 4471"));
        let with_short_ret = Summary {
            short_returns: 18,
            ..base.clone()
        };
        assert!(with_short_ret.to_string().contains("short_returns: 18"));
    }

    #[test]
    fn a_raw_distance_engine_is_normalized_before_comparing() {
        // euclid/manhattan: the engine returns a RAW distance (smaller is
        // nearer) while the cutoff is stored larger-is-nearer. "z" is at
        // distance 0.5 against a cutoff of -0.5, i.e. exactly tied — but only
        // once the engine's score is negated.
        let returned = vec!["a".to_string(), "z".to_string()];
        let scores = [0.2f32, 0.5f32]; // raw distances, ascending = better first
        let cutoff = GtCutoff {
            score: 0.5,
            ties: 3,
            ascending: Some(true),
        }; // raw distance
        let r = recall_at_k(
            &returned,
            Some(&scores),
            &gt(&["a", "b"]),
            2,
            Some(cutoff),
            2,
            2e-4,
            false,
        )
        .unwrap();
        assert_eq!(r.recall, 0.5);
        assert_eq!(r.tolerant, 1.0, "tied once both sides face the same way");
        assert_eq!(r.missing_from_gt, 0);
    }

    #[test]
    fn a_raw_distance_miss_is_not_flagged_as_a_mismatch() {
        // The regression this whole orientation fix exists for: a legitimately
        // WORSE result (larger distance) must not trip `missing_from_gt`.
        // Without normalization `0.9 > -0.5` fires on essentially every miss,
        // turning the stale-ground-truth alarm into noise.
        let returned = vec!["a".to_string(), "z".to_string()];
        let scores = [0.2f32, 0.9f32];
        let cutoff = GtCutoff {
            score: 0.5,
            ties: 1,
            ascending: Some(true),
        }; // raw distance
        let r = recall_at_k(
            &returned,
            Some(&scores),
            &gt(&["a", "b"]),
            2,
            Some(cutoff),
            2,
            2e-4,
            false,
        )
        .unwrap();
        assert_eq!(r.missing_from_gt, 0, "a worse distance is an ordinary miss");
        assert_eq!(r.tolerant, 0.5);

        // And a genuinely BETTER one (smaller distance) still is flagged.
        let better = [0.2f32, 0.1f32];
        let r = recall_at_k(
            &returned,
            Some(&better),
            &gt(&["a", "b"]),
            2,
            Some(cutoff),
            2,
            2e-4,
            false,
        )
        .unwrap();
        assert_eq!(r.missing_from_gt, 1, "nearer than the k-th yet unknown");
    }

    #[test]
    fn every_bucket_gets_a_tie_tolerant_bound() {
        // Ties are not a full-bucket phenomenon. Under a selective filter a
        // shallow ground truth is the norm, and those cutoffs are the ones
        // most likely to be tied — so the short bucket and the blended total
        // must carry an upper bound too, not just `full_recall`.
        let results = StormResults {
            latencies_ms: vec![1.0, 2.0],
            recalls: vec![
                RecallSample { recall: 0.50, tolerant: 0.90, short: false, missing_from_gt: 0 },
                RecallSample { recall: 0.40, tolerant: 0.80, short: true, missing_from_gt: 0 },
            ],
            empty_ground_truth: 0,
            filter_overreturn: 0,
            short_returns: 0,
            missing_from_gt: 0,
            full_recall_queries: 1,
            ties: Some(TieStats { mean: 3.0, max: 5, fraction_of_queries: 1.0, queries: 2 }),
            top_k: 10,
            tie_epsilon: 2e-3,
            tie_epsilon_source: "auto, float16".to_string(),
            tie_disabled_reason: None,
            scores_configured: true,
            n_ok: 2, n_err: 0, n_timeout: 0, wall_s: 1.0, batch_size: 1, dropped_samples: 0,
        };
        let summary = results.summary();
        assert_eq!(summary.full_recall_tolerant, Some(0.90));
        assert_eq!(summary.short_recall_tolerant, Some(0.80), "short bucket bounded too");
        // float sum, so compare with a tolerance rather than for equality
        assert!((summary.total_recall_tolerant.unwrap() - 0.85).abs() < 1e-9, "and the blend");

        let text = summary.to_string();
        assert!(text.contains("recall@10: 0.5000 – 0.9000"), "{text}");
        assert!(text.contains("recall@10_short: 0.4000 – 0.8000"), "{text}");
        assert!(text.contains("recall_total: 0.4500 – 0.8500"), "{text}");
        // The tie line states its OWN denominator rather than borrowing the
        // recall line's, which counts a different population.
        assert!(text.contains("of 2 queries with a cutoff"), "{text}");
    }

    #[test]
    fn a_run_without_a_score_column_emits_no_tie_fields() {
        // Never asking for tie reporting is not the same as being refused it:
        // no banner, but also no tolerance and no "upper bound" implying a
        // comparison that never ran.
        let results = StormResults {
            latencies_ms: vec![1.0],
            recalls: vec![RecallSample {
                recall: 0.5,
                tolerant: 0.5,
                short: false,
                missing_from_gt: 0,
            }],
            empty_ground_truth: 0,
            filter_overreturn: 0,
            short_returns: 0,
            missing_from_gt: 0,
            full_recall_queries: 1,
            ties: None,
            top_k: 10,
            tie_epsilon: 2e-3,
            tie_epsilon_source: "auto, float32".to_string(),
            tie_disabled_reason: None,
            scores_configured: false,
            n_ok: 1,
            n_err: 0,
            n_timeout: 0,
            wall_s: 1.0,
            batch_size: 1,
            dropped_samples: 0,
        };
        let summary = results.summary();
        assert!(
            summary.tie_epsilon.is_none(),
            "no tolerance was ever applied"
        );
        assert!(
            summary.full_recall_tolerant.is_none(),
            "no bound was ever computed"
        );
        let text = summary.to_string();
        assert!(
            !text.contains("tie_reporting"),
            "not refused, just not asked for: {text}"
        );
        assert!(!text.contains("tie_epsilon"), "{text}");
    }

    #[test]
    fn an_unknown_orientation_credits_ties_but_never_fires_the_alarm() {
        // No ordering signal (single hit, or all scores equal) means the
        // orientation is a guess. A tie only widens an upper bound, so it is
        // still credited; `missing_from_gt` is a loud "stale ground truth"
        // claim and must not rest on a guess.
        let returned = vec!["a".to_string(), "z".to_string()];
        let scores = [0.95f32, 0.90f32];
        let cutoff = GtCutoff {
            score: 0.10,
            ties: 1,
            ascending: None,
        };
        let r = recall_at_k(
            &returned,
            Some(&scores),
            &gt(&["a", "b"]),
            2,
            Some(cutoff),
            2,
            2e-4,
            true,
        )
        .unwrap();
        assert_eq!(r.missing_from_gt, 0, "no alarm on a guessed orientation");

        // With the orientation KNOWN, the same shape does fire it.
        let known = GtCutoff {
            score: 0.10,
            ties: 1,
            ascending: Some(false),
        };
        let r = recall_at_k(
            &returned,
            Some(&scores),
            &gt(&["a", "b"]),
            2,
            Some(known),
            2,
            2e-4,
            true,
        )
        .unwrap();
        assert_eq!(r.missing_from_gt, 1, "known orientation, genuine mismatch");
    }

    #[test]
    fn a_short_bucket_recall_of_one_is_reachable_despite_duplicate_ids() {
        // ["a","b","b"] at top_k=10: 3 positions, 2 distinct docs. A perfect
        // engine returns both and must score 1.0 — dividing by the positional
        // depth would cap it at 0.667 with no way to reach the top.
        let returned = vec!["a".to_string(), "b".to_string()];
        let r = recall_at_k(&returned, None, &gt(&["a", "b"]), 3, None, 10, 2e-4, true).unwrap();
        assert!(r.short, "3 < top_k=10");
        assert!(
            (r.recall - 1.0).abs() < 1e-9,
            "perfect engine, got {}",
            r.recall
        );
    }

    /// `short_returns` is decided in `dispatch_sample`, not `recall_at_k`, and
    /// the `filtered` arm is what keeps the alarm from firing on an entire
    /// selective run. Driven through `dispatch_sample` directly, since that is
    /// where the rule lives.
    #[test]
    fn short_returns_bar_adapts_to_filter_and_ground_truth_depth() {
        use crate::targets::BatchOutcome;
        let reported = std::sync::atomic::AtomicBool::new(false);

        let sample =
            |gt: Option<HashSet<String>>, gt_depth: usize, returned: usize, filtered: bool| {
                let vectors = vec![QueryVector {
                    vector: crate::queries::VectorData::Dense(vec![0.0]),
                    ground_truth: gt,
                    gt_cutoff: None,
                    gt_depth,
                    filter_values: HashMap::new(),
                }];
                let ids: Vec<String> = (0..returned).map(|i| format!("h{i}")).collect();
                let out = BatchOutcome {
                    latency: Duration::from_micros(10),
                    ok: true,
                    ids: vec![Some(ids)],
                    scores: vec![None],
                    error: None,
                    timed_out: false,
                };
                dispatch_sample(
                    &out,
                    &[0],
                    &vectors,
                    10,
                    2e-4,
                    true,
                    true,
                    filtered,
                    Instant::now(),
                    &reported,
                )
                .short_returns
            };

        let deep: HashSet<String> = (0..10).map(|i| format!("h{i}")).collect();
        let shallow: HashSet<String> = ["h0".to_string(), "h1".to_string()].into();

        // Full-depth ground truth, engine came up short -> a real finding.
        assert_eq!(sample(Some(deep.clone()), 10, 4, false), 1);
        assert_eq!(sample(Some(deep.clone()), 10, 10, false), 0);
        // Shallow ground truth: the corpus only holds 2, so returning 2 is
        // complete, not short.
        assert_eq!(sample(Some(shallow.clone()), 2, 2, false), 0);
        assert_eq!(sample(Some(shallow.clone()), 2, 1, false), 1);
        // No ground truth at all: unknowable under a filter, so no alarm;
        // without one, top_k is the honest bar.
        assert_eq!(sample(None, 0, 3, true), 0);
        assert_eq!(sample(None, 0, 3, false), 1);
        // An empty ground truth is counted as `empty_ground_truth` instead.
        assert_eq!(sample(Some(HashSet::new()), 0, 0, false), 0);
    }

    #[test]
    fn a_distance_valued_ground_truth_against_a_similarity_engine_is_skipped() {
        // Ascending through POSITIVE values against a larger-is-better engine
        // means a distance-valued ground truth (e.g. `1 - cos`), not a negated
        // similarity. Negating it would leave the two sides a constant apart
        // and fire `missing_from_gt` on every result; the query is skipped
        // instead, exactly as an unknown orientation is.
        let returned = vec!["a".to_string(), "z".to_string()];
        let scores = [0.95f32, 0.90f32]; // cosine similarities from the engine
        let cutoff = GtCutoff {
            score: 0.10,
            ties: 1,
            ascending: Some(true),
        };
        let r = recall_at_k(
            &returned,
            Some(&scores),
            &gt(&["a", "b"]),
            2,
            Some(cutoff),
            2,
            2e-4,
            true,
        )
        .unwrap();
        assert_eq!(r.recall, 0.5, "exact recall is unaffected");
        assert_eq!(r.tolerant, 0.5, "no bogus tie credit");
        assert_eq!(
            r.missing_from_gt, 0,
            "and no false stale-ground-truth alarm"
        );
    }

    #[test]
    fn a_distance_valued_cutoff_of_exactly_zero_is_still_skipped() {
        // A `1 - cos` ground truth over a near-duplicate corpus bottoms out AT
        // zero. `> 0.0` would let that through, flip the cutoff to `-0.0`, and
        // fire `missing_from_gt` on every returned result — the false alarm
        // the guard exists to prevent, at the one value it is most likely to
        // take on a duplicate-heavy corpus.
        let returned = vec!["a".to_string(), "z".to_string()];
        let scores = [0.99f32, 0.98f32]; // cosine similarities
        let cutoff = GtCutoff { score: 0.0, ties: 4, ascending: Some(true) };
        let r = recall_at_k(
            &returned, Some(&scores), &gt(&["a", "b"]), 2, Some(cutoff), 2, 2e-4, true,
        )
        .unwrap();
        assert_eq!(r.missing_from_gt, 0, "no alarm on an unreconcilable orientation");
        assert_eq!(r.tolerant, 0.5, "and no bogus tie credit either");
    }

    #[test]
    fn a_negated_similarity_ground_truth_is_still_recovered() {
        // The quadrant negation IS for: ascending through NEGATIVE values is a
        // negated similarity, and flipping it back is exact.
        let returned = vec!["a".to_string(), "z".to_string()];
        let scores = [0.95f32, 0.50f32];
        let cutoff = GtCutoff {
            score: -0.50,
            ties: 2,
            ascending: Some(true),
        };
        let r = recall_at_k(
            &returned,
            Some(&scores),
            &gt(&["a", "b"]),
            2,
            Some(cutoff),
            2,
            2e-4,
            true,
        )
        .unwrap();
        assert_eq!(r.tolerant, 1.0, "-(-0.5) = 0.5 ties the returned 0.50");
    }

    #[test]
    fn an_all_equal_ground_truth_falls_back_to_the_sign() {
        // No ordering signal (every score identical, or a single hit under a
        // selective filter). For a raw-distance engine a POSITIVE cutoff can
        // only be a raw distance, since nova-bf stores those negated — so the
        // fallback must still detect the tie rather than silently collapsing
        // the tolerant bound onto exact recall.
        let returned = vec!["a".to_string(), "z".to_string()];
        let scores = [0.2f32, 0.5f32]; // raw distances
        let cutoff = GtCutoff {
            score: 0.5,
            ties: 1,
            ascending: None,
        };
        let r = recall_at_k(
            &returned,
            Some(&scores),
            &gt(&["a", "b"]),
            2,
            Some(cutoff),
            2,
            2e-4,
            false,
        )
        .unwrap();
        assert_eq!(r.tolerant, 1.0, "sign fallback must still see the tie");
        assert_eq!(r.missing_from_gt, 0);
    }

    #[test]
    fn duplicate_gt_ids_do_not_demote_a_full_depth_query_to_short() {
        // The set holds 9 after a repeat inside the top-10, but the ground
        // truth is still 10 deep. Classifying on the set length would move the
        // query into the forgiving `short` bucket and divide by 9.
        let returned: Vec<String> = (0..10).map(|i| format!("h{i}")).collect();
        let truth: HashSet<String> = (0..9).map(|i| format!("h{i}")).collect();
        let r = recall_at_k(&returned, None, &truth, 10, None, 10, 2e-4, true).unwrap();
        assert!(!r.short, "10-deep ground truth is not short");
        assert!(
            (r.recall - 0.9).abs() < 1e-9,
            "9 hits / k=10, got {}",
            r.recall
        );
    }

    #[test]
    fn incomparable_scores_withhold_every_tie_derived_number() {
        // A quantized collection queried with rescore=false returns scores from
        // quantized space (measured 3.6e-02 .. 26.4 relative error). Reporting a
        // tie-tolerant bound or a `missing_from_gt` count from those would be
        // confidently wrong, so both are withheld and the reason is printed.
        let results = StormResults {
            latencies_ms: vec![1.0],
            recalls: vec![RecallSample {
                recall: 0.5,
                tolerant: 0.9,
                short: false,
                missing_from_gt: 7,
            }],
            empty_ground_truth: 0,
            filter_overreturn: 0,
            short_returns: 0,
            missing_from_gt: 7,
            full_recall_queries: 1,
            ties: Some(TieStats {
                mean: 6.8,
                max: 41,
                fraction_of_queries: 0.9,
                queries: 10,
            }),
            top_k: 10,
            tie_epsilon: 2e-3,
            tie_epsilon_source: "auto, float16".to_string(),
            tie_disabled_reason: Some("test reason: scores are in quantized space".to_string()),
            scores_configured: true,
            n_ok: 1,
            n_err: 0,
            n_timeout: 0,
            wall_s: 1.0,
            batch_size: 1,
            dropped_samples: 0,
        };
        let summary = results.summary();
        assert!(summary.full_recall_tolerant.is_none(), "no tolerant bound");
        assert!(summary.ties.is_none(), "no tie stats");
        assert_eq!(
            summary.missing_from_gt, 0,
            "the stale-ground-truth alarm must not fire on incomparable scores"
        );
        // The tolerance was never applied, so reporting it beside a null bound
        // would imply a comparison that did not happen. Checked on the struct
        // serde serializes, so the JSON consumer sees the same thing.
        assert!(
            summary.tie_epsilon.is_none(),
            "no tolerance when it went unused"
        );
        assert!(summary.tie_epsilon_source.is_none());
        let json = serde_json::to_value(&summary).expect("serializes");
        assert!(json["tie_epsilon"].is_null(), "{json}");
        assert!(json["ties"].is_null(), "{json}");
        assert!(json["full_recall_tolerant"].is_null(), "{json}");
        assert_eq!(
            summary.full_recall.unwrap().mean,
            0.5,
            "exact recall is unaffected"
        );

        let text = summary.to_string();
        assert!(text.contains("tie_reporting"), "{text}");
        // The reason travels from wherever it was decided to the summary
        // verbatim, so a new suppression cause needs no display change.
        assert!(
            text.contains("test reason: scores are in quantized space"),
            "must echo WHY it was disabled: {text}"
        );
        assert!(
            !text.contains("–"),
            "no range when there is no upper bound: {text}"
        );
        assert!(!text.contains("ties_at_cutoff"), "{text}");
        assert!(!text.contains("tie_epsilon:"), "{text}");
    }

    #[test]
    fn comparable_scores_still_report_ties() {
        let results = StormResults {
            latencies_ms: vec![1.0],
            recalls: vec![RecallSample {
                recall: 0.5,
                tolerant: 0.9,
                short: false,
                missing_from_gt: 0,
            }],
            empty_ground_truth: 0,
            filter_overreturn: 0,
            short_returns: 0,
            missing_from_gt: 0,
            full_recall_queries: 1,
            ties: Some(TieStats {
                mean: 6.8,
                max: 41,
                fraction_of_queries: 0.9,
                queries: 10,
            }),
            top_k: 10,
            tie_epsilon: 2e-3,
            tie_epsilon_source: "auto, float16".to_string(),
            tie_disabled_reason: None,
            scores_configured: true,
            n_ok: 1,
            n_err: 0,
            n_timeout: 0,
            wall_s: 1.0,
            batch_size: 1,
            dropped_samples: 0,
        };
        let summary = results.summary();
        assert_eq!(summary.full_recall_tolerant, Some(0.9));
        assert!(summary.ties.is_some());
        let text = summary.to_string();
        assert!(text.contains("0.5000 – 0.9000"), "{text}");
        assert!(text.contains("ties_at_cutoff"), "{text}");
        assert!(!text.contains("tie_reporting"), "{text}");
    }

    #[test]
    fn summary_serializes_to_json_for_a_calling_tool_to_parse() {
        let summary = Summary {
            requests: 10,
            errors: 1,
            timeouts: 0,
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
            full_recall_tolerant: Some(0.95),
            short_recall_tolerant: None,
            total_recall_tolerant: None,
            short_returns: 1,
            missing_from_gt: 0,
            full_recall_queries: 8,
            ties: Some(TieStats {
                mean: 3.5,
                max: 9,
                fraction_of_queries: 0.5,
                queries: 10,
            }),
            top_k: 10,
            tie_epsilon: Some(2e-3),
            tie_epsilon_source: Some("auto (float16)".to_string()),
            tie_disabled_reason: None,
            schema_version: 2,
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
                        Some(
                            match q.vector.as_dense().expect("test queries are dense")[0] as i64 {
                                0 => vec![],                                 // 0/2 in ground truth -> recall 0.0
                                1 => vec!["a".to_string()],                  // 1/2 -> recall 0.5
                                _ => vec!["a".to_string(), "b".to_string()], // 2/2 -> recall 1.0
                            },
                        )
                    })
                    .collect::<Vec<Option<Vec<String>>>>();
                let scores = vec![None; ids.len()];
                BatchOutcome {
                    latency: Duration::from_micros(100),
                    ok: true,
                    scores,
                    ids,
                    error: None,
                    timed_out: false,
                }
            }
        }
        let gt = Some(HashSet::from(["a".to_string(), "b".to_string()]));
        let vectors = vec![
            QueryVector {
                vector: crate::queries::VectorData::Dense(vec![0.0]),
                ground_truth: gt.clone(),
                gt_cutoff: None,
                gt_depth: 2,
                filter_values: HashMap::new(),
            },
            QueryVector {
                vector: crate::queries::VectorData::Dense(vec![1.0]),
                ground_truth: gt.clone(),
                gt_cutoff: None,
                gt_depth: 2,
                filter_values: HashMap::new(),
            },
            QueryVector {
                vector: crate::queries::VectorData::Dense(vec![2.0]),
                ground_truth: gt,
                gt_cutoff: None,
                gt_depth: 2,
                filter_values: HashMap::new(),
            },
        ];
        // duration long enough to cycle through all 3 at concurrency=1 several times
        let profile = LoadProfile {
            concurrency: 1,
            duration_s: 0.1,
            target_rps: 0.0,
            batch_size: 1,
            passes: 0,
        };
        let results = run_storm(
            Arc::new(PerQueryTarget),
            vectors,
            &profile,
            2,
            cmp(None),
            false,
            None,
        )
        .await;

        // Only asserts the pipeline actually produced all 3 distinct values --
        // NOT their exact proportions, which depend on how many times each of
        // the 3 round-robin slots happened to be hit inside a fixed wall-clock
        // window (not guaranteed 1:1:1). The exact per-bucket mean math itself
        // is covered deterministically, with no timing dependency, by
        // `summary_aggregates_recall_buckets_correctly` below. gt is 2 ids at
        // k=2, so every sample is full-depth (not short).
        assert!(results.recalls.iter().all(|s| !s.short));
        assert!(
            results
                .recalls
                .iter()
                .any(|s| (s.recall - 0.0).abs() < 1e-9)
        );
        assert!(
            results
                .recalls
                .iter()
                .any(|s| (s.recall - 0.5).abs() < 1e-9)
        );
        assert!(
            results
                .recalls
                .iter()
                .any(|s| (s.recall - 1.0).abs() < 1e-9)
        );
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
                RecallSample {
                    recall: 0.5,
                    tolerant: 0.5,
                    short: false,
                    missing_from_gt: 0,
                },
                RecallSample {
                    recall: 1.0,
                    tolerant: 1.0,
                    short: false,
                    missing_from_gt: 0,
                },
                RecallSample {
                    recall: 0.4,
                    tolerant: 0.4,
                    short: true,
                    missing_from_gt: 0,
                },
            ],
            empty_ground_truth: 2,
            filter_overreturn: 0,
            short_returns: 0,
            missing_from_gt: 0,
            full_recall_queries: 3,
            ties: None,
            top_k: 10,
            tie_epsilon: 2e-4,
            tie_epsilon_source: "configured".to_string(),
            tie_disabled_reason: None,
            scores_configured: true,
            n_ok: 3,
            n_err: 0,
            n_timeout: 0,
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
        let cfg = ReportConfig {
            format: ReportFormat::Csv,
            path: path.clone(),
        };
        let mut recorder = cfg.build();
        recorder.begin().expect("begin");

        let profile = LoadProfile {
            concurrency: 4,
            duration_s: 0.2,
            target_rps: 0.0,
            batch_size: 1,
            passes: 0,
        };
        let target = Arc::new(MockTarget::ok(vec![]));
        let results = run_storm(
            target,
            vectors(),
            &profile,
            10,
            cmp(None),
            false,
            Some(recorder),
        )
        .await;
        let summary = results.summary();

        let text = std::fs::read_to_string(&path).expect("csv written");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "t_s,latency_ms,ok,recalls_full,recalls_short");
        // one row per dispatch that reached the sink — the time series IS the
        // raw run, minus any samples dropped when the writer queue was full
        // (with an instant mock target the load loop can briefly outrun the
        // writer; the summary still counts every dispatch).
        assert_eq!(
            lines.len() as u64,
            1 + summary.requests - results.dropped_samples
        );
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

        let profile = LoadProfile {
            concurrency: 4,
            duration_s: 0.2,
            target_rps: 0.0,
            batch_size: 1,
            passes: 0,
        };
        let target = Arc::new(MockTarget::ok(vec![]));
        let results = run_storm(
            target,
            vectors(),
            &profile,
            10,
            cmp(None),
            false,
            Some(Box::new(FailingRecorder)),
        )
        .await;
        let summary = results.summary();

        // Load ran to completion despite the sink failing on the very first row.
        assert!(
            summary.requests > 0,
            "the load test must complete even with a dead sink"
        );
        assert_eq!(
            summary.errors, 0,
            "dispatch errors are unrelated to sink failure"
        );
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
                    .collect::<Vec<Option<Vec<String>>>>();
                let scores = vec![None; ids.len()];
                BatchOutcome {
                    latency: Duration::from_micros(100),
                    ok: true,
                    scores,
                    ids,
                    error: None,
                    timed_out: false,
                }
            }
        }

        let gt = Some(HashSet::from(["a".to_string(), "b".to_string()]));
        let vectors: Vec<QueryVector> = (0..9)
            .map(|i| QueryVector {
                gt_cutoff: None,
                gt_depth: 2,
                vector: crate::queries::VectorData::Dense(vec![i as f32]),
                ground_truth: gt.clone(),
                filter_values: HashMap::new(),
            })
            .collect();
        let batch_size = 3;
        let profile = LoadProfile {
            concurrency: 1,
            duration_s: 0.15,
            target_rps: 0.0,
            batch_size,
            passes: 0,
        };
        let target = Arc::new(BatchCapturingTarget {
            call_lens: std::sync::Mutex::new(Vec::new()),
        });
        let results = run_storm(target.clone(), vectors, &profile, 2, cmp(None), false, None).await;

        let call_lens = target.call_lens.lock().unwrap();
        assert!(!call_lens.is_empty());
        assert!(
            call_lens.iter().all(|&len| len == batch_size),
            "{call_lens:?}"
        );

        assert!(
            results
                .recalls
                .iter()
                .any(|s| (s.recall - 0.0).abs() < 1e-9)
        );
        assert!(
            results
                .recalls
                .iter()
                .any(|s| (s.recall - 0.5).abs() < 1e-9)
        );
        assert!(
            results
                .recalls
                .iter()
                .any(|s| (s.recall - 1.0).abs() < 1e-9)
        );
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
                scores: vec![None; queries.len()],
                error: None,
                timed_out: false,
            }
        }
    }

    fn indexed_vectors(n: usize) -> Vec<QueryVector> {
        (0..n)
            .map(|i| QueryVector {
                gt_cutoff: None,
                gt_depth: 2,
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
        let target = Arc::new(CountingTarget {
            counts: std::sync::Mutex::new(HashMap::new()),
        });
        let vectors = indexed_vectors(10);
        let profile = LoadProfile {
            concurrency: 4,
            duration_s: 0.001,
            target_rps: 0.0,
            batch_size: 3,
            passes: 2,
        };
        let results = run_storm(
            target.clone(),
            vectors,
            &profile,
            10,
            cmp(None),
            false,
            None,
        )
        .await;

        let counts = target.counts.lock().unwrap();
        assert_eq!(counts.len(), 10);
        assert!(counts.values().all(|&c| c == 2), "{counts:?}");
        // 20 firings at batch 3 = 6 full batches + a 2-query tail = 7 dispatches
        assert_eq!(results.n_ok, 7);
    }

    /// Fixed work, paced: the launch budget ends the run, not the clock.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fixed_work_paced_fires_each_query_exactly_once() {
        let target = Arc::new(CountingTarget {
            counts: std::sync::Mutex::new(HashMap::new()),
        });
        let vectors = indexed_vectors(5);
        // 1000 rps so the schedule is not the bottleneck; duration absurd both ways.
        let profile = LoadProfile {
            concurrency: 2,
            duration_s: 10_000.0,
            target_rps: 1000.0,
            batch_size: 2,
            passes: 1,
        };
        let results = run_storm(target.clone(), vectors, &profile, 5, cmp(None), false, None).await;

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
        assert!(report_first_error(&flag, "boom", false)); // first: logs
        assert!(!report_first_error(&flag, "boom", false)); // second: counted only
        assert!(!report_first_error(&flag, "different", false)); // still counted only

        // a NEW flag (a new run) reports again — per-run, not per-process
        let fresh = AtomicBool::new(false);
        assert!(report_first_error(&fresh, "next run's error", false));
    }
}
