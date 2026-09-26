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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::TrySendError;

use tokio::sync::{Semaphore, mpsc};
use tokio::task::JoinSet;
use tokio::time::{Duration, Instant, sleep_until};

use crate::config::LoadProfile;
use crate::queries::{CutoffTies, GtCutoff, GtRank, QueryVector, scores_tied};
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
    /// Queries excluded from recall because their ground truth was present but
    /// empty (`truth_len == 0`) — see [`DispatchSample::empty_ground_truth`].
    /// A distinct-QUERY count, consistent with the `n` in the recall buckets:
    /// counted at the firing that scored the query, not at every firing.
    pub empty_ground_truth: u64,
    /// Suspected filter-leak queries (filter configured AND the vdb returned
    /// more ids than the ground truth holds) — see
    /// [`DispatchSample::filter_overreturn`]. A distinct-QUERY count, like the
    /// recall bucket `n`s.
    pub filter_overreturn: u64,
    /// Queries whose scoring firing returned fewer than `top_k` ids. A
    /// distinct-QUERY count: a target that starts truncating part-way through
    /// a looped run does not move it.
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
    /// RBO's persistence parameter for this run — carried so `summary()` can
    /// report it and derive the truncation residual without a `LoadProfile`
    /// or config in scope.
    pub rbo_p: RboP,
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
    /// Distinct queries that were actually scored. A query is scored at the
    /// first firing that OBSERVED it — a repeat against an unchanged collection
    /// returns the same documents, so scoring it again would copy the
    /// measurement rather than add one, while a firing that errored or timed
    /// out observed nothing and so does not claim the query. Below the loaded
    /// count when a run ended before cycling through them all, which also
    /// means the scored set is the FRONT of the query file rather than a
    /// sample of it.
    pub scored_queries: u64,
    /// Query vectors actually fired, summed over dispatches. NOT
    /// `requests * batch_size`: fixed-work mode trims the final batch, so the
    /// product overstates on any run where `passes * queries` is not a
    /// multiple of `batch_size`.
    pub firings: u64,
    /// The collector thread died, so every measurement this run took was lost.
    /// Distinct from "the target answered nothing": the counts below are all
    /// zero because nothing survived to be counted, and a caller must not
    /// record that as a result. `run()` turns this into an error rather than
    /// returning a clean-looking empty summary.
    pub collector_failed: bool,
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
    /// Actual query throughput: query vectors fired per second (`firings /
    /// wall_s`). Equals `requests_per_sec * batch_size` except where the final
    /// batch was trimmed, which the product overstates.
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
    /// Queries excluded from every recall bucket above because their ground
    /// truth was present but empty (`truth_len == 0`) — nothing to score
    /// against. `0` in the common case; a non-zero value tells the operator some
    /// queries silently sat out recall (distinct from queries with no ground
    /// truth configured at all, which were never in scope for recall).
    pub empty_ground_truth: u64,
    /// Suspected filter leaks: queries where, **with a filter configured**, the
    /// vdb returned MORE result ids than the ground truth holds
    /// (`returned.len() > truth_len`). Recall is unchanged. Only tallied under a
    /// filter — unfiltered over-return is benign truncation (a shallow ground
    /// truth vs a deeper `top_k`), not a bug — and it's a valid leak signal only
    /// when the filtered ground truth is the EXHAUSTIVE match set (bf found all
    /// matching docs, i.e. wasn't itself capped). `0` when it never happened.
    pub filter_overreturn: u64,
    /// Tie-tolerant mean recall over the same queries as `full_recall` — the
    /// UPPER bound (see [`RecallSample::tolerant`]). Equals `full_recall` when
    /// no score column is configured or nothing ties.
    pub full_recall_tolerant: Option<f64>,
    /// Tie-tolerant mean over the SHORT bucket's queries — the same upper
    /// bound as `full_recall_tolerant`, for queries whose ground truth is
    /// shallower than `top_k`. Ties are not a full-bucket phenomenon: under a
    /// selective filter a shallow ground truth is the norm, and those are
    /// exactly the queries whose cutoff is most likely to be tied.
    pub short_recall_tolerant: Option<f64>,
    /// Tie-tolerant mean over both buckets, matching `total_recall`.
    pub total_recall_tolerant: Option<f64>,
    /// Mean Rank-Biased Overlap over the same queries as `full_recall` — how
    /// well the engine reproduced the ground truth's ORDERING, not just its
    /// membership (see [`rbo_at_k`]). `None` when no such query ran.
    pub full_rbo: Option<RecallBucket>,
    /// Mean RBO over the SHORT bucket's queries. Each was compared only as
    /// deep as its own ground truth, so these carry a larger truncation
    /// residual than `rbo_residual` (which describes the full bucket's fixed
    /// depth) — a shallower list leaves more of the weight unobserved.
    pub short_rbo: Option<RecallBucket>,
    /// Mean RBO over both buckets, matching `total_recall`.
    pub total_rbo: Option<RecallBucket>,
    /// Tie-tolerant mean RBO over `full_rbo`'s queries — the UPPER bound,
    /// forgiving reordering among equally-scored ground-truth entries.
    ///
    /// Gated ONLY on a ground-truth score column being configured, NOT on
    /// `tie_disabled_reason`: it is derived from the ground truth's own scores
    /// at load time and never compares an engine score to one, so unlike
    /// `full_recall_tolerant` it survives a quantized collection queried with
    /// `rescore: false`. That asymmetry is deliberate, not an oversight.
    pub full_rbo_tolerant: Option<f64>,
    /// Tie-tolerant mean RBO over the SHORT bucket's queries.
    pub short_rbo_tolerant: Option<f64>,
    /// Tie-tolerant mean RBO over both buckets, matching `total_rbo`.
    pub total_rbo_tolerant: Option<f64>,
    /// `full_rbo`, rescaled so that **1.0 means a perfect ranking**.
    ///
    /// Raw RBO is an infinite sum whose observable part is only `1 - p^depth`,
    /// so a flawless engine scores that ceiling, not 1.0 (see
    /// [`Summary::rbo_residual`]). Dividing each sample by its own ceiling
    /// turns the metric into the expected agreement CONDITIONAL on the reader
    /// stopping within the observed depth — `E[A_D | D <= depth]`. That is an
    /// exact restatement, not an extrapolation: unlike the paper's `RBO_EXT`
    /// it claims nothing whatsoever about the unobserved tail.
    ///
    /// Report this where a reader expects 1.0 to mean perfect (a results
    /// table); report the raw value and residual where the metric itself is
    /// under discussion.
    pub full_rbo_normalized: Option<f64>,
    /// `short_rbo`, normalized per sample. Each short-ground-truth query has
    /// its OWN ceiling, since it was compared only as deep as its own list.
    pub short_rbo_normalized: Option<f64>,
    /// `total_rbo`, normalized per sample.
    pub total_rbo_normalized: Option<f64>,
    /// The tie-tolerant upper bound of `full_rbo_normalized`.
    pub full_rbo_normalized_tolerant: Option<f64>,
    /// The tie-tolerant upper bound of `short_rbo_normalized`.
    pub short_rbo_normalized_tolerant: Option<f64>,
    /// The tie-tolerant upper bound of `total_rbo_normalized`.
    pub total_rbo_normalized_tolerant: Option<f64>,
    /// The `p` the RBO lines were computed with. `None` when no RBO was
    /// scored, so a reader never sees a parameter for a metric that is absent.
    pub rbo_p: Option<f64>,
    /// The LARGEST share of RBO's weight any query in the FULL bucket left
    /// unobserved: `max(p^depth)` over the depths that bucket measured. That is the width of
    /// the gap between the reported (truncated) raw values and the
    /// infinite-depth metric they bound from below, for the worst query in the
    /// run.
    ///
    /// Taken over observed depths rather than assumed to be `p^top_k`, because
    /// a query is measured only as deep as its ground truth can answer: a
    /// short ground truth, or one holding a repeated id, has a SHALLOWER depth
    /// and therefore a LARGER residual. Quoting the full bucket's residual
    /// beside a short bucket's number understated it by 25x in a run whose
    /// ground truths were all 3 deep.
    ///
    /// Per bucket, not one global figure: a single query with a 2-deep ground
    /// truth would otherwise set the run's residual to `p^2` and have the
    /// summary claim that ceiling directly beneath a full-depth `rbo@k` line
    /// whose real ceiling is far higher.
    ///
    /// `None` when that bucket scored nothing. Shrink it by lowering `rbo_p`
    /// or raising `top_k`; the normalized values already account for it.
    pub rbo_residual: Option<f64>,
    /// The same for the SHORT bucket, whose queries are measured at their own
    /// ground-truth depths and so leave more unobserved. Typically much larger
    /// than `rbo_residual`, which is exactly why they are reported apart.
    pub short_rbo_residual: Option<f64>,
    /// Mean of Webber et al.'s `RBO_MIN` over the full bucket — the TIGHTEST
    /// honest lower bound, as opposed to `full_rbo`, which is the bare
    /// truncated sum and therefore assumes the unseen depths agree on nothing.
    /// Quote this, not `full_rbo`, against anyone who knows the paper.
    pub full_rbo_min: Option<f64>,
    /// Mean of `RBO_EXT` over the full bucket — a point ESTIMATE, not a bound:
    /// it assumes agreement continues below the cutoff at the rate seen there.
    /// Reported because it is what the literature quotes, and because the gap
    /// to `full_rbo_min` is the honest width of what the run does not know.
    pub full_rbo_ext: Option<f64>,
    /// Mean of `RBO_RES` over the full bucket: how much room is left between
    /// `full_rbo_min` and the largest value the unobserved depths could
    /// support. `full_rbo_min + full_rbo_res` is the top of that range.
    pub full_rbo_res: Option<f64>,
    /// `RBO_MIN` over the SHORT bucket.
    pub short_rbo_min: Option<f64>,
    /// `RBO_EXT` over the SHORT bucket.
    pub short_rbo_ext: Option<f64>,
    /// `RBO_RES` over the SHORT bucket.
    pub short_rbo_res: Option<f64>,
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
    /// Distinct queries behind every correctness number above — see
    /// [`StormResults::scored_queries`]. Recall and rank agreement are measured
    /// once per query; latency and throughput are per firing.
    pub scored_queries: u64,
    /// Query vectors fired — see [`StormResults::firings`]. Differs from
    /// `requests * batch_size` on a trimmed final batch.
    pub firings: u64,
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

/// One recall bucket's headline: how many queries fell in it and their mean.
/// `n` counts distinct QUERIES, not firings — a query is scored at its first
/// firing only — so the count travels with the mean and a mean over 3 queries
/// can't be read as if it were over 3000.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct RecallBucket {
    pub n: u64,
    pub mean: f64,
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
    /// Rank-Biased Overlap against the ground truth's own ordering — the
    /// ORDER-aware companion to `recall`, which is a set measure and scores a
    /// perfectly-reversed response 1.0. See [`rbo_at_k`].
    pub rbo: f64,
    /// Tie-tolerant RBO — the UPPER bound, forgiving reordering among
    /// ground-truth entries that scored identically. Equals `rbo` when no
    /// ground-truth score column is configured or nothing ties. Unlike
    /// `tolerant`, this is independent of whether ENGINE scores are comparable
    /// (see [`rbo_at_k`]).
    pub rbo_tolerant: f64,
    /// The depth `rbo` was actually compared at — `min(top_k, gt_depth)`.
    /// Carried per sample rather than assumed from `top_k`, because a SHORT
    /// ground truth is compared only as deep as it can answer, which gives it
    /// a different ceiling (`1 - p^depth`) from a full-depth one. The
    /// normalized means in the summary divide by each sample's own ceiling;
    /// dividing the blended mean by one global ceiling would be wrong for any
    /// run holding both bucket kinds.
    pub rbo_depth: u32,
    /// How many ids the two rankings shared at the measured depth (`X_depth`).
    /// Carried so the summary can derive the bounds in [`rbo_bounds`] without
    /// re-walking the response — an id shared at one depth stays shared at
    /// every deeper one, which is the fact that makes a tighter lower bound
    /// possible. Fits in the struct's existing padding, so it costs nothing.
    pub rbo_overlap: u32,
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
        // Every bucket mean is a running (sum, count) rather than a
        // materialized Vec. Collecting them meant eighteen full-length
        // `Vec<f64>` alive at once — measured at 1.8 GB of transient
        // allocation for a 20M-firing run, on top of the samples themselves,
        // and spiking AFTER the load window closed, where an OOM would throw
        // away every number the run had already measured.
        let agg = |pick: &dyn Fn(&RecallSample) -> f64, short: Option<bool>| -> (u64, f64) {
            self.recalls
                .iter()
                .filter(|s| short.is_none_or(|want| s.short == want))
                .fold((0u64, 0.0), |(n, sum), s| (n + 1, sum + pick(s)))
        };
        let bucket = |pick: &dyn Fn(&RecallSample) -> f64, short: Option<bool>| {
            let (n, sum) = agg(pick, short);
            (n > 0).then(|| RecallBucket {
                n,
                mean: sum / n as f64,
            })
        };
        let mean_of = |pick: &dyn Fn(&RecallSample) -> f64, short: Option<bool>| {
            let (n, sum) = agg(pick, short);
            (n > 0).then(|| sum / n as f64)
        };
        // Reporting a tie-tolerant number requires that ties were both ASKED
        // for and possible.
        let ties_reported = self.scores_configured && self.tie_disabled_reason.is_none();
        // The RBO upper bound needs only the ground truth's OWN scores, so it
        // is gated on those being configured — NOT on `ties_reported`, which
        // also requires engine scores to be comparable. See `Summary::full_rbo_tolerant`.
        let rbo_ties_reported = self.scores_configured;
        // Each sample divided by ITS OWN ceiling, before averaging. A short
        // ground truth was compared at a shallower depth and so has a lower
        // ceiling; rescaling the blended mean by one global ceiling would
        // silently overstate those queries.
        let normalized = |pick: &'static dyn Fn(&RecallSample) -> f64, short: Option<bool>| {
            let (n, sum) = self
                .recalls
                .iter()
                .filter(|s| short.is_none_or(|want| s.short == want))
                .fold((0u64, 0.0), |(n, sum), s| {
                    let ceiling = 1.0 - self.rbo_p.0.powi(s.rbo_depth as i32);
                    // Unreachable for a real sample (`p < 1` and `depth >= 1`
                    // both hold), but dividing by a zero ceiling would emit a
                    // NaN or an infinity into a published mean.
                    if ceiling <= 0.0 {
                        return (n, sum);
                    }
                    let normalized = pick(s) / ceiling;
                    // Provably <= 1: each id is charged once at an index >=
                    // its position, and positions are distinct, so `X_d <= d`
                    // and the raw value cannot exceed the ceiling. A silent
                    // 1.0000 would be indistinguishable from a perfect
                    // ranking, so a debug build fails loudly instead.
                    debug_assert!(
                        normalized <= 1.0 + 1e-9,
                        "rbo {} exceeds its ceiling {ceiling} at depth {} — tie-slot \
                         accounting is wrong",
                        pick(s),
                        s.rbo_depth,
                    );
                    (n + 1, sum + normalized.min(1.0))
                });
            (n > 0).then(|| sum / n as f64)
        };
        // `RBO_MIN` / `RBO_EXT` / `RBO_RES` per bucket. The tail factor depends
        // only on `(p, depth)`, and a run's depths take very few distinct
        // values, so it is computed once per depth rather than per sample —
        // for `f` as well, which moves with the data.
        let mut tails: HashMap<u32, f64> = HashMap::new();
        for sample in &self.recalls {
            let depth = sample.rbo_depth as usize;
            let f = 2 * depth - (sample.rbo_overlap as usize).min(depth);
            for d in [depth, f] {
                tails
                    .entry(d as u32)
                    .or_insert_with(|| rbo_tail_factor(d, self.rbo_p.0));
            }
        }
        let bucket_bounds = |short: bool| {
            let (n, min_s, ext_s, res_s) = self
                .recalls
                .iter()
                .filter(|s| s.short == short)
                .fold((0u64, 0.0, 0.0, 0.0), |(n, a, b, c), s| {
                    let (min, ext, res) = rbo_bounds_with_tail(
                        s.rbo,
                        s.rbo_overlap,
                        s.rbo_depth as usize,
                        self.rbo_p.0,
                        |d| tails[&(d as u32)],
                    );
                    (n + 1, a + min, b + ext, c + res)
                });
            let avg = |sum: f64| (n > 0).then(|| sum / n as f64);
            (avg(min_s), avg(ext_s), avg(res_s))
        };
        let bounded = (bucket_bounds(false), bucket_bounds(true));
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
            // Query throughput from the firings that actually happened, not
            // `requests_per_sec * batch_size`: fixed-work mode trims the final
            // batch, so the product credits the run with queries it never
            // fired (200 queries in batches of 32 is 7 dispatches, and the
            // product calls that 224).
            qps: if self.wall_s > 0.0 {
                self.firings as f64 / self.wall_s
            } else {
                0.0
            },
            p50_ms: percentile(&ms, 50.0),
            p95_ms: percentile(&ms, 95.0),
            p99_ms: percentile(&ms, 99.0),
            max_ms: ms.last().copied().unwrap_or(0.0),
            full_recall: bucket(&|s| s.recall, Some(false)),
            short_recall: bucket(&|s| s.recall, Some(true)),
            total_recall: bucket(&|s| s.recall, None),
            empty_ground_truth: self.empty_ground_truth,
            filter_overreturn: self.filter_overreturn,
            full_recall_tolerant: ties_reported
                .then(|| mean_of(&|s| s.tolerant, Some(false)))
                .flatten(),
            short_recall_tolerant: ties_reported
                .then(|| mean_of(&|s| s.tolerant, Some(true)))
                .flatten(),
            total_recall_tolerant: ties_reported
                .then(|| mean_of(&|s| s.tolerant, None))
                .flatten(),
            full_rbo: bucket(&|s| s.rbo, Some(false)),
            short_rbo: bucket(&|s| s.rbo, Some(true)),
            total_rbo: bucket(&|s| s.rbo, None),
            full_rbo_tolerant: rbo_ties_reported
                .then(|| mean_of(&|s| s.rbo_tolerant, Some(false)))
                .flatten(),
            short_rbo_tolerant: rbo_ties_reported
                .then(|| mean_of(&|s| s.rbo_tolerant, Some(true)))
                .flatten(),
            total_rbo_tolerant: rbo_ties_reported
                .then(|| mean_of(&|s| s.rbo_tolerant, None))
                .flatten(),
            // Both describe the RBO lines, so both are withheld when no query
            // produced one — a `p` beside an absent metric reads as a promise
            // the summary didn't keep.
            full_rbo_normalized: normalized(&|s| s.rbo, Some(false)),
            short_rbo_normalized: normalized(&|s| s.rbo, Some(true)),
            total_rbo_normalized: normalized(&|s| s.rbo, None),
            full_rbo_normalized_tolerant: rbo_ties_reported
                .then(|| normalized(&|s| s.rbo_tolerant, Some(false)))
                .flatten(),
            short_rbo_normalized_tolerant: rbo_ties_reported
                .then(|| normalized(&|s| s.rbo_tolerant, Some(true)))
                .flatten(),
            total_rbo_normalized_tolerant: rbo_ties_reported
                .then(|| normalized(&|s| s.rbo_tolerant, None))
                .flatten(),
            rbo_p: (!self.recalls.is_empty()).then_some(self.rbo_p.0),
            // Worst (shallowest-measured) depth in each bucket — `powi`
            // throughout, so these and the per-sample ceilings above agree to
            // the last bit.
            rbo_residual: worst_residual(&self.recalls, self.rbo_p.0, false),
            short_rbo_residual: worst_residual(&self.recalls, self.rbo_p.0, true),
            full_rbo_min: bounded.0.0,
            full_rbo_ext: bounded.0.1,
            full_rbo_res: bounded.0.2,
            short_rbo_min: bounded.1.0,
            short_rbo_ext: bounded.1.1,
            short_rbo_res: bounded.1.2,
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
            scored_queries: self.scored_queries,
            firings: self.firings,
            ties: ties_reported.then_some(self.ties).flatten(),
            top_k: self.top_k,
            tie_epsilon: ties_reported.then_some(self.tie_epsilon),
            tie_epsilon_source: ties_reported.then(|| self.tie_epsilon_source.clone()),
            tie_disabled_reason: self.tie_disabled_reason.clone(),
            schema_version: 3,
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
        // Say it plainly when firings outnumber the queries behind the
        // numbers. Nobody loops a query set to measure recall — the repeats
        // are there to sustain load — but a reader seeing 24,000 firings next
        // to a recall figure will assume 24,000 measurements unless told.
        // What a batch hides, said whenever there IS a batch. This is a
        // statement about `batch_size`, so it must not be gated on looping:
        // `passes: 1` is the configuration recommended for measuring
        // correctness, and it is exactly where the repeat gate below goes
        // quiet — leaving `p95_ms` on the page with nothing saying it covers
        // `batch_size` queries at a time.
        if self.batch_size > 1 {
            lines.push(format!(
                "{:>16}: requests, errors and the latency percentiles above are per \
                 DISPATCH — one round-trip carrying {} queries — so p95_ms is not \
                 per-query service time. Only qps is per query.",
                "latency_basis", self.batch_size
            ));
        }
        // Printed whenever ANY per-query number is on the page — not just
        // recall. `short_returns` and friends are per query too, and a run
        // with no ground truth prints those with no recall line to hang this
        // off: 100 short returns out of 5000 firings reads as 2% unless the
        // basis says the denominator is the query set.
        let per_query_numbers = self.full_recall.is_some()
            || self.short_recall.is_some()
            || self.total_recall.is_some()
            || self.ties.is_some()
            || self.short_returns > 0
            || self.empty_ground_truth > 0
            || self.filter_overreturn > 0;
        if per_query_numbers && self.firings > self.scored_queries {
            lines.push(format!(
                "{:>16}: correctness is per QUERY, scored at the first firing that \
                 OBSERVED it ({} distinct queries out of {} firings): repeats would score \
                 identically against an unchanged collection, and a query whose firings \
                 all errored is never scored at all. requests and errors count \
                 DISPATCHES, not firings.",
                "scoring_basis", self.scored_queries, self.firings
            ));
        }
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
                "{:>16}: {}  ({} eligible queries, {} scored)",
                label,
                value(b.mean, self.full_recall_tolerant),
                self.full_recall_queries,
                b.n
            ));
        }
        if let Some(b) = self.short_recall {
            lines.push(format!(
                "{:>16}: {} ({} queries whose ground truth held <{} ids)",
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
        // RBO sits directly under recall: it scores the SAME queries, and the
        // pair is the point — recall says whether the right ids came back, RBO
        // says whether they came back in the right order. Recall 1.0 next to a
        // low RBO means a perfectly-retrieved, badly-ranked response.
        let rbo_label = format!("rbo@{}", self.top_k);
        // The normalized value rides on the SAME line as the raw one rather
        // than doubling the number of RBO lines. Raw first (it is the metric
        // as defined), `norm` second (it is the one where 1.0 means perfect).
        let norm = |n: Option<f64>, t: Option<f64>| {
            n.map_or_else(String::new, |n| format!("norm {}, ", value(n, t)))
        };
        if let Some(b) = self.full_rbo {
            lines.push(format!(
                "{:>16}: {}  ({}p={}, {} queries)",
                rbo_label,
                value(b.mean, self.full_rbo_tolerant),
                norm(self.full_rbo_normalized, self.full_rbo_normalized_tolerant),
                self.rbo_p.map_or_else(|| "?".into(), |p| format!("{p:.3}")),
                b.n
            ));
        }
        // The ladder, when the truncation gap is wide enough for the rungs to
        // differ: raw assumes the unseen depths agree on NOTHING, min assumes
        // they keep what is already proved, ext assumes agreement continues.
        // Printed together so nobody quotes one as if it were another.
        // One line per bucket, like the residual below. The SHORT bucket's
        // bounds are the wide ones — its queries are measured at their own
        // shallow depths — so printing only the full bucket's hid the run's
        // largest uncertainty, and a short-only run printed none at all.
        //
        // The bar is "the rungs differ at the precision printed", which is
        // what 4 decimals means: 1e-4. An earlier bar of 1.5% was above every
        // residual the derived default can produce, so this block never ran
        // on a default-configured run at all.
        for (label, min, ext, res) in [
            ("rbo_bounds", self.full_rbo_min, self.full_rbo_ext, self.full_rbo_res),
            (
                "rbo_short_bounds",
                self.short_rbo_min,
                self.short_rbo_ext,
                self.short_rbo_res,
            ),
        ] {
            let Some(((min, ext), res)) = min.zip(ext).zip(res).filter(|(_, res)| *res > 1e-4)
            else {
                continue;
            };
            lines.push(format!(
                "{:>16}: min {:.4} / max {:.4} / ext {:.4}  (Webber bounds on the EXACT \
                 value above — it assumes the unobserved depths agree on nothing, `min` \
                 keeps what is already proved, `ext` extrapolates. Truncation only: the \
                 tie-tolerant endpoint is a different quantity and can sit ABOVE `max`)",
                label,
                min,
                min + res,
                ext
            ));
        }
        if let Some(b) = self.short_rbo {
            // No `@k` on this label: every sample in the bucket was measured
            // at its OWN ground-truth depth, never at `top_k`, so naming `k`
            // here would attach a depth to a number that does not have it.
            // `p` repeats on every line — one of these can be the only RBO a
            // run prints, and an RBO without its `p` is not reproducible.
            lines.push(format!(
                "{:>16}: {} ({}p={}, {} queries, each compared only as deep as its own \
                 ground truth)",
                "rbo_short",
                value(b.mean, self.short_rbo_tolerant),
                norm(self.short_rbo_normalized, self.short_rbo_normalized_tolerant),
                self.rbo_p.map_or_else(|| "?".into(), |p| format!("{p:.3}")),
                b.n
            ));
        }
        if let (Some(t), true, true) = (
            self.total_rbo,
            self.short_rbo.is_some(),
            self.full_rbo.is_some(),
        ) {
            lines.push(format!(
                "{:>16}: {} ({}n={})",
                "rbo_total",
                value(t.mean, self.total_rbo_tolerant),
                norm(self.total_rbo_normalized, self.total_rbo_normalized_tolerant),
                t.n
            ));
        }
        // Only worth a line when it is big enough to matter. The bar sits
        // clear of the DERIVED default's own residual, which lands on 1%
        // to within a ULP — a bare `>= 0.01` against it decides by rounding,
        // printing for some `top_k` and not others with no pattern a reader
        // could predict. One line per bucket, each naming its own: the two
        // differ by orders of magnitude in a mixed run.
        for (label, residual) in [
            ("rbo_residual", self.rbo_residual),
            ("rbo_short_residual", self.short_rbo_residual),
        ] {
            let Some(residual) =
                residual.filter(|r| *r > crate::config::RBO_DEFAULT_RESIDUAL * 1.5)
            else {
                continue;
            };
            lines.push(format!(
                "{:>16}: {:.3}  (the SHALLOWEST query in that bucket left this share of \
                 RBO's weight below its own measured depth, so that one query's raw value \
                 tops out at {:.4} rather than 1.0. The line above is a MEAN over the \
                 bucket, whose queries may be measured deeper; `norm` accounts for each \
                 one's own ceiling)",
                label,
                residual,
                1.0 - residual,
            ));
        }
        // The ties that make the range a range, and the tolerance that decided
        // what counts as tied — both only when scores were actually available.
        if let Some(reason) = &self.tie_disabled_reason {
            lines.push(format!(
                "{:>16}: disabled — {reason}. Exact recall above is unaffected, and so is \
                 every rbo line: its range comes from the ground truth's OWN scores and \
                 never compares one to an engine score.",
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

/// The largest `p^depth` among the samples of one bucket — the most RBO
/// weight any of its queries left below the depth it was measured at. `None`
/// when the bucket is empty.
fn worst_residual(recalls: &[RecallSample], p: f64, short: bool) -> Option<f64> {
    recalls
        .iter()
        .filter(|s| s.short == short)
        .map(|s| p.powi(s.rbo_depth as i32))
        .fold(None::<f64>, |worst, r| Some(worst.map_or(r, |w| w.max(r))))
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
/// reaches here). `ground_truth` is already a `HashMap` (built once at load time
/// in `queries.rs`, not per call) since this runs on every query firing.
/// `returned` is deduped before counting hits — a target that ever repeated an
/// id within one query's results must not let that repeat count twice, which
/// would push recall above the `1.0` ceiling a fraction is supposed to have.
#[allow(clippy::too_many_arguments)]
fn recall_at_k(
    returned: &[String],
    scores: Option<&[f32]>,
    ground_truth: &HashMap<String, GtRank>,
    cutoff_ties: Option<&CutoffTies>,
    gt_depth: usize,
    cutoff: Option<GtCutoff>,
    k: u64,
    tie_epsilon: f64,
    engine_higher_is_better: bool,
    rbo_p: RboP,
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
    // Where each returned id sits in the ground truth, resolved HERE and
    // handed to `rbo_at_k`. That kernel used to hash every id a second time,
    // which measured as 81-88% of its cost (17.3 us vs 2.1 us per query at
    // depth 1000) — and it runs on the collector thread concurrently with the
    // load, so its CPU competes with the workers on a saturated box.
    let mut resolved: Vec<Resolved> = Vec::with_capacity(returned.len().min(k as usize));
    for (i, id) in returned.iter().enumerate() {
        let rank = ground_truth.get(id.as_str()).copied();
        // A REPEAT resolves to nothing, so every downstream consumer is
        // automatically deduplicated and none of them needs its own `seen`.
        let first_time = seen.insert(id.as_str());
        if resolved.len() < resolved.capacity() {
            resolved.push(match (first_time, rank) {
                (false, _) => Resolved::Repeat,
                (true, Some(gt)) => Resolved::Ranked(gt),
                // Only ids absent from the ground truth can be cutoff ties, so
                // this second lookup is the minority path.
                (true, None) if cutoff_ties.is_some_and(|t| t.contains(id.as_str())) => {
                    Resolved::CutoffTie
                }
                (true, None) => Resolved::Absent,
            });
        }
        if !first_time {
            continue;
        }
        if rank.is_some() {
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
    // Compared only as deep as this query's ground truth can answer: past its
    // own depth the ground-truth prefix stops growing while the engine's keeps
    // going, so every further depth would score an agreement the ground truth
    // is structurally unable to supply. Same reasoning as the `short` bucket's
    // denominator.
    // DEDUPED depth, not the positional one: ranks are dense over distinct
    // ids, so a ground truth holding a repeat has one rank fewer than its
    // positional depth. Measuring to the positional depth would leave a rank
    // no id can occupy, putting the ceiling out of reach for a response that
    // is actually perfect — recall makes the same adjustment for its short
    // denominator, and for the same reason.
    let rbo_depth = truth_len.min(k) as usize;
    // Without a ground-truth score column every `tied_rank` equals its `rank`
    // and no cutoff ties exist, so the tolerant pass can only reproduce the
    // exact value — and the summary withholds it anyway. Skipping it drops an
    // allocation and a full depth loop from the inline path.
    let (rbo, rbo_tolerant, rbo_overlap) = rbo_at_k(
        &resolved,
        cutoff_ties.map_or(0, |t| t.group_start as usize),
        rbo_depth,
        rbo_p,
        cutoff.is_some(),
    );
    Some(RecallSample {
        recall,
        tolerant,
        rbo,
        rbo_tolerant,
        rbo_depth: rbo_depth as u32,
        rbo_overlap,
        short,
        missing_from_gt,
    })
}

/// Where one returned id stands relative to the ground truth, decided once in
/// [`recall_at_k`]'s pass over the response and reused by [`rbo_at_k`] rather
/// than hashed a second time — that duplicate lookup measured as 81-88% of the
/// kernel's cost, on a path that runs on the collector thread concurrently
/// with the load.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Resolved {
    /// In the ground truth's measured prefix, at this rank.
    Ranked(GtRank),
    /// Equally correct, but the truncation put it outside the prefix — see
    /// [`CutoffTies`].
    CutoffTie,
    /// Not in the ground truth at all.
    Absent,
    /// An id the target already returned at an earlier position. Resolving it
    /// to nothing is what deduplicates every consumer at once.
    Repeat,
}

/// RBO's persistence parameter — the probability the notional reader continues
/// past each rank, so `1/(1-p)` is their expected reading depth and `p^d` is
/// the weight still sitting below depth `d`. A newtype, not a bare `f64`,
/// because it is threaded through the same calls as `tie_epsilon`: two
/// adjacent `f64` arguments transpose silently and both are "small tuning
/// constant" shaped, which is exactly the failure [`ScoreComparison`] exists
/// to prevent elsewhere.
#[derive(Debug, Clone, Copy)]
pub struct RboP(pub f64);

/// Rank-Biased Overlap between the engine's ranking and the ground truth's,
/// as an (exact, tie-tolerant) pair — the same lower/upper bound idiom as
/// [`RecallSample::recall`] / [`RecallSample::tolerant`].
///
/// RBO compares the two rankings by SET OVERLAP at every prefix depth
/// `d = 1..=depth`, then averages those agreements with geometrically decaying
/// weights (`(1-p) * p^(d-1)`), so shallow depths dominate:
///
/// ```text
/// RBO = (1-p) * sum_d  p^(d-1) * (X_d / d)      X_d = |engine[..d] ∩ gt[..d]|
/// ```
///
/// Order is never inspected directly — it is penalized as a side effect. An id
/// returned deeper than its true rank is MISSING from every prefix in between,
/// so the cost scales automatically with how far it moved and how shallow the
/// error was: a rank 40 <-> 41 swap corrupts one prefix, a rank 1 <-> 50 swap
/// corrupts forty-nine of them, including the heaviest.
///
/// Both values are the TRUNCATED sum over the depths actually observed, which
/// is a lower bound on the infinite-depth RBO the metric is defined as; the
/// weight left unobserved is `p^depth`, reported once per run as
/// [`Summary::rbo_residual`] rather than per sample. The paper's extrapolated
/// variant (`RBO_EXT`) is deliberately NOT implemented — it estimates the
/// unseen tail by assuming the observed agreement continues, and a benchmark
/// is better served by an honest bound plus its residual than by an
/// extrapolation whose assumption the run cannot check.
///
/// The tie-tolerant value is the best score achievable under ANY ordering of
/// documents the ground truth scored IDENTICALLY — the ground truth had to
/// break those ties somehow, and an engine that broke them differently is not
/// wrong. It is computed by filling each tie group's slots in the order the
/// engine returned them: the group's `j`-th returned member can only start
/// agreeing once the ground-truth prefix is deep enough to hold `j` of them,
/// i.e. from depth `tied_rank + j`. Charging every member to `tied_rank`
/// instead would let one rank hold the whole group at once — not a reordering
/// of anything, and it credits overlap no real ordering could produce.
/// [`CutoffTies`] members fill the cutoff group's slots on the same footing,
/// since they are equally correct answers that the truncation happened to
/// exclude.
///
/// Note this needs only the ground truth's OWN scores, resolved at load time —
/// unlike [`RecallSample::tolerant`], it never compares an engine score to a
/// ground-truth score, so it stays valid on a quantized collection queried
/// with `rescore: false`, where `tie_disabled_reason` withholds every other
/// tolerant number.
///
/// Webber, Moffat & Zobel, "A Similarity Measure for Indefinite Rankings",
/// ACM TOIS 28(4), 2010.
fn rbo_at_k(
    resolved: &[Resolved],
    tie_group_start: usize,
    depth: usize,
    RboP(p): RboP,
    want_tolerant: bool,
) -> (f64, f64, u32) {
    // Nothing returned agrees with anything at any depth, and the loop below
    // would allocate to prove it. A zero-hit firing under a selective filter
    // is routine.
    if depth == 0 || resolved.is_empty() {
        return (0.0, 0.0, 0);
    }
    // A returned id at position `pos` holding ground-truth rank `r` is inside
    // BOTH prefixes exactly when `d > pos` AND `d > r` — that is, from depth
    // `max(pos, r) + 1` onwards, and forever after. So instead of intersecting
    // two sets at every depth (quadratic), each id is charged ONCE to the
    // depth where it starts agreeing, and `X_d` falls out as a running sum.
    // `.0` is keyed on the exact rank, `.1` on the tie-group slot.
    //
    // Two buffers, not four: `Resolved::Repeat` already deduplicates the
    // response upstream, so the per-firing `charged` / `charged_tie` flag
    // arrays this used to allocate are unnecessary — and the second of those
    // was sized by the ENTIRE cutoff-tie tail, which is unbounded.
    let mut starts = vec![(0u32, 0u32); depth];
    // How many members of the tie group starting at each rank the engine has
    // returned so far. The `j`-th of them takes the group's `j`-th slot.
    let mut group_filled = vec![0u32; if want_tolerant { depth } else { 0 }];
    let mut fill_group = |start: usize, pos: usize, starts: &mut [(u32, u32)]| {
        if !want_tolerant {
            return;
        }
        let slot = start + group_filled[start] as usize;
        group_filled[start] += 1;
        // A group can be handed more members than the prefix has room for;
        // those agree at no depth we measure.
        if slot < depth {
            starts[pos.max(slot)].1 += 1;
        }
    };
    for (pos, entry) in resolved.iter().take(depth).enumerate() {
        match *entry {
            Resolved::Ranked(gt) => {
                let rank = gt.rank as usize;
                // Ranks are dense, so `rank < depth` iff it is inside the
                // measured prefix; a deeper one still belongs to its tie group.
                if rank < depth {
                    starts[pos.max(rank)].0 += 1;
                }
                let tied = gt.tied_rank as usize;
                if tied < depth {
                    fill_group(tied, pos, &mut starts);
                }
            }
            // Equally correct but outside the ground truth's own top-k: it can
            // fill a slot in the cutoff's tie group, and counts toward no
            // exact agreement at all.
            Resolved::CutoffTie => {
                if tie_group_start < depth {
                    fill_group(tie_group_start, pos, &mut starts);
                }
            }
            Resolved::Absent | Resolved::Repeat => {}
        }
    }

    let (mut rbo, mut rbo_tolerant) = (0.0, 0.0);
    let (mut x_exact, mut x_tied) = (0u32, 0u32);
    let mut weight = 1.0 - p;
    for (d, (exact, tied)) in starts.iter().copied().enumerate().map(|(i, s)| (i + 1, s)) {
        x_exact += exact;
        x_tied += tied;
        rbo += weight * (x_exact as f64 / d as f64);
        if want_tolerant {
            rbo_tolerant += weight * (x_tied as f64 / d as f64);
        }
        weight *= p;
    }
    // The two coincide when there was nothing to forgive.
    (rbo, if want_tolerant { rbo_tolerant } else { rbo }, x_exact)
}

/// [`rbo_bounds_with_tail`] computing its own tail factor. Test-only: the
/// summary memoizes the factor per depth instead, since a run's samples take
/// very few distinct depths.
#[cfg(test)]
fn rbo_bounds(truncated: f64, overlap: u32, depth: usize, p: f64) -> (f64, f64, f64) {
    rbo_bounds_with_tail(truncated, overlap, depth, p, |d| rbo_tail_factor(d, p))
}

/// `sum_{d > depth} p^(d-1)/d` — the weight the metric puts on every depth the
/// run never observed, per unit of agreement found there.
///
/// Closed form via `sum_{d>=1} p^d/d = -ln(1-p)`, rather than summing the tail
/// directly, which would need thousands of terms for a large `p`. The
/// subtraction loses relative precision once `p^depth` is tiny — but the term
/// it feeds is then tiny too (bounded by `p^depth`), so the ABSOLUTE error
/// stays around 1e-16. `max(0.0)` keeps a rounding artefact from turning the
/// correction negative, which would break the bound it exists to tighten.
fn rbo_tail_factor(depth: usize, p: f64) -> f64 {
    let mut partial = 0.0;
    let mut p_pow = 1.0;
    for d in 1..=depth {
        p_pow *= p;
        partial += p_pow / d as f64;
    }
    ((-(1.0 - p).ln()) - partial).max(0.0) / p
}

/// The quantities Webber et al. report alongside a truncated RBO, all derived
/// from one query's observations. They differ ONLY in what they assume about
/// the depths the run could not see:
///
/// * `min` — the unseen depths agree no better than what is already proved,
///   but no worse either: an id shared at one depth stays shared at every
///   deeper one, so `X_d >= X_depth` for `d > depth`. The tightest honest LOWER
///   bound, and strictly better than the bare truncated sum, which assumes the
///   unseen depths agree on nothing at all.
/// * `ext` — agreement continues at the rate seen AT the cutoff. A point
///   estimate rather than a bound (the paper's `RBO_EXT`): an extrapolation
///   the run cannot check.
/// * `res` — the width between `min` and the most the unseen depths could
///   possibly hold, so `min + res` is the top of the true value's range.
///
/// The overlap matters to `res`, which is easy to get wrong: stepping one
/// depth deeper adds an id to BOTH prefixes, so the intersection can gain
/// **two**, not one — the engine's new id may already sit in the ground
/// truth's prefix while the ground truth's new id already sits in the
/// engine's. Growth is therefore `min(X_depth + 2(d - depth), d)`, saturating
/// at `f = 2*depth - X_depth`.
///
/// That growth is ACHIEVABLE, not merely an envelope, so `min + res` is tight
/// rather than loose: let the engine's next id be one the ground truth already
/// listed and the ground truth's next id be one the engine already listed, and
/// repeat. After `depth - X_depth` such steps the two prefixes hold the same
/// set, `X_f = f`, and both lists can continue identically. Nor can a step
/// gain three — it adds exactly one id to each prefix, so at most those two
/// can newly enter the intersection. (Worth stating because a bound checked
/// only against its own growth assumption proves nothing; this one is checked
/// against explicit ranking pairs.) Assuming one per depth understates the ceiling
/// by up to 2x as the overlap approaches zero, and the two agree ONLY at a
/// perfect prefix (`overlap == depth`) — which is exactly the shape a
/// perfect-ranking test exercises, so that blind spot is easy to keep.
///
/// Two deliberate simplifications, both valid but worth knowing:
///
/// * Webber treats lists of UNEQUAL length separately. An engine that returns
///   fewer than `depth` ids is scored here as if it had returned `depth` (the
///   even case), so a short response is charged as a ranking deficiency rather
///   than normalized away. That is the honest reading for a benchmark — the
///   engine really did fail to supply an answer — but it makes `raw` and `ext`
///   read lower than the paper's uneven-case numbers for such a query. Those
///   queries are counted separately as `short_returns`.
/// * `res` assumes both rankings continue indefinitely. When a ground truth
///   genuinely ENDS (the short bucket: the corpus holds no more correct
///   answers), the overlap can never grow again, the true value equals `min`
///   exactly, and the whole residual is slack. So the widest band in a summary
///   often sits on the queries with the least real uncertainty.
///
/// `tail(d)` is [`rbo_tail_factor`] at depth `d`. Taken as a closure so the
/// caller can memoize it: this needs the factor at `f` as well as at `depth`,
/// and `f` moves with the data.
fn rbo_bounds_with_tail(
    truncated: f64,
    overlap: u32,
    depth: usize,
    p: f64,
    tail: impl Fn(usize) -> f64,
) -> (f64, f64, f64) {
    // `p` must be STRICTLY inside the unit interval. Written out rather than
    // as `(0.0..1.0).contains(&p)`, which admits `p == 0.0` because a Rust
    // range includes its start — and at `p == 0` the tail factor is `0.0/0.0`,
    // so `min` became a NaN in the published summary, which is precisely what
    // this guard exists to stop. Unreachable through the config, which rejects
    // the endpoints, but `QueryConfig` is public and `run()` does not
    // re-validate.
    if depth == 0 || !(p > 0.0 && p < 1.0) {
        return (truncated, truncated, 0.0);
    }
    // `f` reaches `2 * depth`, so both exponents are clamped: a `top_k` past
    // `i32::MAX / 2` would otherwise wrap NEGATIVE and give `p^-n`, a huge
    // number, rather than failing.
    let exp = |n: usize| -> i32 { n.min(i32::MAX as usize) as i32 };
    let p_depth = p.powi(exp(depth));
    let tail_depth = tail(depth);
    let overlap = (overlap as usize).min(depth);
    // Where the two-per-depth growth meets the `X_d <= d` ceiling.
    let f = 2 * depth - overlap;
    let res = 2.0 * p_depth - p.powi(exp(f)) + (1.0 - p) * f as f64 * tail(f)
        - 2.0 * depth as f64 * (1.0 - p) * tail_depth;
    (
        truncated + (1.0 - p) * overlap as f64 * tail_depth,
        truncated + (overlap as f64 / depth as f64) * p_depth,
        res.max(0.0),
    )
}

/// One dispatch exactly as it came back, before anything has been scored.
///
/// This is what crosses the channel from the workers to the collector thread:
/// the worker does no scoring at all, so the only CPU it spends between a
/// response arriving and the next request going out is a channel send.
#[derive(Debug)]
pub struct RawDispatch {
    /// Seconds since the run started, stamped at dispatch COMPLETION — the
    /// same moment the latency sample exists. Carried rather than recomputed,
    /// because the collector may be behind the dispatch that produced it.
    pub t_s: f64,
    /// Which loaded queries this dispatch fired, positionally paired with
    /// `out.ids`.
    pub idxs: Vec<usize>,
    pub out: crate::targets::BatchOutcome,
}

/// One batch dispatch's observation, built by the collector from a worker's
/// [`RawDispatch`] (and forwarded verbatim to a configured [`Recorder`](crate::report::Recorder) —
/// this IS the time-series row). `latency_ms`/`ok` describe the one
/// round-trip; `recalls` holds 0..N values, one per query in the batch that
/// had both ground truth and returned ids.
#[derive(Debug, Clone)]
pub struct DispatchSample {
    /// Seconds since the run started, stamped at dispatch COMPLETION (the
    /// same moment the latency sample exists) in the worker and carried on the
    /// [`RawDispatch`] — not at collector receive time, which could lag behind
    /// under load.
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
    /// how many queries were excluded from recall for this reason, separately
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
    rbo_p: RboP,
    t_s: f64,
    // Which positions in this batch to score. A query is scored at its FIRST
    // firing only: against a static collection a repeat returns the same
    // documents, so a second scoring is a copy, not an observation. Latency
    // stays per firing; correctness is per query.
    score: &[bool],
) -> DispatchSample {

    // A query scores recall only when the dispatch returned ids for it AND it
    // had ground truth. `recall_at_k` returning `None` there means the ground
    // truth was present but empty — count those separately rather than lose them.
    let mut recalls = Vec::new();
    let mut empty_ground_truth = 0u64;
    let mut filter_overreturn = 0u64;
    let mut short_returns = 0u64;
    for (pos, (&i, ids)) in idxs.iter().zip(out.ids.iter()).enumerate() {
        if !score.get(pos).copied().unwrap_or(false) {
            continue;
        }
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
                vectors[i].gt_cutoff_ties.as_ref(),
                vectors[i].gt_depth,
                vectors[i].gt_cutoff,
                top_k,
                tie_epsilon,
                engine_higher_is_better,
                rbo_p,
            ) {
                Some(sample) => recalls.push(sample),
                None => empty_ground_truth += 1,
            }
        }
    }
    let missing_from_gt = recalls.iter().map(|r| r.missing_from_gt as u64).sum();
    DispatchSample {
        t_s,
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

/// Everything the collector hands back: the raw latency distribution and
/// per-query samples, plus the counts the summary is built from. In the order
/// `run_storm` destructures them.
type Collected = (
    Vec<f64>,
    Vec<RecallSample>,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
);

/// The collector loop, run on its own OS thread (see `run_storm`). Blocks on
/// the channel rather than awaiting it: this thread is not a runtime worker,
/// so parking it costs the load nothing.
///
/// `score` turns one dispatch plus its scoring mask into a sample — taken as a
/// closure so the scoring parameters stay in `run_storm`, where they are
/// resolved.
fn collect(
    mut rx: mpsc::UnboundedReceiver<RawDispatch>,
    vectors: &[QueryVector],
    mut writer_tx: Option<std::sync::mpsc::SyncSender<DispatchSample>>,
    window_closed: &std::sync::atomic::AtomicBool,
    score: impl Fn(&RawDispatch, &[bool]) -> DispatchSample,
) -> Collected {
    let mut latencies = Vec::new();
    let mut recalls = Vec::new();
    let (mut empty_gt, mut over_gt, mut short_ret, mut missing_gt) = (0u64, 0u64, 0u64, 0u64);
    let (mut n_ok, mut n_err, mut n_timeout, mut dropped) = (0u64, 0u64, 0u64, 0u64);
    // One flag per loaded query: has it been scored yet? A looped run re-fires
    // the same queries against an unchanged collection, so every firing after
    // the first returns the same documents and would score identically.
    // Measuring once per query keeps the retained samples and the scoring CPU
    // bounded by the QUERY SET rather than by how long the run was left going.
    let mut scored = vec![false; vectors.len()];
    let mut scored_queries = 0u64;
    let mut firings = 0u64;

    while let Some(raw) = rx.blocking_recv() {
        latencies.push(raw.out.latency.as_secs_f64() * 1000.0);
        // Actual firings, not `requests * batch_size`: fixed-work mode TRIMS
        // the last batch, so the product overstates whenever
        // `passes * queries` is not a multiple of `batch_size` — enough to
        // make a single-pass run look looped.
        firings += raw.idxs.len() as u64;
        if raw.out.ok {
            n_ok += 1;
        } else {
            n_err += 1;
            if raw.out.timed_out {
                n_timeout += 1;
            }
        }
        // Claim the queries this dispatch is the first to OBSERVE. A
        // dispatch that errored or timed out carries `None` for every id, so
        // claiming on arrival would burn the query's one scoring opportunity
        // on a response that contained nothing — and every later success
        // would be masked out for the rest of the run. A cold target, an
        // unwarmed collection or an early `timeout_s` would then report no
        // recall at all for a run whose remaining firings were perfect.
        let mask: Vec<bool> = raw
            .idxs
            .iter()
            .enumerate()
            .map(|(pos, &i)| {
                let observed =
                    raw.out.ok && raw.out.ids.get(pos).is_some_and(|ids| ids.is_some());
                let first = observed && !scored[i];
                if first {
                    scored[i] = true;
                    scored_queries += 1;
                }
                first
            })
            .collect();
        // A dispatch that owes nothing still yields a sample: the time-series
        // carries one row per DISPATCH, because latency is per round-trip.
        // The mask makes its scoring a no-op.
        let s = score(&raw, &mask);
        drop(raw);
        recalls.extend(s.recalls.iter().copied());
        empty_gt += s.empty_ground_truth;
        over_gt += s.filter_overreturn;
        short_ret += s.short_returns;
        missing_gt += s.missing_from_gt;
        let Some(wtx) = writer_tx.as_ref() else {
            continue;
        };
        // While the load window is open a full sink is dropped rather than
        // waited on: stalling here would let the backlog of unscored
        // dispatches grow. Once it has closed there is nothing left to
        // protect, and whatever backlog remains drains at CPU speed rather
        // than dispatch speed — dropping then would decimate the trace, and
        // always its LATER rows, the half an operator reads for degradation
        // under sustained load.
        if !window_closed.load(Ordering::Acquire) {
            match wtx.try_send(s) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => dropped += 1,
                // Writer stopped (a record() error disabled it, or it already
                // finished): stop forwarding.
                Err(TrySendError::Disconnected(_)) => writer_tx = None,
            }
        } else if wtx.send(s).is_err() {
            writer_tx = None;
        }
    }
    // Drop the sender so the writer thread's `recv` ends and it runs finish().
    drop(writer_tx);
    (
        latencies, recalls, empty_gt, over_gt, short_ret, missing_gt, n_ok, n_err, n_timeout,
        dropped, scored_queries, firings,
    )
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
    rbo_p: RboP,
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
    let (tx, rx) = mpsc::unbounded_channel::<RawDispatch>();
    let score_vectors = Arc::clone(&vectors);
    let (c_top_k, c_eps, c_cmp, c_hib, c_filtered, c_rbo_p) = (
        top_k,
        cmp.epsilon,
        scores_comparable,
        engine_higher_is_better,
        filtered,
        rbo_p,
    );

    // Hand the (already-`begin()`-ed) recorder to a dedicated OS thread so its
    // blocking writes never land on a runtime worker — see `report::spawn_writer`
    // for why that matters (especially on a 1-vCPU box). The collector forwards
    // to it over a bounded channel and drops-on-full while the load window is
    // open, so a sink slower than dispatch can neither grow memory unbounded
    // nor backpressure the measurement.
    let (writer_tx, writer_handle) = match recorder {
        Some(r) => {
            let (wtx, handle) = crate::report::spawn_writer(r);
            (Some(wtx), Some(handle))
        }
        None => (None, None),
    };
    // Set once the load loop returns. Until then a full report sink is dropped
    // rather than waited on; after it, nothing measured can be perturbed, so
    // the collector waits instead of decimating the tail of the trace.
    let window_closed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let c_window_closed = Arc::clone(&window_closed);

    // Collector: drain dispatches into the raw distributions + counts, score
    // each query's first observed firing, and forward one row per dispatch to
    // the writer. The accumulation here is the authoritative summary and never
    // loses a sample; only the (auxiliary) time-series file does, and only when
    // its sink can't keep up.
    //
    // It runs on its OWN OS THREAD, not as a tokio task. Scoring is CPU work
    // with no `.await` in it, and on a runtime worker a burst of it delays
    // polling every response that completes beside it — a delay that lands in
    // the latency attributed to the database. Off the runtime, scoring happens
    // concurrently with the load, as results arrive, so nothing has to be held
    // until the window closes: memory stays at whatever backlog the scorer is
    // behind by (normally none — scoring is microseconds per query against a
    // millisecond round-trip), the trace streams in arrival order, and there
    // is no scoring phase after the load. The one thing a thread cannot buy is
    // CPU the machine does not have: on a saturated box it still competes for
    // cores, just through the kernel's preemptive scheduler rather than by
    // holding a runtime worker until it finishes.
    //
    // The result comes back over a oneshot so `run_storm` can await it without
    // blocking a runtime worker on `join()`. A panic drops the sender, which is
    // how the failure is detected below.
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let collector = std::thread::Builder::new()
        .name("nova-storm-collector".into())
        .spawn(move || {
            let collected = collect(
                rx,
                &score_vectors,
                writer_tx,
                &c_window_closed,
                |raw: &RawDispatch, mask: &[bool]| {
                    dispatch_sample(
                        &raw.out,
                        &raw.idxs,
                        &score_vectors,
                        c_top_k,
                        c_eps,
                        c_cmp,
                        c_hib,
                        c_filtered,
                        c_rbo_p,
                        raw.t_s,
                        mask,
                    )
                },
            );
            let _ = done_tx.send(collected);
        })
        .expect("failed to spawn the nova-storm collector thread");

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
            &tx,
            &error_reported,
        )
        .await;
    }

    // Drop the last sender so the collector's `recv` loop ends.
    window_closed.store(true, Ordering::Release);
    drop(tx);
    let wall_s = started.elapsed().as_secs_f64();

    let mut collector_failed = false;
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
        scored_queries,
        firings,
    ) = match done_rx.await {
        Ok(collected) => {
            // Already finished: it sent its result as its last act.
            let _ = collector.join();
            collected
        }
        Err(_) => {
            // The sender was dropped without sending, so the thread panicked.
            // It is exiting, so this join is immediate and recovers the panic
            // message for the log.
            let reason = match collector.join() {
                Err(panic) => panic
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".into()),
                Ok(()) => "exited without a result".into(),
            };
            // All scoring happens in this one thread, so swallowing a panic
            // here discards every latency, count and sample the run took and
            // reports `requests: 0, errors: 0, qps: 0` — a plausible-looking
            // "the target answered nothing" that a scripted caller would
            // record as a real result. Say what happened instead.
            tracing::error!(
                "the collector thread failed ({reason}) — every measurement from this run is \
                 lost. The summary is empty because nothing survived, NOT because the target \
                 returned nothing."
            );
            collector_failed = true;
            Default::default()
        }
    };
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
             couldn't keep pace (bounded writer queue full). Every drop happens INSIDE the \
             measured window, so it is paced by the dispatch rate: a faster sink, a lower \
             offered load, or a shorter run all reduce it. Nothing is dropped after the \
             window closes, where the sink is waited on instead. The summary is \
             unaffected"
        );
    }
    let _ = target.close().await;

    StormResults {
        collector_failed,
        scored_queries,
        firings,
        latencies_ms,
        recalls,
        empty_ground_truth,
        short_returns,
        missing_from_gt,
        full_recall_queries: queries,
        ties,
        top_k,
        rbo_p,
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
    tx: &mpsc::UnboundedSender<RawDispatch>,
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
                if let Some(error) = &out.error {
                    report_first_error(&error_reported, error, out.timed_out);
                }
                let _ = tx.send(RawDispatch {
                    t_s: started.elapsed().as_secs_f64(),
                    idxs,
                    out,
                });
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
    tx: &mpsc::UnboundedSender<RawDispatch>,
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
            if let Some(error) = &out.error {
                report_first_error(&error_reported, error, out.timed_out);
            }
            let _ = tx.send(RawDispatch {
                t_s: started.elapsed().as_secs_f64(),
                idxs,
                out,
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

    /// Fails its first `fail_first` dispatches, then answers correctly — a cold
    /// target, an unwarmed collection, or a `timeout_s` that bites before the
    /// cache is hot.
    #[derive(Debug)]
    struct ColdTarget {
        ids: Vec<String>,
        fail_first: std::sync::atomic::AtomicUsize,
    }

    impl std::fmt::Display for ColdTarget {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "cold-mock")
        }
    }

    #[async_trait]
    impl QueryTarget for ColdTarget {
        async fn query_batch(&self, queries: &[&QueryVector]) -> BatchOutcome {
            let remaining = self
                .fail_first
                .fetch_update(
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst,
                    |n| Some(n.saturating_sub(1)),
                )
                .expect("never returns None");
            if remaining > 0 {
                return BatchOutcome {
                    latency: Duration::from_micros(100),
                    ok: false,
                    ids: vec![None; queries.len()],
                    scores: vec![None; queries.len()],
                    error: Some("cold".into()),
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
                gt_cutoff_ties: None,
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
        let results =
            run_storm(target, vectors(), &profile, 10, cmp(None), TEST_P, false, None).await;
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
        let results =
            run_storm(target, vectors(), &profile, 10, cmp(None), TEST_P, false, None).await;
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

    /// Answers each query perfectly on the FIRST time it sees it and wrongly
    /// on every later firing, with a per-query answer (odd queries only ever
    /// half-right). So a collector that scores anything other than each
    /// query's first observed firing — a repeat, or the wrong position in an
    /// interleaved batch — reports a different number.
    #[derive(Debug, Default)]
    struct FirstFiringTarget {
        seen: std::sync::Mutex<HashSet<u32>>,
    }

    impl std::fmt::Display for FirstFiringTarget {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "first-firing-mock")
        }
    }

    #[async_trait]
    impl QueryTarget for FirstFiringTarget {
        async fn query_batch(&self, queries: &[&QueryVector]) -> BatchOutcome {
            let ids = {
                let mut seen = self.seen.lock().unwrap();
                queries
                .iter()
                .map(|q| {
                    let crate::queries::VectorData::Dense(v) = &q.vector else {
                        unreachable!("fixture is dense")
                    };
                    let i = v[0] as u32;
                    let first = seen.insert(i);
                    Some(match (first, i % 2) {
                        (true, 0) => vec!["a".to_string(), "z".to_string()],
                        (true, _) => vec!["a".to_string(), "b".to_string()],
                        (false, _) => vec!["x".to_string(), "y".to_string()],
                    })
                })
                .collect::<Vec<_>>()
            };
            // Yield so concurrent dispatches genuinely interleave.
            tokio::task::yield_now().await;
            BatchOutcome {
                latency: Duration::from_micros(100),
                ok: true,
                scores: vec![None; ids.len()],
                ids,
                error: None,
                timed_out: false,
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scoring_does_not_depend_on_how_dispatches_interleave() {
        // The collector claims and scores dispatches in ARRIVAL order, which
        // concurrency scrambles. The target answers each query right only on
        // its first firing, so the metrics are the same across interleavings
        // ONLY if every query is scored off exactly its first observed firing.
        let truth = |i: usize| QueryVector {
            vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
            ground_truth: Some(gt(&["a", "z"])),
            gt_cutoff: None,
            gt_cutoff_ties: None,
            gt_depth: 2,
            filter_values: HashMap::new(),
        };
        let mut out = Vec::new();
        for concurrency in [1, 4] {
            let target = Arc::new(FirstFiringTarget::default());
            let profile = LoadProfile {
                concurrency,
                duration_s: 0.0,
                target_rps: 0.0,
                batch_size: 2,
                // Fixed work, so both runs fire exactly the same queries.
                passes: 4,
            };
            let vectors: Vec<QueryVector> = (0..6).map(truth).collect();
            let r = run_storm(target, vectors, &profile, 2, cmp(None), TEST_P, false, None).await;
            let s = r.summary();
            out.push((
                s.requests,
                s.scored_queries,
                s.full_recall.map(|b| (b.n, b.mean)),
                s.full_rbo.map(|b| (b.n, b.mean)),
            ));
        }
        assert_eq!(out[0], out[1], "interleaving changed the reported metrics");
        // 3 even queries at 1.0, 3 odd at 0.5; any later firing would score 0.
        assert_eq!(out[0].1, 6);
        assert_eq!(out[0].2, Some((6, 0.75)), "must score first firings only");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_run_without_ground_truth_still_counts_per_query_diagnostics() {
        // No ground truth means no recall, but `short_returns` is still per
        // query and must be tallied off each query's first firing — and the
        // trace must still be one row per dispatch, in arrival order.
        use crate::report::{ReportConfig, ReportFormat};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ts.csv").to_string_lossy().into_owned();
        let mut recorder = ReportConfig {
            format: ReportFormat::Csv,
            path: path.clone(),
        }
        .build();
        recorder.begin().expect("begin");

        let target = Arc::new(MockTarget::ok(vec!["a".into()])); // 1 id, top_k 2
        let profile = LoadProfile {
            concurrency: 1, // serial, so dispatch order is the pass order
            duration_s: 0.0,
            target_rps: 0.0,
            batch_size: 1,
            passes: 3,
        };
        let vectors: Vec<QueryVector> = (0..4)
            .map(|i| QueryVector {
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: None,
                gt_cutoff: None,
                gt_cutoff_ties: None,
                gt_depth: 0,
                filter_values: HashMap::new(),
            })
            .collect();
        let results = run_storm(
            target,
            vectors,
            &profile,
            2,
            cmp(None),
            TEST_P,
            false,
            Some(recorder),
        )
        .await;
        let summary = results.summary();

        assert_eq!(summary.requests, 12, "4 queries x 3 passes all fired");
        assert!(results.recalls.is_empty(), "nothing to score without gt");
        assert_eq!(
            summary.short_returns, 4,
            "short returns are per QUERY (first firing), and must not be lost \
             when nothing is scorable"
        );

        let text = std::fs::read_to_string(&path).expect("csv written");
        let t: Vec<f64> = text
            .lines()
            .skip(1)
            .map(|l| l.split(',').next().unwrap().parse().unwrap())
            .collect();
        assert_eq!(results.dropped_samples, 0);
        assert_eq!(t.len(), 12);
        // Serial dispatch, so arrival order IS time order.
        assert!(
            t.windows(2).all(|w| w[0] <= w[1]),
            "the trace must stream in arrival order: {t:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_failed_first_firing_does_not_burn_the_query() {
        // Scoring once per query only works if the one scoring goes to a firing
        // that actually SAW something. Claiming on arrival would spend it on an
        // error — which carries `None` for every id — and mask out every later
        // success, so a run whose remaining firings were perfect would report
        // no recall at all. Warm-up 5xx, cold collection, early `timeout_s`.
        let target = Arc::new(ColdTarget {
            ids: vec!["a".into(), "z".into()],
            fail_first: std::sync::atomic::AtomicUsize::new(4),
        });
        let profile = LoadProfile {
            concurrency: 1, // serial, so the first 4 dispatches ARE the failures
            duration_s: 0.0,
            target_rps: 0.0,
            batch_size: 1,
            passes: 10,
        };
        let vectors: Vec<QueryVector> = (0..4)
            .map(|i| QueryVector {
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: Some(gt(&["a", "z"])),
                gt_cutoff: None,
                gt_cutoff_ties: None,
                gt_depth: 2,
                filter_values: HashMap::new(),
            })
            .collect();
        let results = run_storm(target, vectors, &profile, 2, cmp(None), TEST_P, false, None).await;
        let summary = results.summary();

        assert_eq!(summary.requests, 40);
        assert_eq!(summary.errors, 4, "every query's FIRST firing failed");
        assert_eq!(
            summary.scored_queries, 4,
            "all four must still be scored, off a later successful firing"
        );
        assert_eq!(
            summary.full_recall.map(|b| (b.n, b.mean)),
            Some((4, 1.0)),
            "the 36 perfect responses must be what gets reported"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_trimmed_final_batch_is_not_mistaken_for_looping() {
        // 10 queries in batches of 4 = 3 dispatches (4, 4, 2): every query
        // fires exactly once. `requests * batch_size` is 12, so sizing the
        // disclosure off the product claims repeats that never happened.
        let target = Arc::new(MockTarget::ok(vec!["a".into(), "b".into()]));
        let profile = LoadProfile {
            concurrency: 1,
            duration_s: 0.0,
            target_rps: 0.0,
            batch_size: 4,
            passes: 1,
        };
        let vectors: Vec<QueryVector> = (0..10)
            .map(|i| QueryVector {
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: Some(gt(&["a", "z"])),
                gt_cutoff: None,
                gt_cutoff_ties: None,
                gt_depth: 2,
                filter_values: HashMap::new(),
            })
            .collect();
        let results = run_storm(target, vectors, &profile, 2, cmp(None), TEST_P, false, None).await;
        let summary = results.summary();

        assert_eq!(summary.requests, 3, "trimmed: 4 + 4 + 2");
        assert_eq!(summary.firings, 10, "not 3 x 4");
        assert_eq!(summary.scored_queries, 10);
        assert!(
            !summary.to_string().contains("scoring_basis"),
            "nothing looped, so there is nothing to disclose: {summary}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn qps_counts_the_queries_actually_fired() {
        // 10 queries in batches of 4 is 3 dispatches carrying 10 queries, not
        // 12. Crediting the trimmed batch inflates the headline throughput
        // number by 20% here.
        let target = Arc::new(MockTarget::ok(vec!["a".into()]));
        let profile = LoadProfile {
            concurrency: 1,
            duration_s: 0.0,
            target_rps: 0.0,
            batch_size: 4,
            passes: 1,
        };
        let vectors: Vec<QueryVector> = (0..10)
            .map(|i| QueryVector {
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: None,
                gt_cutoff: None,
                gt_cutoff_ties: None,
                gt_depth: 0,
                filter_values: HashMap::new(),
            })
            .collect();
        let results = run_storm(target, vectors, &profile, 2, cmp(None), TEST_P, false, None).await;
        let summary = results.summary();
        assert_eq!(summary.requests, 3);
        assert_eq!(summary.firings, 10);
        let expected = 10.0 / (summary.requests as f64 / summary.requests_per_sec);
        assert!(
            (summary.qps - expected).abs() < 1e-6,
            "qps {} should be firings/wall, not {} ",
            summary.qps,
            summary.requests_per_sec * 4.0
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_batched_run_says_its_latency_is_per_round_trip() {
        // p95 over 8-query round-trips is not per-query service time, and the
        // page must say so on the config a correctness run actually uses:
        // `passes: 1`, where nothing repeats and the scoring_basis line is
        // (correctly) silent.
        let target = Arc::new(MockTarget::ok(vec!["a".into(), "b".into()]));
        let profile = LoadProfile {
            concurrency: 1,
            duration_s: 0.0,
            target_rps: 0.0,
            batch_size: 8,
            passes: 1,
        };
        let vectors: Vec<QueryVector> = (0..16)
            .map(|i| QueryVector {
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: Some(gt(&["a", "z"])),
                gt_cutoff: None,
                gt_cutoff_ties: None,
                gt_depth: 2,
                filter_values: HashMap::new(),
            })
            .collect();
        let results = run_storm(target, vectors, &profile, 2, cmp(None), TEST_P, false, None).await;
        let summary = results.summary();
        let text = summary.to_string();
        assert_eq!(summary.firings, summary.scored_queries, "nothing repeated");
        assert!(
            !text.contains("scoring_basis"),
            "nothing looped, so that line stays quiet: {text}"
        );
        assert!(
            text.contains("latency_basis") && text.contains("carrying 8 queries"),
            "but the batch caveat must still print: {text}"
        );

        // ...and not when there is no batch to caveat.
        let target = Arc::new(MockTarget::ok(vec!["a".into(), "b".into()]));
        let unbatched = LoadProfile {
            batch_size: 1,
            ..profile
        };
        let vectors: Vec<QueryVector> = (0..16)
            .map(|i| QueryVector {
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: Some(gt(&["a", "z"])),
                gt_cutoff: None,
                gt_cutoff_ties: None,
                gt_depth: 2,
                filter_values: HashMap::new(),
            })
            .collect();
        let plain = run_storm(
            target, vectors, &unbatched, 2, cmp(None), TEST_P, false, None,
        )
        .await;
        assert!(!plain.summary().to_string().contains("latency_basis"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_all_empty_ground_truth_set_is_tallied_once_per_query() {
        // `Some(vec![])` is "this query has no correct answer", tallied as
        // recall_empty_gt and scored by nobody — once per query, not once per
        // firing, with the trace still one row per dispatch in order.
        use crate::report::{ReportConfig, ReportFormat};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ts.csv").to_string_lossy().into_owned();
        let mut recorder = ReportConfig {
            format: ReportFormat::Csv,
            path: path.clone(),
        }
        .build();
        recorder.begin().expect("begin");

        let target = Arc::new(MockTarget::ok(vec!["a".into(), "b".into()]));
        let profile = LoadProfile {
            concurrency: 1, // serial, so dispatch order is the pass order
            duration_s: 0.0,
            target_rps: 0.0,
            batch_size: 1,
            passes: 2,
        };
        let vectors: Vec<QueryVector> = (0..3)
            .map(|i| QueryVector {
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: Some(HashMap::new()),
                gt_cutoff: None,
                gt_cutoff_ties: None,
                gt_depth: 0,
                filter_values: HashMap::new(),
            })
            .collect();
        let results = run_storm(
            target,
            vectors,
            &profile,
            2,
            cmp(None),
            TEST_P,
            false,
            Some(recorder),
        )
        .await;
        let summary = results.summary();
        assert!(results.recalls.is_empty());
        assert_eq!(
            summary.empty_ground_truth, 3,
            "tallied once per query, not per firing"
        );

        let text = std::fs::read_to_string(&path).expect("csv written");
        let t: Vec<f64> = text
            .lines()
            .skip(1)
            .map(|l| l.split(',').next().unwrap().parse().unwrap())
            .collect();
        assert_eq!(results.dropped_samples, 0);
        assert_eq!(t.len(), 6);
        assert!(
            t.windows(2).all(|w| w[0] <= w[1]),
            "the trace must stream in arrival order: {t:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn per_query_diagnostics_state_their_basis_without_a_recall_line() {
        // `short_returns` is per QUERY under schema 3, and it prints with no
        // ground truth at all — where there is no recall line to hang the
        // basis off. 4 short returns beside 40 firings reads as 10% unless the
        // page says the denominator is the query set: here EVERY firing
        // returned short.
        let target = Arc::new(MockTarget::ok(vec!["a".into()])); // 1 id, top_k 2
        let profile = LoadProfile {
            concurrency: 2,
            duration_s: 0.0,
            target_rps: 0.0,
            batch_size: 1,
            passes: 10,
        };
        let vectors: Vec<QueryVector> = (0..4)
            .map(|i| QueryVector {
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: None,
                gt_cutoff: None,
                gt_cutoff_ties: None,
                gt_depth: 0,
                filter_values: HashMap::new(),
            })
            .collect();
        let results = run_storm(target, vectors, &profile, 2, cmp(None), TEST_P, false, None).await;
        let summary = results.summary();
        let text = summary.to_string();

        assert_eq!(summary.firings, 40);
        assert_eq!(summary.short_returns, 4, "one per query, not per firing");
        assert!(!text.contains("recall@"), "no ground truth: {text}");
        assert!(
            text.contains("scoring_basis") && text.contains("4 distinct queries out of 40"),
            "the basis has to print for a page whose only numbers are per-query: {text}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_summary_stamps_the_current_schema_version() {
        // Schema 3 = scoring once per query. A consumer reading a sweep
        // parquet cannot otherwise tell a bucket `n` counting queries from one
        // counting firings, so an accidental revert of the stamp is worse than
        // an accidental revert of the behaviour.
        let target = Arc::new(MockTarget::ok(vec!["a".into(), "b".into()]));
        let profile = LoadProfile {
            concurrency: 1,
            duration_s: 0.0,
            target_rps: 0.0,
            batch_size: 1,
            passes: 1,
        };
        let vectors: Vec<QueryVector> = (0..2)
            .map(|i| QueryVector {
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: Some(gt(&["a", "z"])),
                gt_cutoff: None,
                gt_cutoff_ties: None,
                gt_depth: 2,
                filter_values: HashMap::new(),
            })
            .collect();
        let results = run_storm(target, vectors, &profile, 2, cmp(None), TEST_P, false, None).await;
        let summary = results.summary();
        assert_eq!(summary.schema_version, 3);
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&summary).unwrap()).unwrap();
        assert_eq!(json["schema_version"], 3);
        assert_eq!(json["scored_queries"], 2, "serialized for sweep");
        assert_eq!(json["firings"], 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_disclosure_is_silent_when_nothing_was_scored() {
        // No ground truth: no recall line is printed, so a line explaining the
        // BASIS of the recall figures describes something that is not there.
        let target = Arc::new(MockTarget::ok(vec!["a".into(), "b".into()]));
        let profile = LoadProfile {
            concurrency: 2,
            duration_s: 0.0,
            target_rps: 0.0,
            batch_size: 1,
            passes: 5,
        };
        let results = run_storm(
            target,
            vectors(),
            &profile,
            2,
            cmp(None),
            TEST_P,
            false,
            None,
        )
        .await;
        let text = results.summary().to_string();
        assert!(!text.contains("recall@"), "no gt, no recall: {text}");
        assert!(!text.contains("scoring_basis"), "{text}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_looped_run_scores_each_query_once_and_says_so() {
        // The property that makes a looped run safe: scoring, buffered memory
        // and retained samples are bounded by the QUERY SET, not by how long
        // the run was left going. And the summary has to say so, because a
        // reader seeing the firing count beside a recall figure will otherwise
        // assume that many measurements.
        let target = Arc::new(MockTarget::ok(vec!["a".into(), "b".into()]));
        let profile = LoadProfile {
            concurrency: 2,
            duration_s: 0.0,
            target_rps: 0.0,
            batch_size: 1,
            passes: 40,
        };
        let vectors: Vec<QueryVector> = (0..5)
            .map(|i| QueryVector {
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: Some(gt(&["a", "z"])),
                gt_cutoff: None,
                gt_cutoff_ties: None,
                gt_depth: 2,
                filter_values: HashMap::new(),
            })
            .collect();
        let results = run_storm(target, vectors, &profile, 2, cmp(None), TEST_P, false, None).await;
        let summary = results.summary();

        assert_eq!(summary.requests, 200, "5 queries x 40 passes all fired");
        assert_eq!(results.recalls.len(), 5, "but scored once each");
        assert_eq!(summary.scored_queries, 5);
        assert_eq!(
            summary.full_recall.map(|b| b.n),
            Some(5),
            "the recall bucket counts queries, not firings"
        );
        let text = summary.to_string();
        assert!(
            text.contains("scoring_basis") && text.contains("5 distinct queries"),
            "a looped run must disclose the basis: {text}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_single_fire_run_does_not_print_the_looped_disclosure() {
        // One firing per query: firings == scored, so there is nothing to
        // disclose and the line would be noise.
        let target = Arc::new(MockTarget::ok(vec!["a".into(), "b".into()]));
        let profile = LoadProfile {
            concurrency: 2,
            duration_s: 0.0,
            target_rps: 0.0,
            batch_size: 1,
            passes: 1,
        };
        let vectors: Vec<QueryVector> = (0..6)
            .map(|i| QueryVector {
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: Some(gt(&["a", "z"])),
                gt_cutoff: None,
                gt_cutoff_ties: None,
                gt_depth: 2,
                filter_values: HashMap::new(),
            })
            .collect();
        let summary = run_storm(target, vectors, &profile, 2, cmp(None), TEST_P, false, None)
            .await
            .summary();
        assert_eq!(summary.requests, 6);
        assert_eq!(summary.scored_queries, 6);
        assert!(!summary.to_string().contains("scoring_basis"), "nothing to disclose");
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
                gt_cutoff_ties: None,
                gt_depth: 4, // matches the 4-id ground truth below
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: if i % 2 == 0 {
                    Some(gt(&["a", "z", "y", "x"]))
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
        let results = run_storm(target, vectors, &profile, 4, cmp(None), TEST_P, false, None).await;
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
                gt_cutoff_ties: None,
                gt_depth: 1, // matches the 1-id ground truth on this fixture
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: Some(gt(&["a"])),
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
        let results = run_storm(target, vectors, &profile, 1, cmp(None), TEST_P, false, None).await;
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
                gt_cutoff_ties: None,
                gt_depth: 2,
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                // even: real ground truth (recall@2 = 1/2); odd: present-but-empty.
                ground_truth: Some(if i % 2 == 0 {
                    gt(&["a", "zzz"])
                } else {
                    HashMap::new()
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
        let results = run_storm(target, vectors, &profile, 2, cmp(None), TEST_P, false, None).await;
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
                gt_cutoff_ties: None,
                    gt_depth: 1, // matches the 1-id ground truth on this fixture
                    vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                    ground_truth: Some(gt(&["a"])), // 1 id, "a" is a hit
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
        let filtered =
            run_storm(target, vectors(), &profile, 5, cmp(None), TEST_P, true, None).await;
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
        let unfiltered =
            run_storm(target, vectors(), &profile, 5, cmp(None), TEST_P, false, None).await;
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
        let ground_truth = gt(&["a", "b", "z", "y"]);
        // gt has 4 ids (>= k), so denominator is k, not gt len or returned len.
        // 2 of the 3 returned ids are in ground_truth, k=4 -> 2/4.
        let r =
            recall_at_k(&returned, None, &ground_truth, None, 4, None, 4, 2e-4, true,
                TEST_P).unwrap();
        assert_eq!((r.recall, r.short), (0.5, false));
        // k=2 (<= gt len) still divides by k -> 2/2 = 1.0, still full-depth.
        let r =
            recall_at_k(&returned, None, &ground_truth, None, 4, None, 2, 2e-4, true,
                TEST_P).unwrap();
        assert_eq!((r.recall, r.short), (1.0, false));
        // no returned ids -> 0 hits over k, still a real (full-depth) sample.
        let r =
            recall_at_k(&[], None, &ground_truth, None, 4, None, 4, 2e-4, true, TEST_P).unwrap();
        assert_eq!((r.recall, r.short), (0.0, false));
    }

    #[test]
    fn recall_at_k_short_ground_truth_divides_by_its_own_length_and_is_flagged() {
        // gt has 2 ids but k=4 -> can never fill k. Score against gt len (2),
        // not k, and flag it short so the summary keeps it in its own bucket.
        let returned = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let ground_truth = gt(&["a", "b"]);
        let r =
            recall_at_k(&returned, None, &ground_truth, None, 2, None, 4, 2e-4, true,
                TEST_P).unwrap();
        assert_eq!((r.recall, r.short), (1.0, true)); // 2 hits / 2 gt, NOT 2/4
        // one hit of the two -> 1/2, still short.
        let one = vec!["a".to_string(), "zzz".to_string()];
        let r =
            recall_at_k(&one, None, &ground_truth, None, 2, None, 4, 2e-4, true, TEST_P).unwrap();
        assert_eq!((r.recall, r.short), (0.5, true));
    }

    #[test]
    fn recall_at_k_empty_ground_truth_is_no_sample_not_a_nan() {
        // Dividing by an empty gt's length would be NaN and poison the mean;
        // an empty (or absent) ground truth is simply "nothing to measure".
        let returned = vec!["a".to_string()];
        assert!(
            recall_at_k(&returned, None, &HashMap::new(), None, 0, None, 4, 2e-4, true,
                TEST_P).is_none()
        );
    }

    #[test]
    fn recall_at_k_dedupes_returned_so_a_repeated_id_cannot_exceed_1_0() {
        // "a" appears 3 times in `returned` -- must still count as a single
        // hit, not 3, or recall would read 1.5 for k=2 (impossible for a
        // fraction that's supposed to be capped at 1.0).
        let returned = vec!["a".to_string(), "a".to_string(), "a".to_string()];
        let ground_truth = gt(&["a", "b"]);
        assert_eq!(
            recall_at_k(&returned, None, &ground_truth, None, 2, None, 2, 2e-4, true, TEST_P)
                .unwrap()
                .recall,
            0.5
        );
    }

    /// Closed form of RBO over a PERFECT ranking of `depth` ids: every
    /// agreement is 1.0, so the sum telescopes to `1 - p^depth` — the whole
    /// weight except the unobserved residual. Every expectation below is
    /// checked against this rather than a hard-coded decimal, so the tests say
    /// what the metric means instead of what it happened to print.
    fn perfect_rbo(depth: u32, p: f64) -> f64 {
        1.0 - p.powi(depth as i32)
    }

    #[test]
    fn rbo_is_one_minus_the_residual_for_an_exactly_correct_ranking() {
        let truth = gt(&["a", "b", "c"]);
        let returned: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let (rbo, tol, _) = rbo_at_k(&resolve_all(&returned, &truth), 0, 3, RboP(0.5), true);
        assert!((rbo - perfect_rbo(3, 0.5)).abs() < 1e-12, "got {rbo}");
        assert_eq!(rbo, tol, "nothing tied -> the bounds collapse");
    }

    #[test]
    fn rbo_penalizes_order_that_recall_cannot_see() {
        // The SAME set every time, so recall is 1.0 for all three. RBO is the
        // only thing that can tell them apart — the whole reason it exists.
        let truth = gt(&["a", "b", "c"]);
        let of = |ids: [&str; 3]| {
            let v: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
            rbo_at_k(&resolve_all(&v, &truth), 0, 3, RboP(0.5), true).0
        };
        let perfect = of(["a", "b", "c"]);
        let swapped = of(["b", "a", "c"]); // adjacent swap at the top
        let reversed = of(["c", "b", "a"]);
        assert!(
            perfect > swapped && swapped > reversed,
            "perfect {perfect} > swapped {swapped} > reversed {reversed}"
        );
        // Worked by hand: agreements are 0, 1/2, 1 at depths 1..3, weighted
        // (1-p)p^(d-1) = 0.5, 0.25, 0.125.
        assert!((reversed - 0.25).abs() < 1e-12, "got {reversed}");
        // Recall, for contrast, calls all three identical and perfect.
        for ids in [["a", "b", "c"], ["b", "a", "c"], ["c", "b", "a"]] {
            let v: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
            let r = recall_at_k(&v, None, &truth, None, 3, None, 3, 2e-4, true, RboP(0.5)).unwrap();
            assert_eq!(r.recall, 1.0, "recall is a SET measure: {ids:?}");
        }
    }

    #[test]
    fn rbo_charges_more_for_a_deeper_displacement_than_a_shallow_one() {
        // The mechanism: an id returned below its true rank is missing from
        // every prefix in between, so the cost scales with how far it moved.
        // Both responses hold the same set and misplace exactly one pair.
        let truth = gt(&["a", "b", "c", "d", "e", "f"]);
        let of = |ids: [&str; 6]| {
            let v: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
            rbo_at_k(&resolve_all(&v, &truth), 0, 6, RboP(0.9), true).0
        };
        let deep_pair = of(["a", "b", "c", "d", "f", "e"]); // ranks 4 <-> 5
        let top_pair = of(["b", "a", "c", "d", "e", "f"]); // ranks 0 <-> 1
        let far = of(["f", "b", "c", "d", "e", "a"]); // rank 0 <-> 5
        assert!(
            deep_pair > top_pair,
            "a swap deep down must cost less than the same swap at the top: \
             {deep_pair} vs {top_pair}"
        );
        assert!(
            top_pair > far,
            "an adjacent swap must cost less than dragging rank 0 to the bottom: \
             {top_pair} vs {far}"
        );
    }

    #[test]
    fn rbo_tolerant_forgives_reordering_within_a_tie_group_but_exact_does_not() {
        // Ranks 2 and 3 scored identically, so the ground truth's choice of
        // which came first is arbitrary — an engine returning the other order
        // is not wrong, and the UPPER bound must say so.
        let truth = gt_tied(&["a", "b", "c", "d"], &[0, 1, 2, 2]);
        let returned: Vec<String> = ["a", "b", "d", "c"].iter().map(|s| s.to_string()).collect();
        let (rbo, tol, _) = rbo_at_k(&resolve_all(&returned, &truth), 0, 4, RboP(0.9), true);
        assert!(
            (tol - perfect_rbo(4, 0.9)).abs() < 1e-12,
            "a swap inside a tie group is free at the upper bound: {tol}"
        );
        assert!(rbo < tol, "the exact bound still charges for it: {rbo} < {tol}");
    }

    #[test]
    fn rbo_does_not_credit_a_repeated_id_twice() {
        // A target that returns the same id twice has a SHORTER distinct
        // prefix, not a fuller one — charging both positions would let the
        // overlap exceed what was really returned.
        let truth = gt(&["a", "b"]);
        let repeated: Vec<String> = ["a", "a"].iter().map(|s| s.to_string()).collect();
        let honest: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let (dup, _, _) = rbo_at_k(&resolve_all(&repeated, &truth), 0, 2, RboP(0.9), true);
        let (clean, _, _) = rbo_at_k(&resolve_all(&honest, &truth), 0, 2, RboP(0.9), true);
        assert!(dup < clean, "{dup} < {clean}");
        // Depth 1 agrees (both returned "a"); depth 2 is 1/2, not 2/2.
        let expected = 0.1 * 1.0 + 0.1 * 0.9 * 0.5;
        assert!((dup - expected).abs() < 1e-12, "got {dup}");
    }

    #[test]
    fn rbo_is_zero_when_nothing_returned_is_in_the_ground_truth() {
        let truth = gt(&["a", "b"]);
        let returned: Vec<String> = ["y", "z"].iter().map(|s| s.to_string()).collect();
        assert_eq!(rbo_at_k(&resolve_all(&returned, &truth), 0, 2, RboP(0.9), true), (0.0, 0.0, 0));
        // ...and an empty response is the same: no agreement, not a NaN.
        assert_eq!(rbo_at_k(&resolve_all(&[], &truth), 0, 2, RboP(0.9), true), (0.0, 0.0, 0));
    }

    #[test]
    fn a_short_ground_truth_is_compared_only_as_deep_as_it_can_answer() {
        // gt holds 2 ids but top_k is 10. Comparing to depth 10 would score
        // eight depths the ground truth is structurally unable to fill, which
        // is the same unfairness the `short` recall bucket exists to avoid.
        let truth = gt(&["a", "b"]);
        let returned: Vec<String> = ["a", "b", "x", "y"].iter().map(|s| s.to_string()).collect();
        let r =
            recall_at_k(&returned, None, &truth, None, 2, None, 10, 2e-4, true, RboP(0.9)).unwrap();
        assert!(r.short);
        assert!(
            (r.rbo - perfect_rbo(2, 0.9)).abs() < 1e-12,
            "the first two ranks are exactly right, so it is a perfect score \
             AT depth 2: {}",
            r.rbo
        );
    }

    #[test]
    fn rbo_p_controls_how_hard_the_top_of_the_ranking_is_weighted() {
        // One error, at rank 0. A larger p spreads weight deeper, so the same
        // shallow mistake costs proportionally less of the total.
        let truth = gt(&["a", "b", "c", "d"]);
        let returned: Vec<String> = ["z", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let loss = |p: f64| {
            let (rbo, _, _) = rbo_at_k(&resolve_all(&returned, &truth), 0, 4, RboP(p), true);
            // As a fraction of what a perfect ranking scores at that p, so the
            // two are comparable despite having different residuals.
            1.0 - rbo / perfect_rbo(4, p)
        };
        assert!(
            loss(0.5) > loss(0.9),
            "top-weighted p must punish a rank-0 miss harder: {} vs {}",
            loss(0.5),
            loss(0.9)
        );
    }

    /// RBO for one CONCRETE ground-truth ordering, by the textbook definition:
    /// intersect the two prefixes at every depth. No tie logic whatsoever.
    fn naive_rbo(returned: &[String], truth: &[&str], depth: usize, p: f64) -> f64 {
        (1..=depth)
            .map(|d| {
                let eng: HashSet<&str> = returned.iter().take(d).map(|s| s.as_str()).collect();
                let gt: HashSet<&str> = truth.iter().take(d).copied().collect();
                (1.0 - p) * p.powi(d as i32 - 1) * (eng.intersection(&gt).count() as f64 / d as f64)
            })
            .sum()
    }

    /// Every ordering the ground truth could have chosen among documents it
    /// scored equally — i.e. every ground truth that is EXACTLY as correct as
    /// the one on disk. `groups` lists the ids of each tie group in rank order.
    fn tie_resolutions(groups: &[Vec<&'static str>]) -> Vec<Vec<&'static str>> {
        let mut out = vec![Vec::new()];
        for group in groups {
            let mut perms = vec![Vec::new()];
            // All permutations of this group, grown one position at a time.
            for _ in 0..group.len() {
                let mut next = Vec::new();
                for partial in &perms {
                    for id in group {
                        if !partial.contains(id) {
                            let mut p = partial.clone();
                            p.push(*id);
                            next.push(p);
                        }
                    }
                }
                perms = next;
            }
            out = out
                .iter()
                .flat_map(|prefix| {
                    perms.iter().map(move |perm| {
                        let mut combined = prefix.clone();
                        combined.extend(perm.iter().copied());
                        combined
                    })
                })
                .collect();
        }
        out
    }

    #[test]
    fn rbo_tolerant_is_exactly_the_best_any_tie_resolution_could_score() {
        // THE property the upper bound claims. An earlier implementation
        // charged every member of a tie group to the group's first rank at
        // once — which is not a reordering of anything, since one rank cannot
        // hold several documents, and it credited overlap no real ground truth
        // could produce. Checked here against brute force over every ordering
        // the ground truth was free to choose.
        // (ground truth in rank order, its tie groups, the engine's response)
        type TieCase = (Vec<&'static str>, Vec<Vec<&'static str>>, Vec<&'static str>);
        let cases: Vec<TieCase> = vec![
            (
                vec!["z", "a", "b"],
                vec![vec!["z"], vec!["a", "b"]],
                vec!["a", "b", "z"],
            ),
            (
                vec!["b", "e", "a", "c", "d"],
                vec![vec!["b"], vec!["e", "a", "c", "d"]],
                vec!["c", "d", "e", "a", "b"],
            ),
            (
                vec!["x", "a", "b", "c"],
                vec![vec!["x"], vec!["a", "b", "c"]],
                vec!["c", "x", "a", "b"],
            ),
            (
                vec!["a", "b", "c", "d"],
                vec![vec!["a", "b"], vec!["c", "d"]],
                vec!["b", "d", "a", "c"],
            ),
            (
                vec!["a", "b", "c"],
                vec![vec!["a", "b", "c"]],
                vec!["c", "b", "a"],
            ),
            (
                vec!["a", "b", "c", "d"],
                vec![vec!["a", "b", "c"], vec!["d"]],
                vec!["q", "c", "a", "d"],
            ),
        ];
        for (gt_order, groups, engine) in cases {
            // tied_rank = the rank the group starts at.
            let mut tied = Vec::new();
            let mut start = 0u32;
            for g in &groups {
                for _ in 0..g.len() {
                    tied.push(start);
                }
                start += g.len() as u32;
            }
            let truth = gt_tied(&gt_order, &tied);
            let resp: Vec<String> = engine.iter().map(|s| s.to_string()).collect();
            let depth = gt_order.len();
            for p in [0.3, 0.5, 0.7, 0.9] {
                let (exact, tolerant, _) = rbo_at_k(&resolve_all(&resp, &truth), 0, depth, RboP(p), true);
                let best = tie_resolutions(&groups)
                    .iter()
                    .map(|order| naive_rbo(&resp, order, depth, p))
                    .fold(f64::MIN, f64::max);
                assert!(
                    (tolerant - best).abs() < 1e-12,
                    "gt={gt_order:?} engine={engine:?} p={p}: tolerant {tolerant} != best \
                     attainable {best}"
                );
                assert!(
                    exact <= tolerant + 1e-12,
                    "the lower bound must not exceed the upper: {exact} > {tolerant}"
                );
                // And the exact value is one specific resolution: the one on disk.
                let on_disk = naive_rbo(&resp, &gt_order, depth, p);
                assert!((exact - on_disk).abs() < 1e-12, "{exact} != {on_disk}");
            }
        }
    }

    #[test]
    fn straddling_ties_are_also_exactly_the_best_any_resolution_could_score() {
        // The brute-force test above passes `cutoff_ties: None`, so it never
        // exercised the straddling path — which is how a cap that kept only
        // the first few tail members by position survived. Here the ground
        // truth's tie group extends past the measured depth, and the oracle
        // enumerates every ordering of the WHOLE group, including the members
        // that truncation happened to exclude.
        let p_values = [0.3, 0.5, 0.9];
        // Group {b,c,d,e} all tie; only `b` fits inside depth 2.
        let truth = gt_tied(&["a", "b"], &[0, 1]);
        let extras = ["c", "d", "e"];
        let group = ["b", "c", "d", "e"];
        for engine in [
            vec!["a", "e"],
            vec!["e", "a"],
            vec!["a", "d"],
            vec!["d", "c"],
            vec!["a", "b"],
            vec!["c", "z"],
        ] {
            let resp: Vec<String> = engine.iter().map(|s| s.to_string()).collect();
            for p in p_values {
                let (_, tolerant, _) =
                    rbo_at_k(&resolve_with_ties(&resp, &truth, &extras), 1, 2, RboP(p), true);
                // Every ordering of the tie group is an equally valid ground
                // truth; the best of them is what the upper bound claims.
                let best = tie_resolutions(&[vec!["a"], group.to_vec()])
                    .iter()
                    .map(|order| naive_rbo(&resp, order, 2, p))
                    .fold(f64::MIN, f64::max);
                assert!(
                    (tolerant - best).abs() < 1e-12,
                    "engine={engine:?} p={p}: tolerant {tolerant} != best attainable {best}"
                );
            }
        }
    }

    #[test]
    fn a_ground_truth_repeating_an_id_still_lets_a_perfect_response_reach_1_0() {
        // Ranks are compacted over DISTINCT ids, so a repeat leaves no hole.
        // With a positional depth the response below could never fill rank 1,
        // and a flawless answer scored ~0.73 normalized.
        let mut truth = HashMap::new();
        truth.insert("a".to_string(), GtRank { rank: 0, tied_rank: 0 });
        truth.insert("b".to_string(), GtRank { rank: 1, tied_rank: 1 });
        let resp: Vec<String> = ["a", "b", "x", "y"].iter().map(|s| s.to_string()).collect();
        // gt_depth is POSITIONAL (3: "a", "a", "b") while the map holds 2.
        let s = recall_at_k(&resp, None, &truth, None, 3, None, 10, 2e-4, true, RboP(0.9)).unwrap();
        assert_eq!(s.recall, 1.0, "both distinct ids were returned");
        assert_eq!(s.rbo_depth, 2, "measured over the distinct ids, not the repeat");
        assert!(
            (s.rbo - perfect_rbo(2, 0.9)).abs() < 1e-12,
            "a perfect response must reach the ceiling, got {}",
            s.rbo
        );
    }

    #[test]
    fn loader_output_actually_widens_the_bound_end_to_end() {
        // The straddling-tie allowance was broken for weeks of edits while
        // every test passed, because the kernel tests hand-built `CutoffTies`
        // and the loader tests never scored anything. This one crosses the
        // seam: a real parquet, through `load_query_vectors`, into
        // `recall_at_k`.
        use duckdb::Connection;
        let dir = std::env::temp_dir().join(format!("nova_storm_e2e_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("q.parquet");
        let conn = Connection::open_in_memory().unwrap();
        // Ranks 1 and 2 tie; top_k=2 keeps only `b`, so `c` is an equally
        // correct answer the ground truth happened to exclude. `d` is worse,
        // and its presence is what used to clobber the group start.
        conn.execute_batch(&format!(
            "COPY (SELECT [1.0::FLOAT, 2.0::FLOAT, 3.0::FLOAT] AS embedding, \
             ['a', 'b', 'c', 'd'] AS hit_ids, \
             [3.0::FLOAT, 2.0::FLOAT, 2.0::FLOAT, 1.0::FLOAT] AS hit_scores) \
             TO '{}' (FORMAT PARQUET)",
            file.display()
        ))
        .unwrap();
        let source = crate::config::QuerySource {
            uri: file.display().to_string(),
            column: "embedding".into(),
            limit: 10,
            ground_truth_column: Some("hit_ids".into()),
            ground_truth_score_column: Some("hit_scores".into()),
        };
        let vectors =
            crate::queries::load_query_vectors(&source, crate::config::VectorType::Dense, None, 2, usize::MAX)
                .unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let v = &vectors[0];

        // The engine returned `c` where the ground truth wrote `b`. Same
        // score, so it is not a miss — recall says so, and RBO must agree.
        let returned: Vec<String> = ["a", "c"].iter().map(|s| s.to_string()).collect();
        let sample = recall_at_k(
            &returned,
            None,
            v.ground_truth.as_ref().unwrap(),
            v.gt_cutoff_ties.as_ref(),
            v.gt_depth,
            v.gt_cutoff,
            2,
            2e-4,
            true,
            RboP(0.9),
        )
        .unwrap();
        assert!(
            sample.rbo_tolerant > sample.rbo + 1e-9,
            "swapping one member of the cutoff's tie group for another must widen \
             the bound: got {} – {}",
            sample.rbo,
            sample.rbo_tolerant
        );
        assert!(
            (sample.rbo_tolerant - perfect_rbo(2, 0.9)).abs() < 1e-12,
            "and under the ordering that lists `c` inside top_k the response is \
             perfect, so the upper bound is the ceiling: got {}",
            sample.rbo_tolerant
        );
    }

    #[test]
    fn a_tie_group_straddling_top_k_widens_the_upper_bound_like_recall_does() {
        // The ground truth's top-k cut through a tie group, so which members
        // landed inside was arbitrary. An engine returning the others is
        // equally correct — recall says so via `near_ties`, and the rank
        // agreement upper bound must not disagree with it.
        let truth = gt_tied(&["a", "b", "c"], &[0, 1, 1]);
        let resp: Vec<String> = ["a", "d", "e"].iter().map(|s| s.to_string()).collect();
        let (exact, without, _) = rbo_at_k(&resolve_all(&resp, &truth), 0, 3, RboP(0.9), true);
        let (_, with, _) =
            rbo_at_k(&resolve_with_ties(&resp, &truth, &["d", "e"]), 1, 3, RboP(0.9), true);
        assert!(
            (without - exact).abs() < 1e-12,
            "without the straddling ties there is nothing to forgive"
        );
        assert!(
            (with - perfect_rbo(3, 0.9)).abs() < 1e-12,
            "returning other members of the cutoff's tie group is a perfect \
             answer under some valid ordering, got {with}"
        );
    }

    #[test]
    fn the_bounds_ladder_is_ordered_and_matches_the_closed_form() {
        // raw <= min <= ext <= norm, all from the same observations, differing
        // only in what they assume about depths the run never saw. The `min`
        // values are checked against figures computed independently from
        // Webber's definition by summing the infinite tail directly.
        for (p, depth, expected_min, expected_res) in [
            (0.630_957_344_480_193_4_f64, 10usize, 0.998_0, 0.002_0),
            (0.9, 10, 0.855_6, 0.144_4),
            (0.954_992_586_021_436_1, 100, 0.998_4, 0.001_6),
            (0.98, 100, 0.963_1, 0.036_9),
        ] {
            // A PERFECT ranking: overlap is complete at the measured depth.
            let raw = 1.0 - p.powi(depth as i32);
            let (min, ext, res) = rbo_bounds(raw, depth as u32, depth, p);
            assert!(
                (min - expected_min).abs() < 5e-5,
                "p={p} depth={depth}: min {min} != {expected_min}"
            );
            assert!(
                (res - expected_res).abs() < 5e-5,
                "p={p} depth={depth}: res {res} != {expected_res}"
            );
            // A flawless ranking can reach exactly 1.0 and no more, so the
            // top of the range pins down both numbers at once.
            assert!(
                (min + res - 1.0).abs() < 1e-9,
                "p={p} depth={depth}: min+res = {} must be exactly 1.0",
                min + res
            );
            assert!(ext <= 1.0 + 1e-12 && ext >= min - 1e-12, "{ext} outside [{min}, 1]");
            assert!(raw <= min + 1e-12, "raw {raw} must not exceed min {min}");
            // And the normalized value is the top rung: perfect reads 1.0.
            assert!((raw / (1.0 - p.powi(depth as i32)) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn degenerate_p_yields_no_bounds_rather_than_a_nan() {
        // `QueryConfig.rbo_p` is public and `run()` does not re-validate, so a
        // library caller can hand these in. A NaN here would reach the JSON
        // summary and the sweep parquet. `0.0` is the trap: a Rust range
        // includes its start, so an `(0.0..1.0).contains()` guard let it past
        // and `min` came back NaN.
        for p in [0.0, -0.0, 1.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
            let (min, ext, res) = rbo_bounds(0.5, 3, 10, p);
            assert!(
                min.is_finite() && ext.is_finite() && res.is_finite(),
                "p={p} produced non-finite bounds: {min} / {ext} / {res}"
            );
            assert_eq!((min, ext, res), (0.5, 0.5, 0.0), "p={p} must fall back to the raw value");
        }
    }

    #[test]
    fn max_really_bounds_the_best_case_at_every_overlap() {
        // The residual must hold for a PARTIAL prefix overlap, not just a
        // perfect one. Stepping a depth deeper adds an id to both prefixes, so
        // the intersection can gain two — assuming one understates the ceiling
        // by nearly 2x as the overlap goes to zero, and the two assumptions
        // agree only at `overlap == depth`, which is what a perfect-ranking
        // test pins. Checked against the best case summed out to depth 20000.
        for (depth, p) in [(10usize, 0.9_f64), (10, 0.5), (7, 0.7), (25, 0.95)] {
            for overlap in 0..=depth {
                // Overlap accrues as fast as it can, then holds — the shape the
                // `min`/`res` model is defined against.
                let truncated: f64 = (1..=depth)
                    .map(|d| {
                        (1.0 - p) * p.powi(d as i32 - 1) * (overlap.min(d) as f64 / d as f64)
                    })
                    .sum();
                let (min, _ext, res) = rbo_bounds(truncated, overlap as u32, depth, p);
                // Best case: X_d grows by two per depth until it hits d.
                let best: f64 = truncated
                    + (depth + 1..20_000)
                        .map(|d| {
                            let x = (overlap + 2 * (d - depth)).min(d);
                            (1.0 - p) * p.powi(d as i32 - 1) * (x as f64 / d as f64)
                        })
                        .sum::<f64>();
                assert!(
                    min + res >= best - 1e-9,
                    "depth={depth} p={p} overlap={overlap}: printed max {} is BELOW the \
                     best achievable {best}",
                    min + res
                );
                assert!(
                    min + res <= 1.0 + 1e-9,
                    "depth={depth} p={p} overlap={overlap}: max {} exceeds 1.0",
                    min + res
                );
                // And it stays tight — within a hair of the true best case.
                assert!(
                    min + res - best < 1e-6,
                    "depth={depth} p={p} overlap={overlap}: max {} is loose against {best}",
                    min + res
                );
            }
        }
    }

    #[test]
    fn the_bounds_ladder_holds_for_an_imperfect_ranking_too() {
        // Half the top-4 recovered, so the unseen depths are genuinely
        // uncertain and the rungs must separate in the documented order.
        let truth = gt(&["a", "b", "c", "d"]);
        let returned: Vec<String> = ["a", "b", "y", "z"].iter().map(|s| s.to_string()).collect();
        for p in [0.3, 0.5, 0.9] {
            let (raw, _, overlap) = rbo_at_k(&resolve_all(&returned, &truth), 0, 4, RboP(p), true);
            let (min, ext, res) = rbo_bounds(raw, overlap, 4, p);
            let norm = raw / (1.0 - p.powi(4));
            assert_eq!(overlap, 2, "two of the four are in the ground truth");
            assert!(
                raw <= min && min <= ext,
                "p={p}: ladder out of order — raw {raw}, min {min}, ext {ext}"
            );
            assert!(
                min + res <= 1.0 + 1e-12,
                "p={p}: min+res {} exceeds 1.0",
                min + res
            );
            // Both bounds sit inside the range the residual admits.
            assert!(ext <= min + res + 1e-12, "p={p}: ext {ext} above max {}", min + res);
            assert!(norm >= raw, "p={p}: normalizing cannot lower the value");
        }
    }

    #[test]
    fn rbo_matches_a_naive_per_depth_set_intersection() {
        // `rbo_at_k` charges each id ONCE to the depth where it starts
        // agreeing, instead of intersecting two prefixes at every depth. That
        // is the one piece of non-obvious math here, so it is checked against
        // the obvious quadratic implementation over a spread of shuffles,
        // truncations and repeats — not just the cases picked by hand above.
        fn naive(returned: &[String], truth: &[&str], depth: usize, p: f64) -> f64 {
            let mut rbo = 0.0;
            for d in 1..=depth {
                // Distinct ids, so a repeat shrinks the engine's real prefix.
                let eng: HashSet<&str> =
                    returned.iter().take(d).map(|s| s.as_str()).collect();
                let gt: HashSet<&str> = truth.iter().take(d).copied().collect();
                let x = eng.intersection(&gt).count();
                rbo += (1.0 - p) * p.powi(d as i32 - 1) * (x as f64 / d as f64);
            }
            rbo
        }

        let ids: Vec<String> = (0..12).map(|i| format!("d{i}")).collect();
        let truth_ids: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let truth = gt(&truth_ids);
        // Deterministic LCG — a fixed sweep, reproducible on failure.
        let mut seed = 0x2545_F491u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 16) as usize
        };
        for case in 0..200 {
            // A shuffle of the ground truth, then perturbed: some ids dropped
            // for unknown ones, some repeated, and the response truncated.
            let mut resp: Vec<String> = ids.clone();
            for i in (1..resp.len()).rev() {
                resp.swap(i, next() % (i + 1));
            }
            if case % 3 == 0 {
                let at = next() % resp.len();
                resp[at] = format!("unknown{case}");
            }
            if case % 4 == 0 && resp.len() > 1 {
                resp[0] = resp[1].clone(); // a repeated id
            }
            resp.truncate(1 + next() % ids.len());
            for p in [0.3, 0.7, 0.95] {
                let (fast, tol, _) = rbo_at_k(&resolve_all(&resp, &truth), 0, ids.len(), RboP(p), true);
                let slow = naive(&resp, &truth_ids, ids.len(), p);
                assert!(
                    (fast - slow).abs() < 1e-12,
                    "case {case} p={p}: {fast} vs naive {slow} for {resp:?}"
                );
                assert_eq!(fast, tol, "nothing tied -> bounds collapse");
            }
        }
    }

    #[test]
    fn summary_reports_rbo_beside_recall_and_flags_a_large_residual() {
        let sample = |recall: f64, rbo: f64, short: bool| RecallSample {
            recall,
            tolerant: recall,
            rbo,
            rbo_tolerant: rbo,
            rbo_depth: 10,
            rbo_overlap: 0,
            short,
            missing_from_gt: 0,
        };
        let results = StormResults {
            firings: 2,
            collector_failed: false,
            scored_queries: 1,
            latencies_ms: vec![1.0, 2.0],
            recalls: vec![sample(1.0, 0.40, false), sample(1.0, 0.60, false)],
            empty_ground_truth: 0,
            filter_overreturn: 0,
            short_returns: 0,
            missing_from_gt: 0,
            full_recall_queries: 2,
            ties: None,
            rbo_p: RboP(0.9),
            top_k: 10,
            tie_epsilon: 2e-4,
            tie_epsilon_source: "configured".into(),
            tie_disabled_reason: None,
            scores_configured: false,
            n_ok: 2,
            n_err: 0,
            n_timeout: 0,
            wall_s: 1.0,
            batch_size: 1,
            dropped_samples: 0,
        };
        let summary = results.summary();
        let full = summary.full_rbo.expect("full-depth rbo present");
        assert!((full.mean - 0.5).abs() < 1e-12, "{}", full.mean);
        assert_eq!(full.n, 2);
        assert_eq!(summary.rbo_p, Some(0.9));
        // 0.9^10 ~= 0.349 — well over the 1% floor, so it must be stated.
        let residual = summary.rbo_residual.expect("residual reported");
        assert!((residual - 0.9f64.powi(10)).abs() < 1e-12);
        let text = summary.to_string();
        assert!(text.contains("rbo@10: 0.5000"), "{text}");
        assert!(text.contains("rbo_residual"), "{text}");
        // No score column -> no tolerant bound invented, so no range printed.
        assert!(summary.full_rbo_tolerant.is_none());
        assert!(!text.contains("–"), "{text}");
    }

    #[test]
    fn the_normalized_rbo_is_1_0_for_a_perfect_ranking_at_any_p() {
        // The point of normalizing: the raw ceiling moves with `p`, so a
        // flawless engine scores a different number at every setting, which is
        // indefensible in a results table. Normalized, perfect is 1.0 always.
        let truth = gt(&["a", "b", "c", "d"]);
        let returned: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        for p in [0.3, 0.5, 0.7, 0.9, 0.99] {
            let results = StormResults {
            firings: 1,
                collector_failed: false,
            scored_queries: 1,
            latencies_ms: vec![1.0],
                recalls: vec![
                    recall_at_k(&returned, None, &truth, None, 4, None, 4, 2e-4, true,
                        RboP(p)).unwrap(),
                ],
                empty_ground_truth: 0,
                filter_overreturn: 0,
                short_returns: 0,
                missing_from_gt: 0,
                full_recall_queries: 1,
                ties: None,
                rbo_p: RboP(p),
                top_k: 4,
                tie_epsilon: 2e-4,
                tie_epsilon_source: "configured".into(),
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
            let raw = summary.full_rbo.unwrap().mean;
            let normalized = summary.full_rbo_normalized.unwrap();
            assert!(
                (raw - perfect_rbo(4, p)).abs() < 1e-12,
                "p={p}: raw tracks the ceiling ({raw})"
            );
            assert!(
                (normalized - 1.0).abs() < 1e-12,
                "p={p}: normalized must be exactly 1.0, got {normalized}"
            );
        }
    }

    #[test]
    fn normalization_divides_each_sample_by_its_own_ceiling_not_a_global_one() {
        // A short-ground-truth query was compared at a SHALLOWER depth, so it
        // has a lower ceiling. Rescaling the blended mean by one global
        // ceiling would understate the short bucket; both perfect samples here
        // must normalize to 1.0 despite very different raw values.
        let p = 0.7;
        let deep = RecallSample {
            recall: 1.0,
            tolerant: 1.0,
            rbo: perfect_rbo(10, p),
            rbo_tolerant: perfect_rbo(10, p),
            rbo_depth: 10,
            rbo_overlap: 0,
            short: false,
            missing_from_gt: 0,
        };
        let shallow = RecallSample {
            rbo: perfect_rbo(2, p),
            rbo_tolerant: perfect_rbo(2, p),
            rbo_depth: 2,
            rbo_overlap: 0,
            short: true,
            ..deep
        };
        // The raw values are far apart — that difference is pure truncation.
        assert!(shallow.rbo < deep.rbo * 0.6, "{} vs {}", shallow.rbo, deep.rbo);
        let results = StormResults {
            firings: 2,
            collector_failed: false,
            scored_queries: 1,
            latencies_ms: vec![1.0, 2.0],
            recalls: vec![deep, shallow],
            empty_ground_truth: 0,
            filter_overreturn: 0,
            short_returns: 0,
            missing_from_gt: 0,
            full_recall_queries: 1,
            ties: None,
            rbo_p: RboP(p),
            top_k: 10,
            tie_epsilon: 2e-4,
            tie_epsilon_source: "configured".into(),
            tie_disabled_reason: None,
            scores_configured: false,
            n_ok: 2,
            n_err: 0,
            n_timeout: 0,
            wall_s: 1.0,
            batch_size: 1,
            dropped_samples: 0,
        };
        let summary = results.summary();
        for (label, got) in [
            ("full", summary.full_rbo_normalized),
            ("short", summary.short_rbo_normalized),
            ("total", summary.total_rbo_normalized),
        ] {
            let got = got.unwrap_or_else(|| panic!("{label} bucket missing"));
            assert!(
                (got - 1.0).abs() < 1e-12,
                "{label}: both samples are perfect at their own depth, got {got}"
            );
        }
        // A global rescale would have given the short sample
        // perfect_rbo(2)/perfect_rbo(10) ~= 0.53, dragging the total to ~0.77.
        let global_rescale = perfect_rbo(2, p) / perfect_rbo(10, p);
        assert!(global_rescale < 0.6, "the wrong answer really is wrong: {global_rescale}");
    }

    #[test]
    fn a_short_only_run_states_its_own_depth_and_residual_not_top_k_s() {
        // The selective-filter shape: every ground truth is shallower than
        // top_k, so there is no full bucket at all. The residual must describe
        // the depth actually measured — quoting `p^top_k` here understated the
        // truncation gap by 25x — and the one RBO line printed must still
        // carry the `p` that defines it.
        let p = 0.630_957_344_480_193_4; // the derived default at top_k=10
        let sample = RecallSample {
            recall: 1.0,
            tolerant: 1.0,
            rbo: perfect_rbo(3, p),
            rbo_tolerant: perfect_rbo(3, p),
            rbo_depth: 3,
            rbo_overlap: 0,
            short: true,
            missing_from_gt: 0,
        };
        let results = StormResults {
            firings: 1,
            collector_failed: false,
            scored_queries: 1,
            latencies_ms: vec![1.0],
            recalls: vec![sample],
            empty_ground_truth: 0,
            filter_overreturn: 0,
            short_returns: 0,
            missing_from_gt: 0,
            full_recall_queries: 0,
            ties: None,
            rbo_p: RboP(p),
            top_k: 10,
            tie_epsilon: 2e-4,
            tie_epsilon_source: "configured".into(),
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
        assert!(summary.full_rbo.is_none(), "no full-depth query ran");
        assert!(
            summary.rbo_residual.is_none(),
            "no full-depth query ran, so that bucket has no residual to report"
        );
        let residual = summary
            .short_rbo_residual
            .expect("the short bucket's residual is reported");
        assert!(
            (residual - p.powi(3)).abs() < 1e-12,
            "residual must describe depth 3, the depth measured — got {residual}, \
             p^top_k would be {}",
            p.powi(10)
        );
        assert!(residual > 0.2, "depth 3 really does leave a lot unobserved: {residual}");
        // Perfect at its own depth, so the normalized value says 1.0 even
        // though the raw value is far below it.
        let normalized = summary.short_rbo_normalized.expect("normalized short bucket");
        assert!((normalized - 1.0).abs() < 1e-12, "got {normalized}");

        let text = summary.to_string();
        assert!(text.contains("rbo_short"), "{text}");
        assert!(
            !text.contains("rbo@10_short"),
            "nothing here was measured at depth 10: {text}"
        );
        assert!(text.contains("p=0.631"), "every rbo line states its p: {text}");
        assert!(
            text.contains("rbo_short_residual"),
            "a 25% residual must be stated, against the bucket it describes: {text}"
        );
        assert!(
            !text.contains("\n    rbo_residual"),
            "and not against the full bucket, which scored nothing: {text}"
        );
    }

    #[test]
    fn the_residual_line_stays_quiet_at_the_derived_default() {
        // The derived `p` puts the residual ON the 1% target, to within a ULP.
        // A bare `>= 1%` test decided by rounding — printing for some `top_k`
        // and not others — so the bar sits clear of it.
        for top_k in [1u64, 5, 10, 20, 50, 100, 137, 1000] {
            let p = crate::config::default_rbo_p_for(top_k);
            let results = StormResults {
            firings: 1,
                collector_failed: false,
            scored_queries: 1,
            latencies_ms: vec![1.0],
                recalls: vec![RecallSample {
                    recall: 1.0,
                    tolerant: 1.0,
                    rbo: 0.5,
                    rbo_tolerant: 0.5,
                    rbo_depth: top_k as u32,
                    rbo_overlap: 0,
                    short: false,
                    missing_from_gt: 0,
                }],
                empty_ground_truth: 0,
                filter_overreturn: 0,
                short_returns: 0,
                missing_from_gt: 0,
                full_recall_queries: 1,
                ties: None,
                rbo_p: RboP(p),
                top_k,
                tie_epsilon: 2e-4,
                tie_epsilon_source: "configured".into(),
                tie_disabled_reason: None,
                scores_configured: false,
                n_ok: 1,
                n_err: 0,
                n_timeout: 0,
                wall_s: 1.0,
                batch_size: 1,
                dropped_samples: 0,
            };
            assert!(
                !results.summary().to_string().contains("rbo_residual"),
                "top_k={top_k}: the derived default's own residual must not trip the line"
            );
        }
    }

    #[test]
    fn rbo_upper_bound_survives_incomparable_engine_scores() {
        // A quantized collection queried with rescore=false withholds every
        // recall-side tolerant number, because those compare an ENGINE score
        // to a ground-truth one. RBO's upper bound is derived from the ground
        // truth's own scores at load time, so it stays valid — and withholding
        // it here would throw away the only order-aware bound such a run has.
        let results = StormResults {
            firings: 1,
            collector_failed: false,
            scored_queries: 1,
            latencies_ms: vec![1.0],
            recalls: vec![RecallSample {
                recall: 0.5,
                tolerant: 0.5,
                rbo: 0.40,
                rbo_tolerant: 0.55,
                rbo_depth: 10,
                rbo_overlap: 0,
                short: false,
                missing_from_gt: 0,
            }],
            empty_ground_truth: 0,
            filter_overreturn: 0,
            short_returns: 0,
            missing_from_gt: 0,
            full_recall_queries: 1,
            ties: None,
            rbo_p: RboP(0.9),
            top_k: 10,
            tie_epsilon: 2e-4,
            tie_epsilon_source: "configured".into(),
            tie_disabled_reason: Some("scores are in quantized space".into()),
            scores_configured: true,
            n_ok: 1,
            n_err: 0,
            n_timeout: 0,
            wall_s: 1.0,
            batch_size: 1,
            dropped_samples: 0,
        };
        let summary = results.summary();
        assert!(
            summary.full_recall_tolerant.is_none(),
            "recall's upper bound needs comparable engine scores"
        );
        assert_eq!(
            summary.full_rbo_tolerant,
            Some(0.55),
            "RBO's does not — it never looks at an engine score"
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

    /// Ground truth from ids in RANK ORDER, nothing tied (`tied_rank == rank`).
    fn gt(ids: &[&str]) -> HashMap<String, GtRank> {
        ids.iter()
            .enumerate()
            .map(|(rank, id)| {
                (
                    id.to_string(),
                    GtRank {
                        rank: rank as u32,
                        tied_rank: rank as u32,
                    },
                )
            })
            .collect()
    }

    /// Ground truth in rank order where `tied[i]` is each id's tie-group start
    /// — exercises the tie-tolerant bound without a parquet round trip.
    fn gt_tied(ids: &[&str], tied: &[u32]) -> HashMap<String, GtRank> {
        ids.iter()
            .zip(tied)
            .enumerate()
            .map(|(rank, (id, &tied_rank))| {
                (
                    id.to_string(),
                    GtRank {
                        rank: rank as u32,
                        tied_rank,
                    },
                )
            })
            .collect()
    }

    /// Resolve a response against a ground-truth map the way `recall_at_k`
    /// does before calling the kernel, so tests can keep speaking in maps.
    fn resolve_all(returned: &[String], gt: &HashMap<String, GtRank>) -> Vec<Resolved> {
        let mut seen = HashSet::new();
        returned
            .iter()
            .map(|id| match (seen.insert(id.as_str()), gt.get(id.as_str())) {
                (false, _) => Resolved::Repeat,
                (true, Some(r)) => Resolved::Ranked(*r),
                (true, None) => Resolved::Absent,
            })
            .collect()
    }

    /// As `resolve_all`, but marking `ties` as cutoff-tie members.
    fn resolve_with_ties(
        returned: &[String],
        gt: &HashMap<String, GtRank>,
        ties: &[&str],
    ) -> Vec<Resolved> {
        resolve_all(returned, gt)
            .into_iter()
            .zip(returned)
            .map(|(r, id)| match r {
                Resolved::Absent if ties.contains(&id.as_str()) => Resolved::CutoffTie,
                other => other,
            })
            .collect()
    }

    /// RBO persistence for tests that aren't about the weighting itself.
    const TEST_P: RboP = RboP(0.9);

    #[test]
    fn without_scores_the_bounds_collapse_to_exact_recall() {
        let returned = vec!["a".to_string(), "z".to_string()];
        let r =
            recall_at_k(&returned, None, &gt(&["a", "b"]), None, 2, None, 2, 2e-4, true,
                TEST_P).unwrap();
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
            None,
            2,
            Some(cutoff),
            2,
            2e-4,
            true,
            TEST_P,
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
            None,
            2,
            Some(cutoff),
            2,
            2e-4,
            true,
            TEST_P,
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
            None,
            2,
            Some(cutoff),
            2,
            2e-4,
            true,
            TEST_P,
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
            None,
            1,
            Some(cutoff),
            1,
            2e-4,
            true,
            TEST_P,
        )
        .unwrap();
        assert_eq!(loose.tolerant, 1.0, "within tolerance -> tied");
        let tight = recall_at_k(
            &returned,
            Some(&near),
            &gt(&["a"]),
            None,
            1,
            Some(cutoff),
            1,
            1e-9,
            true,
            TEST_P,
        )
        .unwrap();
        assert_eq!(tight.tolerant, 0.0, "outside tolerance -> not tied");
        assert_eq!(tight.missing_from_gt, 1, "and it scored above the cutoff");
    }

    #[test]
    fn summary_display_appends_recall_lines_only_when_present() {
        let base = Summary {
            firings: 10,
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
            full_rbo: None,
            short_rbo: None,
            total_rbo: None,
            full_rbo_tolerant: None,
            short_rbo_tolerant: None,
            total_rbo_tolerant: None,
            full_rbo_normalized: None,
            short_rbo_normalized: None,
            total_rbo_normalized: None,
            full_rbo_normalized_tolerant: None,
            short_rbo_normalized_tolerant: None,
            total_rbo_normalized_tolerant: None,
            rbo_p: None,
            rbo_residual: None,
            short_rbo_residual: None,
            full_rbo_min: None,
            full_rbo_ext: None,
            full_rbo_res: None,
            short_rbo_min: None,
            short_rbo_ext: None,
            short_rbo_res: None,
            short_returns: 0,
            missing_from_gt: 0,
            full_recall_queries: 10,
            scored_queries: 10,
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
        assert!(s.contains("(10 eligible queries, 8 scored)"), "{s}");
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
            None,
            2,
            Some(cutoff),
            2,
            2e-4,
            false,
            TEST_P,
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
            None,
            2,
            Some(cutoff),
            2,
            2e-4,
            false,
            TEST_P,
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
            None,
            2,
            Some(cutoff),
            2,
            2e-4,
            false,
            TEST_P,
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
            firings: 2,
            collector_failed: false,
            scored_queries: 1,
            latencies_ms: vec![1.0, 2.0],
            recalls: vec![
                RecallSample {
                    recall: 0.50,
                    tolerant: 0.90,
                    rbo: 0.50,
                    rbo_tolerant: 0.50,
                    rbo_depth: 10,
                    rbo_overlap: 0,
                    short: false,
                    missing_from_gt: 0,
                },
                RecallSample {
                    recall: 0.40,
                    tolerant: 0.80,
                    rbo: 0.40,
                    rbo_tolerant: 0.40,
                    rbo_depth: 10,
                    rbo_overlap: 0,
                    short: true,
                    missing_from_gt: 0,
                },
            ],
            empty_ground_truth: 0,
            filter_overreturn: 0,
            short_returns: 0,
            missing_from_gt: 0,
            full_recall_queries: 1,
            ties: Some(TieStats { mean: 3.0, max: 5, fraction_of_queries: 1.0, queries: 2 }),
            rbo_p: TEST_P,
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
            firings: 1,
            collector_failed: false,
            scored_queries: 1,
            latencies_ms: vec![1.0],
            recalls: vec![RecallSample {
                recall: 0.5,
                tolerant: 0.5,
                rbo: 0.5,
                rbo_tolerant: 0.5,
                rbo_depth: 10,
                rbo_overlap: 0,
                short: false,
                missing_from_gt: 0,
            }],
            empty_ground_truth: 0,
            filter_overreturn: 0,
            short_returns: 0,
            missing_from_gt: 0,
            full_recall_queries: 1,
            ties: None,
            rbo_p: TEST_P,
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
            None,
            2,
            Some(cutoff),
            2,
            2e-4,
            true,
            TEST_P,
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
            None,
            2,
            Some(known),
            2,
            2e-4,
            true,
            TEST_P,
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
        let r =
            recall_at_k(&returned, None, &gt(&["a", "b"]), None, 3, None, 10, 2e-4, true,
                TEST_P).unwrap();
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

        let sample =
            |gt: Option<HashMap<String, GtRank>>, gt_depth: usize, returned: usize, filtered: bool| {
                let vectors = vec![QueryVector {
                    vector: crate::queries::VectorData::Dense(vec![0.0]),
                    ground_truth: gt,
                    gt_cutoff: None,
                gt_cutoff_ties: None,
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
                    TEST_P,
                    0.0,
                    &[true],
                )
                .short_returns
            };

        let deep: HashMap<String, GtRank> = gt(&[
            "h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9",
        ]);
        let shallow: HashMap<String, GtRank> = gt(&["h0", "h1"]);

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
        assert_eq!(sample(Some(HashMap::new()), 0, 0, false), 0);
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
            None,
            2,
            Some(cutoff),
            2,
            2e-4,
            true,
            TEST_P,
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
            &returned,
            Some(&scores),
            &gt(&["a", "b"]),
            None,
            2,
            Some(cutoff),
            2,
            2e-4,
            true,
            TEST_P,
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
            None,
            2,
            Some(cutoff),
            2,
            2e-4,
            true,
            TEST_P,
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
            None,
            2,
            Some(cutoff),
            2,
            2e-4,
            false,
            TEST_P,
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
        let truth: HashMap<String, GtRank> = (0..9u32)
            .map(|i| {
                (
                    format!("h{i}"),
                    GtRank {
                        rank: i,
                        tied_rank: i,
                    },
                )
            })
            .collect();
        let r =
            recall_at_k(&returned, None, &truth, None, 10, None, 10, 2e-4, true, TEST_P).unwrap();
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
            firings: 1,
            collector_failed: false,
            scored_queries: 1,
            latencies_ms: vec![1.0],
            recalls: vec![RecallSample {
                recall: 0.5,
                tolerant: 0.9,
                rbo: 0.5,
                rbo_tolerant: 0.5,
                rbo_depth: 10,
                rbo_overlap: 0,
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
            rbo_p: TEST_P,
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
            firings: 1,
            collector_failed: false,
            scored_queries: 1,
            latencies_ms: vec![1.0],
            recalls: vec![RecallSample {
                recall: 0.5,
                tolerant: 0.9,
                rbo: 0.5,
                rbo_tolerant: 0.5,
                rbo_depth: 10,
                rbo_overlap: 0,
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
            rbo_p: TEST_P,
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
            firings: 40,
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
            full_rbo: Some(RecallBucket { n: 6, mean: 0.81 }),
            short_rbo: Some(RecallBucket { n: 2, mean: 0.66 }),
            total_rbo: Some(RecallBucket { n: 8, mean: 0.77 }),
            full_rbo_tolerant: Some(0.84),
            short_rbo_tolerant: None,
            total_rbo_tolerant: None,
            full_rbo_normalized: None,
            short_rbo_normalized: None,
            total_rbo_normalized: None,
            full_rbo_normalized_tolerant: None,
            short_rbo_normalized_tolerant: None,
            total_rbo_normalized_tolerant: None,
            rbo_p: Some(0.95),
            rbo_residual: Some(0.0059),
            short_rbo_residual: Some(0.21),
            full_rbo_min: Some(0.8300),
            full_rbo_ext: Some(0.8400),
            full_rbo_res: Some(0.0050),
            short_rbo_min: Some(0.70),
            short_rbo_ext: Some(0.72),
            short_rbo_res: Some(0.18),
            short_returns: 1,
            missing_from_gt: 0,
            full_recall_queries: 8,
            scored_queries: 8,
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
        let gt = Some(gt(&["a", "b"]));
        let vectors = vec![
            QueryVector {
                vector: crate::queries::VectorData::Dense(vec![0.0]),
                ground_truth: gt.clone(),
                gt_cutoff: None,
                gt_cutoff_ties: None,
                gt_depth: 2,
                filter_values: HashMap::new(),
            },
            QueryVector {
                vector: crate::queries::VectorData::Dense(vec![1.0]),
                ground_truth: gt.clone(),
                gt_cutoff: None,
                gt_cutoff_ties: None,
                gt_depth: 2,
                filter_values: HashMap::new(),
            },
            QueryVector {
                vector: crate::queries::VectorData::Dense(vec![2.0]),
                ground_truth: gt,
                gt_cutoff: None,
                gt_cutoff_ties: None,
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
            TEST_P,
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
            firings: 3,
            collector_failed: false,
            scored_queries: 1,
            latencies_ms: vec![1.0, 2.0, 3.0],
            recalls: vec![
                RecallSample {
                    recall: 0.5,
                    tolerant: 0.5,
                    rbo: 0.5,
                    rbo_tolerant: 0.5,
                    rbo_depth: 10,
                    rbo_overlap: 0,
                    short: false,
                    missing_from_gt: 0,
                },
                RecallSample {
                    recall: 1.0,
                    tolerant: 1.0,
                    // Not 1.0: this fixture's depth (10) and `p` (0.9) cap a
                    // raw RBO at 1 - 0.9^10 = 0.6513, and `summary()` asserts
                    // that ceiling in debug builds.
                    rbo: 0.65,
                    rbo_tolerant: 0.65,
                    rbo_depth: 10,
                    rbo_overlap: 0,
                    short: false,
                    missing_from_gt: 0,
                },
                RecallSample {
                    recall: 0.4,
                    tolerant: 0.4,
                    rbo: 0.4,
                    rbo_tolerant: 0.4,
                    rbo_depth: 10,
                    rbo_overlap: 0,
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
            rbo_p: TEST_P,
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
            TEST_P,
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
    async fn a_scored_run_streams_one_row_per_dispatch_in_order() {
        // The test above runs with `vectors()`, which has no ground truth, so
        // nothing is ever scored there. With ground truth, the dispatches that
        // owe a score and the ones that do not must both reach the sink,
        // exactly once each, and in arrival order: scoring happens on the
        // collector thread as results arrive, so nothing is held back to the
        // end of the run and the first pass's rows come FIRST.
        use crate::report::{ReportConfig, ReportFormat};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ts.csv").to_string_lossy().into_owned();
        let mut recorder = ReportConfig {
            format: ReportFormat::Csv,
            path: path.clone(),
        }
        .build();
        recorder.begin().expect("begin");

        let target = Arc::new(MockTarget::ok(vec!["a".into(), "b".into()]));
        let profile = LoadProfile {
            concurrency: 1,
            duration_s: 0.0,
            target_rps: 0.0,
            batch_size: 1,
            passes: 5,
        };
        let vectors: Vec<QueryVector> = (0..8)
            .map(|i| QueryVector {
                vector: crate::queries::VectorData::Dense(vec![i as f32; 4]),
                ground_truth: Some(gt(&["a", "z"])),
                gt_cutoff: None,
                gt_cutoff_ties: None,
                gt_depth: 2,
                filter_values: HashMap::new(),
            })
            .collect();
        let results = run_storm(
            target,
            vectors,
            &profile,
            2,
            cmp(None),
            TEST_P,
            false,
            Some(recorder),
        )
        .await;
        let summary = results.summary();
        assert_eq!(summary.requests, 40);
        assert_eq!(summary.scored_queries, 8);

        let text = std::fs::read_to_string(&path).expect("csv written");
        let rows: Vec<&str> = text.lines().skip(1).collect();
        assert_eq!(
            rows.len() as u64,
            summary.requests - results.dropped_samples,
            "every dispatch writes exactly one row, scored or not"
        );
        // The 8 rows carrying a recall value are the 8 first firings — serial
        // dispatch, so they are the file's FIRST 8.
        let with_recall: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.split(',').nth(3).unwrap_or("").is_empty())
            .map(|(i, _)| i)
            .collect();
        assert_eq!(with_recall.len(), 8, "one scored row per query");
        assert_eq!(
            with_recall,
            (0..8).collect::<Vec<_>>(),
            "the scored rows must be the head of the file, not held to the end"
        );
        let t: Vec<f64> = rows
            .iter()
            .map(|r| r.split(',').next().unwrap().parse().unwrap())
            .collect();
        assert!(t.windows(2).all(|w| w[0] <= w[1]), "time order: {t:?}");
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
            TEST_P,
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

        let gt = Some(gt(&["a", "b"]));
        let vectors: Vec<QueryVector> = (0..9)
            .map(|i| QueryVector {
                gt_cutoff: None,
                gt_cutoff_ties: None,
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
        let results =
            run_storm(target.clone(), vectors, &profile, 2, cmp(None), TEST_P, false, None).await;

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
                gt_cutoff_ties: None,
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
            TEST_P,
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
        let results =
            run_storm(target.clone(), vectors, &profile, 5, cmp(None), TEST_P, false, None).await;

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



