# nova-opt — Design Document

Cost-aware, reuse-aware, recall-constrained multi-objective Bayesian
optimization for vector database configuration. This document explains what
the tuner does, why each piece is shaped the way it is, and which invariants
must not be broken. It is written for both humans and AI coding agents
working in this repo.

---

## 1. The problem and the core idea

Tuning a vector database means choosing dozens of parameters: how the data
is laid out (segments, shards, vector datatype), how the ANN index is built
(HNSW `m`, `ef_construct`), whether/how vectors are quantized, and how
queries are executed (`ef_search`, batch size, concurrency). The user wants
the **fastest / highest-throughput configuration subject to a recall
constraint** (e.g. recall ≥ 0.90).

A naive grid sweep (what `nova-sweep` does) treats every configuration as
equally expensive to try. It is not:

| Change at level | What must be redone | Typical cost |
|---|---|---|
| layout (segments, dtype, shards, payload) | reinsert corpus + rebuild index + quantization + search | most expensive |
| index (m, ef_construct, on-disk, threshold) | rebuild index + quantization + search | expensive |
| quantization (variant, always_ram) | rebuild quantized vectors + search | moderate |
| search (ef_search, batch, top_k, concurrency, rescore) | rerun queries only | cheap |

**The core research idea:** vector DB tuning is not flat black-box
optimization — it has a natural artifact hierarchy
(`layout → index → quantization → search`), and the optimizer should plan
the *experiment schedule itself*, maximizing expected QPS/latency Pareto
improvement **per unit of marginal rebuild cost**, while a learned recall
classifier keeps it from wasting expensive builds on configurations that
won't meet the recall constraint.

The acquisition function that encodes all of this:

```
              EHVI_qps_latency(x) · W_R(x)
score(x) = ──────────────────────────────────
            C_reuse(x | artifacts)^gamma_t
```

- `EHVI` — expected hypervolume improvement over the (feasible) QPS/latency
  Pareto front (§7).
- `W_R` — recall-feasibility weight from the learned classifier, with an
  exploration bonus where the classifier is uncertain (§8).
- `C_reuse` — the marginal cost of evaluating `x` *given which artifacts
  already exist* (§3).
- `gamma_t` — cost-cooled penalty exponent: high early, annealed to zero as
  the budget depletes (§8).

Recall is a **feasibility constraint, never an objective**. The BO
surrogates model QPS and latency; the classifier (plus online feedback)
handles "will this config hit recall ≥ r₀".

---

## 2. Architecture and data flow

```
data.csv ──► RecallClassifier (recall.py + venn_abers.py)   [offline prior]
                    │ P(recall ≥ t), confidence per t
corpus X ──► stats.py ─► workload feature row ──► joined with each candidate
queries Q ─► stats.py ─┘                                    │
                                                            ▼
        ┌───────────────────────── Optimizer loop (optimizer.py) ─────────────┐
        │ 1. sample candidates (space.py, biased toward existing artifacts)   │
        │ 2. predict recall feasibility (offline → Platt → recall GP blend)   │
        │ 3. price candidates against ArtifactCache (cost.py)                 │
        │ 4. EHVI from GP surrogates (surrogate.py, exact 2-D closed form)    │
        │ 5. score = EHVI·W_R / cost^γ_t  (batch-aware for expensive builds)  │
        │ 6. evaluate winner: build ONLY missing levels (evaluate.py)         │
        │ 7. amortize expensive builds with cheap search-level children       │
        │ 8. record trial; update GPs, cost model, cache, online recalibrator,│
        │    recall GP                                                        │
        └──────────────────────────────────────────────────────────────────────┘
                                                            │
                                    opt_trials.parquet + opt_run.json (record.py)
```

Module map:

| Module | Responsibility |
|---|---|
| `space.py` | config hierarchy, artifact keys, quantization vocabulary, candidate sampling, cheap children |
| `cost.py` | artifact cache + reuse-aware marginal cost model (priors → EMA → ridge regression) |
| `stats.py` | dataset/query statistics extractor (the classifier's geometry features) |
| `recall.py` | XGBoost recall feasibility classifier (threshold-as-feature, LODO ensemble, OOD) |
| `venn_abers.py` | inductive Venn–Abers calibration (probability + validity-backed interval) |
| `online.py` | global Platt recalibration of offline probabilities from this run's measured recalls |
| `recall_online.py` | run-local recall GP (spatial correction), blended into `online.py`'s output by local confidence |
| `surrogate.py` | GP surrogates for log-QPS / log-latency, exact 2-D EHVI, feasible-only front |
| `acquisition.py` | W_R blending, exploration bonus, cost-normalized score, baseline strategies |
| `optimizer.py` | the planning loop, batch-aware scoring, scheduling, trial recording |
| `evaluate.py` | `LiveQdrantEvaluator` (nova-load/nova-storm subprocesses) and `ReplayEvaluator` (offline) |
| `config.py` / `cli.py` / `record.py` | pydantic YAML config, `nova opt` CLI, parquet/JSON outputs |

---

## 3. Configuration space, artifact keys, marginal cost

### Artifact keys (`space.py`)

Every candidate decomposes into nested prefix keys:

```
layout_key = (segments, dtype, shard_count, on_disk_payload)
index_key  = (layout_key, m, ef_construct, index_on_disk, indexing_threshold)
quant_key  = (index_key, quantization_variant, always_ram)
search_key = (quant_key, ef_search, batch_size, top_k, concurrency, rescore)
```

Two candidates sharing a prefix key share the artifact that prefix names —
this is what makes reuse-aware costing possible, **with one asymmetry that
matters**: layouts are independent, permanent artifacts (`LiveQdrantEvaluator`
gives each `layout_key` its own Qdrant collection, never deleted), but
index/quant are `reindex` mutations *in place* on that one collection — a
layout has exactly one currently-live index_key and quant_key, not one per
key ever built into it. The `ArtifactCache` (`cost.py`) reflects this: it
remembers which `layout_key`s exist (a set, since those never go stale) but
tracks index/quant per layout as "whichever key is *current*" (a dict), not
"every key ever seen" — the latter would let a stale key look reusable after
a different one physically overwrote it, corrupting a live measurement (see
invariant #8b). `missing_levels()` returns what must be (re)built, with the
crucial rule that **a missing or stale prefix invalidates everything under
it** (a fresh layout has no index yet; a layout whose *current* index_key
differs from the candidate's needs a rebuild even if that exact index_key
was built once before).

### Quantization vocabulary — a hard contract

`QUANT_VARIANTS` maps variant names to (a) the exact
`quantization_variant` / `quantization` / `quantization_mode` strings in the
training data (`data.csv`) and (b) the nova-load `quantization:` config
block that materializes them. **These strings must remain byte-identical to
data.csv's columns** — they are the join key between candidates and the
classifier's training distribution. Example: `binary_1_5bit` ↔
`BINARY__ONE_AND_HALF_BITS` ↔ `{type: binary, encoding: one_and_half_bits}`.

**Byte-identical does not mean fully covered.** `product_x4`/`x8`/`x16`/`x32`
are declared variants with no labeled rows in the current `data.csv` at all
(only `product_x64` has training data; the rest of the PRODUCT family is
config-space that exists but has never been measured). A candidate using one
of these isn't a vocabulary violation, so nothing crashes or logs — it's
silent extrapolation from the classifier's perspective, and unlike a
genuinely novel *dataset*, the OOD similarity term (§5.4) doesn't cover this
case at all (it compares dataset *geometry*, not config coverage). Neither
of `configs/opt/example.yaml`'s example `space:` blocks include these; if a
config's `space.quant_variant` does, treat the classifier's opinion on those
candidates as unfounded, not merely uncertain.

### Marginal cost (`cost.py`)

```
C_reuse(x | artifacts) = Σ estimate(level)  over missing levels
```

Per-level estimates escalate through three sources, best-available first:

1. **Ridge regression** of `log(seconds)` on level-relevant features
   (index-build time genuinely scales with `m` and `ef_construct`; quant
   time differs by family; pretending each level is a constant makes the
   acquisition's denominator lie). Activates once a level has ≥3
   observations with feature variation; predictions are **clamped to
   [EMA/5, EMA·5]** so a 3-point fit informs the estimate rather than
   replacing it.
2. **EMA** of observed durations (config-blind but tracks the actual
   instance/dataset), seeded by
3. **configured priors** (`cost_priors:` in the YAML).

**The ridge fit itself.** With feature rows stacked as `X` (column 0 is the
constant `1`, so the intercept is one of the fitted weights) and targets
`y = log(seconds)`, `_regress` solves the penalized normal equations

```
w = (XᵀX + Λ)⁻¹ Xᵀy,    Λ = ridge_lambda · I  with Λ₀₀ = 0
```

— ordinary ridge regression, except the `(0,0)` entry of the penalty matrix
is zeroed so the intercept (the log-baseline cost) is never shrunk, only
the slopes on `m`/`ef_construct`/etc. are. `Λ₀₀ = 0` is what makes "3 points,
one per feature value" a well-posed (if noisy) fit rather than one biased
toward shrinking the whole prediction to zero.

---

## 4. Dataset/query statistics (`stats.py`)

The recall classifier needs *dataset geometry* features joined with config
features. The extractor reproduces the schema of the external batch feature
pipeline that produced `data.csv` (feature names and default sampling sizes
are a contract):

**Base matrix X** (`embedding_*`): count, dimensionality, norm stats
(mean/std/p50/p95/min/max); random-pair distance percentiles
(p05/p50/p95/mean/std over `pair_sample_size=100k` pairs); self-kNN rank-1 /
rank-10 / rank-100 neighbor distances (p50/p95, from
`nn_query_sample_size=256` queries × `nn_reference_sample_size=5000`
references, self-matches excluded); intrinsic dimensionality via the
**TwoNN maximum-likelihood estimator** on the first two positive neighbor
distances (`ID = n / Σ ln(d₂/d₁)`).

**Query matrix Q** (`query_*`): count, norm stats, duplicate rate (row
hashing), PCA top-1 / top-10 explained variance (on `sample_size=1000`
sampled rows), pairwise distance percentiles, intrinsic dimensionality.

Two deliberate rules:

- **Q features are computed from Q alone.** No query-to-base retrieval
  features — those would leak X's retrieval difficulty into the query
  representation and contaminate the classifier's notion of "query
  geometry".
- **Full pass vs. sampled fallback with provenance.** Norm/count stats are
  exact when the matrix is under `full_pass_row_limit` (default 2M rows),
  otherwise computed on a seeded row sample. Either way the returned
  provenance records exact/sampled per statistic plus every sampling
  parameter and seed — the training pipeline depends on this, and the code
  never fails just because a dataset is too big to scan.

Distance metrics: `cosine` (1 − cos), `euclidean` (L2), `dot`
(negated inner product so "smaller is closer" holds uniformly), selected by
the workload's `distance_metric` (aliases: L2→euclidean, IP→dot).

**TwoNN always runs under a real metric distance, never raw `dot`.**
`dot`'s "distance" (`-inner_product`) isn't a metric at all — it can be any
sign, isn't translation-invariant, and doesn't correspond to the ball-volume
growth TwoNN's MLE derivation is built on. For real (nonzero-mean)
embeddings a genuine nearest neighbor has a large positive inner product, so
`-inner_product` is negative for it — every row would fail TwoNN's `d1 > 0`
validity check, and `embedding_intrinsic_dimensionality`/
`query_intrinsic_dim_estimate` would come out **silently NaN for every
`dot`-metric workload** (found by an audit, reproduced: 100% of nearest-
neighbor `d1`s negative on a synthetic shifted-Gaussian corpus). `_twonn_metric`
maps `dot → euclidean` for this one estimator specifically (recomputing a
second kNN pass under `euclidean` only when needed); every other feature
(percentiles, rank-k distances) still uses the workload's own metric,
since those are legitimately about *that* metric's retrieval geometry.

#### Where `ID = n / Σ ln(d₂/d₁)` comes from

TwoNN (Facco et al. 2017) models the local neighborhood of each point as a
homogeneous Poisson process of unknown intensity `ρ` in `d` dimensions: the
count of points within radius `r` is `Poisson(ρ·Vd·r^d)`, `Vd` the volume of
the unit `d`-ball. Two facts about this process are `d`-revealing and
`ρ`-free:

1. For a single point, let `r₁ < r₂` be the distances to its 1st and 2nd
   nearest neighbors. The ratio `μ = r₂/r₁` turns out to have a distribution
   that does **not** depend on `ρ` (the unknown, uninteresting local density
   cancels out) — only on `d`: `P(μ ≤ x) = 1 − x^{−d}` for `x ≥ 1`, i.e. `μ`
   is Pareto(`d`) with density `f(μ) = d·μ^{−d−1}`.
2. This is exactly why the estimator only ever needs the first two neighbor
   distances, never the local density — `ρ` washes out of the ratio.

**MLE from the Pareto density.** Given `n` iid ratios `μᵢ = d₂ᵢ/d₁ᵢ`, the
log-likelihood is `L(d) = n·ln d + (−d−1)·Σ ln μᵢ`. Setting `dL/dd = 0`:

```
n/d − Σ ln(μᵢ) = 0   ⟹   d̂ = n / Σ ln(μᵢ) = n / Σ ln(d₂ᵢ/d₁ᵢ)
```

— the formula in `_intrinsic_dim_twonn`. The `d₁ > 0` / `d₂ > d₁` validity
check in the code is exactly "is this pair a genuine `(r₁, r₂)` from a real
metric ball" — which is precisely the condition §4's `dot`-metric caveat
above explains can fail for a non-metric "distance".

---

## 5. Recall feasibility classifier (`recall.py`)

Predicts `P(mean_recall_at_k ≥ t)` for t ∈ {0.25, 0.50, 0.80, 0.90, 0.95}
from dataset geometry + config features. Pure classification (no recall
regression model, by explicit project decision). Four design choices, each
replacing a weaker alternative:

### 5.1 Threshold-as-feature, not five independent heads

Training rows are replicated once per threshold with a `threshold` input
column and label `1[recall ≥ t]`; **one** XGBoost model
(`binary:logistic`, `tree_method=hist`, native categorical support) is
trained with **monotone constraints**:

- `threshold: −1` → P(recall ≥ 0.25) ≥ … ≥ P(recall ≥ 0.95) holds **by
  construction** (no post-hoc clamping of independent heads);
- `ef_search: +1`, `ef_construct: +1` → recall never *drops* as either
  grows, which keeps extrapolation past the training grid sane — the
  optimizer probes exactly those edges.

Compared to five independent per-threshold classifiers this shares
statistical strength (5× rows per tree) and removes an entire class of
inconsistency. Categorical features (`distance_metric`,
`quantization_variant/quantization/quantization_mode`, `rescore`) are passed
as pandas categoricals with a fixed vocabulary stored in the model metadata
(unseen values become "missing", never a crash).

**How a `monotone_constraints` entry of `+1`/`−1` actually forces
monotonicity.** It isn't a post-hoc correction; it constrains *tree
growing* itself. For a feature marked `+1`, every candidate split on that
feature is only accepted if the resulting left/right leaf value ranges keep
the whole tree non-decreasing in that feature — concretely, XGBoost tracks
an `[lower, upper]` bound on each leaf's permissible prediction as it
recurses, and a split on a monotone feature must place the "greater feature
value" side's whole subtree bound above the "lesser" side's (recursively,
so this holds no matter how deep either subtree grows later). Summed across
boosting rounds, this makes the *entire ensemble's* prediction — a sum of
trees — provably non-decreasing in that one feature with every other
feature held fixed. That's a much stronger guarantee than "the training
data happens to be monotone": it holds at every point of the input space,
including extrapolated ef_search/ef_construct values past the training
grid, which is exactly where the optimizer spends its exploration budget.
One caveat straight from XGBoost's own docs, not obvious from the outside:
`monotone_constraints` combined with `tree_method="hist"` (used here) can
reject enough candidate splits to produce noticeably shallower trees than
the same data would get unconstrained — a real accuracy/monotonicity
trade-off, not a free guarantee.

### 5.2 Leave-one-dataset-out, and the fold models ARE the deployed model

Evaluation protocol per fold: one dataset is test, the *next* dataset is the
early-stopping validation set, all others train. This matches deployment
exactly — the tuner always faces a dataset the model never saw. Requires
**at least 3 datasets** — with only 2, test and validation between them
claim every dataset and the fold's training set is empty (`RecallClassifier.train`
rejects this explicitly rather than letting it fail downstream with a NaN).

Rather than discarding the fold models and refitting on everything, the
**16 LODO fold models are kept as the deployed ensemble**: the prediction is
the fold mean, and the **cross-fold spread is a per-prediction epistemic
uncertainty** — literally "how much does this prediction depend on which
datasets were in training". A refit-on-everything model would have no such
signal.

Read that spread for what it actually measures, not more: any two folds
share 15/16 of their training data, so the spread reflects "how much does
leaving *one* dataset out of an otherwise near-identical pool move this
prediction" — a real signal, but much softer than "16 independently trained
models disagreeing" might suggest. On a workload genuinely unlike anything
in `data.csv`, don't expect this spread alone to blow up; that failure mode
is what the OOD similarity term (§5.4) is for, and the fold spread is a
comparatively minor correction on top of it, not the primary defense.

### 5.3 Venn–Abers calibration (`venn_abers.py`)

Raw ensemble scores per threshold are calibrated on the pooled out-of-fold
predictions with **inductive Venn–Abers**: for a score `s`, isotonic
regression is fit twice on the calibration set augmented with the test point
labeled 0 and labeled 1, giving an interval `[p0(s), p1(s)]` that provably
contains a perfectly calibrated probability. The single probability is
Vovk's minimax-log-loss merger `p = p1 / (1 − p0 + p1)`; the **interval
width is a validity-backed per-prediction calibration uncertainty**.

**Why fitting isotonic regression twice brackets the truth.** Isotonic
regression finds the monotone function of the raw score that best fits the
calibration labels (via pool-adjacent-violators, the maximum-likelihood
monotone fit under squared error, which for binary labels is also the
maximum-likelihood monotone *probability* fit). Augmenting the calibration
set with the new point `(s, 0)` and re-fitting gives `g₀`, the best monotone
calibration *if this point turns out negative*; augmenting with `(s, 1)`
gives `g₁`, the best fit *if it turns out positive`. `p0 = g₀(s)` and
`p1 = g₁(s)` are that same point's fitted value under each hypothesis — one
is always `≤` the other (adding a positive label can only push the fit at
`s` up, never down), and Vovk et al.'s result is that the true calibrated
probability of an isotonic-consistent model must lie in `[p0, p1]`,
whichever label turns out to be true. **`p1/(1−p0+p1)` is Vovk & Petej's
minimax merge**: the single probability `p` that minimizes the *worst-case*
log-loss over which of the two labels turns out to be true, not their
average — this repo doesn't re-derive that result, it's cited from "Venn–
Abers Predictors" (Vovk & Petej, 2012); worth knowing it's a specific
minimax construction (not, say, `(p0+p1)/2`) if you're ever re-deriving or
re-deriving-adjacent code near `venn_abers.py`. Treat `p0`/`p1` themselves,
not just the merged point, as the real output when the *interval* (not just
the point estimate) matters — the width `p1 − p0` is what feeds
`confidence(t)`.

Exact per-query IVAP costs a PAVA run per prediction — too slow for
thousands of candidates per iteration — so `p0/p1` are **precomputed on a
~129-point score grid** at train time (sklearn's C-optimized isotonic, two
fits per grid point, calibration set subsampled to 10k) and queried by
monotone interpolation. Isotonic functions are piecewise-constant with
breakpoints only at calibration scores, so grid interpolation error is far
below the interval widths that matter.

One honest gap: each calibrator is *fit* on the pooled single-fold OOF score
(one model's leave-one-dataset-out prediction per row — genuinely
leak-free), but *queried* at inference (`RecallClassifier.predict`) with the
**mean score across all 16 fold models**, a differently-distributed
statistic than what the isotonic curves were built from. This isn't wrong
for deployment — a genuinely new 17th dataset is equally out-of-sample for
every fold, so "mean of 16 fold scores" is the right quantity to ask about —
but the calibration curve itself was never validated against exactly that
statistic. Treat the calibration as good-but-unverified in this specific
sense, not as a correctness bug.

### 5.4 Out-of-distribution similarity

Confidence is scaled by how much the new workload's *geometry* resembles the
training datasets: per-training-dataset centroids over the `embedding_*` /
`query_*` stats features, standardized by across-dataset spread; the new
workload's distance to the nearest centroid, relative to the median
inter-dataset nearest-neighbor spacing, maps to a familiarity factor
`exp(−max(0, d/d_typical − 1))` ∈ (0, 1]. A workload that looks like nothing
in data.csv gets its probabilities *used* but *trusted less*.

Two regimes by construction: for `d ≤ d_typical` (as close to the nearest
training dataset as training datasets typically are to *each other*) the
`max(0, ·)` clamps the exponent to 0, so the factor is exactly `1` — a
workload no more unusual than ordinary inter-dataset variation gets no
penalty at all, deliberately, since anything else would penalize the
*typical* case. Past that point the factor decays exponentially in how many
multiples of `d_typical` past the boundary the workload sits. The value is
continuous at the `d = d_typical` join (both sides equal `1` there), though
the slope isn't — flat at `0` just below, `−1/d_typical` times the factor
just above — a deliberate corner, not a bug: nothing is supposed to change
for `d` at or under the "normal" spread, so the penalty should only start
accruing, not already be accruing, right at that boundary.

### Putting it together

```
probability(t)  = VennAbers_t( mean over fold models of raw score )
confidence(t)   = (1 − min(1, VA_width + fold_spread)) · ood_similarity
```

**This combination formula is a heuristic, not a derivation** — it adds a
calibration-interval width to an ensemble spread (two differently-scaled
notions of uncertainty) and then multiplies by a third quantity. Same for
§8's `W_R = w·p(t0) + (1−w)·p(safety_t) + β·(1−confidence)`: a fixed convex
blend plus an additive exploration bonus, chosen because it behaves
sensibly at the cases that matter (zero confidence → full bonus; safety
threshold agrees → blend barely moves), not because it falls out of a
probabilistic model of how VA width, fold spread, and OOD similarity
compose. Both are reasonable, both are tested for the behavior they're
meant to produce — but don't read them as being in the same category as
Vovk's merger or the exact EHVI closed form, which *are* derived quantities.

Real-data LODO pooled AUC (16 datasets, 41k rows), thresholds
0.25/0.50/0.80/0.90/0.95:

- v1 (5 independent heads + isotonic): 0.63 / 0.79 / 0.78 / 0.84 / 0.84
- **v2 (this design): 0.73 / 0.77 / 0.79 / 0.82 / 0.82**

Large gain at 0.25, par elsewhere; the ~0.015 AUC given up at 0.90/0.95 buys
built-in monotonicity, sane extrapolation, and real uncertainty estimates —
which the acquisition depends on structurally.

**Missing from this comparison: a genuinely simple baseline** (e.g. config
knobs alone with no dataset-geometry features, or nearest-training-dataset
lookup). Both v1 and v2 are XGBoost-plus-calibration architectures of
similar sophistication, so beating v1 doesn't establish how much of the
0.7–0.8 AUC comes from the geometry features actually earning their keep
versus recall-as-a-function-of-(ef_search, ef_construct, quantization) alone
just not being that hard to rank. Worth adding before leaning on this
comparison in a paper.

---

## 6. Online recall feedback (`online.py`)

The offline classifier is a *prior*. Its dominant failure mode on a new
dataset — visible in its own LODO evaluation — is **miscalibration under
dataset shift while the ranking stays useful**. So during the run, every
measured `(config, recall)` pair feeds a per-threshold Platt recalibration
of the offline probability:

```
p_online = sigmoid(a + b · logit(p_offline))
```

fitted by penalized MLE with a **shrinkage prior centered at the identity**
(a=0, b=1; strength configurable via `online_prior_strength`). Zero
measurements → offline probabilities pass through unchanged; each
contradicting measurement pulls (a, b) toward what this workload actually
does; after ~10 measurements the tuner effectively trusts its own eyes over
data.csv — which is the point.

**Why this fit is well-behaved with only a handful of points.** Writing
`z = logit(p_offline)` (fixed, precomputed per observation), the objective
`_fit_platt` minimizes is

```
−ln L(a,b) = −Σ [y·ln σ(a+bz) + (1−y)·ln(1−σ(a+bz))] + ½·λ·(a² + (b−1)²)
```

which is exactly logistic regression on the single feature `z` (jointly
convex in `(a,b)`, since `σ` composed with an affine map is a standard
convex logistic loss) plus a quadratic penalty (strictly convex) — the sum
is strictly convex, so BFGS has a unique minimum to find regardless of how
few observations exist; there's no local-minima risk to guard against here.
The penalty term is exactly the negative log of a Gaussian prior
`(a,b) ~ N((0,1), 1/λ · I)`, so this whole fit is MAP estimation under that
prior — `online_prior_strength = λ` directly controls how many "pseudo-
observations" of the offline model being trustworthy the shrinkage is worth
against real measurements, which is why small measurement counts don't
whipsaw the calibration.

**`p_offline` here always means the untouched classifier output**, never the
already-recalibrated or recall-GP-blended prediction — `optimizer.py` keeps
a separate `raw_pred` alongside the blended `pred` through scoring for
exactly this reason (`Optimizer._predict_recall` returns both;
`_evaluate_one` feeds `raw_pred` to `self.online.add`, never `pred`).
Feeding the recalibrator its own corrected output would make each fit a
correction to an already-shifting target instead of to the fixed offline
prior, and would compound further once the recall GP (§6b) is blended in
too. The slope is clipped positive so the offline
*ranking* (the thing LODO says survives dataset shift) can never invert, and
cross-threshold monotonicity is re-clamped after the per-threshold refits.

Observed working in the smoke run: configs the offline model scored
P(≥0.80) ≈ 0.85 measured recall 0.0; two iterations later the recalibrated
probabilities for similar configs had dropped to ≈ 0.3.

### 6b. Online recall *surrogate* — the spatial half (`recall_online.py`)

The Platt recalibration above is a single global curve: it can learn "this
workload is systematically easier/harder than the classifier thinks" but not
"ef_search=16 is worse than the classifier thinks, ef_search=256 is fine" —
that's a spatial pattern, and a 2-parameter scalar can't represent one.
QPS/latency get real per-config GPs (§7) that improve every iteration;
recall didn't have that.

`RecallSurrogate` fits a GP regressing `logit(measured_recall)` on the same
encoded candidate vector the QPS/latency GPs use
(`surrogate.CandidateEncoder`), via the same kernel (`surrogate.make_gp`).
Because it's a genuine regression rather than five independent per-threshold
classifiers, `P(recall(x) ≥ t) = Φ((μ(x) − logit(t)) / σ(x))` is computable
for *any* t from one fit, and — being the same Gaussian CDF evaluated at
different points — automatically monotone non-increasing in t, with none of
the monotone-constraint machinery §5.1 needs.

**Where the `Φ((μ−logit(t))/σ)` formula comes from.** The GP's posterior at
`x` is (by construction of GP regression) `logit(recall(x)) ~ N(μ(x), σ(x)²)`
— a Gaussian belief over the logit of recall, not recall itself. Since
`logit` is strictly increasing, `recall(x) ≥ t ⟺ logit(recall(x)) ≥ logit(t)`,
so:

```
P(recall(x) ≥ t) = P(N(μ(x), σ(x)²) ≥ logit(t)) = 1 − Φ((logit(t) − μ(x))/σ(x)) = Φ((μ(x) − logit(t))/σ(x))
```

(the last step uses `1 − Φ(z) = Φ(−z)`, standard-normal symmetry). This is
the same "Gaussian survival probability" building block `g(l)`'s derivation
in §7 uses (`Φ̄` there, `1−Φ` here) — both sections are asking "what's the
probability a Gaussian exceeds a threshold", just for different purposes
(a hypervolume integral there, a feasibility probability here).

The posterior marginal being *exactly* Gaussian, given the GP model class,
is a mathematical fact (any marginal of a jointly Gaussian vector is
Gaussian), not an approximation — sklearn computes it in closed form, no
sampling. Two things this formula's exactness doesn't cover, both standard
and both worth knowing about rather than being surprised by: kernel
hyperparameters (length scales, noise level) are point-estimated by type-II
MLE and then treated as known (empirical-Bayes plug-in — hyperparameter
uncertainty itself isn't propagated into `σ(x)`), and measured recall is
really a proportion over a finite query sample (closer to binomial noise,
bounded and heteroscedastic in raw `[0,1]` space) while the GP assumes
homoscedastic Gaussian noise in logit space. Both are conventional modeling
choices, not derivation gaps.

**Trust is the GP's own posterior uncertainty, not a hand-tuned schedule.**
Near observed points σ is small (trust the GP); far from data σ reverts to
the kernel's prior variance (defer entirely to the offline+Platt
prediction). This is the exact mechanism that already makes the QPS/latency
GPs behave sensibly away from data — no new trust-schedule hyperparameter,
only a readiness gate (`recall_gp_min_observations` measured recalls, with
feature variation among them, mirroring `cost.py`'s "≥3 obs with variation"
regression gate). Below that gate the surrogate is a no-op and predictions
are the offline+Platt value unchanged.

Blend, per threshold, with a single per-candidate weight `alpha(x)` (the
GP's local confidence) shared across every threshold:

```
P_final[t](x) = alpha(x) · P_gp[t](x) + (1 − alpha(x)) · P_offline_platt[t](x)
```

Since `alpha` doesn't depend on `t` and both terms are already monotone in
`t`, the blend is monotone in `t` for free.

This surrogate is **local to one optimizer run** and is trained only on this
run's live measurements — never on `data.csv`, never touching the offline
classifier's training path. It feeds only `W_R` (via `_predict_recall` in
`optimizer.py`), exactly like the Platt recalibrator it augments.

---

## 7. MOBO surrogate and exact EHVI (`surrogate.py`)

### Surrogates

Two independent GPs (Matérn ν=2.5 × constant + white noise, `normalize_y`)
model `log(QPS)` and `log(latency_p95_ms)` over an encoded candidate vector
(log₂ for scale-like knobs, one-hot over the space's declared vocabularies,
tri-state encoding for `rescore`). Log-space objectives keep hypervolume
scale-sane (a 2× QPS gain counts the same at 100 or 10k QPS) and make GP
stationarity far less wrong. The acquisition frame maximizes
`f = (log qps, −log latency)`.

**Independence is a modeling choice, not a property of the real system.**
QPS and latency are two GPs fit separately with no cross-covariance, and
real QPS/latency are strongly (negatively) correlated — pushing ef_search up
moves both together. "Exact EHVI" below is exact *for this independent-
marginals surrogate*, not exact expected improvement against the real
joint distribution; don't read the two claims as the same thing. This is
the standard, tractable approximation multi-objective BO libraries make for
the same reason (a full joint model needs Monte Carlo, which is exactly what
the closed form here was built to avoid), so it's a deliberate trade, not an
oversight — but it does mean the acquisition can be somewhat overconfident
about how independently the two objectives can be pushed.

### Feasible-only Pareto front (a correctness requirement)

The front EHVI improves upon contains **only observations whose measured
recall met the target**. GPs still train on *all* observations — an
infeasible point is perfectly good evidence about QPS and latency — but it
must not sit on the front: a sky-high-QPS config at recall 0.4 would
otherwise make every feasible candidate look like a negligible hypervolume
improvement, blinding the acquisition exactly where the answer lives. This
is the standard constrained-MOBO decomposition (feasibility probability as a
multiplier, feasible front as the improvement target) — the same
`EI(x)·P(feasible(x))` shape as Gardner et al. 2014 ("Bayesian Optimization
with Inequality Constraints") for single-objective EI, and Feliot, Bect &
Vazquez 2017 for the EHVI case specifically. **What makes the multiplicative
form exact rather than a heuristic reweighting: the feasibility model and
the objective GPs must be conditionally independent given `x`** — true here
because the recall classifier/GP and the QPS/latency GPs are separate models
with no shared conditioning. If that ever changes (e.g. a single joint model
producing both), this decomposition would need re-deriving, not just reuse.
The vanilla-BO baseline is recall-oblivious *by definition* and explicitly
requests the
all-points front.

### Exact 2-D EHVI (no Monte Carlo)

Via `HVI(z) = ∫ 1[ref ≤ y ≤ z, y undominated] dy` and Fubini:

```
EHVI = ∫_A P(Z₁ ≥ y₁) · P(Z₂ ≥ y₂) dy      (independent GP marginals)
```

where A (the undominated region above the reference point) decomposes, for a
2-D front {(aᵢ, bᵢ)} with a descending / b ascending, into vertical strips:

```
strip 0:  y₁ ∈ (a₁, ∞),        y₂ ∈ (ref₂, ∞)
strip i:  y₁ ∈ (a_{i+1}, aᵢ],  y₂ ∈ (bᵢ, ∞)       (a_{n+1} = ref₁)
```

Each strip integral is a product of Gaussian partial expectations
`g(l) = E[max(Z − l, 0)] = σφ(α) + (μ − l)Φ̄(α)`, `α = (l−μ)/σ`. The result
is exact, vectorized over candidates, and ~100× faster than the Monte-Carlo
version it replaced — verified against brute-force MC (60k samples) and
hand-computed degenerate cases in the tests. An empty feasible front reduces
to one strip over the whole `[ref, ∞)` quadrant (everything is upside).

#### Derivation, step by step

**1. What HVI(z) means.** Adding a hypothetical observation at `z = (z₁, z₂)`
to the current front `F` grows the hypervolume by exactly the volume of
points that are (a) inside the box `[ref, z]` and (b) not already dominated
by some point already in `F`:

```
HVI(z) = HV(F ∪ {z}) − HV(F) = ∫ 1[ref ≤ y ≤ z] · 1[y undominated by F] dy
```

(both "≤" componentwise; maximizing both objectives, so "dominated by F"
means some front point is ≥ y in both coordinates).

**2. Take the expectation over the candidate's predictive distribution and
swap it inside the integral.** `Z = (Z₁, Z₂)` is random (the GPs' posterior
at the candidate); `y` is the integration variable, not random. The
undominated-by-`F` indicator depends only on `y`, so it comes out of the
expectation unchanged; the `1[y ≤ Z]` indicator is the only random part:

```
EHVI = E_Z[HVI(Z)] = ∫ 1[y undominated by F, y ≥ ref] · E_Z[1[y ≤ Z]] dy
     = ∫_A P(Z₁ ≥ y₁) · P(Z₂ ≥ y₂) dy
```

(Fubini/Tonelli justifies the swap — the integrand is non-negative — and
`E_Z[1[y ≤ Z]] = P(Z ≥ y) = P(Z₁ ≥ y₁)·P(Z₂ ≥ y₂)` by the GPs' independence.)
`A` is the undominated region from step 1.

**3. `A` decomposes into rectangles because a 2-D Pareto front is a
descending staircase.** Sort the front by `a` descending (`b` ascending). A
point `y` is dominated iff some front point `(aᵢ, bᵢ)` has `aᵢ ≥ y₁` and
`bᵢ ≥ y₂`; walking the staircase left to right, the *undominated* region
above `ref` is exactly the union of the vertical strips in the docstring
above (`strip i` sits to the right of `aᵢ`'s neighbor and above `bᵢ`) —
each strip is a rectangle `(lo₁, hi₁] × (lo₂, ∞)`, so the double integral
over `A` is a sum of double integrals over rectangles.

**4. A rectangle integral factors into a product of two 1-D integrals**,
because the integrand `P(Z₁≥y₁)·P(Z₂≥y₂)` is itself a product and the
region is a product of intervals:

```
∫_{lo₁}^{hi₁}∫_{lo₂}^{∞} P(Z₁≥y₁)P(Z₂≥y₂) dy₂dy₁ = [∫_{lo₁}^{hi₁} P(Z₁≥y₁)dy₁] · [∫_{lo₂}^{∞} P(Z₂≥y₂)dy₂]
```

**5. `∫_l^∞ P(Z≥y) dy = E[max(Z−l, 0)]`.** This is the standard
"layer-cake" identity: for fixed `z`, `∫_l^∞ 1[z≥y] dy = max(z−l, 0)`
(the integral is over exactly `y ∈ (l, z]` when `z > l`, empty otherwise);
take `E_Z` of both sides and swap with the outer integral (Fubini again) to
get `∫_l^∞ P(Z≥y)dy = E_Z[max(Z−l,0)] =: g(l)`. A finite upper limit `hi₁`
instead of `∞` just subtracts: `∫_{lo₁}^{hi₁} = g(lo₁) − g(hi₁)` — exactly
the `width = g(lo1[i]) − upper` line in `ehvi_2d_exact` (`upper = 0` when
`hi₁ = ∞`, since `g(∞) = 0` trivially).

**6. Closed form for `g(l)` when `Z ~ N(μ, σ²)`.** Write `Z = μ + σW` with
`W ~ N(0,1)` and `α = (l−μ)/σ`, so `Z − l = σ(W − α)`:

```
g(l) = σ·E[max(W−α, 0)] = σ·∫_α^∞ (w−α)φ(w) dw
     = σ·[∫_α^∞ w φ(w) dw − α∫_α^∞ φ(w) dw]
     = σ·[φ(α) − α·Φ̄(α)]                     (since φ'(w) = −wφ(w) ⟹ ∫_α^∞ wφ(w)dw = φ(α))
     = σφ(α) + (μ−l)·Φ̄(α)                    (α = (l−μ)/σ ⟹ −σα = μ−l)
```

matching `_g` in `surrogate.py` exactly. Chaining steps 2–6 is the whole
proof that `ehvi_2d_exact` computes the *exact* expectation in step 2 — no
approximation anywhere except the independent-normal-marginals modeling
choice from step 2 itself (see the surrogates caveat above).

**Reference point:** `min(all observed) − max(0.1, 0.5 · observed span)` per
objective. The margin must scale with the objective range: a fixed epsilon
zeroes out candidates that *extend* the front beyond the observed nadir
(e.g. much higher QPS at worse-than-any-seen latency) — exactly the Pareto
extensions worth finding. (This was a real bug found by a test during
development.)

---

## 8. Acquisition and baseline strategies (`acquisition.py`)

### Recall weight

For target r₀, using the strictest modeled threshold t₀ ≥ r₀:

```
W_R(x) = w·p(t₀) + (1−w)·p(safety_t)  +  β · (1 − confidence(t₀))
```

with target-specific safety blends (0.95→safety 0.80 @ w=0.8; 0.90→0.80 @
0.85; 0.80→0.50 @ 0.85; 0.50→0.25 @ 0.85). A candidate confidently above
0.80 retains value under a strict 0.95 target even when the 0.95 estimate is
shaky.

**Calibrated probabilities enter unweighted by confidence.** Expected-utility
reasoning says a calibrated probability is used as-is; multiplying by
confidence (as in the original starting-point formula) double-counts
uncertainty and biases selection toward regions where the model is
*opinionated* rather than where recall is *likely*. Confidence appears only
in the exploration bonus — uncertain candidates get a nudge, never a veto.

### Cost cooling

`gamma_t = gamma · (remaining budget / total budget)` (on by default).
Early: strong preference for learning from cheap evaluations. Late: a final
expensive rebuild is fine if the predicted payoff justifies it. This is the
same shape as CArBO's cost-model depreciation (Lee et al. 2020, "Cost-aware
Bayesian Optimization"): start at EI-per-unit-cost, anneal toward plain EI
as budget depletes. `remaining` is clamped to `[0, 1]`
(`max(0.0, 1 − spent/budget)` in `optimizer.py`) so `gamma_t` never goes
negative even on an overspend, and `cost^gamma_t` is computed against a
cost floored at `1e-6` — `0**0`/negative-base is never reachable.

### Baseline strategies

The `strategy` knob exists to run the research comparison; each baseline
zeroes out a term of the full acquisition. **Do not remove levels of it for
simplicity.**

| strategy | formula | corresponds to |
|---|---|---|
| `random` | uniform choice | baseline 1: naive sweep |
| `bo` | EHVI (all-points front) | baseline 2: vanilla BO |
| `bo_recall` | EHVI · 1[p(t₀) ≥ prune_min] | baseline 3: recall pruning only |
| `bo_cost` | EHVI / cost^γ (no recall model, recall-free batch value) | baseline 4: cost-aware only |
| `full` | EHVI · W_R / cost^γ (batch-aware) | baseline 5: the proposed method |

---

## 9. The optimizer loop (`optimizer.py`)

Per iteration:

1. **Sample candidates** — ~`bias_fraction` of proposals are search-level
   variations of quant artifacts that already exist (the space's dense cheap
   region, reconstructed from cached `quant_key`s), the rest uniform so
   fresh artifacts keep being proposed. Everything already evaluated is
   excluded by `search_key`.
2. **Predict recall feasibility** — offline classifier → online Platt
   recalibration → recall GP blend (§6b).
3. **Score.** Cheap candidates: `EHVI·W_R / cost^γ_t`. Candidates needing an
   expensive build are scored by the **batch they unlock**: the candidate
   *plus* the cheap search-level children the scheduler would evaluate after
   the build (`Σ EHVI·W_R` over the batch, divided by build cost + the
   children's search costs). This is what the build actually buys; pricing a
   build as one point systematically undervalues rebuilds. (A separate
   recall-free accumulator serves `bo_cost` so that baseline can't smuggle
   the classifier back in.)
   Before the GPs have ≥3 points (cold start), there's no EHVI signal yet, so
   a normalized encoded-space min-distance-to-evaluated **diversity factor**
   stands in for it — but it goes through the exact same `acquisition.score`
   call as post-warmup scoring, with `ehvi=diversity`. This matters because
   the per-strategy zeroing is a contract (§8's baseline table), not just a
   post-warmup nicety: an earlier version hand-rolled `W_R / cost^γ ·
   diversity` for every non-`random` strategy during cold start, which
   silently made `bo` recall- *and* cost-aware, `bo_cost` recall-aware, and
   `bo_recall` a soft weight instead of a hard prune, for however many
   iterations cold start lasts — exactly the iterations a short-budget
   research-comparison run spends the largest *fraction* of its trials in
   (found by an audit; see invariant #6b).
4. **Evaluate the winner** — build only its missing levels; artifacts from a
   successful build are cached even if the subsequent search measurement
   fails; failures are recorded as error trials, never raised.
5. **Amortize** — if an expensive level was built, evaluate up to
   `max_children` cheap search-level siblings (ef_search × batch grid,
   ef-coverage-first ordering) against the fresh artifact.
6. **Update** — GP observations (feasibility-marked), cost model
   (per-level, with the candidate for the regression features), artifact
   cache, online recalibrator, recall GP, spent budget.

Feasibility of a measurement: measured `mean_recall ≥ target`; when the
evaluator produced no recall (storm without ground truth), the predicted
`p(t₀) ≥ 0.5` is the fallback proxy.

**Everything is recorded** per trial (`opt_trials.parquet`): measurements,
per-level build seconds, artifact keys, all five threshold
probabilities/confidences, W_R/EHVI/cost/score at selection time, and the
full data.csv-schema feature row (dataset stats + config) so completed runs
feed straight back into future classifier training. Run-level metadata
(settings, stats provenance, LODO metrics, best feasible config) lands in
`opt_run.json`.

---

## 10. Evaluators (`evaluate.py`)

**`LiveQdrantEvaluator`** drives the same nova-load / nova-storm subprocess
machinery as nova-sweep (and generates the same config shapes — duplicated,
not shared, matching this repo's cross-tool contract convention), but
**level by level**: `layout` = fresh `nova-load run` (collection named by a
hash of the layout key), `index` and `quant` = separate `nova-load reindex`
patches, `search` = one `nova-storm --json` run. The level split is what
lets a cache miss pay for only what it needs. Metric names map between the
classifier vocabulary (COSINE/IP/L2) and nova-load's
(`cosine`/`dot`/`euclid`).

**`ReplayEvaluator`** answers from a table of previously measured runs
(e.g. an exhaustive nova-sweep export) — the offline harness for comparing
the tuner and its baselines without touching a live cluster. Matching is on
the intersection of candidate feature columns; a candidate with no matching
row raises an `EvalError` the optimizer records and moves past.

---

## 11. CLI and configuration

```
nova opt tune <config.yaml> [--dry-run]   # run the tuner (live or replay)
nova opt train-recall --data data.csv --out <dir> [--report lodo.csv]
nova opt stats --vectors X --queries Q [--column ...] [--metric cosine]
```

`configs/opt/example.yaml` documents every knob. Highlights: `space:` (value
lists per axis, grouped by artifact level), `optimizer:` (target_recall,
strategy, budget_seconds, gamma/beta, cost_cooling, bias_fraction,
online_prior_strength, recall_gp_min_observations), `scheduler:` (children
grids, max_children),
`stats:` (sampling sizes, full_pass_row_limit), `cost_priors:`, and either a
live `target:` or a `replay:` table. `${VAR}` / `${VAR:-default}` env
expansion matches the rest of the repo.

---

## 12. Verification status

- **98 tests** pass (`make test` runs them via
  `uv run --directory python/nova-opt --extra dev pytest`). Notable ones:
  exact EHVI vs. brute-force Monte Carlo agreement; infeasible points
  leaving the front (and the `bo` baseline's all-points view); Venn–Abers
  interval validity/monotonicity/held-out calibration; online recalibration
  dynamics (identity at zero observations, prior damping, ranking
  preservation, and — after a review caught it — that the recalibrator is
  fed the raw offline probability, never its own blended output); the
  recall GP's readiness gating, confidence decaying with distance from
  observed points, per-threshold monotonicity from a single fit, and spatial
  correction of an overconfident flat offline prediction (§6b); cost
  regression scaling and EMA-band clamping; the artifact cache correctly
  forcing a rebuild when a *stale* index/quant key is revisited on a layout
  that was mutated in between (§3, invariant #8b); TwoNN not going silently
  NaN under the `dot` metric (§4, invariant #3b); cold-start scoring
  preserving each baseline's contract (`bo` cost/recall-oblivious, `bo_cost`
  recall-free, `bo_recall` hard-pruned with no cost term) before the GP
  surrogate is ready, not just after (§9, invariant #6b); end-to-end
  optimizer runs on a simulated evaluator (artifact reuse never redundantly
  rebuilds, children amortize builds, failures recorded not raised,
  cost-aware beats cost-oblivious BO on rebuild ratio, recall GP becomes
  ready and gets consulted mid-run); config validation rejecting
  non-positive budgets/cost-priors instead of silently no-op'ing or
  distorting the ridge-clamp band; the encoder disambiguating an unset
  `indexing_threshold` from an explicit `0` (they used to collide on the
  same encoded value).
- **Real data.csv training** (16 datasets, 41k rows): LODO pooled AUC
  0.73/0.77/0.79/0.82/0.82 across the five thresholds.
- **End-to-end smoke** (`nova opt tune` with the trained model, `.npy` stats
  extraction, replay table): finds the true optimum; logs show γ cooling,
  build levels shrinking across iterations
  (`layout,index,quant` → `index,quant` → `quant`), and online
  recalibration correcting an overconfident offline prior.
- **Not yet verified live:** `LiveQdrantEvaluator` against a real Qdrant
  (structurally covered; config shapes mirror nova-sweep's).
- **Known modeling limitations** (not bugs, but worth not overclaiming):
  the 2-D EHVI is exact under the surrogate's independent-normal-marginals
  assumption, not under the real system, where QPS and latency are
  correlated (§7 already states the assumption; don't read "exact EHVI" as
  "exact expected improvement against reality"). `make_gp` uses a full ARD
  Matérn kernel (one length scale per encoded feature — `CandidateEncoder`'s
  mixed log2-numeric/one-hot vector has no other way to get each dimension
  its own notion of scale), but sklearn's optimizer still frequently hits
  its length-scale/noise-level bounds on the small batches early in a run
  (visible as `ConvergenceWarning`s in the test suite), so early-run GP
  geometry is cruder than a fully-converged fit would be. §9's batch score
  sums each child's EHVI independently against the *same* fixed front —
  it doesn't account for overlap between children's undominated regions, so
  it's an over-estimate of what an expensive build actually buys (the
  classic batch-acquisition submodularity gap; real q-EHVI needs a joint
  distribution or Monte Carlo, exactly what the closed form here trades
  away). This isn't just typically true, it's **provable**: hypervolume
  improvement is a monotone submodular set function, so for any front `S`
  and two points `a, b`, `f(S∪{a,b}) − f(S) ≤ [f(S∪{a})−f(S)] + [f(S∪{b})−f(S)]`
  always — the sum of marginals against a static front can equal but never
  *underestimate* the joint improvement. Biased toward making rebuilds look
  more attractive than they are, never the reverse.

## 13. Invariants for future agents (do not break)

1. `QUANT_VARIANTS` strings must stay byte-identical to data.csv's
   `quantization_variant`/`quantization`/`quantization_mode` vocabulary.
2. Query statistics are computed from Q alone — never add query-to-base
   retrieval features to the classifier.
3. Stats extraction must keep recording sampling provenance (exact vs.
   sampled, sizes, seed).
3b. TwoNN (`_intrinsic_dim_twonn`) must never be fed raw `dot`-metric
    distances — route it through `_twonn_metric` (which maps `dot ->
    euclidean`) first. `dot`'s "distance" isn't a real metric and is
    negative for exactly the points that are genuinely close, which used to
    make this feature silently NaN for every `dot`-metric workload.
4. Recall is a feasibility constraint, never a third BO objective; the
   recall model is classification-only. This includes `recall_online.py`'s
   run-local GP: it feeds only `W_R` and must never enter EHVI or the
   Pareto front, however tempting "just add it as a third GP" looks once
   `surrogate.py` already has the plumbing.
4b. `recall_online.py`'s GP is trained only on this run's live measurements
    — never on `data.csv`, never through the classifier's LODO training
    path. It's a separate, disposable-per-run object, not an alternative to
    `recall.py`.
5. Cross-threshold monotonicity comes from the `threshold: −1` monotone
   constraint — don't reintroduce independent per-threshold heads.
6. The `strategy` baselines exist for the research comparison; `bo_cost`
   must stay recall-free, `bo` must stay cost- and recall-oblivious (it
   uses the all-points front deliberately).
6b. This contract holds during cold start too, not just once the GP
    surrogate is ready. Cold-start scoring must go through
    `acquisition.score` (with a diversity factor standing in for EHVI), not
    a separate hand-rolled formula — the latter previously baked `W_R` and
    `cost` into every strategy's cold-start score regardless of what that
    strategy is supposed to zero out.
7. The EHVI reference-point margin must scale with the observed objective
   span (a fixed epsilon re-introduces the front-extension blind spot).
8. A missing artifact prefix invalidates all levels beneath it in
   `ArtifactCache.missing_levels`.
8b. Layouts are independent, permanent artifacts (one collection each in
    `LiveQdrantEvaluator`); index/quant are **not** — they're in-place
    `reindex` mutations of a layout's one collection, so a layout can have
    only one *currently live* index_key/quant_key at a time. Do not go back
    to tracking "every index/quant key ever built" as a growing set
    (`ArtifactCache.indexes`/`.quants` used to be exactly this, and it let a
    stale key look reusable after a different one had physically overwritten
    it — a real live-measurement-corrupting bug, not a style issue).
9. `segment_size_kb` is derived with half-up rounding
   (`int(bytes/segments/1024 + 0.5)`) to reproduce the training pipeline.
10. `OnlineRecalibrator.add` must always be fed the raw offline classifier
    probability, never the blended/recalibrated one — `Optimizer._predict_recall`
    returns `(blended, raw)` for exactly this reason. Collapsing that back to
    a single prediction reintroduces a feedback loop (each Platt fit
    correcting its own previous output instead of the fixed offline prior).
