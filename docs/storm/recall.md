# Recall measurement

`nova storm` measures latency and throughput always, and **recall** whenever the
query file carries a ground-truth column. Recall is the subtle half: what it
means depends on how deep the ground truth is, and duplicate documents in a
corpus make the "correct" answer genuinely ambiguous at the top-k boundary.

This page covers what storm reports and why.

## Configuring it

```yaml
query:
  source:
    uri: s3://…/queries_with_gt.parquet
    column: dense_embedding
    ground_truth_column: hit_ids          # nova bf's own output, reused directly
    ground_truth_score_column: hit_scores # optional; enables the tie-aware numbers
  top_k: 10
  # tie_epsilon: 2e-3                     # optional; default is derived per datatype
```

`ground_truth_column` alone gives exact id-set recall. Adding
`ground_truth_score_column` lets storm tell a genuine miss from a tie — see
[Ties](#ties-why-recall-is-a-range).

## Ground truth deeper than `top_k` is truncated

A `nova bf` run costs a full pass over the corpus, so one k=1000 ground truth
is normally reused across many sweeps at smaller `top_k`. Storm truncates it to
the first `top_k` ids at load — `hit_ids` is written best-first, so the prefix
IS the true top-k — and says so:

```
INFO ground truth truncated to top_k=10 for 10000 queries (deepest list held
     1000 ids) — recall is measured against the true top-10, NOT against every
     id in the list
```

Without that truncation, a returned id counted as a hit if it appeared
*anywhere* in the list. For a k=1000 ground truth at `top_k=10` that asks "are
my 10 in the true top-1000" rather than "are they the true top-10" — a far more
forgiving number reported under the same name. On a deliberately degraded
collection the two differ by 4x.

!!! warning "This changed"
    Runs before this change reported the forgiving number. Recall will look
    like it dropped; it didn't — it got measured properly. `schema_version: 2`
    in the JSON marks the new semantics.

A ground truth **shallower** than `top_k` is left alone: it cannot answer a
deeper question. Under a selective filter a query may have only 3 matching
documents in the whole corpus, so there is no "true top-10" to have returned —
dividing by 10 anyway would score a *perfect* engine 0.3 and make it look
broken. Those queries are scored against their own length and reported
separately as `recall@k_short`.

## Ties: why recall is a range

Duplicate and near-duplicate documents produce identical embeddings, so
identical scores. When several documents share the score at the ground truth's
k-th place, they are **equally correct** — the ground truth recorded one of
them, and an engine that returns a different one is not wrong.

With `ground_truth_score_column` set, storm reports recall as a range:

```
       recall@10: 0.8630 – 1.0000  (100 eligible queries, 26377 firings)
  ties_at_cutoff: 6.5 avg, 12 max — 100.0% of queries
     tie_epsilon: 2.0e-3 (auto, float16)
```

* **lower bound** — exact id match. Every id counted is unambiguously correct.
* **upper bound** — results absent from the ground truth but scoring the same
  as its k-th place also count.
Every bucket carries its own bound: `recall@k`, `recall@k_short` and
`recall_total` each print as a range when ties make their value ambiguous.
Ties are not a full-bucket phenomenon — under a selective filter a shallow
ground truth is the norm, and those cutoffs are the ones most likely to be
tied.

* the **gap** is what tie ambiguity could account for — not a retrieval
  failure, but not purely exact ties either: any non-ground-truth result within
  `tie_epsilon` of the cutoff widens it, so a loose tolerance inflates the upper
  bound just as it mutes `missing_from_gt`.

The run above is an exact search over a corpus of 6x duplicates: it is
*perfect*, and plain id-set recall still calls it 0.863. Everything between the
bounds is tie-break disagreement.

Without a score column the bounds collapse to one number — the historical
behavior.

!!! note "Qdrant only, for now"
    Only the Qdrant target reports per-result scores and can describe its own
    collection. Milvus and Elastic use the default `scoring_profile()`, so
    their distance function is unknown and tie reporting is **disabled** with
    that reason — exact recall is unaffected. Left that way deliberately: Milvus returns *distances*
    for L2 (ascending, not descending similarity), and guessing the sign
    convention would silently mis-call ties rather than fail loudly.

## `tie_epsilon`

Scores are not bit-identical between an exact brute force and a live engine:
the engine may store vectors at a narrower `datatype` and accumulates in a
different order. `tie_epsilon` is the RELATIVE tolerance (`|a-b| / (1+|b|)`)
within which two scores count as the same.

Leave it unset and storm probes the collection's `datatype` at startup and
picks a default. Measured worst-case relative gap between nova-bf's published
`hit_scores` and live Qdrant exact search (4,000 dim-128 vectors, 100 queries,
top-20):

| datatype | dot     | cosine  | default |
|----------|---------|---------|---------|
| float32  | 6.5e-07 | 1.5e-07 | `5e-06` |
| float16  | 4.2e-04 | 7.3e-05 | `2e-03` |

Each default carries 5–8x headroom. A collection created without an explicit
`datatype` reports none, and Qdrant's default storage is float32, so that case
takes the float32 value. `Turbo4` (which this client build cannot yet create)
and any backend that cannot report its datatype get the conservative value.

A too-large tolerance is NOT harmless. `recall_at_k` tests the tie window
before the "better than the k-th" test, so a result inside the window is booked
as a tie rather than a mismatch — a tolerance far above the real noise floor
mutes `missing_from_gt`. A too-small one under-reports ties and narrows the
upper bound. Neither error is silent-but-safe; set it explicitly if the
measured table does not describe your setup.

Qdrant has no bfloat16 storage type, so bf16 never applies on the collection
side (it can matter when *producing* embeddings, but those are fp32 by the time
storm sees a score).

### The ground truth's own precision counts too

Those defaults assume the ground truth was computed in exact fp32 — `nova bf`'s
default, since `params.allow_tf32` is off precisely to keep scores bit-exact.

`nova bf` has no bf16 or fp16 compute mode: a corpus stored as fp16 is upcast
to fp32 at load, so storage dtype never affects the ground truth's precision.
Parquet has no bf16 type either — vectors "in bf16" are a float32 column whose
values happen to be bf16-representable, and those are exact float32 values, so
they need no special handling. What matters is that BOTH sides score the same
values: a ground truth built from bf16-valued queries does not match a run that
sends fp32 queries, and that gap (~1e-2 relative) is far past any tolerance.
`allow_tf32` is the only knob that does, and it carries roughly **3e-4**
relative error by nova-bf's own measurement — about 60x the fp32 default here.

So if the ground truth was produced with `allow_tf32: true`, set `tie_epsilon`
explicitly to `1e-3` or looser. Storm cannot detect this: a `nova bf` result
parquet records no provenance, so nothing in the file says how it was
computed.

### When tie reporting is switched off

Some configurations put returned scores in a different space from the ground
truth's rather than merely a noisier one. No tolerance rescues those, so storm
**withholds every tie-derived field** — the range, `ties_at_cutoff`,
`tie_epsilon`, and `missing_from_gt` — and prints why. Exact recall is
unaffected.

```
       recall@10: 0.1020  (100 eligible queries, 25147 firings)
   tie_reporting: disabled — the collection is quantized and this run sets
                  quantization.rescore=false, so returned scores are in
                  quantized space. Exact recall above is unaffected.
```

Two triggers, both probed from the collection at startup:

**Quantization with `rescore: false`.** Normally quantization is invisible to
scores — with rescoring on, Qdrant recomputes finals against the original
vectors. Measured relative error vs an exact fp32 reference, ANN path engaged:

| case | median | max |
|---|---|---|
| scalar int8, `rescore: true` | 2.7e-08 | 2.4e-07 |
| binary, `rescore: true` | 2.5e-08 | 2.3e-07 |
| scalar int8, `rescore: false` | 6.1e-03 | 3.6e-02 |
| binary, `rescore: false` | 5.2e-01 | **26.4** |

Two other things disable it: an unknown distance function (below), and a
backend that cannot describe its collection at all. Everything else that looked
like a mismatch is handled rather than refused.

### Distance conventions are normalized, not rejected

`nova bf` stores euclid/manhattan NEGATED so larger is nearer (its top-k picks
the nearest); Qdrant returns the raw positive distance. Rather than treat those
as incomparable, storm puts both in the same orientation:

* the engine's score is negated when the collection's distance is
  `euclid`/`manhattan`, probed at startup;
* the ground truth's cutoff is normalized using the ordering of its OWN score
  list — descending means larger-is-better, ascending means raw distances.
  Inferred rather than assumed, so a ground truth produced by something other
  than `nova bf` still works.

Verified against nova-bf euclidean ground truth after normalization:

| datatype | median | max | default tolerance |
|---|---|---|---|
| float32 | 3.1e-08 | 1.7e-07 | `5e-06` |
| float16 | 1.6e-05 | 9.1e-05 | `2e-03` |

Both sit inside the existing per-datatype defaults, so euclidean needs no
special tolerance.

A ground truth whose scores are all equal — or that holds a single hit, common
under a selective filter — gives no ordering signal. Orientation then falls
back to the sign: against a raw-distance engine, a positive cutoff can only be
a raw distance, since `nova bf` stores those negated.

If the collection's distance cannot be determined at all, tie reporting is
disabled rather than assuming an orientation. Guessing wrong there does not
merely lose a number — it makes `missing_from_gt` fire on nearly every miss and
prints "stale GT or wrong corpus" over a healthy run.

### `datatype: uint8` warns rather than disables

A uint8 collection fed ordinary floats measured up to **14.1** relative error:
components truncate toward integers, so the collection holds a different vector
space than the ground truth describes. But a uint8 collection whose vectors
were pre-scaled to integers, with a ground truth built from those same values,
measured ~0 error and is perfectly valid. Storm cannot tell the two apart, so
it warns instead of withholding numbers that may be correct.

## Summary fields

| field | meaning |
|---|---|
| `recall@k` | one number, or a `lower – upper` range when ties make it ambiguous |
| `recall@k_short` | queries whose ground truth held fewer than `top_k` ids, scored against their own (deduplicated) length. Carries its own tie-tolerant bound, so it prints as a range too |
| `recall_total` | both buckets blended, with its own bound — printed only when both exist |
| `ties_at_cutoff` | avg/max ground-truth documents sharing the k-th score, and the share affected — over every query with a derived cutoff in EITHER bucket, which is why it states its own denominator rather than reusing the recall line's |
| `tie_epsilon` | the tolerance in use and where it came from; absent (and `null` in JSON) when tie reporting is disabled, or when no score column was configured |
| `missing_from_gt` | results scoring **better** than the k-th place yet absent from the ground truth |
| `short_returns` | firings returning fewer than the corpus can supply for that query — `top_k`, or the ground truth's own depth when it is shallower (as under a selective filter). Empty ground truths excluded. |
| `recall_empty_gt` | firings whose ground-truth list was present but empty |
| `filter_overreturn` | with a filter set, firings returning more ids than the ground truth holds |

### Which number to report

The true value lies **somewhere inside the range**, and no further measurement
narrows it: which member of a tied group is "correct" depends on a tie-break
convention neither side has agreed to. The width tells you how much of the
result is ambiguous rather than wrong.

* **Quote the range.** It is the honest answer, and its width is itself a
  finding — a wide range means your ground truth's cutoff is heavily tied.
* **If you must quote one number, quote the exact (lower) one.** It is the
  defensible floor: everything counted is unambiguously correct. Just never
  compare it against someone else's tie-tolerant figure — that is comparing a
  floor to a ceiling.
* **Avoid quoting `recall_total`.** It averages two quantities computed with
  different denominators, so it is not comparable to either bucket, nor to a
  conventional recall@k from any other tool. It exists to give a single
  headline for a mixed run, not to be published.

A worked example: on an exact search over a corpus with 6x duplicate
documents, storm reports `recall@10: 0.8690 – 1.0000`. The engine is
*perfect* — the 0.131 gap is entirely tie-break disagreement, and the exact
figure alone would understate it by that much.

### Which denominator each number uses

| number | numerator | denominator |
|---|---|---|
| `recall@k` | distinct returned ids present in the ground truth | `top_k` — even when the engine returned fewer, which is what `short_returns` flags |
| `recall@k` upper bound | the same, plus results within `tie_epsilon` of the cutoff | `top_k` |
| `recall@k_short` | as above | the ground truth's own **deduplicated** length |
| `recall_total` | as above | each sample's own denominator, so it blends two different ones |

The counts in parentheses distinguish **eligible queries** from **firings**.
Eligible means loaded with a non-empty ground truth at least `top_k` deep — not
the number of vectors loaded, and not necessarily the number that fired, since
a short or paced run can stop before cycling through them all. Firings is
usually higher because queries cycle round-robin; the recall `n` is firings.

Every diagnostic line is printed only when it is non-zero, so a clean run stays
short.

### `missing_from_gt` is the one to watch

It counts results that scored *better* than the ground truth's k-th place yet
are not in it at all. That is not a tie — it means the ground truth and the
collection disagree: a stale ground-truth file, or a collection built from a
different corpus version. It is never folded into the tolerant upper bound,
where it would masquerade as a recall gain.

A run with plenty of ties but a near-zero tie spread AND a non-zero
`missing_from_gt` is the signature of benchmarking against the wrong data.

## The time-series report

`report.path` writes a per-dispatch trace. Its recall columns carry the **exact**
recall only — the tie-tolerant upper bound, `missing_from_gt` and
`short_returns` live in the end-of-run summary, not in the trace. Recomputing
recall from the trace therefore reproduces the lower bound of the range the
summary printed, not the range itself.

The trace also carries no schema marker, so a file written before ground-truth
truncation cannot be told from one written after, even though the meaning of
its recall values changed. Compare traces only within a single storm version.

## JSON

`--json` emits the same fields for `nova sweep`:

```json
{"full_recall": {"n": 24244, "mean": 0.8630},
 "full_recall_tolerant": 1.0,
 "short_recall": null,
 "ties": {"mean": 6.48, "max": 12, "fraction_of_queries": 1.0},
 "top_k": 10, "full_recall_queries": 100,
 "tie_epsilon": 0.002, "tie_epsilon_source": "auto, float16",
 "missing_from_gt": 0, "short_returns": 0,
 "schema_version": 2}
```

`schema_version` exists because ground-truth truncation changed what
`full_recall` MEANS without changing its type — the only way a consumer can
tell a pre-truncation run's number from a post-truncation one.
