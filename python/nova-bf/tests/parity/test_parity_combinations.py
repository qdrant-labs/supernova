"""Which searches share a run changes the code that runs. So: permutations.

`test_parity_matrix` puts all 119 cases in ONE config, which is a combined run
— but always the SAME combined run, and that pins choices `run_compute` makes
from the spec list as a whole:

  * **The batch grid per vector_type.** If ANY search of a vector_type is
    unfiltered, that vt's grid is the whole file. If EVERY one is filtered, the
    grid is instead the union of their surviving rows (`_union_keep`), and every
    row number a hit reports has to survive a round trip through that
    compaction. The matrix always contains a `nofilter` spec, so it only ever
    exercises the first branch — the compacted one is reached only by a run
    where nothing is unfiltered.
  * **Which vector_types are present at all.** A file's reader decodes one
    column per vt needed, and each vt gets its own shared pass. Three at once
    is not the same code as one.
  * **Filter dedup.** A filter is evaluated once per file and shared by every
    search naming it — so two searches with the SAME filter and two with
    DIFFERENT ones are different bookkeeping.

This file walks all seven non-empty subsets of {dense, sparse, multivector},
each in three filter regimes, and checks every search in every combination
against both oracles. The answer must not depend on what a search was
scheduled alongside.
"""

from __future__ import annotations

import itertools

import pytest

from . import compare, naive, qdrant_ref
from .cases import FILTERS, K
from .runner import run, spec
from .test_parity_matrix import _filter_from_dict

VECTOR_TYPES = ("dense", "sparse", "multivector")
SUBSETS = [
    s for n in range(1, 4) for s in itertools.combinations(VECTOR_TYPES, n)
]
SUBSET_IDS = ["+".join(s) for s in SUBSETS]

# Metrics per vector_type. Dense carries two so the shared-Gram path stays live
# inside every combination, not just in the big matrix run.
METRICS = {"dense": ("cosine", "euclidean"), "sparse": ("dot",),
           "multivector": ("dot",)}

# The three regimes. The distinction that matters is whether ANY search of a
# vector_type is unfiltered — that is the switch between an uncompacted
# whole-file grid and `_union_keep`'s compacted row union.
REGIMES = {
    # some search unfiltered -> whole-file grid for its vector_type
    "baseline": (None, "match", "pqrange"),
    # nothing unfiltered, all static -> compacted union of two static masks
    "filtered_static": ("match", "matchany"),
    # nothing unfiltered, all per-query -> the per-query union path
    "filtered_perquery": ("pqmatch", "pqrange"),
}


def _specs(subset, regime):
    """One config's worth of searches: every metric of every vector_type in
    `subset`, crossed with the regime's filters."""
    out = []
    for vt in subset:
        for metric in METRICS[vt]:
            for fname in REGIMES[regime]:
                out.append((
                    vt, metric, fname,
                    spec(f"{vt[:2]}{metric[:3]}_{fname or 'none'}",
                         vector_type=vt, metric=metric, k=K,
                         filter=FILTERS[fname] if fname else None),
                ))
    return out


@pytest.mark.parametrize("regime", list(REGIMES))
@pytest.mark.parametrize("subset", SUBSETS, ids=SUBSET_IDS)
def test_every_search_survives_every_combination(subset, regime, ds, oracle, device):
    """Each search in the combination must return what it would return alone —
    checked against the naive oracle, so a combination that corrupted every
    search in it identically still fails."""
    entries = _specs(subset, regime)
    got = run(ds, [s for *_, s in entries],
              out_tag=f"combo_{'_'.join(subset)}_{regime}", device=device,
              params={"dense_batch_size": 37, "sparse_batch_size": 29,
                      "multivector_batch_size": 23})
    for vt, metric, fname, s in entries:
        want = oracle.topk(vector_type=vt, metric=metric, k=K,
                           filt=_filter_from_dict(ds, FILTERS[fname] if fname else None))
        for qi in range(len(ds.queries)):
            compare.assert_scores_agree(
                got[s["name"]][qi], want[qi], metric=metric,
                label=f"[{device}] {'+'.join(subset)}/{regime} {s['name']} q{qi}")


@pytest.mark.qdrant
@pytest.mark.parametrize("regime", list(REGIMES))
@pytest.mark.parametrize("subset", SUBSETS, ids=SUBSET_IDS)
def test_every_combination_still_agrees_with_qdrant(
    subset, regime, ds, client, collection, device
):
    entries = _specs(subset, regime)
    got = run(ds, [s for *_, s in entries],
              out_tag=f"comboq_{'_'.join(subset)}_{regime}", device=device,
              params={"dense_batch_size": 37, "sparse_batch_size": 29,
                      "multivector_batch_size": 23})
    for vt, metric, fname, s in entries:
        want = qdrant_ref.topk(
            client, collection, ds, vector_type=vt, metric=metric, k=K,
            filt=_filter_from_dict(ds, FILTERS[fname] if fname else None))
        for qi in range(len(ds.queries)):
            compare.assert_scores_agree(
                got[s["name"]][qi], want[qi], metric=metric,
                label=f"[{device}] {'+'.join(subset)}/{regime} {s['name']} q{qi} vs qdrant")


@pytest.mark.parametrize("subset", SUBSETS, ids=SUBSET_IDS)
def test_removing_the_unfiltered_search_does_not_move_the_filtered_ones(
    subset, ds, device
):
    """The sharpest form of the grid question, as a direct A/B.

    The same filtered searches are run twice: once alongside an unfiltered
    search of the same vector_type (whole-file grid) and once alone (compacted
    `_union_keep` grid). Only the tiling differs, so the answer may move by a
    ULP but no document may enter or leave the ranking — which is exactly what
    `assert_same_membership` refuses to tolerate.

    This is the pair the matrix cannot express: it always contains the
    unfiltered search, so it only ever sees the left-hand side.
    """
    filtered = [
        (vt, metric, fname,
         spec(f"{vt[:2]}{metric[:3]}_{fname}", vector_type=vt, metric=metric,
              k=K, filter=FILTERS[fname]))
        for vt in subset for metric in METRICS[vt]
        for fname in ("match", "matchany")
    ]
    baselines = [
        spec(f"{vt[:2]}{METRICS[vt][0][:3]}_base", vector_type=vt,
             metric=METRICS[vt][0], k=K)
        for vt in subset
    ]
    tag = "_".join(subset)
    with_baseline = run(ds, [s for *_, s in filtered] + baselines,
                        out_tag=f"grid_with_{tag}", device=device)
    alone = run(ds, [s for *_, s in filtered], out_tag=f"grid_alone_{tag}",
                device=device)

    for vt, metric, fname, s in filtered:
        for qi in range(len(ds.queries)):
            label = (f"[{device}] {'+'.join(subset)} {s['name']} q{qi}: "
                     "whole-file grid vs compacted union grid")
            compare.assert_same_membership(
                with_baseline[s["name"]][qi], alone[s["name"]][qi], label=label)
            compare.assert_scores_agree(
                with_baseline[s["name"]][qi], alone[s["name"]][qi],
                metric=metric, label=label)


def test_the_compacted_regime_really_does_compact(ds):
    """A guard on the fixture, not on nova-bf: `filtered_static`'s point is to
    make `_union_keep` produce a STRICT subset of the file. If its filters
    between them happened to cover every row, the union would be the whole
    file, the compaction would be a no-op, and every test above that claims to
    exercise the compacted grid would be exercising the uncompacted one under
    a different name."""
    union = set()
    for fname in REGIMES["filtered_static"]:
        filt = _filter_from_dict(ds, FILTERS[fname])
        for q in ds.queries:
            union |= {d.row for d in ds.docs
                      if naive.filter_matches(filt, d.payload, q["payload"],
                                              ds.date_fields, ds.query_date_fields)}
    assert K < len(union) < len(ds.docs), (
        f"the `filtered_static` regime's row union is {len(union)} of "
        f"{len(ds.docs)} rows — it must be a strict subset for the compacted "
        "grid to differ from the whole-file one")


@pytest.mark.parametrize("subset", SUBSETS, ids=SUBSET_IDS)
def test_the_order_searches_are_listed_in_does_not_matter(subset, ds, device):
    """`searches` is a list, and the run groups it by vector_type and by
    filter. Reversing it changes the grouping order, the order filters are
    first seen in, and the order each vt's shared pass encounters its specs —
    none of which is allowed to reach the results.

    Reordering a list is not a numerical operation, so on every path whose
    scores are bit-reproducible this is asserted bit-for-bit. The exception is
    multivector on CUDA, whose scores are not reproducible even between two
    IDENTICAL runs (see `compare.scores_are_reproducible`) — there the id
    sequence is still compared exactly and only the score bits are given the
    metric's tolerance.
    """
    entries = _specs(subset, "baseline")
    specs = [s for *_, s in entries]
    tag = "_".join(subset)
    forward = run(ds, specs, out_tag=f"order_fwd_{tag}", device=device)
    reverse = run(ds, list(reversed(specs)), out_tag=f"order_rev_{tag}",
                  device=device)
    for (vt, metric, _fname, s) in entries:
        for qi in range(len(ds.queries)):
            compare.assert_same_ranking(
                forward[s["name"]][qi], reverse[s["name"]][qi],
                metric=metric, device=device, vector_type=vt,
                label=f"[{device}] {'+'.join(subset)} {s['name']} q{qi}: "
                      "spec order changed the result")


@pytest.mark.parametrize("subset", SUBSETS, ids=SUBSET_IDS)
def test_one_filter_shared_across_vector_types(subset, ds, oracle, device):
    """A single filter naming ONE corpus column, used by searches of every
    vector_type in the run. The filter is evaluated once per file and shared
    across the vt boundary — so a vt that mutated the shared mask, or read it
    against its own compacted row numbering rather than the file's, would
    corrupt only its neighbours and pass every single-vt test."""
    fname = "compound"
    entries = [
        (vt, METRICS[vt][0],
         spec(f"{vt[:2]}_shared", vector_type=vt, metric=METRICS[vt][0],
              k=K, filter=FILTERS[fname]))
        for vt in subset
    ]
    got = run(ds, [s for *_, s in entries],
              out_tag=f"shared_filter_{'_'.join(subset)}", device=device)
    filt = _filter_from_dict(ds, FILTERS[fname])
    for vt, metric, s in entries:
        want = oracle.topk(vector_type=vt, metric=metric, k=K, filt=filt)
        for qi in range(len(ds.queries)):
            compare.assert_scores_agree(
                got[s["name"]][qi], want[qi], metric=metric,
                label=f"[{device}] {'+'.join(subset)} shared-filter {vt} q{qi}")
