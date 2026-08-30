"""Ground truth computed the way it is computed in production: sharded.

Every real GT run is `nova bf compute --num-jobs W --job-rank r` on W machines,
each writing a partial over its own slice of the corpus, followed by one `nova
bf merge` reducing the W partials into the final top-K. Nothing above this file
exercises that: the rest of the harness runs `run_compute` unsharded, which is
the path production never takes.

The claim under test is that sharding is invisible. A merged W-way result must
be the same answer as the one-shot run and the same answer as both oracles —
for every W, and for both tie-break rules, since the tie-break is precisely
what decides which of two equal-scoring candidates from DIFFERENT ranks
survives the reduce.

Also covered here: the guards that stop a merge from silently producing a
plausible wrong answer out of partials that never belonged together.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from . import compare, qdrant_ref
from .cases import FILTERS, K
from .runner import build_config, read_results, pinned_device, spec
from .test_parity_matrix import _filter_from_dict

from nova_bf.compute import run_compute
from nova_bf.merge import run_merge
from nova_bf.results import partial_dir

# A spread that keeps every reduce-time behaviour in play: several vector
# types (so each rank writes several partial dirs), a filter that removes most
# candidates (so some rank contributes nothing to some query's top-K), and a
# per-query filter (whose mask has to survive the shard split).
SHARDED = [
    ("sh_dense", "dense", "cosine", None),
    ("sh_dense_f", "dense", "euclidean", FILTERS["match"]),
    ("sh_sparse", "sparse", "dot", FILTERS["matchany"]),
    ("sh_mv", "multivector", "dot", FILTERS["pqrange"]),
]
IDS = [s[0] for s in SHARDED]

# 1 = the degenerate shard (a control: the sharded code path with one rank must
# still equal the unsharded one). 3 does not divide the 4 corpus files, so at
# least one rank gets a different number of files than the others. 5 exceeds
# the file count, so at least one rank gets NO files at all and must still
# write a well-formed empty partial rather than be absent.
NUM_JOBS = [1, 3, 5]


def _specs():
    return [spec(name, vector_type=vt, metric=m, k=K, filter=f)
            for name, vt, m, f in SHARDED]


def _run_sharded(ds, num_jobs, *, device, tag, tiebreak="ordinal"):
    """compute × num_jobs ranks, then one merge — the production shape."""
    cfg = build_config(ds, _specs(), out_tag=f"{tag}_{device or 'auto'}",
                       params={"tiebreak": tiebreak})
    with pinned_device(device):
        for rank in range(num_jobs):
            run_compute(cfg, num_jobs=num_jobs, job_rank=rank)
        return read_results(run_merge(cfg)), cfg


@pytest.fixture(scope="session")
def sharded_runs(ds, device):
    """`{num_jobs: results}` — each W computed ONCE and shared by every case.

    Without this the W-way compute+merge would re-run per parametrized search,
    which is the same work repeated for a result that does not depend on which
    search is asserting on it.
    """
    return {w: _run_sharded(ds, w, device=device, tag=f"shard{w}")[0]
            for w in NUM_JOBS}


@pytest.fixture(scope="session")
def unsharded(ds, device):
    cfg = build_config(ds, _specs(), out_tag=f"unsharded_{device or 'auto'}")
    with pinned_device(device):
        return read_results(run_compute(cfg))


@pytest.mark.parametrize("num_jobs", NUM_JOBS)
@pytest.mark.parametrize("entry", SHARDED, ids=IDS)
def test_a_merged_shard_run_matches_the_naive_oracle(
    entry, num_jobs, ds, oracle, sharded_runs, device
):
    """The end-to-end claim, against the reference rather than against another
    nova-bf run — so a sharding bug that corrupted compute and merge
    consistently still fails."""
    name, vt, metric, fdict = entry
    got = sharded_runs[num_jobs]
    want = oracle.topk(vector_type=vt, metric=metric, k=K,
                       filt=_filter_from_dict(ds, fdict))
    for qi in range(len(ds.queries)):
        compare.assert_scores_agree(
            got[name][qi], want[qi], metric=metric,
            label=f"[{device}] {name} merged from {num_jobs} ranks q{qi}")


@pytest.mark.qdrant
@pytest.mark.parametrize("num_jobs", [3])
@pytest.mark.parametrize("entry", SHARDED, ids=IDS)
def test_a_merged_shard_run_matches_qdrant(entry, num_jobs, ds, client,
                                            collection, sharded_runs, device):
    name, vt, metric, fdict = entry
    got = sharded_runs[num_jobs]
    want = qdrant_ref.topk(client, collection, ds, vector_type=vt, metric=metric,
                           k=K, filt=_filter_from_dict(ds, fdict))
    for qi in range(len(ds.queries)):
        compare.assert_scores_agree(
            got[name][qi], want[qi], metric=metric,
            label=f"[{device}] {name} merged from {num_jobs} ranks q{qi} vs qdrant")


@pytest.mark.parametrize("num_jobs", NUM_JOBS)
@pytest.mark.parametrize("entry", SHARDED, ids=IDS)
def test_sharding_does_not_change_the_answer(entry, num_jobs, ds, unsharded,
                                              sharded_runs, device):
    """Sharded vs one-shot directly. How the corpus was divided across ranks
    cannot move a document in or out of the top-K, so membership is asserted
    exactly; scores get the metric's tolerance, since a rank sums over fewer
    rows than the one-shot pass does."""
    name, _vt, metric, _f = entry
    got = sharded_runs[num_jobs]
    for qi in range(len(ds.queries)):
        label = f"[{device}] {name} q{qi}: {num_jobs}-way merged vs one-shot"
        compare.assert_same_membership(got[name][qi], unsharded[name][qi], label=label)
        compare.assert_scores_agree(got[name][qi], unsharded[name][qi],
                                    metric=metric, label=label)


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
def test_both_tiebreak_rules_survive_the_reduce(tiebreak, ds, oracle, device):
    """The tie-break is what decides which of two equal-scoring candidates from
    DIFFERENT ranks survives the merge, so it is the one setting whose meaning
    is specific to sharding. Both rules must still produce a legitimate top-K.

    (Which of two tied documents each rule PICKS is pinned by the dedicated
    tie-break suites; what is checked here is that reducing across ranks under
    either rule still agrees with the oracle.)
    """
    got, _ = _run_sharded(ds, 3, device=device, tag=f"tb_{tiebreak}",
                          tiebreak=tiebreak)
    for name, vt, metric, fdict in SHARDED:
        want = oracle.topk(vector_type=vt, metric=metric, k=K,
                           filt=_filter_from_dict(ds, fdict))
        for qi in range(len(ds.queries)):
            compare.assert_scores_agree(
                got[name][qi], want[qi], metric=metric,
                label=f"[{device}] tiebreak={tiebreak} {name} q{qi}")


def test_merge_refuses_a_run_that_is_missing_a_rank(ds, device):
    """A rank that dies before writing leaves its partial absent. The remaining
    partials merge perfectly cleanly — they are individually valid, they just
    cover less of the corpus than the run claimed — so the result is a wrong
    top-K with nothing anomalous about it.

    This is the failure mode a parity suite can never catch by comparison,
    because a corpus subset's exact top-K is a perfectly self-consistent
    answer. It has to be refused at the boundary instead, and that refusal is
    what is asserted here.
    """
    cfg = build_config(ds, _specs(), out_tag=f"missing_{device or 'auto'}")
    with pinned_device(device):
        for rank in range(3):
            run_compute(cfg, num_jobs=3, job_rank=rank)

    victim = SHARDED[0][0]
    pdir = Path(cfg.output.path) / partial_dir(
        cfg, next(s for s in cfg.searches if s.name == victim))
    ranks = sorted(pdir.glob("rank*.parquet"))
    assert len(ranks) == 3, [p.name for p in ranks]
    ranks[1].unlink()   # rank 1 "died"

    with pytest.raises(Exception) as exc:
        run_merge(cfg)
    assert "rank" in str(exc.value).lower(), (
        f"merge failed, but not with a message identifying the missing rank: "
        f"{exc.value}")


def test_merge_refuses_partials_from_two_different_runs(ds, device):
    """A partial directory is addressed by (queries stem, search name, k), so a
    re-run with different settings writes into the SAME directory and
    overwrites only the ranks it has. A 2-rank re-run over a 3-rank run's
    leftovers leaves one stale partial, and stale slices reduce cleanly —
    double-counting the corpus they overlap, missing what nobody covered.

    Again: not detectable by comparison, only by refusing it.
    """
    cfg3 = build_config(ds, _specs(), out_tag=f"mixed_{device or 'auto'}")
    with pinned_device(device):
        for rank in range(3):
            run_compute(cfg3, num_jobs=3, job_rank=rank)
        # A second run over the same output, 2 ranks — leaves rank2 from the
        # first run behind.
        for rank in range(2):
            run_compute(cfg3, num_jobs=2, job_rank=rank)

    with pytest.raises(Exception) as exc:
        run_merge(cfg3)
    msg = str(exc.value).lower()
    assert "run" in msg or "rank" in msg or "fingerprint" in msg, (
        f"merge failed, but not with a message identifying the mixed run: {exc.value}")
