"""The `params` knobs nothing else in the harness turns.

`test_parity_invariance` pins the axes the matrix run fixes — tiling, sharing,
query-row subsetting. It does not cover every knob, and a knob that silently
changes ground truth is exactly as bad whether or not a test happens to set it.
Cross-referencing `ParamsConfig` against the suite leaves five never varied
here; these are the three that gate a distinct CODE PATH rather than only a
pool size:

  * `multivector_token_budget` — splits the MaxSim similarity tile by
    `max_doc_tokens = budget // max_query_tokens`. Unset, a slice is one tile
    and the per-block loop runs once, which is the only shape the rest of the
    suite ever builds.
  * `merge_batch_size` — rows per fold in `nova bf merge`. The default resolves
    to a ceiling far above any parity corpus, so the merge always folds in ONE
    batch and the multi-batch fold is never exercised against an oracle.
  * `merge_ranged_reads` — how a partial's bytes are fetched. Every production
    config sets it true; the harness has only ever run it false.

Each is checked against the naive oracle, not merely against another nova-bf
run, so a knob that corrupted both arms identically still fails.
"""

from __future__ import annotations

import pytest

from nova_bf.merge import run_merge

from . import cases as cases_mod
from . import compare
from .runner import build_config, read_results, pinned_device, run, spec

try:  # `run_compute` moved once; keep the import obvious if it moves again
    from nova_bf.compute import run_compute
except ImportError:  # pragma: no cover
    from nova_bf import run_compute  # type: ignore

MV_PROBES = [
    cases_mod.CASES_BY_NAME[n] for n in (
        "mudot_nofilter", "mucos_match", "mudot_pqmatch", "mucos_matchtext",
    )
]
MV_IDS = [c.id for c in MV_PROBES]


def _filter_of(case, ds):
    from .test_parity_matrix import _filter_from_dict

    return _filter_from_dict(ds, case.filter_dict)


# ---------------------------------------------------------------------------
# multivector_token_budget
# ---------------------------------------------------------------------------


@pytest.fixture
def budget_spy(monkeypatch):
    """Record what the token budget actually did to the blocking.

    Without this the budget tests are ONE-SIDED: they assert the answer is
    unchanged, which a knob that does NOTHING satisfies perfectly. That was not
    hypothetical — disabling the split branch, and even forcing the config
    field to `None`, left every test in this file passing.

    `_ragged_batch_ranges` is where the budget lands, so spy there and record
    both the blocking it produced and the blocking it WOULD have produced
    unbudgeted. A test can then assert the tile really split.
    """
    from nova_bf import compute

    seen: list[tuple[int | None, int, int]] = []
    real = compute._ragged_batch_ranges

    def spy_ranges(offsets, max_rows, max_tokens):
        got = real(offsets, max_rows, max_tokens)
        unbudgeted = real(offsets, max_rows, None)
        seen.append((max_tokens, len(got), len(unbudgeted)))
        return got

    monkeypatch.setattr(compute, "_ragged_batch_ranges", spy_ranges)
    return seen


def _assert_the_budget_split_something(seen, budget):
    assert seen, "the multivector blocking never ran; this case is not multivector"
    assert any(mt is not None for mt, _, _ in seen), (
        f"budget={budget} never reached `_ragged_batch_ranges` — the knob is "
        f"being ignored, and an 'answer unchanged' assertion cannot see that")
    assert any(got > unb for _, got, unb in seen), (
        f"budget={budget} produced no more blocks than an unbudgeted run "
        f"({[(m, g, u) for m, g, u in seen[:4]]}) — the tile never split, so "
        f"this case proves nothing about split correctness")


@pytest.mark.parametrize("budget", [16, 64, 256, 4096])
@pytest.mark.parametrize("case", MV_PROBES, ids=MV_IDS)
def test_a_token_budget_does_not_change_the_answer(budget, case, ds, oracle, device,
                                                   budget_spy):
    """Splitting the MaxSim tile must be invisible.

    The budget caps `n_query_tokens x n_doc_tokens` per tile, so a small one
    turns each slice into many blocks and the per-query MaxSim is assembled
    from several partial reductions instead of one. `8` is small enough that
    the parity corpus (ragged token counts, some documents with none at all)
    splits on nearly every slice.
    """
    got = run(ds, [case.spec()], out_tag=f"tb{budget}_{case.name}", device=device,
              params={"multivector_token_budget": budget,
                      # The harness defaults `multivector_batch_size` to 2, and
                      # 2 rows of this corpus hold at most 12 tokens -- so the
                      # ROW batch dominates and the token budget never binds.
                      # That is precisely why these tests were one-sided. Give
                      # the budget room to be the constraint that splits.
                      "multivector_batch_size": 64})[case.name]
    _assert_the_budget_split_something(budget_spy, budget)
    want = oracle.topk(vector_type=case.vector_type, metric=case.metric,
                       k=case.k, filt=_filter_of(case, ds))
    for qi in range(len(ds.queries)):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=case.metric,
            label=f"[{device}] token_budget={budget} {case.id} q{qi}")


@pytest.mark.parametrize("case", MV_PROBES, ids=MV_IDS)
def test_a_tiny_token_budget_equals_an_unbudgeted_run(case, ds, device, budget_spy):
    """Exact hit-for-hit, not merely oracle-tolerable.

    A tile split changes the ORDER the per-token maxima are reduced in, so
    scores may move by an ulp — but a document may not enter or leave the
    ranking, and with the parity corpus's well-separated scores the hit list
    must be identical.
    """
    # 64, not 8: at budget 8 `max_doc_tokens` is 1 on this corpus, so every
    # tile is a single row and a tile-BOUNDARY bug cannot show up. 64 gives
    # ~12 tokens per tile, which really blocks.
    small = run(ds, [case.spec()], out_tag=f"tbsmall_{case.name}", device=device,
                params={"multivector_token_budget": 16,
                        "multivector_batch_size": 64})[case.name]
    _assert_the_budget_split_something(budget_spy, 16)
    # Same row batching, no token budget: the ONLY difference is the split.
    none = run(ds, [case.spec()], out_tag=f"tbnone_{case.name}", device=device,
               params={"multivector_batch_size": 64})[case.name]
    for qi in range(len(ds.queries)):
        assert [r for r, _ in small[qi]] == [r for r, _ in none[qi]], (
            f"[{device}] {case.id} q{qi}: a token budget changed the ranking")


# ---------------------------------------------------------------------------
# the merge knobs
# ---------------------------------------------------------------------------

MERGE_SPECS = [
    ("mk_dense", "dense", "cosine", None),
    ("mk_sparse", "sparse", "dot", {"must": [{"field": "language", "match": "eng"}]}),
    ("mk_mv", "multivector", "dot", None),
]
MERGE_IDS = [s[0] for s in MERGE_SPECS]


def _sharded(ds, device, *, tag, params=None, num_jobs=3):
    """`num_jobs` compute ranks, then one merge — the production shape."""
    specs = [spec(n, vector_type=vt, metric=m, k=cases_mod.K, filter=f)
             for n, vt, m, f in MERGE_SPECS]
    cfg = build_config(ds, specs, out_tag=f"{tag}_{device or 'auto'}",
                       params=params)
    with pinned_device(device):
        for rank in range(num_jobs):
            run_compute(cfg, num_jobs=num_jobs, job_rank=rank)
        return read_results(run_merge(cfg))


@pytest.fixture(scope="module")
def merge_default(ds, device):
    return _sharded(ds, device, tag="mk_default")


@pytest.mark.parametrize("batch_rows", [1, 2, 3])
@pytest.mark.parametrize("name", MERGE_IDS)
def test_a_small_merge_batch_gives_the_same_merged_answer(
    batch_rows, name, ds, device, merge_default
):
    """`merge_batch_size` folds the partials in row batches. The default
    resolves to a ceiling far above any parity corpus, so the merge has only
    ever folded in ONE batch here — the multi-batch fold, which is what every
    production merge actually runs, was untested against an oracle.

    Batches of 1-3 rows against 8 queries force several folds per search.
    """
    got = _sharded(ds, device, tag=f"mk_b{batch_rows}",
                   params={"merge_batch_size": batch_rows})
    for qi in range(len(ds.queries)):
        assert [r for r, _ in got[name][qi]] == [r for r, _ in merge_default[name][qi]], (
            f"[{device}] merge_batch_size={batch_rows} changed {name} at q{qi}")


def test_ranged_reads_are_out_of_reach_at_this_scale():
    """`merge_ranged_reads` cannot be exercised by this harness, and saying so
    is worth more than a test that pretends otherwise.

    `io._ranged_download` is gated on a partial being at least
    `_RANGED_GET_MIN_BYTES` (256 MiB). Parity partials are kilobytes, so
    setting the flag changes nothing here: an earlier version of this file ran
    both arms and compared them, which compared byte-identical code to itself
    and claimed to cover "the read path production actually uses". It did not,
    and `tests/test_io.py::test_ranged_path_is_actually_taken_on_a_local_file`
    already covers that path properly.

    This asserts the reason, so the gap is documented where someone would
    otherwise re-add the tautology. If the threshold is ever lowered, this
    fails and a real parity case becomes possible.
    """
    from nova_bf.io import _RANGED_GET_MIN_BYTES

    assert _RANGED_GET_MIN_BYTES >= 1 << 28, (
        "the ranged-read threshold dropped below 256 MiB; a parity case for "
        "merge_ranged_reads is now reachable and should replace this test")


@pytest.mark.parametrize("name", MERGE_IDS)
def test_the_merge_knobs_still_agree_with_the_oracle(name, ds, oracle, device):
    """Both knobs at once, against the oracle rather than against another
    nova-bf run — two merges agreeing with each other and both being wrong is
    the failure a self-comparison cannot see."""
    got = _sharded(ds, device, tag="mk_both",
                   params={"merge_batch_size": 2, "merge_ranged_reads": True})
    idx = MERGE_IDS.index(name)
    _, vt, metric, filt = MERGE_SPECS[idx]
    want = oracle.topk(vector_type=vt, metric=metric, k=cases_mod.K,
                       filt=None if filt is None
                       else _filter_of(cases_mod.CASES_BY_NAME["decos_match"], ds))
    for qi in range(len(ds.queries)):
        compare.assert_scores_agree(
            got[name][qi], want[qi], metric=metric,
            label=f"[{device}] merge knobs {name} q{qi}")
