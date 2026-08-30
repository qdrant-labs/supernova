"""The hand-written Triton kernels, against the portable paths they replace.

nova-bf carries three custom CUDA kernels, each an optimization of something
the portable torch path already computes:

  * `topk_triton`   — pre-top-K with the tie-break applied during selection,
                      instead of packing `(score, ordinal)` into int64 keys and
                      calling `torch.topk` on the expanded matrix;
  * `merge_triton`  — the running top-K fold, fusing concatenate + select +
                      gather so no intermediate concatenation is materialized;
  * the multivector `triton_reduce` MaxSim reduction (covered in
    `test_parity_devices.py`, alongside the other CUDA-only checks).

Each is claimed to be EXACT, not approximate: `topk_triton.topk`'s contract is
that its keys are bit-identical to `tiebreak.pack(...).gather(...)`. So unlike
tiling or device changes, there is no rounding to concede here — the assertions
below are for bit-equality, and anything less would be the kernel failing its
own contract rather than a tolerance question.

nova-bf already exposes the switches this needs, for operators who suspect a
kernel on their hardware: `NOVA_BF_NO_TOPK_KERNEL` and `NOVA_BF_NO_FOLD_KERNEL`.
That makes the A/B an end-to-end one through `run_compute` — the same config,
the same data, one flag — rather than a unit test of the kernel in isolation
(which `tests/test_tiebreak_kernel.py` and `tests/test_tiebreak_fold_kernel.py`
already do well). What is added here is that the kernels are exact *in situ*,
composed with filters, sharing, and every vector_type.

On a CPU box the kernels are unreachable, so the A/B tests skip and only the
tie-break-rule coverage runs.
"""

from __future__ import annotations

import pytest

from . import compare, qdrant_ref
from .cases import FILTERS, K
from .devices import has_cuda, parity_devices
from .runner import env, run, spec
from .test_parity_matrix import _filter_from_dict

# Kernel eligibility depends on the SHAPES a run produces, so the probes span
# every vector_type and mix filtered with unfiltered (the filtered ones compact
# the grid, giving the fold narrower pending parts and the pre-top-K narrower
# slices).
PROBES = [
    ("k_dense", "dense", "cosine", None),
    ("k_dense_f", "dense", "dot", FILTERS["match"]),
    ("k_dense_e", "dense", "euclidean", FILTERS["rangeint"]),
    ("k_sparse", "sparse", "dot", None),
    ("k_sparse_f", "sparse", "cosine", FILTERS["compound"]),
    ("k_mv", "multivector", "dot", None),
    ("k_mv_f", "multivector", "cosine", FILTERS["pqtext"]),
]
IDS = [p[0] for p in PROBES]

# Deliberately small, and deliberately BELOW k. Small so a file becomes many
# slices and the fold is called repeatedly per query rather than once — the
# fold's job is the RUNNING state, and a single call would never exercise the
# state/pending interaction at all. Below k because that is what forces the
# extra merge rounds nova-bf warns about: more fold calls per query, each with
# a partially-filled state, which is the case its sentinel handling exists for.
PARAMS = {"dense_batch_size": 31, "sparse_batch_size": 19,
          "multivector_batch_size": 17}

needs_cuda = pytest.mark.skipif(
    not has_cuda(),
    reason=f"the Triton kernels need CUDA (parity devices: {parity_devices()})",
)

# The four corners of the kernel switch matrix.
KERNEL_MODES = {
    "both": {"NOVA_BF_NO_TOPK_KERNEL": None, "NOVA_BF_NO_FOLD_KERNEL": None},
    "no_topk": {"NOVA_BF_NO_TOPK_KERNEL": "1", "NOVA_BF_NO_FOLD_KERNEL": None},
    "no_fold": {"NOVA_BF_NO_TOPK_KERNEL": None, "NOVA_BF_NO_FOLD_KERNEL": "1"},
    "neither": {"NOVA_BF_NO_TOPK_KERNEL": "1", "NOVA_BF_NO_FOLD_KERNEL": "1"},
}


def _specs():
    return [spec(name, vector_type=vt, metric=m, k=K, filter=f)
            for name, vt, m, f in PROBES]


def _run_mode(ds, mode, tiebreak, *, device="cuda"):
    with env(**KERNEL_MODES[mode]):
        return run(ds, _specs(), out_tag=f"kern_{mode}_{tiebreak}", device=device,
                   params={**PARAMS, "tiebreak": tiebreak})


@pytest.fixture(scope="session")
def kernel_runs(ds):
    """Every switch combination × both tie-break rules, on CUDA. Session-scoped:
    eight compute runs is the whole cost of this file."""
    return {(mode, tb): _run_mode(ds, mode, tb)
            for mode in KERNEL_MODES for tb in ("ordinal", "id")}


@needs_cuda
@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
@pytest.mark.parametrize("mode", [m for m in KERNEL_MODES if m != "neither"])
@pytest.mark.parametrize("entry", PROBES, ids=IDS)
def test_each_kernel_is_bit_identical_to_the_portable_path(
    entry, mode, tiebreak, kernel_runs
):
    """The kernels' actual contract, end to end.

    `neither` (both kernels off) is the reference; every other switch
    combination must reproduce it EXACTLY — same ids, same order, same score
    bits. A kernel is an optimization of a selection rule, not a different
    selection rule, so a tolerance here would be conceding the very thing the
    kernel promises.

    Except where the SCORES going into the selection are themselves not
    reproducible. Multivector on CUDA is that case (see
    `compare.scores_are_reproducible`): its MaxSim sum is an `index_add_`
    atomicAdd, so the two runs being compared are not fed identical inputs and
    no selection rule, kernel or portable, could make their outputs
    bit-identical. The kernels' exactness on identical inputs is what
    `tests/test_tiebreak_kernel.py` and `tests/test_tiebreak_fold_kernel.py`
    pin directly, with fixed tensors; what this test adds is that they behave
    in situ, and for multivector that claim stops at the ranking.

    Run for BOTH tie-break rules because `topk_triton` implements them with the
    same descent over different ordinals (`ordinal` = column position, `id` =
    the rank induced by sorted ids), so a bug in the ordinal ranking shows up
    in one rule and not the other.
    """
    name, vt, metric, _f = entry
    fused = kernel_runs[(mode, tiebreak)][name]
    portable = kernel_runs[("neither", tiebreak)][name]
    for qi in sorted(portable):
        compare.assert_same_ranking(
            fused[qi], portable[qi], metric=metric, device="cuda", vector_type=vt,
            label=f"{name} q{qi}: kernels={mode} vs portable (tiebreak={tiebreak})")


@needs_cuda
@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
@pytest.mark.parametrize("entry", PROBES, ids=IDS)
def test_the_kernels_still_agree_with_the_oracles(entry, tiebreak, ds, oracle,
                                                   kernel_runs):
    """Bit-equality against the portable path proves the kernels agree with
    nova-bf. It does not prove nova-bf is right — both could share a defect in
    the surrounding scoring. So the fully-fused configuration is also checked
    against the naive reference."""
    name, vt, metric, fdict = entry
    got = kernel_runs[("both", tiebreak)][name]
    want = oracle.topk(vector_type=vt, metric=metric, k=K,
                       filt=_filter_from_dict(ds, fdict))
    for qi in sorted(got):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=metric,
            label=f"kernels-on {name} q{qi} (tiebreak={tiebreak}) vs naive")


@needs_cuda
@pytest.mark.qdrant
@pytest.mark.parametrize("entry", PROBES, ids=IDS)
def test_the_kernels_still_agree_with_qdrant(entry, ds, client, collection,
                                              kernel_runs):
    name, vt, metric, fdict = entry
    got = kernel_runs[("both", "ordinal")][name]
    want = qdrant_ref.topk(client, collection, ds, vector_type=vt, metric=metric,
                           k=K, filt=_filter_from_dict(ds, fdict))
    for qi in sorted(got):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=metric,
            label=f"kernels-on {name} q{qi} vs qdrant")


@needs_cuda
def test_the_kernels_were_actually_reached(ds, caplog):
    """The one that keeps the rest of this file honest.

    Every assertion above passes trivially if the kernel never ran — a
    fallback produces the portable answer, which is exactly what the portable
    answer is being compared against. Both modules log once, at INFO, when
    they DECLINE a run, so an unexpected decline is visible; this asserts the
    fused run did not decline, and that the kernels are importable and
    eligible for the shapes this file produces.
    """
    import logging

    from nova_bf import merge_triton, topk_triton

    assert topk_triton._cutfill is not None, "topk kernel failed to compile"
    assert merge_triton._fold is not None, "fold kernel failed to compile"

    # Reset the once-only decline latches (BOTH modules have one) so this
    # run's declines are visible rather than swallowed by an earlier test's.
    topk_triton._DECLINE_LOGGED = False
    merge_triton._DECLINE_LOGGED = False
    with caplog.at_level(logging.INFO, logger="nova_bf.topk_triton"), \
         caplog.at_level(logging.INFO, logger="nova_bf.merge_triton"):
        _run_mode(ds, "both", "ordinal")
    declines = [r.message for r in caplog.records if "does not apply" in r.message
                or "not applicable" in r.message]
    assert not declines, f"a kernel declined this run: {declines}"


# ------------------------------------------------------------ CPU-reachable


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
@pytest.mark.parametrize("entry", PROBES, ids=IDS)
def test_both_tiebreak_rules_agree_with_the_oracle_on_any_device(
    entry, tiebreak, ds, oracle, device
):
    """`params.tiebreak` selects which of two exactly-tied candidates wins.
    The rest of the harness runs the default `ordinal` throughout, so `id`
    would otherwise be exercised only by the dedicated tie-break unit tests
    and never end-to-end through a real config.

    This runs on every device, so on a CPU box it is the only part of this
    file that does anything — and on a GPU box it is also the portable-path
    control for the kernel comparisons above.
    """
    name, vt, metric, fdict = entry
    got = run(ds, _specs(), out_tag=f"tb_{tiebreak}", device=device,
              params={**PARAMS, "tiebreak": tiebreak})[name]
    want = oracle.topk(vector_type=vt, metric=metric, k=K,
                       filt=_filter_from_dict(ds, fdict))
    for qi in sorted(got):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=metric,
            label=f"[{device}] tiebreak={tiebreak} {name} q{qi}")
