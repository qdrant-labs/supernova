"""Does the GPU compute the same ground truth as the CPU?

On a laptop this file is almost entirely skips, and that is the intended
resting state — it exists so that moving the harness to a GPU box is nothing
but running it there. On a machine with CUDA it becomes the sharpest test in
the suite, because it compares two nova-bf runs over IDENTICAL input where the
only difference is the device: no tolerance argument about tokenizers or
metrics can absorb a failure, and the corpus is seeded, so both devices really
do see the same bytes.

What it covers beyond plain CPU-vs-CUDA:

  * `multivector_kernel="triton_reduce"` — the fused ragged MaxSim reduction,
    which only exists on CUDA and is therefore invisible to every CPU run.
  * `allow_tf32` — deliberately checked with a LOOSER tolerance, because
    TF32's 10-bit mantissa is documented as not bit-exact. The claim under
    test is not that TF32 matches f32 exactly (it does not, and a test
    asserting so would be wrong) but that it does not move the RANKING.

`NOVA_BF_PARITY_DEVICES` picks the device list; see `devices.py`.
"""

from __future__ import annotations

import pytest

from . import cases as cases_mod
from . import compare
from .devices import has_cuda, parity_devices
from .runner import run

PROBES = [
    cases_mod.CASES_BY_NAME[n] for n in (
        "dedot_nofilter", "decos_match", "deeuc_rangeint",
        "spdot_matchany", "spcos_pqtext",
        "mudot_nofilter", "mucos_compound", "mudot_pqdatetime",
    )
]
PROBE_IDS = [c.id for c in PROBES]
MV_PROBES = [c for c in PROBES if c.vector_type == "multivector"]

needs_cuda = pytest.mark.skipif(
    not has_cuda(),
    reason=f"no CUDA device available (parity devices: {parity_devices()})",
)


@pytest.fixture(scope="session")
def cpu_probe_run(ds):
    return run(ds, [c.spec() for c in PROBES], out_tag="probe", device="cpu")


@pytest.fixture(scope="session")
def cuda_probe_run(ds):
    return run(ds, [c.spec() for c in PROBES], out_tag="probe", device="cuda")


@pytest.fixture(scope="session")
def triton_run(ds):
    """The fused MaxSim kernel over the multivector probes — one run shared by
    every case, rather than one per parametrized case.

    A CUDA box without Triton installed skips instead of failing: that is a
    missing optional dependency, not a parity defect, and reporting it as the
    latter would be a false alarm on exactly the machine this suite exists to
    reassure.
    """
    try:
        return run(ds, [c.spec() for c in MV_PROBES], out_tag="triton",
                   device="cuda", params={"multivector_kernel": "triton_reduce"})
    except RuntimeError as exc:
        if "Triton" in str(exc):
            pytest.skip(f"triton_reduce kernel unavailable: {exc}")
        raise


@pytest.fixture(scope="session")
def tf32_run(ds):
    return run(ds, [c.spec() for c in PROBES], out_tag="tf32", device="cuda",
               params={"allow_tf32": True})


@needs_cuda
@pytest.mark.parametrize("case", PROBES, ids=PROBE_IDS)
def test_cuda_and_cpu_return_the_same_ranking(case, ds, cpu_probe_run, cuda_probe_run):
    """The same config, the same bytes, two devices.

    Membership is asserted EXACTLY — a document may not drop out of the top-K
    on one device and not the other. Scores are allowed the metric's usual
    tolerance, since cuBLAS and CPU BLAS tile their reductions differently,
    but a document changing eligibility is never a rounding artefact.
    """
    for qi in range(len(ds.queries)):
        cpu, cuda = cpu_probe_run[case.name][qi], cuda_probe_run[case.name][qi]
        compare.assert_same_membership(cpu, cuda, label=f"{case.id} q{qi}: cpu vs cuda")
        compare.assert_scores_agree(cpu, cuda, metric=case.metric,
                                    label=f"{case.id} q{qi}: cpu vs cuda")


@needs_cuda
@pytest.mark.parametrize("case", PROBES, ids=PROBE_IDS)
def test_cuda_agrees_with_the_naive_oracle(case, ds, oracle, cuda_probe_run):
    """CUDA against the reference directly, so a GPU-only defect cannot hide
    behind a CPU run that shares the same bug."""
    want = oracle.topk(vector_type=case.vector_type, metric=case.metric,
                       k=case.k, filt=_filter_of(case, ds))
    for qi in range(len(ds.queries)):
        compare.assert_scores_agree(cuda_probe_run[case.name][qi], want[qi],
                                    metric=case.metric,
                                    label=f"[cuda] {case.id} q{qi} vs naive")


@needs_cuda
@pytest.mark.qdrant
@pytest.mark.parametrize("case", PROBES, ids=PROBE_IDS)
def test_cuda_agrees_with_qdrant(case, ds, client, collection, cuda_probe_run):
    from . import qdrant_ref

    want = qdrant_ref.topk(client, collection, ds, vector_type=case.vector_type,
                           metric=case.metric, k=case.k, filt=_filter_of(case, ds))
    for qi in range(len(ds.queries)):
        compare.assert_scores_agree(cuda_probe_run[case.name][qi], want[qi],
                                    metric=case.metric,
                                    label=f"[cuda] {case.id} q{qi} vs qdrant")


@needs_cuda
@pytest.mark.parametrize("case", MV_PROBES, ids=[c.id for c in MV_PROBES])
def test_triton_reduce_kernel_matches_the_torch_kernel(
    case, ds, cuda_probe_run, triton_run
):
    """The fused ragged MaxSim reduction is a CUDA-only code path — on a CPU
    box it is not merely untested, it is unreachable. It must select the same
    documents as the portable torch path it replaces."""
    fused = triton_run
    for qi in range(len(ds.queries)):
        compare.assert_same_membership(
            fused[case.name][qi], cuda_probe_run[case.name][qi],
            label=f"{case.id} q{qi}: triton_reduce vs torch")
        compare.assert_scores_agree(
            fused[case.name][qi], cuda_probe_run[case.name][qi], metric=case.metric,
            label=f"{case.id} q{qi}: triton_reduce vs torch")


@needs_cuda
@pytest.mark.parametrize("case", PROBES, ids=PROBE_IDS)
def test_tf32_preserves_the_ranking(case, ds, cuda_probe_run, tf32_run):
    """TF32 keeps 10 mantissa bits, so it is NOT bit-exact against f32 and no
    test should claim it is — `params.allow_tf32` documents ~3e-4 relative
    error. What it must not do is reorder the answer, since that is the
    property a recall number computed against this GT depends on. Hence: the
    same ids, compared with a tolerance sized to TF32 rather than to f32.
    """
    tf32 = tf32_run
    for qi in range(len(ds.queries)):
        compare.assert_scores_agree(
            tf32[case.name][qi], cuda_probe_run[case.name][qi], metric=case.metric,
            tol=2e-3,
            label=f"{case.id} q{qi}: tf32 vs f32")


def test_the_device_list_is_what_this_machine_can_do():
    """A guard on the harness itself: if `parity_devices()` ever reported only
    `cpu` on a GPU box, every CUDA test above would SKIP rather than fail, and
    the suite would look green while testing nothing about the GPU."""
    import torch

    devices = parity_devices()
    assert "cpu" in devices or "cuda" in devices, devices
    import os

    if not os.environ.get("NOVA_BF_PARITY_DEVICES"):
        assert has_cuda() == torch.cuda.is_available(), (
            f"parity_devices() says {devices} but torch.cuda.is_available() is "
            f"{torch.cuda.is_available()} — the CUDA tests would silently skip")


def _filter_of(case, ds):
    from .test_parity_matrix import _filter_from_dict

    return _filter_from_dict(ds, case.filter_dict)


@needs_cuda
@pytest.mark.parametrize("kernel", ["torch", "triton_reduce"])
def test_multivector_ranking_is_stable_across_identical_cuda_runs(kernel, ds):
    """Ground truth is re-run. What must survive that?

    Not the score bits, as it turns out. On CUDA the default `torch`
    multivector kernel sums each query token's MaxSim with `index_add_`, an
    atomicAdd, so two runs over identical input disagree on roughly a quarter
    of the scores by ~1e-7 relative (measured on an A10G; see
    `compare.scores_are_reproducible` for the full table and the two ways to
    make it reproducible).

    The RANKING is what recall is computed from, and the ranking must be
    stable — so that is what this asserts, for both kernels. It is written to
    pass whether or not the scores happen to be reproducible, so it keeps
    holding if nova-bf later switches to a deterministic reduction; a test
    that failed on that improvement would be worse than no test.

    Skipping `triton_reduce` when Triton is absent is handled the same way as
    the fixture above: a missing optional dependency is not a parity defect.
    """
    from .cases import K
    from .runner import run, spec

    specs = [spec("mv_det", vector_type="multivector", metric="dot", k=K)]
    try:
        runs = [run(ds, specs, out_tag=f"det_{kernel}_{i}", device="cuda",
                    params={"multivector_kernel": kernel}) for i in range(2)]
    except RuntimeError as exc:
        if "Triton" in str(exc):
            pytest.skip(f"{kernel} unavailable: {exc}")
        raise

    for qi in range(len(ds.queries)):
        a, b = runs[0]["mv_det"][qi], runs[1]["mv_det"][qi]
        assert [int(i) for i, _ in a] == [int(i) for i, _ in b], (
            f"multivector_kernel={kernel} q{qi}: two identical CUDA runs "
            f"returned different rankings — ground truth would not be "
            f"reproducible at all")
        compare.assert_scores_agree(a, b, metric="dot",
                                    label=f"kernel={kernel} q{qi}: run-to-run")
