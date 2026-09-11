"""Two-pass dense scoring, held to the INDEPENDENT oracles.

`tests/test_twopass_pipeline.py` pins the two-pass against a one-pass run of
the same engine. That is the right test for the plumbing, and it cannot see a
bound that is wrong in a way both arms share — a pruning rule that drops the
same real candidate whichever path runs it agrees with itself perfectly. So
the two-pass is also held here, against the naive oracle and, where a server
is up, against live Qdrant.

Reaching the path at fixture scale takes three knobs, none of them a
correctness parameter:

  * `MIN_QUERY_ROWS` is the query height below which pass one cannot pay for
    itself (32768 in production). Nothing in this suite is that tall.
  * `PAD_QUANTUM` / `PAD_FLOOR` / `PAD_SMALL` are the exact-GEMM heights
    measured to keep cuBLAS on one kernel. At 26 query rows every production
    height is above `n_full`, so `pad_height` would return the full height and
    the slice would take the ordinary path.
  * `NOVA_BF_TWOPASS_ON_CPU` opts the CPU in. The CPU path is exact — pass one
    widens the half inputs and multiplies in float32 — it is simply not
    FASTER there, which is why production keeps it CUDA-only.

`_verify_shape` is deliberately left ON: `_scores` is not height-invariant in
general, so a run that skipped the check would be asserting an identity it had
not earned.

`ds_wide` (26 queries), not `ds` (8): with `dense_batch_size=8` it prunes ~26%
of query rows across ~26 two-pass slices. Every test here asserts that
pruning actually happened, because a two-pass that pruned nothing would agree
with every oracle vacuously — which is the failure mode this file exists to
rule out.

WHAT THIS FILE DOES NOT COVER
-----------------------------
The TIGHTNESS of the bound. Measured: replacing `bound()`'s result with ZERO —
removing every scrap of safety margin — leaves all of these tests passing, as
does scaling it to 1%. At this fixture's 16 dimensions and well-separated
random scores, the gap between a slice's best score and the row's threshold is
orders of magnitude larger than any fp16 error, so `eps` never decides
anything and a wrongly-tight bound prunes exactly the same rows.

That is not a hole to be plugged here by making the corpus adversarial — it is
what `tests/test_twopass_bound.py` is for, and that file IS sensitive: measured
on the current suite, `eps = 0` fails 45 of its 84 cases and `eps * 0.01` fails
43. (An earlier version of this sentence quoted "33 of 45", which was wrong in
both numbers — if you edit either file, re-measure rather than adjusting it.) What these tests pin is the part a
numerical test cannot reach — that the decision is wired into a real run
correctly: the narrowed and padded GEMM, the per-member spans, the scatter
back to full height, the eligibility refusals, and the agreement with an
INDEPENDENT implementation of the answer.

To be precise about WHY `eps = 0` survives, because the honest reason is
narrower than "this file cannot see pruning errors": these tests are sensitive
to a row that is wrongly pruned — inflating the threshold by 1.05x, or
dropping a single live row per slice, both fail this file loudly. What they
cannot see is a wrongly-tight BOUND, because `eps` sits two to three orders of
magnitude below the slice-best-to-threshold gap here, so shrinking it never
changes which rows get pruned. The mechanism is guarded; the margin is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova_bf import twopass

from . import cases as cases_mod
from . import compare
from .devices import parity_devices
from .runner import build_config, pinned_device, read_results, spec

try:
    from nova_bf.compute import run_compute
except ImportError:  # pragma: no cover
    from nova_bf import run_compute  # type: ignore

# Dense only — `_twopass_groups` takes `cosine` and `dot` and nothing else.
# One unfiltered case per metric, plus the filter shapes that change how the
# score matrix is masked: a static keyword filter, a static text filter, a
# PER-QUERY text filter (whose mask is `(n_queries, rows)`), and a compound.
PROBES = [
    cases_mod.CASES_BY_NAME[n] for n in (
        "dedot_nofilter", "decos_nofilter",
        "decos_match", "dedot_matchtext",
        "decos_pqtext", "dedot_compound",
    )
]
PROBE_IDS = [c.id for c in PROBES]

BATCH = 8   # small enough that a fresh slice often loses to the running top-K
# k=5, not the suite's 25: with only ~300 corpus rows and a filter that admits
# a fraction of them, a k=25 top-K never fills, every threshold stays at the
# -inf sentinel, and NOTHING is prunable — `dense-cosine-match` and
# `dense-dot-matchtext` both measured 0 two-pass slices at k=25. A smaller k
# fills, the thresholds rise, and the decision is actually exercised.
K = 5


def _at_k(case):
    """The case's own spec at this file's smaller `k` (see `K`)."""
    sp = dict(case.spec())
    sp["k"] = K
    return sp


def _filter_of(case, ds):
    from .test_parity_matrix import _filter_from_dict

    return _filter_from_dict(ds, case.filter_dict)


@pytest.fixture
def forced(monkeypatch):
    """Scale the two-pass to fixture size and turn it on for CPU."""
    monkeypatch.setattr(twopass, "PAD_QUANTUM", 2)
    monkeypatch.setattr(twopass, "PAD_FLOOR", 2)
    monkeypatch.setattr(twopass, "PAD_SMALL", ())
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 4)
    monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    # The guard the byte-identity claims rest on.
    monkeypatch.delenv("NOVA_BF_TWOPASS_NO_VERIFY", raising=False)
    twopass.reset()
    yield
    twopass.reset()


def _run(ds, specs, *, tag, device, params=None):
    _CURRENT_DEVICE["d"] = str(device or "")
    p = {"dense_batch_size": BATCH}
    p.update(params or {})
    cfg = build_config(ds, specs, out_tag=f"{tag}_{device or 'auto'}", params=p)
    with pinned_device(device):
        return read_results(run_compute(cfg))


def device_is_cuda():
    """The label carries the device, so read it from there rather than
    threading a parameter through every call site."""
    return _CURRENT_DEVICE.get("d", "").startswith("cuda")


_CURRENT_DEVICE: dict = {}


def test_the_fixture_is_wide_enough_for_the_bound_to_apply():
    """P6 (64 <= d <= 2^20) is a premise, and the fixture has to satisfy it.

    At `corpus.DIM = 16` the closed-form bound refuses every slice, the
    two-pass never engages, and every test in this file is comparing a
    one-pass run against the oracle — still passing, still green, and no
    longer testing the two-pass at all. `_assert_it_pruned` catches it, but
    only as a puzzling failure in fifteen unrelated cases. This says why in
    one line, and it fails FIRST because it names the actual cause.
    """
    from nova_bf import closed_form
    from . import corpus

    assert closed_form.dimension_ok(corpus.DIM), (
        f"the parity fixture's dense width is {corpus.DIM}, outside the "
        f"bound's admissible range [{closed_form.D_MIN}, {closed_form.D_MAX}] "
        f"(premise P6) — the two-pass will refuse every slice and this file "
        f"will test nothing about it")


def _assert_it_pruned(label):
    """`run_compute` resets the counters, so these describe the run just made.

    Without this a green test is indistinguishable from "the two-pass never
    fired", which is exactly the state the knobs above exist to escape.
    """
    st = twopass.stats()
    assert st["slices_twopass"] > 0, (
        f"{label}: the two-pass never ran ({st}) — this case proves nothing "
        f"about it. Check MIN_QUERY_ROWS / PAD_* against the fixture height.")
    pruned = st["rows_full"] - st["rows_live"]
    assert pruned > 0, (
        f"{label}: the two-pass ran on {st['slices_twopass']} slices but "
        f"pruned no rows ({st['rows_live']} live of {st['rows_full']}), so "
        f"every row still took the exact GEMM and the decision was never "
        f"exercised")
    # `slices_twopass` alone is not enough: a slice whose every row is pruned
    # builds a plan with `scores=None` and launches NO narrowed GEMM, so a run
    # made entirely of those would satisfy the two asserts above while leaving
    # the padded GEMM, the per-member spans and the scatter — everything this
    # file says it covers — completely unexercised.
    assert st["gemms"] > 0, (
        f"{label}: no narrowed GEMM ran ({st}); every two-pass slice pruned "
        f"all of its rows, so the exact-scoring path was never entered")
    # NOT `rows_padded < rows_full`: `_tp_note` is only reached when
    # `M < n_full` (both `M >= n_full` exits return before noting), so that
    # comparison cannot fail once `slices_twopass > 0` and it reads as a
    # guarantee it is not. What IS falsifiable is that padding only ever adds
    # rows to the live set — if it ever came back below `rows_live` the
    # padded GEMM would be scoring fewer rows than were found live.
    assert st["rows_padded"] >= st["rows_live"], (
        f"{label}: rows_padded={st['rows_padded']} is below "
        f"rows_live={st['rows_live']} — padding cannot shrink the live set")
    # A mid-run disable is NOT a failure here, and asserting otherwise made
    # this whole file go red on a real GPU. `_verify_shape` found no padded
    # height bit-identical to the full one on that device and fell back to the
    # one-pass path, which computes the same ground truth — the machinery
    # working, not failing.
    #
    # It is an artifact of the forcing, not of production: this fixture sets
    # `PAD_QUANTUM = PAD_FLOOR = 2` to reach the path at 26 query rows, and
    # measured on an A10G a 12/14/16/18-row GEMM is not bit-identical to the
    # 26-row one at N=8, K=16, while the production heights (multiples of 1024
    # at or above 7168) are. So the assertions that matter are the ones above:
    # the two-pass ran, launched real GEMMs, and pruned — all before any
    # fallback. Whether it then switched off says nothing about correctness,
    # and the oracle comparison in each test covers the answers either way.
    # On CUDA the FUSED kernel must be the one that ran. Without this, every
    # claim this file makes is about the cuBLAS fallback and its LOOSER bound
    # (the fused path reads the float32 accumulator, so it legitimately drops
    # the 2**-11 * T output term) — and on a CPU box `slices_fused` is 0 in
    # every run, so the production kernel was untested by this file entirely.
    if device_is_cuda():
        assert st["slices_fused"] > 0, (
            f"{label}: pass one never took the fused kernel on CUDA, so this "
            f"case covers only the cuBLAS fallback and the looser bound: {st}")

    if st["unavailable"] is not None:
        assert st["gemms"] > 0, (
            f"{label}: the two-pass disabled itself without ever running a "
            f"narrowed GEMM, so nothing was exercised: {st}")


@pytest.mark.parametrize("case", PROBES, ids=PROBE_IDS)
def test_the_two_pass_agrees_with_the_oracle(case, ds_wide, oracle_wide, device,
                                             forced):
    """The decision, against an independent implementation.

    A wrongly-pruned row loses a hit it should have had, so the oracle
    comparison is what actually catches a bound that is too tight.
    """
    got = _run(ds_wide, [_at_k(case)], tag=f"tp_{case.name}", device=device)[case.name]
    _assert_it_pruned(f"[{device}] {case.id}")
    want = oracle_wide.topk(vector_type=case.vector_type, metric=case.metric,
                            k=K, filt=_filter_of(case, ds_wide))
    for qi in range(len(ds_wide.queries)):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=case.metric,
            label=f"[{device}] two-pass {case.id} q{qi}")


@pytest.mark.parametrize("case", PROBES, ids=PROBE_IDS)
def test_the_two_pass_is_hit_identical_to_the_one_pass(case, ds_wide, device,
                                                       forced, monkeypatch):
    """Stronger than agreeing with the oracle: the SAME rows and the SAME
    score bits.

    Pass two runs `_scores` on a narrowed, padded query height and the
    one-pass path runs it at full height. `_verify_shape` proves those agree
    bit-for-bit for the heights used, so anything less than exact equality
    here is a real difference in what was computed, not a tolerance question.
    """
    two = _run(ds_wide, [_at_k(case)], tag=f"tp2_{case.name}", device=device)[case.name]
    _assert_it_pruned(f"[{device}] {case.id}")

    # Block eligibility on ANY device, rather than unsetting the CPU opt-in,
    # so the comparison arm is one-pass on a GPU box too.
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 1 << 30)
    one = _run(ds_wide, [_at_k(case)], tag=f"tp1_{case.name}", device=device)[case.name]
    assert twopass.stats()["slices_twopass"] == 0, "the control arm used the two-pass"

    for qi in range(len(ds_wide.queries)):
        assert two[qi] == one[qi], (
            f"[{device}] {case.id} q{qi}: the two-pass changed the result\n"
            f"  two-pass: {two[qi][:5]}\n  one-pass: {one[qi][:5]}")


def test_a_row_subsetted_search_keeps_its_own_live_rows(ds_wide, oracle_wide,
                                                        device, forced):
    """The `qsel`-slice branch, mixed with a `qsel is None` sibling.

    Both members read ONE score matrix, so the live mask is a union over them
    and each has to map the shared live index back into its own state height
    (`span` / `dst` in `_twopass_prepare`). A member that read its sibling's
    rows, or lost rows live only for the sibling, shows up here.

    The selector is a CONTIGUOUS block of `qid`, deliberately. `_twopass_groups`
    requires a member's rows to be a contiguous block of the query matrix, and
    a contiguous `isin` is normalised to a `slice`; the suite's `query_set`
    column alternates even/odd, which resolves to a gathered index tensor and
    is refused (see the test below).
    """
    n_q = len(ds_wide.queries)
    half = n_q // 2
    # BOTH halves, and the second one is the point. `dst[m] = idx[lo:hi] - base`
    # subtracts the member's `qsel.start`; a `front` block starts at 0, so
    # `base == 0` and dropping the subtraction changes nothing — measured:
    # removing `- base` leaves this file entirely green and is caught only by
    # `test_twopass_pipeline.py`, whose second search happens to land on
    # `slice(32, 64)`. `back` resolves to `slice(13, 26)`, which makes the
    # subtraction load-bearing here.
    front = [str(i) for i in range(half)]
    back = [str(i) for i in range(half, n_q)]
    specs = [
        spec("front", vector_type="dense", metric="cosine", k=K,
             rows={"column": "qid", "isin": front}),
        spec("back", vector_type="dense", metric="cosine", k=K,
             rows={"column": "qid", "isin": back}),
        spec("all", vector_type="dense", metric="cosine", k=K),
    ]
    got = _run(ds_wide, specs, tag="tp_subset", device=device)
    _assert_it_pruned(f"[{device}] row-subsetted")

    want = oracle_wide.topk(vector_type="dense", metric="cosine", k=K, filt=None)
    # Results are keyed by the ORIGINAL query index, so the key set is itself
    # the base assertion for the `back` member.
    for name, rows in (("front", range(half)), ("back", range(half, n_q))):
        assert set(got[name]) == set(rows), (
            f"[{device}] the {name} selector covered {sorted(got[name])}, "
            f"expected {sorted(rows)}")
        for qi in rows:
            compare.assert_scores_agree(
                got[name][qi], want[qi], metric="cosine",
                label=f"[{device}] two-pass subset {name} q{qi}")
    for qi in range(len(ds_wide.queries)):
        compare.assert_scores_agree(
            got["all"][qi], want[qi], metric="cosine",
            label=f"[{device}] two-pass subset all q{qi}")


def test_a_gathered_row_subset_is_refused_not_mishandled(ds_wide, oracle_wide,
                                                         device, forced):
    """A member whose query rows are NOT contiguous must take the ordinary path.

    `_twopass_groups` requires `qsel is None or isinstance(qsel, slice)`,
    because the live index is sorted and a member's live rows are read as a
    contiguous SPAN of it. An interleaved selector (`query_set` is even/odd
    here) resolves to a gathered index tensor, for which that span assumption
    is false — a group admitted anyway would hand each member the wrong rows.

    So this pins the refusal itself: the two-pass must not fire, and the
    answers must still be right. It is the eligibility guard's only test.
    """
    specs = [
        spec("even", vector_type="dense", metric="cosine", k=K,
             rows={"column": "query_set", "isin": ["even"]}),
        spec("odd", vector_type="dense", metric="cosine", k=K,
             rows={"column": "query_set", "isin": ["odd"]}),
    ]
    # CONTROL FIRST. `slices_twopass == 0` is also what a broken forcing knob
    # produces, so without a matching run that MUST engage, this test reports
    # a working refusal when nothing was refused. Measured: with
    # `_twopass_groups` stubbed to return {} — the total kill switch — this
    # test and the euclidean one both still passed.
    _run(ds_wide, [spec("ctl", vector_type="dense", metric="cosine", k=K)],
         tag="tp_gathered_ctl", device=device)
    _assert_it_pruned(f"[{device}] gathered-subset control")

    got = _run(ds_wide, specs, tag="tp_gathered", device=device)
    st = twopass.stats()
    assert st["slices_twopass"] == 0, (
        f"[{device}] a gathered (non-slice) row subset was scored through the "
        f"two-pass ({st}) — `span`/`dst` assume a contiguous slice")

    want = oracle_wide.topk(vector_type="dense", metric="cosine", k=K, filt=None)
    covered = set()
    for name in ("even", "odd"):
        expected = {qi for qi, q in enumerate(ds_wide.queries)
                    if q["payload"]["query_set"] == name}
        assert set(got[name]) == expected, (
            f"[{device}] the {name!r} selector covered {sorted(got[name])}, "
            f"expected {sorted(expected)}")
        covered |= expected
        for qi in expected:
            compare.assert_scores_agree(
                got[name][qi], want[qi], metric="cosine",
                label=f"[{device}] gathered {name} q{qi}")
    assert covered == set(range(len(ds_wide.queries)))


def test_euclidean_never_takes_the_two_pass(ds_wide, oracle_wide, device, forced):
    """`_twopass_groups` admits `cosine` and `dot` only.

    Euclidean's score is not a scaled dot product, so the bound does not
    describe it and pruning on it would be wrong. The eligibility check is one
    `continue`; this pins that it is still there, and that the results are the
    ordinary ones.
    """
    case = cases_mod.CASES_BY_NAME["deeuc_nofilter"]
    # Control: the same shape on a metric the two-pass DOES take, so a
    # `slices_twopass == 0` below means "refused", not "never engaged". See
    # the note in the gathered-subset test.
    _run(ds_wide, [spec("ctl", vector_type="dense", metric="cosine", k=K)],
         tag="tp_euc_ctl", device=device)
    _assert_it_pruned(f"[{device}] euclidean control")

    # Assert the REFUSAL ITSELF, not just that nothing ran. Measured: with
    # `metric not in ("cosine", "dot")` deleted, this test still passed —
    # euclidean's live fraction stays at 1.0 on this fixture, so every slice
    # is discarded for want of a smaller padded height and `slices_twopass`
    # is 0 either way. The control arm above does not save it, because the
    # control is a COSINE run. A test that reports a working guard while the
    # guard is gone is worse than no test.
    import torch as _torch

    from nova_bf import compute as _compute

    _q = _torch.randn(64, 8, dtype=_torch.float32)
    _batch = _compute.DenseCorpusBatch(
        np.random.default_rng(0).standard_normal((32, 8), dtype=np.float32))
    _groups = _compute._twopass_groups(
        batch=_batch, score_groups={("euclidean", False): [0]},
        spec_Q=[_q], spec_q_norms=[_q.norm(dim=1)], spec_qsel=[None],
        device="cpu", prune=True)
    assert _groups == {}, (
        f"`_twopass_groups` admitted a euclidean score group {list(_groups)}; "
        f"the bound describes a scaled DOT product and does not cover a "
        f"distance metric")

    got = _run(ds_wide, [_at_k(case)], tag="tp_euc", device=device)[case.name]
    st = twopass.stats()
    assert st["slices_twopass"] == 0, (
        f"[{device}] euclidean was scored through the two-pass ({st}) — the "
        f"bound does not cover it")
    want = oracle_wide.topk(vector_type="dense", metric="euclidean", k=K,
                            filt=None)
    for qi in range(len(ds_wide.queries)):
        compare.assert_scores_agree(
            got[qi], want[qi], metric="euclidean",
            label=f"[{device}] euclidean q{qi}")


# ---------------------------------------------------------------------------
# GPU-only: the claims that cannot be checked without real cuBLAS.
# ---------------------------------------------------------------------------

_CUDA_ONLY = pytest.mark.skipif(
    "cuda" not in parity_devices(),
    reason="these pin real cuBLAS kernel-selection behaviour, not logic",
)


@_CUDA_ONLY
@pytest.mark.parametrize("n_full,n_live", [(8192, 7000), (12288, 9000)])
def test_a_production_shaped_padded_height_is_bit_identical_on_this_gpu(
    n_full, n_live
):
    """The claim the whole feature rests on, measured on THIS device.

    `_verify_shape` proves each height before trusting it, so a machine where
    the claim fails degrades safely rather than corrupting anything — but the
    suite has never demonstrated that a production-shaped height actually
    PASSES on real hardware, which is the difference between "the feature is
    safe" and "the feature is safe and also runs".

    That matters more after what the fixture-scale runs measured: at 26 query
    rows with `PAD_QUANTUM` forced to 2, an A10G found NO height bit-identical
    (12, 14, 16 and 18 rows all differed from 26 at N=8, K=16) and the
    two-pass disabled itself. Small shapes genuinely do move cuBLAS's kernel
    choice. These are the real heights: multiples of 1024 at or above
    `PAD_FLOOR`, which `docs/brute-force/gpu-profile/2026-09-06-fuse` measured
    bit-identical up to 40960.
    """
    import torch

    from nova_bf import twopass as tp

    assert tp.pad_candidates(n_live, n_full)[0] % tp.PAD_QUANTUM == 0
    M = tp.pad_candidates(n_live, n_full)[0]
    assert M >= tp.PAD_FLOOR and M < n_full, f"{M} is not a real padded height"

    g = torch.Generator(device="cpu").manual_seed(4)
    Q = torch.randn(n_full, 128, generator=g).cuda()
    C = torch.randn(2048, 128, generator=g).cuda()
    tp.reset()
    ok = tp._verify_shape(Q, C, M)
    del Q, C
    torch.cuda.empty_cache()
    assert ok, (
        f"a {M}-row GEMM is NOT bit-identical to the full {n_full}-row one on "
        f"this GPU. The two-pass will disable itself here and fall back to the "
        f"one-pass path — ground truth is unaffected, but the padding ladder's "
        f"measured heights do not hold on this device and the perf claim does "
        f"not either.")


@_CUDA_ONLY
def test_tf32_is_refused_end_to_end_on_cuda(ds_wide, oracle_wide, forced):
    """The TF32 refusal on the path that can actually reach it.

    `params.allow_tf32` only takes effect on CUDA, so the CPU test for this
    guard has to call `_twopass_groups` directly. Here the whole run is
    configured with TF32 on: the two-pass must decline, and the answers must
    still be right — the bound is derived for a float32 exact path and does
    not cover TF32 (measured violation: upper 7.3284766404e-04 against a TF32
    score of 7.3313713074e-04).
    """
    case = cases_mod.CASES_BY_NAME["decos_nofilter"]
    got = _run(ds_wide, [_at_k(case)], tag="tp_tf32", device="cuda",
               params={"allow_tf32": True})[case.name]
    st = twopass.stats()
    assert st["slices_twopass"] == 0 and st["gemms"] == 0, (
        f"the two-pass ran with TF32 enabled: {st}")

    want = oracle_wide.topk(vector_type="dense", metric="cosine", k=K, filt=None)
    for qi in range(len(ds_wide.queries)):
        compare.assert_scores_agree(
            got[qi], want[qi], metric="cosine",
            label=f"[cuda] tf32-refused q{qi}")
