"""The two-pass bound must never call a row dead that has a real candidate.

    max_c score[q, c]  <=  twopass.upper_bounds(...)[q]

for the float32 score the one-pass path computes. Everything downstream rests
on that one inequality: a row the bound calls dead never has its exact scores
computed, so a false "dead" silently loses ground truth and nothing later can
notice.

WHAT THIS FILE IS FOR, NOW THAT THE BOUND IS CLOSED-FORM
--------------------------------------------------------
`closed_form.py` owns the arithmetic and `tests/test_closed_form.py` owns its
verification — differentially against the document's own script, against the
document's published table, structurally, and by running both passes for real.
None of that is repeated here.

What `twopass` owns, and what this file tests, is the PLUMBING around it: which
storage format each axis is actually in, which scale factors pass one actually
applied, which guards refuse and what happens to the slice when one does, and
that every refusal fails towards LIVE. The bound can be perfect and the run
still lose ground truth if the wrong `Format` reaches it or a guard is spelled
so that NaN passes.

So these tests attack it from both sides:

* against a **float64** reference, so the check does not inherit the float32
  path's own rounding, AND against the actual float32 scores, which is the
  comparison the run really makes;
* with thresholds placed deliberately AT the boundary — a hair under, on, and
  a hair over the true row max — so a bound that is too tight by any margin
  fails rather than passing because the test data was easy;
* with the degenerate values the bound has to survive: NaN, inf, zero rows,
  wildly mismatched norms, and an under-filled top-K's -inf sentinel. Under the
  closed form most of those are answered by a GUARD REFUSING rather than by a
  large `eps`, so each of those tests asserts both the safety property (nothing
  dead) and the documented reason for the refusal — a test that only checks
  "not dead" would pass just as happily against a guard that stopped firing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nova_bf import closed_form as cf
from nova_bf import twopass
from nova_bf.compute import _scores
from nova_bf.tiebreak import TIE_WORST, pack, unpack_score

# The dimension every `upper_bounds` fixture in this file uses. P6 admits only
# `64 <= d <= 2**20` — below 64 Lemma 2' does not apply and pass one is refused
# outright, all rows live — so a fixture at d = 32 no longer tests the bound at
# all, it tests the dimension guard. Fixtures that used to run at 32 were moved
# up rather than left to pass for the wrong reason;
# `test_a_dimension_below_the_proven_range_is_refused_not_approximated` is where
# the small-`d` behaviour is pinned deliberately.
DIM = 768


def _scales(C, Q, metric):
    """The per-column and per-query factors of the final score."""
    if metric == "cosine":
        return (C.norm(dim=1).clamp_min(1e-12).reciprocal(),
                Q.norm(dim=1).clamp_min(1e-12).reciprocal())
    return None, None


def _exact64(Q, C, metric):
    """The reference: the same score in float64, from the float32 values."""
    Qd, Cd = Q.double(), C.double()
    if metric == "dot":
        return Qd @ Cd.T
    Cn = Cd / Cd.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (Qd @ Cn.T) / Qd.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _fp32_scores(Q, C, metric):
    """What the one-pass path actually produces, in final units."""
    if metric == "dot":
        return _scores(Q, C, "dot")
    qn = Q.norm(dim=1).clamp_min(1e-12)
    return _scores(Q, C, "cosine", qn, scale_in_packer=False)


@pytest.fixture(autouse=True)
def _fresh():
    twopass.reset()
    yield
    twopass.reset()


def _make(n_q, w, dim, *, fp16_corpus, q_scale, c_scale, seed):
    g = torch.Generator().manual_seed(seed)
    Q = (torch.randn(n_q, dim, generator=g) * q_scale).float()
    C = (torch.randn(w, dim, generator=g) * c_scale).float()
    if fp16_corpus:
        # Exactly what `io.dense_to_2d` hands back for a float16 parquet
        # column, which is the case the corpus term is supposed to vanish for.
        C = C.half().float()
    return Q, C


CASES = [
    # (fp16_corpus, q_scale, c_scale, metric, seed)
    (True, 1.0, 1.0, "cosine", 0),
    (False, 1.0, 1.0, "cosine", 1),
    (True, 1.0, 1.0, "dot", 2),
    (False, 1.0, 1.0, "dot", 3),
    (False, 300.0, 0.003, "cosine", 4),      # wildly mismatched norms
    (False, 0.004, 250.0, "cosine", 5),
    (True, 60.0, 60.0, "dot", 6),            # large dot magnitudes
    (False, 1e-3, 1e-3, "dot", 7),           # tiny dot magnitudes
]


# `None` is what the run actually asks for (`_tp_out_dtype`), and it is the
# harder case: a float16 Gram is rounded before `col_scale` is applied, so the
# bound carries an output term the float32 Gram does not need.
@pytest.mark.parametrize("out_dtype", [None, torch.float32],
                         ids=["gram_f16", "gram_f32"])
@pytest.mark.parametrize("fp16_corpus,q_scale,c_scale,metric,seed", CASES)
def test_no_row_with_a_real_candidate_is_ever_called_dead(
    fp16_corpus, q_scale, c_scale, metric, seed, out_dtype
):
    n_q, w, dim = 512, 256, DIM
    Q, C = _make(n_q, w, dim, fp16_corpus=fp16_corpus, q_scale=q_scale,
                 c_scale=c_scale, seed=seed)
    cs, rs = _scales(C, Q, metric)
    upper = twopass.upper_bounds(Q, C, cs, rs, out_dtype, metric)

    ref64 = _exact64(Q, C, metric).amax(dim=1)
    ref32 = _fp32_scores(Q, C, metric).amax(dim=1)

    # 1. The bound is a bound, cell by cell, on both references.
    assert torch.all(upper.double() >= ref64 - 1e-12), (
        f"upper bound below the float64 row max by up to "
        f"{float((ref64 - upper.double()).max()):.3e}"
    )
    assert torch.all(upper >= ref32)

    # 2. The decision itself, with thresholds walked across the boundary. Any
    #    row the bound calls dead must genuinely have nothing at or above thr.
    eps = (upper - ref32).clamp_min(1e-9)
    for mult in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 8.0):
        thr_score = ref32 + mult * eps
        dead = (upper + 0.0) < thr_score
        if not bool(dead.any()):
            continue
        assert torch.all(ref64[dead] < thr_score[dead].double()), (
            f"mult={mult}: a row called dead has a float64 candidate at or "
            f"above its threshold"
        )
        assert torch.all(ref32[dead] < thr_score[dead])

    # 3. Usefulness: a threshold comfortably above the row max must kill it.
    #    Not when the float16 Gram OVERFLOWS, though — large `dot` magnitudes
    #    put `q_h·c_h` past float16's 65504 and the row max comes out `inf`,
    #    which makes every row live. That is the conservative direction and
    #    the bound is still a bound; there is just nothing to prune.
    if bool(torch.isfinite(upper).all()):
        far = ref32 + 100.0 * eps + 1e-3
        assert bool(((upper) < far).all()), "the bound is too loose to prune at all"


@pytest.mark.parametrize("metric", ["cosine", "dot"])
@pytest.mark.parametrize("seed", range(6))
def test_a_subnormal_fp16_gram_is_still_bounded(metric, seed):
    """The regime the purely-relative output term got wrong.

    With small vectors the raw dot `q_h·c_h` falls below float16's smallest
    normal (`2**-14`), where the rounding error is the absolute subnormal
    quantum rather than `2**-11 * |x|` — and under cosine `col_scale = 1/‖c‖`
    is correspondingly large, so it amplifies that absolute error into the
    scored units. A bound with only the relative term is BELOW the true row
    max here, which silently kills live rows. `out_dtype=None` is the
    production setting, so this is the path the run takes.

    Under the closed form the absolute terms live in `Psi` (Lemma 1's `eta`
    for the conversions, Lemma 2's anchor for the accumulation) plus the
    `ETA16 * rho * sigma_max` output term that `half_out` adds — so the same
    regime still has to be covered, just by different arithmetic.

    Run at `DIM` rather than the original 32: P6 refuses anything below 64, so
    at 32 this would exercise the dimension guard and prove nothing about the
    subnormal regime. The magnitude is what makes the fixture, not the width —
    measured, 97% of the raw `q_h·c_h` entries here are below `2**-14`.
    """
    dim = DIM
    Q, C = _make(64, 48, dim, fp16_corpus=False, q_scale=1e-3, c_scale=1e-3,
                 seed=200 + seed)
    raw = (Q.half() @ C.half().T).float()
    assert float((raw.abs() < 2.0 ** -14).float().mean()) > 0.5, (
        "fixture is not in the subnormal regime, so it does not exercise the "
        "absolute terms it exists for")
    cs, rs = _scales(C, Q, metric)
    upper = twopass.upper_bounds(Q, C, cs, rs, None, metric)
    ref64 = _exact64(Q, C, metric).amax(dim=1)
    ref32 = _fp32_scores(Q, C, metric).amax(dim=1)
    assert torch.all(upper.double() >= ref64), (
        "upper bound below the float64 row max by up to "
        f"{float((ref64 - upper.double()).max()):.3e}"
    )
    assert torch.all(upper >= ref32)
    # And it is still useful: a threshold well above the row max prunes.
    assert bool((upper < ref32 + 1.0).all())


@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_bound_survives_thresholds_within_eps_of_a_real_candidate(metric):
    """Construct the hard case directly: a candidate exactly at, and a hair
    under, each row's threshold."""
    n_q, w, dim = 256, 128, DIM
    Q, C = _make(n_q, w, dim, fp16_corpus=False, q_scale=1.0, c_scale=1.0,
                 seed=11)
    cs, rs = _scales(C, Q, metric)
    upper = twopass.upper_bounds(Q, C, cs, rs, torch.float32, metric)
    exact = _fp32_scores(Q, C, metric)
    ref64 = _exact64(Q, C, metric)

    for delta in (0.0, -1e-7, -1e-6, 1e-7):
        # thr sits on the row's best candidate: the row is genuinely live at
        # delta <= 0, so it must not be pruned.
        thr_score = exact.amax(dim=1) + delta
        dead = upper < thr_score
        assert torch.all(ref64[dead].amax(dim=1) < thr_score[dead].double())
        if delta <= 0:
            assert not bool(dead.any()), (
                f"delta={delta}: pruned a row whose own best candidate reaches "
                f"its threshold"
            )


def test_half_output_bound_is_looser_but_still_valid():
    """With no float32 GEMM output the bound carries an output-rounding term.
    Both variants must bound; the half one must not be tighter."""
    Q, C = _make(256, 128, DIM, fp16_corpus=True, q_scale=1.0, c_scale=1.0,
                 seed=21)
    cs, rs = _scales(C, Q, "cosine")
    ref = _fp32_scores(Q, C, "cosine").amax(dim=1)
    u32 = twopass.upper_bounds(Q, C, cs, rs, torch.float32, "cosine")
    twopass.reset()
    u16 = twopass.upper_bounds(Q, C, cs, rs, None, "cosine")
    assert torch.all(u32 >= ref)
    assert torch.all(u16 >= ref)
    assert torch.all(u16 >= u32 - 1e-6)


def test_an_fp16_stored_corpus_is_charged_no_conversion_error_at_all():
    _, C = _make(4, 512, DIM, fp16_corpus=True, q_scale=1.0, c_scale=1.0,
                 seed=31)
    _, C32 = _make(4, 512, DIM, fp16_corpus=False, q_scale=1.0, c_scale=1.0,
                   seed=31)
    assert twopass.float32_is_exactly_fp16(C)
    assert not twopass.float32_is_exactly_fp16(C32)

    # `Q` is float32 in both cases, so only the corpus term moves.
    Q = torch.randn(8, DIM)
    # An explicit UNIT row scale, not `None`. `bound()` refuses cosine with no
    # row scale, because `eps_cos` is sized for a scaled score of magnitude at
    # most `Lambda(d)` and an unscaled row max has magnitude ~||q||. Here the
    # scores are never computed — only two `eps` values are compared — so
    # `rho = 1` is what is wanted, and saying so explicitly is the point.
    ones = torch.ones(int(Q.shape[0]))
    qs = twopass.query_side(Q, ones)
    exact_c = twopass.bound(qs, DIM, 1.0, False, "cosine", fmt_c=cf.EXACT)
    conv_c = twopass.bound(qs, DIM, 1.0, False, "cosine", fmt_c=cf.FP16)
    assert float(np.max(exact_c)) < float(np.min(conv_c)), (
        "an fp16-STORED corpus is not being charged less than a converted "
        "one, so the whole reason pass one runs on the raw corpus is gone")
    # The document's headline pair, to four figures. `test_closed_form` checks
    # the same table against `cf.eps_cos`; what is added by repeating it HERE
    # is the hardware envelope, which `bound()` supplies rather than takes —
    # it passes `C_HW_ENVELOPE` in, and swapping it for the fitted Ampere
    # coefficient would be invisible to every test that calls `closed_form`
    # directly while shrinking `eps` on real runs.
    assert abs(float(np.max(exact_c)) - 7.187e-4) / 7.187e-4 < 6e-4
    assert abs(float(np.max(conv_c)) - 1.208e-3) / 1.208e-3 < 6e-4

    # END TO END, through `upper_bounds`, because that is where the format is
    # actually CHOSEN. Everything above tests `bound()` with the format handed
    # to it; only this catches `upper_bounds` reading the wrong one, which is
    # the failure that costs 41% of `eps` in the unsafe direction with no
    # symptom. `with_parts` is what makes `eps` visible without re-deriving it.
    Qb = torch.randn(16, DIM)
    for corpus, want_fmt in ((C, cf.EXACT), (C32, cf.FP16)):
        twopass.reset()
        cs = corpus.norm(dim=1).clamp_min(1e-12).reciprocal()
        rs = Qb.norm(dim=1).clamp_min(1e-12).reciprocal()
        _, _, eps = twopass.upper_bounds(Qb, corpus, cs, rs, torch.float32,
                                         "cosine", with_parts=True)
        qs = twopass.query_side(Qb, rs)
        _, sigma_max, _, cn_max = twopass.corpus_side(corpus, cs)
        # `upper_bounds` evaluates at a BUCKETED `sigma_max` — rounded up to
        # the next power of two so `eps` can be reused across the slices of a
        # file — so it is not bit-equal to the exact-sigma bound. Two
        # assertions instead of one, and together they are stronger:
        #
        #   1. it must EQUAL the bucketed bound, which pins the format (the
        #      thing this test is about: reading the wrong one costs 41% of
        #      `eps` in the unsafe direction, silently);
        #   2. it must DOMINATE the exact-sigma bound, which is what makes the
        #      bucketing admissible at all — `Psi` is increasing in
        #      `sigma_max`, so rounding up is safe and rounding down is not.
        bucket = cf.ceil_pow2(sigma_max)
        assert bucket >= sigma_max
        want = twopass.bound(qs, DIM, bucket, False, "cosine",
                             fmt_c=want_fmt, cn_max=cn_max)
        assert np.array_equal(eps.numpy(), np.asarray(want)), (
            f"`upper_bounds` charged the wrong corpus format: expected "
            f"{want_fmt!r}")
        exact_sigma = twopass.bound(qs, DIM, sigma_max, False, "cosine",
                                    fmt_c=want_fmt, cn_max=cn_max)
        assert np.all(np.asarray(eps.numpy()) >= np.asarray(exact_sigma)), (
            "the bucketed `sigma_max` produced a SMALLER eps than the exact "
            "one, so rounding up is not being applied and `Psi` no longer "
            "bounds the absolute terms")

    # And the per-FILE verdict overrides the per-slice check, because that is
    # how `compute` calls it — a file whose first slice happens to be
    # fp16-exact must not license the rest of the file.
    twopass.reset()
    cs = C.norm(dim=1).clamp_min(1e-12).reciprocal()
    rs = Qb.norm(dim=1).clamp_min(1e-12).reciprocal()
    _, _, forced = twopass.upper_bounds(Qb, C, cs, rs, torch.float32, "cosine",
                                        corpus_exact_fp16=False,
                                        with_parts=True)
    assert float(forced.max()) > float(np.max(exact_c)), (
        "`corpus_exact_fp16=False` was ignored; the per-file verdict is what "
        "`compute` passes down and it must be able to be conservative")


def test_the_stored_format_is_read_from_the_data_not_from_configuration():
    """One file arriving in a different width must not understate `eps`.

    `stored_format` and `float32_is_exactly_fp16` are the two places the run
    decides which `Format` each axis gets, and getting it wrong is worth 41% of
    the bound in the UNSAFE direction with no symptom anywhere — which is why
    they read the tensor rather than a config field, per file.
    """
    assert twopass.stored_format(torch.zeros(2, 4, dtype=torch.float16)) is cf.EXACT
    # NOT `cf.BF16`. `query_side` runs `Q.half()` unconditionally, so a
    # bfloat16 axis is converted bf16 -> FP16 and fp16's constants are the
    # ones Lemma 1 needs. Charging bf16's `eta_t = 2**-134` and
    # `lambda_t = 2**-126` to an fp16 conversion understates `Psi`'s anchor
    # term by 2**112, and Lemma 1 is violated outright: a bf16 1.0e-8 flushes
    # to zero under `.half()`, an error of 1e-8 against an allowance of 4e-11.
    assert twopass.stored_format(torch.zeros(2, 4, dtype=torch.bfloat16)) is cf.FP16
    assert twopass.stored_format(torch.zeros(2, 4)) is cf.FP16

    # An empty slice has nothing to convert, so nothing to charge.
    assert twopass.float32_is_exactly_fp16(torch.zeros(0, 8))
    # A single value off the fp16 grid is enough to lose the exemption. 1 + 2**-12
    # is representable in float32 and NOT in float16, and it is exactly the kind
    # of one-element difference a per-file check has to see.
    dirty = torch.zeros(4, 8)
    dirty[2, 5] = 1.0 + 2.0 ** -12
    assert not twopass.float32_is_exactly_fp16(dirty)
    # A dtype the bound has no constants for is refused rather than assumed.
    assert not twopass.float32_is_exactly_fp16(torch.zeros(2, 4, dtype=torch.float64))


def test_the_query_side_reports_the_applied_scale_not_a_recomputed_one():
    """`rho_a` is P1's premise, made into data.

    """
    Q, _ = _make(64, 8, DIM, fp16_corpus=False, q_scale=1.0, c_scale=1.0,
                 seed=41)
    row_scale = Q.norm(dim=1).clamp_min(1e-12).reciprocal()
    qs = twopass.query_side(Q, row_scale)
    assert "dq" not in qs and "qn_ub" not in qs, (
        "the measured residual is back; `bound()` is data-free by "
        "construction and nothing may reintroduce a per-vector measurement")
    assert np.array_equal(qs["rho_a"],
                          row_scale.double().numpy()), (
        "`rho_a` is not the scale pass one applies")
    assert qs["rho_a"].dtype == np.float64

    # A row whose stored reciprocal exceeds the binary64 `1/N` — the case P1 is
    # written for. `rho_a` must carry the LARGER, applied value.
    bigger = row_scale.clone()
    bigger[3] = float(np.nextafter(np.float32(bigger[3]), np.float32(np.inf)))
    twopass.release()
    qs2 = twopass.query_side(Q, bigger)
    assert qs2["rho_a"][3] > 1.0 / float(Q.norm(dim=1)[3].double()), (
        "a stored reciprocal above 1/N was replaced by the recomputed value, "
        "which `Psi` is not evaluated at")

    # `dot` applies no query scale at all, so rho is exactly 1 — not `1/N`.
    twopass.release()
    assert np.array_equal(twopass.query_side(Q)["rho_a"], np.ones(64))


@pytest.mark.parametrize("bad", ["nan_query", "inf_query", "nan_corpus", "zero_corpus"])
def test_degenerate_values_come_out_live_not_dead(bad):
    """Live is always safe; dead is not. Anything the bound cannot reason
    about must fail towards live.

    All four are now answered by a GUARD rather than by a large `eps`, and each
    one goes through a different guard, so both halves are asserted: nothing
    dead, AND the refusal counter moved. "Nothing dead" on its own would pass
    against a guard that had stopped firing and left the rows merely lucky.

    The two AXES fail differently, and the difference is deliberate:

      * `nan_corpus` / `zero_corpus` take the WHOLE slice. `norm_guard(cn)` is
        applied slice-wide because `sigma_max` and the norm extremes are
        slice-wide reductions and there is no per-column residual left to
        absorb one bad column. A zero-norm row is no longer exempt either: its
        `col_scale` is `1/clamp_min(0, 1e-12) = 1e12`, which feeds `sigma_max`
        and hence `Psi`, so it is refused rather than assumed harmless;
      * `nan_query` / `inf_query` take ONLY ROW 3. Every query-side guard is
        per row — `norm_guard`, `overflow_mask`, `product_guard` and the
        prunability cut alike. An earlier version reduced the query axis with
        `norms.max()` for the overflow test, which made one NaN row refuse all
        64; `overflow_mask` is the unreduced form and exists for exactly that
        reason.

    So for the query cases this asserts the STRONGER property: the bad row is
    live and the healthy rows are still free to prune.
    """
    Q, C = _make(64, 32, DIM, fp16_corpus=False, q_scale=1.0, c_scale=1.0,
                 seed=51)
    if bad == "nan_query":
        Q[3] = float("nan")
    elif bad == "inf_query":
        Q[3, 17] = float("inf")
    elif bad == "nan_corpus":
        C[5] = float("nan")
    else:
        C[5] = 0.0
    cs, rs = _scales(C, Q, "cosine")
    upper = twopass.upper_bounds(Q, C, cs, rs, torch.float32, "cosine")
    thr_score = torch.full((Q.shape[0],), 0.5)
    dead = upper < thr_score
    # The offending row is never dead, on any of the four.
    assert not bool(dead[3]), f"{bad}: the degenerate row itself was pruned"
    if bad in ("nan_query", "inf_query"):
        # Per-row: the bad row is live, and it did NOT cost the other 63 rows
        # their pruning. `upper[3]` is +inf; the rest are finite.
        assert not torch.isfinite(upper[3])
        others = torch.arange(upper.numel()) != 3
        assert bool(torch.isfinite(upper[others]).all()), (
            f"{bad}: one degenerate query row refused rows other than itself")
    else:
        # Slice-wide: nothing may be pruned at all.
        assert not bool(dead.any()), (
            f"{bad}: a row was pruned against a slice the bound cannot reason "
            f"about")
        assert twopass.stats()["slices_guard_refused"] == 1, (
            f"{bad}: no guard refused, so the rows above are live by luck "
            f"rather than by the mechanism this test is about")
        assert bool(torch.isinf(upper).all()) and bool((upper > 0).all()), (
            f"{bad}: a slice-wide refusal must be `+inf` on every row — the "
            f"only value the caller's `~(upper < thr)` reads as "
            f"unconditionally live")
    # On every path, a REFUSED row is `+inf` and never `-inf`: `-inf` is below
    # every finite threshold and would be read as dead.
    assert bool(upper[3] > 0)


def test_sentinel_threshold_keeps_every_row_live():
    """An under-filled top-K still holds -inf sentinels; nothing may be pruned
    against one, matching `tiebreak.live_rows`."""
    Q, C = _make(32, 16, DIM, fp16_corpus=True, q_scale=1.0, c_scale=1.0,
                 seed=61)
    cs, rs = _scales(C, Q, "cosine")
    upper = twopass.upper_bounds(Q, C, cs, rs, torch.float32, "cosine")
    sentinel = pack(
        torch.full((Q.shape[0],), float("-inf")),
        torch.tensor(TIE_WORST, dtype=torch.int64),
    )
    thr_score = unpack_score(sentinel)
    assert torch.all(torch.isneginf(thr_score))
    assert not bool((upper < thr_score).any())


def test_negative_zero_threshold_matches_the_packer_convention():
    """`score_order_key` folds -0.0 onto +0.0, so a threshold of -0.0 must
    decide exactly as +0.0 does."""
    neg = pack(torch.tensor([-0.0]), torch.tensor(TIE_WORST, dtype=torch.int64))
    pos = pack(torch.tensor([0.0]), torch.tensor(TIE_WORST, dtype=torch.int64))
    assert torch.equal(neg, pos)
    assert float(unpack_score(neg)[0]) == 0.0
    assert not np.signbit(float(unpack_score(neg)[0]))

_DOC_PAD_QUANTUM = 1024
_DOC_PAD_FLOOR = 7168
_DOC_PAD_SMALL = (2048, 4096, 5120)


def test_the_padding_constants_are_the_documented_ones():
    assert twopass.PAD_QUANTUM == _DOC_PAD_QUANTUM
    assert twopass.PAD_FLOOR == _DOC_PAD_FLOOR
    assert tuple(twopass.PAD_SMALL) == _DOC_PAD_SMALL


def _height_is_measured_identical(M: int) -> bool:
    return (M in _DOC_PAD_SMALL
            or (M >= _DOC_PAD_FLOOR and M % _DOC_PAD_QUANTUM == 0))


def test_pad_height_only_produces_verified_heights():
    for n_live in (1, 10, 999, 2047, 4096, 5121, 7167, 8191, 8192, 8193,
                   20000, 109_999):
        M = twopass.pad_height(n_live, 110_000)
        assert n_live <= M <= 110_000
        assert M == 110_000 or _height_is_measured_identical(M), M
    assert twopass.pad_height(0, 110_000) == 0


def test_pad_candidates_are_an_ascending_ladder_ending_at_full_height():
    """Every rung must be usable on its own: at least `n_live`, at most
    `n_full`, measured identical, and strictly increasing — and the last rung
    is always `n_full`, the height that is trivially identical to itself, so
    the caller can never run out of options."""
    for n_full in (110_000, 40_000):
        for n_live in (1, 3000, 5119, 7000, 15_001, n_full - 1, n_full):
            if n_live > n_full:
                continue
            got = twopass.pad_candidates(n_live, n_full)
            assert got, (n_live, n_full)
            assert got == sorted(set(got)), got
            assert got[-1] == n_full
            assert all(n_live <= M <= n_full for M in got), got
            assert all(M == n_full or _height_is_measured_identical(M)
                       for M in got), got
            assert len(got) <= twopass.MAX_PAD_TRIES + 1


def test_pad_candidates_never_offers_a_height_measured_to_disagree():
    """1024, 3072 and 6144 were MEASURED to disagree with the full-height
    product on the reference machine. A ladder that offered one of them would
    be handing `_verify_shape` a known failure and paying for the escalation."""
    bad = (1024, 3072, 6144)
    for n_live in range(1, 8200, 37):
        assert not (set(twopass.pad_candidates(n_live, 110_000)) & set(bad))


def test_the_quantum_is_what_the_padding_wastes():
    """The whole point of the 1024 quantum: no slice ever pads by more than
    one quantum once it is above the floor."""
    for n_live in range(twopass.PAD_FLOOR, 40_000, 331):
        M = twopass.pad_height(n_live, 110_000)
        assert M - n_live < twopass.PAD_QUANTUM


def test_the_gemm_output_is_half_by_default():
    """Pass one's `G` is written once and read once, and it is 1.8 GB in
    float32 at the production shape. The float16 output halves both the store
    and the row max's load; the cost is the `2**-11 * T` term the bound has
    always carried, which the float64 reference tests above already cover."""
    import os

    from nova_bf import compute as C

    os.environ.pop("NOVA_BF_TWOPASS_FP32_OUT", None)
    assert C._tp_out_dtype() is None


def test_fp32_output_can_be_asked_for(monkeypatch):
    """Someone who wants the tighter bound can have it — subject to the torch
    build actually being able to produce a float32 result from a half GEMM,
    which is what the probe decides."""
    import torch

    from nova_bf import compute as C

    monkeypatch.setenv("NOVA_BF_TWOPASS_FP32_OUT", "1")
    monkeypatch.setattr(C, "_TP_OUT_DTYPE", C._TP_UNSET, raising=False)
    got = C._tp_out_dtype()
    assert got in (None, torch.float32)
    if torch.cuda.is_available():
        assert got is torch.float32, "a CUDA build should manage a float32 out"




def _cf_eps_for(Q, C, cs, rs, metric, half_out, corpus_is_fp16):
    """`eps` derived straight from the tensors, not from anything `twopass` built.

    This is the wiring oracle: it reads the formats and the scales off the data
    the way the document says to, and calls `closed_form` directly. If
    `bound()` picks up the wrong `Format`, forgets `row_scale`, applies the
    cosine theorem to `dot` scores, or drops `half_out`, the two disagree.
    """
    d = int(Q.shape[1])
    u_c, eta_c = (0.0, 0.0) if corpus_is_fp16 else (cf.U16, cf.ETA16)
    if metric == "dot":
        return cf.eps_dot(d, cf.U16, u_c, Q.norm(dim=1).double().numpy(),
                          float(C.norm(dim=1).max()), cf.C_HW_ENVELOPE,
                          half_out, cf.ETA16, eta_c, cf.LAMBDA16)
    return cf.eps_cos(d, cf.U16, u_c, cf.C_HW_ENVELOPE, half_out,
                      cf.ETA16, eta_c,
                      rho_q=rs.double().numpy(),
                      sigma_max=float(cs.max()), lam_t=cf.LAMBDA16)


@pytest.mark.parametrize("fp16_corpus,q_scale,c_scale,metric,seed", CASES)
def test_the_bound_is_the_closed_form_at_the_formats_and_scales_that_ran(
    fp16_corpus, q_scale, c_scale, metric, seed
):
    """`bound()` must be `closed_form` evaluated at what pass one really did.

    Not a restatement of `bound()`'s body — the arithmetic is not repeated
    here at all. What is repeated is the DERIVATION OF ITS ARGUMENTS from the
    raw tensors: the per-axis storage format, the applied row and column
    scales, the metric's theorem, and whether a float16 Gram was written. Every
    one of those is a decision `twopass` makes and `closed_form` cannot check,
    and every one of them fails silently:

      * the wrong `fmt_c` understates `eps` by 41% at d = 768;
      * a `sigma_max` or `rho` below what the kernel applied breaks the
        monotonicity `Psi`'s admissibility rests on;
      * the cosine theorem on `dot` scores is four orders of magnitude small
        on production norms, which the `metric` parameter's docstring records
        as a real bug in an earlier draft;
      * a dropped `half_out` silently forgives the fp16 output rounding.
    """
    twopass.reset()
    Q, C = _make(64, 48, DIM, fp16_corpus=fp16_corpus, q_scale=q_scale,
                 c_scale=c_scale, seed=seed)
    cs, rs = _scales(C, Q, metric)
    qs = twopass.query_side(Q, rs)
    _, sigma_max, _, cn_max = twopass.corpus_side(C, cs)
    fmt_c = cf.EXACT if fp16_corpus else cf.FP16

    for half_out in (False, True):
        got = twopass.bound(qs, DIM, sigma_max, half_out, metric,
                            fmt_c=fmt_c, cn_max=cn_max)
        want = _cf_eps_for(Q, C, cs, rs, metric, half_out, fp16_corpus)
        assert np.array_equal(np.asarray(got), np.asarray(want)), (
            f"{metric} seed={seed} half_out={half_out}: `bound()` is not the "
            f"closed form evaluated at the formats and scales that ran — "
            f"worst relative gap "
            f"{float(np.max(np.abs(np.asarray(got, dtype=np.float64) - want) / want)):.3e}")


def test_the_bound_is_evaluated_in_binary64_and_handed_back_as_binary32():
    """P8's evaluation rule, at the boundary `twopass` is responsible for.

    `thr` is stored in binary32, so `eps` has to be a binary32 value — and it
    has to be the binary32 value ABOVE the real one, since whatever consumes it
    would otherwise round it down and the row would be pruned on an `eps` too
    small. `closed_form.eps_final` does the rounding; what is checked here is
    that `bound()` returns that, on the host, rather than building a torch
    graph whose arithmetic nobody certified.

    The shape matters too: `Psi` is per QUERY, so a cosine `eps` is an array of
    `n_q`, not a scalar. A scalar would silently broadcast and charge every row
    the same absolute term.
    """
    twopass.reset()
    Q, C = _make(32, 16, DIM, fp16_corpus=False, q_scale=1.0, c_scale=1.0,
                 seed=77)
    cs, rs = _scales(C, Q, "cosine")
    qs = twopass.query_side(Q, rs)
    _, sigma_max, _, cn_max = twopass.corpus_side(C, cs)

    eps = twopass.bound(qs, DIM, sigma_max, False, "cosine", cn_max=cn_max)
    assert isinstance(eps, np.ndarray), (
        "`eps` came back as a torch tensor; the certified evaluation is "
        "binary64 on the host and `numpy` is what carries it")
    assert eps.dtype == np.float32
    assert eps.shape == (32,), (
        "a scalar `eps` would broadcast and charge every query row the same "
        "absolute term, which `Psi` is not")
    assert np.all(eps > 0.0) and np.all(np.isfinite(eps))

    # Rows with different applied scales must get different absolute terms —
    # otherwise `rho_a` is not reaching `Psi` at all.
    assert float(eps.max()) > float(eps.min())


def test_the_metric_has_no_default_and_dot_is_not_charged_cosines_constant():
    """A caller that forgets to say which theorem applies must not be guessed for.

    Theorem 1 bounds a quantity of magnitude at most `Lambda(d) ~ 1`; Theorem
    1' bounds one that scales with `N_q * max_S N_c`. On the fixture below that
    is nearly three orders of magnitude, and on production norms four. Applying
    the cosine constant to `dot` scores gives an `eps` that small with no
    symptom anywhere — rows whose real score clears the threshold get an upper
    bound below it and are called dead. The `metric` parameter's own docstring
    records this as a real bug in an earlier draft of the function, fixed by
    making the caller state it.
    """
    twopass.reset()
    Q, C = _make(32, 24, DIM, fp16_corpus=False, q_scale=1.0, c_scale=1.0,
                 seed=88)
    with pytest.raises(TypeError):
        twopass.upper_bounds(Q, C, None, None, None)      # no metric

    # Two DIFFERENT query sides: dot applies no scaling at all (Theorem 1' is
    # stated at rho = sigma = 1 exactly, and `bound` now refuses a dot call
    # carrying a scale), while cosine requires one.
    qs_dot = twopass.query_side(Q, None)
    qs_cos = twopass.query_side(Q, torch.ones(int(Q.shape[0])))
    _, sigma_max, _, cn_max = twopass.corpus_side(C, None)
    e_dot = twopass.bound(qs_dot, DIM, sigma_max, False, "dot", cn_max=cn_max)
    e_cos = twopass.bound(qs_cos, DIM, 1.0, False, "cosine")
    assert float(np.min(e_dot)) > 100.0 * float(np.max(e_cos)), (
        f"the dot bound ({float(np.min(e_dot)):.3e}) is not scaling with the "
        f"norms the way Theorem 1' says; it is within a factor of 100 of the "
        f"cosine constant ({float(np.max(e_cos)):.3e})")

    # euclidean is scored by the one-pass path and has no theorem here, so it
    # is refused rather than given the nearest available constant.
    with pytest.raises(ValueError, match="no bound is derived for metric"):
        twopass.bound(qs_cos, DIM, 1.0, False, "euclidean")


def test_the_measured_residual_machinery_is_really_gone():
    """The ledger of Sec.13, as an assertion.

    Each of these was a constant or a measurement whose sufficiency had to be
    certified at run time, and each was removed because the closed form derives
    what it stood in for. They are named here so that a later edit cannot
    quietly reintroduce one as a fudge multiplier on a bound that is supposed
    to be data-free — which is exactly the shape of change that looks harmless
    in review and puts an unproven constant back in the certified path.
    """
    for gone in ("SLACK", "SLACK_EVAL_RESERVE", "NORM_INFLATE",
                 "ROUNDING_BUDGET", "U_ACC", "gamma_acc",
                 "provable_norm_inflation", "certify_norm_inflation",
                 "measure_norm_inflation", "certify_norms_on", "certify_slack",
                 "_norm_unusable"):
        assert not hasattr(twopass, gone), (
            f"`twopass.{gone}` is back. The closed form derives what it stood "
            f"in for; a hand-chosen margin beside it is an unproven premise in "
            f"the certified path")


def test_the_bound_constants_are_the_documented_ones():
    """The magnitudes `eps` is built from, pinned to literals.

    `twopass` re-exports these from `closed_form` because the rest of the
    module reads them constantly, and the re-export is the thing that can rot:
    it is one line, it looks like a convenience, and a wrong one moves every
    guard comparison in the file. Pinned to literals rather than read back from
    `closed_form`, which would make the assertion circular.
    """
    assert twopass.U == 2.0 ** -24, "fp32 unit roundoff"
    assert twopass.U16 == 2.0 ** -11, "fp16 relative rounding"
    for d in (1, 768, 4096):
        want = (d * 2.0 ** -24) / (1.0 - d * 2.0 ** -24)
        assert twopass.gamma(d) == want, f"gamma({d}) is not d*u/(1-d*u)"
    assert twopass.gamma is cf.gamma, (
        "`gamma` is re-exported so that pass one's accumulation term and the "
        "exact path's are the same function; a local copy is how the two drift")


def test_pass_one_is_charged_a_truncating_accumulator_not_a_rounding_one():
    """The classical constant assumes round-to-nearest. Tensor cores truncate.

    `gamma_d` is Higham's bound for a d-term sum in which every addition rounds
    to NEAREST, `|delta| <= u`. Pass one runs on tensor cores — both `tl.dot`
    and the cuBLAS float16 GEMM — and those do not: products are formed
    exactly, but the addends are aligned to the largest exponent and the
    discarded bits are TRUNCATED (Fasi, Higham, Mikaitis & Pranesh, "Numerical
    behaviour of NVIDIA tensor cores", PeerJ CS 2021).

    This was `U_ACC = 2**-22`, a coefficient CHOSEN as 2x the published
    truncation model. It is now `closed_form.kappa_acc` — the same leading
    `4du`, DERIVED as an envelope over every published generation (Lemma 2')
    rather than picked. The constant went; the property it existed for did not,
    so it is asserted here on the function that replaced it.

    No test on this box can observe the hardware behaviour — CPU matmul rounds
    to nearest — so this pins the RELATIONSHIP between the two constants, which
    is what a fix that simply raised `U` everywhere would break.
    """
    d = DIM
    assert cf.kappa_acc(d) > 2.0 * twopass.gamma(d), (
        "the accumulation envelope must exceed the published truncation model "
        "(2x gamma_d), or the bound is charging tensor cores for arithmetic "
        "they do not perform")
    assert cf.kappa_acc(d) / twopass.gamma(d) < 5.0, (
        "and it must not be so wide that pruning suffers")
    # The exact path keeps `u`: it is a float32 GEMM on FFMA units with TF32
    # refused, and those DO round to nearest. The two must differ.
    assert twopass.U == 2.0 ** -24
    assert cf.C_HW_ENVELOPE == 4.0 and cf.C_HW_AMPERE == 1.375
    assert cf.kappa_acc(d, cf.C_HW_ENVELOPE) > cf.kappa_acc(d, cf.C_HW_AMPERE), (
        "the envelope no longer dominates the fitted Ampere/Ada model, so the "
        "device this run is tuned for is outside the constant it is charged")


def test_sigma_max_is_the_applied_column_scale_over_every_column():
    """What replaced `P` and `cs_max`, and the two ways it is unsafe.

    `Psi` is increasing in `sigma_max`, so an UPPER bound on the factor pass
    one applies is admissible and a LOWER one is not. Two mistakes follow, and
    the module's own docstring names both because both shipped at some point:

      * recomputing `1/min_c N_c` in binary64. A stored binary32 `fl(1/N_c)`
        can exceed `1/N_c` by up to `u`, so the recomputed value can be SMALLER
        than the scale the GEMM used;
      * dropping columns. The shipped `cs_max` excluded columns whose half copy
        was exactly zero — right for the half-OUTPUT term it fed, since those
        have an exactly zero Gram entry, and wrong here, because `Psi`'s anchor
        term is charged against every column pass one scales, zero Gram or not.

    A fixture where one column is far and away the largest scale is what makes
    a dropped column visible: with a flat spread, `max` over a subset equals
    `max` over the whole and either version passes.
    """
    twopass.reset()
    C = torch.nn.functional.normalize(torch.randn(48, DIM), dim=1)
    C[11] = 0.0                       # an empty document: col_scale = 1/1e-12
    C[12] *= 2.0 ** -8                # and one merely small, so the spread is real
    cs = C.norm(dim=1).clamp_min(1e-12).reciprocal()
    assert float(cs.max()) > 10.0 * float(cs.median()), (
        "fixture is not discriminating: with a flat spread, a `sigma_max` that "
        "silently drops columns still equals the true maximum")

    _, sigma_max, cn_min, cn_max = twopass.corpus_side(C, cs)
    assert sigma_max == float(cs.max()), (
        f"sigma_max={sigma_max!r} is not the maximum of the scales pass one "
        f"applies ({float(cs.max())!r}) — a column was dropped or the value "
        f"was recomputed rather than read")
    # And the two really are distinguishable on this fixture: a binary64
    # `1/min N_c` differs from the stored binary32 reciprocal the GEMM applies,
    # in whichever direction this particular value happens to round. Without
    # this the assertion above would hold for a recomputed `sigma_max` too, and
    # would not be testing anything.
    recomputed = 1.0 / float(C.norm(dim=1).clamp_min(1e-12).min())
    assert sigma_max != recomputed, (
        f"the applied scale and a recomputed one are the same number on this "
        f"fixture ({sigma_max!r}), so nothing above distinguishes them")
    assert cn_min == float(C.norm(dim=1).min())
    assert cn_max == float(C.norm(dim=1).max())

    # `dot` applies no column scale at all, so sigma is exactly 1 — not
    # `1/min N_c`, which would charge `Psi` for a scaling that never happened.
    _, sigma_dot, _, _ = twopass.corpus_side(C, None)
    assert sigma_dot == 1.0

    # An empty slice has no columns to reduce over, and used to raise.
    Ch, sigma_e, min_e, max_e = twopass.corpus_side(torch.zeros(0, DIM), None)
    assert Ch.shape == (0, DIM) and sigma_e == 0.0
    assert min_e != min_e and max_e != max_e, "an empty slice has no extremes"


def test_dot_bound_rejects_corpus_norms_for_a_different_slice():
    """Dot has no col_scale/norm pairing check to catch this structurally.

    `cn_max` enters the dot error model and product guard.  A shorter fp32
    vector can therefore look superficially valid while describing different
    corpus rows, so reject it before pass one can use it.
    """
    twopass.reset()
    Q, C = _make(8, 16, DIM, fp16_corpus=True, q_scale=1.0, c_scale=1.0,
                 seed=81)
    wrong_slice_norms = C.norm(dim=1)[:-1]

    with pytest.raises(ValueError, match="corpus norms must have shape"):
        twopass.upper_bounds(
            Q, C, None, None, None, "dot", cn=wrong_slice_norms,
        )
    twopass.reset()


def test_the_norm_guard_refuses_nan_rather_than_accepting_it():
    """The TORCH spelling of the guard, which is a different thing to get wrong.

    `closed_form.norm_range_ok` is the scalar version and has its own test.
    This is the tensor one, and the failure mode is the same and just as easy:
    `~(lo <= N <= hi)` refuses NaN (live, safe) while `N < lo | N > hi` accepts
    it (prunable, wrong), and the second reads like a harmless simplification.

    A non-finite component makes its row's norm non-finite, so this is also
    what establishes P5 on both axes — the fused kernel's `tl.max` lowers to a
    NaN-IGNORING PTX `max.f32` and would otherwise silently drop one.
    """
    norms = torch.tensor([1.0, float("nan"), float("inf"), -float("inf"), 0.0,
                          float(cf.NORM_MIN), float(cf.NORM_MAX),
                          float(np.nextafter(np.float32(cf.NORM_MIN),
                                             np.float32(0.0)))])
    bad = twopass.norm_guard(norms)
    assert bad.tolist() == [False, True, True, True, True, False, False, True], (
        f"the two-sided norm guard is not refusing what it must: {bad.tolist()}")
    # The thresholds are the document's, not a value that drifted with an edit.
    assert cf.NORM_MIN == 2.0 ** -40 and cf.NORM_MAX == 2.0 ** 126
    # And an empty axis is not an error — a zero-column slice is a real slice.
    assert twopass.norm_guard(torch.zeros(0)).shape == (0,)


# ---------------------------------------------------------------------------
# Guards that shipped WITHOUT tests. Each of these was added in response to a
# reproduced defect and each survived a mutation revert.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["query", "corpus"])
def test_a_norm_that_underflowed_forces_every_row_live(side):
    """A float32 norm of exactly 0 for a NONZERO vector is not a norm.

    Summing 768 squares of ~1e-23 flushes to zero in float32 while the
    products of a dot product do not, so every term of the bound collapses
    and `eps` becomes exactly 0 against a nonzero error. Reproduced: 22 of 64
    rows wrongly dead. Being live is the only safe answer.

    Under the closed form the answer is the FLOOR of the two-sided norm guard,
    which is what the measured residual's `dq = inf` used to do implicitly. The
    floor also rose, from 2**-63 to 2**-40, and that is not cosmetic: it is
    what makes Lemma 2's absolute term smaller than the 2**-44 reserve `C(d)`
    carries, what makes (R4)'s fold tight enough for `kappa`'s
    `(1 + d 2**-70)` factor, and what keeps `1/N` normal so the scalings cannot
    themselves overflow. So the refusal is asserted as well as the safety.
    """
    twopass.reset()
    tiny = 2e-23
    if side == "query":
        Q = torch.full((8, DIM), tiny)
        C = torch.randn(16, DIM)
        assert bool((Q.norm(dim=1) == 0).all()), "fixture must underflow"
        assert bool((Q != 0).all()), "fixture must be nonzero"
        assert bool(twopass.norm_guard(Q.norm(dim=1)).all()), (
            "an underflowed query norm must be refused by the norm guard")
    else:
        Q = torch.randn(8, DIM)
        C = torch.full((16, DIM), tiny)
        assert bool((C.norm(dim=1) == 0).all()), "fixture must underflow"
        assert bool(twopass.norm_guard(C.norm(dim=1)).all())

    for metric in ("dot", "cosine"):
        twopass.reset()
        cs, rs = _scales(C, Q, metric)
        upper = twopass.upper_bounds(Q, C, cs, rs, None, metric)
        ref = _exact64(Q, C, metric).amax(dim=1)
        assert bool((upper.double() >= ref).all()), (
            f"{side}/{metric}: the bound is below the true best score "
            f"(upper={float(upper[0]):.3e} vs {float(ref[0]):.3e})")
        # Live, and live for the documented reason: every row is `+inf` and a
        # guard recorded the refusal. A finite bound here would mean the guard
        # stopped firing and the rows survived on the size of `eps` instead.
        assert bool(torch.isinf(upper).all()) and bool((upper > 0).all())
        assert twopass.stats()["slices_guard_refused"] == 1, (
            f"{side}/{metric}: no guard refused an unusable norm")


def test_an_unsupported_output_dtype_is_refused():
    """The output-rounding term is written in float16's constants
    (`2**-11`, and `2**-25` for its subnormal quantum). `bfloat16` rounds at
    `2**-9`, so charging it these numbers UNDERSTATES the error."""
    twopass.reset()
    Q, C = _make(8, 16, DIM, fp16_corpus=False, q_scale=1.0, c_scale=1.0, seed=3)
    cs, rs = _scales(C, Q, "cosine")
    with pytest.raises(ValueError, match="out_dtype"):
        twopass.upper_bounds(Q, C, cs, rs, torch.bfloat16, "cosine")
    # the two the run actually asks for stay fine
    for od in (None, torch.float32):
        assert twopass.upper_bounds(Q, C, cs, rs, od, "cosine").shape == (8,)


def test_an_explicit_float16_output_is_charged_the_output_term():
    """`half_out` is spelled "not float32" rather than "is None" so an
    explicit float16 output is charged the output-rounding term instead of
    silently dropping it.

    Under the closed form that term is `U16 * (1 + theta) * (1 + inp)` inside
    `bracket` plus `ETA16 * rho * sigma_max` inside `Psi` — the relative and
    absolute halves of rounding the Gram to float16 before `sigma` is applied.
    Both are strictly positive for any non-empty slice, so equality between the
    two bounds is itself the bug.
    """
    twopass.reset()
    Q, C = _make(32, 24, DIM, fp16_corpus=False, q_scale=1.0, c_scale=1.0, seed=4)
    cs, rs = _scales(C, Q, "cosine")
    u16 = twopass.upper_bounds(Q, C, cs, rs, torch.float16, "cosine")
    twopass.reset()
    u32 = twopass.upper_bounds(Q, C, cs, rs, torch.float32, "cosine")
    # STRICTLY looser. `>=` is not enough: if `half_out` were spelled
    # `out_dtype is None` again, an explicit float16 would be charged NO
    # output term, `u16` would equal `u32` exactly, and a `>=` assertion
    # would still pass.
    assert bool((u16 > u32).all()), (
        "a float16 Gram must be charged the output-rounding term, so its "
        f"bound must be strictly looser than a float32 Gram's; "
        f"{int((u16 <= u32).sum())} rows were not")
    ref = _exact64(Q, C, "cosine").amax(dim=1)
    assert bool((u16.double() >= ref).all())


def test_skipping_verification_is_recorded_in_the_stats(monkeypatch):
    """`NOVA_BF_TWOPASS_NO_VERIFY` removes the proof that a padded GEMM height
    is bit-identical to the full one — the proof the whole feature rests on.
    It used to leave no trace but `verifications: 0`, which is also what a run
    that never engaged the two-pass reports."""
    twopass.reset()
    assert twopass.stats()["verify_skipped"] is False
    monkeypatch.setenv("NOVA_BF_TWOPASS_NO_VERIFY", "1")
    Q = torch.randn(256, 32)
    C = torch.randn(64, 32)
    assert twopass._verify_shape(Q, C, 64) is True
    st = twopass.stats()
    assert st["verify_skipped"] is True, "the skip must be reported"
    assert st["verifications"] == 0, "and no check was actually run"
    # cleared with the counters it is reported beside
    twopass.reset_stats()
    assert twopass.stats()["verify_skipped"] is False


def test_an_overflowing_fp16_gram_never_reports_minus_inf():
    """A written fp16 Gram can OVERFLOW, and overflow is not a bounded
    rounding. `-inf` is below every finite threshold, so a row whose whole
    slice overflowed negative was called dead — reproduced at ordinary cosine
    scores of -1.0, because it is the RAW dots that leave fp16's range once
    the norms pass ~256."""
    twopass.reset()
    d = DIM
    base = torch.zeros(d)
    base[0] = 1.0
    Q = base.repeat(8, 1) * 600.0 + torch.randn(8, d) * 0.01
    C = -base.repeat(16, 1) * 600.0 + torch.randn(16, d) * 0.01
    # Neither AXIS overflows on conversion — the norms are ~600, far below the
    # ~65502 the overflow guard admits — so the guard does not fire and the
    # defence under test is the one inside `upper_bounds`, not the guard.
    assert twopass.overflow_guard(d, C.norm(dim=1), cf.FP16)
    assert twopass.overflow_guard(d, Q.norm(dim=1), cf.FP16)
    for metric in ("dot", "cosine"):
        twopass.reset()
        cs, rs = _scales(C, Q, metric)
        raw16 = (Q.half() @ C.half().T).float()
        assert bool(torch.isneginf(raw16).all()), "fixture must overflow negative"
        upper = twopass.upper_bounds(Q, C, cs, rs, None, metric)
        assert not bool(torch.isneginf(upper).any()), (
            f"{metric}: upper came back -inf, which marks every row dead")
        ref = _exact64(Q, C, metric).amax(dim=1)
        assert bool((upper.double() >= ref).all())


def test_an_empty_slice_is_all_dead_not_an_exception():
    """A zero-column slice has no candidate, so `-inf` (dead) is the right
    answer. `corpus_side`'s reductions have no identity for an empty tensor
    and used to raise."""
    twopass.reset()
    Q = torch.randn(4, DIM)
    C = torch.zeros(0, DIM)
    for cs in (None, C.norm(dim=1).clamp_min(1e-12).reciprocal()):
        upper = twopass.upper_bounds(Q, C, cs, None, None, "dot")
        assert upper.shape == (4,)
        assert bool(torch.isneginf(upper).all())


def test_one_zero_norm_row_costs_the_slice_its_pruning_and_says_so():
    """An empty document gives `‖c‖ = 0`, so `col_scale = 1/clamp(0) = 1e12`.
    """
    twopass.reset()
    Q = torch.nn.functional.normalize(torch.randn(64, DIM), dim=1)
    C = torch.nn.functional.normalize(torch.randn(48, DIM), dim=1)
    C = C.half().float()
    cs, rs = _scales(C, Q, "cosine")
    clean = int((twopass.upper_bounds(Q, C, cs, rs, None, "cosine") < 0.9).sum())
    assert clean > 0, "fixture must be prunable before the zero row is added"
    assert twopass.stats()["slices_guard_refused"] == 0, (
        "the clean slice was refused, so the comparison below is between two "
        "refusals and proves nothing")

    twopass.reset()
    C2 = C.clone()
    C2[7] = 0.0
    cs2, rs2 = _scales(C2, Q, "cosine")
    upper = twopass.upper_bounds(Q, C2, cs2, rs2, None, "cosine")
    assert bool(twopass.norm_guard(C2.norm(dim=1))[7]), (
        "an exactly-zero corpus row is exempt from the norm guard again; its "
        "`col_scale` of 1e12 then feeds `sigma_max` and `Psi` unnoticed")
    assert bool(torch.isinf(upper).all()) and bool((upper > 0).all()), (
        "the slice must go LIVE rather than prune on a 1e12 scale")
    assert twopass.stats()["slices_guard_refused"] == 1, (
        "the refusal was not counted, so a run losing its pruning to empty "
        "documents looks identical to one whose live fraction never fell")
    ref = _fp32_scores(Q, C2, "cosine").amax(dim=1)
    assert bool((upper.double() >= ref.double()).all())


class _HeightSensitive:
    """A corpus stand-in whose matmul depends on the LEFT operand's height.

    `_verify_shape` is the proof the padded exact GEMM is safe, and the only
    tests of it monkeypatch it to `False` — so they pin what happens when the
    answer is no, and nothing pins that the answer IS no when a height really
    disagrees. Reverting it to `return True` survived every test.

    One float32 ulp of disagreement is the smallest thing it must catch,
    because `torch.equal` is the whole point of the check.
    """

    def __init__(self, C, full_height):
        self._C = C
        self._full = full_height

    def __getattr__(self, name):
        return getattr(self._C, name)

    @property
    def T(self):
        return self

    def __rmatmul__(self, other):
        got = other @ self._C.T
        if other.shape[0] != self._full:
            return torch.nextafter(got, torch.full_like(got, float("inf")))
        return got


def test_verify_shape_says_no_when_a_height_really_disagrees():
    twopass.reset()
    Q = torch.randn(256, 32)
    C = torch.randn(64, 32)
    assert twopass._verify_shape(Q, _HeightSensitive(C, 256), 64) is False, (
        "a height whose product differs by one ulp must be REFUSED")
    twopass.reset()
    assert twopass._verify_shape(Q, C, 64) is True, (
        "and an honest height must be accepted, or the ladder never runs")


def test_the_shape_cache_is_keyed_not_global():
    """One failing height must not condemn the others — `pad_candidates` is a
    ladder precisely so a bad rung is stepped over."""
    twopass.reset()
    Q = torch.randn(256, 32)
    C = torch.randn(64, 32)
    assert twopass._verify_shape(Q, C, 64) is True
    poisoned = dict(twopass._SHAPE_OK)
    assert len(poisoned) == 1
    twopass._SHAPE_OK[next(iter(poisoned))] = False
    assert twopass._verify_shape(Q, C, 128) is True, (
        "a different height must be judged on its own")


@pytest.mark.parametrize("raw", ["nan", "inf", "-0.1", "1.5", "abc", "-1"])
def test_a_nonsense_threshold_falls_back_to_the_default(monkeypatch, raw):
    """A live FRACTION, so only [0, 1] means anything. `nan` compares false
    against everything and would run pass one on every slice however live it
    was; a negative value disabled the two-pass with no message."""
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", raw)
    monkeypatch.setattr(twopass, "_THRESHOLD_WARNED", False)
    assert twopass.threshold() == twopass.DEFAULT_THRESHOLD


@pytest.mark.parametrize("raw,want", [("0.0", 0.0), ("0.5", 0.5), ("1.0", 1.0),
                                      ("1", 1.0)])
def test_a_valid_threshold_is_honoured(monkeypatch, raw, want):
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", raw)
    assert twopass.threshold() == want


@pytest.mark.parametrize("side", ["query", "corpus"])
def test_a_partially_underflowed_norm_forces_the_row_live(side):
    """A float32 norm can be badly wrong while staying comfortably NONZERO.

    `kappa(d)`'s relative-error argument needs the sum of squares to stay in
    the normal range. Build a row whose element squares straddle the subnormal
    quantum — most round to zero, the rest lose most of their bits — and the
    float32 norm comes back ~4.6x BELOW the true one. The earlier guard tested
    `norms == 0`, which never fires here, and `upper_bounds` came back 4.57x
    too small: a live row called dead, on the production `half_out=False` path.

    The closed form answers this with the RAISED floor rather than with a
    special case. At 1.6e-22 the norm is nine orders below `NORM_MIN = 2**-40`,
    so the two-sided guard refuses it — and the same floor is what keeps
    `1/N` normal, which is the reason it was raised. Both are asserted: a test
    that only checked "not dead" would pass against a guard that had stopped
    firing.
    """
    import math

    twopass.reset()
    d = DIM
    S = 2.0 ** -149                            # smallest fp32 subnormal
    lo = math.sqrt(0.499 * S)                  # square rounds to zero
    hi = math.sqrt(1.499 * S)                  # square loses ~half its value
    row = torch.full((d,), lo, dtype=torch.float32)
    row[: d // 40] = hi
    assert float(row.norm()) > 0.0, "fixture must NOT flush to zero"
    ratio = float(row.double().norm() / row.norm())
    assert ratio > 2.0, (
        f"fixture is not adversarial: the float32 norm is only {ratio:.3f}x "
        f"below the true one, so it would not have fooled the old guard")
    assert float(row.norm()) < cf.NORM_MIN, (
        "the fixture must sit below the two-sided guard's floor, or it is not "
        "testing the mechanism that replaced the zero check")

    # Q aligned with C, so the true dot is the maximal one.
    if side == "corpus":
        Q, C = torch.full((4, d), 1.0), row.repeat(8, 1)
    else:
        Q, C = row.repeat(4, 1), torch.full((8, d), 1.0)
    upper = twopass.upper_bounds(Q, C, None, None, torch.float32, "dot")
    true = (Q.double() @ C.double().T).amax(dim=1)
    dead = upper.double() < true
    assert not bool(dead.any()), (
        f"{side}: {int(dead.sum())} rows marked DEAD against a real "
        f"candidate (upper={float(upper[0]):.4e}, true={float(true[0]):.4e})")
    assert not bool(torch.isfinite(upper).any()), (
        f"{side}: an unusable norm must produce a non-finite bound, got "
        f"{upper[:4].tolist()}")
    assert twopass.stats()["slices_guard_refused"] == 1, (
        f"{side}: the rows are live but no guard refused them, so they are "
        f"live on the size of `eps` rather than on the floor this tests")


def test_one_degenerate_query_row_does_not_cost_the_others_their_pruning():
    """The query-side guards are per ROW, so one unusable row costs only itself.

    Blanketing the whole vector took a clean 64-row slice from 64/64 rows
    prunable to 0/64 — and because `query_side` is cached per query matrix, for
    the rest of the run. `norm_guard`, `product_guard` and `prunability_cut`
    all return per-row masks for exactly this reason.

    The corpus side cannot be local in the same way: `sigma_max` and the norm
    extremes are slice-wide reductions with no per-column residual left to
    absorb one bad column, so one untrustworthy column does keep every query
    live. That is correct, not a gap — see
    `test_one_zero_norm_row_costs_the_slice_its_pruning_and_says_so`.

    A row whose norm is NaN or infinite is the exception, and it is not this
    test: `overflow_guard` reduces the query axis with `norms.max()`, which is
    then non-finite and refuses the whole axis. Hence a tiny-but-finite norm
    here, which is the case the per-row masks actually handle.
    """
    twopass.reset()
    d = DIM
    Q = torch.nn.functional.normalize(torch.randn(64, d), dim=1)
    C = torch.nn.functional.normalize(torch.randn(48, d), dim=1).half().float()
    cs, rs = _scales(C, Q, "cosine")
    clean = int(torch.isfinite(
        twopass.upper_bounds(Q, C, cs, rs, None, "cosine")).sum())
    assert clean == 64, "fixture must start with every bound finite"

    twopass.reset()
    Q2 = Q.clone()
    Q2[17] = 2e-23                              # one unusable row
    assert torch.isfinite(Q2.norm(dim=1)).all(), (
        "the row must stay FINITE, or the axis-wide overflow guard takes the "
        "slice and the per-row masks are never reached")
    cs2, rs2 = _scales(C, Q2, "cosine")
    upper = twopass.upper_bounds(Q2, C, cs2, rs2, None, "cosine")
    assert bool(twopass.norm_guard(Q2.norm(dim=1))[17])
    assert not torch.isfinite(upper[17]), "the degenerate row must be live"
    finite = int(torch.isfinite(upper).sum())
    assert finite == 63, (
        f"one bad query row left only {finite}/64 bounds finite; the "
        f"neutralisation is not per row")


def test_an_exactly_zero_corpus_row_is_no_longer_exempt_from_the_guard():
    """The exemption that had to go, and why keeping it would be unsafe.

    `_norm_unusable` used to let an exactly-zero row through on the argument
    that its half copy is exactly zero and its Gram entry is exactly zero, so
    `eps = 0` is correct for it. That argument was about the RESIDUAL, and the
    residual is gone. What the closed form charges instead is `Psi`, whose
    anchor term is proportional to `sigma_max` — and `sigma_max` for such a row
    is `1/clamp_min(0, 1e-12) = 1e12`, because the clamp sits ABOVE the
    `2**-40 ~ 9.09e-13` floor. So the row passes the floor, `Psi` comes out
    around 1e4, and the bound is valid and useless.

    Refusing it is the answer, and `prunability_cut` is the second half of the
    same answer: a slice whose smallest corpus norm is below `1e-3` cannot pay
    for pass one at all, so pass one is not run.
    """
    twopass.reset()
    d = DIM
    C = torch.nn.functional.normalize(torch.randn(16, d), dim=1).half().float()
    C[3] = 0.0
    cn = C.norm(dim=1)
    assert bool(twopass.norm_guard(cn)[3]), (
        "an exactly-zero row is exempt again; `clamp_min(1e-12)` then hands "
        "`Psi` a scale of 1e12 that every safety check accepts")
    assert float(cn.min()) < cf.PRUNE_MIN_CNORM
    qn = torch.ones(8)
    assert bool(twopass.prunability_cut(qn, float(cn.min())).all()), (
        "a slice this degenerate cannot prune, so pass one must not be run "
        "for it at all")

    Q = torch.nn.functional.normalize(torch.randn(8, d), dim=1)
    cs, rs = _scales(C, Q, "cosine")
    upper = twopass.upper_bounds(Q, C, cs, rs, None, "cosine")
    assert bool(torch.isinf(upper).all()) and bool((upper > 0).all())
    assert twopass.stats()["slices_guard_refused"] == 1


def test_a_zero_width_matrix_does_not_crash_the_guards():
    """A zero-row axis has no norms to reduce over, and the reductions used to
    raise. Both guards have to answer it without an identity element.

    Kept as a shape test after `_norm_unusable` was removed: what used to need
    a `vecs[suspect].amax(dim=1)` over a zero-width row is now a comparison on
    the norm vector alone, but the empty case still reaches `norm_guard` from
    an empty slice and still has to come back as an empty mask rather than a
    RuntimeError halfway down a rank.
    """
    assert twopass.norm_guard(torch.zeros(0)).shape == (0,)
    assert twopass.norm_guard(torch.zeros(0)).dtype is torch.bool
    # A (4, 0) matrix has four norms, all exactly zero, and zero is refused.
    mask = twopass.norm_guard(torch.zeros(4, 0).norm(dim=1))
    assert mask.shape == (4,) and bool(mask.all())
    assert twopass.product_guard(torch.zeros(0), 1.0).shape == (0,)
    assert twopass.prunability_cut(torch.zeros(0), 1.0).shape == (0,)


def test_the_query_cache_checks_identity_not_just_the_address():
    """`_QCACHE` is keyed on `id(Q)`, and CPython reuses addresses.

    The entry itself holds `Q` alive, though, so the recycle cannot actually
    happen while the entry lives — measured: the cached `Q`'s refcount stays
    at 2 after every other reference is dropped, and 200,000 fresh
    allocations never landed on the address. So be precise about what this
    test is for: the `got["Q"] is Q` re-check is currently UNREACHABLE in
    production, and this pins that the line still exists rather than proving
    it prevents a live bug. That is worth doing — the line looks redundant
    beside an address-keyed dict and is exactly what a tidy-up deletes — but
    it is defence in depth, not a guard over a reachable path.

    Simulated rather than raced: planting the entry under the other matrix's
    id is the same state an address recycle produces, without depending on the
    allocator to reuse an address during a test.
    """
    twopass.reset()
    Q1 = torch.full((4, 32), 1.0)
    Q2 = torch.full((4, 32), 7.0)          # different data, different norms

    first = twopass.query_side(Q1)
    assert first["Q"] is Q1

    # As if Q1 had been freed and Q2 landed on its address.
    twopass._QCACHE[id(Q2)] = first

    got = twopass.query_side(Q2)
    assert got["Q"] is Q2, (
        "the cache returned another matrix's entry for this address; the "
        "identity re-check is missing and the bound would be computed for the "
        "wrong rows")
    assert torch.equal(got["qn"], twopass.query_side(Q2)["qn"])
    # and the values really are Q2's, not Q1's
    assert float(got["qn"][0]) > float(first["qn"][0]) * 5

def test_the_overflow_guard_refuses_a_slice_that_would_overflow_on_conversion():
    """`kappa_bar(d) * max N <= Omega_t (1 - 2**-40)`, per axis, per file.

    MONOTONE in the norm, which is what makes one scalar comparison settle a
    whole axis — and why this can be a per-file check rather than a per-vector
    one. Evaluated with the INFLATED `kappa_bar` rather than a re-rounded
    `kappa`: an inward-rounded comparison would not establish Theorem 1's
    hypothesis, and the `2**-40` margin absorbs the single binary64 rounding of
    the product itself.
    """
    twopass.reset()
    d = DIM
    limit = cf.OMEGA16 * (1.0 - cf.GUARD_MARGIN) / cf.kappa_bar(d)
    Q = torch.randn(8, d)
    C = torch.nn.functional.normalize(torch.randn(16, d), dim=1) * (limit * 1.01)
    assert not twopass.overflow_guard(d, C.norm(dim=1), cf.FP16)
    upper = twopass.upper_bounds(Q, C, None, None, torch.float32, "dot")
    assert bool(torch.isinf(upper).all()) and bool((upper > 0).all()), (
        "a slice that could overflow on conversion must go live; an overflow "
        "is not a bounded rounding and `eps` does not describe it")
    assert twopass.stats()["slices_guard_refused"] == 1

    # Just inside the limit, the same shape is admitted — a guard that refused
    # everything would satisfy the assertion above and prune nothing ever.
    twopass.reset()
    C_ok = torch.nn.functional.normalize(torch.randn(16, d), dim=1) * (limit * 0.99)
    assert twopass.overflow_guard(d, C_ok.norm(dim=1), cf.FP16)

    # An axis ALREADY stored in the first-pass format is not converted, so it
    # cannot overflow on conversion and the guard does not apply to it. Getting
    # this backwards would refuse every fp16 corpus in the fleet.
    assert twopass.overflow_guard(d, C.norm(dim=1), cf.EXACT)
    # And an empty axis has no maximum to compare.
    assert twopass.overflow_guard(d, torch.zeros(0), cf.FP16)


def test_the_product_guard_is_per_row_and_decided_without_rounding():
    """`N_q * max_S N_c <= 2**126`: no binary32 overflow in either pass.

    Per ROW rather than per slice, so one enormous query does not cost the
    slice its pruning — the same argument as the per-row norm guard. And
    decided in binary64, where the product of two binary32 norms is EXACT (48
    significand bits, exponents well inside range): in binary32 a real product
    just above the threshold rounds DOWN onto it and passes.
    """
    qn = torch.tensor([1.0, 3e30, 1.0])
    assert twopass.product_guard(qn, 1e10).tolist() == [False, True, False], (
        "one enormous query row took the whole slice; the guard is per row")
    # A NaN on the corpus side refuses the whole axis — there is no per-row
    # answer to a slice whose maximum norm is not a number.
    assert bool(twopass.product_guard(qn, float("nan")).all())
    # Exactly at the threshold is admitted; one binary32 ulp past it is not.
    at = float(np.float32(2.0 ** 126))
    assert not bool(twopass.product_guard(torch.tensor([at]), 1.0).any())
    over = float(np.nextafter(np.float32(at), np.float32(np.inf)))
    assert bool(twopass.product_guard(torch.tensor([over]), 1.0).all())


def test_the_prunability_cut_declines_to_run_pass_one_where_it_cannot_pay():
    """Not a safety guard — a WASTE guard, and the answer to `clamp_min(1e-12)`.

    `Psi` is perfectly valid at these magnitudes; it is just enormous. It grows
    as `1/N_q` and `1/min_S N_c`, so once it exceeds the typical gap between a
    slice's best score and the cutoff the slice cannot prune and pass one is
    pure overhead. The cuts are deliberately more conservative than the
    crossovers at which `Psi` overtakes `C` (1.2e-3 and 2.0e-5): at d = 768
    they keep `Psi < 1e-4`.

    The corpus half is slice-wide because `cn_min` is; the query half is per
    row, like every other query-side mask.
    """
    assert cf.PRUNE_MIN_QNORM == 1e-2 and cf.PRUNE_MIN_CNORM == 1e-3
    qn = torch.tensor([1.0, 1e-3, 1.0])
    assert twopass.prunability_cut(qn, 1.0).tolist() == [False, True, False]
    # A single near-zero corpus norm takes the slice: there is no useful bound
    # to compute for it, so pass one is skipped rather than run and discarded.
    assert bool(twopass.prunability_cut(qn, 1e-4).all())
    assert bool(twopass.prunability_cut(qn, float("nan")).all())
    # On the cut itself the row survives — `~(qn >= cut)`, not `>`.
    assert not bool(twopass.prunability_cut(
        torch.tensor([cf.PRUNE_MIN_QNORM]), cf.PRUNE_MIN_CNORM).any())


def test_a_slice_every_row_refuses_never_runs_pass_one_at_all():
    """The refusals are ordered so that the GEMM is skipped, not wasted.

    `upper_bounds` checks `bad.all()` before calling `approx_rowmax`. Without
    it, a slice of near-zero-norm corpus vectors pays a full pass-one GEMM
    whose every answer is then overwritten with `+inf` — which is exactly the
    shape the prunability cut exists to avoid.
    """
    twopass.reset()
    Q = torch.nn.functional.normalize(torch.randn(16, DIM), dim=1) * 1e-4
    C = torch.nn.functional.normalize(torch.randn(24, DIM), dim=1)
    assert float(Q.norm(dim=1).max()) < cf.PRUNE_MIN_QNORM

    real_matmul = torch.Tensor.__matmul__
    seen = []

    def watch(self, other):
        seen.append(tuple(self.shape))
        return real_matmul(self, other)

    torch.Tensor.__matmul__ = watch
    try:
        upper = twopass.upper_bounds(Q, C, None, None, None, "dot")
    finally:
        torch.Tensor.__matmul__ = real_matmul

    assert bool(torch.isinf(upper).all()) and bool((upper > 0).all())
    assert not seen, (
        f"pass one ran for a slice whose every row was already refused: {seen}")
    assert twopass.stats()["slices_guard_refused"] == 1


def test_a_row_max_kernel_failure_falls_back_instead_of_killing_the_run(monkeypatch):
    """`approx_rowmax`'s Triton launch must degrade, like every other one.

    It was the only launch in the module without a `try`, and the worst one to
    leave bare: this branch is reached exactly when the fused kernel has
    already declined or disabled itself, so it is the fallback of a fallback.
    Nothing between it and `run_compute`'s consumer loop catches — verified by
    injection, which killed the rank at file 2 of 8. On a sharded run that is
    a missing partial and a merge that refuses the directory, which is what
    `disable()`'s "costs speed and nothing else" promise exists to prevent.

    Reaching the branch on a CPU box takes forcing `is_cuda`, which then makes
    the launch itself fail — which is precisely the condition under test. The
    result must still be the correct row max, computed by the portable loop.
    """
    twopass.reset()
    torch.manual_seed(5)
    Qh = torch.randn(16, 32).half()
    Ch = torch.randn(24, 32).half()
    cs = torch.rand(24) + 0.5

    # `out_dtype=None` on purpose: the explicit-dtype `torch.mm` is CUDA-only,
    # so it would raise before reaching the branch under test.
    want = ((Qh @ Ch.T).float() * cs).amax(dim=1)

    # Every real tensor now claims to be on CUDA, so the guarded branch is
    # entered — and then the launch (or the CUDA device context around it)
    # raises on a CPU box, which is exactly the failure being tested.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    try:
        got, fused = twopass.approx_rowmax(Qh, Ch, cs, None)
    finally:
        monkeypatch.undo()
        twopass.reset()

    assert not fused, "this exercises the unfused path"
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-5), (
        f"the fallback did not reproduce the portable row max: worst diff "
        f"{float((got - want).abs().max()):.3e}")


def test_a_transient_oom_in_verification_is_not_cached_as_a_bad_height():
    """An OOM is a fact about the MOMENT; a mismatch is a fact about the HEIGHT.

    Caching the first as the second meant one memory spike blacklisted a
    padded height permanently. And because every rung of the ladder allocates
    the same full-height reference (~1.8 GB at the production shape),
    sustained pressure fails all four together and reaches `disable()` —
    silently costing a multi-hour rank its speedup, logged only as a WARNING.
    """
    twopass.reset()
    Q = torch.randn(64, 32)
    C = torch.randn(16, 32)

    real = torch.Tensor.__matmul__
    calls = []

    def flaky(self, other):
        calls.append(1)
        if len(calls) == 2:                 # the full-height reference GEMM
            raise torch.OutOfMemoryError("simulated transient OOM")
        return real(self, other)

    torch.Tensor.__matmul__ = flaky
    try:
        first = twopass._verify_shape(Q, C, 16)
    finally:
        torch.Tensor.__matmul__ = real
    assert first is twopass.UNVERIFIED, (
        f"an OOM must be reported as `could not check`, not as a verdict "
        f"about the height; got {first!r}")
    assert not first, (
        "the sentinel has to stay falsy — `if _verify_shape(...)` must not "
        "start trusting an unproven height")
    assert twopass.stats()["verifications"] == 0, (
        "a check that never completed was counted as a completed one, which "
        "is what makes the manifest read as a measurement of this device")

    second = twopass._verify_shape(Q, C, 16)
    assert second is True, (
        "the height was still refused with memory healthy — the transient "
        "failure was cached as a permanent verdict about the height")
    assert twopass.stats()["verifications"] == 1


def test_an_oom_that_arrives_as_a_runtime_error_is_still_an_oom():
    """`torch.OutOfMemoryError` is only what the CACHING ALLOCATOR raises.

    The other two ways a GPU runs out during this check both arrive as a bare
    `RuntimeError`: a cuBLAS workspace failure
    (`CUBLAS_STATUS_ALLOC_FAILED`) and a raw driver failure
    (`CUDA error: out of memory`). The second is the likely one here, because
    the reference product is the largest allocation the two-pass makes and
    cuBLAS asks the driver directly for its workspace.

    An isinstance-only test let both through as genuine verdicts, so the very
    failures the OOM branch exists for were the ones that missed it — the
    height got cached as bad, and a full ladder of them called `disable()`.
    """
    twopass.reset()
    assert twopass.is_oom(torch.OutOfMemoryError("allocator"))
    assert twopass.is_oom(MemoryError())
    assert twopass.is_oom(RuntimeError("CUDA error: out of memory"))
    assert twopass.is_oom(RuntimeError(
        "cuBLAS API failed with status 15: CUBLAS_STATUS_ALLOC_FAILED"))
    # torch's HOST allocator words it a third way, and this is the ONE
    # flavour reproducible without a GPU — so it is the flavour the CPU path
    # (`NOVA_BF_DEVICE=cpu`, the whole parity suite) actually meets. Missing
    # it left the original bug alive exactly where it could be observed.
    assert twopass.is_oom(RuntimeError(
        "[enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: "
        "can't allocate memory: you tried to allocate 40000000000000 bytes. "
        "Error code 12 (Cannot allocate memory)"))
    assert not twopass.is_oom(RuntimeError(
        "expected scalar type Half but found Float")), (
        "a caller bug must stay CACHEABLE — treating it as transient means "
        "retrying a deterministic failure on every slice for the whole run")

    Q = torch.randn(32, 16)
    C = torch.randn(8, 16)
    real = torch.Tensor.__matmul__
    calls = []

    def flaky(self, other):
        calls.append(1)
        if len(calls) == 2:                 # the full-height reference GEMM
            raise RuntimeError("CUDA error: out of memory")
        return real(self, other)

    torch.Tensor.__matmul__ = flaky
    try:
        verdict = twopass._verify_shape(Q, C, 8)
    finally:
        torch.Tensor.__matmul__ = real

    assert verdict is twopass.UNVERIFIED, (
        f"a driver-level OOM was taken as a verdict about the height, got "
        f"{verdict!r}")
    assert twopass._verify_shape(Q, C, 8) is True, (
        "it was cached, so the height stays blacklisted for the whole run")


def test_a_fused_row_max_that_cannot_allocate_does_not_disable_the_fused_path():
    """Answering memory pressure by switching to the path that needs MORE.

    `fuse_disable` lasts the whole run, and what it falls back to is the
    cuBLAS pair, which materialises the entire fp16 Gram.

    A real allocation failure has to cost this slice and nothing more.
    """
    twopass.reset()
    Qh = torch.randn(8, 16).half()
    Ch = torch.randn(12, 16).half()
    cs = torch.rand(12) + 0.5

    def oom(*a, **kw):
        raise RuntimeError("CUDA error: out of memory")

    # `fuse_available` gates on CUDA, so the guarded region is only reached
    # with the tensors claiming a device; `torch.cuda.device` is the first
    # thing inside the `try` and stands in for the allocation under it.
    real_is_cuda = torch.Tensor.is_cuda
    real_device = torch.cuda.device
    torch.Tensor.is_cuda = property(lambda self: True)
    torch.cuda.device = oom
    try:
        assert twopass.fuse_available(Qh, Ch, cs), (
            "the fused path declined before the guarded region, so this test "
            "proves nothing")
        with pytest.raises(twopass.PassOneUnavailable):
            twopass.fused_rowmax(Qh, Ch, cs)
        # Read BEFORE the cleanup: `reset()` clears `_FUSE_OFF`, so asserting
        # after it would pass no matter what the handler did.
        status = twopass.fused_off()
    finally:
        torch.Tensor.is_cuda = real_is_cuda
        torch.cuda.device = real_device
        twopass.reset()

    assert status is None, (
        f"an allocation failure disabled the fused kernel for the whole "
        f"process: {status!r}")


def test_a_fused_final_reduction_oom_takes_the_pass_one_fallback(monkeypatch):
    """`part.amax` is both an allocation and a deferred-CUDA-error boundary."""
    from contextlib import nullcontext

    twopass.reset()
    Qh = torch.randn(8, 16).half()
    Ch = torch.randn(17, 16).half()  # BLOCK_N=16 gives two row-max tiles.
    cs = torch.rand(17, dtype=torch.float32)

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(Qh, Ch, cs, part, *args, **kwargs):
                part.zero_()
                return object()
            return launch

    real_amax = torch.Tensor.amax
    seen = []

    def oom_on_tile_reduction(self, *args, **kwargs):
        if tuple(self.shape) == (2, 8):
            seen.append(tuple(self.shape))
            raise torch.OutOfMemoryError("CUDA out of memory in tile reduction")
        return real_amax(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.Tensor, "amax", oom_on_tile_reduction)
    monkeypatch.setattr(twopass, "_gemm_rowmax", FakeKernel())
    try:
        with pytest.raises(twopass.PassOneUnavailable):
            twopass.fused_rowmax(
                Qh, Ch, cs, config=(16, 16, 16, 1, 1, 1),
            )
        status = twopass.fused_off()
    finally:
        twopass.reset()

    assert seen, "the final tile reduction was not reached"
    assert status is None, "a transient allocation failure disabled fused pass one"


def test_the_row_max_fallback_warns_once_not_once_per_slice(caplog):
    """Per slice per score group is ~10^5 identical lines on a production rank.

    The failure this handler catches is a property of the Triton install or
    the shape, so it is deterministic: once it fires it fires every time. The
    module already learned this twice — `_THRESHOLD_WARNED` and
    `_FUSE_CONFIG_WARNED` — and this handler was added later without it.
    """
    import logging

    twopass.reset()
    torch.manual_seed(5)
    Qh = torch.randn(8, 16).half()
    Ch = torch.randn(12, 16).half()
    cs = torch.rand(12) + 0.5

    prop = property(lambda self: True)
    real_is_cuda = torch.Tensor.is_cuda
    torch.Tensor.is_cuda = prop
    try:
        with caplog.at_level(logging.WARNING, logger="nova_bf.twopass"):
            for _ in range(4):
                twopass.approx_rowmax(Qh, Ch, cs, None)
    finally:
        torch.Tensor.is_cuda = real_is_cuda
        twopass.reset()

    hits = [r for r in caplog.records if "row-max kernel failed" in r.getMessage()]
    assert len(hits) == 1, (
        f"the row-max fallback logged {len(hits)} times for 4 slices; at "
        f"production rates that is a warning per score group forever")


def test_a_new_run_clears_the_warn_once_flags(caplog):
    """Warn-ONCE has to mean once per RUN, not once per process.

    `reset()` already argues this for `_DISABLED_REASON`: a second
    `run_compute` in one process (`nova dist` in-process, and this test
    suite) is a different run, and its operator needs to be told that its
    two-pass is degraded. Left set, the second run's log is indistinguishable
    from a run where nothing went wrong at all.
    """
    import logging

    twopass.reset()
    torch.manual_seed(5)
    Qh = torch.randn(8, 16).half()
    Ch = torch.randn(12, 16).half()
    cs = torch.rand(12) + 0.5

    prop = property(lambda self: True)
    real_is_cuda = torch.Tensor.is_cuda
    torch.Tensor.is_cuda = prop
    try:
        with caplog.at_level(logging.WARNING, logger="nova_bf.twopass"):
            twopass.approx_rowmax(Qh, Ch, cs, None)
            twopass.reset()                       # a new run starts here
            caplog.clear()
            twopass.approx_rowmax(Qh, Ch, cs, None)
    finally:
        torch.Tensor.is_cuda = real_is_cuda
        twopass.reset()

    hits = [r for r in caplog.records if "row-max kernel failed" in r.getMessage()]
    assert len(hits) == 1, (
        "the second run inherited the first run's warn-once flag, so it "
        "never reported that its row-max had fallen back")


def test_a_pass_one_that_cannot_allocate_skips_the_slice_instead_of_retrying_bigger():
    """ Pass one gives up for this slice, `upper_bounds`
    answers all-live (`+inf`), and the caller's `pad_height` guard sends the
    slice down the ordinary one-pass path — which allocates no more than the
    run was always going to. All-live is the safe direction by construction:
    the two-pass can only lose ground truth by calling a row DEAD.
    """
    twopass.reset()
    # `DIM`, not 16: P6 refuses anything below 64 BEFORE pass one is reached,
    # so a narrow fixture would exercise the dimension guard and never touch
    # the allocation handler under test.
    Q = torch.randn(8, DIM)
    Cb = torch.randn(12, DIM)

    def oom(*a, **kw):
        raise RuntimeError("CUDA error: out of memory")

    real_is_cuda = torch.Tensor.is_cuda
    real_device = torch.cuda.device
    real_matmul = torch.Tensor.__matmul__
    seen = []

    def watch_matmul(self, other):
        seen.append(tuple(self.shape))
        return real_matmul(self, other)

    torch.Tensor.is_cuda = property(lambda self: True)
    torch.cuda.device = oom
    torch.Tensor.__matmul__ = watch_matmul
    try:
        got = twopass.upper_bounds(Q, Cb, None, None, None, "dot")
        st = twopass.stats()
    finally:
        torch.Tensor.is_cuda = real_is_cuda
        torch.cuda.device = real_device
        torch.Tensor.__matmul__ = real_matmul
        twopass.reset()

    assert torch.isinf(got).all() and (got > 0).all(), (
        f"the bound must be all-live after a pass-one OOM, got {got}")
    assert st["slices_pass_one_oom"] == 1, (
        f"the slice was not counted, so a rank in this state looks identical "
        f"to one whose two-pass simply never triggered: {st}")
    assert not seen, (
        f"a matmul ran after the allocation failure — that is the 0.9 GB "
        f"Gram this branch exists to avoid: {seen}")


@pytest.mark.parametrize("which", ("query", "corpus"))
def test_a_pass_one_input_copy_oom_forces_the_slice_live(monkeypatch, which):
    """The Q/C fp16 copies are pass-one allocations, not setup outside it.

    `approx_rowmax` cannot catch an OOM from either conversion because both
    happen first.  This pins the actual `upper_bounds` contract instead: each
    one must count as a pass-one OOM and conservatively return all-live.
    """
    twopass.reset()
    Q = torch.randn(8, DIM)
    Cb = torch.randn(12, DIM)
    victim = Q if which == "query" else Cb
    real_half = torch.Tensor.half
    seen = []

    def oom_on_victim(self, *args, **kwargs):
        if self is victim:
            seen.append(which)
            raise torch.OutOfMemoryError("CUDA out of memory in fp16 copy")
        return real_half(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "half", oom_on_victim)
    try:
        got = twopass.upper_bounds(Q, Cb, None, None, None, "dot")
        st = twopass.stats()
    finally:
        twopass.reset()

    assert seen, f"the {which} fp16 conversion was never reached"
    assert torch.isinf(got).all() and bool((got > 0).all())
    assert st["slices_pass_one_oom"] == 1


def test_a_non_oom_input_copy_failure_is_not_disguised_as_memory(monkeypatch):
    """Only allocation failures may take the silent all-live fallback."""
    twopass.reset()
    Q = torch.randn(8, DIM)
    Cb = torch.randn(12, DIM)
    real_half = torch.Tensor.half
    marker = "deliberate fp16-copy bug"

    def broken_query_copy(self, *args, **kwargs):
        if self is Q:
            raise RuntimeError(marker)
        return real_half(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "half", broken_query_copy)
    try:
        with pytest.raises(RuntimeError, match=marker):
            twopass.upper_bounds(Q, Cb, None, None, None, "dot")
        st = twopass.stats()
    finally:
        twopass.reset()

    assert st["slices_pass_one_oom"] == 0


def test_the_row_max_fallback_actually_stops_launching_not_just_warning():
    """"Using the portable reduction from here on" has to be TRUE.

    Warn-once silenced the log and left the launch retried once per slice per
    score group. The failure this handler catches is deterministic — a Triton
    version issue, a shape that will not compile — so those ~10^5 retries all
    fail the same way, each paying the compile pipeline again. Silence plus
    the full cost is worse than noise plus the full cost, because nothing is
    left to notice it.
    """
    twopass.reset()
    torch.manual_seed(5)
    Qh = torch.randn(8, 16).half()
    Ch = torch.randn(12, 16).half()
    cs = torch.rand(12) + 0.5

    launches = []
    real_device = torch.cuda.device

    def counting(dev):
        launches.append(1)
        return real_device(dev)

    real_is_cuda = torch.Tensor.is_cuda
    torch.Tensor.is_cuda = property(lambda self: True)
    torch.cuda.device = counting
    try:
        for _ in range(6):
            twopass.approx_rowmax(Qh, Ch, cs, None)
    finally:
        torch.Tensor.is_cuda = real_is_cuda
        torch.cuda.device = real_device
        twopass.reset()

    # One launch attempt for the fused kernel and one for `_rowmax_scaled` on
    # the first call; nothing on the five after it.
    assert len(launches) <= 2, (
        f"{len(launches)} kernel launch attempts over 6 slices — the "
        f"fallback is not sticky, so a deterministic failure is re-paid "
        f"every slice for the life of the rank")


def test_a_completed_check_clears_the_unchecked_streak():
    """The limiter must count CONSECUTIVE failures, not lifetime ones.

    A long run on a healthy device sees the occasional transient spike. If
    those accumulate, the limiter eventually fires on a machine that has been
    verifying happily for hours — turning the safeguard against thrashing
    into a slow-acting version of the bug it replaced.

    Both resets are exercised, and the SECOND one is the one that matters.
    Clearing the streak inside `_verify_shape_locked` alone looked right and
    was almost inert in production: that function runs only on a cache MISS,
    and at steady state `n_live` is stable, so the ladder asks for the same
    height every slice and nearly every healthy slice is a cache HIT that
    returns before the lock. 
    """
    twopass.reset()
    Q = torch.randn(32, 16)
    C = torch.randn(8, 16)

    # (a) a completed check on a novel shape
    for _ in range(twopass.MAX_UNCHECKED_SLICES - 1):
        assert not twopass.note_unchecked_slice()
    assert twopass._verify_shape(Q, C, 8) is True, "premise: a check completed"
    for _ in range(twopass.MAX_UNCHECKED_SLICES - 1):
        assert not twopass.note_unchecked_slice(), (
            "the streak survived a completed verification")

    # Each arm below starts from a clean streak: the increments left over
    # from the previous arm would otherwise carry across and trip the limit
    # on their own, for reasons that have nothing to do with the reset.
    twopass.note_checked_slice()

    # (b) a check that COMPLETED with a verdict of False. The docstring's
    # claim is "whatever the verdict, the device ANSWERED", and a fixture
    # that only ever verifies True cannot tell that apart from "resets only
    # on a matching height" — `if ok: _UNCHECKED_STREAK = 0` passed
    # everything.
    for _ in range(twopass.MAX_UNCHECKED_SLICES - 1):
        assert not twopass.note_unchecked_slice()
    # `_HeightSensitive` is parametrised by the FULL height it should agree
    # at, which must be Q's own (32) — and M must be a height arm (a) has not
    # already cached, or `_verify_shape` returns the cached True without
    # running the check at all.
    assert twopass._verify_shape(Q, _HeightSensitive(C, 32), 16) is False, (
        "premise: this height must genuinely disagree")
    for _ in range(twopass.MAX_UNCHECKED_SLICES - 1):
        assert not twopass.note_unchecked_slice(), (
            "a completed check that returned False left the streak standing "
            "— the device answered, so the memory pressure the streak counts "
            "had demonstrably passed")

    twopass.note_checked_slice()

    # (c) a CACHE HIT — the common case, and the one the inner reset misses.
    before = twopass.stats()["verifications"]
    assert twopass._verify_shape(Q, C, 8) is True
    assert twopass.stats()["verifications"] == before, (
        "premise: this call must be a CACHE HIT, not a fresh check — the "
        "whole point of this arm is the path that returns before the lock")
    twopass.note_checked_slice()
    for _ in range(twopass.MAX_UNCHECKED_SLICES - 1):
        assert not twopass.note_unchecked_slice(), (
            "a healthy slice whose height was already proven did not clear "
            "the streak, so `16 consecutive` really means `16 for the whole "
            "run` and unrelated spikes hours apart add up to a disable")
    twopass.reset()


def test_config_ok_rejects_a_group_m_large_enough_to_lose_output_lanes():
    """`_config_ok`'s UPPER bound, which its own comment is the argument for.

    The existing test covers `gm` of 0, -1 and 1 — the lower end. The comment
    at the check describes the other end in detail: a huge `GROUP_M` leaves
    most output lanes unwritten with no exception, and near `2**31` the
    derived `pid_m` goes NEGATIVE, at which point the epilogue's `offs_m < M`
    mask — one-sided — lets the store land below the buffer. An unwritten
    lane is `+inf` and costs a slice; a negative one is memory corruption.

    Deleting `<= 1024` passed the whole file, so the half of the check that
    guards the worse failure was the untested half.
    """
    base = twopass.fuse_config(torch.randn(8, 16).half(),
                               torch.randn(12, 16).half())
    bm, bn, bk, gm, stages, warps = base
    assert twopass._config_ok(base), "premise: the derived config is accepted"

    for bad_gm in (1025, 4096, 2 ** 20, 2 ** 30):
        assert not twopass._config_ok((bm, bn, bk, bad_gm, stages, warps)), (
            f"GROUP_M={bad_gm} was accepted; past 1024 the tile grid stops "
            f"covering the output and near 2**31 pid_m goes negative")

    for bad_warps in (3, 5, 6, 7, 9, 12):
        assert not twopass._config_ok((bm, bn, bk, gm, stages, bad_warps)), (
            f"num_warps={bad_warps} is not a power of two; Triton's warp "
            f"scheduling assumes it is")

    for bad_block in (0, 8, 24, 100):
        assert not twopass._config_ok((bad_block, bn, bk, gm, stages, warps)), (
            f"BLOCK_M={bad_block} is not a power of two >= 16")
        assert not twopass._config_ok((bm, bad_block, bk, gm, stages, warps)), (
            f"BLOCK_N={bad_block} is not a power of two >= 16")
        assert not twopass._config_ok((bm, bn, bad_block, gm, stages, warps)), (
            f"BLOCK_K={bad_block} is not a power of two >= 16")

    for malformed in ((16, 16, 16, 1, 2), (16, 16, 16, 1, 2, 4, 8),
                      (16, 16, 16, 1, 2, "8"), ()):
        assert not twopass._config_ok(malformed), (
            f"{malformed!r} is not six integers")


def test_a_disable_keeps_the_first_reason():
    """The reason has to name what went wrong FIRST.

    `disable()` is called per slice on the failing path, so without the
    `is None` guard the manifest records whichever slice happened to fail
    last — and a later, more generic reason then overwrites the specific one
    that would tell an operator what actually happened. `fuse_disable`'s
    identical claim is tested; this one was not.
    """
    twopass.reset()
    assert twopass.enabled()
    twopass.disable("the first and real reason")
    twopass.disable("a later, less useful one")
    assert twopass.stats()["unavailable"] == "the first and real reason", (
        f"the reason was overwritten: {twopass.stats()['unavailable']!r}")
    assert not twopass.enabled()
    twopass.reset()
    assert twopass.stats()["unavailable"] is None
    assert twopass.enabled()


def test_a_real_host_allocation_failure_is_not_cached_as_a_bad_height():
    """The end-to-end version of the message match, on the real allocator.

    Not a hand-written string: this asks torch for an impossible host
    allocation and feeds whatever it actually raises to `_verify_shape`. If
    torch rewords the message, this test fails rather than quietly going
    green against a message nobody produces any more.
    """
    try:
        torch.empty((2 ** 44,), dtype=torch.float64)
    except Exception as real:          # noqa: BLE001 - whatever torch raises
        exc = real
    else:
        pytest.skip("this box actually allocated 128 TiB")

    assert twopass.is_oom(exc), (
        f"the real host allocator's message is not recognised as an OOM, so "
        f"a genuine allocation failure on the CPU path is cached as a "
        f"permanently-bad height: {type(exc).__name__}: {str(exc)[:120]}")

    twopass.reset()
    Q = torch.randn(32, 16)
    C = torch.randn(8, 16)
    real_matmul = torch.Tensor.__matmul__
    calls = []

    def flaky(self, other):
        calls.append(1)
        if len(calls) == 2:            # the full-height reference GEMM
            raise exc
        return real_matmul(self, other)

    torch.Tensor.__matmul__ = flaky
    try:
        verdict = twopass._verify_shape(Q, C, 8)
    finally:
        torch.Tensor.__matmul__ = real_matmul

    assert verdict is twopass.UNVERIFIED, f"got {verdict!r}"
    assert twopass._verify_shape(Q, C, 8) is True, (
        "the height stayed blacklisted for the run")
    twopass.reset()


def test_the_fp16_split_k_reduction_is_pinned_off_on_every_pass_one():
    """The one setting the bound cannot survive being wrong about.

    torch defaults `allow_fp16_reduced_precision_reduction` to True, which
    lets cuBLAS accumulate a split-K reduction in FLOAT16. `bound()`'s
    `gamma_d = d*u/(1 - d*u)` is the classic float32 dot-product bound and
    does not cover that at all — with a float16 accumulator the error grows
    with `2**-11` per step instead of `2**-24`, roughly 8000x the budget the
    bound allocates. `eps` would then UNDERSTATE the true error, which is the
    unsafe direction: a row whose real score clears the threshold gets an
    upper bound below it and is called dead. Silent lost ground truth.

    Two things are asserted, and the second is the reason the function does
    not use a once-only guard: the flag is re-pinned on EVERY pass one, so
    anything else in the process flipping it back on is corrected before the
    next approximate GEMM rather than silently invalidating every bound
    computed after it. torch exposes the attribute whether or not there is a
    CUDA device, so this holds on both.
    """
    import torch

    original = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    twopass.reset()
    # At `DIM`, so pass one actually RUNS — below 64 the dimension guard
    # returns all-live before `_pin_accumulation` is ever called and the
    # assertions below would hold vacuously.
    Q = torch.randn(8, DIM)
    Cb = torch.randn(12, DIM)
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        twopass.upper_bounds(Q, Cb, None, None, None, "dot")
        assert not torch.backends.cuda.matmul.\
            allow_fp16_reduced_precision_reduction, (
            "pass one ran with the fp16 split-K reduction ENABLED; gamma_d "
            "does not cover it, so every eps this run computes understates "
            "the error and can call a live row dead")

        # Again, after something else flips it back. A once-only guard passes
        # the assertion above and fails this one.
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        twopass.upper_bounds(Q, Cb, None, None, None, "dot")
        assert not torch.backends.cuda.matmul.\
            allow_fp16_reduced_precision_reduction, (
            "the flag was pinned once and never re-checked, so anything in "
            "the process that turns it back on silently invalidates every "
            "bound computed from then on")
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = original
        twopass.reset()


def _force_unfused(monkeypatch):
    """Make `approx_rowmax` take the cuBLAS branch, as three production
    configurations do: any non-OOM kernel failure (`fuse_disable`), Triton
    absent, or the `NOVA_BF_NO_FUSED_ROWMAX` kill switch."""
    monkeypatch.setattr(twopass, "_gemm_rowmax", None)


def test_an_unfused_pass_one_that_cannot_allocate_skips_the_slice_too():
    twopass.reset()
    Q = torch.randn(8, DIM)
    Cb = torch.randn(12, DIM)

    real_matmul = torch.Tensor.__matmul__
    calls = []

    def oom_on_the_gram(self, other):
        calls.append(tuple(self.shape))
        raise torch.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 900.00 MiB")

    real_is_cuda = torch.Tensor.is_cuda
    torch.Tensor.is_cuda = property(lambda self: True)
    torch.Tensor.__matmul__ = oom_on_the_gram
    try:
        got = twopass.upper_bounds(Q, Cb, None, None, None, "dot")
        st = twopass.stats()
    finally:
        torch.Tensor.is_cuda = real_is_cuda
        torch.Tensor.__matmul__ = real_matmul
        twopass.reset()

    assert calls, "the cuBLAS branch was never reached; this proves nothing"
    assert torch.isinf(got).all() and (got > 0).all(), (
        f"an OOM in the unfused Gram escaped instead of degrading to the "
        f"all-live bound: {got}")
    assert st["slices_pass_one_oom"] == 1, (
        f"the slice was not counted as a pass-one OOM: {st}")


def test_a_transient_oom_does_not_permanently_pick_the_hungrier_row_max():
    """`_ROWMAX_OFF` must be sticky for a DETERMINISTIC failure only.

    The case for stickiness is that a Triton version issue or a shape that
    will not compile fails identically every time, so retrying it once per
    slice re-pays the compile pipeline for an answer that cannot change. An
    OOM is the opposite kind of fact, and the path stickiness commits to
    allocates a 256 MiB fp32 temporary per chunk where the kernel allocated
    nothing — so answering memory pressure by permanently choosing the
    hungrier path is the same inversion the fused handler was rewritten to
    avoid, one level down.
    """
    twopass.reset()
    Qh = torch.randn(8, 16).half()
    Ch = torch.randn(12, 16).half()
    cs = torch.rand(12) + 0.5

    def oom(*a, **kw):
        raise RuntimeError("CUDA error: out of memory")

    real_is_cuda = torch.Tensor.is_cuda
    real_device = torch.cuda.device
    real_gemm = twopass._gemm_rowmax
    # The FUSED kernel shares `torch.cuda.device`, so without this it catches
    # the injected OOM first and `_rowmax_scaled` — the handler under test —
    # never runs at all. `_gemm_rowmax = None` is also the real configuration
    # of a box without Triton, which is one of the ways this path is reached.
    twopass._gemm_rowmax = None
    torch.Tensor.is_cuda = property(lambda self: True)
    torch.cuda.device = oom
    try:
        with pytest.raises(twopass.PassOneUnavailable):
            twopass.approx_rowmax(Qh, Ch, cs, None)
        sticky_after_oom = twopass._ROWMAX_OFF
    finally:
        torch.Tensor.is_cuda = real_is_cuda
        torch.cuda.device = real_device
        twopass._gemm_rowmax = real_gemm
        twopass.reset()

    assert not sticky_after_oom, (
        "one transient OOM permanently committed the run to the portable "
        "reduction, which needs MORE memory than the kernel it replaced")


def test_a_deterministic_row_max_failure_is_still_sticky():
    """The other half: a non-OOM failure must still stop the retries."""
    twopass.reset()
    Qh = torch.randn(8, 16).half()
    Ch = torch.randn(12, 16).half()
    cs = torch.rand(12) + 0.5

    def boom(*a, **kw):
        raise RuntimeError("PTX assembly failed: unsupported instruction")

    real_is_cuda = torch.Tensor.is_cuda
    real_device = torch.cuda.device
    real_gemm = twopass._gemm_rowmax
    twopass._gemm_rowmax = None          # see the sibling test
    torch.Tensor.is_cuda = property(lambda self: True)
    torch.cuda.device = boom
    try:
        twopass.approx_rowmax(Qh, Ch, cs, None)
        sticky = twopass._ROWMAX_OFF
        reported = twopass.stats()["row_max"]
    finally:
        torch.Tensor.is_cuda = real_is_cuda
        torch.cuda.device = real_device
        twopass._gemm_rowmax = real_gemm
        twopass.reset()

    assert sticky, "a deterministic failure must not be retried every slice"
    assert reported == "portable", (
        f"the manifest still reports row_max={reported!r}; `slices_unfused` "
        f"alone cannot distinguish the Triton row max from the Python loop, "
        f"and one Triton version problem produces the loop for a whole run")


def test_a_non_allocation_failure_in_pass_one_is_not_disguised_as_memory():
    """`PassOneUnavailable` is for allocation failures and nothing else.

    It is answered by an all-live bound and a silent fall back to the
    one-pass path — the right response to memory pressure, and completely
    wrong for a bug. A dtype mismatch, a broken device, a shape error: each
    would be swallowed once per slice, the run would finish, and the only
    trace would be a WARNING claiming the machine was short of memory.
    Anything that is not an allocation failure has to keep its own type and
    its own traceback.
    """
    twopass.reset()
    Q = torch.randn(8, DIM)
    Cb = torch.randn(12, DIM)

    real_matmul = torch.Tensor.__matmul__
    marker = "expected scalar type Half but found Float"

    def bug(self, other):
        raise RuntimeError(marker)

    real_is_cuda = torch.Tensor.is_cuda
    real_gemm = twopass._gemm_rowmax
    twopass._gemm_rowmax = None
    torch.Tensor.is_cuda = property(lambda self: True)
    torch.Tensor.__matmul__ = bug
    try:
        with pytest.raises(RuntimeError) as caught:
            twopass.upper_bounds(Q, Cb, None, None, None, "dot")
        st = twopass.stats()
    finally:
        torch.Tensor.is_cuda = real_is_cuda
        torch.Tensor.__matmul__ = real_matmul
        twopass._gemm_rowmax = real_gemm
        twopass.reset()

    assert not isinstance(caught.value, twopass.PassOneUnavailable), (
        "a caller bug was reported as memory pressure and silently degraded")
    assert marker in str(caught.value), f"the original was lost: {caught.value}"
    assert st["slices_pass_one_oom"] == 0, (
        f"a bug was counted as an allocation failure: {st}")


def test_the_unchecked_limit_is_the_documented_one_and_fires_on_the_boundary():
    """The shipped `MAX_UNCHECKED_SLICES`, which no other test can see.

    Both tests that exercise the limiter monkeypatch this constant — to
    10,000 so it cannot fire, and to 2 or 3 so it fires quickly. That is
    correct for each of them and it leaves the value the run actually uses
    pinned by nothing: `MAX_UNCHECKED_SLICES = 1` passes the whole suite, and
    1 is precisely the round-7 bug restored — a single transient spike takes
    the two-pass down for the rest of the run.

    Same argument as `test_the_padding_constants_are_the_documented_ones`.
    The boundary is asserted too: `>=` silently becoming `>` also passes
    everything, and off-by-one on a threshold like this is invisible in
    production because both readings "work".
    """
    assert twopass.MAX_UNCHECKED_SLICES == 16, (
        "the limit is a judgement about how many consecutive failures stop "
        "being plausibly transient; 1 is the round-7 behaviour this replaced")

    twopass.reset()
    for i in range(twopass.MAX_UNCHECKED_SLICES - 1):
        assert not twopass.note_unchecked_slice(), (
            f"gave up after {i + 1} of {twopass.MAX_UNCHECKED_SLICES}")
    assert twopass.note_unchecked_slice(), (
        "did not give up ON the limit — an off-by-one here means the run "
        "thrashes for one more slice than documented, or gives up one early")
    twopass.reset()


def test_a_new_run_clears_every_piece_of_degradation_state():
    """`reset()` is the per-run boundary, and it has to be complete.

    Each flag here was added separately and only two of them had coverage.
    A missed one is not cosmetic: a warn-once flag that survives makes run 2
    of an in-process `nova dist` silent about its own degradation, and a
    surviving `_UNCHECKED_STREAK` is worse than silent — run 2 inherits run
    1's failures and can disable itself early on evidence that is not its
    own.

    Enumerated by NAME rather than by driving each path, so that a flag added
    later and forgotten shows up here as an obvious omission rather than as
    silence.
    """
    twopass.reset()
    flags = ("_THRESHOLD_WARNED", "_FUSE_CONFIG_WARNED", "_ROWMAX_WARNED",
             "_UNCHECKED_WARNED", "_FUSE_OOM_WARNED", "_VERIFY_OOM_WARNED",
             "_ROWMAX_OFF")
    for name in flags:
        assert hasattr(twopass, name), f"{name} no longer exists — update this test"
        setattr(twopass, name, True)
    twopass._UNCHECKED_STREAK = twopass.MAX_UNCHECKED_SLICES - 1
    twopass.disable("a reason from the previous run")
    twopass.fuse_disable("a fused failure from the previous run")

    twopass.reset()

    still_set = [n for n in flags if getattr(twopass, n)]
    assert not still_set, (
        f"{still_set} survived reset(); the next run in this process starts "
        f"degraded and says nothing about it")
    assert twopass._UNCHECKED_STREAK == 0, (
        "the next run inherits this run's unchecked failures and can give up "
        "early on evidence that is not its own")
    assert twopass.stats()["unavailable"] is None
    assert twopass.fused_off() is None
    assert twopass.enabled()


def test_the_verification_oom_counter_and_its_warning_are_both_bounded():
    """The counter two WARNINGs tell the operator to go read.

    `verifications_incomplete` is referenced by name in two log messages
    ("See verifications_incomplete for the count") and by the counter's own
    argument that `verifications: 0` alone is ambiguous — and nothing in the
    suite asserted it ever moved. The warn-once around it was equally
    unpinned, which matters because this line fires once per LADDER RUNG,
    four times per slice, underneath a caller line that is already gated.
    """
    import logging

    twopass.reset()
    Q = torch.randn(32, 16)
    C = torch.randn(8, 16)

    real_matmul = torch.Tensor.__matmul__
    calls = []

    def always_oom(self, other):
        calls.append(1)
        if len(calls) % 2 == 0:            # the full-height reference GEMM
            raise torch.OutOfMemoryError("simulated")
        return real_matmul(self, other)

    caplog_records = []

    class Grab(logging.Handler):
        def emit(self, record):
            caplog_records.append(record.getMessage())

    handler = Grab()
    logger = logging.getLogger("nova_bf.twopass")
    logger.addHandler(handler)
    torch.Tensor.__matmul__ = always_oom
    try:
        for M in (8, 16, 24, 32):          # four distinct heights = four rungs
            assert twopass._verify_shape(Q, C, M) is twopass.UNVERIFIED
        st = twopass.stats()
    finally:
        torch.Tensor.__matmul__ = real_matmul
        logger.removeHandler(handler)
        twopass.reset()

    assert st["verifications_incomplete"] == 4, (
        f"the counter two WARNINGs point the operator at did not move: {st}")
    assert st["verifications"] == 0, "no check completed"
    hits = [m for m in caplog_records if "ran out of" in m]
    assert len(hits) == 1, (
        f"{len(hits)} warnings for 4 rungs; this line fires per rung, four "
        f"times per slice, for as long as the pressure holds")


def test_a_dimension_below_the_proven_range_is_refused_not_approximated():
    """Past what is proven, decline. This is what `NORM_INFLATE` used to be about.
    """
    assert (cf.D_MIN, cf.D_MAX) == (64, 2 ** 20)
    assert cf.dimension_ok(64) and cf.dimension_ok(2 ** 20)
    assert not cf.dimension_ok(63) and not cf.dimension_ok(2 ** 20 + 1)

    twopass.reset()
    Q = torch.randn(16, 32)
    C = torch.randn(24, 32)
    real_matmul = torch.Tensor.__matmul__
    seen = []

    def watch(self, other):
        seen.append(tuple(self.shape))
        return real_matmul(self, other)

    torch.Tensor.__matmul__ = watch
    try:
        upper = twopass.upper_bounds(Q, C, None, None, None, "dot")
    finally:
        torch.Tensor.__matmul__ = real_matmul

    assert bool(torch.isinf(upper).all()) and bool((upper > 0).all()), (
        "d = 32 is outside P6, so there is no bound to evaluate and every row "
        "must be live")
    assert not seen, (
        f"pass one ran at an unproven dimension: {seen}")
    assert twopass.stats()["slices_guard_refused"] == 1

    # And the dimension the run actually uses is inside it, or nothing prunes.
    twopass.reset()
    Q, C = _make(16, 24, DIM, fp16_corpus=True, q_scale=1.0, c_scale=1.0, seed=9)
    assert bool(torch.isfinite(
        twopass.upper_bounds(Q, C, None, None, None, "dot")).all())


def test_the_structural_self_test_gates_pruning_and_reports_why():
    assert cf.self_test() is None
    assert cf.certified() is None

    twopass.reset()
    Q, C = _make(16, 24, DIM, fp16_corpus=True, q_scale=1.0, c_scale=1.0, seed=10)
    assert bool(torch.isfinite(
        twopass.upper_bounds(Q, C, None, None, None, "dot")).all())

    # With the self-test failing, every row goes live rather than pruning on
    # arithmetic nobody checked.
    twopass.reset()
    real = cf._SELF_TEST
    try:
        cf._SELF_TEST = "a structural property does not hold"
        upper = twopass.upper_bounds(Q, C, None, None, None, "dot")
    finally:
        cf._SELF_TEST = real
    assert bool(torch.isinf(upper).all()) and bool((upper > 0).all()), (
        "the bound pruned while its own structural self-test was failing")
    assert twopass.stats()["slices_guard_refused"] == 1


def test_certification_state_is_three_valued_and_per_run():
    """`never attempted` is not the same as `certified clean`.

    A manifest reader has to be able to tell a run that checked its bound and
    passed from one that never checked — otherwise the field is worthless on
    exactly the runs where it matters. And the state is per-RUN, like every
    other verdict in this module, because the next run may be on different
    hardware, a different torch, or a different dimension.
    """
    twopass.reset()
    assert twopass.certified() is None
    assert twopass.stats()["certified"] is None

    twopass.note_certified(None)
    assert twopass.certified() == ""
    assert twopass.stats()["certified"] == ""

    twopass.note_certified("a reason")
    assert twopass.stats()["certified"] == "a reason"

    twopass.reset()
    assert twopass.certified() is None, (
        "certification leaked into the next run, so run 2 would prune on run "
        "1's evidence")


def test_a_scale_that_is_not_the_reciprocal_of_its_own_norm_is_refused():
    """
    P1 is `(1-u)/N_i <= s_i <= (1+u)/N_i`, ELEMENTWISE, against the norm the
    exact pass divides by. An earlier version of this checked only
    `max(s) <= (1+u)/min(N)` — an upper bound on the largest scale — and that
    is not a technicality. It admits `row_scale = 0`, which is what this test
    now exists for:

        q = c, unit norm, col_scale = 1, row_scale = 0

    passes every norm and overflow guard, passes `0 <= 1+u`, and gives
    `A = 0` against an exact cosine score of `1` with `eps ~ 1.2e-3`. At any
    threshold above `eps` the row is pruned with a perfect match sitting in
    the slice. Measured before the fix: 8 of 8 rows wrongly dead.

    `_check_rowmax_magnitude` cannot catch it either — a too-SMALL scale makes
    the row max smaller, not anomalously large — so the elementwise bound is
    the only thing standing between this and lost ground truth.
    """
    torch.manual_seed(4)
    d = DIM
    C = torch.randn(1024, d).half().float()
    Q = torch.randn(256, d)
    cn = C.norm(dim=1)
    qn = Q.norm(dim=1)

    # The honest stored reciprocal, on both axes, must pass — otherwise the
    # probe refuses every real run.
    assert twopass.probe_scales(cn.reciprocal(), cn) is None
    assert twopass.probe_scales(cn.reciprocal(), cn,
                                qn.reciprocal(), qn) is None
    # `None` on an axis means pass one applies no scale there: the `dot` case,
    # not a violation.
    assert twopass.probe_scales(None, cn, None, qn) is None

    # TOO LARGE, on either axis, with the axis named — an operator cannot fix
    # what they cannot locate.
    why = twopass.probe_scales(cn.reciprocal() * 1.001, cn)
    assert why is not None and "column scale" in why, why
    why = twopass.probe_scales(cn.reciprocal(), cn, qn.reciprocal() * 1.001, qn)
    assert why is not None and "query scale" in why, why

    # TOO SMALL — the half the old check was blind to.
    for factor, label in ((0.0, "zero"), (0.5, "half"), (1e-6, "tiny")):
        why = twopass.probe_scales(cn.reciprocal(), cn,
                                   qn.reciprocal() * factor, qn)
        assert why is not None, f"a {label} query scale was accepted"
    why = twopass.probe_scales(cn.reciprocal() * 0.0, cn)
    assert why is not None, "a zero column scale was accepted"

    # A UNIFORM scale at the largest admitted value is also refused: it is a
    # valid bound on the maximum but is not `fl(1/N_i)` for any other row, and
    # P1 is a statement about each entry against its OWN norm.
    uniform = torch.full_like(cn, float((1.0 + cf.U) / float(cn.min())))
    assert twopass.probe_scales(uniform, cn) is not None

    # Non-finite scales are mismatches by definition.
    for v in (float("nan"), float("inf"), -1.0):
        s_bad = cn.reciprocal().clone()
        s_bad[3] = v
        assert twopass.probe_scales(s_bad, cn) is not None, v

    # THE BAND HAS NO SLACK, and that is deliberate. A correctly-rounded
    # `fl(1/N)` sits within `u` of `1/N` — exactly the width P1 admits — so it
    # is already AT the edge. One further ulp in either direction is `~3u` from
    # `1/N` and is genuinely outside the hypothesis, so it is refused:
    for direction in (0.0, float("inf")):
        jittered = torch.nextafter(cn.reciprocal(),
                                   torch.full_like(cn, direction))
        assert twopass.probe_scales(jittered, cn) is not None, direction
    # Which is only safe to be this tight because the real thing passes — the
    # first assertion in this test — and it does because `torch.reciprocal` is
    # IEEE division and therefore correctly rounded. If a future backend
    # returned a reciprocal accurate to 2 ulp instead of 0.5, this probe would
    # refuse every run rather than silently admit a scale `Psi` was not
    # evaluated at, which is the right way round to fail.


def test_a_mismatched_scale_cannot_prune_because_the_guard_is_per_slice():
    """The probe above runs once per execution configuration. THIS is what
    makes the false prune impossible rather than merely detectable: the same
    elementwise test is a guard inside `upper_bounds`, so a bad scale forces
    rows live on the slice that carries it."""
    twopass.reset()
    d = DIM
    v = torch.randn(d)
    v = (v / v.norm()).half().float()
    v = v / v.norm()
    Q = v.repeat(8, 1)
    C = v.repeat(8, 1)
    qn = Q.norm(dim=1)
    cn = C.norm(dim=1)
    cs = cn.clamp_min(1e-12).reciprocal()

    # The exact score is 1.0 for every pair; a threshold of 0.5 must not kill
    # anything.
    upper = twopass.upper_bounds(Q, C, cs, torch.zeros(8), torch.float32,
                                 "cosine", cn=cn)
    exact = (Q @ C.T).div(cn[None, :]).div(qn[:, None]).amax(dim=1)
    assert float(exact[0]) > 0.99
    assert not bool((upper < 0.5).any()), (
        "a row with a perfect match in the slice was pruned on a zero query "
        "scale — the P1 guard is not firing")
    assert bool(torch.isinf(upper).all()), "the refusal must be +inf (live)"

    # And the honest scale still prunes, so the guard has not simply disabled
    # the feature.
    twopass.reset()
    ok = twopass.upper_bounds(Q, C, cs, qn.reciprocal(), torch.float32,
                              "cosine", cn=cn)
    assert bool(torch.isfinite(ok).all())

def test_pruning_is_refused_where_the_accumulator_belongs_to_cublas():
    """The last assumption that was somebody else's promise, made fail-closed.

    `gamma_d` needs pass one to accumulate in float32. That is a FACT for the
    fused Triton kernel (it declares a float32 accumulator) and for the CPU
    path (it widens and multiplies in float32) — both are code in this repo.
    For cuBLAS it is a torch flag we set and cannot read back, and a float16
    split-K reduction would break the bound silently.

    The end-to-end check detects a simulated version of that fault in 0.8-2%
    of rows, which is evidence. The standard here is proof, so the case where
    the accumulator is not ours is declined instead.

    Tested as a predicate rather than end to end because spoofing a CUDA
    device makes torch attempt to use one.
    """
    assert twopass.accumulator_is_ours("cuda:0", used_fused=True), (
        "the fused kernel's accumulator is ours and must be allowed")
    assert not twopass.accumulator_is_ours("cuda:0", used_fused=False), (
        "cuBLAS pass one on CUDA must be refused — nothing verifies that the "
        "float32 accumulation we requested actually happened")
    # CPU has no fused kernel either, and must NOT be caught by this: its
    # accumulation is equally ours. A guard written as `refuse unless fused`
    # would fail here.
    assert twopass.accumulator_is_ours("cpu", used_fused=False)
    assert twopass.accumulator_is_ours(torch.device("cpu"), used_fused=False)


# =============================================================================
# The continuous live-row audit
# =============================================================================

def test_the_audit_checks_the_rows_that_were_scored_and_catches_a_violation():
    """`audit_live_rows` is the only check that runs on real production data
    every slice, so the thing to prove is that it would FIRE.

    A check that never fires is worse than no check: it reports "0 violations"
    for the whole run and reads as evidence.
    """
    twopass.reset()
    n_live, n_cols = 32, 16
    scores = torch.randn(n_live, n_cols)
    idx = torch.arange(n_live)
    top = scores.amax(dim=1)

    # A bound that genuinely dominates: no violations, and the rows are counted.
    upper = top + 1.0
    assert twopass.audit_live_rows(scores, n_live, idx, upper) == 0
    st = twopass.stats()
    assert st["audit_slices"] == 1 and st["audit_rows"] == n_live
    assert st["audit_violations"] == 0

    # A bound one ulp too small on three rows: it must be caught, and the
    # worst shortfall recorded.
    bad_upper = top.clone() + 1.0
    bad_upper[3] = top[3] - 0.5
    bad_upper[11] = top[11] - 0.25
    bad_upper[29] = top[29] - 1e-6
    assert twopass.audit_live_rows(scores, n_live, idx, bad_upper) == 3
    st = twopass.stats()
    assert st["audit_violations"] == 3
    assert abs(st["audit_worst"] - 0.5) < 1e-6


def test_the_audit_reapplies_a_deferred_query_scale():
    """`scale_in_packer` leaves `scores` in RAW units while `upper` is scaled.
    Comparing them directly compares two different quantities — and in the
    unsafe direction whenever the query norms exceed 1, which they always do
    at production dimension."""
    twopass.reset()
    n_live, n_cols = 24, 8
    raw = torch.rand(n_live, n_cols) + 1.0
    idx = torch.arange(n_live)
    row_scale = torch.full((n_live,), 0.01)       # ||q|| = 100
    scaled_top = (raw.amax(dim=1) * row_scale)
    upper = scaled_top + 1e-3                     # correct in SCALED units

    # With the scale applied: no violation.
    assert twopass.audit_live_rows(raw, n_live, idx, upper, row_scale) == 0
    # Without it, every row looks violated — which is what the caller would
    # see if it forgot to pass `row_scale`, and is why the parameter exists.
    twopass.reset()
    assert twopass.audit_live_rows(raw, n_live, idx, upper) == n_live


def test_a_non_finite_score_is_not_counted_as_a_bound_violation():
    """NaN/inf rows are forced live and never pruned — the two-pass's OTHER
    safety mechanism — so the bound makes no claim about them. Counting them
    would make any corpus holding one NaN permanently 'failing'."""
    twopass.reset()
    n_live, n_cols = 16, 8
    scores = torch.randn(n_live, n_cols)
    scores[2] = float("nan")
    scores[5] = float("inf")
    idx = torch.arange(n_live)
    upper = scores.nan_to_num(0.0, 0.0, 0.0).amax(dim=1) + 1.0
    assert twopass.audit_live_rows(scores, n_live, idx, upper) == 0
    assert twopass.stats()["audit_rows"] == n_live - 2


def test_the_audit_can_be_sampled_or_switched_off(monkeypatch):
    twopass.reset()
    monkeypatch.setenv("NOVA_BF_TWOPASS_AUDIT", "0")
    twopass._AUDIT_RATE = None
    assert twopass.audit_rate() == 0, "0 must disable the audit outright"
    monkeypatch.setenv("NOVA_BF_TWOPASS_AUDIT", "16")
    twopass._AUDIT_RATE = None
    assert twopass.audit_rate() == 16
    monkeypatch.setenv("NOVA_BF_TWOPASS_AUDIT", "nonsense")
    twopass._AUDIT_RATE = None
    assert twopass.audit_rate() == 1, "a malformed value must not disable it"
    twopass._AUDIT_RATE = None


# =============================================================================
# The caches — and specifically, that they HIT
# =============================================================================

def test_the_query_side_cache_hits_when_nothing_changed():
    twopass.reset()
    Q = torch.randn(64, DIM)
    rs = Q.norm(dim=1).reciprocal()
    a = twopass.query_side(Q, rs)
    b = twopass.query_side(Q, rs)
    assert a is b, "the query side was rebuilt for an unchanged (Q, row_scale)"


def test_an_equal_but_distinct_row_scale_still_misses_and_that_is_why_it_is_hoisted():
    """ `query_side` CANNOT hit on a new object — identity is all it has,
    and comparing values would cost more than the rebuild. The fix therefore
    has to be upstream: the caller must not manufacture a new tensor per file.
    `test_the_row_scale_is_reused_for_the_whole_run` is that half.
    """
    twopass.reset()
    Q = torch.randn(64, DIM)
    qn = Q.norm(dim=1)
    a = twopass.query_side(Q, qn.reciprocal())
    b = twopass.query_side(Q, qn.reciprocal())      # equal, not identical
    assert a is not b
    assert torch.equal(a["rho_a"] if isinstance(a["rho_a"], torch.Tensor)
                       else torch.as_tensor(a["rho_a"]),
                       torch.as_tensor(b["rho_a"])), (
        "the two entries disagree numerically, which would be a real bug "
        "rather than a missed cache")


def test_the_eps_cache_hits_across_slices_of_one_query_matrix():
    """many slices, one query matrix, one `eps`.

    `sigma_max` moves slice to slice but its power-of-two bucket does not, so
    after the first slice every one of these must be a hit. Asserting a RATE
    rather than `>= 1` is deliberate — one hit would also have passed while the
    cache was 94% useless.
    """
    twopass.reset()
    n_slices = 12
    Q = torch.randn(256, DIM)
    rs = Q.norm(dim=1).reciprocal()          # ONE object, as the run now does
    g = torch.Generator().manual_seed(3)
    for _ in range(n_slices):
        C = torch.randn(128, DIM, generator=g).half().float()
        cn = C.norm(dim=1)
        twopass.upper_bounds(Q, C, cn.clamp_min(1e-12).reciprocal(), rs,
                             torch.float32, "cosine", cn=cn)
    st = twopass.stats()
    total = st["eps_evaluations"] + st["eps_cache_hits"]
    assert total == n_slices, (st, "every slice must consult the eps cache")
    assert st["eps_evaluations"] <= 2, (
        f"eps was evaluated {st['eps_evaluations']} times for {n_slices} "
        f"slices of ONE query matrix; the bucket is stable across these, so "
        f"anything above 1-2 means the cache is being discarded")
    assert st["eps_cache_hits"] / n_slices >= 0.8, st


def test_the_eps_cache_is_dropped_when_the_query_matrix_changes():
    """The other half: it must not hit across DIFFERENT query matrices, or it
    would be serving one matrix's `eps` to another."""
    twopass.reset()
    g = torch.Generator().manual_seed(4)
    C = torch.randn(128, DIM, generator=g).half().float()
    cn = C.norm(dim=1)
    cs = cn.clamp_min(1e-12).reciprocal()
    for _ in range(4):
        Q = torch.randn(256, DIM, generator=g)      # a NEW matrix each time
        twopass.upper_bounds(Q, C, cs, Q.norm(dim=1).reciprocal(),
                             torch.float32, "cosine", cn=cn)
    st = twopass.stats()
    assert st["eps_cache_hits"] == 0, (
        "eps computed for one query matrix was reused for another")
    assert st["eps_evaluations"] == 4


def test_a_slice_whose_sigma_max_changes_bucket_re_evaluates():
    """The cache is keyed on the BUCKET, so a genuinely different scale must
    miss — otherwise it would serve an `eps` evaluated at the wrong
    `sigma_max`, which is the unsafe direction."""
    twopass.reset()
    Q = torch.randn(64, DIM)
    rs = Q.norm(dim=1).reciprocal()
    g = torch.Generator().manual_seed(5)
    C = torch.randn(128, DIM, generator=g).half().float()
    cn = C.norm(dim=1)
    twopass.upper_bounds(Q, C, cn.clamp_min(1e-12).reciprocal(), rs,
                         torch.float32, "cosine", cn=cn)
    # A corpus 1000x shorter: `sigma_max` jumps ~10 octaves, a different bucket.
    C2 = (C * 1e-3)
    cn2 = C2.norm(dim=1)
    assert cf.ceil_pow2(float(cn2.clamp_min(1e-12).reciprocal().max())) != \
        cf.ceil_pow2(float(cn.clamp_min(1e-12).reciprocal().max()))
    twopass.upper_bounds(Q, C2, cn2.clamp_min(1e-12).reciprocal(), rs,
                         torch.float32, "cosine", cn=cn2)
    st = twopass.stats()
    assert st["eps_evaluations"] == 2 and st["eps_cache_hits"] == 0, st


def test_bound_refuses_the_two_argument_combinations_that_understate_eps():
    """Both refusals in `bound()` exist because the alternative is a SILENT
    under-bound, and neither was tested — reverting either passed the suite.

    They are the same class of trap as `metric` having no default, which IS
    tested, and were simply missed when that one was added.
    """
    twopass.reset()
    Q = torch.randn(32, DIM)
    qs_scaled = twopass.query_side(Q, Q.norm(dim=1).reciprocal())
    qs_unscaled = twopass.query_side(Q, None)

    # cosine with no row scale: `eps_cos` is sized for a scaled score of
    # magnitude at most `Lambda(d)`, and an unscaled row max is ~||q|| times
    # larger. Measured before the refusal existed: 17 of 32 rows violated
    # (SAFE) at ||q|| = 1e3.
    with pytest.raises(ValueError, match="row scale"):
        twopass.bound(qs_unscaled, DIM, 1.0, False, "cosine")

    # dot with no `cn_max`: under Theorem 1' `E` is PROPORTIONAL to
    # `max_S N_c`, so a default of 1.0 understates `eps` by exactly that
    # factor — ~28x at production norms.
    with pytest.raises(ValueError, match="cn_max"):
        twopass.bound(qs_unscaled, DIM, 1.0, False, "dot")
    # and with it supplied, it works and scales with the norm.
    small = twopass.bound(qs_unscaled, DIM, 1.0, False, "dot", cn_max=1.0)
    big = twopass.bound(qs_unscaled, DIM, 1.0, False, "dot", cn_max=100.0)
    assert float(np.max(big)) > 50.0 * float(np.max(small)), (
        "the dot bound is not scaling with max_S N_c, so `cn_max` is being "
        "ignored and the refusal above is guarding nothing")
    assert qs_scaled is not qs_unscaled


def test_a_row_max_above_what_theorem_1_admits_is_recorded_and_warned(caplog):
    """`_check_rowmax_magnitude` is described as "a necessary condition of
    Theorem 1's hypotheses" and could be deleted with no test failing.

    It is the only thing that notices a scale too LARGE on the corpus axis
    after the guards have passed — `probe_scales` catches it at certification,
    this catches it on the slice.
    """
    twopass.reset()
    Q = torch.randn(64, DIM)
    C = torch.randn(128, DIM).half().float()
    cn = C.norm(dim=1)
    rs = Q.norm(dim=1).reciprocal()
    # An honest run leaves the ratio well under 1.
    twopass.upper_bounds(Q, C, cn.clamp_min(1e-12).reciprocal(), rs,
                         torch.float32, "cosine", cn=cn)
    healthy = twopass.stats()["rowmax_worst_ratio"]
    assert 0.0 < healthy < 1.0, healthy

    # Now feed the ROW MAX an inflated scale directly. `upper_bounds` would
    # refuse an out-of-band `col_scale` at the P1 guard, so the magnitude check
    # is exercised through `_check_rowmax_magnitude` itself — which is exactly
    # the layer that has to notice when P1 has been satisfied and the arithmetic
    # is still not what the theorem describes.
    twopass.reset()
    with caplog.at_level("WARNING"):
        twopass._check_rowmax_magnitude(
            torch.full((64,), 5.0), DIM, cf.FP16, cf.EXACT, False, "cosine")
    assert twopass.stats()["rowmax_worst_ratio"] > 1.0, (
        "a row max far above what Theorem 1 admits was not recorded")
    assert any("exceeds" in r.message for r in caplog.records), (
        "nothing warned about a row max outside the theorem's range")


def test_p3_refuses_every_way_the_exact_pass_can_stop_being_ieee_binary32(monkeypatch):
    """P3 — (R2) and the `gamma_d` term are stated for IEEE binary32
    round-to-nearest. If the "exact" pass is TF32, its inputs carry an extra
    ~2**-11 conversion error the bound does not model.

    `_verify_shape` cannot catch this: it compares a padded GEMM to a
    full-height one, and both would run TF32, agree with each other perfectly,
    and differ together from the binary32 arithmetic the theorem defines.

    Four ways it can go wrong, all checked. The env-var pair is the important
    one: `CUBLAS_EMULATE_SINGLE_PRECISION` is what ENABLES emulation, while
    `CUBLAS_EMULATION_STRATEGY` only selects which one — an earlier version
    checked only the latter, so `CUBLAS_EMULATE_SINGLE_PRECISION=1` with no
    strategy set walked straight past. And bf16x9 emulation is accurate to
    roughly binary32, so the empirical `1 + 2**-13` discriminator would NOT
    catch it: for emulation the flags are the only defence.
    """
    # CPU fp32 is IEEE binary32 by construction, so nothing to check.
    assert twopass.exact_math_mode_flags("cpu") is None

    monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "1")
    why = twopass.exact_math_mode_flags("cuda:0")
    assert why is not None and "EMULATE_SINGLE_PRECISION" in why, why
    monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "0")
    assert twopass.exact_math_mode_flags("cuda:0") is None, (
        "explicitly disabled emulation must not be refused")
    monkeypatch.delenv("CUBLAS_EMULATE_SINGLE_PRECISION")

    monkeypatch.setenv("CUBLAS_EMULATION_STRATEGY", "performant")
    why = twopass.exact_math_mode_flags("cuda:0")
    assert why is not None and "EMULATION_STRATEGY" in why, why
    monkeypatch.delenv("CUBLAS_EMULATION_STRATEGY")

    # And the torch flags. A fake backend object stands in for a CUDA build.
    class _M:
        fp32_precision = "tf32"
        allow_tf32 = True

    monkeypatch.setattr(torch.backends.cuda, "matmul", _M)
    why = twopass.exact_math_mode_flags("cuda:0")
    assert why is not None and "fp32_precision" in why, why
    _M.fp32_precision = "ieee"
    why = twopass.exact_math_mode_flags("cuda:0")
    assert why is not None and "allow_tf32" in why, why
    _M.allow_tf32 = False
    assert twopass.exact_math_mode_flags("cuda:0") is None

    # An UNREADABLE state is refused, not assumed benign — torch raises on a
    # mixed legacy/new API state, and "cannot read" is not "off".
    class _Raises:
        @property
        def fp32_precision(self):
            raise RuntimeError("mixed legacy and new TF32 APIs")

    monkeypatch.setattr(torch.backends.cuda, "matmul", _Raises())
    why = twopass.exact_math_mode_flags("cuda:0")
    assert why is not None and "unknown" in why, why


def test_the_guards_get_the_RAW_norms_not_the_clamped_ones():
    """`col_norms()` clamps at 1e-12; `col_norms_raw()` does not. The guards
    must see the raw ones, and nothing tested that.

    The clamp sits ABOVE `NORM_MIN = 2**-40 = 9.09e-13`, so a zero-norm corpus
    row arrives at the clamped vector as 1e-12 and passes `norm_guard` — the
    guard whose whole job is to refuse it. Feeding the guards the clamped norms
    is invisible to the whole suite: a mutation doing exactly that passed 177
    tests.

    It is currently harmless only because `prunability_cut` refuses any slice
    with `min N_c < 1e-3`, which catches the 1e-12 as a side effect. That
    threshold is a PERFORMANCE choice (`Psi` grows too large to prune usefully
    below it), not a safety one, and lowering it — a perfectly reasonable thing
    to want — would silently open the hole. So the distinction is pinned here
    rather than left resting on an unrelated constant.
    """
    C = torch.randn(64, DIM)
    C[7] = 0.0
    raw = C.norm(dim=1)
    clamped = raw.clamp_min(1e-12)

    # The two disagree exactly where it matters, and in the unsafe direction.
    assert bool(twopass.norm_guard(raw).any()), (
        "the raw norms do not trip the guard on a zero row")
    assert not bool(twopass.norm_guard(clamped).any()), (
        "the clamp no longer hides a zero row from the guard — if this starts "
        "failing the clamp or NORM_MIN moved, and this test's premise with it")
    assert 1e-12 > cf.NORM_MIN, (
        f"the clamp {1e-12:.1e} is no longer above NORM_MIN {cf.NORM_MIN:.3e}")

    # And the slice really does hand the raw ones down.
    from nova_bf import compute as C_mod
    import numpy as np
    arr = np.ascontiguousarray(
        np.random.default_rng(4).standard_normal((64, DIM)).astype(np.float32))
    arr[7] = 0.0
    sl = C_mod.DenseCorpusBatch(arr).transfer(0, 64, "cpu")
    assert float(sl.col_norms_raw()[7]) == 0.0
    # `== float(np.float32(1e-12))`, not `== 1e-12`: the clamp is applied in
    # float32 and 1e-12 is not representable there (it lands on
    # 9.99999996e-13). Comparing against the binary64 literal fails for a
    # reason that has nothing to do with what this test is about.
    assert float(sl.col_norms()[7]) == float(np.float32(1e-12))
    assert bool(twopass.norm_guard(sl.col_norms_raw()).any())

    # End to end: a zero-norm row must be refused, and `slices_guard_refused`
    # must be the thing that says so — not the prunability cut alone.
    twopass.reset()
    Q = torch.randn(16, DIM)
    cn = sl.Cb.norm(dim=1)
    upper = twopass.upper_bounds(
        Q, sl.Cb, cn.clamp_min(1e-12).reciprocal(), Q.norm(dim=1).reciprocal(),
        torch.float32, "cosine", cn=cn)
    assert bool(torch.isinf(upper).all()), "a zero-norm corpus row was not refused"
    assert twopass.stats()["slices_guard_refused"] == 1


# --------------------------------------------------------------------------
# THE METRIC/SCALE CONTRACT, both axes.
# --------------------------------------------------------------------------

def _tiny(metric, col_scale, row_scale, norm=1.0):
    import torch
    from nova_bf import twopass
    torch.manual_seed(0)
    d, n, m = 128, 4, 32
    Q = torch.randn(n, d)
    C = torch.randn(m, d)
    C = C / C.norm(dim=1, keepdim=True) * norm
    cn = C.norm(dim=1)
    cs = cn.reciprocal() if col_scale else None
    rs = Q.norm(dim=1).reciprocal() if row_scale else None
    return twopass.upper_bounds(Q, C, cs, rs, torch.float32, metric=metric,
                                cn=cn, corpus_exact_fp16=False)


def test_dot_refuses_a_column_scale():
    """Theorem 1' is stated at sigma = 1 and `eps_dot` never reads sigma_max.

    Pass one's epilogue applies whatever `col_scale` it is given, so handing
    dot a `1/||c||` scales the row max down while `eps` stays sized for the raw
    dot magnitude. Measured before this raised, at ||c|| = 100: true tops
    [261, 150, 186, 330] against upper [4.1, 2.7, 3.3, 5.0] — the bound
    violated on 4 of 4 rows, every one of which a threshold of 3.0 would have
    pruned.
    """
    import pytest
    with pytest.raises(ValueError, match="applies no corpus scaling"):
        _tiny("dot", col_scale=True, row_scale=False, norm=100.0)


def test_cosine_refuses_a_missing_column_scale():
    """Without it the row max is a raw Gram entry, not a cosine.

    `eps_cos` is then evaluated at the default `sigma_max = 1.0` while pass one
    never divided by ||c||. Measured before this raised, at ||c|| = 0.01: true
    tops [0.153, 0.221, 0.209, 0.184] against upper [0.0025, 0.0032, 0.0031,
    0.0029] — violated on 4 of 4 rows.
    """
    import pytest
    with pytest.raises(ValueError, match="needs the column scale"):
        _tiny("cosine", col_scale=False, row_scale=True, norm=0.01)


def test_the_correctly_paired_calls_still_work():
    """The guard must refuse the mismatch and nothing else."""
    import torch
    assert torch.isfinite(_tiny("cosine", col_scale=True, row_scale=True)).all()
    assert torch.isfinite(_tiny("dot", col_scale=False, row_scale=False)).all()


def test_a_zero_scale_is_inadmissible_not_merely_non_negative():
    """`> 0`, not `>= 0`, for every SCALE the bound is evaluated at.

    A norm that has passed `norm_range_ok` is at least 2^-40, so a reciprocal
    of exactly zero is never a valid upper bound — and it is the one value that
    breaks the monotonicity the bound's safety rests on. With `cn_ub = 0` the
    whole `Qn * Cn * bracket` term of `E` vanishes: `eps_dot` returned 8.40e-7
    where the correct value is 1.21e-3, 1440x too small.

    TOLERANCES are the opposite and must stay `>= 0`: `u_c == 0` is exactly
    what an fp16-stored corpus means, and requiring positivity there refuses
    every ordinary production call.
    """
    import numpy as np
    from nova_bf import closed_form as cf

    inf = float("inf")
    assert cf.eps_dot(768, cf.U16, cf.U16, 1.0, 0.0,
                      eta_q=cf.ETA16, eta_c=cf.ETA16) == inf
    assert cf.eps_cos(768, cf.U16, 0.0, eta_q=cf.ETA16,
                      rho_q=0.0, sigma_max=1.0) == inf
    assert cf.eps_cos(768, cf.U16, 0.0, eta_q=cf.ETA16,
                      rho_q=1.0, sigma_max=0.0) == inf
    # per-row neutralisation, not a wholesale refusal of the slice
    per_row = cf.eps_dot(768, cf.U16, cf.U16, np.array([0.0, 1.0]), 1.0,
                         eta_q=cf.ETA16, eta_c=cf.ETA16)
    assert per_row[0] == inf and np.isfinite(per_row[1])
    # `lam_t` is a scale too — the smallest normal of the first-pass format,
    # which no format has as zero (`EXACT` carries binary16's 2^-14). It sits
    # beside the tolerances in every signature, which is why it was grouped
    # with them at first.
    assert cf.eps_cos(768, cf.U16, 0.0, eta_q=cf.ETA16, rho_q=1.0,
                      sigma_max=1.0, lam_t=0.0) == inf
    assert cf.EXACT.lam_t > 0.0 and cf.FP16.lam_t > 0.0 and cf.BF16.lam_t > 0.0
    # and the fp16-corpus call, whose u_c and eta_c ARE zero, is untouched
    assert np.isfinite(cf.eps_cos(768, cf.U16, 0.0, eta_q=cf.ETA16,
                                  rho_q=1.0, sigma_max=1.0))
    assert np.isfinite(cf.eps_dot(768, cf.U16, cf.U16, 1.0, 1.0,
                                  eta_q=cf.ETA16, eta_c=cf.ETA16))


# --------------------------------------------------------------------------
# THE QUERY-SIDE CACHE'S INVALIDATION
# --------------------------------------------------------------------------

def test_an_in_place_write_to_Q_invalidates_the_cached_query_side():
    """A cached `Qh`/`qn`/`rho_a` describing the OLD contents is a wrong bound.

    The cache is keyed on `id(Q)` plus `_version`, and it holds `Q` strongly so
    the id cannot be recycled underneath it. This pins the `_version` half for
    every shape of torch-dispatched write: the base tensor, a view of it, and a
    non-contiguous transposed view.
    """
    import torch
    from nova_bf import twopass

    twopass.reset()
    Q = torch.randn(8, 128)
    rs = Q.norm(dim=1).reciprocal()
    first = twopass.query_side(Q, rs)
    assert twopass.query_side(Q, rs) is first, "a clean re-read must hit"

    Q.mul_(3.0)                                   # base tensor, in place
    after = twopass.query_side(Q, rs)
    assert after is not first
    assert not torch.equal(after["qn"], first["qn"])

    Q[0].mul_(5.0)                                # through a view
    assert twopass.query_side(Q, rs) is not after

    twopass.reset()
    Qt = torch.randn(128, 8).t()                  # non-contiguous view
    rst = Qt.norm(dim=1).reciprocal()
    g = twopass.query_side(Qt, rst)
    Qt.mul_(2.0)
    assert twopass.query_side(Qt, rst) is not g


def test_an_in_place_write_to_row_scale_invalidates_it_too():
    """P1's failure shape, and the one the cache is least likely to be
    watched for: `probe_scales` reads the live tensor while `bound()` reads
    the cached `rho_a`, so a stale `rho_a` is a scale mismatch nothing sees.
    """
    import torch
    from nova_bf import twopass

    twopass.reset()
    Q = torch.randn(8, 128)
    rs = Q.norm(dim=1).reciprocal()
    first = twopass.query_side(Q, rs)
    rs.mul_(0.5)
    after = twopass.query_side(Q, rs)
    assert after is not first
    assert after["rho_a"][0] != first["rho_a"][0]


def test_the_cache_holds_Q_strongly_so_its_id_cannot_be_recycled():
    """`id()` alone would be unsound: free the tensor and the next allocation
    can land on the same address. The entry keeps `Q` alive, and the lookup
    re-checks `is`, so a collision cannot be reached.
    """
    import torch
    from nova_bf import twopass

    twopass.reset()
    Q = torch.randn(8, 128)
    got = twopass.query_side(Q, None)
    assert got["Q"] is Q
    key = id(Q)
    del Q
    # The entry still pins the object, so the id remains live and owned by it.
    assert twopass._QCACHE[key]["Q"] is not None
    twopass.reset()


# --------------------------------------------------------------------------
# `probe_accumulation` and the three ways a probe can fail to be a probe.
# --------------------------------------------------------------------------

def test_a_nan_from_the_probed_kernel_refuses_rather_than_passes():
    """NaN is a third state, and the dangerous one.

    `inf` propagates into the relative error and trips the Volta refusal, so
    an overflowing kernel is caught. NaN does not: `NaN > worst` is FALSE, so
    the running maximum silently declines to record it, `worst` keeps its
    initial 0.0, and the probe returns None while logging "worst relative
    error 0.000e+00" — a PERFECT accumulator, from a kernel that produced no
    number at all.

    Same shape as this function's own "DECLINED IS NOT PASSED" case, one step
    later: there the kernel never ran, here it ran and said nothing. (R5) is
    the one hardware premise the bound rests on and this probe is its only
    in-process backstop.
    """
    import torch
    from nova_bf import twopass

    twopass.reset()
    why = twopass.probe_accumulation(
        768, "cpu",
        rowmax_fn=lambda Qh, Ch, cs: torch.full((Qh.shape[0],), float("nan")))
    assert isinstance(why, str), (
        f"an all-NaN kernel passed the accumulation probe: {why!r}")
    assert "non-finite" in why
    assert twopass.stats()["probe_accum_worst_rel"] is None, (
        "a refused probe must leave the statistic UNMEASURED (None), not 0.0 "
        "— 0.0 reads as a perfect accumulator")


def test_a_partially_nan_kernel_is_also_refused():
    """One NaN row is enough. Every row of a construction holds the same
    operands, so a kernel that is exact on some rows and NaN on others has
    still failed to answer for the ones it skipped — and a plain `max` over
    the finite rows would report the exact ones as the whole story.
    """
    import torch
    from nova_bf import twopass

    def half_nan(Qh, Ch, cs):
        out = torch.zeros(Qh.shape[0], dtype=torch.float32)
        out[0] = float("nan")
        return out

    twopass.reset()
    why = twopass.probe_accumulation(768, "cpu", rowmax_fn=half_nan)
    assert isinstance(why, str) and "1 of" in why, why


def test_an_infinite_kernel_is_still_refused():
    """`inf` was already caught, via the Volta comparison. It must stay
    caught now that the finiteness check runs first — the verdict is the
    same, only the message changes."""
    import torch
    from nova_bf import twopass

    twopass.reset()
    why = twopass.probe_accumulation(
        768, "cpu",
        rowmax_fn=lambda Qh, Ch, cs: torch.full((Qh.shape[0],), float("inf")))
    assert isinstance(why, str), "an infinite kernel must be refused"


def test_an_accurate_kernel_still_passes_the_probe():
    """The three refusals must not swallow the ordinary case: a kernel that
    reproduces the float64 reference has to certify."""
    import numpy as np
    import torch
    from nova_bf import twopass

    def exact(Qh, Ch, cs):
        g = (Qh.double() @ Ch.double().T) * cs.double()[None, :]
        return g.amax(dim=1).float()

    twopass.reset()
    assert twopass.probe_accumulation(768, "cpu", rowmax_fn=exact) is None
    assert twopass.stats()["probe_accum_worst_rel"] is not None, (
        "a probe that ran must record a measurement")


# --------------------------------------------------------------------------
# `exact_math_mode_flags` and the UNDOCUMENTED `'none'` precision state.
# --------------------------------------------------------------------------

def test_the_accepted_fp32_precision_values_match_this_torch():
    """`'none'` is not a documented value of `matmul.fp32_precision`.

    PyTorch's docs list `'ieee'`, `'tf32'` and `'bfx9'`. `exact_math_mode_flags`
    nonetheless accepts `'none'`, on the stated grounds that it is "the unset
    default, which behaves as IEEE". That is an assumption about a specific
    torch build, not a documented contract, so it is pinned here rather than
    left in a comment.

    Measured on torch 2.12.1+cu130:

        default                -> 'none'      <- the reason accepting it is
                                                REQUIRED, not merely allowed:
                                                refusing it would disable the
                                                two-pass on every machine that
                                                has not touched the setting
        allow_tf32 = True      -> 'tf32'
        allow_tf32 = False     -> 'ieee'
        setting 'bfx9'         -> RuntimeError('Unknown precision: bfx9')

    Note the last one: the documented value list does not match this build
    either, which is the point — this is a moving API and the code's guard
    has to be checked against the torch actually installed.

    IF THIS TEST FAILS after a torch upgrade, do not widen the accepted set to
    make it pass. Work out what the new state MEANS first: accepting a mode
    that is not binary32 round-to-nearest silently invalidates `gamma_d` and
    the (R2) term, which is an under-bound.
    """
    import torch
    from nova_bf import twopass

    m = torch.backends.cuda.matmul
    try:
        default = m.fp32_precision
    except Exception:                              # pragma: no cover
        pytest.skip("this torch cannot report matmul.fp32_precision")

    accepted = ("ieee", "none")
    assert default in accepted, (
        f"this torch's DEFAULT matmul.fp32_precision is {default!r}, which "
        f"`exact_math_mode_flags` refuses. The two-pass would be off on every "
        f"untouched machine. Establish what {default!r} means before adding "
        f"it to the accepted set.")

    prev_tf32 = m.allow_tf32
    prev_prec = default
    try:
        m.allow_tf32 = True
        assert m.fp32_precision == "tf32", (
            f"the legacy and modern APIs are no longer linked: allow_tf32=True "
            f"reports {m.fp32_precision!r}, not 'tf32'. The flag check reads "
            f"one and the bound depends on the other.")
        assert m.fp32_precision not in accepted, (
            "TF32 must be refused by the accepted set")

        m.allow_tf32 = False
        assert m.fp32_precision == "ieee"
        assert m.fp32_precision in accepted
    finally:
        m.fp32_precision = prev_prec
        m.allow_tf32 = prev_tf32


def test_a_non_binary32_precision_mode_refuses_the_exact_path():
    """Whatever the mode is called, anything that is not full binary32 has to
    refuse: `gamma_d` and the (R2) term are stated for IEEE round-to-nearest,
    and a narrower input format adds a conversion error the bound does not
    carry.

    Driven through `exact_math_mode_flags` on a CUDA device string, since it
    returns None immediately for CPU — there is no cuBLAS mode to check.
    """
    import torch
    from nova_bf import twopass

    m = torch.backends.cuda.matmul
    prev_tf32, prev_prec = m.allow_tf32, m.fp32_precision
    try:
        m.allow_tf32 = False
        m.fp32_precision = "ieee"
        assert twopass.exact_math_mode_flags("cuda:0") is None, (
            "control: plain IEEE must be accepted")

        m.fp32_precision = "tf32"
        why = twopass.exact_math_mode_flags("cuda:0")
        assert isinstance(why, str) and "fp32_precision" in why, why
    finally:
        m.fp32_precision = prev_prec
        m.allow_tf32 = prev_tf32


def test_the_cpu_path_has_no_cublas_mode_to_check():
    """`exact_math_mode_flags` is about cuBLAS. On CPU there is nothing to
    read, and refusing there would take out the opt-in test path for a
    setting that does not apply to it."""
    from nova_bf import twopass

    assert twopass.exact_math_mode_flags("cpu") is None


def test_the_tf32_override_env_var_refuses_the_exact_path(monkeypatch):
    """`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` makes torch build its context with
    `float32_matmul_precision = HIGH` (TF32) instead of HIGHEST.

    Measured on torch 2.12.1+cu130 this IS mirrored into the flags —
    `fp32_precision` reads `'tf32'`, which the precision check already
    refuses — so it is not a live hole. It is checked explicitly anyway,
    because relying on an environment variable being faithfully reflected in a
    readable flag is an assumption about one build, and the flag is the thing
    this function is trying not to trust. `CUBLAS_EMULATE_SINGLE_PRECISION`
    and `CUBLAS_EMULATION_STRATEGY` are checked for the same reason.
    """
    from nova_bf import twopass

    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", raising=False)
    assert twopass.exact_math_mode_flags("cuda:0") is None, "control"

    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    why = twopass.exact_math_mode_flags("cuda:0")
    assert isinstance(why, str) and "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in why

    # An explicit off value is not a reason to refuse.
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")
    assert twopass.exact_math_mode_flags("cuda:0") is None

    # And it says nothing about the CPU path, which uses no cuBLAS.
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    assert twopass.exact_math_mode_flags("cpu") is None


def test_the_arithmetic_probe_refuses_rather_than_raising(monkeypatch):
    """A certification probe must not be the thing that kills the run.

    The measurement allocates two 64x64 float32 tensors and multiplies them.
    Trivial — but still a CUDA allocation and a library call, and it was
    unguarded, so an OOM or driver fault propagated out of
    `certify_closed_form` -> `_twopass_prepare` -> `run_compute`, ending a
    multi-hour rank and losing its partial (which then makes the merge refuse
    the whole directory). Refusing is both the safe verdict and the cheap one.
    """
    import torch
    from nova_bf import twopass

    real = torch.zeros

    def boom(*a, **kw):
        if kw.get("dtype") is torch.float32 and a and a[0] == 64:
            raise RuntimeError("CUDA error: out of memory")
        return real(*a, **kw)

    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", raising=False)
    monkeypatch.setattr(torch, "zeros", boom)
    why = twopass.probe_exact_math_mode("cuda:0")
    assert isinstance(why, str), (
        f"an allocation failure in the probe must REFUSE, not raise; got {why!r}")
    assert "did not complete" in why
