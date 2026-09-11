"""Tests for `nova_bf.closed_form` — the closed-form, data-free error bound.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

from nova_bf import closed_form as cf

# --- the oracle ---------------------------------------------------------------

def _find_oracle():
    """Locate the document's companion script.

    Searched rather than hard-coded, because this file is run from two
    layouts: the repo (where it sits three levels under the root) and a GPU box
    where only `python/nova-bf` and `docs/brute-force/tc-probe` are mounted, at
    paths that have nothing to do with each other. `NOVA_BF_CF_ORACLE` wins if
    set, so a caller can always say where it is.
    """
    import os

    env = os.environ.get("NOVA_BF_CF_ORACLE")
    if env:
        return pathlib.Path(env)
    name = "two-pass-closed-form-check.py"
    here = pathlib.Path(__file__).resolve()
    for base in here.parents:
        for cand in (base / "docs" / "brute-force" / name,
                     base / "docs" / "brute-force" / "tc-probe" / name,
                     base / name):
            if cand.exists():
                return cand
    return here.parents[3] / "docs" / "brute-force" / name


_ORACLE_PATH = _find_oracle()


def _load_oracle():
    """Load the companion script, or return `None` if it is not present.

    A module-level `pytest.skip` used to live here, and it took the WHOLE FILE
    with it — including the published-table, guard, structural and empirical
    tests, none of which need the oracle at all. On a GPU box that mounts
    `python/nova-bf` but not `docs/`, that silently reduced 262 tests to zero
    while still reporting success. Only the differential tests depend on this,
    so only they are skipped.
    """
    if not _ORACLE_PATH.exists():         # pragma: no cover - environment-dependent
        return None
    spec = importlib.util.spec_from_file_location("_cf_oracle", _ORACLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cf_oracle"] = mod
    spec.loader.exec_module(mod)
    return mod


ORACLE = _load_oracle()
needs_oracle = pytest.mark.skipif(
    ORACLE is None,
    reason=f"the document's companion script is not at {_ORACLE_PATH}; set "
           f"NOVA_BF_CF_ORACLE to point at it")


def test_the_oracle_is_present():
    """A visible FAILURE rather than a silent skip when the differential layer
    is not running. Skips are easy to miss in a green run; this is not."""
    assert ORACLE is not None, (
        f"the document's companion script was not found at {_ORACLE_PATH}, so "
        f"every differential test in this file is being skipped. Set "
        f"NOVA_BF_CF_ORACLE, or mount docs/brute-force alongside the package")

# The four storage cases of the Sec.0.1 table, as
# (label, u_q, u_c, eta_q, eta_c, lambda_t).
CASES = [
    ("fp32 q, fp32 c", cf.U16, cf.U16, cf.ETA16, cf.ETA16, cf.LAMBDA16),
    ("fp32 q, fp16-stored c", cf.U16, 0.0, cf.ETA16, 0.0, cf.LAMBDA16),
    ("fp16-exact q and c", 0.0, 0.0, 0.0, 0.0, cf.LAMBDA16),
    ("bf16 first pass", cf.UBF, cf.UBF, cf.ETA_BF, cf.ETA_BF, cf.LAMBDA32),
]
DIMS = [64, 128, 384, 768, 1024, 1536, 3072, 4096, 8192, 65536, 2 ** 20]
C_HWS = [4.0, 1.375, 1.0]
SCALES = [(1.0, 1.0), (3.7, 11.9), (1e2, 1e3), (1e-3, 1e-2), (1e5, 1.0)]


# =============================================================================
# 1. Differential against the document's own script
# =============================================================================

@needs_oracle
@needs_oracle
def test_the_format_constants_are_the_documents_own():
    for name in ("U", "U16", "UBF", "ETA16", "LAMBDA16", "LAMBDA32", "FLOOR_COS"):
        assert getattr(cf, name) == getattr(ORACLE, name), name


@pytest.mark.parametrize("d", DIMS)
@needs_oracle
@needs_oracle
def test_kappa_lambda_theta_match_the_oracle_bit_for_bit(d):
    # `kappa` is the one the shipped `provable_norm_inflation` got subtly wrong:
    # it omitted the `(1 + d 2**-70)` fold factor, which is what makes kappa
    # valid down to the 2**-40 guard rather than only to the old 2**-63.
    assert cf.kappa(d) == ORACLE.kappa_norm(d)
    assert cf.Lambda(d) == ORACLE.Lambda(d)
    for c in C_HWS:
        assert cf.kappa_acc(d, c) == ORACLE.kappa_acc(d, c)
        assert cf.theta(d, c) == ORACLE.theta(d, c)


@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("half_out", [False, True])
@needs_oracle
@needs_oracle
def test_eps_matches_the_oracle_bit_for_bit(d, case, half_out):
    _, u_q, u_c, eta_q, eta_c, lam_t = case
    for c_hw in C_HWS:
        assert cf.bracket(d, u_q, u_c, c_hw, half_out) == ORACLE.bracket(
            d, u_q, u_c, c_hw, half_out)
        assert cf.C_const(d, u_q, u_c, c_hw, half_out) == ORACLE.C_const(
            d, u_q, u_c, c_hw, half_out)
        for rho, sig in SCALES:
            assert float(cf.Psi(d, u_q, u_c, eta_q, eta_c, rho, sig, c_hw,
                                half_out, lam_t)) == ORACLE.Psi(
                d, u_q, u_c, eta_q, eta_c, rho, sig, c_hw, half_out, lam_t)
            assert float(cf.eps_cos(d, u_q, u_c, c_hw, half_out, eta_q, eta_c,
                                    rho, sig, lam_t)) == ORACLE.eps_cos(
                d, u_q, u_c, c_hw, half_out, eta_q, eta_c, rho, sig, lam_t)


def test_the_vectorised_psi_agrees_with_the_scalar_one_elementwise():
    # `Psi` is evaluated per query, so the production path hands it an array.
    # Broadcasting must not reassociate anything: same operations, same order,
    # same results, or Lemma 5's weight count stops describing what ran.
    rho = np.array([1.0, 3.7, 1e2, 1e-3, 1e5], dtype=np.float64)
    vec = cf.eps_cos(768, cf.U16, 0.0, eta_q=cf.ETA16, rho_q=rho, sigma_max=11.9)
    for i, r in enumerate(rho):
        one = cf.eps_cos(768, cf.U16, 0.0, eta_q=cf.ETA16, rho_q=float(r),
                         sigma_max=11.9)
        assert float(vec[i]) == float(one)


DOT_NORMS = [(1.0, 1.0), (27.7, 27.7), (1e-2, 1e-3), (1e4, 1.0), (1.0, 1e4)]


@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("case", CASES[:3], ids=[c[0] for c in CASES[:3]])
@pytest.mark.parametrize("half_out", [False, True])
@needs_oracle
@needs_oracle
def test_eps_dot_matches_the_oracle_bit_for_bit(d, case, half_out):
    _, u_q, u_c, eta_q, eta_c, lam_t = case
    for c_hw in C_HWS:
        for nq, nc in DOT_NORMS:
            if nq * nc > 2.0 ** 126:
                continue
            got = float(cf.eps_dot(d, u_q, u_c, nq, nc, c_hw, half_out,
                                   eta_q, eta_c, lam_t))
            want = ORACLE.eps_dot(d, u_q, u_c, nq, nc, c_hw, half_out,
                                  eta_q, eta_c, lam_t)
            assert got == want, (d, c_hw, half_out, nq, nc, got, want)


# Hard-coded, so the module and the oracle cannot drift TOGETHER. These are not
# independent of the implementation on their own — the differential test above
# is what establishes agreement with the document's transcription, and these
# freeze the agreed value so a later edit to both must be deliberate. The
# structural test below is the independent check: `E` is proportional to
# `Qn*Cn`, so scaling both norms by 10 must scale `eps` by ~100, and that is
# the property the historical bug destroyed.
PUBLISHED_DOT = {
    # (d, N_q, max N_c): eps, for fp32 q + fp32 c (u_q = u_c = 2**-11), fused
    (768, 1.0, 1.0): 1.208171e-03,
    (768, 27.7, 27.7): 9.257739e-01,
    (768, 10.0, 10.0): 1.206658e-01,
    (4096, 1.0, 1.0): 2.204517e-03,
}


@pytest.mark.parametrize("key", sorted(PUBLISHED_DOT))
def test_the_dot_bound_reproduces_its_frozen_values(key):
    d, nq, nc = key
    got = float(cf.eps_dot(d, cf.U16, cf.U16, nq, nc, eta_q=cf.ETA16,
                           eta_c=cf.ETA16))
    want = PUBLISHED_DOT[key]
    assert abs(got - want) / want < 1e-5, (key, got, want)


def test_the_dot_bound_scales_with_the_norm_product():
    """
    Theorem 1' has `E` proportional to `Qn*Cn = kappa^2 N_q max N_c`. If the
    absolute terms lose their norm factors — which is what borrowing Theorem
    1's `Psi` does — `eps` stops tracking the product and the shortfall grows
    with the norms. A golden table alone would not catch that at other norms;
    this does.
    """
    base = float(cf.eps_dot(768, cf.U16, cf.U16, 1.0, 1.0, eta_q=cf.ETA16,
                            eta_c=cf.ETA16))
    for f in (10.0, 100.0, 1000.0):
        got = float(cf.eps_dot(768, cf.U16, cf.U16, f, f, eta_q=cf.ETA16,
                               eta_c=cf.ETA16))
        # Quadratic in the norms, to within the absolute terms' contribution.
        assert 0.98 <= got / (base * f * f) <= 1.02, (f, got, base * f * f)
    # And strictly increasing in each norm separately.
    assert (float(cf.eps_dot(768, cf.U16, cf.U16, 2.0, 1.0)) >
            float(cf.eps_dot(768, cf.U16, cf.U16, 1.0, 1.0)))
    assert (float(cf.eps_dot(768, cf.U16, cf.U16, 1.0, 2.0)) >
            float(cf.eps_dot(768, cf.U16, cf.U16, 1.0, 1.0)))


# =============================================================================
# 2. The document's published numbers
# =============================================================================

# Sec.0.1, "eps at unit norms and unit applied scales, fused first pass,
# envelope". Transcribed by hand from the document, NOT generated.
PUBLISHED = {
    384: (1.093e-3, 6.039e-4, 1.150e-4, 7.944e-3),
    768: (1.208e-3, 7.187e-4, 2.295e-4, 8.059e-3),
    1536: (1.438e-3, 9.482e-4, 4.585e-4, 8.290e-3),
    4096: (2.205e-3, 1.714e-3, 1.223e-3, 9.060e-3),
}


@pytest.mark.parametrize("d", sorted(PUBLISHED))
def test_the_published_table_is_reproduced(d):
    for want, (label, u_q, u_c, eta_q, eta_c, lam_t) in zip(PUBLISHED[d], CASES):
        got = float(cf.eps_cos(d, u_q, u_c, cf.C_HW_ENVELOPE, False,
                               eta_q, eta_c, 1.0, 1.0, lam_t))
        # The table is quoted to four significant figures.
        assert abs(got - want) / want < 6e-4, (d, label, want, got)


def test_the_production_constant_is_the_one_the_document_headlines():
    # d = 768, fp32 queries, fp16-stored corpus, fused first pass, envelope.
    # This is the number the whole document is about; if it moves, something
    # that was reviewed has changed.
    got = float(cf.eps_cos(768, cf.U16, 0.0, eta_q=cf.ETA16))
    assert f"{got:.6e}" == "7.187119e-04"


# =============================================================================
# 3. Structural properties
# =============================================================================

def test_the_load_time_self_test_passes():
    assert cf.self_test() is None
    assert cf.certified() is None


def test_no_power_operator_survives_in_the_certified_path():
    # P8: IEEE 754 requires correct rounding for + - * / sqrt but NOT for a
    # generic library `pow`, so a `**` inside any of these would be an
    # unstated premise about libm rather than a rounding Lemma 5 can charge.
    assert cf._binary_power_users() == []


def test_eps_is_increasing_in_every_argument_it_must_be():
    base = float(cf.eps_cos(768, cf.U16, 0.0, eta_q=cf.ETA16))
    assert float(cf.eps_cos(1536, cf.U16, 0.0, eta_q=cf.ETA16)) > base
    assert float(cf.eps_cos(768, cf.U16, cf.U16, eta_q=cf.ETA16,
                            eta_c=cf.ETA16)) > base
    # `Psi` increasing in both scales is what makes an UPPER bound on the
    # applied scales admissible and a lower one inadmissible (P1).
    assert float(cf.eps_cos(768, cf.U16, 0.0, eta_q=cf.ETA16, rho_q=2.0)) > base
    assert float(cf.eps_cos(768, cf.U16, 0.0, eta_q=cf.ETA16, sigma_max=2.0)) > base
    assert float(cf.eps_cos(768, cf.U16, 0.0, cf.C_HW_ENVELOPE, True,
                            cf.ETA16)) > base


def test_eps_can_never_go_below_the_conversion_floor():
    # Sec.7 exhibits a vector attaining `u_q + u_c` to 0.999, so no data-free
    # constant smaller than this exists — including at tiny `d`.
    for d in (64, 128, 768):
        assert cf.C_const(d, cf.U16, cf.U16) > cf.U16 + cf.U16 + cf.U16 * cf.U16
        assert cf.C_const(d, cf.U16, 0.0) > cf.U16


@needs_oracle
@needs_oracle
def test_the_envelope_dominates_every_published_generation():
    # Lemma 2': `c_hw = 4` must cover every row of the hardware table, with
    # `m <= ceil(d/b) + s_x` blocks and `s_x <= d/16` extra additions.
    for d in (64, 128, 768, 4096, 65536, 2 ** 20):
        env = cf.kappa_acc(d, cf.C_HW_ENVELOPE)
        for name, (b, nu, lam, e_min) in ORACLE.HW_MODELS.items():
            fitted = ORACLE.kappa_acc_model(d, b, nu, extra_blocks=d // 16)
            assert env >= fitted, (d, name, env, fitted)


def test_kappa_bar_is_an_inflation_not_a_re_rounding():
    # Rounding `kappa_hat` toward +inf at the end proves nothing: the roundings
    # inside its own evaluation can already have moved it down by several ulps.
    # `kappa_bar` must dominate by the full 2**-40, not by one ulp.
    for d in (64, 768, 4096, 2 ** 20):
        k, kb = cf.kappa(d), cf.kappa_bar(d)
        assert kb > k
        assert kb >= k * (1.0 + cf.KAPPA_INFLATE)
        assert kb < k * (1.0 + math.ldexp(1.0, -39))
        # And it must be strictly more than a single nextafter would give.
        assert kb > math.nextafter(k, math.inf)


def test_eps_final_never_rounds_down():
    rng = np.random.default_rng(20260909)
    Es = np.concatenate([np.geomspace(1e-12, 1e-1, 3000),
                         rng.random(3000) * 1e-3])
    for E in Es:
        B = 1.0 + float(E)
        want = (float(E) * (1.0 + cf.U) + cf.U * B) / (1.0 - cf.U) * (1.0 + cf.EVAL_INFLATE)
        got = float(cf.eps_final(float(E), B))
        assert got >= want, (E, got, want)


def test_eps_final_returns_a_binary32_value():
    # `thr` is stored in binary32; an `eps` that is not representable there
    # would be silently rounded — possibly DOWN — by whatever consumes it.
    out = cf.eps_final(7.0e-4, 1.0)
    assert out.dtype == np.float32


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0])
def test_every_guard_refuses_a_degenerate_norm(bad):
    assert not cf.norm_range_ok(bad)


def test_the_guards_refuse_nan_rather_than_accepting_it():
    # The doc spells every guard `not (...)` for exactly this reason: NaN makes
    # the inner comparison False, so the refusal fires and the row goes live.
    # Spelled `N < lo or N > hi` instead, NaN would PASS and be pruned on.
    nan = float("nan")
    assert not cf.norm_range_ok(nan)
    assert not cf.overflow_ok(768, nan, cf.FP16)
    assert not cf.product_ok(nan, 1.0)
    assert not cf.product_ok(1.0, nan)


def test_the_norm_range_guard_is_two_sided_at_the_documented_thresholds():
    assert cf.NORM_MIN == math.ldexp(1.0, -40)
    assert cf.NORM_MAX == math.ldexp(1.0, 126)
    assert cf.norm_range_ok(cf.NORM_MIN)
    assert cf.norm_range_ok(cf.NORM_MAX)
    assert not cf.norm_range_ok(math.nextafter(cf.NORM_MIN, 0.0))
    assert not cf.norm_range_ok(math.nextafter(cf.NORM_MAX, math.inf))
    # The shipped bound's threshold was 2**-63 and one-sided. Both halves of
    # that changed, and the raise is what makes Lemma 2's floor term negligible.
    assert cf.NORM_MIN > math.ldexp(1.0, -63)


def test_the_overflow_guard_binds_before_an_fp16_conversion_can_overflow():
    # A norm at the guard's limit must convert without reaching 65504; a norm
    # comfortably past it must be refused.
    d = 768
    limit = cf.OMEGA16 * (1.0 - cf.GUARD_MARGIN) / cf.kappa_bar(d)
    assert cf.overflow_ok(d, limit * 0.999, cf.FP16)
    assert not cf.overflow_ok(d, limit * 1.001, cf.FP16)
    # AT THE MARGIN, which is what makes `kappa_bar` load-bearing rather than
    # decorative. The inflation is `2**-40 ~ 9.1e-13`, so a probe at +/-0.1%
    # cannot tell `kappa_bar` from `kappa` — and a mutation swapping them
    # passed the whole suite. These two straddle the actual boundary.
    assert not cf.overflow_ok(d, limit * (1.0 + 5e-13), cf.FP16)
    assert cf.overflow_ok(d, limit * (1.0 - 5e-13), cf.FP16)
    # An axis already stored in the first-pass format is not converted, so it
    # cannot overflow on conversion and the guard does not apply.
    assert cf.overflow_ok(d, 1e30, cf.EXACT)
    # bfloat16's Omega is finite too — the guard is needed there as well, just
    # ~5e33 later.
    assert cf.overflow_ok(d, 1e30, cf.BF16)
    assert not cf.overflow_ok(d, 1e39, cf.BF16)


def test_the_norm_product_guard_is_decided_without_rounding():
    # The binary64 product of two binary32 norms is exact (48 significand bits),
    # so a value one ulp over the threshold must be refused, not rounded onto it.
    a32 = np.float32(math.ldexp(1.0, 63))
    b32 = np.float32(math.ldexp(1.0, 63))
    assert cf.product_ok(float(a32), float(b32))            # exactly 2**126
    # The NEXT BINARY32 up, not the next binary64: the guard's inputs are
    # binary32 norms, and the point is that their binary64 product is exact, so
    # one binary32 ulp of excess must be seen rather than rounded away.
    up = np.nextafter(a32, np.float32(np.inf))
    assert float(up) > float(a32)
    assert not cf.product_ok(float(up), float(b32))

    # THE CASE THAT SEPARATES THE TWO PRECISIONS. The pair above is
    # representable in binary32 as well, so it cannot see the difference — a
    # mutation deciding this guard in binary32 passed the whole suite. Here the
    # exact product exceeds 2**126 but rounds DOWN to exactly 2**126 in
    # binary32, so a binary32 comparison accepts what must be refused.
    qn = float(np.float32(math.ldexp(1.0, 63) * (1.0 + 2.0 ** -23)))
    cn = float(np.float32(math.ldexp(1.0, 63) * (1.0 - 2.0 ** -24)))
    assert qn * cn > cf.PROD_MAX, "fixture no longer straddles the threshold"
    assert np.float32(qn) * np.float32(cn) <= np.float32(cf.PROD_MAX), (
        "fixture no longer rounds down in binary32")
    assert not cf.product_ok(qn, cn), (
        "the norm-product guard is being decided in binary32: a real product "
        "above 2**126 rounded down onto it and was accepted")


def test_the_dimension_guard_is_the_documented_range():
    assert cf.dimension_ok(64) and cf.dimension_ok(768) and cf.dimension_ok(2 ** 20)
    assert not cf.dimension_ok(63)
    assert not cf.dimension_ok(2 ** 20 + 1)


# =============================================================================
# 4. Empirical falsification of (SAFE)
# =============================================================================
#
# (SAFE): `A(q) + eps >= max_c s_e(q, c)` for the fp32 score the one-pass path
# computes. Everything below runs both passes for real and checks it.

def _two_passes_torch(Q, C, corpus_is_fp16):
    """Both passes, spelled exactly as the production path spells them.

    Pass one:  Qh @ Ch^T in float32, times `col_scale` (a STORED binary32
               reciprocal, as `_gemm_rowmax`'s epilogue applies it), row max,
               then times `row_scale` on the host.
    Pass two:  the float32 Gram, DIVIDED by the same computed norms — the
               asymmetry (reciprocal vs divide) is what P1 allows and what
               `bracket`'s `(1+u)^4/(1-u)^2` mismatch factor pays for.
    """
    qn = Q.norm(dim=1)
    cn = C.norm(dim=1).clamp_min(1e-12)
    row_scale = qn.reciprocal()
    col_scale = cn.reciprocal()

    Qh = Q.half()
    Ch = C.half()
    acc = (Qh.float() @ Ch.float().T)          # float32 accumulation, no fp16 output
    approx = (acc * col_scale[None, :]).amax(dim=1) * row_scale

    exact = (Q @ C.T).div(cn[None, :]).div(qn[:, None])
    return approx, exact.amax(dim=1), qn, cn, row_scale, col_scale


def _eps_for(Q, C, corpus_is_fp16, row_scale, col_scale):
    d = int(Q.shape[1])
    u_c, eta_c = (0.0, 0.0) if corpus_is_fp16 else (cf.U16, cf.ETA16)
    return cf.eps_cos(
        d, cf.U16, u_c, cf.C_HW_ENVELOPE, False, cf.ETA16, eta_c,
        rho_q=row_scale.double().numpy(),
        sigma_max=float(col_scale.max()),
        lam_t=cf.LAMBDA16,
    )


@pytest.mark.parametrize("d", [64, 128, 768])
@pytest.mark.parametrize("corpus_is_fp16", [True, False])
@pytest.mark.parametrize("seed", range(6))
def test_safe_holds_on_ordinary_data(d, corpus_is_fp16, seed):
    g = torch.Generator().manual_seed(seed * 977 + d)
    Q = torch.randn(48, d, generator=g, dtype=torch.float32)
    C = torch.randn(96, d, generator=g, dtype=torch.float32)
    if corpus_is_fp16:
        C = C.half().float()
    approx, exact, qn, cn, rs, cs = _two_passes_torch(Q, C, corpus_is_fp16)
    eps = torch.from_numpy(np.asarray(_eps_for(Q, C, corpus_is_fp16, rs, cs)))
    assert torch.all(approx + eps >= exact), (approx + eps - exact).min()


@pytest.mark.parametrize("d", [64, 128, 768])
def test_safe_holds_when_every_product_has_the_same_sign(d):
    # The adversarial case for the ACCUMULATION term: with no cancellation the
    # aligned addends are all pushed away from zero, so the truncating
    # accumulator's error is maximal relative to the result.
    g = torch.Generator().manual_seed(4242 + d)
    Q = torch.rand(32, d, generator=g, dtype=torch.float32) + 0.5
    C = torch.rand(64, d, generator=g, dtype=torch.float32) + 0.5
    approx, exact, qn, cn, rs, cs = _two_passes_torch(Q, C, False)
    eps = torch.from_numpy(np.asarray(_eps_for(Q, C, False, rs, cs)))
    assert torch.all(approx + eps >= exact)


@pytest.mark.parametrize("d", [64, 128, 768])
def test_safe_holds_when_the_components_straddle_the_fp16_subnormal_range(d):
    # The adversarial case for the CONVERSION term and for `Psi`'s `eta`: below
    # 2**-14 the fp16 quantum stops being relative, and the absolute floor
    # `eta = 2**-25` is the only thing bounding the residual.
    g = torch.Generator().manual_seed(99 + d)
    Q = (torch.rand(32, d, generator=g, dtype=torch.float32) - 0.5) * (2.0 ** -13)
    C = (torch.rand(64, d, generator=g, dtype=torch.float32) - 0.5) * (2.0 ** -13)
    approx, exact, qn, cn, rs, cs = _two_passes_torch(Q, C, False)
    eps = torch.from_numpy(np.asarray(_eps_for(Q, C, False, rs, cs)))
    assert torch.all(approx + eps >= exact)


@pytest.mark.parametrize("d", [64, 128, 768])
def test_safe_holds_with_one_dominant_component_and_a_tail_of_tiny_ones(d):
    # The adversarial case for the ANCHOR term of Lemma 2: the block's largest
    # product sets the alignment, and everything more than 24 + nu bits below it
    # is chopped entirely. This is the construction that annihilates the tail.
    g = torch.Generator().manual_seed(7 + d)
    Q = torch.full((16, d), 2.0 ** -13, dtype=torch.float32)
    Q[:, 0] = 1.0
    Q += torch.randn(16, d, generator=g) * (2.0 ** -20)
    C = torch.full((32, d), 2.0 ** -13, dtype=torch.float32)
    C[:, 0] = 1.0
    C += torch.randn(32, d, generator=g) * (2.0 ** -20)
    approx, exact, qn, cn, rs, cs = _two_passes_torch(Q, C, False)
    eps = torch.from_numpy(np.asarray(_eps_for(Q, C, False, rs, cs)))
    assert torch.all(approx + eps >= exact)


@pytest.mark.parametrize("qscale", [1e-2, 1.0, 1e2])
@pytest.mark.parametrize("cscale", [1e-3, 1.0, 1e3])
@needs_oracle
def test_safe_holds_across_the_admitted_range_of_norms(qscale, cscale):
    # `Psi` grows as 1/N_q and 1/min N_c, and these are the magnitudes at which
    # the prunability cut starts to bite. The bound must hold throughout, even
    # where it is too loose to be useful.
    d = 128
    g = torch.Generator().manual_seed(int(qscale * 1e6) + int(cscale * 1e6))
    Q = torch.randn(24, d, generator=g, dtype=torch.float32) * qscale
    C = torch.randn(48, d, generator=g, dtype=torch.float32) * cscale
    approx, exact, qn, cn, rs, cs = _two_passes_torch(Q, C, False)
    assert cf.norm_range_ok(float(qn.min())) and cf.norm_range_ok(float(cn.max()))
    eps = torch.from_numpy(np.asarray(_eps_for(Q, C, False, rs, cs)))
    assert torch.all(approx + eps >= exact)


# --- the exact-rational tensor-core emulation ---------------------------------
#
# Everything above runs pass one on a float32 CPU accumulator that rounds to
# NEAREST. Real tensor cores align to the block maximum and TRUNCATE, which is
# the whole reason `kappa_acc` sits above `gamma_d`. This is the only way to
# exercise that without a GPU: replay the published model in exact rationals.

def _rational_first_pass(q_h, c_h, model="Turing/Ampere/Ada", zero_anchors=True):
    from fractions import Fraction

    b, nu, lam, e_min = ORACLE.HW_MODELS[model]
    return ORACLE.block_fma_dot(list(q_h), list(c_h), b, nu, lam,
                                zero_anchors, e_min)


@needs_oracle
@pytest.mark.parametrize("model", ["Volta m8n8k4", "Turing/Ampere/Ada",
                                   "Hopper/Blackwell"])
@pytest.mark.parametrize("kind", ["random", "same_sign", "dominant_plus_tail",
                                  "subnormal_band"])
def test_safe_holds_against_the_emulated_tensor_core_accumulator(model, kind):
    """(SAFE) end to end with pass one replaced by the published hardware model.

    `d` is small because every product is an exact `Fraction`; the terms that
    scale with `d` are checked by `envelope_check()` in the doc's script over
    the whole admitted range, so what this adds is the END-TO-END composition —
    conversion, truncating accumulation, both scalings and the wrapper together,
    against a float64 reference for the exact pass.
    """
    from fractions import Fraction

    d, n_q, n_c = 64, 8, 12
    rng = np.random.default_rng(hash((model, kind)) % (2 ** 32))
    if kind == "random":
        Q = rng.standard_normal((n_q, d))
        C = rng.standard_normal((n_c, d))
    elif kind == "same_sign":
        Q = rng.random((n_q, d)) + 0.5
        C = rng.random((n_c, d)) + 0.5
    elif kind == "dominant_plus_tail":
        Q = np.full((n_q, d), 2.0 ** -13) + rng.standard_normal((n_q, d)) * 2.0 ** -20
        C = np.full((n_c, d), 2.0 ** -13) + rng.standard_normal((n_c, d)) * 2.0 ** -20
        Q[:, 0] = 1.0
        C[:, 0] = 1.0
    else:
        Q = (rng.random((n_q, d)) - 0.5) * 2.0 ** -13
        C = (rng.random((n_c, d)) - 0.5) * 2.0 ** -13

    Qt = torch.tensor(Q, dtype=torch.float32)
    Ct = torch.tensor(C, dtype=torch.float32)
    qn = Qt.norm(dim=1)
    cn = Ct.norm(dim=1).clamp_min(1e-12)
    row_scale = qn.reciprocal()
    col_scale = cn.reciprocal()
    Qh = Qt.half().float().numpy().astype(np.float64)
    Ch = Ct.half().float().numpy().astype(np.float64)

    # Pass one, through the model, then the two float32 scalings the kernel and
    # the host apply — in that order.
    approx = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        best = -np.inf
        for j in range(n_c):
            g = _rational_first_pass(Qh[i], Ch[j], model)
            v = np.float32(np.float32(float(g)) * np.float32(col_scale[j]))
            best = max(best, float(v))
        approx[i] = np.float32(np.float32(best) * np.float32(row_scale[i]))

    # Pass two: what the one-pass float32 path computes.
    exact = (Qt @ Ct.T).div(cn[None, :]).div(qn[:, None]).amax(dim=1).numpy()

    eps = cf.eps_cos(d, cf.U16, cf.U16, cf.C_HW_ENVELOPE, False,
                     cf.ETA16, cf.ETA16,
                     rho_q=row_scale.double().numpy(),
                     sigma_max=float(col_scale.max()))
    slack = (approx.astype(np.float64) + eps.astype(np.float64)) - exact.astype(np.float64)
    assert np.all(slack >= 0.0), (model, kind, slack.min())


@needs_oracle
def test_the_emulated_accumulator_really_is_worse_than_a_rounding_one():
    """A guard on the guard: if the emulation silently degenerated into ordinary
    float32 arithmetic, every test above it would pass for the wrong reason."""
    from fractions import Fraction

    d = 64
    rng = np.random.default_rng(11)
    a = (rng.random(d) + 0.5).astype(np.float32)
    b = (rng.random(d) + 0.5).astype(np.float32)
    ah = torch.tensor(a).half().float().numpy().astype(np.float64)
    bh = torch.tensor(b).half().float().numpy().astype(np.float64)
    exact = sum(Fraction(float(x)) * Fraction(float(y)) for x, y in zip(ah, bh))
    emulated = _rational_first_pass(ah, bh)
    rounded = float(np.float32(np.dot(ah.astype(np.float32), bh.astype(np.float32))))
    err_emul = abs(float(emulated - exact))
    err_round = abs(float(Fraction(rounded) - exact))
    assert err_emul > err_round, (err_emul, err_round)
    # And it must still be inside the envelope the bound charges for.
    assert err_emul / abs(float(exact)) < cf.kappa_acc(d, cf.C_HW_ENVELOPE)


# --------------------------------------------------------------------------
# Findings from the whole-file audit of 2026-09-10.
# --------------------------------------------------------------------------

def test_eps_final_evaluates_in_binary64_whatever_dtype_it_is_handed():
    """P8 says the bound is computed in binary64. That has to be true of the
    FUNCTION, not of its callers.

    numpy takes the dtype from the operands, so a float32 `E` or `B` evaluated
    the whole expression in binary32 and the `asarray(..., float64)` that
    followed only re-labelled a value whose precision was already gone.
    Measured before the fix: the inner expression came out 7.100596558e-4
    against the binary64 7.100596913e-4 — SMALLER, an under-bound.

    `Psi` and `eps_dot` widen their own inputs, so no shipped path reached
    this; a caller handing in float32 is not being unreasonable, and nothing
    told it otherwise.
    """
    import numpy as np
    from nova_bf import closed_form as cf

    # The float32 arrays must hold EXACTLY the same values as the float64
    # ones, or the two calls differ in their inputs and the comparison says
    # nothing about how the expression is evaluated. Round-trip to fix that.
    E32 = np.array([7.1e-4, 1.2e-3, 9.5e-4], dtype=np.float32)
    B32 = np.array([1.0000229, 2.5, 0.5], dtype=np.float32)
    E64, B64 = E32.astype(np.float64), B32.astype(np.float64)
    assert np.array_equal(E32.astype(np.float64), E64)  # same values, two dtypes

    narrow = cf.eps_final(E32, B32)
    wide = cf.eps_final(E64, B64)
    assert np.array_equal(wide, narrow), (
        f"identical values in two dtypes gave different results: {narrow} vs "
        f"{wide}. The expression is being evaluated in the operands' dtype.")
    # A scalar float32, which numpy promotes by yet another rule.
    assert cf.eps_final(np.float32(7.1e-4), np.float32(1.0)) == \
        cf.eps_final(np.float64(np.float32(7.1e-4)), 1.0)


def test_ceil_pow2_is_inside_the_certified_arithmetic():
    """It uses `frexp`/`ldexp` deliberately: a `log2`/`**` rewrite can round
    DOWNWARD, and `sigma_max` bucketed downward is an under-bound.

    It was absent from `_CERTIFIED_FUNCS`, so the scanner never looked at it —
    a `log2` implementation passed `self_test()` unchallenged.
    """
    from nova_bf import closed_form as cf

    assert "ceil_pow2" in cf._CERTIFIED_FUNCS


def test_the_arithmetic_scanner_is_an_allowlist_not_a_denylist():
    """A denylist only rejects the spellings someone thought of.

    Both of these passed the old scanner, which searched for banned NAMES:
    the bytecode loads `_p` and `helper`, never `pow` or `log2`.
    """
    from nova_bf import closed_form as cf

    # Compiled from SOURCE, not defined as nested functions: a nested `def`
    # closes over `_p`/`helper`, so the bytecode is LOAD_DEREF (a local cell)
    # rather than LOAD_GLOBAL, and the scanner deliberately does not police
    # local names — allowlisting those would mean allowlisting every local
    # variable in the module. The real certified functions are module level,
    # which is what this reproduces.
    def _scan(src):
        ns = {}
        exec(compile(src, "<probe>", "exec"), ns)
        return cf._code_uses_power(ns["f"].__code__)

    assert "un-vetted name" in _scan(
        "import math\n_p = math.pow\ndef f(x):\n    return _p(x, 2.0)\n")
    assert "un-vetted name" in _scan(
        "import math\n"
        "def helper(x):\n    return math.log2(x)\n"
        "def f(x):\n    return helper(x)\n")
    # `**` is caught by the opcode check, independently of any name.
    assert _scan("def f(x):\n    return x ** 2\n") == "uses **"
    # The real module must be clean, or the allowlist is too tight to ship.
    assert cf._binary_power_users() == []


@pytest.mark.parametrize("const", ["FLOOR_COS", "KAPPA_INFLATE", "EVAL_INFLATE"])
def test_self_test_refuses_a_weakened_safety_constant(monkeypatch, const):
    """`self_test()` is the RUNTIME certification barrier — the only one that
    runs in a deployment, where pytest and the mutation harness do not.

    All three of these are pure margin: each covers an error the formula does
    not otherwise model, and each is safe larger and unsafe smaller. Setting
    any to zero left `self_test()` returning None, because nothing else in it
    constrains them from below (the `kappa_bar` check asserts the inflation is
    not too LARGE).
    """
    from nova_bf import closed_form as cf

    assert cf.self_test() is None, "control: a clean module must certify"
    monkeypatch.setattr(cf, const, 0.0)
    cf.reset()
    why = cf.self_test()
    cf.reset()
    assert isinstance(why, str) and const in why, (
        f"zeroing {const} left self_test() returning {why!r}")
