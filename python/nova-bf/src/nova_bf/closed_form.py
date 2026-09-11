"""Closed-form error bound for the half-precision first pass.

This formulation requires no per-slice conversion residuals or 
empirical slack constants.

For cosine, the bound has the form

```
eps = fl_up[(E(1 + u) + uB) / (1 - u)],
E = C(d) + Psi(q, S)
```

where `C(d)` depends only on dimension, formats, and the hardware error model,
while `Psi` carries the data-dependent absolute terms through pass one's scale
factors.

The arithmetic below is part of the proof, not merely an implementation choice:

1. Certified expressions are evaluated in binary64 using only `+`, `-`, `*`,
   `/`, `sqrt`, and exact `ldexp` powers of two. Generic `pow`/libm operations
   are excluded because the proof does not assume they are correctly rounded.
2. Expressions avoid cancelling subtractions.
3. Expression shape is load-bearing: algebraically equivalent rewrites can
   change the binary64 evaluation-error bound.
4. Final `eps` values are inflated by `(1 + 2^-30)` and rounded upward to
   binary32.

Guard products are also evaluated in binary64, using the outward-inflated
`kappa_bar` where required to establish the theorem hypotheses.

`self_test()` checks structural invariants of the certified path.
"""


from __future__ import annotations

import math

# --- IEEE format constants ----------------------------------------------------

# Use `ldexp` for exact powers of two; generic `pow` is excluded from the
# certified arithmetic because its rounding behavior is not assumed.
U = math.ldexp(1.0, -24)          # binary32 unit roundoff
U16 = math.ldexp(1.0, -11)        # binary16 unit roundoff
UBF = math.ldexp(1.0, -8)         # bfloat16 unit roundoff

# Absolute conversion floors in
#
#     |fl_t(x) - x| <= u_t |x| + eta_t.
#
# For round-to-nearest, `eta_t` is half the target format's subnormal quantum.
ETA16 = math.ldexp(1.0, -25)      # binary16
ETA_BF = math.ldexp(1.0, -134)    # bfloat16

# Smallest normal operand values used by the tensor-core absolute-error term.
LAMBDA16 = math.ldexp(1.0, -14)   # binary16
LAMBDA32 = math.ldexp(1.0, -126)  # bfloat16 / TF32

# Largest finite values of the first-pass formats.
OMEGA16 = 65504.0
OMEGA_BF = math.ldexp(
    2.0 - math.ldexp(1.0, -7), 127
)  # (2 - 2^-7) * 2^127

# Absolute reserve covering the remaining cosine underflow terms.
FLOOR_COS = math.ldexp(1.0, -44)


# --- admissibility and pruning thresholds ------------------------------------

# Dimension range over which the accumulation and computed-norm bounds hold.
D_MIN, D_MAX = 64, 2 ** 20

# Norm range required by the bound. The lower limit controls absolute-error
# and reciprocal-scaling terms; the upper limit keeps intermediates finite.
NORM_MIN = math.ldexp(1.0, -40)
NORM_MAX = math.ldexp(1.0, 126)

# Maximum norm product permitted without binary32 overflow.
PROD_MAX = math.ldexp(1.0, 126)

# Outward margins used by the norm-inflation and conversion-overflow guards.
KAPPA_INFLATE = math.ldexp(1.0, -40)
GUARD_MARGIN = math.ldexp(1.0, -40)

# Covers binary64 evaluation error before upward rounding to binary32.
EVAL_INFLATE = math.ldexp(1.0, -30)

# Below these norms the absolute-error terms make pruning unprofitable, so the
# affected query rows or corpus slices are kept live without running pass one.
PRUNE_MIN_QNORM = 1e-2
PRUNE_MIN_CNORM = 1e-3

# Conservative tensor-core accumulation envelope and Ampere coefficient.
C_HW_ENVELOPE = 4.0
C_HW_AMPERE = 1.375

# --- floating-point helper bounds --------------------------------------------

def gamma(n: int, unit: float = U) -> float:
    """Return gamma_n = n*unit / (1 - n*unit).

    This is the standard relative-error factor for a length-`n`
    floating-point accumulation when `n*unit < 1`.
    """
    nu = n * unit
    if nu >= 1.0:
        raise ValueError(f"gamma: n={n} too large for unit={unit}")
    return nu / (1.0 - nu)


def kappa(d: int) -> float:
    """Return the inflation factor relating a true norm to its binary32 norm.

    For `N >= NORM_MIN`,

        ||v|| <= kappa(d) * N.

    The first two factors bound binary32 norm computation error; the final
    `(1 + d * 2^-70)` term absorbs the remaining absolute underflow error into
    the relative bound.

    The expression intentionally uses `sqrt` and `ldexp`, not `pow`, so it
    stays within the certified arithmetic model.
    """
    k0 = (
        1.0 / ((1.0 - U) * math.sqrt(1.0 - U))
        * (1.0 / math.sqrt(1.0 - gamma(d - 1)))
    )
    return k0 * (1.0 + d * math.ldexp(1.0, -70))

def kappa_bar(d: int) -> float:
    """Return an outward-rounded upper bound on `kappa(d)` for guard checks.

    `KAPPA_INFLATE` covers binary64 evaluation error accumulated inside
    `kappa()`. The final `nextafter` covers rounding of the inflating multiply,
    ensuring the result remains an upper bound.

    This inflated form is used only for guards; the bound itself uses `kappa`.
    """
    return math.nextafter(
        kappa(d) * (1.0 + KAPPA_INFLATE),
        math.inf,
    )

def kappa_acc(d: int, c_hw: float = C_HW_ENVELOPE) -> float:
    """Return the tensor-core accumulation error envelope.

    The bound is

        c_hw * d * U * (1 + c_hw * d * U),

    where `c_hw` accounts for tensor-core truncation effects beyond the
    standard floating-point accumulation model. `C_HW_ENVELOPE` is chosen
    conservatively over the supported hardware range.
    """
    x = c_hw * d * U
    return x * (1.0 + x)

def theta(d: int, c_hw: float = C_HW_ENVELOPE) -> float:
    """Return `(1 + kappa_acc)(1 + U)^2 - 1` without cancellation."""
    k = kappa_acc(d, c_hw)
    return k + (2.0 * U + U * U) * (1.0 + k)

def bracket(
    d: int,
    u_q: float,
    u_c: float,
    c_hw: float = C_HW_ENVELOPE,
    half_out: bool = False,
) -> float:
    """Return the dimensionless relative-error bracket shared by both metrics.

    Terms cover the exact pass's accumulation and scale mismatch, post-scaling
    roundoff, tensor-core accumulation, input conversion, and optionally the
    fp16 write of the first-pass Gram matrix.
    """
    th = theta(d, c_hw)
    inp = u_q + u_c + u_q * u_c

    # Covers the maximum scale-factor mismatch between the two scoring paths.
    mism = (
        ((1.0 + U) * (1.0 + U))
        * ((1.0 + U) * (1.0 + U))
        / ((1.0 - U) * (1.0 - U))
    )

    core = (
        mism * gamma(d)
        + 6.0 * U
        + 22.0 * U * U
        + th
        + inp * (1.0 + th)
    )

    if half_out:
        # Charge the extra fp16 rounding before the final scale is applied.
        core += U16 * (1.0 + th) * (1.0 + inp)

    return core

def Lambda(d: int) -> float:
    """Return the norm-and-scale inflation factor for cosine scoring.

    The `kappa(d)^2` term bounds both true norms from their computed norms.
    The `(1 + U)^2` term covers upward rounding of the two stored reciprocal
    scale factors used by pass one.
    """
    k = kappa(d)
    return k * k * ((1.0 + U) * (1.0 + U))

def C_const(d: int, u_q: float, u_c: float, c_hw: float = C_HW_ENVELOPE,
            half_out: bool = False) -> float:
    """Return the data-independent cosine error constant for this configuration."""
    return Lambda(d) * bracket(d, u_q, u_c, c_hw, half_out) + FLOOR_COS


def Psi(d: int, u_q: float, u_c: float, eta_q: float, eta_c: float,
        rho_q, sigma_max, c_hw: float = C_HW_ENVELOPE, half_out: bool = False,
        lam_t: float = LAMBDA16):
    """Return the data-dependent absolute-error term.

    `rho_q` and `sigma_max` are pass-one scale factors or safe upper bounds on
    them. The term is increasing in both, so underestimating either is unsafe.

    Scale factors are widened to binary64 before evaluation; `rho_q` may be
    per-query and is handled elementwise.
    """
    import numpy as np

    rho_q = np.asarray(rho_q, dtype=np.float64)
    sigma_max = np.asarray(sigma_max, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _psi_inner(d, u_q, u_c, eta_q, eta_c, rho_q, sigma_max,
                          c_hw, half_out, lam_t)


def _psi_inner(d, u_q, u_c, eta_q, eta_c, rho_q, sigma_max, c_hw, half_out,
               lam_t):
    th = theta(d, c_hw)
    k = kappa(d)
    sd = math.sqrt(d)
    # Absolute error from input conversion, including subnormal rounding.
    conv = (1.0 + th) * (
        k * (1.0 + U) * sd * (eta_q * (1.0 + u_c) * rho_q + eta_c * (1.0 + u_q) * sigma_max)
        + d * eta_q * eta_c * rho_q * sigma_max
    )

    # Absolute tensor-core error from exponent anchoring and zero-anchored blocks.
    anchor = (kappa_acc(d, c_hw) + d * U) * ((1.0 + U) * (1.0 + U)) * (
        lam_t * (
            k * (1.0 + U) * ((1.0 + u_q) * sigma_max + (1.0 + u_c) * rho_q)
            + sd * (eta_q + eta_c) * rho_q * sigma_max
        )
        + lam_t * lam_t * rho_q * sigma_max
    )
    out = (ETA16 * rho_q * sigma_max * ((1.0 + U) * (1.0 + U))) if half_out else 0.0
    return (conv + anchor) * ((1.0 + U16) if half_out else 1.0) + out


def eps_final(E, B):
    """Return the final binary32 error allowance.

    Computes

        eps >= (E * (1 + U) + U * B) / (1 - U),

    so `fl(A + eps) >= A + E` for every `|A| <= B`.

    The expression is evaluated in binary64, inflated by `EVAL_INFLATE`, then
    rounded upward to binary32. Accepts scalars or arrays.
    """
    import numpy as np

    E = np.asarray(E, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    e = (E * (1.0 + U) + U * B) / (1.0 - U) * (1.0 + EVAL_INFLATE)
    e = np.asarray(e, dtype=np.float64)

    # Overflow to +inf is conservative: it forces the corresponding row live.
    with np.errstate(over="ignore", invalid="ignore"):
        f = e.astype(np.float32)

    # If round-to-nearest went below `e`, advance one binary32 value upward.
    low = f.astype(np.float64) < e
    if low.any():
        f = np.where(low, np.nextafter(f, np.float32(np.inf)), f)
    return f


def _refuse_bad_inputs(d, scales=(), tols=()):
    """Return whether any input lies outside the bound's admitted domain.

    Scales must be finite and strictly positive because the bound is monotone
    in them and zero would understate the error. Tolerances may be zero, but
    must otherwise be finite and non-negative.
    """
    import numpy as np

    if not dimension_ok(d):
        return True
    
    # Scale factors and norm bounds must be strictly positive.
    for v in scales:
        a = np.asarray(v, dtype=np.float64)
        if not bool(np.all(np.isfinite(a) & (a > 0.0))):
            return True
    # Roundoff and absolute-error constants may legitimately be zero.
    for v in tols:
        a = np.asarray(v, dtype=np.float64)
        if not bool(np.all(np.isfinite(a) & (a >= 0.0))):
            return True
    return False


def eps_cos(d: int, u_q: float, u_c: float, c_hw: float = C_HW_ENVELOPE,
            half_out: bool = False, eta_q: float = 0.0, eta_c: float = 0.0,
            rho_q=1.0, sigma_max=1.0, lam_t: float = LAMBDA16):
    """Return the per-query cosine error allowance in threshold units.

    Invalid shared inputs force the whole result to `+inf`. Invalid per-query
    scales are neutralized individually so unaffected rows remain prunable.
    """
    import numpy as np

    if _refuse_bad_inputs(d, scales=(sigma_max, lam_t),
                          tols=(u_q, u_c, eta_q, eta_c)):
        return np.full(np.shape(np.asarray(rho_q)), np.float32(np.inf),
                       dtype=np.float32)
    rho_arr = np.asarray(rho_q, dtype=np.float64)

    # Non-positive per-query scales are unsafe; NaN propagates and forces live.
    bad_rho = ~(rho_arr > 0.0) & ~np.isnan(rho_arr)      # negative, -inf or 0
    E = (C_const(d, u_q, u_c, c_hw, half_out)
         + Psi(d, u_q, u_c, eta_q, eta_c, rho_q, sigma_max, c_hw, half_out, lam_t))
    B = (Lambda(d) * (1.0 + theta(d, c_hw)) * (1.0 + u_q) * (1.0 + u_c)
         * ((1.0 + U16) if half_out else 1.0)) + E
    out = eps_final(E, B)
    if bad_rho.any():
        out = np.where(bad_rho, np.float32(np.inf), out).astype(np.float32)
    return out


def eps_dot(d: int, u_q: float, u_c: float, qn_ub, cn_ub,
            c_hw: float = C_HW_ENVELOPE, half_out: bool = False,
            eta_q: float = 0.0, eta_c: float = 0.0, lam_t: float = LAMBDA16):
    """Return the per-query dot-product error allowance.

    `qn_ub` and `cn_ub` bound the computed query and corpus norms. Unlike
    cosine, dot scoring has no reciprocal-norm scaling, so its absolute-error
    terms must retain these norm factors rather than reuse `Psi()`.

    Invalid shared inputs force the whole result to `+inf`; non-positive
    per-query norm bounds are neutralized individually.
    """
    import numpy as np

    # Keep the certified arithmetic in binary64. 
    qn_ub = np.asarray(qn_ub, dtype=np.float64)
    cn_ub = np.asarray(cn_ub, dtype=np.float64)

    if _refuse_bad_inputs(d, scales=(cn_ub, lam_t),
                          tols=(u_q, u_c, eta_q, eta_c)):
        return np.full(np.shape(qn_ub), np.float32(np.inf), dtype=np.float32)
    
    # Neutralize unsafe query norms during evaluation, then force them live.
    bad_qn = ~(qn_ub > 0.0) & ~np.isnan(qn_ub)           # negative, -inf or 0
    if bad_qn.any():
        qn_ub = np.where(bad_qn, 0.0, qn_ub)

    k = kappa(d)
    th = theta(d, c_hw)
    sd = math.sqrt(d)

    # Non-finite intermediates are conservative and ultimately force rows live.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = _eps_dot_inner(d, u_q, u_c, k, th, sd, qn_ub, cn_ub,
                             c_hw, half_out, eta_q, eta_c, lam_t)
    if bad_qn.any():
        out = np.where(bad_qn, np.float32(np.inf), out).astype(np.float32)
    return out


def _eps_dot_inner(d, u_q, u_c, k, th, sd, qn_ub, cn_ub, c_hw, half_out,
                   eta_q, eta_c, lam_t):
    Qn = k * qn_ub
    Cn = k * cn_ub

    rel = Qn * Cn * bracket(d, u_q, u_c, c_hw, half_out)
    conv = (1.0 + th) * (
        sd * (eta_q * (1.0 + u_c) * Cn + eta_c * (1.0 + u_q) * Qn)
        + d * eta_q * eta_c
    )
    anchor = (kappa_acc(d, c_hw) + d * U) * ((1.0 + U) * (1.0 + U)) * (
        lam_t * ((1.0 + u_q) * Qn + (1.0 + u_c) * Cn + sd * (eta_q + eta_c))
        + lam_t * lam_t
    )
    # Absolute underflow allowances not proportional to the vector norms.
    zeta = (2.0 * d - 1.0) * math.ldexp(1.0, -150) / (1.0 - d * U)
    floor = (3.01 * (d + 8.0) * math.ldexp(1.0, -149)
             + ((1.0 + U) * (1.0 + U)) * ((1.0 + U) * (1.0 + U))
             / ((1.0 - U) * (1.0 - U)) * zeta)
    absolute = conv + anchor
    if half_out:
        # Charge the relative and absolute rounding of the fp16-written Gram.
        absolute = absolute * (1.0 + U16) + ETA16 * ((1.0 + U) * (1.0 + U))
    E = rel + absolute + floor
    # Bound the magnitude of the first-pass score before the final wrapper.
    B = (Qn * Cn * (1.0 + th) * (1.0 + u_q) * (1.0 + u_c)
         * ((1.0 + U16) if half_out else 1.0)) + E
    return eps_final(E, B)


def ceil_pow2(x: float) -> float:
    """Return the smallest power of two >= `x` for finite positive `x`.

    Used to bucket `sigma_max` while preserving a safe upper bound for `Psi`.
    `frexp` and `ldexp` avoid `pow` and logarithms in the certified path.
    """
    if not (x > 0.0) or x != x or x == float("inf"):
        # Let downstream guards refuse invalid inputs.
        return x
    m, e = math.frexp(x)          # x == m * 2**e, 0.5 <= m < 1
    if m == 0.5:
        return math.ldexp(1.0, e - 1)
    if e >= 1024:
        # The next power of two is not finite in binary64.
        return math.inf
    return math.ldexp(1.0, e)


# --- the format descriptor ----------------------------------------------------
class Format:
    """Describe the error constants for one first-pass numeric format.

    `u_t` and `eta_t` bound conversion error, `lam_t` is the smallest normal
    operand value, and `omega_t` is the largest finite value. For data already
    stored in the first-pass format, `u_t = eta_t = 0` because no conversion
    error is introduced.
    """

    __slots__ = ("name", "u_t", "eta_t", "lam_t", "omega_t")

    def __init__(self, name, u_t, eta_t, lam_t, omega_t):
        self.name = name
        self.u_t = u_t
        self.eta_t = eta_t
        self.lam_t = lam_t
        self.omega_t = omega_t

    def __repr__(self):
        return f"Format({self.name})"


FP16 = Format("binary16", U16, ETA16, LAMBDA16, OMEGA16)
BF16 = Format("bfloat16", UBF, ETA_BF, LAMBDA32, OMEGA_BF)

# No conversion error when the stored data is already binary16.
EXACT = Format("already-fp16", 0.0, 0.0, LAMBDA16, OMEGA16)
# --- bound guards -------------------------------------------------------------

# Guards are written so NaN fails closed: an invalid value disables pruning
# rather than allowing an unsupported case through.

def dimension_ok(d: int) -> bool:
    """Return whether `d` lies in the dimension range covered by the bound."""
    return isinstance(d, int) and D_MIN <= d <= D_MAX


def norm_range_ok(n: float) -> bool:
    """Return whether a scalar norm lies in the admitted finite range.

    The chained comparison intentionally rejects NaN.
    """
    return NORM_MIN <= n <= NORM_MAX


def overflow_ok(d: int, n_max: float, fmt: Format) -> bool:
    """Return whether conversion to `fmt` is guaranteed not to overflow.

    The largest norm is sufficient because the guard is monotone in `n_max`.
    Data already stored in the first-pass format needs no conversion guard.
    """
    if not (n_max == n_max and n_max > 0.0 and n_max != float("inf")):
        return False

    if fmt.u_t == 0.0:
        return True

    return (
        kappa_bar(d) * n_max
        <= fmt.omega_t * (1.0 - GUARD_MARGIN)
    )


def product_ok(qn: float, cn_max: float) -> bool:
    """Return whether the norm product is safe from binary32 overflow.

    The comparison is performed in binary64 so the product guard cannot pass
    because of downward binary32 rounding.
    """
    if not (qn > 0.0 and cn_max > 0.0):
        return False

    return qn * cn_max <= PROD_MAX

# --- certification self-test -------------------------------------------------

_SELF_TEST: str | None = None


def self_test() -> str | None:
    """Check data- and device-independent invariants of the closed-form bound.

    Returns `None` on success or the first failure message. These checks cover
    monotonicity, representative golden values, hardware-envelope dominance,
    guard behavior, upward rounding, and the certified arithmetic restrictions.
    """
    for _name, _got, _want in (
        ("FLOOR_COS", FLOOR_COS, math.ldexp(1.0, -44)),
        ("KAPPA_INFLATE", KAPPA_INFLATE, math.ldexp(1.0, -40)),
        ("EVAL_INFLATE", EVAL_INFLATE, math.ldexp(1.0, -30)),
    ):
        if _got != _want:
            return (
                f"safety constant {_name} is {_got!r}, expected {_want!r}. "
                f"These are pure margin — smaller is unsafe — and each covers "
                f"an error the formula does not otherwise model. Changing one "
                f"changes the bound; see Sec.12."
            )

    # 1. The error allowance must increase with dot-product length.
    prev = -1.0
    for d in (64, 128, 384, 768, 1536, 4096, 16384, 65536, 2 ** 20):
        e = float(eps_cos(d, U16, 0.0, eta_q=ETA16))
        if not (e > prev):
            return f"eps is not increasing in d: eps({d}) = {e:.6e} <= {prev:.6e}"
        prev = e

    # 2. Increasing conversion error or scale bounds must not shrink `eps`.
    base = float(eps_cos(768, U16, 0.0, eta_q=ETA16))
    if not (float(eps_cos(768, U16, U16, eta_q=ETA16, eta_c=ETA16)) > base):
        return "eps is not increasing in u_c"
    if not (float(eps_cos(768, U16, 0.0, eta_q=ETA16, rho_q=10.0)) > base):
        return "Psi is not increasing in rho_q"
    if not (float(eps_cos(768, U16, 0.0, eta_q=ETA16, sigma_max=10.0)) > base):
        return "Psi is not increasing in sigma_max"
    if not (float(eps_cos(768, U16, 0.0, eta_q=ETA16, half_out=True)) > base):
        return "the fp16-written-Gram route is not charged more than the fused one"

    # 3. The data-independent term must exceed the unavoidable input-conversion floor.
    for u_q, u_c in ((U16, 0.0), (U16, U16), (UBF, UBF)):
        c = C_const(768, u_q, u_c)
        if not (c > u_q + u_c + u_q * u_c):
            return f"C(768) = {c:.6e} is below the conversion floor for ({u_q}, {u_c})"

    # 4. The conservative hardware coefficient must dominate independently
    # computed generation-specific coefficients.
    VOLTA_COEFF = 3.0
    if not (C_HW_ENVELOPE > VOLTA_COEFF):
        return (f"C_HW_ENVELOPE = {C_HW_ENVELOPE} does not dominate Volta's "
                f"published beta/b = {VOLTA_COEFF}; the envelope is not one")

    for d in (64, 768, 4096, 2 ** 20):
        x = VOLTA_COEFF * d * U
        volta = x * (1.0 + x)
        if not (kappa_acc(d, C_HW_ENVELOPE) > volta):
            return (f"kappa_acc at the envelope does not dominate an "
                    f"independently computed Volta value at d = {d}")
        if not (kappa_acc(d, C_HW_ENVELOPE) > kappa_acc(d, C_HW_AMPERE)):
            return f"the c_hw = 4 envelope does not dominate the Ampere model at d = {d}"

    # 5. Golden values catch dropped or altered terms that structural checks cannot.
    for d, want in ((384, 6.039e-4), (768, 7.187119e-4), (1536, 9.482e-4)):
        got = float(eps_cos(d, U16, 0.0, eta_q=ETA16))
        if not (abs(got - want) / want < 6e-4):
            return (f"eps_cos({d}) = {got:.7e} does not match "
                    f"expected {want:.7e}")

    for (dd, nq, nc), want in (((768, 1.0, 1.0), 1.208171e-03),
                               ((768, 27.7, 27.7), 9.257739e-01)):
        got = float(eps_dot(dd, U16, U16, nq, nc, eta_q=ETA16, eta_c=ETA16))
        if not (abs(got - want) / want < 1e-5):
            return (f"eps_dot({dd}, {nq}, {nc}) = {got:.7e} != {want:.7e}")

    # Asymmetric formats ensure query and corpus norm terms are not swapped.
    a = float(eps_dot(768, U16, 0.0, 1.0, 10.0, eta_q=ETA16, eta_c=0.0))
    b = float(eps_dot(768, U16, 0.0, 10.0, 1.0, eta_q=ETA16, eta_c=0.0))
    if not (a > b):
        return ("eps_dot does not distinguish Qn from Cn under asymmetric "
                "formats; the two norms are interchangeable")

    # Dot-product error should scale approximately with the norm product.
    base = float(eps_dot(768, U16, U16, 1.0, 1.0, eta_q=ETA16, eta_c=ETA16))
    scaled = float(eps_dot(768, U16, U16, 10.0, 10.0, eta_q=ETA16, eta_c=ETA16))
    if not (0.98 <= scaled / (base * 100.0) <= 1.02):
        return (f"eps_dot does not scale with Qn*Cn: {scaled:.4e} against "
                f"{base * 100.0:.4e} expected")

    # 6. Guard-side `kappa_bar` must remain a small outward inflation of `kappa`.
    for d in (64, 768, 4096, 2 ** 20):
        k, kb = kappa(d), kappa_bar(d)
        if not (kb > k):
            return f"kappa_bar({d}) = {kb!r} does not exceed kappa({d}) = {k!r}"
        if not (kb < k * (1.0 + math.ldexp(1.0, -39))):
            return f"kappa_bar({d}) is inflated by far more than 2**-40"

    # 7. `eps_final` must never round below its binary64 target.
    import numpy as np

    rng = np.random.default_rng(0)
    Es = np.concatenate([
        np.geomspace(1e-9, 1e-1, 400),
        rng.random(400) * 1e-3,
    ])
    for E in Es:
        B = 1.0 + E
        want = (float(E) * (1.0 + U) + U * B) / (1.0 - U) * (1.0 + EVAL_INFLATE)
        got = float(eps_final(float(E), B))
        if not (got >= want):
            return (f"eps_final rounded DOWN at E={E!r}: "
                    f"{got!r} < {want!r}")

    # 8. Guards must fail closed on NaN and invalid norms.
    if norm_range_ok(float("nan")):
        return "the norm range guard accepted NaN"
    if overflow_ok(768, float("nan"), FP16):
        return "the overflow guard accepted NaN"
    if product_ok(float("nan"), 1.0) or product_ok(1.0, float("nan")):
        return "the norm product guard accepted NaN"
    if norm_range_ok(float("inf")) or norm_range_ok(0.0):
        return "the norm range guard accepted a non-finite or zero norm"

    # 9. Certified expressions must use only the permitted arithmetic operations.
    bad = _binary_power_users()
    if bad:
        return f"certified arithmetic violation: {', '.join(bad)}"

    return None

_CERTIFIED_FUNCS = (
    "gamma", "kappa", "kappa_bar", "kappa_acc", "theta", "bracket",
    "Lambda", "C_const", "Psi", "eps_final", "eps_cos", "eps_dot",
    "overflow_ok", "product_ok", "_eps_dot_inner", "_psi_inner",
    "ceil_pow2",
)


# Every global or attribute name used by the certified path must be explicitly
# allowed. This makes new helpers and library operations fail certification
# until their rounding behavior has been reviewed.
_ALLOWED_NAMES = frozenset({
    # binary64 primitives the model is stated for
    "math", "sqrt", "fabs", "ldexp", "frexp", "isfinite", "isnan", "isinf",
    # numpy, for the array paths
    "np", "numpy", "asarray", "float64", "float32", "shape", "full",
    "errstate", "astype", "nextafter", "where", "isnan", "isfinite",
    "zeros_like", "full_like", "all", "any", "ones", "abs", "maximum",
    "minimum", "inf", "nan", "atleast_1d", "result_type", "ndarray",
    # this module's own certified helpers and constants
    *_CERTIFIED_FUNCS,
    "U", "U16", "U32", "ETA16", "ETA32", "LAMBDA16", "LAMBDA32", "OMEGA16",
    "C_HW_ENVELOPE", "FLOOR_COS", "KAPPA_INFLATE", "EVAL_INFLATE",
    "D_MIN", "D_MAX", "NORM_MIN", "NORM_MAX", "PROD_MAX",
    "PRUNE_MIN_QNORM", "PRUNE_MIN_CNORM",
    "dimension_ok", "norm_range_ok", "_refuse_bad_inputs", "ceil_pow2",
    "Format", "FP16", "BF16", "EXACT",
    # `Format`'s attributes, loaded as LOAD_ATTR on a format object.
    "u_t", "eta_t", "lam_t", "omega_t", "name",
    "GUARD_MARGIN",
    # builtins that carry no rounding of their own
    "bool", "float", "int", "len", "range", "tuple", "list", "min", "max",
    "ValueError", "TypeError", "isinstance",
})


def _binary_power_users() -> list:
    """Certified functions that use arithmetic outside the model, or a name
    that has not been vetted.

    This method restricts the bound's evaluation to `+ - * / sqrt` in binary64. This
    enforces it two ways: no `**`/`BINARY_POWER` opcode anywhere, and every
    loaded name drawn from `_ALLOWED_NAMES`.
    """
    out = []
    g = globals()

    for name in _CERTIFIED_FUNCS:
        fn = g.get(name)
        if fn is None:
            out.append(f"{name} (missing)")
            continue

        try:
            code = fn.__code__
        except AttributeError:
            continue

        bad = _code_uses_power(code)
        if bad:
            out.append(f"{name} ({bad})")

    return out


def _code_uses_power(code) -> str:
    """`""` if `code` stays inside the certified arithmetic, else why not.

    Checks nested code objects too — comprehensions, lambdas and generators
    compile to their own, and a banned operation hidden in one would otherwise
    be invisible.
    """
    import dis

    for ins in dis.get_instructions(code):
        if ins.opname == "BINARY_POWER" or (
            ins.opname in ("BINARY_OP", "BINARY_OPERATION")
            and isinstance(ins.argrepr, str)
            and "**" in ins.argrepr
        ):
            return "uses **"

        if (
            ins.opname in ("LOAD_METHOD", "LOAD_ATTR", "LOAD_GLOBAL", "LOAD_NAME")
            and isinstance(ins.argval, str)
            and ins.argval not in _ALLOWED_NAMES
            and not ins.argval.startswith("__")
        ):
            return f"loads un-vetted name {ins.argval!r}"

    for const in code.co_consts:
        if hasattr(const, "co_code"):
            bad = _code_uses_power(const)
            if bad:
                return bad

    return ""


def reset() -> None:
    """Clear the cached certification result so the next call recomputes it."""
    global _SELF_TEST
    _SELF_TEST = None


def certified() -> str | None:
    """Return `None` if certified, otherwise the cached failure message."""
    global _SELF_TEST
    if _SELF_TEST is None:
        _SELF_TEST = self_test() or ""
    return _SELF_TEST or None