"""Two-pass dense scoring with safe reduced-precision pruning.

Pass one uses half-precision inputs to cheaply bound the best score each query
row can obtain from a corpus slice. A row is pruned only when

    approx_max[q] + eps[q] < threshold[q]

where `eps[q]` is a closed-form upper bound on the difference between pass
one and the exact fp32 scoring path. Non-finite or otherwise unsupported cases
are forced live rather than pruned.

The bound requires no measured conversion residuals. Its data-independent
terms depend on dimension and numeric format, while its absolute-error terms
depend only on the scale and norm summaries needed by the scoring path.

Rows that remain live are recomputed with the ordinary fp32 `_scores` path.
Because changing the GEMM height can change cuBLAS's accumulation order, live
rows are padded to candidate heights and each execution shape is verified
bit-for-bit against the full-height product before its scores are trusted.

On CUDA, pruning requires the fused Triton pass-one kernel. `_gemm_rowmax`
accumulates the half-input product in float32 and computes the scaled row max
without materializing the full Gram matrix. If the fused path cannot be used,
the slice is conservatively kept live and falls back to ordinary fp32 scoring.
"""

from __future__ import annotations

import logging
import math
import os
import threading

logger = logging.getLogger(__name__)

# The bound itself lives in `closed_form`
from . import closed_form as _cf

# Re-exported because the rest of the module and its tests read them constantly.
U = _cf.U
U16 = _cf.U16

# Run-level certification state.
# None = not attempted, "" = passed, otherwise the failure reason.
_CERTIFIED: str | None = None

# Execution configurations certified during this run.
_CERTIFIED_KEYS: set = set()

def certified() -> str | None:
    """Return the run-level certification status.

    `None` means unattempted, `""` means passed, otherwise the value is the
    failure reason.
    """
    return _CERTIFIED


def is_certified(key: tuple) -> bool:
    """Return whether this execution configuration was certified this run."""
    return key in _CERTIFIED_KEYS

# --- runtime hardware certification ------------------------------------------

# Only GPU generations covered by the tensor-core accumulation model are
# admitted. Unknown generations are refused rather than extrapolated.
#
# Runtime probes then check the actual device for behavior inconsistent with
# the model; passing a finite probe cannot prove the model for every input.
_R5_GENERATIONS = {
    (7, 0): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 7): "Ampere",
    (8, 9): "Ada",
    (9, 0): "Hopper",
    (10, 0): "Blackwell",
    (12, 0): "Blackwell",
}

def probe_device_generation(device) -> str | None:
    """Return whether this device generation is covered by the accumulation model."""
    import torch

    if not str(device).startswith("cuda"):
        # CPU pass one widens the half inputs and accumulates in float32.
        return None

    cap = torch.cuda.get_device_capability(torch.device(device))
    if cap not in _R5_GENERATIONS:
        name = torch.cuda.get_device_name(torch.device(device))
        return (
            f"the two-pass has no accumulation model for {name} "
            f"(compute capability {cap[0]}.{cap[1]}); pass one is refused"
        )

    return None

def probe_conversion(device, dtype=None) -> str | None:
    """Check that conversion to the first-pass format matches the bound's model.

    For float16, the conversion must preserve subnormals and use
    round-to-nearest-even. A mismatch refuses pass one rather than applying a
    bound derived for different conversion behavior.
    """
    import torch

    dt = dtype or torch.float16
    dev = torch.device(device)

    # Verify that exactly representable fp16 subnormals survive conversion.
    sub = torch.tensor([2.0 ** -20, 2.0 ** -24], dtype=torch.float32, device=dev)
    got = sub.to(dt).float()
    if not bool(torch.equal(got, sub)):
        return (
            f"the {dt} conversion on {device} is not subnormal-preserving: "
            f"{sub.tolist()} came back as {got.tolist()}; pass one is refused"
        )

    if dt is torch.float16:
        # Distinguish round-to-nearest-even from truncation at the fp16 limit.
        edge = torch.tensor([65519.0, 65520.0], dtype=torch.float32, device=dev)
        e = edge.to(dt).float()
        if not (float(e[0]) == 65504.0 and math.isinf(float(e[1]))):
            return (
                f"the float16 conversion on {device} is not round-to-nearest-even "
                f"at the top of the range: 65519 -> {float(e[0])}, "
                f"65520 -> {float(e[1])} (expected 65504 and inf); "
                f"pass one is refused"
            )

    return None


# The constructions `probe_accumulation` drives through the first pass. Each is
# `(name, builder)`; the builder returns a `(d,)` float32 pair `(q, c)` whose
# products are all the SAME SIGN, which is the adversarial case for a
# truncating accumulator -- no cancellation, so every aligned addend is pushed
# away from zero and the discarded bits all accumulate in one direction.
def _accumulation_probes(d: int):
    import numpy as np

    rng = np.random.default_rng(0x5EED)
    out = []
    # 1. One dominant product sets the block anchor high; the tail is then
    #    truncated against a quantum far above its own magnitude.
    q = np.full(d, 2.0 ** -13, dtype=np.float32)
    c = np.full(d, 1.0, dtype=np.float32)
    q[0] = 1.0
    out.append(("dominant+tail", q, c))
    # 2. Uniform positive products: the running sum grows monotonically, so the
    #    anchor rises through the accumulation and every later addend loses more.
    q = (rng.random(d).astype(np.float32) + 0.5)
    c = (rng.random(d).astype(np.float32) + 0.5)
    out.append(("same-sign uniform", q, c))
    # 3. A geometric spread from unit magnitude down to fp16's minimum normal,
    #    largest first, which maximises how much of each addend falls below the
    #    anchor's quantum.
    q = np.geomspace(1.0, 2.0 ** -14, d).astype(np.float32)
    c = np.ones(d, dtype=np.float32)
    out.append(("geometric decay", q, c))
    return out

def probe_accumulation(d: int, device, rowmax_fn=None) -> str | None:
    """Check the first-pass accumulation against adversarial finite inputs.

    Each construction is evaluated by the actual pass-one kernel and compared
    with a float64 reference formed from the same fp16-rounded operands. The
    measured error must remain below the largest accumulation error admitted
    by the hardware model.

    This probe can detect behavior inconsistent with the model, but cannot
    establish the bound for every possible operand pattern.
    """
    import numpy as np
    import torch

    fn = rowmax_fn or fused_rowmax
    dev = torch.device(device)
    probes = _accumulation_probes(d)

    # Run each construction separately with identical corpus columns. The row
    # max is therefore the construction's dot product rather than a competing
    # column.
    n_q = n_c = 128
    worst = 0.0
    worst_label = ""

    for name, q, c in probes:
        Q = np.repeat(q[None, :], n_q, axis=0)
        C = np.repeat(c[None, :], n_c, axis=0)
        Qh = torch.from_numpy(Q).to(dev).half().contiguous()
        Ch = torch.from_numpy(C).to(dev).half().contiguous()
        cs = torch.ones(n_c, dtype=torch.float32, device=dev)

        try:
            got = fn(Qh, Ch, cs)
        except Exception as exc:                    # noqa: BLE001
            return f"the accumulation probe could not run pass one: {exc!r}"

        if got is None:
            _STATS["probe_accum_worst_rel"] = None
            if not str(device).startswith("cuda"):
                # CPU pass one widens the half inputs and accumulates in float32.
                return None
            return (
                f"the fused kernel declined the accumulation probe's shape on "
                f"{device}; the accumulation model was not tested, so pass one "
                f"is refused"
            )

        if got.shape != (n_q,):
            return (
                f"the accumulation probe on {device} returned shape "
                f"{tuple(got.shape)}, expected ({n_q},); pass one is refused"
            )

        # Compare against the same fp16-rounded inputs so this measures
        # accumulation error rather than conversion error.
        qh64 = Qh.double().cpu().numpy()[0]
        ch64 = Ch.double().cpu().numpy()[0]
        ref = float(qh64 @ ch64)
        absum = float(np.abs(qh64) @ np.abs(ch64))

        if not (np.isfinite(ref) and np.isfinite(absum)):
            return (
                f"the accumulation probe's float64 reference for '{name}' is "
                f"not finite (ref={ref!r}, absum={absum!r}); pass one is refused"
            )

        measured = got.double().cpu().numpy()

        # A non-finite result cannot establish the accumulation model. NaN
        # comparisons are false, so allowing one through could leave `worst`
        # unchanged and incorrectly make the probe appear to pass.
        if not np.isfinite(measured).all():
            n_bad = int((~np.isfinite(measured)).sum())
            return (
                f"the first pass on {device} returned {n_bad} of "
                f"{measured.size} non-finite values for the '{name}' "
                f"accumulation construction; pass one is refused"
            )

        rel = float(np.abs(measured - ref).max()) / max(
            absum, np.finfo(np.float64).tiny
        )
        if not np.isfinite(rel):
            return (
                f"the accumulation probe's relative error for '{name}' on "
                f"{device} is {rel!r}; pass one is refused"
            )

        if rel > worst:
            worst, worst_label = rel, name

    fitted = _cf.kappa_acc(d, _cf.C_HW_AMPERE)
    volta = _cf.kappa_acc(d, 3.0)
    envelope = _cf.kappa_acc(d, _cf.C_HW_ENVELOPE)

    # Preserve the worst accumulation error observed across certifications.
    _STATS["probe_accum_worst_rel"] = max(
        _STATS.get("probe_accum_worst_rel") or 0.0, worst
    )
    _STATS["probe_accum_fitted"] = fitted

    if worst > volta:
        return (
            f"the first pass on {device} accumulates worse than the largest "
            f"published generation model at d={d}: the '{worst_label}' "
            f"construction measured {worst:.3e}, against {volta:.3e} for "
            f"Volta and {envelope:.3e} for the conservative envelope; "
            f"pass one is refused"
        )

    if worst > fitted:
        logger.warning(
            "two-pass: the first pass measured %.3e relative accumulation error "
            "at d=%d, above the Ampere/Ada model's %.3e but inside the %.3e "
            "conservative envelope.",
            worst, d, fitted, envelope,
        )
    else:
        logger.info(
            "two-pass: accumulation probe passed on %s at d=%d — worst relative "
            "error %.3e against the Ampere/Ada model's %.3e "
            "(envelope %.3e).",
            device, d, worst, fitted, envelope,
        )

    return None


def probe_scales(col_scale, cn, row_scale=None, qn=None) -> str | None:
    """Check that the scales applied by pass one match the corresponding norms.

    Each scale must be a valid rounded reciprocal of the same computed norm
    used by the exact path. A mismatch means the two paths are scoring
    different quantities, so pass one is refused.
    """

    def _check(scale, norms, axis):
        if scale is None:
            return None

        if norms is None:
            return (
                f"pass one's {axis} scale was provided without the "
                f"corresponding norms; pass one is refused"
            )

        bad, why = scale_mismatch(scale, norms)
        if why:
            n = int(bad.sum())
            return (
                f"pass one's {axis} scale does not match the corresponding "
                f"computed norm on {n} of {scale.numel()} entries: {why}"
            )

        return None

    return _check(col_scale, cn, "column") or _check(row_scale, qn, "query")

def scale_mismatch(scale, norms):
    """Return entries whose scale is not a valid rounded reciprocal of its norm.

    For each finite positive norm `N`, the corresponding scale must satisfy

        (1-u) / N <= scale <= (1+u) / N

    where `u` is binary32 unit roundoff. The interval is rounded outward in
    binary64 so a legitimate binary32 reciprocal is never rejected.

    Invalid norms are handled by the separate norm guards and are therefore
    excluded from the mismatch mask.
    """
    import torch

    # Scales and norms must correspond entry-for-entry; broadcasting would make
    # the reciprocal check meaningless.
    if scale.shape != norms.shape:
        return (
            torch.ones(norms.shape, dtype=torch.bool, device=norms.device),
            f"scale has shape {tuple(scale.shape)} against norms of shape "
            f"{tuple(norms.shape)}; scale and norm entries must correspond"
        )

    # Flatten so scalar tensors and vectors share the same reporting path.
    n = norms.reshape(-1).double()
    s = scale.reshape(-1).double()
    scale = scale.reshape(-1)
    norms = norms.reshape(-1)

    ok_norm = torch.isfinite(n) & (n > 0.0)

    # Round the admissible interval outward.
    lo = torch.nextafter((1.0 - _cf.U) / n, torch.zeros_like(n))
    hi = torch.nextafter(
        (1.0 + _cf.U) / n,
        torch.full_like(n, float("inf")),
    )

    ok = torch.isfinite(s) & (s > 0.0) & (s >= lo) & (s <= hi)
    bad = ok_norm & ~ok

    if not bool(bad.any()):
        return bad, None

    i = int(bad.double().argmax())
    return (
        bad,
        f"scale[{i}] = {float(scale[i])!r} is outside "
        f"[{float(lo[i])!r}, {float(hi[i])!r}] for norm "
        f"{float(norms[i])!r}",
    )
# --- continuous live-row audit ------------------------------------------------
#
# Live rows are scored exactly by pass two, so their exact slice maxima can be
# compared against `upper = approx + eps` without another GEMM. This provides a
# continuous check of the bound on real production inputs.
#
# Pruned rows are not observed by this audit, so it cannot detect every unsafe
# pruning decision and is not a substitute for an audit that scores dead rows.

_AUDIT_RATE: int | None = None
_AUDIT_WARNED = False


def audit_rate() -> int:
    """Return the live-row audit sampling rate.

    `NOVA_BF_TWOPASS_AUDIT=N` audits one eligible slice in N. Zero disables
    auditing; the default is 1 (every eligible slice).
    """
    global _AUDIT_RATE

    if _AUDIT_RATE is None:
        raw = os.environ.get("NOVA_BF_TWOPASS_AUDIT", "1")
        try:
            v = int(raw)
            _AUDIT_RATE = v if v >= 0 else 1
        except ValueError:
            _AUDIT_RATE = 1

    return _AUDIT_RATE
def audit_decisions(Q, Cb, metric, live, thr, q_norms=None) -> dict:
    """Exactly score every row and grade the pass-one pruning decisions.

    A row should remain live when the exact slice maximum is at or above its
    threshold. Equality counts as live because pruning uses the strict rule
    `upper < threshold`.

    This audit observes both kept and pruned rows, but costs the full-height
    exact GEMM that two-pass pruning is intended to avoid. It is therefore a
    verification tool rather than a production safety guard.
    """
    import torch

    n = int(Q.shape[0])
    if n == 0 or int(Cb.shape[0]) == 0:
        return {}

    if live.shape != (n,):
        raise ValueError(
            f"two-pass decision audit: live has shape {tuple(live.shape)}, "
            f"expected ({n},)"
        )

    if thr.numel() == 1:
        thr_v = thr.reshape(1).expand(n)
    elif thr.shape == (n,):
        thr_v = thr
    else:
        raise ValueError(
            f"two-pass decision audit: threshold has shape {tuple(thr.shape)}, "
            f"expected scalar or ({n},)"
        )

    from .compute import _scores

    # This optional audit requires an additional full-height exact GEMM. An OOM
    # therefore skips the audit rather than affecting the production result.
    try:
        exact = _scores(Q, Cb, metric, q_norms, scale_in_packer=False)
        top = exact.amax(dim=1)
        del exact
    except Exception as exc:                       # noqa: BLE001
        if not is_oom(exc):
            raise
        _STATS["dead_audit_oom"] += 1
        logger.warning(
            "two-pass: the decision audit could not allocate its exact GEMM "
            "for a %d x %d slice; that slice is not graded",
            int(Q.shape[0]), int(Cb.shape[0]),
        )
        return {}

    # Only finite exact scores and thresholds are graded.
    ok = torch.isfinite(top) & torch.isfinite(thr_v)
    should_live = ok & (top >= thr_v)

    correct_prune = int((ok & ~live & ~should_live).sum())
    false_prune = ok & ~live & should_live
    n_false = int(false_prune.sum())
    correct_live = int((ok & live & should_live).sum())
    wasted_live = int((ok & live & ~should_live).sum())

    _STATS["dead_audit_members"] += 1
    _STATS["dead_audit_rows"] += int(ok.sum())
    _STATS["audit_correct_prune"] += correct_prune
    _STATS["audit_correct_live"] += correct_live
    _STATS["audit_wasted_live"] += wasted_live

    if n_false:
        worst = float((top - thr_v)[false_prune].max())
        _STATS["dead_audit_violations"] += n_false
        _STATS["dead_audit_worst"] = max(
            _STATS.get("dead_audit_worst") or 0.0, worst
        )
        disable(
            f"the decision audit found {n_false} rows pruned by pass one "
            f"that contain a candidate at or above their threshold, by as "
            f"much as {worst:.3e}"
        )
        logger.error(
            "two-pass DECISION AUDIT FAILURE: pass one pruned %d of %d graded "
            "rows that contain a candidate reaching their threshold, by as "
            "much as %.3e. Treat this run's output as incomplete.",
            n_false, int(ok.sum()), worst,
        )

    return {
        "checked": int(ok.sum()),
        "correct_prune": correct_prune,
        "false_prune": n_false,
        "correct_live": correct_live,
        "wasted_live": wasted_live,
    }


def dead_audit_rate() -> int:
    """Sample one slice in N for the dead-row audit. 0 (the default) disables.

    Off by default because it costs the exact GEMM the two-pass exists to
    avoid. `NOVA_BF_TWOPASS_DEAD_AUDIT=N` turns it on for a verification run.
    """
    raw = os.environ.get("NOVA_BF_TWOPASS_DEAD_AUDIT", "0")
    try:
        v = int(raw)
        return v if v >= 0 else 0
    except ValueError:
        return 0

def audit_live_rows(scores, n_live: int, idx, upper, row_scale=None,
                    eps_used=None) -> int:
    """Check `upper >= max_c s_e` on rows scored exactly.

    `scores` contains the padded exact scores; only the first `n_live` rows are
    real. `idx` maps those rows back to `upper` and `row_scale`.

    If `row_scale` is provided, apply it before comparing against `upper`.
    Non-finite rows are excluded from the audit.
    """
    global _AUDIT_WARNED
    import torch

    if n_live <= 0 or scores is None:
        return 0

    # Prevent truncated inputs from silently broadcasting during the audit.
    if scores.shape[0] < n_live or idx.shape[0] < n_live:
        raise ValueError(
            f"two-pass audit: n_live={n_live} exceeds scores rows "
            f"{scores.shape[0]} or idx length {idx.shape[0]}; the audit would "
            f"silently broadcast one row's score across the others")

    _STATS["audit_slices"] += 1
    live_idx = idx[:n_live]

    # Ensure every live-row index addresses `upper`.
    if live_idx.numel():
        highest = int(live_idx.max())
        if highest >= upper.shape[0]:
            raise ValueError(
                f"two-pass audit: idx addresses row {highest} of an upper "
                f"bound with only {upper.shape[0]} rows")

    top = scores[:n_live].amax(dim=1)
    if row_scale is not None:
        top = top * row_scale.index_select(0, live_idx)

    ub = upper.index_select(0, live_idx)
    ok = torch.isfinite(top) & torch.isfinite(ub)
    bad = ok & ~(ub >= top)
    n_bad = int(bad.sum())

    _STATS["audit_rows"] += int(ok.sum())

    # Record the smallest remaining bound margin relative to `eps`.
    if bool(ok.any()) and eps_used:
        closest = float((ub - top)[ok].min()) / eps_used
        prev = _STATS.get("audit_closest_margin")
        _STATS["audit_closest_margin"] = (
            closest if prev is None else min(prev, closest))

    if n_bad:
        worst = float((top - ub)[bad].max())
        _STATS["audit_violations"] += n_bad
        _STATS["audit_worst"] = max(
            _STATS.get("audit_worst") or 0.0, worst)

        # A confirmed violation disables further pruning.
        disable(
            f"the continuous audit found {n_bad} live rows whose exact top "
            f"score exceeds their upper bound, by as much as {worst:.3e}"
        )

        if not _AUDIT_WARNED:
            _AUDIT_WARNED = True
            logger.error(
                "two-pass AUDIT FAILURE: %d of %d live rows on this slice have "
                "an exact top score ABOVE their upper bound, by as much as "
                "%.3e. The bound does not hold on this machine for this data. "
                "Live rows are still scored exactly so THESE results are "
                "correct, but a row that was PRUNED under the same bound may "
                "have been lost. Re-run with NOVA_BF_TWOPASS=0 to get ground "
                "truth, and treat this run's output as suspect.",
                n_bad, int(ok.sum()), worst,
            )

    return n_bad

def exact_math_mode_flags(device) -> str | None:
    """Check that the exact CUDA path uses the arithmetic assumed by the bound.

    This cheap check runs per slice because PyTorch's precision settings are
    mutable global state. Non-IEEE or unknown configurations refuse pass one.
    """
    import torch

    if not str(device).startswith("cuda"):
        return None

    # cuBLAS single-precision emulation is outside the exact-path model.
    emu = os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION")
    if emu is not None and emu != "0":
        return (f"CUBLAS_EMULATE_SINGLE_PRECISION={emu!r} enables cuBLAS "
                f"single-precision emulation. bf16x9 and friends are not the "
                f"IEEE binary32 arithmetic (R2) describes, however accurate "
                f"they are, and the `1 + 2**-13` probe cannot see them")

    strategy = os.environ.get("CUBLAS_EMULATION_STRATEGY")
    if strategy:
        return (f"CUBLAS_EMULATION_STRATEGY={strategy!r} is set; even with the "
                f"enable flag unset this signals an emulation configuration "
                f"P3 does not admit")

    override = os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
    if override and override not in ("0", "false", "False", ""):
        return (
            f"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE={override!r} is set, which "
            f"makes cuBLAS matmul default to TF32. (R2) and the `gamma_d` "
            f"term are stated for IEEE binary32 round-to-nearest"
        )

    m = torch.backends.cuda.matmul
    try:
        prec = m.fp32_precision
    except Exception as exc:           # noqa: BLE001
        return (f"torch cannot report `matmul.fp32_precision` ({exc!r}), so "
                f"the exact pass's arithmetic mode is unknown; P3 needs it "
                f"KNOWN, not assumed")

    # Accept only settings that resolve to full binary32 computation.
    if prec not in ("ieee", "none"):
        return (f"the exact pass would run with matmul.fp32_precision="
                f"{prec!r}. (R2) and the `gamma_d` term are stated for IEEE "
                f"binary32 round-to-nearest; anything else adds an input "
                f"conversion error the bound does not carry")

    try:
        if m.allow_tf32:
            return ("the exact pass would run with "
                    "torch.backends.cuda.matmul.allow_tf32=True, which is not "
                    "the binary32 arithmetic (R2) describes")
    except Exception:                  # noqa: BLE001
        # An unreadable legacy/new API state is not evidence that TF32 is off.
        return ("torch reports a mixed legacy/new TF32 API state for cuBLAS "
                "matmul, so `allow_tf32` cannot be read; P3 is not established")

    return None


def probe_exact_math_mode(device) -> str | None:
    """Check that the exact CUDA GEMM uses IEEE binary32 round-to-nearest.

    The configured precision mode is checked first, then a small GEMM provides
    a runtime check against reduced-precision input arithmetic. This probe can
    detect inconsistent behavior but cannot establish the arithmetic for every
    possible GEMM shape.
    """
    import torch

    if not str(device).startswith("cuda"):
        return None                    # CPU fp32 is IEEE binary32 RN

    why = exact_math_mode_flags(device)
    if why:
        return why

    dev = torch.device(device)
    x = 1.0 + 2.0 ** -13               # exact in binary32; 1.0 in TF32 and bf16
    try:
        A = torch.zeros(64, 64, dtype=torch.float32, device=dev)
        A[:, 0] = x
        B = torch.zeros(64, 64, dtype=torch.float32, device=dev)
        B[0, :] = 1.0
        got = float((A @ B)[0, 0])
    except Exception as exc:                       # noqa: BLE001
        return (
            f"the exact-pass arithmetic probe could not run on {device} "
            f"({exc!r}). P3 needs the arithmetic mode KNOWN, and a probe that "
            f"did not complete has established nothing"
        )
    if got != x:
        return (
            f"the exact-pass GEMM on {device} returned {got!r} for a product "
            f"whose binary32 value is {x!r}. `1 + 2**-13` survives binary32 "
            f"and collapses to 1.0 under TF32 (10 mantissa bits) or bfloat16 "
            f"(7), so this GEMM is not doing the arithmetic (R2) assumes -- "
            f"whatever the precision flags report"
        )
    return None


def certify_closed_form(d: int, device, rowmax_fn=None) -> str | None:
    """Run the certification checks required before pass one may prune.

    Checks are ordered from cheapest to most expensive and return the first
    failure reason. Checks that require a real corpus slice run separately.
    """
    return (
        _cf.certified()
        or probe_device_generation(device)
        or probe_exact_math_mode(device)
        or probe_conversion(device)
        or probe_accumulation(d, device, rowmax_fn)
    )


def accumulator_is_ours(device, used_fused: bool) -> bool:
    """Return whether pass-one accumulation is implemented by a trusted path.

    The bound requires float32 accumulation. The fused Triton kernel and CPU
    fallback enforce that directly; the unfused CUDA/cuBLAS path is therefore
    not allowed to prune.
    """
    return used_fused or not str(device).startswith("cuda")


def note_certified(reason: str | None, key: tuple | None = None) -> None:
    """Record a certification outcome."""
    global _CERTIFIED

    if reason:
        _CERTIFIED = reason
        return
    _CERTIFIED = ""
    if key is not None:
        _CERTIFIED_KEYS.add(key)

def norm_guard(norms):
    """Return rows whose norms fall outside the bound's admissible range.

    Finite norms must satisfy

        NORM_MIN <= norm <= NORM_MAX

    Non-finite values are refused automatically because they fail both
    comparisons.
    """
    import torch

    if norms.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool, device=norms.device)

    ok = (norms >= _cf.NORM_MIN) & (norms <= _cf.NORM_MAX)
    return ~ok
def overflow_guard(d: int, norms, fmt) -> bool:
    """Return whether conversion to the first-pass format cannot overflow.

    The check uses the largest norm on the axis because the overflow condition
    is monotone in the norm.
    """
    if fmt.u_t == 0.0:
        # No conversion is performed for an already-exact input format.
        return True
    if norms.numel() == 0:
        return True
    return _cf.overflow_ok(d, float(norms.max()), fmt)
def overflow_mask(d: int, norms, fmt):
    """Return query rows that cannot safely convert to the first-pass format.

    Unlike `overflow_guard`, this check is per row so one invalid query does
    not disable pruning for every query in the slice.
    """
    import torch

    if norms.numel() == 0:
        return torch.zeros_like(norms, dtype=torch.bool)

    # Invalid norms are always refused.
    unusable = ~(torch.isfinite(norms) & (norms > 0.0))

    if fmt.u_t == 0.0:
        return unusable

    limit = fmt.omega_t * (1.0 - _cf.GUARD_MARGIN)
    return unusable | ~(norms.double() * _cf.kappa_bar(d) <= limit)
def product_guard(qn, cn_max: float):
    """Return query rows whose query/corpus norm product can exceed `PROD_MAX`.

    The check is per query row so one oversized or invalid query does not
    disable pruning for the whole slice.
    """
    import torch
    import math as _m

    if not (_m.isfinite(cn_max) and cn_max > 0.0):
        # An invalid corpus norm refuses every query row.
        return torch.ones_like(qn, dtype=torch.bool)

    prod = qn.double() * float(cn_max)
    return ~(torch.isfinite(qn) & (qn > 0.0)) | ~(prod <= _cf.PROD_MAX)

def prunability_cut(qn, cn_min: float):
    """Return query rows where pass one is unlikely to prune enough to pay off.

    Very small query or corpus norms make the absolute error term large, so
    pass one is skipped for those rows even though the bound remains valid.
    """
    import torch
    import math as _m

    if not (_m.isfinite(cn_min) and cn_min >= _cf.PRUNE_MIN_CNORM):
        return torch.ones_like(qn, dtype=torch.bool)

    return ~(torch.isfinite(qn) & (qn >= _cf.PRUNE_MIN_QNORM))

# Exact live-row GEMMs are padded to candidate heights before execution.
# `_verify_shape` still checks every height actually used against the full
# query height, so these constants only choose which shapes are attempted.
PAD_QUANTUM = 1024
PAD_FLOOR = 7168
PAD_SMALL = (2048, 4096, 5120)

# The candidate ladder assumes the floor is aligned to the padding quantum.
assert PAD_FLOOR % PAD_QUANTUM == 0, \
    "PAD_FLOOR must be a multiple of PAD_QUANTUM"

# Maximum number of candidate padded heights attempted before falling back to
# the full-height path.
MAX_PAD_TRIES = 4

# Below this query height, two-pass pruning is not expected to pay for itself.
MIN_QUERY_ROWS = 32768

# Maximum live fraction for using the two-pass path by default.
DEFAULT_THRESHOLD = 0.50

_UNAVAILABLE: str | None = None
_DISABLED_REASON: str | None = None

# Warn once for malformed runtime configuration values.
_THRESHOLD_WARNED = False
_FUSE_CONFIG_WARNED = False
_ROWMAX_WARNED = False

# Disable further Triton row-max launches after a launch failure.
_ROWMAX_OFF = False

# Whether the Triton row-max kernel successfully ran during this run.
_ROWMAX_USED = False

_FUSE_OOM_WARNED = False
_VERIFY_OOM_WARNED = False
_UNCHECKED_WARNED = False

# Stop repeatedly attempting unverifiable padded shapes after sustained
# verification failure rather than continually thrashing the allocator.
MAX_UNCHECKED_SLICES = 16
_UNCHECKED_STREAK = 0


def note_unchecked_slice() -> bool:
    """Record an incomplete shape check and report whether retries must stop.

    Only an ``UNVERIFIED`` result reaches this path: a completed check, whether
    it matched or not, is evidence that the transient allocation pressure has
    passed and is recorded by :func:`note_checked_slice` instead.
    """
    global _UNCHECKED_STREAK
    _UNCHECKED_STREAK += 1
    return _UNCHECKED_STREAK >= MAX_UNCHECKED_SLICES


def note_checked_slice() -> None:
    """Reset the incomplete-verification streak after any completed check.

    This is deliberately called for cached successful shapes too.  The limit
    is for *consecutive slices* whose verification could not be completed,
    not a lifetime counter of transient allocation failures.
    """
    global _UNCHECKED_STREAK
    _UNCHECKED_STREAK = 0


def warn_unchecked_once() -> bool:
    """True the first time a slice gives up because it could not COMPLETE the
    bit-identity check; False every time after, within one run."""
    global _UNCHECKED_WARNED
    if _UNCHECKED_WARNED:
        return False
    _UNCHECKED_WARNED = True
    return True


def _load():
    import triton
    import triton.language as tl

    return triton, tl

try:
    _triton, _tl = _load()

    @_triton.jit
    def _rowmax_scaled(G, CS, OUT, stride_g, n_cols,
                       BLOCK: _tl.constexpr, IS_F16: _tl.constexpr):
        """Compute the scaled maximum of each row of `G` in one read pass."""
        row = _tl.program_id(0)
        acc = _tl.full([BLOCK], float("-inf"), _tl.float32)
        for c0 in range(0, n_cols, BLOCK):
            offs = c0 + _tl.arange(0, BLOCK)
            m = offs < n_cols
            g = _tl.load(G + row * stride_g + offs, mask=m, other=0.0)
            if IS_F16:
                g = g.to(_tl.float32)
            cs = _tl.load(CS + offs, mask=m, other=0.0)

            # Reapply the mask after multiplication so padded lanes remain -inf.
            v = _tl.where(m, g * cs, float("-inf"))
            acc = _tl.maximum(acc, v)

        _tl.store(OUT + row, _tl.max(acc))

    @_triton.jit
    def _gemm_rowmax(Q, C, CS, OUT, M, N, K,
                    stride_qm, stride_cn, stride_ot,
                    BLOCK_M: _tl.constexpr, BLOCK_N: _tl.constexpr,
                    BLOCK_K: _tl.constexpr, GROUP_M: _tl.constexpr,
                    EVEN_K: _tl.constexpr):
        """Compute pass-one GEMM tiles and reduce each row before writing.

        The fp16 product is accumulated in float32. Each program writes the scaled
        row maximum for its column tile; the caller then reduces across column
        tiles. The full approximate score matrix is never materialized.

        Wrapped row and column indices keep GEMM loads in bounds. Wrapped columns
        are masked out in the epilogue, and wrapped-row stores are masked.
        """
        pid = _tl.program_id(0)
        num_pid_m = _tl.cdiv(M, BLOCK_M)
        num_pid_n = _tl.cdiv(N, BLOCK_N)
        num_in_group = GROUP_M * num_pid_n
        group_id = pid // num_in_group
        first_m = group_id * GROUP_M
        group_m = min(num_pid_m - first_m, GROUP_M)
        pid_m = first_m + ((pid % num_in_group) % group_m)
        pid_n = (pid % num_in_group) // group_m

        offs_m = pid_m * BLOCK_M + _tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + _tl.arange(0, BLOCK_N)
        rm = offs_m % M
        rn = offs_n % N
        offs_k = _tl.arange(0, BLOCK_K)
        a_ptrs = Q + rm[:, None] * stride_qm + offs_k[None, :]
        b_ptrs = C + rn[None, :] * stride_cn + offs_k[:, None]

        acc = _tl.zeros((BLOCK_M, BLOCK_N), dtype=_tl.float32)
        for k0 in range(0, K, BLOCK_K):
            if EVEN_K:
                a = _tl.load(a_ptrs)
                b = _tl.load(b_ptrs)
            else:
                km = (k0 + offs_k) < K
                a = _tl.load(a_ptrs, mask=km[None, :], other=0.0)
                b = _tl.load(b_ptrs, mask=km[:, None], other=0.0)
            acc = _tl.dot(a, b, acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        # Exclude wrapped columns after scaling so they cannot affect the maximum.
        nm = offs_n < N
        cs = _tl.load(CS + offs_n, mask=nm, other=0.0)
        v = _tl.where(nm[None, :], acc * cs[None, :], float("-inf"))
        _tl.store(OUT + pid_n * stride_ot + offs_m, _tl.max(v, 1),
                mask=offs_m < M)

except Exception as exc:  # no triton, or a version whose API moved
    _triton = _tl = None
    _rowmax_scaled = None
    _gemm_rowmax = None
    _UNAVAILABLE = f"{type(exc).__name__}: {exc}"


def enabled() -> bool:
    """Is the two-pass permitted at all in this process?"""
    if os.environ.get("NOVA_BF_NO_TWOPASS"):
        return False
    return _DISABLED_REASON is None


def disable(reason: str) -> None:
    """Turn the two-pass off for the rest of the RUN.
    """
    global _DISABLED_REASON
    if _DISABLED_REASON is None:
        _DISABLED_REASON = reason
        logger.warning(
            "two-pass dense scoring disabled for the rest of this run: %s. "
            "Results are unaffected — the one-pass fp32 path produces the "
            "same ground truth, more slowly.", reason,
        )


def threshold() -> float:
    global _THRESHOLD_WARNED

    raw = os.environ.get("NOVA_BF_TWOPASS_THRESHOLD")
    if not raw:
        return DEFAULT_THRESHOLD

    def _bad(why: str) -> float:
        global _THRESHOLD_WARNED
        if not _THRESHOLD_WARNED:
            _THRESHOLD_WARNED = True
            logger.warning("NOVA_BF_TWOPASS_THRESHOLD=%r %s; using %s",
                           raw, why, DEFAULT_THRESHOLD)
        return DEFAULT_THRESHOLD

    try:
        got = float(raw)
    except ValueError:
        return _bad("is not a number")
    # A live FRACTION, so only [0, 1] is meaningful. `nan` compares false
    # against everything, which would make `hint > thresh` never fire and run
    # pass one on every slice however live it was; a negative value turns the
    # two-pass off with no message at all. Both are configuration mistakes
    # worth a line rather than a silent change of behaviour.
    if not (0.0 <= got <= 1.0):
        return _bad("is not a live fraction in [0, 1]")
    return got
# --- fused pass one -----------------------------------------------------------
#
# The fused kernel computes each query row's scaled maximum directly from the
# float32 GEMM accumulator, avoiding materialization of the full approximate
# Gram matrix. Because no float16 Gram is written, this path also avoids the
# output-rounding term required by a materialized float16 result.

# BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_stages, num_warps.
# Runtime configuration may override or autotune this default.
FUSE_CONFIG = (128, 128, 32, 8, 3, 4)

_FUSE_OFF: str | None = None
_FUSE_LAUNCHES = 0
_AUTOTUNED: dict[tuple, tuple] = {}
_FUSE_KERNEL_INFO: dict = {}

# Candidate configurations for optional in-process autotuning.
_FUSE_SPACE = [
    (bm, bn, bk, gm, st, wp)
    for bm, bn, bk, wp in (
        (32, 128, 64, 4), (32, 256, 64, 4), (32, 256, 64, 8),
        (64, 64, 64, 4), (64, 128, 32, 4), (64, 128, 64, 4),
        (64, 128, 128, 4), (64, 128, 64, 8), (64, 256, 32, 8),
        (64, 256, 64, 4), (64, 256, 64, 8), (128, 64, 64, 4),
        (128, 128, 32, 4), (128, 128, 64, 4), (128, 128, 64, 8),
        (128, 256, 32, 8), (256, 64, 64, 4), (256, 128, 32, 8),
    )
    for gm in (4, 8, 16)
    for st in (2, 3, 4)
]


def fused_off() -> str | None:
    """Why the fused pass one is not in use, or `None` if it is."""
    if os.environ.get("NOVA_BF_NO_FUSED_ROWMAX"):
        return "NOVA_BF_NO_FUSED_ROWMAX"
    if _gemm_rowmax is None:
        return f"triton unavailable ({_UNAVAILABLE})"
    return _FUSE_OFF

def fuse_disable(reason: str) -> None:
    """Disable the fused pass-one kernel for the rest of this run.

    `reset()` clears this state so the next run can attempt the fused path
    again under its own device and workload conditions.
    """
    global _FUSE_OFF
    if _FUSE_OFF is None:
        _FUSE_OFF = reason
        logger.warning(
            "two-pass: the fused pass-one kernel is off for the rest of this "
            "run: %s. Falling back to the unfused path.", reason,
        )
def _config_ok(got) -> bool:
    """Return whether a fused-kernel configuration is safe to launch."""
    if len(got) != 6 or any(not isinstance(v, int) for v in got):
        return False

    bm, bn, bk, gm, stages, warps = got

    # Triton tile dimensions must be powers of two.
    for b in (bm, bn, bk):
        if b < 16 or (b & (b - 1)):
            return False

    # Bound GROUP_M to keep the program-id arithmetic well behaved. Warps must
    # be a positive power of two; zero stages is permitted by Triton.
    return (
        1 <= gm <= 1024
        and stages >= 0
        and warps >= 1
        and not (warps & (warps - 1))
    )

def fuse_config(Qh, Ch) -> tuple:
    """The tile configuration to launch with, honouring the env overrides."""
    global _FUSE_CONFIG_WARNED

    raw = os.environ.get("NOVA_BF_FUSE_CONFIG")
    if raw:
        try:
            got = tuple(int(x) for x in raw.split(","))
            if _config_ok(got):
                return got
        except ValueError:
            pass
        if not _FUSE_CONFIG_WARNED:
            _FUSE_CONFIG_WARNED = True
            # The requirements are spelled to match `_config_ok` exactly; an
            # earlier version said "stages >= 1" while the checker accepted 0.
            logger.warning(
                "NOVA_BF_FUSE_CONFIG=%r is not six LAUNCHABLE integers "
                "(BLOCK_M,BLOCK_N,BLOCK_K powers of two >= 16, "
                "1 <= GROUP_M <= 1024, stages >= 0, warps a power of two); "
                "using the tuned default", raw)
    if os.environ.get("NOVA_BF_FUSE_AUTOTUNE"):
        return autotune(Qh, Ch)
    return FUSE_CONFIG


def fuse_available(Qh, Ch, cs) -> bool:
    """Everything the kernel assumes, checked rather than trusted.
    """
    import torch

    if fused_off() is not None:
        return False
    if not (Qh.is_cuda and Ch.is_cuda and cs.is_cuda):
        return False
    if Qh.device != Ch.device or Qh.device != cs.device:
        return False
    if Qh.dtype is not torch.float16 or Ch.dtype is not torch.float16:
        return False
    if cs.dtype is not torch.float32:
        return False
    if Qh.ndim != 2 or Ch.ndim != 2 or cs.ndim != 1:
        return False
    if not (Qh.is_contiguous() and Ch.is_contiguous() and cs.is_contiguous()):
        return False
    if Qh.shape[1] != Ch.shape[1] or cs.shape[0] != Ch.shape[0]:
        return False
    M, K, N = int(Qh.shape[0]), int(Qh.shape[1]), int(Ch.shape[0])
    if M <= 0 or N <= 0 or K <= 0:
        return False
    # Every offset the kernel forms is int32 in Triton unless the pointer
    # arithmetic is widened; the largest is (M-1)*K for `Q`.
    if M * K >= 2 ** 31 or N * K >= 2 ** 31:
        return False
    return True

def fused_rowmax(Qh, Ch, cs, config=None, probe=False):
    """Compute pass-one row maxima without materializing the Gram matrix.

    Returns the float32 `(M,)` row maxima, or `None` when the fused kernel
    declines. `probe=True` prevents candidate-specific failures during
    autotuning from disabling the fused path for the run.
    """
    import torch

    global _FUSE_LAUNCHES
    if not fuse_available(Qh, Ch, cs):
        return None
    M, K = int(Qh.shape[0]), int(Qh.shape[1])
    N = int(Ch.shape[0])
    bm, bn, bk, gm, stages, warps = config or fuse_config(Qh, Ch)
    if not _config_ok((bm, bn, bk, gm, stages, warps)):
        return None
    n_tiles = _triton.cdiv(N, bn)

    # Keep the flattened output offset within signed int32 range.
    if n_tiles * M >= 2 ** 31:
        return None

    try:
        with torch.cuda.device(Qh.device):
            # Initialize unwritten lanes to +inf so any coverage failure can
            # only force a row live, never incorrectly prune it.
            part = torch.full((n_tiles, M), float("inf"),
                              dtype=torch.float32, device=Qh.device)
            grid = (_triton.cdiv(M, bm) * n_tiles,)
            compiled = _gemm_rowmax[grid](
                Qh, Ch, cs, part, M, N, K,
                Qh.stride(0), Ch.stride(0), part.stride(0),
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm,
                EVEN_K=(K % bk == 0),
                num_stages=stages, num_warps=warps,
            )

        if not probe:
            _FUSE_LAUNCHES += 1

            # Record the configuration that actually ran.
            _FUSE_KERNEL_INFO.setdefault(
                "config", [bm, bn, bk, gm, stages, warps])

        if not probe and len(_FUSE_KERNEL_INFO) <= 1:
            # Record basic kernel resource usage once per run.
            _FUSE_KERNEL_INFO.update(
                n_regs=getattr(compiled, "n_regs", None),
                n_spills=getattr(compiled, "n_spills", None),
                shared=getattr(compiled, "metadata", None)
                and getattr(compiled.metadata, "shared", None),
            )

        # This can allocate the final (M,) output, and is also a CUDA
        # synchronization point where a deferred launch error may surface.
        # It is part of pass one, so it must share the allocation fallback
        # above rather than letting an OOM escape after the protected launch.
        if n_tiles == 1:
            return part[0]
        return part.amax(dim=0)

    except Exception as exc:
        if not probe:
            if is_oom(exc):
                # An OOM abandons pass one for this slice rather than switching
                # immediately to a larger temporary allocation.
                global _FUSE_OOM_WARNED
                if not _FUSE_OOM_WARNED:
                    _FUSE_OOM_WARNED = True
                    logger.warning(
                        "two-pass: the fused row-max could not allocate "
                        "(%s); this slice takes the one-pass path. Later "
                        "slices may retry fused pass one. Results are "
                        "unaffected.", exc,
                    )
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001 - best effort only
                    pass
                raise PassOneUnavailable(str(exc)) from exc

            fuse_disable(f"{type(exc).__name__}: {exc}")

        return None


def autotune(Qh, Ch) -> tuple:
    """Time every candidate configuration on THIS device and shape, once.

    Only reachable under `NOVA_BF_FUSE_AUTOTUNE`. The winner is cached per
    (M, N, K) so a rank pays for the sweep at most once per shape; a
    configuration that fails to compile or launch is skipped rather than
    fatal.
    """
    import torch

    # `_AUTOTUNED` is the only cache that survives `reset()`, justified as "a
    # fact about the device" — so the device belongs in the key. It was the
    # one piece of cross-run state keyed most loosely; `_SHAPE_OK`, which
    # does NOT outlive a run, already keys on device and dtype.
    key = (int(Qh.shape[0]), int(Ch.shape[0]), int(Qh.shape[1]),
           str(Qh.device), str(Qh.dtype))
    got = _AUTOTUNED.get(key)
    if got is not None:
        return got
    cs = torch.ones(int(Ch.shape[0]), dtype=torch.float32, device=Ch.device)
    best, best_ms = FUSE_CONFIG, float("inf")
    for cfg in _FUSE_SPACE:
        try:
            if fused_rowmax(Qh, Ch, cs, config=cfg, probe=True) is None:
                continue
            fused_rowmax(Qh, Ch, cs, config=cfg, probe=True)
            torch.cuda.synchronize(Qh.device)
            ev0, ev1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            ev0.record()
            for _ in range(5):
                fused_rowmax(Qh, Ch, cs, config=cfg, probe=True)
            ev1.record()
            torch.cuda.synchronize(Qh.device)
            ms = ev0.elapsed_time(ev1) / 5.0
        except Exception:
            continue
        if ms < best_ms:
            best, best_ms = cfg, ms
    logger.info("two-pass: fused pass one autotuned at %s to %s (%.3f ms)",
                key, best, best_ms)
    _AUTOTUNED[key] = best
    return best


def fuse_usage() -> dict:
    """The manifest's three-field shape, plus the tile config that ran.
    """
    off = fused_off()
    return {
        "permitted": not os.environ.get("NOVA_BF_NO_FUSED_ROWMAX"),
        "launches": _FUSE_LAUNCHES,
        "unavailable": off,
        "config": list(FUSE_CONFIG),
        **_FUSE_KERNEL_INFO,
    }


# --- counters for the manifest -----------------------------------------------

_UNMEASURED_IS_NONE = frozenset((
    "audit_closest_margin", "probe_accum_worst_rel", "shortfall_worst_ratio",
    "rowmax_worst_ratio", "dead_audit_worst",
))

_STATS = {
    "slices_twopass": 0,   # slices scored through the two-pass
    # Slices the two-pass CONSIDERED and declined, for any of three reasons:
    # no live-fraction measurement yet (the per-batch-group warm-up), the
    # measured fraction being too high to pay, or — after pass one has
    # already run — no padded height below the full one. `slices_discarded`
    # separates that third case, which is the only one that wasted work.
    "slices_plain": 0,
    "rows_full": 0,        # query rows those two-pass slices spanned
    "rows_live": 0,        # query rows that reached the exact GEMM (pre-pad)
    "rows_padded": 0,      # query rows the exact GEMM actually ran on
    "slices_fused": 0,     # pass ones that ran the fused kernel
    "slices_unfused": 0,   # pass ones that fell back to cuBLAS + row max
    "verifications": 0,    # bit-identity checks that COMPLETED
    # Attempts that did NOT complete (allocation failure).
    "verifications_incomplete": 0,
    # Slices that gave up on pass one because it could not allocate.
    "slices_pass_one_oom": 0,
    # Guard refusals, and the two always-on health measurements the closed form
    # makes cheap.
    "slices_guard_refused": 0,
    "rowmax_worst_ratio": None,
    "probe_accum_worst_rel": None,
    "probe_accum_fitted": 0.0,
    "shortfall_worst_ratio": None,
    # How often the O(n_q) binary64 `eps` evaluation actually ran, against how
    # often a bucketed one was reused.
    # per-slice host cost is back.
    "eps_evaluations": 0,
    "eps_cache_hits": 0,
    # The continuous audit (`audit_live_rows`): how many slices and live rows
    # were checked, and whether the bound ever failed on real scored data.
    # `audit_violations` must stay 0; anything else means the bound did not
    # hold and the run's pruned rows are suspect.
    "audit_slices": 0,
    "audit_rows": 0,
    "audit_violations": 0,
    "audit_worst": 0.0,
    # Smallest `(upper - exact) / eps` seen on an audited live row.
    # Negative means the bound was violated; `None` means nothing was audited.
    "audit_closest_margin": None,

    # Dead-row decision audit coverage and failures.
    "dead_audit_oom": 0,
    "dead_audit_offered": 0,          # slices considered for sampling
    "dead_audit_sampled": 0,          # slices selected by the sampler
    "dead_audit_members": 0,          # member-slices actually graded
    "dead_audit_rows": 0,             # finite rows graded
    "dead_audit_violations": 0,       # false-pruned rows
    "dead_audit_worst": None,         # largest false-prune margin

    # Members skipped because filtering prevents a comparable full-slice audit.
    "dead_audit_skipped_filtered": 0,

    # Confusion-matrix counts from `audit_decisions`.
    "audit_correct_prune": 0,
    "audit_correct_live": 0,
    "audit_wasted_live": 0,

    # Two-pass eligibility decisions at the score-group level.
    "groups_seen": 0,
    "groups_refused": 0,

    # Certifications that could not obtain a usable verdict and must be retried.
    "certify_inconclusive": 0,

    "gemms": 0,                       # narrowed exact GEMMs launched

    # Slices where pass one ran but padding removed any exact-GEMM savings.
    "slices_discarded": 0,
}

def stats() -> dict:
    """Return two-pass counters, audit settings, and runtime status."""
    out = dict(_STATS)

    # Record the configured live-row audit sampling rate.
    out["live_row_audit_rate"] = audit_rate()

    # Run-level reason two-pass pruning was disabled, if any.
    out["unavailable"] = _DISABLED_REASON

    # Whether padded-shape verification was explicitly bypassed.
    out["verify_skipped"] = _VERIFY_SKIPPED

    # Report which row-max implementation actually ran.
    if _ROWMAX_OFF:
        out["row_max"] = "portable"
    elif _ROWMAX_USED:
        out["row_max"] = "kernel"
    else:
        out["row_max"] = "not attempted"

    # None = unattempted, "" = passed, otherwise the failure reason.
    out["certified"] = _CERTIFIED

    # Number of execution configurations that passed certification this run.
    out["certified_configs"] = len(_CERTIFIED_KEYS)

    return out

def reset_stats() -> None:
    global _VERIFY_SKIPPED

    for key in _STATS:
        # Preserve `None` for statistics where zero would imply a measurement.
        _STATS[key] = None if key in _UNMEASURED_IS_NONE else 0

    # Clear verification status reported alongside the statistics.
    _VERIFY_SKIPPED = False


# --- the query side, built once per query matrix ------------------------------

# Keyed by `id(Q)`, with `Q` itself held in the value so the id cannot be
# recycled while the entry is alive.
_QCACHE: dict[int, dict] = {}

def release() -> None:
    """Release cached query tensors while preserving run statistics.

    Use this at the end of a run to free device memory before the manifest
    consumes the accumulated counters.
    """
    _QCACHE.clear()
def reset() -> None:
    """Reset all state scoped to a single run.

    Device-level autotuning results are retained, while caches, counters,
    certification, verification, disable state, and warn-once flags are reset.
    """
    global _DISABLED_REASON, _FUSE_OFF, _FUSE_LAUNCHES, _VERIFY_SKIPPED, _ROWMAX_MAG_WARNED, _AUDIT_WARNED, _AUDIT_RATE, _ROWMAX_USED
    global _THRESHOLD_WARNED, _FUSE_CONFIG_WARNED, _ROWMAX_WARNED
    global _UNCHECKED_WARNED, _ROWMAX_OFF, _FUSE_OOM_WARNED
    global _VERIFY_OOM_WARNED, _UNCHECKED_STREAK, _CERTIFIED

    _QCACHE.clear()
    reset_stats()
    _SHAPE_OK.clear()
    _VERIFY_SKIPPED = False
    _DISABLED_REASON = None

    # Retry fused execution on the next run; keep device autotuning results.
    _FUSE_OFF = None
    _FUSE_LAUNCHES = 0
    _FUSE_KERNEL_INFO.clear()

    # Reset per-run warning state and cached configuration.
    _THRESHOLD_WARNED = False
    _FUSE_CONFIG_WARNED = False
    _ROWMAX_WARNED = False
    _ROWMAX_MAG_WARNED = False
    _AUDIT_WARNED = False
    _AUDIT_RATE = None

    # Re-establish closed-form certification for each run.
    _cf.reset()

    _UNCHECKED_WARNED = False
    _ROWMAX_OFF = False
    _ROWMAX_USED = False
    _FUSE_OOM_WARNED = False
    _VERIFY_OOM_WARNED = False
    _UNCHECKED_STREAK = 0
    _CERTIFIED = None
    _CERTIFIED_KEYS.clear()

def query_side(Q, row_scale=None) -> dict:
    """Prepare and cache the query-side inputs required by pass one and its bound.

    The cache stores the half-precision query, binary32 norms, guard mask, and
    the actual query scale applied by pass one.
    """
    import numpy as np
    import torch

    def _ver(t):
        # Detect in-place mutations performed through PyTorch. Writes through
        # aliased external storage are not visible, so callers must keep cached
        # query tensors and scales immutable.
        return None if t is None else t._version

    key = id(Q)
    got = _QCACHE.get(key)
    if (got is not None and got["Q"] is Q and got["row_scale"] is row_scale
            and got["versions"] == (_ver(Q), _ver(row_scale))):
        return got

    Qh = Q.half()

    # Compute norms in binary32, as required by the bound.
    qn = Q.float().norm(dim=1)
    bad = norm_guard(qn)

    if row_scale is None:
        # Dot product applies no query-side scaling.
        rho_a = np.ones(int(Q.shape[0]), dtype=np.float64)
    else:
        # Use the scale actually applied by pass one rather than recomputing it.
        rho_a = row_scale.detach().to("cpu", torch.float64).numpy()

    got = {
        "Q": Q, "Qh": Qh.contiguous(), "qn": qn, "bad": bad,
        "row_scale": row_scale, "rho_a": rho_a,
        "versions": (_ver(Q), _ver(row_scale)),
        # Bound values cached for this query-side state.
        "eps_cache": {},
    }
    _QCACHE[key] = got
    return got


# --- corpus side, per slice ---------------------------------------------------


def corpus_side(Cb, col_scale, cn=None):
    """Prepare the corpus-side inputs and scalars required by the bound.

    Returns `(C_h, sigma_max, cn_min, cn_max)`, where `sigma_max` is the
    largest column scale actually applied by pass one.
    """
    import torch

    Ch = Cb.half()

    if cn is None:
        # Compute norms in binary32, as required by the bound.
        cn = Cb.float().norm(dim=1)
    elif cn.dtype is not torch.float32:
        raise ValueError(
            f"two-pass: corpus norms must be binary32 (R4 is stated for the "
            f"computed binary32 norm); got {cn.dtype}")
    elif cn.shape != (Cb.shape[0],):
        raise ValueError(
            f"two-pass: corpus norms must have shape ({Cb.shape[0]},), "
            f"got {tuple(cn.shape)}")

    if int(Cb.shape[0]) == 0:
        return Ch.contiguous(), 0.0, float("nan"), float("nan")

    if col_scale is None:
        # Dot product applies no corpus-side scaling.
        sigma_max = 1.0
    else:
        # Use the largest scale actually applied by pass one.
        sigma_max = float(col_scale.max())

    return Ch.contiguous(), sigma_max, float(cn.min()), float(cn.max())
def bound(qs: dict, d: int, sigma_max, half_out: bool, metric, fmt_q=None,
          fmt_c=None, cn_max=None):
    """Compute the per-query error bound in the same units as `thr`.

    The bound is evaluated in binary64 on the host and rounded upward to
    binary32 by the closed-form implementation.
    """
    import torch

    fq = fmt_q or _cf.FP16
    fc = fmt_c or _cf.FP16
    rho = qs["rho_a"]

    # Dot product applies no query-side scaling.
    if metric == "dot" and qs["row_scale"] is not None:
        raise ValueError(
            "two-pass: metric='dot' applies no scaling, so Theorem 1' is "
            "stated at rho = sigma = 1; passing a row_scale means the bound "
            "would describe different arithmetic than the pass performed"
        )

    # Cosine requires the query-side scale applied by pass one.
    if metric == "cosine" and qs["row_scale"] is None:
        raise ValueError(
            "two-pass: metric='cosine' needs the row scale pass one's output "
            "is multiplied by; without it `eps` is sized for unit-norm scores "
            "and the bound does not hold. Pass `row_scale`, or use metric='dot'"
        )

    if metric not in ("cosine", "dot"):
        raise ValueError(
            f"two-pass: no bound is derived for metric={metric!r}. Theorem 1 "
            f"covers cosine and Theorem 1' covers dot; euclidean is scored by "
            f"the one-pass path"
        )

    if metric == "dot":
        # Dot-product error depends on the largest corpus norm in the slice.
        if cn_max is None:
            raise ValueError(
                "two-pass: metric='dot' needs cn_max = max_S N_c; Theorem 1' "
                "has no scale-free constant and E is proportional to it"
            )

        qn_ub = qs["qn"].detach().to("cpu", torch.float64).numpy()
        return _cf.eps_dot(
            d, fq.u_t, fc.u_t, qn_ub, float(cn_max),
            _cf.C_HW_ENVELOPE, half_out, fq.eta_t, fc.eta_t, fc.lam_t,
        )

    return _cf.eps_cos(
        d, fq.u_t, fc.u_t, _cf.C_HW_ENVELOPE, half_out,
        fq.eta_t, fc.eta_t, rho, sigma_max, fc.lam_t,
    )

def stored_format(t) -> object:
    """Return the format model for converting `t` to pass-one float16.

    A float16 tensor requires no conversion. Other supported input types are
    modeled as conversion to float16.
    """
    import torch

    if t.dtype is torch.float16:
        return _cf.EXACT

    if t.dtype is torch.bfloat16:
        # Pass one converts bfloat16 inputs to float16.
        return _cf.FP16

    return _cf.FP16

def float32_is_exactly_fp16(t) -> bool:
    """Return whether every value is exactly representable in float16."""
    import torch

    if t.numel() == 0:
        return True
    if t.dtype is torch.float16:
        return True
    if t.dtype is not torch.float32:
        return False
    return bool(torch.equal(t.half().float(), t))


def is_oom(exc: BaseException) -> bool:
    """Return whether an exception represents a memory-allocation failure."""
    import torch

    if isinstance(exc, (MemoryError, getattr(torch, "OutOfMemoryError", ()))):
        return True

    text = str(exc).lower()
    return (
        "out of memory" in text
        or "alloc_failed" in text
        # Cover CPU allocator failures that arrive as RuntimeError.
        or "cannot allocate memory" in text
        or "can't allocate memory" in text
    )
# Classical binary32 dot-product error bound used by the exact pass.
# Tensor-core accumulation is modeled separately because its truncating
# arithmetic does not satisfy the same round-to-nearest assumptions.
gamma = _cf.gamma

def _pin_accumulation() -> None:
    """Require float32 accumulation for the unfused CUDA half GEMM.

    Reduced-precision split-K accumulation is outside the error model used by
    the bound, so the setting is reasserted before each pass-one GEMM.
    """
    import torch

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
def approx_rowmax(Qh, Ch, col_scale, out_dtype):
    """Compute the approximate scaled row maximum for pass one.

    Returns `(out, fused)`. When `fused` is true, the row max is taken directly
    from the float32 GEMM accumulator and no approximate Gram is materialized.
    Otherwise the Gram is materialized and reduced separately.
    """
    import torch

    global _ROWMAX_WARNED, _ROWMAX_OFF

    _pin_accumulation()
    n_cols = int(Ch.shape[0])
    if col_scale is None:
        cs = torch.ones(n_cols, dtype=torch.float32, device=Ch.device)
    else:
        cs = col_scale.contiguous()

    got = fused_rowmax(Qh, Ch, cs)
    if got is not None:
        _STATS["slices_fused"] += 1
        return got, True
    _STATS["slices_unfused"] += 1

    if not Qh.is_cuda:
        # CPU widens the half inputs and performs the product in float32.
        try:
            g = Qh.float() @ Ch.float().T
            if out_dtype is None:
                g = g.half().float()
        except Exception as exc:  # noqa: BLE001 - see the note below
            raise _as_pass_one_failure(exc, "the widened CPU product") from exc
    else:
        # Materialize the approximate Gram when the fused CUDA path is unavailable.
        try:
            if out_dtype is None:
                g = Qh @ Ch.T
            else:
                g = torch.mm(Qh, Ch.T, out_dtype=out_dtype)
        except Exception as exc:  # noqa: BLE001 - see above
            raise _as_pass_one_failure(exc, "the fp16 Gram") from exc

    n_q = int(g.shape[0])
    out = torch.empty(n_q, dtype=torch.float32, device=g.device)
    global _ROWMAX_USED

    if (_rowmax_scaled is not None and not _ROWMAX_OFF
            and g.is_cuda and g.is_contiguous()):
        try:
            block = min(2048, _triton.next_power_of_2(n_cols))
            with torch.cuda.device(g.device):
                _rowmax_scaled[(n_q,)](
                    g, cs, out, g.stride(0), n_cols,
                    BLOCK=block, IS_F16=g.dtype is not torch.float32,
                    num_warps=8,
                )

            # Record success only after the kernel completes.
            _ROWMAX_USED = True
            return out, False
        except Exception as exc:  # noqa: BLE001 - fall through to the portable loop
            # OOM is transient; deterministic kernel failures disable this path.
            if is_oom(exc):
                raise _as_pass_one_failure(exc, "the row-max kernel") from exc
            _ROWMAX_OFF = True
            if not _ROWMAX_WARNED:
                _ROWMAX_WARNED = True
                logger.warning(
                    "two-pass: the row-max kernel failed (%s: %s); using the "
                    "portable reduction from here on. Results are unaffected.",
                    type(exc).__name__, exc,
                )

    # Portable chunked reduction limits the temporary scaled matrix size.
    chunk = max(1, (1 << 26) // max(1, n_cols))
    try:
        for r0 in range(0, n_q, chunk):
            blk = g[r0 : r0 + chunk].float() * cs[None, :]
            out[r0 : r0 + chunk] = blk.amax(dim=1)
            del blk
    except Exception as exc:  # noqa: BLE001 - see above
        raise _as_pass_one_failure(exc, "the portable row-max copy") from exc

    return out, False

def upper_bounds(Q, Cb, col_scale, row_scale, out_dtype, metric,
                 cn=None, corpus_exact_fp16=None, with_parts=False):
    """Return a safe per-query upper bound on this corpus slice's best score.

    Guard failures force affected rows live by returning `+inf`. An empty
    corpus slice returns `-inf`. `corpus_exact_fp16` may provide a previously
    established per-file format verdict; otherwise it is checked here.
    """
    import numpy as np
    import torch

    n_q = int(Q.shape[0])
    d = int(Q.shape[1])
    dev = Q.device

    def _fill(v, reason=None):
        if reason:
            _STATS["slices_guard_refused"] += 1
        t = torch.full((n_q,), v, dtype=torch.float32, device=dev)

        # Certification needs the approximate value and bound separately.
        return ((t, t.clone(), torch.zeros_like(t)) if with_parts else t)

    def all_live(reason=None):
        return _fill(float("inf"), reason)

    if int(Cb.shape[0]) == 0:
        # The maximum over an empty corpus is -inf.
        return _fill(float("-inf"))

    # Refuse dimensions outside the closed form's admitted range.
    if not _cf.dimension_ok(d):
        return all_live("dimension")

    # Require the structural closed-form checks to pass.
    why = _cf.certified()
    if why is not None:
        return all_live(why)

    if out_dtype is not None and out_dtype is not torch.float32:
        if out_dtype is not torch.float16:
            # The output-rounding term is defined only for float16.
            raise ValueError(
                f"two-pass: out_dtype={out_dtype} is not supported; the "
                f"output-rounding term is derived for float16 (or a float32 "
                f"output, which needs no term at all)"
            )

    # Enforce the metric's corpus-scaling contract.
    if metric == "dot" and col_scale is not None:
        raise ValueError(
            "two-pass: metric='dot' applies no corpus scaling, so Theorem 1' "
            "is stated at sigma = 1 and `eps_dot` never reads sigma_max; "
            "passing a col_scale means pass one's output is scaled by a "
            "factor the bound does not account for"
        )
    if metric == "cosine" and col_scale is None:
        raise ValueError(
            "two-pass: metric='cosine' needs the column scale pass one "
            "divides by; without it the row max is a raw Gram entry while "
            "`eps` is sized for a normalised score, and the bound does not "
            "hold. Pass `col_scale`, or use metric='dot'"
        )

    # These preparations allocate device memory too: in particular Qh is a
    # run-cached half copy and Ch is a fresh half copy for every corpus slice.
    # Keep them inside the same OOM-to-all-live boundary as the row-max GEMM.
    # Otherwise a conversion OOM escapes before `approx_rowmax` gets its
    # chance to translate it into the safe one-pass fallback.
    try:
        qs = query_side(Q, row_scale)
        if cn is None:
            # Compute corpus norms in binary32, as required by the bound.
            cn = Cb.float().norm(dim=1)

        # Select the conversion-error model from the actual corpus representation.
        if corpus_exact_fp16 is None:
            corpus_exact_fp16 = float32_is_exactly_fp16(Cb)
        fmt_c = _cf.EXACT if corpus_exact_fp16 else _cf.FP16
        fmt_q = stored_format(Q)

        # Corpus guards are slice-wide; query guards are per row.
        Ch, sigma_max, cn_min, cn_max = corpus_side(Cb, col_scale, cn)
    except Exception as exc:  # noqa: BLE001 - preserve non-allocation errors
        failure = _as_pass_one_failure(exc, "the pass-one input copies")
        if isinstance(failure, PassOneUnavailable):
            _STATS["slices_pass_one_oom"] += 1
            return all_live()
        raise failure

    if bool(norm_guard(cn).any()):
        return all_live("corpus norm")

    if not overflow_guard(d, cn, fmt_c):
        return all_live("corpus overflow")

    if not _cf.norm_range_ok(sigma_max) and sigma_max != 1.0:
        return all_live("corpus scale")

    # Verify applied normalization scales against the computed norms.
    if col_scale is not None:
        bad_cols, why = scale_mismatch(col_scale, cn)
        if why:
            return all_live("column scale")

    bad = qs["bad"]

    if row_scale is not None:
        bad_rows, why = scale_mismatch(row_scale, qs["qn"])
        if why:
            # A bad query scale forces only that query row live.
            bad = bad | bad_rows

    bad = bad | overflow_mask(d, qs["qn"], fmt_q)
    bad = bad | product_guard(qs["qn"], cn_max)
    bad = bad | prunability_cut(qs["qn"], cn_min)

    if bool(bad.all()):
        # Avoid pass one when every query row is already forced live.
        return all_live("all rows refused")

    try:
        approx, fused = approx_rowmax(qs["Qh"], Ch, col_scale, out_dtype)
    except PassOneUnavailable:
        _STATS["slices_pass_one_oom"] += 1
        return all_live()

    del Ch

    # Only trusted accumulation paths may contribute pruning decisions.
    if not accumulator_is_ours(dev, fused):
        return all_live("cublas accumulator")

    # Recheck mutable CUDA exact-math state for every slice.
    why = exact_math_mode_flags(dev)
    if why:
        return all_live("exact math mode")

    # Any non-finite approximate result is conservatively forced live.
    approx = torch.where(torch.isfinite(approx), approx,
                         torch.full_like(approx, float("inf")))

    # Fused output is reduced directly from float32 accumulation; an unfused
    # non-float32 Gram requires the output-rounding term.
    half_out = (out_dtype is not torch.float32) and not fused

    # Bucket slice-wide parameters upward so cached eps remains conservative.
    sigma_key = _cf.ceil_pow2(sigma_max)
    cn_key = _cf.ceil_pow2(cn_max) if metric == "dot" else 0.0
    ekey = (sigma_key, cn_key, fmt_q.name, fmt_c.name, half_out, metric, d)

    eps_np = qs["eps_cache"].get(ekey)
    if eps_np is None:
        eps_np = bound(qs, d, sigma_key, half_out, metric, fmt_q, fmt_c, cn_key)
        qs["eps_cache"][ekey] = eps_np
        _STATS["eps_evaluations"] += 1
    else:
        _STATS["eps_cache_hits"] += 1

    eps = torch.as_tensor(np.ascontiguousarray(eps_np), dtype=torch.float32,
                          device=dev)
    if eps.ndim == 0:
        eps = eps.expand(n_q)

    if row_scale is not None:
        approx = approx * row_scale

    out = approx + eps

    # Guarded rows are always live.
    out = torch.where(bad, torch.full_like(out, float("inf")), out)

    # Reject rows whose approximate magnitude violates the bound's hypotheses.
    over = _check_rowmax_magnitude(approx, d, fmt_q, fmt_c, half_out, metric)
    if over is not None:
        out = torch.where(over, torch.full_like(out, float("inf")), out)

    if with_parts:
        # `approx` is in the same scaled units as the exact score.
        return out, approx, eps

    return out

# Largest magnitude admitted for a scaled approximate cosine score.
# Exceeding it means the pass-one result violates the bound's assumptions.
def _check_rowmax_magnitude(approx, d, fmt_q, fmt_c, half_out, metric):
    """Return a per-row mask of approximate scores above the admitted cap."""
    global _ROWMAX_MAG_WARNED
    import torch

    if metric != "cosine":
        return None

    finite = approx[torch.isfinite(approx)]
    if finite.numel() == 0:
        return None

    worst = float(finite.abs().max())
    cap = (_cf.Lambda(d) * (1.0 + _cf.theta(d, _cf.C_HW_ENVELOPE))
           * (1.0 + fmt_q.u_t) * (1.0 + fmt_c.u_t)
           * ((1.0 + _cf.U16) if half_out else 1.0))

    _STATS["rowmax_worst_ratio"] = max(
        _STATS.get("rowmax_worst_ratio") or 0.0, worst / cap)

    if worst > cap and not _ROWMAX_MAG_WARNED:
        _ROWMAX_MAG_WARNED = True
        logger.warning(
            "two-pass: a first-pass row maximum of %.6g exceeds the %.6g that "
            "Theorem 1 admits at d=%d. The affected rows will be forced live; "
            "check the applied scales and `probe_scales`.",
            worst, cap, d,
        )

    return torch.isfinite(approx) & (approx.abs() > cap)

_ROWMAX_MAG_WARNED = False


# --- exact-shape verification -------------------------------------------------
def _as_pass_one_failure(exc: BaseException, what: str) -> BaseException:
    """Convert an allocation failure into a per-slice pass-one fallback."""
    if is_oom(exc):
        logger.warning(
            "two-pass: pass one could not allocate %s (%s); this slice takes "
            "the one-pass path. Results are unaffected.", what, exc,
        )
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best effort only
            pass
        return PassOneUnavailable(str(exc))
    return exc


class PassOneUnavailable(Exception):
    """Pass one could not run for this slice because of memory pressure."""


# Third verification state: the check could not complete.
class _Unverified:
    __slots__ = ()

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "UNVERIFIED"


UNVERIFIED = _Unverified()

# Cached per-run verification results by GEMM shape and device.
_SHAPE_OK: dict[tuple, bool] = {}

# Serialize expensive verification and re-check the cache after acquiring it.
_SHAPE_LOCK = threading.Lock()

# True when runtime shape verification was explicitly bypassed.
_VERIFY_SKIPPED = False


def _verify_shape(Q, Cn, M) -> bool | _Unverified:
    """Check whether an M-row GEMM matches the corresponding full-height rows."""
    global _VERIFY_SKIPPED

    n_full = int(Q.shape[0])
    key = (M, n_full, Cn.shape[0], Q.shape[1], str(Q.dtype), str(Q.device))
    got = _SHAPE_OK.get(key)
    if got is not None:
        return got

    if os.environ.get("NOVA_BF_TWOPASS_NO_VERIFY"):
        if not _VERIFY_SKIPPED:
            _VERIFY_SKIPPED = True
            logger.warning(
                "NOVA_BF_TWOPASS_NO_VERIFY is set: the padded exact-GEMM "
                "heights this run uses are NOT being proven bit-identical to "
                "the full-height product on this device. That proof is what "
                "makes the two-pass exact; without it, scores from padded and "
                "full-height slices can disagree and reorder near-ties. The "
                "manifest records verify_skipped for this run."
            )
        _SHAPE_OK[key] = True
        return True

    with _SHAPE_LOCK:
        # Another thread may have checked this shape while we waited.
        got = _SHAPE_OK.get(key)
        if got is not None:
            return got
        return _verify_shape_locked(Q, Cn, M, key, n_full)


def _verify_shape_locked(Q, Cn, M, key, n_full: int) -> bool | _Unverified:
    import torch

    # Match pass two by testing a gathered, contiguous query matrix.
    idx = (torch.arange(M, device=Q.device) * max(1, n_full // M))[:M].clamp_max(n_full - 1)

    try:
        sub = Q.index_select(0, idx) @ Cn.T
        ref = Q @ Cn.T
        ok = torch.equal(sub, ref.index_select(0, idx))
        del sub, ref, idx
    except Exception as exc:  # noqa: BLE001 - handle verification OOMs separately
        if is_oom(exc):
            _STATS["verifications_incomplete"] += 1

            global _VERIFY_OOM_WARNED
            if not _VERIFY_OOM_WARNED:
                _VERIFY_OOM_WARNED = True
                logger.warning(
                    "two-pass: the %d-row bit-identity check ran out of "
                    "memory (%s); NOT caching that as a failed height. See "
                    "verifications_incomplete for the count.", M, exc,
                )

            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - best effort only
                pass

            # Memory pressure is transient, so allow a later slice to retry.
            return UNVERIFIED

        logger.warning(
            "two-pass: the %d-row bit-identity check did not complete "
            "(%s: %s); treating that height as unusable and falling back to "
            "the next one, or to the one-pass path.",
            M, type(exc).__name__, exc,
        )
        _SHAPE_OK[key] = False
        return False

    # Count only checks that actually completed.
    _STATS["verifications"] += 1

    # A completed check ends the transient verification-failure streak.
    note_checked_slice()

    _SHAPE_OK[key] = ok
    if not ok:
        logger.warning(
            "two-pass: a %d-row GEMM does not match the full %d-row one "
            "bit-for-bit on this device (N=%d, K=%d) — cuBLAS chose a "
            "different kernel and the accumulation order moved.",
            M, Q.shape[0], Cn.shape[0], Q.shape[1],
        )
    return ok

def pad_candidates(n_live: int, n_full: int) -> list[int]:
    """Return ascending GEMM heights to try for `n_live` rows.

    Candidate heights follow the preferred padding pattern and are verified
    before use. `n_full` is always the final fallback.
    """
    if n_live <= 0:
        return []
    if n_live > n_full:
        raise ValueError(f"{n_live} live rows in a {n_full}-row matrix")

    out = [h for h in PAD_SMALL if n_live <= h < PAD_FLOOR]
    h = max(PAD_FLOOR, -(-n_live // PAD_QUANTUM) * PAD_QUANTUM)

    while len(out) < MAX_PAD_TRIES and h < n_full:
        out.append(h)
        h += PAD_QUANTUM

    out = [x for x in out if x < n_full][:MAX_PAD_TRIES]

    # Full height is the final fallback.
    out.append(n_full)
    return out


def pad_height(n_live: int, n_full: int) -> int:
    """Return the first candidate padded height."""
    got = pad_candidates(n_live, n_full)
    return got[0] if got else 0
