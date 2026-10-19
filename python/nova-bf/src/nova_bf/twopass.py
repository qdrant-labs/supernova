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

import contextlib
import logging
import math
import os
import threading
import typing

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

# Only GPU generations supported by the tensor-core accumulation model
# are admitted. Runtime probes check for incompatible behavior but cannot
# prove correctness for every input. Unknown generations fall back to
# one-pass FP32 scoring rather than assuming hardware compatibility.
_R5_GENERATIONS = {
    (7, 0): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 9): "Ada",
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
    if scale.shape != norms.shape:
        return (
            scale_mismatch_mask(scale, norms),
            f"scale has shape {tuple(scale.shape)} against norms of shape "
            f"{tuple(norms.shape)}; scale and norm entries must correspond"
        )

    bad, lo, hi = _scale_mismatch_parts(scale, norms)
    scale = scale.reshape(-1)
    norms = norms.reshape(-1)

    if not bool(bad.any()):
        return bad, None

    i = int(bad.double().argmax())
    return (
        bad,
        f"scale[{i}] = {float(scale[i])!r} is outside "
        f"[{float(lo[i])!r}, {float(hi[i])!r}] for norm "
        f"{float(norms[i])!r}",
    )


def _finite(x):
    """Equivalent to `torch.isfinite` for floating tensors, using fewer
    GPU kernels. Both NaN and infinity fail the comparison."""
    return x.abs() < float("inf")

# Cache scalar constants per (device, dtype, value) to avoid repeated
# allocations and fill kernels in `torch.where`.
# Safe without `record_stream` while all readers use the allocating stream.
# Revisit if accessed from another stream.
_CONSTS: dict = {}


def _clear_device_caches() -> None:
    """Drop the small per-device caches (`_CONSTS`, and those of `compute` /
    `tile_rescore` if loaded -- looked up, not imported, to keep this module
    free of import cycles)."""
    import sys

    _CONSTS.clear()
    for mod, name in (("nova_bf.compute", "_TP_BOUNDS"),
                      ("nova_bf.tile_rescore", "_QIDS")):
        cache = getattr(sys.modules.get(mod), name, None)
        if cache is not None:
            cache.clear()


def _const(value: float, like):
    """`value` as a 0-d tensor of `like`'s dtype on `like`'s device, cached.
    `torch.where(c, x, _const(v, x))` is `torch.where(c, x, v)`."""
    import torch

    key = (str(like.device), like.dtype, repr(float(value)))
    got = _CONSTS.get(key)
    if got is None:
        got = _CONSTS[key] = torch.full((), value, dtype=like.dtype,
                                        device=like.device)
    return got


def _scale_mismatch_parts(scale, norms):
    """`(bad, lo, hi)` of `scale_mismatch` for equal shapes, flattened. Device
    ops only, in binary64 -- no host read."""
    import torch

    # Flatten so scalar tensors and vectors share the same reporting path.
    n = norms.reshape(-1).double()
    s = scale.reshape(-1).double()

    ok_norm = _finite(n) & (n > 0.0)

    # Round the admissible interval outward.
    lo = torch.nextafter((1.0 - _cf.U) / n, torch.zeros_like(n))
    hi = torch.nextafter(
        (1.0 + _cf.U) / n,
        torch.full_like(n, float("inf")),
    )

    ok = _finite(s) & (s > 0.0) & (s >= lo) & (s <= hi)
    return ok_norm & ~ok, lo, hi


def scale_mismatch_mask(scale, norms):
    """Return a device-side mismatch mask without host synchronization.
    Shape mismatches mark all entries invalid."""
    import torch

    # Require matching shapes; broadcasting would invalidate the check.
    if scale.shape != norms.shape:
        return torch.ones(norms.shape, dtype=torch.bool, device=norms.device)
    return _scale_mismatch_parts(scale, norms)[0]

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
def audit_decisions(Q, Cb, metric, live, thr, q_norms=None, tile_live=None,
                    tile_width=None) -> dict:
    """Exactly score every row and grade the pass-one pruning decisions.

    A row should remain live when the exact slice maximum is at or above its
    threshold. Equality counts as live because pruning uses the strict rule
    `upper < threshold`.

    This audit observes both kept and pruned rows, but costs the full-height
    exact GEMM that two-pass pruning is intended to avoid. It is therefore a
    verification tool rather than a production safety guard.

    In tiles mode pass `tile_live` (`(n_tiles, n)` bool) and `tile_width`: the
    decisions graded are then the (query, tile) ones -- a pair should be live
    when its exact TILE maximum reaches the threshold -- in the same counters,
    plus `dead_audit_tile_pairs`. `live` must still be its any-tile reduction.
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
        if tile_live is not None:
            # Graded per pair: (n, n_tiles), thresholds broadcast per row.
            top = tile_amax(exact, int(tile_width))
            if tuple(tile_live.shape) != (int(top.shape[1]), n):
                raise ValueError(
                    f"two-pass decision audit: tile_live has shape "
                    f"{tuple(tile_live.shape)}, expected "
                    f"({int(top.shape[1])}, {n})")
            live = tile_live.T
            thr_v = thr_v[:, None]
        else:
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
    if tile_live is not None:
        _STATS["dead_audit_rows"] += int(ok.any(dim=1).sum())
        _STATS["dead_audit_tile_pairs"] += int(ok.sum())
    else:
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
    ok = _finite(top) & _finite(ub)
    bad = ok & ~(ub >= top)
    n_bad, n_ok, margin, eps_used = _audit_read(bad, ok, ub, top, eps_used)

    _STATS["audit_rows"] += n_ok

    # Record the smallest remaining bound margin relative to `eps`.
    if n_ok > 0 and eps_used:
        closest = margin / eps_used
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

def _audit_read(bad, ok, ub, top, eps_used, *extra):
    """Everything an audit decides from on the host, in ONE device->host
    transfer (it took five): `(n_bad, n_ok, margin, eps_used, *extra)`.

    `margin` is `float((ub - top)[ok].min())` when any entry is `ok` (else
    +inf, unused): an `ok` entry has finite `ub` and `top`, so `ub - top` is
    never NaN and filling the others with +inf leaves the minimum -- one of
    the `ok` values -- as it was, with no boolean index (a sync of its own).
    `n_ok > 0` is `bool(ok.any())`. `eps_used` may be a 0-d device tensor
    (`compute` defers that read to here) or a host float, returned as a
    float. Counts and binary32 values are exact in binary64."""
    import torch

    xs = [bad.sum(), ok.sum()]
    if ok.numel():
        d = ub - top
        xs.append(torch.where(ok, d, _const(float("inf"), d)).min())
    eps_t = isinstance(eps_used, torch.Tensor)
    if eps_t:
        xs.append(eps_used)
    xs.extend(extra)
    vals = _read_scalars(*xs)
    n_bad, n_ok = int(vals.pop(0)), int(vals.pop(0))
    margin = vals.pop(0) if ok.numel() else float("inf")
    if eps_t:
        eps_used = vals.pop(0)
    return (n_bad, n_ok, margin, eps_used, *(int(v) for v in vals))


def audit_live_tiles(scores, idx, tile_upper, width: int, row_scale=None,
                     eps_used=None) -> int:
    """Verify tile bounds against exact scores for all live query/tile pairs.

    Dead pairs are `-inf` and excluded from the audit. Updates the same
    counters as `audit_live_rows`, plus `audit_tile_pairs`.
    """
    global _AUDIT_WARNED
    import torch

    n_live = int(idx.shape[0])
    if n_live <= 0 or scores is None:
        return 0
    if scores.shape[0] != n_live:
        raise ValueError(
            f"two-pass tile audit: {scores.shape[0]} score rows for {n_live} "
            f"live queries")
    n_tiles = -(-int(scores.shape[1]) // width)
    if tile_upper.shape[0] != n_tiles:
        raise ValueError(
            f"two-pass tile audit: {n_tiles} tiles of {width} in the scores, "
            f"{tile_upper.shape[0]} in the bound")

    _STATS["audit_slices"] += 1
    top = tile_amax(scores, width)                       # (n_live, n_tiles)
    if row_scale is not None:
        top = top * row_scale.index_select(0, idx)[:, None]
    ub = tile_upper.index_select(1, idx).T
    ok = _finite(top) & _finite(ub)
    bad = ok & ~(ub >= top)
    n_bad, n_ok, margin, eps_used, n_rows_ok = _audit_read(
        bad, ok, ub, top, eps_used, ok.any(dim=1).sum())

    _STATS["audit_rows"] += n_rows_ok
    _STATS["audit_tile_pairs"] += n_ok

    if n_ok > 0 and eps_used:
        closest = margin / eps_used
        prev = _STATS.get("audit_closest_margin")
        _STATS["audit_closest_margin"] = (
            closest if prev is None else min(prev, closest))

    if n_bad:
        worst = float((top - ub)[bad].max())
        _STATS["audit_violations"] += n_bad
        _STATS["audit_worst"] = max(_STATS.get("audit_worst") or 0.0, worst)
        disable(
            f"the continuous audit found {n_bad} live (query, tile) pairs "
            f"whose exact maximum exceeds their tile upper bound, by as much "
            f"as {worst:.3e}"
        )
        if not _AUDIT_WARNED:
            _AUDIT_WARNED = True
            logger.error(
                "two-pass AUDIT FAILURE: %d of %d live (query, tile) pairs on "
                "this slice have an exact maximum ABOVE their tile upper "
                "bound, by as much as %.3e. Live pairs are still scored "
                "exactly so THESE results are correct, but a pair PRUNED "
                "under the same bound may have been lost. Re-run with "
                "NOVA_BF_TWOPASS=0 to get ground truth, and treat this run's "
                "output as suspect.", n_bad, int(ok.sum()), worst,
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


# --- device-side guards -------------------------------------------------------

# Evaluate guards on the GPU to avoid host synchronization before pass one.
# Match host-side comparisons and precision, using float64 where required
# by the proof. Refused slices receive +inf bounds, preventing pruning.
# Guard results are read back together after the kernel is launched.

# Refusal priority matches the original host-side checks.
SLICE_REFUSALS = ("corpus norm", "corpus overflow", "corpus scale",
                  "column scale", "all rows refused")

# Device probe layout: refusal flags, sigma_max, and max N_c.
# Read back in one D2H transfer for refusal reporting and the eps key.
_PROBE_SIGMA = len(SLICE_REFUSALS)
_PROBE_CN_MAX = _PROBE_SIGMA + 1


def overflow_refused_t(d: int, n_max, fmt):
    """Device twin of `not overflow_guard(d, norms, fmt)` given the reduced
    `n_max = norms.max()` (a 0-d tensor): `_cf.overflow_ok`, in binary64."""
    import torch

    if fmt.u_t == 0.0:
        return torch.zeros((), dtype=torch.bool, device=n_max.device)
    m = n_max.double()
    limit = fmt.omega_t * (1.0 - _cf.GUARD_MARGIN)
    ok = (m == m) & (m > 0.0) & (m != float("inf")) & (_cf.kappa_bar(d) * m <= limit)
    return ~ok


def scale_range_refused_t(sigma_max):
    """Device twin of `not norm_range_ok(sigma_max) and sigma_max != 1.0`."""
    s = sigma_max.double()
    return ~((s >= _cf.NORM_MIN) & (s <= _cf.NORM_MAX)) & (s != 1.0)


def query_guard_halves(qn):
    """The query-only halves of `product_guard_t` / `prunability_cut_t`, for
    a query state: `(qn.double(), product rows, prunability rows)`. Fixed for
    the state, so `_upper_bounds` computes them once per query matrix (cached
    in `query_side`'s entry, dropped with it) instead of on every slice."""
    return (qn.double(), ~(_finite(qn) & (qn > 0.0)),
            ~(_finite(qn) & (qn >= _cf.PRUNE_MIN_QNORM)))


def product_guard_t(qn, cn_max, halves=None):
    """`product_guard(qn, float(cn_max))` with `cn_max` a 0-d device tensor.
    `halves`: `query_guard_halves(qn)`, precomputed -- the same values."""
    c = cn_max.double()
    cn_ok = _finite(c) & (c > 0.0)
    qn_d, q_rows, _ = query_guard_halves(qn) if halves is None else halves
    prod = qn_d * c
    rows = q_rows | ~(prod <= _cf.PROD_MAX)
    return rows | ~cn_ok


def prunability_cut_t(qn, cn_min, halves=None):
    """`prunability_cut(qn, float(cn_min))` with `cn_min` a 0-d device tensor.
    `halves`: `query_guard_halves(qn)`, precomputed -- the same values."""
    m = cn_min.double()
    cn_ok = _finite(m) & (m >= _cf.PRUNE_MIN_CNORM)
    q_rows = (query_guard_halves(qn) if halves is None else halves)[2]
    return q_rows | ~cn_ok


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
# At 8192 rows the smallest padded exact height (PAD_SMALL's 2048) is a quarter
# of the full GEMM, so a late-run live fraction of a few percent still costs
# ~0.25 (pass one) + 0.25 (padded pass two) of plain fp32.
MIN_QUERY_ROWS = 8192

# Maximum live fraction for using the two-pass path by default.
DEFAULT_THRESHOLD = 0.50

# --- pass-two rescoring granularity -------------------------------------------

# `slice` rescores live queries with a verified-height cuBLAS GEMM,
# producing byte-identical one-pass scores.
# `tiles` (default) rescores only live query/tile pairs in FP32, safely
# reducing work but potentially changing near-tie rankings due to
# different summation order. Pruned pairs are filled with -inf.
# Falls back to slice mode when tile rescoring is inefficient or unavailable.
# Configured via `two_pass_rescore`; NOVA_BF_TWOPASS_RESCORE overrides it.
RESCORE_MODES = ("slice", "tiles")
_RESCORE: str | None = None
_RESCORE_WARNED = False

# Engage tiles mode when the live (query, tile) fraction is below this
# threshold. Slice mode uses DEFAULT_THRESHOLD.
DEFAULT_TILE_THRESHOLD = 0.20

# Fall back to slice mode when tile rescoring becomes more expensive
# than a padded GEMM. Both paths preserve correctness.
TILE_CROSSOVER = 1.0 / 3.0

# CPU tile width; CUDA uses the fused kernel's BLOCK_N.
# Affects performance, not correctness.
CPU_TILE = 128

_UNAVAILABLE: str | None = None
_DISABLED_REASON: str | None = None

# Warn once for malformed runtime configuration values.
_THRESHOLD_WARNED = False
_TILE_THRESHOLD_WARNED = False
_FORCE_ENGAGE_WARNED = False
_FUSE_CONFIG_WARNED = False
_ROWMAX_WARNED = False

# Disable further Triton row-max launches after a launch failure.
_ROWMAX_OFF = False

# Whether the Triton row-max kernel successfully ran during this run.
_ROWMAX_USED = False
# How many launches used the row-max kernel; `_ROWMAX_USED` is "> 0", kept as
# its own flag for the readers that predate the count (see `_take_back`).
_ROWMAX_RUNS = 0

_FUSE_OOM_WARNED = False
_VERIFY_OOM_WARNED = False
_UNCHECKED_WARNED = False

# Stop repeatedly attempting unverifiable padded shapes after sustained
# verification failure rather than continually thrashing the allocator.
MAX_UNCHECKED_SLICES = 16
_UNCHECKED_STREAK = 0

# The same shape as MAX_UNCHECKED_SLICES, deliberately a SEPARATE counter.
# Certification can now come back INCONCLUSIVE -- no kernel ran, so there is
# nothing to judge -- and the caller retries on the next slice instead of
# disabling the run. Unbounded, that is worse than the bug it replaced: a
# corpus whose every slice is guard-refused (nytimes-256: 70 zero-norm rows in
# 102,400, so every 4096-row slice contains one) re-pays the full probe
# battery forever and never prunes.
#
# NOT shared with _UNCHECKED_STREAK, for three reasons: that streak means "the
# bit-identity check could not COMPLETE" (a resource condition) while this one
# means "there was nothing to certify" (a data condition), so summing them
# would trip a cap neither reached; `note_checked_slice()` resets it on an
# unrelated event; and its disable message names the wrong mechanism.
MAX_INCONCLUSIVE_SLICES = 16

# Keyed by CONFIGURATION (`compute._certify_budget_key`), NOT by `cert_key`:
# a streak is evidence about ONE execution
# configuration, so a sibling group's verdict must not spend or clear it.
_INCONCLUSIVE_STREAKS: dict = {}


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


def note_inconclusive_certification(key) -> bool:
    """Record an inconclusive certification for `key`; True when retries must
    stop.

    Consecutive slices, not a lifetime count: one degenerate slice in an
    otherwise clean corpus must not push the run toward a disable, so any
    slice that reaches a verdict clears the streak via
    :func:`note_conclusive_certification`.

    Per CONFIGURATION (`compute._certify_budget_key`), not per run and NOT per
    `cert_key` -- `cert_key` carries the corpus slice height and the per-file
    `exact_fp16`, both of which churn, which made the cap unfireable in the
    default configuration. A run carries several execution
    configurations; with one shared counter a healthy group would clear an
    unhealthy sibling's streak (so its cap could never fire) and N groups per
    slice would trip a "16 consecutive slices" cap after 16/N slices.
    """
    n = _INCONCLUSIVE_STREAKS.get(key, 0) + 1
    _INCONCLUSIVE_STREAKS[key] = n
    return n >= MAX_INCONCLUSIVE_SLICES


def note_conclusive_certification(key) -> None:
    """Clear `key`'s streak after a certification that reached a verdict.

    Called for a PASS and for a refusal alike: both are evidence that
    certification is able to judge this configuration, which is what the
    streak is counting the absence of.
    """
    _INCONCLUSIVE_STREAKS.pop(key, None)


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


def set_rescore_mode(mode: str | None) -> None:
    """Record the configured pass-two mode for this run (`None` = default)."""
    global _RESCORE
    if mode is not None and mode not in RESCORE_MODES:
        raise ValueError(
            f"two-pass: rescore mode must be one of {RESCORE_MODES}, got "
            f"{mode!r}")
    _RESCORE = mode


def rescore_mode() -> str:
    """The pass-two mode in effect: env override, else config, else default."""
    global _RESCORE_WARNED

    raw = os.environ.get("NOVA_BF_TWOPASS_RESCORE")
    if raw:
        v = raw.strip().lower()
        if v in RESCORE_MODES:
            return v
        if not _RESCORE_WARNED:
            _RESCORE_WARNED = True
            logger.warning("NOVA_BF_TWOPASS_RESCORE=%r is not one of %s; "
                           "ignoring it", raw, RESCORE_MODES)
    if _RESCORE is not None:
        return _RESCORE
    from nova_bf.config import TWO_PASS_RESCORE_DEFAULT

    return TWO_PASS_RESCORE_DEFAULT


def tiles_mode() -> bool:
    return rescore_mode() == "tiles"


# `params.pipeline_pass_one`, recorded per run by `run_compute`.
_PIPELINE = None


def set_pipeline_pass_one(on: bool | None) -> None:
    """Record the configured pass-one pipelining for this run (`None` = default)."""
    global _PIPELINE
    _PIPELINE = None if on is None else bool(on)


def pipeline_pass_one() -> bool:
    """Whether to launch dense pass one a slice ahead.

    NOVA_BF_PIPELINE_PASS_ONE overrides the configured value.
    Results remain exact, but added memory pressure may change
    tiles-mode fallback behavior and numerical reproducibility.
    """
    raw = os.environ.get("NOVA_BF_PIPELINE_PASS_ONE")
    if raw is not None and raw.strip() != "":
        v = raw.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
        raise ValueError(
            f"NOVA_BF_PIPELINE_PASS_ONE must be a boolean (1/0, true/false), "
            f"got {raw!r}")
    if _PIPELINE is not None:
        return _PIPELINE
    from nova_bf.config import PIPELINE_PASS_ONE_DEFAULT

    return PIPELINE_PASS_ONE_DEFAULT


def threshold(mode: str | None = None) -> float:
    """The engage cutoff on the live-fraction hint for `mode` (default: the
    mode in effect). Slice mode compares a ROW fraction against
    `NOVA_BF_TWOPASS_THRESHOLD` (default 0.5); tiles mode compares a TILE-pair
    fraction against `NOVA_BF_TWOPASS_TILE_THRESHOLD` (default 0.2)."""
    if (mode or rescore_mode()) == "tiles":
        return _fraction_env("NOVA_BF_TWOPASS_TILE_THRESHOLD",
                             DEFAULT_TILE_THRESHOLD)
    return _fraction_env("NOVA_BF_TWOPASS_THRESHOLD", DEFAULT_THRESHOLD)


def _fraction_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default

    def _bad(why: str) -> float:
        # Warn once per threshold, independently.
        global _THRESHOLD_WARNED, _TILE_THRESHOLD_WARNED
        if name == "NOVA_BF_TWOPASS_TILE_THRESHOLD":
            if not _TILE_THRESHOLD_WARNED:
                _TILE_THRESHOLD_WARNED = True
                logger.warning("%s=%r %s; using %s", name, raw, why, default)
        elif not _THRESHOLD_WARNED:
            _THRESHOLD_WARNED = True
            logger.warning("%s=%r %s; using %s", name, raw, why, default)
        return default

    try:
        got = float(raw)
    except ValueError:
        return _bad("is not a number")
    # Require a valid live fraction; NaN or out-of-range values could
    # silently change when two-pass engages.
    if not (0.0 <= got <= 1.0):
        return _bad("is not a live fraction in [0, 1]")
    return got


def force_engage() -> bool:
    """Engage the two-pass on a slice whose live-fraction hint is not seeded yet.

    `NOVA_BF_TWOPASS_FORCE_ENGAGE=1` treats a missing hint as 0.0. This only
    ever makes the two-pass run MORE often; it does not force any row to be
    pruned, which would manufacture false prunes and prove nothing. Off by
    default, so production behaviour is unchanged.
    """
    global _FORCE_ENGAGE_WARNED
    raw = os.environ.get("NOVA_BF_TWOPASS_FORCE_ENGAGE")
    if raw is None or raw == "":
        return False
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    # Validate and warn, like `threshold()` above. An unrecognised spelling
    # used to mean ON, so `...=off` silently removed the warm-up on a
    # production job with no message at all.
    if not _FORCE_ENGAGE_WARNED:
        _FORCE_ENGAGE_WARNED = True
        logger.warning("NOVA_BF_TWOPASS_FORCE_ENGAGE=%r is not a boolean; "
                       "using False", raw)
    return False
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
    """Why the fused pass one is not in use, or `None` if it is.

    Inside `deferred_fuse_disable()` a failure deferred there (`fuse_pending`)
    also counts, so later launches do not re-run a kernel that just failed."""
    if os.environ.get("NOVA_BF_NO_FUSED_ROWMAX"):
        return "NOVA_BF_NO_FUSED_ROWMAX"
    if _gemm_rowmax is None:
        return f"triton unavailable ({_UNAVAILABLE})"
    if _FUSE_OFF is None and _FUSE_DEFERRING and _FUSE_PENDING is not None:
        return _FUSE_PENDING
    return _FUSE_OFF

# Defer fused-kernel failures during pipelined pass one so they do not
# affect earlier slices. Apply or discard the failure in phase B,
# preserving unpipelined behavior.d -- or drops it
# (`fuse_drop_pending`) when phase B discards it.
_FUSE_DEFERRING = False
_FUSE_PENDING: str | None = None


@contextlib.contextmanager
def deferred_fuse_disable():
    """Defer `fuse_disable` for the duration (phase A of a pipelined slice)."""
    global _FUSE_DEFERRING
    prev, _FUSE_DEFERRING = _FUSE_DEFERRING, True
    try:
        yield
    finally:
        _FUSE_DEFERRING = prev


def fuse_pending() -> str | None:
    """A fused-kernel failure deferred by a launch and not yet applied/dropped."""
    return _FUSE_PENDING


def fuse_apply_pending(reason: str) -> None:
    """Apply a deferred failure: the consumer reached the point where the
    unpipelined run's own pass one would have hit it."""
    global _FUSE_PENDING
    if _FUSE_PENDING == reason:
        _FUSE_PENDING = None
    fuse_disable(reason)


def fuse_drop_pending(reason: str) -> None:
    """Forget a deferred failure whose launch nothing used. If the unpipelined
    run would still reach that kernel, its inline pass one fails (and disables
    the kernel) there itself."""
    global _FUSE_PENDING
    if _FUSE_PENDING == reason:
        _FUSE_PENDING = None


def fuse_disable(reason: str) -> None:
    """Disable the fused pass-one kernel for the rest of this run.

    `reset()` clears this state so the next run can attempt the fused path
    again under its own device and workload conditions. Inside
    `deferred_fuse_disable()` the first reason is recorded instead (see
    `_FUSE_PENDING`).
    """
    global _FUSE_OFF, _FUSE_PENDING
    if _FUSE_DEFERRING and _FUSE_OFF is None:
        if _FUSE_PENDING is None:
            _FUSE_PENDING = reason
            logger.debug("two-pass: fused pass-one failure deferred to the "
                         "slice's decision: %s", reason)
        return
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

def fused_rowmax(Qh, Ch, cs, config=None, probe=False, with_tiles=False):
    """Compute pass-one row maxima without materializing the Gram matrix.

    Returns the float32 `(M,)` row maxima, or `None` when the fused kernel
    declines. `probe=True` prevents candidate-specific failures during
    autotuning from disabling the fused path for the run.

    `with_tiles=True` returns `(row_max, part, BLOCK_N)` instead: `part` is
    the kernel's own `(n_tiles, M)` buffer of per-(BLOCK_N-column tile, row)
    maxima that the row max is reduced from -- returned, not recomputed, so
    the kernel and its arithmetic are unchanged.
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
        row_max = part[0] if n_tiles == 1 else part.amax(dim=0)
        if with_tiles:
            return row_max, part, bn
        return row_max

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
    # A string, not a count: the reset loop would otherwise write integer 0.
    "last_refusal_reason",
    "audit_closest_margin", "probe_accum_worst_rel", "shortfall_worst_ratio",
    "rowmax_worst_ratio", "dead_audit_worst",
    "tile_width", "tiles_fallback_reason", "tiles_kernel_decline_reason",
    "tiles_reference_reason",
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
    # The reason string from the most recent guard refusal, PURELY for
    # diagnosis. Every refusal site already computes one ("corpus norm",
    # "column scale", ...) and `_fill` used it only as a truthy flag, dropping
    # the text -- so a declined slice said nothing about WHICH guard fired.
    # Working that out cost three GPU round-trips on nytimes-256, and the
    # first answer was wrong (prunability_cut blamed; norm_guard actual).
    #
    # NOTHING READS THIS FOR CONTROL FLOW, and it must stay that way: the
    # decision of what pass one did is taken from the `PassOneOutcome` that
    # `upper_bounds` RETURNS. A string that steers behaviour is a classifier,
    # and a classifier has to be right at every site. A string that steers behaviour is a
    # classifier, and a classifier has to be right at every site.
    "last_refusal_reason": None,
    # Worst absolute pass-one maximum relative to Theorem 1's cap,
    # including tile maxima. Values above 1 force rows live.
    "rowmax_worst_ratio": None,
    "probe_accum_worst_rel": None,
    "probe_accum_fitted": 0.0,
    # Worst observed underestimation relative to the bound's leading margin,
    # normalized by score scale. Values above 0.75 trigger a warning.
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

    # Tiles-mode pass two; also counted in `slices_twopass`.
    "slices_tiles": 0,                # slices rescored using tiles
    "tile_pairs_full": 0,             # total (query, tile) pairs
    "tile_pairs_live": 0,             # pairs rescored exactly
    "tile_rows_rescored": 0,          # live pairs x tile width (upper bound)
    "tiles_kernel": 0,                # Triton rescoring calls
    "tiles_reference": 0,             # torch reference calls

    # Tile rescoring fallbacks and most recent reason.
    "tiles_fallbacks": 0,
    "tiles_fallback_reason": None,
    "tile_width": None,               # most recent pass-one tile width

    # Tile-level bound and pruning audits.
    "audit_tile_pairs": 0,
    "dead_audit_tile_pairs": 0,

    # Fallback to slice mode when tiles are inefficient or unsupported.
    "tiles_crossover_bails": 0,
    "tiles_kernel_declined": 0,
    "tiles_kernel_decline_reason": None,
    "tiles_reference_reason": None,   # why the CPU reference ran
    "tiles_bound_oom": 0,             # tile-bound OOM; use coarser bounds

    # Kernel validation against the reference.
    "tiles_selfchecks": 0,
    "tiles_selfcheck_failures": 0,

    # Pass-one pipelining: launches are consumed or discarded exactly once.
    # Retries are discarded launches rerun inline. Discarded launches that
    # executed a kernel also contribute to `slices_discarded`.
    "pipeline_launched": 0,
    "pipeline_used": 0,
    "pipeline_discarded": 0,
    "pipeline_retried": 0,
}

# Bookings that wait for a device read nobody has made yet (a discarded
# pipelined launch's verdict: `compute._tp_discard`). Settled at the next
# point that has just synchronised anyway, and always before counters are
# read (`stats`).
_DEFERRED_BOOKINGS: list = []


def defer_booking(fn) -> None:
    """Run `fn()` (it may read the device) at the next `flush_bookings`."""
    _DEFERRED_BOOKINGS.append(fn)


def flush_bookings() -> None:
    """Settle every deferred booking, in the order it was deferred."""
    while _DEFERRED_BOOKINGS:
        _DEFERRED_BOOKINGS.pop(0)()


def stats() -> dict:
    """Return two-pass counters, audit settings, and runtime status."""
    flush_bookings()
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

    # Pass-two mode, and the engage gate it used: which fraction the hint is
    # (row or tile pairs) and the cutoff it was compared against.
    mode = rescore_mode()
    out["rescore"] = mode
    out["engage_fraction"] = "tile" if mode == "tiles" else "row"
    out["engage_threshold"] = threshold(mode)

    # The RESOLVED pass-one pipelining policy (env over config). It only had an
    # effect if `pipeline_launched > 0`.
    out["pipeline_pass_one"] = pipeline_pass_one()

    return out

def reset_stats() -> None:
    global _VERIFY_SKIPPED

    for key in _STATS:
        # Preserve `None` for statistics where zero would imply a measurement.
        _STATS[key] = None if key in _UNMEASURED_IS_NONE else 0

    # Bookings deferred until a verdict is read belong to the counters just
    # cleared: settling them later would book the old ones into the new.
    _DEFERRED_BOOKINGS.clear()


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
    _clear_device_caches()
def reset() -> None:
    """Reset all state scoped to a single run.

    Device-level autotuning results are retained, while caches, counters,
    certification, verification, disable state, and warn-once flags are reset.
    """
    global _DISABLED_REASON, _FUSE_OFF, _FUSE_LAUNCHES, _VERIFY_SKIPPED, _ROWMAX_MAG_WARNED, _AUDIT_WARNED, _AUDIT_RATE, _ROWMAX_USED, _FORCE_ENGAGE_WARNED
    global _THRESHOLD_WARNED, _FUSE_CONFIG_WARNED, _ROWMAX_WARNED
    global _UNCHECKED_WARNED, _ROWMAX_OFF, _FUSE_OOM_WARNED
    global _VERIFY_OOM_WARNED, _UNCHECKED_STREAK, _CERTIFIED
    global _RESCORE, _RESCORE_WARNED, _TILE_THRESHOLD_WARNED

    _QCACHE.clear()
    _clear_device_caches()
    # A previous run's unsettled bookings belong to that run (it raised, or
    # never read its counters): dropped, not booked into this one.
    _DEFERRED_BOOKINGS.clear()
    reset_stats()
    _SHAPE_OK.clear()
    # The configured pass-two mode is per run; `run_compute` sets it again.
    _RESCORE = None
    # So is the pass-one pipelining.
    global _PIPELINE
    _PIPELINE = None
    _RESCORE_WARNED = False
    try:
        from nova_bf import tile_rescore as _tr

        _tr.reset()
    except ImportError:  # pragma: no cover - torch-free installs
        pass
    _VERIFY_SKIPPED = False
    _DISABLED_REASON = None
    # Per-RUN, like every other warn-once flag here: soak_twopass.run() calls
    # reset() between configurations, and a malformed value should warn on each.
    _FORCE_ENGAGE_WARNED = False

    # Retry fused execution on the next run; keep device autotuning results.
    _FUSE_OFF = None
    global _FUSE_PENDING, _FUSE_DEFERRING
    _FUSE_PENDING = None
    _FUSE_DEFERRING = False
    _FUSE_LAUNCHES = 0
    _FUSE_KERNEL_INFO.clear()

    # Reset per-run warning state and cached configuration.
    _THRESHOLD_WARNED = False
    _TILE_THRESHOLD_WARNED = False
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
    global _ROWMAX_RUNS
    _ROWMAX_RUNS = 0
    _FUSE_OOM_WARNED = False
    _VERIFY_OOM_WARNED = False
    _UNCHECKED_STREAK = 0
    _INCONCLUSIVE_STREAKS.clear()
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

    # Cache one entry per Q; changing `row_scale` replaces it.
    # Pass one relies on cache hits to avoid repeated host synchronization.
    # Mixed scales for the same Q are currently excluded from two-pass.
    # Use (id(Q), id(row_scale)) if that changes.
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


def corpus_side(Cb, col_scale, cn=None, device_scalars=False):
    """Prepare the corpus-side inputs and scalars required by the bound.

    Returns `(C_h, sigma_max, cn_min, cn_max)`, where `sigma_max` is the
    largest column scale actually applied by pass one.

    `device_scalars=True` (pass one's own path) returns the three scalars as
    0-d binary32 DEVICE tensors -- the same reductions, not read back -- so
    nothing here synchronises; `sigma_max` is then `None` when no column
    scale is applied (it is exactly 1.0).
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

    if device_scalars:
        # The same reductions as below, left on the device. `max()` raises on
        # an empty `col_scale` exactly as `float(col_scale.max())` did.
        sig = None if col_scale is None else col_scale.max()
        return Ch.contiguous(), sig, cn.min(), cn.max()

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

def tile_amax(x, width: int):
    """Per-tile maxima of a 2-D `(rows, n)` tensor: `(rows, ceil(n / width))`.

    A partial last tile is padded with `-inf`, so it reduces over its real
    columns only -- the same masking `_gemm_rowmax` applies to wrapped columns.
    """
    import torch

    rows, n = int(x.shape[0]), int(x.shape[1])
    n_tiles = -(-n // width)
    pad = n_tiles * width - n
    if pad:
        x = torch.nn.functional.pad(x, (0, pad), value=float("-inf"))
    return x.reshape(rows, n_tiles, width).amax(dim=2)


def _tile_scaled(part, row_scale):
    """Tile maxima in score units: `approx * row_scale`'s operation, per tile."""
    return part * row_scale[None, :] if row_scale is not None else part


def default_tile_width(device) -> int:
    """Return the tile width when pass one has not produced tile maxima.

    CUDA uses the last kernel width or configured fallback; CPU uses
    CPU_TILE. Width affects decision granularity, not pruning safety.
    """
    if str(device).startswith("cuda"):
        got = _STATS.get("tile_width")
        if got:
            return int(got)
        ran = _FUSE_KERNEL_INFO.get("config")
        if ran:
            return int(ran[1])
        raw = os.environ.get("NOVA_BF_FUSE_CONFIG")
        if raw:
            try:
                cfg = tuple(int(x) for x in raw.split(","))
                if _config_ok(cfg):
                    return int(cfg[1])
            except ValueError:
                pass
        return int(FUSE_CONFIG[1])
    return int(CPU_TILE)


def approx_rowmax(Qh, Ch, col_scale, out_dtype, with_tiles=False):
    """Compute the approximate scaled row maximum for pass one.

    Returns `(out, fused)`. When `fused` is true, the row max is taken directly
    from the float32 GEMM accumulator and no approximate Gram is materialized.
    Otherwise the Gram is materialized and reduced separately.

    `with_tiles=True` returns `(out, fused, part, width)`, where `part` is the
    `(n_tiles, n_q)` per-(tile of `width` corpus rows, query) maximum in the
    same scaled units as `out` (whose row max it is). The fused path hands
    back its kernel buffer with `width = BLOCK_N`; the unfused path reduces
    the materialized product over `CPU_TILE`-wide tiles.
    """
    import torch

    global _ROWMAX_WARNED, _ROWMAX_OFF

    _pin_accumulation()
    n_cols = int(Ch.shape[0])
    if col_scale is None:
        cs = torch.ones(n_cols, dtype=torch.float32, device=Ch.device)
    else:
        cs = col_scale.contiguous()

    got = fused_rowmax(Qh, Ch, cs, with_tiles=with_tiles)
    if got is not None:
        _STATS["slices_fused"] += 1
        if with_tiles:
            return got[0], True, got[1], got[2]
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
    global _ROWMAX_USED, _ROWMAX_RUNS

    if (_rowmax_scaled is not None and not _ROWMAX_OFF and not with_tiles
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
            _ROWMAX_RUNS += 1
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
    width = int(CPU_TILE)
    part = None
    try:
        if with_tiles:
            part = torch.empty((-(-n_cols // width), n_q), dtype=torch.float32,
                               device=g.device)
        for r0 in range(0, n_q, chunk):
            blk = g[r0 : r0 + chunk].float() * cs[None, :]
            out[r0 : r0 + chunk] = blk.amax(dim=1)
            if part is not None:
                # The same scaled values the row max reads, so the row max is
                # exactly the max over these tiles.
                part[:, r0 : r0 + chunk] = tile_amax(blk, width).T
            del blk
    except Exception as exc:  # noqa: BLE001 - see above
        raise _as_pass_one_failure(exc, "the portable row-max copy") from exc

    if with_tiles:
        return out, False, part, width
    return out, False

def upper_bounds(Q, Cb, col_scale, row_scale, out_dtype, metric,
                 cn=None, corpus_exact_fp16=None, with_parts=False, with_outcome=False,
                 tiles=None):
    """Return a safe per-query upper bound on this corpus slice's best score.

    Exactly `upper_bounds_finish(upper_bounds_begin(...))`; see `_upper_bounds`
    for the contract. The split exists only so `compute` can launch a slice's
    pass one before the previous slice's bookkeeping (`params.pipeline_pass_one`).
    """
    return upper_bounds_finish(upper_bounds_begin(
        Q, Cb, col_scale, row_scale, out_dtype, metric, cn=cn,
        corpus_exact_fp16=corpus_exact_fp16, with_parts=with_parts,
        with_outcome=with_outcome, tiles=tiles, speculate=False))


class PendingBound:
    """A pass-one launch awaiting completion or discard.

    Holds the suspended `_upper_bounds` generator after the row-max launch,
    before guard readback and bound computation. Must be consumed exactly
    once by `upper_bounds_finish` or `upper_bounds_discard`.

    `event` synchronizes side-stream results with the consumer.
    `probe` holds device-side guard results; `fused` and `delta` track
    counter changes for rollback. `refusal` stores the resolved guard reason.
    """

    __slots__ = ("_gen", "_result", "event", "device", "probe", "fused",
                 "verdict", "delta", "taken_back")

    def __init__(self, gen, device):
        self._gen = gen
        self._result = None
        self.event = None
        self.device = device
        self.probe = None
        self.fused = None
        self.verdict = [None, None]
        self.delta = None
        self.taken_back = False

    @property
    def refusal(self):
        """The guard refusal reason, once read; None otherwise."""
        return self.verdict[0]

    @property
    def launched(self) -> bool:
        """Whether a row-max launch is pending (False: an early exit, whose
        result is already final)."""
        return self._gen is not None

    def _join(self) -> None:
        if self.event is not None:
            import torch

            # Device-side only: the consumer's stream waits for the side
            # stream's launch. No host synchronization.
            torch.cuda.current_stream(self.device).wait_event(self.event)
            self.event = None


class _Launch(typing.NamedTuple):
    """What `_upper_bounds` yields at its launch/finish boundary."""

    event: object      # side-stream completion event, or None
    probe: object      # (len(SLICE_REFUSALS) + 2,) float64 device tensor
    fused: object      # which counter approx_rowmax incremented (True:
                       # slices_fused, False: slices_unfused); None: no launch
    verdict: list      # [refusal reason or None, probe values once read]
    delta: object      # `_launch_delta` of the launch, or None (no launch)


def upper_bounds_begin(Q, Cb, col_scale, row_scale, out_dtype, metric,
                       cn=None, corpus_exact_fp16=None, with_parts=False,
                       with_outcome=False, tiles=None, stream=None,
                       speculate=True):
    """Run pass one through the row-max kernel launch, deferring bound computation.

    Guards run on the device without host synchronization; pass
    `corpus_exact_fp16` to avoid a readback. Host-side errors still raise
    here. With `stream`, the kernel runs on a side stream after waiting
    for the current stream. No thresholds are read.

    `speculate=True` launches before resolving guards, rolling back work
    on refusal. `False` checks guards first and skips refused launches.
    """
    verdict = [None, None]
    gen = _upper_bounds(Q, Cb, col_scale, row_scale, out_dtype, metric,
                        cn=cn, corpus_exact_fp16=corpus_exact_fp16,
                        with_parts=with_parts, with_outcome=with_outcome,
                        tiles=tiles, stream=stream, speculate=speculate,
                        _verdict=verdict)
    pending = PendingBound(gen, Q.device)
    pending.verdict = verdict
    try:
        got = next(gen)
    except StopIteration as stop:
        pending._gen = None
        pending._result = stop.value
        return pending
    (pending.event, pending.probe, pending.fused, pending.verdict,
     pending.delta) = got
    return pending


def resolve_refusal(pending: PendingBound):
    """Read a pending launch's guard verdict without consuming it.

    Synchronizes and returns the refusal reason or None. Used by phase B
    to resolve deferred failures before finishing the bound.
    """
    if (pending.refusal is None and pending.probe is not None
            and pending.verdict[1] is None):
        # Kept, so the finish half reuses this read instead of repeating it.
        pending.verdict[1] = _read_probe(pending.probe)
        pending.verdict[0] = _probe_refusal(pending.verdict[1])
    return pending.refusal


def upper_bounds_finish(pending: PendingBound):
    """The bound `upper_bounds` would have returned for this pending pass one."""
    gen, pending._gen = pending._gen, None
    if gen is None:
        res, pending._result = pending._result, None
        return res
    pending._join()
    try:
        gen.send(None)
    except StopIteration as stop:
        return stop.value
    raise RuntimeError("two-pass: _upper_bounds yielded more than once")


def upper_bounds_discard(pending: PendingBound, resolve: bool = False):
    """Discard a pending pass one without computing its bound. Idempotent.

    Waits on the launch to ensure safe GPU buffer reuse.
    `resolve=True` synchronizes and resolves guard refusals, rolling back
    speculative side effects. Otherwise, resolution can be deferred via
    `discard_resolved`.

    Unresolved discards (e.g., exception cleanup) may retain speculative
    kernel state.
    """
    gen, pending._gen = pending._gen, None
    pending._result = None
    if gen is None:
        return pending.refusal
    pending._join()
    gen.close()
    if resolve:
        discard_resolved(pending)
    return pending.refusal


def discard_resolved(pending: PendingBound):
    """The verdict of a DISCARDED launch, read now (a sync), with the launch's
    non-`_STATS` side effects taken back if it was refused. Idempotent."""
    resolve_refusal(pending)
    if (pending.refusal is not None and pending.delta is not None
            and not pending.taken_back):
        pending.taken_back = True
        _take_back(pending.delta, stats=False)
    return pending.refusal


def _read_scalars(*xs):
    """0-d device scalars as host floats, in ONE device->host transfer."""
    import torch

    return torch.stack([x.reshape(()).to(torch.float64) for x in xs]).tolist()


def _read_probe(probe, *extra):
    """The probe (and any `extra` 0-d device scalars) as host floats, in ONE
    device->host transfer. The only synchronisation of a pass one's guards."""
    import torch

    if extra:
        probe = torch.cat([probe] + [e.reshape(1).to(probe.dtype)
                                     for e in extra])
    return probe.tolist()


def _probe_refusal(vals) -> str | None:
    """The first `SLICE_REFUSALS` entry that fired, in the host order."""
    for reason, hit in zip(SLICE_REFUSALS, vals):
        if hit != 0.0:
            return reason
    return None


# Side effects restored when a speculative launch is guard-refused.
# Counters are rolled back by per-launch deltas, preserving other launches.
# Kernel state and warn-once flags are restored only when no later launch
# depends on them. Logged warnings and autotuning state are not reverted.
# Deferred fused-kernel failures are handled separately by phase B.
_LAUNCH_FLAGS = ("_FUSE_OFF", "_FUSE_OOM_WARNED", "_ROWMAX_OFF",
                 "_ROWMAX_WARNED")

# Track launch order to avoid restoring flags used by later launches.
# Such flags remain set to preserve fallback behavior, though a
# data-specific failure may disable a kernel earlier than necessary.
_LAUNCH_GEN = 0


def _launch_mark():
    global _LAUNCH_GEN
    g = globals()
    _LAUNCH_GEN += 1
    return ({k: _STATS[k] for k in ("slices_fused", "slices_unfused")},
            _FUSE_LAUNCHES, _ROWMAX_RUNS, {n: g[n] for n in _LAUNCH_FLAGS},
            _LAUNCH_GEN)


def _launch_delta(mark):
    """What the launch since `mark` changed: counter deltas, and the flags it
    flipped as `name: (before, after)`."""
    g = globals()
    stats0, fl0, rr0, flags0, gen = mark
    return {
        "gen": gen,
        "stats": {k: _STATS[k] - v for k, v in stats0.items() if _STATS[k] != v},
        "fuse_launches": _FUSE_LAUNCHES - fl0,
        "rowmax_runs": _ROWMAX_RUNS - rr0,
        "flags": {n: (v, g[n]) for n, v in flags0.items() if g[n] != v},
    }


def _take_back(delta, stats: bool = True) -> None:
    """Undo a refused launch's side effects (see `_LAUNCH_FLAGS`). `stats=False`
    leaves `_STATS` to the caller (a discarded pipelined launch whose counter
    writes were never booked)."""
    global _FUSE_LAUNCHES, _ROWMAX_RUNS, _ROWMAX_USED
    g = globals()
    if stats:
        for k, v in delta["stats"].items():
            _STATS[k] -= v
    if delta["fuse_launches"]:
        _FUSE_LAUNCHES -= delta["fuse_launches"]
        if _FUSE_LAUNCHES <= 0:
            _FUSE_LAUNCHES = 0
            _FUSE_KERNEL_INFO.clear()
    if delta["rowmax_runs"]:
        _ROWMAX_RUNS -= delta["rowmax_runs"]
        if _ROWMAX_RUNS <= 0:
            _ROWMAX_RUNS = 0
            _ROWMAX_USED = False
    if _LAUNCH_GEN != delta["gen"]:
        # A later launch already ran under these flags: see `_LAUNCH_GEN`.
        return
    for n, (before, after) in delta["flags"].items():
        if g[n] == after:
            g[n] = before


def _query_refusals(qs, d, fmt_q, row_scale):
    """Return query-side refusal mask and whether all rows are refused.

    Checks norms, scaling, overflow, and prunability. Results are cached
    per query state, avoiding repeated host synchronization across slices.
    """
    import torch

    key = ("query_refusals", fmt_q.name, d)
    got = qs.get(key)
    if got is None:
        bad = qs["bad"]
        if row_scale is not None:
            # A bad query scale forces only that query row live.
            bad = bad | scale_mismatch_mask(row_scale, qs["qn"])
        bad = bad | overflow_mask(d, qs["qn"], fmt_q)
        qn = qs["qn"]
        bad = bad | ~(torch.isfinite(qn) & (qn >= _cf.PRUNE_MIN_QNORM))
        got = qs[key] = (bad, bool(bad.all()))
    return got


def _upper_bounds(Q, Cb, col_scale, row_scale, out_dtype, metric,
                  cn=None, corpus_exact_fp16=None, with_parts=False,
                  with_outcome=False, tiles=None, stream=None,
                  speculate=False, _verdict=None):
    """Compute safe per-query upper bounds for a corpus slice.

    Yields once after the row-max launch, allowing bound computation and
    guard readback to finish later without synchronizing before launch.

    Guard failures return +inf for affected rows; empty slices return -inf.
    `corpus_exact_fp16` avoids repeating the per-file format check.

    `tiles` receives per-tile bounds and width when available. Tile bounds
    never exceed their per-query bounds; invalid values fall back to the
    per-query bound. An empty dict means no tile maxima were computed.

    Guards run on the device before launch, with refusals resolved afterward.
    Speculative launches roll back kernel side effects on refusal;
    non-speculative launches check the verdict before running the kernel.
    """
    import numpy as np
    import torch

    n_q = int(Q.shape[0])
    d = int(Q.shape[1])
    dev = Q.device
    # [guard refusal reason], shared with the PendingBound (`_verdict`) so a
    # refusal reported BEFORE the yield (a failed speculative launch) is
    # visible to phase B too.
    verdict = [None, None] if _verdict is None else _verdict

    # The outcome travels OUT of this function rather than being reconstructed
    # from module globals by the caller. A one-element list because the helpers
    # below are closures, not because anything is shared.
    # Seeded None, not a kind: every exit today assigns one, but the WRONG
    # default here fails open. GUARD_REFUSED would mean "no kernel ran, retry",
    # so a `return _wrap(...)` added before the success assignment would
    # silently burn the retry budget and then disable the run naming a guard
    # that never fired. `None` makes that a loud AssertionError instead.
    _outcome = [None]

    def _wrap(res):
        if not with_outcome:
            return res
        if _outcome[0] is None:
            # FAIL CLOSED, and not with `assert`: `python -O` strips asserts,
            # after which this would return `(res, None)` and the caller would
            # raise AttributeError out of `_twopass_prepare`, killing the rank
            # mid-run instead of falling back to one-pass. A GUARD_REFUSED
            # outcome is the safe reading -- it certifies nothing and prunes
            # nothing -- and the reason says where to look.
            logger.error(
                "two-pass: upper_bounds reached a return without recording a "
                "pass-one outcome; treating the slice as refused. A new exit "
                "path is missing its _fill/all_live or success assignment.")
            return (res, PassOneOutcome(PASS_ONE_GUARD_REFUSED,
                                        "no outcome recorded"))
        return (res, _outcome[0])

    def _book(reason, kind):
        if reason:
            _STATS["slices_guard_refused"] += 1
            _STATS["last_refusal_reason"] = reason
        _outcome[0] = PassOneOutcome(kind, reason)

    def _fill(v, reason=None, kind=PASS_ONE_GUARD_REFUSED):
        _book(reason, kind)
        t = torch.full((n_q,), v, dtype=torch.float32, device=dev)

        # Certification needs the approximate value and bound separately.
        return _wrap((t, t.clone(), torch.zeros_like(t)) if with_parts else t)

    def all_live(reason=None, kind=PASS_ONE_GUARD_REFUSED):
        return _fill(float("inf"), reason, kind)

    if int(Cb.shape[0]) == 0:
        # The maximum over an empty corpus is -inf.
        return _fill(float("-inf"), kind=PASS_ONE_EMPTY)

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
    # run-cached half copy and Ch is a fresh half copy for every corpus slice,
    # and the guards below are small device tensors. Keep them inside the same
    # OOM-to-all-live boundary as the row-max GEMM. Otherwise a conversion OOM
    # escapes before `approx_rowmax` gets its chance to translate it into the
    # safe one-pass fallback.
    try:
        qs = query_side(Q, row_scale)
        if cn is None:
            # Compute corpus norms in binary32, as required by the bound.
            cn = Cb.float().norm(dim=1)

        # Select the conversion-error model from the actual corpus representation.
        if corpus_exact_fp16 is None:
            # NOTE: reads the corpus back (a sync). Callers on the pipelined
            # path pass the per-file verdict instead.
            corpus_exact_fp16 = float32_is_exactly_fp16(Cb)
        fmt_c = _cf.EXACT if corpus_exact_fp16 else _cf.FP16
        fmt_q = stored_format(Q)

        # Corpus guards are slice-wide; query guards are per row. The
        # scalars stay on the device (no `float()`): see SLICE_REFUSALS.
        Ch, sig_t, cn_min_t, cn_max_t = corpus_side(Cb, col_scale, cn,
                                                    device_scalars=True)
        # Widened ONCE
        cn_max_t = cn_max_t.double()
        if sig_t is not None:
            sig_t = sig_t.double()

        false = torch.zeros((), dtype=torch.bool, device=dev)
        refused = [
            # "corpus norm": every N_c within [NORM_MIN, NORM_MAX] (NaN fails).
            norm_guard(cn).any(),
            # "corpus overflow": kappa_bar(d) * max N_c <= Omega (1 - margin).
            overflow_refused_t(d, cn_max_t, fmt_c),
            # "corpus scale": sigma_max in range (or exactly 1).
            false if sig_t is None else scale_range_refused_t(sig_t),
        ]
        # "column scale": each sigma within (1 +- u) / N_c of the SAME norms.
        if col_scale is None:
            refused.append(false)
        elif col_scale.shape != cn.shape:
            # Decided on the host, from shapes: no read.
            refused.append(torch.ones((), dtype=torch.bool, device=dev))
        else:
            refused.append(scale_mismatch_mask(col_scale, cn).any())

        bad, q_refuses_all = _query_refusals(qs, d, fmt_q, row_scale)
        # A refusal the HOST already knows is certain -- from shapes, or from
        # the cached query side -- launches nothing, as the guards-first code
        # did; the device verdict still names the first reason that fired.
        certain = q_refuses_all or (col_scale is not None
                                    and col_scale.shape != cn.shape)
        # The guards' query halves are fixed for the query state: computed
        # once per query matrix, not per slice (`query_guard_halves`).
        halves = qs.get("guard_halves")
        if halves is None:
            halves = qs["guard_halves"] = query_guard_halves(qs["qn"])
        bad = bad | product_guard_t(qs["qn"], cn_max_t, halves)
        bad = bad | prunability_cut_t(qs["qn"], cn_min_t, halves)
        # "all rows refused": checked (and reported) only after the corpus
        # guards, as before -- the reason is the FIRST that fired.
        refused.append(bad.all())

        flags = torch.stack(refused)
        # THE FOLD: a slice-wide refusal marks every row, so the bound is
        # `+inf` (prunes nothing) on the device, before anyone reads anything.
        bad = bad | flags.any()

        probe = torch.cat((
            flags.to(torch.float64),
            (torch.ones((), dtype=torch.float64, device=dev) if sig_t is None
             else sig_t.double()).reshape(1),
            cn_max_t.double().reshape(1),
        ))
        del refused, false, cn_min_t
    except Exception as exc:  # noqa: BLE001 - preserve non-allocation errors
        failure = _as_pass_one_failure(exc, "the pass-one input copies")
        if isinstance(failure, PassOneUnavailable):
            _STATS["slices_pass_one_oom"] += 1
            return all_live(kind=PASS_ONE_OOM)
        raise failure

    def _rowmax():
        if tiles is not None:
            return approx_rowmax(qs["Qh"], Ch, col_scale, out_dtype,
                                 with_tiles=True)
        approx, fused = approx_rowmax(qs["Qh"], Ch, col_scale, out_dtype)
        return approx, fused, None, None

    def _refused(reason, delta):
        """The guards-first report for a slice the verdict refused: take back
        whatever a speculative launch changed (`delta`), book the refusal
        exactly as before, and return `+inf` everywhere."""
        if delta is not None:
            _take_back(delta)
        verdict[0] = reason
        _book(reason, PASS_ONE_GUARD_REFUSED)
        if approx is None:
            t = torch.full((n_q,), float("inf"), dtype=torch.float32, device=dev)
        else:
            # The device fold -- every row `bad`, so `+inf` -- not a fresh
            # constant: the fold is what makes a speculative launch safe.
            t = torch.where(bad, torch.full_like(approx, float("inf")), approx)
        return _wrap((t, t.clone(), torch.zeros_like(t)) if with_parts else t)

    # Non-speculative runs read guard results before launch, avoiding wasted
    # work on refused slices. Only pipelined phase A launches without readback.
    pre_vals = None
    approx = None
    if not speculate and not certain:
        pre_vals = verdict[1] = _read_probe(probe)
        reason = _probe_refusal(pre_vals)
        if reason is not None:
            return _refused(reason, None)

    event = None
    mark = None if certain else _launch_mark()
    try:
        if certain:
            # Host-known refusal: skip the kernel without synchronization.
            # The reported reason may still be an earlier corpus guard.
            fused, part, tile_w = None, None, None
        elif stream is None or not str(dev).startswith("cuda"):
            approx, fused, part, tile_w = _rowmax()
        else:
            approx, fused, part, tile_w, event = _rowmax_on(
                stream, dev, _rowmax, (qs["Qh"], Ch, col_scale))
        _outcome[0] = None if fused is None else PassOneOutcome(
            PASS_ONE_FUSED if fused
            else (PASS_ONE_CPU_WIDENED if not str(dev).startswith("cuda")
                  else PASS_ONE_UNFUSED_CUBLAS))
    except Exception as exc:  # noqa: BLE001 - re-raised unless refused
        if speculate:
            # A speculative failure may belong to a guard-refused slice that would
            # never have launched. Synchronize to resolve the verdict; ignore the
            # failure if refused.
            delta = _launch_delta(mark)
            approx = None
            verdict[1] = _read_probe(probe)
            reason = _probe_refusal(verdict[1])
            if reason is not None:
                return _refused(reason, delta)
        if not isinstance(exc, PassOneUnavailable):
            raise
        # `slices_unfused` may ALREADY have been incremented by approx_rowmax
        # before the unfused GEMM raised, so a counter delta would read this as
        # "cuBLAS ran" and take the hard refusal. The outcome says what
        # happened instead of what a side effect implies.
        _STATS["slices_pass_one_oom"] += 1
        return all_live(kind=PASS_ONE_OOM)

    # Capture this launch's side effects so a later refusal can roll them back.
    delta = None if mark is None else _launch_delta(mark)
    del Ch

    # Launch/finish boundary: speculative execution has not synchronized yet.
    yield _Launch(event, probe, fused, verdict, delta)

    if pre_vals is None and verdict[1] is not None:
        # Reuse the verdict if phase B already resolved it.
        pre_vals = verdict[1]
    try:
        extras = []
        if fused is None:
            # No kernel launched; the guard verdict determines the outcome.
            worst_t = None
        else:
            # Force non-finite approximate scores live.
            approx = torch.where(_finite(approx), approx,
                                _const(float("inf"), approx))
            if row_scale is not None:
                approx = approx * row_scale
            worst_t = _rowmax_worst(approx, metric)
            if worst_t is not None:
                extras.append(worst_t)

        # Compute scaled tile maxima and their magnitude cap.
        # Tile OOMs are handled only after reaching this step.
        t = t_abs = None
        tile_oom = False
        worst_tile_t = None
        if part is not None:
            try:
                t = _tile_scaled(part, row_scale)
                t_abs = t.abs()
                t_abs = torch.where(t_abs < float("inf"), t_abs,
                                    _const(0.0, t_abs)).amax(dim=0)
                worst_tile_t = _rowmax_worst(t_abs, metric)
            except Exception as exc:  # noqa: BLE001 - only an OOM is tolerated
                if not is_oom(exc):
                    raise
                tile_oom = True
                part = t = t_abs = None
            if worst_tile_t is not None:
                extras.append(worst_tile_t)

        # Read guard results, eps inputs, and magnitude caps together.
        if pre_vals is None:
            vals = _read_probe(probe, *extras)
            tail = vals[_PROBE_CN_MAX + 1:]
        else:
            vals = pre_vals
            tail = _read_scalars(*extras) if extras else []
    except Exception:
        # Ignore speculative errors on guard-refused slices, which would
        # never have executed this work. Otherwise propagate the error.
        got = pre_vals
        if got is None:
            got = verdict[1] = _read_probe(probe)
        reason = _probe_refusal(got)
        if reason is not None:
            approx = None
            return _refused(reason, delta)
        raise
    worst = tail.pop(0) if worst_t is not None else None
    worst_tile = tail.pop(0) if worst_tile_t is not None else None
    del probe, worst_t, worst_tile_t, extras

    reason = verdict[0] = _probe_refusal(vals)
    if reason is None and fused is None:
        # Unreachable: no launch means a certain refusal, which the verdict
        # reports. Fail closed rather than bound with no kernel.
        reason = verdict[0] = "all rows refused"
    if reason is not None:
        return _refused(reason, delta)

    # Only trusted accumulation paths may contribute pruning decisions.
    if not accumulator_is_ours(dev, fused):
        # A kernel DID run and the accumulator was not ours -- a definite
        # verdict, not an absence of one. Labelling it GUARD_REFUSED would make
        # certification call it INCONCLUSIVE and retry forever on a box that
        # genuinely cannot be trusted.
        return all_live("cublas accumulator", kind=PASS_ONE_UNFUSED_CUBLAS)

    # Recheck mutable CUDA exact-math state for every slice.
    why = exact_math_mode_flags(dev)
    if why:
        # A kernel DID run -- reaching here requires `accumulator_is_ours`
        # above, which on CUDA means the fused path. Letting this take
        # `all_live`'s GUARD_REFUSED default would report "no kernel ran" for a
        # slice where one demonstrably did, and point an operator at degenerate
        # corpus rows when the real cause is a TF32/emulation flag.
        return all_live("exact math mode", kind=_outcome[0].kind)

    # Fused output is reduced directly from float32 accumulation; an unfused
    # non-float32 Gram requires the output-rounding term.
    half_out = (out_dtype is not torch.float32) and not fused

    # Round this slice's parameters upward for conservative eps caching.
    # Each slice computes its own key from post-launch reductions, ensuring
    # cached eps always bounds its values. Only eps values are reused, not keys.
    sigma_max = 1.0 if col_scale is None else vals[_PROBE_SIGMA]
    cn_max = vals[_PROBE_CN_MAX]
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

    # Cache the device copy of eps to avoid repeated blocking H2D transfers.
    # Reuse only if the source array is unchanged and the tensor unmodified.
    got = qs.setdefault("eps_dev", {}).get(ekey)
    if got is not None and got[0] is eps_np and got[1]._version == got[2]:
        eps = got[1]
    else:
        eps = torch.as_tensor(np.ascontiguousarray(eps_np),
                              dtype=torch.float32, device=dev)
        qs["eps_dev"][ekey] = (eps_np, eps, eps._version)
    if eps.ndim == 0:
        eps = eps.expand(n_q)

    out = approx + eps

    # Guarded rows are always live.
    out = torch.where(bad, torch.full_like(out, float("inf")), out)

    # Reject rows whose approximate magnitude violates the bound's hypotheses.
    over = _check_rowmax_magnitude(approx, d, fmt_q, fmt_c, half_out, metric,
                                   worst=worst)

    if tile_oom:
        # Fall back to per-query bounds, which are safe for every tile.
        _STATS["tiles_bound_oom"] += 1
    elif part is not None:
        # Check every tile's magnitude, including large negative maxima.
        # Force entire rows live if any tile exceeds the bound's cap.
        over_t = _check_rowmax_magnitude(t_abs, d, fmt_q, fmt_c, half_out,
                                        metric, worst=worst_tile, what="tile")
        if over_t is not None:
            over = over_t if over is None else (over | over_t)
    del t_abs

    if over is not None:
        out = torch.where(over, torch.full_like(out, float("inf")), out)

    if part is not None:
        # Same operations in the same order as `approx`/`out` above, applied to
        # each tile maximum. Rounding is monotone, so a tile max <= the row max
        # gives a tile bound <= the row bound, bit for bit.
        try:
            t = t + eps[None, :]
            ok = _finite(part) & _finite(out)[None, :]
            tiles["upper"] = torch.where(ok, t, out[None, :].expand_as(t))
            tiles["width"] = int(tile_w)
            _STATS["tile_width"] = int(tile_w)
            del ok
        except Exception as exc:  # noqa: BLE001 - only an OOM is tolerated
            if not is_oom(exc):
                raise
            _STATS["tiles_bound_oom"] += 1
            tiles.clear()
        del part, t

    if with_parts:
        # `approx` is in the same scaled units as the exact score.
        return _wrap((out, approx, eps))

    return _wrap(out)


def _rowmax_on(stream, dev, rowmax, inputs):
    """Launch `rowmax()` on side `stream`; return its outputs plus an event.

    Cross-stream ownership, so the caching allocator never hands a block to
    one stream while the other may still use it:
    - `stream` first waits for the current stream, which produced `inputs`;
    - `inputs` (current-stream allocations the kernel reads) are recorded on
      `stream`, so freeing them waits for the kernel;
    - the outputs (allocated on `stream`) are recorded on the current stream,
      which is where they are consumed and freed. That consumer must wait on
      the returned event before touching them -- `upper_bounds_finish` and
      `upper_bounds_discard` both do.
    """
    import torch

    current = torch.cuda.current_stream(dev)
    stream.wait_stream(current)
    for t in inputs:
        if t is not None:
            t.record_stream(stream)
    with torch.cuda.stream(stream):
        approx, fused, part, tile_w = rowmax()
        event = torch.cuda.Event()
        event.record(stream)
    for t in (approx, part):
        if t is not None:
            t.record_stream(current)
    return approx, fused, part, tile_w, event


# Largest magnitude admitted for a scaled approximate cosine score.
# Exceeding it means the pass-one result violates the bound's assumptions.
def _rowmax_worst(approx, metric):
    """The largest finite `|approx|` as a 0-d DEVICE tensor (`-inf` when none
    is finite), for `_check_rowmax_magnitude(worst=...)`: what it reads
    with `float(finite.abs().max())`, left unread so the caller can batch the
    transfer. `None` when there is nothing to check (dot, or no rows)."""
    import torch

    if metric != "cosine" or approx.numel() == 0:
        return None
    # `|x| < inf` is `isfinite(x)` (see `_finite`), sharing the one `abs`.
    a = approx.abs()
    return torch.where(a < float("inf"), a, _const(float("-inf"), a)).amax()


_NOT_READ = object()


def _check_rowmax_magnitude(approx, d, fmt_q, fmt_c, half_out, metric,
                            worst=_NOT_READ, what="row"):
    """Return a per-row mask of approximate scores above the admitted cap.

    `worst`, if given, is `_rowmax_worst(approx, metric)` already read back
    (a float, or None for nothing to check), so this reads nothing itself.
    """
    global _ROWMAX_MAG_WARNED
    import torch

    if metric != "cosine":
        return None

    if worst is _NOT_READ:
        finite = approx[torch.isfinite(approx)]
        if finite.numel() == 0:
            return None
        worst = float(finite.abs().max())
    elif worst is None or worst == float("-inf"):
        # No rows, or no finite row max: nothing to cap, nothing to record.
        return None
    cap = (_cf.Lambda(d) * (1.0 + _cf.theta(d, _cf.C_HW_ENVELOPE))
           * (1.0 + fmt_q.u_t) * (1.0 + fmt_c.u_t)
           * ((1.0 + _cf.U16) if half_out else 1.0))

    _STATS["rowmax_worst_ratio"] = max(
        _STATS.get("rowmax_worst_ratio") or 0.0, worst / cap)

    if worst > cap and not _ROWMAX_MAG_WARNED:
        _ROWMAX_MAG_WARNED = True
        logger.warning(
            "two-pass: a first-pass %s maximum of magnitude %.6g exceeds the "
            "%.6g that Theorem 1 admits at d=%d. The affected rows will be "
            "forced live; check the applied scales and `probe_scales`.",
            what, worst, cap, d,
        )

    # `|x| < inf` is `isfinite(x)` (see `_finite`), sharing the one `abs`.
    a = approx.abs()
    return (a < float("inf")) & (a > cap)

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


class PassOneOutcome(typing.NamedTuple):
    """What pass one actually did, RETURNED rather than inferred.

    The caller used to reconstruct this by diffing module-global counters
    across the call (`slices_fused`, `slices_unfused`). That protocol rested on
    prose-only invariants -- exactly one increment site, read before a
    `finally` rolled the counters back -- and produced three defects in a row:
    a whole run disabled because "no kernel ran" was read as "cuBLAS ran"; a
    transient OOM *after* the unfused counter had already been bumped taking
    the hard cuBLAS refusal; and a diagnostic string that was always `None`
    because it was rolled back before the only reader saw it.

    A returned value cannot be rolled back, cannot be attributed to the wrong
    call, and cannot silently acquire a second writer.
    """

    kind: str
    reason: str | None = None

    @property
    def ran_a_kernel(self) -> bool:
        """Whether pass one actually computed anything.

        False means there is NOTHING to say about how the accumulation was
        done -- not that it was done badly. Conflating the two is what
        disabled a 102,400-row run over 70 degenerate rows.
        """
        return self.kind in (PASS_ONE_FUSED, PASS_ONE_CPU_WIDENED,
                             PASS_ONE_UNFUSED_CUBLAS)

    @property
    def accumulator_is_ours(self) -> bool:
        """Whether the float32 accumulation the bound assumes was under our
        control. Only meaningful when `ran_a_kernel`."""
        return self.kind in (PASS_ONE_FUSED, PASS_ONE_CPU_WIDENED)


# Pass-one outcome kinds.
PASS_ONE_FUSED = "fused"                    # our float32 accumulator
PASS_ONE_CPU_WIDENED = "cpu_widened"        # our float32, widened on CPU
PASS_ONE_UNFUSED_CUBLAS = "unfused_cublas"  # cuBLAS ran; accumulation not ours
PASS_ONE_GUARD_REFUSED = "guard_refused"    # no kernel ran; carries a reason
PASS_ONE_OOM = "oom"                        # no usable result; transient
PASS_ONE_EMPTY = "empty"                    # empty corpus; -inf is a real answer


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
