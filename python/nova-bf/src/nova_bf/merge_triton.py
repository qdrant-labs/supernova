"""Triton fold for the running top-K state (`compute._merge_topk`).

Fuses concatenation, top-K selection, and gathering into one kernel: each query
row selects directly from the existing state and pending part and writes the
surviving `(key, id)` pairs without materializing an intermediate concatenation.

Packed keys are ordered lexicographically by their high and low 32-bit halves.
Selection therefore uses a 32-bit descent on the score half and, only when the
cut falls within a score tie, a second descent on the tie-break half.

Padding lanes are excluded explicitly. Duplicate sentinel keys require an exact
cumulative fill at the cutoff so a real candidate cannot be displaced by an
under-filled state's sentinels.

An early-out for converged rows is intentionally omitted: the kernel remains
memory-bound because the pending part must still be read, and the current
out-of-place design must still write the result.
"""


from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_UNAVAILABLE: str | None = None

try:
    import triton as _triton
    import triton.language as _tl

    @_triton.jit
    def _fold(SK, SE, PK, PE, OK, OE,
              sk_s, se_s, pk_s, pe_s, ok_s, oe_s,
              k, w, BLOCK: _tl.constexpr):
        row = _tl.program_id(0)
        offs = _tl.arange(0, BLOCK)
        n = k + w
        m = offs < n
        from_state = offs < k
        src = _tl.where(from_state, offs, offs - k)

        # One effective load per lane: the masked-off side costs no traffic.
        key = _tl.where(
            from_state,
            _tl.load(SK + row * sk_s + src, mask=m & from_state, other=0),
            _tl.load(PK + row * pk_s + src, mask=m & (offs >= k), other=0),
        )

        # int64 order == lexicographic (high int32, low uint32)
        hi = (key >> 32).to(_tl.int32)
        u = hi.to(_tl.uint32, bitcast=True) ^ 0x80000000
        lo = (key & 0xFFFFFFFF).to(_tl.uint32)

        prefix = _tl.zeros([], dtype=_tl.uint32)
        for i in _tl.static_range(32):
            cand = prefix | _tl.full([], 1 << (31 - i), _tl.uint32)
            prefix = _tl.where(_tl.sum((m & (u >= cand)).to(_tl.int32)) >= k, cand, prefix)

        definite = m & (u > prefix)
        tied = m & (u == prefix)
        need = k - _tl.sum(definite.to(_tl.int32))
        n_tied = _tl.sum(tied.to(_tl.int32))

        if n_tied > need:
            p2 = _tl.zeros([], dtype=_tl.uint32)
            for i in _tl.static_range(32):
                c2 = p2 | _tl.full([], 1 << (31 - i), _tl.uint32)
                p2 = _tl.where(_tl.sum((tied & (lo >= c2)).to(_tl.int32)) >= need, c2, p2)
            # `lo >= p2` alone can select MORE than `need`, because low halves
            # are not distinct here: every sentinel is `pack(-inf, TIE_WORST)`,
            # whose low half is 0. When the cut lands among sentinels the
            # descent bottoms out at p2 == 0 and `lo >= 0` takes all of them.
            # The `pos < k` store mask would then truncate in LANE order — and
            # the state occupies the low lanes, so what gets truncated is always
            # the PART. A real new candidate would be silently displaced by
            # sentinels it outranks.
            #
            # So fill the last places explicitly: everything strictly above the
            # cut, then the earliest lanes AT the cut, exactly as the pre-top-K
            # kernel fills its tied slots. (That kernel needs no such step —
            # its ranks are a permutation, hence distinct.)
            strictly = tied & (lo > p2)
            at_cut = tied & (lo == p2)
            room = need - _tl.sum(strictly.to(_tl.int32))
            keep = definite | strictly | (at_cut & (_tl.cumsum(at_cut.to(_tl.int32)) <= room))
        else:
            keep = definite | tied

        enc = _tl.where(
            from_state,
            _tl.load(SE + row * se_s + src, mask=m & from_state, other=0),
            _tl.load(PE + row * pe_s + src, mask=m & (offs >= k), other=0),
        )
        pos = _tl.cumsum(keep.to(_tl.int32)) - 1
        _tl.store(OK + row * ok_s + pos, key, mask=keep & (pos < k))
        _tl.store(OE + row * oe_s + pos, enc, mask=keep & (pos < k))

except Exception as exc:
    _triton = _tl = None
    _fold = None
    _UNAVAILABLE = f"{type(exc).__name__}: {exc}"


MAX_BLOCK = 8192


def _warps_for(block: int) -> int:
    """Warps to launch for a given BLOCK.
    """
    return max(1, min(8, block // 128))



# Kernel offsets are computed as `base + row * row_stride + col`. Both `row`
# (`tl.program_id`) and sufficiently small strides are int32, so the product
# can overflow once the largest row offset exceeds 2**31 - 1, silently
# addressing the wrong row.
#
# Promoting `row` to int64 avoids the overflow but significantly increases
# register pressure and reduces performance on production shapes. Instead,
# reject shapes whose offsets cannot be represented safely in int32 and let
# them use the portable path, which has no such limitation.
_INT32_MAX = (1 << 31) - 1


def _offsets_fit_int32(n_q: int, *strides: int) -> bool:
    """Can `(n_q - 1) * stride + col` be computed in int32 for every pointer?"""
    return n_q <= 0 or (n_q - 1) * max(strides) + MAX_BLOCK <= _INT32_MAX


def disable(exc: BaseException) -> None:
    """Turn the fold kernel off after a launch failure.

    Scope: this exists for the failures that are DETECTABLE AND RECOVERABLE —
    a JIT that compiles nowhere on this box, a Triton/driver mismatch, an API
    that moved. For those the portable path computes the identical answer and
    the cost is speed only.

    It is NOT a general recovery mechanism for an arbitrary kernel fault. An
    asynchronous CUDA execution error surfaces at some later synchronization
    point, may be attributed to whatever operation happened to be running then,
    and can leave the context unusable — in which case falling back is not
    possible and the rank dies regardless. Do not read the reassurance below as
    covering that case.
    """
    global _fold, _UNAVAILABLE
    _fold = None
    _UNAVAILABLE = f"{type(exc).__name__}: {exc}"
    logger.warning(
        "tie-break fold: the Triton kernel failed to launch (%s); using the "
        "portable path for the rest of this run — results are unaffected", exc,
    )


def enabled(sample) -> bool:
    """Could this kernel serve this run AT ALL? A cheap precheck, before inputs
    are prepared for it.
    """
    import os

    import torch

    if _fold is None or os.environ.get("NOVA_BF_NO_FOLD_KERNEL"):
        return False
    if getattr(torch.version, "hip", None) is not None:
        return False
    return sample.is_cuda


def _shape_of(t) -> str:
    """Return `t`'s shape for a log line, never raising.
    """
    try:
        return str(tuple(t.shape))
    except Exception:
        return "?"


# Whether the first decline has been reported — see `available`.
_DECLINE_LOGGED = False


def available(state_key, state_enc, part_key, part_enc, k) -> bool:
    """Is the kernel usable for THIS fold? Anything false falls back.
    """
    ok = _available(state_key, state_enc, part_key, part_enc, k)
    global _DECLINE_LOGGED
    if not ok and not _DECLINE_LOGGED:
        _DECLINE_LOGGED = True
        logger.info(
            "tie-break fold: the Triton kernel does not apply to this run "
            "(%s; state %s, part %s, k=%s) — using the portable path, which "
            "computes the identical answer more slowly. Logged once.",
            _why_declined(state_key, state_enc, part_key, part_enc, k),
            _shape_of(state_key), _shape_of(part_key), k,
        )
    return ok


def _why_declined(state_key, state_enc, part_key, part_enc, k) -> str:
    """A short reason for the log. The shapes alone are not enough to act on:
    a transposed tensor and an oversized block both look perfectly ordinary
    printed, and those are the two most likely causes.

    Best-effort and never raising — this runs only to explain a fallback that
    has already been decided, so it must not become a second failure.
    """
    import os

    try:
        import torch

        if _fold is None:
            return "kernel unavailable or disabled earlier in this run"
        if os.environ.get("NOVA_BF_NO_FOLD_KERNEL"):
            return "NOVA_BF_NO_FOLD_KERNEL is set"
        if getattr(torch.version, "hip", None) is not None:
            return "ROCm build, which these kernels are untuned for"
        ts = {"state_key": state_key, "state_enc": state_enc,
              "part_key": part_key, "part_enc": part_enc}
        for name, t in ts.items():
            if not t.is_cuda:
                return f"{name} is not on CUDA"
            if t.dtype is not torch.int64:
                return f"{name} is {t.dtype}, not int64"
            if not t.is_contiguous():
                return (f"{name} is not contiguous (strides {t.stride()}) — sparse "
                        "score matrices arrive transposed")
        w = part_key.shape[1] if part_key.ndim == 2 else 0
        if k + w > MAX_BLOCK:
            return f"k+w = {k + w} exceeds MAX_BLOCK = {MAX_BLOCK}"
        if not _offsets_fit_int32(state_key.shape[0], *(t.stride(0) for t in ts.values()), k):
            return (f"n_q = {state_key.shape[0]} makes row offsets overflow int32; "
                    "the portable path has no such limit")
        return "shape or device mismatch"
    except Exception:
        return "reason unavailable"


def _available(state_key, state_enc, part_key, part_enc, k) -> bool:
    """`available`'s body — see there."""
    import os

    import torch

    if _fold is None or os.environ.get("NOVA_BF_NO_FOLD_KERNEL"):
        return False
    if getattr(torch.version, "hip", None) is not None:
        return False
    for t in (state_key, state_enc, part_key):
        if not t.is_cuda or t.dtype is not torch.int64 or t.ndim != 2:
            return False
    if not part_enc.is_cuda or part_enc.dtype is not torch.int64:
        return False
    if part_enc.ndim not in (1, 2):
        return False
    # The kernel assumes contiguous rows (`col_stride == 1`); reject transposed or
    # otherwise strided inputs, which would silently produce incorrect reads.
    # Full contiguity, deliberately, though the pointer math only needs a unit
    # COLUMN stride (the row stride is passed explicitly). Relaxing this to
    # `stride(1) == 1` would admit an EXPANDED tensor — `t.expand(n_q, k)` has
    # strides (0, 1) and passes that check — whose row stride of zero makes
    # every query row read row 0's data. That is silently wrong ground truth
    # for every query but the first, which is exactly the failure this gate
    # exists to prevent; the shapes involved look entirely ordinary. The cost of
    # being conservative is a fallback that computes the same answer.
    for t in (state_key, state_enc, part_key, part_enc):
        if not t.is_contiguous():
            return False
    n_q = state_key.shape[0]
    if n_q <= 0 or state_key.shape[1] != k or state_enc.shape != state_key.shape:
        return False
    if part_key.shape[0] != n_q:
        return False
    w = part_key.shape[1]
    if part_enc.ndim == 2 and part_enc.shape != part_key.shape:
        return False
    if part_enc.ndim == 1 and part_enc.numel() != w:
        return False
    if any(t.device != state_key.device for t in (state_enc, part_key, part_enc)):
        return False
    if not _offsets_fit_int32(
        n_q, state_key.stride(0), state_enc.stride(0), part_key.stride(0),
        part_enc.stride(0) if part_enc.ndim == 2 else 1, k,
    ):
        return False
    return 0 < w and k + w <= MAX_BLOCK


def fold(state_key, state_enc, part_key, part_enc, k):
    """Select the top-k of `state ++ part` on the packed key.

    PRECONDITION: `available(...)` must have returned True for these exact
    arguments. This function validates NOTHING — not dtype, not layout, not
    device, not `k + w <= MAX_BLOCK`, not the int32 offset bound. It is the hot
    path (once per flush, per search, for the length of a rank) and every one of
    those checks lives in the gate instead. Calling it directly bypasses guards
    whose failure mode is silently wrong ground truth rather than an exception —
    a transposed `part_key` reads across query rows and still returns k
    plausible hits per row.

    Returns `(new_key, new_enc)`, each `(n_q, k)`. Unordered within a row — the
    caller either folds again or sorts once at decode.
    """
    import torch

    n_q = state_key.shape[0]
    w = part_key.shape[1]
    out_k = torch.empty((n_q, k), dtype=torch.int64, device=state_key.device)
    out_e = torch.empty((n_q, k), dtype=torch.int64, device=state_key.device)
    # A 1-D part id vector is shared by every query row; stride 0 broadcasts it
    # in place instead of materializing n_q copies the way `expand`+`cat` does.
    pe = part_enc if part_enc.ndim == 2 else part_enc.unsqueeze(0)
    pe_s = pe.stride(0) if part_enc.ndim == 2 else 0
    block = _triton.next_power_of_2(k + w)
    with torch.cuda.device(state_key.device):
        _fold[(n_q,)](
            state_key, state_enc, part_key, pe, out_k, out_e,
            state_key.stride(0), state_enc.stride(0), part_key.stride(0), pe_s,
            out_k.stride(0), out_e.stride(0),
            k, w,
            BLOCK=block,
            num_warps=_warps_for(block),
        )
    return out_k, out_e
