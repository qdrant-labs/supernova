"""Triton fold for `compute._merge_topk`.

Fuses state/part concatenation, top-K selection, and id gathering into one
in-place kernel over packed `(score, tiebreak)` keys.

With pruning enabled, dead rows are never read or written; their part buffers
may contain garbage. Live rows update the running state and `thr` in place.

Padding is excluded explicitly, and tied score cutoffs are resolved by the
packed tiebreak so results remain deterministic.
"""



from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_UNAVAILABLE: str | None = None

try:
    import triton as _triton
    import triton.language as _tl

    @_triton.jit
    def _fold(SK, SE, PK, PE, OK, OE, LIVE, THR,
              sk_s, se_s, pk_s, pe_s, ok_s, oe_s,
              k, w, BLOCK: _tl.constexpr, HAS_LIVE: _tl.constexpr):
        row = _tl.program_id(0)
        offs = _tl.arange(0, BLOCK)

        if HAS_LIVE:
            if _tl.load(LIVE + row) == 0:
                # Dead rows leave state and threshold unchanged and must not read the
                # part buffer, whose contents are undefined.
                return

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
            # Low halves can tie, so select exactly `need` lanes at
            # the cutoff instead of letting lane-order truncation choose arbitrarily.
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

        if HAS_LIVE:
            # Update the row's prune threshold from keys already in registers, avoiding
            # a separate reduction over the full top-k state.
            _tl.store(THR + row, _tl.min(
                _tl.where(keep, key, _tl.full([BLOCK], 0x7FFFFFFFFFFFFFFF, _tl.int64)),
                axis=0,
            ))

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
# See `topk_triton.usage` — the switch says PERMITTED, this says RAN.
_LAUNCHES = 0


def usage() -> dict:
    """`permitted` / `launches` / `unavailable` for the run manifest."""
    import os

    return {
        "permitted": not os.environ.get("NOVA_BF_NO_FOLD_KERNEL"),
        "launches": _LAUNCHES,
        "unavailable": _UNAVAILABLE,
    }


# Triton errors that occur before kernel enqueue. Match MRO class names rather
# than importing Triton types so this module works across Triton versions and
# when Triton is unavailable.
_PRE_ENQUEUE_ERRORS = frozenset({
    "CompilationError",
    "CompileTimeAssertionFailure",
    "OutOfResources",
})


def is_pre_enqueue(exc: BaseException) -> bool:
    """Whether `exc` is classified as a pre-enqueue Triton failure.

    These failures can safely fall back to the portable fold because the
    in-place state has not been written. Matching by MRO name avoids depending
    on Triton's version-specific exception modules.
    """
    return any(c.__name__ in _PRE_ENQUEUE_ERRORS for c in type(exc).__mro__)


def reset_usage() -> None:
    global _LAUNCHES
    _LAUNCHES = 0


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


def available(state_key, state_enc, part_key, part_enc, k, live=None, thr=None) -> bool:
    """Is the kernel usable for THIS fold? Anything false falls back.
    """
    ok = _available(state_key, state_enc, part_key, part_enc, k, live, thr)
    global _DECLINE_LOGGED
    if not ok and not _DECLINE_LOGGED:
        _DECLINE_LOGGED = True
        logger.info(
            "tie-break fold: the Triton kernel does not apply to this run "
            "(%s; state %s, part %s, k=%s) — using the portable path, which "
            "computes the identical answer more slowly. Logged once.",
            _why_declined(state_key, state_enc, part_key, part_enc, k,
                          live, thr),
            _shape_of(state_key), _shape_of(part_key), k,
        )
    return ok


def _why_declined(state_key, state_enc, part_key, part_enc, k,
                  live=None, thr=None) -> str:
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
        # Pruning inputs, checked in the same order as `_available` so the
        # reason matches the branch that actually declined. Without these the
        # function falls through to the catch-all below and blames shapes.
        if (live is None) != (thr is None):
            missing = "thr" if live is not None else "live"
            return (f"pruning inputs must arrive as a pair; {missing} is None "
                    "while the other is not")
        if live is not None:
            n_q = state_key.shape[0] if state_key.ndim == 2 else -1
            for name, t, dt in (("live", live, torch.uint8),
                                ("thr", thr, torch.int64)):
                if t.ndim != 1:
                    return f"{name} is {t.ndim}-D, not 1-D"
                if t.numel() != n_q:
                    return (f"{name} has {t.numel()} entries, not one per query "
                            f"row (n_q = {n_q})")
                if t.dtype is not dt:
                    return f"{name} is {t.dtype}, not {dt}"
                if not t.is_contiguous():
                    return f"{name} is not contiguous (strides {t.stride()})"
                if t.device != state_key.device:
                    return (f"{name} is on {t.device}, not state_key's "
                            f"{state_key.device}")
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


def _available(state_key, state_enc, part_key, part_enc, k, live=None, thr=None) -> bool:
    """`available`'s body — see there."""
    import os

    import torch

    if _fold is None or os.environ.get("NOVA_BF_NO_FOLD_KERNEL"):
        return False
    # Pruning inputs travel as a pair: `live` decides the skip, `thr` receives
    # the by-product min. 
    if (live is None) != (thr is None):
        return False
    if live is not None:
        n_q = state_key.shape[0] if state_key.ndim == 2 else -1
        if not (
            live.ndim == 1 and live.numel() == n_q and live.dtype is torch.uint8
            and live.is_contiguous() and live.device == state_key.device
        ):
            return False
        if not (
            thr.ndim == 1 and thr.numel() == n_q and thr.dtype is torch.int64
            and thr.is_contiguous() and thr.device == state_key.device
        ):
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


def fold(state_key, state_enc, part_key, part_enc, k, live=None, thr=None,
         _out=None):
    """Fold `part` into the packed-key top-k state.

    Preconditions are enforced by `available(...)`; this hot path performs no
    validation.

    With `live`/`thr`, dead rows are untouched and their part rows are never
    read. Live rows update both the state and `thr` in place.

    Production folds in place and returns the updated state tensors. Results are
    unordered within each row; callers sort only when decoding.
    """
    import torch

    n_q = state_key.shape[0]
    w = part_key.shape[1]

    # Test-only seam for comparing aliased and unaliased execution.
    out_k, out_e = (state_key, state_enc) if _out is None else _out

    # A 1-D part-id vector is shared across query rows via stride-0 broadcast.
    pe = part_enc if part_enc.ndim == 2 else part_enc.unsqueeze(0)
    pe_s = pe.stride(0) if part_enc.ndim == 2 else 0
    block = _triton.next_power_of_2(k + w)
    global _LAUNCHES
    _LAUNCHES += 1
    with torch.cuda.device(state_key.device):
        _fold[(n_q,)](
            state_key, state_enc, part_key, pe, out_k, out_e,
            # state_key doubles as the LIVE/THR placeholder when unpruned
            live if live is not None else state_key,
            thr if thr is not None else state_key,
            state_key.stride(0), state_enc.stride(0), part_key.stride(0), pe_s,
            out_k.stride(0), out_e.stride(0),
            k, w,
            BLOCK=block,
            HAS_LIVE=live is not None,
            num_warps=_warps_for(block),
        )
    return out_k, out_e
