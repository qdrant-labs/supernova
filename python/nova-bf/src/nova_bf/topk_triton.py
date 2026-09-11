"""Triton top-K with an integrated deterministic tie-break.

The portable implementation encodes each `(score, tiebreak)` pair into an
int64 key before calling `torch.topk`. This kernel instead applies the
tie-break directly during selection, avoiding the expanded int64 matrix.

Each Triton program processes one query row:

1. Find the k-th largest score key (`cut`) using a 32-bit MSB-first descent.
2. Keep all scores above `cut`.
3. For scores tied at `cut`, select the remaining `need` entries with the
   smallest tiebreak ordinals.

The same selection supports both tie-break modes:

* `ordinal`: the ordinal is the column position.
* `id`: the ordinal is the rank induced by sorted IDs.

This produces the same deterministic top-K ordering as the portable packed-key
path while reducing its compute and memory overhead.

Scope: this kernel is used only for the pre-top-K stage, where one ordinal
vector is shared across each query row. It is not valid for `_merge_topk`,
which maintains per-cell ordinals and sentinel values. `available()` and the
wrapper assertions enforce this restriction.
"""


from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_UNAVAILABLE: str | None = None


def _load():
    """Import triton lazily; returns (triton, tl) or raises."""
    import triton
    import triton.language as tl

    return triton, tl


try:
    _triton, _tl = _load()

    @_triton.jit
    def _cutfill(S, NRM, RANK, ORD, ENC, OUTK, OUTI, THR, LIVE, stride_s, n_cols, k,
                 BLOCK: _tl.constexpr, RBITS: _tl.constexpr,
                 HAS_NRM: _tl.constexpr, HAS_THR: _tl.constexpr,
                 HAS_ENC: _tl.constexpr):
        row = _tl.program_id(0)
        offs = _tl.arange(0, BLOCK)
        m = offs < n_cols

        s = _tl.load(S + row * stride_s + offs, mask=m, other=float("-inf"))
        if HAS_NRM:
            # Fuse cosine's per-query norm divide into the score load, avoiding a
            # separate read/write pass over the score matrix.
            #
            # Divide before building the order key so ties reflect the final reported
            # score. Use div_rn to match torch division bit-for-bit across fallback paths.
            s = _tl.math.div_rn(s, _tl.load(NRM + row))
        # Fold -0.0 onto +0.0 before the bit transform, exactly as
        # `score_order_key` does. They are numerically EQUAL, so the ordinal
        # must decide between them -- but their bit patterns differ, and
        # untransformed they order +0.0 strictly above -0.0. Euclidean negates
        # its distance (`.sqrt_().neg_()`), so a self-hit really does produce
        # -0.0; omitting this made the kernel disagree with the portable path
        # on real scores, which the GPU-gated tests caught.
        s = s + 0.0
        b = s.to(_tl.int32, bitcast=True)
        key = b ^ ((b >> 31) & 0x7FFFFFFF)              # IEEE-754 total order
        u = key.to(_tl.uint32, bitcast=True) ^ 0x80000000   # -> unsigned order
        # Padding sinks to the minimum AND is masked out of every compare below:
        # u == 0 is also a REAL value (key INT32_MIN, i.e. a negative NaN), so an
        # unmasked compare could let a pad lane consume a winner slot.
        u = _tl.where(m, u, 0)

        if HAS_THR:
            # Prune rows whose best score cannot reach the running top-K threshold.
            # `>=` preserves exact ties and under-filled sentinel states; see
            # `tiebreak.live_rows`.
            thr_hi = (_tl.load(THR + row) >> 32).to(_tl.int32)
            thr_u = thr_hi.to(_tl.uint32, bitcast=True) ^ 0x80000000
            # "any lane >= thr_u" == "row max >= thr_u", using the same
            # masked-compare-and-sum idiom as the descents below. Pad lanes
            # sit at u == 0 and are excluded by `m` regardless.
            alive = _tl.sum((m & (u >= thr_u)).to(_tl.int32)) > 0
            _tl.store(LIVE + row, alive.to(_tl.uint8))
            if alive == 0:
                # Dead rows write nothing; their outputs are undefined. Downstream folds
                # gate on `live` and never read them; dead-row poisoning tests enforce this.
                return

        prefix = _tl.zeros([], dtype=_tl.uint32)
        for i in _tl.static_range(32):
            cand = prefix | _tl.full([], 1 << (31 - i), _tl.uint32)
            prefix = _tl.where(_tl.sum((u >= cand).to(_tl.int32)) >= k, cand, prefix)

        definite = m & (u > prefix)
        tied = m & (u == prefix)
        # need >= 1 and n_tied >= need, both from the maximality of `prefix`:
        # count(u >= prefix) >= k > count(u > prefix).
        need = k - _tl.sum(definite.to(_tl.int32))

        rk = _tl.load(RANK + offs, mask=m, other=0).to(_tl.uint32)
        # rank ascending -> w descending, shifted into [1, n_cols] so no TIED
        # lane can be 0. If one could, the descent might land on p2 == 0 and
        # `w >= 0` would keep EVERY tied lane, silently degrading to
        # first-k-by-position. Non-tied lanes sit at 0 and are excluded by the
        # `tied &` on every compare.
        w = _tl.where(tied, n_cols - rk, 0)
        p2 = _tl.zeros([], dtype=_tl.uint32)
        for i in _tl.static_range(RBITS):
            cand2 = p2 | _tl.full([], 1 << (RBITS - 1 - i), _tl.uint32)
            p2 = _tl.where(_tl.sum((tied & (w >= cand2)).to(_tl.int32)) >= need, cand2, p2)

        keep = definite | (tied & (w >= p2))
        # Compiled away unless TRITON_DEBUG is set, so this costs nothing in a
        # real run. It guards the one assumption the output buffers lean on:
        # exactly k slots are written, which is why they are `torch.empty`.
        _tl.device_assert(_tl.sum(keep.to(_tl.int32)) == k, "tie-break top-K kept != k")
        pos = _tl.cumsum(keep.to(_tl.int32)) - 1
        # Emit the PACKED KEY as well as the index. The caller stores keys, and
        # rebuilding them host-side meant gathering the scores back, re-running
        # the order transform, and widening to int64 over (n_q, k) -- 1.29 ms
        # against the kernel's own 1.73. Everything needed is already in
        # registers here, so it costs one extra store.
        #
        # `u` carries the sign flip that made compares cheap; undo it to recover
        # the int32 order key, then combine EXACTLY as `pack` does so the two
        # paths produce bit-identical keys.
        ordv = _tl.load(ORD + offs, mask=m, other=0)
        key32 = (u ^ 0x80000000).to(_tl.int32, bitcast=True)
        packed = key32.to(_tl.int64) * 4294967296 + (0xFFFFFFFF - ordv)
        _tl.store(OUTK + row * k + pos, packed, mask=keep & (pos < k))
        if HAS_ENC:
            # Emit encoded row IDs directly, avoiding the caller's index widening
            # and post-top-K gather.
            _tl.store(OUTI + row * k + pos, _tl.load(ENC + offs, mask=m, other=0),
                      mask=keep & (pos < k))
        else:
            _tl.store(OUTI + row * k + pos, offs.to(_tl.int32), mask=keep & (pos < k))


except Exception as exc:  # no triton, or a version whose API moved
    _triton = _tl = None
    _cutfill = None
    _UNAVAILABLE = f"{type(exc).__name__}: {exc}"


# Maximum measured kernel width. Each program holds a full row in registers, so
# wider blocks risk spilling and fall back to the portable path.
#
# A10G / Triton 3.8.0, num_warps=8: BLOCK=8192 has no spills but is near the
# register limit (254 regs without encoded IDs). Re-measure before increasing
# this limit or materially increasing kernel state.
MAX_BLOCK = 8192


def _warps_for(block: int) -> int:
    """Chosen number of warps to launch for a given BLOCK.
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
# How many times the kernel ACTUALLY launched, for the manifest.
_LAUNCHES = 0


def usage() -> dict:
    """`permitted` / `launches` / `unavailable` for the run manifest.

    `permitted and launches == 0` means it never ran; a nonzero `launches`
    together with an `unavailable` reason means it ran and then STOPPED, which
    is the case the old env-var-only report hid completely.
    """
    import os

    return {
        "permitted": not os.environ.get("NOVA_BF_NO_TOPK_KERNEL"),
        "launches": _LAUNCHES,
        "unavailable": _UNAVAILABLE,
    }


def reset_usage() -> None:
    global _LAUNCHES
    _LAUNCHES = 0


_INT32_MAX = (1 << 31) - 1


def _offsets_fit_int32(n_q: int, *strides: int) -> bool:
    """Can `(n_q - 1) * stride + col` be computed in int32 for every pointer?"""
    return n_q <= 0 or (n_q - 1) * max(strides) + MAX_BLOCK <= _INT32_MAX


def disable(exc: BaseException) -> None:
    """Turn the kernel off for the rest of the process after a launch failure.

    `available()` can only see what is inspectable up front; a JIT that fails
    when it actually runs is not. One warning, then every later call takes the
    portable path — which computes the identical answer, so this costs speed
    and nothing else.
    """
    global _cutfill, _UNAVAILABLE
    _cutfill = None
    _UNAVAILABLE = f"{type(exc).__name__}: {exc}"
    logger.warning(
        "tie-break top-K: the Triton kernel failed to launch (%s); using the "
        "portable path for the rest of this PROCESS",
        exc,
    )


def _shape_of(t) -> str:
    """Return `t`'s shape for logging without raising.
    """
    try:
        return str(tuple(t.shape))
    except Exception:
        return "?"


# Whether the first decline has been reported. `available()` is consulted once
# per slice per search — millions of times in a run — so this is logged exactly
# once; the interesting fact is THAT the run fell back, not how often.
_DECLINE_LOGGED = False


def available(scores, ordinal, k, scale=None, thr=None, enc=None) -> bool:
    """Is the kernel usable for THIS call? Anything false falls back.
    """
    ok = _available(scores, ordinal, k, scale, thr, enc)
    global _DECLINE_LOGGED
    if not ok and not _DECLINE_LOGGED:
        _DECLINE_LOGGED = True
        logger.info(
            "tie-break top-K: the Triton kernel does not apply to this run "
            "(scores %s, k=%s%s) — using the portable path, which computes the "
            "identical answer roughly 4x slower. Logged once.",
            _shape_of(scores), k, _why_declined(scores, ordinal, k, scale, thr, enc),
        )
    return ok


def _enc_ok(enc, n_cols: int, device) -> bool:
    """Whether `enc` satisfies the kernel's direct-indexing assumptions.

    The kernel treats `enc` as a contiguous int64 vector with one value per
    column on the same device. Kept separate so these guards are CPU-testable.
    """
    import torch

    return bool(
        enc.ndim == 1
        and enc.numel() == n_cols
        and enc.dtype is torch.int64
        and enc.is_contiguous()
        and enc.device == device
    )


def _why_declined(scores, ordinal, k, scale, thr, enc) -> str:
    """Return the first reason the Triton top-K path is unavailable. 
    
    Checks should stay in the same order as `_available` so the 
    logged reason matches the condition that actually declined the kernel. 
    """
    import os

    import torch

    if _cutfill is None:
        return f"; the kernel did not load ({_UNAVAILABLE})"
    if os.environ.get("NOVA_BF_NO_TOPK_KERNEL"):
        return "; NOVA_BF_NO_TOPK_KERNEL is set"
    if getattr(torch.version, "hip", None) is not None:
        return "; ROCm/HIP build"
    if scores.ndim != 2 or ordinal.ndim != 1:
        return f"; scores.ndim={scores.ndim}, ordinal.ndim={ordinal.ndim}, want 2 and 1"
    if not scores.is_cuda:
        return f"; scores are on {scores.device}, not CUDA"
    if scores.dtype is not torch.float32:
        return f"; scores are {scores.dtype}, not float32"
    if not scores.is_contiguous():
        return "; scores are not contiguous"
    n_cols = scores.shape[1]
    if enc is not None and not _enc_ok(enc, n_cols, scores.device):
        for cond, why in (
            (enc.ndim == 1, f"enc.ndim={enc.ndim}, want 1"),
            (enc.numel() == n_cols, f"enc has {enc.numel()} ids for {n_cols} columns"),
            (enc.dtype is torch.int64, f"enc is {enc.dtype}, not int64"),
            (enc.is_contiguous(), "enc is not contiguous"),
            (enc.device == scores.device,
             f"enc is on {enc.device}, scores on {scores.device}"),
        ):
            if not cond:
                return f"; {why}"
    if ordinal.numel() != n_cols:
        return f"; {ordinal.numel()} ordinals for {n_cols} columns"
    if not 0 < k <= n_cols:
        return f"; k={k} outside (0, n_cols={n_cols}]"
    if n_cols > MAX_BLOCK:
        return f"; n_cols={n_cols} exceeds MAX_BLOCK={MAX_BLOCK}"
    return ""


def _available(scores, ordinal, k, scale=None, thr=None, enc=None) -> bool:
    """`available`'s body — see there.

    This is the contract boundary: everything the kernel ASSUMES is checked
    here, because a wrong `True` is a silent correctness bug while a wrong
    `False` only costs speed. See `topk` for the invariants callers must hold
    that are too expensive to verify per call (uniqueness, ordinal range).
    """
    import os

    import torch

    # `NOVA_BF_NO_TOPK_KERNEL` mirrors the fold's `NOVA_BF_NO_FOLD_KERNEL`, so an
    # operator suspecting either kernel on their hardware can switch it off in
    # the field. Without this one, disabling BOTH needed a code change.
    if _cutfill is None or os.environ.get("NOVA_BF_NO_TOPK_KERNEL"):
        return False
    # `torch.cuda` also fronts ROCm, so `is_cuda` alone would let an AMD tensor
    # through. Triton may well compile for HIP, but `num_warps` and MAX_BLOCK
    # were tuned against NVIDIA warps and register files; a wavefront is 64
    # lanes, so those numbers mean something different there. Untested = off.
    if getattr(torch.version, "hip", None) is not None:
        return False
    if scores.ndim != 2 or ordinal.ndim != 1:
        return False
    if not scores.is_cuda or scores.dtype is not torch.float32:
        return False
    if not scores.is_contiguous():
        return False
    # The wrapper ranks the ordinal on `scores.device`; a tensor from another
    # device (or the host) would make that scatter cross devices.
    if ordinal.device != scores.device or ordinal.dtype is not torch.int64:
        return False
    n_q, n_cols = scores.shape
    if n_q <= 0:
        return False
    if scale is not None and not (
        scale.ndim == 1 and scale.numel() == n_q
        and scale.dtype is torch.float32 and scale.is_contiguous()
        and scale.device == scores.device
    ):
        return False
    if thr is not None and not (
        thr.ndim == 1 and thr.numel() == n_q
        and thr.dtype is torch.int64 and thr.is_contiguous()
        and thr.device == scores.device
    ):
        return False
    if not _offsets_fit_int32(n_q, scores.stride(0), k):
        return False
    # `enc` is indexed directly by score-column offset, so require one contiguous
    # int64 value per column on the same device.
    if enc is not None and not _enc_ok(enc, n_cols, scores.device):
        return False
    return ordinal.numel() == n_cols and 0 < k <= n_cols <= MAX_BLOCK




# Test-only values for dead rows. `POISON_KEY` is the largest int64 key, so any
# accidental read wins selection and fails loudly; `POISON_ID` is recognizable.
POISON_KEY = 0x7FFFFFFFFFFFFFFF
POISON_ID = -0x5EEDDEAD


def _poisoning() -> bool:
    """Whether to fill undefined dead-row outputs with deterministic poison.

    Used to verify that downstream paths gate on `live` and never consume them.
    """
    import os

    return bool(os.environ.get("NOVA_BF_POISON_DEAD_ROWS"))


def rank_of(ordinal):
    """Return each column's 0-based rank in ascending ordinal order.

    The rank depends only on `ordinal`, so callers may compute it once and reuse
    it across members sharing the same columns.
    """
    import torch

    n_cols = ordinal.numel()
    perm = torch.argsort(ordinal)
    rank = torch.empty(n_cols, dtype=torch.int32, device=ordinal.device)
    rank.scatter_(0, perm,
                  torch.arange(n_cols, dtype=torch.int32, device=ordinal.device))
    return rank


def topk(scores, ordinal, k, scale=None, thr=None, enc=None, rank=None):
    """Select top-K packed `(score, ordinal)` keys per query row.

    Returns `(keys, values, live)`. `values` contains `enc[column]` when
    provided, otherwise column indices. `scale` applies a per-query divisor and
    `thr` enables pruning.

    Dead rows have undefined outputs when pruning; callers must gate on `live`.
    `rank` may provide a precomputed `rank_of(ordinal)` for reuse.

    `rank` is `rank_of(ordinal)`; pass it to share one computation across the
    members of a slice, which all see the same ordinal vector.

    Requires unique ordinals in `[0, 0xFFFFFFFF]`.
    """
    import torch

    n_q, n_cols = scores.shape
    if rank is None:
        rank = rank_of(ordinal)
    elif not (rank.dtype is torch.int32 and rank.numel() == n_cols
              and rank.is_contiguous() and rank.device == scores.device):
        # Reject structurally invalid ranks rather than silently recomputing.
        raise ValueError(
            f"rank must be int32, contiguous, {n_cols} long, on "
            f"{scores.device}; got dtype={rank.dtype} "
            f"shape={tuple(rank.shape)} device={rank.device}"
        )

    _block = _triton.next_power_of_2(n_cols)
    outk = torch.empty((n_q, k), dtype=torch.int64, device=scores.device)
    outi = torch.empty((n_q, k),
                       dtype=torch.int64 if enc is not None else torch.int32,
                       device=scores.device)
    live = None if thr is None else torch.empty(n_q, dtype=torch.uint8, device=scores.device)
    if thr is not None and _poisoning():
        # Poison undefined dead-row outputs so accidental reads fail loudly.
        outk.fill_(POISON_KEY)
        outi.fill_(POISON_ID if enc is not None else 0x7FFFFFFF)

    # Launch on the tensors' CUDA device, not the process's current device.
    global _LAUNCHES
    _LAUNCHES += 1
    with torch.cuda.device(scores.device):
        _cutfill[(n_q,)](
            # `scores` doubles as the NRM placeholder when unscaled, as the
            # THR/LIVE placeholders when unpruned, and as ENC when the caller
            # wants column indices back.
            scores, scale if scale is not None else scores,
            rank, ordinal.contiguous(),
            enc if enc is not None else scores,
            outk, outi,
            thr if thr is not None else scores,
            live if live is not None else scores,
            scores.stride(0), n_cols, k,
            BLOCK=_block,
            RBITS=max(1, int(n_cols).bit_length()),   # w lands in [1, n_cols]
            HAS_NRM=scale is not None,
            HAS_THR=thr is not None,
            HAS_ENC=enc is not None,
            num_warps=_warps_for(_block),  # 4 spills at BLOCK=8192; 8 does not.
        )
    return outk, outi if enc is not None else outi.to(torch.int64), live
