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
    def _cutfill(S, RANK, ORD, OUTK, OUTI, stride_s, n_cols, k,
                 BLOCK: _tl.constexpr, RBITS: _tl.constexpr):
        row = _tl.program_id(0)
        offs = _tl.arange(0, BLOCK)
        m = offs < n_cols

        s = _tl.load(S + row * stride_s + offs, mask=m, other=float("-inf"))
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
        _tl.store(OUTI + row * k + pos, offs.to(_tl.int32), mask=keep & (pos < k))


except Exception as exc:  # no triton, or a version whose API moved
    _triton = _tl = None
    _cutfill = None
    _UNAVAILABLE = f"{type(exc).__name__}: {exc}"


# Widest slice the kernel will take. Each program holds the whole row in
# registers, so past some width the register file spills and the win evaporates.
# MEASURED on an A10G with `num_warps=8`: 4096 -> n_regs=104, 8192 -> n_regs=209,
# zero spills at both. (At 8192 with num_warps=4 it spills 8 bytes, which is why
# the launch below pins 8.) Anything wider is unmeasured and takes the portable
# path.
MAX_BLOCK = 8192


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
        "portable path for the rest of this run — results are unaffected, this "
        "is a ~4x slower select",
        exc,
    )


def available(scores, ordinal, k) -> bool:
    """Is the kernel usable for THIS call? Anything false falls back.

    This is the contract boundary: everything the kernel ASSUMES is checked
    here, because a wrong `True` is a silent correctness bug while a wrong
    `False` only costs speed. See `topk` for the invariants callers must hold
    that are too expensive to verify per call (uniqueness, ordinal range).
    """
    import torch

    if _cutfill is None:
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
    return ordinal.numel() == n_cols and 0 < k <= n_cols <= MAX_BLOCK


def topk(scores, ordinal, k):
    """(n_q, n_cols) float32 -> `(packed_keys, column_indices)`, both (n_q, k).

    Highest score wins; among bit-identical scores, the smallest ordinal wins.
    Unordered within a row (the caller re-selects). The keys are bit-identical
    to `tiebreak.pack(scores, ordinal).gather(1, idx)`.

    CALLER INVARIANTS — checked by tests, not per call, because verifying them
    here would cost a reduction and a host sync on every slice:

      ordinal values are UNIQUE within the slice. Descent 2 selects on the
        ordinal's RANK, so duplicates would be given an arbitrary order by the
        underlying (unstable) argsort while the portable path leaves them
        genuinely equal — the two would diverge. Both modes satisfy this by
        construction: `ordinal` is `base + row` and `id` is a permutation rank,
        and a subset of distinct values stays distinct under filtering.

      0 <= ordinal <= 0xFFFFFFFF. The packed key puts the ordinal in the low
        32 bits as `0xFFFFFFFF - ordv`; anything outside that range makes the
        low half negative or overflowing and corrupts the score half. Held by
        `tiebreak.MAX_ROWS_PER_WORKER`, which is exactly `0xFFFFFFFF`.
    """
    import torch

    n_q, n_cols = scores.shape
    perm = torch.argsort(ordinal)
    rank = torch.empty(n_cols, dtype=torch.int32, device=scores.device)
    rank.scatter_(0, perm, torch.arange(n_cols, dtype=torch.int32, device=scores.device))

    outk = torch.empty((n_q, k), dtype=torch.int64, device=scores.device)
    outi = torch.empty((n_q, k), dtype=torch.int32, device=scores.device)
    # Triton launches on the CURRENT device; pin it to the tensors' own so a
    # process whose current device differs (multi-GPU) cannot launch elsewhere.
    with torch.cuda.device(scores.device):
        _cutfill[(n_q,)](
            scores, rank, ordinal.contiguous(), outk, outi,
            scores.stride(0), n_cols, k,
            BLOCK=_triton.next_power_of_2(n_cols),
            RBITS=max(1, int(n_cols).bit_length()),   # w lands in [1, n_cols]
            num_warps=8,   # 4 spills at BLOCK=8192; 8 does not. See MAX_BLOCK.
        )
    return outk, outi.to(torch.int64)
