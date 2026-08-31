"""Deterministic tie-breaking for top-K selection.

`torch.topk` does not define the order of EXACTLY equal scores, so tied results
can change with batching, tiling, worker count, or device. To make selection
deterministic, each candidate is assigned one packed `int64` key:

    packed = score_order_key(score) * 2**32 + (0xFFFFFFFF - ordinal)

The high 32 bits contain a lossless, order-preserving encoding of the float32
score. The low 32 bits contain the inverted ordinal, so `topk`-largest selects
the smallest ordinal when scores tie. Because the score transform is bijective,
`unpack_score` recovers the original score bit-for-bit.

ORDINALS
--------
Each row owned by a worker gets a dense ordinal in `[0, n_rows)`:

  ordinal   position in corpus order.
  id        position in sorted-ID order, so the lowest ID wins ties.

Ordinals are worker-local. They only need to preserve the requested order among
that worker's rows. Cross-worker ties are resolved later by `merge` using the
full ID, or `hit_tie` for numeric IDs.

WHY RANK IDS
------------
IDs themselves may be too large to fit in the key. Instead, `id` tie-breaking
uses each ID's rank in sorted order. This preserves exact ordering regardless
of whether the ID is a UUID, integer, or string, while keeping the tie value
within 32 bits.

REPRODUCIBILITY
---------------
This only resolves scores that are bit-for-bit equal. Changing matmul batching
or tiling can change a score by an ULP and therefore change whether a tie
exists. Pin the batch size when bit-reproducible output is required.
"""

from __future__ import annotations

import logging

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

logger = logging.getLogger(__name__)

# The ordinal field is the low 32 bits of the packed key.
U32 = 0xFFFFFFFF
_TWO32 = 1 << 32
# Mask of everything below the sign bit — flipping it is what reverses the
# negative floats' bit order into value order (see `score_order_key`).
_LOW31 = 0x7FFFFFFF
_INT64_MAX = (1 << 63) - 1

# Largest ordinal value, i.e. the LOWEST priority. Used for the initial `-inf`
# sentinel state, which every real candidate must outrank. Held back from real
# rows, so a worker may hold at most `U32` of them — see `MAX_ROWS_PER_WORKER`.
TIE_WORST = U32
MAX_ROWS_PER_WORKER = U32


def score_order_key(scores):
    """Convert float32 scores to int64 keys with the same numeric ordering.

    The float32 bit pattern is transformed so integer comparison matches float
    comparison: positive values keep their bit order, while negative values
    have their low 31 bits flipped.

    `-0.0` is normalized to `+0.0` so equal zeros are ordered by the tiebreak,
    not by their sign bit. The transform is performed in int32 and widened to
    int64 only at the end.

    The mapping is lossless for non-NaN scores, so the original float32 value
    can be recovered exactly.
    """
    import torch

    b = (scores + 0.0).view(torch.int32)
    return (b ^ ((b >> 31) & _LOW31)).to(torch.int64)


def _order_key_to_bits(key):
    """Inverse of `score_order_key`'s bit transform, on int32 keys."""
    import torch

    return torch.where(key >= 0, key, key ^ _LOW31)


def unpack_score(packed):
    """Inverse of `pack`'s high half: int64 key -> the exact float32 score.

    Exact because `score_order_key` is a bijection on bit patterns, with the
    single deliberate exception of `-0.0`, which comes back as `+0.0`.
    """
    import torch

    key = torch.div(packed, _TWO32, rounding_mode="floor").to(torch.int32)
    return _order_key_to_bits(key).view(torch.float32)


def _pack_eager(scores, ordinal):
    """`pack`'s body, as plain tensor ops — see `pack` for what it computes."""
    key = score_order_key(scores)
    # In-place from here: `score_order_key` already returned a fresh tensor, and
    # the multiply/add would otherwise allocate two more full-size int64 copies
    # of what is often the largest tensor in the run.
    key *= _TWO32
    key += (U32 - ordinal)
    return key


# `torch.compile(_pack_eager)`, built on first use. `None` = not tried yet,
# `False` = unavailable or it raised on first use, so don't try again.
_compiled_pack = None
# Set once the compiled callable has actually returned. Until then a failure is
# "inductor doesn't work here" and degrades; after it, a failure is a real error
# and propagates.
_compiled_proven = False


def pack(scores, ordinal):
    """Pack each score and ordinal into one int64 key.

    Keys sort by higher score first, then lower ordinal for exact ties.
    `ordinal` broadcasts against `scores`, so a 1-D ordinal vector can be used
    for all queries in a slice.

    The score occupies the high 32 bits and the inverted ordinal occupies the
    low 32 bits. The construction stays within the int64 range.

    The elementwise operations are compiled into a single kernel when possible;
    the uncompiled path is used if compilation is unavailable.
    """
    import torch

    # The ordering transform reinterprets each float32 score as an int32.
    # Other dtypes can change the tensor shape and produce invalid packed keys,
    # so require float32 explicitly.
    if scores.dtype != torch.float32:
        raise TypeError(
            f"tie-break key needs float32 scores, got {scores.dtype}; the order "
            "transform reinterprets each score as an int32, so any other width "
            "silently reshapes the packed key. nova-bf upcasts every vector to "
            "float32 before scoring, so this means something upstream did not."
        )

    global _compiled_pack, _compiled_proven

    if _compiled_pack is None:
        try:
            _compiled_pack = torch.compile(_pack_eager, dynamic=True)
        except Exception as exc:  # no torch.compile on this build/backend
            logger.info("tie-break key: torch.compile unavailable (%s); using eager ops", exc)
            _compiled_pack = False

    if _compiled_pack is False:
        return _pack_eager(scores, ordinal)

    if _compiled_proven:
        # Past the one call that proves inductor works here, errors are real.
        return _compiled_pack(scores, ordinal)

    try:
        key = _compiled_pack(scores, ordinal)
    except Exception as exc:
        # Compilation can fail lazily, on first call rather than at wrap time.
        logger.warning(
            "tie-break key: torch.compile failed (%s); using eager ops for the "
            "rest of this run — results are unaffected, this is a ~3x slower "
            "path for building the packed key",
            exc,
        )
        _compiled_pack = False
        return _pack_eager(scores, ordinal)
    _compiled_proven = True
    return key


def sentinel_key(shape, device):
    """The packed key for `(-inf, worst ordinal)` — the initial running top-K
    state, which every real candidate must outrank."""
    import torch

    neg_inf = torch.full(shape, float("-inf"), dtype=torch.float32, device=device)
    return pack(neg_inf, torch.tensor(TIE_WORST, dtype=torch.int64, device=device))


# Materializing the packed key doubles a score matrix's footprint (int64 next
# to float32). Where the key is only an intermediate — the wide pre-top-K in
# `compute.process_slice` — rows are processed in chunks so that transient
# stays bounded regardless of query count. 2**28 slots = 2 GiB of int64.
PACK_TARGET_SLOTS = 1 << 28


def pack_topk(scores, ordinal, k):
    """`torch.topk` over `pack(scores, ordinal)`, returning `(keys, idx)`.

    Chunked over query rows so the transient packed key never exceeds
    `PACK_TARGET_SLOTS` elements — the per-row top-K is independent, so
    chunking is exact, not an approximation.

    On CUDA this hands off to `nova_bf.topk_triton`, which applies the same rule
    INSIDE the select instead of encoding it into a doubled-width key. 

    The body below stays the portable path, and runs whenever the kernel does
    not apply: no triton, non-CUDA or non-float32 tensors, a slice wider than
    the kernel's register budget, or an ordinal that is not the 1-D per-slice
    vector. Both paths select the SAME SET of winners; only the cost differs.
    """
    import torch

    from nova_bf import topk_triton

    if topk_triton.available(scores, ordinal, k):
        # Use the optimized Triton path when available; fall back permanently if
        # compilation or launch fails.
        try:
            return topk_triton.topk(scores, ordinal, k)
        except torch.cuda.OutOfMemoryError:
            # OOM does not indicate an unsupported kernel configuration, and
            # the portable path requires even more temporary memory. Preserve
            # the original error rather than permanently disabling Triton.
            raise
        except Exception as exc:
            topk_triton.disable(exc)

    n_rows, n_cols = scores.shape
    chunk = max(1, min(n_rows, PACK_TARGET_SLOTS // max(1, n_cols)))
    if chunk >= n_rows:
        return torch.topk(pack(scores, ordinal), k=k, dim=1, sorted=False)

    key_parts, idx_parts = [], []
    for r0 in range(0, n_rows, chunk):
        block = pack(scores[r0 : r0 + chunk], ordinal)
        kp, ip = torch.topk(block, k=k, dim=1, sorted=False)
        del block
        key_parts.append(kp)
        idx_parts.append(ip)
    return torch.cat(key_parts, dim=0), torch.cat(idx_parts, dim=0)


# --- id-order ordinals --------------------------------------------------------


def build_ordinals(id_arrays: list) -> list[np.ndarray]:
    """Build per-file uint32 ordinals for `tiebreak='id'`.

    IDs from all files owned by the worker are ranked together in ascending ID
    order. Equal IDs are ordered by their original corpus position, giving each
    row a deterministic total order.

    The returned list matches `id_arrays`, with one ordinal array per file.
    Null IDs are rejected because they cannot be ordered consistently across
    local selection and final merge.
    """
    arrays = [
        a.combine_chunks() if isinstance(a, pa.ChunkedArray) else a for a in id_arrays
    ]
    lengths = [len(a) for a in arrays]
    total = sum(lengths)
    if total == 0:
        return [np.zeros(0, dtype=np.uint32) for _ in arrays]
    if total > MAX_ROWS_PER_WORKER:
        raise RuntimeError(
            f"this worker's corpus slice is {total:,} rows, above the "
            f"{MAX_ROWS_PER_WORKER:,} that fit the 32-bit tie-break field. Ties "
            "would stop being deterministic. Split the work further with a "
            "larger `--num-jobs`."
        )

    combined = arrays[0] if len(arrays) == 1 else pa.concat_arrays(arrays)
    if combined.null_count:
        raise ValueError(
            f"params.tiebreak='id': the id column has {combined.null_count:,} null "
            "value(s) in this worker's files, which have no ordering position and "
            "no distinct identity in the output (every one reports the hit id "
            "'None'). Fill or drop those rows, or use params.tiebreak='ordinal', "
            "which orders by corpus position and needs no id column."
        )
    table = pa.table({
        "id": combined,
        "pos": pa.array(np.arange(total, dtype=np.int64)),
    })
    # Nulls land at the end by default, which is the placement we want and the
    # one `id_order_scalar` matches; passing `null_placement` explicitly is
    # deprecated as of pyarrow 25 in favour of per-sort-key placement, so the
    # default is relied on and pinned by a test instead.
    perm = np.asarray(
        pc.sort_indices(
            table, sort_keys=[("id", "ascending"), ("pos", "ascending")]
        )
    )
    ordinals = np.empty(total, dtype=np.uint32)
    # Invert the permutation: `perm[i]` is where the i-th smallest id lives, so
    # scattering `arange` through it answers the question we actually want —
    # what position in sorted order does the id at row j hold? One O(n) scatter,
    # no comparisons and no hash map.
    ordinals[perm] = np.arange(total, dtype=np.uint32)

    out, off = [], 0
    for n in lengths:
        out.append(ordinals[off : off + n])
        off += n
    return out


def id_order_array(values, unsigned: bool):
    """Vectorized `id_order_scalar`.

    Used on the n_q*k decode hot path to avoid per-hit Python calls. The scalar
    rule remains authoritative; tests pin this implementation to it.

    Nulls sort last. Unsigned IDs flip the high bit, mapping uint64 to int64
    while preserving unsigned order.
    """
    import numpy as np
    import pyarrow as pa

    # Fill nulls in Arrow: converting nullable uint64 through NumPy can route
    # through float64 and lose precision. The filled values are masked below.
    nulls = np.asarray(values.is_null()) if values.null_count else None
    filled = values.fill_null(0) if values.null_count else values
    if unsigned:
        u = np.asarray(filled.cast(pa.uint64()).to_numpy(zero_copy_only=False),
                       dtype=np.uint64)
        out = (u ^ np.uint64(1 << 63)).astype(np.int64)
    else:
        out = np.asarray(filled.cast(pa.int64()).to_numpy(zero_copy_only=False),
                         dtype=np.int64).copy()
    if nulls is not None:
        out[nulls] = np.int64(_INT64_MAX)
    return pa.array(out, type=pa.int64())


def id_order_scalar(v, unsigned: bool) -> int:
    """The int64 whose ascending order DEFINES `tiebreak='id'` for one id.

    `merge` reduces across workers and must apply the identical rule a worker
    applied within one, but a worker's ordinal is worker-local and meaningless
    outside it. For a STRING id column `merge` compares `hit_ids` directly; for
    a NUMERIC one it cannot, because `hit_ids` arrive already stringified —
    where `"10"` precedes `"9"` — so the numeric order has to travel alongside
    them as this ordinate rather than be re-derived from the text.

    A null sorts last, the same end `build_ordinals` puts it at. Unsigned
    values are shifted, not wrapped, so the whole `uint64` range maps onto
    `int64` in order.
    """
    if v is None:
        return _INT64_MAX
    v = int(v)
    return v - (1 << 63) if unsigned else v
