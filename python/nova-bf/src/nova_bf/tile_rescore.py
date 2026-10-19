"""Tile-level pass two: exact FP32 rescoring of live (query, tile) pairs.

Unlike slice mode, which rescores entire slices for each live query, tiles
mode uses pass one's per-tile bounds to rescore only live query/tile pairs.
The result is an (n_live, n_rows) FP32 matrix, with pruned pairs set to -inf,
matching the downstream convention for excluded candidates.

Scores use the same arithmetic as `_scores`: FP32 dot products followed
by corpus and optional query norm division. The Triton kernel uses IEEE
FP32 arithmetic (`input_precision="ieee"` and `div_rn`), but a different
summation order from cuBLAS. Scores may therefore differ slightly, potentially
changing near-tie rankings without affecting pruning correctness.

Tile bounds use the same conservative error margin and guards as slice
bounds. The kernel reuses the original corpus and query norms, preserving
the assumptions of the two-pass error bound.

Two implementations are provided:
  * `_tile_rescore`: Triton kernel using a device-built CSR work list of
    live query/tile pairs, avoiding unnecessary scoring and host sync.
  * `reference`: PyTorch GEMM with dead pairs masked to -inf, used on CPU.

The Triton kernel validates its first launch per compiled configuration
against the reference using bounded, deterministic coverage of scores and
the full live/dead mask. A mismatch disables the kernel for the run.
Failures propagate to the caller, which falls back to slice-mode scoring.
"""

import logging
import os

logger = logging.getLogger(__name__)

BQ = 32                # live queries per program block
BK = 32                # dimension chunk
MAX_TILES_P2 = 4096    # register limit for per-tile indices
MAX_TB = 256           # tile width limit for GPU resource usage

# Limit full-row self-checks to bound validation cost.
# Additional tile-level coverage comes from `_self_check_pairs`.
SELF_CHECK_ROWS = 256

_OFF: str | None = None
_LAUNCHES = 0
_CHECKED: set = set()          # `_compile_key`s self-checked
_SELFCHECKS = 0
_SELFCHECK_FAILURES = 0

try:
    import triton
    import triton.language as tl

    # Require correctly rounded FP32 division; plain `/` is approximate.
    tl.math.div_rn

    @triton.jit
    def _tile_rescore(Q, C, CN, QN, QLIST, TOFF, TCNT, BCUM, NBLK, OUT, POS,
                      n_rows, d, stride_q, stride_c, stride_out, n_tiles,
                      NT_P2: tl.constexpr,
                      TB: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr,
                      HAS_CN: tl.constexpr, HAS_QN: tl.constexpr,
                      NPROG: tl.constexpr):
        pid = tl.program_id(0)
        nblk = tl.load(NBLK)
        tiles = tl.arange(0, NT_P2)

        # Find each work item's tile using cumulative block counts.
        bc_end = tl.load(BCUM + 1 + tiles, mask=tiles < n_tiles, other=2 ** 30)
        for w in range(pid, nblk, NPROG):
            t = tl.sum((bc_end <= w).to(tl.int32))
            b = w - tl.load(BCUM + t)
            cnt = tl.load(TCNT + t)
            off = tl.load(TOFF + t)
            qi = b * BQ + tl.arange(0, BQ)
            qv = qi < cnt
            qidx = tl.load(QLIST + off + qi, mask=qv, other=0)
            rows = t * TB + tl.arange(0, TB)
            rv = rows < n_rows
            acc = tl.zeros((BQ, TB), dtype=tl.float32)

            for k0 in range(0, d, BK):
                ks = k0 + tl.arange(0, BK)
                kv = ks < d
                a = tl.load(Q + qidx[:, None] * stride_q + ks[None, :],
                            mask=qv[:, None] & kv[None, :], other=0.0)
                c = tl.load(C + rows[:, None] * stride_c + ks[None, :],
                            mask=rv[:, None] & kv[None, :], other=0.0)
                acc = tl.dot(a, tl.trans(c), acc, input_precision="ieee")

            s = acc
            if HAS_CN:
                cn = tl.load(CN + rows, mask=rv, other=1.0)
                s = tl.math.div_rn(s, tl.broadcast_to(cn[None, :], (BQ, TB)))
            if HAS_QN:
                qn = tl.load(QN + qidx, mask=qv, other=1.0)
                s = tl.math.div_rn(s, tl.broadcast_to(qn[:, None], (BQ, TB)))

            p = tl.load(POS + qidx, mask=qv, other=0)
            tl.store(OUT + p[:, None].to(tl.int64) * stride_out + rows[None, :],
                     s, mask=qv[:, None] & rv[None, :])

    _UNAVAILABLE = None
except Exception as exc:  # Triton unavailable or incompatible.
    triton = tl = None
    _tile_rescore = None
    _UNAVAILABLE = f"{type(exc).__name__}: {exc}"


class KernelMismatch(RuntimeError):
    """The kernel's first launch disagreed with the reference."""


def reset() -> None:
    """Per-run state: a kernel disabled by one run is retried by the next,
    and re-checked."""
    global _OFF, _LAUNCHES, _SELFCHECKS, _SELFCHECK_FAILURES
    _OFF = None
    _LAUNCHES = 0
    _SELFCHECKS = 0
    _SELFCHECK_FAILURES = 0
    _CHECKED.clear()
    _QIDS.clear()


def kernel_off() -> str | None:
    """Why the Triton kernel is not in use for the run, or `None`."""
    if os.environ.get("NOVA_BF_NO_TILE_KERNEL"):
        return "NOVA_BF_NO_TILE_KERNEL"
    if _tile_rescore is None:
        return f"triton unavailable ({_UNAVAILABLE})"
    return _OFF


def kernel_disable(reason: str) -> None:
    global _OFF
    if _OFF is None:
        _OFF = reason
        logger.warning(
            "two-pass: the tile re-score kernel is off for the rest of this "
            "run: %s. Tiles-mode slices take the slice-mode pass two (CPU: "
            "the torch reference) from here on.", reason)


def usage() -> dict:
    return {"launches": _LAUNCHES, "unavailable": kernel_off(),
            "selfchecks": _SELFCHECKS,
            "selfcheck_failures": _SELFCHECK_FAILURES}


def build_worklist(tile_live, bq: int = BQ):
    """Build a device-side CSR work list without host synchronization.

    Returns live query IDs grouped by tile (`qlist`), per-tile offsets
    and counts (`toff`, `tcnt`), cumulative block counts (`bcum`), and
    total blocks (`nblk`). Query IDs are sorted within each tile.
    """
    import torch

    n_tiles, nq = tile_live.shape
    dev = tile_live.device
    flat = tile_live.reshape(-1)
    tcnt = tile_live.sum(dim=1, dtype=torch.int32)
    toff = (torch.cumsum(tcnt, 0, dtype=torch.int32) - tcnt).to(torch.int32)

    # Map live pairs to consecutive slots; dead pairs use a shared sink.
    sink = n_tiles * nq
    target = torch.where(flat, torch.cumsum(flat, 0, dtype=torch.int64) - 1,
                         sink)
    qids = _query_ids(nq, n_tiles, dev)
    qlist = torch.zeros(sink + 1, device=dev, dtype=torch.int32)
    qlist.scatter_(0, target, qids)

    blocks = (tcnt + (bq - 1)) // bq
    bcum = torch.zeros(n_tiles + 1, device=dev, dtype=torch.int32)
    bcum[1:] = torch.cumsum(blocks, 0, dtype=torch.int32)

    # Clone to preserve pointer alignment across tile counts.
    return qlist, toff, tcnt, bcum, bcum[-1:].clone()


# Cache query IDs per shape to avoid rebuilding them for each slice.
# Read-only and used on a single stream; cleared on reset/release.
_QIDS: dict = {}


def _query_ids(nq: int, n_tiles: int, dev):
    import torch

    key = (int(nq), int(n_tiles), str(dev))
    got = _QIDS.get(key)
    if got is None:
        if len(_QIDS) >= 4:
            _QIDS.clear()
        got = _QIDS[key] = torch.arange(
            nq, device=dev, dtype=torch.int32).repeat(n_tiles)
    return got


def decline_reason(Q, C, tile_live, width, cn=None, qn=None) -> str | None:
    """Why the kernel would not run on these operands, or `None` if it would.
    Everything the kernel assumes, checked rather than trusted."""
    import torch

    off = kernel_off()
    if off is not None:
        return off
    if not (Q.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"):
        return "not a CUDA device"
    if Q.dtype is not torch.float32 or C.dtype is not torch.float32:
        return f"operands are {Q.dtype}/{C.dtype}, not float32"
    if not (Q.is_contiguous() and C.is_contiguous()):
        return "non-contiguous operands"
    if Q.device != C.device or tile_live.device != Q.device:
        return "operands on different devices"
    if width < 16 or (width & (width - 1)) or width > MAX_TB:
        return f"tile width {width} is not a power of two in [16, {MAX_TB}]"
    n_tiles = int(tile_live.shape[0])
    if triton.next_power_of_2(max(1, n_tiles)) > MAX_TILES_P2:
        return (f"{n_tiles} tiles in the slice (more than {MAX_TILES_P2}); "
                f"lower dense_batch_size")
    # The kernel's row offsets are int32: (rows - 1) * stride must fit.
    if Q.shape[0] * Q.stride(0) >= 2 ** 31 or C.shape[0] * C.stride(0) >= 2 ** 31:
        return "an operand exceeds int32 offsets"
    if n_tiles * Q.shape[0] >= 2 ** 31:
        return "the tile mask exceeds int32 offsets"
    for v in (cn, qn):
        if v is not None and (v.dtype is not torch.float32 or v.ndim != 1
                              or not v.is_contiguous() or v.device != Q.device):
            return "a norm vector is not a contiguous float32 vector"
    return None


def _kernel_ok(Q, C, tile_live, width, cn, qn) -> bool:
    return decline_reason(Q, C, tile_live, width, cn, qn) is None


def _nprog(dev) -> int:
    import torch

    if dev.type != "cuda":
        return 4
    return torch.cuda.get_device_properties(dev).multi_processor_count * 4


def reference(Q, C, idx, tile_live, width: int, cn=None, qn=None):
    """The torch implementation of `rescore`'s contract (see the module doc).

    The slice-mode arithmetic -- `_scores`' GEMM and divides, in its order --
    over the live queries, with every dead pair then masked to `-inf`.
    """
    import torch

    s = Q.index_select(0, idx) @ C.T
    if cn is not None:
        s.div_(cn[None, :])
    if qn is not None:
        s.div_(qn.index_select(0, idx)[:, None])
    n_rows = int(C.shape[0])
    keep = tile_live.index_select(1, idx).T.repeat_interleave(width, dim=1)
    s.masked_fill_(~keep[:, :n_rows], float("-inf"))
    return s


def _tolerance(d: int, mag, score=None):
    """Bound the difference between two FP32 scoring implementations.

    Each dot product has error bounded by gamma_d * M, where M is the
    normalized sum of absolute products. The tolerance accounts for
    both accumulation errors and up to two rounded divisions.

    The (1 + 4u) factor covers division errors on the accumulated
    error, while 8u * |score| covers rounding of the reference score.
    The FP64 magnitude is inflated to account for its own rounding.
    """
    u = 2.0 ** -24
    gamma = d * u / (1.0 - d * u)
    mag = mag * (1.0 + (d + 4) * 2.0 ** -52)
    tol = 2.0 * gamma * mag * (1.0 + 4.0 * u) + 1e-37
    if score is not None:
        tol = tol + 8.0 * u * score.abs()
    return tol


def score_tolerance(Q, C, idx, cn=None, qn=None, score=None):
    """Compute per-score FP64 error bounds between the kernel and reference.

    Accounts for differing FP32 accumulation orders and normalization
    rounding. Returns (len(idx), n_rows); see `_tolerance` for the bound.
    """
    mag = Q.index_select(0, idx).double().abs() @ C.double().abs().T
    if cn is not None:
        mag = mag / cn.double()[None, :]
    if qn is not None:
        mag = mag / qn.index_select(0, idx).double()[:, None]
    return _tolerance(int(Q.shape[1]), mag,
                      None if score is None else score.double())


def _self_check_rows(n_live: int, cap: int | None = None) -> list[int]:
    """Select deterministic rows for kernel validation.

    Check all rows when small; otherwise sample evenly spaced rows
    (including endpoints) plus BQ consecutive rows to cover every
    query-block lane. Sparse tile coverage is handled by `_self_check_pairs`.
    """
    cap = SELF_CHECK_ROWS if cap is None else int(cap)
    if n_live <= cap + BQ:
        return list(range(n_live))
    if cap <= 1:
        even = {0}
    else:
        even = {(i * (n_live - 1)) // (cap - 1) for i in range(cap)}
    mid = (n_live - BQ) // 2
    return sorted(even | set(range(mid, mid + BQ)))


def _self_check_pairs(tile_live, bq: int = BQ):
    """Select (tile, query) pairs for deterministic kernel validation.

    Samples the first and last live query of every tile, the final
    work-list block, and the busiest tile's first and last blocks.
    This covers tile boundaries and partial blocks that row sampling
    may miss. Returns two int64 device tensors.
    """
    import torch

    dev = tile_live.device
    n_tiles, nq = tile_live.shape
    cnt = tile_live.sum(dim=1)
    live_t = (cnt > 0).nonzero().flatten()
    if live_t.numel() == 0:
        e = torch.empty(0, dtype=torch.int64, device=dev)
        return e, e
    rows = tile_live.index_select(0, live_t).to(torch.uint8)
    first = rows.argmax(dim=1)                       # first maximal index
    last = (nq - 1) - rows.flip(1).argmax(dim=1)
    del rows
    tiles, qids = [live_t, live_t], [first, last]
    cnt_h = cnt.tolist()
    last_tile = int(live_t[-1])
    hot = max(range(n_tiles), key=lambda t: (cnt_h[t], -t))

    def block(t, b):
        q = tile_live[t].nonzero().flatten()[b * bq:(b + 1) * bq]
        tiles.append(torch.full_like(q, t))
        qids.append(q)

    block(last_tile, (cnt_h[last_tile] - 1) // bq)
    block(hot, 0)
    block(hot, (cnt_h[hot] - 1) // bq)
    return (torch.cat(tiles).to(torch.int64), torch.cat(qids).to(torch.int64))


def _check_pairs(Q, C, out, idx, tiles, qids, width, cn, qn):
    """Kernel output of each (tile, query) pair against an fp32 dot of the
    same operands (an independent summation order, so `_tolerance` applies
    as for the reference), over that tile's columns only -- no full-row
    GEMM. Returns the worst `err - tol` (<= 0 on success; +inf for a live
    pair left at -inf or NaN), read back once after every chunk."""
    import torch

    n_rows, d = int(C.shape[0]), int(Q.shape[1])
    if int(tiles.numel()) == 0:
        return -float("inf")
    pos = torch.searchsorted(idx, qids)
    ar = torch.arange(width, device=C.device)
    # Bounded gathers: at most ~4M fp32 corpus elements per chunk.
    chunk = max(1, (1 << 22) // max(1, width * d))
    worst = torch.full((), -float("inf"), dtype=torch.float64, device=C.device)
    for s in range(0, int(tiles.numel()), chunk):
        t, q, p = tiles[s:s + chunk], qids[s:s + chunk], pos[s:s + chunk]
        cols = t[:, None] * width + ar[None, :]
        valid = cols < n_rows
        cols = cols.clamp_max(n_rows - 1)
        Cg = C[cols]                                       # (p, width, d)
        Qg = Q.index_select(0, q)[:, :, None]              # (p, d, 1)
        want = torch.bmm(Cg, Qg).squeeze(2)
        mag = torch.bmm(Cg.double().abs(), Qg.double().abs()).squeeze(2)
        del Cg, Qg
        if cn is not None:
            want = want / cn[cols]
            mag = mag / cn.double()[cols]
        if qn is not None:
            want = want / qn.index_select(0, q)[:, None]
            mag = mag / qn.double().index_select(0, q)[:, None]
        got = out[p[:, None], cols]
        want = want.double()
        err = (got.double() - want).abs() - _tolerance(d, mag, want)
        # NaN (a -inf or NaN written to a live pair) must count as a failure.
        err = torch.where(torch.isnan(err), float("inf"), err)
        err = torch.where(valid, err, -float("inf"))
        worst = torch.maximum(worst, err.max())
    return float(worst)


def _dead_mask_ok(out, idx, tile_live, width) -> bool:
    """Over the WHOLE launch: every dead pair exactly -inf and every live
    pair finite (no -inf left unwritten, no NaN / +inf). Elementwise, no
    GEMM, in bounded row chunks, read back once."""
    import torch

    n_live, n_rows = int(out.shape[0]), int(out.shape[1])
    step = max(1, (1 << 24) // max(1, n_rows))
    bad = torch.zeros((), dtype=torch.bool, device=out.device)
    for r0 in range(0, n_live, step):
        keep = tile_live.index_select(1, idx[r0:r0 + step]).T
        keep = keep.repeat_interleave(width, dim=1)[:, :n_rows]
        o = out[r0:r0 + step]
        bad |= (torch.isneginf(o) != ~keep).any()
        bad |= (torch.isfinite(o) != keep).any()
    return not bool(bad)


def _self_check(Q, C, idx, tile_live, width, cn, qn, out) -> None:
    """Validate the first kernel launch for each compiled configuration.

    Compare the actual output against the reference within `score_tolerance`
    using bounded, deterministic coverage:
      - All live/dead pairs for correct masking and finite scores.
      - Sampled query rows across the full slice.
      - Tile boundaries and work-list edge blocks.

    A mismatch disables the kernel and raises `KernelMismatch`.
    Other errors propagate without marking the configuration checked,
    allowing a later launch to retry validation.
    """
    global _SELFCHECKS, _SELFCHECK_FAILURES
    import torch

    mask_ok = _dead_mask_ok(out, idx, tile_live, width)
    rows = _self_check_rows(int(idx.shape[0]))
    r_idx, r_out = idx, out
    if len(rows) < int(idx.shape[0]):
        pos = torch.tensor(rows, dtype=torch.int64, device=idx.device)
        r_idx = idx.index_select(0, pos)
        r_out = out.index_select(0, pos)
    want = reference(Q, C, r_idx, tile_live, width, cn, qn)
    dead = torch.isneginf(want)
    tol = score_tolerance(Q, C, r_idx, cn, qn, score=want)
    err = (r_out.double() - want.double()).abs()
    rows_mask_ok = bool(torch.equal(torch.isneginf(r_out), dead))
    over = (err - tol)[~dead]
    over = torch.where(torch.isnan(over), float("inf"), over)
    row_worst = float(over.max()) if over.numel() else -float("inf")
    del want, dead, tol, err, over
    tiles, qids = _self_check_pairs(tile_live)
    pair_worst = _check_pairs(Q, C, out, idx, tiles, qids, width, cn, qn)
    failed = [name for name, bad in (
        ("mask (a dead pair not -inf, or a live pair not finite)", not mask_ok),
        ("rows' dead-pair mask", not rows_mask_ok),
        (f"rows (a score off by {row_worst:.3e} beyond the bound)",
         row_worst > 0.0),
        (f"tile pairs (a score off by {pair_worst:.3e} beyond the bound)",
         pair_worst > 0.0),
    ) if bad]
    _SELFCHECKS += 1
    if failed:
        _SELFCHECK_FAILURES += 1
        kernel_disable(f"its first launch disagreed with the reference: "
                       f"{'; '.join(failed)}")
        raise KernelMismatch("tile re-score kernel failed its self-check: "
                             + "; ".join(failed))


def rescore(Q, C, idx, tile_live, width: int, cn=None, qn=None):
    """Exact scores of the live (query, tile) pairs; `-inf` everywhere else.

    `Q` is the FULL `(n_q, d)` query matrix, `C` the `(n_rows, d)` slice,
    `tile_live` `(ceil(n_rows / width), n_q)` bool, and `idx` the sorted
    queries live in at least one tile (`tile_live.any(0).nonzero()`). `cn`
    (cosine: the clamped corpus norms) divides each column, `qn` (cosine
    without packer scaling: the FULL query norms) each row. Returns
    `(scores, used_kernel)`, scores `(len(idx), n_rows)` float32. The kernel
    runs when `decline_reason` is None, the reference otherwise; callers on a
    GPU check `decline_reason` first and do not come here when it declines.
    """
    import torch

    global _LAUNCHES

    n_live, n_rows = int(idx.shape[0]), int(C.shape[0])
    if tuple(tile_live.shape) != (-(-n_rows // width), int(Q.shape[0])):
        raise ValueError(
            f"tile re-score: tile_live has shape {tuple(tile_live.shape)} for "
            f"{n_rows} rows in tiles of {width} and {int(Q.shape[0])} queries")
    if n_live == 0:
        return torch.empty((0, n_rows), dtype=torch.float32,
                           device=Q.device), False
    if not _kernel_ok(Q, C, tile_live, width, cn, qn):
        return reference(Q, C, idx, tile_live, width, cn, qn), False

    dev = Q.device
    out = torch.full((n_live, n_rows), float("-inf"), dtype=torch.float32,
                     device=dev)
    pos = torch.zeros(int(Q.shape[0]), dtype=torch.int32, device=dev)
    pos.index_copy_(0, idx, torch.arange(n_live, dtype=torch.int32, device=dev))
    qlist, toff, tcnt, bcum, nblk = build_worklist(tile_live)
    n_tiles = int(tile_live.shape[0])
    dummy = torch.empty(1, dtype=torch.float32, device=dev)
    nprog = _nprog(dev)
    try:
        ctx = torch.cuda.device(dev) if dev.type == "cuda" else _null()
        with ctx:
            _tile_rescore[(nprog,)](
                Q, C, cn if cn is not None else dummy,
                qn if qn is not None else dummy,
                qlist, toff, tcnt, bcum, nblk, out, pos,
                n_rows, int(Q.shape[1]), Q.stride(0), C.stride(0), out.stride(0),
                n_tiles,
                NT_P2=triton.next_power_of_2(n_tiles),
                TB=int(width), BQ=BQ, BK=BK,
                HAS_CN=cn is not None, HAS_QN=qn is not None, NPROG=nprog,
                num_warps=4, num_stages=2,
            )
    except Exception as exc:
        from nova_bf.twopass import is_oom

        if not is_oom(exc):
            # Deterministic compile/launch failure: off for the run. This
            # slice still propagates, so the caller takes the slice path.
            kernel_disable(f"{type(exc).__name__}: {exc}")
        raise
    _LAUNCHES += 1
    key = _compile_key(dev, width, cn, qn, n_tiles,
                       (n_rows, int(Q.shape[1]), Q.stride(0), C.stride(0),
                        out.stride(0), n_tiles),
                       (Q, C, cn if cn is not None else dummy,
                        qn if qn is not None else dummy,
                        qlist, toff, tcnt, bcum, nblk, out, pos))
    if key not in _CHECKED:
        # An OOM in the check propagates without marking the key, so the
        # caller falls back and a later slice re-checks.
        _self_check(Q, C, idx, tile_live, width, cn, qn, out)
        _CHECKED.add(key)
    return out, True


def _compile_key(dev, width, cn, qn, n_tiles, ints, tensors=()):
    """What selects a separately compiled kernel, so each compile is
    self-checked on its first launch: the device (and so NPROG), every
    constexpr that varies (TB, HAS_CN, HAS_QN and NT_P2 -- a short last
    slice falls in a smaller tile-count bucket), and Triton's
    specializations: of the integer arguments (== 1, divisible by 16) and
    of every pointer argument (`data_ptr() % 16 == 0`, in `tensors`, in
    launch order). BQ and BK are module constants."""
    return (str(dev), int(width), cn is not None, qn is not None,
            triton.next_power_of_2(max(1, int(n_tiles))),
            tuple((int(v) == 1, int(v) % 16 == 0) for v in ints),
            tuple(t.data_ptr() % 16 == 0 for t in tensors))


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
