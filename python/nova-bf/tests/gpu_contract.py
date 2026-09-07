"""Make a CPU test take the GPU-shaped prune paths.

Two things a CPU run never exercises, both of which hide exactly the bug
class per-query pruning can introduce:

  1. `topk_triton._cutfill` leaves a DEAD row's output keys AND IDS
     UNINITIALIZED. The portable `pack_topk` leaves them fully valid, so a CPU
     test reads correct values precisely where the GPU would read garbage.
     (Both halves matter now: since G2 the kernel emits the encoded row id
     itself, so a dead row's id column is garbage too, not a harmless zero.)
  2. `merge_triton.enabled()` is `sample.is_cuda`, so on CPU the entire
     Triton branch of `_merge_topk` is dead code — the single-part
     no-neutralize case, the multi-part `(live_any != 0) & (l == 0)`
     sanitize, and the `available() is False` fall-through.

`install()` closes both: `pack_topk` poisons dead rows with a key that
OUTRANKS everything (so a read shows up as a wrong answer, not a silent
pass), and `merge_triton` is replaced by a CPU emulation of `_fold` that
reproduces the dead-row skip. Three dispatch modes become reachable:

    "fold"    the kernel path, dead rows skipped by the emulated fold
    "decline" `available()` says no -> `_merge_topk`'s sanitize fall-through
    "nofold"  `enabled()` says no -> the portable per-part branch
    "native"  patch NOTHING but the counters — on a GPU box this runs the
              REAL Triton kernel against REAL uninitialized dead rows, which
              is the thing the other three modes only imitate

Not a substitute for running on a GPU: this emulates the kernel's CONTRACT,
not the kernel. It catches callers that violate the contract, not bugs
inside the Triton source itself.
"""
from __future__ import annotations

import torch

import nova_bf.compute as compute_mod
from nova_bf import merge_triton
from nova_bf.tiebreak import live_rows as _real_live_rows
from nova_bf.tiebreak import pack_topk as _real_pack_topk

# A dead row's key is uninitialized memory. The adversarial choice is a value
# that BEATS every real candidate: if any path reads it, it wins the fold and
# the wrong answer is visible. Zero or int64-min would be indistinguishable
# from correct pruning.
POISON_KEY = 2**62

# And its id. `_cutfill` no longer writes the id column of a dead row either,
# so it holds whatever the fresh buffer did. Encoded row ids are
# `gidx * MAX_ROWS_PER_FILE + row`, hence non-negative and bounded by the
# corpus; this is neither, so if one reaches the decode it either indexes past
# the file list or emerges as an id nobody can explain — both loud. Zero, the
# old fill, would decode to a real corpus row.
POISON_ID = 2**61 + 12345


def poisoning_pack_topk(scores, ordinal, k, scale=None, thr=None, encoded=None,
                        rank=None):
    """`pack_topk`, with `_cutfill`'s uninitialized-dead-row contract."""
    keys, vals, live = _real_pack_topk(scores, ordinal, k, scale, thr=thr,
                                       encoded=encoded, rank=rank)
    if live is not None:
        dead = live == 0
        if dead.any():
            _state["dead_rows_seen"] += int(dead.sum())
            _state["dead_rows_via_pack_topk"] += int(dead.sum())
            keys = keys.clone()
            keys[dead] = POISON_KEY
            # BOTH halves: the kernel writes neither for a dead row.
            vals = vals.clone()
            vals[dead] = POISON_ID
    return keys, vals, live


_state = {"folds": 0, "dead_rows_seen": 0, "available_false": 0,
          "dead_rows_via_pack_topk": 0, "dead_rows_via_live_rows": 0}


def counting_pack_topk(scores, ordinal, k, scale=None, thr=None, encoded=None,
                       rank=None):
    """Count without poisoning — for `native`, where the real kernel has
    already left dead rows uninitialized on its own."""
    keys, idx, live = _real_pack_topk(scores, ordinal, k, scale, thr=thr,
                                      encoded=encoded, rank=rank)
    if live is not None:
        dead = int((live == 0).sum())
        _state["dead_rows_seen"] += dead
        _state["dead_rows_via_pack_topk"] += dead
    return keys, idx, live


def counting_live_rows(keys, thr):
    """`compute.py` has TWO prune entry points: `pack_topk` for slices wide
    enough to pre-top-K, and a direct `live_rows` for narrow ones (a tight
    multivector token budget takes this branch for every slice). Counting only
    the first makes a working prune look inert."""
    live = _real_live_rows(keys, thr)
    dead = int((live == 0).sum())
    _state["dead_rows_via_live_rows"] += dead
    _state["dead_rows_seen"] += dead
    return live


def _emul_available(state_key, state_enc, part_key, part_enc, k,
                    live=None, thr=None):
    """Mirror `merge_triton.available`'s gate on CPU tensors."""
    if (live is None) != (thr is None):
        return False
    n_q = state_key.shape[0]
    if live is not None:
        if not (live.ndim == 1 and live.numel() == n_q
                and live.dtype is torch.uint8 and live.is_contiguous()):
            return False
        if not (thr.ndim == 1 and thr.numel() == n_q
                and thr.dtype is torch.int64 and thr.is_contiguous()):
            return False
    for t in (state_key, state_enc, part_key):
        if t.dtype is not torch.int64 or t.ndim != 2 or not t.is_contiguous():
            return False
    if part_enc.dtype is not torch.int64 or part_enc.ndim not in (1, 2):
        return False
    if not part_enc.is_contiguous():
        return False
    if n_q <= 0 or state_key.shape[1] != k or state_enc.shape != state_key.shape:
        return False
    if part_key.shape[0] != n_q:
        return False
    w = part_key.shape[1]
    if part_enc.ndim == 2 and part_enc.shape != part_key.shape:
        return False
    if part_enc.ndim == 1 and part_enc.numel() != w:
        return False
    return 0 < w and k + w <= merge_triton.MAX_BLOCK


def _emul_fold(state_key, state_enc, part_key, part_enc, k, live=None, thr=None):
    """CPU emulation of `_fold`, including the dead-row skip and the in-place
    `thr` update the kernel performs as a by-product.

    IN PLACE, like the kernel: a dead row is not written, not merely written
    back unchanged. The difference is the whole point of the emulation — if a
    caller kept a reference to the pre-fold state expecting it to survive, an
    out-of-place emulation would hide that.
    """
    _state["folds"] += 1
    n_q = state_key.shape[0]
    pe = (part_enc if part_enc.ndim == 2
          else part_enc.unsqueeze(0).expand(n_q, -1))
    if live is None:
        merged_k = torch.cat([state_key, part_key], dim=1)
        merged_e = torch.cat([state_enc, pe], dim=1)
        nk, idx = torch.topk(merged_k, k=k, dim=1, sorted=False)
        state_key.copy_(nk)
        state_enc.copy_(merged_e.gather(1, idx))
        return state_key, state_enc

    alive = live.bool()
    # Dead rows: NOT TOUCHED AT ALL — not read, not written, `thr` unchanged.
    if alive.any():
        merged_k = torch.cat([state_key[alive], part_key[alive]], dim=1)
        merged_e = torch.cat([state_enc[alive], pe[alive]], dim=1)
        nk, idx = torch.topk(merged_k, k=k, dim=1, sorted=False)
        new_e = merged_e.gather(1, idx)
        state_key[alive] = nk
        state_enc[alive] = new_e
        thr[alive] = nk.min(dim=1).values
    return state_key, state_enc


def install(monkeypatch, mode="fold"):
    """Patch compute/merge_triton so a CPU run takes a GPU-shaped path.

    Returns the counter dict; assert on `dead_rows_seen` to prove the test
    actually reached the regime it claims to cover.
    """
    if mode not in ("fold", "decline", "nofold", "native"):
        raise ValueError(f"unknown mode {mode!r}")
    for key in _state:
        _state[key] = 0

    if mode == "native":
        # Leave every dispatch decision and the kernel itself alone.
        monkeypatch.setattr(compute_mod, "pack_topk", counting_pack_topk)
        monkeypatch.setattr(compute_mod, "live_rows", counting_live_rows)
        return _state

    monkeypatch.setattr(compute_mod, "pack_topk", poisoning_pack_topk)
    monkeypatch.setattr(compute_mod, "live_rows", counting_live_rows)

    if mode == "nofold":
        monkeypatch.setattr(merge_triton, "enabled", lambda sample: False)
        return _state

    monkeypatch.setattr(merge_triton, "enabled", lambda sample: True)
    if mode == "decline":
        def _never(*a, **kw):
            _state["available_false"] += 1
            return False
        monkeypatch.setattr(merge_triton, "available", _never)
    else:
        monkeypatch.setattr(merge_triton, "available", _emul_available)
    monkeypatch.setattr(merge_triton, "fold", _emul_fold)
    return _state
