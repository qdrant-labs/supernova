"""Prune coverage for paths a CPU-only run would otherwise never reach.

`test_topk_prune.py` builds its parts by hand and, on a CPU box, only ever
takes `_merge_topk`'s portable per-part branch with fully-valid dead rows.
These tests use `gpu_contract.install()` to poison dead rows the way
`_cutfill` leaves them and to route through each of the three dispatch
modes, then assert the answers are unchanged.

Every test that claims to cover the dead-row regime asserts
`dead_rows_seen > 0`, so a change that stops pruning fails loudly here
instead of passing vacuously.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import gpu_contract
from nova_bf import tiebreak
from nova_bf.tiebreak import SENTINEL_KEY, live_rows, pack, pack_topk
from test_topk_prune import (
    DIM, _many_file_corpus, _oracle_topk_ids, _run_dense, _write_queries,
)

# "native" runs the REAL kernel and is meaningful only where there is one:
# on CPU there is no Triton path and dead rows are already valid, so it would
# duplicate `test_topk_prune.py` rather than add coverage.
MODES = ["fold", "decline", "nofold"]
if __import__("torch").cuda.is_available():
    MODES.append("native")


def _corpus_and_queries(tmp_path, n_files=8, per_file=200, n_q=6, seed=5):
    cdir, cvecs, ids = _many_file_corpus(
        tmp_path, n_files=n_files, per_file=per_file, seed=seed)
    qv = np.random.default_rng(seed + 90).normal(size=(n_q, DIM)).astype(np.float32)
    return cdir, cvecs, ids, qv, _write_queries(tmp_path, qv)


# ---------------------------------------------------------------------------
# the three dispatch modes, with dead rows poisoned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_dispatch_modes_preserve_answers(tmp_path, monkeypatch, mode):
    """With dead rows holding a key that outranks everything, every dispatch
    mode must still agree with an independent f64 oracle. A path that reads a
    dead row promotes the poison and fails here."""
    monkeypatch.delenv("NOVA_BF_NO_PRUNE", raising=False)
    state = gpu_contract.install(monkeypatch, mode=mode)
    cdir, cvecs, ids, qv, qpath = _corpus_and_queries(tmp_path)

    t = _run_dense(tmp_path, cdir, qpath, f"o_{mode}", k=10, batch=32)

    assert state["dead_rows_seen"] > 0, f"{mode}: no dead row ever produced"
    for row, qid in enumerate(t["query_id"]):
        want = _oracle_topk_ids(qv[int(qid[1:])], cvecs, 10, ids)
        assert t["hit_ids"][row] == want, f"{mode}/{qid}: wrong hits"


@pytest.mark.parametrize("mode", MODES)
def test_dispatch_modes_match_prune_disabled(tmp_path, monkeypatch, mode):
    """Pruning stays a pure perf knob under each dispatch mode: identical
    hits and identical scores against a NOVA_BF_NO_PRUNE run."""
    cdir, _, _, _, qpath = _corpus_and_queries(tmp_path, n_files=6, per_file=150,
                                               n_q=5, seed=11)
    monkeypatch.setenv("NOVA_BF_NO_PRUNE", "1")
    base = _run_dense(tmp_path, cdir, qpath, f"base_{mode}", k=8, batch=32)

    monkeypatch.delenv("NOVA_BF_NO_PRUNE", raising=False)
    state = gpu_contract.install(monkeypatch, mode=mode)
    got = _run_dense(tmp_path, cdir, qpath, f"got_{mode}", k=8, batch=32)

    assert state["dead_rows_seen"] > 0, f"{mode}: prune never fired"
    assert got["hit_ids"] == base["hit_ids"]
    assert got["hit_scores"] == base["hit_scores"]


def test_decline_mode_actually_declines(tmp_path, monkeypatch):
    """Guard the guard: `decline` must reach the fall-through, not silently
    behave like `fold`."""
    monkeypatch.delenv("NOVA_BF_NO_PRUNE", raising=False)
    state = gpu_contract.install(monkeypatch, mode="decline")
    cdir, _, _, _, qpath = _corpus_and_queries(tmp_path, n_files=5, per_file=120,
                                               n_q=4, seed=3)
    _run_dense(tmp_path, cdir, qpath, "decl", k=6, batch=32)
    assert state["available_false"] > 0, "available() was never consulted"


# ---------------------------------------------------------------------------
# operator kill switches, with pruning on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("switch",
                         ["NOVA_BF_NO_FOLD_KERNEL", "NOVA_BF_NO_TOPK_KERNEL"])
def test_kill_switches_keep_results_identical(tmp_path, monkeypatch, switch):
    """The change has to stay correct under the operator escape hatches it
    now shares the code path with."""
    cdir, cvecs, ids, qv, qpath = _corpus_and_queries(tmp_path, n_files=6,
                                                      per_file=150, n_q=5, seed=17)
    monkeypatch.delenv("NOVA_BF_NO_PRUNE", raising=False)
    base = _run_dense(tmp_path, cdir, qpath, f"nb_{switch}", k=8, batch=32)

    monkeypatch.setenv(switch, "1")
    got = _run_dense(tmp_path, cdir, qpath, f"sw_{switch}", k=8, batch=32)

    assert got["hit_ids"] == base["hit_ids"]
    assert got["hit_scores"] == base["hit_scores"]
    for row, qid in enumerate(got["query_id"]):
        want = _oracle_topk_ids(qv[int(qid[1:])], cvecs, 8, ids)
        assert got["hit_ids"][row] == want


# ---------------------------------------------------------------------------
# pack_topk edges that only matter once `thr` is passed
# ---------------------------------------------------------------------------


def test_pack_topk_chunk_boundary_with_thr(monkeypatch):
    """`live` is computed after the chunked `cat`. Chunking must not change
    which rows come back live."""
    g = torch.Generator().manual_seed(4)
    scores = torch.randn(12, 9, generator=g)
    ordinal = torch.arange(9, dtype=torch.int64)
    thr = pack(torch.randn(12, 1, generator=g), torch.zeros(1, dtype=torch.int64))
    thr = thr.squeeze(1).contiguous()

    whole = pack_topk(scores, ordinal, k=4, thr=thr)
    monkeypatch.setattr(tiebreak, "PACK_TARGET_SLOTS", 9 * 2)  # force chunking
    chunked = pack_topk(scores, ordinal, k=4, thr=thr)

    assert torch.equal(chunked[2], whole[2]), "chunking changed the live mask"
    assert torch.equal(chunked[0].sort(dim=1).values,
                       whole[0].sort(dim=1).values)


def test_pack_topk_k_equals_n_cols_with_thr():
    """k == n_cols leaves the top-K exactly saturated; the live rule must
    still agree with computing it directly off the keys."""
    g = torch.Generator().manual_seed(6)
    scores = torch.randn(5, 4, generator=g)
    ordinal = torch.arange(4, dtype=torch.int64)
    thr = pack(torch.randn(5, 1, generator=g),
               torch.zeros(1, dtype=torch.int64)).squeeze(1).contiguous()

    keys, _, live = pack_topk(scores, ordinal, k=4, thr=thr)
    assert torch.equal(live, live_rows(keys, thr))


# ---------------------------------------------------------------------------
# the documented exception, and the new guards
# ---------------------------------------------------------------------------


def test_negative_nan_sorts_below_the_sentinel():
    """`live_rows`' docstring calls this out: a negative NaN (0xFFC00000) is
    the one score class dead even against a sentinel-only state."""
    neg_nan = torch.tensor([-0x00400000], dtype=torch.int32).view(torch.float32)
    key = pack(neg_nan.reshape(1, 1), torch.zeros(1, dtype=torch.int64))
    assert int(key[0, 0]) >> 32 < SENTINEL_KEY >> 32

    thr = torch.tensor([SENTINEL_KEY], dtype=torch.int64)
    assert int(live_rows(key, thr)[0]) == 0


def test_live_rows_rejects_a_malformed_thr():
    """The kernel gate rejects a bad `thr` by DECLINING, which lands here —
    where a length-1 `thr` would otherwise broadcast into a wrong mask."""
    keys = pack(torch.randn(4, 3), torch.arange(3, dtype=torch.int64))
    with pytest.raises(ValueError, match="thr must be int64"):
        live_rows(keys, torch.tensor([SENTINEL_KEY], dtype=torch.int64))
    with pytest.raises(ValueError, match="thr must be int64"):
        live_rows(keys, torch.zeros(4, dtype=torch.int32))


def test_live_rows_handles_a_zero_width_part():
    """A part with no candidates cannot displace anything, so no row is live.
    `amax` would raise on the empty reduction dim."""
    keys = torch.zeros(5, 0, dtype=torch.int64)
    thr = torch.full((5,), SENTINEL_KEY, dtype=torch.int64)
    live = live_rows(keys, thr)
    assert live.shape == (5,) and int(live.sum()) == 0


# ---------------------------------------------------------------------------
# dead rows are DEFINED, not uninitialised
# ---------------------------------------------------------------------------




def test_kernel_takes_no_sentinel_argument_any_more():
    """The sentinel FILL is gone, and this pins that it stays gone.

    `_cutfill` used to write k sentinel keys and k zero indices into every
    pruned row so that a consumer which read one would lose rather than
    corrupt. At the production shape that was 1.2 GB of stores per slice for
    output nobody reads — and once the run reaches steady state ~97% of rows
    are dead, so it was most of the kernel's write traffic. See G1 in
    `docs/brute-force/perf-design-2026-09-05.md`.

    What replaced it is not a weaker guarantee, it is a different one: NOBODY
    READS A DEAD ROW. `merge_triton._fold` returns before touching it and
    `compute._merge_topk`'s portable fallback folds live rows only, and
    `tests/test_dead_row_poison.py` proves both by filling those rows with
    values engineered to WIN any selection they leak into. A test beats a fill:
    the fill made a leak survivable, the test makes a leak fail.

    Keeping `SENTINEL` as a parameter would be actively harmful now — it would
    read as though the fill were still happening. Its absence also means a
    stale launch site still passing `SENTINEL=` dies with a TypeError instead
    of being silently ignored.

    This runs without CUDA on purpose. `test_dead_rows_are_not_written_at_all`
    below is the real proof, but it is GPU-gated, so on a CPU box — which is
    where this suite usually runs — nothing else here would notice the fill
    coming back.
    """
    import inspect

    from nova_bf import topk_triton

    if topk_triton._cutfill is None:
        pytest.skip(f"triton did not import: {topk_triton._UNAVAILABLE}")

    # `@triton.jit` wraps the function; the original is on `.fn`.
    fn = getattr(topk_triton._cutfill, "fn", topk_triton._cutfill)
    params = inspect.signature(fn).parameters
    assert "SENTINEL" not in params, (
        "topk_triton._cutfill still takes SENTINEL; dead rows are no longer "
        "filled, so the argument would be a lie"
    )
    # ...and the pointer that replaced the work it used to do is there.
    assert "ENC" in params and "HAS_ENC" in params, (
        "the kernel should emit encoded ids directly (G2)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="exercises the kernel")
def test_dead_rows_are_not_written_at_all():
    """A pruned row must come back EXACTLY as the buffer was handed over.

    Poison mode fills the output before the launch, so anything the kernel
    wrote is visible as a difference. Live rows must overwrite their own k
    slots completely (the kernel asserts it selects exactly k); dead rows must
    still be pure poison afterwards.
    """
    from nova_bf import topk_triton

    dev = "cuda"
    n_q, n_cols, k = 8, 64, 4
    g = torch.Generator(device="cpu").manual_seed(11)
    scores = torch.randn(n_q, n_cols, generator=g).float().to(dev)
    ordinal = torch.arange(n_cols, dtype=torch.int64, device=dev)
    enc = (torch.arange(n_cols, dtype=torch.int64, device=dev) + 10_000)

    # Threshold above every score for half the rows -> those rows are dead.
    thr = pack(torch.full((n_q, 1), 1e9), torch.zeros(1, dtype=torch.int64))
    thr = thr.squeeze(1).contiguous().to(dev)
    thr[: n_q // 2] = SENTINEL_KEY          # keep the first half alive

    os.environ["NOVA_BF_POISON_DEAD_ROWS"] = "1"
    try:
        keys, vals, live = topk_triton.topk(scores, ordinal, k, thr=thr, enc=enc)
    finally:
        del os.environ["NOVA_BF_POISON_DEAD_ROWS"]

    dead = (live == 0).nonzero(as_tuple=True)[0]
    assert dead.numel() > 0, "no row was pruned; the test proves nothing"
    assert (keys[dead] == topk_triton.POISON_KEY).all(), (
        "the kernel wrote into a dead row's keys"
    )
    assert (vals[dead] == topk_triton.POISON_ID).all(), (
        "the kernel wrote into a dead row's ids"
    )

    alive = (live != 0).nonzero(as_tuple=True)[0]
    assert alive.numel() > 0
    assert (keys[alive] != topk_triton.POISON_KEY).all(), (
        "a live row kept poison — the kernel did not fill all k slots"
    )
    # G2: a live row's values are the ENCODED IDS, not column indices.
    assert (vals[alive] >= 10_000).all() and (vals[alive] < 10_000 + n_cols).all()
