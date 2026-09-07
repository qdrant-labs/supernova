"""Nobody reads a dead row.

`topk_triton._cutfill` no longer writes anything into a pruned row — not the
k sentinel keys, not the k ids. That fill was hardening, and at the production
shape it cost 1.2 GB of stores per slice on output nobody reads, on ~97% of
rows once a rank reaches steady state (G1 in
`docs/brute-force/perf-design-2026-09-05.md`). What replaced it is this file.

The fill made a stray read SURVIVABLE — a sentinel loses to every real
candidate, so reading one was wasteful rather than wrong. These tests make a
stray read FAIL: dead rows are filled with a key that beats everything and an
id that exists nowhere, so any path that reads one produces a visibly wrong
answer instead of a quietly correct-looking one. That is a strictly stronger
guarantee, and it costs nothing at run time.

Two readers have to be proven, because they are separate implementations of
the same rule:

  * `merge_triton._fold` — returns on a dead row before touching the part.
  * `compute._merge_topk`'s PORTABLE fallback — gathers the live rows and
    folds only those. This one is easy to get wrong. Rows dead in EVERY part
    are excluded by that gather, but a row live overall and dead in ONE part
    still has its whole concatenated row read, so those cells are neutralized
    instead — with `NEUTRAL_KEY`, which is strictly below both real keys and
    the state's sentinels. `SENTINEL_KEY` tied with an under-filled state's own
    slots, and `sorted=False` promises no order, so the neutralized cell could
    take the slot and carry a dead row's garbage id along. That was containable
    rather than unsound (the id rode a -inf key and the output gate dropped it,
    and the Triton fold never lost the tie at all) — `NEUTRAL_KEY` makes it
    impossible instead of device-dependent.

Both halves of a dead row are poisoned. Poisoning only the keys would miss
G2's change entirely: the kernel emits the encoded row id itself now, so a
dead row's id column is garbage too, where it used to be a harmless zero.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from nova_bf.compute import _merge_topk
from nova_bf.tiebreak import SENTINEL_KEY, sentinel_key

# A key that outranks every real candidate, and an id that is not a corpus row.
POISON_KEY = 2**62
POISON_ID = 2**61 + 999

# Prefill for an UNALIASED fold's output buffers. Distinct from POISON_KEY so a
# fold that fails to write a slot cannot be mistaken for one that wrote a
# poisoned value, and outside `_fixture`'s key range so it can never be a
# legitimate winner.
_UNWRITTEN = 2**62 + 1


def _fixture(n_q=16, k=4, w=6, seed=0, dead_frac=0.5):
    """A running top-K state, one pending part, and a liveness mask.

    Returns everything twice over: the poisoned inputs the code under test
    sees, and the clean inputs the reference uses. `live` is chosen
    independently of the scores on purpose — the point is that the CALLER's
    liveness decision is honoured verbatim, not re-derived.
    """
    g = torch.Generator().manual_seed(seed)
    state_key = torch.randint(-(2**60), 2**60, (n_q, k), generator=g,
                              dtype=torch.int64)
    state_enc = torch.randint(0, 10**6, (n_q, k), generator=g, dtype=torch.int64)
    part_key = torch.randint(-(2**60), 2**60, (n_q, w), generator=g,
                             dtype=torch.int64)
    part_enc = torch.randint(0, 10**6, (n_q, w), generator=g, dtype=torch.int64)
    live = (torch.rand(n_q, generator=g) > dead_frac).to(torch.uint8)
    # Guarantee both regimes exist however the dice fell -- but ONLY when the
    # caller asked for a mix. Forcing these unconditionally made `dead_frac`
    # 0.0 and 1.0 silent duplicates of the mixed case, so the all-live and
    # all-dead regimes were never actually exercised by anything
    # parametrized over it.
    if 0.0 < dead_frac < 1.0:
        live[0] = 0
        live[1] = 1
    elif dead_frac <= 0.0:
        live[:] = 1
    else:
        live[:] = 0
    thr = state_key.min(dim=1).values.clone()
    return state_key, state_enc, part_key, part_enc, live, thr


def _poison(part_key, part_enc, live):
    """What the kernel actually leaves behind in a dead row."""
    pk, pe = part_key.clone(), part_enc.clone()
    dead = live == 0
    pk[dead] = POISON_KEY
    pe[dead] = POISON_ID
    return pk, pe


def _reference(state_key, state_enc, part_key, part_enc, live, k):
    """What the fold must produce: live rows folded, dead rows untouched.

    Deliberately not a rearrangement of the implementation — a plain
    concatenate-and-select over the CLEAN part, applied only where `live`.
    """
    out_k, out_e = state_key.clone(), state_enc.clone()
    alive = live.bool()
    if alive.any():
        mk = torch.cat([state_key[alive], part_key[alive]], dim=1)
        me = torch.cat([state_enc[alive], part_enc[alive]], dim=1)
        nk, idx = torch.topk(mk, k=k, dim=1, sorted=False)
        out_k[alive] = nk
        out_e[alive] = me.gather(1, idx)
    return out_k, out_e


def _sorted_rows(t):
    """Fold output is unordered within a row, so compare as sets."""
    return torch.sort(t, dim=1).values


class TestPortableFallback:
    """`_merge_topk` on CPU: `merge_triton.enabled()` is `sample.is_cuda`, so
    every one of these takes the portable path by construction."""

    @pytest.mark.parametrize("dead_frac", [0.0, 0.5, 0.97, 1.0])
    def test_dead_rows_survive_bit_identical(self, dead_frac):
        sk, se, pk, pe, live, thr = _fixture(n_q=32, seed=7, dead_frac=dead_frac)
        before_k, before_e = sk.clone(), se.clone()
        before_thr = thr.clone()
        ppk, ppe = _poison(pk, pe, live)

        got_k, got_e = _merge_topk(sk, se, [(ppk, ppe, live)], 4, thr=thr)

        dead = (live == 0)
        assert torch.equal(got_k[dead], before_k[dead]), "a dead row's keys moved"
        assert torch.equal(got_e[dead], before_e[dead]), "a dead row's ids moved"
        assert torch.equal(thr[dead], before_thr[dead]), (
            "a dead row's threshold moved, so its state must have been re-folded"
        )
        # The poison must not be anywhere in the result.
        assert not (got_k == POISON_KEY).any()
        assert not (got_e == POISON_ID).any()

    @pytest.mark.parametrize("dead_frac", [0.0, 0.5, 0.97])
    def test_live_rows_match_the_reference(self, dead_frac):
        sk, se, pk, pe, live, thr = _fixture(n_q=32, seed=11, dead_frac=dead_frac)
        ref_k, ref_e = _reference(sk.clone(), se.clone(), pk, pe, live, 4)
        ppk, ppe = _poison(pk, pe, live)

        got_k, got_e = _merge_topk(sk, se, [(ppk, ppe, live)], 4, thr=thr)

        alive = live.bool()
        assert torch.equal(_sorted_rows(got_k[alive]), _sorted_rows(ref_k[alive]))
        # ids follow their keys, so compare the (key, id) pairing, not the sets
        for r in alive.nonzero(as_tuple=True)[0].tolist():
            got = dict(zip(got_k[r].tolist(), got_e[r].tolist()))
            ref = dict(zip(ref_k[r].tolist(), ref_e[r].tolist()))
            assert got == ref, f"row {r}: {got} != {ref}"

    def test_thr_tracks_the_new_state_min_for_live_rows(self):
        sk, se, pk, pe, live, thr = _fixture(n_q=24, seed=3)
        before_thr = thr.clone()
        ppk, ppe = _poison(pk, pe, live)
        got_k, _ = _merge_topk(sk, se, [(ppk, ppe, live)], 4, thr=thr)
        alive = live.bool()
        assert torch.equal(thr[alive], got_k[alive].min(dim=1).values)
        # And ONLY for live rows: the kernel path leaves a dead row's `thr`
        # alone (its state did not move, so neither did its minimum), so the
        # portable path must too, or the two diverge on the next slice.
        assert torch.equal(thr[~alive], before_thr[~alive])

    def test_a_row_live_overall_but_dead_in_one_part(self):
        """The multi-part case, which is the one place a `masked_fill_` is
        still needed: the row folds, so its whole concatenated row is read,
        and one of the parts contributing to it is garbage."""
        sk, se, pk1, pe1, live1, thr = _fixture(n_q=20, w=3, seed=5)
        _, _, pk2, pe2, live2, _ = _fixture(n_q=20, w=3, seed=6)
        # Make the liveness masks genuinely disagree.
        live1 = torch.zeros(20, dtype=torch.uint8)
        live2 = torch.zeros(20, dtype=torch.uint8)
        live1[:10] = 1
        live2[5:15] = 1
        live_any = live1 | live2
        assert ((live1 == 0) & (live_any != 0)).any(), "premise broke"

        ppk1, ppe1 = _poison(pk1, pe1, live1)
        ppk2, ppe2 = _poison(pk2, pe2, live2)

        # Reference: each part contributes only where IT is live.
        neg = torch.full((1,), SENTINEL_KEY, dtype=torch.int64)
        rk1 = torch.where(live1.bool().unsqueeze(1), pk1, neg)
        rk2 = torch.where(live2.bool().unsqueeze(1), pk2, neg)
        ref_k, ref_e = _reference(
            sk.clone(), se.clone(),
            torch.cat([rk1, rk2], dim=1), torch.cat([pe1, pe2], dim=1),
            live_any, 4,
        )

        before_k = sk.clone()
        got_k, got_e = _merge_topk(
            sk, se, [(ppk1, ppe1, live1), (ppk2, ppe2, live2)], 4, thr=thr)

        dead = live_any == 0
        assert torch.equal(got_k[dead], before_k[dead])
        assert not (got_k == POISON_KEY).any(), "poison entered the top-K"
        assert not (got_e == POISON_ID).any(), "a poisoned id entered the top-K"
        alive = live_any.bool()
        assert torch.equal(_sorted_rows(got_k[alive]), _sorted_rows(ref_k[alive]))

    def test_an_all_dead_flush_is_a_no_op(self):
        sk, se, pk, pe, live, thr = _fixture(n_q=8, seed=9)
        live = torch.zeros(8, dtype=torch.uint8)
        before_k, before_e, before_thr = sk.clone(), se.clone(), thr.clone()
        ppk, ppe = _poison(pk, pe, live)
        got_k, got_e = _merge_topk(sk, se, [(ppk, ppe, live)], 4, thr=thr)
        assert torch.equal(got_k, before_k) and torch.equal(got_e, before_e)
        assert torch.equal(thr, before_thr)

    def test_an_under_filled_sentinel_state_still_rejects_poison(self):
        """The specific tie the old `masked_fill_(SENTINEL_KEY)` could lose.

        A state made entirely of sentinels has no real key to beat a
        neutralized dead part row, so whether the part won came down to
        `torch.topk`'s unspecified tie order. Folding live rows only removes
        the question.
        """
        n_q, k, w = 12, 4, 5
        sk = sentinel_key((n_q, k), "cpu")
        se = torch.zeros((n_q, k), dtype=torch.int64)
        g = torch.Generator().manual_seed(21)
        pk = torch.randint(-(2**60), 2**60, (n_q, w), generator=g, dtype=torch.int64)
        pe = torch.randint(0, 10**6, (n_q, w), generator=g, dtype=torch.int64)
        live = torch.zeros(n_q, dtype=torch.uint8)
        live[::2] = 1
        thr = sk.min(dim=1).values.clone()
        ppk, ppe = _poison(pk, pe, live)

        got_k, got_e = _merge_topk(sk, se, [(ppk, ppe, live)], k, thr=thr)

        dead = live == 0
        assert (got_k[dead] == SENTINEL_KEY).all(), (
            "a dead row's sentinel state was displaced"
        )
        assert not (got_e == POISON_ID).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="exercises the kernel")
class TestKernelFold:
    """The same rule, through the real Triton fold."""

    @pytest.mark.parametrize("dead_frac", [0.5, 0.97])
    def test_kernel_fold_ignores_poisoned_dead_rows(self, dead_frac):
        from nova_bf import merge_triton

        sk, se, pk, pe, live, thr = _fixture(
            n_q=512, k=32, w=64, seed=13, dead_frac=dead_frac)
        sk, se = sk.cuda(), se.cuda()
        pk, pe, live, thr = pk.cuda(), pe.cuda(), live.cuda(), thr.cuda()
        ppk, ppe = _poison(pk, pe, live)
        ref_k, ref_e = _reference(sk.clone(), se.clone(), pk, pe, live, 32)
        before_k, before_e = sk.clone(), se.clone()
        before_thr = thr.clone()

        assert merge_triton.available(sk, se, ppk, ppe, 32, live, thr), (
            "the kernel declined; this test would prove nothing"
        )
        got_k, got_e = merge_triton.fold(sk, se, ppk, ppe, 32, live, thr)

        dead = live == 0
        assert torch.equal(got_k[dead], before_k[dead])
        assert torch.equal(got_e[dead], before_e[dead])
        assert torch.equal(thr[dead], before_thr[dead])
        assert not (got_k == POISON_KEY).any()
        assert not (got_e == POISON_ID).any()
        alive = live.bool()
        assert torch.equal(_sorted_rows(got_k[alive]),
                           _sorted_rows(ref_k[alive].cuda()))

    def test_fold_is_in_place(self):
        """`fold` returns the state tensors themselves. A caller that kept the
        old state would be reading the new one — pinned so the contract is
        explicit rather than incidental."""
        from nova_bf import merge_triton

        sk, se, pk, pe, live, thr = _fixture(n_q=64, k=8, w=8, seed=17)
        sk, se = sk.cuda(), se.cuda()
        pk, pe, live, thr = pk.cuda(), pe.cuda(), live.cuda(), thr.cuda()
        got_k, got_e = merge_triton.fold(sk, se, pk, pe, 8, live, thr)
        assert got_k is sk and got_e is se

    def test_matches_an_out_of_place_fold_exactly(self):
        """In-place aliasing (`OK=SK`, `OE=SE`) is safe only because every
        load in the kernel precedes every store. If a compiler ever reordered
        one, or a future edit moved a load below a store, this is what
        catches it: the same fold with SEPARATE outputs must agree bit for
        bit, on inputs wide enough that state and part really overlap."""
        from nova_bf import merge_triton

        for seed in range(8):
            sk, se, pk, pe, live, thr = _fixture(
                n_q=1024, k=64, w=128, seed=100 + seed, dead_frac=0.3)
            sk, se = sk.cuda(), se.cuda()
            pk, pe, live, thr = pk.cuda(), pe.cuda(), live.cuda(), thr.cuda()

            # UNALIASED: the same kernel writing into buffers that are not its
            # inputs, via the `_out` test seam. Prefilled with a value the fold
            # cannot produce, so a slot the kernel fails to write is caught
            # here rather than reading as agreement.
            ref_k = torch.full_like(sk, _UNWRITTEN)
            ref_e = torch.full_like(se, _UNWRITTEN)
            merge_triton.fold(sk.clone(), se.clone(), pk, pe, 64, live,
                              thr.clone(), _out=(ref_k, ref_e))
            alive = live.bool()
            assert alive.any() and not alive.all(), f"seed {seed}: degenerate fixture"
            # Every LIVE row must have been written. Dead rows are deliberately
            # not: the kernel returns on them before any store, so they keep
            # the prefill -- which is itself the dead-row contract.
            assert not (ref_k[alive] == _UNWRITTEN).any(), (
                f"seed {seed}: the unaliased fold left a live row unwritten")
            assert (ref_k[~alive] == _UNWRITTEN).all(), (
                f"seed {seed}: the unaliased fold WROTE a dead row")

            # ALIASED: the production call, writing through its own inputs.
            in_k, in_e = merge_triton.fold(sk, se, pk, pe, 64, live, thr)
            assert in_k.data_ptr() == sk.data_ptr(), "expected the in-place fold"
            assert torch.equal(in_k[alive], ref_k[alive]), (
                f"seed {seed}: aliasing changed the keys — a load moved below a store")
            assert torch.equal(in_e[alive], ref_e[alive]), (
                f"seed {seed}: aliasing changed the ids — a load moved below a store")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="exercises the kernel")
def test_poison_env_is_output_neutral_end_to_end():
    """The switch itself must change nothing but what dead rows contain.

    `NOVA_BF_POISON_DEAD_ROWS` fills the pre-top-K output before the launch.
    If any code path read a dead row, this run would differ from the clean
    one — which is exactly the assertion, made against a real selection rather
    than a synthetic fixture.
    """
    from nova_bf.tiebreak import pack_topk

    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(31)
    n_q, n_cols, k = 256, 512, 16
    scores = torch.randn(n_q, n_cols, generator=g).float().to(dev)
    ordinal = torch.arange(n_cols, dtype=torch.int64, device=dev)
    enc = torch.arange(n_cols, dtype=torch.int64, device=dev) * 7 + 3
    # A threshold that really does kill most rows, like steady state. It has to
    # be built in PACKED-KEY space: a plain large int (this used to be `2**40`)
    # is a tiny SCORE once the high half is read off, so it pruned nothing at
    # all and the poison below was never exercised.
    from nova_bf.tiebreak import pack
    unreachable = pack(torch.full((n_q, 1), 10.0, device=dev),
                       torch.zeros(1, dtype=torch.int64, device=dev))[:, 0]
    thr = unreachable.clone()          # scores are ~N(0,1), so 10.0 is out of reach
    thr[:16] = SENTINEL_KEY            # ...except these, which stay live

    state = sentinel_key((n_q, k), dev)
    state_e = torch.zeros((n_q, k), dtype=torch.int64, device=dev)

    def run(poison: bool):
        if poison:
            os.environ["NOVA_BF_POISON_DEAD_ROWS"] = "1"
        try:
            sk, se = state.clone(), state_e.clone()
            t = thr.clone()
            key, val, live = pack_topk(scores, ordinal, k, thr=t, encoded=enc)
            return _merge_topk(sk, se, [(key, val, live)], k, thr=t) + (key, live)
        finally:
            os.environ.pop("NOVA_BF_POISON_DEAD_ROWS", None)

    clean_k, clean_e, _, clean_live = run(False)
    dirty_k, dirty_e, dirty_part, dirty_live = run(True)

    # PREMISES. Without these the comparison below passes trivially whenever
    # the poison never gets applied — a broken `_poisoning()` would make both
    # runs identical and the test would still be green.
    from nova_bf import topk_triton
    assert topk_triton.usage()["launches"] > 0, "the kernel never ran"
    n_dead = int((dirty_live == 0).sum())
    assert n_dead > 0, "no row was pruned, so nothing was poisoned"
    assert (dirty_part[dirty_live == 0] == topk_triton.POISON_KEY).all(), (
        "dead rows are not actually carrying the poison — the switch did nothing")
    assert torch.equal(clean_live, dirty_live), "poisoning changed the prune decisions"

    assert torch.equal(clean_k, dirty_k), "poisoning dead rows changed the keys"
    assert torch.equal(clean_e, dirty_e), "poisoning dead rows changed the ids"


# ---------------------------------------------------------------------------
# the invariants the fold's safety actually rests on
# ---------------------------------------------------------------------------


def test_pruned_parts_in_an_unpruned_flush_raise():
    """`thr=None` means nothing was pruned, so no part may carry a liveness
    mask. If one does, some rows are dead, the unpruned path reads whole rows,
    and it would read a dead row's garbage. `_merge_topk` guards that with a
    real `raise` (not `assert`, which `python -O` drops) — nothing else
    constructs the illegal state, so this is what keeps the guard honest."""
    sk, se, pk, pe, live, _ = _fixture(n_q=8, seed=5)
    with pytest.raises(RuntimeError, match="unpruned flush"):
        _merge_topk(sk, se, [(pk, pe, live)], 4, thr=None)
    # and the legal shape still works
    got_k, _ = _merge_topk(sk, se, [(pk, pe, None)], 4, thr=None)
    assert got_k.shape == (8, 4)


def test_the_neutraliser_is_strictly_below_the_sentinel():
    """A cell neutralized to exactly `SENTINEL_KEY` TIES with an under-filled
    state's own sentinels, and `torch.topk(sorted=False)` may return either —
    handing back the neutralized cell's id, which for a dead row is garbage.
    `NEUTRAL_KEY` has to lose, not tie."""
    from nova_bf.tiebreak import NEUTRAL_KEY, unpack_score

    assert NEUTRAL_KEY < SENTINEL_KEY
    assert int(sentinel_key((1,), "cpu").item()) == SENTINEL_KEY
    # It must also be safe if it somehow survives into a result: the output
    # gate is `sc > -inf`, and this must not pass it.
    sc = unpack_score(torch.tensor([NEUTRAL_KEY], dtype=torch.int64))
    assert not bool(sc > float("-inf")), "a neutralized slot could reach output"


def test_the_neutraliser_is_actually_USED_at_the_neutralise_sites():
    """Pinning `NEUTRAL_KEY`'s VALUE is not enough — the sites in `_merge_topk`
    have to reach for it. Reverting just those two `masked_fill_` calls to
    `SENTINEL_KEY`, leaving the constant alone, changed no other test in this
    file, so this is the one that notices.

    The discriminating shape is subtle: the state must be under-filled AND the
    live part must supply FEWER THAN k real candidates, so the state's own
    sentinels still compete for slots. Give the part k-or-more winners and it
    fills the top-k by itself, the sentinels never enter the comparison, and a
    neutraliser equal to `SENTINEL_KEY` looks perfectly safe."""
    k = 4
    sk = sentinel_key((1, k), "cpu").clone()
    se = torch.tensor([[101, 102, 103, 104]], dtype=torch.int64)
    thr = sk.min(dim=1).values.clone()
    # ONE real candidate against k=4 slots: three sentinels survive the fold.
    live_k = torch.tensor([[5 << 40]], dtype=torch.int64)
    live_e = torch.tensor([[777]], dtype=torch.int64)
    dead_k = torch.full((1, 3), POISON_KEY, dtype=torch.int64)
    dead_e = torch.full((1, 3), POISON_ID, dtype=torch.int64)

    got_k, got_e = _merge_topk(
        sk, se,
        [(live_k, live_e, torch.ones(1, dtype=torch.uint8)),
         (dead_k, dead_e, torch.zeros(1, dtype=torch.uint8))],
        k, thr=thr,
    )
    assert POISON_ID not in got_e[0].tolist(), (
        f"a dead row's id reached the top-K: {got_e[0].tolist()} — the "
        f"neutralised cells tied with the state's sentinels and won")
    assert POISON_KEY not in got_k[0].tolist()


def test_a_neutralised_cell_loses_to_an_under_filled_sentinel_state():
    """The configuration that made `SENTINEL_KEY` unsound, end to end: a state
    that still holds sentinels, and a row live in one part but dead in another
    so its cells get neutralized rather than excluded."""
    n_q, k, w = 8, 4, 4
    sk = sentinel_key((n_q, k), "cpu").clone()
    se = torch.zeros((n_q, k), dtype=torch.int64)
    thr = sk.min(dim=1).values.clone()
    ordv = torch.arange(w, dtype=torch.int64)

    live_part = torch.randint(0, 2**40, (n_q, w), dtype=torch.int64)
    live_enc = ordv.expand(n_q, w).contiguous().clone()
    l_live = torch.ones(n_q, dtype=torch.uint8)

    dead_part = torch.full((n_q, w), POISON_KEY, dtype=torch.int64)
    dead_enc = torch.full((n_q, w), POISON_ID, dtype=torch.int64)
    l_dead = torch.zeros(n_q, dtype=torch.uint8)

    got_k, got_e = _merge_topk(
        sk, se,
        [(live_part, live_enc, l_live), (dead_part, dead_enc, l_dead)],
        k, thr=thr,
    )
    assert not (got_e == POISON_ID).any(), "a dead row's id entered the top-K"
    assert not (got_k == POISON_KEY).any(), "a dead row's key entered the top-K"


def test_the_retry_is_allowed_only_for_pre_enqueue_failures(monkeypatch):
    """`merge_triton.fold` writes THROUGH the state, so re-folding the same
    part after a failure would count it twice — duplicate ids in the top-K and
    a real k-th candidate pushed out, silently.

    What makes the retry sound is that THIS attempt wrote nothing, and that is
    a property of the EXCEPTION TYPE: Triton raises compile and
    launch-configuration failures before any work reaches the device. It is NOT
    a property of history. A shape whose specialization has not been compiled
    yet can fail that way at any point in a run (`BLOCK` follows
    `k + pending_width`, which changes between a steady flush and the final
    drain), so gating on "the kernel has succeeded before" would kill a ~2h
    rank for a failure that provably wrote nothing.
    """
    from nova_bf import merge_triton

    sk, se, pk, pe, live, thr = _fixture(n_q=8, seed=9)

    # `enabled` is CUDA-only, so the fold branch is unreachable on CPU without
    # this. The recovery logic is pure control flow and needs no real kernel.
    monkeypatch.setattr(merge_triton, "enabled", lambda sample: True)
    monkeypatch.setattr(merge_triton, "available", lambda *a, **kw: True)
    monkeypatch.setattr(merge_triton, "disable", lambda exc: None)

    def raise_(exc):
        def _f(*a, **kw):
            raise exc
        monkeypatch.setattr(merge_triton, "fold", _f)

    def fold_it():
        return _merge_topk(sk.clone(), se.clone(), [(pk, pe, live)], 4,
                           thr=thr.clone())

    # PRE-ENQUEUE: triton's own class names, so these must DEGRADE.
    for name, msg in (("OutOfResources", "out of resource: shared memory"),
                      ("CompilationError", "could not compile"),
                      ("CompileTimeAssertionFailure", "static assert")):
        raise_(type(name, (RuntimeError,), {})(msg))
        got_k, _ = fold_it()
        assert got_k.shape == (8, 4), f"{name} must degrade, not kill the rank"

    # ANYTHING ELSE may already have written, so it must PROPAGATE.
    raise_(RuntimeError("an illegal memory access was encountered"))
    with pytest.raises(RuntimeError, match="illegal memory access"):
        fold_it()

    # OOM propagates too, and is matched by MESSAGE as well as type: a CUDA OOM
    # from inside Triton's module load arrives as a plain RuntimeError, which
    # `torch.cuda.OutOfMemoryError` alone would miss.
    raise_(RuntimeError("CUDA error: out of memory"))
    with pytest.raises(RuntimeError, match="out of memory"):
        fold_it()


def test_is_pre_enqueue_classifies_triton_errors_by_name():
    """The classifier matches over the MRO by CLASS NAME because triton moves
    these between modules across releases — on 3.8.0 `CompilationError` lives
    in `triton.compiler.errors` while `OutOfResources` lives in
    `triton.runtime.errors` — and this module must stay importable on a box
    with no triton at all."""
    from nova_bf import merge_triton as mt

    for name in ("CompilationError", "OutOfResources", "CompileTimeAssertionFailure"):
        exc = type(name, (RuntimeError,), {})("boom")
        assert mt.is_pre_enqueue(exc), f"{name} should be pre-enqueue"
        sub = type("Sub" + name, (type(exc),), {})("boom")
        assert mt.is_pre_enqueue(sub), f"a subclass of {name} should be pre-enqueue"

    for exc in (RuntimeError("illegal memory access"), ValueError("x"),
                MemoryError("oom"), KeyError("k")):
        assert not mt.is_pre_enqueue(exc), f"{exc!r} must NOT be pre-enqueue"
