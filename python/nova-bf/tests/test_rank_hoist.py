"""`rank_of` is shared across a slice's members, so it has to be right once.

`topk_triton._cutfill`'s tie-break descent ranks the ordinals, and that rank
depends only on the COLUMNS — every member of a slice was computing the same
argsort and scatter over again. Hoisting it means the value is now supplied by
the caller, and a wrong one breaks only tie-breaking among exactly-equal
scores: the results stay plausible and nothing raises. These pin both halves —
that the hoisted value is what the kernel used to compute, and that a
mismatched one is refused rather than used.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nova_bf import topk_triton
from nova_bf.tiebreak import pack_topk

CUDA = torch.cuda.is_available()


def _reference_rank(ordinal):
    perm = torch.argsort(ordinal)
    rank = torch.empty(ordinal.numel(), dtype=torch.int32, device=ordinal.device)
    rank.scatter_(0, perm, torch.arange(ordinal.numel(), dtype=torch.int32,
                                        device=ordinal.device))
    return rank


@pytest.mark.parametrize("n", [1, 2, 17, 512])
def test_rank_of_matches_the_argsort_scatter_it_replaced(n):
    g = torch.Generator().manual_seed(n)
    ordinal = torch.randperm(n, generator=g).to(torch.int64)
    assert torch.equal(topk_triton.rank_of(ordinal), _reference_rank(ordinal))


def test_rank_of_is_a_dense_permutation_even_with_gaps():
    """Ordinals are worker-local ranks, not positions: `tiebreak='id'` hands
    over arbitrary uint32 values with gaps. The rank has to be dense in
    `[0, n)` regardless, because `_cutfill` shifts it into `[1, n_cols]`."""
    ordinal = torch.tensor([900, 3, 77, 4_000_000_000, 5], dtype=torch.int64)
    rank = topk_triton.rank_of(ordinal)
    assert sorted(rank.tolist()) == list(range(5))
    assert rank.tolist() == [3, 0, 2, 4, 1]


@pytest.mark.skipif(not CUDA, reason="the kernel path needs CUDA")
@pytest.mark.parametrize("bad", ["short", "wrong_dtype", "strided"])
def test_a_mismatched_rank_is_refused(bad):
    dev = "cuda"
    n_q, n_cols, k = 4, 64, 8
    scores = torch.randn(n_q, n_cols, device=dev)
    ordinal = torch.randperm(n_cols, device=dev).to(torch.int64)
    good = topk_triton.rank_of(ordinal)
    if bad == "short":
        rank = good[: n_cols // 2].contiguous()
    elif bad == "wrong_dtype":
        rank = good.to(torch.int64)
    else:
        rank = torch.empty(n_cols * 2, dtype=torch.int32, device=dev)[::2]
    with pytest.raises(ValueError, match="rank must be int32"):
        topk_triton.topk(scores, ordinal, k, rank=rank)


@pytest.mark.skipif(not CUDA, reason="the kernel path needs CUDA")
def test_passing_the_rank_changes_nothing_about_the_answer():
    dev = "cuda"
    n_q, n_cols, k = 32, 256, 10
    torch.manual_seed(3)
    # Heavy ties: without them the rank never decides anything and this test
    # would pass with the argument ignored.
    scores = torch.randint(0, 4, (n_q, n_cols), device=dev).float()
    ordinal = torch.randperm(n_cols, device=dev).to(torch.int64)
    enc = torch.arange(n_cols, dtype=torch.int64, device=dev) + 7
    a = pack_topk(scores, ordinal, k, encoded=enc)
    b = pack_topk(scores, ordinal, k, encoded=enc,
                  rank=topk_triton.rank_of(ordinal))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_rank_for_never_serves_a_stale_rank_after_an_id_is_reused():
    """The per-slice cache is keyed on `id()`, and CPython reuses an id as soon
    as the object is freed. Retaining the tensor in the value prevents that,
    and re-checking identity on a hit is what makes it impossible rather than
    unlikely — so this fails if the entry is ever reduced to a bare rank."""
    import torch

    from nova_bf.compute import _rank_for

    cache = {}
    base = torch.arange(64, dtype=torch.int64)

    a = base.flip(0).contiguous()
    ra = _rank_for(cache, a)
    assert torch.equal(ra, topk_triton.rank_of(a))
    assert _rank_for(cache, a) is ra, "a genuine repeat should hit the cache"

    # Forge the collision the retention exists to prevent: plant an entry whose
    # key is some OTHER live tensor's id but whose retained tensor is not it.
    b = base.clone()
    cache[id(b)] = (a, ra)
    rb = _rank_for(cache, b)
    assert torch.equal(rb, topk_triton.rank_of(b)), (
        "served a rank belonging to a different ordinal vector — the identity "
        "re-check is missing, and a wrong rank corrupts tie-breaking silently")
    assert not torch.equal(rb, ra), "the two vectors must not share a rank here"


def test_rank_for_computes_once_per_distinct_vector():
    """The whole point of the hoist: members of one slice share an ordinal
    vector, so `rank_of` must run once, not once per member."""
    import torch

    from nova_bf import compute

    calls = []
    real = topk_triton.rank_of

    def counting(o):
        calls.append(o.shape[0])
        return real(o)

    cache = {}
    ordinals = torch.arange(32, dtype=torch.int64).flip(0).contiguous()
    orig = topk_triton.rank_of
    topk_triton.rank_of = counting
    try:
        for _ in range(5):          # five members, one slice
            compute._rank_for(cache, ordinals)
    finally:
        topk_triton.rank_of = orig
    assert len(calls) == 1, f"rank_of ran {len(calls)} times for one vector"


def test_a_bad_rank_propagates_instead_of_disabling_the_kernel(monkeypatch):
    """`topk`'s `rank` validation says "this is a raise, not a silent
    recompute", because a wrong rank breaks tie-breaking among equal scores
    while leaving results plausible. `pack_topk`'s blanket `except Exception ->
    disable()` used to swallow it: the caller bug became a much slower run for
    the whole process, announced by a log line blaming the kernel and promising
    "results are unaffected". Only compile/launch failures may be swallowed."""
    import torch

    from nova_bf import topk_triton as tt

    scores = torch.zeros((4, 16), dtype=torch.float32)
    ordinal = torch.arange(16, dtype=torch.int64)
    bad_rank = torch.arange(16, dtype=torch.int64)      # int64, not int32

    monkeypatch.setattr(tt, "available", lambda *a, **kw: True)
    monkeypatch.setattr(tt, "topk", lambda *a, **kw: (_ for _ in ()).throw(
        ValueError("rank must be int32, contiguous, 16 long")))
    disabled = []
    monkeypatch.setattr(tt, "disable", lambda exc: disabled.append(exc))

    with pytest.raises(ValueError, match="rank must be int32"):
        pack_topk(scores, ordinal, 4, rank=bad_rank)
    assert not disabled, "a caller bug must not permanently disable the kernel"


def test_a_compile_failure_still_falls_back_quietly(monkeypatch):
    """The other side of the same gate: a genuine compile/launch failure is
    exactly what `disable` exists for, and must still degrade silently."""
    import torch

    from nova_bf import topk_triton as tt

    scores = torch.zeros((4, 16), dtype=torch.float32)
    ordinal = torch.arange(16, dtype=torch.int64)

    monkeypatch.setattr(tt, "available", lambda *a, **kw: True)
    monkeypatch.setattr(tt, "topk", lambda *a, **kw: (_ for _ in ()).throw(
        type("OutOfResources", (RuntimeError,), {})("out of resource: shared memory")))
    disabled = []
    monkeypatch.setattr(tt, "disable", lambda exc: disabled.append(exc))

    keys, values, live = pack_topk(scores, ordinal, 4)
    assert keys.shape == (4, 4), "a compile failure must degrade to the portable path"
    assert live is None, "no threshold was given, so every row stays valid"
    assert len(disabled) == 1, "and it should disable the kernel for the run"


def test_the_subset_ordinals_are_shared_so_the_rank_hoist_can_fire():
    """The hoist keys on the ordinal vector's IDENTITY, so a filtered search
    that rebuilds `ordinals[sel_cols]` per member could never hit it — it
    never hit once across a suite run before this. `sel_cols` is what
    `select()` memoizes per filter, so keying the subset on it makes every
    member of the slice see the SAME ordinal object, which is what lets
    `_rank_for` hit."""
    import torch

    from nova_bf.compute import _rank_for, _subset_for

    ordinals = torch.arange(64, dtype=torch.int64).flip(0).contiguous()
    sel_cols = torch.tensor([1, 5, 9, 30, 61], dtype=torch.int64)

    subset_cache, rank_cache = {}, {}
    first = _subset_for(subset_cache, ordinals, sel_cols)
    assert torch.equal(first, ordinals[sel_cols]), "must equal the plain subset"

    # Four more members of the same slice: same object every time.
    for _ in range(4):
        again = _subset_for(subset_cache, ordinals, sel_cols)
        assert again is first, "each member must see the SAME subset object"

    # ...which is precisely what makes the rank cache hit.
    r = _rank_for(rank_cache, first)
    for _ in range(4):
        assert _rank_for(rank_cache, _subset_for(subset_cache, ordinals, sel_cols)) is r

    # A different filter's columns must NOT share.
    other = torch.tensor([2, 3], dtype=torch.int64)
    assert _subset_for(subset_cache, ordinals, other) is not first
    assert torch.equal(_subset_for(subset_cache, ordinals, other), ordinals[other])
