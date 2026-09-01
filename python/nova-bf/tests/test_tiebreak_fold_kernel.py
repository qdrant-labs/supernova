"""The Triton fold kernel, against the portable fold as oracle.

Skips without a GPU, so this is only real coverage on hardware.

The fold is riskier than the pre-top-K in one specific way: it carries the
RUNNING STATE, so a wrong answer here corrupts every later slice rather than
one. The cases below are written from the semantics of what the state contains
— sentinels, under-filled rows, broadcast id vectors — rather than from
whatever a benchmark happens to generate.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import nova_bf.merge_triton as mt
from nova_bf.tiebreak import TIE_WORST, pack, sentinel_key

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or mt._fold is None,
    reason=f"needs CUDA + triton (built: {mt._fold is not None}, {mt._UNAVAILABLE})",
)
DEV = "cuda"


def _portable(sk, se, pk, pe, k):
    """What `_merge_topk`'s body does: concatenate, select, gather."""
    mk = torch.cat([sk, pk], dim=1)
    me = torch.cat([se, pe if pe.ndim == 2 else pe.unsqueeze(0).expand(sk.shape[0], -1)], dim=1)
    nk, idx = torch.topk(mk, k=k, dim=1, sorted=False)
    return nk, me.gather(1, idx)


def _same(a, b):
    return torch.equal(torch.sort(a, dim=1).values, torch.sort(b, dim=1).values)


def _check(sk, se, pk, pe, k, label):
    gk, ge = mt.fold(sk, se, pk, pe, k)
    ek, ee = _portable(sk, se, pk, pe, k)
    assert _same(gk, ek), f"{label}: wrong surviving keys"
    # Ids must travel with their own key AND stay in their own query's row, so
    # compare (key, id) pairs ROW BY ROW. Flattening the grid first would pass
    # even if the kernel read the right values out of the WRONG row — and every
    # pointer in it is `base + row * stride + col`, so a row mix-up is precisely
    # what this kernel can get wrong. (It is what the transposed-`part_key` bug
    # actually did.) The fixtures give each row distinct ids for the same
    # reason; see `_rand_state`.
    assert gk.shape == ge.shape and gk.shape[0] == sk.shape[0]
    for r in range(gk.shape[0]):
        got = sorted(zip(gk[r].tolist(), ge[r].tolist()))
        want = sorted(zip(ek[r].tolist(), ee[r].tolist()))
        assert got == want, f"{label}: row {r}: an id did not travel with its key"


def _rand_state(n_q, k, seed=0, sentinels=0):
    """A state with `sentinels` trailing sentinel columns, like an under-filled
    row partway through a run."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.randn(n_q, k, generator=g, device=DEV)
    ordinal = torch.arange(k, dtype=torch.int64, device=DEV)
    sk = pack(scores, ordinal)
    # Row-DISTINCT ids. `arange(k).expand(n_q, k)` would hand every query the
    # same id vector, and then a kernel that read enc from the wrong ROW would
    # still produce the right ids and pass — the one bug class these row-strided
    # pointers are most able to produce. Give each row its own id space so a
    # row mix-up cannot hide.
    se = (
        torch.arange(n_q, dtype=torch.int64, device=DEV)[:, None] * 1_000_000
        + torch.arange(1, k + 1, dtype=torch.int64, device=DEV)[None, :]
    )
    if sentinels:
        sk[:, k - sentinels:] = sentinel_key((n_q, sentinels), DEV)
        se[:, k - sentinels:] = 0          # sentinels carry a zero id
    return sk.contiguous(), se.contiguous()


def _rand_part(n_q, w, seed=1, offset=10_000, scale=1.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.randn(n_q, w, generator=g, device=DEV) * scale
    ordinal = offset + torch.arange(w, dtype=torch.int64, device=DEV)
    return pack(scores, ordinal).contiguous(), (offset + torch.arange(w, dtype=torch.int64, device=DEV))


@pytest.mark.parametrize("k,w", [(1000, 1000), (1000, 300), (64, 64), (1, 1), (500, 1500)])
def test_matches_the_portable_fold(k, w):
    n_q = 128
    sk, se = _rand_state(n_q, k)
    pk, pe = _rand_part(n_q, w)
    _check(sk, se, pk, pe, k, f"k={k} w={w}")


@pytest.mark.parametrize("sentinels", [1, 500, 999, 1000])
def test_a_state_full_of_sentinels(sentinels):
    """An under-filled row: the cut lands among sentinels, which share BOTH key
    and id. Descent 2's distinct-low argument does not hold there, so `keep` can
    exceed k and the store mask truncates — this pins that the result still
    matches the portable fold."""
    k, n_q = 1000, 64
    sk, se = _rand_state(n_q, k, sentinels=sentinels)
    pk, pe = _rand_part(n_q, 400)
    _check(sk, se, pk, pe, k, f"sentinels={sentinels}")


def test_an_entirely_sentinel_state_with_a_tiny_part():
    """The very first fold of a run: nothing real in the state yet."""
    k, n_q, w = 1000, 32, 5
    sk = sentinel_key((n_q, k), DEV)
    se = torch.zeros((n_q, k), dtype=torch.int64, device=DEV)
    pk, pe = _rand_part(n_q, w)
    _check(sk, se, pk, pe, k, "all-sentinel state")


def test_a_broadcast_1d_part_id_vector():
    """A pre-top-K'd part carries a 1-D id vector shared by every query row; the
    kernel broadcasts it with stride 0 rather than materializing n_q copies."""
    k, n_q, w = 256, 48, 256
    sk, se = _rand_state(n_q, k)
    pk, _ = _rand_part(n_q, w)
    pe_1d = 10_000 + torch.arange(w, dtype=torch.int64, device=DEV)
    _check(sk, se, pk, pe_1d, k, "1-D part enc")


def test_parts_that_displace_nothing_and_everything():
    """The two extremes of a fold: a part beaten by the whole state, and one
    that beats all of it."""
    k, n_q, w = 512, 64, 512
    sk, se = _rand_state(n_q, k)
    weak_k = pack(torch.full((n_q, w), -1e30, device=DEV),
                  99_000 + torch.arange(w, dtype=torch.int64, device=DEV))
    strong_k = pack(torch.full((n_q, w), 1e30, device=DEV),
                    99_000 + torch.arange(w, dtype=torch.int64, device=DEV))
    pe = 99_000 + torch.arange(w, dtype=torch.int64, device=DEV)
    for lbl, pk in (("displaces nothing", weak_k.contiguous()),
                    ("displaces everything", strong_k.contiguous())):
        _check(sk, se, pk, pe, k, lbl)


@pytest.mark.parametrize("seed", range(25))
def test_fuzz_against_the_portable_fold(seed):
    r = np.random.default_rng(seed)
    n_q = int(r.integers(1, 24))
    k = int(r.integers(1, 400))
    w = int(r.integers(1, 400))
    sents = int(r.integers(0, k + 1))
    sk, se = _rand_state(n_q, k, seed=seed, sentinels=sents)
    # a narrow score alphabet so exact ties across state and part are common
    g = torch.Generator(device=DEV).manual_seed(seed + 77)
    vals = torch.tensor(r.choice([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf],
                                 size=max(1, int(r.integers(1, 4)))).astype(np.float32), device=DEV)
    idx = torch.randint(0, len(vals), (n_q, w), generator=g, device=DEV)
    pk = pack(vals[idx], 50_000 + torch.arange(w, dtype=torch.int64, device=DEV)).contiguous()
    pe = 50_000 + torch.arange(w, dtype=torch.int64, device=DEV)
    _check(sk, se, pk, pe, k, f"fuzz seed={seed}")


def test_the_gate_rejects_what_the_kernel_cannot_serve():
    k, n_q, w = 64, 8, 64
    sk, se = _rand_state(n_q, k)
    pk, pe = _rand_part(n_q, w)
    assert mt.available(sk, se, pk, pe, k)
    assert not mt.available(sk.cpu(), se, pk, pe, k), "CPU state"
    assert not mt.available(sk.float(), se, pk, pe, k), "non-int64 key"
    assert not mt.available(sk, se, pk, pe.float(), k), "non-int64 part enc"
    assert not mt.available(sk, se, pk, pe, k + 1), "k does not match state width"
    assert not mt.available(sk[:0], se[:0], pk[:0], pe, k), "n_q == 0"
    wide = torch.zeros((n_q, mt.MAX_BLOCK), dtype=torch.int64, device=DEV)
    assert not mt.available(sk, se, wide, pe[:1].expand(mt.MAX_BLOCK), k), "k + w over MAX_BLOCK"


# --------------------------------------------------------------------------
# the property the fold exists for
# --------------------------------------------------------------------------
#
# `_merge_topk`'s docstring claims the result is "independent of slice
# boundaries, candidate grouping, and flush timing". That is the whole reason
# the amortization is allowed to exist, and it is a stronger claim than
# "matches the portable fold on one call" — it has to hold across a CHAIN of
# folds, in any grouping. These test that directly.


def _one_shot(sk, se, parts, k):
    """The answer with no folding at all: select once over everything."""
    mk = torch.cat([sk] + [p for p, _ in parts], dim=1)
    me = torch.cat(
        [se] + [e if e.ndim == 2 else e.unsqueeze(0).expand(sk.shape[0], -1) for p, e in parts],
        dim=1,
    )
    nk, idx = torch.topk(mk, k=k, dim=1, sorted=False)
    return nk, me.gather(1, idx)


def _pairs(kk, ee):
    return sorted(zip(kk.flatten().tolist(), ee.flatten().tolist()))


@pytest.mark.parametrize("n_parts", [2, 3, 8])
def test_a_chain_of_folds_equals_one_big_selection(n_parts):
    """Fold parts in one at a time; the result must equal selecting over the
    concatenation of everything at once. This is what makes flush timing an
    implementation detail rather than part of the answer."""
    k, n_q, w = 200, 32, 150
    sk, se = _rand_state(n_q, k, seed=5, sentinels=60)
    parts = [_rand_part(n_q, w, seed=100 + i, offset=10_000 * (i + 1)) for i in range(n_parts)]
    ck, ce = sk, se
    for pk, pe in parts:
        ck, ce = mt.fold(ck, ce, pk, pe, k)
    ek, ee = _one_shot(sk, se, parts, k)
    assert _pairs(ck, ce) == _pairs(ek, ee), "chained folds != one-shot selection"


def test_grouping_of_parts_does_not_change_the_answer():
    """Same parts, different flush boundaries: [a][b][c] vs [ab][c] vs [abc].
    All three must agree, or `pending_cols >= k` would be part of the result."""
    k, n_q, w = 128, 24, 100
    sk, se = _rand_state(n_q, k, seed=9, sentinels=30)
    parts = [_rand_part(n_q, w, seed=200 + i, offset=10_000 * (i + 1)) for i in range(3)]

    def run(groups):
        ck, ce = sk, se
        for grp in groups:
            pk = torch.cat([parts[i][0] for i in grp], dim=1)
            pe = torch.cat([parts[i][1].unsqueeze(0).expand(n_q, -1) for i in grp], dim=1)
            ck, ce = mt.fold(ck, ce, pk, pe, k)
        return _pairs(ck, ce)

    a = run([[0], [1], [2]])
    b = run([[0, 1], [2]])
    c = run([[0, 1, 2]])
    assert a == b == c, "the answer depends on how parts were grouped"


def test_mixed_convergence_within_one_launch():
    """The early-out is per ROW, so a single launch can have some programs take
    it and others run both descents. Half the rows here get a part that cannot
    displace anything; the other half get one that displaces everything."""
    k, n_q, w = 256, 64, 256
    sk, se = _rand_state(n_q, k, seed=11)
    lo = torch.full((n_q // 2, w), -1e30, device=DEV)
    hi = torch.full((n_q - n_q // 2, w), 1e30, device=DEV)
    ordinal = 77_000 + torch.arange(w, dtype=torch.int64, device=DEV)
    pk = pack(torch.cat([lo, hi], dim=0), ordinal).contiguous()
    _check(sk, se, pk, ordinal, k, "mixed convergence")


@pytest.mark.parametrize("fill", ["-inf", "+inf", "nan", "zeros", "-0.0"])
def test_degenerate_score_fills(fill):
    """Whole-matrix fills of the values that have bitten this code before."""
    k, n_q, w = 128, 16, 128
    v = {"-inf": float("-inf"), "+inf": float("inf"), "nan": float("nan"),
         "zeros": 0.0, "-0.0": -0.0}[fill]
    sk, se = _rand_state(n_q, k, seed=13)
    pk = pack(torch.full((n_q, w), v, device=DEV),
              88_000 + torch.arange(w, dtype=torch.int64, device=DEV)).contiguous()
    pe = 88_000 + torch.arange(w, dtype=torch.int64, device=DEV)
    _check(sk, se, pk, pe, k, f"part all {fill}")


def test_state_and_part_sharing_identical_keys():
    """Real rows cannot collide (disjoint slices, unique ordinals), but the
    kernel should not depend on that: identical keys must still yield the
    portable answer, and every surviving id must belong to a surviving key."""
    k, n_q, w = 64, 8, 64
    sk, se = _rand_state(n_q, k, seed=17)
    pk = sk.clone()                                  # byte-identical keys
    pe = torch.full((n_q, w), -1, dtype=torch.int64, device=DEV)
    gk, ge = mt.fold(sk, se, pk, pe, k)
    ek, _ = _portable(sk, se, pk, pe, k)
    assert _same(gk, ek)
    allowed = set(se.flatten().tolist()) | {-1}
    assert set(ge.flatten().tolist()) <= allowed


def test_ordinals_at_the_edges_of_the_packed_range():
    """The low half is `0xFFFFFFFF - ordinal`; 0 and TIE_WORST are its edges,
    and TIE_WORST is the sentinel value that makes the low half zero."""
    k, n_q, w = 128, 16, 128
    scores = torch.zeros(n_q, k, device=DEV)          # everything ties on score
    sk = pack(scores, torch.arange(k, dtype=torch.int64, device=DEV)).contiguous()
    se = torch.arange(k, dtype=torch.int64, device=DEV).expand(n_q, k).contiguous()
    for lbl, od in (
        ("ordinal 0..w", torch.arange(w, dtype=torch.int64, device=DEV)),
        ("ordinal at TIE_WORST", torch.full((w,), TIE_WORST, dtype=torch.int64, device=DEV)),
        ("ordinal near TIE_WORST", TIE_WORST - torch.arange(w, dtype=torch.int64, device=DEV)),
    ):
        pk = pack(torch.zeros(n_q, w, device=DEV), od).contiguous()
        _check(sk, se, pk, od, k, lbl)


def test_repeated_folds_never_lose_a_real_hit_to_a_sentinel():
    """A real candidate must always outrank a sentinel, however many folds it
    survives — the sentinel is `pack(-inf, TIE_WORST)` and every real row has a
    smaller ordinal, so this is the invariant the initial state depends on."""
    from nova_bf.tiebreak import unpack_score

    k, n_q, w = 500, 16, 50
    sk = sentinel_key((n_q, k), DEV)
    se = torch.zeros((n_q, k), dtype=torch.int64, device=DEV)
    real_ids = set()
    for i in range(6):
        pk, pe = _rand_part(n_q, w, seed=300 + i, offset=1_000 * (i + 1))
        real_ids |= set(pe.tolist())
        sk, se = mt.fold(sk, se, pk, pe, k)
    kept = unpack_score(sk) > float("-inf")
    # EVERY row, not just row 0: each Triton program handles one query row, so a
    # bug that strands one row's state leaves the others perfect. Checking row 0
    # alone is the shape of assertion that lets that through.
    per_row = kept.sum(dim=1).tolist()
    assert per_row == [6 * w] * n_q, (
        f"real hits were displaced by sentinels: kept per row {per_row}, want {6 * w}"
    )
    for r in range(n_q):
        assert set(se[r][kept[r]].tolist()) == real_ids, f"row {r}: a real id went missing"


# --------------------------------------------------------------------------
# does the REAL pipeline reach the fold kernel, and is the artifact identical?
# --------------------------------------------------------------------------


def _tiny_corpus(tmp_path, n_files=3, per_file=900, dim=8, seed=0):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    cdir = tmp_path / "c"
    cdir.mkdir(exist_ok=True)
    for f in range(n_files):
        pq.write_table(
            pa.table({
                "dense_embedding": pa.array(
                    rng.standard_normal((per_file, dim)).astype(np.float32).tolist(),
                    pa.list_(pa.float32()),
                ),
                "sid": pa.array([f"f{f}_r{i}" for i in range(per_file)]),
            }),
            str(cdir / f"f{f}.parquet"),
        )
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array(
                rng.standard_normal((24, dim)).astype(np.float32).tolist(),
                pa.list_(pa.float32()),
            ),
            "qid": pa.array([f"q{i}" for i in range(24)]),
        }),
        str(tmp_path / "q.parquet"),
    )
    return cdir


def _cfg(cdir, tmp_path, out, tiebreak, k=64, batch=512):
    from nova_bf.config import (
        BruteForceConfig, CorpusConfig, OutputConfig, ParamsConfig,
        QueriesConfig, SearchSpec,
    )
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="sid"),
        queries=QueriesConfig(path=str(tmp_path / "q.parquet"), id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, tiebreak=tiebreak, dense_batch_size=batch),
        searches=[SearchSpec(name="t", k=k, metric="dot")],
    )


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
def test_run_compute_reaches_the_fold_kernel(tmp_path, tiebreak, monkeypatch):
    """`available()` guards a performance path, so a gate that wrongly rejects
    is invisible — the run just gets slower. Assert the real pipeline reaches
    the fold in BOTH modes and is never declined."""
    import pyarrow.parquet as pq

    import nova_bf.merge_triton as mtm
    from nova_bf.compute import run_compute

    seen = {"calls": 0, "declined": 0, "why": None}
    real_avail, real_fold = mtm.available, mtm.fold

    def spy_avail(sk, se, pk, pe, k, live=None, thr=None):
        ok = real_avail(sk, se, pk, pe, k, live, thr)
        if not ok:
            seen["declined"] += 1
            seen["why"] = dict(
                sk=tuple(sk.shape), pk=tuple(pk.shape), k=k,
                pe_dim=pe.ndim, dtypes=(str(sk.dtype), str(pk.dtype), str(pe.dtype)),
                cuda=(sk.is_cuda, pk.is_cuda, pe.is_cuda),
            )
        return ok

    def spy_fold(*a, **kw):
        seen["calls"] += 1
        return real_fold(*a, **kw)

    monkeypatch.setattr(mtm, "available", spy_avail)
    monkeypatch.setattr(mtm, "fold", spy_fold)

    cdir = _tiny_corpus(tmp_path)
    out = tmp_path / f"o-{tiebreak}"
    out.mkdir()
    run_compute(_cfg(cdir, tmp_path, out, tiebreak))
    assert seen["calls"] > 0, (
        f"tiebreak={tiebreak}: fold kernel never reached "
        f"(declined={seen['declined']}, first={seen['why']})"
    )
    assert seen["declined"] == 0, f"gate declined {seen['declined']}: {seen['why']}"


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
def test_the_artifact_is_identical_with_and_without_the_fold_kernel(tmp_path, tiebreak, monkeypatch):
    """The strongest end-to-end check: a full ground-truth artifact produced
    with the fold kernel must be byte-identical to one produced by the portable
    fold. If these ever differ, a GT computed on a CUDA box differs from one
    computed without — the machine-dependent answer this feature exists to
    remove."""
    import pyarrow.parquet as pq

    import nova_bf.merge_triton as mtm
    from nova_bf.compute import run_compute

    cdir = _tiny_corpus(tmp_path, seed=3)

    out_a = tmp_path / f"kern-{tiebreak}"
    out_a.mkdir()
    a = pq.read_table(run_compute(_cfg(cdir, tmp_path, out_a, tiebreak))["t"]).to_pydict()

    monkeypatch.setattr(mtm, "available", lambda *a, **k: False)
    out_b = tmp_path / f"port-{tiebreak}"
    out_b.mkdir()
    b = pq.read_table(run_compute(_cfg(cdir, tmp_path, out_b, tiebreak))["t"]).to_pydict()

    assert a["hit_ids"] == b["hit_ids"], "fold kernel changed which hits were kept"
    assert a["hit_scores"] == b["hit_scores"], "fold kernel changed the scores"


def test_the_artifact_survives_every_batch_size(tmp_path, monkeypatch):
    """Batch size changes how many folds happen and how wide each part is. The
    artifact must not move — that independence is the point of the feature."""
    import pyarrow.parquet as pq

    from nova_bf.compute import run_compute

    cdir = _tiny_corpus(tmp_path, seed=7)
    ref = None
    for batch in (128, 300, 512, 1024):
        out = tmp_path / f"b{batch}"
        out.mkdir()
        got = pq.read_table(
            run_compute(_cfg(cdir, tmp_path, out, "id", batch=batch))["t"]
        ).to_pydict()
        if ref is None:
            ref = got
        else:
            assert got["hit_ids"] == ref["hit_ids"], f"batch={batch} changed the answer"
            assert got["hit_scores"] == ref["hit_scores"], f"batch={batch} changed the scores"


def test_a_real_candidate_is_never_displaced_by_sentinels():
    """The exact shape that broke the first fold kernel: a state that is almost
    all sentinels plus one real hit, and a part carrying one more real hit. The
    cut lands among sentinels, whose low halves are all zero, so the tie
    descent cannot separate them — and an over-selection there truncates the
    PART, because the state holds the low lanes. Both real hits must survive."""
    from nova_bf.tiebreak import unpack_score

    k, n_q = 1000, 8
    sk = sentinel_key((n_q, k), DEV)
    se = torch.zeros((n_q, k), dtype=torch.int64, device=DEV)
    sk[:, -1] = pack(torch.full((n_q, 1), 5.0, device=DEV),
                     torch.tensor([7], dtype=torch.int64, device=DEV))[:, 0]
    se[:, -1] = 7
    pk = pack(torch.full((n_q, 1), 3.0, device=DEV),
              torch.tensor([9], dtype=torch.int64, device=DEV))
    pe = torch.tensor([9], dtype=torch.int64, device=DEV)
    gk, ge = mt.fold(sk, se, pk.contiguous(), pe, k)
    real = unpack_score(gk) > float("-inf")
    assert int(real.sum(dim=1)[0]) == 2, "a real hit was displaced by sentinels"
    assert set(ge[real].tolist()) == {7, 9}


def test_a_transposed_part_key_is_never_read_with_the_wrong_stride():
    """The bug that broke tiling invariance on sparse corpora.

    Sparse scoring returns a transposed score matrix and `pack` propagates the
    stride pattern, so `part_key` arrives with stride (1, n_q). The kernel
    indexes `ptr + row * row_stride + col`, which assumes a column stride of 1
    — a transposed tensor makes it read ACROSS ROWS, returning keys that are not
    in that row's input at all. `available()` must decline it."""
    k, n_q, w = 3, 4, 3
    sk, se = _rand_state(n_q, k)
    pk_c, pe = _rand_part(n_q, w)
    pk_t = pk_c.t().contiguous().t()          # same values, transposed strides
    assert not pk_t.is_contiguous() and pk_t.stride() == (1, n_q)
    assert torch.equal(pk_t, pk_c), "fixture must differ only in layout"

    assert mt.available(sk, se, pk_c, pe, k), "the contiguous form should be served"
    assert not mt.available(sk, se, pk_t, pe, k), "the transposed form must decline"

    # and once made contiguous, it is served AND correct
    _check(sk, se, pk_t.contiguous(), pe, k, "transposed -> contiguous")


def test_transposed_inputs_of_every_kind_are_declined():
    k, n_q, w = 8, 6, 8
    sk, se = _rand_state(n_q, k)
    pk, pe = _rand_part(n_q, w)
    tr = lambda t: t.t().contiguous().t()
    assert mt.available(sk, se, pk, pe, k)
    assert not mt.available(tr(sk), se, pk, pe, k), "transposed state key"
    assert not mt.available(sk, tr(se), pk, pe, k), "transposed state enc"
    assert not mt.available(sk, se, tr(pk), pe, k), "transposed part key"
