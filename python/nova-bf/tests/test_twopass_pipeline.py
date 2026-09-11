"""Two-pass dense scoring must not change a single result.

`test_twopass_bound.py` pins the DECISION (no row with a real candidate is
ever called dead). This pins the PLUMBING around it: the narrowed exact GEMM,
the per-member spans into it, `_tp_scatter_part`'s full-height part with
uninitialized dead rows, and the fold that has to skip exactly those rows.

Everything here is end to end through `run_compute`, because that is the only
level at which the pieces are wired the way a run wires them — a direct
tensor-level test would re-implement `process_slice` and pin the
re-implementation instead.

The two-pass thresholds are scaled down (`PAD_QUANTUM`/`PAD_FLOOR`,
`MIN_QUERY_ROWS`) so a fixture-sized run reaches the path at all; nothing else
is changed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")
import pyarrow as pa
import pyarrow.parquet as pq

from nova_bf import compute
from nova_bf import twopass
from nova_bf.compute import run_compute
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    OutputConfig,
    ParamsConfig,
    QueriesConfig,
    SearchSpec,
)

# DIM is 64 and not 32 because the closed-form bound's P6 requires
# `64 <= d <= 2**20`: Lemma 2' needs `d >= 64` for its block count, and below
# it there is no bound to evaluate, so `upper_bounds` refuses every row and the
# two-pass never engages. A fixture under the floor would test nothing.
DIM, K, NQ = 64, 5, 64
BATCH = 64


def _write(path, vectors, **columns):
    data = {"dense_embedding": pa.array(vectors.tolist(),
                                        type=pa.list_(pa.float32()))}
    data.update({k: pa.array(v) for k, v in columns.items()})
    pq.write_table(pa.table(data), str(path))


@pytest.fixture
def ds(tmp_path):
    rng = np.random.default_rng(7)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    g = 0
    # Enough files that the running top-K fills and thresholds start pruning —
    # a two-pass that never prunes anything would pass this test vacuously,
    # which is what `slices_twopass` and the live fraction assert against.
    for fi in range(8):
        n = 200
        vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        vecs = vecs.astype(np.float16).astype(np.float32)
        _write(cdir / f"f{fi}.parquet", vecs,
               id=[f"c{g + r:06d}" for r in range(n)])
        g += n
    qpath = tmp_path / "queries.parquet"
    _write(qpath, rng.standard_normal((NQ, DIM)).astype(np.float32),
           qid=[f"q{i}" for i in range(NQ)],
           qset=["a"] * (NQ // 2) + ["b"] * (NQ - NQ // 2))
    return {"cdir": str(cdir), "qpath": str(qpath)}


def _cfg(ds, out, metric="cosine", tiebreak="ordinal") -> BruteForceConfig:
    return BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid",
                              payload_fields=["qset"]),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, dense_batch_size=BATCH,
                            tiebreak=tiebreak),
        searches=[
            # One search over every query row and one over a contiguous
            # subset: the two-pass has to map a member's live rows back into
            # its OWN state height, and a `qsel` of `None` and a `slice` take
            # different branches doing it.
            SearchSpec(name="all", metric=metric, k=K),
            SearchSpec(name="half", metric=metric, k=K,
                       rows={"column": "qset", "isin": ["b"]}),
        ],
    )


@pytest.fixture(autouse=True)
def _small_twopass(monkeypatch):
    """Scale the two-pass down to fixture size.

    `PAD_FLOOR` / `PAD_QUANTUM` are the GEMM heights that keep cuBLAS on one
    kernel; `MIN_QUERY_ROWS` is the height below which pass one cannot pay for
    itself. Neither is a correctness parameter.
    """
    monkeypatch.setattr(twopass, "PAD_QUANTUM", 8)
    monkeypatch.setattr(twopass, "PAD_FLOOR", 8)
    monkeypatch.setattr(twopass, "PAD_SMALL", ())
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 16)
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
    # `_verify_shape` stays ON. It is the guard the byte-identity of these
    # runs actually rests on: `_scores` is NOT height-invariant in general
    # (on CPU at realistic shapes a narrowed GEMM disagrees with the
    # full-height one in a large fraction of random triples), so a fixture
    # that skipped the check would be asserting identity it had not earned.
    # It is cheap at this size — one extra full-height GEMM per shape.
    monkeypatch.delenv("NOVA_BF_TWOPASS_NO_VERIFY", raising=False)
    twopass.reset()
    yield
    twopass.reset()


def test_the_hint_is_a_fraction_and_does_not_scale_with_member_count(
    ds, tmp_path, monkeypatch
):
    """`_TP_LIVE_HINT` must be a live FRACTION of the query matrix.
    """
    from nova_bf import compute as compute_mod

    seen = {}
    for members in (1, 2, 3, 4):
        out = tmp_path / f"hint{members}"
        out.mkdir(parents=True, exist_ok=True)
        cfg = BruteForceConfig(
            corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
            queries=QueriesConfig(path=ds["qpath"], id_column="qid",
                                  payload_fields=["qset"]),
            output=OutputConfig(path=str(out)),
            params=ParamsConfig(io_workers=1, dense_batch_size=BATCH),
            searches=[SearchSpec(name=f"s{i}", metric="cosine", k=K)
                      for i in range(members)],
        )
        run_compute(cfg)
        # `_TP_LIVE_HINT` is cleared at the START of a run, so it survives for
        # inspection afterwards.
        hints = list(compute_mod._TP_LIVE_HINT.values())
        assert hints, f"{members} members: the probe never seeded a hint"
        for h in hints:
            assert 0.0 <= h <= 1.0, (
                f"{members} members: hint {h} is not a fraction — the probe is "
                f"summing per-member masks instead of unioning them")
        seen[members] = max(hints)

    base = seen[1]
    for members, got in seen.items():
        assert got == pytest.approx(base, abs=1e-12), (
            f"the hint moved from {base} to {got} when the same search was "
            f"repeated {members} times; identical members share one live mask, "
            f"so the union — and the fraction — must be unchanged")


def _tables(out, names=("all", "half")):
    got = {}
    for name in names:
        hits = list((out).rglob(f"*_{name}_k{K}/rank000.parquet"))
        assert len(hits) == 1, f"{name}: {hits}"
        got[name] = pq.read_table(hits[0])
    return got


def _run(ds, out, monkeypatch, *, twopass_on: bool, metric="cosine",
         tiebreak="ordinal"):
    out.mkdir(parents=True, exist_ok=True)
    if twopass_on:
        monkeypatch.delenv("NOVA_BF_NO_TWOPASS", raising=False)
    else:
        monkeypatch.setenv("NOVA_BF_NO_TWOPASS", "1")
    twopass.reset()
    run_compute(_cfg(ds, out, metric, tiebreak), num_jobs=1, job_rank=0)
    return _tables(out), twopass.stats()


@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_results_are_identical_with_and_without_the_two_pass(
    ds, tmp_path, monkeypatch, metric
):
    off, _ = _run(ds, tmp_path / "off", monkeypatch, twopass_on=False,
                  metric=metric)
    on, stats = _run(ds, tmp_path / "on", monkeypatch, twopass_on=True,
                     metric=metric)
    assert stats["slices_twopass"] > 0, (
        "the two-pass never engaged, so this test proved nothing: "
        f"{stats}"
    )
    assert stats["rows_live"] < stats["rows_full"], (
        "the two-pass engaged but pruned nothing, so the narrowed GEMM path "
        f"was never exercised: {stats}"
    )
    for name in off:
        assert on[name].equals(off[name]), f"{name} differs"


def test_identical_with_dead_rows_poisoned(ds, tmp_path, monkeypatch):
    """`_tp_scatter_part` leaves dead rows UNINITIALIZED and marks them 0 in
    `live` — the contract `topk_triton._cutfill` has had since G1. Filling
    them with a key engineered to WIN any selection it leaks into is how that
    contract is proven rather than assumed."""
    off, _ = _run(ds, tmp_path / "off", monkeypatch, twopass_on=False)
    monkeypatch.setenv("NOVA_BF_POISON_DEAD_ROWS", "1")
    on, stats = _run(ds, tmp_path / "on", monkeypatch, twopass_on=True)
    assert stats["slices_twopass"] > 0
    for name in off:
        assert on[name].equals(off[name]), f"{name} differs under poisoning"


def test_a_failed_shape_verification_disables_the_feature(
    ds, tmp_path, monkeypatch
):
    """The safeguard, not the fast path: if an M-row GEMM cannot be shown
    bit-identical to the full-height one, the run must fall back rather than
    quietly shift a score."""
    monkeypatch.setattr(twopass, "_verify_shape",
                        lambda *a, **kw: False)
    out = tmp_path / "fail"
    out.mkdir()
    twopass.reset()
    run_compute(_cfg(ds, out), num_jobs=1, job_rank=0)
    assert not twopass.enabled()
    assert twopass.stats()["unavailable"]
    # And the results are still the one-pass results.
    ref, _ = _run(ds, tmp_path / "ref", monkeypatch, twopass_on=False)
    got = _tables(out)
    for name in ref:
        assert got[name].equals(ref[name])


def _cfg_union(ds, out, k_sub, k_all):
    """A row-subsetted member FIRST, a whole-matrix member second.

    Both halves of the live-mask union need their own arrangement to be
    observable, and this is the one for the `qsel is None` half. `_cfg` puts
    the whole-matrix member first, where `live = ...` and `live |= ...` are
    indistinguishable because the mask it writes into is still all-zero.

    The k values are the other half of the setup: the whole-matrix member
    needs FEWER live rows than the subsetted one, so it must have the higher
    running threshold, so it must have the SMALLER k. With both members on
    the same k their live masks coincide and an overwrite is again invisible
    — which is why `test_a_row_subsetted_search_keeps_its_own_live_rows` does
    not cover this despite saying it does.
    """
    return BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid",
                              payload_fields=["qset"]),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, dense_batch_size=BATCH),
        searches=[
            SearchSpec(name="sub", metric="cosine", k=k_sub,
                       rows={"column": "qset", "isin": ["b"]}),
            SearchSpec(name="whole", metric="cosine", k=k_all),
        ],
    )


def test_a_whole_matrix_member_unions_its_live_rows_instead_of_replacing_them(
    ds, tmp_path, monkeypatch
):
    """`live |= ~(upper < thr_s)`, not `live = ~(upper < thr_s)`.

    The two members of a score group share one exact GEMM, so the rows it
    computes must be the UNION of what each member needs. The `qsel is None`
    branch writes the whole mask at once, so a plain `=` there does not just
    add nothing — it DISCARDS every row an earlier member asked for, and
    those rows are then never scored. Silent lost ground truth, no exception,
    no counter that moves.

    Both existing tests miss it, in two different ways, and both ways are
    properties of their fixtures rather than of the code:

      - `test_members_with_different_k_union_their_live_rows` says in its own
        docstring that it covers the `live[qsel] |=` branch. Its members are
        all row-subsetted, so the branch under test here never runs.
      - `test_a_row_subsetted_search_keeps_its_own_live_rows` in the parity
        suite claims a member "losing rows live only for the sibling" shows
        up. Its three members all share one k, so their live masks are equal
        and the overwrite is a no-op.

    So this arranges the two things needed to see it: the whole-matrix member
    goes SECOND (writing into a mask that is already populated) and gets the
    smaller k (so its own live set is strictly smaller than its sibling's).
    Measured with `=` substituted in: 433 of 1408 query-rows wrongly dead and
    the output table differs.
    """
    out = tmp_path / "union_on"
    out.mkdir()
    twopass.reset()
    run_compute(_cfg_union(ds, out, k_sub=25, k_all=2), num_jobs=1, job_rank=0)
    st = twopass.stats()

    # Premises. Without both of these the run never reaches the branch and
    # the comparison below would hold for the wrong reason.
    assert st["slices_twopass"] > 0, (
        f"the two-pass never ran, so no live mask was ever built: {st}")
    assert st["rows_live"] < st["rows_full"], (
        f"nothing was pruned, so every row is live and an overwrite would "
        f"restore the same mask it destroyed: {st}")

    ref = tmp_path / "union_off"
    ref.mkdir()
    monkeypatch.setenv("NOVA_BF_NO_TWOPASS", "1")
    twopass.reset()
    run_compute(_cfg_union(ds, ref, k_sub=25, k_all=2), num_jobs=1, job_rank=0)
    monkeypatch.delenv("NOVA_BF_NO_TWOPASS")

    for name in ("sub", "whole"):
        a = list(out.rglob(f"*_{name}_k*/rank000.parquet"))
        b = list(ref.rglob(f"*_{name}_k*/rank000.parquet"))
        assert len(a) == 1 and len(b) == 1, f"{name}: {a} {b}"
        assert pq.read_table(a[0]).equals(pq.read_table(b[0])), (
            f"{name} differs from the one-pass result — rows a sibling "
            f"member needed were dropped from the shared live mask")


def _cfg_narrow(ds, out, k, batch):
    """A corpus batch NARROWER than k, so the pre-top-K is skipped."""
    return BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid",
                              payload_fields=["qset"]),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, dense_batch_size=batch),
        searches=[SearchSpec(name="all", metric="cosine", k=k)],
    )


def test_a_slice_narrower_than_k_scatters_its_shared_id_vector(
    ds, tmp_path, monkeypatch
):
    """The two-pass crossed with a slice that has at most k columns.
    """
    from nova_bf import compute as compute_mod

    seen = []
    real = compute_mod._tp_scatter_part

    def watch(part_key, part_enc, live, dst, height):
        seen.append(part_enc.ndim)
        return real(part_key, part_enc, live, dst, height)

    monkeypatch.setattr(compute_mod, "_tp_scatter_part", watch)

    # `dense_batch_size` below k makes every full slice narrow; the two-pass
    # still engages because it gates on the QUERY height, not the corpus one.
    out = tmp_path / "narrow_on"
    out.mkdir()
    twopass.reset()
    run_compute(_cfg_narrow(ds, out, k=8, batch=4), num_jobs=1, job_rank=0)
    st = twopass.stats()

    assert st["slices_twopass"] > 0, (
        f"the two-pass never engaged, so `dst` was None and the scatter was "
        f"never called at all: {st}")
    assert 1 in seen, (
        f"every scattered part was {sorted(set(seen))}-dimensional — the "
        f"narrow-slice branch still has no coverage")

    monkeypatch.undo()
    ref = tmp_path / "narrow_off"
    ref.mkdir()
    monkeypatch.setenv("NOVA_BF_NO_TWOPASS", "1")
    twopass.reset()
    run_compute(_cfg_narrow(ds, ref, k=8, batch=4), num_jobs=1, job_rank=0)

    a = list(out.rglob("*_all_k8/rank000.parquet"))
    b = list(ref.rglob("*_all_k8/rank000.parquet"))
    assert len(a) == 1 and len(b) == 1, f"{a} {b}"
    assert pq.read_table(a[0]).equals(pq.read_table(b[0])), (
        "a slice narrower than k produced different results under the "
        "two-pass than under the one-pass path")


def test_a_verification_that_never_completed_does_not_disable_the_feature(
    ds, tmp_path, monkeypatch, caplog
):
    """`could not check` is not `this device disagrees`.

    The test above is the real safeguard firing. This is the case that looks
    identical from the caller and must NOT fire it: every rung of the ladder
    returned UNVERIFIED, meaning the bit-identity check ran out of memory
    rather than finding a difference.

    The two are easy to conflate and expensive to conflate. The whole ladder
    is walked inside one call, microseconds apart, and each rung allocates
    the same full-height reference — the largest allocation the two-pass ever
    makes — so a memory spike lasting a few milliseconds fails all four
    together. Folding that into `False` reached `disable()`, which is
    run-lifetime: a transient spike cost a multi-hour rank its speedup and
    stamped the manifest with "no padded GEMM height is bit-identical on this
    device", an accusation about the hardware that had never been tested.

    The contract: results unchanged (the one-pass path runs), the two-pass
    still ARMED at the end, and no false reason in the stats.
    """
    import logging

    monkeypatch.setattr(twopass, "_verify_shape",
                        lambda *a, **kw: twopass.UNVERIFIED)
    # Above the slice count of this fixture, so the streak limiter cannot
    # fire. That limiter is the OTHER half of the contract and has its own
    # test below; here it would mask the half being tested.
    monkeypatch.setattr(twopass, "MAX_UNCHECKED_SLICES", 10_000)
    out = tmp_path / "unverified"
    out.mkdir()
    twopass.reset()
    with caplog.at_level(logging.WARNING, logger="nova_bf.compute"):
        run_compute(_cfg(ds, out), num_jobs=1, job_rank=0)

    assert twopass.stats()["slices_discarded"] > 0, (
        "no slice reached the end of the ladder, so the branch under test "
        "never ran and this test proves nothing")
    assert twopass.stats()["unavailable"] is None, (
        f"a check that never completed took the two-pass down for the run: "
        f"{twopass.stats()['unavailable']!r}")
    assert twopass.enabled(), "the two-pass must stay armed for later slices"
    hits = [r for r in caplog.records if "out of memory" in r.getMessage()]
    assert hits, (
        "the operator was given no way to tell this apart from a device that "
        "really does disagree")
    assert len(hits) == 1, (
        f"{len(hits)} warnings for {twopass.stats()['slices_discarded']} "
        f"discarded slices. Memory pressure lasts, and while it lasts every "
        f"slice fails the whole ladder, so a per-slice line is a WARNING per "
        f"score group for as long as it holds")

    ref, _ = _run(ds, tmp_path / "ref_unverified", monkeypatch, twopass_on=False)
    got = _tables(out)
    for name in ref:
        assert got[name].equals(ref[name])


def test_the_query_cache_is_released_even_when_the_run_raises(
    ds, tmp_path, monkeypatch
):
    """`release()` sits in a `finally` for one reason, and it was untested.

    The sibling test covers a run that SUCCEEDS. But the argument for moving
    the call off the tail and into the scan/score `finally` is the opposite
    case: `twopass.reset()` only clears `_QCACHE` at the START of a run, so a
    run that raises used to leave the cached query matrices and their half
    copies — ~0.5 GB of device memory at the production shape — pinned until
    the next run, or for the life of the process if there was no next one.
    """
    from nova_bf import compute as compute_mod

    real = compute_mod._scores
    calls = []

    def blow_up_midway(*a, **kw):
        calls.append(1)
        if len(calls) == 6:
            raise RuntimeError("injected failure partway through the run")
        return real(*a, **kw)

    monkeypatch.setattr(compute_mod, "_scores", blow_up_midway)
    out = tmp_path / "raised"
    out.mkdir()
    twopass.reset()
    with pytest.raises(RuntimeError, match="injected failure"):
        run_compute(_cfg(ds, out), num_jobs=1, job_rank=0)

    assert len(calls) >= 6, "the injected failure never fired"
    assert not twopass._QCACHE, (
        f"{len(twopass._QCACHE)} cached query matrices survived a failed "
        f"run — at the production shape that is ~0.5 GB of device memory "
        f"pinned until the next run, or forever if there is not one")


def test_a_run_certifies_the_bound_on_its_own_data_before_pruning(ds, tmp_path):
    """The bound is checked on this machine, this data, before it is trusted.

    Every other guard checks a precondition. This one checks the CONCLUSION:
    for one real slice, no query row's exact top score exceeds its upper
    bound. It is a joint test — the dot-product bound, the norm inflation,
    the accumulation mode, the scale factors and the slack all have to be
    right on this machine for it to pass.
    """
    out = tmp_path / "certified"
    out.mkdir()
    twopass.reset()
    assert twopass.certified() is None, "a fresh run starts uncertified"
    run_compute(_cfg(ds, out), num_jobs=1, job_rank=0)
    st = twopass.stats()

    assert st["slices_twopass"] > 0, f"the two-pass never ran: {st}"
    assert st["certified"] == "", (
        f"the run pruned without certifying its bound: {st['certified']!r}")
    assert st["unavailable"] is None


def test_a_bound_that_does_not_hold_disables_the_two_pass_before_it_prunes(
    ds, tmp_path, monkeypatch
):
    """The whole point: an unsound bound must cost speed, never ground truth.

    Simulated by shrinking `eps` so the bound stops dominating the exact
    score — which is what a wrong constant, an unhonoured accumulation-mode
    flag, or a `NORM_INFLATE` that has expired at this dimension would each
    produce. The run must notice on its first slice, disable, fall back to
    the one-pass path, and still emit correct results.
    """
    real = twopass.bound

    def too_small(*a, **kw):
        return real(*a, **kw) * 1e-4

    monkeypatch.setattr(twopass, "bound", too_small)
    out = tmp_path / "bad_bound"
    out.mkdir()
    twopass.reset()
    run_compute(_cfg(ds, out), num_jobs=1, job_rank=0)
    st = twopass.stats()

    assert st["certified"], "an unsound bound certified clean"
    assert "does not hold on this machine" in st["certified"]
    assert st["unavailable"], "the two-pass was not disabled"
    assert st["slices_twopass"] == 0, (
        f"{st['slices_twopass']} slices were pruned on a bound that had "
        f"already failed certification")

    monkeypatch.undo()
    ref, _ = _run(ds, tmp_path / "ref_bad_bound", monkeypatch, twopass_on=False)
    got = _tables(out)
    for name in ref:
        assert got[name].equals(ref[name]), (
            f"{name}: falling back after a failed certification changed the "
            f"results")


def test_every_distinct_execution_configuration_is_certified_separately(
    ds, tmp_path, monkeypatch
):
    """Certification is empirical, so its evidence does not travel.

    A numerical library dispatches its kernel from the problem shape, and a
    different score group brings a different query matrix down a path the
    check never exercised. Certifying once per run and extrapolating to every
    later shape and group is exactly the inference this mechanism exists to
    avoid — and it was what the first version did, which left the safety
    claim conditional on something the run never established.

    Two configurations are forced here. `dense_batch_size` unset (permitted)
    makes the step each file's own row count, so three files of different
    sizes give three slice widths; and `_cfg` has two searches, giving two
    score groups. Each combination that actually prunes must have its own
    certification.
    """
    import numpy as np

    rng = np.random.default_rng(11)
    cdir = tmp_path / "ragged"
    cdir.mkdir()
    g = 0
    for n in (192, 224, 256, 192, 224, 256):
        v = rng.standard_normal((n, DIM)).astype(np.float32)
        v = v.astype(np.float16).astype(np.float32)
        _write(cdir / f"f{g:06d}.parquet", v,
               id=[f"c{g + r:06d}" for r in range(n)])
        g += n
    ds2 = {"cdir": str(cdir), "qpath": ds["qpath"]}

    def cfg(out):
        c = _cfg(ds2, out)
        c.params.dense_batch_size = None      # step = each file's row count
        return c

    out = tmp_path / "ragged_on"
    out.mkdir()
    twopass.reset()
    run_compute(cfg(out), num_jobs=1, job_rank=0)
    st = twopass.stats()

    assert st["certified"] == "", f"certification did not run: {st}"
    assert st["slices_twopass"] > 0, "the two-pass never engaged"
    assert st["certified_configs"] > 1, (
        f"only {st['certified_configs']} configuration was certified across "
        f"three slice widths and two score groups — evidence from one is "
        f"being extended to the others: {st}")

    ref_out = tmp_path / "ragged_off"
    ref_out.mkdir()
    monkeypatch.setenv("NOVA_BF_NO_TWOPASS", "1")
    twopass.reset()
    run_compute(cfg(ref_out), num_jobs=1, job_rank=0)
    monkeypatch.delenv("NOVA_BF_NO_TWOPASS")

    for name in ("all", "half"):
        x = list(out.rglob(f"*_{name}_k{K}/rank000.parquet"))
        y = list(ref_out.rglob(f"*_{name}_k{K}/rank000.parquet"))
        assert len(x) == 1 and len(y) == 1
        assert pq.read_table(x[0]).equals(pq.read_table(y[0])), name


def test_a_dense_batch_can_never_have_two_eligible_score_groups(ds, tmp_path):
    """Why certifying per configuration is enough on the score-group axis.

    Two facts combine. A score key is `(metric, scale_in_packer)`, so a batch
    scored by one metric collapses to ONE key. And a dense batch scored by two
    or more DISTINCT metrics has `share_gram` set — the two metrics are derived
    from a single shared raw Gram — which the two-pass refuses outright,
    because it narrows the query axis and the two cannot then share a matrix.

    So a dense batch either has one eligible group or none. If a future change
    lets two eligible groups coexist, this test fails and the coverage
    argument needs revisiting.
    """
    def build(out, searches):
        out.mkdir()
        return BruteForceConfig(
            corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
            queries=QueriesConfig(path=ds["qpath"], id_column="qid",
                                  payload_fields=["qset"]),
            output=OutputConfig(path=str(out)),
            params=ParamsConfig(io_workers=1, dense_batch_size=BATCH),
            searches=searches)

    # One metric, two searches: ONE group, two members, and it engages.
    one = tmp_path / "one_metric"
    twopass.reset()
    run_compute(build(one, [SearchSpec(name="a", metric="cosine", k=K),
                            SearchSpec(name="b", metric="cosine", k=K)]),
                num_jobs=1, job_rank=0)
    st_one = twopass.stats()
    assert st_one["slices_twopass"] > 0, f"one metric did not engage: {st_one}"
    assert st_one["certified_configs"] == 1, (
        f"one metric produced {st_one['certified_configs']} configurations; "
        f"the score key was expected to collapse to one: {st_one}")

    # Two metrics: `share_gram`, refused wholesale — so the two-pass never
    # sees two groups, and never prunes one on another's evidence.
    two = tmp_path / "two_metrics"
    twopass.reset()
    run_compute(build(two, [SearchSpec(name="cos", metric="cosine", k=K),
                            SearchSpec(name="dot", metric="dot", k=K)]),
                num_jobs=1, job_rank=0)
    st_two = twopass.stats()
    assert st_two["groups_seen"] > 0
    assert st_two["groups_refused"] == st_two["groups_seen"], (
        f"a batch with two distinct metrics was admitted; `share_gram` was "
        f"expected to refuse every one: {st_two}")
    assert st_two["slices_twopass"] == 0
    assert st_two["certified_configs"] == 0


def test_certification_actually_consults_the_accumulator_guard(
    ds, tmp_path, monkeypatch
):
    """The predicate is tested directly; this pins that it is CONSULTED.

    Spoofing a CUDA device makes torch try to use one, so the refusal itself
    cannot be driven end to end on this box. What can be: force the predicate
    to say "not ours" and require the run to decline. Without this, deleting
    the call site passes every other test.
    """
    monkeypatch.setattr(twopass, "accumulator_is_ours",
                        lambda device, used_fused: False)
    out = tmp_path / "not_ours"
    out.mkdir()
    twopass.reset()
    run_compute(_cfg(ds, out), num_jobs=1, job_rank=0)
    st = twopass.stats()

    assert st["certified"], "the guard said `not ours` and nothing declined"
    assert "cuBLAS" in st["certified"], (
        f"declined for the wrong reason: {st['certified']!r}")
    assert st["slices_twopass"] == 0, (
        f"{st['slices_twopass']} slices pruned after the accumulator guard "
        f"refused")

    monkeypatch.undo()
    ref, _ = _run(ds, tmp_path / "not_ours_ref", monkeypatch, twopass_on=False)
    got = _tables(out)
    for name in ref:
        assert got[name].equals(ref[name])


def test_the_cpu_path_is_not_refused_since_its_accumulation_is_also_ours(
    ds, tmp_path
):
    """The counterpart, so the guard cannot be written as `refuse unless
    fused` and pass.

    The CPU path has no fused kernel either, but it does not run cuBLAS: it
    widens the half inputs and multiplies in float32, in this repo's own code.
    The distinction the guard has to make is whose accumulator it is, not
    whether the fast kernel ran.
    """
    out = tmp_path / "cpu_ok"
    out.mkdir()
    twopass.reset()
    run_compute(_cfg(ds, out), num_jobs=1, job_rank=0)
    st = twopass.stats()

    assert st["certified"] == "", (
        f"the CPU path was refused, but its accumulation is ours: "
        f"{st['certified']!r}")
    assert st["slices_twopass"] > 0


def test_certification_does_not_pollute_the_slice_counters(ds, tmp_path):
    """Its pass one is overhead, not work.

    `slices_fused + slices_unfused` is how many pass ones ran for SLICES, and
    the manifest and other tests diff it against
    `slices_twopass + slices_discarded`. Certification runs one more pass one
    that belongs to no slice, so it has to be excluded or that identity breaks.
    """
    out = tmp_path / "counters"
    out.mkdir()
    twopass.reset()
    run_compute(_cfg(ds, out), num_jobs=1, job_rank=0)
    st = twopass.stats()

    assert st["certified"] == "", "premise: certification ran"
    assert st["slices_fused"] + st["slices_unfused"] == (
        st["slices_twopass"] + st["slices_discarded"]), (
        f"pass-one count does not match the slices that used one: {st}")


def test_an_eligibility_refusal_is_distinguishable_in_the_manifest(
    ds, tmp_path, monkeypatch
):
    """Three very different runs used to produce one all-zero block.

    All twelve refusals in `_twopass_groups` return `{}` without calling
    `disable()` and without moving a counter. So a run whose two-pass was
    turned away by a guard, a run whose live fraction never fell, and a run
    that had no dense slice at all were byte-identical in the manifest — and
    the manifest's own comment told the reader to interpret
    `permitted and launches == 0` as the second of the three.

    That is the difference between "your config silently disabled the
    optimisation" and "the optimisation looked and decided not to bother".
    """
    # The query-height floor, not TF32: `run_compute` sets `allow_tf32` from
    # the config on its way in, so monkeypatching the torch flag here is
    # overwritten before `_twopass_groups` ever reads it. This guard is a
    # module constant the run does not touch.
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 10 ** 9)
    out = tmp_path / "refused"
    out.mkdir()
    twopass.reset()
    run_compute(_cfg(ds, out), num_jobs=1, job_rank=0)
    refused = twopass.stats()

    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 16)
    ok = tmp_path / "eligible"
    ok.mkdir()
    twopass.reset()
    run_compute(_cfg(ds, ok), num_jobs=1, job_rank=0)
    eligible = twopass.stats()

    assert refused["groups_seen"] > 0, "the eligibility check never ran"
    assert refused["groups_refused"] == refused["groups_seen"], (
        f"a guard turned every group away but the manifest does not say so: "
        f"{refused['groups_refused']}/{refused['groups_seen']}")
    assert refused["slices_twopass"] == 0

    assert eligible["groups_refused"] == 0, (
        f"the control run was refused too, so this test cannot tell a "
        f"working signal from a broken fixture: {eligible}")
    assert eligible["slices_twopass"] > 0

    # The point of the whole exercise: the two runs are now distinguishable.
    assert refused["groups_refused"] != eligible["groups_refused"], (
        "an eligibility refusal is still indistinguishable from a run that "
        "was eligible and simply never triggered")


def test_a_pass_one_oom_does_not_fabricate_a_live_hint(ds, tmp_path, monkeypatch):
    """An all-live bound after a failed allocation is not a MEASUREMENT.

    `_TP_LIVE_HINT` is what the gate in `_twopass_groups` reads to decide the
    two-pass is worth running. When pass one cannot allocate, `upper_bounds`
    answers all-live, so `n_live == n_full` and the naive write records 1.0 —
    a fraction nothing measured. 1.0 is above every threshold, so it switches
    the two-pass off for the rest of the batch group, and the one-pass probe
    that seeds the hint is already off by then, so the group cannot recover
    until the next one.

    The oracle is "one fewer write than there were slices", NOT "no hint is
    ever 1.0": a run's first slices are legitimately all-live, because the
    top-K state is empty and every threshold is still -inf. The OOM is
    injected well after that, so a real 1.0 cannot be confused with a
    fabricated one.
    """
    import torch

    from nova_bf import compute as compute_mod

    real = twopass.upper_bounds
    calls = []
    OOM_ON = 5                      # past the naturally all-live warm-up

    def oom_on_the_fifth(Q, Cb, col_scale, row_scale, out_dtype, **kw):
        calls.append(1)
        if len(calls) == OOM_ON:
            twopass._STATS["slices_pass_one_oom"] += 1
            live = torch.full((int(Q.shape[0]),), float("inf"),
                              dtype=torch.float32, device=Q.device)
            # `with_parts` is what certification asks for; an all-live return
            # has no pass one, so the parts are the refusal and a zero bound.
            if kw.get("with_parts"):
                return live, live, torch.zeros_like(live)
            return live
        return real(Q, Cb, col_scale, row_scale, out_dtype, **kw)

    class Recording(dict):
        """Tag each write with how many pass ones have run when it happens.

        Three simpler oracles do not work. Counting writes fails because
        `_TP_LIVE_HINT` is also written by the one-pass probe that seeds it,
        so there are more writes than two-pass slices. Reading the dict at
        the end fails because a later healthy slice in the same group
        overwrites the value. And "no write is ever 1.0" fails because a
        run's first slices are LEGITIMATELY all-live — the top-K state is
        empty, so every threshold is still -inf.

        What identifies the fabrication exactly: a write of 1.0 made while
        the OOM'd slice is the most recent pass one.
        """

        writes = []

        def __setitem__(self, k, v):
            Recording.writes.append((v, len(calls)))
            super().__setitem__(k, v)

    Recording.writes = []
    monkeypatch.setattr(twopass, "upper_bounds", oom_on_the_fifth)
    monkeypatch.setattr(compute_mod, "_TP_LIVE_HINT", Recording())
    out = tmp_path / "no_fabricated_hint"
    out.mkdir()
    twopass.reset()
    run_compute(_cfg(ds, out), num_jobs=1, job_rank=0)

    assert len(calls) > OOM_ON, (
        f"only {len(calls)} slices reached pass one, so the injected OOM "
        f"was never the non-warm-up case this test is about")
    assert Recording.writes, "no hint was ever recorded"
    fabricated = [v for v, n in Recording.writes if n == OOM_ON and v == 1.0]
    assert not fabricated, (
        f"a live hint of 1.0 was written while the OOM'd slice was the most "
        f"recent pass one. Nothing measured it — pass one never ran — and "
        f"1.0 is above every threshold, so it switches the two-pass off for "
        f"the rest of the batch group. All writes: {Recording.writes}")
    # And the legitimate warm-up 1.0s are still there, so the assertion above
    # is not passing merely because no 1.0 is ever recorded.
    assert any(v == 1.0 for v, _ in Recording.writes), (
        "no 1.0 was recorded at all, so the check above proves nothing")
    assert twopass.stats()["slices_twopass"] > 0, (
        "the two-pass never recovered after the pass-one OOM")


def test_healthy_slices_clear_the_unchecked_streak_end_to_end(
    ds, tmp_path, monkeypatch
):
    """The streak must be broken by real slices, not just by novel shapes.

    The unit test pins `note_checked_slice`; this pins that the PIPELINE
    calls it. Without that call the limiter counts lifetime failures rather
    than consecutive ones, because at steady state the ladder keeps asking
    for a height it has already proven and the reset inside
    `_verify_shape_locked` never runs.

    Arranged so the fixture cannot hide it: `MAX_UNCHECKED_SLICES` is 2, and
    the first slice fails its ladder while every slice after it succeeds. If
    healthy slices clear the streak the run finishes armed; if they do not,
    the second failure — or in the lifetime reading, the accumulation —
    disables it.
    """
    calls = []
    # ONE rung per slice, so a call index IS a slice index. With the real
    # four-rung ladder, failing "call 1" only fails the first RUNG — the
    # second rung then succeeds and the slice is healthy, which is why the
    # first version of this test survived deleting the reset.
    real_pc = twopass.pad_candidates
    monkeypatch.setattr(
        twopass, "pad_candidates",
        # One real rung plus the full height. Built from the real ladder, not
        # from `pad_height` — that calls `pad_candidates` itself, so a lambda
        # using it recurses forever.
        lambda n_live, n_full: (
            [c for c in real_pc(n_live, n_full) if c < n_full][:1] + [n_full]),
    )
    # TWO failures, far apart, with healthy slices in between: one failure
    # can never reach a limit of 2 no matter what the reset does.
    fail_on = {1, 12}

    def fail_apart(Q, Cn, M):
        calls.append(M)
        # `True`, not the real check: a real call would run
        # `_verify_shape_locked`, whose own reset would clear the streak and
        # hide whether the PIPELINE resets it. Returning the verdict directly
        # is exactly what a CACHE HIT does, and that is the case in question —
        # at steady state the ladder keeps asking for a height it has already
        # proven, so almost every healthy slice takes that path.
        return twopass.UNVERIFIED if len(calls) in fail_on else True

    monkeypatch.setattr(twopass, "_verify_shape", fail_apart)
    monkeypatch.setattr(twopass, "MAX_UNCHECKED_SLICES", 2)

    out = tmp_path / "streak_e2e"
    out.mkdir()
    twopass.reset()
    run_compute(_cfg(ds, out), num_jobs=1, job_rank=0)
    st = twopass.stats()

    assert st["slices_twopass"] > 0, (
        f"no slice ever succeeded, so nothing could have cleared the "
        f"streak and this test proves nothing: {st}")
    assert len(calls) > max(fail_on), (
        f"only {len(calls)} ladder walks; the second injected failure never "
        f"happened, so the accumulation under test was never possible")
    assert st["unavailable"] is None, (
        f"two unchecked slices separated by healthy ones tripped a limit "
        f"documented as CONSECUTIVE: {st['unavailable']!r}")


def test_a_verification_that_never_completes_eventually_gives_up(
    ds, tmp_path, monkeypatch, caplog
):
    """Never disabling is the other way to get this wrong.
    """
    import logging

    monkeypatch.setattr(twopass, "_verify_shape",
                        lambda *a, **kw: twopass.UNVERIFIED)
    monkeypatch.setattr(twopass, "MAX_UNCHECKED_SLICES", 3)
    out = tmp_path / "gives_up"
    out.mkdir()
    twopass.reset()
    with caplog.at_level(logging.WARNING, logger="nova_bf.compute"):
        run_compute(_cfg(ds, out), num_jobs=1, job_rank=0)

    reason = twopass.stats()["unavailable"]
    assert reason, (
        "the run never gave up, so a device that can never verify would "
        "thrash the allocator for the whole run")
    assert "COMPLETED" in reason and "consecutive" in reason, (
        f"the reason has to say the check never completed: {reason!r}")
    assert "is bit-identical" not in reason, (
        f"this accuses the device of a mismatch that was never measured — "
        f"the exact confusion the three-state verdict exists to prevent: "
        f"{reason!r}")

    # And the results are still the one-pass results.
    ref, _ = _run(ds, tmp_path / "ref_gives_up", monkeypatch, twopass_on=False)
    got = _tables(out)
    for name in ref:
        assert got[name].equals(ref[name])


def test_a_disable_does_not_leak_into_the_next_run(ds, tmp_path, monkeypatch):
    """A disable is a statement about ONE run's shapes. A process that runs
    `run_compute` twice — the test suite, `nova dist` in process — must not
    have the second run silently slowed by the first, with nothing saying so."""
    twopass.disable("test")
    assert not twopass.enabled()
    twopass.reset()
    assert twopass.enabled()
    assert twopass.stats()["unavailable"] is None


def test_two_run_computes_in_one_process_report_their_own_work(
    ds, tmp_path, monkeypatch
):
    """The same thing through `run_compute`, which is where it has to hold.

    Nothing here resets the two-pass by hand: `run_compute` does it, and this
    pins that it does. Two halves, and both have bitten:

    * the counters are per-run, so the second manifest must report the second
      run's work and not the sum of both;
    * a `disable()` from before the second run must not survive into it — the
      next run re-verifies its own shapes, and leaving the flag set would make
      one bad run silently slow every later one in the process.
    """
    def _run_one(name):
        out = tmp_path / name
        out.mkdir()
        run_compute(_cfg(ds, out), num_jobs=1, job_rank=0)
        hits = list(out.rglob("_bf_manifest_*/rank000.json"))
        assert len(hits) == 1, hits
        doc = json.loads(hits[0].read_text())
        return doc["params"]["kernels"]["twopass"]

    first = _run_one("first")
    assert first["launches"] > 0, first
    assert first["rows_full"] > 0, first

    twopass.disable("a shape from the previous run")
    assert not twopass.enabled()

    second = _run_one("second")
    assert second["unavailable"] is None, second
    assert second["launches"] == first["launches"], (first, second)
    assert second["rows_full"] == first["rows_full"], (first, second)


def test_disabled_by_env(ds, tmp_path, monkeypatch):
    monkeypatch.setenv("NOVA_BF_NO_TWOPASS", "1")
    assert not twopass.enabled()
    out = tmp_path / "envoff"
    out.mkdir()
    run_compute(_cfg(ds, out), num_jobs=1, job_rank=0)
    assert twopass.stats()["slices_twopass"] == 0


def test_a_mid_run_disable_reports_itself_and_does_not_kill_the_run(
    ds, tmp_path, monkeypatch, caplog
):
    """`disable()` is the feature's graceful-degradation path, and it has to
    actually be graceful.
    """
    import logging

    real = twopass._verify_shape
    calls = []

    def flaky(Q, Cn, M):
        calls.append(M)
        return real(Q, Cn, M) if len(calls) <= 1 else False

    monkeypatch.setattr(twopass, "_verify_shape", flaky)

    out = tmp_path / "disabled_run"
    out.mkdir(parents=True, exist_ok=True)
    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        paths = run_compute(_cfg(ds, out))

    assert len(calls) > 1, (
        f"only {len(calls)} verification(s) ran, so the failing branch was "
        f"never reached and this test proves nothing")
    assert twopass.stats()["unavailable"] is not None, (
        "the two-pass should have disabled itself")

    lines = [r.getMessage() for r in caplog.records if "twopass file=" in r.getMessage()]
    assert lines, "no per-file two-pass line was emitted"
    disabled = [l for l in lines if "DISABLED:" in l]
    assert disabled, (
        f"no per-file line reported the disable, so the branch that used to "
        f"raise was not exercised: {lines[:3]}")
    assert "is bit-identical" in disabled[0], (
        f"the line must carry the reason, got: {disabled[0]}")

    # and the run still produced its results
    assert paths, "the run produced no output despite promising results are unaffected"

    # Every pass one is accounted for. `slices_fused + slices_unfused` counts
    # the pass ones that ran; each either produced a two-pass slice or had its
    # result discarded. The disable path used to increment neither, so a run
    # that ended this way silently under-reported the work it had wasted —
    # which is the one path where the waste is certain.
    st = twopass.stats()
    pass_ones = st["slices_fused"] + st["slices_unfused"]
    assert pass_ones == st["slices_twopass"] + st["slices_discarded"], (
        f"{pass_ones} pass one(s) ran but "
        f"{st['slices_twopass']} + {st['slices_discarded']} are accounted "
        f"for: {st}")
    # NOT `slices_discarded > 0` on its own: the OTHER discard path
    # (`M >= n_full`) fires on ordinary slices in this same run, so that
    # assertion stays true even with the disable-path increment deleted. The
    # identity above is what actually catches it — verified by reverting the
    # increment, which leaves `slices_discarded == 2` and fails only on the
    # identity.
    assert st["slices_discarded"] >= 1



# --- the eligibility guards -------------------------------------------------
#
# `_twopass_groups` refuses a dozen situations its bound does not describe.
# Each refusal is a one-line check, and the ones below had NO test at all: no
# fixture in this suite builds a `share_gram` batch, a sparse batch reaching a
# dense group, or a non-float32 query matrix, so deleting any of those lines
# broke nothing visible. They are tested here directly rather than end to end,
# because a run that produced those shapes would fail for other reasons first.


def _groups_args(**over):
    """A minimal, ELIGIBLE call to `_twopass_groups`, for a guard to spoil.

    Each test below overrides exactly one thing, and every test first asserts
    the unmodified args ARE accepted — otherwise `== {}` would pass for any
    reason at all, which is how the euclidean test managed to certify a guard
    that had been deleted.
    """
    import numpy as np
    import torch

    from nova_bf import compute as compute_mod

    n_q, dim, rows = 64, 64, 32
    Q = torch.randn(n_q, dim, dtype=torch.float32)
    args = dict(
        batch=compute_mod.DenseCorpusBatch(
            np.random.default_rng(0).standard_normal((rows, dim), dtype=np.float32)),
        score_groups={("cosine", True): [0]},
        spec_Q=[Q],
        spec_q_norms=[Q.norm(dim=1)],
        spec_qsel=[None],
        device="cpu",
        prune=True,
    )
    args.update(over)
    return args


@pytest.fixture
def _eligible(monkeypatch):
    """Make the minimal args actually eligible on a CPU box."""
    import torch

    monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 4)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    twopass.reset()


def _refuses(label, **over):
    from nova_bf import compute as compute_mod

    assert compute_mod._twopass_groups(**_groups_args()), (
        f"{label}: the CONTROL args were refused, so this test cannot tell a "
        f"working guard from a broken fixture")
    got = compute_mod._twopass_groups(**_groups_args(**over))
    assert got == {}, f"{label}: admitted {list(got)}"


def test_config_policy_off_refuses_an_otherwise_eligible_group(_eligible):
    _refuses("two_pass=off", two_pass=False)


def test_the_tf32_refusal_is_reported_once_per_run_not_once_per_process(
    _eligible, caplog, monkeypatch
):
    """"...for this run" has to mean this run.

    The TF32 refusal is the quietest of all the eligibility guards: it
    returns `{}` without calling `disable()` and without moving a counter, so
    the WARNING is the only evidence it happened. `twopass.reset()` clears
    its own warn-once flags for exactly this reason, and this one lives in
    `compute` and was missed. Left set, run 2 of an in-process `nova dist` is
    silent and its manifest is byte-identical to a run whose live fraction
    simply never fell.
    """
    import logging

    import torch

    from nova_bf import compute as compute_mod

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)

    def one_run():
        compute_mod._reset_twopass_hints()
        twopass.reset()
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="nova_bf.compute"):
            # `device="cuda:0"` as a STRING, with no GPU present: the TF32
            # refusal is now keyed on the device, because the flag governs the
            # CUDA matmul path and says nothing about the CPU one. The
            # function only ever reads `str(device).startswith("cuda")`, so a
            # string reaches the branch a GPU-less box otherwise cannot.
            got = compute_mod._twopass_groups(
                **_groups_args(device="cuda:0"))
        assert got == {}, "premise: TF32 must refuse the two-pass"
        return [r for r in caplog.records if "tf32" in r.getMessage().lower()]

    assert len(one_run()) == 1, "run 1 did not report the refusal at all"
    assert len(one_run()) == 1, (
        "run 2 inherited run 1's warn-once flag, so it never reported that "
        "its two-pass was off")


def test_members_reading_different_query_norms_are_refused(_eligible):
    """The `Q` half of this pair is tested; the `q_norms` half was not.

    `row_scale` is built from `members[0]`'s norms alone and then used for
    every member's bound. If two members reached the same score group with
    different norm vectors — a filtered search whose selector changed the
    norm set, say — the second member's `eps` would be computed against the
    first member's `1/‖q‖`. Wrong in either direction, and the unsafe one
    under-bounds and kills a live row. No crash, no counter, no symptom.
    """
    import torch

    from nova_bf import compute as compute_mod

    Q = torch.randn(64, 64, dtype=torch.float32)
    norms = Q.norm(dim=1)
    shared = dict(score_groups={("cosine", True): [0, 1]},
                  spec_Q=[Q, Q], spec_qsel=[None, None])

    assert compute_mod._twopass_groups(
        **_groups_args(spec_q_norms=[norms, norms], **shared)), (
        "the CONTROL was refused, so this test cannot tell a working guard "
        "from a broken fixture")

    # Equal VALUES, different object. The check is identity on purpose: the
    # two are indistinguishable to the bound but not to the reader, and an
    # identity check is the one that cannot be fooled by a later in-place
    # write to one of them.
    got = compute_mod._twopass_groups(
        **_groups_args(spec_q_norms=[norms, norms.clone()], **shared))
    assert got == {}, f"admitted members with different norm vectors: {got}"


@pytest.mark.parametrize("dtype", ["float64", "float16", "bfloat16"])
def test_a_non_float32_query_matrix_is_refused(_eligible, dtype):
    """The bound's `gamma_d` term is the float32 dot-product bound, and `dq`
    is measured as the float16 residual OF a float32 matrix. A query matrix
    stored at lower precision rounds worse than either assumes, so `eps` would
    under-bound the score the run actually reports."""
    import torch

    Q = torch.randn(64, 64).to(getattr(torch, dtype))
    _refuses(f"Q dtype {dtype}", spec_Q=[Q],
             spec_q_norms=[Q.float().norm(dim=1)])


def test_a_shared_gram_batch_is_refused(_eligible):
    """The shared-Gram derivation caches one raw `Q @ Cb.T` across metrics; the
    two-pass narrows the query axis, so the two cannot use the same matrix."""
    args = _groups_args()
    args["batch"].share_gram = True
    from nova_bf import compute as compute_mod

    assert compute_mod._twopass_groups(**_groups_args()), "control was refused"
    assert compute_mod._twopass_groups(**args) == {}, (
        "a share_gram batch was admitted; the narrowed pass two cannot share "
        "the cached full-height Gram")


def test_a_non_dense_corpus_batch_is_refused(_eligible):
    """Sparse and multivector batches have neither the layout nor the metric
    the bound is written for."""
    class _NotDense:
        share_gram = False
        n_rows = 32

    _refuses("non-dense batch", batch=_NotDense())


def test_pruning_disabled_refuses_the_two_pass(_eligible):
    """`NOVA_BF_NO_PRUNE` turns off the selection the two-pass exists to
    accelerate; running pass one anyway would be pure cost."""
    _refuses("prune=False", prune=False)


def test_tf32_turns_the_two_pass_off_rather_than_widening_the_bound(monkeypatch):
    """TF32 is refused, because the bound does not cover it.
    """
    import numpy as np
    import torch

    from nova_bf import compute as compute_mod

    # `device="cuda:0"` as a STRING on a GPU-less box. The refusal is keyed on
    # the device now: `allow_tf32` governs the CUDA matmul path and has no
    # bearing on the CPU GEMM `_scores` runs, so letting it veto the CPU
    # opt-in path meant any stray global could silently switch off the only
    # route a GPU-less suite has into this plumbing. `_twopass_groups` reads
    # the device solely via `str(device).startswith("cuda")`, so a string
    # reaches the branch that a CPU box otherwise cannot.
    args = _groups_args(device="cuda:0")
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 4)

    # Control: with TF32 off the group is eligible, or the assertion below
    # would pass for any reason at all.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    assert compute_mod._twopass_groups(**args), (
        "control: the group should be eligible with TF32 off")

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(compute_mod, "_TP_TF32_WARNED", False)
    assert compute_mod._twopass_groups(**args) == {}, (
        "the two-pass accepted a group with TF32 enabled, and its bound does "
        "not cover TF32 arithmetic")


def test_a_nan_bound_keeps_its_row_live(ds, tmp_path, monkeypatch):
    """The pruning predicate is `~(upper < thr)`, not `upper >= thr`.

    The difference is only visible for NaN, and NaN is reachable: an unusable
    query norm makes `dq` infinite, and against an fp16-exact corpus (`Dc = 0`)
    the bound computes `inf * 0 = NaN`. `~(NaN < thr)` is True — live, which is
    the safe direction — while `NaN >= thr` is False, which marks a row DEAD
    against a real candidate.

    Nothing covered this: `test_a_partially_underflowed_norm_forces_the_row_live`
    builds exactly this data but asserts on `upper_bounds`' output, never on
    the predicate in `compute.py`. Rewriting `~(a < b)` as `a >= b` is a
    textbook "simplification" of a double negative, and it left the whole
    suite green.
    """
    import pyarrow.parquet as pq

    rng = np.random.default_rng(11)
    qpath = tmp_path / "queries_nan.parquet"
    vecs = rng.standard_normal((NQ, DIM)).astype(np.float32)
    # Two such rows, one in each half of the query set. Note the second buys
    # no extra detection HERE and the comment that once claimed otherwise was
    # wrong: measured, mutating only the `live[qsel] |= ...` branch still
    # leaves this test green, because the group also contains a full-matrix
    # member whose correct branch marks the row live and masks it. That branch
    # is covered by `test_members_with_different_k_union_their_live_rows`,
    # whose members are ALL row-subsetted so nothing can cover for it. The
    # second row is kept only so the fixture exercises both spans.
    vecs[7] = 2e-23                      # in the `all` member only
    vecs[NQ // 2 + 3] = 2e-23            # also inside `half`
    _write(qpath, vecs, qid=[f"q{i}" for i in range(NQ)],
           qset=["a"] * (NQ // 2) + ["b"] * (NQ - NQ // 2))
    ds2 = {"cdir": ds["cdir"], "qpath": str(qpath)}

    on, st = _run(ds2, tmp_path / "nan_on", monkeypatch, twopass_on=True)
    off, _ = _run(ds2, tmp_path / "nan_off", monkeypatch, twopass_on=False)
    assert st["slices_twopass"] > 0, (
        f"the two-pass never ran, so the predicate was never exercised: {st}")
    for name in ("all", "half"):
        assert on[name].equals(off[name]), (
            f"{name}: a query row whose bound is NaN was scored differently "
            f"with the two-pass on — the predicate is dropping it instead of "
            f"keeping it live")


def test_members_with_different_k_union_their_live_rows(ds, tmp_path, monkeypatch):
    """The shared live mask is a UNION over members: `live[qsel] |= ...`.

    With `=` a later member overwrites an earlier member's contribution and
    the earlier one loses rows it needed. Two conditions are both required to
    see it, and missing either leaves the mutation invisible:

      * the members must be ROW-SUBSETTED, or they take the `live |= ...`
        branch and `live[qsel]` is never written at all;
      * their subsets must OVERLAP and their k must DIFFER, or their live
        masks coincide and the overwrite is a no-op. Every other fixture here
        gives its members the same k.

    `qid` blocks are contiguous, so they resolve to slices, which is what
    `_twopass_groups` requires.
    """
    n = NQ
    front = [f"q{i}" for i in range(0, 3 * n // 4)]        # q0 .. q47
    back = [f"q{i}" for i in range(n // 4, n)]             # q16 .. q63 (overlap 16..47)

    # One query row in the overlap whose float32 norm underflows, so its bound
    # is NaN. That makes this fixture cover a SECOND predicate mutation as
    # well: `live[qsel] |= ~(x < thr)` rewritten as `>=`. The NaN-specific test
    # above cannot catch that one, because its group also contains a
    # full-matrix member whose (unmutated) branch marks the row live and masks
    # the subsetted member's mistake. Here every member is subsetted, so
    # nothing covers for it.
    rng = np.random.default_rng(23)
    qpath = tmp_path / "queries_k.parquet"
    vecs = rng.standard_normal((n, DIM)).astype(np.float32)
    vecs[n // 2] = 2e-23
    _write(qpath, vecs, qid=[f"q{i}" for i in range(n)],
           qset=["a"] * (n // 2) + ["b"] * (n - n // 2))

    out_on, out_off = tmp_path / "k_on", tmp_path / "k_off"
    for out, on in ((out_on, True), (out_off, False)):
        out.mkdir(parents=True, exist_ok=True)
        if not on:
            monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 1 << 30)
        cfg = BruteForceConfig(
            corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
            queries=QueriesConfig(path=str(qpath), id_column="qid",
                                  payload_fields=["qset"]),
            output=OutputConfig(path=str(out)),
            params=ParamsConfig(io_workers=1, dense_batch_size=BATCH),
            searches=[
                SearchSpec(name="wide", metric="cosine", k=25,
                           rows={"column": "qid", "isin": front}),
                SearchSpec(name="narrow", metric="cosine", k=2,
                           rows={"column": "qid", "isin": back}),
            ],
        )
        run_compute(cfg, num_jobs=1, job_rank=0)

    def _one(out, name, k):
        hits = list(out.rglob(f"*_{name}_k{k}/rank000.parquet"))
        assert len(hits) == 1, f"{name} k={k}: {hits}"
        return pq.read_table(hits[0])

    for name, k in (("wide", 25), ("narrow", 2)):
        a, b = _one(out_on, name, k), _one(out_off, name, k)
        assert a.equals(b), (
            f"{name} (k={k}) differs between the two-pass and one-pass runs; "
            f"overlapping members with different k have different live masks, "
            f"so the shared mask must be a union, not an assignment")


def test_the_prune_counter_does_not_move_with_the_two_pass(ds, tmp_path, monkeypatch):
    """`params.kernels.prune.launches` must be two-pass neutral.

    The feature's whole contract is that it changes nothing observable, and a
    manifest field documented as "what actually happened" is observable. The
    fully-pruned-slice path skipped the increment that both one-pass branches
    perform, so the field moved with the flag.

    A fully-pruned slice IS a pruned selection, and the most complete one
    there is; not counting it was the bug.
    """
    import json

    def _launches(out, twopass_on):
        out.mkdir(parents=True, exist_ok=True)
        if not twopass_on:
            monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 1 << 30)
        twopass.reset()
        # A SMALL k and a SMALL batch, deliberately: the missing increment
        # lives on the fully-pruned-slice path, and the module's default
        # fixture (k=5, batch=64) produces slices too large to ever prune
        # every row — measured zero of them, which made a first version of
        # this test pass with the fix reverted.
        cfg = BruteForceConfig(
            corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
            queries=QueriesConfig(path=ds["qpath"], id_column="qid",
                                  payload_fields=["qset"]),
            output=OutputConfig(path=str(out)),
            params=ParamsConfig(io_workers=1, dense_batch_size=4),
            searches=[SearchSpec(name="all", metric="cosine", k=2)],
        )
        run_compute(cfg)
        doc = json.loads(next(out.rglob("*manifest*.json")).read_text())
        st = twopass.stats()
        return (doc["params"]["kernels"]["prune"]["launches"],
                st["slices_twopass"], st["gemms"])

    on, slices_on, gemms_on = _launches(tmp_path / "prune_on", True)
    off, slices_off, _ = _launches(tmp_path / "prune_off", False)

    assert slices_on > 0, "the two-pass never ran; this comparison is vacuous"
    assert slices_off == 0, "the control arm used the two-pass"
    # The premise that makes this test able to fail: at least one slice must
    # have pruned EVERY row, since that is the only path missing the counter.
    # `slices_twopass` counts plans; `gemms` counts only those that launched a
    # narrowed GEMM, so a gap between them is a fully-pruned slice.
    assert slices_on > gemms_on, (
        f"no slice was fully pruned ({slices_on} plans, {gemms_on} GEMMs), so "
        f"the code path with the missing increment was never reached")
    assert on == off, (
        f"prune.launches is {on} with the two-pass and {off} without it — a "
        f"manifest field is moving with a flag that is supposed to be "
        f"invisible")


def test_a_query_matrix_below_the_minimum_height_is_refused(_eligible, monkeypatch):
    """Below `MIN_QUERY_ROWS` the padded exact GEMM is most of the full one,
    so pass one cannot pay for itself. Performance only — but it is also the
    gate that decides whether any of this machinery runs at all, and every
    other test in this file monkeypatches it, so nothing was checking that it
    still works."""
    from nova_bf import compute as compute_mod

    # Control FIRST, at the fixture's low minimum — `_refuses` cannot be used
    # here because raising the threshold is a global change that would refuse
    # its control too, so the helper would fail for the wrong reason.
    assert compute_mod._twopass_groups(**_groups_args()), (
        "control: the 64-row Q should be eligible at the fixture's minimum")

    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 128)   # above the 64-row Q
    assert compute_mod._twopass_groups(**_groups_args()) == {}, (
        "a query matrix shorter than MIN_QUERY_ROWS was admitted")


def test_members_reading_different_query_matrices_are_refused(_eligible):
    """Every member of a score group must read the SAME `Q` object.

    The two-pass narrows the query axis once for the whole group and hands
    each member a span of that one narrowed result. If two members were
    actually reading different matrices, each would receive rows computed
    from the other's queries — wrong answers, not a crash. The score cache
    already assumes this; here it is load-bearing.
    """
    import torch

    from nova_bf import compute as compute_mod

    args = _groups_args()
    other = torch.randn(64, 64, dtype=torch.float32)       # same shape, different object
    args["score_groups"] = {("cosine", True): [0, 1]}
    args["spec_Q"] = [args["spec_Q"][0], other]
    args["spec_q_norms"] = [args["spec_q_norms"][0], args["spec_q_norms"][0]]
    args["spec_qsel"] = [None, None]

    assert compute_mod._twopass_groups(**args) == {}, (
        "a group whose members read different query matrices was admitted; "
        "each member would get rows computed from the other's queries")

    # Control: the same two-member group sharing one matrix IS admitted, so
    # the refusal above is about the identity check and not the second member.
    shared = _groups_args()
    shared["score_groups"] = {("cosine", True): [0, 1]}
    shared["spec_Q"] = [shared["spec_Q"][0], shared["spec_Q"][0]]
    shared["spec_q_norms"] = [shared["spec_q_norms"][0], shared["spec_q_norms"][0]]
    shared["spec_qsel"] = [None, None]
    assert compute_mod._twopass_groups(**shared), (
        "control: two members sharing one Q should be eligible")


def test_a_gathered_query_selector_is_refused(_eligible):
    """A member's rows must be `None` or a contiguous `slice`.

    `_twopass_prepare` reads each member's live rows as a contiguous SPAN of
    the sorted live index and maps them back with `idx[lo:hi] - qsel.start`.
    Neither holds for a gathered index tensor: the span assumption is false,
    and `qsel.start` does not exist — so admitting one gives wrong rows, or an
    `AttributeError` from inside the plan builder.
    """
    import torch

    from nova_bf import compute as compute_mod

    gathered = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int64)
    assert compute_mod._twopass_groups(**_groups_args(spec_qsel=[gathered])) == {}, (
        "a gathered (non-slice) query selector was admitted; its live rows "
        "are not a contiguous span and it has no `.start`")

    # A real slice IS admitted, so this is about the gather and not the qsel.
    assert compute_mod._twopass_groups(**_groups_args(spec_qsel=[slice(0, 64)])), (
        "control: a contiguous slice selector should be eligible")


def test_the_query_cache_is_released_when_the_run_finishes(ds, tmp_path):
    """`_QCACHE` holds each query matrix and its half copy and `reset()` 
    clears it at the START of a run. In a process that runs `run_compute`
    more than once (`nova dist` in-process, this test suite) that 
    pinned it for the entire gap between runs, long after the matrices were useful.

    The counters must survive, though: the manifest is written from them, so
    the release happens after that and is separate from `reset()`.
    """
    out = tmp_path / "release"
    out.mkdir(parents=True, exist_ok=True)
    twopass.reset()
    run_compute(_cfg(ds, out))

    assert twopass.stats()["slices_twopass"] > 0, (
        "the two-pass never ran, so nothing was cached and this is vacuous")
    assert not twopass._QCACHE, (
        f"the query cache still holds {len(twopass._QCACHE)} entry(ies) after "
        f"the run; its device tensors stay pinned until the next run starts")


def test_the_live_row_audit_actually_runs_and_passes_on_a_real_pipeline(
    ds, tmp_path, monkeypatch
):
    """The audit is the only bound check that runs on every production slice,
    so the failure that matters is it silently never running: a run would then
    report `audit_violations == 0` for the whole rank and that would read as
    evidence when it is nothing at all.

    So this asserts it FIRED (slices and rows counted) as well as that it
    passed, and that the rows it checked are a real fraction of the work.
    """
    _, stats = _run(ds, tmp_path / "audit", monkeypatch, twopass_on=True)
    assert stats["slices_twopass"] > 0, "the two-pass never engaged"
    assert stats["audit_slices"] > 0, (
        "the audit never ran on any slice, so `audit_violations == 0` below "
        "would be vacuous")
    assert stats["audit_rows"] > 0, (
        "the audit ran but checked no rows — every slice must have had zero "
        "live rows, which the `slices_twopass` assertion above rules out")
    assert stats["audit_violations"] == 0, (
        f"the bound FAILED on {stats['audit_violations']} live rows "
        f"(worst {stats['audit_worst']:.3e}) — rows pruned under the same "
        f"bound may have been lost")
    assert stats["audit_worst"] in (0.0, 0)


def test_the_audit_can_be_switched_off_without_changing_results(
    ds, tmp_path, monkeypatch
):
    """It is a check, not a mechanism: disabling it must change the counters
    and nothing else."""
    monkeypatch.setenv("NOVA_BF_TWOPASS_AUDIT", "0")
    twopass._AUDIT_RATE = None
    off, s_off = _run(ds, tmp_path / "off", monkeypatch, twopass_on=True)
    monkeypatch.setenv("NOVA_BF_TWOPASS_AUDIT", "1")
    twopass._AUDIT_RATE = None
    on, s_on = _run(ds, tmp_path / "on", monkeypatch, twopass_on=True)
    twopass._AUDIT_RATE = None
    assert s_off["audit_slices"] == 0 and s_on["audit_slices"] > 0
    assert off == on, "the audit changed the run's results, which it must not"


def test_the_row_scale_is_reused_for_the_whole_run(ds, tmp_path, monkeypatch):
    """`_twopass_groups` must not manufacture a new `row_scale` per file.

    It used to. `twopass.query_side` keys its cache on the IDENTITY of `Q` and
    `row_scale`, so an equal-but-new tensor missed once per file, rebuilt
    `Q.half()` and the norms, and discarded the `eps` cache that lives inside
    the entry.
    """
    seen = []
    real = compute._row_scale_for

    def spy(q_norms):
        rs = real(q_norms)
        seen.append(id(rs))
        return rs

    monkeypatch.setattr(compute, "_row_scale_for", spy)
    _, stats = _run(ds, tmp_path / "rs", monkeypatch, twopass_on=True)
    assert stats["slices_twopass"] > 0
    assert len(seen) > 1, "the fixture produced only one call; it proves nothing"
    assert len(set(seen)) == 1, (
        f"{len(set(seen))} distinct row_scale objects across {len(seen)} calls "
        f"— a fresh tensor per file misses the query-side cache every time and "
        f"takes the eps cache with it")


def test_the_eps_cache_actually_pays_over_a_whole_run(ds, tmp_path, monkeypatch):
    """A HIT-RATE assertion, which is the only kind that catches this class of
    bug: `eps` was being recomputed for every slice while every result stayed
    correct, so the sole symptom was an optimisation quietly doing nothing.

    `eps` depends on the query matrix and on `sigma_max`'s power-of-two bucket.
    Across the slices of one run both are stable, so evaluations should be a
    small fraction of the slices that consulted the cache.
    """
    _, stats = _run(ds, tmp_path / "eps", monkeypatch, twopass_on=True)
    ev, hits = stats["eps_evaluations"], stats["eps_cache_hits"]
    total = ev + hits
    assert stats["slices_twopass"] > 0
    assert total > 1, (
        f"only {total} eps lookups, so a hit rate proves nothing here")
    assert hits > 0, (
        f"eps was evaluated {ev} times and NEVER reused across {total} "
        f"lookups. Either the query-side cache is missing (a new `Q` or "
        f"`row_scale` object per file) or the sigma_max bucket is unstable; "
        f"both make the O(n_q) binary64 evaluation run on every slice")
    # ABSOLUTE evaluations, not a rate. A rate threshold is too weak to catch
    # the bug this test exists for: with N slices per file, a per-file cache
    # miss still hits (N-1)/N of the time, which on this fixture is 67% and
    # sails past any sane rate bar. The run has ONE query matrix and a stable
    # sigma_max bucket, so there is no legitimate reason to evaluate more than
    # once — twice allows for a single bucket step.
    assert ev <= 2, (
        f"eps was evaluated {ev} times across {total} lookups for ONE query "
        f"matrix. Expected 1. This is what a per-file cache miss looks like: "
        f"the hit rate stays high ({100 * hits / total:.0f}%) because slices "
        f"WITHIN a file still hit, while the entry is rebuilt at every file "
        f"boundary")


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
def test_results_are_identical_under_both_tiebreak_modes(
    ds, tmp_path, monkeypatch, tiebreak
):
    """`tiebreak='id'` orders ties by the point id rather than by position.

    The two-pass only decides whether a QUERY ROW is scored at all for a
    slice; it never touches the ordinals or ids attached to corpus columns, so
    it should not be able to change which of two equal-scoring candidates
    wins. That is an argument, and this is the test — the combination is a
    legitimate production one and had no coverage, so the argument was all
    there was.

    An id difference with identical scores is exactly the divergence a
    score-only comparison would miss; `pa.Table.equals` compares the id column
    too, which is why the whole table is compared rather than the scores.
    """
    off, _ = _run(ds, tmp_path / "off", monkeypatch, twopass_on=False,
                  tiebreak=tiebreak)
    on, stats = _run(ds, tmp_path / "on", monkeypatch, twopass_on=True,
                     tiebreak=tiebreak)
    assert stats["slices_twopass"] > 0, "the two-pass never engaged"
    assert stats["rows_live"] < stats["rows_full"], "nothing was pruned"
    for name in off:
        assert on[name].equals(off[name]), f"{name} differs under {tiebreak}"


def test_a_group_with_no_live_rows_at_all_still_matches(tmp_path, monkeypatch):
    """`n_live == 0` — every row of a score group pruned for a slice — takes a
    branch that pushes NOTHING to the pending list, rather than pushing an
    empty or zero-scored entry.
    """
    rng = np.random.default_rng(31)
    cdir = tmp_path / "z_corpus"
    cdir.mkdir()
    g = 0
    for fi in range(24):
        vecs = rng.standard_normal((200, DIM)).astype(np.float32)
        vecs = vecs.astype(np.float16).astype(np.float32)
        _write(cdir / f"f{fi}.parquet", vecs,
               id=[f"c{g + r:06d}" for r in range(200)])
        g += 200
    qpath = tmp_path / "q.parquet"
    _write(qpath, rng.standard_normal((NQ, DIM)).astype(np.float32),
           qid=[f"q{i}" for i in range(NQ)],
           qset=["a"] * (NQ // 2) + ["b"] * (NQ - NQ // 2))
    ds2 = {"cdir": str(cdir), "qpath": str(qpath)}

    off, _ = _run(ds2, tmp_path / "z_off", monkeypatch, twopass_on=False)
    monkeypatch.setenv("NOVA_BF_POISON_DEAD_ROWS", "1")
    on, stats = _run(ds2, tmp_path / "z_on", monkeypatch, twopass_on=True)
    assert stats["slices_twopass"] > 0
    for name in off:
        assert on[name].equals(off[name]), (
            f"{name} differs with dead rows poisoned — an uninitialised dead "
            f"row leaked into a result")


def test_the_pipeline_hands_the_guards_the_raw_corpus_norms(ds, tmp_path,
                                                            monkeypatch):
    """`_twopass_prepare` must pass `col_norms_raw()` to `upper_bounds`, not
    `col_norms()`.

    `col_norms()` clamps at 1e-12, which sits ABOVE `NORM_MIN = 2**-40`
    (9.09e-13) — so a zero-norm corpus row reaches the guard as 1e-12 and
    passes the check whose entire job is to refuse it. Swapping the two at this
    call site passed the entire suite (177 tests) when first tried.

    Asserted by OBJECT IDENTITY rather than by values. Two earlier attempts at
    this test compared contents and both were vacuous: on healthy data the
    clamped and raw vectors are bit-identical, so there is nothing to see, and
    seeding a zero-norm row does not help either — the slice holding it takes
    the plain path (the live-fraction hint does not exist yet on the first
    slice), so it never reaches `upper_bounds` at all. Identity has neither
    problem and is the property actually wanted.

    Masked in production by `prunability_cut` refusing `min N_c < 1e-3`, but
    that threshold is a PERFORMANCE choice, so the distinction is pinned here
    rather than left resting on it.
    """
    raws, seen = [], []
    real_raw = compute.DenseBatchSlice.col_norms_raw
    real_ub = twopass.upper_bounds

    def spy_raw(self):
        out = real_raw(self)
        raws.append(id(out))
        return out

    def spy_ub(Q, Cb, col_scale, row_scale, out_dtype, metric, cn=None, **kw):
        seen.append(None if cn is None else id(cn))
        return real_ub(Q, Cb, col_scale, row_scale, out_dtype, metric, cn=cn,
                       **kw)

    monkeypatch.setattr(compute.DenseBatchSlice, "col_norms_raw", spy_raw)
    monkeypatch.setattr(twopass, "upper_bounds", spy_ub)
    _, stats = _run(ds, tmp_path / "rawn", monkeypatch, twopass_on=True)

    assert stats["slices_twopass"] > 0, "the two-pass never engaged"
    # `_certify_two_pass` legitimately passes no `cn` — `upper_bounds` then
    # computes `Cb.float().norm(dim=1)` itself, which is already raw.
    supplied = [c for c in seen if c is not None]
    assert supplied, (
        "no call supplied `cn`, so the raw-vs-clamped choice was never made "
        "and this test proved nothing")
    assert raws, "col_norms_raw() was never called at all"
    for c in supplied:
        assert c in raws, (
            "the `cn` handed to `upper_bounds` is not the object returned by "
            "`col_norms_raw()` — the CLAMPED norms are reaching the guards, so "
            "a zero-norm corpus row would pass `norm_guard` as 1e-12")


# --------------------------------------------------------------------------
# The decision audit, end to end. `test_twopass_decision_audit.py` pins how
# `audit_decisions` GRADES; these pin how a run drives it — that it grades
# every fp16 decision the run makes, that `=N` really means one slice in N,
# and that a filtered member is skipped rather than mis-graded.
# --------------------------------------------------------------------------

def test_the_decision_audit_grades_every_fp16_decision_a_run_makes(
    ds, tmp_path, monkeypatch
):
    """Both passes on every sampled slice, and the whole confusion matrix.

    The check that matters is the pair of totals: the four outcomes have to
    account for every decision, and `correct_prune + false_prune` has to match
    the rows the run actually denied. If the audit graded a different set of
    rows from the one the run pruned, every "0 false prunes" it ever prints
    would be about the wrong rows.
    """
    monkeypatch.setenv("NOVA_BF_TWOPASS_DEAD_AUDIT", "1")
    _, st = _run(ds, tmp_path / "on", monkeypatch, twopass_on=True)
    assert st["slices_twopass"] > 0 and st["rows_live"] < st["rows_full"], (
        f"the two-pass never engaged or never pruned, so the audit graded "
        f"nothing interesting: {st}")
    assert st["dead_audit_members"] > 0, f"the audit never ran: {st}"
    graded = (st["audit_correct_prune"] + st["dead_audit_violations"]
              + st["audit_correct_live"] + st["audit_wasted_live"])
    assert graded == st["dead_audit_rows"], (
        f"the four outcomes do not account for every decision: {graded} "
        f"classified of {st['dead_audit_rows']} checked — {st}")
    assert st["dead_audit_violations"] == 0, (
        f"the fp16 pass denied {st['dead_audit_violations']} rows that held a "
        f"candidate: results this run would have LOST — {st}")
    # Both halves of the matrix are non-empty, or the test is only exercising
    # one column of it. Denials must exist (the run pruned), and so must keeps.
    assert st["audit_correct_prune"] > 0, f"nothing was denied: {st}"
    assert st["audit_correct_live"] > 0, f"nothing was kept: {st}"


def test_the_audit_grades_a_slice_whose_pass_one_was_discarded(
    ds, tmp_path, monkeypatch
):
    """Pass one's decisions are graded whether or not the run kept them.

    When the padded height saves nothing (`M >= n_full`) the run throws pass
    one away and scores the full slice. The fp16 pass still RAN and still made
    a decision about every row, and that decision is exactly as worth checking
    as one the run acted on — it is the same arithmetic on the same data. So
    the audit sits BEFORE the discard, and `dead_audit_offered` can exceed
    `slices_twopass`.
    """
    monkeypatch.setenv("NOVA_BF_TWOPASS_DEAD_AUDIT", "1")
    _, st = _run(ds, tmp_path / "on", monkeypatch, twopass_on=True)
    assert st["slices_discarded"] > 0, (
        f"no slice was discarded here, so this test did not reach the case it "
        f"is about: {st}")
    assert st["dead_audit_offered"] == st["dead_audit_sampled"] > 0
    # Every slice that reached the sampling point was offered, so the offers
    # must cover the kept slices AND any discarded ones.
    assert st["dead_audit_offered"] >= st["slices_twopass"], (
        f"fewer audit offers than two-passed slices: {st}")


def test_the_audit_samples_one_slice_in_n(ds, tmp_path, monkeypatch):
    """`=N` must actually sample.
    """
    monkeypatch.setenv("NOVA_BF_TWOPASS_DEAD_AUDIT", "1")
    _, every = _run(ds, tmp_path / "a1", monkeypatch, twopass_on=True)
    assert every["dead_audit_sampled"] == every["dead_audit_offered"] > 2, (
        f"=1 must grade every offer: {every}")

    monkeypatch.setenv("NOVA_BF_TWOPASS_DEAD_AUDIT", "3")
    _, third = _run(ds, tmp_path / "a3", monkeypatch, twopass_on=True)
    offers = third["dead_audit_offered"]
    assert offers == every["dead_audit_offered"], (
        f"the rate changed how many slices ran, not just how many were "
        f"graded: {offers} offers at =3 vs {every['dead_audit_offered']} at =1")
    assert third["dead_audit_sampled"] == -(-offers // 3), (
        f"=3 sampled {third['dead_audit_sampled']} of {offers} offers, "
        f"expected {-(-offers // 3)}")
    assert third["dead_audit_sampled"] < every["dead_audit_sampled"], (
        "=3 graded as much as =1, so the rate is being ignored")


def test_the_audit_is_off_unless_asked_for(ds, tmp_path, monkeypatch):
    """It costs the exact GEMM the whole feature exists to avoid."""
    monkeypatch.delenv("NOVA_BF_TWOPASS_DEAD_AUDIT", raising=False)
    _, st = _run(ds, tmp_path / "off", monkeypatch, twopass_on=True)
    assert st["slices_twopass"] > 0
    assert st["dead_audit_offered"] == 0 and st["dead_audit_members"] == 0


def test_a_decision_audit_failure_recomputes_its_current_slice(
    ds, tmp_path, monkeypatch
):
    """An audit detects a bad decision before pass two, so retain exactness.

    Disabling only future slices is insufficient: returning the plan already
    assembled for this slice would still fold its known-bad dead rows into the
    output.  Make the audit fail synthetically and require the plan to be
    discarded, which sends this slice through the ordinary full-height path.
    """
    monkeypatch.setenv("NOVA_BF_TWOPASS_DEAD_AUDIT", "1")
    off, _ = _run(ds, tmp_path / "off", monkeypatch, twopass_on=False)

    calls = 0

    def fail_audit(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        twopass.disable("synthetic decision-audit failure")
        return {"false_prune": 1}

    monkeypatch.setattr(twopass, "audit_decisions", fail_audit)
    on, st = _run(ds, tmp_path / "on", monkeypatch, twopass_on=True)

    assert calls == 1, "the decision audit never reached a two-pass slice"
    assert not twopass.enabled()
    assert st["slices_twopass"] == 0, (
        "the plan from the failed audit slice was still used instead of "
        "being recomputed through the exact path"
    )
    assert st["slices_discarded"] == 1
    for name in off:
        assert on[name].equals(off[name]), name


@pytest.fixture
def ds_filt(tmp_path):
    """A corpus with a filterable column, so a FILTERED member can share a
    score group with an unfiltered one."""
    rng = np.random.default_rng(19)
    cdir = tmp_path / "corpus_filt"
    cdir.mkdir()
    g = 0
    for fi in range(8):
        n = 200
        vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        vecs = vecs.astype(np.float16).astype(np.float32)
        _write(cdir / f"f{fi}.parquet", vecs,
               id=[f"c{g + r:06d}" for r in range(n)],
               lang=["eng" if (g + r) % 2 else "fra" for r in range(n)])
        g += n
    qpath = tmp_path / "queries_filt.parquet"
    _write(qpath, rng.standard_normal((NQ, DIM)).astype(np.float32),
           qid=[f"q{i}" for i in range(NQ)])
    return {"cdir": str(cdir), "qpath": str(qpath)}


def _cfg_filt(ds_filt, out):
    """One unfiltered search and one column-filtered search, same metric and
    vector type, so they share a single dense score group and a single GEMM."""
    from nova_bf.config import Filter, FilterCondition
    return BruteForceConfig(
        corpus=CorpusConfig(path=ds_filt["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds_filt["qpath"], id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, dense_batch_size=BATCH),
        searches=[
            SearchSpec(name="all", metric="cosine", k=K),
            SearchSpec(name="eng", metric="cosine", k=K,
                       filter=Filter(must=[FilterCondition(field="lang",
                                                           match="eng")])),
        ],
    )


def test_a_column_filtered_member_is_skipped_by_the_audit_not_mis_graded(
    ds_filt, tmp_path, monkeypatch
):
    """The audit must not grade a member against columns its filter removes.

    The bound is computed over the WHOLE slice, which is correct — a bound over
    the unfiltered slice dominates any filtered subset's max, so pruning stays
    conservative. But GROUND TRUTH for the audit is `max_c s_e >= thr`, and if
    that max is taken over all columns while the member only ever sees the
    surviving ones, the audit will find a "candidate" the member could never
    have returned and report a FALSE PRUNE — a violation that disables the
    two-pass and stamps the manifest with a claim that the bound is unsound.

    So filtered members are skipped and COUNTED, which is what makes
    "0 false prunes" readable: without the count it would be indistinguishable
    from "everything was checked".

    This is the only assertion anywhere on `dead_audit_skipped_filtered`;
    before it, deleting the skip entirely passed the whole suite.
    """
    monkeypatch.setenv("NOVA_BF_TWOPASS_DEAD_AUDIT", "1")
    out = tmp_path / "on"
    out.mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv("NOVA_BF_NO_TWOPASS", raising=False)
    twopass.reset()
    run_compute(_cfg_filt(ds_filt, out), num_jobs=1, job_rank=0)
    st = twopass.stats()

    assert st["slices_twopass"] > 0 and st["rows_live"] < st["rows_full"], (
        f"the two-pass never engaged or never pruned: {st}")
    assert st["dead_audit_skipped_filtered"] > 0, (
        f"no member was skipped as column-filtered, so this test did not "
        f"reach the case it is about: {st}")
    # The unfiltered sibling must STILL be graded — skipping the filtered
    # member must not take the whole slice's audit with it.
    assert st["dead_audit_members"] > 0, (
        f"the filtered member was skipped and so was everything else: {st}")
    assert st["dead_audit_violations"] == 0, st


def test_grading_a_filtered_member_cannot_raise_a_false_violation(
    ds_filt, tmp_path, monkeypatch
):
    """The skip is ACCURACY hygiene, not a safety guard. Proven, not assumed.

    The obvious story — "grading a filtered member against columns it never
    saw would report a candidate that was never at risk, a false violation
    that disables the run" — is what this skip's comment used to say, and it
    is WRONG. `upper` is a bound on the max over the FULL slice, and the
    audit's ground truth is the max over that same full slice, so:

        dead          <=>  upper < thr
        upper         >=   full_max        (the bound covers every column)
        ground truth  <=>  full_max >= thr

    A false prune would need `full_max >= thr > upper >= full_max`. There is
    no such row. Filtering cannot manufacture a violation for any member.

    What the skip actually prevents is a MIS-COUNT: a filtered member whose
    own surviving columns held nothing, but whose slice held something for
    somebody, is graded `correct_live` when it was really `wasted_live`. That
    inflates the "optimal" share of the confusion matrix — a statistic, not a
    lost result, and not a spurious disable.

    Forcing every member to be graded confirms it: the split moves, the
    violation count does not.
    """
    from nova_bf import compute as compute_mod

    monkeypatch.setenv("NOVA_BF_TWOPASS_DEAD_AUDIT", "1")
    monkeypatch.setattr(compute_mod, "_is_unfiltered", lambda f: True)
    out = tmp_path / "nofilterskip"
    out.mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv("NOVA_BF_NO_TWOPASS", raising=False)
    twopass.reset()
    run_compute(_cfg_filt(ds_filt, out), num_jobs=1, job_rank=0)
    st = twopass.stats()

    assert st["dead_audit_skipped_filtered"] == 0, (
        f"the skip still fired, so nothing extra was graded and this test "
        f"proves nothing: {st}")
    assert st["dead_audit_members"] > 0
    assert st["dead_audit_violations"] == 0, (
        f"grading a filtered member against the unfiltered slice produced "
        f"{st['dead_audit_violations']} violations. That contradicts the "
        f"domination argument above — either the bound is not covering the "
        f"whole slice, or the audit is not maxing over it: {st}")
    assert twopass.enabled()


def test_tf32_does_not_veto_the_cpu_opt_in_path(monkeypatch):
    """`allow_tf32` is a statement about the CUDA matmul path, nothing else.

    On CPU, `_scores` runs a CPU GEMM the flag has no bearing on — and the
    flag is readable AND settable in a build with no GPU at all. Read
    unconditionally, it let any stray global (a test module, a notebook, an
    imported library) silently switch off the CPU opt-in path, which exists
    precisely so a GPU-less suite can exercise this plumbing.

    That is the silent-non-engagement shape that has already cost this feature
    two separate suites' worth of coverage — a d=16 parity fixture below P6,
    and an audit that reported zero violations over zero rows.
    """
    import torch
    from nova_bf import compute as compute_mod

    monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 4)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)

    assert compute_mod._twopass_groups(**_groups_args(device="cpu")), (
        "a CUDA-only flag refused the CPU path, which its arithmetic never "
        "touches")
    # ...and it must still refuse on CUDA, or the guard has just been deleted.
    assert compute_mod._twopass_groups(**_groups_args(device="cuda:0")) == {}


@pytest.mark.parametrize("qsel, why", [
    (slice(0, 8, 2), "a stride of 2 owns 4 rows while `stop - start` says 8"),
    (slice(None), "`start`/`stop` of None crash `torch.tensor(bnds)`"),
    (slice(-4, -1), "`base = start = -4` is a scatter offset from the END"),
    (slice(5, 2), "`stop - start` is a NEGATIVE height"),
    (slice(0, 1_000_000),
     "torch TRUNCATES to Q.shape[0] while `stop - start` claims a million"),
    (slice(0.5, 4.5),
     "float endpoints pass `0 <= start <= stop` and raise on the real slice"),
])
def test_only_a_contiguous_unit_stride_query_block_is_admitted(
    monkeypatch, qsel, why
):
    """`isinstance(qsel, slice)` alone is not the check the comment claimed.

    Downstream reads the selection as three separate things, all of which
    assume a plain forward block:

        height[m] = qsel.stop - qsel.start     the member's state height
        bnds.extend((qsel.start, qsel.stop))   fed to `searchsorted`
        base      = qsel.start                 the scatter offset

    Every slice below satisfies `isinstance` and breaks at least one of them.
    The stride case is the dangerous one: it is SILENT, giving the member a
    span covering rows it does not own. Refusing costs one group scored the
    ordinary way.
    """
    import torch
    from nova_bf import compute as compute_mod

    monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 4)

    assert compute_mod._twopass_groups(**_groups_args(spec_qsel=[slice(0, 8)])), (
        "control: a plain forward block must be admitted")
    assert compute_mod._twopass_groups(**_groups_args(spec_qsel=[qsel])) == {}, (
        f"{qsel!r} was admitted: {why}")


def test_a_numpy_built_slice_is_still_admitted():
    """The bounds check must not be an `isinstance(start, int)` check.

    `isinstance(np.int64(0), int)` is False, so a numpy-built slice would be
    refused for being the wrong TYPE rather than the wrong SHAPE — a guard
    that costs performance on perfectly valid input.
    """
    import numpy as np
    from nova_bf import compute as compute_mod

    import os
    os.environ["NOVA_BF_TWOPASS_ON_CPU"] = "1"
    try:
        real_min = twopass.MIN_QUERY_ROWS
        twopass.MIN_QUERY_ROWS = 4
        got = compute_mod._twopass_groups(
            **_groups_args(spec_qsel=[slice(np.int64(0), np.int64(8))]))
    finally:
        twopass.MIN_QUERY_ROWS = real_min
        os.environ.pop("NOVA_BF_TWOPASS_ON_CPU", None)
    assert got, "a numpy-built slice is a valid contiguous block"


@pytest.mark.parametrize("bad_shape", [(64, 1), (1, 64), (8, 8)])
def test_a_threshold_that_is_not_a_flat_vector_is_refused(
    monkeypatch, bad_shape
):
    """`numel()` does not pin a shape, and this guard is about broadcasting.

    A `(want, 1)` threshold has `numel() == want` and passes an element-count
    check — then `upper < thr_s` broadcasts a length-`want` mask into a
    `want x want` MATRIX, and the live mask is computed from the wrong thing
    entirely. So the check has to test `ndim` and `shape[0]`.

    Driven through `_twopass_prepare` itself, not by re-deriving the
    condition: an earlier version of this test re-implemented the comparison
    and asserted on its own arithmetic, which meant reverting the guard to
    `numel()` left it passing. The mutation harness caught that.
    """
    import numpy as np
    import torch

    from nova_bf import compute as compute_mod

    monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
    # Certification runs before the threshold is validated and can come back
    # INCONCLUSIVE on a fixture-sized slice, which `continue`s past the guard.
    # SCOPED TO THIS TEST: an earlier version of this line landed in the
    # module's autouse fixture instead and silently switched certification off
    # for every test in the file — seven of which exist to check exactly that.
    monkeypatch.setenv("NOVA_BF_TWOPASS_NO_CERTIFY", "1")
    # Force engagement: the group is skipped before the threshold is ever
    # validated unless the live-fraction hint has fallen past the gate, and on
    # a fresh 32-row slice it has not. Without this the test reaches nothing
    # and "DID NOT RAISE" is the wrong reason for a failure.
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 4)
    monkeypatch.setattr(twopass, "PAD_QUANTUM", 8)
    monkeypatch.setattr(twopass, "PAD_FLOOR", 8)
    monkeypatch.setattr(twopass, "PAD_SMALL", ())
    twopass.reset()

    n_q, dim, rows = 64, 64, 32
    Q = torch.randn(n_q, dim, dtype=torch.float32)
    batch = compute_mod.DenseCorpusBatch(
        np.random.default_rng(0).standard_normal((rows, dim), dtype=np.float32))
    groups = compute_mod._twopass_groups(
        batch=batch, score_groups={("cosine", True): [0]}, spec_Q=[Q],
        spec_q_norms=[Q.norm(dim=1)], spec_qsel=[None], device="cpu",
        prune=True)
    assert groups, "control: the group must be eligible or this proves nothing"

    # Seed the live-fraction hint. It is normally set by a preceding one-pass
    # slice, and `hint is None` skips the group outright — before the
    # threshold is looked at — so without this the test reaches nothing and
    # "DID NOT RAISE" would be the wrong reason for a pass or a failure.
    # `monkeypatch.setitem`, not a bare assignment: `_TP_LIVE_HINT` is a
    # module-level dict that `twopass.reset()` does not touch (it lives in
    # `compute`, and only `run_compute` clears it). A leaked entry makes a
    # later test find a hint where it expects none — which is exactly what
    # happened, and the mutation harness's baseline check is what caught it.
    for _g in groups.values():
        monkeypatch.setitem(compute_mod._TP_LIVE_HINT, _g["key"], 0.1)

    sl = batch.transfer(0, rows, "cpu")
    good = torch.zeros(n_q, dtype=torch.float32)
    # Control: the correctly shaped threshold is accepted.
    compute_mod._twopass_prepare(groups, sl, [good], [None], "cpu")

    bad = torch.zeros(bad_shape, dtype=torch.float32)
    if bad.numel() == n_q:
        assert bad.numel() == good.numel(), (
            "premise: this shape passes an element-count check")
    with pytest.raises(RuntimeError, match="threshold has shape"):
        compute_mod._twopass_prepare(groups, sl, [bad], [None], "cpu")


def test_the_decision_audit_takes_no_scale_arguments():
    """`audit_decisions` must not carry parameters it does not read.

    It used to take `col_scale` and `row_scale`, both unused. The call site
    subsets `Q` and `q_norms` to a member's row span but passed `row_scale`
    whole — which an external review read as a live bug, correctly in
    mechanism and wrongly in consequence: it was harmless only because nothing
    looked at it. A dead parameter that appears load-bearing is worse than a
    bug, because whoever starts using it inherits the mismatch with no signal.
    """
    import inspect
    from nova_bf import twopass

    params = list(inspect.signature(twopass.audit_decisions).parameters)
    assert params == ["Q", "Cb", "metric", "live", "thr", "q_norms"], params
