"""Pruning in the shapes the ground-truth run actually uses.

The other prune suites run unfiltered, single-job, at k<=50 with batch sizes
of 16-64. The production GT run is none of those: it runs filtered and
structured searches, 32-64 sharded ranks, k=1000 and dense_batch_size=4096.
Those are not "bigger" versions of the tested shapes, they are DIFFERENT code
paths:

  * a filter `-inf`s non-matching cells exactly like the sparse no-overlap
    gate, so a selective filter leaves the state under-filled and its
    threshold pinned at the sentinel for the whole scan;
  * `merge_triton.available()` gates on `k + w <= MAX_BLOCK` (8192). At k=10
    that can never decline; at k=1000 with wide parts it can, silently
    routing every flush through the fall-through instead of the kernel;
  * each shard prunes against its OWN partial state, and the merge has to
    reduce partials that were produced with rows skipped.

`_assert_same` compares against a `NOVA_BF_NO_PRUNE` run of the identical
config, so any divergence is the prune's doing.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")
import torch

import gpu_contract
from nova_bf.compute import run_compute
from nova_bf.config import (
    BruteForceConfig, CorpusConfig, Filter, FilterCondition, OutputConfig,
    ParamsConfig, QueriesConfig, SearchSpec,
)
from nova_bf.merge import run_merge
from test_prune_search_paths import MODES, _assert_same, _read

DIM = 8


def _filter_corpus(tmp_path, n_files=8, per_file=250, seed=0):
    """A `tenant` column with a deliberately skewed distribution: tenant 'rare'
    lands on ~1 row in 60, so a search filtered to it can never fill k=20 and
    keeps a sentinel threshold throughout."""
    rng = np.random.default_rng(seed)
    cdir = tmp_path / "cf"
    cdir.mkdir(exist_ok=True)
    all_vecs, all_ids, all_tenants = [], [], []
    for f in range(n_files):
        v = rng.normal(size=(per_file, DIM)).astype(np.float32)
        ten = ["rare" if (j % 60) == 0 else ("a" if j % 2 else "b")
               for j in range(per_file)]
        ids = [f"c{f:02d}_{j:04d}" for j in range(per_file)]
        pq.write_table(pa.table({
            "dense_embedding": pa.array(v.tolist(), pa.list_(pa.float32())),
            "sid": pa.array(ids),
            "tenant": pa.array(ten),
        }), str(cdir / f"f{f:02d}.parquet"))
        all_vecs.append(v)
        all_ids.extend(ids)
        all_tenants.extend(ten)
    qv = rng.normal(size=(6, DIM)).astype(np.float32)
    qpath = tmp_path / "qf.parquet"
    pq.write_table(pa.table({
        "dense_embedding": pa.array(qv.tolist(), pa.list_(pa.float32())),
        "qid": pa.array([f"q{i}" for i in range(len(qv))]),
        # Half the queries want a common tenant, half the rare one.
        "tenant_want": pa.array(["a" if i % 2 else "rare" for i in range(len(qv))]),
    }), str(qpath))
    return cdir, qpath, np.concatenate(all_vecs), all_ids, all_tenants


def _filter_cfg(cdir, qpath, out, k=20, batch=64):
    """Three filter flavours at once, since they take different routes:
    a uniform literal match, a per-query match pulled from the query row, and
    an unfiltered control sharing the same batch group."""
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="sid"),
        queries=QueriesConfig(path=str(qpath), id_column="qid",
                              payload_fields=["tenant_want"]),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, dense_batch_size=batch, tiebreak="id"),
        searches=[
            SearchSpec(name="plain", k=k, metric="cosine"),
            SearchSpec(name="uniform", k=k, metric="cosine",
                       filter=Filter(must=[FilterCondition(field="tenant",
                                                           match="a")])),
            SearchSpec(name="perquery", k=k, metric="cosine",
                       filter=Filter(must=[
                           FilterCondition(field="tenant",
                                           match_from_query="tenant_want")])),
        ],
    )


@pytest.mark.parametrize("mode", MODES)
def test_filtered_prune_matches_prune_disabled(tmp_path, monkeypatch, mode):
    """Filtered specs share a batch group with an unfiltered one. Each keeps
    its own threshold; a filter's `-inf` cells must not be pruned away, and a
    mask must not leak between specs."""
    cdir, qpath, _, _, _ = _filter_corpus(tmp_path, seed=4)

    monkeypatch.setenv("NOVA_BF_NO_PRUNE", "1")
    base = _read(run_compute(_filter_cfg(cdir, qpath, tmp_path / f"fb{mode}")))

    monkeypatch.delenv("NOVA_BF_NO_PRUNE", raising=False)
    state = gpu_contract.install(monkeypatch, mode=mode)
    got = _read(run_compute(_filter_cfg(cdir, qpath, tmp_path / f"fg{mode}")))

    assert state["dead_rows_seen"] > 0, f"filtered/{mode}: prune never fired"
    assert set(got) == {"plain", "uniform", "perquery"}
    _assert_same(got, base, f"filtered/{mode}")


def test_filtered_selective_query_keeps_its_few_candidates(tmp_path, monkeypatch):
    """The regime a filter creates that nothing else does: so few rows survive
    that the state never fills, so the threshold stays the sentinel. Every
    surviving row must still be returned."""
    cdir, qpath, cvecs, ids, tenants = _filter_corpus(tmp_path, n_files=4,
                                                      per_file=250, seed=8)
    n_rare = sum(1 for t in tenants if t == "rare")
    # k must exceed the surviving-row count or the state fills and the
    # threshold leaves the sentinel, losing the regime this test exists for.
    k = n_rare + 10
    assert n_rare > 0, "fixture produced no 'rare' rows at all"

    monkeypatch.delenv("NOVA_BF_NO_PRUNE", raising=False)
    gpu_contract.install(monkeypatch, mode="fold")
    got = _read(run_compute(_filter_cfg(cdir, qpath, tmp_path / "fsel", k=k)))

    by_q = dict(zip(got["perquery"]["query_id"], got["perquery"]["hit_ids"]))
    rare_ids = {i for i, t in zip(ids, tenants) if t == "rare"}
    for qid, hits in by_q.items():
        if qid in ("q0", "q2", "q4"):          # the 'rare' half
            assert set(h for h in hits if h) == rare_ids, \
                f"{qid}: an under-filled filtered query lost candidates"


# ---------------------------------------------------------------------------
# sharded compute + merge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_jobs", [2, 3])
def test_sharded_prune_matches_prune_disabled(tmp_path, monkeypatch, n_jobs):
    """Each rank prunes against its OWN partial state, then the merge reduces
    partials that were produced with rows skipped."""
    cdir, qpath, _, _, _ = _filter_corpus(tmp_path, n_files=6, per_file=200, seed=6)

    def _run(out_name, prune: bool):
        cfg = _filter_cfg(cdir, qpath, tmp_path / out_name, k=10)
        for r in range(n_jobs):
            run_compute(cfg, num_jobs=n_jobs, job_rank=r)
        return _read(run_merge(cfg))

    monkeypatch.setenv("NOVA_BF_NO_PRUNE", "1")
    base = _run(f"shb{n_jobs}", prune=False)

    monkeypatch.delenv("NOVA_BF_NO_PRUNE", raising=False)
    state = gpu_contract.install(monkeypatch, mode="fold")
    got = _run(f"shg{n_jobs}", prune=True)

    assert state["dead_rows_seen"] > 0, f"sharded/{n_jobs}: prune never fired"
    _assert_same(got, base, f"sharded/{n_jobs}")


# ---------------------------------------------------------------------------
# production shape: k=1000, dense_batch_size=4096
# ---------------------------------------------------------------------------


def _dense_list_array(v: np.ndarray) -> pa.Array:
    """Build the list<float32> column straight off the numpy buffer — at a few
    million rows `pa.array(v.tolist())` dominates the test's runtime."""
    offsets = np.arange(0, v.shape[0] * v.shape[1] + 1, v.shape[1], dtype=np.int32)
    return pa.ListArray.from_arrays(
        pa.array(offsets), pa.array(v.reshape(-1), pa.float32()))


def _big_corpus(tmp_path, n_files=8, per_file=300_000, seed=0):
    """~2.4M rows. At k=1000 the corpus has to be MILLIONS of rows before
    anything is prunable: a slice is only dead if its whole max falls below
    the 1000th best score, and with a small corpus almost every slice still
    holds a top-1000 candidate. Measured prunability for dense cosine at
    k=1000 is ~8% at 2^21 rows, so anything smaller makes this test vacuous.
    """
    rng = np.random.default_rng(seed)
    cdir = tmp_path / "cbig"
    cdir.mkdir(exist_ok=True)
    for f in range(n_files):
        v = rng.normal(size=(per_file, DIM)).astype(np.float32)
        pq.write_table(pa.table({
            "dense_embedding": _dense_list_array(v),
            "sid": pa.array([f"b{f:02d}_{j:06d}" for j in range(per_file)]),
        }), str(cdir / f"f{f:02d}.parquet"))
    qv = rng.normal(size=(4, DIM)).astype(np.float32)
    qpath = tmp_path / "qbig.parquet"
    pq.write_table(pa.table({
        "dense_embedding": _dense_list_array(qv),
        "qid": pa.array([f"q{i}" for i in range(len(qv))]),
    }), str(qpath))
    return cdir, qpath


def _big_cfg(cdir, qpath, out, k, batch):
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="sid"),
        queries=QueriesConfig(path=str(qpath), id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, dense_batch_size=batch, tiebreak="id"),
        searches=[SearchSpec(name="big", k=k, metric="cosine")],
    )


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="production shape is far too slow on CPU")
@pytest.mark.parametrize("k,batch", [(1000, 4096), (1000, 1024)])
def test_production_k_and_batch_match_prune_disabled(tmp_path, monkeypatch,
                                                     k, batch):
    """k=1000 is where `merge_triton.available()`'s `k + w <= MAX_BLOCK` gate
    can actually decline — at the k<=50 the other suites use it never can, so
    this is a genuinely different dispatch than anything else we test."""
    cdir, qpath = _big_corpus(tmp_path, seed=2)
    tag = f"{k}_{batch}"

    monkeypatch.setenv("NOVA_BF_NO_PRUNE", "1")
    base = _read(run_compute(_big_cfg(cdir, qpath, tmp_path / f"bb{tag}", k, batch)))

    monkeypatch.delenv("NOVA_BF_NO_PRUNE", raising=False)
    state = gpu_contract.install(monkeypatch, mode="native")
    got = _read(run_compute(_big_cfg(cdir, qpath, tmp_path / f"bg{tag}", k, batch)))

    assert state["dead_rows_seen"] > 0, f"prod/{tag}: prune never fired"
    _assert_same(got, base, f"production/{tag}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the kernel")
def test_fold_gate_declines_at_production_k(tmp_path, monkeypatch):
    """Pin the claim above: at k=1000 the fold gate must actually be consulted
    with widths that can exceed MAX_BLOCK. If this stops holding, the test
    over it is quietly measuring the k<=50 dispatch again."""
    from nova_bf import merge_triton

    widths = []
    real = merge_triton.available

    def spy(state_key, state_enc, part_key, part_enc, k, live=None, thr=None):
        widths.append((k, part_key.shape[1]))
        return real(state_key, state_enc, part_key, part_enc, k,
                    live=live, thr=thr)

    monkeypatch.delenv("NOVA_BF_NO_PRUNE", raising=False)
    monkeypatch.setattr(merge_triton, "available", spy)
    cdir, qpath = _big_corpus(tmp_path, n_files=2, per_file=20_000, seed=12)
    run_compute(_big_cfg(cdir, qpath, tmp_path / "gate", k=1000, batch=4096))

    assert widths, "the fold gate was never consulted"
    assert max(k + w for k, w in widths) > 1000, \
        "no flush was wide enough to exercise the MAX_BLOCK gate"
