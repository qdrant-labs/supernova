"""Cross-worker threshold exchange (nova_bf.thresholds).

Three layers:
  * the exchange object on a local store: publish/poll, the max fold, the
    background poller, and that torn / foreign / NaN peer files can only
    prune LESS;
  * end to end on CPU: ranks run back to back with the exchange on, so later
    ranks inherit converged thresholds and prune against them, and the merged
    ground truth must equal a single unsharded run bit for bit (ids AND
    scores: the CPU path is deterministic), across dense/filtered/rows-subset
    searches of different heights;
  * the two hazards the reviews found: a leftover `_thresholds/` from an
    EARLIER run on the same output path (different k) must be ignored, and a
    merge missing a rank must be refused even when forced.
"""
from __future__ import annotations

import glob
import io as _io
import json
import os
import shutil
import time

import types
import numpy as np
import pytest

pytest.importorskip("torch")
import pyarrow as pa
import pyarrow.parquet as pq

from nova_bf import thresholds
from nova_bf.compute import run_compute
from nova_bf.config import (
    BruteForceConfig, CorpusConfig, Filter, FilterCondition, OutputConfig, ParamsConfig,
    QueriesConfig, RowSelector, SearchSpec,
)
from nova_bf.io import Store
from nova_bf.merge import run_merge
from nova_bf.thresholds import ThresholdExchange, file_name

DIM, K = 8, 3
RUN = "a" * 64


def _wait_for(pred, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def _npz(**arrays) -> bytes:
    buf = _io.BytesIO()
    np.savez(buf, seq=np.int64(1), **{thresholds._KEY_PREFIX + k: v for k, v in arrays.items()})
    return buf.getvalue()


# Device the end-to-end tests score on: CPU by default; set NOVA_BF_TEST_DEVICE=cuda
# on a GPU box to run the same suite against the real kernels.
TEST_DEVICE = os.environ.get("NOVA_BF_TEST_DEVICE", "cpu")


# ---------------------------------------------------------------- exchange object

def test_publish_then_peer_polls_lists(tmp_path):
    store = Store(str(tmp_path))
    sizes = {"dense": (5, 2), "sparse": (2, 1)}
    a = ThresholdExchange(store, RUN, 0, 2, sizes, refresh_s=100, publish_min_interval_s=0)
    b = ThresholdExchange(store, RUN, 1, 2, sizes, refresh_s=100, publish_min_interval_s=0)
    try:
        la = {"dense": np.array([[1, 0.5], [-np.inf, -np.inf], [3, 2], [0.5, 0.25], [2, 1]], np.float32),
              "sparse": np.array([[0.1], [0.2]], np.float32)}
        a.publish(la)
        assert _wait_for(lambda: a.stats["publishes"] == 1)
        lb = {"dense": np.array([[0, -1], [4, 3], [-np.inf, -np.inf], [0.5, 0.5], [1, 0]], np.float32),
              "sparse": np.array([[0.3], [-np.inf]], np.float32)}
        b.publish(lb)
        assert _wait_for(lambda: b.stats["publishes"] == 1)
        # Each side sees only the OTHER side's file (its own is excluded), as
        # float16 rounded toward -inf (exact for these values).
        assert b.poll_once() == 1
        v, lists, changed = b.latest(0)
        assert v == 1 and changed and len(lists["dense"]) == 1
        np.testing.assert_array_equal(lists["dense"][0].astype(np.float32), la["dense"])
        assert lists["dense"][0].dtype == np.float16
        assert a.poll_once() == 1
        _, lists_a, _ = a.latest(0)
        np.testing.assert_array_equal(lists_a["dense"][0].astype(np.float32), lb["dense"])
        # Unchanged peer file: nothing folded, no new version, lists still served.
        assert b.poll_once() == 0
        v2, lists2, changed2 = b.latest(v)
        assert v2 == v and not changed2 and lists2 is not None
        # Identical re-publish is skipped; a changed one REPLACES the peer's list.
        a.publish(la)
        assert _wait_for(lambda: a.stats["publishes_skipped_unchanged"] == 1)
        la2 = {"dense": np.full((5, 2), -np.inf, np.float32), "sparse": np.zeros((2, 1), np.float32)}
        a.publish(la2)
        assert _wait_for(lambda: a.stats["publishes"] == 2)
        assert b.poll_once() == 1
        _, lists3, changed3 = b.latest(v)
        assert changed3
        np.testing.assert_array_equal(lists3["dense"][0].astype(np.float32), la2["dense"])
        assert b.stats["peer_ranks_seen"] == 1
        assert b.stats["publish_width"] == {"dense": 2, "sparse": 1}
    finally:
        a.close(); b.close()


def test_background_poller_delivers_without_manual_poll(tmp_path):
    store = Store(str(tmp_path))
    b = ThresholdExchange(store, RUN, 1, 2, {"dense": (3, 2)}, refresh_s=0.05)
    try:
        arr = np.array([[1, 0], [2, 1], [3, 2]], np.float16)
        (tmp_path / b.subdir / file_name(0)).write_bytes(_npz(dense=arr))
        assert _wait_for(lambda: b.stats["versions"] == 1)
        _, lists, _ = b.latest(0)
        np.testing.assert_array_equal(lists["dense"][0], arr)
    finally:
        b.close()


def test_torn_foreign_nan_and_out_of_range_peers_only_prune_less(tmp_path):
    store = Store(str(tmp_path))
    b = ThresholdExchange(store, RUN, 1, 3, {"dense": (3, 2)}, refresh_s=100)
    try:
        root = tmp_path / b.subdir
        (root / file_name(0)).write_bytes(b"\x00\x01garbage")            # torn
        (root / file_name(2)).write_bytes(_npz(dense=np.ones((7, 2), np.float16)))  # wrong height
        (root / file_name(7)).write_bytes(_npz(dense=np.full((3, 2), 9.0, np.float16)))  # rank >= num_jobs
        assert b.poll_once() == 0
        assert b.stats["peer_files_skipped"] == 2
        assert b.latest(0) == (0, None, False)
        (root / file_name(0)).write_bytes(_npz(dense=np.array([[np.nan, 1.0], [2.0, np.inf], [np.nan, -np.inf]], np.float32)))
        assert b.poll_once() == 1
        _, lists, _ = b.latest(0)
        np.testing.assert_array_equal(lists["dense"][0].astype(np.float32),
                                      [[-np.inf, 1.0], [2.0, -np.inf], [-np.inf, -np.inf]])
    finally:
        b.close()


def test_other_run_directory_is_invisible(tmp_path):
    """Files are scoped by run id: another run's thresholds are never read."""
    store = Store(str(tmp_path))
    other = ThresholdExchange(store, "b" * 64, 0, 2, {"dense": (3, 1)}, refresh_s=100, publish_min_interval_s=0)
    other.publish({"dense": np.full((3, 1), 5.0, np.float32)})
    assert _wait_for(lambda: other.stats["publishes"] == 1)
    other.close()
    mine = ThresholdExchange(store, RUN, 1, 2, {"dense": (3, 1)}, refresh_s=100)
    try:
        assert mine.poll_once() == 0
        assert mine.latest(0) == (0, None, False)
    finally:
        mine.close()


def test_s3_style_write_path(tmp_path):
    """On S3 the exchange PUTs the object directly (atomic); exercise that branch."""
    store = Store(str(tmp_path))
    store.is_s3 = True   # pretend: pyarrow local fs still backs the writes
    a = ThresholdExchange(store, RUN, 0, 2, {"dense": (2, 1)}, refresh_s=100, publish_min_interval_s=0)
    try:
        # The first poll ran against a prefix that does not exist yet: that is
        # "no peers", not a listing error (S3 has no directories).
        assert a.stats["poll_errors"] == 0
        os.makedirs(tmp_path / a.subdir)   # S3 needs no prefix creation; the local stand-in does
        a.publish({"dense": np.array([[1.0], [2.0]], np.float32)})
        assert _wait_for(lambda: a.stats["publishes"] == 1)
        assert (tmp_path / a.subdir / file_name(0)).exists()
        assert not (tmp_path / a.subdir / (file_name(0) + ".tmp")).exists()
    finally:
        a.close()


def test_publish_width_fp16_floor_and_union_kth():
    import torch
    assert thresholds.publish_width(1000, 8) == 250
    assert thresholds.publish_width(1000, 2) == 1000
    assert thresholds.publish_width(3, 2) == 3
    assert thresholds.publish_width(1000, 32) == 63
    a = np.array([0.1, 1 / 3, 2 / 3, 1e5, -1e5, np.nan, np.inf, -np.inf, 0.0, -0.1], np.float32)
    f = thresholds.fp16_floor(a)
    assert f.dtype == np.float16
    assert np.all(f.astype(np.float32)[np.isfinite(a)] <= a[np.isfinite(a)])     # never above
    assert np.all(f.astype(np.float32)[~np.isfinite(a)] == -np.inf)
    assert f[3] == np.float16(65504) and f[4] == -np.inf
    # The union's k-th largest over own + peers; fewer than k entries -> -inf.
    own = torch.tensor([[5.0, 1.0], [np.nan, 2.0]], dtype=torch.float32)
    p1 = np.array([[4.0, 3.0], [9.0, 8.0]], np.float16)
    p2 = np.array([[np.float16(-np.inf), 2.0], [1.0, 0.5]], np.float16)
    got = thresholds.union_kth(own, [p1, p2], 3, "cpu")
    assert got.tolist() == [3.0, 2.0]            # row 0: {5,4,3,2,1,-inf}; row 1: NaN dropped -> {9,8,2,1,.5}
    assert thresholds.union_kth(own, [], 3, "cpu").tolist() == [-np.inf, -np.inf]
    assert thresholds.union_kth(own, [p1], 2, "cpu", chunk=1).tolist() == [4.0, 8.0]


def test_keys_from_scores_tie_stays_live_and_slack_is_small():
    import torch
    from nova_bf.tiebreak import live_rows, pack, unpack_score

    ext = thresholds.keys_from_scores(np.array([0.5, -np.inf], np.float32), "cpu")
    got = unpack_score(ext).tolist()
    assert got[1] == float("-inf")
    assert 0.5 - 2e-4 < got[0] < 0.5          # lowered by the re-run allowance only
    # A local candidate that ties (or sits within the allowance of) the external
    # threshold is still live, whatever its ordinal.
    cand = pack(torch.tensor([[0.5], [0.2]]), torch.tensor([[7], [7]]))
    assert live_rows(cand, ext).tolist() == [1, 1]
    cand = pack(torch.tensor([[0.49], [0.2]]), torch.tensor([[0], [0]]))
    assert live_rows(cand, ext).tolist() == [0, 1]


# ---------------------------------------------------------------- end to end

def _dataset(tmp_path, n_files=6, rows=40, n_q=20, seed=0):
    rng = np.random.default_rng(seed)
    cdir = tmp_path / "corpus"; cdir.mkdir()
    for fi in range(n_files):
        v = rng.standard_normal((rows, DIM)).astype(np.float32)
        pq.write_table(pa.table({
            "dense_embedding": pa.array(v.tolist(), pa.list_(pa.float32())),
            "id": pa.array([f"f{fi}r{r}" for r in range(rows)]),
            "lang": pa.array([("eng" if (fi + r) % 3 else "fra") for r in range(rows)]),
        }), str(cdir / f"part{fi}.parquet"))
    q = rng.standard_normal((n_q, DIM)).astype(np.float32)
    pq.write_table(pa.table({
        "dense_embedding": pa.array(q.tolist(), pa.list_(pa.float32())),
        "qid": pa.array([f"q{i}" for i in range(n_q)]),
        "qset": pa.array(["a" if i % 2 else "b" for i in range(n_q)]),
    }), str(tmp_path / "queries.parquet"))
    return str(cdir), str(tmp_path / "queries.parquet")


def _searches(k=K, multi=False):
    s = [SearchSpec(name="dense", vector_type="dense", metric="cosine", k=k)]
    if multi:
        s += [
            SearchSpec(name="eng_only", vector_type="dense", metric="dot", k=k,
                       filter=Filter(must=[FilterCondition(field="lang", match="eng")])),
            SearchSpec(name="half", vector_type="dense", metric="cosine", k=k + 1,
                       rows=RowSelector(column="qset", isin=["a"])),
        ]
    return s


def _cfg(cdir, qpath, out, exchange: bool, k=K, multi=False):
    return BruteForceConfig(
        corpus=CorpusConfig(path=cdir, id_column="id"),
        queries=QueriesConfig(path=qpath, id_column="qid", payload_fields=["qset"]),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(dense_batch_size=16, threshold_exchange=exchange,
                            threshold_exchange_refresh_s=0.2),
        searches=_searches(k, multi),
    )


def _manifests(out):
    files = sorted(glob.glob(os.path.join(out, "_bf_manifest_*_compute", "rank*.json")))
    return [json.load(open(f)) for f in files]


def _run_sharded(cfg, num_jobs):
    for r in range(num_jobs):
        run_compute(cfg, num_jobs=num_jobs, job_rank=r)
    return {name: pq.read_table(p).to_pydict() for name, p in run_merge(cfg).items()}


def _assert_same(merged, single):
    for name in single:
        assert merged[name]["query_id"] == single[name]["query_id"], name
        assert merged[name]["hit_ids"] == single[name]["hit_ids"], name
        assert merged[name]["hit_scores"] == single[name]["hit_scores"], name


def test_two_ranks_exchange_matches_single_run(tmp_path, monkeypatch):
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path)
    single_out = tmp_path / "single"; single_out.mkdir()
    single = {n: pq.read_table(p).to_pydict()
              for n, p in run_compute(_cfg(cdir, qpath, single_out, False), num_jobs=None).items()}

    out = tmp_path / "sharded"; out.mkdir()
    merged = _run_sharded(_cfg(cdir, qpath, out, True), 2)
    _assert_same(merged, single)

    m0, m1 = _manifests(str(out))
    x0, x1 = m0["params"]["threshold_exchange_stats"], m1["params"]["threshold_exchange_stats"]
    assert m0["params"]["threshold_exchange"] is True          # the configured bool survives
    assert x0["enabled"] and x1["enabled"]
    # Rank 0 ran alone: published, saw no peer. Rank 1 inherited rank 0's
    # finished thresholds at construction, BEFORE its first file.
    assert x0["publishes"] >= 1 and x0["peer_files_read"] == 0 and x0["versions_applied"] == 0
    assert x1["peer_files_read"] >= 1 and x1["versions_applied"] >= 1
    assert x1["first_applied_file"] == -1
    # ...and pruned MORE than rank 0 did on the same amount of corpus.
    lf0 = m0["params"]["kernels"]["prune"]["live_fraction"]
    lf1 = m1["params"]["kernels"]["prune"]["live_fraction"]
    assert lf1 < lf0, (lf0, lf1)


def test_multi_search_heights_rows_and_filter(tmp_path, monkeypatch):
    """Three searches with different heights (full, `rows` subset) and metrics,
    one filtered: thresholds are exchanged per search and the merge is exact."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=8, rows=30, n_q=30, seed=5)
    single_out = tmp_path / "single"; single_out.mkdir()
    single = {n: pq.read_table(p).to_pydict()
              for n, p in run_compute(_cfg(cdir, qpath, single_out, False, multi=True), num_jobs=None).items()}
    out = tmp_path / "sharded"; out.mkdir()
    merged = _run_sharded(_cfg(cdir, qpath, out, True, multi=True), 4)
    _assert_same(merged, single)
    last = _manifests(str(out))[-1]["params"]["threshold_exchange_stats"]
    assert last["peer_files_read"] >= 3 and last["versions_applied"] >= 1


def test_exchange_partials_may_be_short_but_merge_is_exact(tmp_path, monkeypatch):
    """With peers' thresholds applied, a rank can legitimately return fewer than
    k hits for a query (it pruned rows that would have made its LOCAL top-k but
    not the global one; the writer drops still-empty slots, so the list is
    shorter). The merge must still be exact."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=8, rows=30, n_q=50, seed=3)
    ref_out = tmp_path / "ref"; ref_out.mkdir()
    ref = {n: pq.read_table(p).to_pydict()
           for n, p in run_compute(_cfg(cdir, qpath, ref_out, False), num_jobs=None).items()}
    out = tmp_path / "x"; out.mkdir()
    merged = _run_sharded(_cfg(cdir, qpath, out, True), 4)
    _assert_same(merged, ref)
    partial_dirs = glob.glob(os.path.join(str(out), "_bf_partial_*dense*"))
    assert partial_dirs
    short = 0
    for f in glob.glob(os.path.join(partial_dirs[0], "*.parquet")):
        t = pq.read_table(f).to_pydict()
        short += sum(1 for h in t["hit_ids"] if len(h) < K)
    assert short > 0


def test_stale_thresholds_from_an_earlier_run_are_ignored(tmp_path, monkeypatch):
    """The hazard both reviews reproduced: run k=2 with 4 ranks, keep ONLY the
    `_thresholds/` directory, then run k=10 with 2 ranks on the same output
    path. The old thresholds are not lower bounds for k=10 and would corrupt
    the result if trusted."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=8, rows=30, n_q=20, seed=11)
    out = tmp_path / "out"; out.mkdir()
    _run_sharded(_cfg(cdir, qpath, out, True, k=2), 4)
    for entry in os.listdir(out):
        if entry != thresholds.SUBDIR:
            p = out / entry
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    assert os.listdir(out / thresholds.SUBDIR)        # the stale run's files remain

    ref_out = tmp_path / "ref"; ref_out.mkdir()
    ref = {n: pq.read_table(p).to_pydict()
           for n, p in run_compute(_cfg(cdir, qpath, ref_out, False, k=10), num_jobs=None).items()}
    merged = _run_sharded(_cfg(cdir, qpath, out, True, k=10), 2)
    _assert_same(merged, ref)
    # Two run ids now live side by side under _thresholds/.
    assert len(os.listdir(out / thresholds.SUBDIR)) == 2


def test_missing_rank_merge_is_refused_even_when_forced(tmp_path, monkeypatch):
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=6, rows=20, n_q=10)
    out = tmp_path / "o"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, True)
    run_compute(cfg, num_jobs=3, job_rank=0)
    run_compute(cfg, num_jobs=3, job_rank=1)
    monkeypatch.setenv("NOVA_BF_MERGE_FORCE", "1")
    with pytest.raises(RuntimeError, match="threshold_exchange"):
        run_merge(cfg)


def test_single_worker_warns_and_runs(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=2, rows=10, n_q=4)
    out = tmp_path / "o"; out.mkdir()
    with caplog.at_level("WARNING"):
        run_compute(_cfg(cdir, qpath, out, True), num_jobs=None)
    assert any("threshold_exchange" in r.message and "single-worker" in r.message for r in caplog.records)
    (m,) = [json.load(open(f)) for f in glob.glob(os.path.join(str(out), "_bf_manifest_*_compute*.json"))]
    assert m["params"]["threshold_exchange_stats"] == {
        "enabled": False, "configured": True, "reason": "single-worker run"}


def _write_corpus_like_queries(cdir, qpath, n_files, rows, seed):
    """Corpus files whose vectors are near-copies of the queries: a run on it
    ends with very HIGH thresholds that are not lower bounds for any other
    corpus. Same file names and row counts as `_dataset`."""
    rng = np.random.default_rng(seed)
    q = np.asarray(pq.read_table(qpath).column("dense_embedding").to_pylist(), np.float32)
    for fi in range(n_files):
        base = q[rng.integers(0, len(q), rows)]
        v = (base + 0.01 * rng.standard_normal(base.shape)).astype(np.float32)
        pq.write_table(pa.table({
            "dense_embedding": pa.array(v.tolist(), pa.list_(pa.float32())),
            "id": pa.array([f"f{fi}r{r}" for r in range(rows)]),
            "lang": pa.array([("eng" if (fi + r) % 3 else "fra") for r in range(rows)]),
        }), os.path.join(cdir, f"part{fi}.parquet"))


def test_same_identity_content_change_is_caught_at_merge(tmp_path, monkeypatch):
    """Round-two hazard: the SAME config on the SAME output path after the
    corpus was regenerated in place under the same file names. The run id is
    identical, so the old (too high) thresholds are read; the merge must
    detect that the result cannot be exact and refuse, unforceably."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=6, rows=30, n_q=20, seed=21)
    out = tmp_path / "out"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, True)
    # First run: corpus = near-copies of the queries -> thresholds near 1.0.
    _write_corpus_like_queries(cdir, qpath, 6, 30, seed=22)
    _run_sharded(cfg, 2)
    # Regenerate the corpus in place (random vectors), keep only _thresholds/.
    (tmp_path / "fresh").mkdir()
    _dataset_files = _dataset(tmp_path / "fresh", n_files=6, rows=30, n_q=20, seed=23)
    for fi in range(6):
        shutil.copy(os.path.join(_dataset_files[0], f"part{fi}.parquet"),
                    os.path.join(cdir, f"part{fi}.parquet"))
    for entry in os.listdir(out):
        if entry != thresholds.SUBDIR:
            p = out / entry
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    run_compute(cfg, num_jobs=2, job_rank=0)
    run_compute(cfg, num_jobs=2, job_rank=1)
    monkeypatch.setenv("NOVA_BF_MERGE_FORCE", "1")
    with pytest.raises(RuntimeError, match="VERIFICATION FAILED"):
        run_merge(cfg)
    # Nothing wrong is left behind under a finished-looking name.
    assert not glob.glob(os.path.join(str(out), "bf_*_dense_k*.parquet"))
    assert not glob.glob(os.path.join(str(out), "_bf_manifest_*merge*", "*.json"))


def test_thresholds_dir_is_scratch_and_partials_relocate(tmp_path, monkeypatch):
    """What the merge verifies against travels INSIDE the partial
    (`applied_threshold` column): deleting `_thresholds/` after the ranks
    finish, or moving the partials to another output path, must still merge
    and verify. Rank 1 runs after rank 0 and so really applied its thresholds."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=4, rows=20, n_q=10)
    ref = tmp_path / "ref"; ref.mkdir()
    single = {n: pq.read_table(p).to_pydict()
              for n, p in run_compute(_cfg(cdir, qpath, ref, False), num_jobs=None).items()}
    out = tmp_path / "o"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, True)
    run_compute(cfg, num_jobs=2, job_rank=0)
    run_compute(cfg, num_jobs=2, job_rank=1)
    (p1,) = glob.glob(os.path.join(str(out), "**", "rank001.parquet"), recursive=True)
    col = pq.read_table(p1, columns=["applied_threshold"]).column(0).to_numpy()
    assert np.isfinite(col).any(), "rank 1 should have applied rank 0's thresholds"
    shutil.rmtree(out / thresholds.SUBDIR)
    moved = tmp_path / "moved"
    shutil.copytree(out, moved)
    shutil.rmtree(out)
    cfg2 = _cfg(cdir, qpath, moved, True)
    merged = {name: pq.read_table(p).to_pydict() for name, p in run_merge(cfg2).items()}
    _assert_same(merged, single)
    (m,) = [json.load(open(f)) for f in glob.glob(os.path.join(str(moved), "_bf_manifest_*merge*", "*.json"))]
    assert m["searches"][0]["threshold_exchange_verified_queries"] == int(np.isfinite(col).sum())


def test_missing_rank_may_be_forced_only_when_nothing_was_applied(tmp_path, monkeypatch):
    """Rank 0 of a sequential run sees no peer (all -inf applied): a forced
    merge of it alone is as safe as without the exchange. Once rank 1 (which
    applied rank 0's thresholds) is present, a missing rank 2 is unforceable."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=6, rows=20, n_q=10)
    out = tmp_path / "o"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, True)
    run_compute(cfg, num_jobs=3, job_rank=0)
    monkeypatch.setenv("NOVA_BF_MERGE_FORCE", "1")
    run_merge(cfg)
    assert glob.glob(os.path.join(str(out), "bf_*_dense_k*.parquet"))
    run_compute(cfg, num_jobs=3, job_rank=1)
    with pytest.raises(RuntimeError, match="threshold_exchange"):
        run_merge(cfg)


def test_merge_check_catches_inf_and_nan():
    """Safety-net edges (round 5): an applied +inf must be checked, not
    skipped, and a NaN score in the merged list must not mask a short k-th."""
    from nova_bf.merge import _check_exchange_batch, _exchange_active
    spec = types.SimpleNamespace(name="d", k=2)
    def lists(rows):
        return pa.array(rows, pa.list_(pa.float32()))
    z = np.zeros(1, np.int64)
    def bad(need, arr, nan=None, k=2):
        s = types.SimpleNamespace(name="d", k=k)
        r = _check_exchange_batch(s, np.asarray(need, np.float32),
                                  np.zeros(len(need), np.int64) if nan is None else np.asarray(nan, np.int64),
                                  0, arr)
        return r["rows"].tolist()
    assert bad([np.inf], lists([[0.5, 0.1]])) == [0]
    assert bad([0.9], lists([[np.nan, 0.1]])) == [0]
    assert bad([0.1, -np.inf], lists([[0.5, 0.1], [0.0]])) == []
    # A sliced ListArray: `values` is the whole child, offsets must be rebased.
    full = lists([[0.9, 0.55], [0.5, 0.1], [0.2, 0.0]])
    assert bad([0.5, 0.1], full.slice(0, 2)) == []
    assert bad([0.5, 0.2], full.slice(1, 2)) == [0, 1]   # both rows' minima fall short
    # Dropped NaN hits relax the rule to the top (k - N): k=3, one NaN dropped
    # -> 2 hits whose 2nd score clears the threshold pass; a 3rd below it is
    # irrelevant; only 1 hit, or a 2nd below the threshold, fails; N >= k
    # cannot be checked at all.
    assert bad([0.5], lists([[0.9, 0.6]]), nan=[1], k=3) == []
    assert bad([0.5], lists([[0.9, 0.6, 0.1]]), nan=[1], k=3) == []
    assert bad([0.5], lists([[0.9]]), nan=[1], k=3) == [0]
    assert bad([0.5], lists([[0.9, 0.4, 0.3]]), nan=[1], k=3) == [0]
    assert bad([0.5], lists([[0.9]]), nan=[3], k=3) == []
    assert not _exchange_active(np.array([-np.inf, np.nan], np.float32))
    assert _exchange_active(np.array([-np.inf, 0.0], np.float32))


def test_poller_drops_inf_entries(tmp_path):
    store = Store(str(tmp_path))
    ex = ThresholdExchange(store, RUN, 0, 2, {"d": (3, 1)}, refresh_s=100.0)
    try:
        root = tmp_path / ex.subdir
        root.mkdir(parents=True, exist_ok=True)
        (root / file_name(1)).write_bytes(_npz(d=np.array([[np.inf], [0.5], [np.nan]], np.float32)))
        ex.poll_once()
        _, lists, _ = ex.latest(0)
        assert lists is not None
        assert lists["d"][0].astype(np.float32).reshape(-1).tolist() == [-np.inf, 0.5, -np.inf]
    finally:
        ex.close(final=False)


def test_hash_array_handles_zero_size_and_dtype():
    import hashlib
    def hx(a):
        h = hashlib.sha256(); thresholds.hash_array(h, a); return h.hexdigest()
    hx(np.zeros((0, 8), np.float32)); hx(np.zeros((4, 0), np.float32))
    assert hx(np.zeros(4, np.float32)) != hx(np.zeros(4, np.float64))
    assert hx(np.zeros((2, 2), np.float32)) != hx(np.zeros(4, np.float32))
    assert hx(np.arange(4, dtype=np.float32)[::2]) == hx(np.array([0, 2], np.float32))


def test_merge_manifest_records_verification(tmp_path, monkeypatch):
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=4, rows=20, n_q=10)
    out = tmp_path / "o"; out.mkdir()
    _run_sharded(_cfg(cdir, qpath, out, True), 2)
    (m,) = [json.load(open(f)) for f in glob.glob(os.path.join(str(out), "_bf_manifest_*merge*", "*.json"))]
    (e,) = m["searches"]
    assert e["threshold_exchange_verified_queries"] > 0


def test_mixed_on_off_ranks_merge_and_verify_only_the_exchanging_ranks(tmp_path, monkeypatch):
    """A rank re-run with the exchange OFF pruned only against its own
    thresholds: it needs no applied record, and the merge is still exact."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=6, rows=30, n_q=20, seed=31)
    single_out = tmp_path / "single"; single_out.mkdir()
    single = {n: pq.read_table(p).to_pydict()
              for n, p in run_compute(_cfg(cdir, qpath, single_out, False), num_jobs=None).items()}
    out = tmp_path / "x"; out.mkdir()
    run_compute(_cfg(cdir, qpath, out, True), num_jobs=2, job_rank=0)
    run_compute(_cfg(cdir, qpath, out, False), num_jobs=2, job_rank=1)
    merged = {n: pq.read_table(p).to_pydict() for n, p in run_merge(_cfg(cdir, qpath, out, True)).items()}
    _assert_same(merged, single)


def _mk_modes(cdir, qpath, out, exchange, tiebreak):
    return BruteForceConfig(
        corpus=CorpusConfig(path=cdir, id_column="id"),
        queries=QueriesConfig(path=qpath, id_column="qid", payload_fields=["qset"]),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(dense_batch_size=16, threshold_exchange=exchange,
                            threshold_exchange_refresh_s=0.2, tiebreak=tiebreak),
        searches=[SearchSpec(name="cos", vector_type="dense", metric="cosine", k=3),
                  SearchSpec(name="euc", vector_type="dense", metric="euclidean", k=4),
                  SearchSpec(name="dot", vector_type="dense", metric="dot", k=2,
                             rows=RowSelector(column="qset", isin=["a"]))],
    )


@pytest.mark.parametrize("twopass", [False, True])
@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
@pytest.mark.parametrize("nj", [2, 3])
def test_metrics_tiebreaks_and_two_pass(tmp_path, monkeypatch, twopass, tiebreak, nj):
    """Cosine, euclidean (negative distances) and dot with a `rows` subset;
    both tie-break modes; the FP16 two-pass forced on (CPU) with dead rows
    poisoned so a wrongly pruned row cannot pass unnoticed."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    monkeypatch.setenv("NOVA_BF_POISON_DEAD_ROWS", "1")
    if twopass:
        monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
        monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
        monkeypatch.setenv("NOVA_BF_TWOPASS_FORCE_ENGAGE", "1")
    cdir, qpath = _dataset(tmp_path, n_files=9, rows=48, n_q=24, seed=5)
    s_out = tmp_path / "s"; s_out.mkdir()
    single = {n: pq.read_table(p).to_pydict()
              for n, p in run_compute(_mk_modes(cdir, qpath, s_out, False, tiebreak), num_jobs=None).items()}
    out = tmp_path / "x"; out.mkdir()
    cfg = _mk_modes(cdir, qpath, out, True, tiebreak)
    for r in range(nj):
        run_compute(cfg, num_jobs=nj, job_rank=r)
    merged = {n: pq.read_table(p).to_pydict() for n, p in run_merge(cfg).items()}
    _assert_same(merged, single)
    json.dumps(_manifests(str(out)))


def _concurrent_child(args):
    tmp, r, nj = args
    # Same device as the parent's reference run (a CPU child against a GPU
    # reference differs by fp32 GEMM rounding in the scores).
    os.environ["NOVA_BF_DEVICE"] = os.environ.get("NOVA_BF_TEST_DEVICE", "cpu")
    os.environ["NOVA_BF_POISON_DEAD_ROWS"] = "1"
    import torch
    from nova_bf import thresholds as _thr
    from nova_bf.compute import run_compute as _rc
    if os.environ["NOVA_BF_DEVICE"].startswith("cuda"):
        torch.zeros(1, device="cuda")          # CUDA init before the barrier
    # Start barrier: every rank constructs its exchange at about the same
    # time, so nobody can inherit at construction and the pickup is mid-scan.
    open(f"{tmp}/ready_{r}", "w").close()
    deadline = time.monotonic() + 120
    while not all(os.path.exists(f"{tmp}/ready_{i}") for i in range(nj)):
        if time.monotonic() > deadline:
            raise RuntimeError("barrier timeout")
        time.sleep(0.02)
    # Slow every file boundary (the publish runs once per file on every
    # reader path, CPU or GPU) so the scans genuinely interleave: 8 files at
    # 0.5 s each is ~4 s per rank against publishes >= 1 s apart polled every
    # 0.2 s. Without this a GPU box scans these files in ~1 s.
    orig_publish = _thr.ThresholdExchange.publish
    def slow_publish(self, arrays):
        time.sleep(0.5)
        return orig_publish(self, arrays)
    _thr.ThresholdExchange.publish = slow_publish
    _rc(_mk_modes(f"{tmp}/corpus", f"{tmp}/queries.parquet", f"{tmp}/x", True, "id"),
        num_jobs=nj, job_rank=r)


def test_concurrent_ranks_exchange_mid_scan(tmp_path, monkeypatch):
    """Ranks running at the SAME time in separate processes, picking up each
    other's thresholds mid-scan via the background poller."""
    import multiprocessing as mp
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    monkeypatch.setenv("NOVA_BF_POISON_DEAD_ROWS", "1")
    cdir, qpath = _dataset(tmp_path, n_files=24, rows=48, n_q=24, seed=9)
    (tmp_path / "s").mkdir(); (tmp_path / "x").mkdir()
    single = {n: pq.read_table(p).to_pydict()
              for n, p in run_compute(_mk_modes(cdir, qpath, tmp_path / "s", False, "id"), num_jobs=None).items()}
    nj = 3
    with mp.get_context("spawn").Pool(nj) as pool:
        pool.map(_concurrent_child, [(str(tmp_path), r, nj) for r in range(nj)])
    cfg = _mk_modes(cdir, qpath, tmp_path / "x", True, "id")
    merged = {n: pq.read_table(p).to_pydict() for n, p in run_merge(cfg).items()}
    _assert_same(merged, single)
    stats = [m["params"]["threshold_exchange_stats"] for m in _manifests(str(tmp_path / "x"))]
    assert sum(s["versions_applied"] for s in stats) >= 1
    # All ranks started together, so nobody could inherit at construction:
    # the pickup must have happened MID-SCAN (file index >= 0).
    assert any((s.get("first_applied_file") or -1) >= 0 for s in stats), stats


def test_nan_corpus_row_does_not_poison_peers(tmp_path, monkeypatch):
    """Round 6: a NaN corpus row sorts into a rank's top-k slots, so its
    k-th best is no lower bound on the global k-th REAL score, and the
    partial writer drops the NaN hit. The rank must publish -inf for such
    queries; otherwise peers prune true neighbours and the merge (correctly)
    refuses, unforceably, with no re-run that helps."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=4, rows=20, n_q=10)
    f0 = os.path.join(cdir, "part0.parquet")          # rank 0's first file
    t = pq.read_table(f0).to_pydict()
    t["dense_embedding"][3] = [float("nan")] * DIM
    pq.write_table(pa.table({
        "dense_embedding": pa.array(t["dense_embedding"], pa.list_(pa.float32())),
        "id": pa.array(t["id"]), "lang": pa.array(t["lang"]),
    }), f0)
    ref = tmp_path / "ref"; ref.mkdir()
    single = {n: pq.read_table(p).to_pydict()
              for n, p in run_compute(_cfg(cdir, qpath, ref, False), num_jobs=None).items()}
    out = tmp_path / "o"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, True)
    run_compute(cfg, num_jobs=2, job_rank=0)
    run_compute(cfg, num_jobs=2, job_rank=1)          # inherits rank 0's thresholds
    merged = {n: pq.read_table(p).to_pydict() for n, p in run_merge(cfg).items()}
    # The merged result equals the single run's: the NaN hit is dropped and
    # the list is cut to the top (k - 1) rather than padded with leftovers.
    _assert_same(merged, single)


def test_verification_failure_names_the_rank_at_fault(tmp_path, monkeypatch):
    """Round 7: a failing merge must name the rank(s) whose applied
    threshold the result does not clear, and re-running ONLY those (exchange
    off) must then merge; the other partials stay valid."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=4, rows=20, n_q=10)
    ref = tmp_path / "ref"; ref.mkdir()
    single = {n: pq.read_table(p).to_pydict()
              for n, p in run_compute(_cfg(cdir, qpath, ref, False), num_jobs=None).items()}
    out = tmp_path / "o"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, True)
    run_compute(cfg, num_jobs=2, job_rank=0)
    run_compute(cfg, num_jobs=2, job_rank=1)
    (p1,) = glob.glob(os.path.join(str(out), "**", "rank001.parquet"), recursive=True)
    t = pq.read_table(p1)
    col = t.column("applied_threshold").to_numpy().copy()
    col[0] = 50.0                                   # not a lower bound for query 0
    t = t.set_column(t.schema.get_field_index("applied_threshold"), "applied_threshold",
                     pa.array(col, pa.float32()))
    pq.write_table(t, p1)
    with pytest.raises(RuntimeError, match=r"re-run ONLY rank\(s\) \[1\]"):
        run_merge(cfg)
    run_compute(_cfg(cdir, qpath, out, False), num_jobs=2, job_rank=1)
    merged = {n: pq.read_table(p).to_pydict() for n, p in run_merge(cfg).items()}
    _assert_same(merged, single)


def test_late_nan_row_exempts_queries_instead_of_refusing(tmp_path, monkeypatch, caplog):
    """Round 7: rank 0 published a finite k-th best, THEN scored a NaN row
    that displaced a real hit (the writer drops the NaN, so its list is
    short) while rank 1 had already pruned against the published value. The
    merge must not refuse: rank 0's partial records the dropped NaN hits and
    those queries are exempted from verification, with a warning. Rank 0's
    intermediate publish is reproduced by rewriting its exchange file after
    it finished (its final publish was -inf for every query)."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=4, rows=20, n_q=10)
    # Rank 0's second file (part2) gets the NaN row, so file 0 publishes finite.
    f2 = os.path.join(cdir, "part2.parquet")
    t = pq.read_table(f2).to_pydict()
    t["dense_embedding"][0] = [float("nan")] * DIM
    pq.write_table(pa.table({
        "dense_embedding": pa.array(t["dense_embedding"], pa.list_(pa.float32())),
        "id": pa.array(t["id"]), "lang": pa.array(t["lang"]),
    }), f2)
    ref = tmp_path / "ref"; ref.mkdir()
    single = {n: pq.read_table(p).to_pydict()
              for n, p in run_compute(_cfg(cdir, qpath, ref, False), num_jobs=None).items()}
    out = tmp_path / "o"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, True)
    run_compute(cfg, num_jobs=2, job_rank=0)
    (p0,) = glob.glob(os.path.join(str(out), "**", "rank000.parquet"), recursive=True)
    t0 = pq.read_table(p0)
    assert (t0.column("nan_hits").to_numpy() > 0).all()
    (thr0,) = glob.glob(os.path.join(str(out), thresholds.SUBDIR, "*", file_name(0)))
    pub = np.load(thr0)[thresholds._KEY_PREFIX + "dense"].astype(np.float32)
    assert pub.shape == (10, thresholds.publish_width(K, 2)) and np.isfinite(pub).any()
    # The intermediate publish: rank 0's real k-th best after file 0 (a valid
    # lower bound at the time), which rank 1 will now prune against.
    # Rank 0's list as the single run's (k-1) real best per query padded to
    # width k (the published width at 2 ranks), which rank 1 will union with.
    w = thresholds.publish_width(K, 2)
    lst = np.full((len(single["dense"]["hit_scores"]), w), -np.inf, np.float32)
    for i, s in enumerate(single["dense"]["hit_scores"]):
        lst[i, :min(w, len(s))] = sorted(s, reverse=True)[:w]
    buf = _io.BytesIO()
    np.savez(buf, seq=np.int64(time.time_ns() // 1_000_000 + 10 ** 9),
             **{thresholds._KEY_PREFIX + "dense": thresholds.fp16_floor(lst)})
    open(thr0, "wb").write(buf.getvalue())
    run_compute(cfg, num_jobs=2, job_rank=1)
    (p1,) = glob.glob(os.path.join(str(out), "**", "rank001.parquet"), recursive=True)
    assert np.isfinite(pq.read_table(p1).column("applied_threshold").to_numpy()).all()
    with caplog.at_level("WARNING"):
        merged = {n: pq.read_table(p).to_pydict() for n, p in run_merge(cfg).items()}
    assert any("NaN-scored corpus hits dropped" in r.getMessage() for r in caplog.records)
    _assert_same(merged, single)
    # Those queries were still verified (top k-1), not skipped wholesale.
    (m,) = [json.load(open(f)) for f in glob.glob(os.path.join(str(out), "_bf_manifest_*merge*", "*.json"))]
    assert m["searches"][0]["threshold_exchange_verified_queries"] == 10


def test_nan_row_does_not_switch_off_verification(tmp_path, monkeypatch):
    """Round 8: one NaN corpus row scores NaN for EVERY query, so a blanket
    exemption would disable the check wholesale. With stale thresholds from
    an earlier run (same names, same sizes) and a NaN row, the merge must
    still refuse."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=4, rows=20, n_q=10, seed=1)
    out = tmp_path / "o"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, True)
    # Run A, with a NaN row on rank 0 from the start (so both runs look alike).
    f0 = os.path.join(cdir, "part0.parquet")
    def _nanify(path, row):
        t = pq.read_table(path).to_pydict()
        t["dense_embedding"][row] = [float("nan")] * DIM
        pq.write_table(pa.table({
            "dense_embedding": pa.array(t["dense_embedding"], pa.list_(pa.float32())),
            "id": pa.array(t["id"]), "lang": pa.array(t["lang"]),
        }), path)
    _nanify(f0, 5)
    run_compute(cfg, num_jobs=2, job_rank=0)
    run_compute(cfg, num_jobs=2, job_rank=1)
    # Regenerate the corpus in place (same names/sizes: same exchange id) with
    # much weaker vectors, keep the NaN row, and delete the partials only.
    rng = np.random.default_rng(7)
    for fi in range(4):
        path = os.path.join(cdir, f"part{fi}.parquet")
        t = pq.read_table(path).to_pydict()
        v = (rng.standard_normal((20, DIM)) * 0.01).astype(np.float32)
        t["dense_embedding"] = v.tolist()
        pq.write_table(pa.table({
            "dense_embedding": pa.array(t["dense_embedding"], pa.list_(pa.float32())),
            "id": pa.array(t["id"]), "lang": pa.array(t["lang"]),
        }), path)
    _nanify(f0, 5)
    for d in glob.glob(os.path.join(str(out), "_bf_partial_*")):
        shutil.rmtree(d)
    # Make the stale (run A) thresholds bite: a fresh query set would not be
    # scored by the same vectors, so just force the weak corpus under the old
    # files -- cosine scores of random vectors are ~N(0, 1/sqrt(DIM)) either
    # way, but run A's file holds run A's k-th best which the regenerated
    # corpus must now clear with different rows; the lists come out wrong.
    run_compute(cfg, num_jobs=2, job_rank=0)
    run_compute(cfg, num_jobs=2, job_rank=1)
    monkeypatch.setenv("NOVA_BF_MERGE_FORCE", "1")
    with pytest.raises(RuntimeError, match="VERIFICATION FAILED"):
        run_merge(cfg)


def test_remedy_rerun_of_nan_rank_with_exchange_off_still_merges(tmp_path, monkeypatch):
    """Round 9: the NaN rank re-run with the exchange off (the merge's own
    remedy) carries nan_hits but no applied_threshold; its counts must still
    relax the check, or a correct result is refused and an innocent rank
    blamed."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=4, rows=20, n_q=10)
    f2 = os.path.join(cdir, "part2.parquet")
    t = pq.read_table(f2).to_pydict()
    t["dense_embedding"][0] = [float("nan")] * DIM
    pq.write_table(pa.table({
        "dense_embedding": pa.array(t["dense_embedding"], pa.list_(pa.float32())),
        "id": pa.array(t["id"]), "lang": pa.array(t["lang"]),
    }), f2)
    ref = tmp_path / "ref"; ref.mkdir()
    single = {n: pq.read_table(p).to_pydict()
              for n, p in run_compute(_cfg(cdir, qpath, ref, False), num_jobs=None).items()}
    out = tmp_path / "o"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, True)
    run_compute(cfg, num_jobs=2, job_rank=0)
    (thr0,) = glob.glob(os.path.join(str(out), thresholds.SUBDIR, "*", file_name(0)))
    # Rank 0's list as the single run's (k-1) real best per query padded to
    # width k (the published width at 2 ranks), which rank 1 will union with.
    w = thresholds.publish_width(K, 2)
    lst = np.full((len(single["dense"]["hit_scores"]), w), -np.inf, np.float32)
    for i, s in enumerate(single["dense"]["hit_scores"]):
        lst[i, :min(w, len(s))] = sorted(s, reverse=True)[:w]
    buf = _io.BytesIO()
    np.savez(buf, seq=np.int64(time.time_ns() // 1_000_000 + 10 ** 9),
             **{thresholds._KEY_PREFIX + "dense": thresholds.fp16_floor(lst)})
    open(thr0, "wb").write(buf.getvalue())
    run_compute(cfg, num_jobs=2, job_rank=1)
    run_merge(cfg)
    # Now the remedy re-run of the NaN rank, exchange OFF.
    run_compute(_cfg(cdir, qpath, out, False), num_jobs=2, job_rank=0)
    (p0,) = glob.glob(os.path.join(str(out), "**", "rank000.parquet"), recursive=True)
    names = pq.read_table(p0).column_names
    assert "nan_hits" in names and "applied_threshold" not in names
    merged = {n: pq.read_table(p).to_pydict() for n, p in run_merge(cfg).items()}
    _assert_same(merged, single)


def test_truncate_lists_unit():
    from nova_bf.merge import _truncate_lists
    arr = pa.array([[1, 2, 3], [], [4, 5], [6]], pa.list_(pa.int64()))
    out = _truncate_lists(arr, np.array([2, 3, 0, 1]))
    assert out.to_pylist() == [[1, 2], [], [], [6]]
    assert _truncate_lists(arr, np.array([3, 3, 3, 3])).to_pylist() == arr.to_pylist()
    sl = arr.slice(1, 3)
    assert _truncate_lists(sl, np.array([0, 1, 0])).to_pylist() == [[], [4], []]


def test_exchange_setup_failure_degrades(tmp_path, monkeypatch):
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=4, rows=20, n_q=10)
    out = tmp_path / "o"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, True)
    def boom(*a, **k):
        raise PermissionError("denied")
    monkeypatch.setattr(thresholds, "ThresholdExchange", boom)
    run_compute(cfg, num_jobs=2, job_rank=0)
    run_compute(cfg, num_jobs=2, job_rank=1)
    for m in _manifests(str(out)):
        st = m["params"]["threshold_exchange_stats"]
        assert st["enabled"] is False and st["configured"] is True
        assert st["reason"].startswith("setup failed")
    run_merge(cfg)


@pytest.mark.parametrize("exch", [False, True])
@pytest.mark.parametrize("nanmode", ["query", "corpus"])
def test_multi_batch_merge_mixes_cut_and_uncut_batches(tmp_path, monkeypatch, exch, nanmode):
    """Round 10: with `merge_batch_size` small, batches with and without
    NaN-truncated rows share one output file (same list types), exchange on
    or off, across all three searches; the result equals the single-node run."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=4, rows=20, n_q=10)
    if nanmode == "query":
        t = pq.read_table(qpath).to_pydict(); t["dense_embedding"][7] = [float("nan")] * DIM
        pq.write_table(pa.table({"dense_embedding": pa.array(t["dense_embedding"], pa.list_(pa.float32())),
                                 "qid": pa.array(t["qid"]), "qset": pa.array(t["qset"])}), qpath)
    else:
        f = os.path.join(cdir, "part2.parquet"); t = pq.read_table(f).to_pydict()
        t["dense_embedding"][0] = [float("nan")] * DIM
        pq.write_table(pa.table({"dense_embedding": pa.array(t["dense_embedding"], pa.list_(pa.float32())),
                                 "id": pa.array(t["id"]), "lang": pa.array(t["lang"])}), f)
    ref = tmp_path / "ref"; ref.mkdir()
    single = {n: pq.read_table(p).to_pydict()
              for n, p in run_compute(_cfg(cdir, qpath, ref, False, multi=True), num_jobs=None).items()}
    out = tmp_path / "o"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, exch, multi=True)
    cfg.params.merge_batch_size = 3
    merged = _run_sharded(cfg, 3)
    _assert_same(merged, single)
    ms = [json.load(open(f)) for f in glob.glob(os.path.join(str(out), "_bf_manifest_*merge*", "*.json"))]
    cut = {e["name"]: e.get("nan_truncated_queries", 0) for m in ms for e in m["searches"]}
    if nanmode == "corpus":
        assert all(v > 0 for v in cut.values()), cut

