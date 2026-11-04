"""Cross-worker threshold exchange (nova_bf.thresholds).

Three layers:
  * the exchange object on a local store: publish/poll, the union fold, the
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


def _thr(seq=1, **arrays) -> bytes:
    """A peer file in the exchange's raw format (arrays cast to float16 as
    is: NaN/inf survive, which is what the tests want to feed in)."""
    return thresholds.encode(seq, {k: np.asarray(v).astype(np.float16) for k, v in arrays.items()})


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
        # Every hand-over is written (no content comparison): an identical
        # re-publish is a new seq, folded as a new version.
        a.publish(la)
        assert _wait_for(lambda: a.stats["publishes"] == 2)
        assert b.poll_once() == 1
        v3, _, changed_same = b.latest(v)
        assert changed_same and v3 == v + 1
        # A changed one REPLACES the peer's list.
        la2 = {"dense": np.full((5, 2), -np.inf, np.float32), "sparse": np.zeros((2, 1), np.float32)}
        a.publish(la2)
        assert _wait_for(lambda: a.stats["publishes"] == 3)
        assert b.poll_once() == 1
        _, lists3, changed3 = b.latest(v3)
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
        (tmp_path / b.subdir / file_name(0)).write_bytes(_thr(dense=arr))
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
        (root / file_name(2)).write_bytes(_thr(dense=np.ones((7, 2), np.float16)))  # wrong height
        (root / file_name(7)).write_bytes(_thr(dense=np.full((3, 2), 9.0, np.float16)))  # rank >= num_jobs
        assert b.poll_once() == 0
        assert b.stats["peer_files_skipped"] == 2
        assert b.latest(0) == (0, None, False)
        nan_inf = np.array([[np.nan, 1.0], [2.0, np.inf], [np.nan, -np.inf]], np.float32)
        (root / file_name(0)).write_bytes(_thr(dense=nan_inf))
        assert b.poll_once() == 1
        _, lists, _ = b.latest(0)
        # Stored raw (no host-side sanitisation); the union treats every
        # non-finite peer entry as -inf on the device.
        np.testing.assert_array_equal(lists["dense"][0].astype(np.float32), nan_inf)
        import torch
        own = torch.full((3, 1), float("-inf"))
        assert thresholds.union_kth(own, lists["dense"], 1, "cpu").tolist() == [1.0, 2.0, -np.inf]
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
    # Compute-thread time in the exchange hooks is recorded (per-file publish
    # + apply, the construction-time apply, the final publish).
    for x in (x0, x1):
        assert x["publish_calls"] >= 2 and x["apply_calls"] >= 2
        assert 0.0 <= x["apply_seconds_max"] <= x["apply_seconds_total"]
        assert 0.0 <= x["publish_seconds_max"] <= x["publish_seconds_total"]
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


def test_the_exchange_and_trimming_are_on_by_default():
    from nova_bf.config import ParamsConfig
    p = ParamsConfig()
    assert p.threshold_exchange is True and p.trim_partials is True


@pytest.mark.parametrize("num_jobs", [None, 1])
def test_single_worker_turns_the_exchange_off_quietly(tmp_path, monkeypatch, caplog, num_jobs):
    """On by default, so a lone worker is the NORMAL case: the exchange turns
    itself off with an info line, never a warning, and the run completes."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=2, rows=10, n_q=4)
    out = tmp_path / "o"; out.mkdir()
    with caplog.at_level("INFO"):
        run_compute(_cfg(cdir, qpath, out, True), num_jobs=num_jobs,
                    **({"job_rank": 0} if num_jobs else {}))
    assert any("threshold exchange off" in r.message and "single-worker" in r.message
               and r.levelname == "INFO" for r in caplog.records)
    assert not any(r.levelname == "WARNING" and "exchange" in r.message.lower()
                   for r in caplog.records)
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


def test_peer_inf_and_nan_never_raise_the_bound(tmp_path):
    """A peer's +inf or NaN reaches `union_kth` raw and must count as -inf
    there (device-side sanitisation): never above any finite candidate,
    whatever k, and the poller's arrays are left untouched."""
    import torch
    store = Store(str(tmp_path))
    ex = ThresholdExchange(store, RUN, 0, 3, {"d": (4, 2)}, refresh_s=100.0)
    try:
        root = tmp_path / ex.subdir
        root.mkdir(parents=True, exist_ok=True)
        p1 = np.array([[np.inf, np.inf], [0.5, np.nan], [np.nan, np.nan], [np.inf, 0.25]], np.float32)
        p2 = np.array([[0.75, -np.inf], [np.inf, 0.125], [-np.inf, -np.inf], [np.nan, np.inf]], np.float32)
        (root / file_name(1)).write_bytes(_thr(d=p1))
        (root / file_name(2)).write_bytes(_thr(d=p2))
        assert ex.poll_once() == 2
        _, lists, _ = ex.latest(0)
        before = [a.copy() for a in lists["d"]]
        own = torch.tensor([[0.375, -np.inf], [np.nan, -np.inf], [0.0625, 0.0], [-np.inf, -np.inf]])
        # Finite union per row: {.75,.375} {.5,.125} {.0625,0} {.25}.
        assert thresholds.union_kth(own, lists["d"], 1, "cpu").tolist() == [0.75, 0.5, 0.0625, 0.25]
        assert thresholds.union_kth(own, lists["d"], 2, "cpu").tolist() == [0.375, 0.125, 0.0, -np.inf]
        assert thresholds.union_kth(own, lists["d"], 2, "cpu", chunk=1).tolist() == \
            thresholds.union_kth(own, lists["d"], 2, "cpu").tolist()
        for a, b in zip(lists["d"], before):
            np.testing.assert_array_equal(a, b)            # raw views untouched
    finally:
        ex.close(final=False)


def test_file_format_round_trip_multiple_searches_odd_shapes():
    rng = np.random.default_rng(0)
    arrays = {"a": rng.standard_normal((7, 3)).astype(np.float16),
              "bé": rng.standard_normal((1, 1)).astype(np.float16),
              "c": rng.standard_normal((13, 5)).astype(np.float16),
              "empty": np.zeros((0, 4), np.float16)}
    arrays["c"][2, 1] = np.nan; arrays["c"][3, 0] = np.inf
    raw = thresholds.encode(123456789012, arrays)
    seq, got = thresholds.decode(raw, {n: a.shape for n, a in arrays.items()})
    assert seq == 123456789012
    for n, a in arrays.items():
        assert got[n].shape == a.shape and got[n].dtype == np.float16
        np.testing.assert_array_equal(got[n], a)                 # NaN/inf bits kept too
        if a.size:
            off = got[n].__array_interface__["data"][0] - np.frombuffer(raw, np.uint8).__array_interface__["data"][0]
            assert off % thresholds.ALIGN == 0
    # The streamed parts write the same bytes as `encode`.
    import pyarrow as pa_
    header, parts = thresholds.encode_header(123456789012, arrays)
    sink = pa_.BufferOutputStream()
    thresholds.write_stream(sink, header, parts)
    assert sink.getvalue().to_pybytes() == raw
    # Zero-copy views of a pyarrow Buffer (what the poller reads into).
    seq2, got2 = thresholds.decode(pa_.py_buffer(raw), {n: a.shape for n, a in arrays.items()})
    np.testing.assert_array_equal(got2["c"], arrays["c"])


def test_header_validation_rejects_bad_files(tmp_path):
    import struct
    sizes = {"d": (3, 2), "e": (2, 1)}
    good = thresholds.encode(5, {"d": np.ones((3, 2)), "e": np.zeros((2, 1))})
    assert thresholds.decode(good, sizes)[0] == 5
    bad = {
        "magic": b"XXXXXXXX" + good[8:],
        "version": good[:8] + struct.pack("<I", 99) + good[12:],
        "truncated": good[:-1],
        "appended": good + b"\0",
        "header only": good[:20],
        "empty": b"",
        "wrong shape": thresholds.encode(5, {"d": np.ones((3, 3)), "e": np.zeros((2, 1))}),
        "wrong name": thresholds.encode(5, {"d": np.ones((3, 2)), "x": np.zeros((2, 1))}),
        "missing search": thresholds.encode(5, {"d": np.ones((3, 2))}),
        "extra search": thresholds.encode(5, {"d": np.ones((3, 2)), "e": np.zeros((2, 1)),
                                              "f": np.zeros((1, 1))}),
    }
    for why, raw in bad.items():
        with pytest.raises(ValueError):
            thresholds.decode(raw, sizes)
    # Through the poller: each is skipped (counted), nothing folded.
    store = Store(str(tmp_path))
    ex = ThresholdExchange(store, RUN, 0, 2, sizes, refresh_s=100.0)
    try:
        path = tmp_path / ex.subdir / file_name(1)
        for i, raw in enumerate(bad.values()):
            path.write_bytes(raw)
            os.utime(path, ns=(10 ** 9 * (i + 1), 10 ** 9 * (i + 1)))   # a new listing sig each time
            assert ex.poll_once() == 0
        assert ex.stats["peer_files_skipped"] == len(bad)
        assert ex.latest(0) == (0, None, False)
        path.write_bytes(good)
        assert ex.poll_once() == 1
    finally:
        ex.close(final=False)


def test_change_detection_is_by_seq(tmp_path):
    """Same seq is not refolded (even with new bytes and a new mtime), an
    older seq is ignored, a newer one is folded as a new version; an
    unchanged listing signature costs no read at all."""
    store = Store(str(tmp_path))
    ex = ThresholdExchange(store, RUN, 0, 2, {"d": (2, 1)}, refresh_s=100.0)
    try:
        path = tmp_path / ex.subdir / file_name(1)
        def put(seq, val, t):
            path.write_bytes(_thr(seq=seq, d=np.full((2, 1), val)))
            os.utime(path, ns=(t * 10 ** 9, t * 10 ** 9))
        put(10, 1.0, 1)
        assert ex.poll_once() == 1
        v, lists, _ = ex.latest(0)
        assert v == 1 and lists["d"][0].tolist() == [[1.0], [1.0]]
        reads = ex.stats["peer_files_read"]
        assert ex.poll_once() == 0 and ex.stats["peer_files_stale"] == 0   # same sig: no GET
        put(10, 2.0, 2)                    # same seq, different bytes: not refolded
        assert ex.poll_once() == 0 and ex.stats["peer_files_stale"] == 1
        put(9, 3.0, 3)                     # older object won a racing write
        assert ex.poll_once() == 0 and ex.stats["peer_files_stale"] == 2
        v2, lists2, changed = ex.latest(v)
        assert v2 == v and not changed and lists2["d"][0].tolist() == [[1.0], [1.0]]
        put(11, 4.0, 4)                    # newer: folded, new version
        assert ex.poll_once() == 1
        v3, lists3, changed3 = ex.latest(v)
        assert v3 == v + 1 and changed3 and lists3["d"][0].tolist() == [[4.0], [4.0]]
        assert ex.stats["peer_files_read"] == reads + 1
    finally:
        ex.close(final=False)


def test_publisher_seq_is_monotone_and_writes_are_atomic(tmp_path):
    """The written file carries an increasing seq (starting at the wall
    clock, so a re-run of the rank outranks the first attempt) and no tmp
    file is left behind."""
    store = Store(str(tmp_path))
    sizes = {"d": (3, 2)}
    a = ThresholdExchange(store, RUN, 0, 2, sizes, refresh_s=100, publish_min_interval_s=0)
    try:
        path = tmp_path / a.subdir / file_name(0)
        seqs = []
        for i in range(3):
            a.publish({"d": np.full((3, 2), float(i), np.float32)})
            assert _wait_for(lambda: a.stats["publishes"] == i + 1)
            seq, got = thresholds.decode(path.read_bytes(), sizes)
            seqs.append(seq)
            assert got["d"].tolist() == [[float(i)] * 2] * 3
        assert seqs == sorted(seqs) and len(set(seqs)) == 3
        assert seqs[0] > time.time_ns() // 1_000_000 - 60_000
        assert sorted(os.listdir(tmp_path / a.subdir)) == [file_name(0)]
    finally:
        a.close(final=False)


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


@pytest.mark.parametrize("twopass", [False, "slice", "tiles"])
@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
@pytest.mark.parametrize("nj", [2, 3])
def test_metrics_tiebreaks_and_two_pass(tmp_path, monkeypatch, twopass, tiebreak, nj):
    """Cosine, euclidean (negative distances) and dot with a `rows` subset;
    both tie-break modes; the FP16 two-pass forced on (CPU) with dead rows
    poisoned so a wrongly pruned row cannot pass unnoticed -- in both pass-two
    modes (tiles mode compared near-tie-aware: its scores may move an ULP)."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    monkeypatch.setenv("NOVA_BF_POISON_DEAD_ROWS", "1")
    if twopass:
        monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
        monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
        monkeypatch.setenv("NOVA_BF_TWOPASS_FORCE_ENGAGE", "1")
        monkeypatch.setenv("NOVA_BF_TWOPASS_RESCORE", twopass)
        if twopass == "tiles":
            from nova_bf import twopass as _tp
            monkeypatch.setenv("NOVA_BF_TWOPASS_TILE_THRESHOLD", "1.0")
            monkeypatch.setattr(_tp, "CPU_TILE", 4)
            monkeypatch.setattr(_tp, "TILE_CROSSOVER", 1.0)
    cdir, qpath = _dataset(tmp_path, n_files=9, rows=48, n_q=24, seed=5)
    s_out = tmp_path / "s"; s_out.mkdir()
    single = {n: pq.read_table(p).to_pydict()
              for n, p in run_compute(_mk_modes(cdir, qpath, s_out, False, tiebreak), num_jobs=None).items()}
    out = tmp_path / "x"; out.mkdir()
    cfg = _mk_modes(cdir, qpath, out, True, tiebreak)
    for r in range(nj):
        run_compute(cfg, num_jobs=nj, job_rank=r)
    merged = {n: pq.read_table(p).to_pydict() for n, p in run_merge(cfg).items()}
    if twopass == "tiles":
        from twopass_compare import assert_same_results, norm_product, tolerance
        cols = ("query_id", "hit_ids", "hit_scores")
        # The gamma_d bound at the dot search's B = max|q|max|c| (>= cosine's 1).
        tol = tolerance(DIM, max(1.0, norm_product(cdir, qpath)))
        for name in single:
            assert_same_results({c: merged[name][c] for c in cols},
                                {c: single[name][c] for c in cols},
                                abs_tol=tol, label=name)
    else:
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


def _run_spawned(fn, arg_list, timeout=600.0):
    """Run ``fn(arg)`` per arg, each in its own spawned process, and fail
    (never hang) if any child dies or hangs. Not `multiprocessing.Pool.map`:
    a child that dies WITHOUT raising (native abort, OOM kill, os._exit)
    leaves `Pool.map` waiting forever."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=fn, args=(a,), daemon=True) for a in arg_list]
    for p in procs:
        p.start()
    deadline = time.monotonic() + timeout
    try:
        for p in procs:
            p.join(max(0.0, deadline - time.monotonic()))
        codes = [p.exitcode for p in procs]
        assert all(c == 0 for c in codes), f"child exit codes {codes} (None = hung)"
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
                p.join(5)


def _dead_child(code):
    os._exit(code)


def test_spawn_helper_fails_fast_when_a_child_dies_silently():
    t0 = time.monotonic()
    with pytest.raises(AssertionError, match="exit codes"):
        _run_spawned(_dead_child, [0, 3], timeout=120)
    assert time.monotonic() - t0 < 100


def test_concurrent_ranks_exchange_mid_scan(tmp_path, monkeypatch):
    """Ranks running at the SAME time in separate processes, picking up each
    other's thresholds mid-scan via the background poller."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    monkeypatch.setenv("NOVA_BF_POISON_DEAD_ROWS", "1")
    cdir, qpath = _dataset(tmp_path, n_files=24, rows=48, n_q=24, seed=9)
    (tmp_path / "s").mkdir(); (tmp_path / "x").mkdir()
    single = {n: pq.read_table(p).to_pydict()
              for n, p in run_compute(_mk_modes(cdir, qpath, tmp_path / "s", False, "id"), num_jobs=None).items()}
    nj = 3
    _run_spawned(_concurrent_child, [(str(tmp_path), r, nj) for r in range(nj)])
    cfg = _mk_modes(cdir, qpath, tmp_path / "x", True, "id")
    merged = {n: pq.read_table(p).to_pydict() for n, p in run_merge(cfg).items()}
    _assert_same(merged, single)
    stats = [m["params"]["threshold_exchange_stats"] for m in _manifests(str(tmp_path / "x"))]
    assert sum(s["versions_applied"] for s in stats) >= 1
    # All ranks started together, so nobody could inherit at construction:
    # the pickup must have happened MID-SCAN (file index >= 0).
    assert any(isinstance(s.get("first_applied_file"), int) and s["first_applied_file"] >= 0
               for s in stats), stats


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
    _, pub_d = thresholds.decode(open(thr0, "rb").read(), {"dense": (10, thresholds.publish_width(K, 2))})
    pub = pub_d["dense"].astype(np.float32)
    assert pub.shape == (10, thresholds.publish_width(K, 2)) and np.isfinite(pub).any()
    # The intermediate publish: rank 0's real k-th best after file 0 (a valid
    # lower bound at the time), which rank 1 will now prune against.
    # Rank 0's list as the single run's (k-1) real best per query padded to
    # width k (the published width at 2 ranks), which rank 1 will union with.
    w = thresholds.publish_width(K, 2)
    lst = np.full((len(single["dense"]["hit_scores"]), w), -np.inf, np.float32)
    for i, s in enumerate(single["dense"]["hit_scores"]):
        lst[i, :min(w, len(s))] = sorted(s, reverse=True)[:w]
    open(thr0, "wb").write(thresholds.encode(time.time_ns() // 1_000_000 + 10 ** 9,
                                             {"dense": thresholds.fp16_floor(lst)}))
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
    open(thr0, "wb").write(thresholds.encode(time.time_ns() // 1_000_000 + 10 ** 9,
                                             {"dense": thresholds.fp16_floor(lst)}))
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



def test_union_kth_never_mutates_float32_peer(tmp_path):
    import torch
    from nova_bf import thresholds as T
    peer = np.array([[np.inf, 0.0, np.nan, 0.0]], dtype=np.float32)
    before = peer.copy()
    own = torch.tensor([[1.0, 0.5]], dtype=torch.float32)
    T.union_kth(own, [peer], 2, "cpu")
    np.testing.assert_array_equal(peer, before)


def test_encode_rounds_non_float16_scores_down_never_up():
    """`encode` must not be an unsafe shortcut past `publish`: a float32
    score not exactly representable in float16 is written rounded toward
    -inf (round-to-nearest rounds about half of them UP, above the true
    score, which is no longer a lower bound). Float16 input is written
    untouched, NaN/inf bits included."""
    from nova_bf import thresholds as T
    rng = np.random.default_rng(1)
    a = rng.uniform(-2, 2, (64, 8)).astype(np.float32)
    _, got = T.decode(T.encode(3, {"s": a}), {"s": a.shape})
    assert (got["s"].astype(np.float32) <= a).all(), "a score was rounded up"
    np.testing.assert_array_equal(got["s"], T.fp16_floor(a))
    h = np.array([[np.nan, np.inf, 0.5]], np.float16)
    _, got = T.decode(T.encode(4, {"s": h}), {"s": h.shape})
    np.testing.assert_array_equal(got["s"], h)


def test_decode_rejects_overlapping_arrays():
    from nova_bf import thresholds as T
    import struct
    a = np.zeros((2, 4), np.float16)
    raw = bytearray(T.encode(7, {"s": a, "t": a}))
    sizes = {"s": (2, 4), "t": (2, 4)}
    T.decode(bytes(raw), sizes)                       # well-formed: accepted
    # Point t's offset at s's payload.
    hdr = T._FIXED.size
    s_entry = hdr + T._NAME_LEN.size + 1
    s_off = struct.unpack_from("<Q", raw, s_entry + 16)[0]
    t_entry = s_entry + T._ENTRY.size + T._NAME_LEN.size + 1
    struct.pack_into("<Q", raw, t_entry + 16, s_off)
    with pytest.raises(ValueError, match="overlapping"):
        T.decode(bytes(raw), sizes)


def test_bad_peer_file_is_not_redownloaded(tmp_path):
    from nova_bf import thresholds as T
    store = Store(str(tmp_path))
    ex = T.ThresholdExchange(store, "e" * 64, 0, 2, {"s": (3, 2)}, refresh_s=3600)
    try:
        good = T.encode(5, {"s": np.zeros((3, 2), np.float16)})
        path = tmp_path / T.subdir_for("e" * 64) / T.file_name(1)
        path.write_bytes(good[:-4])                   # truncated object
        ex.poll_once(); ex.poll_once()
        assert ex.stats["peer_files_skipped"] == 1    # read once, not every poll
        path.write_bytes(good)
        os.utime(path, ns=(10**18, 10**18))           # a new object: new signature
        assert ex.poll_once() == 1
    finally:
        ex.close(final=False)


# ------------------------------------------------- concurrent poll-round fetches

class _GatedFS:
    """Delegates to a real pyarrow filesystem; `open_input_file` can be held
    per peer rank (`hold[rank]` = an Event the GET waits on), made to fail
    (`fail` = set of ranks, one failure each), and tracks how many GETs run
    at once."""

    def __init__(self, fs):
        self._fs = fs
        self.hold: dict[int, "threading.Event"] = {}
        self.fail: set[int] = set()
        self.lock = __import__("threading").Lock()
        self.active = 0
        self.max_active = 0
        self.started: list[int] = []
        self.done: list[int] = []

    def __getattr__(self, name):
        return getattr(self._fs, name)

    def open_input_file(self, path):
        import re
        m = re.search(r"thr_rank(\d+)\.thr$", path)
        r = int(m.group(1)) if m else -1
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.started.append(r)
        try:
            ev = self.hold.get(r)
            if ev is not None:
                ev.wait(30)
            with self.lock:
                if r in self.fail:
                    self.fail.discard(r)
                    raise OSError(f"injected GET failure for rank {r}")
            time.sleep(0.01)          # long enough for several GETs to overlap
            return self._fs.open_input_file(path)
        finally:
            with self.lock:
                self.active -= 1
                self.done.append(r)


def _gated_exchange(tmp_path, nj, conc, refresh_s=3600.0, h=4, m=2):
    store = Store(str(tmp_path))
    gfs = _GatedFS(store.fs)
    store.fs = gfs
    ex = ThresholdExchange(store, RUN, 0, nj, {"d": (h, m)}, refresh_s=refresh_s,
                           fetch_concurrency=conc)
    return ex, gfs


def _put_peer(ex, tmp_path, r, seq, val, h=4, m=2, t=None):
    path = tmp_path / ex.subdir / file_name(r)
    path.write_bytes(_thr(seq=seq, d=np.full((h, m), val)))
    if t is not None:
        os.utime(path, ns=(t * 10 ** 9, t * 10 ** 9))
    return path


def _stats_consistent(st):
    assert st["peer_gets"] == st["peer_files_read"] + st["peer_files_skipped"] + st["peer_files_stale"], st


def test_concurrent_fetch_folds_every_changed_peer_in_one_version(tmp_path):
    nj, conc = 13, 4
    ex, gfs = _gated_exchange(tmp_path, nj, conc)
    try:
        for r in range(1, nj):
            _put_peer(ex, tmp_path, r, seq=100 + r, val=float(r))
        assert ex.poll_once() == nj - 1
        assert 1 < gfs.max_active <= conc            # really concurrent, and bounded
        v, lists, changed = ex.latest(0)
        assert v == 1 and changed and ex.stats["versions"] == 1     # one version per round
        assert sorted(float(a[0, 0]) for a in lists["d"]) == [float(r) for r in range(1, nj)]
        st = ex.stats
        assert st["peer_files_read"] == nj - 1 and st["peer_gets"] == nj - 1
        assert st["peer_ranks_seen"] == nj - 1 and st["fetch_concurrency"] == conc
        _stats_consistent(st)
        # Nothing changed: no GET at all.
        assert ex.poll_once() == 0 and ex.stats["peer_gets"] == nj - 1
        # Only some peers change: only those are fetched, one more version.
        for r in (3, 7, 11):
            _put_peer(ex, tmp_path, r, seq=500 + r, val=-float(r), t=10 ** 6 + r)
        assert ex.poll_once() == 3 and ex.stats["peer_gets"] == nj - 1 + 3
        v2, lists2, _ = ex.latest(v)
        assert v2 == 2 and sorted(float(a[0, 0]) for a in lists2["d"]) == sorted(
            [float(r) for r in range(1, nj) if r not in (3, 7, 11)] + [-3.0, -7.0, -11.0])
        _stats_consistent(ex.stats)
    finally:
        ex.close(final=False)


def test_concurrent_bad_files_sig_recorded_failed_get_retried(tmp_path):
    nj = 9
    ex, gfs = _gated_exchange(tmp_path, nj, 4)
    try:
        good = {r: _put_peer(ex, tmp_path, r, seq=10, val=1.0) for r in range(1, nj)}
        for r in (2, 5):                              # truncated objects: bad
            good[r].write_bytes(good[r].read_bytes()[:-4])
        gfs.fail = {6}                                # a GET that fails once
        assert ex.poll_once() == nj - 1 - 3
        st = ex.stats
        assert st["peer_files_skipped"] == 3 and st["peer_gets"] == nj - 1
        assert set(ex._peer_sig) == set(range(1, nj)) - {6}   # bad recorded, failed GET not
        _stats_consistent(st)
        # Next round: the bad files are NOT downloaded again; the failed GET is retried.
        assert ex.poll_once() == 1
        assert ex.stats["peer_gets"] == nj - 1 + 1 and ex.stats["peer_files_skipped"] == 3
        # A rank's next publish (new signature) is read normally.
        _put_peer(ex, tmp_path, 2, seq=11, val=2.0, t=10 ** 6)
        assert ex.poll_once() == 1
        _stats_consistent(ex.stats)
        assert ex.stats["poll_errors"] == 0
    finally:
        ex.close(final=False)


def test_slow_peer_get_does_not_block_the_others(tmp_path):
    import threading
    nj, conc = 8, 3
    ex, gfs = _gated_exchange(tmp_path, nj, conc)
    try:
        for r in range(1, nj):
            _put_peer(ex, tmp_path, r, seq=10, val=float(r))
        gfs.hold[1] = ev = threading.Event()
        res = {}
        t = threading.Thread(target=lambda: res.setdefault("n", ex.poll_once()))
        t.start()
        # Every other peer's GET completes while rank 1's is still held.
        assert _wait_for(lambda: len([r for r in gfs.done if r != 1]) == nj - 2, timeout=10)
        assert 1 not in gfs.done and t.is_alive()
        ev.set()
        t.join(10)
        assert res["n"] == nj - 1 and gfs.done[-1] == 1
        _stats_consistent(ex.stats)
    finally:
        ex.close(final=False)


def test_close_does_not_hang_on_inflight_get_and_final_refresh_works(tmp_path):
    """A poller round stuck in a GET: close() still returns (bounded joins),
    and the end-of-scan `poll_once` on the caller's thread works afterwards,
    on a transient pool that does not outlive the round."""
    import threading
    nj = 6
    ex, gfs = _gated_exchange(tmp_path, nj, 4, refresh_s=0.05)
    ev = threading.Event()
    try:
        gfs.hold[1] = ev
        _put_peer(ex, tmp_path, 1, seq=10, val=1.0)
        assert _wait_for(lambda: gfs.active >= 1)    # the poller is inside rank 1's GET
        t0 = time.monotonic()
        ex.close(final=False)
        assert time.monotonic() - t0 < 10.0
        # Idle fetch workers exit; only the one inside the held GET remains.
        assert _wait_for(lambda: sum(t.name.startswith("thr-exchange-get-")
                                     for t in threading.enumerate()) <= 1)
        # The stuck poller still holds the round: the final refresh gives up
        # after the (shortened) round timeout instead of hanging.
        ex._ROUND_LOCK_TIMEOUT_S = 0.3
        assert ex.poll_once() == 0 and ex.stats["poll_errors"] == 1
        ev.set()                                      # the stuck GET completes
        assert _wait_for(lambda: not ex._round_lock.locked(), timeout=10)
        # Now the final refresh: every changed peer is fetched concurrently.
        gfs.max_active = 0
        for r in range(2, nj):
            _put_peer(ex, tmp_path, r, seq=10, val=float(r))
        assert ex.poll_once() == nj - 2
        assert gfs.max_active > 1
        _, lists, _ = ex.latest(0)
        assert sorted(float(a[0, 0]) for a in lists["d"]) == [float(r) for r in range(1, nj)]
        assert _wait_for(lambda: not any(t.name.startswith("thr-exchange-get")
                                         for t in threading.enumerate()), timeout=5)
        _stats_consistent(ex.stats)
    finally:
        ev.set()
        ex.close(final=False)


def test_fetch_concurrency_one_is_sequential(tmp_path):
    """1 = no pool, and at most one GET in flight at any time (each runs on a
    short-lived fetch thread, so the round's deadline bounds the caller)."""
    import threading
    nj = 6
    ex, gfs = _gated_exchange(tmp_path, nj, 1)
    try:
        assert ex._pool is None
        assert not any(t.name.startswith("thr-exchange-get") for t in threading.enumerate())
        for r in range(1, nj):
            _put_peer(ex, tmp_path, r, seq=10, val=float(r))
        assert ex.poll_once() == nj - 1 and gfs.max_active == 1
        _stats_consistent(ex.stats)
    finally:
        ex.close(final=False)


def test_stuck_get_round_hits_deadline_and_folds_what_arrived(tmp_path):
    """A GET that never returns must not hold the round forever: the round
    gives up at its deadline, folds the peers that did arrive, and leaves
    the stuck peer to be retried."""
    import threading
    nj = 6
    ex, gfs = _gated_exchange(tmp_path, nj, 4)
    ev = threading.Event()
    try:
        ex._ROUND_DEADLINE_S = 0.5
        for r in range(1, nj):
            _put_peer(ex, tmp_path, r, seq=10, val=float(r))
        gfs.hold[1] = ev
        t0 = time.monotonic()
        assert ex.poll_once() == nj - 2                 # everyone but rank 1
        assert time.monotonic() - t0 < 5.0
        assert ex.stats["poll_rounds_timed_out"] == 1 and ex.stats["poll_errors"] == 1
        assert 1 not in ex._peer_sig                     # retried next round
        _stats_consistent(ex.stats)
        ev.set()
        assert _wait_for(lambda: gfs.active == 0, timeout=10)
        assert ex.poll_once() == 1
        _stats_consistent(ex.stats)
    finally:
        ev.set()
        ex.close(final=False)


def test_pool_never_larger_than_peer_count(tmp_path):
    import threading
    ex, _ = _gated_exchange(tmp_path, 2, 8)              # one peer: no pool at all
    try:
        assert ex._pool is None
        assert not any(t.name.startswith("thr-exchange-get") for t in threading.enumerate())
    finally:
        ex.close(final=False)
    ex, _ = _gated_exchange(tmp_path / "b", 4, 8)        # three peers: three workers
    try:
        assert len(ex._pool._threads) == 3
    finally:
        ex.close(final=False)


# ------------------------------------------- robustness: bounded waits, cleanup

def test_single_peer_stuck_get_is_bounded_on_the_calling_thread(tmp_path, monkeypatch):
    """N=2 (one peer, no pool): the GET used to run on the calling thread
    with no deadline -- the compute thread, for the first poll in __init__
    and the end-of-scan refresh. A stuck GET now only means "no update this
    round", and is not re-issued while it is still stuck."""
    import threading
    monkeypatch.setattr(ThresholdExchange, "_ROUND_DEADLINE_S", 0.4)
    store = Store(str(tmp_path))
    gfs = _GatedFS(store.fs)
    store.fs = gfs
    ev = threading.Event()
    gfs.hold[1] = ev
    d = tmp_path / thresholds.subdir_for(RUN)
    d.mkdir(parents=True)
    (d / file_name(1)).write_bytes(_thr(seq=10, d=np.full((4, 2), 1.0)))
    t0 = time.monotonic()
    ex = ThresholdExchange(store, RUN, 0, 2, {"d": (4, 2)}, refresh_s=3600)
    try:
        assert time.monotonic() - t0 < 5.0                 # construction did not hang
        assert ex.stats["poll_rounds_timed_out"] == 1
        assert ex.latest(0)[1] is None
        t0 = time.monotonic()
        assert ex.poll_once() == 0                          # still stuck: skipped
        assert time.monotonic() - t0 < 2.0
        assert ex.stats["peer_fetches_still_running"] == 1
        assert gfs.started == [1]                           # one GET, never re-issued
        ev.set()
        assert _wait_for(lambda: not ex._inflight, timeout=10)
        assert ex.poll_once() == 1                          # retried once it returned
        assert gfs.max_active == 1
        _stats_consistent(ex.stats)
    finally:
        ev.set()
        ex.close(final=False)


def test_sequential_stuck_peer_is_skipped_others_still_fetched(tmp_path, monkeypatch):
    """fetch_concurrency=1: one GET in flight at a time, even across rounds;
    a stuck peer does not hold the others back after its round."""
    import threading
    monkeypatch.setattr(ThresholdExchange, "_ROUND_DEADLINE_S", 0.4)
    nj = 4
    ex, gfs = _gated_exchange(tmp_path, nj, 1)
    ev = threading.Event()
    try:
        gfs.hold[1] = ev
        for r in range(1, nj):
            _put_peer(ex, tmp_path, r, seq=10, val=float(r))
        t0 = time.monotonic()
        first = ex.poll_once()                    # stops at rank 1's stuck GET
        assert time.monotonic() - t0 < 5.0
        assert ex.stats["poll_rounds_timed_out"] == 1
        # The rest arrive next round, while rank 1 is still stuck.
        assert first + ex.poll_once() == nj - 2
        assert gfs.max_active == 1                # never two GETs at once
        assert gfs.started.count(1) == 1
        _stats_consistent(ex.stats)
    finally:
        ev.set()
        ex.close(final=False)


def test_stuck_peer_is_not_resubmitted_and_pool_queue_stays_bounded(tmp_path, monkeypatch):
    """Pooled rounds: a peer whose GET outlived its round's deadline is not
    queued again every round (that stranded one more worker per round until
    none was free, and grew the queue without bound)."""
    import threading
    monkeypatch.setattr(ThresholdExchange, "_ROUND_DEADLINE_S", 0.3)
    nj = 4
    ex, gfs = _gated_exchange(tmp_path, nj, 2)
    ev = threading.Event()
    try:
        gfs.hold[1] = ev
        for r in range(1, nj):
            _put_peer(ex, tmp_path, r, seq=10, val=float(r))
        assert ex.poll_once() == nj - 2
        for rnd in range(4):
            for r in range(2, nj):                # peers 2.. publish again
                _put_peer(ex, tmp_path, r, seq=20 + rnd, val=float(r), t=10 ** 6 + rnd)
            assert ex.poll_once() == nj - 2, rnd
            assert ex._pool._q.qsize() == 0
        assert gfs.started.count(1) == 1
        assert ex.stats["peer_fetches_still_running"] == 4
        ev.set()
        assert _wait_for(lambda: not ex._inflight, timeout=10)
        assert ex.poll_once() == 1
        _stats_consistent(ex.stats)
    finally:
        ev.set()
        ex.close(final=False)


def test_task_queued_behind_stuck_workers_never_blocks_its_peer(tmp_path, monkeypatch):
    """Every persistent pool worker stuck in a GET: later rounds' tasks for
    OTHER peers sit unstarted in the queue when their round hits its
    deadline. They used to stay claimed while queued, so every later round
    -- including the end-of-scan refresh on a transient pool with free
    workers, after close() -- skipped those peers as "still running". Now a
    round cancels its unstarted tasks: the peers are free next round, the
    queue is left empty, and a cancelled task never runs a useless GET once
    a worker frees up. A GET that STARTED still keeps its peer claimed."""
    import threading
    monkeypatch.setattr(ThresholdExchange, "_ROUND_DEADLINE_S", 0.3)
    nj = 6
    ex, gfs = _gated_exchange(tmp_path, nj, 2)       # two workers
    ev = threading.Event()
    try:
        gfs.hold[1] = ev
        gfs.hold[2] = ev
        for r in (1, 2):
            _put_peer(ex, tmp_path, r, seq=10, val=float(r))
        assert ex.poll_once() == 0                    # both workers stuck on 1 and 2
        assert _wait_for(lambda: gfs.active == 2)
        for rnd in range(2):
            for r in range(3, nj):
                _put_peer(ex, tmp_path, r, seq=20 + rnd, val=float(r) + 1, t=10 ** 6 + rnd)
            assert ex.poll_once() == 0, rnd           # queued behind the stuck workers
            assert ex._pool._q.qsize() == 0, rnd      # ... and cancelled at the deadline
            assert ex._inflight == {1, 2}, rnd        # only the STARTED GETs stay claimed
        for r in range(3, nj):
            _put_peer(ex, tmp_path, r, seq=30, val=float(r) + 2, t=10 ** 6 + 9)
        ex.close(final=False)
        # The end-of-scan refresh's 3 local GETs only need to finish within
        # the deadline; give them room on a loaded machine (the deadline
        # was only short to make the earlier rounds expire quickly).
        monkeypatch.setattr(ThresholdExchange, "_ROUND_DEADLINE_S", 10.0)
        running = ex.stats["peer_fetches_still_running"]
        assert ex.poll_once() == nj - 3               # end-of-scan refresh: 3, 4, 5
        assert ex.stats["peer_fetches_still_running"] == running + 2   # only 1 and 2
        _, lists, _ = ex.latest(0)
        assert sorted(float(a[0, 0]) for a in lists["d"]) == [float(r) + 2 for r in range(3, nj)]
        ev.set()
        assert _wait_for(lambda: gfs.active == 0 and not ex._inflight, timeout=10)
        time.sleep(0.2)                               # a freed worker would start a queued task now
        assert sorted(gfs.started) == [1, 2, 3, 4, 5]  # no GET ever ran for a cancelled task
        _stats_consistent(ex.stats)
    finally:
        ev.set()
        ex.close(final=False)


def test_fetch_pool_cancel_only_unstarted_tasks():
    import queue as _queue
    import threading
    gate, started = threading.Event(), threading.Event()

    def fn(a):
        started.set()
        gate.wait(10)
        return a

    pool = thresholds._FetchPool(fn, 1, "t-cancel")
    out = _queue.SimpleQueue()
    try:
        t1 = pool.submit("a", out)
        assert started.wait(5)                        # "a" is running
        t2, t3 = pool.submit("b", out), pool.submit("c", out)
        assert pool.cancel([t1, t2]) == ["b"]         # a started: left alone
        gate.set()
        assert out.get(timeout=5) == ("a", "a")
        assert out.get(timeout=5) == ("c", "c")       # c was not cancelled
        assert pool.cancel([t3]) == [] and out.empty()
    finally:
        gate.set()
        pool.close()


def test_exchange_thread_start_failure_stops_the_publisher(tmp_path, monkeypatch):
    import threading
    orig = threading.Thread.start

    def start(self):
        if self.name == "thr-exchange-poll":
            raise RuntimeError("can't start new thread")
        return orig(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    with pytest.raises(RuntimeError, match="can't start"):
        ThresholdExchange(Store(str(tmp_path)), RUN, 0, 4, {"d": (2, 1)}, refresh_s=1)
    monkeypatch.setattr(threading.Thread, "start", orig)
    assert _wait_for(lambda: not any(t.name.startswith("thr-exchange")
                                     for t in threading.enumerate()), timeout=5)


def test_fetch_pool_thread_start_failure_stops_started_workers(monkeypatch):
    import threading
    orig = threading.Thread.start

    def start(self):
        if self.name == "t-pool-2":
            raise RuntimeError("can't start new thread")
        return orig(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    with pytest.raises(RuntimeError, match="can't start"):
        thresholds._FetchPool(lambda a: a, 4, "t-pool")
    monkeypatch.setattr(threading.Thread, "start", orig)
    assert _wait_for(lambda: not any(t.name.startswith("t-pool")
                                     for t in threading.enumerate()), timeout=5)


def test_stats_keys_are_fixed_at_construction(tmp_path, monkeypatch):
    """Every stats key the exchange's OWN threads (poller, fetch, publisher)
    touch exists from the start, so a thread that outlives close() only ever
    updates values, never adds keys. (The compute loop adds its own keys --
    `configured`, `apply_*`, `publish_*`, `*_failures`, `final_refresh` --
    on the compute thread, the thread that also takes the manifest's
    snapshot.)"""
    import threading
    monkeypatch.setattr(ThresholdExchange, "_ROUND_DEADLINE_S", 0.3)
    ex, gfs = _gated_exchange(tmp_path, 4, 2)
    ev = threading.Event()
    try:
        keys = set(ex.stats)
        assert ex.stats["poll_rounds_timed_out"] == 0
        assert ex.stats["peer_fetches_still_running"] == 0
        gfs.hold[1] = ev
        for r in range(1, 4):
            _put_peer(ex, tmp_path, r, seq=10, val=float(r))
        ex.poll_once()
        ex.poll_once()
        assert ex.stats["poll_rounds_timed_out"] == 1
        assert set(ex.stats) == keys
    finally:
        ev.set()
        ex.close(final=False)


@pytest.mark.parametrize("kw", [
    {"refresh_s": 0}, {"refresh_s": -1.0}, {"refresh_s": float("nan")},
    {"refresh_s": float("inf")}, {"refresh_s": __import__("threading").TIMEOUT_MAX * 2},
    {"publish_min_interval_s": -1.0},
    {"publish_min_interval_s": float("nan")}, {"fetch_concurrency": 0},
    {"fetch_concurrency": 1.5}, {"fetch_concurrency": True},
    {"sizes": {"d": (3, 0)}}, {"sizes": {"d": (-1, 2)}},
])
def test_constructor_validates_its_arguments(tmp_path, kw):
    import threading
    kw = dict(kw)
    sizes = kw.pop("sizes", {"d": (3, 2)})
    before = {t.ident for t in threading.enumerate()}
    with pytest.raises(ValueError):
        ThresholdExchange(Store(str(tmp_path)), RUN, 0, 2, sizes, **kw)
    assert not [t for t in threading.enumerate()
                if t.ident not in before and t.name.startswith("thr-exchange")]


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 0.0, -1.0,
                                 __import__("threading").TIMEOUT_MAX * 2])
def test_config_rejects_refresh_the_exchange_would_refuse(bad):
    """A refresh the exchange refuses at setup (where compute degrades to no
    exchange, no trim, with only a warning) is a config error at load time."""
    from pydantic import ValidationError
    from nova_bf.config import ParamsConfig
    with pytest.raises(ValidationError, match="threshold_exchange_refresh_s"):
        ParamsConfig(threshold_exchange_refresh_s=bad)
    assert ParamsConfig(threshold_exchange_refresh_s=0.5).threshold_exchange_refresh_s == 0.5
    assert ParamsConfig(threshold_exchange_refresh_s=5).threshold_exchange_refresh_s == 5.0


def test_exit_drain_stops_exchanges_and_waits_for_inflight_io(tmp_path, monkeypatch):
    """The atexit hook: stop every live exchange, refuse new I/O, and wait
    for in-flight GETs -- briefly: a stuck one never hangs exit."""
    import threading
    monkeypatch.setitem(thresholds._IO, "closing", False)   # restored afterwards
    ex, gfs = _gated_exchange(tmp_path, 2, 1, refresh_s=0.05)
    ev = threading.Event()
    try:
        gfs.hold[1] = ev
        _put_peer(ex, tmp_path, 1, seq=10, val=1.0)
        assert _wait_for(lambda: gfs.active >= 1)          # the poller is inside a GET
        threading.Timer(0.3, ev.set).start()
        t0 = time.monotonic()
        thresholds._drain_at_exit(timeout_s=10.0)
        assert time.monotonic() - t0 < 5.0
        assert gfs.active == 0 and thresholds._IO["n"] == 0  # waited for it
        assert ex._stop.is_set()
        assert _wait_for(lambda: not ex._poll_thread.is_alive() and not ex._pub_thread.is_alive())
        # New I/O is refused once closing.
        n = len(gfs.started)
        assert ex._fetch(str(tmp_path / ex.subdir / file_name(1)))[0] == "get"
        assert len(gfs.started) == n
        with pytest.raises(RuntimeError, match="exiting"):
            ex._write({"d": np.zeros((4, 2), np.float16)})
    finally:
        ev.set()
        ex.close(final=False)


def test_exit_drain_is_bounded_by_its_timeout(tmp_path, monkeypatch):
    import threading
    monkeypatch.setitem(thresholds._IO, "closing", False)
    ex, gfs = _gated_exchange(tmp_path, 2, 1)
    ev = threading.Event()
    try:
        gfs.hold[1] = ev
        path = str(_put_peer(ex, tmp_path, 1, seq=10, val=1.0))
        threading.Thread(target=ex._fetch, args=(path,), daemon=True).start()
        assert _wait_for(lambda: gfs.active >= 1)
        t0 = time.monotonic()
        thresholds._drain_at_exit(timeout_s=0.3)
        assert time.monotonic() - t0 < 3.0 and thresholds._IO["n"] >= 1
    finally:
        ev.set()
        assert _wait_for(lambda: thresholds._IO["n"] == 0, timeout=10)
        ex.close(final=False)


# ------------------------------------- compute hooks: a failing step degrades

def _oom(*a, **k):
    import torch
    raise torch.OutOfMemoryError("injected OOM in the exchange")


@pytest.mark.parametrize("where", ["union_kth", "union_kth_third_search", "publish",
                                   "final_only"])
def test_exchange_hook_failures_degrade_never_fail(tmp_path, monkeypatch, where):
    """An exception inside the exchange's own per-file / startup / end-of-scan
    steps (e.g. an OOM in `union_kth`) used to kill the rank. Now it is
    logged once, counted, and skipped; partial progress (some searches'
    thresholds updated before the failure) stays valid, and the merge stays
    exact with `trim_partials` on (the default)."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=6, rows=30, n_q=20, seed=4)
    single_out = tmp_path / "single"; single_out.mkdir()
    single = {n: pq.read_table(p).to_pydict()
              for n, p in run_compute(_cfg(cdir, qpath, single_out, False, multi=True),
                                      num_jobs=None).items()}
    real_union, real_publish = thresholds.union_kth, ThresholdExchange.publish
    calls = {"n": 0}
    if where == "union_kth":
        monkeypatch.setattr(thresholds, "union_kth", _oom)
    elif where == "union_kth_third_search":
        # The `half` search has k + 1: the first two searches' thresholds
        # are updated, then the apply fails.
        def flaky(own, peers, k, device, **kw):
            if k == K + 1:
                _oom()
            return real_union(own, peers, k, device, **kw)
        monkeypatch.setattr(thresholds, "union_kth", flaky)
    elif where == "publish":
        def bad_publish(self, arrays):
            calls["n"] += 1
            if calls["n"] % 2:
                raise RuntimeError("injected publish failure")
            return real_publish(self, arrays)
        monkeypatch.setattr(ThresholdExchange, "publish", bad_publish)
    else:
        # Only the end-of-scan refresh's apply fails (after the final poll).
        real_poll = ThresholdExchange.poll_once
        def poll_then_break(self):
            if self._closed:
                monkeypatch.setattr(thresholds, "union_kth", _oom)
            return real_poll(self)
        monkeypatch.setattr(ThresholdExchange, "poll_once", poll_then_break)
    out = tmp_path / "sharded"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, True, multi=True)
    for r in range(3):
        run_compute(cfg, num_jobs=3, job_rank=r)
        if where == "final_only":
            monkeypatch.setattr(thresholds, "union_kth", real_union)
    merged = {n: pq.read_table(p).to_pydict() for n, p in run_merge(cfg).items()}
    _assert_same(merged, single)
    stats = [m["params"]["threshold_exchange_stats"] for m in _manifests(str(out))]
    assert all(s["enabled"] for s in stats)
    key = "publish_failures" if where == "publish" else "apply_failures"
    assert sum(s[key] for s in stats) >= 1, stats
    if where in ("publish", "final_only"):
        # The exchange still did something useful around the failures.
        assert sum(s["applies"] for s in stats) >= 1
    if where == "union_kth_third_search":
        # No apply completed, but the first searches' thresholds were
        # updated (and pruned against) before each failure.
        assert sum(s["applies"] for s in stats) == 0
        assert any(s["first_applied_file"] is not None for s in stats), stats


def test_exchange_hook_device_errors_are_not_swallowed(tmp_path, monkeypatch):
    """A sticky device fault surfacing at the hook's sync is not the
    exchange's own failure: it propagates."""
    import torch
    if not hasattr(torch, "AcceleratorError"):
        pytest.skip("torch has no AcceleratorError")
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=4, rows=20, n_q=10)
    out = tmp_path / "o"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, True)
    run_compute(cfg, num_jobs=2, job_rank=0)

    def fault(*a, **k):
        raise torch.AcceleratorError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(thresholds, "union_kth", fault)
    with pytest.raises(torch.AcceleratorError):
        run_compute(cfg, num_jobs=2, job_rank=1)


def test_device_fault_classification():
    """torch is unpinned: before `torch.AcceleratorError` (~2.8) a sticky CUDA
    fault is a plain RuntimeError. Those propagate too; an OOM (a
    RuntimeError subclass) and ordinary errors do not."""
    import torch
    from nova_bf.compute import _is_device_fault
    assert _is_device_fault(RuntimeError(
        "CUDA error: an illegal memory access was encountered\nCUDA kernel errors ..."))
    assert _is_device_fault(RuntimeError("CUDA error: device-side assert triggered"))
    assert _is_device_fault(RuntimeError("an illegal memory access was encountered"))
    assert not _is_device_fault(torch.OutOfMemoryError("CUDA out of memory. Tried to allocate"))
    assert not _is_device_fault(RuntimeError("CUDA error: out of memory"))
    assert not _is_device_fault(RuntimeError("shape mismatch"))
    assert not _is_device_fault(ValueError("CUDA error: not a RuntimeError"))
    # Allocation failures stay recoverable whatever type they arrive as (on
    # torch >= 2.8 a failed cudaMalloc outside the caching allocator is an
    # AcceleratorError), and cuBLAS allocation failures likewise.
    assert not _is_device_fault(RuntimeError(
        "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`"))
    # Other device-fault spellings that are not "CUDA error: ..."
    assert _is_device_fault(RuntimeError("Triton Error [CUDA]: unknown error"))
    assert _is_device_fault(RuntimeError("cuDNN error: CUDNN_STATUS_EXECUTION_FAILED"))
    if hasattr(torch, "AcceleratorError"):
        assert _is_device_fault(torch.AcceleratorError("anything"))
        assert not _is_device_fault(torch.AcceleratorError("CUDA error: out of memory"))


def test_exchange_hook_plain_runtime_cuda_error_is_not_swallowed(tmp_path, monkeypatch):
    """The pre-AcceleratorError form of a sticky device fault propagates."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=4, rows=20, n_q=10)
    out = tmp_path / "o"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, True)
    run_compute(cfg, num_jobs=2, job_rank=0)

    def fault(*a, **k):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(thresholds, "union_kth", fault)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        run_compute(cfg, num_jobs=2, job_rank=1)


def test_exchange_hook_swallowed_error_keeps_its_traceback(tmp_path, monkeypatch, caplog):
    """A degraded hook failure (possibly a programming error) is logged once
    WITH its traceback."""
    import logging
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=4, rows=20, n_q=10)
    out = tmp_path / "o"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, True)
    run_compute(cfg, num_jobs=2, job_rank=0)
    monkeypatch.setattr(thresholds, "union_kth", _oom)
    with caplog.at_level(logging.WARNING, logger="nova_bf"):
        run_compute(cfg, num_jobs=2, job_rank=1)
    recs = [r for r in caplog.records if "apply step failed" in r.getMessage()]
    assert len(recs) == 1 and recs[0].exc_info is not None
    assert recs[0].exc_info[0].__name__ == "OutOfMemoryError"


def test_manifest_records_a_snapshot_of_the_exchange_stats(tmp_path, monkeypatch):
    from nova_bf import compute as C
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=3, rows=20, n_q=10)
    out = tmp_path / "o"; out.mkdir()
    made, docs = [], []

    class Recording(ThresholdExchange):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    real_write = C.run_manifest.write
    def capture(store, name, doc):
        docs.append(doc)
        return real_write(store, name, doc)
    monkeypatch.setattr(thresholds, "ThresholdExchange", Recording)
    monkeypatch.setattr(C.run_manifest, "write", capture)
    run_compute(_cfg(cdir, qpath, out, True), num_jobs=2, job_rank=0)
    rec = docs[-1]["params"]["threshold_exchange_stats"]
    assert rec is not made[-1].stats and rec == made[-1].stats
