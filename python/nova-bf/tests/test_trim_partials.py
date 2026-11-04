"""Write-time partial trimming (`params.trim_partials`, threshold exchange on).

A rank drops every entry of its final top-k that lies strictly below the
external threshold it pruned against, records that threshold in
`applied_threshold`, and the merge verifies it. These tests check:

  * exactness: trimmed sharded output == unsharded output (ids AND scores),
    across dense / filtered / rows-subset / sparse / multivector searches in
    one run, both tie-break rules, 2 and 4 ranks, and both merge folds
    (`numpy`, and `torch` so the device-fold code runs on CPU tensors);
  * that partials really shrink (entries and bytes vs `trim_partials: false`);
  * two-pass engagement, NaN corpora, and a remedy re-run (exchange off,
    untrimmed) mixed with trimmed partials;
  * tampering: a too-high threshold trims true neighbours and the merge
    must refuse;
  * `_trim_mask` semantics: ties kept whatever the ordinal, NaN/+inf rows
    untouched.
"""
from __future__ import annotations

import glob
import json
import os

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
from nova_bf.merge import run_merge

DIM, VOCAB, NNZ, TOK_DIM = 8, 16, 5, 4
TEST_DEVICE = os.environ.get("NOVA_BF_TEST_DEVICE", "cpu")

SPARSE_T = pa.struct([pa.field("indices", pa.list_(pa.uint32())),
                      pa.field("values", pa.list_(pa.float32()))])
MV_T = pa.list_(pa.list_(pa.float32()))


def _sparse_rows(rng, n):
    out = []
    for _ in range(n):
        idx = np.sort(rng.choice(VOCAB, size=NNZ, replace=False))
        out.append({"indices": idx.tolist(),
                    "values": rng.standard_normal(NNZ).astype(np.float32).tolist()})
    return out


def _mv_rows(rng, n):
    return [rng.standard_normal((int(rng.integers(1, 4)), TOK_DIM)).astype(np.float32).tolist()
            for _ in range(n)]


def _dataset(tmp_path, n_files=8, rows=40, n_q=24, seed=0, nan_row=None):
    """Corpus files with dense + sparse + multivector columns, an id column
    and a filterable `lang`; queries carry all three vector columns."""
    rng = np.random.default_rng(seed)
    cdir = tmp_path / "corpus"; cdir.mkdir()
    for fi in range(n_files):
        dense = rng.standard_normal((rows, DIM)).astype(np.float32)
        if nan_row is not None and nan_row[0] == fi:
            dense[nan_row[1]] = np.nan
        pq.write_table(pa.table({
            "dense_embedding": pa.array(dense.tolist(), pa.list_(pa.float32())),
            "sparse_embedding": pa.array(_sparse_rows(rng, rows), SPARSE_T),
            "multivector_embedding": pa.array(_mv_rows(rng, rows), MV_T),
            "id": pa.array([f"f{fi}r{r}" for r in range(rows)]),
            "lang": pa.array([("eng" if (fi + r) % 3 else "fra") for r in range(rows)]),
        }), str(cdir / f"part{fi}.parquet"))
    q = rng.standard_normal((n_q, DIM)).astype(np.float32)
    qpath = tmp_path / "queries.parquet"
    pq.write_table(pa.table({
        "dense_embedding": pa.array(q.tolist(), pa.list_(pa.float32())),
        "sparse_embedding": pa.array(_sparse_rows(rng, n_q), SPARSE_T),
        "multivector_embedding": pa.array(_mv_rows(rng, n_q), MV_T),
        "qid": pa.array([f"q{i}" for i in range(n_q)]),
        "qset": pa.array(["a" if i % 2 else "b" for i in range(n_q)]),
    }), str(qpath))
    return str(cdir), str(qpath)


def _searches(k, types=("dense", "sparse", "multivector")):
    s = []
    if "dense" in types:
        s += [
            SearchSpec(name="cos", vector_type="dense", metric="cosine", k=k),
            SearchSpec(name="euc_eng", vector_type="dense", metric="euclidean", k=k + 1,
                       filter=Filter(must=[FilterCondition(field="lang", match="eng")])),
            SearchSpec(name="dot_half", vector_type="dense", metric="dot", k=k - 1,
                       rows=RowSelector(column="qset", isin=["a"])),
        ]
    if "sparse" in types:
        s.append(SearchSpec(name="sp", vector_type="sparse", metric="dot", k=k))
    if "multivector" in types:
        s.append(SearchSpec(name="mv", vector_type="multivector", metric="dot", k=k))
    return s


def _cfg(cdir, qpath, out, exchange, k=6, tiebreak="ordinal", trim=True,
         types=("dense", "sparse", "multivector")):
    return BruteForceConfig(
        corpus=CorpusConfig(path=cdir, id_column="id"),
        queries=QueriesConfig(path=qpath, id_column="qid", payload_fields=["qset"]),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(dense_batch_size=16, threshold_exchange=exchange,
                            threshold_exchange_refresh_s=0.2, tiebreak=tiebreak,
                            trim_partials=trim),
        searches=_searches(k, types),
    )


def _read(results):
    return {n: pq.read_table(p).to_pydict() for n, p in results.items()}


# On CUDA a sharded run's scores are not bit-identical to the single run's:
# the scoring kernels' fp32 reduction order depends on the batch shape, and
# sharding changes it (seen on an A10G for multivector MaxSim, with exchange
# and trimming OFF too). The ids and their order must still match exactly; the
# scores may differ by a few ULP (the fp32 exactness definition). On the CPU
# the scores stay bit-identical.
_SCORE_ULPS = 0 if TEST_DEVICE == "cpu" else 4


def _ulp_diff(a, b) -> int:
    a = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    b = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    # Map the sign-magnitude float bits onto one monotone integer line.
    a = np.where(a < 0, -(a & 0x7FFFFFFF), a)
    b = np.where(b < 0, -(b & 0x7FFFFFFF), b)
    return int(np.abs(a - b).max(initial=0))


def _assert_same(merged, single):
    assert set(merged) == set(single)
    for name in single:
        assert merged[name]["query_id"] == single[name]["query_id"], name
        assert merged[name]["hit_ids"] == single[name]["hit_ids"], name
        if _SCORE_ULPS == 0:
            assert merged[name]["hit_scores"] == single[name]["hit_scores"], name
            continue
        for r, (m, s1) in enumerate(zip(merged[name]["hit_scores"],
                                        single[name]["hit_scores"])):
            assert len(m) == len(s1), (name, r)
            assert _ulp_diff(m, s1) <= _SCORE_ULPS, (name, r, m, s1)


def _partials(out):
    return sorted(glob.glob(os.path.join(str(out), "_bf_partial_*", "rank*.parquet")))


def _entries(out):
    return sum(sum(len(h) for h in pq.read_table(f, columns=["hit_ids"]).column(0).to_pylist())
               for f in _partials(out))


def _bytes(out):
    return sum(os.path.getsize(f) for f in _partials(out))


def _shard(cfg, nj, second_pass=True):
    """Ranks back to back (each later rank inherits its predecessors'
    thresholds), then -- `second_pass` -- every rank again, now seeing every
    peer's FINAL list: the state a concurrent fleet ends in."""
    for r in range(nj):
        run_compute(cfg, num_jobs=nj, job_rank=r)
    if second_pass:
        for r in range(nj):
            run_compute(cfg, num_jobs=nj, job_rank=r)


@pytest.mark.parametrize("fold", ["numpy", "torch"])
@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
@pytest.mark.parametrize("nj", [2, 4])
def test_trimmed_sharded_equals_single_all_vector_types(tmp_path, monkeypatch, nj, tiebreak, fold):
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    monkeypatch.setenv("NOVA_BF_POISON_DEAD_ROWS", "1")
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", fold)
    cdir, qpath = _dataset(tmp_path, seed=nj)
    (tmp_path / "s").mkdir(); (tmp_path / "x").mkdir()
    single = _read(run_compute(_cfg(cdir, qpath, tmp_path / "s", False, tiebreak=tiebreak)))
    cfg = _cfg(cdir, qpath, tmp_path / "x", True, tiebreak=tiebreak)
    _shard(cfg, nj)
    # Trimming really happened, and on every search.
    ms = [json.load(open(f)) for f in glob.glob(
        os.path.join(str(tmp_path / "x"), "_bf_manifest_*_compute", "rank*.json"))]
    kept = {}
    for m in ms:
        for e in m["searches"]:
            kept.setdefault(e["name"], []).append(e["trim"])
    for name, ts in kept.items():
        assert sum(t["entries_before"] - t["entries_kept"] for t in ts) > 0, name
    merged = _read(run_merge(cfg))
    _assert_same(merged, single)


def test_partials_shrink_vs_untrimmed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    nj, k, n_q = 4, 20, 60
    cdir, qpath = _dataset(tmp_path, n_files=8, rows=200, n_q=n_q, seed=7)
    sizes = {}
    merged = {}
    for trim in (False, True):
        out = tmp_path / f"trim{trim}"; out.mkdir()
        cfg = _cfg(cdir, qpath, out, True, k=k, trim=trim, types=("dense",))
        _shard(cfg, nj)
        sizes[trim] = (_entries(out), _bytes(out))
        merged[trim] = _read(run_merge(cfg))
    _assert_same(merged[True], merged[False])
    (e0, b0), (e1, b1) = sizes[False], sizes[True]
    with capsys.disabled():
        print(f"\n[trim sizes] dense searches, {nj} ranks, k={k}, NQ={n_q}: "
              f"entries {e0} -> {e1} ({e1 / e0:.3f}), partial bytes {b0} -> {b1} "
              f"({b1 / b0:.3f})")
    assert e1 < 0.5 * e0
    assert b1 < b0


def test_trim_with_two_pass_engaged(tmp_path, monkeypatch):
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    monkeypatch.setenv("NOVA_BF_POISON_DEAD_ROWS", "1")
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
    monkeypatch.setenv("NOVA_BF_TWOPASS_FORCE_ENGAGE", "1")
    cdir, qpath = _dataset(tmp_path, n_files=9, rows=48, seed=5)
    (tmp_path / "s").mkdir(); (tmp_path / "x").mkdir()
    single = _read(run_compute(_cfg(cdir, qpath, tmp_path / "s", False, types=("dense",))))
    cfg = _cfg(cdir, qpath, tmp_path / "x", True, types=("dense",))
    _shard(cfg, 3)
    _assert_same(_read(run_merge(cfg)), single)


@pytest.mark.parametrize("fold", ["numpy", "torch"])
def test_trim_nan_corpus_rows_untouched_and_exact(tmp_path, monkeypatch, fold):
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", fold)
    cdir, qpath = _dataset(tmp_path, n_files=6, rows=30, seed=3, nan_row=(1, 4))
    (tmp_path / "s").mkdir(); (tmp_path / "x").mkdir()
    single = _read(run_compute(_cfg(cdir, qpath, tmp_path / "s", False, types=("dense",))))
    cfg = _cfg(cdir, qpath, tmp_path / "x", True, types=("dense",))
    _shard(cfg, 2)
    # Rank 1 owns file 1 (the NaN row): its NaN-affected queries are written
    # whole (k minus the dropped NaN hits), never trimmed.
    for f in _partials(tmp_path / "x"):
        if f.endswith("rank001.parquet"):
            t = pq.read_table(f).to_pydict()
            k = 6 + 1 if "euc_eng" in f else 6 - 1 if "dot_half" in f else 6
            for hits, nn in zip(t["hit_ids"], t["nan_hits"]):
                if nn:
                    assert len(hits) == k - nn, f
    _assert_same(_read(run_merge(cfg)), single)


def test_trimmed_mixed_with_untrimmed_remedy_rerun(tmp_path, monkeypatch):
    """The merge's remedy (re-run a rank with the exchange off) writes a full,
    untrimmed partial beside trimmed ones; the merge stays exact."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, seed=13)
    (tmp_path / "s").mkdir(); (tmp_path / "x").mkdir()
    single = _read(run_compute(_cfg(cdir, qpath, tmp_path / "s", False)))
    cfg = _cfg(cdir, qpath, tmp_path / "x", True)
    _shard(cfg, 3)
    run_compute(_cfg(cdir, qpath, tmp_path / "x", False), num_jobs=3, job_rank=1)
    _assert_same(_read(run_merge(cfg)), single)


def test_too_high_trim_threshold_is_caught_by_merge(tmp_path, monkeypatch):
    """Tampering: the last rank lowers its external thresholds by a NEGATIVE
    slack (i.e. raises them past the true k-th), so it prunes and trims true
    neighbours. The recorded applied threshold is the same value, so the
    merge must refuse, naming that rank."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=4, rows=30, n_q=12, seed=17)
    out = tmp_path / "x"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, True, types=("dense",))
    run_compute(cfg, num_jobs=2, job_rank=0)
    orig = thresholds.lower_for_slack
    monkeypatch.setattr(thresholds, "lower_for_slack",
                        lambda s, d: np.where(np.isfinite(orig(s, d)), orig(s, d) + 0.25,
                                              orig(s, d)).astype(np.float32))
    run_compute(cfg, num_jobs=2, job_rank=1)
    monkeypatch.setattr(thresholds, "lower_for_slack", orig)
    monkeypatch.setenv("NOVA_BF_MERGE_FORCE", "1")
    with pytest.raises(RuntimeError, match=r"VERIFICATION FAILED.*\[1\]"):
        run_merge(cfg)


def test_trim_off_without_exchange_and_param_not_fingerprinted(tmp_path, monkeypatch):
    """`trim_partials` changes no fingerprint: partials of a trim-on and a
    trim-off rank merge together."""
    from nova_bf.results import config_identity
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=4, rows=20, n_q=10, seed=19)
    a = _cfg(cdir, qpath, tmp_path / "x", True, trim=True)
    b = _cfg(cdir, qpath, tmp_path / "x", True, trim=False)
    c = _cfg(cdir, qpath, tmp_path / "x", True, trim=False)
    c.params.threshold_exchange_fetch_concurrency = 1       # speed only, too
    for s in a.searches:
        assert config_identity(a, s) == config_identity(b, s) == config_identity(c, s)
    (tmp_path / "s").mkdir(); (tmp_path / "x").mkdir()
    single = _read(run_compute(_cfg(cdir, qpath, tmp_path / "s", False)))
    run_compute(a, num_jobs=2, job_rank=0)
    run_compute(b, num_jobs=2, job_rank=1)
    run_compute(a, num_jobs=2, job_rank=0)
    _assert_same(_read(run_merge(a)), single)


def test_trim_mask_semantics():
    import torch
    from nova_bf.compute import _trim_mask
    from nova_bf.tiebreak import pack

    ext = thresholds.keys_from_scores(np.array([0.5, 0.5, 0.5, -np.inf], np.float32), "cpu")
    from nova_bf.tiebreak import unpack_score
    t = float(unpack_score(ext)[0])                      # the lowered threshold
    sc = torch.tensor([[0.9, t, np.nextafter(np.float32(t), np.float32(-1)), -0.3],
                       [0.9, float("nan"), 0.1, -0.3],
                       [float("inf"), 0.6, 0.1, -0.3],
                       [0.9, 0.2, 0.1, -0.3]], dtype=torch.float32)
    ords = torch.tensor([[0, 4294967294, 0, 1]] * 4, dtype=torch.int64)  # worst real ordinal on the tie
    keys = pack(sc, ords)
    m = _trim_mask(keys, ext).tolist()
    assert m[0] == [False, False, True, True]     # tie at the threshold kept, below dropped
    assert m[1] == [False] * 4                    # NaN row untouched
    assert m[2] == [False] * 4                    # +inf row untouched
    assert m[3] == [False] * 4                    # no external threshold


def _compute_manifests(out):
    return {os.path.basename(f): json.load(open(f)) for f in glob.glob(
        os.path.join(str(out), "_bf_manifest_*_compute", "rank*.json"))}


def test_manifest_records_whether_trim_ran(tmp_path, monkeypatch):
    """`params.trim_partials` stays the configured value; the manifest's
    `trim_partials_effective` says whether the trim RAN (exchange active)."""
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=4, rows=20, n_q=10, seed=23)
    for exchange in (False, True):
        out = tmp_path / f"x{exchange}"; out.mkdir()
        cfg = _cfg(cdir, qpath, out, exchange, trim=True, types=("dense",))
        for r in range(2):
            run_compute(cfg, num_jobs=2, job_rank=r)
        ms = _compute_manifests(out)
        assert len(ms) == 2
        for m in ms.values():
            assert m["params"]["trim_partials"] is True
            assert m["params"]["trim_partials_effective"] is exchange
            assert all(("trim" in e) is exchange for e in m["searches"])


def test_first_apply_by_final_refresh_is_recorded_as_final(tmp_path, monkeypatch):
    """A rank whose only peer pickup is the end-of-scan refresh records
    `first_applied_file == "final"`, not None (which means never applied)."""
    from nova_bf import thresholds
    monkeypatch.setenv("NOVA_BF_DEVICE", TEST_DEVICE)
    cdir, qpath = _dataset(tmp_path, n_files=4, rows=20, n_q=10, seed=29)
    out = tmp_path / "x"; out.mkdir()
    cfg = _cfg(cdir, qpath, out, True, types=("dense",))
    run_compute(cfg, num_jobs=2, job_rank=0)            # publishes; sees nobody
    orig = thresholds.ThresholdExchange.poll_once

    def blind_until_closed(self):
        # Every poll during the scan (constructor + poller) sees nothing; only
        # the end-of-scan refresh, which runs after close(), reads the peer.
        return orig(self) if self._closed else 0
    monkeypatch.setattr(thresholds.ThresholdExchange, "poll_once", blind_until_closed)
    run_compute(cfg, num_jobs=2, job_rank=1)
    st = [m for m in _compute_manifests(out).values()
          if m["sharding"]["job_rank"] == 1][0]["params"]["threshold_exchange_stats"]
    assert st["first_applied_file"] == "final"
    assert st["applies"] == 1 and st["final_refresh"]["peer_files_folded"] == 1
