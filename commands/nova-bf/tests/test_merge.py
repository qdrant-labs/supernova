"""Correctness tests for the brute-force merge phase.

Covers the streaming lockstep reduce over row-aligned partials:
  - the merged top-K equals the global top-K over the union of all partials,
  - payload is carried through and queries with fewer than k total candidates
    keep only their real hits (variable-length output, no -inf padding), and
  - partials written as MULTIPLE row groups (what a large partial becomes) merge
    correctly — the regression for the `to_pylist` "Nested data conversions"
    crash the old merge hit at 1M queries.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    OutputConfig,
    QueriesConfig,
    SearchSpec,
)
from nova_bf.merge import run_merge
from nova_bf.results import build_result_table, partial_dir, result_name

Q, W, K = 5, 3, 4


def _make_cfg(tmp) -> BruteForceConfig:
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(tmp / "corpus")),
        queries=QueriesConfig(path=str(tmp / "queries.parquet")),
        output=OutputConfig(path=str(tmp / "out")),
        searches=[SearchSpec(name="test", k=K)],
    )


@pytest.fixture
def scenario(tmp_path):
    """W row-aligned partials + the reference global top-K per query."""
    rng = np.random.default_rng(0)
    qids = [f"q{i}" for i in range(Q)]
    # per (query) accumulate every candidate across partials for the reference
    all_cands: dict[str, list[tuple[float, str]]] = {q: [] for q in qids}
    # per partial: hit_ids / hit_scores lists aligned to qids
    partials: list[tuple[list[list[str]], list[list[float]]]] = []
    score = 100.0
    for p in range(W):
        p_ids, p_scores = [], []
        for q in qids:
            # query "q0" gets only 1 candidate per partial → <K total (variable len);
            # others get a random 1..K so several queries exceed K total.
            n = 1 if q == "q0" else int(rng.integers(1, K + 1))
            ids = [f"{q}_p{p}_{i}" for i in range(n)]
            scores = [score := score - 1.0 for _ in range(n)]  # globally unique, no ties
            # a partial's own list is already sorted desc (as compute emits it)
            order = np.argsort(-np.array(scores))
            ids = [ids[j] for j in order]
            scores = [scores[j] for j in order]
            p_ids.append(ids)
            p_scores.append(scores)
            all_cands[q].extend(zip(scores, ids))
        partials.append((p_ids, p_scores))

    reference = {}
    for q in qids:
        top = sorted(all_cands[q], reverse=True)[:K]
        reference[q] = ([h for _, h in top], [s for s, _ in top])

    cfg = _make_cfg(tmp_path)
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True)
    for p, (p_ids, p_scores) in enumerate(partials):
        payload = {"src": [f"payload-{q}" for q in qids]}  # identical across partials
        table = build_result_table(qids, payload, p_ids, p_scores)
        # row_group_size < Q forces MULTIPLE row groups → multi-chunk nested columns
        # on read: the exact shape that crashed the old to_pylist-based merge.
        pq.write_table(table, str(pdir / f"rank{p:03d}.parquet"), row_group_size=2)

    return cfg, qids, reference


def _read_result(cfg) -> dict[str, tuple[list[str], list[float], str]]:
    t = pq.read_table(f"{cfg.output.path}/{result_name(cfg, cfg.searches[0])}").to_pydict()
    return {
        q: (hi, hs, src)
        for q, hi, hs, src in zip(t["query_id"], t["hit_ids"], t["hit_scores"], t["src"])
    }


def test_merge_matches_global_topk(scenario):
    cfg, qids, reference = scenario
    run_merge(cfg)
    got = _read_result(cfg)
    assert sorted(got) == sorted(qids)
    for q in qids:
        hi, hs, src = got[q]
        ref_ids, ref_scores = reference[q]
        assert hi == ref_ids  # identical hit ids, identical (score-desc) order
        assert np.allclose(hs, ref_scores)
        assert src == f"payload-{q}"  # payload carried from partial 0


def test_short_query_keeps_only_real_hits(scenario):
    """q0 has 1 candidate per partial (W total < K) → no -inf padding leaks out."""
    cfg, _, reference = scenario
    run_merge(cfg)
    hi, hs, _ = _read_result(cfg)["q0"]
    assert len(hi) == W < K
    assert hi == reference["q0"][0]


def test_explicit_batch_size_is_invariant(scenario):
    """A tiny merge_batch_size (many batches) gives the same result as one batch."""
    cfg, qids, reference = scenario
    cfg.params.merge_batch_size = 2  # < Q → several lockstep batches
    run_merge(cfg)
    got = _read_result(cfg)
    for q in qids:
        assert got[q][0] == reference[q][0]
        assert np.allclose(got[q][1], reference[q][1])


def test_merge_prefetch_shares_one_pool_across_searches(scenario, monkeypatch, tmp_path):
    """merge_prefetch's download pool must be created ONCE for the whole
    `run_merge` call, not once per search — a search whose downloads land
    early frees its share of that shared pool for a slower search's downloads
    instead of sitting on a dedicated pool nobody else can reach (see
    merge.py's `run_merge` docstring). Exercises the real S3-shaped prefetch
    code path locally by forcing `Store.is_s3 = True` on top of a real local
    filesystem — `_plan_prefetch`/`_fetch_range` only ever call generic
    pyarrow FileSystem methods, so this is a faithful exercise of the same
    code an S3 run would take, not a mock of it."""
    import nova_bf.merge as merge_mod

    cfg, qids, reference = scenario

    # A second search, reusing the first search's partial bytes verbatim —
    # this test is about pool sharing/correctness of the merge path, not
    # distinct per-search ground truth (other tests already cover that).
    second = SearchSpec(name="test2", k=K)
    cfg.searches = [*cfg.searches, second]
    src_dir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    dst_dir = tmp_path / "out" / partial_dir(cfg, second)
    dst_dir.mkdir(parents=True)
    for f in src_dir.iterdir():
        (dst_dir / f.name).write_bytes(f.read_bytes())

    cfg.params.merge_prefetch = True

    real_store_cls = merge_mod.Store

    def _forced_s3_store(uri):
        store = real_store_cls(uri)
        store.is_s3 = True  # force the prefetch branch over a real local Store
        return store

    monkeypatch.setattr(merge_mod, "Store", _forced_s3_store)

    pool_sizes: list[int | None] = []

    class _CountingExecutor(merge_mod.ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            pool_sizes.append(kwargs.get("max_workers"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(merge_mod, "ThreadPoolExecutor", _CountingExecutor)

    run_merge(cfg)

    assert len(pool_sizes) == 1, f"expected exactly ONE shared thread pool, got {len(pool_sizes)}"

    for spec in cfg.searches:
        t = pq.read_table(f"{cfg.output.path}/{result_name(cfg, spec)}").to_pydict()
        got = {q: (hi, hs) for q, hi, hs in zip(t["query_id"], t["hit_ids"], t["hit_scores"])}
        for q in qids:
            ref_ids, ref_scores = reference[q]
            assert got[q][0] == ref_ids
            assert np.allclose(got[q][1], ref_scores)


def test_mismatched_partial_counts_across_searches_raises(scenario, tmp_path):
    """Every search in one `compute` run is written by the same set of ranks,
    so a mismatched partial count between two searches means some rank died
    partway through writing its per-search outputs — this must raise loudly
    at merge time instead of silently merging the short search from fewer
    ranks than it actually had."""
    cfg, qids, reference = scenario

    second = SearchSpec(name="test2", k=K)
    cfg.searches = [*cfg.searches, second]
    src_dir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    dst_dir = tmp_path / "out" / partial_dir(cfg, second)
    dst_dir.mkdir(parents=True)
    # Copy only W-1 of the W partials — simulates a rank that wrote the first
    # search's partial but died before writing this second search's.
    for f in sorted(src_dir.iterdir())[:-1]:
        (dst_dir / f.name).write_bytes(f.read_bytes())

    with pytest.raises(RuntimeError, match="mismatched partial counts"):
        run_merge(cfg)
