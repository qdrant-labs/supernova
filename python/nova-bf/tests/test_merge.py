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


def test_reduce_bounds_how_many_partials_are_resident(scenario, monkeypatch, tmp_path):
    """The reduce must hold at most `_MERGE_WINDOW` partials at once.

    This is the property the whole partial-major rewrite exists for. The old
    shape opened ALL W partials and read the same query batch from each in
    lockstep; because parquet's smallest read unit is the row group, that cost
    W x row-group, not W x batch -- 176 GB for a 32-rank dense merge, which is
    what actually OOMed. Asserting the fold's ORDER would be wrong (the reduce
    is commutative on purpose); the invariant worth pinning is the ceiling on
    concurrent readers.
    """
    import nova_bf.merge as merge_mod

    cfg, qids, reference = scenario
    cfg.params.merge_ranged_reads = True          # -> Store(ranged_get=True)

    live = 0
    peak = 0
    real_read = merge_mod.Store.read_columns

    def counting_read(self, read_path, columns):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        try:
            return real_read(self, read_path, columns)
        finally:
            live -= 1

    monkeypatch.setattr(merge_mod.Store, "read_columns", counting_read)
    merge_mod.run_merge(cfg)

    assert peak <= merge_mod._MERGE_WINDOW_MAX, (
        f"{peak} partials were resident at once; the cap is "
        f"{merge_mod._MERGE_WINDOW_MAX}. An unbounded reduce is what OOMed at scale."
    )
    assert peak >= 1, "no partial was ever read — the test proves nothing"

    # ...and the answer is still the global top-K, folded partial-by-partial.
    got = _read_result(cfg)
    assert sorted(got) == sorted(qids)
    for q in qids:
        hi, hs, src = got[q]
        ref_ids, ref_scores = reference[q]
        assert hi == ref_ids
        assert np.allclose(hs, ref_scores)
        assert src == f"payload-{q}"
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


def test_merge_window_is_derived_from_bytes_not_a_fixed_count():
    """The window must scale with how big a partial actually is.

    A fixed count is wrong in both directions: one partial is ~0.2 GB for a
    small search and ~5.5 GB for a 100k-query dense one, so the same number is
    either wasteful or an OOM. This pins the shape of the derivation rather
    than a magic value.
    """
    import nova_bf.merge as m

    class _Col:
        def __init__(self, path, n):
            self.path_in_schema = path
            self.total_uncompressed_size = n
            self.total_compressed_size = n // 2      # raw-file term
    class _RG:
        def __init__(self, cols): self._c = cols; self.num_columns = len(cols)
        def column(self, i): return self._c[i]
    class _MD:
        def __init__(self, rgs): self._r = rgs; self.num_row_groups = len(rgs)
        def row_group(self, i): return self._r[i]
    class _R:
        def __init__(self, per_col): self.metadata = _MD([_RG([
            _Col("hit_ids.list.element", per_col), _Col("hit_scores.list.element", per_col),
            _Col("query", 10**9),          # payload: must NOT be counted
        ])])

    hit = ["hit_ids", "hit_scores"]
    small = m._hit_bytes_per_partial([_R(1 << 20)], hit)      # 1 MiB per col
    big   = m._hit_bytes_per_partial([_R(1 << 30)], hit)      # 1 GiB per col
    # Scales with the hit columns and EXCLUDES payload (the 1 GB `query` column
    # must not appear), and is >= the encoded size because a parsed table is
    # bigger than its dictionary-encoded form.
    assert small >= 2 << 20, small
    assert small < (1 << 30), "payload column leaked into the estimate"
    assert big >= 2 << 30, big
    assert big > small * 100, (small, big)

    # ranged_get adds the whole raw file on top -- it is buffered while parsing.
    assert m._hit_bytes_per_partial([_R(1 << 20)], hit, ranged=True) > small

    # A tiny partial gets a deeper window than a huge one, from the same budget.
    w_small = m._merge_window([_R(1 << 20)], hit, 64)
    w_big   = m._merge_window([_R(8 << 30)], hit, 64)
    assert w_small > w_big, (w_small, w_big)
    assert w_small <= m._MERGE_WINDOW_MAX, "must stay under the concurrency cap"
    # A partial larger than the whole budget drops to ONE reader on purpose:
    # keeping the 2-partial overlap floor there would just double an overshoot
    # the budget already says will not fit. Overlap is the thing worth losing.
    assert w_big == 1, (w_big, "huge partials must not keep the overlap floor")
    # ...but a partial that comfortably fits still gets real overlap.
    assert m._merge_window([_R(1 << 20)], hit, 64) >= m._MERGE_WINDOW_MIN
    # and the window never exceeds the number of partials there are to read
    assert m._merge_window([_R(1 << 20)], hit, 1) == 1
