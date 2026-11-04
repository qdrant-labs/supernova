"""The multi-way (deferred) fold must reproduce the pairwise fold bit for bit.

`_reduce` used to fold every partial into the k-wide running state on arrival,
so with trimmed partials (~k/N hits per query each) the merge did ~N*k work per
query. It now buffers short partials per batch and folds the state plus every
buffered partial in ONE `_topk_merge` call, flushing before the buffer would
hold more than k entries in any row (so the candidate grid stays <= 2k wide,
the width `_resolve_batch_rows` sizes for). Full-width partials still flush
one at a time, exactly as before.

Exactness rests on `_topk_merge` being a top-k under ONE total order -- score
descending, tiebreak ascending, then arrival position -- after dropping -inf
padding and NaN scores. Top-k under a total order is associative, so how the
partials are grouped cannot change the result. (The generated ids and ties
are globally unique, so the arrival-position key itself never decides a
selection here; what is pinned is everything above it.) These tests pin that
claim against (a) the old pairwise chain in HEAD's padded k-per-input layout, (b) an independent brute-force oracle, and
(c) whole `run_merge` outputs under every window size, both tiebreaks and both
fold backends (`torch` runs the packed-key code a GPU would on CPU tensors).
"""

from __future__ import annotations

import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import nova_bf.merge as merge_mod
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    OutputConfig,
    ParamsConfig,
    QueriesConfig,
    SearchSpec,
)
from nova_bf.merge import _topk_merge
from nova_bf.results import build_result_table, partial_dir, result_name


def _seed(*params) -> int:
    """A per-case seed that is the same in every process (`hash()` of a
    tuple with strings is salted by PYTHONHASHSEED, so a failure could not
    be replayed)."""
    return zlib.crc32(repr(params).encode())


def _lists(rows, typ):
    return pa.array(rows, pa.list_(typ))


def _gen(rng, n_partials, n_q, k, shape, distinct, nan_p=0.0, id_width=None):
    """Per-partial, per-query candidate lists of (id, score, tie).

    `distinct` small -> exact score ties inside AND across partials, so the
    tiebreak decides most selections. Ties and ids are globally unique but
    their orders deliberately disagree, so the wrong tiebreak cannot pass.
    """
    uid = 0
    perm = rng.permutation(10**6)
    parts = []
    for p in range(n_partials):
        rows = []
        for _ in range(n_q):
            if shape == "full":
                m = k
            elif shape == "trimmed":
                m = int(rng.binomial(k, min(1.0, 1.1 / n_partials)))
            elif shape == "mixed":
                m = k if p % 3 == 0 else int(rng.integers(0, max(1, k // 2) + 1))
            else:  # ragged
                m = int(rng.integers(0, k + 1))
            row = []
            for _ in range(m):
                s = float(rng.integers(0, distinct))
                if nan_p and rng.random() < nan_p:
                    s = float("nan") if rng.random() < 0.5 else float("-inf")
                tag = int(perm[uid])
                ident = (f"{tag:0{id_width}d}" if id_width else f"id{tag}")
                row.append((ident, s, -uid))
                uid += 1
            # a partial's own list is score-descending, as compute writes it
            row.sort(key=lambda c: -c[1] if c[1] == c[1] else np.inf)
            rows.append(row)
        parts.append(rows)
    return parts


def _arrow(part):
    return (_lists([[s for _, s, _ in r] for r in part], pa.float32()),
            _lists([[i for i, _, _ in r] for r in part], pa.large_string()),
            _lists([[t for _, _, t in r] for r in part], pa.int64()))


def _py(out):
    ids, sc, ti = out
    ids = merge_mod._as_ids_array(ids)
    return (ids.to_pylist(), sc.to_pylist(),
            ti.to_pylist() if ti is not None else None)


_PAD = ("", float("-inf"), np.iinfo(np.int64).max)


def _pad_to_k(part, k):
    """Every row padded to exactly k entries with what HEAD's grid padding
    held (-inf score, worst tie). Fed to the row-packed `_topk_merge`, input
    w's entry j then lands at column w*k + j -- HEAD's layout -- so this
    reference reproduces the OLD grid, not the new packed one."""
    return [list(r) + [_PAD] * (k - len(r)) for r in part]


def _rows(st):
    ids, sc, ti = _py(st)
    return [list(zip(i, s, t if t is not None else [0] * len(i)))
            for i, s, t in zip(ids, sc, ti if ti is not None else [None] * len(ids))]


def _pairwise(parts, k, use_tie):
    """The OLD reduce in the OLD layout: seed with partial 0, then fold one
    partial at a time, every input padded to k columns (HEAD's
    `col = w*k + within` grid)."""
    st_rows = None
    st = None
    for p in parts:
        ins = [p] if st_rows is None else [st_rows, p]
        arrs = [_arrow(_pad_to_k(x, k)) for x in ins]
        st = _topk_merge([a[0] for a in arrs], [a[1] for a in arrs],
                         [a[2] for a in arrs] if use_tie else None, k)
        st_rows = _rows(st)
    return _py(st)


def _grouped(parts, k, use_tie, group):
    """The NEW shape: state plus `group` partials per call."""
    st = None
    for g in range(0, len(parts), group):
        arrs = [_arrow(p) for p in parts[g:g + group]]
        sc = [a[0] for a in arrs]
        ids = [a[1] for a in arrs]
        ti = [a[2] for a in arrs]
        if st is not None:
            sc, ids, ti = [st[1]] + sc, [st[0]] + ids, [st[2]] + ti
        st = _topk_merge(sc, ids, ti if use_tie else None, k)
    return _py(st)


def _oracle(parts, k, use_tie):
    """Brute force: drop NaN/-inf, sort by (-score, tiebreak, arrival)."""
    ids_out, sc_out = [], []
    for q in range(len(parts[0])):
        cands = []
        for pi, p in enumerate(parts):
            for j, (i, s, t) in enumerate(p[q]):
                if s == s and s > -np.inf:
                    cands.append((-s, t if use_tie else i, pi, j, i, s))
        cands.sort(key=lambda c: c[:4])
        top = cands[:k]
        ids_out.append([c[4] for c in top])
        sc_out.append([c[5] for c in top])
    return ids_out, sc_out


@pytest.mark.parametrize("fold", ["numpy", "torch"])
@pytest.mark.parametrize("use_tie", [True, False])
@pytest.mark.parametrize("shape", ["trimmed", "ragged", "full", "mixed"])
@pytest.mark.parametrize("k", [1, 3, 9])
def test_grouped_fold_is_bit_identical_to_pairwise(monkeypatch, fold, use_tie, shape, k):
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", fold)
    rng = np.random.default_rng(_seed(fold, use_tie, shape, k))
    n = 7
    parts = _gen(rng, n, n_q=23, k=k, shape=shape, distinct=3,
                 nan_p=0.15, id_width=None if fold == "numpy" else 9)
    ref = _pairwise(parts, k, use_tie)
    want_ids, want_sc = _oracle(parts, k, use_tie)
    assert ref[0] == want_ids
    np.testing.assert_array_equal(
        np.array([x for r in ref[1] for x in r], np.float32),
        np.array([x for r in want_sc for x in r], np.float32))
    for group in (1, 2, 3, n):
        got = _grouped(parts, k, use_tie, group)
        assert got == ref, f"group={group}"


def test_compact_layout_handles_all_empty_and_k_above_candidates():
    """Rows with no candidates anywhere, and rows with fewer than k in total."""
    k = 5
    sc = [_lists([[], [1.0], []], pa.float32()),
          _lists([[], [], [2.0, 2.0]], pa.float32())]
    ids = [_lists([[], ["b"], []], pa.large_string()),
           _lists([[], [], ["z", "a"]], pa.large_string())]
    i, s, _ = _topk_merge(sc, ids, None, k)
    assert merge_mod._as_ids_array(i).to_pylist() == [[], ["b"], ["a", "z"]]
    assert s.to_pylist() == [[], [1.0], [2.0, 2.0]]
    sc0 = [_lists([[], []], pa.float32())] * 3
    ids0 = [_lists([[], []], pa.large_string())] * 3
    i, s, _ = _topk_merge(sc0, ids0, None, k)
    assert s.to_pylist() == [[], []]


# ---------------------------------------------------------------------------
# End to end: run_merge under every window size == the pairwise chain.
# ---------------------------------------------------------------------------

def _cfg(root, k, tiebreak, window, batch):
    return BruteForceConfig(
        corpus=CorpusConfig(path=f"{root}/corpus", id_column="id"),
        queries=QueriesConfig(path=f"{root}/queries.parquet"),
        output=OutputConfig(path=root),
        params=ParamsConfig(tiebreak=tiebreak, merge_window=window,
                            merge_batch_size=batch),
        searches=[SearchSpec(name="s", k=k)],
    )


def _write(cfg, parts, use_tie, nan_hits=None):
    pdir = f"{cfg.output.path}/{partial_dir(cfg, cfg.searches[0])}"
    import os
    os.makedirs(pdir, exist_ok=True)
    n_q = len(parts[0])
    qids = [f"q{i}" for i in range(n_q)]
    for r, p in enumerate(parts):
        tbl = build_result_table(
            qids, {"src": [f"pl{i}" for i in range(n_q)]},
            [[i for i, _, _ in row] for row in p],
            [[s for _, s, _ in row] for row in p],
            hit_tie=[[t for _, _, t in row] for row in p] if use_tie else None,
            nan_hits=nan_hits[r] if nan_hits is not None else None,
        )
        pq.write_table(tbl, f"{pdir}/rank{r:03d}.parquet")


@pytest.mark.parametrize("fold", ["numpy", "torch"])
@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
@pytest.mark.parametrize("shape", ["trimmed", "ragged", "full", "mixed"])
@pytest.mark.parametrize("window", [1, 2, 9])
def test_run_merge_matches_pairwise_for_every_window(
        tmp_path, monkeypatch, fold, tiebreak, shape, window):
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", fold)
    use_tie = tiebreak == "ordinal"
    k, n, n_q = 6, 9, 31
    rng = np.random.default_rng(_seed(fold, tiebreak, shape, window))
    parts = _gen(rng, n, n_q, k, shape, distinct=4, nan_p=0.05,
                 id_width=None if shape == "ragged" else 8)
    cfg = _cfg(str(tmp_path), k, tiebreak, window, batch=7)
    _write(cfg, parts, use_tie)

    calls: list[int] = []
    real = merge_mod._topk_merge

    def spy(score_lists, id_lists, tie_lists, k, **kw):
        calls.append(len(score_lists))
        # the grid is never wider than 2k: state (<= k) + pending (<= k)
        tot = sum(np.asarray(sl.value_lengths().to_numpy(zero_copy_only=False),
                             np.int64) for sl in score_lists)
        assert tot.max(initial=0) <= 2 * k
        return real(score_lists, id_lists, tie_lists, k, **kw)

    monkeypatch.setattr(merge_mod, "_topk_merge", spy)
    merge_mod.run_merge(cfg)
    out = pq.read_table(f"{cfg.output.path}/{result_name(cfg, cfg.searches[0])}")

    # The merge reads partials in completion order; the ORDER does not matter
    # for the result (pinned by the grouped test) so compare to rank order.
    ref_ids, ref_sc, _ = _pairwise(parts, k, use_tie)
    assert out.column("hit_ids").to_pylist() == ref_ids
    np.testing.assert_array_equal(
        np.array([x for r in out.column("hit_scores").to_pylist() for x in r], np.float32),
        np.array([x for r in ref_sc for x in r], np.float32))
    assert out.column("src").to_pylist() == [f"pl{i}" for i in range(n_q)]
    if shape == "trimmed":
        # the point of the change: far fewer folds than partials per batch
        assert max(calls) > 2, calls


@pytest.mark.parametrize("fold", ["numpy", "torch"])
def test_nan_truncation_after_deferred_fold(tmp_path, monkeypatch, fold):
    """`nan_hits` cuts each affected query to k - N AFTER the fold; deferral
    must hand it the same final state."""
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", fold)
    k, n, n_q = 5, 6, 12
    rng = np.random.default_rng(3)
    parts = _gen(rng, n, n_q, k, "trimmed", distinct=3)
    nan_hits = [np.zeros(n_q, np.int64) for _ in range(n)]
    nan_hits[1][[0, 4, 7]] = [1, 2, 9]
    cfg = _cfg(str(tmp_path), k, "ordinal", 3, batch=5)
    _write(cfg, parts, True, nan_hits=nan_hits)
    merge_mod.run_merge(cfg)
    out = pq.read_table(f"{cfg.output.path}/{result_name(cfg, cfg.searches[0])}")
    ref_ids, _, _ = _pairwise(parts, k, True)
    tot = sum(nan_hits)
    want = [r[:max(0, k - int(tot[q]))] for q, r in enumerate(ref_ids)]
    assert out.column("hit_ids").to_pylist() == want


def test_full_width_partials_still_fold_one_at_a_time(tmp_path, monkeypatch):
    """Untrimmed partials keep today's call shape (seed, then 2-way folds), so
    the dense device fold and its memory bound are untouched."""
    k, n, n_q = 4, 5, 10
    rng = np.random.default_rng(5)
    parts = _gen(rng, n, n_q, k, "full", distinct=50)
    cfg = _cfg(str(tmp_path), k, "ordinal", 2, batch=None)
    _write(cfg, parts, True)
    calls: list[int] = []
    real = merge_mod._topk_merge

    def spy(score_lists, *a, **kw):
        calls.append(len(score_lists))
        return real(score_lists, *a, **kw)

    monkeypatch.setattr(merge_mod, "_topk_merge", spy)
    merge_mod.run_merge(cfg)
    assert calls == [1] + [2] * (n - 1), calls


def _sparse_parts(n, n_q, every):
    """`n` partials whose rows are almost all empty: partial p holds one hit
    for query p % n_q when p % every == 0. No row ever reaches k, so only
    the pending-count cap bounds how many partials wait for a fold."""
    parts = []
    for p in range(n):
        rows = [[] for _ in range(n_q)]
        if p % every == 0:
            rows[p % n_q] = [(f"id{p}", float(p % 5), -p)]
        parts.append(rows)
    return parts


def test_pending_partials_are_capped_by_count(tmp_path, monkeypatch):
    """Mostly-empty rows (a filtered search) never reach k pending entries;
    the merge still flushes every `_MAX_PENDING_PARTIALS` partials so it
    does not keep the whole run pending. Same result as the pairwise chain."""
    monkeypatch.setattr(merge_mod, "_MAX_PENDING_PARTIALS", 4)
    k, n, n_q = 5, 30, 4
    parts = _sparse_parts(n, n_q, every=3)
    cfg = _cfg(str(tmp_path), k, "ordinal", 3, batch=None)
    _write(cfg, parts, True)
    calls: list[int] = []
    real = merge_mod._topk_merge

    def spy(score_lists, *a, **kw):
        calls.append(len(score_lists))
        return real(score_lists, *a, **kw)

    monkeypatch.setattr(merge_mod, "_topk_merge", spy)
    merge_mod.run_merge(cfg)
    # state + at most the cap
    assert max(calls) <= 4 + 1 and len(calls) >= n // 4, calls
    out = pq.read_table(f"{cfg.output.path}/{result_name(cfg, cfg.searches[0])}")
    ref_ids, ref_sc, _ = _pairwise(parts, k, True)
    assert out.column("hit_ids").to_pylist() == ref_ids
    assert out.column("hit_scores").to_pylist() == ref_sc


def test_failed_group_fold_names_every_pending_partial(tmp_path, monkeypatch):
    """A flush folds several pending partials at once; the bad data may be in
    any of them, so the error names the whole group, not only the partial
    whose arrival triggered the flush."""
    monkeypatch.setattr(merge_mod, "_MAX_PENDING_PARTIALS", 4)
    k, n, n_q = 5, 8, 4
    parts = _sparse_parts(n, n_q, every=1)
    cfg = _cfg(str(tmp_path), k, "ordinal", 1, batch=None)
    _write(cfg, parts, True)
    real = merge_mod._topk_merge
    seen: list[int] = []

    def boom(score_lists, *a, **kw):
        if len(score_lists) >= 3:
            seen.append(len(score_lists))
            raise ValueError("injected bad partial")
        return real(score_lists, *a, **kw)

    monkeypatch.setattr(merge_mod, "_topk_merge", boom)
    with pytest.raises(ValueError, match="injected") as ei:
        merge_mod.run_merge(cfg)
    notes = getattr(ei.value, "__notes__", [])
    group = [x for x in notes if x.startswith("while folding batch")]
    assert group, notes
    n_pending = seen[0] - (1 if "running state" in group[0] else 0)
    assert n_pending >= 2
    assert group[0].count(".parquet") == n_pending, group[0]
