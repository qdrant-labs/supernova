"""The packed-key fold and the numpy fold must decide identically.

`merge`'s numpy path is the one validated against a live Qdrant, so the GPU
path earns its place only by matching it exactly -- not approximately, and not
just on the scores. A ground truth that changed its tie-breaking depending on
whether the reducer had a GPU would be worse than a slow one.

`NOVA_BF_MERGE_FOLD=torch` runs the packed-key fold on CPU tensors:
`merge_triton.enabled()` is `sample.is_cuda`, so this takes the portable branch
of `_merge_topk` while everything above it -- `tiebreak.pack`, the ordinal
ranking, the argsort of packed keys -- is the same code a GPU would run.
"""
import numpy as np
import pyarrow as pa
import pytest

from nova_bf.merge import _topk_merge


def _lists(rows, typ):
    return pa.array(rows, pa.list_(typ))


def _case(rng, n_partials, b, k, distinct_scores, want_tie, width=None):
    """Row-aligned partials with DELIBERATELY few distinct scores, so almost
    every selection has to be settled by the tiebreak rather than the score."""
    scores, ids, ties = [], [], []
    n = 0
    for _ in range(n_partials):
        srow, irow, trow = [], [], []
        for _ in range(b):
            m = int(rng.integers(0, k + 1))          # ragged: some rows short
            srow.append(rng.integers(0, distinct_scores, size=m).astype(np.float32).tolist())
            if width:
                irow.append([f"{x:0{width}d}" for x in rng.integers(0, 10**6, size=m)])
            else:
                irow.append([f"id-{x}" for x in rng.integers(0, 10**6, size=m)])
            trow.append(list(range(n, n + m)))
            n += m
        scores.append(_lists(srow, pa.float32()))
        ids.append(_lists(irow, pa.large_string()))
        ties.append(_lists(trow, pa.int64()))
    return scores, ids, (ties if want_tie else None)


def _run(monkeypatch, mode, args, k):
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", mode)
    i, s, t = _topk_merge(*args, k)
    return i.to_pylist(), s.to_pylist(), (t.to_pylist() if t is not None else None)


@pytest.mark.parametrize("want_tie", [False, True])
@pytest.mark.parametrize("n_partials", [1, 2, 3])
@pytest.mark.parametrize("width", [None, 12])
def test_torch_fold_matches_numpy_exactly(monkeypatch, want_tie, n_partials, width):
    rng = np.random.default_rng(17 + n_partials + int(want_tie) + (width or 0))
    k = 8
    args = _case(rng, n_partials, b=40, k=k, distinct_scores=3,
                 want_tie=want_tie, width=width)
    a = _run(monkeypatch, "numpy", args, k)
    b_ = _run(monkeypatch, "torch", args, k)
    assert a[1] == b_[1], "scores diverge"
    assert a[0] == b_[0], "ids diverge -- the tiebreak decided differently"
    assert a[2] == b_[2], "tie ordinates diverge"


@pytest.mark.parametrize("mode", ["numpy", "torch"])
@pytest.mark.parametrize("want_tie", [False, True])
def test_fold_matches_a_plain_python_reference(monkeypatch, mode, want_tie):
    """Both backends against the rule stated in prose: highest score first, and
    among equal scores the lowest id (or lowest `hit_tie`). Agreement between
    two implementations is not correctness if both share a mistake."""
    rng = np.random.default_rng(5)
    k = 6
    scores, ids, ties = _case(rng, 3, b=30, k=k, distinct_scores=2, want_tie=want_tie)
    got_i, got_s, _ = _run(monkeypatch, mode, (scores, ids, ties), k)

    tl = ties if want_tie else None
    for r in range(30):
        cand = []
        for w in range(3):
            sw, iw = scores[w][r].as_py(), ids[w][r].as_py()
            tw = tl[w][r].as_py() if want_tie else None
            for j in range(len(sw)):
                cand.append((-sw[j], tw[j] if want_tie else iw[j], iw[j], sw[j]))
        cand.sort(key=lambda c: (c[0], c[1]))
        assert got_i[r] == [c[2] for c in cand[:k]], f"row {r}"
        assert got_s[r] == [c[3] for c in cand[:k]], f"row {r}"


@pytest.mark.parametrize("mode,expect", [("numpy", "numpy"), ("torch", "torch:cpu")])
def test_manifest_records_the_fold_that_actually_ran(tmp_path, monkeypatch, mode, expect):
    """Not the env switch -- what executed. A run that asked for a GPU fold and
    silently got numpy must not claim the GPU in its provenance."""
    import json

    from nova_bf import manifest as run_manifest
    from nova_bf import merge as merge_mod
    from nova_bf.results import partial_dir
    from tests.test_merge_regressions import _cfg, _write_partials

    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", mode)
    cfg = _cfg(str(tmp_path / "out"))
    _write_partials(cfg, tmp_path / "out" / partial_dir(cfg, cfg.searches[0]),
                    n_partials=3, n_queries=20)
    merge_mod.run_merge(cfg)

    doc = json.loads(
        (tmp_path / "out" / run_manifest.manifest_name(cfg, "merge")).read_text())
    assert [e["merge_fold"] for e in doc["searches"]] == [[expect]], doc["searches"]


def test_variable_width_ids_decline_the_packed_fold_unless_asked(monkeypatch):
    """Ranking every candidate only pays when the ids pack into lanes. With
    ragged ids the lane path declines, so the extra ranking would land on
    Arrow's CPU string sort -- worse than the ambiguous-row ranking it replaced.
    An explicit request is still obeyed."""
    import torch

    from nova_bf import merge as merge_mod

    monkeypatch.delenv("NOVA_BF_MERGE_FOLD", raising=False)
    monkeypatch.setattr(merge_mod, "_fold_device",
                        lambda forced_only=False: False if forced_only
                        else torch.device("cpu"))
    rng = np.random.default_rng(1)
    k = 4
    ragged = _case(rng, 2, b=10, k=k, distinct_scores=2, want_tie=False)
    fixed = _case(rng, 2, b=10, k=k, distinct_scores=2, want_tie=False, width=12)

    merge_mod._reset_fold_used()
    _topk_merge(*ragged, k)
    assert merge_mod._FOLD_USED == {"numpy"}, merge_mod._FOLD_USED

    merge_mod._reset_fold_used()
    _topk_merge(*fixed, k)
    assert merge_mod._FOLD_USED == {"torch:cpu"}, merge_mod._FOLD_USED


def test_ranking_every_row_does_not_copy_the_id_buffers():
    """`take` on a complete selection reproduces an array we already hold. At
    the production batch shape that copy is ~940 MB per fold."""
    from nova_bf import merge as merge_mod

    flat = pa.array(["b", "a", "d", "c"], pa.large_string())
    scatter = [(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), flat)]
    seen = []
    real = merge_mod.build_ordinals
    try:
        merge_mod.build_ordinals = lambda subs: seen.append(subs) or real(subs)
        merge_mod._id_tie_grid(scatter, np.array([0, 1]), 2, 2)   # every row
        assert seen[0][0] is flat, "complete selection must not copy"
        seen.clear()
        merge_mod._id_tie_grid(scatter, np.array([1]), 2, 2)      # one row
        assert seen[0][0] is not flat, "a partial selection must still narrow"
        assert len(seen[0][0]) == 2
    finally:
        merge_mod.build_ordinals = real


# ---------------------------------------------------------------------------
# Defects found by adversarial review, 2026-09-05.
# ---------------------------------------------------------------------------

def _nan_case():
    S = [_lists([[5.0, float("nan")]], pa.float32()),
         _lists([[3.0, 1.0]], pa.float32())]
    I = [_lists([["aaa", "bbb"]], pa.large_string()),
         _lists([["ccc", "ddd"]], pa.large_string())]
    return S, I


@pytest.mark.parametrize("want_tie", [False, True])
def test_a_nan_score_is_not_selected_by_either_fold(monkeypatch, want_tie):
    """`score_order_key` maps a positive NaN ABOVE +inf, so a packed NaN wins a
    top-k slot and is then dropped by the `> -inf` filter -- costing a REAL hit,
    silently. The numpy fold sorts NaN last (every comparison is false), and the
    packed-key fold has to agree or a shard boundary changes the answer."""
    S, I = _nan_case()
    T = [_lists([[1, 2]], pa.int64()), _lists([[3, 4]], pa.int64())] if want_tie else None
    a = _run(monkeypatch, "numpy", (S, I, T), 2)
    b = _run(monkeypatch, "torch", (S, I, T), 2)
    assert a[0] == [["aaa", "ccc"]], a[0]      # the NaN slot goes to a real hit
    assert a == b


def test_nan_does_not_make_the_result_depend_on_fold_order(monkeypatch):
    """Partials arrive off a queue fed by concurrent readers, so fold order is
    nondeterministic. `_topk_merge` is claimed to be a commutative monoid; a
    mis-ordered NaN broke that, giving three different answers for one input."""
    import itertools

    P = [(_lists([[9.0, float("nan")]], pa.float32()),
          _lists([["00000001", "00000009"]], pa.large_string())),
         (_lists([[8.0, 2.0]], pa.float32()),
          _lists([["00000002", "00000003"]], pa.large_string())),
         (_lists([[6.0, 1.0]], pa.float32()),
          _lists([["00000005", "00000004"]], pa.large_string()))]
    for mode in ("numpy", "torch"):
        seen = {str(_run(monkeypatch, mode, ([P[i][0] for i in p],
                                             [P[i][1] for i in p], None), 2))
                for p in itertools.permutations(range(3))}
        assert len(seen) == 1, f"{mode}: {len(seen)} answers for one input"


def test_null_ids_are_refused_by_both_folds(monkeypatch):
    """A null hit id has no ordering position and is meaningless in ground
    truth. It used to be TOLERATED by the numpy fold -- which ranks only
    ambiguous rows, so a null in an unambiguous row was never looked at -- and
    rejected by the packed-key fold, which ranks every row. So whether a null
    was caught depended on whether some other candidate happened to tie. Now the
    scatter refuses it outright, before either fold is chosen."""
    S = [_lists([[5.0, 4.0]], pa.float32()), _lists([[3.0, 2.0]], pa.float32())]
    I = [_lists([["aaa", None]], pa.large_string()),
         _lists([["ccc", "ddd"]], pa.large_string())]
    for mode in ("numpy", "torch", "cpu"):
        with pytest.raises(RuntimeError, match="null value"):
            _run(monkeypatch, mode, (S, I, None), 2)


def test_hit_ids_split_differently_from_hit_scores_is_refused(monkeypatch):
    """The id grid is addressed by the SCORE lengths, so a longer hit_ids row
    slid every later partial's ids against the scores -- each score reported
    under another document's id, with no error. The object-array scatter this
    replaced caught it by accident (numpy refuses the broadcast)."""
    S = [_lists([[9.0, 8.0]], pa.float32()), _lists([[7.0, 6.0]], pa.float32())]
    I = [_lists([["a", "b", "EXTRA"]], pa.large_string()),
         _lists([["c", "d"]], pa.large_string())]
    for mode in ("numpy", "torch"):
        with pytest.raises(RuntimeError, match="split differently"):
            _run(monkeypatch, mode, (S, I, None), 4)
