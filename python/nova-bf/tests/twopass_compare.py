"""Tie-aware result comparison for tiles-mode pass two.

Tiles mode (`params.two_pass_rescore: tiles`) sums exact scores in a
different order from the one-pass GEMM. Each side is an fp32 dot within
`gamma_d * sum_i |q_i c_i|` of the real value (Higham's order-free bound,
(R2) in the two-pass proof), so the two can differ by up to

    tol = 2 * gamma_d * B  (+ a few ulp of the score for the divides)

where `B` bounds `sum_i |q_i c_i|` in final score units: 1 for cosine
(`|q||c| / (|q||c|)`), `max|q| * max|c|` for dot. That -- not "a few ULP" --
is the comparison tolerance. Two candidates within `tol` of each other, or of
the k-th score, may swap. `assert_same_results` checks exactly that, on two
result tables (or `to_pydict()` dicts) of the same search:

  * every non-hit column is identical;
  * no hit id appears twice in one list;
  * per query, the same number of hits, and the score lists agree position
    by position within `tol` (both are sorted best first);
  * a hit present on one side only must be a provable near-tie: its score is
    within `tol` of the OTHER side's k-th (last) score;
  * a common hit's score agrees within `tol`;
  * an id at a different POSITION is allowed only where the two scores at
    that position are themselves within `tol` (a reordered near-tie).

`slice` mode keeps the byte-identity assertion (`Table.equals`);
`assert_same(mode, a, b)` picks between them.
"""

from __future__ import annotations

import numpy as np

HIT_COLUMNS = ("hit_ids", "hit_scores")
U = 2.0 ** -24


def tolerance(d: int, norm_product: float = 1.0) -> float:
    """`2 * gamma_d * B` for dimension `d` and `B = norm_product` (see the
    module doc): the largest honest |tiles - one-pass| difference."""
    gamma = d * U / (1.0 - d * U)
    return 2.0 * gamma * float(norm_product)


def norm_product(corpus_dir, query_path, column="dense_embedding") -> float:
    """`max|q| * max|c|` over a fixture's parquet files (the dot metric's B)."""
    import pathlib

    import pyarrow.parquet as pq

    def max_norm(paths):
        best = 0.0
        for p in paths:
            col = pq.read_table(str(p), columns=[column]).column(column)
            v = np.asarray([r for r in col.to_pylist() if r is not None],
                           dtype=np.float64)
            if v.size:
                best = max(best, float(np.linalg.norm(v, axis=1).max()))
        return best

    cdir = pathlib.Path(corpus_dir)
    return (max_norm(sorted(cdir.rglob("*.parquet")))
            * max_norm([pathlib.Path(query_path)]))


def _tol(abs_tol: float, x: float) -> float:
    # The divides add a few roundings relative to the score itself.
    return abs_tol + 8 * float(np.spacing(np.float32(abs(x))))


def assert_same_results(got, want, *, abs_tol: float, label: str = "") -> int:
    """Assert `got` matches `want` up to near-ties within `abs_tol` (use
    `tolerance`); returns the number of queries whose hit lists were not
    bit-identical (for the caller to check the comparison was not vacuous,
    if it wants)."""
    a = got.to_pydict() if hasattr(got, "to_pydict") else dict(got)
    b = want.to_pydict() if hasattr(want, "to_pydict") else dict(want)
    assert set(a) == set(b), (label, sorted(a), sorted(b))
    for col in a:
        if col in HIT_COLUMNS:
            continue
        assert a[col] == b[col], f"{label}: column {col!r} differs"
    n_diff = 0
    for qi, (ia, sa, ib, sb) in enumerate(zip(a["hit_ids"], a["hit_scores"],
                                               b["hit_ids"], b["hit_scores"])):
        where = f"{label}: query {qi}"
        assert len(set(ia)) == len(ia), f"{where}: a hit id repeats in {ia}"
        assert len(set(ib)) == len(ib), f"{where}: a hit id repeats in {ib}"
        if ia == ib and sa == sb:
            continue
        n_diff += 1
        assert len(ia) == len(ib), f"{where}: {len(ia)} vs {len(ib)} hits"
        for x, y in zip(sa, sb):
            assert abs(x - y) <= _tol(abs_tol, y), (
                f"{where}: scores {sa} vs {sb} differ beyond {abs_tol:.3e}")
        if not ia:
            continue
        ka, kb = sa[-1], sb[-1]
        da, db = dict(zip(ia, sa)), dict(zip(ib, sb))
        for hid in set(ia) ^ set(ib):
            mine, other_k = (da[hid], kb) if hid in da else (db[hid], ka)
            assert abs(mine - other_k) <= _tol(abs_tol, other_k), (
                f"{where}: hit {hid!r} (score {mine}) is on one side only and "
                f"is not a near-tie of the k-th score {other_k}")
        for hid in set(ia) & set(ib):
            assert abs(da[hid] - db[hid]) <= _tol(abs_tol, db[hid]), (
                f"{where}: hit {hid!r} scored {da[hid]} vs {db[hid]}")
        for p, (x, y) in enumerate(zip(ia, ib)):
            if x != y:
                assert abs(sa[p] - sb[p]) <= _tol(abs_tol, sb[p]), (
                    f"{where}: position {p} holds {x!r} vs {y!r} at scores "
                    f"{sa[p]} vs {sb[p]}, not a near-tie")
    return n_diff


def assert_same(mode: str, got, want, label: str = "", *,
                abs_tol: float | None = None) -> None:
    """Byte identity in slice mode; near-tie-aware equality in tiles mode."""
    if mode == "tiles":
        assert abs_tol is not None, "tiles mode needs a bound-based abs_tol"
        assert_same_results(got, want, abs_tol=abs_tol, label=label)
    else:
        assert got.equals(want), f"{label} differs"
