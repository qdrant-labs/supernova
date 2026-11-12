"""Static `range` and `match` on integer columns are exact at any size.

`range`: pyarrow used to cast the column to float64 to compare it with the
config's float bound, which raises on values above 2**53 and killed the run.
Now the bound is turned into the integer bound keeping the same rows, so the
column is never rounded (Qdrant's behaviour on an indexed integer field, up to
bounds beyond the type's limits, where Qdrant clips the bound to them).
`match`: the value set is built in the column's type, so a uint64 value
>= 2**63 works and values the type can't hold match nothing.

The oracle is Python's own int-vs-float comparison, which is exact."""
from __future__ import annotations

import math
import operator
import random

import numpy as np
import pyarrow as pa
import pytest

from nova_bf.config import Filter, FilterCondition
from nova_bf.filters import FilterPlan, evaluate

OPS = {"gt": operator.gt, "gte": operator.ge, "lt": operator.lt, "lte": operator.le}
TYPES = [pa.int8(), pa.int16(), pa.int32(), pa.int64(),
         pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]


def _limits(t):
    i = np.iinfo(t.to_pandas_dtype())
    return int(i.min), int(i.max)


def _values(t, r):
    lo, hi = _limits(t)
    pts = [lo, lo + 1, hi, hi - 1, 0, 1, 2, 7, 2**53, 2**53 + 1, 2**60, 2**60 + 1,
           2**60 + 255, 2**63 - 1, 2**63, 2**63 + 5, -2**53 - 1, -2**60]
    pts = [p for p in pts if lo <= p <= hi]
    vals = [r.choice(pts) + r.choice([0, 0, 1, -1]) for _ in range(r.randint(1, 30))]
    vals = [v if lo <= v <= hi else lo for v in vals]
    return [None if r.random() < 0.15 else v for v in vals]


def _bounds(t, r):
    lo, hi = _limits(t)
    b = [float(lo), float(hi), lo - 0.5, hi + 0.5, 2.5, -2.5, 0.0, 7.0, 2.0**53, 2.0**53 + 2,
         2.0**60, 2.0**63, -2.0**63, 2.0**64, 1e20, -1e20, math.inf, -math.inf, math.nan]
    return r.choice(b + [float(r.choice(_values(t, r)) or 0)])


def _dense(m, table):
    if isinstance(m, FilterPlan):     # static-only plans aren't built; keep for safety
        raise AssertionError("static filter returned a plan")
    return np.asarray(m.unpack() if hasattr(m, "unpack") else m)


@pytest.mark.parametrize("t", TYPES, ids=str)
def test_static_range_exact(t):
    for seed in range(300):
        r = random.Random(seed)
        vals = _values(t, r)
        table = pa.table({"v": pa.array(vals, t)})
        rng = {}
        for name in r.sample(list(OPS), r.randint(1, 2)):
            rng[name] = _bounds(t, r)
        got = _dense(evaluate(Filter(must=[FilterCondition(field="v", range=rng)]), table), table)
        want = [v is not None and all(OPS[k](v, b) for k, b in rng.items()) for v in vals]
        assert got.tolist() == want, (str(t), seed, vals, rng)


@pytest.mark.parametrize("t", TYPES, ids=str)
def test_static_match_exact(t):
    lo, hi = _limits(t)
    for seed in range(200):
        r = random.Random(seed)
        vals = _values(t, r)
        table = pa.table({"v": pa.array(vals, t)})
        present = [v for v in vals if v is not None]
        pool = present + [lo - 1, hi + 1, 2**63 + 5, -1, 2**64, 3, 2.5, 5.0, float(2**63),
                          float(2**64), math.inf, math.nan, -0.0]
        pool += [float(v) for v in present[:2]]
        m = r.sample(pool, r.randint(1, 3)) if len(pool) >= 3 else pool
        m = m[0] if len(m) == 1 and r.random() < 0.5 else tuple(m)
        got = _dense(evaluate(Filter(must=[FilterCondition(field="v", match=m)]), table), table)
        ms = list(m) if isinstance(m, tuple) else [m]
        want = [v is not None and any(v == x for x in ms) for v in vals]
        assert got.tolist() == want, (str(t), seed, vals, m)
        dic = pa.table({"v": pa.array(vals, t).dictionary_encode()})
        got_d = _dense(evaluate(Filter(must=[FilterCondition(field="v", match=m)]), dic), dic)
        assert got_d.tolist() == want, ("dict", str(t), seed, vals, m)


@pytest.mark.parametrize("t", TYPES, ids=str)
def test_static_range_exact_dictionary(t):
    for seed in range(100):
        r = random.Random(seed)
        vals = _values(t, r)
        table = pa.table({"v": pa.array(vals, t).dictionary_encode()})
        rng = {name: _bounds(t, r) for name in r.sample(list(OPS), r.randint(1, 2))}
        got = _dense(evaluate(Filter(must=[FilterCondition(field="v", range=rng)]), table), table)
        want = [v is not None and all(OPS[k](v, b) for k, b in rng.items()) for v in vals]
        assert got.tolist() == want, (str(t), seed, vals, rng)


@pytest.mark.parametrize("t", TYPES, ids=str)
def test_range_at_type_limits(t):
    lo, hi = _limits(t)
    vals = [lo, lo + 1, 0, hi - 1, hi]
    table = pa.table({"v": pa.array(vals, t)})
    for name, op in OPS.items():
        for b in (float(lo), float(hi), lo - 0.5, hi + 0.5, float(lo + 1), float(hi - 1)):
            got = evaluate(Filter(must=[FilterCondition(field="v", range={name: b})]), table)
            assert got.tolist() == [op(v, b) for v in vals], (str(t), name, b)


def test_float_match_on_large_int_column_no_longer_crashes():
    t = pa.table({"v": pa.array([2**60 + 1, 1, 5, None], pa.int64())})

    def got(m):
        return evaluate(Filter(must=[FilterCondition(field="v", match=m)]), t).astype(int).tolist()
    assert got(2.5) == [0, 0, 0, 0]
    assert got((5, 2.5)) == [0, 0, 1, 0]
    assert got(5.0) == [0, 0, 1, 0]
    assert got(float(2**63)) == [0, 0, 0, 0]
    assert got(float(2**60)) == [0, 0, 0, 0]          # exact: 2**60 != 2**60 + 1


def test_large_int64_range_no_longer_crashes_and_matches_qdrant_indexed():
    """The values/bounds checked against a live Qdrant 1.18.2 (integer
    payload index) on 2026-11-10."""
    B = 2**60
    t = pa.table({"v": pa.array([B, B + 1, B + 2, B + 64, B + 200, 7, 2**63 - 1, -2**63, None],
                                pa.int64())})

    def ids(rng):
        m = evaluate(Filter(must=[FilterCondition(field="v", range=rng)]), t)
        return [int(i) + 1 for i in np.flatnonzero(m)]
    assert ids({"gt": B}) == [2, 3, 4, 5, 7]
    assert ids({"lte": B + 1}) == [1, 6, 8]            # bound rounds to 2**60
    assert ids({"gte": B + 100}) == [1, 2, 3, 4, 5, 7]
    assert ids({"gte": 2.5}) == [1, 2, 3, 4, 5, 6, 7]
    assert ids({"lt": 7.5}) == [6, 8]


def test_uint64_match_at_and_above_2_63():
    t = pa.table({"v": pa.array([2**63 + 5, 3, None, 2**64 - 1], pa.uint64())})

    def got(m):
        return evaluate(Filter(must=[FilterCondition(field="v", match=m)]), t).astype(int).tolist()
    assert got(2**63 + 5) == [1, 0, 0, 0]
    assert got((2**63 + 5, -1, 2**64)) == [1, 0, 0, 0]     # unrepresentable values dropped
    assert got(-1) == [0, 0, 0, 0]
    assert got(2**64 - 1) == [0, 0, 0, 1]


@pytest.mark.parametrize("combo", [False, True])
def test_static_int_conditions_inside_per_query_filters(combo):
    """The same conditions next to a per-query one (the plan's static step)."""
    from test_combo_row_mask import _mixed_dense
    B = 2**60
    vals = [B, B + 1, None, 2**63 - 1, 5]
    t = pa.table({"v": pa.array(vals, pa.int64()), "c": pa.array(["a", "b", "a", "a", "b"])})
    q = np.array(["a", "b"], dtype=object)
    f = Filter(must=[FilterCondition(field="v", range={"gt": B}),
                     FilterCondition(field="v", match=(B + 1, 2**63 - 1, 2**64)),
                     FilterCondition(field="c", match_from_query="q")])
    m = evaluate(f, t, {"q": q}, combo=combo)
    d = _mixed_dense(m, t, {"q": q}) if isinstance(m, FilterPlan) else m.unpack()
    assert d.astype(int).tolist() == [[0, 0, 0, 1, 0], [0, 1, 0, 0, 0]]
