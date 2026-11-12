"""`compute._plan_mask_device` (the one GPU mask function for per-query
filters) against the CPU reference `filters.evaluate(combo=False)` and the
legacy `compute._gpu_evaluate`, on device "cpu": structured-only and mixed
plans with static conditions in every group (static text included), int /
string / dictionary columns with nulls, MatchAny lists, ranges with NaN
bounds, query-row narrowing, slices and compacted rows; plus the per-file
static cache (mutation-free, shared by concurrent filters, and static text
tokenized once per file). Adopted from review r13's fuzz harness."""
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pytest

torch = pytest.importorskip("torch")

from nova_bf import compute as C
from nova_bf import filters as F
from nova_bf.config import Filter, FilterCondition, RangeCondition, RangeFromQuery
from nova_bf.filters import TextPart, evaluate

from test_combo_row_mask import WORDS, DOMS, _rand_text, _rand_phrase, _text_cond


@pytest.fixture(autouse=True)
def _no_guard(monkeypatch):
    monkeypatch.setattr(F, "_COMBO_PARTS_GUARD", False)


def table(rng, n):
    cats = [rng.choice(["a", "b", "c", None]) for _ in range(n)]
    return pa.table({
        "text": pa.array([_rand_text(rng, WORDS) for _ in range(n)], type=pa.string()),
        "url": pa.array([_rand_text(rng, DOMS, 1, 3) for _ in range(n)], type=pa.string()),
        "cat": pa.array(cats, type=pa.string()),
        "dcat": pa.array(cats, type=pa.string()).dictionary_encode(),
        "num": pa.array([None if rng.random() < 0.15 else rng.randint(0, 6) for _ in range(n)],
                        type=pa.int64()),
        "cost": pa.array([None if rng.random() < 0.1 else rng.uniform(0, 10) for _ in range(n)],
                         type=pa.float64()),
    })


def qvals(rng, nq):
    cols = {}
    for c in ("kw", "kw2", "neg"):
        cols[c] = [_rand_phrase(rng, WORDS) for _ in range(nq)]
    for i in range(4):
        cols[f"d{i}"] = [_rand_phrase(rng, DOMS) for _ in range(nq)]
    cols["cat_q"] = [rng.choice(["a", "b", "c", "zz", None]) for _ in range(nq)]
    cols["num_q"] = [rng.choice([0, 1, 2, 3, 9, None]) for _ in range(nq)]
    cols["cats_q"] = [rng.choice([None, [], ["a"], ["a", "b"], ["c", None], ["zz"], ["b", "c"]])
                      for _ in range(nq)]
    cols["lo_q"] = [float("nan") if rng.random() < 0.15 else rng.uniform(0, 8) for _ in range(nq)]
    cols["hi_q"] = [float("nan") if rng.random() < 0.15 else rng.uniform(2, 10) for _ in range(nq)]
    cols["nlo_q"] = [float(rng.randint(0, 5)) for _ in range(nq)]
    return {k: C._to_query_array(v) for k, v in cols.items()}


def static_cond(rng):
    k = rng.randrange(6)
    if k == 0:
        return FilterCondition(field="text", match_text=rng.choice(["alpha", "beta gamma", "eta"]))
    if k == 1:
        return FilterCondition(field="cat", match=rng.choice(["a", "b"]))
    if k == 2:
        return FilterCondition(field="dcat", match=("a", "c"))
    if k == 3:
        return FilterCondition(field="num", match=rng.choice([1, 2, (0, 3)]))
    if k == 4:
        return FilterCondition(field="cost", range=RangeCondition(lt=rng.uniform(2, 9)))
    return FilterCondition(field="num", range=RangeCondition(gte=rng.randint(0, 4)))


def struct_cond(rng):
    k = rng.randrange(7)
    if k == 0:
        return FilterCondition(field="cat", match_from_query="cat_q")
    if k == 1:
        return FilterCondition(field="dcat", match_from_query="cat_q")
    if k == 2:
        return FilterCondition(field="num", match_from_query="num_q")
    if k == 3:
        return FilterCondition(field=rng.choice(["cat", "dcat"]), match_from_query="cats_q")
    if k == 4:
        b = rng.choice([dict(gt="lo_q"), dict(gte="lo_q", lte="hi_q"), dict(lt="hi_q"),
                        dict(gt="lo_q", lt="hi_q")])
        return FilterCondition(field="cost", range_from_query=RangeFromQuery(**b))
    if k == 5:
        return FilterCondition(field="num", range_from_query=RangeFromQuery(gte="nlo_q"))
    return FilterCondition(field="num", range_from_query=RangeFromQuery(lt="lo_q"))


def rand_filter(rng, text_p):
    groups = {"must": [], "should": [], "must_not": []}
    for g in groups:
        for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):
            r = rng.random()
            if r < 0.4:
                groups[g].append(struct_cond(rng))
            elif r < 0.4 + text_p:
                groups[g].append(_text_cond(rng))
            else:
                groups[g].append(static_cond(rng))
    if not any(F._is_pq_struct(c) for cs in groups.values() for c in cs):
        groups[rng.choice(list(groups))].append(struct_cond(rng))
    return Filter(**{g: tuple(dict.fromkeys(cs)) for g, cs in groups.items()})


def dense_ref(filt, t, qv):
    m = evaluate(filt, t, qv, combo=False)
    return m.unpack() if hasattr(m, "unpack") else m


def leaf_state(filt, t, qv_full, idx=None, include_static=False):
    leaf_arrays, qgpu = {}, {}
    for c in filt.all_conditions():
        if F._is_pq_struct(c):
            st, vocab = C._build_gpu_leaf_state(c, qv_full, "cpu")
            if idx is not None:
                st = C._narrow_gpu_leaf_state(st, torch.as_tensor(idx, dtype=torch.long))
            qgpu[c] = st
            leaf_arrays[c] = C._corpus_leaf_array(c, t, vocab, None)
        elif include_static and c.match_text_from_query is None:
            leaf_arrays[c] = C._corpus_leaf_array(c, t, None, None)
    return leaf_arrays, qgpu


def row_sets(rng, n):
    out = [slice(0, n), slice(0, 1), slice(n - 1, n)]
    for _ in range(4):
        r0 = rng.randrange(n)
        out.append(slice(r0, rng.randint(r0 + 1, n)))
    for _ in range(3):
        out.append(np.array(sorted(rng.sample(range(n), rng.randint(1, n))), dtype=np.int64))
    return out


def rows_t(tr):
    if isinstance(tr, slice):
        return torch.arange(tr.start, tr.stop, dtype=torch.long)
    return torch.from_numpy(tr)


def check(got, want, ctx):
    if not want.any():
        # nothing allowed: None, or an all-false mask (the union is a superset)
        assert got is None or not bool(got.any()), ctx
        return
    assert got is not None, ctx
    assert got.dtype == torch.bool and tuple(got.shape) == want.shape, ctx
    assert np.array_equal(got.numpy(), want), ctx


@pytest.mark.parametrize("seed", range(600))
def test_plan_device_vs_reference(seed, monkeypatch):
    rng = random.Random(130000 + seed)
    n = rng.choice([1, 7, 8, 9, rng.randint(1, 150)])
    t = table(rng, n)
    nq = rng.randint(1, 14)
    qv_full = qvals(rng, nq)
    text_p = rng.choice([0.0, 0.3])
    filt = rand_filter(rng, text_p)
    narrow = rng.random() < 0.4
    idx = np.array(sorted(rng.sample(range(nq), rng.randint(1, nq))), dtype=np.int64) if narrow else None
    qv = {k: v[idx] for k, v in qv_full.items()} if narrow else qv_full
    want = dense_ref(filt, t, qv)
    cache = {}
    plan = evaluate(filt, t, qv, combo=True, static_cache=cache)
    assert isinstance(plan, (F.FilterPlan, F.PackedRowMask)), type(plan)
    if not isinstance(plan, F.FilterPlan):
        return
    snap = {c: m.copy() for c, m in cache.items()}
    assert plan.n_queries == want.shape[0]
    leaf_arrays, qgpu = leaf_state(filt, t, qv_full, idx)
    leaf_gpu = {c: torch.from_numpy(a) for c, a in leaf_arrays.items()}
    plan.struct_union = C._row_union_from_gpu_leaves(filt, leaf_arrays, qv_full, n,
                                                     include_static=False)
    assert not (want.any(axis=0) & ~plan.union_rows()).any(), "union not superset"
    assert plan.any() or not want.any()
    eligible = C._gpu_eligible(filt)
    if eligible:
        la2, _ = leaf_state(filt, t, qv_full, idx, include_static=True)
        lg2 = {c: torch.from_numpy(a) for c, a in la2.items()}
        u_old = C._row_union_from_gpu_leaves(filt, la2, qv_full, n)
        if u_old is not None:
            assert not (want.any(axis=0) & ~u_old).any()
    for tr in row_sets(rng, n):
        w = want[:, tr]
        got = C._plan_mask_device(plan, rows_t(tr), tr, leaf_gpu, qgpu, "cpu")
        check(got, w, f"seed={seed} {filt} tr={tr}")
        if eligible:
            old = C._gpu_evaluate(filt, lg2, rows_t(tr), qgpu, "cpu")
            assert np.array_equal(old.numpy(), np.broadcast_to(w, old.shape)), "legacy"
    # second evaluation through the warmed cache: same answer
    plan2 = evaluate(filt, t, qv, combo=True, static_cache=cache)
    got = C._plan_mask_device(plan2, rows_t(slice(0, n)), slice(0, n), leaf_gpu, qgpu, "cpu")
    check(got, want, f"seed={seed} cached")
    for c, m in snap.items():
        assert np.array_equal(cache[c], m), "cached mask mutated"
    plan.release_device()
    assert not plan._dev


@pytest.mark.parametrize("seed", range(30))
def test_shared_cache_threads(seed):
    rng = random.Random(9100 + seed)
    n = rng.randint(50, 400)
    t = table(rng, n)
    qv = qvals(rng, rng.randint(2, 10))
    filts = list(dict.fromkeys(rand_filter(rng, 0.3) for _ in range(12)))
    want = [dense_ref(f, t, qv) for f in filts]
    for rep in range(3):
        cache = {}
        barrier = threading.Barrier(len(filts))

        def go(f):
            barrier.wait()
            return evaluate(f, t, qv, combo=True, static_cache=cache)
        with ThreadPoolExecutor(len(filts)) as pool:
            plans = list(pool.map(go, filts))
        for f, p, w in zip(filts, plans, want):
            if not isinstance(p, F.FilterPlan):
                assert np.array_equal(p.unpack(), w)
                continue
            la, qg = leaf_state(f, t, qv)
            lg = {c: torch.from_numpy(a) for c, a in la.items()}
            got = C._plan_mask_device(p, rows_t(slice(0, n)), slice(0, n), lg, qg, "cpu")
            check(got, w, f"seed={seed} rep={rep} {f}")


def test_static_text_tokenized_once_per_file_across_filters(monkeypatch):
    """Structured-only filters sharing a static `match_text`: with the per-file
    cache, only the first filter tokenizes; with a full cache none does."""
    rng = random.Random(5)
    t = table(rng, 80)
    qv = qvals(rng, 5)
    static = FilterCondition(field="text", match_text="alpha")
    filts = [Filter(must=[static, FilterCondition(field="cat", match_from_query="cat_q")]),
             Filter(must=[static], should=[FilterCondition(field="num", match_from_query="num_q")]),
             Filter(must=[static], must_not=[FilterCondition(field="dcat", match_from_query="cat_q")])]
    want = [dense_ref(f, t, qv) for f in filts]     # before counting
    calls = []
    real = F._text_prep
    monkeypatch.setattr(F, "_text_prep", lambda *a, **k: calls.append(1) or real(*a, **k))
    cache = {}
    for f, w in zip(filts, want):
        plan = evaluate(f, t, qv, combo=True, static_cache=cache)
        assert isinstance(plan, F.FilterPlan) and plan.text is None
        la, qg = leaf_state(f, t, qv)
        got = C._plan_mask_device(plan, rows_t(slice(0, 80)), slice(0, 80),
                                  {c: torch.from_numpy(a) for c, a in la.items()}, qg, "cpu")
        check(got, w, str(f))
    assert len(calls) == 1, calls


def test_lazy_tokenization_memoized_within_a_filter(monkeypatch):
    """Two static `match_text` conditions that both miss the cache: the
    filter's text fields are tokenized once, not once per condition."""
    rng = random.Random(8)
    t = table(rng, 60)
    qv = qvals(rng, 4)
    f = Filter(must=[FilterCondition(field="text", match_text="alpha"),
                     FilterCondition(field="text", match_text="beta gamma"),
                     FilterCondition(field="cat", match_from_query="cat_q")])
    want = dense_ref(f, t, qv)
    calls = []
    real = F._text_prep
    monkeypatch.setattr(F, "_text_prep", lambda *a, **k: calls.append(1) or real(*a, **k))
    plan = evaluate(f, t, qv, combo=True, static_cache={})
    assert len(calls) == 1, calls
    la, qg = leaf_state(f, t, qv)
    got = C._plan_mask_device(plan, rows_t(slice(0, 60)), slice(0, 60),
                              {c: torch.from_numpy(a) for c, a in la.items()}, qg, "cpu")
    check(got, want, "two static text conditions")


def test_static_cache_keys_by_condition_not_field():
    """Two filters with different static conditions on the SAME field share
    one cache: each gets its own condition's rows."""
    rng = random.Random(9)
    t = table(rng, 90)
    qv = qvals(rng, 5)
    f1 = Filter(must=[FilterCondition(field="cat", match="a"),
                      FilterCondition(field="num", match_from_query="num_q")])
    f2 = Filter(must=[FilterCondition(field="cat", match="b"),
                      FilterCondition(field="num", match_from_query="num_q")])
    cache = {}
    for f in (f1, f2, f1):
        plan = evaluate(f, t, qv, combo=True, static_cache=cache)
        la, qg = leaf_state(f, t, qv)
        got = C._plan_mask_device(plan, rows_t(slice(0, 90)), slice(0, 90),
                                  {c: torch.from_numpy(a) for c, a in la.items()}, qg, "cpu")
        check(got, dense_ref(f, t, qv), str(f))


def test_mixed_plan_rejects_query_columns_of_different_lengths():
    rng = random.Random(10)
    t = table(rng, 20)
    qv = qvals(rng, 4)
    qv["cat_q"] = qv["cat_q"][:3]                    # one column shorter
    f = Filter(must=[_text_cond(rng), FilterCondition(field="cat", match_from_query="cat_q")])
    with pytest.raises(ValueError, match="differing lengths"):
        evaluate(f, t, qv, combo=True)
