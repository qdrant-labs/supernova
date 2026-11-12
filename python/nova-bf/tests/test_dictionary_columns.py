"""Dictionary-encoded corpus columns behave exactly like plain ones in every
filter kind (pyarrow's own `to_numpy` decodes a dictionary null to a real
value and raises on an all-null chunk; the tokenizer walks string offsets),
including all-null and empty columns."""
from __future__ import annotations

import random

import numpy as np
import pyarrow as pa
import pytest

from nova_bf.config import Filter, FilterCondition
from nova_bf.filters import TextPart, evaluate

FILTERS = {
    "match_from_query": (Filter(must=[FilterCondition(field="c", match_from_query="q")]),
                         ["a", "b", "c", None]),
    "match": (Filter(must=[FilterCondition(field="c", match="a")]), ["a", "b", None]),
    "matchany": (Filter(must=[FilterCondition(field="c", match=["a", "c"])]), ["a", "b", "c", None]),
    "match_text": (Filter(must=[FilterCondition(field="c", match_text="alpha")]),
                   ["alpha beta", "beta", None, "Alpha", ""]),
    "match_text_from_query": (Filter(must=[FilterCondition(field="c", match_text_from_query="t")]),
                              ["alpha beta", "beta", None, "Alpha", ""]),
}


def _dense(m, table, qv):
    from nova_bf import filters as F
    if isinstance(m, F.FilterPlan):
        from test_combo_row_mask import _mixed_dense   # CPU formula of a plan
        return _mixed_dense(m, table, qv)
    if isinstance(m, TextPart):
        m = m.to_packed()
    return m.unpack() if hasattr(m, "unpack") else np.asarray(m)


@pytest.mark.parametrize("name", list(FILTERS))
@pytest.mark.parametrize("combo", [False, True])
def test_dictionary_equals_plain(name, combo):
    filt, pool = FILTERS[name]
    for seed in range(300):
        r = random.Random(seed)
        n = r.choice([1, 2, r.randint(1, 12)])
        vals = [r.choice(pool) for _ in range(n)]
        if seed % 10 == 0:
            vals = [None] * n                  # all-null: empty dictionary
        nq = r.randint(1, 3)
        q = np.empty(nq, dtype=object)
        q[:] = [r.choice([None, [], ["a"], ["z"], ["a", None], ["b", "c"], "a", "b"])
                for _ in range(nq)]
        qv = {"q": q, "t": np.array([r.choice(["alpha", "beta", "gamma"]) for _ in range(nq)],
                                    dtype=object)}
        plain = pa.table({"c": pa.array(vals, pa.string())})
        dic = pa.table({"c": pa.array(vals, pa.string()).dictionary_encode()})
        a = _dense(evaluate(filt, plain, qv, combo=combo), plain, qv)
        b = _dense(evaluate(filt, dic, qv, combo=combo), dic, qv)
        assert a.shape == b.shape and np.array_equal(a, b), (name, seed, vals, list(q))


@pytest.mark.parametrize("combo", [False, True])
def test_null_inside_dictionary_values_never_matches(combo):
    """A null can sit in the dictionary VALUES (not only the indices): such a
    row is a null corpus value and must not match a null query value."""
    d = pa.DictionaryArray.from_arrays(pa.array([0, 1, 2, 1]), pa.array(["a", None, "b"]))
    t = pa.table({"c": d})
    filt = Filter(must=[FilterCondition(field="c", match_from_query="q")])
    q = np.empty(3, dtype=object)
    q[:] = [None, "a", ["a", None]]
    got = _dense(evaluate(filt, t, {"q": q}, combo=combo), t, {"q": q})
    assert got.astype(int).tolist() == [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
