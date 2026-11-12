"""Corpus nulls on the GPU per-query path: a null corpus value never matches
`match_from_query` and never satisfies `range_from_query` (and so is kept
under must_not), for plain, dictionary-encoded and int-with-null columns --
the same answer as the CPU reference `filters.evaluate(combo=False)`."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")

from nova_bf import compute as C
from nova_bf.config import (BruteForceConfig, CorpusConfig, Filter, FilterCondition,
                            OutputConfig, ParamsConfig, QueriesConfig, RangeFromQuery,
                            SearchSpec)

N = 64


def _corpus(kind: str) -> dict:
    rng = np.random.default_rng(3)
    cats = [None if i % 3 == 0 else ["a", "b", "c"][i % 3] for i in range(N)]
    nums = [None if i % 4 == 0 else int(i) for i in range(N)]
    if kind == "dict":
        cat = pa.array(cats, pa.string()).dictionary_encode()

    else:
        cat = pa.array(cats, pa.string())
    return {"dense_embedding": pa.array(rng.standard_normal((N, 4)).astype(np.float32).tolist(),
                                        type=pa.list_(pa.float32())),
            "id": [f"c{i}" for i in range(N)], "cat": cat, "num": pa.array(nums, pa.int64())}


FILTERS = {
    "match": Filter(must=[FilterCondition(field="cat", match_from_query="q_cat")]),
    "match_not": Filter(must_not=[FilterCondition(field="cat", match_from_query="q_cat")]),
    "range": Filter(must=[FilterCondition(field="num", range_from_query=RangeFromQuery(gte="q_lo"))]),
    "range_not": Filter(must_not=[FilterCondition(field="num",
                                                  range_from_query=RangeFromQuery(gte="q_lo"))]),
    # mixed with per-query text: the MixedPlan path uses the same leaves
    "mixed": Filter(must=[FilterCondition(field="cat", match_from_query="q_cat"),
                          FilterCondition(field="txt", match_text_from_query="q_kw")]),
}


@pytest.mark.parametrize("kind", ["plain", "dict"])
@pytest.mark.parametrize("fname", list(FILTERS))
def test_null_corpus_values_match_like_the_cpu_reference(tmp_path, monkeypatch, kind, fname):
    cdir = tmp_path / "c"; cdir.mkdir()
    cols = _corpus(kind)
    cols["txt"] = pa.array(["alpha beta" if i % 2 else "beta" for i in range(N)])
    pq.write_table(pa.table(cols), str(cdir / "f.parquet"))
    nq = 5
    q = {"dense_embedding": pa.array(np.eye(nq, 4, dtype=np.float32).tolist(), pa.list_(pa.float32())),
         "qid": [f"q{i}" for i in range(nq)],
         "q_cat": pa.array(["a", "b", "c", "a", "b"]),
         "q_lo": pa.array([0.0, 10.0, 20.0, 40.0, 63.0]),
         "q_kw": pa.array(["beta", "alpha", "beta", "alpha", "beta"])}
    pq.write_table(pa.table(q), str(tmp_path / "q.parquet"))

    def run(tag):
        out = tmp_path / tag
        cfg = BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), id_column="id"),
            queries=QueriesConfig(path=str(tmp_path / "q.parquet"), id_column="qid"),
            output=OutputConfig(path=str(out)), params=ParamsConfig(dense_batch_size=16),
            searches=[SearchSpec(name="s", vector_type="dense", metric="dot", k=N,
                                 filter=FILTERS[fname])])
        return {n: pq.read_table(p).to_pydict() for n, p in C.run_compute(cfg).items()}

    # reference: the CPU mask path for every filter (GPU leaves off)
    monkeypatch.setenv("NOVA_BF_NO_GPU_COMBO", "1")
    monkeypatch.setattr(C, "_gpu_eligible", lambda f: False)
    want = run("ref")
    monkeypatch.undo()
    got = run("gpu")
    for qi, (a, b) in enumerate(zip(got["s"]["hit_ids"], want["s"]["hit_ids"])):
        assert sorted(a) == sorted(b), f"{kind} {fname} q{qi}"


@pytest.mark.parametrize("matchany", [False, True])
def test_nullable_int64_matches_exactly_above_2_pow_53(matchany):
    """An int64 corpus column with a null must not be widened to float64:
    2**53 and 2**53 + 1 are different values, on the CPU reference and on the
    GPU leaf path alike."""
    import torch
    from nova_bf import filters as F
    big = 2 ** 53
    t = pa.table({"c": pa.array([big, big + 1, None, 7], pa.int64())})
    q = np.empty(2, dtype=object)
    q[:] = [[big + 1], [7]] if matchany else [big + 1, 7]
    qv = {"q": q}
    cond = FilterCondition(field="c", match_from_query="q")
    want = [[0, 1, 0, 0], [0, 0, 0, 1]]
    ref = F.evaluate(Filter(must=[cond]), t, qv, combo=False).unpack().astype(int).tolist()
    assert ref == want
    st, vocab = C._build_gpu_leaf_state(cond, qv, "cpu")
    leaf = torch.from_numpy(C._corpus_leaf_array(cond, t, vocab, None))
    got = C._gpu_cond_mask(cond, {cond: leaf}, torch.arange(4), {cond: st})
    assert got.int().tolist() == want
