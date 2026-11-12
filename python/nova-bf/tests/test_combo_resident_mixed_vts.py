"""Dense + sparse + multivector searches sharing lazy text filters, with the
`parts` held on the device whole or in column windows: outputs identical to NOVA_BF_NO_GPU_COMBO=1,
one upload per (file, filter) shared across vector types, and every
lazy mask released."""
import json
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nova_bf import compute as C, filters as F
from nova_bf.config import (BruteForceConfig, CorpusConfig, Filter, FilterCondition,
                            OutputConfig, ParamsConfig, QueriesConfig, SearchSpec)
from nova_bf.filters import TextPart
import test_combo_row_mask as T  # sibling test module (pytest puts tests/ on sys.path)

DIM, VOCAB = 5, 40


def _sp(rng):
    idx = rng.choice(VOCAB, size=6, replace=False)
    return {"indices": idx.astype(np.uint32).tolist(),
            "values": rng.standard_normal(6).astype(np.float32).tolist()}


def _mv(rng):
    return rng.standard_normal((int(rng.integers(1, 4)), DIM)).astype(np.float32).tolist()


SPT = pa.struct([pa.field("indices", pa.list_(pa.uint32())), pa.field("values", pa.list_(pa.float32()))])


def _write(path, n, rng, prng, **cols):
    d = {"dense_embedding": pa.array(rng.standard_normal((n, DIM)).astype(np.float32).tolist(),
                                     type=pa.list_(pa.float32())),
         "sparse_embedding": pa.array([_sp(rng) for _ in range(n)], type=SPT),
         "multivector_embedding": pa.array([_mv(rng) for _ in range(n)],
                                           type=pa.list_(pa.list_(pa.float32())))}
    d.update({k: pa.array(v) for k, v in cols.items()})
    pq.write_table(pa.table(d), str(path))


@pytest.fixture
def ds(tmp_path):
    prng, rng = random.Random(5), np.random.default_rng(5)
    cdir = tmp_path / "c"; cdir.mkdir()
    base = 0
    for i, n in enumerate([97, 150, 61]):
        _write(cdir / f"f{i}.parquet", n, rng, prng, id=[f"c{base + r}" for r in range(n)],
               text=[T._rand_text(prng, T.WORDS) for _ in range(n)],
               url=[T._rand_text(prng, T.DOMS, 1, 3) for _ in range(n)],
               cat=[prng.choice(["a", "b", None]) for _ in range(n)])
        base += n
    nq = 24
    cols = {"kw": [T._rand_phrase(prng, T.WORDS) for _ in range(nq)],
            "neg": [T._rand_phrase(prng, T.WORDS) for _ in range(nq)]}
    for i in range(3):
        cols[f"d{i}"] = [T._rand_phrase(prng, T.DOMS) for _ in range(nq)]
    _write(tmp_path / "q.parquet", nq, rng, prng, qid=[f"q{i}" for i in range(nq)], **cols)
    return tmp_path


def _run(tmp, name, batch, with_mv):
    out = tmp / name; out.mkdir()
    s = []
    for vt in (("dense", "sparse", "multivector") if with_mv else ("dense", "sparse")):
        for fn, f, k in (("fw", T.FINEWEB_LIKE, 20), ("not", T.WITH_NOT, 30)):
            s.append(SearchSpec(name=f"{vt}_{fn}", vector_type=vt, metric="dot", k=k, filter=f))
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(tmp / "c"), id_column="id"),
        queries=QueriesConfig(path=str(tmp / "q.parquet"), id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(dense_batch_size=batch), searches=s)
    res = {n: pq.read_table(p).to_pydict() for n, p in C.run_compute(cfg).items()}
    man = json.loads(next(out.rglob("*manifest*.json")).read_text())
    return res, man["params"]["text_filter_combine"]


@pytest.mark.parametrize("window", [None, 1])
@pytest.mark.parametrize("with_mv", [False, True])
@pytest.mark.parametrize("batch", [29, None])
def test_mixed_vts_resident(ds, monkeypatch, batch, with_mv, window):
    monkeypatch.setattr(C, "_RESIDENT_WINDOW_BYTES", window)
    monkeypatch.setattr(F, "_COMBO_PARTS_GUARD", False)
    made = []
    real = C.evaluate
    def spy(*a, **kw):
        m = real(*a, **kw)
        if isinstance(m, F.FilterPlan):
            made.extend(m.masks())
        return m
    monkeypatch.setattr(C, "evaluate", spy)
    monkeypatch.setenv("NOVA_BF_NO_GPU_COMBO", "1")
    off, _ = _run(ds, f"off{batch}{with_mv}{window}", batch, with_mv)
    monkeypatch.delenv("NOVA_BF_NO_GPU_COMBO")
    on, u = _run(ds, f"on{batch}{with_mv}{window}", batch, with_mv)
    assert u["combo_masks"] > 0 and u["device_builds"] > 0
    # one buffer per (file, filter) at most, even with 2-3 vts sharing it; the
    # whole table is then uploaded once (windows reload per vector-type pass)
    taken = u["resident_allocs"] + u["resident_reuses"]
    assert 0 < taken <= u["combo_masks"]
    if window is None:
        assert u["resident_uploads"] == taken
    assert all(not m._parts_dev and not m._dev for m in made)
    assert set(on) == set(off)
    for n in on:
        if n.startswith("multivector_"):
            # multivector CUDA scoring is not bit-reproducible run to run (~1 ULP,
            # same ids and order; see tests/parity/test_parity_text_combo.py), so
            # ids/order exact, scores to 1e-6 relative
            assert on[n]["hit_ids"] == off[n]["hit_ids"], n
            for a, b in zip(on[n]["hit_scores"], off[n]["hit_scores"]):
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7, err_msg=n)
            rest = [k for k in on[n] if k not in ("hit_ids", "hit_scores")]
            assert all(on[n][k] == off[n][k] for k in rest), n
        else:
            assert on[n] == off[n], n
