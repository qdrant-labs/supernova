"""End-to-end date-filter tests: drive `run_compute` on a synthetic corpus with
an RFC-3339 `date` column and confirm both static `range` and per-query
`range_from_query` over that column return exactly the in-range neighbors —
exercising the corpus-side and query-side epoch conversions in compute.py."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import pyarrow as pa
import pyarrow.parquet as pq

from nova_bf.compute import run_compute
from nova_bf.config import load_config
from nova_bf.dates import parse_scalar_epoch_us

DIM = 8

CORPUS_DATES = [
    "2010-01-01T00:00:00Z",  # c0
    "2011-01-01T00:00:00Z",  # c1
    "2012-06-15T12:00:00Z",  # c2
    "2013-01-01T00:00:00Z",  # c3
    "2014-07-04T00:00:00Z",  # c4
    "2015-12-31T23:59:59Z",  # c5
    "2016-01-01T00:00:00Z",  # c6
    "2017-03-03T03:03:03Z",  # c7
    None,                    # c8  (null date -> never matches)
    "2019-09-09T09:09:09Z",  # c9
]
N = len(CORPUS_DATES)


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    rng = np.random.default_rng(7)
    tmp = tmp_path_factory.mktemp("bfdate")
    cdir = tmp / "corpus"
    cdir.mkdir()
    corpus = rng.standard_normal((N, DIM)).astype(np.float32)
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array(corpus.tolist(), type=pa.list_(pa.float32())),
            "id": pa.array([f"c{i}" for i in range(N)]),
            "date": pa.array(CORPUS_DATES, type=pa.string()),
        }),
        str(cdir / "f0.parquet"),
    )

    n_q = 3
    queries = rng.standard_normal((n_q, DIM)).astype(np.float32)
    afters = ["2015-01-01T00:00:00Z", "2018-01-01T00:00:00Z", "2011-06-01T00:00:00Z"]
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array(queries.tolist(), type=pa.list_(pa.float32())),
            "qid": pa.array([f"q{i}" for i in range(n_q)]),
            "after": pa.array(afters, type=pa.string()),
        }),
        str(tmp / "queries.parquet"),
    )

    corpus_us = [parse_scalar_epoch_us(d) if d else None for d in CORPUS_DATES]
    return {
        "tmp": tmp, "cdir": str(cdir), "qpath": str(tmp / "queries.parquet"),
        "qids": [f"q{i}" for i in range(n_q)], "corpus": corpus, "queries": queries,
        "corpus_us": corpus_us, "afters": [parse_scalar_epoch_us(a) for a in afters],
    }


def _run(ds, filter_yaml, corpus_dates="[date]", query_dates=""):
    out = ds["tmp"] / "out"
    out.mkdir(exist_ok=True)
    cfgtext = f"""
corpus:
  path: {ds["cdir"]}
  dense_column: dense_embedding
  id_column: id
  date_fields: {corpus_dates}
queries:
  path: {ds["qpath"]}
  dense_column: dense_embedding
  id_column: qid
  {("date_fields: " + query_dates) if query_dates else ""}
output:
  path: {out}
params:
  io_workers: 2
searches:
  - name: dated
    k: {N}
    metric: dot
    filter:
{filter_yaml}
"""
    cfgpath = ds["tmp"] / "cfg.yaml"
    cfgpath.write_text(cfgtext)
    cfg = load_config(str(cfgpath))
    t = pq.read_table(run_compute(cfg)["dated"]).to_pydict()
    return {q: list(zip(hi, hs)) for q, hi, hs in
            zip(t["query_id"], t["hit_ids"], t["hit_scores"])}


def _ids(res_for_q):
    return [i for i, _ in res_for_q]


def _scores_descending(res_for_q):
    s = [sc for _, sc in res_for_q]
    return all(a >= b - 1e-4 for a, b in zip(s, s[1:]))


def test_static_range_returns_only_in_range(ds):
    lo, hi = "2013-01-01T00:00:00Z", "2016-01-01T00:00:00Z"
    res = _run(ds, f"""      must:
        - field: date
          range: {{gte: "{lo}", lt: "{hi}"}}
""")
    lo_us, hi_us = parse_scalar_epoch_us(lo), parse_scalar_epoch_us(hi)
    expected = {f"c{g}" for g, us in enumerate(ds["corpus_us"])
                if us is not None and lo_us <= us < hi_us}
    assert expected == {"c3", "c4", "c5"}  # sanity on the fixture
    for q in ds["qids"]:
        assert set(_ids(res[q])) == expected     # exactly the in-range docs
        assert _scores_descending(res[q])         # ranked by dot within the filter
        assert "c8" not in _ids(res[q])           # null date never matches


def test_static_range_single_bound(ds):
    res = _run(ds, """      must:
        - field: date
          range: {gt: "2016-01-01T00:00:00Z"}
""")
    cut = parse_scalar_epoch_us("2016-01-01T00:00:00Z")
    expected = {f"c{g}" for g, us in enumerate(ds["corpus_us"]) if us is not None and us > cut}
    assert expected == {"c7", "c9"}
    for q in ds["qids"]:
        assert set(_ids(res[q])) == expected


def test_range_from_query_per_query_cutoff(ds):
    res = _run(
        ds,
        """      must:
        - field: date
          range_from_query: {gte: after}
""",
        query_dates="[after]",
    )
    for qi, q in enumerate(ds["qids"]):
        after_us = ds["afters"][qi]
        expected = {f"c{g}" for g, us in enumerate(ds["corpus_us"])
                    if us is not None and us >= after_us}
        assert set(_ids(res[q])) == expected, f"{q}: after={after_us}"
        assert _scores_descending(res[q])
    # concrete per-query expectations from the fixture dates
    assert set(_ids(res["q0"])) == {"c5", "c6", "c7", "c9"}   # >= 2015-01-01
    assert set(_ids(res["q1"])) == {"c9"}                      # >= 2018-01-01
    assert set(_ids(res["q2"])) == {"c2", "c3", "c4", "c5", "c6", "c7", "c9"}  # >= 2011-06-01
