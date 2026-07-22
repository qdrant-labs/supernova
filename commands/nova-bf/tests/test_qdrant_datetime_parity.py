"""Live-Qdrant parity for datetime `range` / `range_from_query` filters.

Confirms nova-bf's epoch-µs date comparison selects EXACTLY the same corpus
points as Qdrant's native `DatetimeRange` on the same data and the same bounds —
including boundary inclusivity (gt vs gte, lt vs lte). Uses k >= corpus size so
both engines return every point passing the filter, making the comparison a
deterministic SET equality that can't be perturbed by score ties.

Skipped automatically unless `qdrant-client` is importable AND a Qdrant server
is reachable (env `QDRANT_URL`, default http://localhost:6333). So it's inert in
the default `pytest` run / CI, and active wherever a live Qdrant exists.

Run it explicitly:
    uv run --with qdrant-client --directory python/nova-bf --extra dev \
        pytest -q tests/test_qdrant_datetime_parity.py
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("qdrant_client")
import pyarrow as pa
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

from nova_bf.compute import run_compute
from nova_bf.config import load_config
from nova_bf.dates import parse_scalar_epoch_us

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DIM = 32
M = 2000       # corpus points
N_Q = 12       # queries
EPOCH0 = datetime(2010, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def client():
    try:
        c = QdrantClient(url=QDRANT_URL, timeout=10)
        c.get_collections()
    except Exception as e:  # pragma: no cover - environment gate
        pytest.skip(f"no reachable Qdrant at {QDRANT_URL}: {e}")
    return c


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    rng = np.random.default_rng(20260717)
    tmp = tmp_path_factory.mktemp("qparity")
    vecs = rng.standard_normal((M, DIM)).astype(np.float32)
    # dates spread over ~10 years at hour granularity; a few nulls
    hours = rng.integers(0, 24 * 365 * 10, size=M)
    dts = [EPOCH0 + timedelta(hours=int(h)) for h in hours]
    date_strs = [d.strftime("%Y-%m-%dT%H:%M:%SZ") for d in dts]
    for i in rng.choice(M, size=30, replace=False):
        date_strs[i] = None

    cdir = tmp / "corpus"
    cdir.mkdir()
    pq.write_table(pa.table({
        "dense_embedding": pa.array(vecs.tolist(), type=pa.list_(pa.float32())),
        "id": pa.array([str(i) for i in range(M)]),
        "date": pa.array(date_strs, type=pa.string()),
    }), str(cdir / "c0.parquet"))

    qidx = rng.choice(M, size=N_Q, replace=False)
    qvecs = vecs[qidx]
    # per-query cutoff dates for the range_from_query test
    afters = [(EPOCH0 + timedelta(hours=int(h))).strftime("%Y-%m-%dT%H:%M:%SZ")
              for h in rng.integers(0, 24 * 365 * 10, size=N_Q)]
    pq.write_table(pa.table({
        "dense_embedding": pa.array(qvecs.tolist(), type=pa.list_(pa.float32())),
        "qid": pa.array([str(i) for i in range(N_Q)]),
        "after": pa.array(afters, type=pa.string()),
    }), str(tmp / "queries.parquet"))

    return {
        "tmp": tmp, "cdir": str(cdir), "qpath": str(tmp / "queries.parquet"),
        "vecs": vecs, "qvecs": qvecs, "date_strs": date_strs, "afters": afters,
    }


@pytest.fixture(scope="module")
def collection(client, data):
    name = f"nova_bf_date_parity_{uuid.uuid4().hex[:8]}"
    client.create_collection(
        name, vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
    )
    client.create_payload_index(name, "date", models.PayloadSchemaType.DATETIME)
    points = [
        models.PointStruct(id=i, vector=data["vecs"][i].tolist(),
                           payload=({"date": d} if d is not None else {}))
        for i, d in enumerate(data["date_strs"])
    ]
    client.upsert(name, points=points, wait=True)
    yield name
    client.delete_collection(name)


def _nova(data, filter_yaml, query_dates=""):
    out = data["tmp"] / "out"
    out.mkdir(exist_ok=True)
    cfg_text = f"""
corpus:
  path: {data["cdir"]}
  dense_column: dense_embedding
  id_column: id
  date_fields: [date]
queries:
  path: {data["qpath"]}
  dense_column: dense_embedding
  id_column: qid
  {("date_fields: " + query_dates) if query_dates else ""}
output:
  path: {out}
params:
  io_workers: 2
searches:
  - name: p
    k: {M}
    metric: cosine
    filter:
{filter_yaml}
"""
    p = data["tmp"] / "cfg.yaml"
    p.write_text(cfg_text)
    t = pq.read_table(run_compute(load_config(str(p)))["p"]).to_pydict()
    return {q: {int(i) for i in hi} for q, hi in zip(t["query_id"], t["hit_ids"])}


def _qdrant(client, collection, data, qflt):
    res = {}
    for qi in range(N_Q):
        pts = client.query_points(
            collection, query=data["qvecs"][qi].tolist(),
            query_filter=qflt(qi), limit=M,
            search_params=models.SearchParams(exact=True),  # exact, not HNSW
            with_payload=False,
        ).points
        res[str(qi)] = {int(p.id) for p in pts}
    return res


@pytest.mark.parametrize("lo,hi,gte,lt", [
    ("2013-01-01T00:00:00Z", "2016-01-01T00:00:00Z", True, True),
    ("2015-06-01T00:00:00Z", None, True, None),
    (None, "2012-01-01T00:00:00Z", None, True),
    ("2014-01-01T00:00:00Z", "2018-01-01T00:00:00Z", False, False),  # gt / lte boundary
])
def test_static_range_parity(client, collection, data, lo, hi, gte, lt):
    conds, qbounds = [], {}
    if lo is not None:
        key = "gte" if gte else "gt"
        conds.append(f'{key}: "{lo}"')
        qbounds[key if gte else "gt"] = lo
    if hi is not None:
        key = "lt" if lt else "lte"
        conds.append(f'{key}: "{hi}"')
        qbounds[key if lt else "lte"] = hi
    nova = _nova(data, "      must:\n        - field: date\n          range: {%s}\n" % ", ".join(conds))

    dr = models.DatetimeRange(**{k: v for k, v in qbounds.items()})
    qflt = lambda qi: models.Filter(must=[models.FieldCondition(key="date", range=dr)])
    qd = _qdrant(client, collection, data, qflt)

    for qi in range(N_Q):
        assert nova[str(qi)] == qd[str(qi)], (
            f"q{qi} lo={lo} hi={hi} gte={gte} lt={lt}: "
            f"nova-only={sorted(nova[str(qi)] - qd[str(qi)])[:5]} "
            f"qdrant-only={sorted(qd[str(qi)] - nova[str(qi)])[:5]}"
        )


def test_range_from_query_parity(client, collection, data):
    nova = _nova(
        data,
        "      must:\n        - field: date\n          range_from_query: {gte: after}\n",
        query_dates="[after]",
    )
    qflt = lambda qi: models.Filter(must=[models.FieldCondition(
        key="date", range=models.DatetimeRange(gte=data["afters"][qi]))])
    qd = _qdrant(client, collection, data, qflt)
    for qi in range(N_Q):
        assert nova[str(qi)] == qd[str(qi)], (
            f"q{qi} after={data['afters'][qi]}: "
            f"nova-only={sorted(nova[str(qi)] - qd[str(qi)])[:5]} "
            f"qdrant-only={sorted(qd[str(qi)] - nova[str(qi)])[:5]}"
        )
