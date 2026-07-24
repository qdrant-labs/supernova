"""Live-Qdrant parity for multivector (ColBERT / late-interaction MaxSim) GT.

Confirms nova-bf's MaxSim ranking selects EXACTLY the same top-K points, with
matching scores, as Qdrant's native multivector MaxSim comparator on the same
data — for both `dot` (Qdrant's default) and `cosine` (per-token L2
normalization). Qdrant runs with `exact=True` so it computes the true MaxSim
ranking, not an HNSW approximation.

Scores are compared position-agnostically with a boundary-aware tolerance: an
id present in one engine's top-K but not the other's is only accepted when its
score sits within tolerance of the K-th score (a genuine boundary near-tie
between two engines' independent float32 reductions), never when it's a real
disagreement inside the ranking.

Skipped automatically unless `qdrant-client` is importable AND a Qdrant server
is reachable (env `QDRANT_URL`, default http://localhost:6333). Inert in the
default `pytest` run / CI; active wherever a live Qdrant exists.

Run it explicitly:
    uv run --with qdrant-client --directory python/nova-bf --extra dev \
        pytest -q tests/test_qdrant_multivector_parity.py
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("qdrant_client")
import pyarrow as pa
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

from nova_bf.compute import run_compute
from nova_bf.config import load_config

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DIM = 16
M = 400        # corpus points
N_Q = 10       # queries
K = 50         # top-K (< M, so this is a real ranking, not set equality)
TOK_LO, TOK_HI = 2, 9   # tokens per doc / query (Qdrant rejects zero-token multivectors)


@pytest.fixture(scope="module")
def client():
    try:
        c = QdrantClient(url=QDRANT_URL, timeout=30)
        c.get_collections()
    except Exception as e:  # pragma: no cover - environment gate
        pytest.skip(f"no reachable Qdrant at {QDRANT_URL}: {e}")
    return c


def _mv_array(docs: list[np.ndarray]) -> pa.Array:
    """list of (n_tokens, DIM) arrays -> list<list<float32>> Arrow column."""
    tok_counts = [len(d) for d in docs]
    total = sum(tok_counts)
    flat = np.concatenate([d.reshape(-1) for d in docs]).astype(np.float32)
    inner = pa.ListArray.from_arrays(
        pa.array(np.arange(0, total * DIM + 1, DIM, dtype=np.int32)),
        pa.array(flat, type=pa.float32()),
    )
    outer_off = np.concatenate([[0], np.cumsum(tok_counts)]).astype(np.int32)
    return pa.ListArray.from_arrays(pa.array(outer_off), inner)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    rng = np.random.default_rng(20260724)
    tmp = tmp_path_factory.mktemp("qmvparity")
    cdocs = [rng.standard_normal((int(rng.integers(TOK_LO, TOK_HI)), DIM)).astype(np.float32)
             for _ in range(M)]
    qdocs = [rng.standard_normal((int(rng.integers(TOK_LO, TOK_HI)), DIM)).astype(np.float32)
             for _ in range(N_Q)]

    cdir = tmp / "corpus"
    cdir.mkdir()
    pq.write_table(pa.table({
        "multivector_embedding": _mv_array(cdocs),
        "id": pa.array([str(i) for i in range(M)]),
    }), str(cdir / "c0.parquet"))
    pq.write_table(pa.table({
        "multivector_embedding": _mv_array(qdocs),
        "qid": pa.array([str(i) for i in range(N_Q)]),
    }), str(tmp / "queries.parquet"))

    return {"tmp": tmp, "cdir": str(cdir), "qpath": str(tmp / "queries.parquet"),
            "cdocs": cdocs, "qdocs": qdocs}


def _collection(client, data, distance):
    name = f"nova_bf_mv_parity_{uuid.uuid4().hex[:8]}"
    client.create_collection(
        name,
        vectors_config=models.VectorParams(
            size=DIM, distance=distance,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM),
        ),
    )
    client.upsert(name, points=[
        models.PointStruct(id=i, vector=d.tolist()) for i, d in enumerate(data["cdocs"])
    ], wait=True)
    return name


def _qdrant_topk(client, name, data):
    out = {}
    for qi in range(N_Q):
        pts = client.query_points(
            name, query=data["qdocs"][qi].tolist(), limit=K,
            search_params=models.SearchParams(exact=True), with_payload=False,
        ).points
        out[qi] = {int(p.id): float(p.score) for p in pts}
    return out


def _nova(data, metric):
    out = data["tmp"] / f"out_{metric}"
    out.mkdir(exist_ok=True)
    cfg_text = f"""
corpus:
  path: {data["cdir"]}
  multivector_column: multivector_embedding
  id_column: id
queries:
  path: {data["qpath"]}
  multivector_column: multivector_embedding
  id_column: qid
output:
  path: {out}
params:
  io_workers: 2
  multivector_batch_size: 128
  multivector_query_block: 4
searches:
  - name: mv
    k: {K}
    metric: {metric}
    vector_type: multivector
"""
    p = data["tmp"] / f"cfg_{metric}.yaml"
    p.write_text(cfg_text)
    t = pq.read_table(run_compute(load_config(str(p)))["mv"]).to_pydict()
    return {int(q): {int(i): float(s) for i, s in zip(hi, hs)}
            for q, hi, hs in zip(t["query_id"], t["hit_ids"], t["hit_scores"])}


def _assert_parity(nova, qd, metric):
    for qi in range(N_Q):
        n, q = nova[qi], qd[qi]
        # K-th (worst) score per engine — the boundary a near-tie can straddle.
        n_floor = min(n.values())
        q_floor = min(q.values())
        tol = 1e-3
        # scores must match for every id both engines returned
        for i in set(n) & set(q):
            assert abs(n[i] - q[i]) <= tol * (1 + abs(q[i])), (
                f"{metric} q{qi} id={i}: nova={n[i]} qdrant={q[i]}")
        # an id only one engine returned must be a boundary near-tie, not a
        # real ranking disagreement
        for i in set(n) - set(q):
            assert abs(n[i] - q_floor) <= tol * (1 + abs(q_floor)), (
                f"{metric} q{qi} nova-only id={i} score={n[i]} not at qdrant "
                f"boundary {q_floor}")
        for i in set(q) - set(n):
            assert abs(q[i] - n_floor) <= tol * (1 + abs(n_floor)), (
                f"{metric} q{qi} qdrant-only id={i} score={q[i]} not at nova "
                f"boundary {n_floor}")


def test_multivector_dot_parity(client, data):
    name = _collection(client, data, models.Distance.DOT)
    try:
        _assert_parity(_nova(data, "dot"), _qdrant_topk(client, name, data), "dot")
    finally:
        client.delete_collection(name)


def test_multivector_cosine_parity(client, data):
    name = _collection(client, data, models.Distance.COSINE)
    try:
        _assert_parity(_nova(data, "cosine"), _qdrant_topk(client, name, data), "cosine")
    finally:
        client.delete_collection(name)
