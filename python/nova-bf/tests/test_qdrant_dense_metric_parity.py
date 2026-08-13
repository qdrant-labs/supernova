"""Live-Qdrant parity for the SHARED-GRAM dense scoring path.

`DenseBatchSlice` derives `dot`/`cosine`/`euclidean` from one raw Gram whenever
2+ distinct metrics score the same batch. These tests confirm that sharing the
product does not change which points a search returns, by checking every metric
against Qdrant's own exact search — the system whose recall this ground truth
exists to measure.

Every test runs the metrics BOTH ways:
  - one config carrying all three searches   -> shared Gram (the new path)
  - one config per search                    -> unshared `_scores` (the old path)
and requires both to agree with Qdrant, and with each other.

Qdrant score conventions, probed rather than assumed (see `_qdrant_topk`):
  dot        raw dot product, descending
  cosine     cosine similarity (Qdrant normalizes at insert), descending
  euclidean  plain L2 DISTANCE, ascending, computed directly (a self-hit comes
             back as exactly 0.0) — negated here into nova-bf's
             larger-is-nearer convention

Skipped automatically unless `qdrant-client` is importable AND a Qdrant server
is reachable (env `QDRANT_URL`, default http://localhost:6333).

Run it explicitly:
    QDRANT_URL=http://localhost:6333 \
      uv run --with qdrant-client --directory python/nova-bf --extra dev \
      pytest -q tests/test_qdrant_dense_metric_parity.py
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
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    Filter,
    FilterCondition,
    OutputConfig,
    ParamsConfig,
    QueriesConfig,
    SearchSpec,
)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DIM = 32
M = 600          # corpus points, across 3 files
N_Q = 12         # queries
K = 50           # top-K (<< M, so this is a real ranking, not set equality)
SELF_HIT_QUERIES = 4      # queries 0..3 are exact copies of corpus rows
SELF_HIT_ROWS = (0, 137, 401, 599)
LANGS = ("eng", "fra", "deu")

METRIC_DISTANCE = {
    "dot": models.Distance.DOT,
    "cosine": models.Distance.COSINE,
    "euclidean": models.Distance.EUCLID,
}

# Per-metric absolute tolerance. dot/cosine are O(1)-scale sums, so float32
# resolution dominates. euclidean is looser BY CONSTRUCTION: both nova paths
# (derived-from-Gram, and `torch.cdist` above its own mm threshold) expand to
# `‖q‖² + ‖c‖² − 2qc`, which resolves a near-zero distance to ~sqrt(eps)·‖q‖
# — about 2e-3 at ‖q‖≈sqrt(DIM). Qdrant computes the distance directly and is
# exact there, so the gap is real, pre-existing, and bounded.
TOL = {"dot": 2e-4, "cosine": 2e-5, "euclidean": 2e-2}


@pytest.fixture(scope="module")
def client():
    try:
        c = QdrantClient(url=QDRANT_URL, timeout=60)
        c.get_collections()
    except Exception as e:  # pragma: no cover - environment gate
        pytest.skip(f"no reachable Qdrant at {QDRANT_URL}: {e}")
    return c


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """Dense corpus over 3 parquet files + queries, some of which are EXACT
    copies of corpus rows (the worst case for the euclidean expansion)."""
    rng = np.random.default_rng(20260810)
    tmp = tmp_path_factory.mktemp("qdense")
    C = rng.standard_normal((M, DIM)).astype(np.float32)
    langs = [LANGS[i % len(LANGS)] for i in range(M)]

    Q = rng.standard_normal((N_Q, DIM)).astype(np.float32)
    for qi, row in enumerate(SELF_HIT_ROWS[:SELF_HIT_QUERIES]):
        Q[qi] = C[row]

    cdir = tmp / "corpus"
    cdir.mkdir()
    bounds = [0, 211, 437, M]        # uneven files; none divides the batch size
    for fi in range(3):
        lo, hi = bounds[fi], bounds[fi + 1]
        pq.write_table(
            pa.table({
                "dense_embedding": pa.array(C[lo:hi].tolist(), type=pa.list_(pa.float32())),
                "id": pa.array([str(i) for i in range(lo, hi)]),
                "language": pa.array(langs[lo:hi]),
            }),
            str(cdir / f"f{fi}.parquet"),
        )
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array(Q.tolist(), type=pa.list_(pa.float32())),
            "qid": pa.array([str(i) for i in range(N_Q)]),
        }),
        str(tmp / "queries.parquet"),
    )
    return {"tmp": tmp, "cdir": str(cdir), "qpath": str(tmp / "queries.parquet"),
            "C": C, "Q": Q, "langs": langs}


@pytest.fixture(scope="module")
def collections(client, data):
    """One collection per distance, upserted once and shared by every test."""
    names = {}
    for metric, dist in METRIC_DISTANCE.items():
        name = f"nova_bf_dense_{metric}_{uuid.uuid4().hex[:8]}"
        client.create_collection(
            name, vectors_config=models.VectorParams(size=DIM, distance=dist)
        )
        client.upsert(name, points=[
            models.PointStruct(id=i, vector=data["C"][i].tolist(),
                               payload={"language": data["langs"][i]})
            for i in range(M)
        ], wait=True)
        names[metric] = name
    yield names
    for name in names.values():
        client.delete_collection(name)


def _qdrant_topk(client, collections, metric, data, k=K, language=None):
    """{query_index: {point_id: score}} in nova-bf's larger-is-nearer
    convention (euclidean distances negated)."""
    qfilter = None
    if language is not None:
        qfilter = models.Filter(must=[models.FieldCondition(
            key="language", match=models.MatchValue(value=language))])
    out = {}
    for qi in range(N_Q):
        pts = client.query_points(
            collections[metric], query=data["Q"][qi].tolist(), limit=k,
            query_filter=qfilter,
            search_params=models.SearchParams(exact=True), with_payload=False,
        ).points
        sign = -1.0 if metric == "euclidean" else 1.0
        out[qi] = {int(p.id): sign * float(p.score) for p in pts}
    return out


def _nova(data, specs, tag, batch_size=64):
    """Run one config and return {spec_name: {query_index: {point_id: score}}}."""
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=data["cdir"], id_column="id"),
        queries=QueriesConfig(path=data["qpath"], id_column="qid"),
        output=OutputConfig(path=str(data["tmp"] / f"out_{tag}")),
        params=ParamsConfig(io_workers=2, dense_batch_size=batch_size),
        searches=specs,
    )
    paths = run_compute(cfg)
    result = {}
    for name, path in paths.items():
        t = pq.read_table(path).to_pydict()
        result[name] = {
            int(q): {int(i): float(s) for i, s in zip(hi, hs)}
            for q, hi, hs in zip(t["query_id"], t["hit_ids"], t["hit_scores"])
        }
    return result


def _spec(metric, k=K, language=None):
    f = None
    if language is not None:
        f = Filter(must=[FilterCondition(field="language", match=language)])
    return SearchSpec(name=metric, vector_type="dense", metric=metric, k=k, filter=f)


def _assert_parity(nova, qd, metric, label):
    """Position-agnostic comparison with a boundary-aware tolerance: an id only
    one engine returned is accepted only when its score sits within tolerance
    of the OTHER engine's K-th score (a genuine near-tie straddling the top-K
    boundary), never when it is a real ranking disagreement."""
    tol = TOL[metric]
    for qi in range(N_Q):
        n, q = nova[qi], qd[qi]
        assert n, f"{label} {metric} q{qi}: nova returned nothing"
        assert q, f"{label} {metric} q{qi}: qdrant returned nothing"
        n_floor, q_floor = min(n.values()), min(q.values())
        for i in set(n) & set(q):
            assert abs(n[i] - q[i]) <= tol * (1 + abs(q[i])), (
                f"{label} {metric} q{qi} id={i}: nova={n[i]!r} qdrant={q[i]!r}")
        for i in set(n) - set(q):
            assert abs(n[i] - q_floor) <= tol * (1 + abs(q_floor)), (
                f"{label} {metric} q{qi} nova-only id={i} score={n[i]!r} not at "
                f"qdrant top-K boundary {q_floor!r}")
        for i in set(q) - set(n):
            assert abs(q[i] - n_floor) <= tol * (1 + abs(n_floor)), (
                f"{label} {metric} q{qi} qdrant-only id={i} score={q[i]!r} not at "
                f"nova top-K boundary {n_floor!r}")


# ------------------------------------------------------------------ shared
@pytest.fixture(scope="module")
def shared_run(data):
    """All three metrics in ONE config -> the shared-Gram path is live."""
    return _nova(data, [_spec(m) for m in METRIC_DISTANCE], "shared_all3")


@pytest.fixture(scope="module")
def solo_runs(data):
    """Each metric alone -> the unshared `_scores` path, as a control."""
    return {m: _nova(data, [_spec(m)], f"solo_{m}")[m] for m in METRIC_DISTANCE}


@pytest.mark.parametrize("metric", list(METRIC_DISTANCE))
def test_shared_gram_matches_qdrant(client, collections, data, shared_run, metric):
    """The change under test: with all three metrics sharing one Gram, every
    one of them must still agree with Qdrant's exact search."""
    _assert_parity(shared_run[metric], _qdrant_topk(client, collections, metric, data),
                   metric, "shared")


@pytest.mark.parametrize("metric", list(METRIC_DISTANCE))
def test_unshared_solo_matches_qdrant(client, collections, data, solo_runs, metric):
    """Control: the pre-existing unshared path agrees with Qdrant too, so a
    shared-path failure above could not be blamed on the harness."""
    _assert_parity(solo_runs[metric], _qdrant_topk(client, collections, metric, data),
                   metric, "solo")


@pytest.mark.parametrize("metric", list(METRIC_DISTANCE))
def test_shared_and_solo_return_the_same_ranking(shared_run, solo_runs, metric):
    """Sharing the product must not move any point in or out of the top-K, and
    must not reorder it — the ids are compared as ordered sequences."""
    for qi in range(N_Q):
        shared_ids = list(shared_run[metric][qi])
        solo_ids = list(solo_runs[metric][qi])
        assert shared_ids == solo_ids, f"{metric} q{qi} ranking moved under sharing"
        for i in shared_ids:
            a, b = shared_run[metric][qi][i], solo_runs[metric][qi][i]
            assert abs(a - b) <= TOL[metric] * (1 + abs(b)), (
                f"{metric} q{qi} id={i}: shared={a!r} solo={b!r}")


# ----------------------------------------------------------------- filtered
@pytest.mark.parametrize("metric", list(METRIC_DISTANCE))
def test_filtered_shared_gram_matches_qdrant(client, collections, data, metric):
    """Sharing composed with filtering: three filtered searches of the same
    vector type, none unfiltered — so the batch is the compacted row union AND
    the Gram is shared across metrics on that compacted grid. Each must match
    Qdrant's exact search under the equivalent payload filter."""
    specs = [_spec(m, language="eng") for m in METRIC_DISTANCE]
    nova = _nova(data, specs, "shared_filtered")
    _assert_parity(nova[metric],
                   _qdrant_topk(client, collections, metric, data, language="eng"),
                   metric, "shared+filtered")


def test_filtered_results_only_contain_matching_rows(data):
    """Cheap independent guard on the filter itself, so a filter that silently
    matched everything could not make the parity test above pass."""
    specs = [_spec(m, language="fra") for m in METRIC_DISTANCE]
    nova = _nova(data, specs, "shared_filtered_fra")
    for metric in METRIC_DISTANCE:
        for qi in range(N_Q):
            for pid in nova[metric][qi]:
                assert data["langs"][pid] == "fra", (
                    f"{metric} q{qi} returned id={pid} with language="
                    f"{data['langs'][pid]}")


# ------------------------------------------------------------- self-hit / K
def test_euclidean_self_hit_is_rank_one_under_sharing(shared_run, data):
    """A query that IS a corpus row must come back as its own nearest
    neighbour under the derived euclidean, despite the expansion resolving that
    true-zero distance to ~sqrt(eps)·‖q‖ rather than exactly 0. This is the
    ordering consequence of the cancellation, and the property that actually
    matters for ground truth."""
    for qi, row in enumerate(SELF_HIT_ROWS[:SELF_HIT_QUERIES]):
        ranked = list(shared_run["euclidean"][qi])
        assert ranked[0] == row, (
            f"q{qi} should be its own nearest neighbour (row {row}), got {ranked[:3]}")
        d = -shared_run["euclidean"][qi][row]
        assert d < TOL["euclidean"], f"q{qi} self-distance {d} exceeds tolerance"


def test_all_metrics_agree_on_self_hit_rank_one(shared_run):
    """dot is not a metric where a self-hit must win (a longer vector can beat
    it), so only cosine and euclidean are asserted here — cosine's self-hit is
    exactly 1.0, its maximum."""
    for qi, row in enumerate(SELF_HIT_ROWS[:SELF_HIT_QUERIES]):
        assert list(shared_run["cosine"][qi])[0] == row, f"cosine q{qi}"
        assert abs(shared_run["cosine"][qi][row] - 1.0) <= 1e-4, (
            f"cosine self-similarity should be 1.0, got "
            f"{shared_run['cosine'][qi][row]}")
        assert list(shared_run["euclidean"][qi])[0] == row, f"euclidean q{qi}"


@pytest.mark.parametrize("metric", list(METRIC_DISTANCE))
@pytest.mark.parametrize("k", [1, 5, 200])
def test_parity_holds_across_k(client, collections, data, metric, k):
    """K interacts with the amortized top-K merge (`_merge_topk` flushes once
    ~k columns accumulate) and with the pre-topk of wide slices, so parity is
    re-checked at k below, near and above `dense_batch_size`."""
    nova = _nova(data, [_spec(m, k=k) for m in METRIC_DISTANCE], f"shared_k{k}")
    _assert_parity(nova[metric],
                   _qdrant_topk(client, collections, metric, data, k=k),
                   metric, f"shared k={k}")


@pytest.mark.parametrize("batch_size", [7, 64, 1024])
def test_parity_holds_across_batch_size(client, collections, data, batch_size):
    """`dense_batch_size` decides how many slices a file becomes, and the Gram
    is per SLICE — so a tiny batch exercises many shared Grams per file, and a
    large one a single Gram spanning whole files."""
    nova = _nova(data, [_spec(m) for m in METRIC_DISTANCE],
                 f"shared_bs{batch_size}", batch_size=batch_size)
    for metric in METRIC_DISTANCE:
        _assert_parity(nova[metric], _qdrant_topk(client, collections, metric, data),
                       metric, f"shared batch_size={batch_size}")
