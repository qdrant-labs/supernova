"""Live-Qdrant parity for a corpus that is FULL of exact ties.

A tie-break changes which of several equally-scored points nova-bf keeps, so the
thing to verify against a real engine is that it changed only that, and nothing
about the ranking itself. The corpus here is built with deliberate duplicate
vectors — every doc is one of a small set, so each score is shared by dozens of
points — which is precisely where a tie-break could silently diverge.

What parity means here, and what it does not:

  * nova-bf must return the same SCORE SEQUENCE as Qdrant. That is the real
    ranking claim, and it is unaffected by tie-breaking.
  * nova-bf's hits must be a valid selection: every hit strictly above the cut
    score must also be in Qdrant's answer, and every hit AT the cut score must
    be one of the points that legitimately tie there.
  * The two need NOT return the same point ids at the cut. Qdrant's
    `impl Ord for ScoredPoint` compares score only, so it has no tie-break of
    its own — this makes nova's artifact reproducible, not the two engines
    identical. Disagreement at a tie boundary is what nova-storm's tie-tolerant
    recall exists to absorb.

Run with a reachable Qdrant:

    QDRANT_URL=http://localhost:6333 pytest tests/test_qdrant_tiebreak_parity.py
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")
pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient, models

from nova_bf.compute import run_compute
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    OutputConfig,
    ParamsConfig,
    QueriesConfig,
    SearchSpec,
)
from nova_bf.merge import run_merge

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DIM = 16
N_DISTINCT = 12      # only this many distinct vectors...
M = 480              # ...spread over this many points, so ties are everywhere
N_Q = 8
K = 40
TOL = 2e-4


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
    rng = np.random.default_rng(20260828)
    tmp = tmp_path_factory.mktemp("qtie")
    palette = rng.standard_normal((N_DISTINCT, DIM)).astype(np.float32)
    # Point i takes palette vector (i % N_DISTINCT): every score is shared by
    # exactly M/N_DISTINCT = 40 points.
    C = np.stack([palette[i % N_DISTINCT] for i in range(M)])
    Q = rng.standard_normal((N_Q, DIM)).astype(np.float32)

    cdir = tmp / "corpus"
    cdir.mkdir()
    bounds = [0, 173, 344, M]        # uneven files
    for fi in range(3):
        lo, hi = bounds[fi], bounds[fi + 1]
        pq.write_table(
            pa.table({
                "dense_embedding": pa.array(C[lo:hi].tolist(), pa.list_(pa.float32())),
                # DESCENDING with corpus position, so the `id` rule and the
                # `ordinal` rule disagree on every tie. Identical ids and
                # positions would make each rule-specific assertion below pass
                # without testing anything.
                "id": pa.array([f"{M - 1 - i:05d}" for i in range(lo, hi)]),
            }),
            str(cdir / f"f{fi}.parquet"),
        )
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array(Q.tolist(), pa.list_(pa.float32())),
            "qid": pa.array([str(i) for i in range(N_Q)]),
        }),
        str(tmp / "queries.parquet"),
    )
    return {"tmp": tmp, "cdir": str(cdir), "qpath": str(tmp / "queries.parquet"),
            "C": C, "Q": Q}


@pytest.fixture(scope="module")
def collection(client, data):
    name = f"nova_bf_tie_{uuid.uuid4().hex[:8]}"
    client.create_collection(
        name, vectors_config=models.VectorParams(size=DIM, distance=models.Distance.DOT)
    )
    client.upsert(name, points=[
        models.PointStruct(id=i, vector=data["C"][i].tolist()) for i in range(M)
    ], wait=True)
    yield name
    client.delete_collection(name)


def _qdrant(client, collection, data):
    out = {}
    for qi in range(N_Q):
        pts = client.query_points(
            collection, query=data["Q"][qi].tolist(), limit=K,
            search_params=models.SearchParams(exact=True), with_payload=False,
        ).points
        out[qi] = [(int(p.id), float(p.score)) for p in pts]
    return out


def _nova(data, tag, tiebreak, n_jobs=None, batch=64):
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=data["cdir"], id_column="id"),
        queries=QueriesConfig(path=data["qpath"], id_column="qid"),
        output=OutputConfig(path=str(data["tmp"] / f"out_{tag}")),
        params=ParamsConfig(io_workers=2, dense_batch_size=batch, tiebreak=tiebreak),
        searches=[SearchSpec(name="t", k=K, metric="dot")],
    )
    if n_jobs is None:
        path = run_compute(cfg)["t"]
    else:
        for r in range(n_jobs):
            run_compute(cfg, num_jobs=n_jobs, job_rank=r)
        path = run_merge(cfg)["t"]
    t = pq.read_table(path).to_pydict()
    by_q = dict(zip(t["query_id"], zip(t["hit_ids"], t["hit_scores"])))
    return {int(q): list(zip(*v)) for q, v in by_q.items()}


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
def test_the_score_sequence_matches_qdrant_exactly(client, collection, data, tiebreak):
    """The ranking claim, which a tie-break must not touch: same scores, same
    order, same count — whichever points carry them."""
    q = _qdrant(client, collection, data)
    n = _nova(data, f"seq_{tiebreak}", tiebreak)
    for qi in range(N_Q):
        qs = [s for _, s in q[qi]]
        ns = [s for _, s in n[qi]]
        assert len(ns) == len(qs) == K
        assert np.allclose(ns, qs, atol=TOL), f"query {qi} score sequence diverged"


def _pid(hit_id: str) -> int:
    """nova hit id -> Qdrant point id. The corpus writes ids DESCENDING with
    corpus position, so the two numbering schemes are mirror images."""
    return M - 1 - int(hit_id)


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
def test_novas_hits_are_a_legitimate_selection(client, collection, data, tiebreak):
    """Above the cut, the two engines must agree on the exact point set — no tie
    can excuse a difference there. AT the cut, nova may pick different points,
    but only from among those that genuinely hold that score."""
    q = _qdrant(client, collection, data)
    n = _nova(data, f"sel_{tiebreak}", tiebreak)
    for qi in range(N_Q):
        cut = q[qi][-1][1]
        q_above = {pid for pid, s in q[qi] if s > cut + TOL}
        n_above = {_pid(i) for i, s in n[qi] if s > cut + TOL}
        assert n_above == q_above, f"query {qi} disagrees ABOVE the cut score"

        at_cut = {pid for pid, s in q[qi] if abs(s - cut) <= TOL}
        n_at = {_pid(i) for i, s in n[qi] if abs(s - cut) <= TOL}
        # Qdrant returns only K, so it need not list every tied point; what nova
        # keeps at the cut must at least carry that score.
        true_scores = data["C"] @ data["Q"][qi]
        assert all(abs(float(true_scores[p]) - cut) <= TOL for p in n_at), (
            f"query {qi} kept a point that does not tie at the cut"
        )
        assert at_cut, f"query {qi} has no tie at the cut — the corpus is not exercising this"


def test_the_two_rules_pick_different_points_here(client, collection, data):
    """Guards every rule-specific assertion from passing vacuously: this corpus
    must actually contain ties the two rules resolve differently."""
    a = _nova(data, "ra", "ordinal")
    b = _nova(data, "rb", "id")
    assert any(
        [i for i, _ in a[qi]] != [i for i, _ in b[qi]] for qi in range(N_Q)
    ), "the corpus produced no tie the two rules resolve differently"


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
def test_the_rule_holds_within_every_group_of_equal_scores(client, collection, data, tiebreak):
    """The rule itself, checked against nova's OWN scores.

    Deliberately not against a numpy oracle: numpy and torch reduce a dot
    product in different orders, so ~60% of these scores differ between them in
    the last bit — which regroups the ties and would make the oracle disagree
    for reasons that have nothing to do with the tie-break.
    """
    n = _nova(data, f"rule_{tiebreak}", tiebreak)
    for qi in range(N_Q):
        hits = n[qi]
        assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)
        start = 0
        for j in range(1, len(hits) + 1):
            if j == len(hits) or hits[j][1] != hits[start][1]:
                group = [i for i, _ in hits[start:j]]
                if tiebreak == "id":
                    assert group == sorted(group), (
                        f"query {qi}: tied ids not ascending: {group}"
                    )
                else:
                    pos = [_pid(i) for i in group]
                    assert pos == sorted(pos), (
                        f"query {qi}: tied rows not in corpus order: {pos}"
                    )
                start = j


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
def test_the_answer_survives_sharding(client, collection, data, tiebreak):
    """The invariance actually on offer: at a FIXED batch size, the result must
    not depend on how many workers produced it.

    Batch size is deliberately held constant. Re-tiling the matmul changes its
    reduction order, and on this corpus it really does move a score by one ULP
    (8.478569984436035 vs 8.478569030761719 at batch 64 vs 512), which changes
    whether two points tie AT ALL. No tie-break can make that invariant — see
    `test_a_moved_score_bit_is_what_batch_size_changes`.
    """
    for bs in (64, 128, 512):
        answers = {
            tuple(tuple(i for i, _ in _nova(data, f"inv_{tiebreak}{nj}_{bs}",
                                            tiebreak, nj, bs)[qi])
                  for qi in range(N_Q))
            for nj in (None, 2, 3)
        }
        assert len(answers) == 1, f"batch={bs}: {len(answers)} answers across shard counts"


def test_a_moved_score_bit_is_what_batch_size_changes(client, collection, data):
    """Pins the boundary as a documented property rather than folklore: where
    the batch size changes the answer, it is because it changed the SCORES."""
    a = _nova(data, "ulp64", "id", None, 64)
    b = _nova(data, "ulp512", "id", None, 512)
    for qi in range(N_Q):
        ids_a = [i for i, _ in a[qi]]
        ids_b = [i for i, _ in b[qi]]
        if ids_a == ids_b:
            continue
        bits_a = np.array([s for _, s in a[qi]], np.float32).view(np.int32)
        bits_b = np.array([s for _, s in b[qi]], np.float32).view(np.int32)
        assert not np.array_equal(bits_a, bits_b), (
            f"query {qi}: the answer moved while every score bit stayed identical "
            "— that would be a real tie-break defect, not float re-association"
        )
