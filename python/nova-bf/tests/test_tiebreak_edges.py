"""Edges of the packed key, and the guards that keep a run honest.

Grouped here rather than scattered: the score half of the key has to survive
values that are not ordinary finite floats (`+inf` is a real hit, `-inf` is the
padding, `-0.0` is what a euclidean self-hit produces), and the config/merge
guards have to refuse the mistakes that would silently apply the wrong rule.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

pytest.importorskip("torch")

import nova_bf.compute as cp
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
from nova_bf.results import RESERVED, TIEBREAK_KEY

DIM = 4


def _write(tmp, corpus_vecs, ids, qvec):
    cdir = tmp / "corpus"
    cdir.mkdir(exist_ok=True)
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array([list(map(float, v)) for v in corpus_vecs],
                                        pa.list_(pa.float32())),
            "sid": pa.array(ids, pa.string()),
        }),
        str(cdir / "f0.parquet"),
    )
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array([list(map(float, qvec))], pa.list_(pa.float32())),
            "qid": pa.array(["q0"]),
        }),
        str(tmp / "q.parquet"),
    )
    return cdir


def _cfg(cdir, tmp, out, k, tiebreak="id", metric="dot"):
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="sid"),
        queries=QueriesConfig(path=str(tmp / "q.parquet"), id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, tiebreak=tiebreak),
        searches=[SearchSpec(name="t", k=k, metric=metric)],
    )


def _hits(path):
    t = pq.read_table(path).to_pydict()
    d = dict(zip(t["query_id"], zip(t["hit_ids"], t["hit_scores"])))
    return list(zip(*d["q0"]))


# --------------------------------------------------------------------------
# the score half at its extremes
# --------------------------------------------------------------------------


def test_an_infinite_score_is_a_real_hit(tmp_path):
    """`+inf` comes out of float32 overflow on a legitimately huge dot product.
    It must rank first and survive the write path — the `-inf` padding gate is
    `> -inf`, not `isfinite`, precisely so this is not silently dropped."""
    big = 3.0e38
    cdir = _write(tmp_path, [[big, 0, 0, 0], [1, 0, 0, 0]], ["a", "b"], [big, 0, 0, 0])
    out = tmp_path / "inf"
    out.mkdir()
    got = _hits(run_compute(_cfg(cdir, tmp_path, out, 2))["t"])
    assert got[0][0] == "a" and np.isinf(got[0][1])


def test_two_infinite_scores_are_separated_by_the_tie_break(tmp_path):
    big = 3.0e38
    cdir = _write(tmp_path, [[big, 0, 0, 0]] * 3, ["c", "a", "b"], [big, 0, 0, 0])
    out = tmp_path / "inf2"
    out.mkdir()
    got = _hits(run_compute(_cfg(cdir, tmp_path, out, 2))["t"])
    assert [i for i, _ in got] == ["a", "b"], "ties at +inf still follow the rule"


@pytest.mark.parametrize("n_jobs", [None, 2])
def test_an_infinite_score_survives_the_reduce(tmp_path, n_jobs):
    """The merge gate had to change with it: `np.isfinite` there would have
    dropped a `+inf` hit, making a sharded result differ from a single-node one
    on the same corpus."""
    big = 3.0e38
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    for fi, ids in enumerate([["a"], ["b"]]):
        pq.write_table(
            pa.table({
                "dense_embedding": pa.array([[big, 0.0, 0.0, 0.0]], pa.list_(pa.float32())),
                "sid": pa.array(ids, pa.string()),
            }),
            str(cdir / f"f{fi}.parquet"),
        )
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array([[big, 0.0, 0.0, 0.0]], pa.list_(pa.float32())),
            "qid": pa.array(["q0"]),
        }),
        str(tmp_path / "q.parquet"),
    )
    out = tmp_path / f"i{n_jobs}"
    out.mkdir()
    cfg = _cfg(cdir, tmp_path, out, 2)
    if n_jobs is None:
        got = _hits(run_compute(cfg)["t"])
    else:
        for r in range(n_jobs):
            run_compute(cfg, num_jobs=n_jobs, job_rank=r)
        got = _hits(run_merge(cfg)["t"])
    assert [i for i, _ in got] == ["a", "b"]
    assert all(np.isinf(s) for _, s in got)


def test_a_euclidean_self_hit_reports_positive_zero(tmp_path):
    """Euclidean negates its distance, so an exact self-hit produces `-0.0`.
    It is numerically EQUAL to `+0.0`, so the ordinal — not the sign bit — has
    to decide between them; the key folds it, and the reported score comes back
    as `+0.0`."""
    cdir = _write(tmp_path, [[1, 0, 0, 0], [1, 0, 0, 0]], ["b", "a"], [1, 0, 0, 0])
    out = tmp_path / "euc"
    out.mkdir()
    got = _hits(run_compute(_cfg(cdir, tmp_path, out, 2, metric="euclidean"))["t"])
    assert [i for i, _ in got] == ["a", "b"], "the tie-break, not the sign bit, decides"
    assert all(np.copysign(1.0, s) > 0 for _, s in got), "reported as +0.0"


def test_a_query_with_fewer_than_k_candidates_is_truncated(tmp_path):
    """The padding must not leak out as real hits with a good-looking ordinal."""
    cdir = _write(tmp_path, [[1, 0, 0, 0], [1, 0, 0, 0]], ["b", "a"], [1, 0, 0, 0])
    out = tmp_path / "short"
    out.mkdir()
    got = _hits(run_compute(_cfg(cdir, tmp_path, out, 50))["t"])
    assert [i for i, _ in got] == ["a", "b"], "exactly the two real rows"


# --------------------------------------------------------------------------
# duplicate ids
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [3, 4])
def test_duplicate_ids_give_an_identical_artifact_at_every_shard_count(tmp_path, k):
    """Within a worker, equal ids fall back to corpus position; across workers,
    `merge` orders them by partial — which is NOT corpus order under stride
    partitioning. That is unobservable and must stay so: equal ids carry equal
    scores, so whichever underlying row is chosen the emitted artifact is
    byte-identical."""
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    groups = [["dup", "m0"], ["dup", "m1"], ["dup", "m2"], ["m3", "m4"]]
    for fi, ids in enumerate(groups):
        pq.write_table(
            pa.table({
                "dense_embedding": pa.array([[1.0, 0.0, 0.0, 0.0]] * len(ids),
                                            pa.list_(pa.float32())),
                "sid": pa.array(ids, pa.string()),
            }),
            str(cdir / f"f{fi}.parquet"),
        )
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array([[1.0, 0.0, 0.0, 0.0]], pa.list_(pa.float32())),
            "qid": pa.array(["q0"]),
        }),
        str(tmp_path / "q.parquet"),
    )
    answers = set()
    for nj in (None, 1, 2, 3):
        out = tmp_path / f"d{k}_{nj}"
        out.mkdir()
        cfg = _cfg(cdir, tmp_path, out, k)
        if nj is None:
            got = _hits(run_compute(cfg)["t"])
        else:
            for r in range(nj):
                run_compute(cfg, num_jobs=nj, job_rank=r)
            got = _hits(run_merge(cfg)["t"])
        answers.add(tuple(got))
    assert len(answers) == 1, f"{len(answers)} distinct artifacts: {answers}"


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------


def test_id_mode_without_an_id_column_is_rejected_at_config_time(tmp_path):
    with pytest.raises(ValueError, match="needs `corpus.id_column`"):
        BruteForceConfig(
            corpus=CorpusConfig(path=str(tmp_path)),
            queries=QueriesConfig(path=str(tmp_path / "q.parquet")),
            output=OutputConfig(path=str(tmp_path / "o")),
            params=ParamsConfig(tiebreak="id"),
            searches=[SearchSpec(name="t", k=2, metric="dot")],
        )


def test_an_unorderable_id_column_is_rejected_at_startup(tmp_path):
    """Binary specifically: the ordinals would order it by raw bytes, but
    `hit_ids` render it through Python's bytes repr, whose order differs — so
    the two sides of the reduce would disagree."""
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array([[1.0, 0.0, 0.0, 0.0]], pa.list_(pa.float32())),
            "sid": pa.array([b"\x01"], pa.binary()),
        }),
        str(cdir / "f0.parquet"),
    )
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array([[1.0, 0.0, 0.0, 0.0]], pa.list_(pa.float32())),
            "qid": pa.array(["q0"]),
        }),
        str(tmp_path / "q.parquet"),
    )
    out = tmp_path / "bin"
    out.mkdir()
    with pytest.raises(ValueError, match="integer or string"):
        run_compute(_cfg(cdir, tmp_path, out, 2))


def test_a_worker_slice_above_the_key_width_is_rejected(tmp_path, monkeypatch):
    """It takes 4.3B rows on one worker to fire for real, so the ceiling is
    lowered here. When it does fire the alternative is silent: the ordinal wraps
    and ties stop being deterministic, the one thing the field exists for."""
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    for fi in range(3):
        pq.write_table(
            pa.table({
                "dense_embedding": pa.array([[1.0, 0.0, 0.0, 0.0]] * 2,
                                            pa.list_(pa.float32())),
                "sid": pa.array([f"a{fi}{r}" for r in range(2)], pa.string()),
            }),
            str(cdir / f"f{fi}.parquet"),
        )
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array([[1.0, 0.0, 0.0, 0.0]], pa.list_(pa.float32())),
            "qid": pa.array(["q0"]),
        }),
        str(tmp_path / "q.parquet"),
    )
    monkeypatch.setattr(cp, "MAX_ROWS_PER_WORKER", 4)      # 6 corpus rows > 4
    out = tmp_path / "ovf"
    out.mkdir()
    with pytest.raises(RuntimeError, match="larger `--num-jobs`"):
        run_compute(_cfg(cdir, tmp_path, out, 2, tiebreak="ordinal"))


def test_merge_refuses_partials_computed_under_a_different_rule(tmp_path):
    """Editing `params.tiebreak` between compute and merge would otherwise
    reduce ties by a rule the partials were never built for."""
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    for fi, ids in enumerate([["b", "a"], ["d", "c"]]):
        pq.write_table(
            pa.table({
                "dense_embedding": pa.array([[1.0, 0.0, 0.0, 0.0]] * 2,
                                            pa.list_(pa.float32())),
                "sid": pa.array(ids, pa.string()),
            }),
            str(cdir / f"f{fi}.parquet"),
        )
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array([[1.0, 0.0, 0.0, 0.0]], pa.list_(pa.float32())),
            "qid": pa.array(["q0"]),
        }),
        str(tmp_path / "q.parquet"),
    )
    out = tmp_path / "stamp"
    out.mkdir()
    cfg = _cfg(cdir, tmp_path, out, 2, tiebreak="id")
    for r in range(2):
        run_compute(cfg, num_jobs=2, job_rank=r)

    other = _cfg(cdir, tmp_path, out, 2, tiebreak="ordinal")
    with pytest.raises(RuntimeError, match="was computed with params.tiebreak"):
        run_merge(other)


def test_the_rule_is_stamped_on_the_output(tmp_path):
    cdir = _write(tmp_path, [[1, 0, 0, 0]], ["a"], [1, 0, 0, 0])
    out = tmp_path / "st"
    out.mkdir()
    path = run_compute(_cfg(cdir, tmp_path, out, 1, tiebreak="id"))["t"]
    meta = pq.ParquetFile(path).schema_arrow.metadata or {}
    assert meta.get(TIEBREAK_KEY) == b"id"


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
def test_the_rule_is_stamped_on_a_MERGED_output_too(tmp_path, tiebreak):
    """The sharded path is the one that produces the artifacts anyone ships, and
    it went out unstamped while the single-node path carried the rule — so the
    only results whose provenance you could read were the ones least likely to
    exist."""
    cdir = _write(tmp_path, [[1, 0, 0, 0]] * 4, ["a", "b", "c", "d"], [1, 0, 0, 0])
    out = tmp_path / f"stm-{tiebreak}"
    out.mkdir()
    cfg = _cfg(cdir, tmp_path, out, 2, tiebreak=tiebreak)
    for r in range(2):
        run_compute(cfg, num_jobs=2, job_rank=r)
    merged = run_merge(cfg)["t"]
    meta = pq.ParquetFile(merged).schema_arrow.metadata or {}
    assert meta.get(TIEBREAK_KEY) == tiebreak.encode()


def test_a_payload_field_may_not_shadow_an_output_column(tmp_path):
    """`hit_tie` joined the reserved names, so a queries column of that name now
    collides: it is overwritten in the partial and dropped from the merge under
    `ordinal`, and under `id` on a string id column it survives as a STRING that
    `merge` mistakes for the int64 ordinate and dies on. Caught at config time
    instead, before a run is spent."""
    for name in RESERVED:
        with pytest.raises(ValidationError, match="reserved output column"):
            QueriesConfig(path="q.parquet", id_column="qid", payload_fields=[name])
    # an ordinary column is still fine
    QueriesConfig(path="q.parquet", id_column="qid", payload_fields=["lang", "year"])
