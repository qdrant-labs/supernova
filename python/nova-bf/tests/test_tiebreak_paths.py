"""The tie-break through the vector types that are NOT plain dense rows.

`process_slice` is shared, but each vector type reaches it differently, and each
had its own way of breaking the ordinal→row correspondence:

  * sparse slices carry a no-overlap gate that `-inf`s whole cells, and a
    coalesced sparse batch is rebuilt from several files' COO parts;
  * multivector slices are tiled by TOKEN budget, not by row, so one corpus
    row's position inside a slice has nothing to do with its position in the
    file — the case most likely to mis-index an ordinal array.

Every corpus here makes many candidates score exactly equal, so the tie-break
alone decides the answer, and the answer must not move with the tiling.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")

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

SPARSE_TYPE = pa.struct([
    pa.field("indices", pa.list_(pa.uint32())),
    pa.field("values", pa.list_(pa.float32())),
])
DIM = 4


def _read(path):
    t = pq.read_table(path).to_pydict()
    return dict(zip(t["query_id"], t["hit_ids"]))["q0"]


def _run(cfg, tmp, tag, n_jobs=None):
    if n_jobs is None:
        return _read(run_compute(cfg)["t"])
    for r in range(n_jobs):
        run_compute(cfg, num_jobs=n_jobs, job_rank=r)
    return _read(run_merge(cfg)["t"])


# --------------------------------------------------------------------------
# sparse
# --------------------------------------------------------------------------


def _sparse_corpus(tmp, per_file_ids):
    """Every row shares term 1 with weight 1.0, so every score is exactly 1.0."""
    cdir = tmp / "corpus"
    cdir.mkdir(exist_ok=True)
    for fi, ids in enumerate(per_file_ids):
        pq.write_table(
            pa.table({
                "sparse_embedding": pa.array(
                    [{"indices": [1], "values": [1.0]}] * len(ids), SPARSE_TYPE
                ),
                "sid": pa.array(ids, pa.string()),
            }),
            str(cdir / f"f{fi}.parquet"),
        )
    pq.write_table(
        pa.table({
            "sparse_embedding": pa.array(
                [{"indices": [1], "values": [1.0]}], SPARSE_TYPE
            ),
            "qid": pa.array(["q0"]),
        }),
        str(tmp / "q.parquet"),
    )
    return cdir


def _sparse_cfg(cdir, tmp, out, k, tiebreak, batch=None):
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), sparse_column="sparse_embedding",
                            id_column="sid"),
        queries=QueriesConfig(path=str(tmp / "q.parquet"),
                              sparse_column="sparse_embedding", id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=2, tiebreak=tiebreak, sparse_batch_size=batch),
        searches=[SearchSpec(name="t", k=k, metric="dot",
                             vector_type="sparse")],
    )


SP = [["s9", "s8"], ["s7", "s6"], ["s5", "s4"]]
SP_FLAT = [i for f in SP for i in f]


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
@pytest.mark.parametrize("n_jobs,batch", [
    (None, None), (None, 1), (None, 2), (2, None), (2, 1), (3, 2),
])
def test_sparse_ties_do_not_move(tmp_path, tiebreak, n_jobs, batch):
    cdir = _sparse_corpus(tmp_path, SP)
    out = tmp_path / f"sp{tiebreak}{n_jobs}{batch}"
    out.mkdir()
    cfg = _sparse_cfg(cdir, tmp_path, out, 3, tiebreak, batch)
    expected = sorted(SP_FLAT)[:3] if tiebreak == "id" else SP_FLAT[:3]
    assert _run(cfg, tmp_path, "sp", n_jobs) == expected


def test_a_sparse_row_sharing_no_term_is_still_excluded(tmp_path):
    """The no-overlap gate `-inf`s those cells, and `-inf` is also the sentinel
    the empty state carries — so an excluded row must not be resurrected by
    having a better ordinal than the padding."""
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    # 'a' shares no term with the query; 'b' and 'c' do.
    rows = [({"indices": [99], "values": [1.0]}, "a"),
            ({"indices": [1], "values": [1.0]}, "b"),
            ({"indices": [1], "values": [1.0]}, "c")]
    pq.write_table(
        pa.table({
            "sparse_embedding": pa.array([r[0] for r in rows], SPARSE_TYPE),
            "sid": pa.array([r[1] for r in rows]),
        }),
        str(cdir / "f0.parquet"),
    )
    pq.write_table(
        pa.table({
            "sparse_embedding": pa.array([{"indices": [1], "values": [1.0]}], SPARSE_TYPE),
            "qid": pa.array(["q0"]),
        }),
        str(tmp_path / "q.parquet"),
    )
    out = tmp_path / "gate"
    out.mkdir()
    got = _run(_sparse_cfg(cdir, tmp_path, out, 5, "id"), tmp_path, "gate")
    assert got == ["b", "c"], "the non-overlapping row must stay excluded"


# --------------------------------------------------------------------------
# multivector
# --------------------------------------------------------------------------


def _mv_array(docs: list[np.ndarray]) -> pa.Array:
    tok = [len(d) for d in docs]
    total = sum(tok)
    flat = (
        np.concatenate([d.reshape(-1) for d in docs if len(d)])
        if total else np.empty(0, np.float32)
    )
    inner = pa.ListArray.from_arrays(
        pa.array(np.arange(0, total * DIM + 1, DIM, dtype=np.int32)),
        pa.array(flat.astype(np.float32), pa.float32()),
    )
    outer_off = np.zeros(len(docs) + 1, dtype=np.int32)
    np.cumsum(tok, out=outer_off[1:])
    return pa.ListArray.from_arrays(pa.array(outer_off), inner)


def _mv_corpus(tmp, per_file_ids, tokens_per_doc):
    """Every token is the SAME unit vector, so every maxsim score is exactly
    the query's token count — identical for every doc regardless of how many
    tokens it has. Docs deliberately differ in token count so the token-budget
    tiling produces ragged, row-misaligned slices."""
    cdir = tmp / "corpus"
    cdir.mkdir(exist_ok=True)
    one = np.zeros(DIM, np.float32)
    one[0] = 1.0
    for fi, ids in enumerate(per_file_ids):
        docs = [np.tile(one, (tokens_per_doc(i), 1)) for i in ids]
        pq.write_table(
            pa.table({"multivector_embedding": _mv_array(docs),
                      "sid": pa.array(ids, pa.string())}),
            str(cdir / f"f{fi}.parquet"),
        )
    pq.write_table(
        pa.table({"multivector_embedding": _mv_array([np.tile(one, (2, 1))]),
                  "qid": pa.array(["q0"])}),
        str(tmp / "q.parquet"),
    )
    return cdir


def _mv_cfg(cdir, tmp, out, k, tiebreak, budget=None, batch=None):
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir),
                            multivector_column="multivector_embedding",
                            id_column="sid"),
        queries=QueriesConfig(path=str(tmp / "q.parquet"),
                              multivector_column="multivector_embedding",
                              id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=2, tiebreak=tiebreak,
                            multivector_token_budget=budget,
                            multivector_batch_size=batch),
        searches=[SearchSpec(name="t", k=k, metric="dot",
                             vector_type="multivector")],
    )


MV = [["m9", "m8"], ["m7", "m6"], ["m5", "m4"]]
MV_FLAT = [i for f in MV for i in f]


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
@pytest.mark.parametrize("budget,batch,n_jobs", [
    (None, None, None), (3, None, None), (5, None, None), (100, None, None),
    (None, 1, None), (None, 2, None),
    (None, None, 2), (3, None, 2), (None, 1, 3),
])
def test_multivector_ties_do_not_move_with_the_token_tiling(
    tmp_path, tiebreak, budget, batch, n_jobs
):
    """The token budget slices by TOKENS, so a doc's index inside a slice is
    unrelated to its row in the file. Docs here have 1-3 tokens each, so the
    slice boundaries land differently at every budget."""
    cdir = _mv_corpus(tmp_path, MV, lambda i: 1 + (int(i[1:]) % 3))
    out = tmp_path / f"mv{tiebreak}{budget}{batch}{n_jobs}"
    out.mkdir()
    cfg = _mv_cfg(cdir, tmp_path, out, 3, tiebreak, budget, batch)
    expected = sorted(MV_FLAT)[:3] if tiebreak == "id" else MV_FLAT[:3]
    assert _run(cfg, tmp_path, "mv", n_jobs) == expected


def test_every_multivector_tiling_gives_one_answer(tmp_path):
    cdir = _mv_corpus(tmp_path, MV, lambda i: 1 + (int(i[1:]) % 3))
    answers = set()
    for budget in (None, 2, 3, 7, 50):
        for n_jobs in (None, 2):
            out = tmp_path / f"all{budget}{n_jobs}"
            out.mkdir()
            cfg = _mv_cfg(cdir, tmp_path, out, 4, "id", budget)
            answers.add(tuple(_run(cfg, tmp_path, "a", n_jobs)))
    assert len(answers) == 1, f"{len(answers)} distinct answers: {answers}"


def test_a_zero_token_doc_stays_a_non_candidate(tmp_path):
    """A zero-token doc scores `-inf` — the same value the empty state carries,
    so a good ordinal must not float it into the results."""
    cdir = _mv_corpus(tmp_path, [["z1", "z0", "z2"]],
                      lambda i: 0 if i == "z0" else 2)
    out = tmp_path / "zero"
    out.mkdir()
    got = _run(_mv_cfg(cdir, tmp_path, out, 5, "id"), tmp_path, "zero")
    assert got == ["z1", "z2"], "the zero-token doc must not be a hit"
