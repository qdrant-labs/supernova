"""The result parquet records HOW it was computed.

A ground-truth file outlives the run that made it, and nothing else in it says
which corpus, metric, k, or scoring precision produced it. Recovering
`allow_tf32` from the stored values' bit patterns is possible but absurd, so it
is written into the schema metadata instead — on the partials as well as the
merged output, since a partial is a parquet someone can pick up alone.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import pyarrow as pa
import pyarrow.parquet as pq

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
from nova_bf.merge import run_merge

DIM = 8


def _dataset(tmp_path, n_corpus=12, n_q=3):
    rng = np.random.default_rng(0)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array(
                rng.standard_normal((n_corpus, DIM)).astype(np.float32).tolist(),
                type=pa.list_(pa.float32()),
            ),
            "id": pa.array([f"c{i}" for i in range(n_corpus)]),
            "language": pa.array(["eng"] * n_corpus),
        }),
        str(cdir / "f0.parquet"),
    )
    qpath = tmp_path / "queries.parquet"
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array(
                rng.standard_normal((n_q, DIM)).astype(np.float32).tolist(),
                type=pa.list_(pa.float32()),
            ),
            "qid": pa.array([str(i) for i in range(n_q)]),
        }),
        str(qpath),
    )
    return str(cdir), str(qpath)


def _meta(path):
    raw = pq.ParquetFile(path).schema_arrow.metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items()}


def test_result_records_how_it_was_computed(tmp_path):
    cdir, qpath = _dataset(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=cdir, id_column="id"),
        queries=QueriesConfig(path=qpath, id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[SearchSpec(name="gt", vector_type="dense", metric="dot", k=5)],
    )
    meta = _meta(run_compute(cfg)["gt"])

    assert meta["nova_bf.search"] == "gt"
    assert meta["nova_bf.metric"] == "dot"
    assert meta["nova_bf.k"] == "5"
    assert meta["nova_bf.vector_type"] == "dense"
    assert meta["nova_bf.corpus_path"] == cdir
    assert meta["nova_bf.queries_path"] == qpath
    assert meta["nova_bf.corpus_column"] == "dense_embedding"
    assert meta["nova_bf.filtered"] == "false"
    # The field this exists for: exact f32 unless explicitly opted out of.
    assert meta["nova_bf.allow_tf32"] == "false"


def test_storage_dtypes_are_recorded(tmp_path):
    """The motivating case: a corpus stored `float16` and one stored `float32`
    produce byte-identical ground truth for the same vectors, because nova-bf
    upcasts before scoring. Nothing in the output reveals the difference — so
    the file has to say which it was, or a consumer is left inferring it from
    the stored values' bit patterns."""
    rng = np.random.default_rng(1)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    corpus = rng.standard_normal((12, DIM)).astype(np.float32)
    pq.write_table(
        pa.table({
            # fp16 on disk, like the FineWeb corpus
            "dense_embedding": pa.array(corpus.tolist(), type=pa.list_(pa.float16())),
            "id": pa.array([f"c{i}" for i in range(12)]),
        }),
        str(cdir / "f0.parquet"),
    )
    qpath = tmp_path / "queries.parquet"
    pq.write_table(
        pa.table({
            # fp64 queries, deliberately different from the corpus
            "dense_embedding": pa.array(
                rng.standard_normal((3, DIM)).tolist(), type=pa.list_(pa.float64())
            ),
            "qid": pa.array(["0", "1", "2"]),
        }),
        str(qpath),
    )
    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="id"),
        queries=QueriesConfig(path=str(qpath), id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[SearchSpec(name="gt", vector_type="dense", metric="dot", k=5)],
    )
    meta = _meta(run_compute(cfg)["gt"])
    assert meta["nova_bf.corpus_dtype"] == "float16"
    assert meta["nova_bf.queries_dtype"] == "float64"
    # The output's own dtype, so a consumer reads it instead of assuming.
    assert meta["nova_bf.scores_dtype"] == "float32"


def test_allow_tf32_is_recorded_when_enabled(tmp_path):
    """A TF32 run's scores carry ~3e-4 relative error, so a consumer comparing
    them against a live engine needs a looser tolerance — and can only know to
    if the file says so."""
    cdir, qpath = _dataset(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=cdir, id_column="id"),
        queries=QueriesConfig(path=qpath, id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(allow_tf32=True),
        searches=[SearchSpec(name="gt", vector_type="dense", metric="dot", k=5)],
    )
    assert _meta(run_compute(cfg)["gt"])["nova_bf.allow_tf32"] == "true"


def test_a_filtered_search_says_so(tmp_path):
    cdir, qpath = _dataset(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=cdir, id_column="id"),
        queries=QueriesConfig(path=qpath, id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[SearchSpec(
            name="gt", vector_type="dense", metric="cosine", k=5,
            filter=Filter(must=[FilterCondition(field="language", match="eng")]),
        )],
    )
    meta = _meta(run_compute(cfg)["gt"])
    assert meta["nova_bf.filtered"] == "true"
    assert meta["nova_bf.metric"] == "cosine"


def test_partials_and_the_merged_output_both_carry_it(tmp_path):
    """The merged file is what people consume, but a partial is a parquet
    someone can pick up on its own — and a merge that mixed partials from two
    different runs is exactly what this makes visible."""
    cdir, qpath = _dataset(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=cdir, id_column="id"),
        queries=QueriesConfig(path=qpath, id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[SearchSpec(name="gt", vector_type="dense", metric="dot", k=5)],
    )
    for rank in range(2):
        run_compute(cfg, num_jobs=2, job_rank=rank)
    merged = _meta(run_merge(cfg)["gt"])
    assert merged["nova_bf.metric"] == "dot"
    # Carried through the merge, which never opens the corpus itself.
    assert merged["nova_bf.corpus_dtype"] == "float32"
    assert merged["nova_bf.queries_dtype"] == "float32"
    assert merged["nova_bf.allow_tf32"] == "false"
    assert merged["nova_bf.k"] == "5"

    partials = sorted((out / "_bf_partial_queries_gt_k5").glob("*.parquet"))
    assert partials, "sharded run wrote partials"
    for p in partials:
        assert _meta(str(p))["nova_bf.metric"] == "dot"
