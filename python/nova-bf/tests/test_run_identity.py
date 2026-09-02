"""`merge` must refuse partials that did not come from one complete run.

A search's partial directory is addressed by (queries stem, search name, k)
alone, so any two runs agreeing on those three write into it, and rank files
overwrite only the ranks the newer run has. The resulting mixtures merge
CLEANLY — the schema, the query rows and the hit-id shape are all identical —
and produce a wrong top-K that looks entirely normal. These tests build each
such mixture on purpose and assert the merge refuses it.
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
    OutputConfig,
    ParamsConfig,
    QueriesConfig,
    SearchSpec,
)
from nova_bf.merge import run_merge
from nova_bf.results import RUN_KEY, partial_dir

DIM, K = 8, 3


def _write(path, vectors, **columns):
    data = {"dense_embedding": pa.array(vectors.tolist(), type=pa.list_(pa.float32()))}
    data.update({k: pa.array(v) for k, v in columns.items()})
    pq.write_table(pa.table(data), str(path))


@pytest.fixture
def ds(tmp_path):
    rng = np.random.default_rng(0)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    g = 0
    for fi, n in enumerate((5, 4, 6, 3)):
        _write(
            cdir / f"f{fi}.parquet",
            rng.standard_normal((n, DIM)).astype(np.float32),
            id=[f"c{g + r}" for r in range(n)],
        )
        g += n
    qpath = tmp_path / "queries.parquet"
    _write(qpath, rng.standard_normal((4, DIM)).astype(np.float32), qid=[f"q{i}" for i in range(4)])
    return {"cdir": str(cdir), "qpath": str(qpath)}


def _cfg(ds, out, **params) -> BruteForceConfig:
    return BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=2, **params),
        searches=[SearchSpec(name="dense", metric="dot", k=K)],
    )


def _run(cfg, num_jobs, **kwargs):
    for rank in range(num_jobs):
        run_compute(cfg, num_jobs=num_jobs, job_rank=rank, **kwargs)


def test_merge_refuses_partials_from_two_runs(ds, tmp_path):
    """The headline case: a 2-rank run lands on a 4-rank run's leftovers.

    Rank files are named by rank, so the second run overwrites ranks 0-1 and
    leaves ranks 2-3 of the first behind. Both runs' slices are valid on their
    own; together they double-count the files ranks 0-1 covered under the
    2-way stride and that ranks 2-3 covered under the 4-way one.
    """
    out = tmp_path / "out"
    out.mkdir()
    cfg = _cfg(ds, out)
    _run(cfg, 4)
    _run(cfg, 2)  # overwrites rank000/rank001, leaves rank002/rank003 stale

    assert len(list((out / partial_dir(cfg, cfg.searches[0])).glob("*.parquet"))) == 4
    with pytest.raises(RuntimeError, match="MORE THAN ONE run"):
        run_merge(cfg)


def test_merge_refuses_a_missing_rank(ds, tmp_path):
    """A rank that died before writing anything leaves every search short by
    exactly one — which is uniform, and so invisible to the "same partial count
    across searches" check. Its slice of the corpus would simply be absent from
    the merged top-K, lowering every recall number computed against it."""
    out = tmp_path / "out"
    out.mkdir()
    cfg = _cfg(ds, out)
    _run(cfg, 4)
    dead = sorted((out / partial_dir(cfg, cfg.searches[0])).glob("*.parquet"))[2]
    dead.unlink()

    with pytest.raises(RuntimeError, match="missing \\[2\\]"):
        run_merge(cfg)


def test_merge_refuses_a_config_edited_between_phases(ds, tmp_path):
    """The `tiebreak` check generalized: any config field that changes results
    (here `allow_tf32`, which perturbs scores) must not differ between the run
    that produced the partials and the merge that reduces them."""
    out = tmp_path / "out"
    out.mkdir()
    _run(_cfg(ds, out), 2)

    with pytest.raises(RuntimeError, match="different config"):
        run_merge(_cfg(ds, out, allow_tf32=True))


def test_a_benchmark_slice_never_merges_with_a_full_run(ds, tmp_path):
    """`--max-files` reads only part of a rank's slice, so its output is not
    ground truth. It fingerprints as a different run precisely so it can never
    be folded into one."""
    out = tmp_path / "out"
    out.mkdir()
    cfg = _cfg(ds, out)
    run_compute(cfg, num_jobs=2, job_rank=0)
    run_compute(cfg, num_jobs=2, job_rank=1, max_files=1)

    with pytest.raises(RuntimeError, match="MORE THAN ONE run"):
        run_merge(cfg)


def test_a_re_run_rank_merges_cleanly(ds, tmp_path):
    """The flip side, and why the fingerprint is content-derived rather than a
    per-invocation uuid: the documented recovery path is to re-run just the
    failed rank. That rank must fingerprint identically to its siblings, or the
    fix would look exactly like the corruption."""
    out = tmp_path / "out"
    out.mkdir()
    cfg = _cfg(ds, out)
    _run(cfg, 3)
    run_compute(cfg, num_jobs=3, job_rank=1)  # rerun one rank, same config

    merged = run_merge(cfg)
    table = pq.read_table(merged["dense"])
    assert table.num_rows == 4
    # and the artifact records which run it came from
    assert RUN_KEY in (table.schema.metadata or {})


def test_single_node_output_carries_a_run_fingerprint(ds, tmp_path):
    """No ranks to check, but the fingerprint still identifies the run — it is
    what a later merge of re-sharded partials would be compared against."""
    out = tmp_path / "out"
    out.mkdir()
    cfg = _cfg(ds, out)
    meta = pq.read_table(run_compute(cfg)["dense"]).schema.metadata or {}
    assert len(meta[RUN_KEY].decode()) == 64
    assert b"nova_bf.num_jobs" not in meta  # a single-node run has no rank set
