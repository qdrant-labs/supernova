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
from nova_bf.results import (
    RUN_KEY, config_identity, partial_dir, provenance, run_identity,
)

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


# ---------------------------------------------------------------------------
# what the fingerprints cover
#
# The tests above drive whole runs through `merge`. These pin the hash inputs
# directly, so a field silently dropping out of a fingerprint fails here rather
# than showing up as a merge that should have been refused.
# ---------------------------------------------------------------------------


def _ident_cfg(**corpus_kw):
    return BruteForceConfig(
        corpus=CorpusConfig(path="s3://c/", id_column="sid", **corpus_kw),
        queries=QueriesConfig(path="s3://q/q.parquet", id_column="qid"),
        output=OutputConfig(path="s3://o/"),
        params=ParamsConfig(),
        searches=[SearchSpec(name="t", k=10, metric="cosine")],
    )


def test_run_identity_separates_different_max_files():
    """`--max-files` truncates each rank's OWN slice, so two ranks given
    different values covered different corpora. As a boolean they hashed the
    same and merged cleanly into a top-K short exactly where the truncated
    ranks never reached."""
    base = dict(config_sha="cfg", corpus_sha="corp", num_jobs=4, tiebreak="id")
    a = run_identity(max_files=5, **base)
    b = run_identity(max_files=50, **base)
    assert a != b, "--max-files 5 and 50 must not share a run fingerprint"

    full = run_identity(max_files=None, **base)
    assert full not in (a, b), "a full run must differ from any truncated one"
    assert run_identity(max_files=5, **base) == a, "hash must be stable"


# --- 14: date_fields change what a filter compares ------------------------


def test_config_identity_covers_date_fields():
    """`convert_table_date_columns` rewrites declared date columns to int64
    epoch us BEFORE any filter reads them, so the declaration decides which
    rows survive."""
    spec = _ident_cfg().searches[0]
    plain = config_identity(_ident_cfg(), spec)
    dated = config_identity(_ident_cfg(date_fields={"published": "%Y-%m-%d"}), spec)
    assert plain != dated, "corpus.date_fields must reach the config fingerprint"

    other = config_identity(_ident_cfg(date_fields={"published": "%d/%m/%Y"}), spec)
    assert other != dated, "a different date FORMAT parses differently"


def test_config_identity_covers_query_date_fields():
    def cfg(qdates):
        c = _ident_cfg()
        c.queries.date_fields = qdates
        return c

    spec = _ident_cfg().searches[0]
    assert config_identity(cfg(None), spec) != config_identity(
        cfg({"asked_at": "%Y-%m-%d"}), spec)


def test_config_identity_still_ignores_speed_only_knobs():
    """Guard the guard: the fingerprint must not start tracking things that
    only change speed, or every batch-size tweak invalidates a merge."""
    spec = _ident_cfg().searches[0]
    base = _ident_cfg()
    fast = _ident_cfg()
    fast.params.io_workers = 7
    fast.params.dense_batch_size = 4096
    fast.params.merge_batch_size = 99
    assert config_identity(base, spec) == config_identity(fast, spec)


# --- 16: git must be answering about THIS package -------------------------


def _prov(**kw):
    cfg = _ident_cfg()
    return provenance(cfg, cfg.searches[0], **kw), RUN_KEY


def test_merge_omits_the_run_key_when_no_partial_carried_one():
    """Partials that predate the fingerprint leave `merge` with nothing to
    record. Hashing its own empty inputs would mint a sha matching no compute
    run, claiming `num_jobs=None` for what may have been a sharded one — a
    later consumer then sees a mismatch it cannot explain."""
    meta, RUN_KEY = _prov(run_sha=None, reducing=True)
    assert RUN_KEY not in meta, "merge invented a run fingerprint"
    # everything else it CAN vouch for must still be stamped
    assert any(k.startswith(b"nova_bf.") for k in meta)


def test_merge_passes_through_a_carried_run_key():
    meta, RUN_KEY = _prov(run_sha="abc123", reducing=True)
    assert meta[RUN_KEY] == b"abc123"


def test_compute_still_mints_a_run_key_from_real_inputs():
    """`compute` holds the actual identifying inputs, so it is the one place a
    fingerprint is legitimately created — that must not regress."""
    meta, RUN_KEY = _prov(corpus_sha="deadbeef", num_jobs=4, max_files=None)
    assert RUN_KEY in meta and len(meta[RUN_KEY]) == 64

    other, _ = _prov(corpus_sha="deadbeef", num_jobs=4, max_files=7)
    assert other[RUN_KEY] != meta[RUN_KEY], "#13 must still hold"


def test_merge_of_unstamped_partials_leaves_no_run_key_on_the_artifact(tmp_path):
    """End to end: strip the fingerprint from every partial, merge, and check
    the artifact does not claim one."""
    pytest.importorskip("torch")

    import pyarrow.parquet as pq
    from nova_bf.compute import run_compute
    from nova_bf.merge import run_merge
    from nova_bf.results import RUN_KEY
    from test_result_decode import _cfg_for_merge

    cfg = _cfg_for_merge(tmp_path)
    for r in range(2):
        run_compute(cfg, num_jobs=2, job_rank=r)

    stripped = 0
    for part in sorted(tmp_path.rglob("rank*.parquet")):
        t = pq.read_table(part)
        md = {k: v for k, v in (t.schema.metadata or {}).items() if k != RUN_KEY}
        stripped += 1
        pq.write_table(t.replace_schema_metadata(md), part)
    assert stripped >= 2, "no partials found to strip"

    merged = run_merge(cfg)["t"]
    md = pq.read_schema(merged).metadata or {}
    assert RUN_KEY not in md, "merge stamped a fabricated run fingerprint"
