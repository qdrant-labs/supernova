"""`merge` must refuse partials that did not come from one complete run.

A search's partial directory is addressed by (queries stem, search name, k)
alone, so any two runs agreeing on those three write into it, and rank files
overwrite only the ranks the newer run has. The resulting mixtures merge
CLEANLY — the schema, the query rows and the hit-id shape are all identical —
and produce a wrong top-K that looks entirely normal. These tests build each
such mixture on purpose and assert the merge refuses it.
"""

from __future__ import annotations

import json
import os
import pathlib

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
from nova_bf.manifest import manifest_name
from nova_bf.merge import _inputs_forced, run_merge
from nova_bf.results import (
    CONFIG_KEY, FORCED_KEY, RUN_KEY, config_identity, merge_forced, partial_dir,
    provenance, result_name, run_identity,
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
    # Provenance tests: pin the exchange OFF. With it on (the default), a merge
    # missing a rank is refused even when forced -- tested in
    # test_threshold_exchange.py -- which would mask the checks tested here.
    params.setdefault("threshold_exchange", False)
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


def test_disabling_the_twopass_verification_changes_the_run_identity(ds, tmp_path,
                                                                     monkeypatch):
    """`NOVA_BF_TWOPASS_NO_VERIFY` must split the fingerprint.

    It is the one switch that makes the two-pass NOT output-neutral: it
    disables the proof that a padded GEMM height is bit-identical to the
    full-height one, and its own warning says padded and full-height scores
    "can disagree and reorder near-ties". Without it in the identity, a
    partial produced with the proof disabled merges silently with partials
    produced without it, and nothing downstream can tell them apart — the
    exact mixing `merge` refuses for every other reproducibility-affecting
    setting.

    `allow_tf32` is in the identity for the same reason, and like it this is
    the SETTING rather than the outcome: it splits even on a run where the
    switch happened to have no effect, which is the safe direction.
    """
    from nova_bf.results import config_identity

    cfg = _cfg(ds, tmp_path / "out")
    monkeypatch.delenv("NOVA_BF_TWOPASS_NO_VERIFY", raising=False)
    clean = config_identity(cfg, cfg.searches[0])

    monkeypatch.setenv("NOVA_BF_TWOPASS_NO_VERIFY", "1")
    unverified = config_identity(cfg, cfg.searches[0])

    assert clean != unverified, (
        "a run with the exactness proof disabled has the same identity as one "
        "without it, so their partials would merge silently")


# --------------------------------------------------------------------------
# NOVA_BF_MERGE_FORCE: one blunt, recorded escape hatch.
#
# Every test below drives `run_merge` (or, where the property is about what a
# LATER merge sees, a real artifact `run_merge` wrote). The mechanism these
# replaced was tested by poking module globals, and a reviewer showed four
# mutations -- including deleting the mechanism outright -- passing the whole
# suite. There is no module state left to poke.
# --------------------------------------------------------------------------

def _config_drift(ds, tmp_path, **params):
    """One complete run, merged under a DIFFERENT config. Returns (cfg, out).

    This is the motivating case for the flag, and the shape it is safe on: all
    partials come from one run, cover distinct slices, and are complete -- only
    the fingerprint the merge recomputes disagrees, which is exactly what a
    field added to `config_identity` does to perfectly good partials.

    NOT the mixed-leftovers shape (`_run(cfg, 4)` then `_run(cfg, 2)`), which
    these tests used to use. That one is DOUBLE COVERAGE, not unverified
    provenance: sharding is `i % num_jobs == job_rank`, so rank numbers from
    two different `num_jobs` name slices of different PARTITIONS and the
    overlap is folded twice. It is refused whatever the flag says -- see
    `test_disagreeing_num_jobs_is_refused_even_when_forced`.
    """
    out = tmp_path / "out"
    out.mkdir()
    _run(_cfg(ds, out), 2)
    return _cfg(ds, out, allow_tf32=True, **params), out


def _artifact_meta(path):
    return pq.ParquetFile(str(path)).schema_arrow.metadata or {}


def test_merge_is_not_forced_by_default(ds, tmp_path, monkeypatch):
    """Off unless asked, from either source."""
    monkeypatch.delenv("NOVA_BF_MERGE_FORCE", raising=False)
    cfg, _ = _config_drift(ds, tmp_path)
    assert merge_forced(cfg) is False
    with pytest.raises(RuntimeError, match="different config"):
        run_merge(cfg)


def test_forcing_merges_a_drifted_config_and_says_so_everywhere(
    ds, tmp_path, monkeypatch
):
    """The motivating case, end to end: a merge that would refuse is forced
    through, and BOTH the artifact and the manifest record that it was."""
    cfg, out = _config_drift(ds, tmp_path)
    monkeypatch.setenv("NOVA_BF_MERGE_FORCE", "1")

    paths = run_merge(cfg)
    meta = _artifact_meta(out / result_name(cfg, cfg.searches[0]))
    assert meta.get(FORCED_KEY) == b"true", (
        "a forced artifact must carry the marker; without it the file is "
        "indistinguishable from verified ground truth")

    doc = json.loads((out / manifest_name(
        cfg, "merge", search=cfg.searches[0].name)).read_text())
    assert doc.get("merge_forced") is True, "the manifest is what a human reads first"
    assert paths, "the merge still produced its output"


def test_a_clean_merge_carries_no_forced_marker(ds, tmp_path, monkeypatch):
    """The control. A marker that is always present says nothing."""
    monkeypatch.delenv("NOVA_BF_MERGE_FORCE", raising=False)
    out = tmp_path / "out"
    out.mkdir()
    cfg = _cfg(ds, out)
    _run(cfg, 2)
    run_merge(cfg)

    assert FORCED_KEY not in _artifact_meta(out / result_name(cfg, cfg.searches[0]))
    doc = json.loads((out / manifest_name(
        cfg, "merge", search=cfg.searches[0].name)).read_text())
    assert "merge_forced" not in doc


def test_forcing_still_stamps_the_config_fingerprint(ds, tmp_path, monkeypatch):
    """The marker INVALIDATES the provenance block; it does not edit it.

    The previous mechanism omitted `config_fingerprint` and went on stamping
    `metric`, `k`, the paths and `allow_tf32` from the same config it had just
    admitted might not describe these rows — so the hash a consumer cannot
    read was withheld while the human-readable claims it summarised were still
    asserted. One marker covers all of it, and, unlike an absent key, cannot
    be confused with a partial that predates the key.
    """
    cfg, out = _config_drift(ds, tmp_path)
    monkeypatch.setenv("NOVA_BF_MERGE_FORCE", "true")
    run_merge(cfg)

    meta = _artifact_meta(out / result_name(cfg, cfg.searches[0]))
    assert meta.get(FORCED_KEY) == b"true"
    assert CONFIG_KEY in meta, "the block is marked untrusted, not partly deleted"
    assert meta[b"nova_bf.allow_tf32"] == b"true", (
        "still copied from this merge's config — which is exactly what the "
        "marker warns the reader about")


def test_forcing_does_not_launder_through_a_re_merge(ds, tmp_path, monkeypatch):
    """A forced artifact re-merged by a CLEAN merge stays marked.

    Without propagation the second merge would stamp a fresh fingerprint for
    whatever config it was handed and drop the marker, so the taint would be
    washed out by one extra hop.
    """
    from nova_bf.results import JOB_RANK_KEY, NUM_JOBS_KEY

    cfg, out = _config_drift(ds, tmp_path)
    monkeypatch.setenv("NOVA_BF_MERGE_FORCE", "1")
    run_merge(cfg)
    forced = out / result_name(cfg, cfg.searches[0])
    assert _artifact_meta(forced).get(FORCED_KEY) == b"true"
    assert _inputs_forced([pq.ParquetFile(str(forced))]) is True

    # Re-stage the forced artifact as a clean single-rank partial directory.
    out2 = tmp_path / "out2"
    pdir = out2 / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True)
    t = pq.read_table(str(forced))
    md = dict(t.schema.metadata or {})
    md[NUM_JOBS_KEY], md[JOB_RANK_KEY] = b"1", b"0"
    pq.write_table(t.replace_schema_metadata(md), str(pdir / "rank000.parquet"))

    monkeypatch.delenv("NOVA_BF_MERGE_FORCE", raising=False)
    # Matching config, so the second merge trips NOTHING: the marker it ends up
    # with can only have come from its input.
    cfg2 = _cfg(ds, out2, allow_tf32=True)
    assert merge_forced(cfg2) is False, "the second merge is clean"
    run_merge(cfg2)

    # Through run_merge, not by calling `provenance` by hand: dropping
    # `inputs_forced=` from `_reduce`'s call site -- the only place it is ever
    # wired -- passed the whole suite while the propagation was unreachable.
    assert _artifact_meta(
        out2 / result_name(cfg2, cfg2.searches[0])).get(FORCED_KEY) == b"true", (
        "a clean re-merge laundered the forced marker off its input")
    doc2 = json.loads((out2 / manifest_name(
        cfg2, "merge", search=cfg2.searches[0].name)).read_text())
    assert doc2.get("merge_forced") is True, (
        "the parquet kept the marker but the manifest -- what a human reads "
        "first -- came out clean")


def test_the_force_flag_reads_from_the_config_too(ds, tmp_path, monkeypatch):
    """A rescue that outlives one shell needs to be committable."""
    monkeypatch.delenv("NOVA_BF_MERGE_FORCE", raising=False)
    cfg, out = _config_drift(ds, tmp_path)
    forced_cfg = _cfg(ds, out, allow_tf32=True, merge_force=True)
    assert merge_forced(forced_cfg) is True
    run_merge(forced_cfg)
    assert _artifact_meta(out / result_name(cfg, cfg.searches[0])).get(FORCED_KEY) == b"true"


@pytest.mark.parametrize("val", ["0", "false", "no", "off"])
def test_an_explicit_false_env_beats_a_forced_config(ds, tmp_path, monkeypatch, val):
    """`NOVA_BF_MERGE_FORCE=0` must be able to turn OFF a committed
    `merge_force: true`, not merely fail to turn it on. An operator disarming
    a config they inherited has no other lever."""
    monkeypatch.setenv("NOVA_BF_MERGE_FORCE", val)
    cfg, out = _config_drift(ds, tmp_path)
    forced_cfg = _cfg(ds, out, allow_tf32=True, merge_force=True)
    assert merge_forced(forced_cfg) is False
    with pytest.raises(RuntimeError, match="different config"):
        run_merge(forced_cfg)


@pytest.mark.parametrize("val", ["maybe", "yes please", "2", "all", "config"])
def test_an_unparseable_force_value_is_refused_not_guessed(
    ds, tmp_path, monkeypatch, val
):
    """Both defaults are wrong: treating it as false refuses a merge the
    operator meant to force, treating it as true forces one they did not.
    `config`/`all` are included because they were valid under the per-check
    variable this replaced, so a stale invocation must fail loudly rather than
    read as either answer.
    """
    monkeypatch.setenv("NOVA_BF_MERGE_FORCE", val)
    cfg, _ = _config_drift(ds, tmp_path)
    with pytest.raises(RuntimeError, match="NOVA_BF_MERGE_FORCE"):
        run_merge(cfg)


def test_the_force_value_is_validated_before_any_reduce(ds, tmp_path, monkeypatch):
    """A typo must surface before the merge spends its I/O, not after.

    Asserted on the READS. `raises` plus "no output file" does not show this:
    `merge_forced` was also reachable from `provenance`, and `_reduce`'s
    `finally` deletes the incomplete output on any failure -- so both held with
    the preamble check removed entirely.
    """
    import nova_bf.merge as m

    out = tmp_path / "out"
    out.mkdir()
    cfg = _cfg(ds, out)
    _run(cfg, 2)                               # a merge that would SUCCEED

    reads: list[str] = []
    real = m.Store.read_columns

    def spy(self, read_path, columns, *args, **kwargs):
        reads.append(read_path)
        return real(self, read_path, columns, *args, **kwargs)

    monkeypatch.setattr(m.Store, "read_columns", spy)
    monkeypatch.setenv("NOVA_BF_MERGE_FORCE", "yepp")
    with pytest.raises(RuntimeError, match="NOVA_BF_MERGE_FORCE"):
        run_merge(cfg)
    assert reads == [], (
        f"the merge read {len(reads)} partial(s) before noticing the typo; the "
        "flag must be resolved in run_merge's preamble")
    assert not list(out.glob("bf_*.parquet")), "nothing was written"


def test_merge_force_is_a_config_field(ds, tmp_path):
    """It must survive a YAML round trip, or it cannot be committed."""
    import yaml

    cfg = _cfg(ds, tmp_path / "out")
    assert cfg.params.merge_force is False, "off by default"
    raw = yaml.safe_load(yaml.safe_dump(cfg.model_dump(mode="json")))
    raw["params"]["merge_force"] = True
    assert BruteForceConfig(**raw).params.merge_force is True


def _stamped(job_rank, num_jobs=2, run=b"R", tie=b"id"):
    """A minimal partial/reader pair carrying just the metadata the rank
    checks read — enough to drive `_validate_one_run` without a compute run."""
    import types
    from nova_bf.results import RUN_KEY, NUM_JOBS_KEY, JOB_RANK_KEY, TIEBREAK_KEY

    meta = {RUN_KEY: run, TIEBREAK_KEY: tie, NUM_JOBS_KEY: str(num_jobs).encode()}
    if job_rank is not None:
        meta[JOB_RANK_KEY] = str(job_rank).encode()
    schema = types.SimpleNamespace(
        metadata=meta, names=["query_id", "hit_ids", "hit_scores"])
    reader = types.SimpleNamespace(schema_arrow=schema)
    r = job_rank if job_rank is not None else 9
    parquet = types.SimpleNamespace(read_path=f"dir/rank{r:03d}.parquet")
    return parquet, reader


@pytest.mark.parametrize("ranks,why", [
    ([0, 0, 1], "a duplicated rank double-counts its corpus slice"),
    ([0, 1, 5], "a rank outside 0..num_jobs-1 is not part of this run"),
    ([0, 1, None], "a partial with no job_rank cannot be shown not to duplicate"),
])
def test_double_coverage_is_refused_even_when_forced(
    ds, tmp_path, monkeypatch, ranks, why
):
    """The one carve-out. Every other check says "I cannot VERIFY this is
    right"; this one says "I can prove it is wrong" — the merged top-K would
    hold the same document id in several slots and fewer than k distinct
    documents per query. Forcing buys unverified output, never corrupt output.

    Driven through `_validate_one_run`, not by grepping the source. An earlier
    version of this test asserted only that the string `"if dupes or extra:"`
    appeared in merge.py — it passed while a reviewer demonstrated a live
    counterexample.
    """
    from nova_bf import merge as m

    monkeypatch.setenv("NOVA_BF_MERGE_FORCE", "1")
    cfg = _cfg(ds, tmp_path / "out")
    spec = cfg.searches[0]
    assert merge_forced(cfg) is True, "the carve-out is what refuses, not the flag"
    # Every partial carries the config fingerprint this merge computes, so the
    # config check passes and the rank checks are what the test reaches.
    sha = config_identity(cfg, spec).encode()
    pairs = [_stamped(r, num_jobs=2) for r in ranks]
    for _, rd in pairs:
        rd.schema_arrow.metadata[CONFIG_KEY] = sha
    # Each of the three has its own message: duplicates and unstamped partials
    # say "counted TWICE", an out-of-range rank says it is not part of this
    # run. All three refuse, which is what the test is about.
    with pytest.raises(RuntimeError,
                       match="counted TWICE|refused even under|not part of this run"):
        m._validate_one_run(cfg, spec, [f for f, _ in pairs], [r for _, r in pairs])


def test_disagreeing_num_jobs_is_refused_even_when_forced(ds, tmp_path, monkeypatch):
    """Two runs' leftovers are DOUBLE COVERAGE, not unverified provenance.

    Sharding is `i % num_jobs == job_rank`, so a rank NUMBER only names a
    corpus slice relative to its own `num_jobs`. A 4-way run overwritten by a
    2-way one leaves ranks {0,1,2,3} all distinct AS NUMBERS while the 4-way
    rank2/rank3 slices ({2,6,...}/{3,7,...}) sit inside the 2-way rank0/rank1
    slices ({0,2,4,...}/{1,3,5,...}) -- files 2 and 3 folded twice.

    An earlier version routed this through the flag on the argument that
    distinct rank numbers meant distinct slices. Forced, it returned normally
    and produced a k=3 top-K holding the same document id in two of three
    slots for 4 of 4 queries.
    """
    out = tmp_path / "out"
    out.mkdir()
    cfg = _cfg(ds, out)
    _run(cfg, 4)
    _run(cfg, 2)      # overwrites rank000/rank001, leaves rank002/rank003 stale

    monkeypatch.setenv("NOVA_BF_MERGE_FORCE", "1")
    with pytest.raises(RuntimeError, match="folded twice|counted TWICE|DIFFERENT"):
        run_merge(cfg)
    assert not list(out.glob("bf_*.parquet")), "nothing was written"


def test_a_duplicate_is_caught_with_no_num_jobs_stamp(ds, tmp_path):
    """Rank IDENTITY must not sit behind the `num_jobs` early return.

    The pre-f050cf5 ground-truth directories carry `job_rank` but predate the
    `num_jobs` stamp. While the two checks shared `if declared == {None}:
    return`, such a directory plus one copied partial merged CLEANLY and
    double-counted that rank's slice -- 4 of 4 queries came back with a
    repeated document id, nothing forced, nothing logged. Every other rank
    test in this file stamps `num_jobs`, so this shape was never constructed
    and reverting the restructure passed the whole suite.

    The copy is named `dup.parquet`, not `rank002.parquet`, so it is the
    duplicate-rank check that refuses and not the filename-vs-metadata one.
    """
    from nova_bf.results import NUM_JOBS_KEY

    out = tmp_path / "out"
    out.mkdir()
    cfg = _cfg(ds, out)
    _run(cfg, 2)
    pdir = out / partial_dir(cfg, cfg.searches[0])

    for f in sorted(pdir.glob("rank*.parquet")):       # strip num_jobs ONLY
        t = pq.read_table(str(f))
        md = {k: v for k, v in (t.schema.metadata or {}).items()
              if k != NUM_JOBS_KEY}
        pq.write_table(t.replace_schema_metadata(md), str(f))
    (pdir / "dup.parquet").write_bytes((pdir / "rank000.parquet").read_bytes())

    with pytest.raises(RuntimeError, match="counted TWICE"):
        run_merge(cfg)


def test_a_missing_rank_by_contrast_is_forceable(ds, tmp_path, monkeypatch):
    """The other side of the carve-out: omission is merely pessimistic, so it
    is exactly what the flag is for. If this refused too, the flag would be
    unusable for the case that motivated it."""
    out = tmp_path / "out"
    out.mkdir()
    cfg = _cfg(ds, out)
    _run(cfg, 4)
    sorted((out / partial_dir(cfg, cfg.searches[0])).glob("*.parquet"))[2].unlink()

    monkeypatch.setenv("NOVA_BF_MERGE_FORCE", "1")
    run_merge(cfg)
    assert _artifact_meta(out / result_name(cfg, cfg.searches[0])).get(FORCED_KEY) == b"true"


# --------------------------------------------------------------------------
# `params.merge_window` / `NOVA_BF_MERGE_WINDOW`: the operator names the
# in-flight count. There is no estimate to fall back to -- see
# `test_merge_window_is_exactly_what_the_operator_set` in test_merge.py for
# why the metadata-derived one was deleted.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("window,want", [(1, 1), (3, 3), (9, 4)])
def test_the_merge_window_reaches_run_merge_from_the_config(
    ds, tmp_path, monkeypatch, window, want
):
    """The number in the YAML is the number of readers -- BOTH directions.

    Counted through concurrent `read_columns` calls, not by inspecting what
    `_merge_window` returned: the pin has to survive the clamp, the Semaphore
    and the Queue to mean anything. Parametrized because only trying
    `merge_window=1` pins a ceiling and not a floor -- hard-coding
    `window_n = 1` at the call site passed that version.
    """
    import threading
    import time
    import nova_bf.merge as m

    out = tmp_path / "out"
    out.mkdir()
    cfg = _cfg(ds, out, merge_window=window)
    _run(cfg, 4)

    live = peak = 0
    lock = threading.Lock()
    real = m.Store.read_columns

    def counting(self, read_path, columns, *args, **kwargs):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        try:
            time.sleep(0.4)          # hold the permit so readers overlap
            return real(self, read_path, columns)
        finally:
            with lock:
                live -= 1

    monkeypatch.setattr(m.Store, "read_columns", counting)
    run_merge(cfg)
    assert peak == want, (
        f"merge_window={window} over 4 partials admitted {peak} concurrent "
        f"readers, expected {want}")


def test_the_merge_window_is_a_config_field(ds, tmp_path):
    """It has to be settable from the YAML the run is launched with, not only
    from an env var that lives in someone's shell history."""
    import yaml
    import pydantic

    cfg = _cfg(ds, tmp_path / "out")
    assert cfg.params.merge_window == 2, "read/fold overlap by default"

    raw = yaml.safe_load(yaml.safe_dump(cfg.model_dump(mode="json")))
    raw["params"]["merge_window"] = 3
    assert BruteForceConfig(**raw).params.merge_window == 3

    # Non-positive values are refused at parse time rather than deadlocking on
    # `Semaphore(0)` or silently unbounding `Queue(maxsize=0)` at merge time.
    for bad in (0, -1):
        raw["params"]["merge_window"] = bad
        with pytest.raises(pydantic.ValidationError):
            BruteForceConfig(**raw)


# --------------------------------------------------------------------------
# `--search`: reduce one search per process, so four searches can use a box
# that one merge leaves ~97% idle.
# --------------------------------------------------------------------------

def _two_search_cfg(ds, out, **params):
    params.setdefault("threshold_exchange", False)   # see `_cfg`
    return BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=2, **params),
        searches=[SearchSpec(name="alpha", metric="dot", k=K),
                  SearchSpec(name="beta", metric="cosine", k=K)],
    )


def test_only_reduces_the_named_search(ds, tmp_path):
    """One process, one search: the others' outputs must not appear."""
    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)

    got = run_merge(cfg, only={"alpha"})
    assert set(got) == {"alpha"}
    assert (out / result_name(cfg, cfg.searches[0])).exists()
    assert not (out / result_name(cfg, cfg.searches[1])).exists()

    # ...and the second process completes the set without disturbing the first.
    run_merge(cfg, only={"beta"})
    assert (out / result_name(cfg, cfg.searches[1])).exists()


def test_each_restricted_merge_writes_its_own_manifest(ds, tmp_path):
    """Four processes must not race for one manifest key.

    Asserted by running both restricted merges and checking BOTH manifests
    survive: a single shared name would leave only the last writer's.
    """
    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)

    run_merge(cfg, only={"alpha"})
    run_merge(cfg, only={"beta"})
    for name in ("alpha", "beta"):
        path = out / manifest_name(cfg, "merge", search=name)
        assert path.exists(), f"{name}'s manifest was clobbered"
        doc = json.loads(path.read_text())
        assert [s["name"] for s in doc["searches"]] == [name], (
            "a per-search manifest must describe only its own search")
    assert not (out / manifest_name(cfg, "merge")).exists(), (
        "merge writes one manifest per search and never a run-level one")


def test_a_restricted_merge_still_validates_every_search(ds, tmp_path):
    """`only` restricts the REDUCE, not the CHECKS.

    A rank that died partway through writing its per-search outputs leaves the
    searches with different partial counts — which is only visible by looking
    at searches this process is not reducing. Scoping the checks to the named
    search would delete exactly the evidence they exist to find.
    """
    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)
    # `beta` loses a rank; `alpha` is untouched and is all this process reduces.
    sorted((out / partial_dir(cfg, cfg.searches[1])).glob("*.parquet"))[1].unlink()

    with pytest.raises(RuntimeError, match="mismatched partial counts"):
        run_merge(cfg, only={"alpha"})


def test_an_unknown_search_name_is_refused(ds, tmp_path):
    """A typo'd --search must not silently merge nothing and report success."""
    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)

    with pytest.raises(RuntimeError, match="does not define|alpha"):
        run_merge(cfg, only={"alhpa"})


# --------------------------------------------------------------------------
# `_validate_one_run`'s metadata-sanity guards. Each of these refuses a shape
# that is CORRUPT rather than merely unverified, so none is forceable.
# --------------------------------------------------------------------------

def _pairs(specs, cfg, spec):
    """(partials, readers) from (rank, num_jobs, filename) triples."""
    from nova_bf.results import CONFIG_KEY, config_identity
    sha = config_identity(cfg, spec).encode()
    ps, rs = [], []
    for rank, num_jobs, name in specs:
        f, r = _stamped(rank, num_jobs=num_jobs) if num_jobs is not None else _stamped(rank)
        if name is not None:
            f = type(f)(read_path="dir/" + name)
        r.schema_arrow.metadata[CONFIG_KEY] = sha
        ps.append(f); rs.append(r)
    return ps, rs


def test_a_negative_job_rank_is_refused(ds, tmp_path):
    """`job_rank=-1` is corrupt metadata. It slips past the 0..num_jobs-1 range
    check in an UNSHARDED directory, where `num_jobs` is absent and that check
    never runs at all — so it needs its own guard, not the range test."""
    from nova_bf import merge as m

    cfg = _cfg(ds, tmp_path / "out")
    spec = cfg.searches[0]
    ps, rs = _pairs([(-1, None, "a.parquet"), (1, None, "b.parquet")], cfg, spec)
    with pytest.raises(RuntimeError, match="non-negative"):
        m._validate_one_run(cfg, spec, ps, rs, forced=True)


@pytest.mark.parametrize("raw,why", [("abc", "not an integer"), ("0", "must be positive"),
                                     ("-4", "must be positive")])
def test_a_corrupt_num_jobs_is_diagnosed_not_crashed(ds, tmp_path, raw, why):
    """`int(declared.pop())` on junk used to raise a bare ValueError traceback.
    An operator needs to be told the metadata is corrupt, not handed a stack."""
    from nova_bf import merge as m

    cfg = _cfg(ds, tmp_path / "out")
    spec = cfg.searches[0]
    ps, rs = _pairs([(0, raw, None), (1, raw, None)], cfg, spec)
    with pytest.raises(RuntimeError, match=why):
        m._validate_one_run(cfg, spec, ps, rs, forced=True)


def test_partials_and_readers_must_be_the_same_length(ds, tmp_path):
    """They are zipped; a mismatch would silently truncate the rank set and
    could turn a missing rank into an invisible one."""
    from nova_bf import merge as m

    cfg = _cfg(ds, tmp_path / "out")
    spec = cfg.searches[0]
    ps, rs = _pairs([(0, "2", None), (1, "2", None)], cfg, spec)
    with pytest.raises(RuntimeError, match="but opened"):
        m._validate_one_run(cfg, spec, ps, rs[:1], forced=True)


def test_an_empty_partial_list_is_refused(ds, tmp_path):
    """Defence in depth: `run_merge` catches this earlier, but a direct caller
    must not get a silent success over nothing."""
    from nova_bf import merge as m

    cfg = _cfg(ds, tmp_path / "out")
    with pytest.raises(RuntimeError, match="no partials"):
        m._validate_one_run(cfg, cfg.searches[0], [], [], forced=True)


def test_partials_with_no_config_fingerprint_warn_rather_than_pass_silently(
    ds, tmp_path, caplog
):
    """A MISSING config fingerprint used to be read as a MATCHING one.

    `(_get(meta, CONFIG_KEY) or want_config) != want_config` treats absent as
    equal, so a directory that could not be checked at all looked exactly like
    one that passed. Absence must be visible — this is the same "absent is not
    verified" trap that produced the forced-marker laundering bug.
    """
    import logging
    from nova_bf import merge as m
    from nova_bf.results import CONFIG_KEY

    cfg = _cfg(ds, tmp_path / "out")
    spec = cfg.searches[0]
    ps, rs = _pairs([(0, "2", None), (1, "2", None)], cfg, spec)
    for r in rs:
        del r.schema_arrow.metadata[CONFIG_KEY]
    with caplog.at_level(logging.WARNING, logger="nova_bf.merge"):
        m._validate_one_run(cfg, spec, ps, rs, forced=False)
    assert any("config fingerprint" in r.message for r in caplog.records), (
        "unverifiable config must be logged, not silently accepted")


@pytest.mark.parametrize("backend", ["numpy", "cpu"])
def test_run_merge_output_is_identical_across_fold_backends(ds, tmp_path, monkeypatch, backend):
    """END TO END through `run_merge`, not through `_topk_merge`.

    The unit tests compare the two folds on hand-built ListArrays. This drives
    a real compute + merge and compares the written parquet, so it also covers
    the batching, the partial-major loop, the running-state handoff between
    folds, and the Arrow assembly — everywhere a fast path can be right in
    isolation and wrong in the pipeline.

    `numpy` is the reference: it predates all of this and shares no code with
    the device path beyond `_assemble`.
    """
    out = tmp_path / "out"
    out.mkdir()
    cfg = _cfg(ds, out)
    _run(cfg, 2)

    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", backend)
    run_merge(cfg)
    t = pq.read_table(str(out / result_name(cfg, cfg.searches[0])))
    got = {r["query_id"]: (r["hit_ids"], r["hit_scores"]) for r in t.to_pylist()}

    # The reference, recomputed in-process so the comparison cannot go stale.
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "numpy")
    out2 = tmp_path / "ref"
    out2.mkdir()
    cfg2 = _cfg(ds, out2)
    _run(cfg2, 2)
    run_merge(cfg2)
    t2 = pq.read_table(str(out2 / result_name(cfg2, cfg2.searches[0])))
    ref = {r["query_id"]: (r["hit_ids"], r["hit_scores"]) for r in t2.to_pylist()}

    assert set(got) == set(ref)
    for q in ref:
        assert got[q][0] == ref[q][0], f"query {q}: hit ids differ under {backend}"
        assert got[q][1] == ref[q][1], f"query {q}: hit scores differ under {backend}"


def _write_cfg(cfg, path):
    import yaml
    pathlib.Path(path).write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return str(path)


def test_jobs_fans_out_one_process_per_search(ds, tmp_path):
    """`--jobs N` must reduce N searches concurrently and produce the FULL set.

    Driven through the real CLI, not by calling the helper: the point of the
    flag is that an operator does not have to orchestrate this, so the thing
    under test is the command, the child invocations, and the outputs landing.
    """
    import subprocess
    import sys

    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)
    path = _write_cfg(cfg, tmp_path / "cfg.yaml")

    r = subprocess.run(
        [sys.executable, "-m", "nova_bf.cli", "merge", path, "--jobs", "2"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(pathlib.Path("src").resolve())})
    assert r.returncode == 0, r.stderr[-2000:]
    for spec in cfg.searches:
        assert (out / result_name(cfg, spec)).exists(), f"{spec.name} output missing"
        assert (out / manifest_name(cfg, "merge", search=spec.name)).exists()
    assert "started" in r.stderr and "finished" in r.stderr


def test_jobs_preflights_whole_run_failures_once(ds, tmp_path):
    """A dead rank fails EVERY search, so it must be diagnosed ONCE.

    The partial-count check is cross-search by nature, so without a pre-flight
    every child runs it and every child fails: measured, four identical
    tracebacks interleaved on stderr with the one useful line buried among
    them. The parent should refuse before spawning anything.
    """
    import subprocess
    import sys

    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)
    sorted((out / partial_dir(cfg, cfg.searches[1])).glob("*.parquet"))[1].unlink()
    path = _write_cfg(cfg, tmp_path / "cfg.yaml")

    r = subprocess.run(
        [sys.executable, "-m", "nova_bf.cli", "merge", path, "--jobs", "2"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(pathlib.Path("src").resolve())})
    assert r.returncode != 0
    assert "mismatched partial counts" in r.stderr
    assert "refusing to fan out" in r.stderr
    assert r.stderr.count("mismatched partial counts") == 1, (
        "the whole-run failure was reported once per child instead of once")
    assert "started" not in r.stderr, "children were spawned despite a whole-run failure"


def test_jobs_reports_a_per_search_failure_as_incomplete(ds, tmp_path):
    """A failure that hits ONE search must say the output set is incomplete.

    This is the dangerous shape: the other searches wrote fresh output, so the
    prefix looks finished. The exit code and message are the only record.
    """
    import subprocess
    import sys

    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)
    # Row-misalign ONE search's partial: realistic after a rank died mid-write.
    # Partial counts still match, so the pre-flight passes and the failure is
    # per-search, inside that child's reduce.
    victim = sorted((out / partial_dir(cfg, cfg.searches[1])).glob("*.parquet"))[1]
    t = pq.read_table(str(victim))
    pq.write_table(t.slice(0, max(1, t.num_rows - 1)).replace_schema_metadata(
        t.schema.metadata), str(victim))
    path = _write_cfg(cfg, tmp_path / "cfg.yaml")

    r = subprocess.run(
        [sys.executable, "-m", "nova_bf.cli", "merge", path, "--jobs", "2"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(pathlib.Path("src").resolve())})
    assert r.returncode != 0, "a failed search must not exit zero"
    assert "INCOMPLETE" in r.stderr, (
        "the operator must be told the output set is incomplete, not just that "
        "one process failed")
    assert "--search" in r.stderr, "the message must name the recovery command"


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_jobs_below_one_is_refused(ds, tmp_path, bad):
    """`-j 0` silently ran a SERIAL merge and, because `only` was then None,
    also wrote the whole-run manifest into a prefix merged per-search. A typo
    for `-j 4` must not quietly change the manifest layout."""
    import subprocess
    import sys

    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)
    path = _write_cfg(cfg, tmp_path / "cfg.yaml")

    r = subprocess.run(
        [sys.executable, "-m", "nova_bf.cli", "merge", path, "--jobs", bad],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(pathlib.Path("src").resolve())})
    assert r.returncode != 0 and "at least 1" in (r.stderr + r.stdout)
    assert not list(out.glob("bf_*.parquet")), "a refused --jobs must not merge"


def test_merge_writes_one_manifest_per_search_and_no_run_level_one(ds, tmp_path):
    """ONE LAYOUT. There used to be two -- a run-level `<stem>_merge.json` and a
    per-search `<stem>_merge/<name>.json` -- and whichever a given merge did not
    write was left beside outputs it no longer described. Reconciling them
    produced a bug in every review round: deleted one way but not the other;
    deleted by config-derived name so a renamed search kept an orphan; a
    carry-over that raced four `--jobs` children; and finally a parent that
    destroyed a still-accurate record before any work had been done.

    Now a whole-run merge writes N per-search manifests and a `--search` merge
    writes one, so re-merging a single search overwrites exactly its own record
    and touches nothing else.
    """
    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)

    run_merge(cfg)
    assert not (out / manifest_name(cfg, "merge")).exists(), (
        "merge must not write a run-level manifest")
    for name in ("alpha", "beta"):
        assert (out / manifest_name(cfg, "merge", search=name)).exists()

    # Re-merging one search leaves the other's record alone.
    beta = out / manifest_name(cfg, "merge", search="beta")
    before = beta.read_text()
    run_merge(cfg, only={"alpha"})
    assert beta.read_text() == before, (
        "a --search merge rewrote a search it did not reduce")


def _plant_legacy(cfg, out):
    """Put the prefix in the state an OLDER BUILD would have left it in.

    That means the parquets exist and the run-level manifest describes them,
    but the per-search manifests this build writes do not exist -- which is the
    whole reason the legacy file is worth keeping. Planting it beside a set of
    per-search manifests would not test anything: the legacy file would be
    redundant and dropping it would lose nothing.
    """
    run_merge(cfg)
    for spec in cfg.searches:
        (out / manifest_name(cfg, "merge", search=spec.name)).unlink()
    legacy = out / manifest_name(cfg, "merge")
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps({"searches": [
        {"name": s.name, "output_file": result_name(cfg, s)}
        for s in cfg.searches]}))
    return legacy


def test_the_legacy_manifest_goes_only_once_nothing_it_describes_is_orphaned(
        ds, tmp_path):
    """A prefix merged by an older build still has `<stem>_merge.json`.

    The rule is about FILES, not about whether the merge was called whole-run:
    a merge that has rewritten every parquet the legacy file names may retire
    it, and one that left any of them untouched may not -- that parquet would
    be left with no record at all, and `--search NAME` is the documented
    recovery path, so a subset merge is a likely first touch on a legacy prefix.
    """
    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)

    legacy = _plant_legacy(cfg, out)
    body = legacy.read_text()
    run_merge(cfg, only={"alpha"})
    assert legacy.exists() and legacy.read_text() == body, (
        "a subset merge destroyed the only record beta's output had")

    run_merge(cfg)
    assert not legacy.exists(), (
        "this merge rewrote every parquet the legacy file named, so it should "
        "have superseded it")

    # An explicit --search set that covers everything counts too.
    legacy = _plant_legacy(cfg, out)
    run_merge(cfg, only={"alpha", "beta"})
    assert not legacy.exists()



def test_jobs_one_does_not_fan_out(ds, tmp_path, monkeypatch):
    """The default path must stay in-process: no subprocesses, no behaviour
    change for anyone who does not ask for concurrency."""
    import nova_bf.cli as cli_mod

    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)
    path = _write_cfg(cfg, tmp_path / "cfg.yaml")

    called = []
    monkeypatch.setattr(cli_mod, "_merge_fanout",
                        lambda *a, **k: called.append(1) or 0)
    from click.testing import CliRunner
    res = CliRunner().invoke(cli_mod.main, ["merge", path])
    assert res.exit_code == 0, res.output
    assert called == [], "--jobs 1 must not spawn child processes"
    assert (out / result_name(cfg, cfg.searches[0])).exists()


def test_a_repeated_search_flag_does_not_start_two_children_for_it(
        ds, tmp_path, monkeypatch):
    """`--search alpha --search alpha` must reduce alpha ONCE.

    `_merge_fanout` used to key its child table by search name, so the second
    `Popen` overwrote the first entry: that child was never polled, never
    waited on and never terminated by the `finally` -- exactly the invisible
    orphan that block exists to prevent -- while two processes wrote the same
    output key concurrently. The command still exited 0. Config validation
    forbids duplicate search NAMES, so only the flag side could produce it
    (scripted flag accumulation, copy-paste), and it gave no signal at all.
    """
    import nova_bf.cli as cli_mod

    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)
    path = _write_cfg(cfg, tmp_path / "cfg.yaml")

    seen: list[list[str]] = []
    monkeypatch.setattr(cli_mod, "_merge_fanout",
                        lambda config, names, jobs: seen.append(names) or 0)
    from click.testing import CliRunner
    res = CliRunner().invoke(
        cli_mod.main,
        ["merge", path, "-j", "2", "--search", "alpha", "--search", "alpha",
         "--search", "beta"])
    assert res.exit_code == 0, res.output
    assert seen == [["alpha", "beta"]], seen


def test_the_fanout_child_table_is_keyed_per_child_not_per_name(
        tmp_path, monkeypatch):
    """Even given a duplicate, every child must stay reachable for the `finally`.

    The dedupe above is the first line of defence; this pins the second. The
    table used to be keyed by search NAME, so two children for one name meant
    the first entry was overwritten and that process was never polled, waited
    on, or terminated -- an orphan merge writing the output prefix invisibly.
    """
    import subprocess
    import sys
    import time

    import nova_bf.cli as cli_mod
    import nova_bf.config as config_mod
    import nova_bf.merge as merge_mod

    # Patch the name `cli` RESOLVES, not the one it came from: cli.py imports
    # `load_config` at module scope, so `nova_bf.config.load_config` is a
    # different binding and patching it no longer intercepts the call.
    monkeypatch.setattr(cli_mod, "load_config", lambda p: object())
    monkeypatch.setattr(merge_mod, "preflight_searches", lambda cfg: None)

    started: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def sleeper(cmd, *a, **kw):
        proc = real_popen([sys.executable, "-c", "import time; time.sleep(30)"])
        started.append(proc)
        return proc

    real_sleep = time.sleep
    fired: list[int] = []

    def stop_waiting(seconds):
        # ONCE. `Popen.wait(timeout=...)` also calls `time.sleep`, so a handler
        # that keeps raising fires again inside the cleanup this test is trying
        # to observe -- the failure then points at `subprocess._wait` instead of
        # at the child that escaped.
        if not fired:
            fired.append(1)             # both children are registered by now
            raise RuntimeError("stop waiting")
        return real_sleep(seconds)

    monkeypatch.setattr(subprocess, "Popen", sleeper)
    monkeypatch.setattr(time, "sleep", stop_waiting)
    try:
        with pytest.raises(RuntimeError, match="stop waiting"):
            cli_mod._merge_fanout(str(tmp_path / "cfg.yaml"), ["a", "a"], 2)
    finally:
        monkeypatch.setattr(subprocess, "Popen", real_popen)

    assert len(started) == 2, "both children must have been launched"
    for proc in started:
        proc.wait(timeout=30)
        assert proc.poll() is not None, "a child escaped the fan-out's cleanup"


def test_a_search_manifest_lands_with_its_own_output_not_at_the_end(
        ds, tmp_path, monkeypatch):
    """A merge that dies on search N must not leave 1..N-1 misdescribed.

    Collecting every entry and writing the manifests at the end meant the
    earlier searches had FRESHLY REWRITTEN parquets and their PREVIOUS merge's
    manifests. Measured two ways: a forced re-merge left alpha's parquet stamped
    `merge_forced=true` beside a manifest saying nothing of the kind, and a
    re-merge after a third rank landed left alpha's parquet carrying
    `num_jobs=3` beside a manifest still claiming `partials: 2`. The parquet
    self-describes correctly in both, so ground truth is intact -- but the
    record contradicts it, and the record is the stale one.
    """
    import nova_bf.merge as merge_mod

    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)
    run_merge(cfg)                                   # a clean baseline

    alpha_man = out / manifest_name(cfg, "merge", search="alpha")
    before = json.loads(alpha_man.read_text())
    assert before.get("merge_forced") is None

    real_reduce = merge_mod._reduce

    def reduce_then_die(cfg_, spec, *a, **kw):
        entry = real_reduce(cfg_, spec, *a, **kw)
        if spec.name == "beta":
            raise RuntimeError("injected failure on the second search")
        return entry

    monkeypatch.setattr(merge_mod, "_reduce", reduce_then_die)
    monkeypatch.setenv("NOVA_BF_MERGE_FORCE", "1")
    with pytest.raises(RuntimeError, match="injected failure"):
        run_merge(cfg)

    after = json.loads(alpha_man.read_text())
    assert after.get("merge_forced") is True, (
        "alpha's parquet was rewritten by a forced merge but its manifest is "
        "still the previous merge's, claiming verified ground truth")
    assert after["started_at"] != before["started_at"]


def test_a_jobs_fanout_retires_the_legacy_run_level_manifest(ds, tmp_path,
                                                             monkeypatch):
    """Every child is a subset merge, so none of them may drop it.

    That left a fan-out unable to complete the migration off a prefix an older
    build merged: the legacy file sat beside N fresh per-search manifests
    describing the same parquets. The parent does it, after the children
    succeed and only when they covered every search.
    """
    import nova_bf.cli as cli_mod

    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)
    legacy = _plant_legacy(cfg, out)
    path = _write_cfg(cfg, tmp_path / "cfg.yaml")

    assert cli_mod._merge_fanout(path, ["alpha", "beta"], 2) == 0
    assert not legacy.exists(), "the fan-out left the legacy manifest behind"
    for name in ("alpha", "beta"):
        assert (out / manifest_name(cfg, "merge", search=name)).exists()


def test_a_partial_fanout_leaves_the_legacy_manifest_alone(ds, tmp_path):
    """Covering only some searches, it is still the others' only record."""
    import nova_bf.cli as cli_mod

    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)
    legacy = _plant_legacy(cfg, out)
    body = legacy.read_text()
    path = _write_cfg(cfg, tmp_path / "cfg.yaml")

    assert cli_mod._merge_fanout(path, ["alpha"], 2) == 0
    assert legacy.exists() and legacy.read_text() == body


def test_the_legacy_manifest_survives_if_it_still_records_a_dropped_search(
        ds, tmp_path):
    """"This merge replaced every output it claimed" is only true while the
    config still covers what the legacy file described.

    Drop a search from the config -- because it is being recomputed, say -- and
    a whole-run merge no longer touches its parquet, which was then left on the
    prefix with no manifest of any kind. The `--search` path is guarded against
    this by construction; this is the same guard for the whole-run path.
    """
    out = tmp_path / "out"
    out.mkdir()
    cfg2 = _two_search_cfg(ds, out)
    _run(cfg2, 2)
    run_merge(cfg2)                                  # alpha + beta on disk

    legacy = _plant_legacy(cfg2, out)

    # beta is dropped from the config; a whole-run merge now covers alpha only.
    cfg1 = _two_search_cfg(ds, out)
    cfg1.searches = [cfg1.searches[0]]
    run_merge(cfg1)
    assert legacy.exists(), (
        "the legacy manifest was dropped although beta's parquet is still on "
        "disk and it is that parquet's only record")

    # Once beta's output is gone too, nothing needs it.
    (out / result_name(cfg2, cfg2.searches[1])).unlink()
    run_merge(cfg1)
    assert not legacy.exists()


def test_the_legacy_manifest_survives_a_k_change_that_orphans_a_parquet(
        ds, tmp_path):
    """The guard is keyed on FILES, not search names.

    `result_name` is `bf_<stem>_<name>_k<K>.parquet`, so bumping `k` -- a
    routine config edit -- leaves the old parquet untouched on the prefix while
    the per-search manifest (which has no `k` in its name) is overwritten by the
    new one. Keyed on names, the guard saw "alpha is in `written`, this run
    rewrote it" and dropped the legacy file, orphaning the k=4 parquet: the very
    state the guard exists to prevent, reached without dropping any search.
    """
    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)
    run_merge(cfg)
    old_alpha = out / result_name(cfg, cfg.searches[0])
    assert old_alpha.exists()

    legacy = _plant_legacy(cfg, out)

    bigger = _two_search_cfg(ds, out)
    for spec in bigger.searches:
        spec.k = cfg.searches[0].k + 1
    _run(bigger, 2)
    run_merge(bigger)

    assert old_alpha.exists(), "fixture: the old-k parquet should still be here"
    assert legacy.exists(), (
        "the legacy manifest was dropped although the previous k's parquets "
        "are still on disk with nothing else describing them")


@pytest.mark.parametrize("body", [
    '{"searches": ["alpha"]}',              # a list of strings
    '{"searches": "alpha"}',                # a string
    '{"searches": 7}',                      # a number
    '[{"name": "alpha"}]',                  # the doc is a list
    '{"searches": [{"output_file": {"a": 1}}]}',   # unhashable output_file
    'not json at all',
    '',
])
def test_a_malformed_legacy_manifest_never_fails_a_finished_merge(
        ds, tmp_path, body):
    """The legacy file is read AFTER every parquet and manifest has landed.

    Anything that raises while inspecting it turns a wholly successful merge
    into a non-zero exit, and the outputs are already on disk by then. Each of
    these shapes raised out of `run_merge` when the iteration sat outside the
    read's `try` -- `AttributeError` on a list of strings, `TypeError:
    unhashable type` on a dict `output_file`.
    """
    out = tmp_path / "out"
    out.mkdir()
    cfg = _two_search_cfg(ds, out)
    _run(cfg, 2)

    legacy = out / manifest_name(cfg, "merge")
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(body)

    paths = run_merge(cfg)                       # must not raise
    assert set(paths) == {"alpha", "beta"}
    for name in ("alpha", "beta"):
        assert (out / manifest_name(cfg, "merge", search=name)).exists()
