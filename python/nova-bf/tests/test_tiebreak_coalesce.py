"""The coalesced path, where one batch's rows come from SEVERAL files.

When a filter leaves few rows per file and a batch size is configured, files are
concatenated into one larger GPU call. That breaks the two assumptions the
cheap ordinal derivation rests on — a row's position in the batch is neither its
position in its file nor its position in the worker — so this path has to be
handed ordinals explicitly, and it is the one most likely to mis-align them.

It also turned out to be where a zero-row file crashed the run outright, which
is checked here too.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")

import nova_bf.compute as cp
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

VEC = [1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def spy(monkeypatch):
    """Records the ordinals handed to every COALESCED call (a coalesced batch is
    the one that gets explicit ordinals with `orig_rows=None`)."""
    seen: list[list[int]] = []
    orig = cp._process_shared_batch

    def wrapped(*a, **kw):
        if kw.get("ordinal_row_ids") is not None and kw.get("orig_rows") is None:
            seen.append(np.asarray(kw["ordinal_row_ids"]).tolist())
        return orig(*a, **kw)

    monkeypatch.setattr(cp, "_process_shared_batch", wrapped)
    return seen


def _corpus(tmp, groups):
    cdir = tmp / "corpus"
    cdir.mkdir(exist_ok=True)
    for fi, ids in enumerate(groups):
        pq.write_table(
            pa.table({
                "dense_embedding": pa.array([VEC] * len(ids), pa.list_(pa.float32())),
                "sid": pa.array(ids, pa.string()),
                "lang": pa.array(["eng"] * len(ids), pa.string()),
            }),
            str(cdir / f"f{fi}.parquet"),
        )
    pq.write_table(
        pa.table({"dense_embedding": pa.array([VEC], pa.list_(pa.float32())),
                  "qid": pa.array(["q0"])}),
        str(tmp / "q.parquet"),
    )
    return cdir


def _run(cdir, tmp, tag, tiebreak, k=4, batch=6, n_jobs=None):
    out = tmp / tag
    out.mkdir(exist_ok=True)
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="sid"),
        queries=QueriesConfig(path=str(tmp / "q.parquet"), id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, tiebreak=tiebreak, dense_batch_size=batch),
        searches=[SearchSpec(name="t", k=k, metric="dot",
                             filter=Filter(must=[FilterCondition(field="lang", match="eng")]))],
    )
    if n_jobs is None:
        t = pq.read_table(run_compute(cfg)["t"]).to_pydict()
    else:
        for r in range(n_jobs):
            run_compute(cfg, num_jobs=n_jobs, job_rank=r)
        t = pq.read_table(run_merge(cfg)["t"]).to_pydict()
    return dict(zip(t["query_id"], t["hit_ids"]))["q0"]


GROUPS = [["a9", "a8"], ["a7", "a6"], ["a5", "a4"], ["a3", "a2"]]
FLAT = [i for g in GROUPS for i in g]


def test_the_coalesced_path_is_actually_reached(tmp_path, spy):
    """Guards every other test in this file from being vacuous."""
    _run(_corpus(tmp_path, GROUPS), tmp_path, "reach", "id")
    assert spy, "no coalesced batch was produced — the fixture stopped exercising it"


@pytest.mark.parametrize("tiebreak,expect_ordinals", [
    # corpus order across the 4 files, in one flat run
    ("ordinal", [[0, 1, 2, 3, 4, 5], [6, 7]]),
    # a2 is the lowest id, so the ordinals run backwards against corpus order
    ("id", [[7, 6, 5, 4, 3, 2], [1, 0]]),
])
def test_ordinals_stay_attached_to_their_rows_across_files(
    tmp_path, spy, tiebreak, expect_ordinals
):
    """The ordinals must interleave across the files in the group, and follow
    the rule — not the batch's own row order."""
    _run(_corpus(tmp_path, GROUPS), tmp_path, f"ord{tiebreak}", tiebreak)
    assert spy == expect_ordinals


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
@pytest.mark.parametrize("batch", [1, 2, 3, 5, 6, 100])
def test_the_answer_survives_every_coalesce_grouping(tmp_path, tiebreak, batch):
    """The batch size decides how many files land in one coalesced group, so it
    decides which rows meet in a pre-top-K. The answer must not follow it."""
    cdir = _corpus(tmp_path, GROUPS)
    expected = sorted(FLAT)[:4] if tiebreak == "id" else FLAT[:4]
    assert _run(cdir, tmp_path, f"g{tiebreak}{batch}", tiebreak, batch=batch) == expected


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
@pytest.mark.parametrize("n_jobs", [None, 2, 3])
def test_coalesced_and_sharded_together(tmp_path, tiebreak, n_jobs):
    cdir = _corpus(tmp_path, GROUPS)
    expected = sorted(FLAT)[:4] if tiebreak == "id" else FLAT[:4]
    assert _run(cdir, tmp_path, f"s{tiebreak}{n_jobs}", tiebreak, n_jobs=n_jobs) == expected


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
def test_a_zero_row_file_does_not_crash_a_coalesced_group(tmp_path, tiebreak):
    """Regression, and a PRE-EXISTING one: an empty shard decodes to a `(0, 0)`
    dense array with no vector width, so concatenating it against real rows
    raised `ValueError: all the input array dimensions except for the
    concatenation axis must match exactly`. Empty shards are ordinary.
    """
    groups = [["a9", "a8"], [], ["a7", "a6"], [], ["a5", "a4"], ["a3", "a2"]]
    cdir = _corpus(tmp_path, groups)
    flat = [i for g in groups for i in g]
    expected = sorted(flat)[:4] if tiebreak == "id" else flat[:4]
    assert _run(cdir, tmp_path, f"z{tiebreak}", tiebreak) == expected


def test_an_all_empty_coalesce_group_is_a_no_op(tmp_path):
    """Every file empty: the group must flush cleanly and reset, not crash and
    not leave rows buffered into the next flush."""
    cdir = _corpus(tmp_path, [[], [], ["a1"]])
    assert _run(cdir, tmp_path, "allempty", "id", k=2) == ["a1"]
