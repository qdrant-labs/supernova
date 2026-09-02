"""Decoding a search's top-K into `hit_ids` / `hit_scores` / `hit_tie`.

This is the one place a per-element Python cost gets multiplied by n_q * k, so
`run_compute` resolves a whole `(n_q, k)` block at once: one Arrow `take` for
the ids, list OFFSETS for the ragged per-query lengths, and — when a numeric id
column needs a tie ordinate — the SAME taken values reused rather than every
element resolved a second time.

The risk in going vectorized is drift: the block form has to mean exactly what
the per-element form meant. `tiebreak.id_order_scalar` stays the authority for
`tiebreak='id'`, and the first test here pins `id_ordinals` against it over the
cases that actually differ (nulls, zero, negatives, the ends of the uint64
range). The rest check the ragged shape, which is where an off-by-one in the
offsets would put one query's hits on another query's row.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from nova_bf.results import build_result_table
from nova_bf.tiebreak import id_order_array, id_order_scalar


@pytest.mark.parametrize("unsigned", [False, True])
def test_id_order_array_matches_the_scalar_rule(unsigned):
    """The vectorized image must equal the scalar rule element for element —
    including the null sentinel and the sign-bit flip that maps the whole
    uint64 range onto int64 in order."""
    if unsigned:
        vals = [0, 1, 2, 2**63 - 1, 2**63, 2**63 + 1, 2**64 - 1, None]
        arr = pa.array(vals, type=pa.uint64())
    else:
        vals = [0, 1, -1, 2**63 - 1, -(2**63), 12345, None]
        arr = pa.array(vals, type=pa.int64())

    got = id_order_array(arr, unsigned).to_pylist()
    want = [id_order_scalar(v, unsigned) for v in vals]
    assert got == want


def test_id_order_array_preserves_order(unsigned=True):
    """The point of the ordinate is that ascending int64 order IS id order.
    A mapping that lost that would reorder ties in `merge` while looking
    perfectly well-formed."""
    vals = [0, 1, 2**63 - 1, 2**63, 2**64 - 1]
    got = id_order_array(pa.array(vals, type=pa.uint64()), True).to_pylist()
    assert got == sorted(got), got


def _ragged(rows):
    """rows -> (offsets, flat values) the way the decoder builds them."""
    counts = np.array([len(r) for r in rows], dtype=np.int64)
    offsets = pa.array(np.concatenate(([0], np.cumsum(counts))).astype(np.int32),
                       type=pa.int32())
    flat = [v for r in rows for v in r]
    return offsets, flat


def test_ragged_offsets_keep_each_query_s_hits_together():
    """Variable hit counts are the normal case (a short top-K, or a filter with
    fewer than k matches). An off-by-one here silently attributes one query's
    documents to another — well-formed output, wrong answer."""
    rows = [["a", "b", "c"], [], ["d"], ["e", "f"]]
    offsets, flat = _ragged(rows)
    arr = pa.ListArray.from_arrays(offsets, pa.array(flat, type=pa.string()))
    assert arr.to_pylist() == rows


def test_build_result_table_accepts_prebuilt_arrow_and_lists():
    """`compute` hands in ready-built ListArrays; `merge` and older callers
    hand in lists of lists. Both must produce the identical table, or the two
    producers would write subtly different parquet for the same data."""
    rows_ids = [["a", "b"], ["c"]]
    rows_sc = [[1.0, 2.0], [3.0]]
    rows_tie = [[10, 20], [30]]
    qids, payload = ["q0", "q1"], {}

    off_i, flat_i = _ragged(rows_ids)
    off_s, flat_s = _ragged(rows_sc)
    off_t, flat_t = _ragged(rows_tie)
    arrow = build_result_table(
        qids, payload,
        pa.ListArray.from_arrays(off_i, pa.array(flat_i, type=pa.string())),
        pa.ListArray.from_arrays(off_s, pa.array(flat_s, type=pa.float32())),
        hit_tie=pa.ListArray.from_arrays(off_t, pa.array(flat_t, type=pa.int64())),
    )
    lists = build_result_table(qids, payload, rows_ids, rows_sc, hit_tie=rows_tie)

    assert arrow.schema == lists.schema
    assert arrow.to_pydict() == lists.to_pydict()


def test_take_reproduces_per_element_lookup():
    """The flat-index remap `base[gidx] + row` must address the same element
    the old `corpus_ids[gidx][row]` did, for every file including gaps in the
    gidx sequence (a filter can drop whole files)."""
    per_file = {0: pa.array(["a0", "a1", "a2"]), 2: pa.array(["c0", "c1"])}
    MAX = 1000

    gidxs = sorted(per_file)
    arrays = [per_file[g] for g in gidxs]
    lens = np.fromiter((len(a) for a in arrays), dtype=np.int64, count=len(arrays))
    base = np.zeros(max(gidxs) + 1, dtype=np.int64)
    base[np.asarray(gidxs)] = np.concatenate(([0], np.cumsum(lens)[:-1]))
    values = pa.concat_arrays(arrays)

    enc = np.array([0 * MAX + 2, 2 * MAX + 0, 2 * MAX + 1, 0 * MAX + 0], dtype=np.int64)
    flat = base[enc // MAX] + (enc % MAX)
    got = values.take(pa.array(flat)).to_pylist()
    want = [str(per_file[int(e) // MAX][int(e) % MAX].as_py()) for e in enc]
    assert got == want == ["a2", "c0", "c1", "a0"]


def test_a_null_id_stringifies_to_None():
    """The old decoder built ids with `str(scalar.as_py())`, so a null id
    became the literal string "None" rather than a null. The block decoder
    casts instead, which would leave a real null — a different ground truth.
    Pinned here because it is easy to "fix" while reading the new code and not
    notice it changes output. See test_tiebreak_determinism.py::
    test_a_null_id_is_still_an_ordinary_row_under_ordinal for the end-to-end
    version."""
    ids = pa.array(["b", None])
    assert ids.cast(pa.string()).fill_null("None").to_pylist() == ["b", "None"]
    assert [str(v.as_py()) for v in ids] == ["b", "None"]


# ---------------------------------------------------------------------------
# 64-bit offsets on the id column
# ---------------------------------------------------------------------------
# A `string` array's 32-bit offsets cap its CHARACTER data at 2 GiB. One
# search's decode is n_q*k ids: the production dense search is 100k queries at
# k=1000 = 1e8 ids of ~47 bytes, i.e. ~4.7 GB. `Array.take` does not raise on
# that -- it returns an array whose offsets have wrapped NEGATIVE, and nothing
# notices until the parquet writer dereferences them and the process dies of
# SIGSEGV, after the entire corpus scan has been paid for and with no output
# written at all. (Observed on an A10G: 8 files scanned, `bf-bench` logged,
# then signal 11 in `pyarrow.parquet.write_table`; the commit before the
# vectorized decode wrote all four partials fine.)
#
# Reproducing the overflow itself would mean materializing >2 GiB of ids in a
# unit test, so these pin the property that prevents it instead: the id column
# reaches parquet with 64-bit offsets. If anything casts back down to
# `pa.string()`, these fail.

def _dense_run(tmp_path, *, id_column):
    """A minimal end-to-end compute, returning the written parquet path."""
    import pyarrow.parquet as pq
    from nova_bf.compute import run_compute
    from nova_bf.config import (
        BruteForceConfig, CorpusConfig, OutputConfig, ParamsConfig,
        QueriesConfig, SearchSpec,
    )

    vec = [1.0, 0.0, 0.0, 0.0]
    cdir = tmp_path / "c"
    cdir.mkdir()
    cols = {"dense_embedding": pa.array([vec] * 6, pa.list_(pa.float32()))}
    if id_column:
        cols["sid"] = pa.array([f"id{i}" for i in range(6)])
    pq.write_table(pa.table(cols), str(cdir / "f0.parquet"))
    pq.write_table(
        pa.table({"dense_embedding": pa.array([vec], pa.list_(pa.float32())),
                  "qid": pa.array(["q0"])}),
        str(tmp_path / "q.parquet"),
    )
    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column=id_column),
        queries=QueriesConfig(path=str(tmp_path / "q.parquet"), id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1),
        searches=[SearchSpec(name="t", k=3, metric="dot")],
    )
    return pq.read_schema(run_compute(cfg)["t"])


def test_hit_ids_use_64_bit_offsets_with_a_corpus_id_column():
    import tempfile
    import pathlib

    with tempfile.TemporaryDirectory() as d:
        schema = _dense_run(pathlib.Path(d), id_column="sid")
    assert schema.field("hit_ids").type == pa.list_(pa.large_string()), (
        "hit_ids must carry 64-bit offsets: at 1e8 hits a 32-bit `string` child "
        "silently wraps and segfaults the parquet writer"
    )


def test_hit_ids_use_64_bit_offsets_for_synthesized_ids():
    """The `make_point_id` branch has the same ceiling: 1e8 x 36-byte UUIDs is
    3.6 GB, also past 2 GiB."""
    import tempfile
    import pathlib

    with tempfile.TemporaryDirectory() as d:
        schema = _dense_run(pathlib.Path(d), id_column=None)
    assert schema.field("hit_ids").type == pa.list_(pa.large_string())


# ---------------------------------------------------------------------------
# the merged artifact must keep the partials' 64-bit id offsets
# ---------------------------------------------------------------------------


def test_merged_hit_ids_keep_64_bit_offsets():
    """`compute` writes `list<large_string>` partials, but `merge` rebuilt the
    ids as plain `string` — a 32-bit-offset child that caps one batch at 2 GiB
    of characters (~44 M ids at 47 B) and silently narrowed the schema between
    the partials and the artifact they merge into."""
    import pathlib
    import tempfile

    import numpy as np
    import pyarrow.parquet as pq
    from nova_bf.compute import run_compute
    from nova_bf.merge import run_merge

    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        cfg = _cfg_for_merge(tmp)
        for r in range(2):
            run_compute(cfg, num_jobs=2, job_rank=r)
        merged = run_merge(cfg)["t"]
        part_dir = pathlib.Path(str(tmp / "out"))
        parts = sorted(part_dir.rglob("*.parquet"))
        merged_schema = pq.read_schema(merged)

    assert merged_schema.field("hit_ids").type == pa.list_(pa.large_string()), (
        "merge narrowed hit_ids back to 32-bit offsets")
    assert parts, "no partials were written"


def _cfg_for_merge(tmp):
    import numpy as np
    import pyarrow as _pa
    import pyarrow.parquet as _pq
    from nova_bf.config import (
        BruteForceConfig, CorpusConfig, OutputConfig, ParamsConfig,
        QueriesConfig, SearchSpec,
    )

    cdir = tmp / "c"
    cdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    for f in range(4):
        v = rng.normal(size=(25, 4)).astype(np.float32)
        _pq.write_table(_pa.table({
            "dense_embedding": _pa.array(v.tolist(), _pa.list_(_pa.float32())),
            "sid": _pa.array([f"c{f}_{j}" for j in range(25)]),
        }), str(cdir / f"f{f}.parquet"))
    qv = rng.normal(size=(3, 4)).astype(np.float32)
    qp = tmp / "q.parquet"
    _pq.write_table(_pa.table({
        "dense_embedding": _pa.array(qv.tolist(), _pa.list_(_pa.float32())),
        "qid": _pa.array(["q0", "q1", "q2"]),
    }), str(qp))
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="sid"),
        queries=QueriesConfig(path=str(qp), id_column="qid"),
        output=OutputConfig(path=str(tmp / "out")),
        params=ParamsConfig(io_workers=1),
        searches=[SearchSpec(name="t", k=3, metric="dot")],
    )


def test_explicit_merge_batch_size_is_clamped_not_obeyed_blindly(caplog):
    """`_topk_merge` allocates `(rows, n_partials * k)` grids, so the knob is
    quadratic in the fan-in. An explicit value past the candidate-slot ceiling
    is clamped — and says so — rather than turning into a host OOM."""
    import logging

    from nova_bf.merge import _TARGET_CANDIDATE_SLOTS, _resolve_batch_rows

    n_partials, k, n_rows = 64, 1000, 1_000_000
    ceiling = _TARGET_CANDIDATE_SLOTS // (n_partials * k)

    with caplog.at_level(logging.WARNING, logger="nova_bf.merge"):
        got = _resolve_batch_rows(50_000, n_rows, n_partials, k)
    assert got == ceiling, f"expected clamp to {ceiling}, got {got}"
    assert "clamping" in caplog.text, "the clamp must be logged"

    # under the ceiling: honoured exactly, and silent
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="nova_bf.merge"):
        assert _resolve_batch_rows(10, n_rows, n_partials, k) == 10
    assert "clamping" not in caplog.text

    # auto path unchanged
    assert _resolve_batch_rows(None, n_rows, n_partials, k) == ceiling
    # a tiny corpus still bounds by n_rows
    assert _resolve_batch_rows(None, 5, n_partials, k) == 5
    # never zero
    assert _resolve_batch_rows(None, 1_000_000, 100_000, 100_000) == 1
