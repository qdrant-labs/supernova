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
