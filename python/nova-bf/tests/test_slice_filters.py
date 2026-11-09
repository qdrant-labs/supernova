"""`_SliceFilters`: the per-slice source of every per-query cell mask and of the
filter liveness derived from it.

Two layers:

* unit tests with a fake mask builder pin memoisation, release, the `select`
  contract and the tile arithmetic (partial last tile, exact tile boundaries);
* an end-to-end test runs `run_compute` on a tiny corpus and checks, at EVERY
  slice and member, that `row_live` / `tile_live` equal liveness recomputed
  independently from `filters.evaluate` over the true corpus rows of that slice
  -- with and without union compaction (`orig_rows`), with slices that start
  off a byte boundary, for a CPU text filter, a GPU-eligible filter and a
  uniform filter, under `rows:` selectors.

Liveness must be EXACT in the safe direction: a row or tile reported dead has no
allowed cell. The end-to-end oracle checks exact equality, which covers both
directions.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from nova_bf import compute
from nova_bf.compute import _SliceFilters, _tile_any, run_compute, _to_query_array
from nova_bf.config import (
    BruteForceConfig, CorpusConfig, Filter, FilterCondition, OutputConfig,
    QueriesConfig, RowSelector, SearchSpec,
)
from nova_bf.filters import PackedRowMask, evaluate


# --- unit layer ---------------------------------------------------------------

F_A = Filter(must=[FilterCondition(field="tenant", match_from_query="tenant")])
F_B = Filter(must=[FilterCondition(field="score", range_from_query={"gte": "lo"})])
F_U = Filter(must=[FilterCondition(field="tenant", match="A")])


def _spec(name, f):
    return SearchSpec(name=name, vector_type="dense", metric="dot", k=3, filter=f)


def _np_tiles(mask: np.ndarray, width: int) -> np.ndarray:
    """(r, n) -> (n_tiles, r) oracle, written independently of `_tile_any`."""
    r, n = mask.shape
    n_tiles = -(-n // width)
    out = np.zeros((n_tiles, r), dtype=bool)
    for t in range(n_tiles):
        out[t] = mask[:, t * width:(t + 1) * width].any(axis=1)
    return out


def _make(masks, *, members, specs, qsel, qrows, heights, uniform=None, n_cols=10):
    calls = []

    def build(f, rows, true_rows):
        calls.append(f)
        m = masks[f]
        return None if m is None else torch.from_numpy(m.copy())

    def uni(f, true_rows):
        return torch.from_numpy(np.nonzero(uniform[f][true_rows])[0])

    sf = _SliceFilters(
        specs, members, torch.arange(n_cols), slice(0, n_cols), n_cols, "cpu",
        build_mask=build, uniform_idx=uni, spec_qsel=qsel, spec_qrows=qrows,
        member_height=lambda m: heights[m],
        per_query=lambda f: f in (F_A, F_B),
    )
    return sf, calls


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    mA = rng.random((6, 10)) < 0.15          # F_A: 6 filter rows, used by member 0 and 1
    mB = rng.random((4, 10)) < 0.3           # F_B: 4 filter rows, member 2 (sub-selected)
    specs = [_spec("a0", F_A), _spec("a1", F_A), _spec("b", F_B), _spec("u", F_U),
             _spec("none", None)]
    # member 1 reads rows 2..4 of F_A's mask, member 2 reads rows [3, 0, 2] of F_B's
    qsel = [None, slice(10, 13), torch.tensor([20, 21, 22]), None, None]
    qrows = [None, slice(2, 5), torch.tensor([3, 0, 2]), None, None]
    heights = [6, 3, 3, 30, 30]
    keep_u = np.zeros(10, dtype=bool)
    keep_u[[4, 9]] = True
    return dict(masks={F_A: mA, F_B: mB}, specs=specs, qsel=qsel, qrows=qrows,
                heights=heights, uniform={F_U: keep_u}, members=[0, 1, 2, 3, 4])


def test_row_and_tile_liveness_match_numpy(setup):
    sf, _ = _make(**{k: setup[k] for k in ("masks", "members", "specs", "qsel",
                                           "qrows", "heights", "uniform")})
    mA, mB = setup["masks"][F_A], setup["masks"][F_B]
    want = {0: mA, 1: mA[2:5], 2: mB[[3, 0, 2]]}
    for m, w in want.items():
        np.testing.assert_array_equal(sf.row_live(m).numpy(), w.any(axis=1))
        for width in (1, 3, 4, 7, 10, 15):
            got = sf.tile_live(m, width)
            assert got.shape == (-(-10 // width), w.shape[0])
            np.testing.assert_array_equal(got.numpy(), _np_tiles(w, width))
    assert sf.row_live(4) is None and sf.tile_live(4, 3) is None      # unfiltered
    assert sf.row_live(3) is None                                     # uniform, some cols
    np.testing.assert_array_equal(                                    # uniform tiles
        sf.tile_live(3, 4).numpy(),
        np.repeat(np.array([[False], [True], [True]]), 30, axis=1))


def test_empty_mask_is_dead_everywhere(setup):
    setup["masks"][F_B] = None
    sf, _ = _make(**{k: setup[k] for k in ("masks", "members", "specs", "qsel",
                                           "qrows", "heights", "uniform")})
    assert not sf.row_live(2).any() and sf.row_live(2).shape == (3,)
    assert sf.tile_live(2, 4).shape == (3, 3) and not sf.tile_live(2, 4).any()
    assert sf.select(2) == (None, None, None)


def test_uniform_with_no_column_is_dead(setup):
    setup["uniform"][F_U][:] = False
    sf, _ = _make(**{k: setup[k] for k in ("masks", "members", "specs", "qsel",
                                           "qrows", "heights", "uniform")})
    assert sf.row_live(3).shape == (30,) and not sf.row_live(3).any()
    assert not sf.tile_live(3, 4).any()
    assert sf.select(3) == (None, None, None)


def test_select_contract_and_memoisation(setup):
    sf, calls = _make(**{k: setup[k] for k in ("masks", "members", "specs", "qsel",
                                               "qrows", "heights", "uniform")})
    # liveness first (what the two-pass / compaction steps will do), then select
    sf.row_live(0); sf.tile_live(1, 3); sf.row_live(2)
    r0 = sf.select(0)
    r1 = sf.select(1)
    assert r0[2] is r1[2]                     # one tensor backs both F_A members
    np.testing.assert_array_equal(r0[2].numpy(), setup["masks"][F_A])  # FILTER rows
    assert r0[1] is None and torch.equal(r0[0], torch.arange(10))
    sf.select(2)
    assert calls == [F_A, F_B] and sf.builds == 2   # built once per filter
    assert sf._masks == {}                    # released after the last select
    rows, cols, cm = sf.select(3)             # uniform: column selection
    np.testing.assert_array_equal(cols.numpy(), [4, 9])
    np.testing.assert_array_equal(rows.numpy(), [4, 9])
    assert cm is None
    assert sf.select(4) == (sf._rows, None, None)   # unfiltered


def test_tile_boundaries_are_exact():
    """A single allowed cell at each column: exactly its tile is live."""
    n = 11
    for width in (1, 2, 3, 4, 5, 11, 12):
        for c in range(n):
            m = torch.zeros((1, n), dtype=torch.bool)
            m[0, c] = True
            got = _tile_any(m, width, -(-n // width))[0]
            want = torch.zeros(-(-n // width), dtype=torch.bool)
            want[c // width] = True
            assert torch.equal(got, want), (width, c)


def test_bad_width_and_guard():
    sf = _SliceFilters([], [], torch.arange(3), slice(0, 3), 3, "cpu",
                       build_mask=None, uniform_idx=None, spec_qsel=[],
                       spec_qrows=[], member_height=None, per_query=lambda f: False)
    with pytest.raises(ValueError):
        sf.tile_live(0, 0)
    batch = compute.DenseCorpusBatch(np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError, match="exactly one"):       # both
        compute._process_batch_group(
            batch, [], [], [], [], [], [], [], None, 0, "cpu", None,
            lambda *a: None, [], [], [], slice_filters=lambda *a: None)
    with pytest.raises(ValueError, match="exactly one"):       # neither
        compute._process_batch_group(
            batch, [], [], [], [], [], [], [], None, 0, "cpu", None,
            None, [], [], [])


def test_release_frees_a_skipped_members_mask(setup):
    """A member skipped before masking calls `release`; the shared mask is then
    dropped after the remaining member, not held to the end of the slice."""
    sf, calls = _make(**{k: setup[k] for k in ("masks", "members", "specs", "qsel",
                                               "qrows", "heights", "uniform")})
    sf.row_live(0)
    sf.row_live(1)
    sf.release(0)                             # member 0 skipped
    assert F_A in sf._masks                   # member 1 still to come
    sf.select(1)
    assert F_A not in sf._masks and calls.count(F_A) == 1
    sf.release(4)                             # unfiltered: a no-op
    sf.release(3)                             # uniform: a no-op


# --- end-to-end layer ---------------------------------------------------------

N, D = 23, 8          # 23 rows, dense_batch_size 5 -> partial slices, off-byte starts
TENANTS = ["A", "B", "C", "A", "B", "A", "C", "C", "B", "A", "A", "B",
           "C", "A", "B", "B", "A", "C", "A", "B", "C", "A", "B"]
WORDS = ["red", "blue", "green", "red blue", "blue", "green red", "x", "red",
         "blue green", "x", "red", "green", "blue", "red x", "x", "green",
         "blue red", "x", "green", "red", "blue", "x", "green blue"]


def _dataset(tmp_path):
    rng = np.random.default_rng(1)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    pq.write_table(pa.table({
        "id": [f"d{i}" for i in range(N)],
        "dense": pa.array(rng.standard_normal((N, D)).astype(np.float32).tolist(),
                          pa.list_(pa.float32())),
        "tenant": TENANTS,
        "score": rng.random(N).round(3).tolist(),
        "text": WORDS,
    }), cdir / "part0.parquet")
    nq = 9
    qpath = tmp_path / "q.parquet"
    pq.write_table(pa.table({
        "qid": [f"q{i}" for i in range(nq)],
        "dense": pa.array(rng.standard_normal((nq, D)).astype(np.float32).tolist(),
                          pa.list_(pa.float32())),
        "grp": ["t", "s", "t", "s", "t", "s", "u", "u", "t"],
        # No query names tenant C (the GPU filter's row union is built from EVERY
        # query's values, see `_row_union_from_gpu_leaves`), so tenant-C rows the
        # text queries reject are outside every filter -> union compaction.
        "tenant": ["A", "B", "Z", "Z", "Z", "B", "A", "B", "B"],
        "lo": [0.1, 0.5, 0.9, 0.6, 0.3, 0.7, 0.2, 0.4, 0.6],
        "kw": ["red", "green", "x", "x", "nothing", "red", "x", "blue", "green"],
    }), qpath)
    return str(cdir), str(qpath)


F_TEXT = Filter(must=[FilterCondition(field="text", match_text_from_query="kw")])
F_GPU = Filter(must=[FilterCondition(field="tenant", match_from_query="tenant"),
                     FilterCondition(field="score", range_from_query={"gte": "lo"})])
F_STATIC = Filter(must=[FilterCondition(field="tenant", match="A")])


@pytest.mark.parametrize("with_baseline", [False, True])
def test_liveness_matches_independent_masks_end_to_end(tmp_path, monkeypatch, with_baseline):
    cdir, qpath = _dataset(tmp_path)
    # Each per-query filter is SHARED: the text filter by two specs over
    # different `rows:` (so its mask spans both and each member maps into it
    # with a non-identity `spec_qrows`), the GPU filter by a `rows:` spec and a
    # spec with no `rows:` (so the filter's rows are ALL queries).
    searches = [
        SearchSpec(name="text_t", vector_type="dense", metric="dot", k=4, filter=F_TEXT,
                   rows=RowSelector(column="grp", isin=["t"])),
        SearchSpec(name="text_s", vector_type="dense", metric="dot", k=4, filter=F_TEXT,
                   rows=RowSelector(column="grp", isin=["s"])),
        SearchSpec(name="gpu", vector_type="dense", metric="dot", k=4, filter=F_GPU,
                   rows=RowSelector(column="grp", isin=["s"])),
        SearchSpec(name="gpu_all", vector_type="dense", metric="dot", k=4, filter=F_GPU),
        SearchSpec(name="static", vector_type="dense", metric="dot", k=4, filter=F_STATIC,
                   rows=RowSelector(column="grp", isin=["u"])),
    ]
    if with_baseline:   # an unfiltered member -> no union compaction, slice true_rows
        searches.append(SearchSpec(name="all", vector_type="dense", metric="dot", k=4))
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=cdir, id_column="id", dense_column="dense"),
        queries=QueriesConfig(path=qpath, id_column="qid", dense_column="dense",
                              payload_fields=["grp", "tenant", "lo", "kw"]),
        output=OutputConfig(path=str(tmp_path / "out")),
        searches=searches,
        params={"dense_batch_size": 5, "io_workers": 1, "threshold_exchange": False},
    )

    seen = []

    class Rec(_SliceFilters):
        __slots__ = ()

        def select(self, m):
            tr = self._true_rows
            cols = (np.arange(tr.start, tr.stop) if isinstance(tr, slice)
                    else np.asarray(tr).copy())
            rl = self.row_live(m)
            tl = {w: self.tile_live(m, w) for w in (1, 2, 3)}
            seen.append((m, cols, None if rl is None else rl.clone(),
                         {w: None if t is None else t.clone() for w, t in tl.items()},
                         isinstance(tr, slice)))
            return super().select(m)

    monkeypatch.setattr(compute, "_SliceFilters", Rec)
    run_compute(cfg)

    # Independent oracle: the CPU evaluator over the whole file and ALL queries
    # (a query's allowed rows do not depend on the other queries), then each
    # member's OWN query rows -- independent of how nova-bf composes filter
    # rows and maps members into them.
    table = pq.read_table(f"{cdir}/part0.parquet")
    qt = pq.read_table(qpath)
    grp = np.asarray(qt["grp"].to_pylist())
    qv_all = {c: _to_query_array(qt[c].to_pylist()) for c in ("tenant", "lo", "kw")}
    expected = {}
    for m, s in enumerate(searches):
        if s.filter is None:
            continue
        own = (np.arange(len(grp)) if s.rows is None
               else np.nonzero(np.isin(grp, s.rows.isin))[0])
        got = evaluate(s.filter, table, qv_all)
        if isinstance(got, PackedRowMask):
            expected[m] = np.unpackbits(got.packed, axis=1, count=N).astype(bool)[own]
        else:
            expected[m] = np.broadcast_to(got, (len(own), N))

    checked = 0
    saw_compacted = saw_offbyte = False
    kinds = set()
    for m, cols, rl, tl, as_slice in seen:
        # Contiguous-rows fast path: a slice whose rows are one contiguous run
        # (including a contiguous run inside a union-compacted batch) must be
        # handed on as a `slice`; only a genuinely scattered one as an array.
        contiguous = np.array_equal(cols, np.arange(cols[0], cols[0] + len(cols)))
        assert as_slice == contiguous, (cols, as_slice)
        kinds.add(as_slice)
        s = searches[m]
        if s.filter is None:
            assert rl is None
            continue
        mask = expected[m][:, cols]
        if s.filter == F_STATIC:
            if mask.any():
                assert rl is None
            else:
                assert not rl.any()
        else:
            np.testing.assert_array_equal(rl.cpu().numpy(), mask.any(axis=1))
        for w, t in tl.items():
            np.testing.assert_array_equal(t.cpu().numpy(), _np_tiles(mask, w))
            assert t.is_contiguous()
        checked += 1
        saw_compacted |= not np.array_equal(cols, np.arange(cols[0], cols[0] + len(cols)))
        saw_offbyte |= bool(cols[0] % 8)
    assert checked >= 3
    assert saw_offbyte
    if not with_baseline:
        assert saw_compacted, "expected union compaction to give non-contiguous rows"
        # Both paths exercised inside one compacted batch: contiguous slices
        # take the packed-slice path, scattered ones the bit gather.
        assert kinds == {True, False}
    else:
        assert kinds == {True}
