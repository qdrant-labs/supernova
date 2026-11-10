"""The lazy per-query text combine (`filters.ComboRowMask`) is EXACT.

`evaluate(..., combo=True)` returns the production-shape per-query text mask
as distinct phrase rows plus a per-combination AND/OR structure; `run_compute`
combines a slice of it on the scoring device (`compute._combo_mask_device`).
Everything here pins it bit-for-bit against the materialising loop
(`evaluate(..., combo=False)` -> `PackedRowMask`) it replaces:

- `to_packed()` (bytes, padding included), `union_rows()`, `any()`, query
  narrowing, over fuzzed filters of the supported shape and edge sizes;
- shapes it cannot express keep returning `PackedRowMask`;
- the device slice mask == the packed path's `build_mask` for random slices
  (unaligned, single-row, whole file) and compacted row-index arrays, on torch
  "cpu" (the same torch code a GPU runs), including chunked combines;
- `_union_keep` on either mask type;
- `run_compute` outputs are identical with `NOVA_BF_NO_GPU_COMBO=1` and
  without, with compaction on and off, several files, short lists.
"""

import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nova_bf import filters as F
from nova_bf.config import (
    BruteForceConfig, CorpusConfig, Filter, FilterCondition, OutputConfig,
    ParamsConfig, QueriesConfig, RangeCondition, RangeFromQuery, SearchSpec,
)
from nova_bf.filters import ComboRowMask, PackedRowMask, evaluate


@pytest.fixture(autouse=True)
def _no_parts_guard(monkeypatch):
    """Small test query sets have more parts than queries; the memory guard
    would send them to the reference loop. Drive the lazy path instead (the
    guard itself is tested in test_combo_row_mask.py)."""
    from nova_bf import filters as _F
    monkeypatch.setattr(_F, "_COMBO_PARTS_GUARD", False)


WORDS = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "Eta", "theta"]
DOMS = ["com", "org", "edu", "gov", "io", "net"]


def _rand_text(rng, words, lo=0, hi=6):
    if rng.random() < 0.08:
        return None
    return rng.choice([" ", "-", ", ", "/"]).join(
        rng.choice(words) for _ in range(rng.randint(lo, hi)))


def _rand_phrase(rng, words):
    r = rng.random()
    if r < 0.1:
        return None
    if r < 0.16:
        return rng.choice(["", "  ", "!!", "-"])          # token-less
    if r < 0.22:
        return "zzznomatch" + rng.choice(["", " alpha"])  # token absent from the file
    return " ".join(rng.choice(words) for _ in range(rng.randint(1, 3)))


def _table(rng, n):
    return pa.table({
        "text": pa.array([_rand_text(rng, WORDS) for _ in range(n)], type=pa.string()),
        "url": pa.array([_rand_text(rng, DOMS, 1, 3) for _ in range(n)], type=pa.string()),
        "cat": pa.array([rng.choice(["a", "b", None]) for _ in range(n)], type=pa.string()),
        "cost": pa.array([None if rng.random() < 0.1 else rng.uniform(0, 10)
                          for _ in range(n)], type=pa.float64()),
    })


def _qv(rng, nq, same_combo=False):
    cols = {}
    for c in ("kw", "kw2", "neg"):
        cols[c] = [_rand_phrase(rng, WORDS) for _ in range(nq)]
    for i in range(4):
        cols[f"d{i}"] = [_rand_phrase(rng, DOMS) for _ in range(nq)]
    cols["cat_q"] = [rng.choice(["a", "b", None]) for _ in range(nq)]
    cols["budget"] = [rng.uniform(0, 10) for _ in range(nq)]
    if same_combo:
        cols = {k: [v[0]] * nq for k, v in cols.items()}
    return {k: np.array(v, dtype=object if k != "budget" else np.float64)
            for k, v in cols.items()}


def _static_cond(rng):
    k = rng.randrange(3)
    if k == 0:
        return FilterCondition(field="text", match_text=rng.choice(["alpha", "beta gamma"]))
    if k == 1:
        return FilterCondition(field="cat", match=rng.choice(["a", "b"]))
    return FilterCondition(field="cost", range=RangeCondition(lt=rng.uniform(2, 9)))


def _text_cond(rng):
    if rng.random() < 0.5:
        return FilterCondition(field="text", match_text_from_query=rng.choice(["kw", "kw2", "neg"]))
    return FilterCondition(field="url", match_text_from_query=f"d{rng.randrange(4)}")


def _rand_filter(rng, allow_per_query_nontext=False):
    groups = {"must": [], "should": [], "must_not": []}
    for g in groups:
        for _ in range(rng.randrange(4)):
            r = rng.random()
            if allow_per_query_nontext and r < 0.25:
                groups[g].append(rng.choice([
                    FilterCondition(field="cat", match_from_query="cat_q"),
                    FilterCondition(field="cost", range_from_query=RangeFromQuery(lt="budget")),
                ]))
            elif r < 0.75:
                groups[g].append(_text_cond(rng))
            else:
                groups[g].append(_static_cond(rng))
    if not any(c.match_text_from_query for cs in groups.values() for c in cs):
        groups[rng.choice(list(groups))].append(_text_cond(rng))
    return Filter(**{g: tuple(cs) for g, cs in groups.items()})


def _pair(filt, t, qv):
    old = evaluate(filt, t, qv, combo=False)
    new = evaluate(filt, t, qv, combo=True)
    return old, new


def _assert_same(old, new, ctx=""):
    assert isinstance(old, PackedRowMask), ctx
    got = new.to_packed() if isinstance(new, ComboRowMask) else new
    assert got.packed.shape == old.packed.shape, ctx
    assert np.array_equal(got.packed, old.packed), ctx   # padding bytes too
    assert new.shape == old.shape and new.n_queries == old.n_queries, ctx
    assert new.any() == old.any(), ctx


# ---------------------------------------------------------------------------
# 1. to_packed / union / any against the materialising loop
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(120))
def test_fuzz_combo_to_packed_is_bit_identical(seed):
    rng = random.Random(7000 + seed)
    n = rng.choice([0, 1, 7, 8, 9, 15, 16, 17, 31, 64, 65, rng.randint(1, 200)])
    nq = rng.choice([1, 2, rng.randint(1, 25)])
    t = _table(rng, n)
    qv = _qv(rng, nq, same_combo=rng.random() < 0.15)
    filt = _rand_filter(rng)
    old, new = _pair(filt, t, qv)
    ctx = f"seed={seed} n={n} nq={nq} filt={filt}"
    assert isinstance(new, ComboRowMask), ctx
    _assert_same(old, new, ctx)
    want_union = (np.unpackbits(np.bitwise_or.reduce(old.packed, axis=0), count=n).astype(bool)
                  if nq else np.zeros(n, bool))
    assert np.array_equal(new.union_rows(), want_union), ctx
    assert np.array_equal(new.unpack(), old.unpack()), ctx
    # query narrowing keeps exactness (and unused combinations drop out)
    qsel = np.array(sorted(rng.sample(range(nq), rng.randint(1, nq))))
    _assert_same(old[qsel], new[qsel], ctx + " narrowed")
    assert np.array_equal(
        new[qsel].union_rows(),
        np.unpackbits(np.bitwise_or.reduce(old[qsel].packed, axis=0), count=n).astype(bool)), ctx


@pytest.mark.parametrize("seed", range(60))
def test_raw_parts_and_device_combine_bytes(seed):
    """What the DEVICE consumes, unfiltered by to_packed()/union's re-masking:
    every `parts` row has zero padding, and the raw `combine_torch` over the
    whole width, expanded by `combo_of_query`, equals the packed bytes. Many
    queries, so union runs spanning several OR groups are exercised too."""
    torch = pytest.importorskip("torch")
    rng = random.Random(31000 + seed)
    n = rng.choice([1, 7, 9, 15, 17, 63, 65, rng.randint(1, 300)])
    nq = rng.randint(20, 80)
    t = _table(rng, n)
    qv = _qv(rng, nq)
    old, new = _pair(_rand_filter(rng), t, qv)
    if not isinstance(new, ComboRowMask):
        return
    ctx = f"seed={seed} n={n} nq={nq}"
    pad = np.uint8(~np.uint8(F._tail_mask(n)))
    assert not (new.parts[:, -1] & pad).any(), ctx + " dirty padding in parts"
    R = new.combine_torch(torch.from_numpy(new.parts))
    got = R.numpy()[new.combo_of_query]
    assert np.array_equal(got, old.packed), ctx
    want_union = np.unpackbits(np.bitwise_or.reduce(old.packed, axis=0), count=n).astype(bool)
    assert np.array_equal(new.union_rows(), want_union), ctx


def test_zero_query_and_one_query_negative_stride_narrowing():
    torch = pytest.importorskip("torch")
    # every query in ONE combination: narrowing then keeps a view of
    # combo_of_query (no unused combos to drop), the case that reached
    # torch.from_numpy with a negative stride
    rng = random.Random(5)
    t = _table(rng, 40)
    qv = _qv(rng, 3, same_combo=True)
    filt = Filter(must=[FilterCondition(field="text", match_text_from_query="kw")])
    old, new = _pair(filt, t, qv)
    assert isinstance(new, ComboRowMask) and new.and_idx.shape[0] == 1
    empty = new[np.array([], dtype=np.int64)]
    assert empty.n_queries == 0 and empty.any() is False and empty.any() == old[np.array([], dtype=np.int64)].any()
    for sl in (slice(0, 1), slice(None, None, -3), slice(2, 1, -1)):
        one = new[sl]
        assert one.n_queries == 1
        R = one.combine_torch(torch.from_numpy(one.parts))     # must not raise on stride
        assert np.array_equal(R.numpy()[one.combo_of_query], old[sl].packed)


@pytest.mark.parametrize("chunk", [1, 3, 64])
@pytest.mark.parametrize("seed", range(10))
def test_host_combine_chunking_is_exact(monkeypatch, seed, chunk):
    monkeypatch.setattr(F, "_COMBINE_CHUNK_BYTES", chunk)
    rng = random.Random(9100 + seed)
    t = _table(rng, rng.randint(1, 150))
    qv = _qv(rng, rng.randint(1, 20))
    old, new = _pair(_rand_filter(rng), t, qv)
    _assert_same(old, new, f"seed={seed}")
    assert np.array_equal(
        new.union_rows(),
        np.unpackbits(np.bitwise_or.reduce(old.packed, axis=0), count=len(t)).astype(bool))


def test_fineweb_shape_and_dead_combos():
    """must keyword + 9 url should slots: combos share OR groups; a null
    keyword is a dead combination (all-zero row)."""
    rng = random.Random(3)
    t = _table(rng, 333)
    nq = 40
    qv = {"kw": np.array([None, ""] + [_rand_phrase(rng, WORDS) for _ in range(nq - 2)], dtype=object)}
    slots = [rng.choice([DOMS[:3], DOMS[2:], DOMS]) for _ in range(nq)]
    for i in range(9):
        qv[f"s{i}"] = np.array([s[i] if i < len(s) else "zzznomatch" for s in slots], dtype=object)
    filt = Filter(
        must=[FilterCondition(field="text", match_text_from_query="kw")],
        should=[FilterCondition(field="url", match_text_from_query=f"s{i}") for i in range(9)],
    )
    old, new = _pair(filt, t, qv)
    assert isinstance(new, ComboRowMask)
    assert new.or_idx.shape[0] <= 4   # 3 slot groups + the dead combos' (ONES,)
    _assert_same(old, new)
    assert not old.packed[:2].any()          # the dead combos really are dead


def test_uniform_and_unsupported_shapes_keep_the_old_types():
    rng = random.Random(5)
    t = _table(rng, 50)
    qv = _qv(rng, 6)
    uni = Filter(must=[FilterCondition(field="text", match_text="alpha")])
    assert isinstance(evaluate(uni, t, qv, combo=True), np.ndarray)
    for seed in range(40):
        r = random.Random(seed)
        filt = _rand_filter(r, allow_per_query_nontext=True)
        old, new = _pair(filt, t, qv)
        if isinstance(new, ComboRowMask):
            _assert_same(old, new)
            continue
        assert isinstance(new, PackedRowMask)
        assert np.array_equal(new.packed, old.packed)
    mixed = Filter(must=[FilterCondition(field="text", match_text_from_query="kw"),
                         FilterCondition(field="cat", match_from_query="cat_q")])
    assert isinstance(evaluate(mixed, t, qv, combo=True), PackedRowMask)


def test_combo_usage_counts():
    F.reset_combo_usage()
    rng = random.Random(1)
    t = _table(rng, 20)
    qv = _qv(rng, 3)
    f = Filter(must=[FilterCondition(field="text", match_text_from_query="kw")])
    evaluate(f, t, qv, combo=True)
    evaluate(f, t, qv, combo=False)
    u = F.combo_usage()
    assert u["combo_masks"] == 1 and u["packed_masks"] == 1


# ---------------------------------------------------------------------------
# 2. the device slice mask == the packed path's build_mask
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")
from nova_bf import compute as C  # noqa: E402


def _old_build(pm, true_rows, device="cpu"):
    """The `PackedRowMask` branch of `_process_shared_batch.build_mask`."""
    if isinstance(true_rows, slice):
        r0, r1 = true_rows.start, true_rows.stop
        if not C._packed_slice_any(pm.packed, r0, r1 - r0):
            return None
        sub = pm.packed[:, r0 >> 3:(r1 + 7) >> 3]
        return C._unpack_row_axis_device(sub, r1 - r0, device, bit_offset=r0 & 7)
    sel = (np.uint8(1) << (7 - (true_rows & 7)).astype(np.uint8))
    bits = pm.packed[:, true_rows >> 3] & sel
    if not bits.any():
        return None
    return torch.from_numpy(np.ascontiguousarray(bits)).to(device) != 0


def _same_dev(a, b, ctx):
    assert (a is None) == (b is None), ctx
    if a is not None:
        assert a.dtype == torch.bool and b.dtype == torch.bool, ctx
        assert tuple(a.shape) == tuple(b.shape), ctx
        assert torch.equal(a, b), ctx


def _slices(rng, n):
    out = [slice(0, n), slice(0, 1), slice(n - 1, n)]
    for _ in range(12):
        r0 = rng.randrange(n)
        out.append(slice(r0, rng.randint(r0 + 1, n)))
    step = rng.randint(1, max(1, n // 3))
    out += [slice(r0, min(n, r0 + step)) for r0 in range(0, n, step)]
    return out


@pytest.mark.parametrize("window", [None, 1, 64])
@pytest.mark.parametrize("empty_check", ["auto", "device"])
@pytest.mark.parametrize("chunk", [None, 5])
@pytest.mark.parametrize("seed", range(40))
def test_device_slice_mask_matches_packed_path(monkeypatch, seed, chunk, empty_check, window):
    """`empty_check="device"` forces the device-side None decision (taken
    when the host union is not cheap); "auto" mostly takes the host one.
    `window`: `parts` held whole (None) or in column windows of at most that
    many bytes (1: one column per window), slices and rows visited in random
    order so windows are reloaded back and forth."""
    monkeypatch.setattr(C, "_RESIDENT_WINDOW_BYTES", window)
    if chunk is not None:
        monkeypatch.setattr(F, "_COMBINE_CHUNK_BYTES", chunk)
        # and the query-chunked unpack: 1..3 queries per chunk
        monkeypatch.setattr(C, "_UNPACK_CHUNK_BYTES", 1 + seed % 3 * 64)
        # and the row-chunked compacted gather: 8 or 16 rows per chunk
        monkeypatch.setattr(C, "_COMBINE_CHUNK_ROWS_BYTES", 1 + seed % 2 * 400)
    if empty_check == "device":
        monkeypatch.setattr(ComboRowMask, "union_is_cheap", lambda self: False)
    rng = random.Random(4200 + seed)
    n = rng.choice([1, 8, 9, 23, rng.randint(1, 180)])
    t = _table(rng, n)
    qv = _qv(rng, rng.randint(1, 18), same_combo=rng.random() < 0.1)
    old, new = _pair(_rand_filter(rng), t, qv)
    assert isinstance(new, ComboRowMask)
    trs = list(_slices(rng, n))
    for _ in range(8):
        k = rng.randint(0, n)
        trs.append(np.array(sorted(rng.sample(range(n), k)), dtype=np.int64))
    # unsorted and with duplicates (production rows are sorted and unique, but
    # the gather must not depend on it)
    for _ in range(3):
        trs.append(np.array([rng.randrange(n) for _ in range(rng.randint(1, 2 * n))],
                            dtype=np.int64))
    rng.shuffle(trs)
    for tr in trs:
        _same_dev(C._combo_mask_device(new, tr, "cpu"), _old_build(old, tr), f"seed={seed} {tr}")
    # narrowed query axis (a filter's own rows) still agrees
    qsel = np.arange(0, new.n_queries, 2)
    _same_dev(C._combo_mask_device(new[qsel], slice(0, n), "cpu"),
              _old_build(old[qsel], slice(0, n)), f"seed={seed} narrowed")


@pytest.mark.parametrize("window", [None, 1])
@pytest.mark.parametrize("empty_check", ["auto", "device"])
def test_device_mask_none_when_nothing_allowed(monkeypatch, empty_check, window):
    monkeypatch.setattr(C, "_RESIDENT_WINDOW_BYTES", window)
    if empty_check == "device":
        monkeypatch.setattr(ComboRowMask, "union_is_cheap", lambda self: False)
    t = pa.table({"text": pa.array(["alpha"] * 5 + ["beta"] * 11)})
    qv = {"kw": np.array(["beta", None, "gamma"], dtype=object)}
    f = Filter(must=[FilterCondition(field="text", match_text_from_query="kw")])
    old, new = _pair(f, t, qv)
    for sl in (slice(0, 5), slice(1, 4), slice(5, 6), slice(0, 6), slice(4, 16)):
        _same_dev(C._combo_mask_device(new, sl, "cpu"), _old_build(old, sl), str(sl))
    assert C._combo_mask_device(new, slice(0, 5), "cpu") is None
    assert new.union_is_cheap() == (empty_check == "auto")
    assert C._combo_mask_device(new, np.array([0, 4], dtype=np.int64), "cpu") is None
    assert C._combo_mask_device(new, np.array([], dtype=np.int64), "cpu") is None


def test_resident_parts_shared_released_and_counted():
    """Whole table: one upload per (mask, device), shared with query
    narrowings and reused by every slice and compacted batch; a copy, not a
    view of the host array; `release_device` drops it (next use re-uploads)."""
    rng = random.Random(77)
    n = 61
    t = _table(rng, n)
    qv = _qv(rng, 9)
    f = Filter(must=[FilterCondition(field="text", match_text_from_query="kw")],
               should=[FilterCondition(field="text", match_text_from_query="kw2")])
    old, new = _pair(f, t, qv)
    rows = np.arange(0, n, 3, dtype=np.int64)
    parts_before = new.parts.copy()
    F.reset_combo_usage()
    narrow = new[np.arange(0, new.n_queries, 2)]
    for cm, pm in ((new, old), (narrow, old[np.arange(0, old.n_queries, 2)])):
        for tr in (slice(0, n), slice(5, 40), rows):
            _same_dev(C._combo_mask_device(cm, tr, "cpu"), _old_build(pm, tr), str(tr))
    assert F.combo_usage()["resident_uploads"] == 1
    assert narrow._parts_dev is new._parts_dev and "cpu" in new._parts_dev
    held = new._parts_dev["cpu"]
    assert held.width == new.parts.shape[1] and held.k == 0
    assert torch.equal(held.view, torch.from_numpy(new.parts))
    assert held.view.data_ptr() != new.parts.ctypes.data
    assert np.array_equal(new.parts, parts_before)
    new.release_device()
    assert not new._parts_dev and not narrow._parts_dev and not new._dev
    _same_dev(C._combo_mask_device(narrow, rows, "cpu"),
              _old_build(old[np.arange(0, old.n_queries, 2)], rows), "after release")
    assert F.combo_usage()["resident_uploads"] == 2


def test_windows_loaded_once_per_sequential_pass(monkeypatch):
    """Windowed: a front-to-back pass of slices (some crossing windows) loads
    each window once; going back reloads; results exact throughout."""
    monkeypatch.setattr(C, "_RESIDENT_WINDOW_BYTES", 1)
    # device-side emptiness check: every slice is built (none skipped on the host)
    monkeypatch.setattr(ComboRowMask, "union_is_cheap", lambda self: False)
    old, cm = _sized_mask(400)
    n_p, nb = cm.parts.shape
    F.reset_combo_usage()
    for r0 in range(0, 400, 24):                # 3-byte slices over 1-byte windows
        sl = slice(r0, min(400, r0 + 24))
        _same_dev(C._combo_mask_device(cm, sl, "cpu"), _old_build(old, sl), str(sl))
    assert cm._parts_dev["cpu"].width == 1
    assert F.combo_usage()["resident_uploads"] == nb
    _same_dev(C._combo_mask_device(cm, slice(0, 8), "cpu"), _old_build(old, slice(0, 8)), "back")
    assert F.combo_usage()["resident_uploads"] == nb + 1


def test_window_width_policy(monkeypatch, fresh_pools):
    """Whole table when it fits the free fraction of free + cached-unallocated
    device memory (boundary inclusive); otherwise the columns that
    min(budget, window cap) holds, at least one; CPU: whole; test override."""
    GiB = 1 << 30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (3 * GiB, 24 * GiB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d=None: 2 * GiB)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: 1 * GiB)
    monkeypatch.setattr(C, "_resident_device_key", lambda d: "cuda:0")   # no GPU here
    assert C._RESIDENT_PARTS_FREE_FRACTION == 0.25
    budget = GiB                                # 0.25 x (3 + 2 - 1) GiB
    n_p = 1000
    for dev in ("cuda", "cuda:0"):
        assert C._resident_window_width(n_p, budget // n_p, dev) == budget // n_p
        nb = budget // n_p + 1                  # one column too many: windowed
        assert C._resident_window_width(n_p, nb, dev) == budget // n_p
    monkeypatch.setattr(C, "_RESIDENT_WINDOW_MAX_BYTES", 10 * n_p)
    assert C._resident_window_width(n_p, 10 ** 9, "cuda") == 10
    assert C._resident_window_width(10 ** 12, 5, "cuda") == 1     # never below one
    assert C._resident_window_width(n_p, 10 ** 9, "cpu") == 10 ** 9
    monkeypatch.setattr(C, "_RESIDENT_WINDOW_BYTES", 3 * n_p)
    assert C._resident_window_width(n_p, 100, "cpu") == 3


def _sized_mask(n, nq=6, seed=0):
    """A lazy mask over an n-row table: the queries are fixed (so the part
    count is), the table content varies with `seed`; parts.nbytes ~ n."""
    t = _table(random.Random(9100 + seed), n)
    f = Filter(must=[FilterCondition(field="text", match_text_from_query="kw")],
               should=[FilterCondition(field="text", match_text_from_query="kw2")])
    return _pair(f, t, _qv(random.Random(9000), nq))


@pytest.fixture
def fresh_pools():
    C._close_resident_pools()
    F.reset_combo_usage()
    yield
    C._close_resident_pools()


def test_resident_pool_reuse_growth_and_bound(monkeypatch, fresh_pools):
    """Released buffers are reused best-fit across files (stale bytes fully
    overwritten), grown with headroom only when nothing free fits (dropping the
    largest free one first, so the pool never exceeds the peak in use), never
    pooled twice, and every mask stays exact."""
    monkeypatch.setattr(C, "_RESIDENT_ALIGN", 1)      # exact headroom arithmetic

    def pool():
        return C._RESIDENT_POOLS.get("cpu", [])

    def use(pair, ctx):
        old, cm = pair
        n = cm.n_rows
        rows = np.arange(0, n, 2, dtype=np.int64)
        for tr in (slice(0, n), rows):
            _same_dev(C._combo_mask_device(cm, tr, "cpu"), _old_build(old, tr), ctx)
        return cm._parts_dev["cpu"]

    a = _sized_mask(400)
    ea = use(a, "a")
    cap_a = ea.buf.numel()
    assert C._RESIDENT_HEADROOM == 1.125
    assert cap_a == int(a[1].parts.nbytes * 1.125) > a[1].parts.nbytes
    C._resident_release(a[1])
    C._resident_release(a[1])                          # idempotent: pooled once
    assert len(pool()) == 1
    # slightly smaller and different content: reuses a's buffer
    b = _sized_mask(390, seed=1)
    assert b[1].parts.nbytes <= cap_a
    eb = use(b, "b")
    assert eb.buf is ea.buf
    # two at once: b holds a's buffer, c needs a new one
    c = _sized_mask(300, seed=2)
    ec = use(c, "c")
    assert ec.buf is not eb.buf
    u = F.combo_usage()
    assert (u["resident_allocs"], u["resident_reuses"]) == (2, 1)
    C._resident_release(b[1]); C._resident_release(c[1])
    assert len(pool()) == 2
    # bigger than every free buffer: the largest free is dropped, one new alloc
    d = _sized_mask(4000, seed=3)
    ed = use(d, "d")
    assert ed.buf is not ea.buf and ed.buf is not ec.buf
    assert len(pool()) == 1 and pool()[0] is ec.buf
    C._resident_release(d[1])
    assert len(pool()) == 2
    assert F.combo_usage()["resident_allocs"] == 3
    C._close_resident_pools()
    assert not C._RESIDENT_POOLS


def test_resident_acquire_is_best_fit(fresh_pools):
    import torch as _t
    small, big = _t.empty(100, dtype=_t.uint8), _t.empty(1000, dtype=_t.uint8)
    C._RESIDENT_POOLS["cpu"] = [big, small]
    assert C._resident_acquire("cpu", 50) is small
    assert C._resident_acquire("cpu", 50) is big


def test_window_override_is_clamped_and_pool_counts_as_free(monkeypatch, fresh_pools):
    monkeypatch.setattr(C, "_RESIDENT_WINDOW_BYTES", 10 ** 12)
    assert C._resident_window_width(7, 100, "cpu") == 100
    monkeypatch.setattr(C, "_RESIDENT_WINDOW_BYTES", None)
    GiB = 1 << 30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (1 * GiB, 24 * GiB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d=None: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: 0)
    monkeypatch.setattr(C, "_resident_device_key", lambda d: "cuda:0")
    # 1 GiB free -> 256 MiB budget: a 512 MiB table is windowed ...
    assert C._resident_window_width(1024, 512 << 10, "cuda") < 512 << 10
    # ... unless an idle pooled buffer (here 2 GiB, counted as free) covers it
    monkeypatch.setattr(C, "_resident_largest_pooled", lambda d: 2 * GiB)
    assert C._resident_window_width(1024, 512 << 10, "cuda") == 512 << 10


def test_largest_pooled_and_device_key(monkeypatch, fresh_pools):
    """Only the largest idle buffer of the device counts (acquire reuses or
    drops one buffer); "cuda" and "cuda:0" are the same pool."""
    import torch as _t
    assert C._resident_largest_pooled("cpu") == 0
    C._RESIDENT_POOLS["cpu"] = [_t.empty(100, dtype=_t.uint8), _t.empty(700, dtype=_t.uint8),
                                _t.empty(300, dtype=_t.uint8)]
    C._RESIDENT_POOLS["meta"] = [_t.empty(10 ** 6, dtype=_t.uint8, device="meta")]
    assert C._resident_largest_pooled("cpu") == 700
    monkeypatch.setattr(C, "_indexed_cuda", lambda d: _t.device("cuda", 0))
    assert C._resident_device_key("cuda") == C._resident_device_key("cuda:0") == "cuda:0"


def test_budget_fallback_warns_once_per_filter_and_counts(monkeypatch, caplog):
    """Over the byte budget: the reference full mask (identical), a warning
    naming the filter and the fixes, logged once per run per filter, and a
    `budget_fallbacks` count per mask."""
    import logging
    monkeypatch.setattr(F, "_COMBO_PARTS_GUARD", True)
    monkeypatch.setattr(F, "_COMBO_PARTS_MAX_BYTES", 0)
    rng = random.Random(5)
    t = _table(rng, 40)
    qv = _qv(rng, 2)                 # few queries: parts > queries -> over budget
    f = Filter(must=[FilterCondition(field="text", match_text_from_query="kw")],
               should=[FilterCondition(field="text", match_text_from_query="kw2")])
    F.reset_combo_usage()
    with caplog.at_level(logging.WARNING, logger="nova_bf.filters"):
        got = [evaluate(f, t, qv, combo=True) for _ in range(3)]
    assert all(type(g) is PackedRowMask for g in got)
    _assert_same(evaluate(f, t, qv, combo=False), got[0], "fallback")
    msgs = [r.getMessage() for r in caplog.records if "budget" in r.getMessage()]
    assert len(msgs) == 1, msgs
    assert "text~kw" in msgs[0] and "_COMBO_PARTS_MAX_BYTES" in msgs[0] and "slower" in msgs[0]
    assert F.combo_usage()["budget_fallbacks"] == 3
    F.reset_combo_usage()            # a new run warns again
    with caplog.at_level(logging.WARNING, logger="nova_bf.filters"):
        evaluate(f, t, qv, combo=True)
    assert len([r for r in caplog.records if "budget" in r.getMessage()]) == 2


def test_zero_rows_and_zero_queries_upload_nothing(fresh_pools):
    _, empty = _sized_mask(0)
    assert empty.parts.shape[1] == 0
    assert C._combo_mask_device(empty, np.array([], dtype=np.int64), "cpu") is None
    assert C._combo_mask_device(empty, slice(0, 0), "cpu") is None
    _, cm = _sized_mask(50)
    nobody = cm[np.array([], dtype=np.int64)]
    assert C._combo_mask_device(nobody, slice(0, 50), "cpu") is None
    assert not empty._parts_dev and not nobody._parts_dev
    # an empty compacted batch uploads nothing either (device-side check path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ComboRowMask, "union_is_cheap", lambda self: False)
        assert C._combo_mask_device(cm, np.array([], dtype=np.int64), "cpu") is None
    assert not cm._parts_dev
    assert F.combo_usage()["resident_uploads"] == 0


@pytest.mark.parametrize("seed", range(25))
def test_union_keep_matches(seed):
    rng = random.Random(800 + seed)
    n = rng.randint(1, 120)
    t = _table(rng, n)
    qv = _qv(rng, rng.randint(1, 15))
    f1, f2 = _rand_filter(rng), _rand_filter(rng)
    o1, n1 = _pair(f1, t, qv)
    o2, n2 = _pair(f2, t, qv)
    uni = Filter(must=[FilterCondition(field="cat", match="a")])
    um = evaluate(uni, t, qv)
    want = C._union_keep([f1, f2, uni], {f1: o1, f2: o2, uni: um}, n)
    got = C._union_keep([f1, f2, uni], {f1: n1, f2: n2, uni: um}, n)
    assert got.dtype == want.dtype and np.array_equal(got, want)


# ---------------------------------------------------------------------------
# 3. run_compute end to end: switch on == switch off
# ---------------------------------------------------------------------------
DIM = 6


def _write(path, dense, **cols):
    data = {"dense_embedding": pa.array(np.asarray(dense).tolist(), type=pa.list_(pa.float32()))}
    data.update({k: pa.array(v) for k, v in cols.items()})
    pq.write_table(pa.table(data), str(path))


@pytest.fixture
def e2e_ds(tmp_path):
    rng = random.Random(77)
    nrng = np.random.default_rng(77)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    base = 0
    for i, n in enumerate([131, 77, 203]):
        _write(cdir / f"f{i}.parquet",
               nrng.standard_normal((n, DIM)).astype(np.float32),
               id=[f"c{base + r}" for r in range(n)],
               text=[_rand_text(rng, WORDS) for _ in range(n)],
               url=[_rand_text(rng, DOMS, 1, 3) for _ in range(n)],
               cat=[rng.choice(["a", "b", None]) for _ in range(n)])
        base += n
    nq = 30
    cols = {"kw": [_rand_phrase(rng, WORDS) for _ in range(nq)],
            "neg": [_rand_phrase(rng, WORDS) for _ in range(nq)],
            "cat_q": [rng.choice(["a", "b"]) for _ in range(nq)]}
    for i in range(3):
        cols[f"d{i}"] = [_rand_phrase(rng, DOMS) for _ in range(nq)]
    qpath = tmp_path / "queries.parquet"
    _write(qpath, nrng.standard_normal((nq, DIM)).astype(np.float32),
           qid=[f"q{i}" for i in range(nq)], **cols)
    return {"cdir": str(cdir), "qpath": str(qpath), "tmp": tmp_path}


FINEWEB_LIKE = Filter(
    must=[FilterCondition(field="text", match_text_from_query="kw")],
    should=[FilterCondition(field="url", match_text_from_query=f"d{i}") for i in range(3)],
)
WITH_NOT = Filter(
    must=[FilterCondition(field="text", match_text_from_query="kw"),
          FilterCondition(field="cat", match="a")],
    must_not=[FilterCondition(field="text", match_text_from_query="neg")],
)
STATIC_SHOULD = Filter(
    must=[FilterCondition(field="url", match_text_from_query="d0")],
    should=[FilterCondition(field="text", match_text_from_query="neg"),
            FilterCondition(field="text", match_text="alpha")],
)
MIXED = Filter(   # per-query non-text alongside text: the PackedRowMask fallback
    must=[FilterCondition(field="text", match_text_from_query="kw"),
          FilterCondition(field="cat", match_from_query="cat_q")],
)


def _searches(with_unfiltered):
    out = [SearchSpec(name=n, vector_type="dense", metric="dot", k=k, filter=f)
           for n, f, k in (("fw", FINEWEB_LIKE, 25), ("not", WITH_NOT, 40),
                           ("st", STATIC_SHOULD, 9), ("mixed", MIXED, 60))]
    if with_unfiltered:
        out.append(SearchSpec(name="all", vector_type="dense", metric="dot", k=5))
    return out


def _run(ds, name, searches, batch):
    import json
    out = ds["tmp"] / name
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(dense_batch_size=batch),
        searches=searches,
    )
    res = {}
    for n, p in C.run_compute(cfg).items():
        t = pq.read_table(p).to_pydict()
        res[n] = t
    manifest = json.loads(next(out.rglob("*manifest*.json")).read_text())
    return res, manifest["params"]["text_filter_combine"]


@pytest.mark.parametrize("window", [None, 1])
@pytest.mark.parametrize("with_unfiltered", [False, True])
@pytest.mark.parametrize("batch", [37, None])
def test_run_compute_identical_with_switch(e2e_ds, monkeypatch, with_unfiltered, batch, window):
    monkeypatch.setattr(C, "_RESIDENT_WINDOW_BYTES", window)
    # a pool left by earlier work must not carry into the run
    # (large enough that every mask here would fit and reuse it)
    stale_buf = torch.empty(1 << 20, dtype=torch.uint8)
    seen_stale = []
    real_acq = C._resident_acquire

    def spy_acq(device, need):
        got = real_acq(device, need)
        if got is stale_buf:
            seen_stale.append(True)
        return got
    monkeypatch.setattr(C, "_resident_acquire", spy_acq)
    # every lazy mask run_compute gets must leave it with no device copies:
    # the per-file `release_device` (not garbage collection) frees them
    made = []
    real_eval = C.evaluate

    def spy_eval(*a, **kw):
        m = real_eval(*a, **kw)
        if isinstance(m, ComboRowMask):
            made.append(m)
        return m
    monkeypatch.setattr(C, "evaluate", spy_eval)
    tag = f"{int(with_unfiltered)}_{batch}_{window}"
    monkeypatch.setenv("NOVA_BF_NO_GPU_COMBO", "1")
    off, u_off = _run(e2e_ds, "off" + tag, _searches(with_unfiltered), batch)
    monkeypatch.delenv("NOVA_BF_NO_GPU_COMBO")
    # injected after the off run (whose own exit clears the pools)
    C._RESIDENT_POOLS.setdefault("cpu", []).append(stale_buf)
    on, u_on = _run(e2e_ds, "on" + tag, _searches(with_unfiltered), batch)
    assert u_off["permitted"] is False and u_off["combo_masks"] == 0
    assert u_on["permitted"] is True
    # 3 lazy-capable filters x 3 files; MIXED falls back on both runs
    assert u_on["combo_masks"] == 9 and u_on["packed_masks"] == 3
    assert u_on["device_builds"] > 0 and u_on["host_materialisations"] == 0
    # pooled: at most one buffer per lazy filter in flight (3), the rest reused;
    # one buffer per mask that was ever used
    assert 0 < u_on["resident_allocs"] <= 3
    taken = u_on["resident_allocs"] + u_on["resident_reuses"]
    assert 0 < taken <= u_on["combo_masks"]
    if window is None:
        assert u_on["resident_uploads"] == taken     # whole table: once per mask
    else:
        assert u_on["resident_uploads"] > taken      # windows: reloaded
    assert not C._RESIDENT_POOLS, "run_compute left resident pools behind"
    assert made and all(not m._parts_dev and not m._dev for m in made), (
        "a lazy mask kept device copies after run_compute")
    assert not seen_stale, "run_compute reused a buffer from before the run"
    assert set(on) == set(off)
    short = 0
    for name in on:
        assert on[name] == off[name], name   # ids, scores, order, lengths
        short += sum(len(h) < (60 if name == "mixed" else 40) for h in on[name]["hit_ids"])
    assert short, "fixture no longer produces short lists"


@pytest.mark.parametrize("cap", [0, None])
def test_parts_guard_threshold_is_exact(monkeypatch, cap):
    """Guard on: lazy iff `parts` bytes <= max(full mask bytes, the byte cap);
    otherwise the reference PackedRowMask. With the cap at 0 that is "part
    rows (ONES/ZERO included) <= queries", fuzzed until both sides of the
    boundary have been hit; at the default cap (4 GiB) small tables are
    always lazy. Bit-identical either way."""
    if cap is not None:
        monkeypatch.setattr(F, "_COMBO_PARTS_MAX_BYTES", cap)
    hit = {"eq": 0, "over": 0}
    for seed in range(400):
        rng = random.Random(52000 + seed)
        t = _table(rng, rng.randint(1, 60))
        nq = rng.randint(1, 12)
        qv = _qv(rng, nq, same_combo=rng.random() < 0.3)
        filt = _rand_filter(rng)
        monkeypatch.setattr(F, "_COMBO_PARTS_GUARD", False)
        lazy = evaluate(filt, t, qv, combo=True)
        if not isinstance(lazy, ComboRowMask):
            continue                     # a shape the lazy type cannot express
        rows = lazy.parts.shape[0]
        monkeypatch.setattr(F, "_COMBO_PARTS_GUARD", True)
        got = evaluate(filt, t, qv, combo=True)
        want_lazy = (rows <= nq) if cap == 0 else True
        assert isinstance(got, ComboRowMask) == want_lazy, (seed, rows, nq)
        _assert_same(evaluate(filt, t, qv, combo=False), got, f"seed={seed}")
        hit["eq"] += rows == nq
        hit["over"] += rows == nq + 1
    assert hit["eq"] and hit["over"], hit
