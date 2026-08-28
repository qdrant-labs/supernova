"""The packed selection key and the ordinals that ride in it.

`nova_bf.tiebreak` turns `(score, ordinal)` into one int64 whose order is the
total order ties are resolved by. Two properties carry the whole design and are
pinned here:

  * the key's high half orders EXACTLY as the float32 does, and is a bijection,
    so the running state can hold keys instead of scores and still recover the
    score bit-for-bit at decode;
  * a worker's ordinals are a dense relabelling of its own rows that is
    order-consistent with the global rule — which is what lets a per-worker
    top-K compose into the correct global one.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

torch = pytest.importorskip("torch")

import nova_bf.tiebreak as tb
from nova_bf.tiebreak import (
    MAX_ROWS_PER_WORKER,
    U32,
    build_ordinals,
    id_order_scalar,
    pack,
    pack_topk,
    score_order_key,
    sentinel_key,
    unpack_score,
)


# --------------------------------------------------------------------------
# the score half
# --------------------------------------------------------------------------

INTERESTING = [
    0.0, -0.0, 1.0, -1.0, 1e-45, -1e-45, 3.4e38, -3.4e38,
    float("inf"), float("-inf"), 1.1754944e-38, 0.1, -0.1,
]


def test_the_order_key_sorts_exactly_as_the_float_does():
    v = torch.tensor(INTERESTING, dtype=torch.float32)
    keys = score_order_key(v)
    by_key = [INTERESTING[i] for i in torch.argsort(keys).tolist()]
    # -0.0 folds onto +0.0, so compare on a sequence where they are one value
    assert by_key == sorted(INTERESTING, key=lambda x: (x, 0.0))


def test_negative_zero_is_folded_onto_positive_zero():
    """They are numerically EQUAL, so the ordinal must decide between them, not
    the sign bit. Euclidean negates its distance, so a self-hit really does
    produce -0.0 — without this fold it would sort below every other 0.0."""
    z = torch.tensor([0.0, -0.0], dtype=torch.float32)
    assert score_order_key(z)[0].item() == score_order_key(z)[1].item()


def test_the_score_survives_the_round_trip_bit_for_bit():
    rng = np.random.default_rng(0)
    bits = rng.integers(-(2**31), 2**31, 200_000, dtype=np.int64).astype(np.int32)
    s = torch.from_numpy(bits).view(torch.float32)
    s = torch.where(torch.isnan(s), torch.zeros_like(s), s)   # NaN has no order
    ordinal = torch.from_numpy(rng.integers(0, U32, len(s)).astype(np.int64))
    back = unpack_score(pack(s, ordinal))
    same = torch.where(
        s == 0.0,                                  # -0.0 deliberately becomes +0.0
        back == 0.0,
        back.view(torch.int32) == s.view(torch.int32),
    )
    assert bool(same.all())


def test_the_packed_key_never_leaves_the_signed_int64_range():
    """`score_order_key` spans [-2**31, 2**31) and is multiplied by 2**32, so
    the extremes are exactly the int64 endpoints — one bit further either way
    and the key would wrap and invert."""
    extremes = torch.tensor(
        [float("-inf"), float("inf"), -3.4e38, 3.4e38], dtype=torch.float32
    )
    for ordinal in (0, U32):
        key = pack(extremes, torch.tensor(ordinal, dtype=torch.int64))
        assert key.dtype == torch.int64
        assert int(key.min()) >= -(2**63)
        assert int(key.max()) <= 2**63 - 1


def test_score_dominates_the_ordinal():
    """A worse score can never win on a better ordinal — the ordinal only ever
    separates candidates the score left equal."""
    s = torch.tensor([1.0, 1.0000001], dtype=torch.float32)
    key = pack(s, torch.tensor([0, U32], dtype=torch.int64))
    assert key[1] > key[0], "the higher score wins despite the worst ordinal"


def test_lower_ordinal_wins_at_equal_score():
    s = torch.tensor([5.0, 5.0], dtype=torch.float32)
    key = pack(s, torch.tensor([7, 3], dtype=torch.int64))
    assert key[1] > key[0]


def test_every_real_candidate_outranks_the_sentinel():
    sent = sentinel_key((1, 3), "cpu")
    worst = pack(
        torch.tensor([[float("-inf")]], dtype=torch.float32),
        torch.tensor(U32 - 1, dtype=torch.int64),
    )
    assert bool((worst > sent).all()), "even -inf at a real ordinal beats the pad"
    assert float(unpack_score(sent)[0, 0]) == float("-inf")


def test_pack_topk_matches_an_unchunked_pack(monkeypatch):
    """The chunking exists only to bound transient memory; per-row top-K is
    independent, so it must be exact, not an approximation."""
    import nova_bf.tiebreak as tb

    rng = np.random.default_rng(3)
    s = torch.from_numpy(rng.random((40, 60), dtype=np.float32))
    o = torch.from_numpy(rng.integers(0, 1000, 60).astype(np.int64))
    whole = torch.topk(pack(s, o), k=7, dim=1)
    monkeypatch.setattr(tb, "PACK_TARGET_SLOTS", 64)     # force many chunks
    chunked = pack_topk(s, o, 7)
    assert torch.equal(whole.values, chunked[0])
    assert torch.equal(whole.indices, chunked[1])


# --------------------------------------------------------------------------
# the ordinals
# --------------------------------------------------------------------------


def test_ordinals_follow_sorted_id_order_and_interleave_across_files():
    """Interleaving is the load-bearing part: the running top-K folds a later
    file's candidates against an earlier one's, so ordinals from different
    files have to be comparable. Ranking each file alone would restart every
    file at 0 and let its first row outrank everything."""
    files = [pa.array(["m", "a", "z"]), pa.array(["c", "q"]), pa.array(["b"])]
    got = build_ordinals(files)
    assert [o.tolist() for o in got] == [[3, 0, 5], [2, 4], [1]]

    flat_ids = [i for f in files for i in f.to_pylist()]
    flat_ord = np.concatenate(got)
    assert [flat_ids[i] for i in np.argsort(flat_ord)] == sorted(flat_ids)


def test_duplicate_ids_fall_back_to_corpus_position():
    """Not left to whether the sort happens to be stable — a multithreaded sort's
    tie order can depend on how the input was partitioned, i.e. on thread count.
    The position key makes it explicit and total."""
    got = build_ordinals([pa.array(["b", "a", "b"]), pa.array(["a", "b"])])
    # flat: b@0 a@1 b@2 | a@3 b@4  ->  a@1=0, a@3=1, b@0=2, b@2=3, b@4=4
    assert [o.tolist() for o in got] == [[2, 0, 3], [1, 4]]


def test_numeric_ids_order_numerically():
    """The whole reason a numeric id column cannot ride on `hit_ids`: as text,
    "10" precedes "9"."""
    got = build_ordinals([pa.array([9, 10, 100, 2], pa.int64())])
    assert got[0].tolist() == [1, 2, 3, 0]


def test_wide_integers_do_not_wrap():
    """A uint64 above 2**63 must not sort below a small one. Ranking never
    inspects the value's width, which is precisely why this is free."""
    vals = [2**64 - 1, 5, 2**63]
    got = build_ordinals([pa.array(vals, pa.uint64())])
    assert got[0].tolist() == [2, 0, 1]


def test_ids_far_from_zero_separate_exactly():
    """Snowflake/epoch-shaped ids sit wholly above 2**32. A projection of the
    id's VALUE into 32 bits collapses them; ranking is indifferent."""
    base = 1_700_000_000_000_000_000
    got = build_ordinals([pa.array([base + 3, base + 1, base + 2], pa.int64())])
    assert got[0].tolist() == [2, 0, 1]


def test_long_shared_prefixes_do_not_reduce_resolution():
    """The case a 4-byte prefix key cannot serve at all: every id identical for
    the first 10 characters, entropy only after."""
    head = "<urn:uuid:"
    ids = [f"{head}{i:032x}>" for i in (9, 3, 7)]
    got = build_ordinals([pa.array(ids)])
    assert got[0].tolist() == [2, 0, 1]


def test_a_null_id_is_rejected_rather_than_ordered():
    """A null cannot be ordered CONSISTENTLY: it would sort last here, but
    `hit_ids` render it as the literal `"None"`, which is what `merge` compares
    across workers — and `"None"` sorts before every lowercase id. Ordering it
    either way would make the two sides of the reduce disagree."""
    with pytest.raises(ValueError, match="null value"):
        build_ordinals([pa.array(["b", None, "a"])])


def test_an_empty_file_contributes_nothing():
    got = build_ordinals([pa.array(["b"]), pa.array([], pa.string()), pa.array(["a"])])
    assert [o.tolist() for o in got] == [[1], [], [0]]


def test_a_worker_with_no_rows_at_all():
    assert [o.tolist() for o in build_ordinals([])] == []
    assert [o.tolist() for o in build_ordinals([pa.array([], pa.string())])] == [[]]


def test_ordinals_are_dense_and_fit_the_key():
    rng = np.random.default_rng(5)
    ids = pa.array([f"id{v:08d}" for v in rng.integers(0, 10**7, 5000)])
    got = build_ordinals([ids])[0]
    assert got.dtype == np.uint32
    assert sorted(got.tolist()) == list(range(len(ids)))
    assert got.max() < MAX_ROWS_PER_WORKER


def test_too_many_rows_for_the_key_is_rejected(monkeypatch):
    """The alternative is silent: the ordinal wraps and ties stop being
    deterministic, which is the one thing the field exists to prevent."""
    import nova_bf.tiebreak as tb

    monkeypatch.setattr(tb, "MAX_ROWS_PER_WORKER", 3)
    with pytest.raises(RuntimeError, match="larger `--num-jobs`"):
        tb.build_ordinals([pa.array(["a", "b", "c", "d"])])


@pytest.mark.parametrize("n_files", [1, 2, 7])
def test_ordinals_are_a_permutation_however_the_rows_are_split(n_files):
    """Splitting the SAME ids across a different number of files must not
    change any row's ordinal — the sort spans the worker, not the file."""
    rng = np.random.default_rng(11)
    ids = [f"x{v:06d}" for v in rng.choice(10**6, 900, replace=False)]
    one = np.concatenate(build_ordinals([pa.array(ids)]))
    chunks = np.array_split(np.array(ids, dtype=object), n_files)
    many = np.concatenate(build_ordinals([pa.array(list(c)) for c in chunks]))
    assert np.array_equal(one, many)


# --------------------------------------------------------------------------
# the cross-worker ordinate
# --------------------------------------------------------------------------


def test_id_order_scalar_orders_the_whole_uint64_range():
    vals = [0, 1, 2**32, 2**63 - 1, 2**63, 2**64 - 1]
    keys = [id_order_scalar(v, unsigned=True) for v in vals]
    assert keys == sorted(keys), "unsigned ids must not wrap negative"
    assert all(-(2**63) <= x < 2**63 for x in keys)


def test_id_order_scalar_puts_nulls_last():
    assert id_order_scalar(None, False) > id_order_scalar(2**62, False)


def test_id_order_scalar_agrees_with_the_ordinals_it_stands_in_for():
    """`merge` applies this rule across workers while `build_ordinals` applies
    it within one; if they disagreed, a sharded run would order differently
    from a single-node one."""
    vals = [7, 2**63, 0, 2**64 - 1, 42]
    ordinals = build_ordinals([pa.array(vals, pa.uint64())])[0]
    scalars = [id_order_scalar(v, unsigned=True) for v in vals]
    assert list(np.argsort(ordinals)) == list(np.argsort(scalars))


# --------------------------------------------------------------------------
# the compiled fast path
# --------------------------------------------------------------------------


def test_the_compiled_pack_agrees_with_the_eager_one():
    """`pack` runs a `torch.compile`d kernel and falls back to the eager body if
    compilation is unavailable. The two must be bit-identical: which of them ran
    is a performance detail, and a divergence would make the ARTIFACT depend on
    whether inductor happened to work on that host."""
    rng = np.random.default_rng(3)
    bits = rng.integers(-(2**31), 2**31, 50_000, dtype=np.int64).astype(np.int32)
    s = torch.from_numpy(bits).view(torch.float32)
    s = torch.where(torch.isnan(s), torch.zeros_like(s), s)  # NaN has no order
    o = torch.from_numpy(rng.integers(0, U32, len(s)).astype(np.int64))
    assert bool((tb.pack(s.clone(), o) == tb._pack_eager(s.clone(), o)).all())


def test_a_failing_compile_degrades_to_eager_instead_of_raising(monkeypatch):
    """The fallback is what keeps this a performance path only."""
    monkeypatch.setattr(tb, "_compiled_pack", None)
    monkeypatch.setattr(tb, "_compiled_proven", False)
    monkeypatch.setattr(
        torch, "compile", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no inductor"))
    )
    s = torch.tensor([[1.0, -0.0, 3.0]], dtype=torch.float32)
    o = torch.arange(3, dtype=torch.int64)
    assert tb.pack(s.clone(), o).tolist() == tb._pack_eager(s.clone(), o).tolist()
    assert tb._compiled_pack is False, "a failed compile must not be retried"


def test_a_failure_AFTER_the_compile_is_proven_propagates(monkeypatch):
    """Only the first call is guarded. Guarding every call would let one
    anomalous input, or a transient CUDA OOM, silently pin the rest of a
    multi-hour run to the slow path with the real error swallowed."""
    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(tb, "_compiled_pack", boom)
    monkeypatch.setattr(tb, "_compiled_proven", True)
    with pytest.raises(RuntimeError, match="out of memory"):
        tb.pack(torch.tensor([[1.0]], dtype=torch.float32), torch.zeros(1, dtype=torch.int64))
    assert tb._compiled_pack is boom, "a proven path must not be torn down by one error"


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_non_float32_scores_are_refused_rather_than_reshaped(dtype):
    """The transform reinterprets each score's BITS as an int32, so a f16 input
    silently HALVES the key's last dim and a f64 input doubles it — the ordinal
    then broadcasts against the wrong axis. Everything nova-bf scores is upcast
    to f32, so this can only mean something upstream broke; it must say so
    rather than produce a differently-shaped key."""
    s = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=dtype)
    with pytest.raises(TypeError, match="needs float32 scores"):
        tb.pack(s, torch.arange(4, dtype=torch.int64))


def test_a_refused_dtype_does_not_disable_the_compiled_path():
    """A bad input must not be blamed on inductor: the dtype gate runs BEFORE
    the compile fallback, so a rejected call leaves the fast path intact."""
    tb.pack(torch.tensor([[1.0]], dtype=torch.float32), torch.zeros(1, dtype=torch.int64))
    was = tb._compiled_pack
    with pytest.raises(TypeError):
        tb.pack(torch.tensor([[1.0]], dtype=torch.float64), torch.zeros(1, dtype=torch.int64))
    assert tb._compiled_pack is was


def test_the_negative_zero_fold_is_exact_over_the_bit_space():
    """The fold moved from a compare-and-select to `+ 0.0`; it must still map
    -0.0 (and ONLY -0.0) onto +0.0."""
    rng = np.random.default_rng(4)
    bits = np.concatenate([
        np.array([0x00000000, 0x80000000, 0x7F800000, 0xFF800000], dtype=np.uint32),
        rng.integers(0, 2**32, size=200_000, dtype=np.uint64).astype(np.uint32),
    ])
    f = torch.from_numpy(bits.view(np.float32))
    keys = score_order_key(f)
    nn = ~torch.isnan(f)
    # order still agrees with the float order
    order = torch.argsort(keys[nn])
    vals = f[nn][order]
    assert bool((vals[:-1] <= vals[1:]).all())
    # the two zeros collapse, nothing else does
    z = torch.tensor([0.0, -0.0], dtype=torch.float32)
    assert score_order_key(z)[0].item() == score_order_key(z)[1].item()
