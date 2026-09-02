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
    independent, so it must be exact, not an approximation.

    Compared as SETS per row, not sequences: `pack_topk` selects with
    `sorted=False` (its caller re-selects the result, so ordering it would be
    work nobody reads), and each chunk is its own `topk` call, so the order
    within a row is not meaningful. What chunking must preserve is exactly
    WHICH k candidates come back.
    """
    import nova_bf.tiebreak as tb

    rng = np.random.default_rng(3)
    s = torch.from_numpy(rng.random((40, 60), dtype=np.float32))
    o = torch.from_numpy(rng.integers(0, 1000, 60).astype(np.int64))
    whole_k, whole_i = torch.topk(pack(s, o), k=7, dim=1, sorted=False)
    monkeypatch.setattr(tb, "PACK_TARGET_SLOTS", 64)     # force many chunks
    chunk_k, chunk_i, _ = pack_topk(s, o, 7)

    assert torch.equal(
        torch.sort(whole_k, dim=1).values, torch.sort(chunk_k, dim=1).values
    )
    assert torch.equal(
        torch.sort(whole_i, dim=1).values, torch.sort(chunk_i, dim=1).values
    )
    # the returned key must still be the key OF the returned index — an
    # order-insensitive comparison alone would not catch a mismatched pairing
    assert torch.equal(chunk_k, pack(s, o).gather(1, chunk_i))


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


# --------------------------------------------------------------------------
# the CUDA kernel path
# --------------------------------------------------------------------------


def test_the_kernel_declines_everything_it_cannot_serve():
    """`available()` is the only thing standing between the kernel and inputs
    it would answer WRONGLY (a per-cell ordinal) or slowly (a slice wider than
    its register budget). Every false here sends the call to the portable path,
    so a gate that wrongly returns True is a silent correctness bug, not a
    performance one."""
    import nova_bf.topk_triton as tk

    s = torch.randn(4, 64)
    o = torch.arange(64, dtype=torch.int64)
    assert not tk.available(s, o, 8), "CPU tensors must never take the kernel"
    assert not tk.available(s.double(), o, 8), "non-float32 must decline"
    assert not tk.available(s, o.unsqueeze(0).expand(4, 64), 8), (
        "a 2-D (per-cell) ordinal must decline — the kernel indexes it by COLUMN, "
        "so it would silently break `_merge_topk`'s state"
    )
    assert not tk.available(s, torch.arange(63, dtype=torch.int64), 8), "length mismatch"
    assert not tk.available(s[:, ::2], o[::2], 8), "non-contiguous must decline"
    wide = torch.empty((1, tk.MAX_BLOCK + 1))
    assert not tk.available(wide, torch.arange(tk.MAX_BLOCK + 1), 8), "over-wide must decline"


def test_pack_topk_falls_back_when_the_kernel_declines(monkeypatch):
    """The portable path must still be reachable and exact — it is what runs on
    CPU, on a build without triton, and for every input `available()` refuses."""
    import nova_bf.topk_triton as tk

    monkeypatch.setattr(tk, "available", lambda *a: False)
    rng = np.random.default_rng(11)
    s = torch.from_numpy(rng.random((32, 128), dtype=np.float32))
    o = torch.from_numpy(rng.integers(0, 10_000, 128).astype(np.int64))
    key, idx, _ = pack_topk(s, o, 9)
    ref = torch.topk(pack(s, o), 9, dim=1, sorted=False)
    assert torch.equal(torch.sort(idx, dim=1).values, torch.sort(ref.indices, dim=1).values)
    assert torch.equal(key, pack(s, o).gather(1, idx)), "key must pair with its index"


# --------------------------------------------------------------------------
# the int32 offset guard both Triton gates share
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mod_name", ["topk_triton", "merge_triton"])
def test_the_gate_declines_shapes_whose_offsets_overflow_int32(mod_name):
    """Inside both kernels every pointer is `base + row * row_stride + col`, and
    BOTH factors are int32 — `tl.program_id` is `tl.int32`, and Triton types a
    kernel int argument as i32 whenever its VALUE fits (verified: `mangle_type`
    returns i32 at 2**31 - 1 and i64 at 2**31). So the product wraps and the
    load silently reads a different query's row: wrong ground truth, no error.

    Widening the row index to int64 fixes it and was measured to cost 27% — it
    took `topk_triton` from 104 registers to 156 on an A10G — so the gate
    declines the oversized shape instead and the portable torch path, which has
    no such limit, takes over.

    Lives here rather than in the GPU-gated kernel suites because it is pure
    arithmetic: it must run on every box, and the shapes it describes are far
    too large to allocate.
    """
    import importlib

    mod = importlib.import_module(f"nova_bf.{mod_name}")

    # everything production actually runs must still be accepted — a false
    # decline is a silent ~4x slowdown that no test or log would show
    assert mod._offsets_fit_int32(10_000, 4096, 1000)
    assert mod._offsets_fit_int32(110_000, 4096, 1000), "the largest real query set"
    assert mod._offsets_fit_int32(110_000, 1024, 1000)

    # w=4096 wraps at n_q >= 2**31 / 4096 = 524,288
    assert not mod._offsets_fit_int32(600_000, 4096, 1000)
    # the k-strided output pointers wrap later, but they do wrap
    assert not mod._offsets_fit_int32(3_000_000, 1000, 1000)
    # an empty query axis has no row to multiply
    assert mod._offsets_fit_int32(0, 4096, 1000)

    # the boundary itself: the guard leaves MAX_BLOCK of slack for the in-row
    # column offset, so the last accepted n_q is that much below the naive one
    limit = (2**31 - 1 - mod.MAX_BLOCK) // 4096 + 1
    assert mod._offsets_fit_int32(limit, 4096)
    assert not mod._offsets_fit_int32(limit + 1, 4096)


def test_triton_really_does_type_a_small_int_argument_as_int32():
    """The premise of the guard above, pinned against the installed Triton so a
    version that changed it would fail loudly here rather than silently reopen
    the overflow."""
    jit = pytest.importorskip("triton.runtime.jit")
    assert jit.mangle_type(4096) == "i32"
    assert jit.mangle_type(2**31 - 1) == "i32", "still i32 right up to the wrap"
    assert jit.mangle_type(2**31) == "i64"


@pytest.mark.parametrize(
    "mod_name,env",
    [("topk_triton", "NOVA_BF_NO_TOPK_KERNEL"), ("merge_triton", "NOVA_BF_NO_FOLD_KERNEL")],
)
def test_each_kernel_has_a_field_kill_switch(mod_name, env, monkeypatch):
    """Both kernels are pure optimizations over a portable path that computes
    the identical answer, so an operator who suspects one on their hardware must
    be able to switch it off WITHOUT a code change — and must be able to switch
    off both, not just one. Only the fold had a switch before."""
    import importlib

    mod = importlib.import_module(f"nova_bf.{mod_name}")
    src = mod.__loader__.get_source(mod.__name__)
    assert env in src, f"{mod_name} has no {env} kill switch"
    monkeypatch.setenv(env, "1")
    # the gate must decline on the env var alone, before it touches any tensor
    args = (None,) * (3 if mod_name == "topk_triton" else 5)
    assert mod.available(*args) is False


@pytest.mark.parametrize("mod_name", ["topk_triton", "merge_triton"])
def test_a_declining_gate_says_so_once(mod_name, monkeypatch, caplog):
    """A wrong `available()` False costs a multiple of the select time and
    nothing else — which is exactly why it has to be logged. `disable()`
    announces a kernel that BROKE; a gate that quietly never matches (sparse's
    transposed score matrices do this for the whole run) would otherwise leave
    the job several times slower with nothing anywhere recording it.

    Once, not per call: these gates are consulted on every slice of every
    search, millions of times in a real run.
    """
    import importlib
    import logging

    mod = importlib.import_module(f"nova_bf.{mod_name}")
    monkeypatch.setattr(mod, "_DECLINE_LOGGED", False)
    monkeypatch.setattr(mod, "_available", lambda *a, **k: False)

    shape = torch.zeros(2, 4)
    args = (shape, shape, 1) if mod_name == "topk_triton" else (shape, shape, shape, shape, 1)
    with caplog.at_level(logging.INFO, logger=mod.__name__):
        for _ in range(5):
            assert mod.available(*args) is False
    lines = [r for r in caplog.records if "portable path" in r.getMessage()]
    assert len(lines) == 1, f"expected exactly one decline log, got {len(lines)}"


def test_two_real_candidates_can_never_share_a_packed_key():
    """The invariant that makes "the kernel and the portable path agree" a
    THEOREM rather than a coincidence.

    Both paths pick k winners by packed key, but they break an exact key tie
    differently — the Triton kernels resolve duplicates at the cut by lane
    order, `torch.topk` by whatever its radix select happens to do. That
    difference is harmless only because two REAL candidates cannot share a key:
    the low half is `0xFFFFFFFF - ordinal` and a worker's ordinals are unique,
    so the key is distinct regardless of how many candidates share a score.

    The one deliberate exception is the `-inf` sentinel, whose duplicates are
    interchangeable in key AND id, so any choice among them is the same answer.

    If this ever stopped holding, the kernels would silently disagree with the
    portable path on which of two equally-ranked hits to keep — and the
    artifact-identity tests are the only thing that would notice.
    """
    rng = np.random.default_rng(0)
    # Unique ordinals spanning the legal range, INCLUDING both edges. Built by
    # sampling and de-duplicating rather than `rng.choice(TIE_WORST,
    # replace=False)`, which would try to enumerate a 4.29-billion population
    # (34 GB) to draw 50,000 values from it.
    top = tb.TIE_WORST
    ordinals_np = np.unique(
        np.concatenate([
            np.array([0, 1, top - 2, top - 1], dtype=np.int64),
            rng.integers(0, top, size=50_000, dtype=np.int64),
        ])
    )
    n = len(ordinals_np)
    ordinals = torch.from_numpy(ordinals_np)
    # the adversarial case for uniqueness: EVERY candidate has the same score,
    # so the ordinal alone has to separate them
    same = pack(torch.zeros(n, dtype=torch.float32), ordinals)
    assert len(set(same.tolist())) == n, "equal scores collided despite unique ordinals"

    # and with scores that themselves repeat heavily
    coarse = torch.from_numpy(rng.integers(0, 4, size=n).astype(np.float32))
    mixed = pack(coarse, ordinals)
    assert len(set(mixed.tolist())) == n, "a real key collision is possible"

    # the sentinel is the sole duplicate-by-design, and it sits below every
    # real candidate rather than tying with one
    sent = int(pack(torch.tensor([float("-inf")]), torch.tensor([top]))[0])
    assert sent not in set(mixed.tolist()), "a real candidate collided with the sentinel"
    assert sent < int(mixed.min()), "the sentinel must lose to every real candidate"
