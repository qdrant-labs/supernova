"""`tiebreak='id'` ranks ids three different ways; all three must agree.

`build_ordinals` picks a path by what the machine can do:

    tier 1  GPU, int64 keys, one pass per 8 id bytes
    tier 2  GPU, int32 key halves, two passes per 8 id bytes
    tier 3  CPU, Arrow's string sort         no CUDA, or ids not fixed-width

Measured on a real shard (315M ids): 10.6s / 24.6s / 503.1s. Speed is the whole
point of the split, which is exactly why it needs pinning -- a fast path that
quietly ranks differently would corrupt ground truth rather than fail.

The rest of the parity harness CANNOT reach tiers 1 and 2: its corpus writes
`"id": str(g)`, giving widths 1..4, and the fixed-width guard declines. So this
module rebuilds the same corpus with zero-padded ids and drives all three paths
end to end -- compute x ranks, then merge -- comparing each against the naive
oracle and against the other two.

Rewriting the id column does not disturb the oracle: `naive.Doc` keys off `row`
(the global corpus row number) and its payload explicitly excludes `id`, so the
expected answer is identical either way.
"""
from __future__ import annotations

import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from . import compare
from .cases import K
from .runner import build_config, pinned_device, read_results, spec
from .test_parity_matrix import _filter_from_dict

from nova_bf.compute import run_compute
from nova_bf.merge import run_merge
from nova_bf.tiebreak import _NO_GPU_ORDINALS, _fixed_width

try:
    import torch
    HAVE_CUDA = torch.cuda.is_available()
except ImportError:
    HAVE_CUDA = False

# One per modality, unfiltered: this module is about the RANKING path, not the
# filter language (which the rest of the harness covers exhaustively).
SPECS = [
    ("dense_cos", "dense", "cosine", None),
    ("sparse_dot", "sparse", "dot", None),
]


# 11 bytes -> 2 lanes, so the lane split is at least exercised. The id MUST
# remain `int(id) == row`: `read_results` hands the raw id to `compare`, and the
# naive oracle keys on `Doc.row`, so scrambling the ids makes the comparison
# match DIFFERENT documents (tried it — every dense query "disagreed").
#
# That constraint forces the ids monotonic, which makes the ordinals the
# identity permutation and therefore unobservable HERE. Observability lives in
# `test_all_three_paths_produce_identical_ORDINALS`, which builds its own
# scrambled, duplicate-bearing id columns and compares ordinals directly rather
# than going through results.
def _wide_id(g: int) -> str:
    return f"{g:011d}"


@pytest.fixture(scope="module")
def ds_fixed(ds, tmp_path_factory):
    """`ds` with the corpus id column widened to a constant 11 bytes.

    Same documents, same order, same oracle — only the id STRING changes, from
    `str(g)` (widths 1..4, which the fixed-width guard declines) to
    `f"{g:011d}"`, which it admits and which needs 2 lanes. See `_wide_id` for
    why these stay monotonic and what that costs.
    """
    root = tmp_path_factory.mktemp("bf_parity_fixed")
    cdir = root / "corpus"
    shutil.copytree(ds.corpus_dir, cdir)
    for p in sorted(cdir.glob("*.parquet")):
        t = pq.read_table(p)
        widened = pa.array([_wide_id(int(v.as_py())) for v in t["id"]],
                           pa.string())
        t = t.set_column(t.schema.get_field_index("id"), "id", widened)
        pq.write_table(t, p)

    import dataclasses

    out = dataclasses.replace(ds, corpus_dir=str(cdir), tmp=str(root))
    # the point of the fixture: the guard must now admit the GPU path
    first = pq.read_table(sorted(cdir.glob("*.parquet"))[0], columns=["id"])["id"]
    W = _fixed_width(first.chunks)
    assert W == 11, f"fixture failed to widen the ids (W={W})"
    assert (W + 7) // 8 == 2, "need >1 lane or the lane split is unexercised"
    # `int(id) == row` is REQUIRED here; see `_wide_id`.
    vals = [int(v.as_py()) for v in first]
    assert vals == sorted(vals), "ds_fixed must keep id == row for the oracle"
    return out


def _run(ds_fixed, *, device, tag):
    cfg = build_config(
        ds_fixed,
        [spec(n, vector_type=vt, metric=m, k=K, filter=f) for n, vt, m, f in SPECS],
        out_tag=tag,
        params={"tiebreak": "id"},
    )
    with pinned_device(device):
        for rank in range(3):
            run_compute(cfg, num_jobs=3, job_rank=rank)
        return read_results(run_merge(cfg))


def _spy_gpu_perm(monkeypatch):
    """Record `_gpu_perm` calls: the fast path fails OPEN, so a correct result
    is no evidence it ran."""
    import nova_bf.tiebreak as tb

    seen = []
    real = tb._gpu_perm
    monkeypatch.setattr(tb, "_gpu_perm",
                        lambda lanes, mode: (seen.append(mode), real(lanes, mode))[1])
    return seen


def _force_mode(monkeypatch, mode):
    """Pin `_gpu_mode`'s answer.

    Forcing free VRAM instead does NOT work here: `_run` shards 3 ways, so the
    per-rank row counts are 160/61/79 and `need64` is 8960/3416/4424 bytes --
    there is no single `free` in [need32, need64) for all three at once, so a
    global VRAM patch silently gives mode 64 on every rank and the "narrow key"
    test becomes a copy of the wide-key one. (It did, until this was fixed.)
    Patching the decision is rank-independent, and it also avoids poking
    `mem_get_info`, which `compute.py` reads for an unrelated decision.
    """
    import nova_bf.tiebreak as tb

    monkeypatch.setattr(tb, "_gpu_mode", lambda total: mode)


def _assert_matches_oracle(got, ds_fixed, oracle, label):
    for name, vt, metric, fdict in SPECS:
        want = oracle.topk(vector_type=vt, metric=metric, k=K,
                           filt=_filter_from_dict(ds_fixed, fdict))
        for qi in range(len(ds_fixed.queries)):
            compare.assert_scores_agree(
                got[name][qi], want[qi], metric=metric,
                label=f"[{label}] {name} q{qi}")


def test_cpu_path_matches_the_oracle(ds_fixed, oracle, device, monkeypatch):
    """Tier 3: the fallback, forced even where CUDA exists. The spy asserts the
    GPU was NOT used, which is the half a result comparison cannot see."""
    monkeypatch.setenv(_NO_GPU_ORDINALS, "1")
    seen = _spy_gpu_perm(monkeypatch)
    _assert_matches_oracle(_run(ds_fixed, device=device, tag="ord_cpu"),
                           ds_fixed, oracle, f"{device}/cpu-ordinals")
    assert seen == [], f"the kill switch did not keep the GPU out: {seen}"


@pytest.mark.skipif(not HAVE_CUDA, reason="needs CUDA")
def test_gpu_wide_key_path_matches_the_oracle(ds_fixed, oracle, monkeypatch):
    """Tier 1 produces a legitimate top-K — and ONLY that.

    This corpus has no exact score ties, so the ordinals cannot move a hit and
    this does not pin the ranking. The spy is what makes it mean anything: the
    fast path fails OPEN, so without it a decline anywhere in
    `_gpu_ready`/`_fixed_width`/`_gpu_mode` would silently run tier 3 and pass.
    """
    seen = _spy_gpu_perm(monkeypatch)
    _assert_matches_oracle(_run(ds_fixed, device="cuda", tag="ord_gpu64"),
                           ds_fixed, oracle, "gpu-int64")
    assert seen and set(seen) == {64}, f"tier 1 did not run: {seen}"


@pytest.mark.skipif(not HAVE_CUDA, reason="needs CUDA")
def test_gpu_narrow_key_path_matches_the_oracle(ds_fixed, oracle, monkeypatch):
    """Tier 2 produces a legitimate top-K. Same caveat as tier 1; the spy
    proves the narrow mode is what ran."""
    _force_mode(monkeypatch, 32)
    seen = _spy_gpu_perm(monkeypatch)
    _assert_matches_oracle(_run(ds_fixed, device="cuda", tag="ord_gpu32"),
                           ds_fixed, oracle, "gpu-int32")
    assert seen and set(seen) == {32}, f"tier 2 did not run: {seen}"


@pytest.mark.skipif(not HAVE_CUDA, reason="needs CUDA")
def test_all_three_ordinal_paths_return_identical_hits(ds_fixed, monkeypatch):
    """The three paths return the same hits end to end (compute x3 + merge).

    NOT the "strong claim" an earlier version of this docstring asserted: this
    corpus has no exact score ties, so the tie-break decides nothing and these
    hits would match even if `build_ordinals` were wrong — measured, by
    reversing the ranking entirely and seeing 0 of 16 results move. What it
    does pin is that swapping the ranking path leaves the sharded pipeline
    undisturbed. The ordinals are pinned by
    `test_all_three_paths_produce_identical_ORDINALS`.
    """
    gpu64 = _run(ds_fixed, device="cuda", tag="same_gpu64")

    _force_mode(monkeypatch, 32)
    gpu32 = _run(ds_fixed, device="cuda", tag="same_gpu32")
    monkeypatch.undo()

    monkeypatch.setenv(_NO_GPU_ORDINALS, "1")
    cpu = _run(ds_fixed, device="cuda", tag="same_cpu")

    for name, _vt, _m, _f in SPECS:
        for qi in range(len(ds_fixed.queries)):
            assert gpu64[name][qi] == gpu32[name][qi], (
                f"{name} q{qi}: int64 and int32 key modes disagree")
            assert gpu64[name][qi] == cpu[name][qi], (
                f"{name} q{qi}: GPU and CPU ordinals disagree")


@pytest.mark.skipif(not HAVE_CUDA, reason="needs CUDA")
def test_all_three_paths_produce_identical_ORDINALS():
    """Compare the ordinals themselves, not just the search results.

    The result-level tests above can pass while the ranking is wrong: this
    corpus has no exact score ties, so the tie-break never decides anything and
    `build_ordinals` could return garbage without moving a single hit. (Measured:
    fully REVERSING the ranking changed 0 of 16 query results.) Ordinals are
    observable regardless of ties, so this is the assertion that actually pins
    the three paths together.
    """
    import nova_bf.tiebreak as tb

    # Purpose-built ids rather than the corpus's: non-monotonic (so the
    # ranking is NOT the identity), duplicated (so the position tie-break
    # decides something), 11 bytes (so nlanes == 2), and split across files
    # with several chunks each (so the per-file split is exercised).
    rng = np.random.default_rng(17)
    vals = [f"{(g * 2654435761) % 10**11:011d}" for g in range(900)]
    vals[100:110] = vals[:10]                      # exact duplicate ids
    rng.shuffle(vals)
    cols = [
        pa.chunked_array([pa.array(vals[0:120], pa.string()),
                          pa.array(vals[120:300], pa.string())]),
        pa.chunked_array([pa.array(vals[300:640], pa.string())]),
        pa.chunked_array([pa.array(vals[640:700], pa.string()),
                          pa.array(vals[700:900], pa.string())]),
    ]
    assert _fixed_width([c for col in cols for c in col.chunks]) == 11
    flatvals = [v for col in cols for v in col.to_pylist()]
    assert flatvals != sorted(flatvals), "ids must not be monotonic"
    assert len(set(flatvals)) < len(flatvals), "need duplicate ids"

    def run(mode=None, kill=False):
        mp = pytest.MonkeyPatch()
        try:
            if kill:
                mp.setenv("NOVA_BF_NO_GPU_ORDINALS", "1")
            else:
                mp.delenv("NOVA_BF_NO_GPU_ORDINALS", raising=False)
                if mode is not None:
                    mp.setattr(tb, "_gpu_mode", lambda total: mode)
            seen = []
            real = tb._gpu_perm
            mp.setattr(tb, "_gpu_perm",
                       lambda lanes, m: (seen.append(m), real(lanes, m))[1])
            out = tb.build_ordinals(cols)
            return out, seen
        finally:
            mp.undo()

    wide, seen64 = run(mode=64)
    narrow, seen32 = run(mode=32)
    cpu, seencpu = run(kill=True)

    assert seen64 == [64], f"wide-key path did not run: {seen64}"
    assert seen32 == [32], f"narrow-key path did not run: {seen32}"
    assert seencpu == [], f"CPU path unexpectedly used the GPU: {seencpu}"

    assert [len(a) for a in wide] == [len(c) for c in cols]
    for i, (a, b, c) in enumerate(zip(wide, narrow, cpu)):
        assert np.array_equal(a, b), f"file {i}: int64 vs int32 key modes differ"
        assert np.array_equal(a, c), f"file {i}: GPU vs CPU ordinals differ"
    # and the ranking must not be the identity, or none of the above means much
    flat = np.concatenate(wide)
    assert not np.array_equal(flat, np.arange(len(flat))), \
        "ordinals are the identity permutation — the fixture is not scrambling ids"
