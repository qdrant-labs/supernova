"""`_build_vocab_lut` / `_vocab_lookup(lut=...)` — the run-wide vocabulary LUT.

`_vocab_lookup`'s fast path used to build its lookup table on every call, and
its hot caller (`_remap_sparse_file`) runs once per corpus file in the reader
threads against a vocabulary that is fixed for the whole run. Hoisting the
build must be a pure performance change: every lookup has to return exactly
what the unhoisted path returned, on every id shape and dtype.
"""
from __future__ import annotations

import numpy as np
import pytest

import nova_bf.compute as C


def _vocab(ids):
    return np.array(sorted(set(ids)), dtype=np.int64)


CASES = [
    ("simple", [3, 7, 11], [7, 0, 11, 99, 3]),
    ("dense_from_zero", [0, 1, 2, 3], [3, 2, 1, 0, 4]),
    ("single_term", [5], [5, 4, 6]),
    ("gaps", [0, 1000, 100000], [0, 999, 1000, 100000, 100001]),
    ("all_absent", [2, 4, 6], [1, 3, 5, 7]),
    ("empty_ids", [1, 2, 3], []),
    ("repeats", [4, 9], [9, 9, 4, 4, 9]),
]


@pytest.mark.parametrize("name,vocab,ids", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.uint32, np.uint64])
def test_prebuilt_lut_matches_the_per_call_path(name, vocab, ids, dtype):
    v = _vocab(vocab)
    i = np.array(ids, dtype=dtype)
    want = C._vocab_lookup(v, i)                      # builds its own
    got = C._vocab_lookup(v, i, C._build_vocab_lut(v))  # prebuilt
    assert np.array_equal(got, want), f"{name}/{dtype.__name__}"
    assert got.dtype == want.dtype


def test_prebuilt_lut_matches_a_searchsorted_oracle():
    """Independent of both implementations: position in the sorted vocab, or
    -1 when absent."""
    rng = np.random.default_rng(4)
    v = _vocab(rng.choice(5000, 400, replace=False))
    i = rng.integers(0, 6000, 3000).astype(np.int64)
    lut = C._build_vocab_lut(v)
    got = C._vocab_lookup(v, i, lut)
    pos = {int(t): n for n, t in enumerate(v)}
    want = np.array([pos.get(int(x), -1) for x in i], dtype=np.int64)
    assert np.array_equal(got, want)


def test_empty_vocab_needs_no_lut():
    v = np.zeros(0, dtype=np.int64)
    assert C._build_vocab_lut(v) is None
    out = C._vocab_lookup(v, np.array([1, 2], dtype=np.int64))
    assert np.array_equal(out, np.array([-1, -1]))


def test_lut_declined_for_ineligible_vocabularies():
    """A LUT is only valid for bounded, non-negative integer vocabularies —
    everything else must still work through `searchsorted`."""
    over = np.array([0, C._VOCAB_LUT_MAX_BYTES // 8 + 10], dtype=np.int64)
    assert C._build_vocab_lut(over) is None, "budget cap not enforced"
    assert np.array_equal(
        C._vocab_lookup(over, np.array([0, 7], dtype=np.int64)),
        np.array([0, -1]))

    neg = np.array([-5, 2], dtype=np.int64)
    assert C._build_vocab_lut(neg) is None
    assert np.array_equal(
        C._vocab_lookup(neg, np.array([2, -5, 1], dtype=np.int64)),
        np.array([1, 0, -1]))

    strs = np.array(["ab", "cd"], dtype=object)
    assert C._build_vocab_lut(strs) is None
    assert np.array_equal(
        C._vocab_lookup(strs, np.array(["cd", "zz"], dtype=object)),
        np.array([1, -1]))


def test_negative_ids_still_take_the_search_path_even_with_a_lut():
    """A prebuilt LUT must not let negative ids index it: numpy wraps, so
    `lut[-2]` reads a REAL vocabulary slot and returns a wrong column.

    `-1` alone is a trap of a test — it wraps onto the table's sentinel slot,
    which already holds -1, so a broken implementation returns the right
    answer by luck. The negatives below wrap onto live entries.
    """
    v = _vocab([1, 2, 3])          # lut = [-1, 0, 1, 2, -1], 5 slots
    lut = C._build_vocab_lut(v)
    assert lut.table.tolist() == [-1, 0, 1, 2, -1], lut.table.tolist()

    for bad in (-1, -2, -3, -4, -5):
        ids = np.array([bad], dtype=np.int64)
        got = C._vocab_lookup(v, ids, lut)
        assert got.tolist() == [-1], (
            f"id {bad} must be absent, got {got.tolist()} "
            f"(lut[{bad}] = {lut[bad]}) — negatives indexed the table")
        assert np.array_equal(got, C._vocab_lookup(v, ids))

    mixed = np.array([-2, 2, -4, 3], dtype=np.int64)
    assert np.array_equal(C._vocab_lookup(v, mixed, lut),
                          C._vocab_lookup(v, mixed))
    assert C._vocab_lookup(v, mixed, lut).tolist() == [-1, 1, -1, 2]


def test_a_mismatched_lut_is_rejected_loudly():
    """Silently remapping every token to the wrong column is the worst
    outcome here, so a LUT from another vocabulary must raise."""
    v1, v2 = _vocab([1, 2, 3]), _vocab([1, 2, 300])
    with pytest.raises(ValueError, match="different vocab"):
        C._vocab_lookup(v1, np.array([2], dtype=np.int64), C._build_vocab_lut(v2))


def test_a_mismatched_lut_with_the_SAME_MAX_ID_is_rejected():
    """The case the old guard could not see, and the reason it was replaced.

    The check used to be `len(lut) != int(vocab[-1]) + 2`, and `len(lut)` IS
    `int(vocab[-1]) + 2` by construction — so it tested the largest id and
    nothing else. These two vocabularies have the same length AND the same
    maximum, so both build a 12-slot table and the old check passed them as
    interchangeable. Every token would then be remapped to the wrong column:
    wrong ground truth, no exception, nothing in the manifest.

    `test_a_mismatched_lut_is_rejected_loudly` above does NOT cover this — its
    two vocabularies differ in maximum, which the size test already caught.
    """
    a, b = _vocab([0, 5, 10]), _vocab([3, 7, 10])
    lut_a = C._build_vocab_lut(a)
    assert len(lut_a.table) == int(b[-1]) + 2, (
        "these vocabularies must produce IDENTICALLY SIZED tables, or this "
        "test is not exercising the hole")

    with pytest.raises(ValueError, match="different vocab"):
        C._vocab_lookup(b, np.array([7], dtype=np.int64), lut_a)


def test_an_EQUAL_vocabulary_in_a_different_array_is_accepted():
    """Identity is the fast path, not the rule. A caller that rebuilt an equal
    vocabulary must still be served rather than refused — otherwise the guard
    trades silent corruption for a spurious hard failure."""
    v = _vocab([0, 5, 10])
    rebuilt = _vocab([0, 5, 10])
    assert rebuilt is not v

    lut = C._build_vocab_lut(v)
    got = C._vocab_lookup(rebuilt, np.array([5, 10, 7], dtype=np.int64), lut)
    assert got.tolist() == [1, 2, -1], got.tolist()


def test_the_lut_keeps_the_vocabulary_OBJECT_it_was_built_from():
    """What makes the check O(1) in production: the LUT holds the very array it
    was built from, so every real call site compares pointers instead of
    contents. Storing a copy would still be CORRECT but would silently move
    every lookup onto the O(n) compare."""
    v = _vocab([0, 5, 10])
    assert C._build_vocab_lut(v).vocab is v


def test_remap_sparse_file_is_identical_with_and_without_a_lut():
    rng = np.random.default_rng(11)
    n_rows, per = 40, 6
    indices = rng.integers(0, 500, n_rows * per).astype(np.int64)
    values = rng.random(n_rows * per).astype(np.float32)
    offsets = np.arange(0, n_rows * per + 1, per, dtype=np.int64)
    v = _vocab(rng.choice(500, 120, replace=False))

    a = C._remap_sparse_file(offsets, indices, values, v)
    b = C._remap_sparse_file(offsets, indices, values, v, C._build_vocab_lut(v))
    for x, y, name in zip(a, b, ("offsets", "idx", "values")):
        assert np.array_equal(x, y), name


# ---------------------------------------------------------------------------
# the point of the change: built once per RUN, not once per FILE
# ---------------------------------------------------------------------------


def test_the_lut_is_built_once_per_run_not_once_per_file(tmp_path, monkeypatch):
    """`_remap_sparse_file` runs once per corpus file in the reader threads.
    Before this change each of those calls built its own table — a full-width
    memset per file, several resident at once with `io_workers > 1`.

    A file-count-independent build count is the whole property, so assert it
    against a corpus with several files.
    """
    pytest.importorskip("torch")
    from nova_bf.compute import run_compute
    from test_prune_search_paths import _sparse_corpus, _sparse_cfg

    builds = {"n": 0}
    real_build = C._build_vocab_lut
    remaps = {"n": 0}
    real_remap = C._remap_sparse_file

    def build_spy(vocab):
        builds["n"] += 1
        return real_build(vocab)

    def remap_spy(offsets, indices, values, vocab, lut=None):
        remaps["n"] += 1
        assert lut is not None, "the hot path did not receive the prebuilt LUT"
        return real_remap(offsets, indices, values, vocab, lut)

    monkeypatch.setattr(C, "_build_vocab_lut", build_spy)
    monkeypatch.setattr(C, "_remap_sparse_file", remap_spy)

    n_files = 6
    cdir, qpath = _sparse_corpus(tmp_path, n_files=n_files, per_file=60, seed=2)
    run_compute(_sparse_cfg(cdir, qpath, tmp_path / "lutrun", k=5))

    assert remaps["n"] == n_files, f"expected one remap per file, got {remaps}"
    assert builds["n"] == 1, (
        f"LUT built {builds['n']} times for {n_files} files — it must be "
        "hoisted out of the per-file path")


def test_hoisting_the_lut_does_not_change_results(tmp_path, monkeypatch):
    """Belt and braces on the end-to-end path: forcing the per-call build
    (lut=None, the old behaviour) must give byte-identical output."""
    pytest.importorskip("torch")
    import pyarrow.parquet as pq
    from nova_bf.compute import run_compute
    from test_prune_search_paths import _sparse_corpus, _sparse_cfg

    cdir, qpath = _sparse_corpus(tmp_path, n_files=5, per_file=80, seed=6)

    def read(paths):
        return {n: pq.read_table(p).to_pydict() for n, p in paths.items()}

    hoisted = read(run_compute(_sparse_cfg(cdir, qpath, tmp_path / "a", k=8)))

    real_remap = C._remap_sparse_file
    monkeypatch.setattr(
        C, "_remap_sparse_file",
        lambda o, i, v, vocab, lut=None: real_remap(o, i, v, vocab, None))
    per_call = read(run_compute(_sparse_cfg(cdir, qpath, tmp_path / "b", k=8)))

    for name in hoisted:
        assert hoisted[name]["hit_ids"] == per_call[name]["hit_ids"], name
        assert hoisted[name]["hit_scores"] == per_call[name]["hit_scores"], name
