"""The per-query filter half of the vocabulary lookup table.

Reachable only for an INTEGER-valued `match_from_query`; every such filter in
this repo matches on strings (`dump` against `dump_set`), which take the
`searchsorted` path and never build a table at all. So these tests construct the
integer case directly rather than relying on a config to reach it.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova_bf.compute import (
    _VOCAB_LUT_MAX_BYTES,
    _build_vocab_lut,
    _encode_against_vocab,
    _vocab_lookup,
)


def test_a_prebuilt_table_gives_the_same_answer_as_building_one():
    """The whole point: this is a cost change, never a result change."""
    rng = np.random.default_rng(3)
    vocab = np.unique(rng.integers(0, 50_000, 500)).astype(np.int64)
    vals = rng.integers(0, 50_000, 20_000).astype(np.int64)

    lut = _build_vocab_lut(vocab)
    assert lut is not None, "an integer vocab under the cap must get a table"
    assert np.array_equal(
        _encode_against_vocab(vocab, vals),
        _encode_against_vocab(vocab, vals, lut),
    )


def test_nulls_and_absent_values_still_encode_to_minus_one():
    """`-1` means 'null, or not in any query's list'. Both must survive the
    prebuilt path, since `_gpu_cond_mask` keys its `valid` masking on it."""
    vocab = np.array([10, 20, 30], dtype=np.int64)
    lut = _build_vocab_lut(vocab)

    present = np.array([10, 30, 20], dtype=np.int64)
    assert _encode_against_vocab(vocab, present, lut).tolist() == [0, 2, 1]

    absent = np.array([11, 999, 0], dtype=np.int64)
    assert _encode_against_vocab(vocab, absent, lut).tolist() == [-1, -1, -1]

    nulls = np.array([10, None, 30], dtype=object)
    assert _encode_against_vocab(vocab, nulls, lut).tolist() == [0, -1, 2]


def test_a_string_vocab_gets_no_table_and_is_unaffected():
    """The common case — `dump` against `dump_set`. `_build_vocab_lut` declines,
    the caller passes `None`, and `searchsorted` answers exactly as before."""
    vocab = np.array(["CC-MAIN-2013-20", "CC-MAIN-2019-04"])
    assert _build_vocab_lut(vocab) is None
    vals = np.array(["CC-MAIN-2019-04", "CC-MAIN-2099-01"])
    assert _encode_against_vocab(vocab, vals, None).tolist() == [1, -1]


def test_a_vocab_over_the_byte_cap_gets_no_table():
    """The table is sized by the LARGEST id, not the vocabulary length, so a
    handful of widely-spread ids can ask for gigabytes. Past the cap there is no
    table and `_vocab_lookup` falls back — the behaviour this fix must preserve,
    because that is the case where prebuilding would be worst."""
    too_big = np.array([0, _VOCAB_LUT_MAX_BYTES // 8 + 10], dtype=np.int64)
    assert _build_vocab_lut(too_big) is None
    vals = np.array([0, 7], dtype=np.int64)
    assert _encode_against_vocab(too_big, vals, None).tolist() == [0, -1]


def test_a_mismatched_table_is_rejected_rather_than_used():
    """A table built for a different vocabulary would remap every value to the
    wrong column, silently. `_vocab_lookup` refuses instead."""
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([1, 2, 3, 400], dtype=np.int64)
    with pytest.raises(ValueError, match="different vocab"):
        _vocab_lookup(a, np.array([1], dtype=np.int64), _build_vocab_lut(b))


def _count_lut_builds(tmp_path, monkeypatch, n_files):
    """Run a real filtered search over `n_files` corpus files and return how
    many times `_build_vocab_lut` ran."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from nova_bf import compute as C
    from nova_bf.config import BruteForceConfig

    per_file, dim = 6, 4
    rng = np.random.default_rng(11)
    tmp_path.mkdir(parents=True, exist_ok=True)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    for i in range(n_files):
        pq.write_table(pa.table({
            "dense_embedding": pa.array(
                rng.standard_normal((per_file, dim)).astype("float32").tolist(),
                type=pa.list_(pa.float32())),
            "id": pa.array([f"f{i}r{r}" for r in range(per_file)]),
            # INTEGER payload -> the LUT path, unlike the repo's string filters
            "tenant": pa.array(rng.integers(0, 5000, per_file), type=pa.int64()),
        }), str(cdir / f"{i}.parquet"))

    pq.write_table(pa.table({
        "dense_embedding": pa.array(
            rng.standard_normal((3, dim)).astype("float32").tolist(),
            type=pa.list_(pa.float32())),
        "qid": pa.array(["a", "b", "c"]),
        "want_tenant": pa.array([1, 2, 3], type=pa.int64()),
    }), str(tmp_path / "q.parquet"))

    cfg = BruteForceConfig.model_validate({
        "corpus": {"path": str(cdir), "id_column": "id"},
        "queries": {"path": str(tmp_path / "q.parquet"), "id_column": "qid"},
        "output": {"path": str(tmp_path / "out")},
        "params": {"io_workers": 2},
        "searches": [{
            "name": "f", "vector_type": "dense", "metric": "dot", "k": 3,
            "filter": {"must": [{"field": "tenant", "match_from_query": "want_tenant"}]},
        }],
    })

    calls = []
    real = C._build_vocab_lut
    monkeypatch.setattr(C, "_build_vocab_lut", lambda v: (calls.append(len(v)), real(v))[1])
    C.run_compute(cfg)
    monkeypatch.undo()
    return len(calls)


def test_the_table_is_not_rebuilt_per_corpus_file(tmp_path, monkeypatch):
    """The behaviour the fix exists for.

    Before it, `_corpus_leaf_array` -> `_encode_against_vocab` ->
    `_vocab_lookup` built a fresh table for EVERY corpus file, on whichever
    reader thread got that file. So the build count scaled with the corpus.
    Now it is setup-only and the count is flat.

    Asserted as `2 files == 8 files` rather than as an absolute number, because
    the absolute number is not 1: `_build_gpu_leaf_state` also builds one while
    encoding the QUERY side of the same vocabulary. That is twice per RUN, does
    not grow, and is not what this guards."""
    pytest.importorskip("torch")

    few = _count_lut_builds(tmp_path / "few", monkeypatch, n_files=2)
    many = _count_lut_builds(tmp_path / "many", monkeypatch, n_files=8)

    assert few == many, (
        f"_build_vocab_lut ran {few}x for 2 corpus files and {many}x for 8 — it "
        "is scaling with the corpus, so it is being rebuilt per file again"
    )
    assert many <= 2, (
        f"expected at most the two setup-time builds, got {many}"
    )


def test_the_predicted_size_matches_the_real_allocation():
    """`_vocab_lut_nbytes` is what lets the prebuild loop decide affordability
    WITHOUT allocating. If it ever disagreed with the real table, the budget
    would drift silently — over-spending or refusing tables that would fit."""
    from nova_bf.compute import _vocab_lut_nbytes

    for vocab in (np.array([0], dtype=np.int64),
                  np.array([3, 9, 4096], dtype=np.int64),
                  np.array([1, 2, 1_000_000], dtype=np.int64)):
        lut = _build_vocab_lut(vocab)
        assert lut is not None
        assert _vocab_lut_nbytes(vocab) == lut.nbytes, f"mispredicted for {vocab[-1]}"

    # 0 means "no table", for every reason that is not size
    assert _vocab_lut_nbytes(np.zeros(0, dtype=np.int64)) == 0
    assert _vocab_lut_nbytes(np.array([-5, 2], dtype=np.int64)) == 0
    assert _vocab_lut_nbytes(np.array(["a", "b"])) == 0


def test_the_two_budgets_are_independent_knobs():
    """Splitting them was the point: the per-table cap is a SPEED decision (past
    it, binary search wins), the prebuild budget is a MEMORY one (past it, keep
    building per file). Moving one must not move the other."""
    from nova_bf.compute import _VOCAB_LUT_PREBUILD_BYTES, _lut_vocab_ok

    assert isinstance(_VOCAB_LUT_PREBUILD_BYTES, int)
    # `_lut_vocab_ok` — the speed decision — must consult only the per-table cap
    fits = np.array([0, _VOCAB_LUT_MAX_BYTES // 8 - 5], dtype=np.int64)
    over = np.array([0, _VOCAB_LUT_MAX_BYTES // 8 + 10], dtype=np.int64)
    assert _lut_vocab_ok(fits) and not _lut_vocab_ok(over)


def test_an_unaffordable_condition_is_never_built_at_setup(tmp_path, monkeypatch):
    """The ordering fix. The loop used to build the table and THEN check the
    budget, so an over-budget condition paid the full allocate-and-fill anyway —
    the very cost this prebuild exists to remove, relocated to setup.

    With the budget set to zero, setup must build NOTHING: the only builds left
    are the one `_build_gpu_leaf_state` does encoding the query side, plus one
    per corpus file. Any extra means the wasted build is back."""
    pytest.importorskip("torch")
    from nova_bf import compute as C

    monkeypatch.setattr(C, "_VOCAB_LUT_PREBUILD_BYTES", 0)
    n_files = 3
    builds = _count_lut_builds(tmp_path / "poor", monkeypatch, n_files=n_files)
    assert builds == 1 + n_files, (
        f"expected 1 setup build (query side) + {n_files} per-file builds, got "
        f"{builds} — an over-budget table is being built and discarded at setup"
    )
